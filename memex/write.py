# -*- coding: utf-8 -*-
"""M3 写入与治理：新记忆入库、去重合并、supersede、衰减分档、edges 维护。

治理约定（全部可解释、不调在线 LLM）：
- 写入必须带 `source`，空则抛 ValueError（不许匿名写入）。
- 去重：同 scope 下 bge-m3 cos > 0.92 视为同一条 → 合并：
  文本取信息量大的那份、importance 取 max、access_count 相加、保留旧 id（身份稳定）。
- supersede：只改 `status='superseded'` + `superseded_by`，**不删行**（历史要能回溯，
  召回侧（recall.load）只取 `status='active'`，所以旧条自然不再返回）。
- importance 规则化四档：0.9 用户明确偏好 / 0.8 技术事实 / 0.6 过程步骤 / 0.4 闲聊。
- 衰减分档沿用现役约定：>=0.9 tier0（永固）、0.8→360 天、0.7→180 天、0.5→60 天、其余 30 天。

用法：
  PYTHONUTF8=1 <venv>python -m memex.write --stats
  PYTHONUTF8=1 <venv>python -m memex.write --add "以后都用 UTF-8 编码" --scope facts --source manual
  PYTHONUTF8=1 <venv>python -m memex.write --rebuild-edges
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from collections import defaultdict

import numpy as np

from .db import blob_to_vec, connect, new_id, vec_to_blob
from .recall import tokenize

DEDUP_COS = 0.92
EDGE_KIND_BATCH = "co_batch"
EDGE_KIND_TOKEN = "co_token"
EDGE_KIND_SOURCE = "co_source"

# ---------- importance 规则 ----------

# 用户明确偏好 / 长期约定：这是「以后都要这么做」的指令，永固级别
_PREF = (
    "以后都", "以后别", "以后要", "记住", "必须", "一律", "统一用", "统一改成", "永远",
    "不许", "不要再", "别再用", "优先用", "优先选", "我的偏好", "我喜欢", "我习惯",
    "我们约定", "约定好", "默认用", "固定用", "从今天起",
    "always use", "never use", "must use", "i prefer", "from now on", "remember that",
)

# 技术事实：端口/路径/配置/版本/命令/结论，这类是「查得到才有用」的硬信息
_TECH = (
    "端口", "路径", "配置", "参数", "版本", "安装", "命令", "报错", "依赖", "环境变量",
    "函数", "脚本", "数据库", "表结构", "字段", "接口", "日志", "服务", "部署", "显卡",
    "显存", "量化", "模型", "缓存", "容器", "镜像", "编译", "调试", "权限", "证书",
    "api", "token", "key", "url", "http", "https", "sqlite", "chroma", "python",
    "docker", "npm", "pip", "git", "json", "yaml", "uvicorn", "mcp", "gpu", "cpu",
    "sql", "css", "html", ".py", ".exe", ".json", ".db", ".md", ".ps1", ".sh",
    "localhost", "127.0.0.1", "windows", "linux", "macos",
)
_TECH_RE = re.compile(r"(?:[A-Za-z]:\\|/[\w.\-]+/|\b\d+(?:\.\d+)?\s*(?:ms|s|gb|mb|kb|g|m)\b)", re.I)

# 过程步骤：怎么做的流水账，半年后基本只对复盘有用
_PROC = (
    "第一步", "第二步", "步骤", "流程", "先", "然后", "接着", "之后", "最后",
    "打开", "点击", "运行", "执行", "保存", "重启", "导入", "导出", "复制",
)


def explain_importance(text: str) -> tuple[float, str]:
    """返回 (importance, 依据)。纯规则、可解释、可单测。"""
    t = (text or "").lower()
    if not t.strip():
        return 0.4, "空文本"

    pref_hits = [k for k in _PREF if k in t]
    if pref_hits:
        return 0.9, f"用户明确偏好/长期约定：{pref_hits[0]}"

    tech_hits = [k for k in _TECH if k in t]
    if _TECH_RE.search(t):
        tech_hits.append("路径或带单位数值")

    proc_hits = [k for k in _PROC if k in t]
    # 流水账优先级高于单个泛化技术词（「先打开…然后点击…」不该被「服务」抬成技术事实）
    if len(proc_hits) >= 2 and len(tech_hits) <= 1:
        return 0.6, f"过程步骤：{proc_hits[0]},{proc_hits[1]}"

    short = len(t) <= 60
    if len(tech_hits) >= 2 or (tech_hits and short):
        return 0.8, f"技术事实：{','.join(str(x) for x in tech_hits[:3])}"

    if proc_hits:
        return 0.6, f"过程步骤：{proc_hits[0]}"

    return 0.4, "闲聊/无硬信息"


def estimate_importance(text: str) -> float:
    """四档：0.9 偏好 / 0.8 技术事实 / 0.6 过程步骤 / 0.4 闲聊。"""
    return explain_importance(text)[0]


def tier_for(importance: float) -> int:
    """importance → decay_tier（天）；0 表示永固。与 import_om.tier_for 同口径。"""
    if importance >= 0.9:
        return 0
    if importance >= 0.8:
        return 360
    if importance >= 0.7:
        return 180
    if importance >= 0.5:
        return 60
    return 30


def stability_for(tier: int) -> float:
    """FSRS 稳定度（天）：永固给 3650，其余按档位一半。与 import_om 同口径。"""
    return 3650.0 if not tier else max(1.0, tier / 2.0)


# ---------- 写入器 ----------

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def _info_size(text: str) -> tuple[int, int]:
    """信息量比较口径：先比有效 token 数，再比字符数。"""
    return len(tokenize(text or "")), len((text or "").strip())


class Writer:
    """写入与治理。所有方法都直接操作传入的 con，便于对临时库跑测试。"""

    def __init__(self, con: sqlite3.Connection, embedder=None, dedup_cos: float = DEDUP_COS):
        self.con = con
        self._embedder = embedder
        self.dedup_cos = float(dedup_cos)
        # scope -> [ {id,text,importance,access_count,decay_tier,meta,source}, ndarray ]
        self._idx: dict[str, tuple[list[dict], np.ndarray]] = {}

    # ---- 依赖 ----
    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import Embedder  # 延迟导入：只有真要写时才需要 key
            self._embedder = Embedder(self.con)
        return self._embedder

    # ---- 索引 ----
    def _index(self, scope: str) -> tuple[list[dict], np.ndarray]:
        if scope in self._idx:
            return self._idx[scope]
        rows = self.con.execute(
            "SELECT id, text, importance, access_count, decay_tier, meta, source FROM memories "
            "WHERE scope=? AND status='active' AND vec IS NOT NULL", (scope,)
        ).fetchall()
        recs = [{"id": r["id"], "text": r["text"], "importance": float(r["importance"]),
                 "access_count": int(r["access_count"]), "decay_tier": int(r["decay_tier"]),
                 "meta": r["meta"] or "{}", "source": r["source"] or ""} for r in rows]
        vecs = self.con.execute(
            "SELECT vec FROM memories WHERE scope=? AND status='active' AND vec IS NOT NULL", (scope,)
        ).fetchall()
        if not rows:
            mat = np.zeros((0, int(self.embedder.dim()) or 1024), dtype=np.float32)
        else:
            mat = np.vstack([blob_to_vec(v["vec"]) for v in vecs]).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        self._idx[scope] = (recs, mat)
        return self._idx[scope]

    def _index_append(self, scope: str, rec: dict, vec: np.ndarray) -> None:
        if scope not in self._idx:
            return
        recs, mat = self._idx[scope]
        recs.append(rec)
        self._idx[scope] = (recs, np.vstack([mat, _normalize(vec)]).astype(np.float32))

    def _find_dup(self, scope: str, vec: np.ndarray) -> tuple[dict, float] | None:
        recs, mat = self._index(scope)
        if not recs:
            return None
        sims = mat @ vec
        i = int(np.argmax(sims))
        s = float(sims[i])
        return (recs[i], s) if s >= self.dedup_cos else None

    # ---- 写入 ----
    def add(self, text: str, scope: str, kind: str, source: str,
            importance: float | None = None, meta: dict | None = None) -> str:
        """写入一条记忆；与现有条目近似（cos>0.92）时合并，返回最终 id。"""
        return self.add_many([{"text": text, "scope": scope, "kind": kind, "source": source,
                               "importance": importance, "meta": meta}])[0]

    def add_many(self, items: list[dict], source: str | None = None) -> list[str]:
        """批量写入。同一次调用里产生的记忆两两加权连边（同 scope +0.5、同 source +0.3）。"""
        now = int(time.time())
        ids: list[str] = []
        created: list[tuple[str, str, str]] = []   # (id, scope, source)
        for it in items:
            src = str(it.get("source") or source or "").strip()
            if not src:
                raise ValueError("写入必须带 source（不许匿名写入）")
            text = str(it.get("text") or "").strip()
            if not text:
                raise ValueError("text 不能为空")
            scope = str(it.get("scope") or "facts")
            kind = str(it.get("kind") or ("note" if scope == "notes" else "fact"))
            if it.get("importance") is None:
                imp, _why = explain_importance(text)
            else:
                imp = min(1.0, max(0.0, float(it["importance"])))
            m = dict(it.get("meta") or {})
            m.setdefault("added_at", now)

            vec = _normalize(self.embedder.embed_one(text))
            dup = self._find_dup(scope, vec)
            if dup:
                ids.append(self._merge(dup[0], text, imp, src, m, vec, now))
                continue

            mid = new_id()
            tier = tier_for(imp)
            self.con.execute(
                "INSERT INTO memories(id, scope, kind, text, importance, stability, decay_tier,"
                " created_at, last_access, access_count, status, superseded_by, source, meta, vec,"
                " valid_from, valid_to, superseded_by_id)"
                " VALUES(?,?,?,?,?,?,?,?,?,0,'active','',?,?,?,?,0,'')",
                (mid, scope, kind, text, imp, stability_for(tier), tier, now, now, src,
                 json.dumps(m, ensure_ascii=False), vec_to_blob(vec), now))
            ids.append(mid)
            created.append((mid, scope, src))
            self._index_append(scope, {"id": mid, "text": text, "importance": imp,
                                       "access_count": 0, "decay_tier": tier,
                                       "meta": json.dumps(m, ensure_ascii=False), "source": src}, vec)

        self.con.commit()
        if len(created) > 1:
            self._link(created)
        return ids

    def _merge(self, old: dict, text: str, imp: float, src: str, meta: dict,
               vec: np.ndarray, now: int) -> str:
        """同条合并：文本取信息量大的、importance 取 max、access_count 相加。"""
        use_new = _info_size(text) > _info_size(old["text"])
        imp = max(float(old["importance"]), imp)
        tier = tier_for(imp)
        try:
            old_meta = json.loads(old.get("meta") or "{}")
        except Exception:
            old_meta = {}
        merged_meta = {**old_meta, **meta, "merges": int(old_meta.get("merges") or 0) + 1}

        if use_new:
            self.con.execute(
                "UPDATE memories SET text=?, vec=?, importance=?, stability=?, decay_tier=?,"
                " access_count=?, last_access=?, meta=?, source=COALESCE(NULLIF(source,''),?)"
                " WHERE id=?",
                (text, vec_to_blob(vec), imp, stability_for(tier), tier,
                 int(old["access_count"]) + 1, now,
                 json.dumps(merged_meta, ensure_ascii=False), src, old["id"]))
        else:
            self.con.execute(
                "UPDATE memories SET importance=?, stability=?, decay_tier=?, access_count=?,"
                " last_access=?, meta=?, source=COALESCE(NULLIF(source,''),?) WHERE id=?",
                (imp, stability_for(tier), tier, int(old["access_count"]) + 1, now,
                 json.dumps(merged_meta, ensure_ascii=False), src, old["id"]))

        scope = self._scope_of(old["id"])
        if scope:
            self._idx.pop(scope, None)   # 合并改了 scope 索引里的内容，下次用时重建
        return old["id"]

    def _scope_of(self, mid: str) -> str | None:
        r = self.con.execute("SELECT scope FROM memories WHERE id=?", (mid,)).fetchone()
        return r["scope"] if r else None

    # ---- 更新 / 取代 ----
    def update(self, mid: str, text: str, importance: float | None = None) -> bool:
        """改文本（重新向量化）。importance 不给则按规则重估。"""
        row = self.con.execute("SELECT scope, importance FROM memories WHERE id=?", (mid,)).fetchone()
        if not row:
            return False
        text = str(text or "").strip()
        if not text:
            raise ValueError("text 不能为空")
        imp = float(importance) if importance is not None else estimate_importance(text)
        imp = min(1.0, max(0.0, imp))
        tier = tier_for(imp)
        vec = _normalize(self.embedder.embed_one(text))
        self.con.execute(
            "UPDATE memories SET text=?, vec=?, importance=?, stability=?, decay_tier=?,"
            " last_access=? WHERE id=?",
            (text, vec_to_blob(vec), imp, stability_for(tier), tier, int(time.time()), mid))
        self.con.commit()
        self._idx.pop(row["scope"], None)
        return True

    def supersede(self, old_id: str, new_id_: str) -> None:
        """旧条保留、标 superseded 并指向新条（召回声明的「正确写」靠它，不删行）。"""
        for x in (old_id, new_id_):
            if not self.con.execute("SELECT 1 FROM memories WHERE id=?", (x,)).fetchone():
                raise ValueError(f"记忆不存在：{x}")
        if old_id == new_id_:
            raise ValueError("supersede 的新旧 id 不能相同")
        old_pin = self.con.execute(
            "SELECT pinned, core_slot FROM memories WHERE id=?", (old_id,)).fetchone()
        nf = self.con.execute(
            "SELECT valid_from, created_at FROM memories WHERE id=?", (new_id_,)).fetchone()
        new_from = (nf[0] or nf[1]) if nf else int(time.time())
        self.con.execute(
            "UPDATE memories SET status='superseded', superseded_by=?, superseded_by_id=?,"
            " valid_to=?, last_access=? WHERE id=?",
            (new_id_, new_id_, int(new_from), int(time.time()), old_id))
        # core 不能指向被 supersede 的旧条：pin 位跟着迁到新条
        if old_pin and old_pin[0]:
            self.con.execute(
                "UPDATE memories SET pinned=1, core_slot=? WHERE id=?", (old_pin[1] or "", new_id_))
            self.con.execute(
                "UPDATE memories SET pinned=0, core_slot='' WHERE id=?", (old_id,))
        self.con.commit()

    # ---- 边 ----
    def _link(self, nodes: list[tuple[str, str, str]]) -> int:
        """同一次写入里两两加权：同 scope +0.5、同 source +0.3。无向边存双向。"""
        n = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, sa, ra = nodes[i]
                b, sb, rb = nodes[j]
                w = (0.5 if sa == sb else 0.0) + (0.3 if ra and ra == rb else 0.0)
                if w <= 0:
                    continue
                n += self._put_edge(a, b, w, EDGE_KIND_BATCH)
        self.con.commit()
        return n

    def _put_edge(self, a: str, b: str, w: float, kind: str) -> int:
        cur = self.con.execute(
            "INSERT INTO edges(a,b,weight,kind) VALUES(?,?,?,?)"
            " ON CONFLICT(a,b,kind) DO UPDATE SET weight=MAX(edges.weight, excluded.weight)",
            (a, b, round(float(w), 4), kind))
        self.con.execute(
            "INSERT INTO edges(a,b,weight,kind) VALUES(?,?,?,?)"
            " ON CONFLICT(a,b,kind) DO UPDATE SET weight=MAX(edges.weight, excluded.weight)",
            (b, a, round(float(w), 4), kind))
        return 1 if cur.rowcount else 0

    def rebuild_edges(self, top_k: int = 12, min_weight: float = 0.15,
                      max_df: int = 40, max_source_group: int = 30,
                      max_pairs: int = 3_000_000, verbose: bool = False) -> dict:
        """按共现统计重建 `co_token` / `co_source` 边（保留 co_batch）。

        co_token：两篇共享的「稀有词」idf 之和 → 饱和映射到 0~1，每节点只留 top_k 邻居。
        co_source：同来源文件（同 source）视为强相关，权重 0.3；来源过大（>30 条）视为
        泛化来源（如 distill）跳过，避免 O(n^2) 爆炸和无意义全连接。
        """
        rows = self.con.execute(
            "SELECT id, scope, text, source FROM memories WHERE status='active'").fetchall()
        docs = [(r["id"], r["scope"], r["text"] or "", r["source"] or "") for r in rows]
        N = len(docs)
        stats = {"nodes": N, "co_token": 0, "co_source": 0, "skipped_pairs": 0}
        if N == 0:
            return stats

        toks = [list(dict.fromkeys(tokenize(d[2]))) for d in docs]
        inv: dict[str, list[int]] = defaultdict(list)
        for i, ts in enumerate(toks):
            for t in ts:
                inv[t].append(i)

        pair: dict[tuple[int, int], float] = defaultdict(float)
        ops = 0
        for t, ids in inv.items():
            if len(ids) < 2 or len(ids) > max_df:
                continue
            idf = math.log(N / (1.0 + len(ids)))
            if idf <= 0:
                continue
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    a, b = ids[x], ids[y]
                    pair[(a, b) if a < b else (b, a)] += idf
                    ops += 1
                    if ops > max_pairs:
                        break
                if ops > max_pairs:
                    break
            if ops > max_pairs:
                stats["skipped_pairs"] = 1
                break

        # 每节点保留 top_k
        per: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for (a, b), raw in pair.items():
            w = raw / (raw + 3.0)
            if w < min_weight:
                continue
            per[a].append((w, b))
            per[b].append((w, a))

        self.con.execute("DELETE FROM edges WHERE kind IN (?,?)", (EDGE_KIND_TOKEN, EDGE_KIND_SOURCE))
        for a, lst in per.items():
            lst.sort(reverse=True)
            for w, b in lst[:top_k]:
                self._put_edge(docs[a][0], docs[b][0], w, EDGE_KIND_TOKEN)

        # 同来源
        by_src: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(docs):
            s = d[3].strip()
            if not s or s == "test":
                continue
            by_src[s].append(i)
        for s, ids in by_src.items():
            if len(ids) < 2 or len(ids) > max_source_group:
                continue
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    self._put_edge(docs[ids[x]][0], docs[ids[y]][0], 0.3, EDGE_KIND_SOURCE)

        self.con.commit()
        # 边无向、存双向：行数 / 2 = 实际边数（别用循环里的调用次数，会被双向写重算错）
        for kind, c in self.con.execute(
                "SELECT kind, count(*) c FROM edges WHERE kind IN (?,?) GROUP BY kind",
                (EDGE_KIND_TOKEN, EDGE_KIND_SOURCE)):
            stats[kind] = int(c) // 2
        if verbose:
            print(json.dumps(stats, ensure_ascii=False))
        return stats

    # ---- 读 ----
    def get(self, mid: str) -> dict | None:
        r = self.con.execute(
            "SELECT id, scope, kind, text, importance, decay_tier, status, superseded_by,"
            " source, access_count, meta FROM memories WHERE id=?", (mid,)).fetchone()
        return dict(r) if r else None

    def stats(self) -> dict:
        c = self.con
        out = {
            "active": c.execute("SELECT count(*) c FROM memories WHERE status='active'").fetchone()["c"],
            "superseded": c.execute(
                "SELECT count(*) c FROM memories WHERE status='superseded'").fetchone()["c"],
            "edges": c.execute("SELECT count(*) c FROM edges").fetchone()["c"],
            "by_importance": {str(r["importance"]): r["c"] for r in c.execute(
                "SELECT importance, count(*) c FROM memories WHERE status='active'"
                " GROUP BY importance ORDER BY importance")},
            "by_source_top": [{"source": r["source"], "n": r["c"]} for r in c.execute(
                "SELECT source, count(*) c FROM memories WHERE status='active'"
                " GROUP BY source ORDER BY c DESC LIMIT 8")],
        }
        return out


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="memex.write", description="M3 写入与治理")
    ap.add_argument("--db", default=None, help="库路径（默认 data/memex.db；测试请指向副本）")
    ap.add_argument("--add", metavar="TEXT", help="写一条记忆")
    ap.add_argument("--scope", default="facts", choices=["facts", "notes"])
    ap.add_argument("--kind", default=None)
    ap.add_argument("--source", default="manual")
    ap.add_argument("--importance", type=float, default=None)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--rebuild-edges", action="store_true")
    ap.add_argument("--explain", metavar="TEXT", help="只看 importance 规则判定")
    args = ap.parse_args(argv)

    if args.explain:
        imp, why = explain_importance(args.explain)
        print(json.dumps({"text": args.explain, "importance": imp, "why": why,
                          "decay_tier": tier_for(imp)}, ensure_ascii=False))
        return 0

    con = connect(args.db) if args.db else connect()
    w = Writer(con)

    if args.add:
        mid = w.add(args.add, args.scope, args.kind or ("note" if args.scope == "notes" else "fact"),
                    args.source, args.importance)
        print(json.dumps({"id": mid, "rec": w.get(mid)}, ensure_ascii=False))
    if args.rebuild_edges:
        print(json.dumps(w.rebuild_edges(), ensure_ascii=False))
    if args.stats or not (args.add or args.rebuild_edges):
        print(json.dumps(w.stats(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
