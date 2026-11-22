# -*- coding: utf-8 -*-
"""memex M6 接入层：stdio MCP（9 个工具）+ HTTP + 共用实现。

铁律遵循：
- 只依赖标准库 + mcp；没装 mcp 时 HTTP 侧仍可用。
- 不删任何记忆行：forget 只把 status 置 archived，可逆。
- 只写自己的文件：本文件 + http.py + plugin/。recall.py / db.py 只 import 不改。
- 中文全链路 UTF-8，返回给模型的一律是可读 JSON 文本。

给 M0/M1/M3 留的接口位：
- Hit 目前还是 dict（types.py 未落地）。`_hit_field()` 同时吃 dict 和 dataclass，
  M0 把 Hit 提出去之后不用改这里。
- remember 优先调用 M3 的 write.add；文件还不存在时走本文件的兜底实现。
- trace 明细单独落 recall_hits 表（本文件建，不改 db.py 的 schema）。

用法：
    $py -m memex.server_mcp                 # stdio MCP（给 DSH / 其他 agent）
    $py -m memex.server_mcp --db <path>     # 指定库（自检用临时库，别写脏真库）
    $py -m memex.server_mcp --selfcheck     # 自检：握手 + 工具 + recall→feedback 回路
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python memex/server_mcp.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from . import core as coremod  # noqa: E402
from .db import DB_PATH, connect  # noqa: E402
from .embed import Embedder  # noqa: E402
from .recall import DEFAULT_POLICY, Recall  # noqa: E402

DB_ENV = "MEMEX_DB"
SCOPES = ("facts", "notes")

# trace(trace_id) 要能回答「那次召回到了什么、每条哪个臂第几名」。
# recall_log.hits 只存 id（M1 的地盘），明细单独一张表，不动 db.py。
HITS_DDL = """
CREATE TABLE IF NOT EXISTS recall_hits (
    trace_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    id TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    dense_sim REAL NOT NULL DEFAULT 0,
    arms TEXT NOT NULL DEFAULT '[]',
    cited INTEGER NOT NULL DEFAULT 0,
    rerank REAL,
    PRIMARY KEY (trace_id, rank)
);
"""


def _ensure_hits_cols(con) -> None:
    """老库里的 recall_hits 没有 rerank 列时补上（只动 M6 自己这张表）。"""
    try:
        cols = {r["name"] for r in con.execute("PRAGMA table_info(recall_hits)")}
        if "rerank" not in cols:
            con.execute("ALTER TABLE recall_hits ADD COLUMN rerank REAL")
            con.commit()
    except Exception:
        pass


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _db_path(arg: str | None = None) -> Path:
    return Path(arg or os.environ.get(DB_ENV) or DB_PATH)


_ROOT = Path(__file__).resolve().parent.parent


def _rel(p: Path | str) -> str:
    """对外只报相对路径（M6 加固要求：响应里不许出现本机绝对路径）。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(_ROOT).as_posix()
    except Exception:
        return p.name


def _hit_field(h, name: str, default=None):
    """同时吃 dict（v0）和 dataclass（M0 的 Hit）。"""
    if isinstance(h, dict):
        return h.get(name, default)
    return getattr(h, name, default)


def _err(msg: str, **extra) -> dict:
    return {"ok": False, "error": msg, **extra}


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def _connect_shared(path: Path) -> sqlite3.Connection:
    """跨线程共用的连接（MCP 宿主把同步工具放工作线程跑）。

    语义与 db.connect 一致（含 M0 的 migrate），只多重开 check_same_thread；
    所有访问都在 ENGINE.lock 里串行，所以关掉同线程断言是安全的。
    """
    from .db import SCHEMA, migrate
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    try:
        migrate(con)
    except Exception:
        pass
    return con


# --------------------------------------------------------------------------
# 引擎（单进程内共享：MCP 进程 / HTTP 进程各一份）
# --------------------------------------------------------------------------
class Engine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.con: sqlite3.Connection | None = None
        self.embedder: Embedder | None = None
        self.recall: Recall | None = None
        self.db_path: Path | None = None
        self.cold_start_ms: float | None = None
        self._policy_stamp: int | None = None
        self.policy_reloads = 0

    # ---- 现役策略 ----
    @staticmethod
    def _stamp() -> int:
        try:
            from .policy_io import active_path
            p = active_path()
            return p.stat().st_mtime_ns if p.exists() else 0
        except Exception:
            return 0

    def refresh_policy(self) -> bool:
        """`data/active_policy.json` 变了就重建召回器（M5 采纳新策略后不必重启服务）。

        MCP 进程是长活的：不重读的话，面板/模型看到的还是进程启动那一刻的参数，
        「默认走 active_policy」就成了一句空话。
        """
        with self._lock:
            if self.recall is None:
                return False
            s = self._stamp()
            if s == self._policy_stamp:
                return False
            cls = type(self.recall)          # 保留 RerankedRecall / Recall 的具体实现
            try:
                self.recall = cls(self.con, self.embedder)  # type: ignore[arg-type]
                self._policy_stamp = s
                self.policy_reloads += 1
                return True
            except Exception:
                return False

    def boot(self, db_path: Path | None = None) -> Recall:
        with self._lock:
            p = Path(db_path or _db_path())
            if self.con is not None and self.db_path == p:
                return self.recall  # type: ignore[return-value]
            t0 = time.perf_counter()
            # MCP 宿主会把同步工具丢到工作线程执行，连接必须跨线程可用；
            # 所有访问都在 ENGINE.lock 里串行，所以关掉同线程断言是安全的。
            self.con = _connect_shared(p)
            self.con.executescript(HITS_DDL)
            self.con.commit()
            _ensure_hits_cols(self.con)
            self.embedder = Embedder(self.con)
            # M1 已把重排钩子做进 `Recall.recall()`（policy["use_rerank"] 说了算），
            # 这里必须用**裸 Recall**：再接 M2 的 `RerankedRecall` 会二次重排（分被算两遍、慢一倍）。
            self.recall = Recall(self.con, self.embedder)
            self.db_path = p
            self._policy_stamp = self._stamp()
            self.cold_start_ms = round((time.perf_counter() - t0) * 1000, 1)
            return self.recall

    def invalidate(self) -> None:
        """写库之后调用：让 Recall 下次重新载入 memories（否则激活臂看旧值）。"""
        with self._lock:
            if self.recall is not None:
                self.recall._loaded = False

    @property
    def lock(self) -> threading.RLock:
        return self._lock


ENGINE = Engine()


# --------------------------------------------------------------------------
# 业务操作（MCP 与 HTTP 共用同一份，别复制逻辑）
# --------------------------------------------------------------------------
def op_recall(q: str = "", k: int = 5, scope: str = "") -> dict:
    q = (q or "").strip()
    if not q:
        return _err("q 不能为空：传一句自然语言问题，例 q=\"TTS 参考音频用的哪个音色\"")
    scope = (scope or "").strip()
    if scope and scope not in SCOPES:
        return _err(f"scope 只能是 {SCOPES} 之一或留空表示不限，收到 {scope!r}")
    try:
        k = max(1, min(int(k or 5), 20))
    except (TypeError, ValueError):
        return _err("k 必须是 1~20 的整数")

    with ENGINE.lock:
        try:
            r = ENGINE.boot()
            if ENGINE.refresh_policy():      # active_policy.json 被 M5 改了 → 用新参，不重启
                r = ENGINE.recall
            hits = r.recall(q, k=k, scope=scope or None, write_log=True)
        except Exception as e:  # 网络/向量化失败也要说清楚的、能照做的中文
            return _err(f"召回失败：{type(e).__name__}: {e}",
                        hint="检查 MEMEX_EMBED_BASE 端点与网络连通；"
                             "embed_cache 命中的 query 不需要网络")
        if not hits:
            return _ok(trace_id="", n=0, hits=[],
                       note="没有相关记忆。按「未命中」如实回答，别编。")
        trace_id = _hit_field(hits[0], "trace_id", "") or ""
        out = []
        rows = []
        for rank, h in enumerate(hits, 1):
            item = {
                "id": _hit_field(h, "id"),
                "scope": _hit_field(h, "scope"),
                "kind": _hit_field(h, "kind"),
                "text": (_hit_field(h, "text") or "").strip(),
                "score": _hit_field(h, "score"),
                "dense_sim": _hit_field(h, "dense_sim"),
                "arms": _hit_field(h, "arms") or [],
                "importance": _hit_field(h, "importance"),
                "cited": 0,   # 0=召回但未回写引用；feedback 后变 1（面板/复放用）
            }
            # M2 开了 use_rerank 时才有这个字段；关着时整键不出现（别让模型把 0 当分数用）
            rs = _hit_field(h, "rerank_score", None)
            if rs is not None:
                item["rerank_score"] = round(float(rs), 4)
            out.append(item)
            rows.append((trace_id, rank, _hit_field(h, "id"),
                         float(_hit_field(h, "score", 0) or 0),
                         float(_hit_field(h, "dense_sim", 0) or 0),
                         json.dumps(_hit_field(h, "arms") or [], ensure_ascii=False),
                         None if rs is None else float(rs)))
        try:
            con = ENGINE.con
            assert con is not None
            # 兜底：某些召回实现（如 M2 的 RerankedRecall）只 UPDATE 不 INSERT recall_log，
            # 结果 trace/feedback 找不到这张单。这里按契约补一行，幂等（已有就不动）。
            if trace_id and not con.execute(
                    "SELECT 1 FROM recall_log WHERE trace_id=?", (trace_id,)).fetchone():
                con.execute(
                    "INSERT OR REPLACE INTO recall_log(trace_id, ts, query, policy_version, hits)"
                    " VALUES(?,?,?,?,?)",
                    (trace_id, int(time.time()), q, None,
                     json.dumps([_hit_field(h, "id") for h in hits], ensure_ascii=False)))
            con.executemany(  # type: ignore[union-attr]
                "INSERT OR REPLACE INTO recall_hits(trace_id, rank, id, score, dense_sim, arms, cited, rerank)"
                " VALUES(?,?,?,?,?,?,0,?)", rows)
            con.commit()  # type: ignore[union-attr]
        except Exception:
            pass  # 明细写失败不影响召回结果；trace 会退化成只有 id 列表
        return _ok(trace_id=trace_id, n=len(out), hits=out,
                   note="用完后调 feedback(trace_id, hit_ids, used) 回写「哪条真用上了」，"
                        "这是激活衰减唯一的真实信号。")


def op_remember(text: str = "", scope: str = "facts", kind: str = "fact",
                importance: str | float = "", source: str = "dsh",
                meta: str = "") -> dict:
    text = (text or "").strip()
    if not text:
        return _err("text 不能为空：传要记住的那句话本身")
    source = (source or "").strip()
    if not source:
        return _err("source 不能为空：写调用方标识（dsh / hermes / doubao），不许匿名写入")
    scope = (scope or "facts").strip()
    if scope not in SCOPES:
        return _err(f"scope 只能是 {SCOPES} 之一，收到 {scope!r}")
    kind = (kind or "fact").strip()

    imp: float | None = None
    if str(importance).strip() != "":
        try:
            imp = max(0.0, min(1.0, float(importance)))
        except ValueError:
            return _err(f"importance 要是 0~1 的数字，收到 {importance!r}")
    meta_obj = None
    if (meta or "").strip():
        try:
            meta_obj = json.loads(meta)
        except json.JSONDecodeError as e:
            return _err(f"meta 要是 JSON 对象字符串，解析失败：{e}")

    # 写入与治理一律走 M3 的 write.py（去重合并 / 衰减分档 / 连边都在里面），
    # M6 只负责把 MCP/HTTP 的入参翻译过去，不自己写一份规则。
    with ENGINE.lock:
        try:
            ENGINE.boot()
            con = ENGINE.con
            assert con is not None
            from .write import Writer
            seen = {r["id"] for r in con.execute(
                "SELECT id FROM memories WHERE scope=? AND status='active'", (scope,))}
            mid = Writer(con, ENGINE.embedder).add(
                text=text, scope=scope, kind=kind, source=source,
                importance=imp, meta=meta_obj)
        except ValueError as e:
            return _err(f"写入被拒：{e}")
        except Exception as e:
            return _err(f"写入失败：{type(e).__name__}: {e}",
                        hint="检查 .siliconflow_key 与外网（新文本要算向量）")
        ENGINE.invalidate()
        rec = con.execute("SELECT importance, decay_tier, status, source FROM memories"
                          " WHERE id=?", (mid,)).fetchone()
        action = "merged" if mid in seen else "inserted"
        return _ok(action=action, id=mid, scope=scope, kind=kind,
                   importance=float(rec["importance"]) if rec else imp,
                   decay_tier=int(rec["decay_tier"]) if rec else None,
                   source=(rec["source"] if rec else source),
                   note=("已有 cos>0.92 的同义记忆，已合并（文本取信息量大的、importance 取 max）"
                         if action == "merged" else "已入库，下一次 recall 就能召回"))


def op_forget(id: str = "", reason: str = "") -> dict:
    mid = (id or "").strip()
    if not mid:
        return _err("id 不能为空：用 recall 返回的 id（32 位 hex，notes 会带 note: 前缀）")
    with ENGINE.lock:
        ENGINE.boot()
        con = ENGINE.con
        assert con is not None
        row = con.execute("SELECT id, status, text FROM memories WHERE id=?", (mid,)).fetchone()
        if row is None:
            return _err(f"没有 id={mid} 这条记忆。先 recall 拿到真实 id 再忘。")
        if row["status"] == "archived":
            return _ok(id=mid, status="archived", changed=False, note="这条已经是 archived，无需再操作")
        # 只归档不删除（治理铁律：不许删行）
        con.execute("UPDATE memories SET status='archived' WHERE id=?", (mid,))
        con.commit()
        ENGINE.invalidate()
        return _ok(id=mid, status="archived", changed=True, reason=reason,
                   note="已归档，召回不再返回。要恢复：UPDATE memories SET status='active' WHERE id=?",
                   preview=(row["text"] or "")[:60])


def op_memory_get(id: str = "") -> dict:
    """读一条记忆的完整字段（面板编辑用）。"""
    mid = (id or "").strip()
    if not mid:
        return _err("id 不能为空")
    with ENGINE.lock:
        ENGINE.boot()
        con = ENGINE.con
        assert con is not None
        row = con.execute(
            "SELECT id, scope, kind, text, importance, status, source, created_at,"
            " last_access, pinned, core_slot, superseded_by, superseded_by_id"
            " FROM memories WHERE id=?", (mid,)).fetchone()
        if row is None:
            return _err(f"没有 id={mid} 这条记忆")
        return _ok(memory=dict(row))


def op_memory_update(id: str = "", text: str = "", importance=None) -> dict:
    """改文本（重新向量化）或改 importance。只改这两样，不动 status。"""
    mid = (id or "").strip()
    if not mid:
        return _err("id 不能为空")
    body = (text or "").strip()
    if not body:
        return _err("text 不能为空：要改文本就传新文本，只改 importance 就别传 text")
    if len(body) > 4000:
        return _err("text 太长（>4000 字），先拆条再写")
    imp = None
    if importance not in (None, ""):
        try:
            imp = float(importance)
        except (TypeError, ValueError):
            return _err("importance 要是 0~1 的数")
    from .write import Writer  # 延迟 import：MCP 冷启动时 write.py 依赖较重
    with ENGINE.lock:
        ENGINE.boot()
        con = ENGINE.con
        assert con is not None
        if not Writer(con, ENGINE.embedder).update(mid, body, imp):
            return _err(f"没有 id={mid} 这条记忆")
        ENGINE.invalidate()
        row = con.execute(
            "SELECT id, scope, kind, text, importance, status, pinned, core_slot"
            " FROM memories WHERE id=?", (mid,)).fetchone()
        return _ok(memory=dict(row), note="已改并重新向量化，下一次 recall 生效")


def op_memory_forget(id: str = "", reason: str = "") -> dict:
    """归档（不删行），面板删除走它。"""
    return op_forget(id=id, reason=reason)


def op_stats() -> dict:
    with ENGINE.lock:
        p = _db_path()
        con = connect(p)
        try:
            def one(sql, *a):
                r = con.execute(sql, a).fetchone()
                return r[0] if r else 0

            by_scope = {r["scope"]: r["c"] for r in con.execute(
                "SELECT scope, count(*) c FROM memories WHERE status='active' GROUP BY scope")}
            by_status = {r["status"]: r["c"] for r in con.execute(
                "SELECT status, count(*) c FROM memories GROUP BY status")}
            today = int(time.time()) - 86400
            n_recall = one("SELECT count(*) FROM recall_log WHERE ts>?", today)
            n_cited = one("SELECT count(*) FROM recall_log WHERE ts>? AND cited IS NOT NULL"
                          " AND cited NOT IN ('', '[]')", today)
            # 策略以 M5 的 policy_io 为准（读不到就回落 DEFAULT_POLICY，它自己保证不抛）
            from .policy_io import load_active, active_path
            policy = {**DEFAULT_POLICY, **load_active()}
            ap = active_path()
            src = _rel(ap) if ap.exists() else "DEFAULT_POLICY"
            return _ok(
                db=_rel(p), db_mb=round(p.stat().st_size / 1e6, 1) if p.exists() else 0,
                memories={"active": sum(by_scope.values()), "by_scope": by_scope, "by_status": by_status},
                edges=one("SELECT count(*) FROM edges"),
                embed_cache=one("SELECT count(*) FROM embed_cache"),
                recall_24h={"total": n_recall, "with_cited": n_cited},
                trace_detail=one("SELECT count(*) FROM recall_hits"),
                policy={"source": src, **{k: policy[k] for k in policy}},
                engine={"loaded": bool(ENGINE.recall and ENGINE.recall._loaded),
                        "cold_start_ms": ENGINE.cold_start_ms,
                        "policy_reloads": ENGINE.policy_reloads,
                        "impl": type(ENGINE.recall).__name__ if ENGINE.recall else None},
            )
        finally:
            con.close()


def op_core(budget: int = 1200) -> dict:
    """渲染 core 块（常驻上下文的那一小块）。不写库。"""
    with ENGINE.lock:
        p = _db_path()
        con = connect(p)
        try:
            out = coremod.render_core(con, budget=int(budget or 1200))
            out["db"] = _rel(p)
            return out
        finally:
            con.close()


def op_core_pin(id: str = "", slot: str = "other") -> dict:
    with ENGINE.lock:
        con = connect(_db_path())
        try:
            out = coremod.core_pin(con, id, slot)
            if out.get("ok"):
                out["core"] = coremod.render_core(con)
            return out
        finally:
            con.close()


def op_core_unpin(id: str = "") -> dict:
    with ENGINE.lock:
        con = connect(_db_path())
        try:
            out = coremod.core_unpin(con, id)
            if out.get("ok"):
                out["core"] = coremod.render_core(con)
            return out
        finally:
            con.close()


def op_traces(limit: int = 20) -> dict:
    """最近 N 次召回（面板「最近召回」列表用）。"""
    n = max(1, min(int(limit or 20), 100))
    with ENGINE.lock:
        con = connect(_db_path())
        try:
            rows = con.execute(
                "SELECT trace_id, ts, query, hits FROM recall_log ORDER BY ts DESC LIMIT ?",
                (n,)).fetchall()
            out = []
            for r in rows:
                try:
                    ids = [h["id"] if isinstance(h, dict) else h for h in json.loads(r["hits"] or "[]")]
                except Exception:
                    ids = []
                cited = con.execute("SELECT COUNT(*) FROM recall_hits WHERE trace_id=? AND cited=1",
                                    (r["trace_id"],)).fetchone()[0]
                out.append({"trace_id": r["trace_id"], "ts": r["ts"],
                            "ts_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])),
                            "query": r["query"], "n": len(ids), "cited_n": cited})
            return _ok(n=len(out), traces=out)
        finally:
            con.close()


def op_trace(trace_id: str = "") -> dict:
    tid = (trace_id or "").strip()
    if not tid:
        return _err("trace_id 不能为空：recall 的返回里带 trace_id（32 位 hex）")
    with ENGINE.lock:
        con = connect(_db_path())
        _ensure_hits_cols(con)  # 老库可能还没有 rerank 列
        try:
            row = con.execute("SELECT * FROM recall_log WHERE trace_id=?", (tid,)).fetchone()
            if row is None:
                return _err(f"没有 trace_id={tid} 的记录。先调 recall，它会返回 trace_id；"
                            "只查最近的召回，历史不会无限保留。")
            try:
                hits = json.loads(row["hits"] or "[]")
            except Exception:
                hits = []
            detail = []
            ids = [h["id"] if isinstance(h, dict) else h for h in hits]
            if ids:
                q = "SELECT h.rank, h.id, h.score, h.dense_sim, h.arms, h.cited, h.rerank, m.text, m.scope, m.kind" \
                    " FROM recall_hits h LEFT JOIN memories m ON m.id=h.id" \
                    " WHERE h.trace_id=? ORDER BY h.rank"
                for r in con.execute(q, (tid,)):
                    try:
                        arms = json.loads(r["arms"] or "[]")
                    except Exception:
                        arms = []
                    item = {"rank": r["rank"], "id": r["id"],
                            "arms": arms, "score": r["score"],
                            "dense_sim": r["dense_sim"], "cited": bool(r["cited"]),
                            "scope": r["scope"], "kind": r["kind"],
                            "text": (r["text"] or "")[:200]}
                    if r["rerank"] is not None:
                        item["rerank_score"] = round(float(r["rerank"]), 4)
                    detail.append(item)
            if not detail:  # 明细缺失（M1 直接调 recall 写日志的情况）
                detail = [{"rank": i + 1, "id": i_, "arms": [], "cited": False}
                          for i, i_ in enumerate(ids)]
            try:
                cited = json.loads(row["cited"] or "[]")
            except Exception:
                cited = []
            return _ok(trace_id=tid, ts=row["ts"],
                       ts_text=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"])),
                       query=row["query"], policy_version=row["policy_version"],
                       n=len(ids), cited=cited, hits=detail)
        finally:
            con.close()


def op_feedback(trace_id: str = "", hit_ids: list[str] | None = None, used: bool = True) -> dict:
    tid = (trace_id or "").strip()
    if not tid:
        return _err("trace_id 不能为空：recall 返回里带")
    ids = [str(i).strip() for i in (hit_ids or []) if str(i).strip()]
    with ENGINE.lock:
        ENGINE.boot()
        con = ENGINE.con
        assert con is not None
        row = con.execute("SELECT hits FROM recall_log WHERE trace_id=?", (tid,)).fetchone()
        if row is None:
            return _err(f"没有 trace_id={tid} 的记录，feedback 无法落库。先 recall 再回写。")
        try:
            logged = json.loads(row["hits"] or "[]")
        except Exception:
            logged = []
        logged = [h["id"] if isinstance(h, dict) else h for h in logged]
        if not ids:
            ids = logged  # 不点名 = 整批都算用上了（最常见：把这些片段拼进了回答）
        unknown = [i for i in ids if i not in logged]
        if unknown:
            return _err("hit_ids 里有本次召回之外的 id，拒写（防止张冠李戴）",
                        unknown=unknown[:5], allowed=logged[:20])
        now = int(time.time())
        con.execute("UPDATE recall_log SET cited=? WHERE trace_id=?",
                    (json.dumps(ids if used else [], ensure_ascii=False), tid))
        if used:
            for i in ids:
                con.execute("UPDATE memories SET access_count=access_count+1, last_access=?"
                            " WHERE id=?", (now, i))
        placed = ",".join("?" * len(ids)) or "''"
        if ids:
            con.execute(f"UPDATE recall_hits SET cited=? WHERE trace_id=? AND id IN ({placed})",
                        (1 if used else 0, tid, *ids))
        con.commit()
        ENGINE.invalidate()
        return _ok(trace_id=tid, used=bool(used), n=len(ids),
                   note=("已记下这些命中被采用，access_count+1、last_access 已更新（激活臂下次生效）"
                         if used else "已标记为未采用，不增计数"))


# --------------------------------------------------------------------------
# MCP 6 个工具
# --------------------------------------------------------------------------
mcp = MCPServer("memex")


@mcp.tool(
    name="recall",
    title="召回共享记忆",
    description=(
        "语义检索本机长期记忆库（2176 条事实+笔记，bge-m3 三臂融合）。\n"
        "什么时候调：回答任何「以前怎么配的 / 上次说到哪 / 有没有现成方案 / 主人的偏好」之前，"
        "先调一次，别猜。一次对话调 1~2 次即可（每次 5~30 ms，最多 20 条）。\n"
        "返回：trace_id + hits[]（id/scope/kind/text/score/dense_sim/arms/importance）。"
        "hit_ids 是后续 feedback 的入参。\n"
        "没命中会返回 n=0：这时如实说「记忆里没有」，不要编造。"
    ),
)
def t_recall(q: str, k: int = 5, scope: str = "") -> str:
    """从共享记忆召回。

    Args:
        q: 一句自然语言问题，例 "TTS 参考音频用的哪个音色"。
        k: 返回条数，1~20，默认 5。
        scope: "facts"（提炼事实）/ "notes"（笔记块）/ 留空=不限。
    """
    return _j(op_recall(q=q, k=k, scope=scope))


@mcp.tool(
    name="remember",
    title="写入一条记忆",
    description=(
        "把一条值得长期记住的结论写进共享记忆（去重：同 scope 下 cos>0.92 会合并而不是新增）。\n"
        "什么时候调：主人明确说了长期约定/偏好、或得出可复用的事实结论时，主动写一条，"
        "别把过程闲聊塞进来。\n"
        "importance 不传会按规则估（偏好 0.9 / 技术事实 0.8 / 步骤 0.6 / 闲聊 0.4）。\n"
        "返回 action=inserted|merged 与 id。source 必填调用方标识（dsh/hermes/doubao）。"
    ),
)
def t_remember(text: str, scope: str = "facts", kind: str = "fact",
               importance: str = "", source: str = "dsh", meta: str = "") -> str:
    """写入记忆。

    Args:
        text: 要记住的那句话本身（一句话讲清，别带对话原文）。
        scope: "facts"（默认，结论级）/ "notes"（原文笔记块）。
        kind: fact / note / profile / lesson / rule，默认 fact。
        importance: 0~1 数字字符串；留空按规则估。
        source: 调用方标识，默认 dsh。
        meta: 可选 JSON 字符串，附加字段（如 {"topic":"TTS"}）。
    """
    return _j(op_remember(text=text, scope=scope, kind=kind, importance=importance,
                          source=source, meta=meta))


@mcp.tool(
    name="forget",
    title="归档一条记忆",
    description=(
        "把一条记忆归档（status=archived），召回不再返回。不删行，可恢复。\n"
        "什么时候调：主人说「这条记错了 / 过时了」，先用 recall 拿到 id 再调。\n"
        "返回 status 与恢复用的 SQL。id 不存在会明确报错，别重复试。"
    ),
)
def t_forget(id: str, reason: str = "") -> str:
    """归档记忆（不删除）。

    Args:
        id: recall 返回的 id。
        reason: 为什么归档（写进结果便于审计）。
    """
    return _j(op_forget(id=id, reason=reason))


@mcp.tool(
    name="stats",
    title="库状态",
    description=(
        "看库有多大、多少条、近 24 小时召回与被引用次数、当前策略参数。\n"
        "什么时候调：怀疑记忆没生效 / 要汇报「记忆库健康度」时。不触发向量化，很便宜。"
    ),
)
def t_stats() -> str:
    """返回库统计与当前策略。"""
    return _j(op_stats())


@mcp.tool(
    name="trace",
    title="回看某次召回",
    description=(
        "按 trace_id 回看那一次召回的完整过程：查询原文、每条命中的排名/各臂名次/dense_sim/是否被引用。\n"
        "什么时候调：主人问「你刚才为什么拿了这条」，或排查为什么某条没召回。trace_id 来自 recall。"
    ),
)
def t_trace(trace_id: str) -> str:
    """回看召回明细。

    Args:
        trace_id: recall 返回的 32 位 hex。
    """
    return _j(op_trace(trace_id=trace_id))


@mcp.tool(
    name="feedback",
    title="回写用了哪条",
    description=(
        "告诉记忆库这次召回里哪几条真的用上了（会用进激活衰减：access_count+1、last_access=now）。\n"
        "什么时候调：每次用 recall 的结果作答之后，紧接着调一次，别攒着。\n"
        "used=true 表示 hit_ids 被采用（不传 hit_ids 视为整批采用）；used=false 表示没用上，只记录不计数。\n"
        "hit_ids 必须是本次召回返回过的 id，传别的会被拒（防止张冠李戴）。"
    ),
)
def t_feedback(trace_id: str, hit_ids: list[str], used: bool = True) -> str:
    """回写引用反馈。

    Args:
        trace_id: recall 返回的 trace_id。
        hit_ids: 本次采用（或未采用）的 id 列表，来自 recall 的 hits[].id。
        used: true=采用了；false=没用上。
    """
    return _j(op_feedback(trace_id=trace_id, hit_ids=hit_ids, used=used))


# --------------------------------------------------------------------------
# 自检：真握手 + 6 工具 + recall→feedback→stats 回路（默认在临时库副本上跑）
# --------------------------------------------------------------------------
@mcp.tool(
    name="core",
    title="核心记忆块（开局常驻）",
    description=(
        "返回 memex 的 core 块：身份/规则/偏好/项目四类里被 pin 的记忆，按 1200 字预算渲染。\n"
        "什么时候调：会话开局、或要确认「我现在该记得什么」时，一次就够；"
        "内容带 version（内容 hash），同一 version 不必重复注入。\n"
        "超预算的条目在 dropped 里列出，不静默截断；返回 over_budget=true 表示 pin 的太多，需要取舍。"
    ),
)
def t_core(budget: int = 1200) -> str:
    """渲染 core 块。

    Args:
        budget: 字数上限，默认 1200（约 900 tokens）。
    """
    return _j(op_core(budget=budget))


@mcp.tool(
    name="core_pin",
    title="把一条记忆 pin 进 core",
    description=(
        "把 id 指定的 active 记忆放进 core 块的某个分区（identity/rule/preference/project/other）。\n"
        "什么时候调：用户说「这个以后一直要记得 / 这条放常驻」，或你自己确认了不可遗忘的硬约束。"
        "pin 是治理动作，不要批量自动 pin。"
    ),
)
def t_core_pin(id: str, slot: str = "other") -> str:
    """pin 一条记忆进 core。

    Args:
        id: recall 返回的 32 位 hex id。
        slot: identity / rule / preference / project / other。
    """
    return _j(op_core_pin(id=id, slot=slot))


@mcp.tool(
    name="core_unpin",
    title="把一条记忆移出 core",
    description="把 id 指定的记忆移出 core 块（记忆本身不受影响，仍可被 recall 召回）。",
)
def t_core_unpin(id: str) -> str:
    """unpin 一条记忆。

    Args:
        id: 记忆 id。
    """
    return _j(op_core_unpin(id=id))


def _copy_db(dst: Path) -> Path:
    src = _db_path()
    shutil.copy2(src, dst)
    return dst


async def _selfcheck_async(db: Path) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    py = sys.executable
    params = StdioServerParameters(
        command=py,
        args=["-m", "memex.server_mcp", "--db", str(db)],
        env={"PYTHONUTF8": "1", "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    out: dict = {"db": str(db)}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            init = await s.initialize()
            out["server"] = getattr(init.server_info, "name", "?")
            tools = (await s.list_tools()).tools
            out["tools"] = sorted(t.name for t in tools)
            out["tool_count"] = len(tools)

            r1 = await s.call_tool("recall", {"q": "自拍出人类耳朵怎么修", "k": 3})
            d1 = _tool_json(r1)
            out["recall"] = {"ok": d1.get("ok"), "n": d1.get("n"),
                             "trace_id": d1.get("trace_id", "")[:8]}
            tid = d1.get("trace_id") or ""
            ids = [h["id"] for h in d1.get("hits", [])]

            if tid and ids:
                st0 = _tool_json(await s.call_tool("stats", {}))
                before = _count_of(db, ids[0])
                d2 = _tool_json(await s.call_tool("feedback",
                                                 {"trace_id": tid, "hit_ids": ids, "used": True}))
                st1 = _tool_json(await s.call_tool("stats", {}))
                after = _count_of(db, ids[0])
                out["feedback"] = {"ok": d2.get("ok"), "n": d2.get("n")}
                out["access_count_delta"] = after - before
                d3 = _tool_json(await s.call_tool("trace", {"trace_id": tid}))
                out["trace"] = {"ok": d3.get("ok"), "cited": d3.get("cited", [])[:2],
                                "n": d3.get("n")}

            d4 = _tool_json(await s.call_tool("remember", {
                "text": "自检写入：memex M6 接入层自检记录，跑完即弃",
                "scope": "facts", "kind": "lesson", "importance": "0.45",
                "source": "selfcheck"}))
            out["remember"] = {"ok": d4.get("ok"), "action": d4.get("action"),
                              "id": (d4.get("id") or "")[:8]}
            newid = d4.get("id") or ""
            d5 = _tool_json(await s.call_tool("remember", {  # 二次写入应合并或高相似
                "text": "自检写入：memex M6 接入层自检记录，跑完即弃",
                "scope": "facts", "kind": "lesson", "source": "selfcheck"}))
            out["remember_twice"] = {"action": d5.get("action"), "same_id": d5.get("id") == newid}
            if newid:
                d6 = _tool_json(await s.call_tool("forget", {"id": newid, "reason": "selfcheck"}))
                out["forget"] = {"ok": d6.get("ok"), "status": d6.get("status")}
            d7 = _tool_json(await s.call_tool("recall", {"q": "随便问一个库里没有的东西 zzzqqq", "k": 2}))
            out["negative"] = {"n": d7.get("n")}
            d8 = _tool_json(await s.call_tool("remember", {"text": " ", "source": "selfcheck"}))
            out["empty_text_rejected"] = d8.get("ok") is False
    return out


def _tool_json(res) -> dict:
    """把 MCP tool 结果还原成 dict。优先 structuredContent，其次解析文本。"""
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict):
        # MCPServer 对 str 返回会包一层 {"result": "..."}
        if set(sc) == {"result"} and isinstance(sc["result"], str):
            try:
                return json.loads(sc["result"])
            except Exception:
                return {"raw": sc["result"]}
        return sc
    for c in getattr(res, "content", []) or []:
        t = getattr(c, "text", None)
        if t:
            try:
                return json.loads(t)
            except Exception:
                return {"raw": t}
    return {}


def _count_of(db: Path, mid: str) -> int:
    con = connect(db)
    try:
        r = con.execute("SELECT access_count FROM memories WHERE id=?", (mid,)).fetchone()
        return int(r[0]) if r else -1
    finally:
        con.close()


def selfcheck() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="memex-selfcheck-"))
    db = _copy_db(scratch / "memex.db")
    import anyio
    try:
        result = anyio.run(_selfcheck_async, db)
    finally:
        pass
    print(_j(result))
    ok = (result.get("tool_count") == 6
          and result.get("recall", {}).get("ok")
          and result.get("feedback", {}).get("ok")
          and result.get("access_count_delta", 0) == 1
          and result.get("remember", {}).get("action") == "inserted"
          and result.get("forget", {}).get("status") == "archived"
          and result.get("empty_text_rejected"))
    shutil.rmtree(scratch, ignore_errors=True)
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="memex MCP 接入层（stdio）")
    ap.add_argument("--db", default=None, help="库路径（默认 data/memex.db 或 $MEMEX_DB）")
    ap.add_argument("--selfcheck", action="store_true", help="自检后退出（临时库副本，不写脏真库）")
    args = ap.parse_args(argv)

    if args.db:
        os.environ[DB_ENV] = str(Path(args.db).resolve())
    if args.selfcheck:
        return selfcheck()
    ENGINE.boot()
    mcp.run()  # stdio
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
