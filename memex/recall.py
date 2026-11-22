# -*- coding: utf-8 -*-
"""四臂召回：dense / bm25 / activation / graph + RRF 融合 + MMR 去重。

权重放在 policy.params，可版本化；演化就是改这里的数并跑 eval 比较。
臂可单独开关：recall(..., arms=["dense","bm25"]) 或 policy["arms"]。
graph 臂依赖 edges 表；没有边时自动返回空（不报错），由 M3 负责产边。
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field, replace

import numpy as np

from .db import activation, blob_to_vec
from .policy_io import load_active
from .types import Hit

DEFAULT_POLICY = {
    # 2026-11-22 实测：dense .70 / bm25 .25 / graph .05 时 dev 73.8/87.5 最佳；
    # act 与 query 无关（加进来 Top1 掉到 70.0），默认关闭。
    "w_dense": 0.70,
    "w_bm25": 0.25,
    "w_act": 0.00,
    "w_graph": 0.05,
    "rrf_k": 60,
    "mmr_lambda": 0.70,
    "arm_dense": 40,
    "arm_bm25": 40,
    "arm_act": 20,
    "arm_graph": 120,
    "graph_seeds": 18,
    "graph_hops": 2,
    "graph_alpha": 1.0,
    "graph_gamma": 0.55,
    "arms": ["dense", "bm25"],
    # 默认不重排：没有重排服务也能跑。配好 HTTP 重排端点后把 use_rerank 设 true。
    "use_rerank": False,
    # 两路召回：全库查询时按 scope 分别召回，再按 scope_boost 放大后融合。
    # 动因：facts 数量是 notes 的 2.8 倍，且笔记块正文长（700~900 字），
    # 一次全库打分时笔记的 dense_sim 系统性偏低 → notes Top1 掉到 57/47。
    # 2026-11-22 12:4x 实测：RRF 口径下 dev 72.5/93.3@1.05、holdout 73.3/86.7（无增益）；
    # 按 dense_sim 排序能到 77.5 但 holdout 掉到 63.3 —— 是过拟合，默认关，保留开关做实验。
    "two_pass": False,
    "scope_boost": {"facts": 1.0, "notes": 1.10},
    "two_pass_per": 15,
    # 父块回溯：命中细块（260 字）时，把同路径的粗块（500 字）正文一起返回。
    # 动因：v2 上 notes/semantic 只有 25~30%，问题多是「为什么这么设计」，答案跨两段。
    # 取值 None（关）/ "append"（细块+粗块）/ "replace"（只回粗块）。
    # 结果层级：把「块」合并回「笔记」再返回。
    # 动因：v2 notes/semantic 30% 里，约 7~8 条是「正确块进了 pool 20、最后排序掉到第 2」，
    # 而重排精度路线已关闭（bge/30B/0.6B/4B 全试过）。同一篇笔记多块分数取 max 合并后，
    # 正确笔记自然浮上来，且不依赖重排器。
    "note_promote": False,
    "note_window": 1,          # 命中块前后各带几个邻块（拼出的正文）
    "note_max_chars": 900,     # 合并后正文上限
    "note_agg": "max",         # max | sum
    "parent_backfill": None,
    "parent_max_chars": 700,
    # 问题空间索引：note_q 条目命中后回填所属细块正文；关掉则整个索引不参与召回。
    "use_note_q": False,
    "note_q_bm25": False,
    # 问题探针：把 query 和问题条目的 dense 相似度当「轻量交叉编码器」，命中就把对应块塞进候选池
    "note_q_probe": False,
    "note_q_probe_sim": 0.72,
    "w_probe": 0.20,
    "recency_weight": 0.05,
}

_ASCII = re.compile(r"[a-z0-9_\.\-/]+", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_STOP = set("的了是我你他她它吗呢吧啊和与就都也很不有个这那什么怎么为什么可以")


_PREFIXES = ("请问", "帮我看下", "帮我看看", "我想问", "想问下", "想知道", "麻烦问下", "咨询一下")
_TAIL_WORDS = ("呢", "吗", "吧", "啊", "呀", "呗", "么")
_TAIL_PHRASES = (
    "放在哪里了", "放在哪儿了", "放在哪了", "放在什么地方", "放哪里了", "放哪儿了", "放哪了",
    "装在哪里了", "装在哪儿了", "装在哪了", "装哪里了", "装哪儿了", "装哪了",
    "存在哪里了", "存在哪儿了", "在哪里的", "在哪儿了", "在哪里", "在哪儿",
    "是什么", "是哪个", "是哪一个", "叫什么", "叫啥", "是啥",
    "怎么改", "怎么弄", "怎么设置", "怎么配", "怎么做", "怎么办", "咋办",
    "能不能", "可不可以", "是否", "有没有", "对不对",
)


def strip_intent(q: str) -> str:
    """剥掉问句外壳，只留实体词；剥完太短（<3 字）就原样返回。"""
    t = (q or "").strip()
    for w in ("？", "?", "。", "！", "!", "，"):
        t = t.replace(w, "")
    t = t.strip()
    for pre in _PREFIXES:
        if t.startswith(pre):
            t = t[len(pre):].strip()
    changed = True
    while changed and len(t) > 3:
        changed = False
        for ph in _TAIL_PHRASES:
            if t.endswith(ph) and len(t) - len(ph) >= 3:
                t = t[: -len(ph)].strip()
                changed = True
                break
        if changed:
            continue
        for w in _TAIL_WORDS:
            if t.endswith(w) and len(t) - len(w) >= 3:
                t = t[: -len(w)].strip()
                changed = True
                break
    return t if len(t) >= 3 else (q or "").strip()


def tokenize(text: str) -> list[str]:
    """中英混合分词：ASCII 词 + 中文 bigram（无依赖，够用）。"""
    t = (text or "").lower()
    toks = _ASCII.findall(t)
    for run in _CJK.findall(t):
        if len(run) == 1:
            if run not in _STOP:
                toks.append(run)
            continue
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg[0] in _STOP and bg[1] in _STOP:
                continue
            toks.append(bg)
    return toks


@dataclass
class Doc:
    id: str
    scope: str
    kind: str
    text: str
    importance: float
    stability: float
    decay_tier: int
    last_access: int
    created_at: int
    vec: np.ndarray
    toks: list[str] = field(default_factory=list)


class Recall:
    def __init__(self, con, embedder, policy: dict | None = None):
        self.con = con
        self.embedder = embedder
        # 生效顺序：DEFAULT_POLICY ← data/active_policy.json（M5 采纳的现役策略）← 显式传入
        self.policy = {**DEFAULT_POLICY, **load_active(), **(policy or {})}
        self._loaded = False
        self.docs: list[Doc] = []
        self.mat: np.ndarray | None = None
        self.df: dict[str, int] = {}
        self.avgdl = 0.0
        self.by_id: dict[str, int] = {}
        self.nbrs: dict[str, list[tuple[str, float]]] = {}
        # 细块 → 同路径粗块正文（父块回溯用；语义答案常跨块，命中细块时把粗块一起给模型）
        self.parent: dict[str, str] = {}
        # 细块 → 前后邻块的上下文片段（只喂给重排器，返回正文不变；policy.rerank_context 控制）
        self.chunk_ctx: dict[str, str] = {}
        # 细块 → 所属笔记标题（只拼进重排输入；policy.rerank_title_anchor 控制）
        self.chunk_title: dict[str, str] = {}
        # 双时态：as_of=秒级时间戳时按 valid_from/valid_to 取候选（只承诺 facts）
        self.as_of: int | None = None

    # ---------- 索引 ----------
    def load(self, force: bool = False, as_of: int | None = None) -> None:
        if self._loaded and not force and as_of == self.as_of:
            return
        self.as_of = as_of
        if as_of:
            rows = self.con.execute(
                "SELECT id, scope, kind, text, importance, stability, decay_tier, last_access, created_at, vec "
                "FROM memories WHERE scope='facts' AND valid_from<=? AND (valid_to=0 OR valid_to>?)",
                (as_of, as_of)).fetchall()
        else:
            rows = self.con.execute(
                "SELECT id, scope, kind, text, importance, stability, decay_tier, last_access, created_at, vec "
                "FROM memories WHERE status='active'"
            ).fetchall()
        self.docs, vecs = [], []
        df: dict[str, int] = {}
        use_q = bool(self.policy.get("use_note_q", True))
        for r in rows:
            if not use_q and r["kind"] == "note_q":
                continue
            d = Doc(r["id"], r["scope"], r["kind"], r["text"], r["importance"], r["stability"],
                    r["decay_tier"], r["last_access"], r["created_at"], blob_to_vec(r["vec"]))
            d.toks = tokenize(d.text)
            for t in dict.fromkeys(d.toks):
                df[t] = df.get(t, 0) + 1
            self.docs.append(d)
            vecs.append(d.vec)
        self.df = df
        self.mat = np.vstack(vecs) if vecs else np.zeros((0, 1024), dtype=np.float32)
        self.avgdl = (sum(len(d.toks) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        self.by_id = {d.id: i for i, d in enumerate(self.docs)}
        self.q_idx = [i for i, d in enumerate(self.docs) if d.kind == "note_q"]
        self._build_parents()
        if self.policy.get("note_promote"):
            self._build_chunk_seq()
        self._build_chunk_neighbors()
        self._build_chunk_titles()

        # 无向邻接（edges 存一条 a→b，两边都可达）
        nbrs: dict[str, list[tuple[str, float]]] = {}
        for e in self.con.execute("SELECT a, b, weight FROM edges"):
            if e["a"] == e["b"]:
                continue
            nbrs.setdefault(e["a"], []).append((e["b"], float(e["weight"])))
            nbrs.setdefault(e["b"], []).append((e["a"], float(e["weight"])))
        self.nbrs = nbrs
        self._loaded = True

    # ---------- 父块回溯 ----------
    def _build_parents(self) -> None:
        """细块（kind=note_chunk）→ 同路径粗块（kind=note）正文。

        只在 policy 打开 parent_backfill 时才真正用到；建表本身很便宜（574 粗块 × 每条细块查 2~5 个候选）。
        """
        self.parent = {}
        coarse: dict[str, list[str]] = {}
        for r in self.con.execute(
                "SELECT text, meta FROM memories WHERE scope='notes' AND kind='note' AND status='active'"):
            try:
                path = json.loads(r["meta"] or "{}").get("path") or ""
            except Exception:
                path = ""
            if path:
                coarse.setdefault(path, []).append(r["text"])
        if not coarse:
            return
        for r in self.con.execute(
                "SELECT id, text, source_id FROM memories WHERE scope='notes' AND kind='note_chunk' "
                "AND status='active'"):
            parent_list = coarse.get(r["source_id"] or "")
            if not parent_list:
                continue
            head = (r["text"] or "").strip()[:30]
            if not head:
                continue
            for pt in parent_list:
                if head in pt:
                    self.parent[r["id"]] = pt
                    break
        # 问题条目 → 它所属的细块正文（问题串本身对模型没用，命中后直接换成正文）
        chunk_text = {r["id"]: r["text"] for r in self.con.execute(
            "SELECT id, text FROM memories WHERE kind='note_chunk' AND status='active'")}
        self.q_chunk: dict[str, str] = {}
        for r in self.con.execute("SELECT id, source_id FROM memories WHERE kind='note_q' AND status='active'"):
            cid = r["source_id"] or ""
            t = chunk_text.get(cid)
            if t:
                self.parent[r["id"]] = t
                self.q_chunk[r["id"]] = cid

    def _build_chunk_neighbors(self) -> None:
        """细块 → 「前块尾部 + 后块头部」的上下文串（只给重排器用）。

        与父块回溯的区别：父块回溯换的是**返回正文**（字数 +75%、实测无增益）；
        这里只扩**判别依据**——重排器看到跨块的上下文，但调用方拿到的仍是本块原文。
        开销：一次全表扫 + 按 path 分组排序，1444 条细块约 20ms，随 load() 一起做。
        """
        self.chunk_ctx = {}
        n = int(self.policy.get("rerank_context") or 0)
        if n <= 0:
            return
        groups: dict[str, list[tuple[int, str, str]]] = {}
        for r in self.con.execute(
                "SELECT id, text, meta FROM memories WHERE kind='note_chunk' AND status='active'"):
            try:
                m = json.loads(r["meta"] or "{}")
            except Exception:
                continue
            path = m.get("path") or ""
            if not path:
                continue
            groups.setdefault(path, []).append((int(m.get("idx") or 0), r["id"], r["text"] or ""))
        for items in groups.values():
            items.sort(key=lambda x: x[0])
            for i, (_idx, cid, text) in enumerate(items):
                prev_tail = items[i - 1][2][-n:] if i > 0 else ""
                next_head = items[i + 1][2][:n] if i + 1 < len(items) else ""
                self.chunk_ctx[cid] = f"{prev_tail}\n{text}\n{next_head}"

    def _build_chunk_titles(self) -> None:
        """细块 → 所属笔记标题（取 meta.path 文件名，≤30 字），只喂重排器。"""
        self.chunk_title = {}
        if not self.policy.get("rerank_title_anchor"):
            return
        from pathlib import PurePosixPath
        for r in self.con.execute(
                "SELECT id, meta FROM memories WHERE kind='note_chunk' AND status='active'"):
            try:
                m = json.loads(r["meta"] or "{}")
            except Exception:
                continue
            stem = PurePosixPath(m.get("path") or "").stem.strip()
            if stem:
                self.chunk_title[r["id"]] = stem[:30]

    def _build_chunk_seq(self) -> None:
        """细块按 path 排序，供 note_promote 拼窗口正文。"""
        self.chunk_seq = {}
        self.chunk_of = {}
        for r in self.con.execute(
                "SELECT id, text, meta FROM memories WHERE kind='note_chunk' AND status='active'"):
            try:
                m = json.loads(r["meta"] or "{}")
            except Exception:
                continue
            path = m.get("path") or ""
            if not path:
                continue
            idx = int(m.get("idx") or 0)
            self.chunk_seq.setdefault(path, []).append((idx, r["id"], r["text"] or ""))
            self.chunk_of[r["id"]] = (path, idx)
        for v in self.chunk_seq.values():
            v.sort(key=lambda x: x[0])

    def _promote_notes(self, hits: list[Hit], p: dict) -> list[Hit]:
        """同 path 的多块合并成一条笔记结果：分数取 max/sum，正文取命中块 ±window 的邻块。"""
        seq = getattr(self, "chunk_seq", None)
        if not seq:
            return hits
        agg = p.get("note_agg") or "max"
        w = int(p.get("note_window") or 1)
        cap = int(p.get("note_max_chars") or 900)
        groups: dict[str, list[Hit]] = {}
        order: list[str] = []
        passthrough: list[Hit] = []
        for h in hits:
            key = self.chunk_of.get(h.id)
            if not key:
                passthrough.append(h)   # 粗块/事实：本来就是一整篇，不动
                continue
            path = key[0]
            if path not in groups:
                groups[path] = []
                order.append(path)
            groups[path].append(h)
        if agg == "none":
            # 只换正文、不动排序：同名笔记只留最先出现的一条，后面同 path 的直接丢掉
            seen: set[str] = set()
            out2: list[Hit] = []
            for h in hits:
                key = self.chunk_of.get(h.id)
                if not key:
                    out2.append(h)
                    continue
                path = key[0]
                if path in seen:
                    continue
                seen.add(path)
                idx = key[1]
                items = seq.get(path, [])
                lo = max(0, idx - w)
                hi = idx + w
                text = "\n".join(t for i, _cid, t in items if lo <= i <= hi)[:cap] or h.text
                out2.append(replace(h, text=text))
            return out2
        merged: list[Hit] = []
        for path in order:
            hs = groups[path]
            hs.sort(key=lambda x: -float(x.score))
            best = hs[0]
            score = float(best.score) if agg == "max" else sum(float(x.score) for x in hs)
            idx = self.chunk_of[best.id][1]
            items = seq.get(path, [])
            lo = max(0, idx - w)
            hi = idx + w
            text = "\n".join(t for i, _cid, t in items if lo <= i <= hi)[:cap] or best.text
            merged.append(replace(best, text=text, score=score))
        merged.extend(passthrough)
        merged.sort(key=lambda x: -float(x.score))
        return merged

    def _dedupe_chunks(self, hits: list[Hit], p: dict) -> list[Hit]:
        """同一个细块可能以「正文块」和「问题条目」两种身份进来，只留排名靠前的那个。"""
        if not getattr(self, "q_chunk", None):
            return hits
        seen: set[str] = set()
        out: list[Hit] = []
        for h in hits:
            key = self.q_chunk.get(h.id, h.id)
            if key.startswith("chunk:") or h.id in self.q_chunk:
                if key in seen:
                    continue
                seen.add(key)
            out.append(h)
        return out

    def _backfill_parents(self, hits: list[Hit], p: dict) -> list[Hit]:
        mode = p.get("parent_backfill")
        if not self.parent:
            return hits
        cap = int(p.get("parent_max_chars") or 700)
        out = []
        for h in hits:
            par = self.parent.get(h.id)
            if h.kind == "note_q":
                # 问题条目：永远换成所属细块正文（一条问题字符串对模型毫无用处）
                out.append(replace(h, text=(par or h.text)[:cap]))
                continue
            if not mode or not par:
                out.append(h)
                continue
            if mode == "replace":
                text = par[:cap]
            else:  # append
                text = (h.text + "\n…\n" + par)[:cap]
            out.append(replace(h, text=text))
        return out

    # ---------- 各臂 ----------
    def _arm_dense(self, qv: np.ndarray, n: int) -> list[tuple[int, float]]:
        assert self.mat is not None
        sims = self.mat @ qv
        idx = np.argpartition(-sims, min(n, len(sims) - 1))[:n]
        return [(int(i), float(sims[i])) for i in idx[np.argsort(-sims[idx])]]

    def _bm25(self, q: str, n: int, k1: float = 1.4, b: float = 0.75) -> list[tuple[int, float]]:
        qts = list(dict.fromkeys(tokenize(q)))
        N = len(self.docs)
        if not N or not qts:
            return []
        avgdl = self.avgdl or 1.0
        scores: list[float] = []
        skip_q = not bool(self.policy.get("note_q_bm25", False))
        for d in self.docs:
            if skip_q and d.kind == "note_q":
                scores.append(0.0)      # 问题条目是短文档，进 BM25 会被长度归一化抬到前排
                continue
            tf: dict[str, int] = {}
            for t in d.toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            dl = len(d.toks) or 1
            for t in qts:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            scores.append(s)
        arr = np.asarray(scores)
        idx = np.argsort(-arr)[:n]
        return [(int(i), float(arr[i])) for i in idx if arr[i] > 0]

    def _arm_act(self, n: int) -> list[tuple[int, float]]:
        now = int(time.time())
        scored = [(i, activation(d.importance, d.stability, d.last_access or now, now))
                  for i, d in enumerate(self.docs)]
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def _arm_graph(self, seeds: dict[int, float], n: int) -> list[tuple[int, float]]:
        """图扩散：A(v) = MAX over paths ∏(w^alpha · gamma^hop) · seed。无边时返回空。"""
        p = self.policy
        if not self.nbrs or not seeds:
            return []
        alpha = float(p["graph_alpha"])
        gamma = float(p["graph_gamma"])
        hops = int(p["graph_hops"])
        # 只留能落到活跃节点上的种子（scope 过滤交给外层）
        acc: dict[str, float] = {}
        frontier: dict[str, float] = {}
        for i, s in seeds.items():
            did = self.docs[i].id
            acc[did] = max(acc.get(did, 0.0), float(s))
            frontier[did] = float(s)
        seen = set(acc)
        for hop in range(1, hops + 1):
            nxt: dict[str, float] = {}
            for nid, sv in frontier.items():
                for nb, w in self.nbrs.get(nid, ()):  # type: ignore[union-attr]
                    if w <= 0:
                        continue
                    val = sv * (w ** alpha) * (gamma ** hop)
                    if val > acc.get(nb, 0.0):
                        acc[nb] = val
                    if val > nxt.get(nb, 0.0):
                        nxt[nb] = val
            frontier = nxt
            if not frontier:
                break
        ranked = sorted(acc.items(), key=lambda x: -x[1])[:n]
        return [(self.by_id[i], v) for i, v in ranked if i in self.by_id]

    
# ---------- 融合 ----------
    def _maybe_reload(self) -> None:
        """指数变了就重建内存索引：note_sync 每 15 分钟会改库，MCP/HTTP 是常驻进程，不能靠重启。"""
        try:
            row = self.con.execute("SELECT count(*), coalesce(sum(length(text)),0) FROM memories "
                                   "WHERE status='active'").fetchone()
            stamp = (int(row[0]), int(row[1]))
        except Exception:
            return
        if getattr(self, "_stamp", None) != stamp:
            self._stamp = stamp
            if self._loaded:
                self.load(force=True, as_of=self.as_of)

    def recall(self, query: str, k: int = 5, scope: str | None = None,
               write_log: bool = True, arms: list[str] | None = None,
               as_of: int | None = None) -> list[Hit]:
        """对外入口：显式给 scope 或关掉 two_pass 时走单路，否则两路召回后融合。"""
        self._maybe_reload()
        self.load(as_of=as_of)
        p = self.policy
        if p.get("query_strip"):
            query = strip_intent(query)
        boost = p.get("scope_boost") or {}
        # 开了重排就必须先取大候选池：只重排 top-k 的话，排 6~pool 的相关块永远上不来
        base = max(k, int(p.get("rerank_pool", 10))) if p.get("use_rerank") else k
        wide = max(base, int(p.get("rerank_pool_wide") or 0)) if p.get("use_rerank") else base

        def _run(pool: int) -> list[Hit]:
            if scope is None and p.get("two_pass") and len(boost) >= 2:
                return self._recall_two_pass(query, pool, arms, boost)
            return self._recall_once(query, pool, scope, arms)

        out = _run(base)
        pool_used = base
        # 自适应池：top1 不够像（长块稀释/正确块排在第 11~20 名）时，才多花一档重排成本
        if p.get("use_rerank") and wide > base and out:
            th = float(p.get("rerank_pool_wide_sim") or 0.0)
            if th and float(out[0].dense_sim) < th:
                wide_out = _run(wide)
                if wide_out:
                    out, pool_used = wide_out, wide
        # E3：融合排序会把「dense 靠前、bm25/图臂不认」的正确块挤出重排池
        # （v2 notes/semantic 实测掉到第 45~47 名，而 dense 单臂本来在第 6~20 名）。
        # 打开 rerank_pool_union_dense 后，把 dense 单臂 top N 并进重排输入：去重、不改融合顺序。
        union_n = int(p.get("rerank_pool_union_dense") or 0)
        if p.get("use_rerank") and union_n and out:
            extra = self._recall_once(query, union_n, scope, ["dense"])
            seen = {h.id for h in out}
            added = [h for h in extra if h.id not in seen]
            if added:
                out = list(out) + added
                pool_used = len(out)
        # E3b：不加大总池，只给 dense 单臂留固定名额（总池不变 ⇒ 重排延迟不变）
        reserve = int(p.get("rerank_pool_dense_reserve") or 0)
        if p.get("use_rerank") and reserve and out:
            total = max(int(p.get("rerank_pool", 20)), reserve)
            head = list(out[: max(0, total - reserve)])
            seen = {h.id for h in head}
            for h in self._recall_once(query, reserve, scope, ["dense"]):
                if len(head) >= total or h.id in seen:
                    continue
                head.append(h)
                seen.add(h.id)
            out = head
            pool_used = len(out)
        # M2 重排钩子（memex/rerank.py HOOK_PATCH）：只在 policy 打开 use_rerank 时生效，
        # 失败由 rerank 内部降级 lexical，绝不让重排把召回搞挂。
        if out and p.get("use_rerank"):
            skip = float(p.get("rerank_skip_sim") or 0.0)
            # 两段门控：命中最像的那条已经足够接近时就不花 180ms 重排
            if not skip or float(out[0].dense_sim) < skip:
                from .rerank import rerank as _rerank
                ctx = self.chunk_ctx
                texts = [ctx.get(h.id, h.text) for h in out] if ctx else None
                if p.get("rerank_title_anchor") and self.chunk_title:
                    base_texts = texts if texts is not None else [h.text for h in out]
                    texts = [f"{self.chunk_title[h.id]}\n{t}" if h.id in self.chunk_title else t
                             for h, t in zip(out, base_texts)]
                out = _rerank(query, out, top_k=k, policy={**p, "rerank_pool": pool_used},
                              df=self.df, n_docs=len(self.docs), texts=texts)
                out = out[:k]
        if out and p.get("note_promote"):
            out = self._promote_notes(out, p)[:k]
        trace = uuid.uuid4().hex
        out = self._dedupe_chunks(out, p)
        out = self._backfill_parents(out, p)
        out = [h if h.trace_id == trace else replace(h, trace_id=trace) for h in out]
        if write_log and out:
            self.con.execute(
                "INSERT INTO recall_log(trace_id, ts, query, policy_version, hits) VALUES(?,?,?,?,?)",
                (trace, int(time.time()), query, None, json.dumps([o.id for o in out])))
            self.con.commit()
        return out

    def _recall_two_pass(self, query: str, k: int,
                         arms: list[str] | None, boost: dict) -> list[Hit]:
        per = max(int(self.policy.get("two_pass_per", 15)), k * 3)
        cand: dict[str, tuple[float, Hit]] = {}
        for sc, mul in boost.items():
            for h in self._recall_once(query, per, sc, arms):
                # 融合键用池内 RRF 分（已含 bm25），不是 dense_sim —— 否则丢掉 bm25 的排序增益
                adj = float(h.score) * float(mul)
                cur = cand.get(h.id)
                if cur is None or adj > cur[0]:
                    h2 = Hit(id=h.id, scope=h.scope, kind=h.kind, text=h.text, score=h.score,
                             dense_sim=h.dense_sim, arms=h.arms, importance=h.importance,
                             trace_id=h.trace_id)
                    cand[h.id] = (adj, h2)
        if not cand:
            return []
        return [h for _adj, h in sorted(cand.values(), key=lambda t: -t[0])[:k]]

    def _recall_once(self, query: str, k: int = 5, scope: str | None = None,
                     arms: list[str] | None = None) -> list[Hit]:
        p = self.policy
        active = list(arms or p.get("arms") or ["dense", "bm25", "act"])
        bad = set(active) - {"dense", "bm25", "act", "graph", "probe"}
        if bad:
            raise ValueError(f"未知臂：{sorted(bad)}；可选 dense/bm25/act/graph")
        qv = self.embedder.embed_one(query)
        qv = qv / (np.linalg.norm(qv) or 1.0)

        got: dict[str, list[tuple[int, float]]] = {}
        if "dense" in active:
            got["dense"] = self._arm_dense(qv, int(p["arm_dense"]))
        # 问题探针：问题条目和用户问法同语体，相似度比正文块灵敏；命中高就把它的块补进候选
        if p.get("note_q_probe") and self.q_idx and self.mat is not None:
            sims = self.mat[self.q_idx] @ qv
            bi = int(np.argmax(sims))
            if float(sims[bi]) >= float(p.get("note_q_probe_sim", 0.72)):
                qdoc = self.docs[self.q_idx[bi]]
                cid = getattr(self, "q_chunk", {}).get(qdoc.id)
                ci = self.by_id.get(cid or "")
                if ci is not None:
                    got.setdefault("probe", []).append((ci, float(sims[bi])))
        # 问题条目本身永远不当结果返回（只通过 probe 影响候选）
        for _k in list(got):
            got[_k] = [(i, sc) for i, sc in got[_k] if self.docs[i].kind != "note_q"]
        if "bm25" in active:
            got["bm25"] = self._bm25(query, int(p["arm_bm25"]))
        if "act" in active:
            got["act"] = self._arm_act(int(p["arm_act"]))
        if "graph" in active:
            seeds: dict[int, float] = {}
            for name, hits in got.items():
                for rank, (i, _s) in enumerate(hits[: int(p["graph_seeds"])]):
                    seeds[i] = max(seeds.get(i, 0.0), 1.0 / (rank + 1))
            got["graph"] = self._arm_graph(seeds, int(p["arm_graph"]))

        # 关闭某些臂时按剩余臂重新归一（只影响尺度，不改变排序）
        def _w(a: str) -> float:
            if f"w_{a}" in p:
                return float(p[f"w_{a}"])
            return float(p.get("w_probe", 0.20)) if a == "probe" else 0.0

        total = sum(_w(a) for a in active) or 1.0
        weights = {a: _w(a) / total for a in active}
        arms = got

        fused: dict[int, float] = {}
        why: dict[int, list[str]] = {}
        for name, hits in arms.items():
            w = float(weights.get(name, _w(name)))
            for rank, (i, _s) in enumerate(hits):
                if scope and self.docs[i].scope != scope:
                    continue
                fused[i] = fused.get(i, 0.0) + w / (float(p["rrf_k"]) + rank + 1)
                why.setdefault(i, []).append(f"{name}#{rank + 1}")

        if not fused:
            return []
        cand = sorted(fused.items(), key=lambda x: -x[1])[: max(k * 6, 30)]

        # MMR：别把同一段的近似重复块都返回
        # 注意：rel 必须和 div 同量纲。RRF 分是 ~0.0x 量级，div 是余弦（0~1），
        # 直接相减会让多样性项完全压过相关性（2026-11-22 修：先按最大值归一）。
        lam = float(p["mmr_lambda"])
        picked: list[int] = []
        pool = [i for i, _ in cand]
        rel = {i: s for i, s in cand}
        top = max(rel.values()) or 1.0
        reln = {i: s / top for i, s in rel.items()}
        while pool and len(picked) < k:
            best, best_v = None, -1e9
            for i in pool:
                div = max((float(self.docs[i].vec @ self.docs[j].vec) for j in picked), default=0.0)
                v = lam * reln[i] - (1 - lam) * div
                if v > best_v:
                    best, best_v = i, v
            picked.append(best)  # type: ignore[arg-type]
            pool.remove(best)  # type: ignore[arg-type]

        trace = uuid.uuid4().hex
        out = []
        for i in picked:
            d = self.docs[i]
            out.append(Hit(
                id=d.id, scope=d.scope, kind=d.kind, text=d.text,
                score=round(float(rel[i]), 6), arms=why.get(i, []),
                # 融合分是 RRF 量纲（~0.0x），不能当相似度阈值用；负例判据要用这个
                dense_sim=round(float(d.vec @ qv), 4),
                importance=d.importance, trace_id=trace,
            ))
        return out
