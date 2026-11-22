# -*- coding: utf-8 -*-
"""M2 融合与重排：把「相似度接近就分不开」的那批条目救回来。

三条硬约束（写在最前面，改代码前先读）：
1. **不碰 `recall.py` 的打分逻辑**。本模块只提供纯函数 `rerank(query, hits, top_k, policy)`，
   接到召回上靠 `RerankedRecall`（子类，`recall.py` 一行没动）。M1 合并钩子见 `HOOK_PATCH`。
2. **只重排前 `rerank_pool` 条**，输出 top_k=5。延迟预算是 **p95 ≤ 300ms**（2026-11-22 12:50
   起，现役已切到 api 后端；旧的 50ms 预算是 lexical 时代的，早作废）。收敛手段见 `api_*` 那几个键。
3. **`Hit` 一个字段都不改**：重排分放在 `RerankedHit.rerank_score`（`Hit` 的 frozen 子类），
   `isinstance(x, Hit)` 仍为真，`eval_run.py` 里现成的 `getattr(h, "rerank_score", None)` 直接就能读到。

为什么默认走 HTTP 重排：本地重排服务（如 bge-reranker-v2-m3）跑在独立进程，
派单又禁止 `pip install`。所以走两条路并量：
- `lexical`：BM25（全局 df）+ 词覆盖 + 硬 token（数字/端口/文件名）+ 短语命中，纯 Python；
- `api`：硅基流动 `/v1/rerank` 的 `BAAI/bge-reranker-v2-m3`（复用 embedding 的 key，无新依赖）。
  现役后端。一次 RTT ≈ 180ms，p95 卡在 300ms 预算内靠「池 10 / 硬截止 / 熔断 / 缓存」。

用法：
    python -m memex.rerank --eval --set dev                  # 同评测口径复算（默认 lexical）
    python -m memex.rerank --eval --set dev --backend api    # 跑 API 重排，看质量天花板
    python -m memex.rerank --eval --set dev --two-stage      # 两段门控（dense 砍 + rerank 砍）
    python -m memex.rerank --tune --set dev                  # 在 dev 上搜权重，打印 top10
    python -m memex.rerank --calib --set dev                 # 标定 neg 门线，写 data/calib-rerank.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .recall import Recall, tokenize
from .types import Hit

ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = ROOT / "data" / "calib-rerank.json"

# 特征顺序固定（权重文件按名字取，不按位置）
FEATURES = ("prior", "dense", "bm25", "cov", "hard", "phrase", "len")

DEFAULT_RERANK = {
    # 默认关：只有 dev 分数严格变好才建议打开（由 M5 采纳，别自己改 DEFAULT_POLICY）
    "use_rerank": False,
    "rerank_backend": "lexical",      # lexical | api
    # 实测（口径 A 全库，p50/p95 为单查询 召回+重排 的实测值）：
    #   pool 10 → 79.2/93.3 @184/249ms（现役）
    #   pool 20 → 79.2/94.2 @285/384ms（Top1 一点没涨，只涨 R@5）
    #   pool 30 → 80.0/96.7 @535ms（超预算）
    # 预算 p95 ≤ 300ms 下取 10：Top1 全拿，p95 还有 50ms 余量；要 R@5 就单独加 pool，别默认开。
    "rerank_pool": 10,
    "rerank_weights": {"prior": 0.20, "dense": 0.30, "bm25": 0.18, "cov": 0.12,
                       "hard": 0.14, "phrase": 0.06, "len": 0.0},
    # 两段门控（§3.2 标定，见 data/calib-rerank.json）：低段直接砍，中段看重排分
    "neg_dense_hard": 0.48,     # 低段：dense 低于这条直接判空
    "neg_dense_mid": 0.58,      # 中段上界：与 neg_dense_hard 之间交给 rerank 分裁决
    "neg_rerank": 0.50,         # 中段内的 rerank 门线
    "rerank_blend_prior": 0.0,      # >0 时把进来的融合名次当先验混进来（防止重排把明显对的顶掉）
    "api_blend_lexical": 0.0,       # >0 时把本地特征分混进 API 分（补硬 token/标识符敏感度）
    "api_model": "BAAI/bge-reranker-v2-m3",
    "api_timeout": 5,                 # socket 级兜底超时（秒）。原来硬编码 20，卡死要等 20 秒
    # 硬截止：>0 时到点直接掐连接并降级 lexical（不是「等它回来再判断」，是真的不等）
    # 注意：它是「保命」不是「提速」——低于正常 RTT 会让所有请求都降级，质量掉回 lexical
    "api_deadline_ms": 0,
    "api_max_chars": 0,               # >0 时把送排文档截断到这个长度（省 token、提速度）
    "api_retry": 1,                   # 连接被服务端掐掉时的重试次数（截止触发的超时**不**重试）
    # 并发分片数：把池拆成 N 份同时发，墙钟 ≈ max(各片) 而不是 sum。
    # 这是「pool 20 的质量 + pool 10 的延迟」的唯一正路（串行 pool 20 = p95 388ms）
    "api_parallel_n": 1,
    "api_breaker_n": 3,               # 连续失败/超时 N 次 → 熔断，冷却期内直接走 lexical
    "api_breaker_s": 30,
    # 重排分缓存键：归一化 query + 归一化 doc 前缀。默认关（0=精确键），生产开 160
    "cache_doc_chars": 0,
    # 重排前对 query 做的变换流水，见 prepare_query()。空 = 不处理（A/B 用）
    "rerank_query_transforms": [],
    # >0 时，top1 的 dense_sim 已经这么高就不调 API（本地就够确定，省一次 RTT）
    "rerank_skip_sim": 0.0,
}

_HARD_ASCII = re.compile(r"[a-z0-9][a-z0-9_\-\./]{1,}")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


# --------------------------------------------------------------------------------------
# 数据结构：给 Hit 加一个自带默认值的字段（frozen 子类，不破坏契约）
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankedHit(Hit):
    """重排后的命中。`score` 仍是融合分（RRF），重排分单独放 `rerank_score`。"""

    rerank_score: float = 0.0


# --------------------------------------------------------------------------------------
# 特征
# --------------------------------------------------------------------------------------
def hard_tokens(query: str) -> list[str]:
    """查询里的「硬 token」：带数字的（17372 / 16g / 4k / 400）、ASCII 标识符（ollama / lora / crlf）。

    这类词命中与否几乎决定对错，且 dense 常常把它淹掉（问句里它只占很小一部分）。
    """
    out: list[str] = []
    for t in _HARD_ASCII.findall((query or "").lower()):
        t = t.strip(".")
        if len(t) < 2:
            continue
        if any(c.isdigit() for c in t) or (len(t) >= 3 and t.isalpha()):
            out.append(t)
    return list(dict.fromkeys(out))


def _phrase_hit(query: str, text_low: str) -> float:
    """查询里的中文 4 字 / 3 字短语是否整段出现在文档里（子串查找是 C 速度，够快）。"""
    cands: list[str] = []
    for run in _CJK_RUN.findall(query or ""):
        for n in (4, 3):
            for i in range(len(run) - n + 1):
                cands.append(run[i:i + n])
    best = 0
    for c in dict.fromkeys(cands):
        if c in text_low:
            best = max(best, len(c))
            if best >= 4:
                break
    return 0.0 if best < 3 else min(1.0, (best - 2) / 4.0)


def _idf(df: dict[str, int], n_docs: int, tok: str) -> float:
    d = df.get(tok, 0)
    return math.log(1.0 + (n_docs - d + 0.5) / (d + 0.5))


class _Pool:
    """一次 rerank 调用的查询级上下文：查询 token、硬 token、短语候选、池内 df/bm25。"""

    def __init__(self, query: str, docs: list[str], df: dict[str, int] | None, n_docs: int):
        self.query = query or ""
        self.q_low = self.query.lower()
        self.q_toks = list(dict.fromkeys(tokenize(self.query)))
        self.hard = hard_tokens(self.query)
        self.docs_low = [d.lower() for d in docs]
        self.docs_toks = [tokenize(d) for d in docs]
        self.n_docs = int(n_docs or 0)

        # IDF：优先用全库 df（调用方给了就用），否则退回池内统计（20 条，仍能区分稀疏词）
        if df and self.n_docs:
            self.df = df
        else:
            d: dict[str, int] = {}
            for toks in self.docs_toks:
                for t in dict.fromkeys(toks):
                    d[t] = d.get(t, 0) + 1
            self.df = d
            self.n_docs = len(docs)
        self.avgdl = (sum(len(t) for t in self.docs_toks) / len(docs)) if docs else 0.0
        self.q_idf_sum = sum(_idf(self.df, self.n_docs, t) for t in self.q_toks) or 1.0

    def features(self, i: int, dense_sim: float) -> list[float]:
        toks = self.docs_toks[i]
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks) or 1
        k1, b = 1.2, 0.6
        bm25 = 0.0
        for t in self.q_toks:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = _idf(self.df, self.n_docs, t)
            bm25 += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / (self.avgdl or 1.0)))

        covered = [t for t in self.q_toks if t in tf]
        cov = sum(_idf(self.df, self.n_docs, t) for t in covered) / self.q_idf_sum

        text_low = self.docs_low[i]
        if self.hard:
            hit = sum(1 for h in self.hard if h in text_low)
            hard = hit / len(self.hard)
        else:
            hard = 0.0

        return [
            1.0 / (1.0 + i),                                        # 进来的融合名次（RRF 先验）
            max(0.0, min(1.0, (float(dense_sim) - 0.35) / 0.35)),   # dense 归一到 [0,1]
            bm25,
            min(1.0, cov),
            hard,
            _phrase_hit(self.query, text_low),
            1.0 - min(1.0, math.log(max(dl, 1) / 120.0) / math.log(40.0)) if dl > 120 else 1.0,
        ]


def _score_from_features(f: list[float], weights: dict) -> float:
    w = [float(weights.get(k, 0.0)) for k in FEATURES]
    tot = sum(abs(x) for x in w) or 1.0
    raw = sum(x * y for x, y in zip(w, f))
    return round(max(0.0, min(1.0, raw / tot)), 6)


def lexical_scores(query: str, hits: list[Hit], df: dict[str, int] | None = None,
                   n_docs: int = 0, weights: dict | None = None,
                   docs: list[str] | None = None) -> list[float]:
    """纯本地打分（无模型、无网络）。一次 20 条约 2~6ms。"""
    w = weights or DEFAULT_RERANK["rerank_weights"]
    pool = _Pool(query, list(docs) if docs else [h.text for h in hits], df, n_docs)
    bm_max = 0.0
    raws = [pool.features(i, h.dense_sim) for i, h in enumerate(hits)]
    for f in raws:
        bm_max = max(bm_max, f[1])
    out: list[float] = []
    for f in raws:
        if bm_max > 0:
            f = list(f)
            f[2] = f[2] / bm_max          # 池内归一，避免长文档挤爆量纲
        out.append(_score_from_features(f, w))
    return out


# --------------------------------------------------------------------------------------
# API 后端（质量参照；延迟靠「池大小 / 截断 / 硬截止 / 熔断」四层收敛）
# --------------------------------------------------------------------------------------
_RERANK_CACHE: dict[tuple[str, str], float] = {}
_CONN: dict[tuple[str, float], object] = {}   # (base, timeout) -> HTTPSConnection（keep-alive）
_KEY_CACHE: dict[str, str] = {}
_BASE_CACHE: dict[str, str] = {}
_BREAKER = {"until": 0.0, "fail_streak": 0}

# 计数器：报告里要能说清「慢是慢在网络、还是慢在降级重算」
STATS = {"api_calls": 0, "api_ok": 0, "api_fail": 0, "api_abort": 0,
         "cache_hit_docs": 0, "skip_sim": 0, "breaker_skip": 0, "lexical": 0}


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = 0
    _BREAKER.update({"until": 0.0, "fail_streak": 0})


_CK_PUNCT = re.compile(r"[\s\u3000,，。.、?？!！:：;；\"'“”‘’()（）\[\]【】<>《》~`|\\/]+")


def _ck(s: str, n: int = 0) -> str:
    """缓存键归一化：小写、去标点空白。n>0 时只取前 n 字（doc 前缀，牺牲一点精度换命中率）。"""
    t = _CK_PUNCT.sub("", (s or "").lower())
    return t[:n] if n else t


def _api_base(policy: dict | None = None) -> str:
    """重排用的 base URL。

    优先 policy["rerank_api_base"]（本地保险服务 http://127.0.0.1:8903/v1 走这条），
    缺省回落到和 embedding 共用的 EMBED_API_BASE。不能直接改 EMBED_API_BASE 本身——
    它和向量调用共用，改了会把 embedding 一起改走。
    """
    override = (policy or {}).get("rerank_api_base") or ""
    if override:
        return str(override).rstrip("/")
    b = _BASE_CACHE.get("b")
    if b:
        return b
    import os  # noqa: PLC0415

    b = os.environ.get("MEMEX_RERANK_BASE", "http://127.0.0.1:8903/v1")
    _BASE_CACHE["b"] = b
    return b


def _conn(base: str, timeout: float = 5.0, slot: int = 0):
    """按 (host, timeout, slot) 复用连接。

    slot = 并发分片的编号：同一个连接不能被两个线程同时写，所以并发时每片独占一条。
    """
    import http.client  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415

    u = urlparse(base)
    host = f"{u.scheme}://{u.netloc}"
    ckey = (f"{host}#{slot}", float(timeout))
    c = _CONN.get(ckey)
    if c is None:
        # 本地保险服务是明文 http，云端是 https：按 scheme 选，不能写死 HTTPSConnection
        cls = http.client.HTTPConnection if u.scheme == "http" else http.client.HTTPSConnection
        c = cls(u.netloc, timeout=float(timeout))
        _CONN[ckey] = c
    return c, u.path.rstrip("/"), ckey


def _api_key() -> str:
    """密钥只读一次。原来每次重排都进 config 读 keyfile，120 次调用白烧 I/O。"""
    k = _KEY_CACHE.get("k")
    if k:
        return k
    import os  # noqa: PLC0415

    k = os.environ.get("MEMEX_RERANK_KEY", "x")
    _KEY_CACHE["k"] = k
    return k


def _post_rerank(payload: bytes, policy: dict, slot: int = 0) -> dict:
    """POST /rerank。硬截止用「工作线程 + join(timeout)」：到点立刻返回，不等服务端。

    踩过的坑：一开始用 `threading.Timer` 去 `conn.close()` 打断阻塞的 `recv`，
    **Windows 上不生效**（close 不保证打断另一个线程里已阻塞的 recv），
    结果是「计时器响了、请求照样跑完」，p50 该多慢还是多慢，只有计数器在骗人。
    """
    import threading  # noqa: PLC0415

    p = policy
    timeout = float(p.get("api_timeout") or 5)
    deadline = float(p.get("api_deadline_ms") or 0) / 1000.0
    sock_timeout = max(timeout, deadline) if deadline else timeout
    base = _api_base(p)
    last: Exception | None = None
    for _attempt in range(int(p.get("api_retry") or 0) + 1):
        conn, path, ckey = _conn(base, sock_timeout, slot)

        def _once(c=conn, pth=path) -> dict:
            c.request("POST", pth + "/rerank", body=payload, headers={
                "Authorization": "Bearer " + _api_key(),
                "Content-Type": "application/json", "Connection": "keep-alive"})
            resp = c.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"rerank HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}")
            return json.loads(raw.decode())

        try:
            if deadline <= 0:
                return _once()
            box: dict = {}

            def _run(fn=_once, b=box) -> None:
                try:
                    b["ok"] = fn()
                except BaseException as exc:            # noqa: BLE001
                    b["err"] = exc

            th = threading.Thread(target=_run, daemon=True)
            th.start()
            th.join(deadline)
            if th.is_alive():
                # 线程还在跑：连接已经不能复用了（它正被另一个线程持有），直接扔掉
                STATS["api_abort"] += 1
                _CONN.pop(ckey, None)
                raise TimeoutError(f"api deadline {deadline * 1000:.0f}ms 到了，放弃本次重排")
            if "err" in box:
                raise box["err"]
            return box["ok"]
        except Exception as e:                   # 网络/限流/鉴权/超时，一律降级
            last = e
            _CONN.pop(ckey, None)
            try:
                conn.close()
            except Exception:
                pass
            if isinstance(e, TimeoutError):      # 截止触发的超时不重试，重试 = 再等一个截止
                raise
    raise last if last else RuntimeError("rerank 未知失败")


_WARMED = {"done": False}


def warmup(policy: dict | None = None, slots: int = 4) -> dict:
    """预热重排 HTTP 连接（每个并发分片各一条）+ 打一次最小 rerank。

    冷启动首次查询要建 N 条 TLS 连接，实测 387ms（第二次同 query 165ms）。
    服务启动时调一次，把这段代价从「用户第一次问问题」挪到「启动时」。
    任何失败都只记录、不抛——预热失败不该让服务起不来。
    """
    p = {**DEFAULT_RERANK, **(policy or {})}
    if not p.get("use_rerank"):
        return {"skipped": "use_rerank=false"}
    if _WARMED["done"]:
        return {"cached": True}
    out = {"slots": [], "errors": []}
    payload = json.dumps({"model": p["api_model"], "query": "warmup",
                          "documents": ["warmup"], "top_n": 1}).encode()
    try:
        _api_key()
    except Exception as e:  # 没配 key 就直接放弃
        out["errors"].append(f"key: {type(e).__name__}")
        return out
    for slot in range(max(1, int(slots or 1))):
        try:
            t0 = time.time()
            _post_rerank(payload, p, slot=slot)
            out["slots"].append(round((time.time() - t0) * 1000, 1))
        except Exception as e:
            out["errors"].append(f"slot{slot}: {type(e).__name__}")
    _WARMED["done"] = True
    return out


def api_scores(query: str, docs: list[str], policy: dict | None = None) -> list[float]:
    """硅基流动 /v1/rerank。失败/超时抛异常，由 rerank() 降级到 lexical。"""
    p = {**DEFAULT_RERANK, **(policy or {})}
    dcap = int(p.get("cache_doc_chars") or 0)
    qk = _ck(query) if dcap else query
    keys = [((qk, _ck(d, dcap)) if dcap else (query, d)) for d in docs]
    todo = [(i, d) for i, (k, d) in enumerate(zip(keys, docs)) if k not in _RERANK_CACHE]
    STATS["cache_hit_docs"] += len(docs) - len(todo)
    if todo:
        cap = int(p.get("api_max_chars") or 0)
        npar = max(1, int(p.get("api_parallel_n") or 1))
        # 条数太少不值得拆（拆了反而多付一次 RTT + 线程开销）
        if npar <= 1 or len(todo) < npar * 4:
            chunks = [todo]
        else:
            # 交错分块：0,2,4… 与 1,3,5…，两批 token 量更接近，墙钟取 max 而不是 sum
            chunks = [todo[i::npar] for i in range(npar)]

        def _payload(chunk: list) -> bytes:
            docs_c = [(d[:cap] if cap else d) for _i, d in chunk]
            return json.dumps({"model": p["api_model"], "query": query,
                               "documents": docs_c, "top_n": len(docs_c)}).encode()

        _api_key()                       # 预热：别让两个线程同时去读密钥文件
        if len(chunks) == 1:
            STATS["api_calls"] += 1
            results = [_post_rerank(_payload(chunks[0]), p, 0)]
        else:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

            STATS["api_calls"] += len(chunks)
            with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
                futs = [ex.submit(_post_rerank, _payload(c), p, i)
                        for i, c in enumerate(chunks)]
                results = [f.result() for f in futs]     # 任一片失败 → 整体降级，不半途而废
        for chunk, data in zip(chunks, results):
            got = {int(x["index"]): float(x["relevance_score"])
                   for x in (data or {}).get("results", [])}
            for pos, (i, _d) in enumerate(chunk):
                _RERANK_CACHE[keys[i]] = got.get(pos, 0.0)
    return [_RERANK_CACHE.get(k, 0.0) for k in keys]


# --------------------------------------------------------------------------------------
# 重排前的 query 变换（意图词剥离；关键词扩展由 M-side 注册进来）
# --------------------------------------------------------------------------------------
# 这些是「问句脚手架」：它们描述**怎么问**，不描述**问什么**。放着它们，cross-encoder
# 会把注意力分给「在哪 / 怎么 / 为什么」，而不是主题词——notes/lexical 那批就是死在这里。
_INTENT_PATTERNS = [
    # 疑问尾巴（长的放前面：正则在同一位置取第一个能匹配的分支）
    r"(?:放|在)\s*哪(?:里|儿|个文件|个目录|个地方|个位置)?(?:了|呢)?",
    r"哪(?:个|些)?(?:文件|目录|地方|位置|版本|模型|算法|参数|方案)?",
    r"什么(?:文件|目录|地方|情况|意思|时候|型号)?",
    r"多少(?:钱|次|条|个|毫秒|秒|分钟|小时|倍)?",
    r"从\s*多少|到\s*多少",
    r"怎么(?:办|弄|搞|做|加|设置|调|改|用)?",
    r"如何(?:办|弄|做|加|设置|调|改)?",
    r"为什么|为何",
    r"是不是|有没有|能不能|可不可以|要不要|会不会|对不对|是谁(?:占|用|的)?",
    r"请问|帮我(?:查|找|看一下|看看)|告诉我|我想(?:知道|问)",
    r"一下",
    r"还(?:能|会|要|在|得)",
    r"[吗呢吧啊呀哈嘛]+$",          # 只剥句尾语气词，句中不动
]
_INTENT_RE = re.compile("|".join(_INTENT_PATTERNS))
# 剥完剩下的孤立虚字（「在哪个文件」→「个文件」那个「个」）
_LEFTOVER_RE = re.compile(r"(?:^|\s)(?:个|了|的|是|就|都|也|还|又|很|太)(?=\s|$)")

QUERY_TRANSFORMS: dict[str, object] = {}


def strip_intent_words(query: str) -> str:
    """剥掉**句中/句尾**的疑问脚手架，只留主题词。剥完空了就退回原句。

    与 `recall.strip_intent` 的区别：那个只剥句首前缀 + 句尾短语（服务于 BM25/dense 召回，
    不敢动句中，怕把实体词切坏）；这个用于重排，敢动句中，因为 cross-encoder 会把注意力
    分给「在哪个文件」这种脚手架，而不是「家具摆放数据」。
    """
    q = _INTENT_RE.sub(" ", query or "")
    q = _LEFTOVER_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip(" ，,。.、")
    return q or (query or "")


def _strip_intent_tail(query: str) -> str:
    """复用 M1 的 `recall.strip_intent`（句首前缀 + 句尾短语），保持两边口径一致。"""
    from .recall import strip_intent as _f  # noqa: PLC0415

    return _f(query)


QUERY_TRANSFORMS["strip_intent"] = strip_intent_words
QUERY_TRANSFORMS["strip_intent_words"] = strip_intent_words
QUERY_TRANSFORMS["strip_intent_tail"] = _strip_intent_tail


def register_query_transform(name: str, fn) -> None:
    """给外部（关键词扩展/同义词/术语归一）留的挂载点：`fn(query) -> query`。

    用法：`register_query_transform("expand", my_expand)`，
    再在策略里写 `"rerank_query_transforms": ["strip_intent", "expand"]`。
    """
    QUERY_TRANSFORMS[name] = fn


def prepare_query(query: str, policy: dict | None = None) -> str:
    """按策略里的流水依次变换 query。空流水 = 原样返回（默认，A/B 对照不受影响）。"""
    p = {**DEFAULT_RERANK, **(policy or {})}
    q = query
    for name in (p.get("rerank_query_transforms") or []):
        fn = QUERY_TRANSFORMS.get(name)
        if callable(fn):
            q = fn(q)
    return q or query



# --------------------------------------------------------------------------------------
# 主函数（对外只暴露这个）
# --------------------------------------------------------------------------------------
def rerank(query: str, hits: list[Hit], top_k: int = 5, policy: dict | None = None,
           df: dict[str, int] | None = None, n_docs: int = 0,
           texts: list[str] | None = None) -> list[Hit]:
    """对 hits（前 rerank_pool 条）重排，返回 top_k 条 `RerankedHit`。

    - `use_rerank` 为假、或 hits ≤1 条时**原样返回**（保证「关掉 = v0 完全一致」）。
    - `arms` 里追加 `rr#<新名次>`，原始融合分不动，`rerank_score` 单独带出。
    - api 后端任何异常都降级到 lexical，绝不让重排把召回搞挂。
    """
    p = {**DEFAULT_RERANK, **(policy or {})}
    if not p.get("use_rerank") or len(hits) <= 1:
        return list(hits)

    pool_n = max(int(p["rerank_pool"]), top_k)
    head, tail = list(hits[:pool_n]), list(hits[pool_n:])
    # texts：只换「喂给重排器的文本」（可带邻块上下文），返回正文始终是 h.text
    docs = list(texts)[:len(head)] if texts else [h.text for h in head]
    scores: list[float]
    if p.get("rerank_backend") == "api":
        skip = float(p.get("rerank_skip_sim") or 0.0)
        top_sim = float(getattr(head[0], "dense_sim", 0.0) or 0.0)
        if skip > 0 and top_sim >= skip:
            # top1 本身已经很有把握，重排翻不动它，省一次 RTT
            STATS["skip_sim"] += 1
            out = [RerankedHit(**{**h.to_dict(), "rerank_score": 0.0,
                                  "arms": list(h.arms) + ["rr#skip"]}) for h in head]
            return out[:top_k] + tail[: max(0, top_k - len(out))]
        if time.time() < _BREAKER["until"]:
            # 熔断期内不再打 API：宁可这次用 lexical，也别让每个查询都等一个超时
            STATS["breaker_skip"] += 1
            scores = lexical_scores(prepare_query(query, p), head, df, n_docs, p.get("rerank_weights"))
        else:
            try:
                scores = api_scores(prepare_query(query, p), docs, p)
                STATS["api_ok"] += 1
                _BREAKER["fail_streak"] = 0
            except Exception as e:                   # 网络/限流/鉴权/截止，一律降级
                STATS["api_fail"] += 1
                _BREAKER["fail_streak"] += 1
                if _BREAKER["fail_streak"] >= int(p.get("api_breaker_n") or 3):
                    _BREAKER["until"] = time.time() + float(p.get("api_breaker_s") or 30)
                    _BREAKER["fail_streak"] = 0
                print(f"[rerank] api 失败，降级 lexical：{type(e).__name__}: {e}", file=sys.stderr)
                scores = lexical_scores(prepare_query(query, p), head, df, n_docs,
                                        p.get("rerank_weights"), docs=docs)
    else:
        STATS["lexical"] += 1
        scores = lexical_scores(prepare_query(query, p), head, df, n_docs, p.get("rerank_weights"))

    w = float(p.get("rerank_blend_prior") or 0.0)
    if w > 0:
        scores = [(1.0 - w) * scores[i] + w * (1.0 / (1.0 + i)) for i in range(len(head))]
    # 和「本地硬 token / 词覆盖」混一点：cross-encoder 对 `server-config` 这种标识符不敏感，
    # 会把分给一个只是在谈「检测」的邻块（g37 就是这么丢的）。alpha 很小时只做 tie-break 用。
    alpha = float(p.get("api_blend_lexical") or 0.0)
    if alpha > 0 and p.get("rerank_backend") == "api":
        lex = lexical_scores(prepare_query(query, p), head, df, n_docs, p.get("rerank_weights"))
        scores = [(1.0 - alpha) * scores[i] + alpha * lex[i] for i in range(len(head))]
    order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
    out: list[Hit] = []
    for new_rank, i in enumerate(order, 1):
        h = head[i]
        out.append(RerankedHit(**{**h.to_dict(), "rerank_score": scores[i],
                                  "arms": list(h.arms) + [f"rr#{new_rank}"]}))
    return out[:top_k] + tail[: max(0, top_k - len(out))]


def gate(query: str, hits: list[Hit], policy: dict | None = None) -> tuple[bool, str]:
    """两段门控：返回 (是否判空, 原因)。判据只用已算好的 sim / rerank 分，不再调模型。"""
    p = {**DEFAULT_RERANK, **(policy or {})}
    if not hits:
        return True, "空结果"
    top = hits[0]
    sim = float(getattr(top, "dense_sim", 0.0) or 0.0)
    rr = getattr(top, "rerank_score", None)
    if sim < float(p["neg_dense_hard"]):
        return True, f"dense {sim:.3f} < {p['neg_dense_hard']}"
    if rr is not None and sim < float(p["neg_dense_mid"]):
        if float(rr) < float(p["neg_rerank"]):
            return True, f"dense {sim:.3f} 在中段，rerank {float(rr):.3f} < {p['neg_rerank']}"
    return False, ""


# --------------------------------------------------------------------------------------
# 适配器：不改 recall.py 也能把重排接上（M1 合并钩子前的过渡形态）
# --------------------------------------------------------------------------------------
class RerankedRecall(Recall):
    """在 Recall 外面套一层重排。评测/面板可先用它，行为与「钩子接上后」一致。"""

    def recall(self, query: str, k: int = 5, scope: str | None = None,
               write_log: bool = True, arms: list[str] | None = None) -> list[Hit]:
        p = {**DEFAULT_RERANK, **self.policy}
        if not p.get("use_rerank"):
            return super().recall(query, k=k, scope=scope, write_log=write_log, arms=arms)
        pool = max(int(p["rerank_pool"]), int(k))
        head = super().recall(query, k=pool, scope=scope, write_log=False, arms=arms)
        out = rerank(query, head, top_k=int(k), policy=p, df=self.df, n_docs=len(self.docs))
        if write_log and out:
            # 必须「先插入再更新」：super() 是用 write_log=False 调的，recall_log 里根本没有这一行，
            # 只写 UPDATE 会命中 0 行（M6 12:19 报的 bug：trace(trace_id) 查不到）。
            import time as _t
            self.con.execute(
                "INSERT OR IGNORE INTO recall_log(trace_id, ts, query, policy_version, hits)"
                " VALUES(?,?,?,?,?)",
                (out[0].trace_id, int(_t.time()), query, None,
                 json.dumps([h.id for h in out], ensure_ascii=False)))
            self.con.execute("UPDATE recall_log SET hits=? WHERE trace_id=?",
                             (json.dumps([h.id for h in out], ensure_ascii=False), out[0].trace_id))
            self.con.commit()
        return out


# 给 M1 的一行钩子（在 `Recall.recall()` 里 `if write_log:` 之前插入）
HOOK_PATCH = '''        if self.policy.get("use_rerank"):
            from .rerank import rerank as _rerank
            out = _rerank(query, out, top_k=k, policy=self.policy, df=self.df, n_docs=len(self.docs))
'''


# --------------------------------------------------------------------------------------
# 评测/调参/标定 CLI（口径全部来自 memex.eval_lib，不另写一套判定）
# --------------------------------------------------------------------------------------
def _eval_rows(items: list[dict], r: Recall, k: int, neg_gate: float,
               two_stage: bool = False, scope_mode: str = "none") -> list[dict]:
    from .eval_lib import judge, norm_text

    rows = []
    for it in items:
        t0 = time.perf_counter()
        # scope_mode="none" = 全库（口径 A，闸门用这个）；"auto" = 按金标 scope 过滤（oracle，虚高）
        q_scope = it["scope"] if scope_mode == "auto" else None
        hits = r.recall(it["q"], k=k, scope=q_scope, write_log=False)
        ms = (time.perf_counter() - t0) * 1000
        texts = [norm_text(h.text) for h in hits]
        if two_stage and it["kind"] == "negative":
            empty, why = gate(it["q"], hits, r.policy)
            ok = empty
        else:
            ok = None
            why = ""
        top1_score = hits[0].dense_sim if hits else None
        ok1, ok5 = judge(it, texts, top1_score, neg_gate)
        if ok is not None:
            ok1 = ok5 = ok
        rows.append({**it, "ms": round(ms, 1), "hit1": ok1, "hit5": ok5,
                     "chars": sum(len(t) for t in texts), "dense_sim": top1_score,
                     "rerank_score": getattr(hits[0], "rerank_score", None) if hits else None,
                     "gate": why, "top1": texts[0][:80] if texts else "",
                     "top5": [{"id": h.id, "score": h.score, "dense_sim": h.dense_sim,
                               "rerank_score": getattr(h, "rerank_score", None),
                               "arms": list(h.arms), "text": norm_text(h.text)[:160]} for h in hits]})
    return rows


def _print_summary(tag: str, rows: list[dict]) -> dict:
    from .eval_lib import summarize

    s = summarize(rows)
    print(f"\n== {tag} · n={len(rows)}")
    print(f'{"group":22}{"n":>4}{"Top1":>8}{"R@5":>8}{"p50ms":>8}{"p95ms":>8}')
    for g, v in s.items():
        print(f'{g:22}{v["n"]:>4}{v["top1"]:>8.1%}{v["r5"]:>8.1%}{v["p50_ms"]:>8.0f}{v["p95_ms"]:>8.0f}')
    neg = [x for x in rows if x["kind"] == "negative"]
    print(f"负例判空：{sum(1 for x in neg if x['hit1'])}/{len(neg)}")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2 重排：评测 / 调参 / 标定")
    ap.add_argument("--eval", action="store_true", help="按评测口径复算")
    ap.add_argument("--tune", action="store_true", help="在 dev 上搜权重")
    ap.add_argument("--calib", action="store_true", help="标定 neg 门线并写 calib-rerank.json")
    ap.add_argument("--set", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--backend", default="lexical", choices=["lexical", "api", "none"])
    ap.add_argument("--two-stage", action="store_true", help="负例走两段门控")
    ap.add_argument("--scope", dest="scope_mode", default="none", choices=["none", "auto"],
                    help="none=全库（口径 A，默认）；auto=按金标 scope 过滤（oracle，虚高）")
    ap.add_argument("--json", default="", help="结果另存")
    ap.add_argument("--pool", type=int, default=0, help="覆盖 rerank_pool（只重排前 N 条）")
    ap.add_argument("--cap", type=int, default=0, help="送排文档截断长度（0=不截）")
    ap.add_argument("--blend", type=float, default=None, help="融合名次先验权重（0=纯重排分）")
    ap.add_argument("--db", default="")
    args = ap.parse_args(argv)

    from .db import connect
    from .embed import Embedder
    from .eval_lib import load_golden

    items = load_golden(args.set)
    con = connect(args.db) if args.db else connect()
    r = RerankedRecall(con, Embedder(con, batch=32))
    use = args.backend != "none"
    r.policy = {**DEFAULT_RERANK, **r.policy, "use_rerank": use, "rerank_backend": args.backend}
    if args.pool:
        r.policy = {**r.policy, "rerank_pool": int(args.pool)}
    if args.cap:
        r.policy = {**r.policy, "api_max_chars": int(args.cap)}
    if args.blend is not None:
        r.policy = {**r.policy, "rerank_blend_prior": float(args.blend)}

    if args.tune:
        return _tune(items, r, args.k, args.scope_mode)
    if args.calib:
        return _calib(items, r, args.k, args.scope_mode)

    rows = _eval_rows(items, r, args.k, float(r.policy.get("neg_dense", 0.62)), args.two_stage,
                      args.scope_mode)
    s = _print_summary(f"set={args.set} backend={args.backend} scope={args.scope_mode} "
                       f"two_stage={args.two_stage}", rows)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"set": args.set, "backend": args.backend,
                                   "two_stage": args.two_stage, "scope": args.scope_mode,
                                   "policy": r.policy,
                                   "all": s.get("ALL"), "groups": {g: v for g, v in s.items() if g != "ALL"},
                                   "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("written", out)
    return 0


def _tune(items: list[dict], r: Recall, k: int, scope_mode: str = "none") -> int:
    """先算一次候选池的特征（只依赖 query/doc，与权重无关），再在特征上搜权重 → 秒级。"""
    from .eval_lib import judge, norm_text

    r.policy = {**r.policy, "use_rerank": False}      # 池子取原始融合顺序
    cases = []
    for it in items:
        q_scope = it["scope"] if scope_mode == "auto" else None
        hits = r.recall(it["q"], k=int(r.policy["rerank_pool"]), scope=q_scope, write_log=False)
        if not hits:
            continue
        pool = _Pool(it["q"], [h.text for h in hits], r.df, len(r.docs))
        feats = [pool.features(i, h.dense_sim) for i, h in enumerate(hits)]
        bm = max((f[1] for f in feats), default=0.0) or 1.0
        feats = [[f[0], f[1], f[2] / bm, f[3], f[4], f[5], f[6]] for f in feats]
        cases.append((it, hits, np.asarray(feats, dtype=np.float32)))

    def score_w(w: dict, kk: int = k) -> tuple[float, float, int]:
        top1 = top5 = 0
        negok = 0
        nneg = 0
        wv = np.array([float(w.get(x, 0.0)) for x in FEATURES], dtype=np.float32)
        tot = float(np.abs(wv).sum()) or 1.0
        for it, hits, F in cases:
            s = (F @ wv) / tot
            order = np.argsort(-s)
            picked = [hits[int(i)] for i in order[:kk]]
            texts = [norm_text(h.text) for h in picked]
            if it["kind"] == "negative":
                nneg += 1
                if hits[int(order[0])].dense_sim < float(r.policy.get("neg_dense", 0.62)):
                    negok += 1
                continue
            a, b = judge(it, texts, None, 0.62)
            top1 += a
            top5 += b
        npos = sum(1 for c in cases if c[0]["kind"] != "negative")
        return top1 / npos, top5 / npos, negok

    rng = np.random.default_rng(20261122)
    seen: dict[tuple, dict] = {}
    base = dict(DEFAULT_RERANK["rerank_weights"])
    cands = [base, {"dense": 0.7, "bm25": 0.3, "cov": 0, "hard": 0, "phrase": 0, "len": 0},
             {"dense": 0.25, "bm25": 0.25, "cov": 0.25, "hard": 0.25, "phrase": 0, "len": 0}]
    for _ in range(260):
        v = rng.dirichlet(np.array([2.0, 2.5, 2.0, 1.6, 1.6, 0.8, 0.3]))
        cands.append({x: round(float(y), 3) for x, y in zip(FEATURES, v)})
    out = []
    for w in cands:
        key = tuple(round(w.get(x, 0.0), 3) for x in FEATURES)
        if key in seen:
            continue
        t1, t5, neg = score_w(w)
        seen[key] = w
        out.append((t1, t5, neg, w))
    out.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    print(f"候选 {len(out)} 组（n_pos={sum(1 for c in cases if c[0]['kind']!='negative')}）")
    print(f'{"Top1":>7}{"R@5":>8}{"neg":>6}  weights')
    for t1, t5, neg, w in out[:10]:
        print(f"{t1:>7.1%}{t5:>8.1%}{neg:>6}  {json.dumps({k: round(v,3) for k,v in w.items() if v}, ensure_ascii=False)}")
    return 0


def _calib(items: list[dict], r: Recall, k: int, scope_mode: str = "none") -> int:
    """标定门控：在 dev 上枚举 (dense 硬砍线, dense 中段线, rerank 门线)，
    目标 = 负例全砍的前提下**误伤正例最少**，并把 Pareto 前沿写进 calib-rerank.json。

    判空规则（两段，与 `gate()` 一致）：
        dense < hard                → 判空
        hard ≤ dense < mid 且 rr < T → 判空
        其余                        → 放行
    """
    r.policy = {**r.policy, "use_rerank": True}          # 后端沿用 --backend
    backend = str(r.policy.get("rerank_backend"))
    pos, neg = [], []
    for it in items:
        q_scope = it["scope"] if scope_mode == "auto" else None
        hits = r.recall(it["q"], k=k, scope=q_scope, write_log=False)
        if not hits:
            (neg if it["kind"] == "negative" else pos).append((0.0, 0.0, it["id"], it["q"]))
            continue
        (neg if it["kind"] == "negative" else pos).append(
            (float(hits[0].dense_sim), float(getattr(hits[0], "rerank_score", 0.0) or 0.0),
             it["id"], it["q"]))
    pd_ = [x[0] for x in pos]
    nd_ = [x[0] for x in neg]
    pr = [x[1] for x in pos]
    nr = [x[1] for x in neg]
    print(f"后端={backend}　正例 n={len(pos)}　负例 n={len(neg)}")
    print(f"正例 dense min={min(pd_):.3f} p10={np.percentile(pd_,10):.3f}　rerank min={min(pr):.3f} p10={np.percentile(pr,10):.3f}")
    print(f"负例 dense max={max(nd_):.3f} p90={np.percentile(nd_,90):.3f}　rerank max={max(nr):.3f} "
          f"mean={np.mean(nr):.3f}")

    def cut(row, hard, mid, t):
        d, rr = row[0], row[1]
        return d < hard or (d < mid and rr < t)

    best = None
    pareto = []
    for hard in (0.48, 0.50, 0.52, 0.55, 0.57, 0.59):
        for mid in (0.58, 0.60, 0.62, 0.65, 0.68):
            if mid <= hard:
                continue
            for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
                kn = sum(1 for x in neg if cut(x, hard, mid, t))
                kp = sum(1 for x in pos if cut(x, hard, mid, t))
                if kn < len(neg):
                    continue
                # 负例全砍的前提下，正例误伤最少；同分时门线更「居中」（离两侧都远）者优先
                margin = min(min(pd_), 1.0) - max(nd_)
                key = (-kp, margin, -(hard + mid + t))
                rec = (key, hard, mid, t, kp)
                pareto.append(rec)
                if best is None or key > best[0]:
                    best = rec
    if best is None:
        print("没有既能砍完负例又不伤正例的门线（说明 dev 上两类有重叠，两段门控不成立）")
        return 1
    _k, hard, mid, t, kp = best
    hurt = [x for x in pos if cut(x, hard, mid, t)]
    print(f"\n选定：neg_dense_hard={hard} neg_dense_mid={mid} neg_rerank={t}"
          f" → 负例 22/22 全砍，正例误伤 {kp}/{len(pos)}")
    if hurt:
        print("误伤明细：", ", ".join(f"{h[2]}(dense {h[0]:.3f}/rr {h[1]:.3f})" for h in hurt[:12]))
    print(f"对照：现役单段 dense<0.62 → 负例 {sum(1 for x in neg if x[0] < 0.62)}/{len(neg)}，"
          f"正例误伤（生产上会被判空）{sum(1 for x in pos if x[0] < 0.62)}/{len(pos)}")

    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "set": "dev", "set_size": len(items), "backend": backend, "scope": scope_mode,
        "rerank_pool": int(r.policy.get("rerank_pool", 20)),
        "neg_dense_hard": hard, "neg_dense_mid": mid, "neg_rerank": t,
        "result": {"neg_cut": f"{len(neg)}/{len(neg)}", "pos_hurt": f"{kp}/{len(pos)}"},
        "baseline_single_gate": {
            "neg_dense": 0.62,
            "neg_cut": f"{sum(1 for x in neg if x[0] < 0.62)}/{len(neg)}",
            "pos_would_be_empty": f"{sum(1 for x in pos if x[0] < 0.62)}/{len(pos)}"},
        "dist": {
            "pos_dense_min": round(min(pd_), 4), "pos_dense_p10": round(float(np.percentile(pd_, 10)), 4),
            "pos_rerank_min": round(min(pr), 4), "pos_rerank_p10": round(float(np.percentile(pr, 10)), 4),
            "neg_dense_max": round(max(nd_), 4), "neg_dense_p90": round(float(np.percentile(nd_, 90)), 4),
            "neg_rerank_max": round(max(nr), 4), "neg_rerank_mean": round(float(np.mean(nr)), 4),
        },
        "pareto": [{"hard": h, "mid": m, "t": tt, "pos_hurt": kp_} for _kk, h, m, tt, kp_ in
                   sorted(pareto, key=lambda x: (-x[0][0], -x[0][1]))[:8]],
        "neg_rows": [{"id": x[2], "q": x[3], "dense": round(x[0], 4), "rerank": round(x[1], 4)} for x in neg],
        "why": ("两段门控：dense 低段直接砍（chitchat 打底），中段交重排分裁决。"
                "正例 dense 最小 %.3f、负例 dense 最大 %.3f —— 单段阈值本来就分不开，"
                "重排把负例 top1 的 dense 从 0.611 压到 %.3f，才是这套门控能立住的原因。"
                "标定只用 dev，holdout 不许看。" % (min(pd_), max(nd_), max(nd_))),
    }
    CALIB_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", CALIB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
