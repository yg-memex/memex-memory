# -*- coding: utf-8 -*-
"""评测口径的唯一实现（所有评测脚本都必须走这里）。

为什么单独抽出来：2026-11-22 上午发现过一次口径事故——`golden.jsonl` 里 expect 有的存字符串、
有的存数组，两个评测脚本都写 `[e.lower() for e in expect]`，遇到字符串就变成**按字符匹配**，
于是 Top1 被虚高（memex 报 85%，真实约 70%）。此后：
- expect 一律用 `norm_expect()` 归一化（字符串 = 单条）；
- 命中判定一律用 `judge()`；
- 新写的评测脚本不许自己再写一遍子串判断。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

GOLDEN_DIR = Path(os.environ.get("MEMEX_GOLDEN_DIR", "eval/golden"))

SETS = {
    "all": "golden.jsonl",          # 120 条，只用于全量体检
    "dev": "golden-dev.jsonl",      # 80 条，调参/日常闸门用这个
    "holdout": "golden-holdout.jsonl",  # 40 条，发版前才跑，调参严禁看
}

SCOPES = {"facts", "notes"}
KINDS = {"lexical", "semantic", "negative"}


def load_golden(set_name: str = "all") -> list[dict]:
    if set_name not in SETS:
        raise ValueError(f"未知集合 {set_name}，可选 {sorted(SETS)}")
    path = GOLDEN_DIR / SETS[set_name]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_expect(expect) -> list[str]:
    """字符串视为单条；数组原样；None/空 → 空表。"""
    if expect is None:
        return []
    if isinstance(expect, str):
        return [expect] if expect else []
    return [str(e) for e in expect if str(e)]


def norm_text(s: str | None) -> str:
    return (s or "").replace("\n", " ").lower()


def judge(item: dict, texts: list[str], top1_score: float | None, neg_gate: float) -> tuple[bool, bool]:
    """返回 (hit1, hit5)。texts 必须是已小写化的 top-k 文本。"""
    if item["kind"] == "negative":
        ok = (top1_score is None) or float(top1_score) < float(neg_gate)
        return ok, ok
    exp = [e.lower() for e in norm_expect(item["expect"])]
    if not exp:
        return False, False
    ok1 = bool(texts) and any(e in texts[0] for e in exp)
    ok5 = any(e in t for t in texts for e in exp)
    return ok1, ok5


def validate_golden(items: list[dict]) -> list[str]:
    """结构体检：只报问题，不改数据（改金标是 M4 owner 的事）。"""
    problems: list[str] = []
    seen: set[str] = set()
    for it in items:
        i = it.get("id", "?")
        if i in seen:
            problems.append(f"{i}: id 重复")
        seen.add(i)
        for key in ("id", "scope", "kind", "q", "expect"):
            if key not in it:
                problems.append(f"{i}: 缺字段 {key}")
        if it.get("scope") not in SCOPES:
            problems.append(f"{i}: scope={it.get('scope')!r} 不合法")
        if it.get("kind") not in KINDS:
            problems.append(f"{i}: kind={it.get('kind')!r} 不合法")
        if not norm_expect(it.get("expect")) and it.get("kind") != "negative":
            problems.append(f"{i}: expect 为空")
        if not str(it.get("q") or "").strip():
            problems.append(f"{i}: q 为空")
    return problems


def summarize(rows: list[dict]) -> dict[str, dict]:
    """rows: [{scope, kind, hit1, hit5, ms, chars}] → 分组汇总。"""
    import statistics
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[f'{r["scope"]}/{r["kind"]}'].append(r)
        groups["ALL"].append(r)
    out: dict[str, dict] = {}
    for g, rs in sorted(groups.items()):
        n = len(rs)
        ms = sorted(r["ms"] for r in rs)
        out[g] = {
            "n": n,
            "top1": round(sum(1 for r in rs if r["hit1"]) / n, 4),
            "r5": round(sum(1 for r in rs if r["hit5"]) / n, 4),
            "p50_ms": round(statistics.median(ms), 1),
            "p95_ms": round(ms[min(n - 1, int(round((n - 1) * 0.95)))], 1),
            "avg_chars": round(statistics.mean(r["chars"] for r in rs), 1),
        }
    return out


# ---- 统计判定（2026-11-22 P0/R4）：配对比较与置信区间 ------------------------------
# 为什么必须配对：同一批题目、同一金标，只有系统变化；比较两次运行就是比较同一组题目上的
# 对/错翻转。单点 84.2 vs 81.7 这种差，在 n=120 上 95% CI 约 ±6.6pt，不看配对就是噪声。

def paired_rows(rows_a: list[dict], rows_b: list[dict], key: str = "hit1") -> list[tuple[dict, dict]]:
    """按 id 对齐两次运行的逐题结果；只返回两边都有的题。"""
    am = {r.get("id"): r for r in rows_a}
    return [(am[r["id"]], r) for r in rows_b if r.get("id") in am]


def mcnemar_exact(rows_a: list[dict], rows_b: list[dict], key: str = "hit1") -> dict:
    """配对二分类的 McNemar 精确检验（双侧）。

    b01 = A 错 B 对，b10 = A 对 B 错。n = b01+b10 很小时只能靠精确二项分布。
    """
    from math import comb

    pairs = paired_rows(rows_a, rows_b, key)
    b01 = sum(1 for a, b in pairs if not a.get(key) and b.get(key))
    b10 = sum(1 for a, b in pairs if a.get(key) and not b.get(key))
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    total = len(pairs)
    return {
        "n": total,
        "b01": b01,
        "b10": b10,
        "delta": round((b01 - b10) / total, 4) if total else 0.0,
        "p": round(p, 4),
    }


def bootstrap_ci(rows_a: list[dict], rows_b: list[dict], key: str = "hit1",
                 iters: int = 5000, seed: int = 42) -> dict:
    """配对差的 bootstrap 95% 置信区间（对题目重采样，固定 seed 可复现）。"""
    import random

    pairs = paired_rows(rows_a, rows_b, key)
    deltas = [(1 if b.get(key) else 0) - (1 if a.get(key) else 0) for a, b in pairs]
    n = len(deltas)
    if n == 0:
        return {"n": 0, "delta": 0.0, "lo": 0.0, "hi": 0.0}
    point = sum(deltas) / n
    rnd = random.Random(seed)
    samples = []
    for _ in range(iters):
        samples.append(sum(deltas[rnd.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return {
        "n": n,
        "delta": round(point, 4),
        "lo": round(samples[int(0.025 * iters)], 4),
        "hi": round(samples[min(iters - 1, int(0.975 * iters))], 4),
    }


def reciprocal_rank(row: dict) -> float:
    """第一条命中的倒数名次；negative 题按「门禁是否通过」记 1/0。

    注意：存档 top5 的 text 只留了前 160 字（判定用的是全文），所以存档文本里看不到
    expect 时，用 hit1/hit5 兜底：hit1=True 视为第 1 名，仅 hit5=True 按第 5 名保守记。
    这样 MRR 不会低于 Top1，也不会虚高；跨运行对比是同一套截断，方向可比。
    """
    if isinstance(row.get("rr"), (int, float)) and not isinstance(row.get("rr"), bool):
        return float(row["rr"])   # 新运行：评测时用全文算好的逐题 rr
    if row.get("kind") == "negative":
        return 1.0 if row.get("hit1") else 0.0
    exp = [e.lower() for e in norm_expect(row.get("expect"))]
    if not exp:
        return 0.0
    for i, t in enumerate(row.get("top5") or [], 1):
        txt = norm_text(t.get("text") if isinstance(t, dict) else str(t))
        if any(e in txt for e in exp):
            return 1.0 / i
    if row.get("hit1"):
        return 1.0
    if row.get("hit5"):
        return 0.2
    return 0.0


def mrr(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return round(sum(reciprocal_rank(r) for r in rows) / len(rows), 4)


def perm_test_rr(rows_a: list[dict], rows_b: list[dict], iters: int = 10000,
                 seed: int = 42) -> dict:
    """MRR 的配对置换检验（H0：两臂逐题 RR 可交换）。

    统计量 T = mean(rr_B - rr_A)；每对差独立以 0.5 翻符号重采样 iters 轮，
    双侧 p = (1 + #{|T*| >= |T|}) / (iters + 1)。只对正题有意义，负例请先剔除。
    """
    import random
    pairs = paired_rows(rows_a, rows_b, key="rr")
    diffs = []
    for a, b in pairs:
        if a.get("kind") == "negative" or (b and b.get("kind") == "negative"):
            continue   # MRR 只在正题上有意义
        ra = reciprocal_rank(a)
        rb = reciprocal_rank(b) if b is not None else None
        if rb is None:
            continue
        diffs.append(rb - ra)
    if not diffs:
        return {"n": 0, "delta": 0.0, "p": 1.0}
    t = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        acc = 0.0
        for d in diffs:
            acc += d if rng.random() < 0.5 else -d
        if abs(acc / len(diffs)) >= abs(t) - 1e-12:
            hits += 1
    return {"n": len(diffs), "delta": round(t, 4),
            "p": round((1 + hits) / (iters + 1), 4), "iters": iters, "seed": seed}


def subset(rows: list[dict], group: str) -> list[dict]:
    """group: ALL 或 'facts/lexical' 这种。"""
    if group == "ALL":
        return list(rows)
    scope, _, kind = group.partition("/")
    return [r for r in rows if r.get("scope") == scope and r.get("kind") == kind]


def judge_change(rows_a: list[dict], rows_b: list[dict], key: str = "hit1",
                 iters: int = 5000) -> dict:
    """把 McNemar + bootstrap 合成一条可用判决。delta 方向 = B 相对 A。"""
    mc = mcnemar_exact(rows_a, rows_b, key)
    ci = bootstrap_ci(rows_a, rows_b, key, iters)
    sig = mc["p"] < 0.05 and ci["lo"] > 0
    return {
        "n": mc["n"],
        "delta": ci["delta"],
        "ci": [ci["lo"], ci["hi"]],
        "p": mc["p"],
        "b01": mc["b01"],
        "b10": mc["b10"],
        "significant_improvement": sig,
        "verdict": "显著变好" if sig else ("不显著" if mc["p"] >= 0.05 else "显著但区间跨 0"),
    }
