#!/usr/bin/env python3
"""统计判定 CLI：单次体检 / 两次配对比较（Top1、R@5、MRR、McNemar、bootstrap CI）。

用法:
  # 单次体检
  python tools/eval_stats.py --a data/memex-eval-xxx-dev.json
  # 两次比较（delta 方向 = B - A）
  python tools/eval_stats.py --a old.json --b new.json
  # 逐分组比较，只列 n>=10 的组
  python tools/eval_stats.py --a old.json --b new.json --groups
  # 机器可读
  python tools/eval_stats.py --a old.json --b new.json --groups --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memex.eval_lib import (  # noqa: E402
    bootstrap_ci,
    judge_change,
    mrr,
    perm_test_rr,
    mcnemar_exact,
    mrr,
    subset,
)


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "rows" not in d:
        raise SystemExit(f"{path} 没有 rows，无法做逐题统计（老格式请重跑评测）")
    return d


def groups_of(rows: list[dict], min_n: int) -> list[str]:
    keys = sorted({f'{r.get("scope")}/{r.get("kind")}' for r in rows})
    out = ["ALL"]
    for k in keys:
        if len(subset(rows, k)) >= min_n:
            out.append(k)
    return out


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def single(d: dict, group: str) -> dict:
    rows = subset(d["rows"], group)
    return {
        "group": group,
        "n": len(rows),
        "top1": round(sum(1 for r in rows if r.get("hit1")) / len(rows), 4),
        "r5": round(sum(1 for r in rows if r.get("hit5")) / len(rows), 4),
        "mrr": mrr(rows),
        "p50_ms": d.get("summary", {}).get(group, {}).get("p50_ms"),
        "p95_ms": d.get("summary", {}).get(group, {}).get("p95_ms"),
    }


def compare(da: dict, db: dict, group: str, key: str, iters: int) -> dict:
    ra, rb = subset(da["rows"], group), subset(db["rows"], group)
    jc = judge_change(ra, rb, key, iters)
    has_rr = any("rr" in r for r in ra) and any("rr" in r for r in rb)
    mp = perm_test_rr(ra, rb) if has_rr else None
    ca = sum(1 for r in ra if r.get(key)) / len(ra) if ra else 0.0
    cb = sum(1 for r in rb if r.get(key)) / len(rb) if rb else 0.0
    return {
        "group": group,
        "n": jc["n"],
        "a": round(ca, 4),
        "b": round(cb, 4),
        "mrr_a": mrr(ra), "mrr_b": mrr(rb),
        "mrr_perm": mp,
        **{k: jc[k] for k in ("delta", "ci", "p", "b01", "b10", "verdict")},
        "significant_improvement": jc["significant_improvement"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b")
    ap.add_argument("--group", default="ALL")
    ap.add_argument("--groups", action="store_true", help="逐分组比较（n>=10）")
    ap.add_argument("--key", default="hit1", choices=["hit1", "hit5"])
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    da = load(a.a)
    db = load(a.b) if a.b else None
    if db is not None and da.get("set") != db.get("set"):
        print(f"警告：两次 run 的 set 不同（{da.get('set')} vs {db.get('set')}），只比较同 id 题", file=sys.stderr)

    gs = groups_of(da["rows"], a.min_n) if (a.groups and db) else [a.group]
    if db is None:
        out = [single(da, g) for g in gs]
        if a.json:
            print(json.dumps(out, ensure_ascii=False, indent=1))
            return
        print("| 分组 | n | Top1 | R@5 | MRR | p50 | p95 |")
        print("|---|---|---|---|---|---|---|")
        for r in out:
            print(f"| {r['group']} | {r['n']} | {fmt_pct(r['top1'])} | {fmt_pct(r['r5'])} | "
                  f"{r['mrr']:.3f} | {r['p50_ms']} | {r['p95_ms']} |")
        return

    out = [compare(da, db, g, a.key, a.iters) for g in gs]
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    print(f"A = {a.a}\nB = {a.b}\n指标 = {a.key}，delta 方向 = B - A\n")
    print("| 分组 | n | A | B | Δ | 95% CI | McNemar p | 翻转 (A错B对/A对B错) | 判决 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in out:
        line = (f"| {r['group']} | {r['n']} | {fmt_pct(r['a'])} | {fmt_pct(r['b'])} | "
                f"{r['delta'] * 100:+.1f}pt | [{r['ci'][0] * 100:+.1f}, {r['ci'][1] * 100:+.1f}] | "
                f"{r['p']:.4f} | {r['b01']}/{r['b10']} | {r['verdict']} |")
        print(line)
        mp = r.get("mrr_perm")
        if mp:
            print(f"  ↳ MRR {r['mrr_a']:.4f} → {r['mrr_b']:.4f}（{mp['delta']:+.4f}，"
                  f"置换 p={mp['p']:.4f}，n={mp['n']}；解释性副指标，不入账）")


if __name__ == "__main__":
    main()
