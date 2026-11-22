# -*- coding: utf-8 -*-
"""memex 评测入口（M4）：固定口径、可重复、可防回归。

用法：
  python -m memex.eval_run                      # 默认 dev（80 条，日常闸门/调参）
  python -m memex.eval_run --set holdout        # 发版前才跑（40 条，调参严禁看）
  python -m memex.eval_run --set all            # 120 条全量体检
  python -m memex.eval_run --json o.json        # 另存结果（M5 evolve 读这个）
  python -m memex.eval_run --golden-file <path> # 复算任意历史金标（审计用）

口径唯一实现在 memex/eval_lib.py（字符串 expect 绝不能被按字符拆）。
产物：data/memex-eval-*.json（历史）+ --json 指定路径 + docs/回归-YYYYmmdd.md（自动追加）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memex.db import connect  # noqa: E402
from memex.embed import Embedder  # noqa: E402
from memex.eval_lib import judge, load_golden, norm_text, summarize, validate_golden  # noqa: E402
from memex.recall import Recall  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Windows 控制台默认 GBK，笔记正文里带 emoji（如 ✅）会让 print 直接崩（UnicodeEncodeError）。
# 统一把标准输出改成 utf-8 + replace，别让一行日志毁掉整次评测。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass


def render_md(now: datetime, set_name: str, rows: list[dict], summary: dict,
              policy: dict, arms: list[str] | None, neg_gate: float, neg_ok: str) -> str:
    lines = [f"## memex 回归报表 {now:%Y-%m-%d %H:%M}（set={set_name}）", "",
             f"- n={len(rows)}　负例判空：{neg_ok}（neg_dense={neg_gate}）",
             f"- arms: `{','.join(arms or policy.get('arms') or ['默认'])}`",
             f"- policy: `{json.dumps(policy, ensure_ascii=False)}`", "",
             "| 分组 | n | Top1 | R@5 | p50 | p95 | chars |", "|---|---|---|---|---|---|---|"]
    for g, s in summary.items():
        lines.append(f"| {g} | {s['n']} | {s['top1']:.1%} | {s['r5']:.1%} | "
                     f"{s['p50_ms']:.0f} | {s['p95_ms']:.0f} | {s['avg_chars']:.0f} |")
    miss = [x for x in rows if not x["hit1"]]
    lines += ["", f"## MISS 共 {len(miss)} 条（每条写明它实际命中了什么）", ""]
    for m in miss:
        lines.append(f"### {m['id']} · {m['scope']}/{m['kind']}")
        lines.append(f"- q: {m['q']}")
        lines.append(f"- expect: {m['expect']}")
        if m["kind"] == "negative":
            lines.append(f"- 误召回：top1 dense_sim={m['dense_sim']}（应判空却返回了内容）")
        elif m["top5"]:
            h = m["top5"][0]
            lines.append(f"- 实际 top1：dense_sim={h['dense_sim']}　score={h['score']}　arms={h['arms']}")
            lines.append(f"  - 文本：{h['text']}")
        else:
            lines.append("- 实际 top1：（空结果）")
        for h in m["top5"][1:3]:
            lines.append(f"  - top{'-'}: dense_sim={h['dense_sim']} score={h['score']} {h['text'][:90]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["all", "dev", "holdout"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neg", type=float, default=None, help="覆盖 neg_dense（默认取 policy，回落 0.62）")
    ap.add_argument("--arms", default="", help="逗号分隔：dense,bm25,act,graph；空=用 policy 默认")
    ap.add_argument("--json", default="", help="结果另存到这个路径（M5 读它）")
    ap.add_argument("--db", default="", help="指向另一个 sqlite 库（测试用临时库）")
    ap.add_argument("--golden-file", default="", help="复算任意历史金标文件")
    ap.add_argument("--policy", default="", help="用指定策略文件跑（M5 演化用：只换参数，不改口径）")
    ap.add_argument("--no-md", action="store_true", help="不写 docs/回归-YYYYmmdd.md")
    ap.add_argument("--scope", default="auto", choices=["auto", "none"],
                    help="auto=按金标 scope 过滤（oracle，默认，兼容旧行为）；none=全库检索（用户实际）")
    args = ap.parse_args()

    if args.golden_file:
        items = [json.loads(l) for l in Path(args.golden_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        set_name = Path(args.golden_file).stem
    else:
        items = load_golden(args.set)
        set_name = args.set
    problems = validate_golden(items)
    if problems:
        print(f"金标结构问题 {len(problems)} 条（不阻断，但要报给 M4）：")
        for p in problems[:10]:
            print("  -", p)

    con = connect(args.db) if args.db else connect()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or None
    pol: dict = {}
    if args.policy:
        # 显式指定策略时**不吞异常**：文件坏了要立刻报，不能静默跑默认策略（否则数字看着没变）
        from memex.policy_io import validate as _validate_policy  # 局部导入，避免 eval 启动变慢
        pol = _validate_policy(json.loads(Path(args.policy).read_text(encoding="utf-8")))
        print(f"policy: {args.policy} → {json.dumps({k: pol[k] for k in sorted(pol) if k.startswith('w_')}, ensure_ascii=False)}")
    r = Recall(con, Embedder(con, batch=32), policy=pol)
    policy = r.policy
    neg_gate = float(args.neg) if args.neg is not None else float(policy.get("neg_dense", 0.62))

    rows = []
    for it in items:
        t0 = time.perf_counter()
        # scope="auto"：按金标标的 scope 过滤（oracle 口径，MCP 传 scope 时等价）
        # scope="none"：全库检索（MCP 不传 scope 时的真实体验，闸门用这个）
        q_scope = it["scope"] if args.scope == "auto" else None
        hits = r.recall(it["q"], k=args.k, scope=q_scope, write_log=False, arms=arms)
        ms = (time.perf_counter() - t0) * 1000
        texts = [norm_text(h.text) for h in hits]
        top1_score = hits[0].dense_sim if hits else None
        ok1, ok5 = judge(it, texts, top1_score, neg_gate)
        # rr 用全文算（存档只留 160 字会低估名次，跨臂截断偏差不一致）
        if it.get("kind") == "negative":
            rr = 1.0 if ok1 else 0.0
        else:
            exp = [e.lower() for e in it.get("expect") or []]
            rr = 0.0
            for i, t in enumerate(texts, 1):
                if any(e in t for e in exp):
                    rr = 1.0 / i
                    break
        rows.append({**it, "ms": round(ms, 1), "hit1": ok1, "hit5": ok5, "rr": round(rr, 4),
                     "chars": sum(len(t) for t in texts),
                     "dense_sim": top1_score,
                     "top1": (hits[0].text if hits else "")[:80],
                     "top5": [{"id": h.id, "score": h.score, "dense_sim": h.dense_sim,
                               "arms": list(getattr(h, "arms", []) or []),
                               "rerank_score": getattr(h, "rerank_score", None),
                               "text": norm_text(h.text)[:160]} for h in hits]})

    summary = summarize(rows)
    print(f"== memex eval · set={set_name} · n={len(rows)} · arms={','.join(arms or policy.get('arms') or []) or '默认'}")
    print(f'{"group":22}{"n":>4}{"Top1":>8}{"R@5":>8}{"p50ms":>9}{"p95ms":>9}{"chars":>8}')
    for g, s in summary.items():
        print(f'{g:22}{s["n"]:>4}{s["top1"]:>8.1%}{s["r5"]:>8.1%}{s["p50_ms"]:>9.0f}{s["p95_ms"]:>9.0f}{s["avg_chars"]:>8.0f}')

    neg = [x for x in rows if x["kind"] == "negative"]
    neg_pass = sum(1 for x in neg if x["hit1"])
    neg_ok = f"{neg_pass}/{len(neg)}"
    print(f"\n负例（判空正确）: {neg_ok}   neg_dense={neg_gate}")

    miss = [x for x in rows if not x["hit1"]]
    print(f"\nMISS: {len(miss)}")
    for m in miss:
        got = m["top5"][0] if m["top5"] else None
        tail = f'→ 实际命中 sim={got["dense_sim"]} {got["text"][:38]}' if got else "→ 空结果"
        print(f'  {m["id"]:5} {m["scope"]}/{m["kind"]:8} {m["q"][:22]:24} {tail}')

    now = datetime.now()
    payload = {"run_at": now.isoformat(timespec="seconds"), "at": now.isoformat(timespec="seconds"),
               "set": set_name, "n": len(rows), "policy": policy,
               "scope": getattr(args, "scope", "auto"),
               "arms": arms or policy.get("arms"), "neg_dense": neg_gate, "neg_pass": neg_ok,
               "all": summary.get("ALL"), "groups": {g: s for g, s in summary.items() if g != "ALL"},
               "miss": [{"id": m["id"], "scope": m["scope"], "kind": m["kind"], "q": m["q"],
                         "expect": m["expect"], "dense_sim": m["dense_sim"],
                         "got": m["top5"][0]["text"] if m["top5"] else "",
                         "got_id": m["top5"][0]["id"] if m["top5"] else "",
                         "top5": m["top5"]} for m in miss],
               "summary": summary, "rows": rows}
    out = Path(args.json) if args.json else ROOT / "data" / f'memex-eval-{now:%Y%m%d-%H%M}-{set_name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwritten", out)

    if not args.no_md:
        docs = ROOT / "docs"
        docs.mkdir(exist_ok=True)
        md = docs / f"回归-{now:%Y%m%d}.md"
        block = render_md(now, set_name, rows, summary, policy, arms, neg_gate, neg_ok)
        header = "" if md.exists() else f"# memex 回归报表 {now:%Y-%m-%d}（按天累积）\n\n"
        with md.open("a", encoding="utf-8") as f:
            f.write(header + block + "\n---\n\n")
        print("written", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

