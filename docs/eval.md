# Evaluation methodology

The point of this project is that "it got better" is a claim you can test.

## Sets

| Set | n | Purpose |
|---|---|---|
| `dev` | 120 (98 positive + 22 negatives) | daily gate |
| `holdout v2` | 186 | harder, notes-heavy |
| `temporal` | 15 | `as_of` queries against superseded facts |

Goldens are JSONL: `{id, q, expect, scope, ...}`. Negatives have `expect: null` and must return no confident hit.

## Metrics

- **Top1** — first hit contains the expected string (case-insensitive substring).
- **R@5** — any of the top 5 does.
- **Negatives** — pass only if no hit clears the negative threshold.
- **MRR** — reciprocal rank computed from **full** text of the hit (not the 160-char cache), so cross-run comparisons are not biased.
- **p50/p95 latency** — always reported; a gate above 300 ms (rerank) / 50 ms (no rerank) fails.

All reported numbers must state the denominator (`--scope none`, all questions vs positives only). Mixing denominators was the single most confusing bug in our own history; `check_all` and `eval_stats` now print both.

## Decision rule (pre-registered)

A change is only "better" if **all** hold:

1. McNemar exact p < 0.05;
2. bootstrap 95% CI lower bound > 0;
3. no significant regression on the other set;
4. negatives all pass;
5. p95 within budget.

On 120 questions the single-set CI is ±6.6 pt, so anything below ~5 pt must be called noise. If you want to detect smaller effects, grow the golden set — not the confidence.

## Reproduce

```bash
python -m memex.eval_run --set dev --scope none
python tools/eval_stats.py --a results-a.json --b results-b.json --key hit1
```

`eval_stats` prints Δ, CI, McNemar p, per-question flips and a verdict. Negative results belong in the repo too: `docs/` in the private repo keeps a "rejected routes" ledger, and every rejected experiment lists its pre-registered threshold.
