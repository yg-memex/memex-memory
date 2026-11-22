# -*- coding: utf-8 -*-
"""统计口径的最小自检（无 pytest，直接 python tests/test_stats.py 跑）。

覆盖：
1. McNemar 精确检验的已知值（5/2 → p=0.4531）；
2. bootstrap CI 确定性（同 seed 两次一致）；
3. MRR 不低于 Top1（存档截断下的兜底逻辑）；
4. 配对对齐只取两边都有的题。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memex.eval_lib import (  # noqa: E402
    bootstrap_ci,
    judge_change,
    mcnemar_exact,
    mrr,
    paired_rows,
)


def rows(spec):
    """spec: [(id, hit1, hit5)] → 评测行。"""
    out = []
    for i, h1, h5 in spec:
        out.append({"id": i, "scope": "facts", "kind": "lexical", "q": "q",
                    "expect": ["x"], "hit1": h1, "hit5": h5, "top5": []})
    return out


def test_mcnemar_known():
    a = rows([(f"q{i}", True, True) for i in range(7)] + [(f"z{i}", False, True) for i in range(5)])
    b = rows([(f"q{i}", True, True) for i in range(7)] + [(f"z{i}", True, True) for i in range(5)])
    r = mcnemar_exact(a, b)
    assert r["b01"] == 5 and r["b10"] == 0, r
    # 5/0 的精确双侧 p = 2 * (1/32) = 0.0625
    assert abs(r["p"] - 0.0625) < 1e-9, r


def test_bootstrap_deterministic():
    a = rows([(f"q{i}", False, False) for i in range(10)])
    b = rows([(f"q{i}", i < 7, i < 7) for i in range(10)])
    c1 = bootstrap_ci(a, b)
    c2 = bootstrap_ci(a, b)
    assert c1 == c2, (c1, c2)
    assert c1["delta"] == 0.7, c1


def test_mrr_not_below_top1():
    a = rows([("q1", True, True), ("q2", False, True), ("q3", False, False)])
    assert mrr(a) >= sum(r["hit1"] for r in a) / len(a)
    jc = judge_change(a, a)
    assert jc["p"] == 1.0 and jc["delta"] == 0.0


def test_pairing_alignment():
    a = rows([("q1", True, True), ("q2", True, True)])
    b = rows([("q2", False, True), ("q3", True, True)])
    assert [x["id"] for x, _ in paired_rows(a, b)] == ["q2"]



def test_perm_test_rr_identical_is_p1():
    from memex.eval_lib import perm_test_rr
    data = [{"id": f"x{i}", "kind": "fact", "hit1": True, "rr": 1.0} for i in range(10)]
    r = perm_test_rr(data, data, iters=500, seed=1)
    assert r["p"] == 1.0, r
    assert r["delta"] == 0.0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} 项全过")
