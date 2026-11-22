"""core 块自检（无 pytest，直接 python tests/test_core.py）。

覆盖：渲染预算、超预算 dropped 不截断、pin/unpin、supersede 时 pin 位迁移。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memex import core                      # noqa: E402
from memex.db import connect                # noqa: E402
from memex.write import Writer              # noqa: E402

import numpy as np                         # noqa: E402


class _FakeEmb:
    """确定性假向量：单测不联网、不需要 embedding 服务。"""

    def _vec(self, t: str) -> np.ndarray:
        import hashlib
        d = hashlib.sha256(t.encode("utf-8")).digest() * 4   # 128 bytes
        v = np.frombuffer(d, dtype=np.uint8).astype(np.float32) + 1.0
        return v / np.linalg.norm(v)

    def embed_many(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_one(self, text):
        return self._vec(text)

    def dim(self):
        return 128


def _fresh():
    con = connect(":memory:")
    return con, Writer(con, _FakeEmb())


def test_budget_counts():
    con, w = _fresh()
    a = w.add("甲" * 100, "facts", "fact", "t")
    core.core_pin(con, a, "rule")
    r = core.render_core(con, budget=1200)
    assert r["chars"] <= 1200, r["chars"]
    assert r["version"] and len(r["version"]) == 8, r["version"]
    assert r["items"], "pinned 条必须进"
    con.close()


def test_dropped_not_truncated():
    con, w = _fresh()
    a = w.add("甲" * 400, "facts", "fact", "t")
    core.core_pin(con, a, "rule")
    r = core.render_core(con, budget=200)
    assert r["over_budget"] is True, "pinned 超预算要如实报"
    assert r["dropped"] == [], "pinned 不能丢"
    assert r["items"][0]["chars"] == 400, "不截断，照实报"

    # 挂了槽位但没 pin 的候选：装不下进 dropped，并带 reason
    b = w.add("乙" * 300, "facts", "fact", "t2")
    con.execute("UPDATE memories SET core_slot='other' WHERE id=?", (b,))
    con.commit()
    r2 = core.render_core(con, budget=450)
    assert r2["over_budget"] is False
    assert [d["id"] for d in r2["dropped"]] == [b], r2["dropped"]
    assert all(d["reason"] for d in r2["dropped"]), "dropped 必须带 reason，不能静默丢"
    assert b not in [i["id"] for i in r2["items"]]
    con.close()


def test_unpin():
    con, w = _fresh()
    a = w.add("丙" * 10, "facts", "fact", "t")
    core.core_pin(con, a, "project")
    assert core.render_core(con)["items"], "pin 后应有内容"
    core.core_unpin(con, a)
    assert core.render_core(con)["items"] == [], "全 unpin 后 core 应为空"
    con.close()


def test_pin_moves_on_supersede():
    con, w = _fresh()
    old = w.add("测试事实甲：pin 迁移用，内容与乙刻意不同。", "facts", "fact", "t")
    new = w.add("另一条无关事实乙：验证 supersede 时 pin 位跟着搬。", "facts", "fact", "t")
    assert old != new
    core.core_pin(con, old, "project")
    w.supersede(old, new)
    items = [i["id"] for i in core.render_core(con)["items"]]
    assert old not in items, "旧条不能留在 core"
    assert new in items, "新条必须接住 pin"
    row = con.execute("SELECT pinned, core_slot FROM memories WHERE id=?", (new,)).fetchone()
    assert (row[0], row[1]) == (1, "project"), (row[0], row[1])
    con.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
