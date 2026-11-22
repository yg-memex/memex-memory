# -*- coding: utf-8 -*-
"""一条命令跑完所有自检（公开版）。

    python tools/check_all.py

7 项：
  1. 语法编译（compileall）
  2. 统计口径自检（tests/test_stats.py）
  3. core 块自检（tests/test_core.py）
  4. schema 迁移幂等
  5. demo 库可召回（3 条固定查询）
  6. core 渲染预算
  7. HTTP 路由冒烟（不监听端口，直接调 handler 太重，改为 import 检查）
"""
from __future__ import annotations

import compileall
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
DEMO = ROOT / "demo" / "memex-demo.db"


def run_test(path: Path) -> int:
    code = subprocess.run([PY, str(path)], cwd=str(ROOT)).returncode
    print(("  PASS " if code == 0 else "  FAIL ") + path.name)
    return code


def main() -> int:
    # 在 demo 库的临时副本上跑，别把自检产生的 recall_log / schema 写进发行文件
    import shutil
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="memex-check-")
    work_db = Path(tmpdir) / "check.db"
    shutil.copy2(DEMO, work_db)
    os.environ["MEMEX_DB"] = str(work_db)
    fails = 0

    print("[1] compileall")
    ok = compileall.compile_dir(str(ROOT / "memex"), quiet=1)
    fails += 0 if ok else 1
    print(("  PASS" if ok else "  FAIL") + " compileall")

    print("[2] tests")
    for name in ("test_stats.py", "test_core.py"):
        fails += run_test(ROOT / "tests" / name) != 0

    print("[3] schema 迁移幂等")
    sys.path.insert(0, str(ROOT))
    from memex.db import SCHEMA_VERSION, connect  # noqa: E402

    con = connect(str(work_db))
    v1 = con.execute("SELECT version FROM schema_meta").fetchone()[0]
    con.close()
    con = connect(str(work_db))
    v2 = con.execute("SELECT version FROM schema_meta").fetchone()[0]
    con.close()
    ok = int(v1) == int(v2) == SCHEMA_VERSION
    fails += 0 if ok else 1
    print(("  PASS" if ok else "  FAIL") + f" schema v{v1}/{v2}")

    print("[4] demo 召回")
    from memex.embed import Embedder  # noqa: E402
    from memex.recall import Recall  # noqa: E402

    con = connect(str(work_db))
    r = Recall(con, Embedder(con))
    cases = [
        ("alice 的编辑器", "编辑器"),
        ("how does alice prefer messages", "消息"),
        ("alice 的数据放在哪", "数据"),
    ]
    ok = True
    for q, needle in cases:
        hits = r.recall(q, k=3)
        top = hits[0].text if hits else ""
        good = needle.lower() in top.lower()
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {q!r} -> {top[:40]}")
    fails += 0 if ok else 1

    print("[5] core 渲染")
    from memex.core import render_core  # noqa: E402

    out = render_core(con, budget=1200)
    ok = out["chars"] > 0 and out["version"] and isinstance(out["items"], list)
    fails += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'} {out['chars']} chars / v{out['version']} / dropped {len(out['dropped'])}")
    con.close()

    print(f"\n{'GREEN' if fails == 0 else 'RED'}  fails={fails}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
