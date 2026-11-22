# -*- coding: utf-8 -*-
"""备份/恢复：SQLite 在线备份，不锁库、不中断召回。

用法：
  python -m memex.backup                 # 备份一次，保留最近 7 份
  python -m memex.backup --list          # 列出现有备份
  python -m memex.backup --restore <文件>  # 恢复（恢复前自动再备份当前库）
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

from .db import DB_PATH, connect


def backup_dir(db_path: Path | str = DB_PATH) -> Path:
    """备份目录。默认 data/backups，但可用 MEMEX_BACKUP_DIR 指到仓库外面。

    2026-11-22 事故：一条误跑的 `git clean -fdx` 把 data/backups（7 份快照）连同库一起删了。
    库能靠数据源重建，快照没了就是没了 —— 所以生产上应该指到仓库外（且不在 .gitignore 的
    覆盖范围里），例如 D:\\memex-backups。
    """
    import os

    env = os.environ.get("MEMEX_BACKUP_DIR", "").strip()
    d = Path(env) if env else Path(db_path).parent / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_backups(db_path: Path | str = DB_PATH) -> list[Path]:
    return sorted(backup_dir(db_path).glob("memex-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)


def backup(db_path: Path | str = DB_PATH, keep: int = 7) -> Path:
    """在线备份（sqlite3 backup API，WAL 也会被折叠进去）。保留最近 keep 份。"""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    out = backup_dir(db_path) / f"memex-{time.strftime('%Y%m%d-%H%M%S')}.db"
    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(out)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    # 校验：备份里行数与主库一致
    with sqlite3.connect(out) as c:
        n_out = c.execute("SELECT count(*) FROM memories").fetchone()[0]
    with sqlite3.connect(db_path) as c:
        n_src = c.execute("SELECT count(*) FROM memories").fetchone()[0]
    if n_out != n_src:
        raise RuntimeError(f"备份校验失败：backup={n_out} source={n_src}")
    for old in list_backups(db_path)[keep:]:
        old.unlink(missing_ok=True)
    return out


def restore(backup_path: Path | str, db_path: Path | str = DB_PATH) -> Path:
    """用备份覆盖当前库；先把当前库再备份一份（safety-*），方便反悔。"""
    backup_path, db_path = Path(backup_path), Path(db_path)
    if not backup_path.exists():
        raise FileNotFoundError(backup_path)
    safety = None
    if db_path.exists():
        safety = backup_dir(db_path) / f"memex-pre-restore-{time.strftime('%Y%m%d-%H%M%S')}.db"
        src = sqlite3.connect(db_path)
        dst = sqlite3.connect(safety)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    # 清掉 WAL/SHM，避免旧页残留
    for suffix in ("-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    shutil.copy2(backup_path, db_path)
    con = connect(db_path)          # 触发 migrate，确保结构对齐
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.commit()
    con.close()
    return safety or Path("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--keep", type=int, default=7)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", metavar="FILE")
    args = ap.parse_args()

    if args.list:
        for p in list_backups(args.db):
            print(f"{p.name}\t{p.stat().st_size/1024:.0f}KB")
        return 0
    if args.restore:
        safety = restore(args.restore, args.db)
        print(f"restored from {args.restore}; safety copy = {safety or '(none)'}")
        return 0
    out = backup(args.db, keep=args.keep)
    print(f"backup ok: {out} ({out.stat().st_size/1024:.0f}KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
