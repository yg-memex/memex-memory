# -*- coding: utf-8 -*-
"""memex 内核 v0：SQLite 主存 + 多臂召回（见 recall.py）。

设计要点（博采众长，见 docs/设计.md）：
- 存储：SQLite 单文件，向量存 BLOB(float32)，事务安全、可单独备份；Markdown 只做镜像。
- 召回：多臂各自 topN → RRF 融合 → MMR 去重。权重/参数存 policy 表，可版本化、可回滚。
- 演化：每条记忆带 importance / stability / decay_tier / last_access，供「用得勤决定寿命」计算。
- 迁移：schema_meta.version 单调递增；connect() 自动 migrate()，旧库无需手工处理。
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memex.db"

SCHEMA_VERSION = 3  # v3：core 块 pinned / core_slot

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,                -- facts | notes
    kind TEXT NOT NULL DEFAULT 'fact',  -- fact | note | episode | profile | rule
    text TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    stability REAL NOT NULL DEFAULT 1.0,      -- FSRS 记忆稳定度（天）
    decay_tier INTEGER NOT NULL DEFAULT 30,   -- 7/14/30/60/180/250/360, 0=永固
    created_at INTEGER NOT NULL,
    last_access INTEGER NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',    -- active | superseded | archived
    superseded_by TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    source_id TEXT NOT NULL DEFAULT '',       -- 上游 id（增量同步用）
    updated_at INTEGER NOT NULL DEFAULT 0,    -- 上游最后变更时间
    meta TEXT NOT NULL DEFAULT '{}',
    pinned INTEGER NOT NULL DEFAULT 0,     -- 1=进 core 块
    core_slot TEXT NOT NULL DEFAULT '',     -- identity|rule|preference|project|other
    vec BLOB
);
CREATE TABLE IF NOT EXISTS edges (
    a TEXT NOT NULL, b TEXT NOT NULL, weight REAL NOT NULL, kind TEXT NOT NULL,
    PRIMARY KEY (a, b, kind)
);
CREATE TABLE IF NOT EXISTS embed_cache (
    h TEXT PRIMARY KEY, vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS policy (
    version INTEGER PRIMARY KEY, created_at INTEGER NOT NULL, params TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS recall_log (
    trace_id TEXT PRIMARY KEY, ts INTEGER NOT NULL, query TEXT NOT NULL,
    policy_version INTEGER, hits TEXT NOT NULL, cited INTEGER
);
CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope, status);
"""


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}


def migrate(con: sqlite3.Connection) -> int:
    """把旧库补到 SCHEMA_VERSION。幂等，可在每次 connect() 安全调用。"""
    cols = _columns(con, "memories")
    added = []
    for col, ddl in (
        ("source_id", "TEXT NOT NULL DEFAULT ''"),
        ("updated_at", "INTEGER NOT NULL DEFAULT 0"),
        ("valid_from", "INTEGER NOT NULL DEFAULT 0"),
        ("valid_to", "INTEGER NOT NULL DEFAULT 0"),
        ("superseded_by_id", "TEXT NOT NULL DEFAULT ''"),
        ("pinned", "INTEGER NOT NULL DEFAULT 0"),
        ("core_slot", "TEXT NOT NULL DEFAULT ''"),
    ):
        if col not in cols:
            con.execute(f"ALTER TABLE memories ADD COLUMN {col} {ddl}")
            added.append(col)
    # 旧行没填 valid_from 的按 created_at 补；新写入由 write.py 直接写 now（幂等）
    con.execute("UPDATE memories SET valid_from=created_at WHERE valid_from=0")
    con.execute("CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(scope, source_id)")
    row = con.execute("SELECT version FROM schema_meta").fetchone()
    if row is None:
        con.execute("INSERT INTO schema_meta(version) VALUES(?)", (SCHEMA_VERSION,))
    else:
        con.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))
    con.commit()
    return SCHEMA_VERSION


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    if path is None:
        path = os.environ.get("MEMEX_DB") or DB_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    migrate(con)
    return con


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def new_id() -> str:
    return uuid.uuid4().hex


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ---------- 打分：用得勤决定寿命 ----------

def activation(importance: float, stability_days: float, last_access: int, now: int | None = None) -> float:
    """FSRS 式激活值：importance × exp(-Δt / S)。S 越大衰减越慢，检索命中后 S 增长。"""
    now = now or int(time.time())
    dt_days = max(0.0, (now - int(last_access or now)) / 86400.0)
    s = max(0.05, float(stability_days))
    return float(importance) * math.exp(-dt_days / s)


def half_life_days(decay_tier: int) -> float:
    """档位 → 半衰期。0 = 永固（用极大值表示）。"""
    return 3650.0 if not decay_tier else float(decay_tier) / 2.0
