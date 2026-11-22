# -*- coding: utf-8 -*-
r"""公共数据类型（契约层）。

**任何模块都不许改这里的字段语义**；加字段必须带默认值，删/改字段等于破坏性变更。
见 docs/architecture.md。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Hit:
    """一次召回的单条命中。"""

    id: str
    scope: str                     # facts | notes
    kind: str                      # fact | note | profile | lesson ...
    text: str
    score: float                   # 融合分（RRF 量纲 ≈0.0x~0.1x，**不能**当相似度阈值）
    dense_sim: float               # 与查询的余弦 [0,1]；负例门控 / 重排先看这个
    arms: list[str] = field(default_factory=list)   # 例 ["dense#3", "bm25#7"]
    importance: float = 0.5
    trace_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Memory:
    """库内一条记忆（用于写入/治理模块回读）。"""

    id: str
    scope: str
    kind: str
    text: str
    importance: float = 0.5
    stability: float = 1.0
    decay_tier: int = 30
    created_at: int = 0
    last_access: int = 0
    access_count: int = 0
    status: str = "active"
    superseded_by: str = ""
    source: str = ""
    source_id: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
