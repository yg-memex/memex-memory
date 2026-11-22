# -*- coding: utf-8 -*-
"""生成演示库 demo/memex-demo.db（全合成数据，无个人内容）。

    MEMEX_EMBED_BASE=... MEMEX_EMBED_KEY=... python tools/make_demo.py

需要一个可用的 OpenAI 兼容 embedding 端点（本机 ollama 即可）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memex.core import core_pin  # noqa: E402
from memex.db import connect  # noqa: E402
from memex.embed import Embedder  # noqa: E402
from memex.write import Writer  # noqa: E402

FACTS = [
    ("alice", "Alice 是一名后端工程师，主要写 Python 和 Go。", 0.9, "identity"),
    ("alice", "Alice 用 VS Code 写 Python，主题是 solarized light。", 0.9, "preference"),
    ("alice", "Alice 的消息偏好：短句、直接、不要铺垫。", 0.9, "preference"),
    ("alice", "Alice 的项目数据一律放 D 盘项目目录，不堆 C 盘。", 0.8, "preference"),
    ("alice", "Alice 的办公机器是 Windows 11，编辑器之外不做定制。", 0.6, "fact"),
    ("alice", "Alice 讨厌软件自动更新弹窗，能关就关。", 0.7, "preference"),
    ("alice", "Alice 每周五下午整理一次笔记，用 Obsidian。", 0.6, "fact"),
    ("memex", "memex 是一个本地单文件 SQLite 记忆库，召回 = dense + BM25 + 图臂融合。", 0.9, "fact"),
    ("memex", "memex 的评测口径必须带 --scope none，否则数字不可比。", 0.8, "rule"),
    ("memex", "memex 的写入口一律走 Writer，禁止直接 INSERT memories。", 0.8, "rule"),
    ("memex", "memex 支持 as_of 时间旅行查询，只对 facts 承诺。", 0.7, "fact"),
    ("infra", "本机 embedding 走 Ollama 的 bge-m3，重排可选 bge-reranker-v2-m3。", 0.8, "fact"),
    ("infra", "所有服务只绑 127.0.0.1，不对外暴露端口。", 0.8, "rule"),
]

PINS = {
    "Alice 是一名后端工程师": "identity",
    "Alice 的消息偏好": "preference",
    "Alice 用 VS Code": "preference",
    "Alice 的项目数据": "project",
    "memex 的评测口径": "rule",
    "memex 支持 as_of": "project",
}


def main() -> int:
    db = ROOT / "demo" / "memex-demo.db"
    db.parent.mkdir(exist_ok=True)
    if db.exists():
        db.unlink()
    con = connect(str(db))
    w = Writer(con, Embedder(con))
    ids = []
    for scope, text, imp, kind in FACTS:
        mid = w.add(text, scope=scope, kind=kind, importance=imp, source="demo")
        ids.append((text, mid))
    for text, mid in ids:
        for k, slot in PINS.items():
            if k in text:
                core_pin(con, mid, slot)
    # 预热：把 check_all / 示例 / 演示金标的查询也写进缓存，这样离线也能跑
    emb = Embedder(con)
    qs = ["alice 的编辑器", "how does alice prefer messages", "alice 的数据放在哪"]
    for g in sorted((ROOT / "eval" / "golden").glob("*.jsonl")):
        import json as _json
        for line in g.read_text(encoding="utf-8").splitlines():
            if line.strip():
                qs.append(_json.loads(line)["q"])
    # 召回入口会对 query 做 strip_intent，缓存键按归一化后的文本算，这里两种都预热
    from memex.recall import strip_intent
    emb.embed_many(qs + [strip_intent(q) for q in qs])
    con.commit()
    print(f"demo ok: {db} ({db.stat().st_size // 1024} KB, {len(ids)} facts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
