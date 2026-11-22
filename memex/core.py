# -*- coding: utf-8 -*-
"""核心记忆块（core）：每次会话开局常驻的那一小块记忆。

设计约束（见 docs/PRD.md R6、docs/提案-schema-v2 与 T06 契约）：
- 硬上限 1200 字（约 900 tokens），**不静默截断**：装不下的进 dropped 并说明原因；
- pinned 的条目必须进（哪怕超预算，也要如实报 over_budget=True）；
- version = 内容 hash，内容一变就变，供面板和 MCP 判断"要不要重新注入"。
"""
from __future__ import annotations

import hashlib
import sqlite3

SLOTS = ("identity", "rule", "preference", "project", "other")
SLOT_TITLES = {
    "identity": "身份",
    "rule": "规则",
    "preference": "偏好",
    "project": "项目",
    "other": "其他",
}
DEFAULT_BUDGET = 1200
DEFAULT_SLOT = "other"


def _rank(slot: str) -> int:
    return SLOTS.index(slot) if slot in SLOTS else len(SLOTS) - 1


def _norm_slot(slot: str) -> str:
    s = (slot or "").strip().lower()
    return s if s in SLOTS else DEFAULT_SLOT


def render_core(con: sqlite3.Connection, budget: int = DEFAULT_BUDGET) -> dict:
    """渲染 core 块。返回 {ok, text, chars, limit, version, items, dropped, over_budget}。"""
    budget = max(200, int(budget or DEFAULT_BUDGET))
    rows = con.execute(
        "SELECT id, kind, text, core_slot, pinned, importance, created_at "
        "FROM memories WHERE status='active' AND (pinned=1 OR core_slot!='')"
    ).fetchall()
    cand = []
    for r in rows:
        cand.append({
            "id": r["id"],
            "slot": _norm_slot(r["core_slot"]),
            "text": (r["text"] or "").strip(),
            "chars": len((r["text"] or "").strip()),
            "pinned": bool(r["pinned"]),
            "importance": float(r["importance"] or 0.5),
            "created_at": int(r["created_at"] or 0),
        })
    cand.sort(key=lambda x: (_rank(x["slot"]), not x["pinned"], -x["importance"], -x["created_at"]))

    items, dropped, chars = [], [], 0
    for c in cand:
        if c["pinned"] or chars + c["chars"] <= budget:
            items.append(c)
            chars += c["chars"]
        else:
            dropped.append({"id": c["id"], "slot": c["slot"], "chars": c["chars"],
                            "reason": "budget"})

    by_slot: dict[str, list[dict]] = {}
    for it in items:
        by_slot.setdefault(it["slot"], []).append(it)
    parts, order = [], []
    for slot in SLOTS:
        group = by_slot.get(slot) or []
        if not group:
            continue
        order.append(slot)
        parts.append(f"【{SLOT_TITLES[slot]}】")
        parts.extend(f"- {it['text']}" for it in group)
    text = "\n".join(parts)

    h = hashlib.sha1()
    h.update(f"{budget}|".encode())
    for it in items:
        h.update(f"{it['id']}:{it['text']}\n".encode())
    return {
        "ok": True,
        "text": text,
        "chars": chars,
        "limit": budget,
        "version": h.hexdigest()[:8],
        "items": [{k: it[k] for k in ("id", "slot", "text", "chars", "pinned")} for it in items],
        "dropped": dropped,
        "over_budget": chars > budget,
        "slots": order,
    }


def core_pin(con: sqlite3.Connection, mid: str, slot: str = DEFAULT_SLOT) -> dict:
    mid = (mid or "").strip()
    if not mid:
        return {"ok": False, "error": "id 不能为空"}
    row = con.execute("SELECT id, status, core_slot, pinned FROM memories WHERE id=?", (mid,)).fetchone()
    if row is None:
        return {"ok": False, "error": f"没有 id={mid} 这条记忆"}
    if row["status"] != "active":
        return {"ok": False, "error": f"id={mid} 当前状态是 {row['status']}，先恢复成 active 再 pin"}
    con.execute("UPDATE memories SET pinned=1, core_slot=? WHERE id=?", (_norm_slot(slot), mid))
    con.commit()
    return {"ok": True, "id": mid, "slot": _norm_slot(slot), "changed": True}


def core_unpin(con: sqlite3.Connection, mid: str) -> dict:
    mid = (mid or "").strip()
    if not mid:
        return {"ok": False, "error": "id 不能为空"}
    row = con.execute("SELECT id, pinned, core_slot FROM memories WHERE id=?", (mid,)).fetchone()
    if row is None:
        return {"ok": False, "error": f"没有 id={mid} 这条记忆"}
    con.execute("UPDATE memories SET pinned=0, core_slot='' WHERE id=?", (mid,))
    con.commit()
    return {"ok": True, "id": mid, "changed": bool(row["pinned"] or row["core_slot"])}
