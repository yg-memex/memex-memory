# -*- coding: utf-8 -*-
"""策略文件 I/O（M5 自演化专用）：`data/active_policy.json` 的读写与校验。

为什么单独一个文件：`recall.py` 属 M1、`evolve.py` 属 M5，两边都要「当前生效的参」。
把读盘与校验放这里，`recall.py` 只需要一行 `from .policy_io import load_active`，
M5 就不用去改 M1 的打分逻辑（铁律 5：一人一模块）。

契约（§4.2）：权重和必须 = 1.0，否则整份策略作废、回落 DEFAULT_POLICY。
`load_active()` **永不抛异常**——策略文件坏了也不能让召回挂掉。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_PATH = ROOT / "data" / "active_policy.json"
ENV_OVERRIDE = "MEMEX_ACTIVE_POLICY"     # 测试用：指向临时策略文件，不污染生产
WEIGHT_KEYS = ("w_dense", "w_bm25", "w_act", "w_graph")


def active_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_OVERRIDE)
    return Path(env) if env else ACTIVE_PATH


def normalize_weights(params: dict) -> dict:
    """把现有权重等比归一化到和为 1，返回新 dict。"""
    p = dict(params)
    keys = [k for k in WEIGHT_KEYS if k in p]
    tot = sum(float(p[k]) for k in keys)
    if tot <= 0:
        raise ValueError("权重和必须为正数")
    for k in keys:
        p[k] = round(float(p[k]) / tot, 6)
    # 补掉四舍五入的尾差，保证严格 = 1.0
    if keys:
        p[keys[0]] = round(1.0 - sum(float(p[k]) for k in keys[1:]), 6)
    return p


def validate(params: dict) -> dict:
    """校验并返回可直接使用的策略 dict（丢弃 version）。不合法抛 ValueError。"""
    if not isinstance(params, dict):
        raise ValueError("策略必须是 dict")
    p = {k: v for k, v in params.items() if k != "version"}
    keys = [k for k in WEIGHT_KEYS if k in p]
    if keys:
        for k in keys:
            if not isinstance(p[k], (int, float)) or float(p[k]) < 0:
                raise ValueError(f"{k} 必须是非负数")
        s = sum(float(p[k]) for k in keys)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"权重和 {s:.6f} != 1.0")
    return p


def load_active(path: Path | str | None = None) -> dict:
    """读当前生效策略；文件缺失/损坏/权重不合法 → 返回 {}（调用方回落 DEFAULT_POLICY）。"""
    p = active_path(path)
    try:
        if not p.exists():
            return {}
        return validate(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return {}


def read_active_meta(path: Path | str | None = None) -> dict:
    """带 version 的完整内容（evolve 需要知道现在是第几版）。读不到返回 {}。"""
    p = active_path(path)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        validate(d)
        return d
    except Exception:
        return {}


def write_active(version: int, params: dict, path: Path | str | None = None) -> Path:
    """原子写 active_policy.json（先写 .tmp 再 replace，避免读到半个文件）。"""
    p = active_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"version": int(version), **validate(params)}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p
