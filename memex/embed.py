# -*- coding: utf-8 -*-
"""Embedding 后端：任意 OpenAI 兼容的 /embeddings 服务 + SQLite 缓存。

默认指向本机 Ollama（http://127.0.0.1:11434/v1，模型 bge-m3），
也可以指向任何兼容端点（硅基流动、vLLM、LM Studio、OpenAI …）。

环境变量：
  MEMEX_EMBED_BASE     /embeddings 的 base（含 /v1），默认 http://127.0.0.1:11434/v1
  MEMEX_EMBED_MODEL    模型名，默认 bge-m3
  MEMEX_EMBED_KEY      API key；本机 ollama 随便填，默认 "ollama"
  MEMEX_EMBED_TIMEOUT  单次请求超时秒，默认 60
  MEMEX_EMBED_BATCH    每批条数，默认 32

缓存键是文本的 sha1，送进 API 的必须是原文（不是 hash）。
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from .db import blob_to_vec, text_hash, vec_to_blob


class Embedder:
    def __init__(self, con, batch: int = 32):
        self.base = os.environ.get("MEMEX_EMBED_BASE", "http://127.0.0.1:11434/v1").rstrip("/")
        self.model = os.environ.get("MEMEX_EMBED_MODEL", "bge-m3")
        self.key = os.environ.get("MEMEX_EMBED_KEY", "ollama")
        self.timeout = float(os.environ.get("MEMEX_EMBED_TIMEOUT", "60"))
        self.batch = int(os.environ.get("MEMEX_EMBED_BATCH", batch))
        self.con = con

    # ---- HTTP ----
    def _post(self, texts: list[str]) -> list[np.ndarray]:
        payload = json.dumps(
            {"model": self.model, "input": texts, "encoding_format": "float"}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/embeddings",
            data=payload,
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as f:
            data = json.loads(f.read().decode("utf-8"))
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [np.asarray(r["embedding"], dtype=np.float32) for r in rows]

    # ---- 公开接口 ----
    def embed_many(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        missing: dict[str, str] = {}
        where: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            h = text_hash(t)
            row = self.con.execute("SELECT vec FROM embed_cache WHERE h=?", (h,)).fetchone()
            if row:
                out[i] = blob_to_vec(row["vec"])
            else:
                missing.setdefault(h, t)
                where.setdefault(h, []).append(i)

        if missing:
            hashes = list(missing)
            for start in range(0, len(hashes), self.batch):
                chunk = hashes[start:start + self.batch]
                vecs = self._post([missing[h] for h in chunk])
                for h, v in zip(chunk, vecs):
                    for i in where[h]:
                        out[i] = v
                    self.con.execute(
                        "INSERT OR REPLACE INTO embed_cache(h, vec) VALUES(?,?)",
                        (h, vec_to_blob(v)),
                    )
            self.con.commit()
        return np.stack(out)  # type: ignore[arg-type]

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def dim(self) -> int:
        row = self.con.execute("SELECT vec FROM embed_cache LIMIT 1").fetchone()
        return len(blob_to_vec(row["vec"])) if row else 1024
