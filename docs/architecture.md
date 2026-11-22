# Architecture

memex is deliberately small: one process, one SQLite file, no background daemons required.

## Data model

| Table | Holds |
|---|---|
| `memories` | facts and notes: `text`, `scope`, `kind`, `importance`, vector `vec`, `status` (`active`/`superseded`/`archived`), `valid_from`, `valid_to`, `superseded_by_id`, `pinned`, `core_slot` |
| `edges` | weighted links between memories (same scope / source / supersede chain), used by the graph arm |
| `embed_cache` | `sha1(text) → vector`, so re-embedding is free |
| `recall_log` / `recall_hits` | every recall with its hits, scores and policy version — the basis for trace replay and feedback |
| `schema_meta` | schema version; migrations are idempotent |

Schema v3. `as_of` recall filters facts with `valid_from <= t AND (valid_to = 0 OR valid_to > t)`. Notes currently have no time chain: they are honestly reported as unsupported instead of guessed.

## Retrieval pipeline

1. **Arms.** Dense (cosine over cached embeddings), BM25 (SQLite FTS-style scoring in Python, no extra service), graph (weighted expansion from top seeds), optional rerank (external HTTP cross-encoder).
2. **Fusion.** Per-arm normalization, then weighted sum from the policy (`w_dense`, `w_bm25`, `w_graph`; `w_act` off by default because activation proved query-independent).
3. **Rerank.** Optional. Pool of 20 candidates, 280 ms deadline, automatic fallback to lexical scoring on timeout. Off by default; enabling it is a policy change, not a code change.
4. **Output.** Hits carry `id/text/score/dense_sim/arms`; every recall writes a `trace_id` so a result can be replayed later.

## Write path

`Writer.add` deduplicates against existing vectors in the same scope (cos > 0.92 merges), computes importance when not supplied, and records provenance (`source`). `supersede()` keeps the old row, stamps `valid_to`/`superseded_by_id`, and moves any `pinned`/`core_slot` marker to the successor — the core block must never point at a dead row.

## Core block

`render_core(con, budget=1200)` sorts pinned-first by slot order (`identity → rule → preference → project → other`), greedily fills up to the character budget, and reports what did not fit in `dropped` with a reason. Nothing is silently truncated; `version` is a hash of `(budget, id:text …)` so consumers can cache.

## Processes

- `python -m memex.http` — HTTP API + panel, binds 127.0.0.1 only.
- `python -m memex.server_mcp` — MCP stdio server (needs `pip install -e ".[mcp]"`).
- Everything else is a library call.
