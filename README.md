# memex

[![ci](https://github.com/2083472126zt/memex-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/2083472126zt/memex-memory/actions/workflows/ci.yml)

**Local-first, verifiable memory for AI agents.** One SQLite file. No cloud, no vendor lock-in, no hidden prompts.

> Status: **alpha (v0.1.0a1)**. The API surface is small and stable enough to use; the schema is versioned and self-migrating. Expect changes between minor versions.

Agents forget everything between sessions. memex is the smallest thing that fixes it without becoming a second brain you can't audit:

- **Store facts, not transcripts.** Each memory is a short, self-contained sentence you can read, edit, and delete.
- **Recall is measurable.** Four retrieval arms (dense + BM25 + graph + optional rerank) fused into one ranking. A golden set and a bootstrap/McNemar harness are part of the repo, not an afterthought.
- **History is explicit.** Facts are superseded, never overwritten; `as_of` queries can replay what was true on a given date.
- **A core block is always in context.** Pin rules/preferences and they get injected every session, with a hard budget and no silent truncation (~743 chars in our own deployment).

![memex panel](docs/images/panel.png)

*The panel: core block by slot, budget usage, a recall sandbox, decay curves, and policy/DB state. Everything shown is editable or deletable in place.*

## Quickstart

```bash
pip install -e .                       # numpy only; Python 3.11+
export MEMEX_EMBED_BASE=http://127.0.0.1:11434/v1   # any OpenAI-compatible /embeddings
export MEMEX_EMBED_MODEL=bge-m3
python tools/make_demo.py              # builds demo/memex-demo.db (synthetic data)
python tools/check_all.py              # offline self-check, should print GREEN
MEMEX_DB=demo/memex-demo.db python -m memex.http --port 8901   # panel: http://127.0.0.1:8901/
```

No embedding server? Install Ollama and `ollama pull bge-m3` — that is the intended offline setup. SiliconFlow / vLLM / LM Studio work too.

### As an MCP server

```bash
pip install -e ".[mcp]"
```

```json
{"mcpServers":{"memex":{"command":"python","args":["-m","memex.server_mcp"]}}}
```

Tools: `recall`, `remember`, `stats`, `core`, `core_pin`, `core_unpin`, `memory_get`, `memory_update`, `memory_forget`.

### HTTP

| Method | Path | What |
|---|---|---|
| POST | `/recall` | `{q, k, scope}` → ranked hits with `trace_id`, each hit carries `cited` |
| POST | `/remember` | `{text, scope, kind, source, importance}` |
| GET | `/core?budget=1200` | rendered core block + `dropped` list |
| POST | `/core/pin`, `/core/unpin` | pin a memory into a slot (`identity/rule/preference/project/other`) |
| GET/PUT/DELETE | `/memory/<id>` | read, edit (re-embeds), archive |
| GET | `/traces?limit=20`, `/trace/<id>` | recent recalls / one replay |
| POST | `/feedback` | `{trace_id, hit_ids, used}` — the only real activation-decay signal |

## How it works

```
query ──► dense (bge-m3) ─┐
      ──► BM25 (SQLite)  ─┼─► normalize + weighted fusion ─► optional rerank ─► top-k
      ──► graph (edges)  ─┘
```

- Weights live in `data/active_policy.json` (falls back to `DEFAULT_POLICY`); every change is a file you can diff and roll back.
- The write path (`memex/write.py`) deduplicates by cosine > 0.92, supersedes instead of overwriting, and keeps a `superseded_by` chain.
- Schema is at v3 with idempotent migrations (`memex/db.py`); `as_of` recall filters on `valid_from/valid_to` (facts only).
- Everything is stdlib + numpy. The reranker is an external HTTP process (`bge-reranker-v2-m3`) and is **off by default**.

## Current numbers (measured on our own golden sets)

| Set | Top1 | R@5 | note |
|---|---|---|---|
| dev (120 q, incl. 22 negatives) | **84.2%** | **95.8%** | positive-only: 80.6% / 94.9% (n=98) |
| holdout v2 (186 q) | **72.6%** | **86.0%** | notes-heavy, harder |
| temporal (15 q, `as_of`) | **86.7%** | 100% | vs 6.7% for "current-only" |

Two honest caveats, because this repo is about verification:

1. On 120 questions the 95% CI is roughly ±6.6 pt. Differences under ~5 pt are noise; the repo therefore ships a paired bootstrap + McNemar harness (`tools/eval_stats.py`) and refuses to call a change "better" unless `p < 0.05` **and** the CI lower bound is above zero.
2. The numbers above were produced with a reranker on a specific machine. They are a measured state, not a leaderboard claim.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `MEMEX_DB` | `data/memex.db` (repo-relative) | database path |
| `MEMEX_EMBED_BASE` | `http://127.0.0.1:11434/v1` | OpenAI-compatible embeddings base |
| `MEMEX_EMBED_MODEL` | `bge-m3` | embedding model |
| `MEMEX_EMBED_KEY` | `ollama` | API key (any value for Ollama) |
| `MEMEX_EMBED_TIMEOUT` / `_BATCH` | `60` / `32` | request tuning |
| `MEMEX_GOLDEN_DIR` | `eval/golden` | golden set directory for `eval_run` |

Rerank: set `"use_rerank": true` in `data/active_policy.json` and point `rerank_api_base` at a bge-reranker server. Off by default so a fresh clone runs with zero external services.

## Repo map

```
memex/        library (recall, write, core, db, embed, http, server_mcp, eval)
tools/        make_demo.py, check_all.py, eval_stats.py
tests/        plain-python tests (no pytest needed)
demo/         synthetic demo database
docs/         architecture, evaluation notes, roadmap
panel/        single-file web panel (pin/unpin, edit, delete, trace replay)
examples/     MCP config, curl recipes
```

## Docs

- [Architecture](docs/architecture.md)
- [Evaluation methodology](docs/eval.md)
- [Roadmap](docs/roadmap.md)

## Contributing

Issues and PRs welcome. Two house rules: every behavioral change needs a falsifiable acceptance criterion, and no test may depend on the author's machine or private data.

MIT licensed.
