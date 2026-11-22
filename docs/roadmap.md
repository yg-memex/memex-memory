# Roadmap

## Now (v0.1.0a1)

- [x] Single-file store, schema v3, idempotent migrations
- [x] Four-arm recall with versioned policy and automatic fallback
- [x] Core block with hard budget, `dropped`, and a version hash
- [x] HTTP + MCP + single-file panel
- [x] Trace log, replay, and the feedback endpoint (activation decay)
- [x] Paired significance testing in-repo

## Next

- [ ] Pluggable note chunking (generic markdown, no vault-specific code)
- [ ] A demo video / GIF of the panel pinning a rule and watching it appear in the next session
- [ ] Docker one-liner (`docker run -v $PWD/data:/app/data memex`) — needs a decision on shipping Ollama or requiring an external endpoint
- [ ] More goldens: 400-question set for detecting ≤3 pt effects
- [ ] Optional local cross-encoder path (ONNX) so rerank works without an extra HTTP process

## Not planned

Multi-tenant, hosted cloud, vector database dependency, or auto-summarising memories away from their source text. If you need those, use a different project.
