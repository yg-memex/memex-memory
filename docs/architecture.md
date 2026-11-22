# 架构

memex 刻意做小：一个进程、一个 SQLite 文件，不需要常驻后台服务。

## 数据模型

| 表 | 存什么 |
|---|---|
| `memories` | facts 和 notes：`text`、`scope`、`kind`、`importance`、向量 `vec`、`status`（`active`/`superseded`/`archived`）、`valid_from`、`valid_to`、`superseded_by_id`、`pinned`、`core_slot` |
| `edges` | 记忆之间的带权连接（同 scope / 同来源 / supersede 链），供 graph 臂使用 |
| `embed_cache` | `sha1(text) → vector`，重复 embedding 不花时间也不花钱 |
| `recall_log` / `recall_hits` | 每次召回及其命中、分数、策略版本——复放和反馈的基础 |
| `schema_meta` | schema 版本；迁移幂等 |

schema 为 v3。`as_of` 召回按 `valid_from <= t AND (valid_to = 0 OR valid_to > t)` 过滤 facts。notes 目前没有时间链：如实标记为 unsupported，不猜。

## 召回流水线

1. **各臂。** dense（对缓存向量做 cosine）、BM25（用 Python 实现 SQLite 风格的打分，不引入额外服务）、graph（从 top 种子做带权扩展）、可选重排（外部 HTTP cross-encoder）。
2. **融合。** 各臂先归一化，再按策略加权求和（`w_dense`、`w_bm25`、`w_graph`；`w_act` 默认关闭，因为实测激活强度与查询无关）。
3. **重排。** 可选。候选池 20 条、280ms 截止，超时自动降级回词法打分。默认关闭；开启是改策略，不是改代码。
4. **输出。** 命中带 `id/text/score/dense_sim/arms`；每次召回写一个 `trace_id`，之后可以复放。

## 写入路径

`Writer.add` 在同一个 scope 内按余弦去重（> 0.92 合并），没给 importance 时自动算，并记录来源（`source`）。`supersede()` 保留旧行，写入 `valid_to`/`superseded_by_id`，并把 `pinned`/`core_slot` 标记迁到后继——core 块永远不能指向一条死行。

## core 块

`render_core(con, budget=1200)` 按槽位顺序（`identity → rule → preference → project → other`）pinned 优先排序，在字符预算内贪心填充，装不下的进 `dropped` 并带原因。不做静默截断；`version` 是 `(budget, id:text …)` 的哈希，方便消费方缓存。

## 进程

- `python -m memex.http` —— HTTP API + 面板，只监听 127.0.0.1。
- `python -m memex.server_mcp` —— MCP stdio 服务（需要 `pip install -e ".[mcp]"`）。
- 其余都是一次库调用。
