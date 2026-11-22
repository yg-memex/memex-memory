# memex

[![ci](https://github.com/yg-memex/memex-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/yg-memex/memex-memory/actions/workflows/ci.yml)

**本地优先、可验证的 AI 记忆层。** 一个 SQLite 文件。不联网、不绑供应商、没有藏在提示词里的暗逻辑。

> 状态：**alpha（v0.1.0a1）**。API 面很小，够用；schema 带版本、自动迁移。小版本之间接口可能变。

Agent 一换会话就失忆。memex 是修这个问题最小的东西——它不会长成你审不了的「第二大脑」：

- **存事实，不存聊天记录。** 每条记忆是一句能读懂、能改、能删的独立事实。
- **召回是可度量的。** 四路召回（dense + BM25 + graph + 可选重排）融合成一个排序；金标集和 bootstrap/McNemar 显著性工具就在仓库里，不是事后补的。
- **历史是显式的。** 事实只会被 supersede（取代），不会被覆盖；`as_of` 能回放某一天「当时是什么说法」。
- **core 块常驻上下文。** 规则/偏好 pin 进去，每次会话自动注入，有硬预算、超了如实报 `dropped`，不静默截断（我们自己部署时约 743 字）。

![memex 面板](docs/images/panel.png)

*面板：按槽位分组的 core 块、预算占用、召回沙盒、衰减曲线、策略与库状态。界面上看到的都能就地改或删。*

## 快速开始

```bash
pip install -e .                       # 只依赖 numpy，Python 3.11+
export MEMEX_EMBED_BASE=http://127.0.0.1:11434/v1   # 任何 OpenAI 兼容 /embeddings
export MEMEX_EMBED_MODEL=bge-m3
python tools/make_demo.py              # 生成 demo/memex-demo.db（全合成数据）
python tools/check_all.py              # 离线自检，应输出 GREEN
MEMEX_DB=demo/memex-demo.db python -m memex.http --port 8901   # 面板：http://127.0.0.1:8901/
```

没有 embedding 服务？装 Ollama 再 `ollama pull bge-m3`，这就是推荐的本地方案。硅基流动 / vLLM / LM Studio 也都能用。

### 作为 MCP 服务

```bash
pip install -e ".[mcp]"
```

```json
{"mcpServers":{"memex":{"command":"python","args":["-m","memex.server_mcp"]}}}
```

工具：`recall`、`remember`、`stats`、`core`、`core_pin`、`core_unpin`、`memory_get`、`memory_update`、`memory_forget`。

### HTTP 接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/recall` | `{q, k, scope}` → 带 `trace_id` 的排序结果，每条 hit 带 `cited` |
| POST | `/remember` | `{text, scope, kind, source, importance}` |
| GET | `/core?budget=1200` | 渲染 core 块 + `dropped` 清单 |
| POST | `/core/pin`、`/core/unpin` | 把一条记忆 pin 进槽位（`identity/rule/preference/project/other`） |
| GET/PUT/DELETE | `/memory/<id>` | 读、改（会重算向量）、归档 |
| GET | `/traces?limit=20`、`/trace/<id>` | 最近召回 / 复放某一次 |
| POST | `/feedback` | `{trace_id, hit_ids, used}` —— 激活衰减唯一的真实信号 |

## 怎么工作的

```
query ──► dense（bge-m3）──┐
      ──► BM25（SQLite）  ─┼─► 归一化 + 加权融合 ─► 可选重排 ─► top-k
      ──► graph（edges）  ─┘
```

- 权重在 `data/active_policy.json`（没有就用 `DEFAULT_POLICY`）；每次改都是一个能 diff、能回滚的文件。
- 写入路径（`memex/write.py`）按 cosine > 0.92 去重，用 supersede 代替覆盖，保留 `superseded_by` 链。
- schema 在 v3，迁移幂等（`memex/db.py`）；`as_of` 召回按 `valid_from/valid_to` 过滤（仅 facts）。
- 全部代码是标准库 + numpy。重排器是外部 HTTP 进程（`bge-reranker-v2-m3`），**默认关闭**。

## 当前数字（在我们自己的金标集上测的）

| 集合 | Top1 | R@5 | 备注 |
|---|---|---|---|
| dev（120 题，含 22 条负例） | **84.2%** | **95.8%** | 仅正题：80.6% / 94.9%（n=98） |
| holdout v2（186 题） | **72.6%** | **86.0%** | 偏 note，更难 |
| temporal（15 题，`as_of`） | **86.7%** | 100% | 现行对照只有 6.7% |

两条老实话，因为这个仓库讲的就是「可验证」：

1. 120 题的 95% 置信区间约 ±6.6 个百分点。小于 ~5pt 的差异一律算噪声；所以仓库自带配对 bootstrap + McNemar 工具（`tools/eval_stats.py`），只有 `p < 0.05` **且** CI 下界大于 0 才允许说「变好了」。
2. 上面的数字是在一台特定机器上、开着重排测出来的。它是一份测量结果，不是排行榜宣言。

## 配置

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `MEMEX_DB` | `data/memex.db`（相对仓库） | 数据库路径 |
| `MEMEX_EMBED_BASE` | `http://127.0.0.1:11434/v1` | OpenAI 兼容的 embeddings 地址 |
| `MEMEX_EMBED_MODEL` | `bge-m3` | embedding 模型 |
| `MEMEX_EMBED_KEY` | `ollama` | API key（Ollama 随便填） |
| `MEMEX_EMBED_TIMEOUT` / `_BATCH` | `60` / `32` | 请求调参 |
| `MEMEX_GOLDEN_DIR` | `eval/golden` | `eval_run` 用的金标目录 |

重排：在 `data/active_policy.json` 里设 `"use_rerank": true`，并把 `rerank_api_base` 指到 bge-reranker 服务。默认关，保证新克隆零外部服务也能跑。

## 目录

```
memex/        库（recall、write、core、db、embed、http、server_mcp、eval）
tools/        make_demo.py、check_all.py、eval_stats.py
tests/        纯 python 测试（不需要 pytest）
demo/         合成数据 demo 库
docs/         架构、评测说明、路线图
panel/        单文件网页面板（pin/unpin、编辑、删除、召回复放）
examples/     MCP 配置、curl 示例
```

## 文档

- [架构](docs/architecture.md)
- [评测方法](docs/eval.md)
- [路线图](docs/roadmap.md)

## 参与

欢迎 issue 和 PR。两条规矩：任何行为改动都要带一条可证伪的验收标准；测试不许依赖作者本机的数据或私有路径。

MIT 许可。
