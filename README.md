# embed-svc

一个用 Rust 写的本地 embedding 服务，提供 OpenAI 兼容的 `/v1/embeddings` 接口。它用 ONNX Runtime 运行 `bge-small-zh-v1.5`（512 维），带 LRU 缓存和 Prometheus 指标，目的是给 [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) 的 Memory Core 提供向量化。

## 接口

| 路径 | 说明 |
|---|---|
| `POST /v1/embeddings`（也支持 `/embeddings`） | `input` 可以是字符串或数组；`dimensions` 与模型维度不一致时返回 400；支持 `encoding_format: base64`；超限返回 429（限流开启时）或 413（请求体超过 `MAX_BODY_MB` 时） |
| `GET /health` | 可用性检查：模型可用返回 200；模型锁被 panic 毒化，或连续推理失败达到 `HEALTH_MAX_FAILS` 次，返回 503 并带上原因；不鉴权，供 Docker HEALTHCHECK 用 |
| `GET /metrics` | Prometheus 文本格式指标；设置 `API_KEY` 后需要同样的 Bearer 鉴权 |
| `GET /v1/models` | 返回当前模型名 |

设置 `API_KEY` 后，`/v1/embeddings` 和 `/metrics` 需要 `Authorization: Bearer <key>`（恒定时间比较）。`/health` 不鉴权。默认只绑 `127.0.0.1`，不要把端口暴露到公网。

响应里 `usage` 增加了向后兼容的 `truncated` 字段：本次请求里被 `MAX_CHARS` 截断的文本数（没有截断时为 0）。

## 快速开始

```bash
bash deploy/fetch-model.sh model      # 下载模型并校验 SHA-256，约 95 MB
cargo run --release                  # 默认只监听 127.0.0.1:8090
python3 scripts/smoke.py             # 冒烟测试，需要服务已在运行
```

`cargo test` 不需要模型：共 42 个测试，其中 13 个在本地回环上启动真实的 HTTP 服务、用假模型验证鉴权、参数校验、缓存、限流、截断统计和健康检查。首次构建会从网络下载 ONNX Runtime（`ort` 的 `download-binaries`）。

CI 里还有一个 `smoke` 任务：下载模型（按 `fetch-model.sh` 的内容缓存，每次都重新校验 SHA-256），用真实模型启动发布版二进制，运行 `scripts/ci-smoke.sh`：冒烟测试一次，缓存淘汰和 TTL 测试各一次，三次使用不同的服务配置。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | 8090 | 监听端口 |
| `BIND_ADDR` | `127.0.0.1` | 监听地址；容器内需用 `0.0.0.0`（Dockerfile / compose 已设置），宿主机端口仍只绑回环 |
| `MODEL_DIR` | `model` | 需要包含 `model.onnx` 和 `tokenizer.json` |
| `MODEL_NAME` | `bge-small-zh-v1.5` | 响应里的 `model` 字段 |
| `DIM` | 512 | 模型输出维度，请求中的 `dimensions` 必须与之一致 |
| `THREADS` | 按核数推导 | 未设置时按 `可用核数 / 会话池大小` 推导，不再默认 2 会话 × 4 线程超配 4 核机 |
| `SESSION_POOL_SIZE` | 按核数推导 | 未设置时 ≥4 核为 2，否则为 1 |
| `MAX_BATCH` | 256 | 单次请求最大条数 |
| `MAX_CHARS` | 5000 | 单条文本超过该字符数会被截断，截断数计入 `usage.truncated` 和 `embed_truncated_texts_total` |
| `MAX_TOKENS` | 512 | 分词后的最大长度 |
| `MAX_TOKENS_PER_RUN` | 8192 | 单次 ONNX 推理的填充后位置数预算（条数 × 块内最长）；一批按 token 长度排序后按此预算切块 |
| `MAX_BODY_MB` | 4 | 请求体大小上限（MB），0 表示不限制，超限返回 413 |
| `RATE_LIMIT_PER_MIN` | 0 | 每分钟 embedding 请求上限，0 表示关闭，超限返回 429 |
| `API_KEY` | 空 | 非空时启用 Bearer 鉴权；`BIND_ADDR=0.0.0.0` 且未设时启动会打印警告 |
| `CACHE_MAX_MB` | 64 | 缓存字节预算，0 表示关闭 |
| `CACHE_MAX_ENTRIES` | 200000 | 缓存条目上限 |
| `CACHE_TTL_SECS` | 0 | 缓存过期秒数，0 表示不过期 |
| `HEALTH_MAX_FAILS` | 3 | 连续推理失败多少次后 `/health` 返回 503，0 表示关闭这项检查 |
| `HEALTH_EXIT_ON_UNHEALTHY` | false | 设为 true 时，健康检查失败（连续失败达阈值或锁毒化）后进程主动退出，让 `restart: unless-stopped` 重启它 |

## 启动自检

监听端口之前，服务会先做两次真实推理：一条短文本（核对输出维度和归一化），一条长到足以达到 `MAX_TOKENS` 的文本（核对模型确实支持这么长的序列）。`DIM` 或 `MAX_TOKENS` 配错时进程会带着原因以非零状态退出，而不是等到第一个请求才报错。

## 缓存与指标

缓存键是文本的 128 位 blake3 哈希，按字节预算淘汰最久未用的条目，可选 TTL。批量请求里的重复文本只推理一次。

大批量推理时先按 token 长度排序，再按填充后位置数（条数 × 块内最长）以 `MAX_TOKENS_PER_RUN` 为预算切块分别跑 ONNX、最后按原顺序还原，避免一条长文本把整批都垫到最长长度。向量语义不变（CLS 池化 + L2 归一化）。

重点关注这几个指标：`embed_responses_server_error_total`（调用方拿不到向量的次数）、`embed_inference_errors_total`、`embed_cache_hits_total` 与 `embed_cache_misses_total`、`embed_cache_evictions_total`、`embed_truncated_texts_total`、`embed_rate_limited_total`，以及 `embed_healthy`（为 0 即服务不可用）。

## 部署

`deploy/` 里有把它和 Memory Core 放在一起运行的 `docker-compose.yml`、网关配置和初始化脚本，步骤见 [deploy/README.md](deploy/README.md)。compose 里 embed-svc 已有 healthcheck，memory-core 的 `depends_on` 是 `condition: service_healthy`。如果把 `HEALTH_EXIT_ON_UNHEALTHY` 设为 true，503 时进程会退出并由 `restart: unless-stopped` 重启；默认 false 时 503 只停留在健康检查层面，需要外部编排处理。

## 已知限制

- 推理用 ONNX 会话池（默认按核数推导，见上表），不是单把互斥锁。在一台 4 vCPU 的沙箱里、缓存未命中时原版吞吐约 190 条/秒（合成负载测量值，不代表你的真实流量；优化版未实测，需用 `scripts/bench.sh` / `load.py` 验证）。
- `ort` 固定在预发布版 `2.0.0-rc.15`，升级时接口可能变化。
- Dockerfile 和 `docker compose` 流程尚未在容器里实际验证（编写环境没有 Docker），详见 `deploy/README.md` 的验证状态表。
- `/health` 返回 503 的路径在 HTTP 层测试里用假模型触发过，没有用真实模型触发过。
- CI 会用 `cargo clippy -- -D warnings` 强制不允许新的 clippy 警告，并检查 `cargo fmt --check`。
- 暂无许可证。

## 目录

```
src/        服务代码（main、embedder、cache、metrics、health）
deploy/     docker-compose、网关配置、模型下载与 admin 初始化脚本
scripts/    冒烟测试、缓存策略测试、压测脚本
```
