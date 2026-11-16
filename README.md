# embed-svc

一个用 Rust 写的本地 embedding 服务，提供 OpenAI 兼容的 `/v1/embeddings` 接口。它用 ONNX Runtime 运行 `bge-small-zh-v1.5`（512 维），带 LRU 缓存和 Prometheus 指标，目的是给 [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) 的 Memory Core 提供向量化。

## 接口

| 路径 | 说明 |
|---|---|
| `POST /v1/embeddings`（也支持 `/embeddings`） | `input` 可以是字符串或数组；`dimensions` 与模型维度不一致时返回 400；支持 `encoding_format: base64` |
| `GET /health` | 存活检查 |
| `GET /metrics` | Prometheus 文本格式指标 |
| `GET /v1/models` | 返回当前模型名 |

设置 `API_KEY` 后，`/v1/embeddings` 需要 `Authorization: Bearer <key>`。`/health` 和 `/metrics` 不鉴权，不要把端口暴露到公网。

## 快速开始

```bash
./deploy/fetch-model.sh model        # 下载模型并校验 SHA-256，约 95 MB
cargo run --release                  # 默认监听 0.0.0.0:8090
python3 scripts/smoke.py             # 冒烟测试，需要服务已在运行
```

单元测试不需要模型：`cargo test`。首次构建会从网络下载 ONNX Runtime（`ort` 的 `download-binaries`）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | 8090 | 监听端口 |
| `MODEL_DIR` | `model` | 需要包含 `model.onnx` 和 `tokenizer.json` |
| `MODEL_NAME` | `bge-small-zh-v1.5` | 响应里的 `model` 字段 |
| `DIM` | 512 | 模型输出维度，请求中的 `dimensions` 必须与之一致 |
| `THREADS` | 4 | ONNX Runtime 线程数 |
| `MAX_BATCH` | 256 | 单次请求最大条数 |
| `MAX_CHARS` | 5000 | 单条文本超过该字符数会被截断 |
| `MAX_TOKENS` | 512 | 分词后的最大长度 |
| `API_KEY` | 空 | 非空时启用 Bearer 鉴权 |
| `CACHE_MAX_MB` | 64 | 缓存字节预算，0 表示关闭 |
| `CACHE_MAX_ENTRIES` | 200000 | 缓存条目上限 |
| `CACHE_TTL_SECS` | 0 | 缓存过期秒数，0 表示不过期 |

## 缓存与指标

缓存键是文本的 128 位 blake3 哈希，按字节预算淘汰最久未用的条目，可选 TTL。批量请求里的重复文本只推理一次。

重点关注这几个指标：`embed_responses_server_error_total`（调用方拿不到向量的次数）、`embed_inference_errors_total`、`embed_cache_hits_total` 与 `embed_cache_misses_total`、`embed_cache_evictions_total`。

## 部署

`deploy/` 里有把它和 Memory Core 放在一起运行的 `docker-compose.yml`、网关配置和初始化脚本，步骤见 [deploy/README.md](deploy/README.md)。

## 已知限制

- 模型会话受一把互斥锁保护，推理是串行的。在一台 4 vCPU 的沙箱里、缓存未命中时吞吐约 190 条/秒。这是合成负载下的测量值，不代表你的真实流量。
- `ort` 固定在预发布版 `2.0.0-rc.15`，升级时接口可能变化。
- Dockerfile 和 `docker compose` 流程尚未在容器里实际验证（编写环境没有 Docker），详见 `deploy/README.md` 的验证状态表。
- CI 会用 `cargo clippy -- -D warnings` 强制不允许新的 clippy 警告；`cargo fmt --check` 目前不通过，CI 没有检查格式。
- 暂无许可证。

## 目录

```
src/        服务代码（main、embedder、cache、metrics）
deploy/     docker-compose、网关配置、模型下载与 admin 初始化脚本
scripts/    冒烟测试、缓存策略测试、压测脚本
```
