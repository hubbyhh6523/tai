# 部署：embed-svc + memory-core（已启用向量检索）

两个容器，同一网络 `tdai-memory-stack`。Core 通过 `http://embed-svc:8090/v1` 调用向量服务。

## 步骤
1. `cp .env.example .env && chmod 600 .env`，填好 `MEMORY_LLM_*`，并用 `openssl rand -hex 24` 生成 `EMBED_API_KEY`。
2. `./fetch-model.sh`（下载约 95 MB 的模型，可设 `HF_ENDPOINT` 走镜像）。
3. `docker compose up -d --build`（embed-svc 健康后才会启动 Core）。
4. `./init-admin.sh`（首次创建 admin，密钥写入 `./.admin-key`，权限 600）。
5. 验证：`curl -s localhost:8420/health` 里应有 `embeddingService: true`；`curl -s localhost:8090/metrics | grep embed_responses`。

## 不要做的事
- 不要运行官方 `start-memory-core.sh` / `start-all.sh`：它们每次启动都会把配置覆盖成 `embedding.provider: none`，并替换 `tdai-memory-core` 容器。
- Hub、Proxy 仍可用官方 `start-memory-hub.sh`、`start-proxy.sh` 启动，它们按网络别名 `memory-core` 找 Core。
- 改 `DIM` 或模型时，必须同步改 `tdai-gateway.yaml` 的 `dimensions`/`model`；Core 会清空并重建向量表，旧向量全部丢失，换之前先备份卷。

## 监控（Core 在 embedding 失败后不重试、也不补写）
- `embed_responses_server_error_total` 增量大于 0：有记忆缺向量，应告警。
- `embed_inference_errors_total`：模型推理失败。
- `embed_cache_hits_total / (hits + misses)`：缓存命中率；`embed_cache_evictions_total` 持续增长说明预算偏小。

## 验证状态
| 项目 | 状态 |
|---|---|
| 网关配置模板被 Core 自带解析器加载（embedding 与 `${VAR}` 替换） | 已验证 |
| 容器内健康检查命令（服务在线返回 0，停止后返回 1） | 已验证 |
| 真实 Core 经 Bearer 鉴权调用 embed-svc，写入与检索 | 已验证（宿主机进程，非容器） |
| `init-admin.sh` 首次、重复、缺密钥文件三种情况 | 已验证 |
| `docker compose up`、镜像构建、Dockerfile | **未验证**（编写环境没有 Docker） |
| 官方 Hub/Proxy 接入同一网络 | 未验证 |
