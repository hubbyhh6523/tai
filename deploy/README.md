# 部署：embed-svc + memory-core + memory-hub + memory-proxy（已启用向量检索）

四个容器，同一网络 `tdai-memory-stack`。Core 通过 `http://embed-svc:8090/v1` 调用向量服务；
Hub 与 Proxy 通过 `http://memory-core:8420` 访问 Core。

| 服务 | 容器名 | 端口 | 用途 | 必需 |
|---|---|---|---|---|
| `embed-svc` | `tdai-embed-svc` | 127.0.0.1:8090 | 本地向量服务（bge-small-zh-v1.5） | 是 |
| `memory-core` | `tdai-memory-core` | 0.0.0.0:8420 | 记忆读写、鉴权、skill/RAG 数据面 | 是 |
| `memory-hub` | `tdai-memory-hub` | 127.0.0.1:8125 / 8424 | Panel UI + Knowledge（Wiki/Code-Graph） | 可选 |
| `memory-proxy` | `tdai-proxy` | 127.0.0.1:8096 | LLM 请求代理（记忆注入） | 可选 |

## 步骤

1. `cp .env.example .env && chmod 600 .env`，填好 `MEMORY_LLM_*`，并用 `openssl rand -hex 24` 生成 `EMBED_API_KEY`。
2. `bash fetch-model.sh`（下载约 95 MB 的模型，可设 `HF_ENDPOINT` 走镜像。仅影响本地 `./models` 挂载；
   上游新 Dockerfile 也会在构建时把模型烧进镜像，两种方式并存不冲突）。
3. `docker compose up -d --build`（embed-svc 健康后才会启动 Core；Rust 构建慢可加
   `CARGO_REGISTRY_MIRROR=https://rsproxy.cn`）。
4. `bash init-admin.sh`（首次创建 admin，密钥写入 `./.admin-key`，权限 600）。

只起了 Core 就到此结束。要 Panel 和 Proxy，再往下：

5. `docker compose up -d memory-hub`（Panel UI：<http://127.0.0.1:8125>）
6. Proxy 需先准备镜像和配置，见下节，然后 `docker compose up -d memory-proxy`。

## Proxy 的镜像与配置（两个都要手动做）

**镜像**：本仓库不含 Proxy 源码，`agentmemory/memory-proxy:latest` 在国内镜像站拉不到，
需先从上游 `MemoryProxy/` 目录自行构建一次，标签保持 `memory-proxy:local`（或改 `MEMORY_PROXY_IMAGE`）：

```bash
# 在上游仓库的 MemoryProxy/ 目录
docker build -t memory-proxy:local .
```

构建注意：`deb.debian.org` 在国内极慢，需在 Dockerfile 的 apt 层真实改源
（`sed -i "s|deb.debian.org|mirrors.tencent.com|g" /etc/apt/sources.list.d/debian.sources`）；
镜像约 2.5 GB，磁盘不足会 `no space left on device`。

**配置**：`cp proxy-config.example.yaml proxy-config.yaml && chmod 644 proxy-config.yaml`，填 `upstream.url` / `upstream.apiKey`。

> 配置文件必须是 644 不能是 600：proxy 镜像以内置 uid 10001 运行，读不了 600 的文件（`EACCES`）。
> 该 yaml **不会展开 `${VAR}`**，占位符会被当成字面值（`upstream` 变成 `"${MEMORY_LLM_BASE_URL}"`，请求直接 502）。
> 必须写真实值，或用 `TDAI_PROXY_UPSTREAM_API_KEY_FILE` 指向密钥文件。

## 不要做的事

- 不要运行官方 `start-memory-core.sh` / `start-all.sh`：它们每次启动都会把配置覆盖成 `embedding.provider: none`，并替换 `tdai-memory-core` 容器。
- 不要再手动 `docker run` Hub/Proxy：本 compose 已包含它们，且已处理下面几个官方文档没写对的点。
- 改 `DIM` 或模型时，必须同步改 `tdai-gateway.yaml` 的 `dimensions`/`model`；Core 会清空并重建向量表，旧向量全部丢失，换之前先备份卷。

## 官方文档没写对的地方（已在 compose 里绕过）

**1. Hub 的 `REMOTE_INSTANCE_KEY` 不能填 `local`。**
官方 INSTALL 写 `-e REMOTE_INSTANCE_KEY=local`，实测 Hub 调 Core 的 `/v3/meta/user/list`、
`/v3/meta/user/create` 全部 `401 invalid_user_key`。它需要一个 Core 认可的**用户 key**，
而该 key 要等 `init-admin.sh` 跑完才有。因此 compose 把 `./.admin-key` 只读挂进容器，
再用 `entrypoint: /bin/sh -c` 在启动时读取——不要改成在 `environment` 里写死，
那样密钥会进 `docker inspect`，而且结构上早于 `.admin-key` 生成。

**2. Hub 要按网络别名找 Core，不是 `host.docker.internal`。**
官方命令用 `--add-host=host.docker.internal:host-gateway`。在同 compose 网络里直接用
`http://memory-core:8420`，少一个 host 依赖。

**3. Proxy 的 `injection.enabled` 默认是 false。**
不显式打开，`injectors` 全部不生效（日志里 `injectionEnabled=false`）。
打开后仍可能看到 `injectedSkipped=true` —— 那不是"正常跳过"，见下一条。

**4. Proxy 用 `local` 就够了，不需要 admin key。**
Core 的 `/v3/session|skill` 数据面接受 `local`（`auth.serviceToken` 也一样），
实测无 401、`POST /v3/session/add` 返回 200。只有 Hub 的 `/v3/meta/*` 需要真 key。

**5. 记忆注入默认是「静默失效」的，要同时满足四个条件才会真的注入。**

这是最容易踩的坑：配置不对时请求照常 200，只是记忆什么都没注入。四个条件缺一不可。

| 条件 | 缺了会怎样 |
|---|---|
| `auth.enabled: true` + `auth.url` 指向 Core | `userId` 为空，注入跳过 |
| `injection.enabled: true` | `injectors` 整张列表不生效 |
| 请求路径为 `/<agent>/<spaceId>/...`，agent 名在白名单内 | 401 `missing service_id` |
| **请求头带 `x-conversation-id`** | `conversationId=null` → 注入恒跳过 |

**路径不能写成 `/proxy/<spaceId>/...`**：该前缀被识别为 legacy，日志会打
`legacy /proxy prefix (skip session-init/mem/injection/L0/skill)`，会话初始化和注入全部跳过。
必须用 `/claude-code/default/v1/chat/completions` 这种 `/<agent>/<spaceId>/` 形式
（agent 白名单：claude-code / codebuddy / codex / cursor / hermes / openclaw / workbuddy / dsh / opencode / pi）。

## 客户端必需请求头

接 Proxy 的客户端每一轮都要带这些头，否则记忆链路不工作：

| 头 | 必需 | 说明 |
|---|---|---|
| `Authorization: Bearer <user_key>` | 是 | 业务用户的 `sk-mem-...`，Proxy 用它换 userId |
| `x-conversation-id` | 是 | 会话标识，**没有它注入直接跳过**（`conversationId` 决定一切） |
| `x-team-id` | 是 | Team，例如 `team-xxx` |
| `x-agent-id` | 是 | Agent，例如 `agt-xxx` |
| `x-task-id` | 可选 | Task，绑定后记忆关联到该任务 |

`x-team-id` / `x-agent-id` / `x-task-id` 命中后走 `headerAutoSelect`，直接注册绑定，
不会弹交互表单（日志 `preset hit ... → register directly`）。

验证注入真的生效（两轮，第一轮建绑定、第二轮注入）：

```bash
curl -s -X POST http://127.0.0.1:8096/claude-code/default/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_KEY" \
  -H "x-conversation-id: conv-001" \
  -H "x-team-id: $TEAM_ID" -H "x-agent-id: $AGENT_ID" -H "x-task-id: $TASK_ID" \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}]}'

docker logs tdai-proxy 2>&1 | grep -E "injected [0-9]+ block|pipeline.done|write-l0"
```

期望看到 `injected 1 block(s)`、`totalBlockCount=3`、`errorCount=0`、`tdai-recorder:write-l0`。
只有 `injectedSkipped=true` 而没有上面几行 = 上面四个条件缺了一个。

**记忆是「工具检索 + L2 索引直注」模型，不是每轮全文召回**（保护 KV cache）：

- L2 场景索引 + L3 画像随 `<tdai_profile_memory>` 块直接注入 system —— 新会话第一轮
  模型就能答出以前学过的事，不需要工具调用。
- L0/L1 原文**不自动召回**。LLM 按需用 Bash 调 `http://127.0.0.1:8096/memory-bridge/v3/{sub}`
  （proxy 自动注入身份，客户端只发业务字段），可用 sub：`atomic/search`（L1 hybrid 检索）、
  `atomic/query`、`conversation/search`（L0）、`conversation/query`、`scenario/ls`（L2 索引）、
  `scenario/read`（L2 全文）。bridge 必需头与 chat 相同（含 `x-conversation-id`）。

验证检索（一轮写入后等 10s 提取，再查）：

```bash
curl -s -X POST http://127.0.0.1:8096/memory-bridge/v3/atomic/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_KEY" -H "x-conversation-id: conv-001" \
  -H "x-team-id: $TEAM_ID" -H "x-agent-id: $AGENT_ID" \
  -d '{"query":"项目部署","limit":5}'
```

期望 `data.items[]`（或同构字段）返回写入内容的改写条目。上游 LLM 用
`MEMORY_LLM_*`（Core 提取用）与 `upstream.url`（Proxy 转发用）两处配置，都要指向真实端点。

## 建 Team / Agent / 用户（先有这些才有 userId 和三元组）

`auth` 打开后请求必须带有效 user key，所以要先造数据。注意鉴权头是
`x-tdai-user-key`，**不是** `Authorization`：

```bash
ADMIN=$(cat ./.admin-key)
H=(-H "x-tdai-user-key: $ADMIN" -H "Authorization: Bearer $ADMIN" -H "x-tdai-service-id: default" -H "Content-Type: application/json")

# admin 自己的 user_id
ADMIN_UID=$(curl -s -X POST http://127.0.0.1:8420/v3/meta/auth/verify "${H[@]}" \
  -d "{\"user_key\":\"$ADMIN\"}" | jq -r .data.user.user_id)

# Team（owner_user_id 必填）
TEAM=$(curl -s -X POST http://127.0.0.1:8420/v3/meta/team/create "${H[@]}" \
  -d "{\"name\":\"my-team\",\"owner_user_id\":\"$ADMIN_UID\"}" | jq -r .data.team_id)

# 业务用户 → 返回里只有一次 default_user_key，务必存下
curl -s -X POST http://127.0.0.1:8420/v3/meta/user/create "${H[@]}" -d '{"username":"alice"}'

# 加入 Team
curl -s -X POST http://127.0.0.1:8420/v3/meta/team-member/add "${H[@]}" \
  -d "{\"team_id\":\"$TEAM\",\"user_id\":\"$USER_ID\",\"role\":\"member\"}"

# Agent（owner_user_id 必填）
curl -s -X POST http://127.0.0.1:8420/v3/meta/agent/create "${H[@]}" \
  -d "{\"team_id\":\"$TEAM\",\"name\":\"my-agent\",\"owner_user_id\":\"$USER_ID\"}"
```

**Task 必须用资源所有者自己的 key 建**，用 admin key 会 `403 permission_denied: caller is not resource owner`；
字段是 `title` 不是 `name`：

```bash
curl -s -X POST http://127.0.0.1:8420/v3/meta/task/create \
  -H "x-tdai-user-key: $USER_KEY" -H "Authorization: Bearer $USER_KEY" \
  -H "x-tdai-service-id: default" -H "Content-Type: application/json" \
  -d "{\"team_id\":\"$TEAM\",\"title\":\"my-task\",\"creator_user_id\":\"$USER_ID\"}"
```

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
| `docker compose up`、镜像构建、Dockerfile | 已验证（bookworm → trixie 修复后） |
| `memory-hub` 起容器、Panel 200、连上 Core、无 401 | 已验证 |
| `memory-proxy` 起容器、health ok、经它转发 chat/completions、Core 收到 session/add 200 | 已验证 |
| Proxy 会话绑定（headerAutoSelect 命中三元组）与记忆注入（3 个 block、L0 写入） | 已验证 |
| 真实 LLM 记忆链路（L1 提取 → L2 场景 → 检索召回 → 新会话冷召回） | 已验证（gpt-4.1-mini 级模型） |
| L3 persona 自动生成 | 已验证触发逻辑（数据量小时正确跳过） |
| 全栈 `docker compose up -d` 四容器同时 healthy | 已验证 |
| MongoDB / ClickHouse / Redis / Opik / Langfuse | 未启用（默认关闭，按需） |
