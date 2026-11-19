mod cache;
mod embedder;
mod health;
mod metrics;
mod session_pool;

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use cache::{EmbedCache, Entry};
use embedder::{Embed, Embedder};
use metrics::Metrics;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, atomic::Ordering::Relaxed},
    time::{Duration, Instant},
};

struct Config {
    port: u16,
    bind_addr: String,
    model_dir: PathBuf,
    model_name: String,
    dim: usize,
    threads: usize,
    pool_size: usize,
    queue_timeout_ms: u64,
    max_batch: usize,
    max_chars: usize,
    max_tokens: usize,
    max_tokens_per_run: usize,
    max_body_mb: usize,
    rate_limit_per_min: u64,
    cache_max_mb: f64,
    cache_max_entries: usize,
    cache_ttl_secs: u64,
    health_max_failures: u64,
    health_exit_on_unhealthy: bool,
    api_key: Option<String>,
    /// Precomputed `Bearer <key>` so no per-request allocation is needed.
    bearer: Option<String>,
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn env_bool(k: &str, d: bool) -> bool {
    match std::env::var(k) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => d,
        },
        Err(_) => d,
    }
}

/// (pool_size, threads_per_session) defaults derived from the machine.
///
/// The old defaults (2 sessions x 4 threads = 8 ONNX threads) oversubscribed
/// a 4-core box. The new rule keeps total ONNX threads at or below the
/// number of available CPUs: 2 sessions on >= 4 cores, otherwise 1, with
/// threads split evenly across sessions. Explicit `SESSION_POOL_SIZE` /
/// `THREADS` always win.
fn default_pool_and_threads() -> (usize, usize) {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);
    let pool_size = if std::env::var("SESSION_POOL_SIZE").is_ok() {
        env_or("SESSION_POOL_SIZE", 2)
    } else if cpus >= 4 {
        2
    } else {
        1
    }
    .max(1);
    let threads = if std::env::var("THREADS").is_ok() {
        env_or("THREADS", 4)
    } else {
        (cpus / pool_size).max(1)
    }
    .max(1);
    (pool_size, threads)
}

impl Config {
    fn from_env() -> Self {
        let (pool_size, threads) = default_pool_and_threads();
        let api_key = std::env::var("API_KEY").ok().filter(|s| !s.is_empty());
        let bearer = api_key.as_ref().map(|k| format!("Bearer {k}"));
        Self {
            port: env_or("PORT", 8090),
            bind_addr: env_or("BIND_ADDR", "127.0.0.1".to_string()),
            model_dir: env_or("MODEL_DIR", PathBuf::from("model")),
            model_name: env_or("MODEL_NAME", "bge-small-zh-v1.5".to_string()),
            dim: env_or("DIM", 512),
            threads,
            pool_size,
            queue_timeout_ms: env_or("EMBED_QUEUE_TIMEOUT_MS", 5000),
            max_batch: env_or("MAX_BATCH", 256),
            max_chars: env_or("MAX_CHARS", 5000),
            max_tokens: env_or("MAX_TOKENS", 512),
            max_tokens_per_run: env_or("MAX_TOKENS_PER_RUN", 8192),
            max_body_mb: env_or("MAX_BODY_MB", 4),
            rate_limit_per_min: env_or("RATE_LIMIT_PER_MIN", 0),
            cache_max_mb: env_or("CACHE_MAX_MB", 64.0),
            cache_max_entries: env_or("CACHE_MAX_ENTRIES", 200_000),
            cache_ttl_secs: env_or("CACHE_TTL_SECS", 0),
            health_max_failures: env_or("HEALTH_MAX_FAILS", 3),
            health_exit_on_unhealthy: env_bool("HEALTH_EXIT_ON_UNHEALTHY", false),
            api_key,
            bearer,
        }
    }
}

/// Fixed-window rate limiter, standard library only.
///
/// One 60-second window, counted per process (not per client). A limit of
/// 0 disables it. This is coarse flood protection for a local service, not
/// a per-user quota system.
struct RateLimiter {
    limit_per_min: u64,
    state: Mutex<(Instant, u64)>,
}

impl RateLimiter {
    fn new(limit_per_min: u64) -> Self {
        Self {
            limit_per_min,
            state: Mutex::new((Instant::now(), 0)),
        }
    }

    fn allow(&self) -> bool {
        if self.limit_per_min == 0 {
            return true;
        }
        let mut guard = match self.state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = Instant::now();
        if now.duration_since(guard.0) >= Duration::from_secs(60) {
            *guard = (now, 1);
            return true;
        }
        if guard.1 >= self.limit_per_min {
            return false;
        }
        guard.1 += 1;
        true
    }
}

struct App {
    cfg: Config,
    embedder: Arc<dyn Embed>,
    cache: EmbedCache,
    metrics: Metrics,
    rate_limiter: RateLimiter,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct EmbReq {
    input: Input,
    #[allow(dead_code)]
    model: Option<String>,
    dimensions: Option<usize>,
    encoding_format: Option<String>,
}

#[derive(Serialize)]
struct EmbResp<'a> {
    object: &'static str,
    data: Vec<EmbItem<'a>>,
    model: &'a str,
    usage: Usage,
}

#[derive(Serialize)]
struct EmbItem<'a> {
    object: &'static str,
    index: usize,
    embedding: EmbeddingOut<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum EmbeddingOut<'a> {
    Float(&'a [f32]),
    Base64(String),
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
    /// Texts shortened by MAX_CHARS in this request (0 when nothing was cut).
    truncated: usize,
}

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": msg.into(), "type": kind, "code": status.as_u16()}})),
    )
        .into_response()
}

/// Constant-time comparison: no early exit on the first differing byte, so
/// the comparison time does not reveal how many leading bytes matched.
/// A length mismatch still returns false immediately (the key length is
/// not treated as secret here).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn authorized(app: &App, h: &HeaderMap) -> bool {
    match &app.cfg.bearer {
        None => true,
        Some(expected) => h
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|v| constant_time_eq(v.as_bytes(), expected.as_bytes()))
            .unwrap_or(false),
    }
}

type JsonBody = Result<Json<EmbReq>, axum::extract::rejection::JsonRejection>;

async fn embeddings(State(app): State<Arc<App>>, headers: HeaderMap, body: JsonBody) -> Response {
    app.metrics.requests.fetch_add(1, Relaxed);
    let resp = embeddings_inner(app.clone(), headers, body).await;
    let status = resp.status().as_u16();
    app.metrics.record_status(status);
    if status >= 500 {
        eprintln!("WARN responded {status}; caller will not get vectors for this batch");
    }
    resp
}

/// Serve hits from the LRU, deduplicate misses, run inference for them, then refill the cache.
///
/// `uniq_texts` borrows from `texts`, so deduplication no longer clones
/// every unique string; the embedder takes `&str` for the same reason.
fn embed_with_cache(app: &App, texts: &[String]) -> anyhow::Result<Vec<Entry>> {
    let mut out: Vec<Option<Entry>> = vec![None; texts.len()];
    let mut uniq: HashMap<&str, usize> = HashMap::new();
    let mut uniq_texts: Vec<&str> = Vec::with_capacity(texts.len());
    let mut pending: Vec<(usize, usize)> = Vec::with_capacity(texts.len());
    for (i, t) in texts.iter().enumerate() {
        if let Some(e) = app.cache.get(t) {
            app.metrics.cache_hits.fetch_add(1, Relaxed);
            out[i] = Some(e);
            continue;
        }
        app.metrics.cache_misses.fetch_add(1, Relaxed);
        let j = match uniq.get(t.as_str()) {
            Some(&j) => j,
            None => {
                uniq_texts.push(t.as_str());
                uniq.insert(t.as_str(), uniq_texts.len() - 1);
                uniq_texts.len() - 1
            }
        };
        pending.push((i, j));
    }
    if !uniq_texts.is_empty() {
        let t0 = Instant::now();
        let emb = app.embedder.embed(&uniq_texts).map_err(|e| {
            app.metrics.inference_errors.fetch_add(1, Relaxed);
            eprintln!(
                "ERROR inference failed (batch of {}): {e:#}",
                uniq_texts.len()
            );
            e
        })?;
        app.metrics.inference_batches.fetch_add(1, Relaxed);
        app.metrics
            .inference_ms
            .fetch_add(t0.elapsed().as_millis() as u64, Relaxed);
        let entries: Vec<Entry> = emb
            .vectors
            .into_iter()
            .zip(emb.tokens)
            .map(|(v, tokens)| Entry {
                vec: Arc::new(v),
                tokens,
            })
            .collect();
        for (t, e) in uniq_texts.iter().zip(&entries) {
            app.cache.put(t, e.clone());
        }
        for (i, j) in pending {
            out[i] = Some(entries[j].clone());
        }
    }
    out.into_iter()
        .map(|e| e.ok_or_else(|| anyhow::anyhow!("missing embedding")))
        .collect()
}

async fn embeddings_inner(app: Arc<App>, headers: HeaderMap, body: JsonBody) -> Response {
    if !authorized(&app, &headers) {
        return err(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "invalid api key",
        );
    }
    if !app.rate_limiter.allow() {
        app.metrics.rate_limited.fetch_add(1, Relaxed);
        return err(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "rate limit exceeded; retry after a minute",
        );
    }
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("bad request body: {e}"),
            );
        }
    };
    if let Some(d) = req.dimensions
        && d != app.cfg.dim
    {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "model produces {} dimensions, requested {d}; set `dimensions` to {} (or sendDimensions: false)",
                app.cfg.dim, app.cfg.dim
            ),
        );
    }
    let b64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(o) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("unsupported encoding_format {o}"),
            );
        }
    };
    let texts: Vec<String> = match req.input {
        Input::One(s) => vec![s],
        Input::Many(v) => v,
    };
    if texts.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "input must not be empty",
        );
    }
    if texts.len() > app.cfg.max_batch {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("batch of {} exceeds max {}", texts.len(), app.cfg.max_batch),
        );
    }
    // Truncation used to be silent: the caller never knew text was lost.
    // Count it, expose it in `usage.truncated` and in the metrics, and log it.
    let max_chars = app.cfg.max_chars;
    let mut truncated = 0usize;
    let texts: Vec<String> = texts
        .into_iter()
        .map(|t| {
            if t.chars().count() > max_chars {
                truncated += 1;
            }
            t.chars().take(max_chars).collect()
        })
        .collect();
    if truncated > 0 {
        app.metrics
            .truncated_texts
            .fetch_add(truncated as u64, Relaxed);
    }
    let n = texts.len();

    let t0 = Instant::now();
    app.metrics.texts.fetch_add(n as u64, Relaxed);
    let app2 = app.clone();
    let res = tokio::task::spawn_blocking(move || embed_with_cache(&app2, &texts)).await;
    let entries = match res {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("inference failed: {e:#}"),
            );
        }
        Err(e) => {
            app.metrics.inference_errors.fetch_add(1, Relaxed);
            eprintln!("ERROR worker panicked: {e}");
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("worker failed: {e}"),
            );
        }
    };
    let prompt_tokens: usize = entries.iter().map(|e| e.tokens).sum();
    eprintln!(
        "embed batch={n} tokens={prompt_tokens} truncated={truncated} ms={}",
        t0.elapsed().as_millis()
    );

    // Serialize the vectors directly from their slices instead of building
    // a serde_json::Value per float (512 Values per text before).
    let data: Vec<EmbItem<'_>> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let v = &*e.vec;
            let embedding = if b64 {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for f in v.iter() {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                EmbeddingOut::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                EmbeddingOut::Float(v)
            };
            EmbItem {
                object: "embedding",
                index: i,
                embedding,
            }
        })
        .collect();
    Json(EmbResp {
        object: "list",
        data,
        model: &app.cfg.model_name,
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
            truncated,
        },
    })
    .into_response()
}

async fn health(State(app): State<Arc<App>>) -> Response {
    match app.embedder.health_check() {
        Ok(()) => {
            Json(json!({"status": "ok", "model": app.cfg.model_name, "dimensions": app.cfg.dim}))
                .into_response()
        }
        Err(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unhealthy", "reason": reason})),
        )
            .into_response(),
    }
}

/// `/metrics` is authenticated when an API key is set: it leaks cache
/// sizes, request counts and health state. `/health` stays open so Docker
/// HEALTHCHECK and load balancers can probe it without a key.
async fn metrics(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if !authorized(&app, &headers) {
        // Unlike /v1/embeddings, this handler has no wrapper that records
        // the status, so count the rejection here to keep the metric honest.
        app.metrics.unauthorized.fetch_add(1, Relaxed);
        return err(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "invalid api key",
        );
    }
    let healthy = app.embedder.health_check().is_ok();
    let body = app.metrics.render(
        &app.cache.stats(),
        healthy,
        app.embedder.consecutive_failures(),
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": app.cfg.model_name, "object": "model", "owned_by": "local"}]}),
    )
}

fn build_router(app: Arc<App>) -> Router {
    let body_limit = if app.cfg.max_body_mb == 0 {
        DefaultBodyLimit::disable()
    } else {
        DefaultBodyLimit::max(app.cfg.max_body_mb * 1024 * 1024)
    };
    Router::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .layer(body_limit)
        .with_state(app)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    anyhow::ensure!(
        cfg.pool_size > 0,
        "SESSION_POOL_SIZE must be greater than zero"
    );
    anyhow::ensure!(cfg.threads > 0, "THREADS must be greater than zero");
    anyhow::ensure!(
        cfg.max_tokens_per_run > 0,
        "MAX_TOKENS_PER_RUN must be greater than zero"
    );
    if (cfg.bind_addr == "0.0.0.0" || cfg.bind_addr == "::") && cfg.api_key.is_none() {
        eprintln!(
            "WARN BIND_ADDR={} with no API_KEY: the embeddings endpoint is unauthenticated on all interfaces; set API_KEY or bind 127.0.0.1",
            cfg.bind_addr
        );
    }
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let queue_timeout =
        (cfg.queue_timeout_ms > 0).then(|| std::time::Duration::from_millis(cfg.queue_timeout_ms));
    let embedder = Arc::new(Embedder::load(
        &cfg.model_dir,
        cfg.pool_size,
        cfg.threads,
        queue_timeout,
        cfg.dim,
        cfg.max_tokens,
        cfg.max_tokens_per_run,
        cfg.health_max_failures,
    )?);
    let t0 = Instant::now();
    embedder
        .self_check(cfg.max_tokens)
        .context("startup self-check failed")?;
    eprintln!(
        "self-check ok: dim={} max_tokens={} max_tokens_per_run={} pool_size={} threads_per_session={} cpus={} queue_timeout_ms={} bind={} in {} ms",
        cfg.dim,
        cfg.max_tokens,
        cfg.max_tokens_per_run,
        cfg.pool_size,
        cfg.threads,
        cpus,
        cfg.queue_timeout_ms,
        cfg.bind_addr,
        t0.elapsed().as_millis()
    );
    let port = cfg.port;
    let bind_addr = cfg.bind_addr.clone();
    let exit_on_unhealthy = cfg.health_exit_on_unhealthy;
    let ttl = (cfg.cache_ttl_secs > 0).then(|| std::time::Duration::from_secs(cfg.cache_ttl_secs));
    let cache = EmbedCache::new(
        (cfg.cache_max_mb * 1024.0 * 1024.0) as usize,
        cfg.cache_max_entries,
        ttl,
    );
    let rate_limit = cfg.rate_limit_per_min;
    let app = Arc::new(App {
        cfg,
        embedder,
        cache,
        metrics: Metrics::default(),
        rate_limiter: RateLimiter::new(rate_limit),
    });
    if exit_on_unhealthy {
        // A 503 used to stick forever: `restart: unless-stopped` only
        // restarts a process that exits. With this flag the process exits
        // as soon as the health check fails, so the orchestrator restarts it.
        let embedder = app.embedder.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                if let Err(reason) = embedder.health_check() {
                    eprintln!(
                        "ERROR unhealthy with HEALTH_EXIT_ON_UNHEALTHY=true: {reason}; exiting so the orchestrator can restart the service"
                    );
                    std::process::exit(1);
                }
            }
        });
    }
    let router = build_router(app);
    let listener = tokio::net::TcpListener::bind((bind_addr.as_str(), port)).await?;
    eprintln!("embed-svc listening on {bind_addr}:{port}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::Embedded;
    use crate::health::Health;
    use std::net::SocketAddr;
    use std::sync::{Mutex, atomic::AtomicBool};
    use std::time::Duration;

    /// Stand-in for the model: vectors are derived from the text, calls are recorded, failures can be injected.
    struct Fake {
        fail: AtomicBool,
        poisoned: AtomicBool,
        health: Health,
        seen: Mutex<Vec<Vec<String>>>,
    }

    impl Embed for Fake {
        fn embed(&self, texts: &[&str]) -> anyhow::Result<Embedded> {
            self.seen
                .lock()
                .unwrap()
                .push(texts.iter().map(|t| t.to_string()).collect());
            let r = if self.fail.load(Relaxed) {
                Err(anyhow::anyhow!("injected failure"))
            } else {
                let vectors = texts
                    .iter()
                    .map(|t| {
                        vec![
                            t.chars().count() as f32,
                            t.bytes().map(u32::from).sum::<u32>() as f32,
                            0.0,
                            1.0,
                        ]
                    })
                    .collect();
                Ok(Embedded {
                    vectors,
                    tokens: texts.iter().map(|t| t.chars().count()).collect(),
                })
            };
            self.health.record(r.is_ok());
            r
        }

        fn health_check(&self) -> Result<(), String> {
            self.health.check(self.poisoned.load(Relaxed))
        }

        fn consecutive_failures(&self) -> u64 {
            self.health.consecutive_failures()
        }
    }

    fn cfg() -> Config {
        Config {
            port: 0,
            bind_addr: "127.0.0.1".into(),
            model_dir: PathBuf::from("unused"),
            model_name: "fake-model".into(),
            dim: 4,
            threads: 1,
            pool_size: 2,
            queue_timeout_ms: 5000,
            max_batch: 3,
            max_chars: 8,
            max_tokens: 512,
            max_tokens_per_run: 8192,
            max_body_mb: 4,
            rate_limit_per_min: 0,
            cache_max_mb: 1.0,
            cache_max_entries: 100,
            cache_ttl_secs: 0,
            health_max_failures: 2,
            health_exit_on_unhealthy: false,
            api_key: None,
            bearer: None,
        }
    }

    type Resp = (u16, String);

    fn json_of(r: &Resp) -> Value {
        serde_json::from_str(&r.1).unwrap_or_else(|e| panic!("not JSON ({e}): {}", r.1))
    }

    /// Minimal blocking HTTP/1.1 client (no extra dependencies); the server closes the connection after replying.
    fn raw_http(
        addr: SocketAddr,
        method: &str,
        path: &str,
        headers: &[(&str, String)],
        body: Option<&str>,
    ) -> Resp {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n");
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        if let Some(b) = body {
            req.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                b.len()
            ));
        }
        req.push_str("\r\n");
        if let Some(b) = body {
            req.push_str(b);
        }
        s.write_all(req.as_bytes()).unwrap();
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).unwrap();
        let text = String::from_utf8(raw).unwrap();
        let (head, body) = text
            .split_once("\r\n\r\n")
            .expect("no header/body separator");
        assert!(
            !head
                .to_ascii_lowercase()
                .contains("transfer-encoding: chunked"),
            "chunked responses are not handled by this client"
        );
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, body.to_string())
    }

    struct Harness {
        addr: SocketAddr,
        fake: Arc<Fake>,
        api_key: Option<String>,
    }

    async fn start(cfg: Config) -> Harness {
        let fake = Arc::new(Fake {
            fail: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            health: Health::new(cfg.health_max_failures),
            seen: Mutex::new(Vec::new()),
        });
        let cache = EmbedCache::new(
            (cfg.cache_max_mb * 1024.0 * 1024.0) as usize,
            cfg.cache_max_entries,
            None,
        );
        let api_key = cfg.api_key.clone();
        let rate_limit = cfg.rate_limit_per_min;
        let app = Arc::new(App {
            cfg,
            embedder: fake.clone(),
            cache,
            metrics: Metrics::default(),
            rate_limiter: RateLimiter::new(rate_limit),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = build_router(app);
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        Harness {
            addr,
            fake,
            api_key,
        }
    }

    impl Harness {
        async fn send(
            &self,
            method: &'static str,
            path: &'static str,
            headers: Vec<(&'static str, String)>,
            body: Option<String>,
        ) -> Resp {
            let addr = self.addr;
            tokio::task::spawn_blocking(move || {
                raw_http(addr, method, path, &headers, body.as_deref())
            })
            .await
            .unwrap()
        }

        async fn get(&self, path: &'static str) -> Resp {
            self.send("GET", path, vec![], None).await
        }

        async fn get_auth(&self, path: &'static str) -> Resp {
            let auth = self
                .api_key
                .as_ref()
                .map(|k| format!("Bearer {k}"))
                .unwrap_or_default();
            self.send("GET", path, vec![("Authorization", auth)], None)
                .await
        }

        async fn post(&self, body: &str) -> Resp {
            self.send("POST", "/v1/embeddings", vec![], Some(body.to_string()))
                .await
        }

        async fn post_auth(&self, body: &str, authorization: &str) -> Resp {
            self.send(
                "POST",
                "/v1/embeddings",
                vec![("Authorization", authorization.to_string())],
                Some(body.to_string()),
            )
            .await
        }

        async fn metric(&self, name: &str) -> u64 {
            // /metrics needs a key when one is configured; send it so the
            // helper keeps working in authenticated harnesses.
            let body = if self.api_key.is_some() {
                self.get_auth("/metrics").await.1
            } else {
                self.get("/metrics").await.1
            };
            let prefix = format!("{name} ");
            body.lines()
                .find_map(|l| l.strip_prefix(prefix.as_str())?.trim().parse::<u64>().ok())
                .unwrap_or_else(|| panic!("metric {name} missing in:\n{body}"))
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.fake.seen.lock().unwrap().clone()
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn health_flips_to_503_after_consecutive_failures_and_recovers() {
        let h = start(cfg()).await;
        assert_eq!(h.get("/health").await.0, 200);
        h.fake.fail.store(true, Relaxed);
        assert_eq!(h.post(r#"{"input":["a"]}"#).await.0, 500);
        assert_eq!(
            h.get("/health").await.0,
            200,
            "one failure is below the threshold of 2"
        );
        assert_eq!(h.post(r#"{"input":["b"]}"#).await.0, 500);
        let r = h.get("/health").await;
        assert_eq!(r.0, 503);
        assert_eq!(json_of(&r)["status"], "unhealthy");
        assert_eq!(json_of(&r)["reason"], "2 consecutive inference failures");
        assert_eq!(h.metric("embed_healthy").await, 0);
        assert_eq!(h.metric("embed_consecutive_inference_failures").await, 2);
        h.fake.fail.store(false, Relaxed);
        assert_eq!(h.post(r#"{"input":["c"]}"#).await.0, 200);
        assert_eq!(h.get("/health").await.0, 200);
        assert_eq!(h.metric("embed_healthy").await, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn health_is_503_when_the_model_lock_is_poisoned() {
        let h = start(cfg()).await;
        h.fake.poisoned.store(true, Relaxed);
        let r = h.get("/health").await;
        assert_eq!(r.0, 503);
        assert!(json_of(&r)["reason"].as_str().unwrap().contains("poisoned"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bearer_auth_protects_embeddings_and_metrics_but_not_health() {
        let mut c = cfg();
        c.api_key = Some("k".into());
        c.bearer = Some("Bearer k".into());
        let h = start(c).await;
        let body = r#"{"input":["a"]}"#;
        assert_eq!(h.post(body).await.0, 401);
        assert_eq!(h.post_auth(body, "Bearer wrong").await.0, 401);
        assert_eq!(
            h.post_auth(body, "k").await.0,
            401,
            "the Bearer prefix is required"
        );
        assert_eq!(
            h.get("/metrics").await.0,
            401,
            "metrics needs a key when one is configured"
        );
        assert_eq!(h.get_auth("/metrics").await.0, 200);
        assert_eq!(h.post_auth(body, "Bearer k").await.0, 200);
        assert_eq!(h.get("/health").await.0, 200);
        assert_eq!(h.metric("embed_responses_unauthorized_total").await, 4);
        assert!(
            h.calls().len() == 1,
            "unauthorized requests must not reach the model"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn metrics_is_open_without_api_key() {
        let h = start(cfg()).await;
        assert_eq!(h.get("/metrics").await.0, 200);
        assert_eq!(h.metric("embed_requests_total").await, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rate_limit_returns_429_and_counts() {
        let mut c = cfg();
        c.rate_limit_per_min = 1;
        let h = start(c).await;
        assert_eq!(h.post(r#"{"input":["a"]}"#).await.0, 200);
        let r = h.post(r#"{"input":["b"]}"#).await;
        assert_eq!(r.0, 429);
        assert_eq!(h.metric("embed_rate_limited_total").await, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn truncated_texts_are_counted_and_reported() {
        let h = start(cfg()).await;
        let r = json_of(&h.post(r#"{"input":["0123456789"]}"#).await);
        assert_eq!(
            r["usage"]["truncated"], 1,
            "10 chars with MAX_CHARS=8 is truncated"
        );
        assert_eq!(h.metric("embed_truncated_texts_total").await, 1);
        let r2 = json_of(&h.post(r#"{"input":["abc"]}"#).await);
        assert_eq!(r2["usage"]["truncated"], 0);
        assert_eq!(h.metric("embed_truncated_texts_total").await, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bad_requests_are_rejected_with_400_before_reaching_the_model() {
        let h = start(cfg()).await;
        let cases: [(&str, &str); 7] = [
            (r#"{"input":["x"],"dimensions":5}"#, "dimensions"),
            (r#"{"input":[]}"#, "must not be empty"),
            (r#"{"model":"m"}"#, "bad request body"),
            ("{", "bad request body"),
            (r#"{"input":["a","b","c","d"]}"#, "exceeds max 3"),
            (
                r#"{"input":["a"],"encoding_format":"fp8"}"#,
                "unsupported encoding_format",
            ),
            (r#"{"input":[1,2]}"#, "bad request body"),
        ];
        for (body, needle) in cases {
            let r = h.post(body).await;
            assert_eq!(r.0, 400, "{body} -> {}", r.1);
            let msg = json_of(&r)["error"]["message"]
                .as_str()
                .unwrap()
                .to_string();
            assert!(
                msg.contains(needle),
                "{body}: {msg:?} should contain {needle:?}"
            );
        }
        assert!(h.calls().is_empty());
        assert_eq!(h.metric("embed_responses_client_error_total").await, 7);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn accepts_string_or_array_input_and_matching_dimensions() {
        let h = start(cfg()).await;
        let one = json_of(&h.post(r#"{"input":"abc"}"#).await);
        assert_eq!(one["data"].as_array().unwrap().len(), 1);
        assert_eq!(one["usage"]["truncated"], 0);
        let many = h.post(r#"{"input":["abc","","de"],"dimensions":4}"#).await;
        assert_eq!(many.0, 200, "{}", many.1);
        let many = json_of(&many);
        let data = many["data"].as_array().unwrap();
        assert_eq!(data.len(), 3);
        for (i, d) in data.iter().enumerate() {
            assert_eq!(d["index"], i);
            assert_eq!(d["embedding"].as_array().unwrap().len(), 4);
        }
        assert_eq!(many["model"], "fake-model");
        assert_eq!(many["usage"]["prompt_tokens"], 5);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn base64_output_decodes_to_the_float_output() {
        let h = start(cfg()).await;
        let f = json_of(&h.post(r#"{"input":["abc"]}"#).await);
        let b = json_of(
            &h.post(r#"{"input":["abc"],"encoding_format":"base64"}"#)
                .await,
        );
        let raw = base64::engine::general_purpose::STANDARD
            .decode(b["data"][0]["embedding"].as_str().unwrap())
            .unwrap();
        let decoded: Vec<f64> = raw
            .chunks(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
            .collect();
        let expected: Vec<f64> = f["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded, expected);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duplicates_in_a_batch_and_repeat_requests_hit_the_model_once() {
        let h = start(cfg()).await;
        let r = json_of(&h.post(r#"{"input":["aa","aa","bb"]}"#).await);
        assert_eq!(r["data"][0]["embedding"], r["data"][1]["embedding"]);
        assert_ne!(r["data"][0]["embedding"], r["data"][2]["embedding"]);
        assert_eq!(
            h.calls(),
            vec![vec!["aa".to_string(), "bb".to_string()]],
            "duplicates are computed once"
        );
        let again = json_of(&h.post(r#"{"input":["aa"]}"#).await);
        assert_eq!(again["data"][0]["embedding"], r["data"][0]["embedding"]);
        assert_eq!(h.calls().len(), 1, "the repeat is served from the cache");
        assert_eq!(h.metric("embed_cache_hits_total").await, 1);
        assert_eq!(h.metric("embed_cache_misses_total").await, 3);
        assert_eq!(again["usage"]["prompt_tokens"], 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn long_texts_are_truncated_to_max_chars_before_the_model() {
        let h = start(cfg()).await;
        assert_eq!(h.post(r#"{"input":["0123456789"]}"#).await.0, 200);
        assert_eq!(h.calls(), vec![vec!["01234567".to_string()]]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inference_failures_return_500_and_are_counted() {
        let h = start(cfg()).await;
        h.fake.fail.store(true, Relaxed);
        let r = h.post(r#"{"input":["a"]}"#).await;
        assert_eq!(r.0, 500);
        assert!(
            json_of(&r)["error"]["message"]
                .as_str()
                .unwrap()
                .contains("injected failure")
        );
        assert_eq!(h.metric("embed_responses_server_error_total").await, 1);
        assert_eq!(h.metric("embed_inference_errors_total").await, 1);
        assert_eq!(h.metric("embed_requests_total").await, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn models_and_embeddings_alias_routes_work() {
        let h = start(cfg()).await;
        let m = h.get("/v1/models").await;
        assert_eq!(m.0, 200);
        assert_eq!(json_of(&m)["data"][0]["id"], "fake-model");
        let r = h
            .send(
                "POST",
                "/embeddings",
                vec![],
                Some(r#"{"input":["a"]}"#.to_string()),
            )
            .await;
        assert_eq!(r.0, 200);
        assert_eq!(h.get("/nope").await.0, 404);
    }
}
