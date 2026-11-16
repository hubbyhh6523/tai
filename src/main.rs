mod cache;
mod embedder;
mod metrics;

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use cache::{EmbedCache, Entry};
use embedder::Embedder;
use metrics::Metrics;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::HashMap, path::PathBuf, sync::{atomic::Ordering::Relaxed, Arc}, time::Instant};

struct Config {
    port: u16,
    model_dir: PathBuf,
    model_name: String,
    dim: usize,
    threads: usize,
    max_batch: usize,
    max_chars: usize,
    max_tokens: usize,
    cache_max_mb: f64,
    cache_max_entries: usize,
    cache_ttl_secs: u64,
    api_key: Option<String>,
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl Config {
    fn from_env() -> Self {
        Self {
            port: env_or("PORT", 8090),
            model_dir: env_or("MODEL_DIR", PathBuf::from("model")),
            model_name: env_or("MODEL_NAME", "bge-small-zh-v1.5".to_string()),
            dim: env_or("DIM", 512),
            threads: env_or("THREADS", 4),
            max_batch: env_or("MAX_BATCH", 256),
            max_chars: env_or("MAX_CHARS", 5000),
            max_tokens: env_or("MAX_TOKENS", 512),
            cache_max_mb: env_or("CACHE_MAX_MB", 64.0),
            cache_max_entries: env_or("CACHE_MAX_ENTRIES", 200_000),
            cache_ttl_secs: env_or("CACHE_TTL_SECS", 0),
            api_key: std::env::var("API_KEY").ok().filter(|s| !s.is_empty()),
        }
    }
}

struct App {
    cfg: Config,
    embedder: Embedder,
    cache: EmbedCache,
    metrics: Metrics,
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

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": kind, "code": status.as_u16()}}))).into_response()
}

fn authorized(app: &App, h: &HeaderMap) -> bool {
    match &app.cfg.api_key {
        None => true,
        Some(k) => h
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {k}"))
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

/// Serve hits from the LRU, deduplicate misses, run one inference call for them, then refill the cache.
fn embed_with_cache(app: &App, texts: &[String]) -> anyhow::Result<Vec<Entry>> {
    let mut out: Vec<Option<Entry>> = vec![None; texts.len()];
    let mut uniq: HashMap<&str, usize> = HashMap::new();
    let mut uniq_texts: Vec<String> = Vec::new();
    let mut pending: Vec<(usize, usize)> = Vec::new();
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
                uniq_texts.push(t.clone());
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
            eprintln!("ERROR inference failed (batch of {}): {e:#}", uniq_texts.len());
            e
        })?;
        app.metrics.inference_batches.fetch_add(1, Relaxed);
        app.metrics.inference_ms.fetch_add(t0.elapsed().as_millis() as u64, Relaxed);
        let entries: Vec<Entry> = emb
            .vectors
            .into_iter()
            .zip(emb.tokens)
            .map(|(v, tokens)| Entry { vec: Arc::new(v), tokens })
            .collect();
        for (t, e) in uniq_texts.iter().zip(&entries) {
            app.cache.put(t, e.clone());
        }
        for (i, j) in pending {
            out[i] = Some(entries[j].clone());
        }
    }
    out.into_iter().map(|e| e.ok_or_else(|| anyhow::anyhow!("missing embedding"))).collect()
}

async fn embeddings_inner(app: Arc<App>, headers: HeaderMap, body: JsonBody) -> Response {
    if !authorized(&app, &headers) {
        return err(StatusCode::UNAUTHORIZED, "invalid_request_error", "invalid api key");
    }
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("bad request body: {e}")),
    };
    if let Some(d) = req.dimensions {
        if d != app.cfg.dim {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("model produces {} dimensions, requested {d}; set `dimensions` to {} (or sendDimensions: false)", app.cfg.dim, app.cfg.dim),
            );
        }
    }
    let b64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(o) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("unsupported encoding_format {o}")),
    };
    let texts: Vec<String> = match req.input {
        Input::One(s) => vec![s],
        Input::Many(v) => v,
    };
    if texts.is_empty() {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", "input must not be empty");
    }
    if texts.len() > app.cfg.max_batch {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("batch of {} exceeds max {}", texts.len(), app.cfg.max_batch));
    }
    let max_chars = app.cfg.max_chars;
    let texts: Vec<String> = texts.into_iter().map(|t| t.chars().take(max_chars).collect()).collect();
    let n = texts.len();

    let t0 = Instant::now();
    app.metrics.texts.fetch_add(n as u64, Relaxed);
    let app2 = app.clone();
    let res = tokio::task::spawn_blocking(move || embed_with_cache(&app2, &texts)).await;
    let entries = match res {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("inference failed: {e:#}")),
        Err(e) => {
            app.metrics.inference_errors.fetch_add(1, Relaxed);
            eprintln!("ERROR worker panicked: {e}");
            return err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("worker failed: {e}"));
        }
    };
    let prompt_tokens: usize = entries.iter().map(|e| e.tokens).sum();
    eprintln!("embed batch={n} tokens={prompt_tokens} ms={}", t0.elapsed().as_millis());

    let data: Vec<Value> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let v = &*e.vec;
            let emb = if b64 {
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(v)
            };
            json!({"object": "embedding", "index": i, "embedding": emb})
        })
        .collect();
    Json(json!({
        "object": "list",
        "data": data,
        "model": app.cfg.model_name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}
    }))
    .into_response()
}

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"status": "ok", "model": app.cfg.model_name, "dimensions": app.cfg.dim}))
}

async fn metrics(State(app): State<Arc<App>>) -> Response {
    let body = app.metrics.render(&app.cache.stats());
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": app.cfg.model_name, "object": "model", "owned_by": "local"}]}))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    let embedder = Embedder::load(&cfg.model_dir, cfg.threads, cfg.dim, cfg.max_tokens)?;
    let port = cfg.port;
    let ttl = (cfg.cache_ttl_secs > 0).then(|| std::time::Duration::from_secs(cfg.cache_ttl_secs));
    let cache = EmbedCache::new((cfg.cache_max_mb * 1024.0 * 1024.0) as usize, cfg.cache_max_entries, ttl);
    let app = Arc::new(App { cfg, embedder, cache, metrics: Metrics::default() });
    let router = Router::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    eprintln!("embed-svc listening on :{port}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
