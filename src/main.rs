mod cache;
mod embedder;
mod health;
mod metrics;

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use anyhow::Context;
use base64::Engine;
use cache::{EmbedCache, Entry};
use embedder::{Embed, Embedder};
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
    health_max_failures: u64,
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
            health_max_failures: env_or("HEALTH_MAX_FAILS", 3),
            api_key: std::env::var("API_KEY").ok().filter(|s| !s.is_empty()),
        }
    }
}

struct App {
    cfg: Config,
    embedder: Arc<dyn Embed>,
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
    if let Some(d) = req.dimensions
        && d != app.cfg.dim
    {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("model produces {} dimensions, requested {d}; set `dimensions` to {} (or sendDimensions: false)", app.cfg.dim, app.cfg.dim),
        );
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

async fn health(State(app): State<Arc<App>>) -> Response {
    match app.embedder.health_check() {
        Ok(()) => Json(json!({"status": "ok", "model": app.cfg.model_name, "dimensions": app.cfg.dim})).into_response(),
        Err(reason) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "unhealthy", "reason": reason}))).into_response(),
    }
}

async fn metrics(State(app): State<Arc<App>>) -> Response {
    let healthy = app.embedder.health_check().is_ok();
    let body = app.metrics.render(&app.cache.stats(), healthy, app.embedder.consecutive_failures());
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": app.cfg.model_name, "object": "model", "owned_by": "local"}]}))
}

fn build_router(app: Arc<App>) -> Router {
    Router::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(app)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    let embedder = Arc::new(Embedder::load(&cfg.model_dir, cfg.threads, cfg.dim, cfg.max_tokens, cfg.health_max_failures)?);
    let t0 = Instant::now();
    embedder.self_check(cfg.max_tokens).context("startup self-check failed")?;
    eprintln!("self-check ok: dim={} max_tokens={} in {} ms", cfg.dim, cfg.max_tokens, t0.elapsed().as_millis());
    let port = cfg.port;
    let ttl = (cfg.cache_ttl_secs > 0).then(|| std::time::Duration::from_secs(cfg.cache_ttl_secs));
    let cache = EmbedCache::new((cfg.cache_max_mb * 1024.0 * 1024.0) as usize, cfg.cache_max_entries, ttl);
    let app = Arc::new(App { cfg, embedder, cache, metrics: Metrics::default() });
    let router = build_router(app);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    eprintln!("embed-svc listening on :{port}");
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
    use std::sync::{atomic::AtomicBool, Mutex};
    use std::time::Duration;

    /// Stand-in for the model: vectors are derived from the text, calls are recorded, failures can be injected.
    struct Fake {
        fail: AtomicBool,
        poisoned: AtomicBool,
        health: Health,
        seen: Mutex<Vec<Vec<String>>>,
    }

    impl Embed for Fake {
        fn embed(&self, texts: &[String]) -> anyhow::Result<Embedded> {
            self.seen.lock().unwrap().push(texts.to_vec());
            let r = if self.fail.load(Relaxed) {
                Err(anyhow::anyhow!("injected failure"))
            } else {
                let vectors = texts
                    .iter()
                    .map(|t| vec![t.chars().count() as f32, t.bytes().map(u32::from).sum::<u32>() as f32, 0.0, 1.0])
                    .collect();
                Ok(Embedded { vectors, tokens: texts.iter().map(|t| t.chars().count()).collect() })
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
            model_dir: PathBuf::from("unused"),
            model_name: "fake-model".into(),
            dim: 4,
            threads: 1,
            max_batch: 3,
            max_chars: 8,
            max_tokens: 512,
            cache_max_mb: 1.0,
            cache_max_entries: 100,
            cache_ttl_secs: 0,
            health_max_failures: 2,
            api_key: None,
        }
    }

    type Resp = (u16, String);

    fn json_of(r: &Resp) -> Value {
        serde_json::from_str(&r.1).unwrap_or_else(|e| panic!("not JSON ({e}): {}", r.1))
    }

    /// Minimal blocking HTTP/1.1 client (no extra dependencies); the server closes the connection after replying.
    fn raw_http(addr: SocketAddr, method: &str, path: &str, headers: &[(&str, String)], body: Option<&str>) -> Resp {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n");
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        if let Some(b) = body {
            req.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n", b.len()));
        }
        req.push_str("\r\n");
        if let Some(b) = body {
            req.push_str(b);
        }
        s.write_all(req.as_bytes()).unwrap();
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).unwrap();
        let text = String::from_utf8(raw).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").expect("no header/body separator");
        assert!(!head.to_ascii_lowercase().contains("transfer-encoding: chunked"), "chunked responses are not handled by this client");
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, body.to_string())
    }

    struct Harness {
        addr: SocketAddr,
        fake: Arc<Fake>,
    }

    async fn start(cfg: Config) -> Harness {
        let fake = Arc::new(Fake {
            fail: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            health: Health::new(cfg.health_max_failures),
            seen: Mutex::new(Vec::new()),
        });
        let cache = EmbedCache::new((cfg.cache_max_mb * 1024.0 * 1024.0) as usize, cfg.cache_max_entries, None);
        let app = Arc::new(App { cfg, embedder: fake.clone(), cache, metrics: Metrics::default() });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = build_router(app);
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        Harness { addr, fake }
    }

    impl Harness {
        async fn send(&self, method: &'static str, path: &'static str, headers: Vec<(&'static str, String)>, body: Option<String>) -> Resp {
            let addr = self.addr;
            tokio::task::spawn_blocking(move || raw_http(addr, method, path, &headers, body.as_deref())).await.unwrap()
        }

        async fn get(&self, path: &'static str) -> Resp {
            self.send("GET", path, vec![], None).await
        }

        async fn post(&self, body: &str) -> Resp {
            self.send("POST", "/v1/embeddings", vec![], Some(body.to_string())).await
        }

        async fn post_auth(&self, body: &str, authorization: &str) -> Resp {
            self.send("POST", "/v1/embeddings", vec![("Authorization", authorization.to_string())], Some(body.to_string())).await
        }

        async fn metric(&self, name: &str) -> u64 {
            let body = self.get("/metrics").await.1;
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
        assert_eq!(h.get("/health").await.0, 200, "one failure is below the threshold of 2");
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
    async fn bearer_auth_protects_embeddings_but_not_health_or_metrics() {
        let mut c = cfg();
        c.api_key = Some("k".into());
        let h = start(c).await;
        let body = r#"{"input":["a"]}"#;
        assert_eq!(h.post(body).await.0, 401);
        assert_eq!(h.post_auth(body, "Bearer wrong").await.0, 401);
        assert_eq!(h.post_auth(body, "k").await.0, 401, "the Bearer prefix is required");
        assert_eq!(h.post_auth(body, "Bearer k").await.0, 200);
        assert_eq!(h.get("/health").await.0, 200);
        assert_eq!(h.metric("embed_responses_unauthorized_total").await, 3);
        assert!(h.calls().len() == 1, "unauthorized requests must not reach the model");
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
            (r#"{"input":["a"],"encoding_format":"fp8"}"#, "unsupported encoding_format"),
            (r#"{"input":[1,2]}"#, "bad request body"),
        ];
        for (body, needle) in cases {
            let r = h.post(body).await;
            assert_eq!(r.0, 400, "{body} -> {}", r.1);
            let msg = json_of(&r)["error"]["message"].as_str().unwrap().to_string();
            assert!(msg.contains(needle), "{body}: {msg:?} should contain {needle:?}");
        }
        assert!(h.calls().is_empty());
        assert_eq!(h.metric("embed_responses_client_error_total").await, 7);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn accepts_string_or_array_input_and_matching_dimensions() {
        let h = start(cfg()).await;
        let one = json_of(&h.post(r#"{"input":"abc"}"#).await);
        assert_eq!(one["data"].as_array().unwrap().len(), 1);
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
        let b = json_of(&h.post(r#"{"input":["abc"],"encoding_format":"base64"}"#).await);
        let raw = base64::engine::general_purpose::STANDARD.decode(b["data"][0]["embedding"].as_str().unwrap()).unwrap();
        let decoded: Vec<f64> = raw.chunks(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap()))).collect();
        let expected: Vec<f64> = f["data"][0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded, expected);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duplicates_in_a_batch_and_repeat_requests_hit_the_model_once() {
        let h = start(cfg()).await;
        let r = json_of(&h.post(r#"{"input":["aa","aa","bb"]}"#).await);
        assert_eq!(r["data"][0]["embedding"], r["data"][1]["embedding"]);
        assert_ne!(r["data"][0]["embedding"], r["data"][2]["embedding"]);
        assert_eq!(h.calls(), vec![vec!["aa".to_string(), "bb".to_string()]], "duplicates are computed once");
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
        assert!(json_of(&r)["error"]["message"].as_str().unwrap().contains("injected failure"));
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
        let r = h.send("POST", "/embeddings", vec![], Some(r#"{"input":["a"]}"#.to_string())).await;
        assert_eq!(r.0, 200);
        assert_eq!(h.get("/nope").await.0, 404);
    }
}
