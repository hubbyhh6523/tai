use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub ok: AtomicU64,
    pub unauthorized: AtomicU64,
    pub client_errors: AtomicU64,
    pub server_errors: AtomicU64,
    pub inference_errors: AtomicU64,
    pub inference_batches: AtomicU64,
    pub inference_ms: AtomicU64,
    pub texts: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
}

impl Metrics {
    pub fn record_status(&self, status: u16) {
        let c = match status {
            200..=299 => &self.ok,
            401 => &self.unauthorized,
            400..=499 => &self.client_errors,
            _ => &self.server_errors,
        };
        c.fetch_add(1, Relaxed);
    }

    /// Prometheus text exposition format.
    pub fn render(&self, c: &crate::cache::CacheStats, healthy: bool, consecutive_failures: u64) -> String {
        let mut s = String::new();
        let mut line = |name: &str, help: &str, kind: &str, v: u64| {
            s.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}\n"));
        };
        line("embed_requests_total", "HTTP requests to the embeddings endpoint", "counter", self.requests.load(Relaxed));
        line("embed_responses_ok_total", "2xx responses", "counter", self.ok.load(Relaxed));
        line("embed_responses_unauthorized_total", "401 responses", "counter", self.unauthorized.load(Relaxed));
        line("embed_responses_client_error_total", "other 4xx responses", "counter", self.client_errors.load(Relaxed));
        line("embed_responses_server_error_total", "5xx responses; the caller loses vectors on these", "counter", self.server_errors.load(Relaxed));
        line("embed_inference_errors_total", "failed model inference calls", "counter", self.inference_errors.load(Relaxed));
        line("embed_inference_batches_total", "model inference calls", "counter", self.inference_batches.load(Relaxed));
        line("embed_inference_milliseconds_total", "time spent in model inference", "counter", self.inference_ms.load(Relaxed));
        line("embed_texts_total", "texts received", "counter", self.texts.load(Relaxed));
        line("embed_cache_hits_total", "texts served from cache", "counter", self.cache_hits.load(Relaxed));
        line("embed_cache_misses_total", "texts needing inference", "counter", self.cache_misses.load(Relaxed));
        line("embed_healthy", "1 when the model is serving normally, 0 otherwise", "gauge", u64::from(healthy));
        line("embed_consecutive_inference_failures", "inference failures since the last success", "gauge", consecutive_failures);
        line("embed_cache_entries", "entries currently cached", "gauge", c.entries as u64);
        line("embed_cache_bytes", "estimated bytes held by the cache", "gauge", c.bytes as u64);
        line("embed_cache_max_bytes", "cache byte budget (0 = disabled)", "gauge", c.max_bytes as u64);
        line("embed_cache_max_entries", "cache entry cap (0 = disabled)", "gauge", c.max_entries as u64);
        line("embed_cache_evictions_total", "entries evicted to stay within budget", "counter", c.evictions);
        line("embed_cache_expired_total", "entries dropped because their TTL elapsed", "counter", c.expired);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_classification() {
        let m = Metrics::default();
        for s in [200, 400, 401, 404, 500, 503] {
            m.record_status(s);
        }
        assert_eq!(m.ok.load(Relaxed), 1);
        assert_eq!(m.unauthorized.load(Relaxed), 1);
        assert_eq!(m.client_errors.load(Relaxed), 2);
        assert_eq!(m.server_errors.load(Relaxed), 2);
    }

    #[test]
    fn render_contains_series() {
        let m = Metrics::default();
        m.cache_hits.fetch_add(3, Relaxed);
        let st = crate::cache::CacheStats { entries: 2, bytes: 4096, max_bytes: 8192, max_entries: 10, evictions: 5, expired: 1 };
        let out = m.render(&st, true, 2);
        assert!(out.contains("embed_cache_hits_total 3"));
        assert!(out.contains("embed_cache_entries 2"));
        assert!(out.contains("embed_cache_evictions_total 5"));
        assert!(out.contains("embed_cache_max_bytes 8192"));
        assert!(out.contains("embed_healthy 1"));
        assert!(out.contains("embed_consecutive_inference_failures 2"));
        assert!(out.contains("# TYPE embed_responses_server_error_total counter"));
    }
}
