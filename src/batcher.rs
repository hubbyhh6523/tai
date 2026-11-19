//! Cross-request dynamic batching.
//!
//! Concurrent embedding requests that arrive within a short window are
//! merged into one model call, so the per-run ONNX overhead (tokenizer
//! setup, session checkout, tensor allocation) is paid once instead of
//! once per request. The merged texts still go through the existing
//! `chunk_by_token_budget` / `MAX_TOKENS_PER_RUN` splitting inside the
//! embedder, so no second chunking scheme is introduced.
//!
//! Design (leader / follower, std-only, no new dependencies):
//!
//! * Each request first does its own cache lookup and per-request
//!   deduplication in `embed_with_cache`; only the cache-miss texts are
//!   submitted here. A request whose texts are all cache hits never
//!   touches the batcher and never waits for a window.
//! * A submission joins the most recent open batch if it still has room
//!   (fewer than `max_requests` jobs and fewer than `max_texts` texts).
//!   Otherwise it opens a new batch and becomes its leader.
//! * The leader waits up to `window` for more jobs, flushing early when
//!   the batch reaches `max_texts` or `max_requests`. A single job that
//!   already fills the batch flushes immediately, so large single
//!   requests pay no window at all. A lone request pays at most one
//!   window (shipped default 0 ms = batching off; set 5–10 ms on
//!   high-concurrency deployments).
//! * While one batch is running, new arrivals open a new batch and run
//!   concurrently on another pooled session, so batching never serialises
//!   the service behind a single inference and never bypasses the
//!   session pool: every merged run still acquires pooled sessions per
//!   chunk, with the existing queue timeout as backpressure.
//! * Results are mapped back per job, in each job's original order.
//!   Duplicate texts across jobs are inferred once.
//! * Failure isolation: if a merged run fails and the batch held more
//!   than one job, each job is retried on its own. A job whose texts are
//!   fine then succeeds even if another job's texts broke the merged
//!   run; a single-job batch fails straight through with no retry, so a
//!   lone bad request costs exactly one inference error.

use crate::cache::Entry;
use crate::embedder::{Embed, Embedded};
use crate::metrics::Metrics;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

/// What a job gets back: its entries in the order it submitted them,
/// or a per-job error string.
type JobReply = Result<Vec<Entry>, String>;

struct PendingJob {
    texts: Vec<String>,
    reply: mpsc::Sender<JobReply>,
}

struct OpenBatch {
    id: u64,
    jobs: Vec<PendingJob>,
    total_texts: usize,
}

struct State {
    batches: Vec<OpenBatch>,
    next_id: u64,
}

pub struct Batcher {
    embedder: Arc<dyn Embed>,
    metrics: Arc<Metrics>,
    window: Duration,
    max_texts: usize,
    max_requests: usize,
    state: Mutex<State>,
    cv: Condvar,
}

impl Batcher {
    pub fn new(
        embedder: Arc<dyn Embed>,
        metrics: Arc<Metrics>,
        window: Duration,
        max_texts: usize,
        max_requests: usize,
    ) -> Self {
        Self {
            embedder,
            metrics,
            window,
            max_texts: max_texts.max(1),
            max_requests: max_requests.max(1),
            state: Mutex::new(State {
                batches: Vec::new(),
                next_id: 0,
            }),
            cv: Condvar::new(),
        }
    }

    /// `true` when batching is on. A zero window disables it: submissions
    /// run directly with no waiting.
    pub fn enabled(&self) -> bool {
        !self.window.is_zero()
    }

    /// One model call, with the shared inference counters recorded.
    fn run_once(&self, texts: &[&str]) -> Result<Embedded> {
        let t0 = Instant::now();
        let r = self.embedder.embed(texts);
        self.metrics.inference_batches.fetch_add(1, Relaxed);
        self.metrics
            .inference_ms
            .fetch_add(t0.elapsed().as_millis() as u64, Relaxed);
        if r.is_err() {
            self.metrics.inference_errors.fetch_add(1, Relaxed);
        }
        r
    }

    fn entries_of(&self, texts: &[&str]) -> Result<Vec<Entry>> {
        let emb = self.run_once(texts)?;
        if emb.vectors.len() != texts.len() || emb.tokens.len() != texts.len() {
            return Err(anyhow!(
                "embedder returned {} vectors for {} texts",
                emb.vectors.len(),
                texts.len()
            ));
        }
        Ok(emb_to_entries(emb))
    }

    /// Submit this request's cache-miss texts and wait for their entries,
    /// in submission order. Blocks the calling (spawn_blocking) thread.
    pub fn submit(&self, texts: Vec<String>) -> Result<Vec<Entry>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if !self.enabled() {
            let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
            return self.entries_of(&refs);
        }
        let (tx, rx) = mpsc::channel::<JobReply>();
        let job = PendingJob { texts, reply: tx };
        // Decide leader vs follower under the lock, but do NOT wait for a
        // reply while holding it: the leader needs the same lock to seal
        // the batch, so waiting here would deadlock leader and follower.
        let leader_id: Option<u64> = {
            let mut guard = self
                .state
                .lock()
                .map_err(|_| anyhow!("batcher lock poisoned"))?;
            match guard.batches.last_mut() {
                Some(batch)
                    if batch.jobs.len() < self.max_requests
                        && batch.total_texts < self.max_texts =>
                {
                    batch.total_texts += job.texts.len();
                    batch.jobs.push(job);
                    if batch.total_texts >= self.max_texts || batch.jobs.len() >= self.max_requests
                    {
                        // The leader may already be waiting for the window;
                        // wake it so the full batch flushes now.
                        self.cv.notify_all();
                    }
                    None
                }
                _ => {
                    let id = guard.next_id;
                    guard.next_id += 1;
                    let total_texts = job.texts.len();
                    guard.batches.push(OpenBatch {
                        id,
                        jobs: vec![job],
                        total_texts,
                    });
                    Some(id)
                }
            }
        };
        match leader_id {
            Some(id) => self.lead(id, rx),
            None => self.wait_reply(rx),
        }
    }

    fn wait_reply(&self, rx: mpsc::Receiver<JobReply>) -> Result<Vec<Entry>> {
        match rx.recv() {
            Ok(Ok(entries)) => Ok(entries),
            Ok(Err(msg)) => Err(anyhow!("{msg}")),
            Err(_) => Err(anyhow!("batcher leader dropped before replying")),
        }
    }

    fn should_flush(&self, id: u64) -> bool {
        let Ok(guard) = self.state.lock() else {
            return true;
        };
        match guard.batches.iter().find(|b| b.id == id) {
            Some(batch) => {
                batch.total_texts >= self.max_texts || batch.jobs.len() >= self.max_requests
            }
            // Already sealed by someone else; nothing left to wait for.
            None => true,
        }
    }

    fn lead(&self, id: u64, rx: mpsc::Receiver<JobReply>) -> Result<Vec<Entry>> {
        let deadline = Instant::now() + self.window;
        loop {
            if self.should_flush(id) {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let guard = self
                .state
                .lock()
                .map_err(|_| anyhow!("batcher lock poisoned"))?;
            let (guard, _) = self
                .cv
                .wait_timeout(guard, remaining)
                .map_err(|_| anyhow!("batcher lock poisoned"))?;
            drop(guard);
        }
        let batch = {
            let mut guard = self
                .state
                .lock()
                .map_err(|_| anyhow!("batcher lock poisoned"))?;
            let pos = guard
                .batches
                .iter()
                .position(|b| b.id == id)
                .ok_or_else(|| anyhow!("batch {id} already sealed"))?;
            let batch = guard.batches.remove(pos);
            self.cv.notify_all();
            batch
        };
        self.run_batch(batch);
        self.wait_reply(rx)
    }

    /// Run one batch and reply to every job in it, including the leader's
    /// own job (the leader picks its reply up from its channel).
    fn run_batch(&self, batch: OpenBatch) {
        let jobs = batch.jobs;
        self.metrics.batch_runs.fetch_add(1, Relaxed);
        self.metrics
            .batch_jobs
            .fetch_add(jobs.len() as u64, Relaxed);
        if jobs.len() == 1 {
            let job = &jobs[0];
            let refs: Vec<&str> = job.texts.iter().map(|t| t.as_str()).collect();
            let reply = self.entries_of(&refs).map_err(|e| format!("{e:#}"));
            let _ = job.reply.send(reply);
            return;
        }
        // Merge and deduplicate across jobs, keeping first-seen order.
        let mut uniq: Vec<&str> = Vec::new();
        let mut index_of: HashMap<&str, usize> = HashMap::new();
        for job in &jobs {
            for t in &job.texts {
                if !index_of.contains_key(t.as_str()) {
                    index_of.insert(t.as_str(), uniq.len());
                    uniq.push(t.as_str());
                }
            }
        }
        self.metrics
            .batch_texts
            .fetch_add(uniq.len() as u64, Relaxed);
        match self.entries_of(&uniq) {
            Ok(entries) => {
                for job in &jobs {
                    let mine: Vec<Entry> = job
                        .texts
                        .iter()
                        .map(|t| entries[index_of[t.as_str()]].clone())
                        .collect();
                    let _ = job.reply.send(Ok(mine));
                }
            }
            Err(e) => {
                // Failure isolation: the merged run failed, but the fault
                // may belong to one job's texts. Retry each job alone so
                // the healthy ones still get their vectors.
                eprintln!(
                    "WARN merged batch of {} jobs failed ({e:#}); retrying each job alone",
                    jobs.len()
                );
                for job in &jobs {
                    let refs: Vec<&str> = job.texts.iter().map(|t| t.as_str()).collect();
                    let reply = self.entries_of(&refs).map_err(|e| format!("{e:#}"));
                    let _ = job.reply.send(reply);
                }
            }
        }
    }
}

fn emb_to_entries(emb: Embedded) -> Vec<Entry> {
    emb.vectors
        .into_iter()
        .zip(emb.tokens)
        .map(|(v, tokens)| Entry {
            vec: Arc::new(v),
            tokens,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::Health;
    use std::collections::HashSet;
    use std::sync::Barrier;

    /// Fake model: vector is [text length, byte sum]; fails on any text
    /// in `fail_texts`, so a merged run containing a bad text fails while
    /// a run without it succeeds.
    struct FakeEmbed {
        seen: Mutex<Vec<Vec<String>>>,
        fail_texts: HashSet<String>,
        health: Health,
    }

    impl FakeEmbed {
        fn new(fail_texts: &[&str]) -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
                fail_texts: fail_texts.iter().map(|s| s.to_string()).collect(),
                health: Health::new(3),
            }
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl Embed for FakeEmbed {
        fn embed(&self, texts: &[&str]) -> Result<Embedded> {
            self.seen
                .lock()
                .unwrap()
                .push(texts.iter().map(|t| t.to_string()).collect());
            if texts.iter().any(|t| self.fail_texts.contains(*t)) {
                return Err(anyhow!("bad text in batch"));
            }
            Ok(Embedded {
                vectors: texts
                    .iter()
                    .map(|t| {
                        vec![
                            t.chars().count() as f32,
                            t.bytes().map(u32::from).sum::<u32>() as f32,
                        ]
                    })
                    .collect(),
                tokens: texts.iter().map(|t| t.chars().count()).collect(),
            })
        }

        fn health_check(&self) -> Result<(), String> {
            self.health.check(false)
        }

        fn consecutive_failures(&self) -> u64 {
            0
        }
    }

    fn batcher(
        fake: Arc<FakeEmbed>,
        window_ms: u64,
        max_texts: usize,
        max_requests: usize,
    ) -> Batcher {
        Batcher::new(
            fake as Arc<dyn Embed>,
            Arc::new(Metrics::default()),
            Duration::from_millis(window_ms),
            max_texts,
            max_requests,
        )
    }

    fn one() -> Arc<FakeEmbed> {
        Arc::new(FakeEmbed::new(&[]))
    }

    #[test]
    fn single_request_preserves_order_and_runs_once() {
        let fake = one();
        let b = batcher(fake.clone(), 50, 32, 8);
        let out = b
            .submit(vec!["bb".into(), "a".into(), "ccc".into()])
            .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].vec[0], 2.0);
        assert_eq!(out[1].vec[0], 1.0);
        assert_eq!(out[2].vec[0], 3.0);
        assert_eq!(fake.calls().len(), 1);
    }

    #[test]
    fn concurrent_requests_merge_into_one_call() {
        let fake = one();
        let b = Arc::new(batcher(fake.clone(), 100, 32, 8));
        let start = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let b = Arc::clone(&b);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    let text = format!("t{i}");
                    let out = b.submit(vec![text]).unwrap();
                    assert_eq!(out.len(), 1);
                    assert_eq!(out[0].vec[0], 2.0, "each job gets its own vector");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            fake.calls().len(),
            1,
            "8 simultaneous single-text jobs should merge into one model call, got {:?}",
            fake.calls()
        );
    }

    #[test]
    fn window_timeout_flushes_a_lone_request() {
        let fake = one();
        let b = batcher(fake.clone(), 50, 32, 8);
        let t0 = Instant::now();
        let out = b.submit(vec!["abc".into()]).unwrap();
        assert_eq!(out[0].vec[0], 3.0);
        assert!(
            t0.elapsed() >= Duration::from_millis(40),
            "a lone request waits out the window: {:?}",
            t0.elapsed()
        );
        assert_eq!(fake.calls().len(), 1);
    }

    #[test]
    fn reaching_max_texts_flushes_before_the_window() {
        let fake = one();
        let b = Arc::new(batcher(fake.clone(), 500, 4, 8));
        let start = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let b = Arc::clone(&b);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    b.submit(vec![format!("x{i}")]).unwrap();
                })
            })
            .collect();
        let t0 = Instant::now();
        for h in handles {
            h.join().unwrap();
        }
        assert!(
            t0.elapsed() < Duration::from_millis(400),
            "4 texts hit max_texts=4 and should flush well before the 500 ms window: {:?}",
            t0.elapsed()
        );
        assert_eq!(fake.calls().len(), 1);
        assert_eq!(fake.calls()[0].len(), 4);
    }

    #[test]
    fn reaching_max_requests_flushes_before_the_window() {
        let fake = one();
        let b = Arc::new(batcher(fake.clone(), 500, 100, 2));
        let start = Arc::new(Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|i| {
                let b = Arc::clone(&b);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    b.submit(vec![format!("y{i}")]).unwrap();
                })
            })
            .collect();
        let t0 = Instant::now();
        for h in handles {
            h.join().unwrap();
        }
        assert!(
            t0.elapsed() < Duration::from_millis(400),
            "2 jobs hit max_requests=2 and should flush well before the 500 ms window: {:?}",
            t0.elapsed()
        );
        assert_eq!(fake.calls().len(), 1);
    }

    #[test]
    fn duplicates_across_jobs_are_inferred_once_and_mapped_back() {
        let fake = one();
        let b = Arc::new(batcher(fake.clone(), 100, 32, 8));
        let start = Arc::new(Barrier::new(2));
        let h1 = {
            let b = Arc::clone(&b);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                let out = b.submit(vec!["aa".into(), "bb".into()]).unwrap();
                assert_eq!(out[0].vec[0], 2.0);
                assert_eq!(out[1].vec[0], 2.0);
                // aa and bb both have length 2; tell them apart by byte sum.
                assert_ne!(out[0].vec[1], out[1].vec[1]);
            })
        };
        let h2 = {
            let b = Arc::clone(&b);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                let out = b.submit(vec!["bb".into(), "ccc".into()]).unwrap();
                assert_eq!(out[0].vec[0], 2.0);
                assert_eq!(out[1].vec[0], 3.0);
            })
        };
        h1.join().unwrap();
        h2.join().unwrap();
        let calls = fake.calls();
        assert_eq!(
            calls.len(),
            1,
            "merged run should be a single call: {calls:?}"
        );
        let mut merged = calls[0].clone();
        merged.sort();
        assert_eq!(merged, vec!["aa", "bb", "ccc"]);
    }

    #[test]
    fn failure_isolation_healthy_job_succeeds_when_a_sibling_fails() {
        let fake = Arc::new(FakeEmbed::new(&["bad"]));
        let b = Arc::new(batcher(fake.clone(), 100, 32, 8));
        let start = Arc::new(Barrier::new(2));
        let good = {
            let b = Arc::clone(&b);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                b.submit(vec!["good".into()])
            })
        };
        let bad = {
            let b = Arc::clone(&b);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                b.submit(vec!["bad".into()])
            })
        };
        let good = good.join().unwrap();
        let bad = bad.join().unwrap();
        assert!(
            good.is_ok(),
            "healthy job must not fail with its batch-mate"
        );
        assert_eq!(good.unwrap()[0].vec[0], 4.0);
        assert!(bad.is_err(), "the bad job must still fail on its own");
        // Merged run (fails) + one retry per job = 3 calls.
        assert_eq!(fake.calls().len(), 3, "{:?}", fake.calls());
    }

    #[test]
    fn single_bad_job_fails_without_a_retry() {
        let fake = Arc::new(FakeEmbed::new(&["bad"]));
        let b = batcher(fake.clone(), 50, 32, 8);
        assert!(b.submit(vec!["bad".into()]).is_err());
        assert_eq!(fake.calls().len(), 1, "a lone job costs exactly one call");
    }

    #[test]
    fn disabled_window_runs_directly() {
        let fake = one();
        let b = batcher(fake.clone(), 0, 32, 8);
        assert!(!b.enabled());
        let t0 = Instant::now();
        let out = b.submit(vec!["abc".into()]).unwrap();
        assert_eq!(out[0].vec[0], 3.0);
        assert!(
            t0.elapsed() < Duration::from_millis(100),
            "window 0 must not wait: {:?}",
            t0.elapsed()
        );
        assert_eq!(fake.calls().len(), 1);
    }

    #[test]
    fn large_single_job_flushes_without_waiting() {
        let fake = one();
        let b = batcher(fake.clone(), 500, 4, 8);
        let t0 = Instant::now();
        let out = b
            .submit(vec![
                "a".into(),
                "b".into(),
                "c".into(),
                "d".into(),
                "e".into(),
            ])
            .unwrap();
        assert_eq!(out.len(), 5);
        assert!(
            t0.elapsed() < Duration::from_millis(400),
            "a job already over max_texts must not wait out the window: {:?}",
            t0.elapsed()
        );
    }
}
