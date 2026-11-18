use anyhow::{anyhow, Context, Result};
use ort::environment::Environment;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub struct PooledSession {
    pool: Arc<PoolInner>,
    index: usize,
}

impl PooledSession {
    pub fn session(&self) -> Result<MutexGuard<'_, Session>> {
        self.pool.slots[self.index]
            .lock()
            .map_err(|_| anyhow!("session {} lock poisoned", self.index))
    }
}

impl Drop for PooledSession {
    fn drop(&mut self) {
        let mut idle = self.pool.idle.lock().unwrap_or_else(|e| e.into_inner());
        idle.push_back(self.index);
        drop(idle);
        self.pool.available.notify_one();
    }
}

struct PoolInner {
    slots: Vec<Mutex<Session>>,
    idle: Mutex<VecDeque<usize>>,
    available: Condvar,
    queue_timeout: Option<Duration>,
}

pub struct SessionPool {
    inner: Arc<PoolInner>,
}

fn oe(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{e}")
}

impl SessionPool {
    pub fn new(
        env: &Environment,
        model_path: &Path,
        pool_size: usize,
        threads_per_session: usize,
        queue_timeout: Option<Duration>,
    ) -> Result<Self> {
        anyhow::ensure!(pool_size > 0, "session pool size must be greater than zero");
        anyhow::ensure!(threads_per_session > 0, "threads per session must be greater than zero");
        let mut slots = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let session = Session::builder(env)
                .map_err(oe)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(oe)?
                .with_intra_threads(threads_per_session)
                .map_err(oe)?
                .commit_from_file(model_path)
                .map_err(oe)
                .with_context(|| format!("load ONNX session {i}"))?;
            slots.push(Mutex::new(session));
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                slots,
                idle: Mutex::new((0..pool_size).collect()),
                available: Condvar::new(),
                queue_timeout,
            }),
        })
    }

    pub fn acquire(&self) -> Result<PooledSession> {
        let mut idle = self.inner.idle.lock().map_err(|_| anyhow!("session pool lock poisoned"))?;
        let start = Instant::now();
        let index = loop {
            if let Some(index) = idle.pop_front() {
                break index;
            }
            idle = match self.inner.queue_timeout {
                Some(timeout) => {
                    let remaining = timeout.checked_sub(start.elapsed())
                        .ok_or_else(|| anyhow!("timed out waiting for an embedding session"))?;
                    let (guard, result) = self.inner.available.wait_timeout(idle, remaining)
                        .map_err(|_| anyhow!("session pool lock poisoned"))?;
                    if result.timed_out() && guard.is_empty() {
                        return Err(anyhow!("timed out waiting for an embedding session"));
                    }
                    guard
                }
                None => self.inner.available.wait(idle)
                    .map_err(|_| anyhow!("session pool lock poisoned"))?,
            };
        };
        Ok(PooledSession { pool: self.inner.clone(), index })
    }

    pub fn session_poisoned(&self) -> bool {
        self.inner.idle.is_poisoned() || self.inner.slots.iter().any(|slot| slot.is_poisoned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pool_times_out_without_model() {
        let pool = SessionPool {
            inner: Arc::new(PoolInner {
                slots: vec![],
                idle: Mutex::new(VecDeque::new()),
                available: Condvar::new(),
                queue_timeout: Some(Duration::from_millis(10)),
            }),
        };
        assert!(pool.acquire().is_err());
        assert!(!pool.session_poisoned());
    }
}
