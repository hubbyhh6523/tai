use anyhow::{anyhow, Result};
use ort::environment::Environment;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A pooled ONNX session.
///
/// Dropping the guard returns its slot to the pool. The underlying session is
/// still protected by a `Mutex` because `ort::Session::run` takes `&mut self`.
pub struct PooledSession {
    pool: Arc<PoolInner>,
    index: usize,
}

impl PooledSession {
    pub fn session(&self) -> Result<&mut Session> {
        let mut slot = self.pool.slots[self.index]
            .lock()
            .map_err(|_| anyhow!("session {} lock poisoned", self.index))?;
        Ok(&mut slot)
    }
}

impl Drop for PooledSession {
    fn drop(&mut self) {
        let mut idle = self.pool.idle.lock().unwrap();
        idle.push_back(self.index);
        self.pool.available.notify_one();
    }
}

struct PoolInner {
    slots: Vec<Mutex<Session>>,
    idle: Mutex<VecDeque<usize>>,
    available: Condvar,
    queue_timeout: Option<Duration>,
}

/// A fixed-size pool of independently created ONNX sessions.
///
/// This removes the single global inference lock: up to `pool_size` requests
/// can run concurrently, while `acquire` bounds how long a caller waits for a
/// free session.
pub struct SessionPool {
    inner: Arc<PoolInner>,
    pool_size: usize,
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
        anyhow::ensure!(
            threads_per_session > 0,
            "threads per session must be greater than zero"
        );

        let mut slots = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let session = Session::builder(env)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads_per_session)?
                .commit_from_file(model_path)?;
            slots.push(Mutex::new(session));
        }

        let idle = (0..pool_size).collect::<VecDeque<_>>();

        Ok(Self {
            inner: Arc::new(PoolInner {
                slots,
                idle: Mutex::new(idle),
                available: Condvar::new(),
                queue_timeout,
            }),
            pool_size,
        })
    }

    /// Acquire an idle session, waiting at most `queue_timeout` when all are busy.
    pub fn acquire(&self) -> Result<PooledSession> {
        let mut idle = self.inner.idle.lock().unwrap();

        let index = match self.inner.queue_timeout {
            Some(timeout) => {
                let deadline = std::time::Instant::now() + timeout;
                loop {
                    if let Some(index) = idle.pop_front() {
                        break index;
                    }

                    let now = std::time::Instant::now();
                    if now >= deadline {
                        return Err(anyhow!(
                            "timed out after {:?} waiting for an embedding session",
                            timeout
                        ));
                    }

                    let (guard, _timeout) = self
                        .inner
                        .available
                        .wait_timeout(idle, deadline - now)
                        .map_err(|_| anyhow!("session pool lock poisoned"))?;
                    idle = guard;
                }
            }
            None => loop {
                if let Some(index) = idle.pop_front() {
                    break index;
                }
                idle = self
                    .inner
                    .available
                    .wait(idle)
                    .map_err(|_| anyhow!("session pool lock poisoned"))?;
            },
        };

        Ok(PooledSession {
            pool: self.inner.clone(),
            index,
        })
    }

    pub fn size(&self) -> usize {
        self.pool_size
    }

    /// True when any underlying session mutex has been poisoned by a panic.
    pub fn session_poisoned(&self) -> bool {
        self.inner.slots.iter().any(|slot| slot.is_poisoned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_tracks_size() {
        // This test does not create ONNX sessions; it only checks the public
        // size accessor contract through a manually constructed pool.
        let pool = SessionPool {
            inner: Arc::new(PoolInner {
                slots: Vec::new(),
                idle: Mutex::new(VecDeque::new()),
                available: Condvar::new(),
                queue_timeout: None,
            }),
            pool_size: 3,
        };

        assert_eq!(pool.size(), 3);
        assert!(!pool.session_poisoned());
    }
}
