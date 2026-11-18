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

#[cfg(test)]
mod regression {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{mpsc, Barrier};
    use std::thread;
    const WAIT: Duration = Duration::from_secs(5);
    fn pool(n: usize, timeout: Duration) -> Arc<SessionPool> {
        Arc::new(SessionPool { inner: Arc::new(PoolInner {
            slots: vec![], idle: Mutex::new((0..n).collect()),
            available: Condvar::new(), queue_timeout: Some(timeout),
        }) })
    }
    fn returned(p: &SessionPool, n: usize) {
        let idle = p.inner.idle.lock().unwrap();
        assert_eq!(idle.len(), n);
        assert_eq!(idle.iter().copied().collect::<HashSet<_>>(), (0..n).collect());
    }
    #[test]
    fn empty_pool_times_out() {
        let timeout = Duration::from_millis(20);
        let p = pool(0, timeout);
        let start = Instant::now();
        assert!(p.acquire().is_err());
        assert!(start.elapsed() >= timeout);
    }
    #[test]
    fn exhausted_pool_times_out_then_reuses_returned_slot() {
        let timeout = Duration::from_millis(30);
        let p = pool(2, timeout);
        let a = p.acquire().unwrap();
        let b = p.acquire().unwrap();
        assert_ne!(a.index, b.index);
        let start = Instant::now();
        assert!(p.acquire().is_err());
        assert!(start.elapsed() >= timeout);
        let index = a.index;
        drop(a);
        let c = p.acquire().unwrap();
        assert_eq!(c.index, index);
        drop(c);
        drop(b);
        returned(&p, 2);
    }
    #[test]
    fn returning_slot_unblocks_borrower() {
        let p = pool(1, WAIT);
        let held = p.acquire().unwrap();
        let index = held.index;
        let (ready_tx, ready_rx) = mpsc::channel();
        let (tx, rx) = mpsc::channel();
        let other = Arc::clone(&p);
        let worker = thread::spawn(move || {
            ready_tx.send(()).unwrap();
            let guard = other.acquire().unwrap();
            tx.send(guard.index).unwrap();
        });
        ready_rx.recv_timeout(WAIT).unwrap();
        let before = rx.recv_timeout(Duration::from_millis(30));
        drop(held);
        let after = rx.recv_timeout(WAIT);
        worker.join().unwrap();
        assert!(matches!(before, Err(mpsc::RecvTimeoutError::Timeout)));
        assert_eq!(after.unwrap(), index);
        returned(&p, 1);
    }
    #[test]
    fn early_error_returns_slot() {
        fn fail(p: &SessionPool) -> Result<()> {
            let _guard = p.acquire()?;
            Err(anyhow!("injected failure"))
        }
        let p = pool(1, WAIT);
        assert!(fail(&p).is_err());
        returned(&p, 1);
        drop(p.acquire().unwrap());
        returned(&p, 1);
    }
    #[test]
    fn panic_unwinding_returns_slot() {
        let p = pool(1, WAIT);
        let other = Arc::clone(&p);
        let worker = thread::spawn(move || {
            let _guard = other.acquire().unwrap();
            panic!("injected panic after checkout");
        });
        assert!(worker.join().is_err());
        returned(&p, 1);
        drop(p.acquire().unwrap());
        returned(&p, 1);
    }
    #[test]
    fn concurrent_borrowers_never_share_slots_or_exceed_capacity() {
        const SIZE: usize = 2;
        const WORKERS: usize = 8;
        let p = pool(SIZE, WAIT);
        let start = Arc::new(Barrier::new(WORKERS));
        let active = Arc::new(Mutex::new(HashSet::new()));
        let workers: Vec<_> = (0..WORKERS).map(|_| {
            let p = Arc::clone(&p);
            let start = Arc::clone(&start);
            let active = Arc::clone(&active);
            thread::spawn(move || {
                start.wait();
                for _ in 0..100 {
                    let guard = p.acquire().unwrap();
                    let index = guard.index;
                    {
                        let mut active = active.lock().unwrap();
                        assert!(index < SIZE);
                        assert!(active.insert(index), "slot {index} borrowed twice");
                        assert!(active.len() <= SIZE);
                    }
                    thread::yield_now();
                    assert!(active.lock().unwrap().remove(&index));
                    drop(guard);
                }
            })
        }).collect();
        let results: Vec<_> = workers.into_iter().map(|w| w.join()).collect();
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(active.lock().unwrap().is_empty());
        returned(&p, SIZE);
    }
}
