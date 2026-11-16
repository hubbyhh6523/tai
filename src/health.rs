use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Passive health tracking: counts consecutive inference failures and reports a poisoned model lock.
pub struct Health {
    consecutive_failures: AtomicU64,
    max_failures: u64,
}

impl Health {
    /// `max_failures == 0` disables the failure-count check (a poisoned lock is still reported).
    pub fn new(max_failures: u64) -> Self {
        Self { consecutive_failures: AtomicU64::new(0), max_failures }
    }

    pub fn record(&self, ok: bool) {
        if ok {
            self.consecutive_failures.store(0, Relaxed);
        } else {
            self.consecutive_failures.fetch_add(1, Relaxed);
        }
    }

    pub fn consecutive_failures(&self) -> u64 {
        self.consecutive_failures.load(Relaxed)
    }

    /// `Ok` while serving normally, otherwise the reason it is not.
    /// `poisoned` means the model session lock was poisoned by a panic during inference.
    pub fn check(&self, poisoned: bool) -> Result<(), String> {
        if poisoned {
            return Err("model session lock poisoned by an earlier panic; restart the service".into());
        }
        let n = self.consecutive_failures();
        if self.max_failures > 0 && n >= self.max_failures {
            return Err(format!("{n} consecutive inference failures"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_by_default() {
        assert!(Health::new(3).check(false).is_ok());
    }

    #[test]
    fn unhealthy_only_at_threshold() {
        let h = Health::new(3);
        h.record(false);
        h.record(false);
        assert!(h.check(false).is_ok());
        h.record(false);
        assert_eq!(h.check(false).unwrap_err(), "3 consecutive inference failures");
    }

    #[test]
    fn success_resets_the_streak() {
        let h = Health::new(2);
        h.record(false);
        h.record(true);
        h.record(false);
        assert!(h.check(false).is_ok());
        assert_eq!(h.consecutive_failures(), 1);
    }

    #[test]
    fn poisoned_lock_is_always_unhealthy() {
        let h = Health::new(0);
        assert!(h.check(true).unwrap_err().contains("poisoned"));
    }

    #[test]
    fn zero_threshold_disables_failure_check() {
        let h = Health::new(0);
        for _ in 0..100 {
            h.record(false);
        }
        assert!(h.check(false).is_ok());
    }
}
