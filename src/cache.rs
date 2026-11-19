use lru::LruCache;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

type Key = [u8; 16];

/// 128-bit content hash: keeps long texts out of memory and makes entry size predictable.
fn key_of(text: &str) -> Key {
    let h = blake3::hash(text.as_bytes());
    let mut k = [0u8; 16];
    k.copy_from_slice(&h.as_bytes()[..16]);
    k
}

#[derive(Clone)]
pub struct Entry {
    pub vec: Arc<Vec<f32>>,
    pub tokens: usize,
}

struct Slot {
    entry: Entry,
    bytes: usize,
    at: Instant,
}

struct Inner {
    map: LruCache<Key, Slot>,
    bytes: usize,
}

pub struct CacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub max_bytes: usize,
    pub max_entries: usize,
    pub evictions: u64,
    pub expired: u64,
}

/// LRU bounded by a byte budget and an entry cap, with an optional TTL checked lazily on read.
/// A zero budget or zero entry cap disables the cache.
pub struct EmbedCache {
    inner: Option<Mutex<Inner>>,
    max_bytes: usize,
    max_entries: usize,
    ttl: Option<Duration>,
    evictions: AtomicU64,
    expired: AtomicU64,
}

const PER_ENTRY_OVERHEAD: usize = 16 + 64; // key + slot/bookkeeping estimate

fn cost(e: &Entry) -> usize {
    e.vec.len() * 4 + PER_ENTRY_OVERHEAD
}

impl EmbedCache {
    pub fn new(max_bytes: usize, max_entries: usize, ttl: Option<Duration>) -> Self {
        let on = max_bytes > 0 && max_entries > 0;
        let inner = on.then(|| {
            Mutex::new(Inner {
                map: LruCache::unbounded(),
                bytes: 0,
            })
        });
        Self {
            inner,
            max_bytes,
            max_entries,
            ttl,
            evictions: AtomicU64::new(0),
            expired: AtomicU64::new(0),
        }
    }

    pub fn get(&self, text: &str) -> Option<Entry> {
        let mut g = self.inner.as_ref()?.lock().ok()?;
        let key = key_of(text);
        let slot = g.map.peek(&key)?;
        let stale = self.ttl.is_some_and(|t| slot.at.elapsed() > t);
        if stale {
            if let Some(s) = g.map.pop(&key) {
                g.bytes -= s.bytes;
            }
            self.expired.fetch_add(1, Relaxed);
            return None;
        }
        g.map.get(&key).map(|s| s.entry.clone())
    }

    pub fn put(&self, text: &str, e: Entry) {
        let Some(Ok(mut g)) = self.inner.as_ref().map(|m| m.lock()) else {
            return;
        };
        let c = cost(&e);
        if c > self.max_bytes {
            return; // a single entry larger than the whole budget is never admitted
        }
        let key = key_of(text);
        if let Some(old) = g.map.pop(&key) {
            g.bytes -= old.bytes;
        }
        g.map.put(
            key,
            Slot {
                entry: e,
                bytes: c,
                at: Instant::now(),
            },
        );
        g.bytes += c;
        while g.bytes > self.max_bytes || g.map.len() > self.max_entries {
            match g.map.pop_lru() {
                Some((_, s)) => {
                    g.bytes -= s.bytes;
                    self.evictions.fetch_add(1, Relaxed);
                }
                None => break,
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        let (entries, bytes) = self
            .inner
            .as_ref()
            .and_then(|m| m.lock().ok())
            .map(|g| (g.map.len(), g.bytes))
            .unwrap_or((0, 0));
        CacheStats {
            entries,
            bytes,
            max_bytes: if self.inner.is_some() {
                self.max_bytes
            } else {
                0
            },
            max_entries: if self.inner.is_some() {
                self.max_entries
            } else {
                0
            },
            evictions: self.evictions.load(Relaxed),
            expired: self.expired.load(Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(dim: usize) -> Entry {
        Entry {
            vec: Arc::new(vec![0.5; dim]),
            tokens: 1,
        }
    }
    fn one() -> usize {
        cost(&e(10))
    }

    #[test]
    fn evicts_least_recently_used_by_bytes() {
        let c = EmbedCache::new(one() * 2, 100, None);
        c.put("a", e(10));
        c.put("b", e(10));
        assert!(c.get("a").is_some()); // a is now most recent
        c.put("c", e(10)); // over budget: evicts b
        assert!(c.get("b").is_none());
        assert!(c.get("a").is_some() && c.get("c").is_some());
        let s = c.stats();
        assert_eq!((s.entries, s.evictions), (2, 1));
        assert!(s.bytes <= s.max_bytes);
    }

    #[test]
    fn entry_cap_applies_independently() {
        let c = EmbedCache::new(1 << 20, 2, None);
        for k in ["a", "b", "c"] {
            c.put(k, e(10));
        }
        assert_eq!(c.stats().entries, 2);
        assert!(c.get("a").is_none());
    }

    #[test]
    fn oversized_entry_is_not_admitted() {
        let c = EmbedCache::new(one(), 10, None);
        c.put("big", e(1000));
        assert_eq!(c.stats().entries, 0);
        assert!(c.get("big").is_none());
    }

    #[test]
    fn overwrite_keeps_byte_accounting_exact() {
        let c = EmbedCache::new(1 << 20, 10, None);
        c.put("a", e(10));
        c.put("a", e(10));
        assert_eq!(c.stats().bytes, one());
        assert_eq!(c.stats().entries, 1);
    }

    #[test]
    fn ttl_expires_lazily() {
        let c = EmbedCache::new(1 << 20, 10, Some(Duration::from_millis(30)));
        c.put("a", e(10));
        assert!(c.get("a").is_some());
        std::thread::sleep(Duration::from_millis(60));
        assert!(c.get("a").is_none());
        let s = c.stats();
        assert_eq!((s.entries, s.bytes, s.expired), (0, 0, 1));
    }

    #[test]
    fn zero_budget_disables() {
        let c = EmbedCache::new(0, 10, None);
        c.put("a", e(10));
        assert!(c.get("a").is_none());
        assert_eq!(c.stats().max_bytes, 0);
    }
}
