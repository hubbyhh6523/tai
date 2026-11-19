#!/usr/bin/env python3
"""Cache policy checks. MODE=evict needs a server started with a tiny CACHE_MAX_MB (e.g. 0.01);
MODE=ttl needs CACHE_TTL_SECS=1. Usage: MODE=evict BASE=... python scripts/smoke_cache.py"""
import json, os, sys, time, urllib.request
BASE = os.getenv("BASE", "http://127.0.0.1:8090"); MODE = os.getenv("MODE", "evict"); KEY = os.getenv("API_KEY", "")
tag = str(os.getpid())

def post(texts):
    h = {"Content-Type": "application/json"}
    if KEY: h["Authorization"] = f"Bearer {KEY}"
    r = urllib.request.Request(BASE + "/v1/embeddings", json.dumps({"model": "m", "input": texts}).encode(), h)
    with urllib.request.urlopen(r, timeout=30) as x: return json.load(x)

def metrics():
    h = {}
    if KEY:
        h["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(BASE + "/metrics", headers=h)
    with urllib.request.urlopen(req, timeout=10) as x:
        return {l.split()[0]: int(l.split()[1]) for l in x.read().decode().splitlines() if l and not l.startswith("#")}

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS" if cond else "FAIL"), name, extra)

if MODE == "evict":
    texts = [f"淘汰测试文本-{tag}-{i}" for i in range(20)]
    for t in texts: post([t])
    m = metrics()
    check("evictions happened", m["embed_cache_evictions_total"] > 0, f"evictions={m['embed_cache_evictions_total']}")
    check("bytes within budget", 0 < m["embed_cache_bytes"] <= m["embed_cache_max_bytes"], f"{m['embed_cache_bytes']}/{m['embed_cache_max_bytes']}")
    check("entries below inserted", m["embed_cache_entries"] < 20, f"entries={m['embed_cache_entries']}")
    h0 = m["embed_cache_hits_total"]; post([texts[-1]]); m1 = metrics()
    check("most recent still cached", m1["embed_cache_hits_total"] == h0 + 1)
    ms0 = m1["embed_cache_misses_total"]; post([texts[0]]); m2 = metrics()
    check("oldest was evicted", m2["embed_cache_misses_total"] == ms0 + 1)
elif MODE == "ttl":
    t = f"过期测试文本-{tag}"; post([t]); m0 = metrics(); post([t]); m1 = metrics()
    check("hit before ttl", m1["embed_cache_hits_total"] == m0["embed_cache_hits_total"] + 1)
    time.sleep(1.6); post([t]); m2 = metrics()
    check("miss after ttl", m2["embed_cache_misses_total"] == m1["embed_cache_misses_total"] + 1)
    check("expiry counted", m2["embed_cache_expired_total"] >= 1)
sys.exit(0 if ok else 1)
