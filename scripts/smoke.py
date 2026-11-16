#!/usr/bin/env python3
"""Smoke test: BASE=http://127.0.0.1:8090 [API_KEY=..] python scripts/smoke.py"""
import base64, json, math, os, struct, sys, urllib.error, urllib.request
BASE = os.getenv("BASE", "http://127.0.0.1:8090"); KEY = os.getenv("API_KEY", ""); DIM = int(os.getenv("DIM", "512"))

def post(body, key=KEY, path="/v1/embeddings"):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = f"Bearer {key}"
    r = urllib.request.Request(BASE + path, json.dumps(body).encode(), h)
    try:
        with urllib.request.urlopen(r, timeout=30) as x: return x.status, json.load(x)
    except urllib.error.HTTPError as e: return e.code, json.load(e)

def dot(a, b): return sum(x * y for x, y in zip(a, b))
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS" if cond else "FAIL"), name, extra)

s, r = post({"model": "m", "input": ["我喜欢用 Rust 写网络服务", "我常用 Rust 做网络编程", "今天天气很好适合野餐"], "dimensions": DIM})
check("batch 200", s == 200)
v = [d["embedding"] for d in r["data"]]
check("indices ordered", [d["index"] for d in r["data"]] == [0, 1, 2])
check("dimension", all(len(x) == DIM for x in v), f"dim={len(v[0])}")
check("unit norm", all(abs(math.sqrt(dot(x, x)) - 1) < 1e-4 for x in v))
sim_close, sim_far = dot(v[0], v[1]), dot(v[0], v[2])
check("semantic order", sim_close > sim_far + 0.05, f"similar={sim_close:.3f} unrelated={sim_far:.3f}")
print("usage", r["usage"])

s, r1 = post({"model": "m", "input": "我喜欢用 Rust 写网络服务"})
check("string input + no dimensions", s == 200 and len(r1["data"]) == 1)
check("single == batch", dot(r1["data"][0]["embedding"], v[0]) > 0.9999, f"cos={dot(r1['data'][0]['embedding'], v[0]):.6f}")

s, rb = post({"model": "m", "input": ["我喜欢用 Rust 写网络服务"], "encoding_format": "base64"})
raw = base64.b64decode(rb["data"][0]["embedding"]); arr = struct.unpack(f"<{len(raw)//4}f", raw)
check("base64 == float", s == 200 and dot(arr, v[0]) > 0.9999)

s, re_ = post({"model": "m", "input": ["x"], "dimensions": DIM + 1}); check("dimension mismatch -> 400", s == 400, re_.get("error", {}).get("message", "")[:60])
s, _ = post({"model": "m", "input": []}); check("empty input -> 400", s == 400)
s, _ = post({"model": "m"}); check("missing input -> 400", s == 400)
s, _ = post({"model": "m", "input": ["a"] * 257}); check("batch 257 -> 400", s == 400)
s, rl = post({"model": "m", "input": ["长" * 20000]}); check("long text truncated, 200", s == 200)
s, rm = post({"model": "m", "input": ["", "完整句子"]}); check("empty string item ok", s == 200 and len(rm["data"]) == 2)
if KEY:
    s, _ = post({"model": "m", "input": ["x"]}, key="wrong"); check("bad key -> 401", s == 401)

def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as x:
        return {l.split()[0]: int(l.split()[1]) for l in x.read().decode().splitlines() if l and not l.startswith("#")}

m0 = metrics()
t = "缓存验证专用文本-" + str(os.getpid())
s, a = post({"model": "m", "input": [t, t, "另一条缓存验证文本-" + str(os.getpid())]})
m1 = metrics()
check("in-batch duplicate computed once", m1["embed_inference_batches_total"] - m0["embed_inference_batches_total"] == 1 and a["data"][0]["embedding"] == a["data"][1]["embedding"])
check("misses counted", m1["embed_cache_misses_total"] - m0["embed_cache_misses_total"] == 3)
s, b = post({"model": "m", "input": [t]})
m2 = metrics()
check("repeat served from cache", m2["embed_cache_hits_total"] - m1["embed_cache_hits_total"] == 1 and m2["embed_inference_batches_total"] == m1["embed_inference_batches_total"])
check("cached vector identical", b["data"][0]["embedding"] == a["data"][0]["embedding"])
u = "用量一致性验证文本-" + str(os.getpid())
_, u1 = post({"model": "m", "input": [u]})
_, u2 = post({"model": "m", "input": [u]})
check("usage identical for miss and hit", u1["usage"] == u2["usage"] and u1["usage"]["prompt_tokens"] > 0, f"miss={u1['usage']['prompt_tokens']} hit={u2['usage']['prompt_tokens']}")
m3a = metrics()
post({"model": "m", "input": []}); post({"model": "m"})
m3b = metrics()
check("client errors counted", m3b["embed_responses_client_error_total"] - m3a["embed_responses_client_error_total"] == 2)
check("no server errors", m3b["embed_responses_server_error_total"] == 0 and m3b["embed_inference_errors_total"] == 0)
check("cache bounded", m3b["embed_cache_bytes"] <= m3b["embed_cache_max_bytes"])

with urllib.request.urlopen(BASE + "/health", timeout=10) as x:
    hb = json.load(x)
check("health endpoint ok", x.status == 200 and hb.get("status") == "ok" and hb.get("dimensions") == DIM, str(hb))
check("healthy gauge", m3b["embed_healthy"] == 1 and m3b["embed_consecutive_inference_failures"] == 0)

sys.exit(0 if ok else 1)
