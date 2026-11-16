#!/usr/bin/env python3
"""Synthetic load: Zipf-popular texts; ~half of the events repeat the text immediately (dedup then write).
Usage: BASE=... REQS=6000 UNIQUE=2000 CONC=8 python scripts/load.py  -> prints one JSON line"""
import http.client, json, os, random, threading, time, urllib.parse
BASE = os.getenv("BASE", "http://127.0.0.1:8090"); REQS = int(os.getenv("REQS", "6000"))
UNIQUE = int(os.getenv("UNIQUE", "2000")); CONC = int(os.getenv("CONC", "8")); ZIPF = float(os.getenv("ZIPF", "1.1"))
REPEAT_P = float(os.getenv("REPEAT_P", "0.5"))
u = urllib.parse.urlparse(BASE)
W = ["Rust", "Tokio", "异步运行时", "网络服务", "P2P", "NAT穿透", "UDP打洞", "隐道协议", "超时控制", "取消安全", "CI流水线", "GitHub Actions", "缓存命中", "内存占用", "记忆检索", "向量索引", "去重写入", "用户偏好", "部署方案", "压力测试", "日志排查", "端到端测试", "跨平台", "发布流程", "Docker镜像", "依赖管理", "安全扫描", "WebAuthn", "SOCKS5", "代码审查"]
rng0 = random.Random(7)
TEXTS = ["，".join(rng0.sample(W, rng0.randint(3, 6))) + f"。编号{i}" for i in range(UNIQUE)]
WEIGHTS = [1.0 / (i + 1) ** ZIPF for i in range(UNIQUE)]
lock = threading.Lock(); lat = []; errs = [0]; sent = [0]

def worker(seed):
    rng = random.Random(seed); c = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
    def post(text):
        t0 = time.perf_counter()
        try:
            c.request("POST", "/v1/embeddings", json.dumps({"model": "m", "input": [text], "dimensions": 512}), {"Content-Type": "application/json"})
            r = c.getresponse(); r.read(); ok = r.status == 200
        except Exception:
            ok = False; c.close(); c.connect()
        dt = (time.perf_counter() - t0) * 1000
        with lock:
            lat.append(dt); sent[0] += 1
            if not ok: errs[0] += 1
    while True:
        with lock:
            if sent[0] >= REQS: return
        text = TEXTS[rng.choices(range(UNIQUE), WEIGHTS)[0]]
        post(text)
        if rng.random() < REPEAT_P: post(text)

t0 = time.perf_counter()
th = [threading.Thread(target=worker, args=(i,)) for i in range(CONC)]
[t.start() for t in th]; [t.join() for t in th]
wall = time.perf_counter() - t0
lat.sort(); p = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 2)
print(json.dumps({"requests": len(lat), "errors": errs[0], "wall_s": round(wall, 1), "rps": round(len(lat) / wall, 1), "p50_ms": p(0.5), "p95_ms": p(0.95), "p99_ms": p(0.99)}))
