#!/bin/bash
# For each cache config: start server, warm up, record RSS, run load, record RSS + metrics. Appends to bench.jsonl
cd "$(dirname "$0")/.."; : > bench.jsonl
rss() { awk '/VmRSS/{print $2}' /proc/$1/status; }
for mb in 0 1 4 64; do
  pkill -f release/embed-svc; sleep 1
  CACHE_MAX_MB=$mb MODEL_DIR=model PORT=8090 THREADS=4 setsid nohup ./target/release/embed-svc > serve.log 2>&1 </dev/null &
  sleep 3; PID=$(pgrep -f release/embed-svc | head -1)
  curl -s -X POST localhost:8090/v1/embeddings -H 'Content-Type: application/json' -d '{"input":["warmup"]}' >/dev/null
  R0=$(rss $PID); M0=$(curl -s localhost:8090/metrics)
  L=$(python3 scripts/load.py)
  R1=$(rss $PID); M1=$(curl -s localhost:8090/metrics)
  g() { echo "$2" | awk -v k="$1" '$1==k{print $2}'; }
  python3 - "$mb" "$R0" "$R1" "$L" "$(g embed_cache_hits_total "$M1")" "$(g embed_cache_misses_total "$M1")" "$(g embed_cache_bytes "$M1")" "$(g embed_cache_entries "$M1")" "$(g embed_cache_evictions_total "$M1")" "$(g embed_inference_batches_total "$M1")" "$(g embed_inference_milliseconds_total "$M1")" >> bench.jsonl <<'PY'
import sys, json
a = sys.argv; d = json.loads(a[4])
h, m = int(a[5]), int(a[6])
d.update(cache_mb=float(a[1]), rss_before_mb=round(int(a[2]) / 1024, 1), rss_after_mb=round(int(a[3]) / 1024, 1), hits=h, misses=m,
         hit_rate=round(h / max(1, h + m), 3), cache_bytes=int(a[7]), entries=int(a[8]), evictions=int(a[9]), inference_calls=int(a[10]), inference_ms=int(a[11]))
print(json.dumps(d))
PY
  pkill -f release/embed-svc
done
echo DONE >> bench.jsonl
