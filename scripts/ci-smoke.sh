#!/usr/bin/env bash
# Start the release binary against the real model, run the smoke checks, stop it.
# Run from the repo root after `cargo build --release` and `deploy/fetch-model.sh model`:
#   bash scripts/ci-smoke.sh
set -euo pipefail
BIN=${BIN:-./target/release/embed-svc}
PORT=${PORT:-8090}
LOG=${LOG:-server.log}
export BASE="http://127.0.0.1:$PORT"
pid=""

stop() {
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    pid=""
  fi
}
trap stop EXIT

start() { # extra VAR=value settings for the server
  : > "$LOG"
  env MODEL_DIR=model PORT="$PORT" API_KEY=ci-smoke "$@" "$BIN" >> "$LOG" 2>&1 &
  pid=$!
  for _ in $(seq 1 60); do
    if curl -fs "$BASE/health" > /dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "service exited during startup" >&2
      cat "$LOG" >&2
      exit 1
    fi
    sleep 1
  done
  echo "service did not become healthy within 60s" >&2
  cat "$LOG" >&2
  exit 1
}

check() { # <name> <command...>
  local name=$1
  shift
  echo "::group::$name"
  if "$@"; then
    echo "::endgroup::"
  else
    echo "::endgroup::"
    echo "FAILED: $name" >&2
    echo "--- server log ---" >&2
    cat "$LOG" >&2
    exit 1
  fi
}

start
check "smoke (auth on, default settings)" env API_KEY=ci-smoke python3 scripts/smoke.py
stop

start CACHE_MAX_MB=0.01
check "cache eviction" env MODE=evict API_KEY=ci-smoke python3 scripts/smoke_cache.py
stop

start CACHE_TTL_SECS=1
check "cache ttl" env MODE=ttl API_KEY=ci-smoke python3 scripts/smoke_cache.py
stop

echo "all smoke checks passed"
