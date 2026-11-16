#!/usr/bin/env bash
# One-time admin bootstrap for the Core started by docker-compose.yml.
# Mirrors what the official start-memory-core.sh does; the key is written to ./.admin-key (mode 600).
# Safe to re-run: an existing admin (HTTP 409) is accepted when the key file is present.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
BASE="http://127.0.0.1:${MEMORY_CORE_PORT:-8420}"
KEY_FILE="${ADMIN_KEY_FILE:-./.admin-key}"
USERNAME="${MEMORY_CORE_ADMIN_USERNAME:-admin}"
AUTH=()
if [ -n "${MEMORY_CORE_GATEWAY_API_KEY:-}" ]; then AUTH=(-H "Authorization: Bearer ${MEMORY_CORE_GATEWAY_API_KEY}"); fi

for i in $(seq 1 30); do
  if curl -fsS -m 3 "$BASE/health" >/dev/null 2>&1; then break; fi
  if [ "$i" = 30 ]; then echo "memory-core not healthy at $BASE after 60s" >&2; exit 1; fi
  sleep 2
done

if [ -s "$KEY_FILE" ]; then KEY="$(cat "$KEY_FILE")"
else KEY="sk-mem-$(openssl rand -base64 48 | LC_ALL=C tr -dc 'A-Za-z0-9' | head -c 32)"; fi

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
code="$(curl -sS -o "$TMP" -w '%{http_code}' -X POST "$BASE/v3/internal/meta/user/init-admin" \
  -H 'Content-Type: application/json' -H 'x-tdai-service-id: default' "${AUTH[@]+"${AUTH[@]}"}" \
  -d "{\"username\":\"$USERNAME\",\"user_key\":\"$KEY\"}")"
case "$code" in
  200) ( umask 077; printf '%s' "$KEY" > "$KEY_FILE" ); echo "admin created; key saved to $KEY_FILE" ;;
  409) if [ -s "$KEY_FILE" ]; then echo "admin already exists; using $KEY_FILE"
       else echo "admin exists but $KEY_FILE is missing; recover the key or recreate the volume" >&2; exit 2; fi ;;
  *)   echo "init-admin failed: HTTP $code" >&2; cat "$TMP" >&2; exit 1 ;;
esac

vcode="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$BASE/v3/meta/auth/verify" \
  -H 'Content-Type: application/json' -H 'x-tdai-service-id: default' "${AUTH[@]+"${AUTH[@]}"}" \
  -d "{\"user_key\":\"$(cat "$KEY_FILE")\"}")"
[ "$vcode" = 200 ] && echo "auth/verify OK (key ${KEY:0:11}****)" || { echo "auth/verify returned HTTP $vcode" >&2; exit 1; }
