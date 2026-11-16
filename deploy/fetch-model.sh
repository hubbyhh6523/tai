#!/usr/bin/env bash
# Download bge-small-zh-v1.5 (ONNX, fp32) + tokenizer into ./models/bge-small-zh-v1.5 and verify SHA-256.
# Idempotent: existing files are not re-downloaded, but they are always verified.
# Behind a mirror: HF_ENDPOINT=https://hf-mirror.com ./fetch-model.sh
set -euo pipefail
DIR="${1:-$(cd "$(dirname "$0")" && pwd)/models/bge-small-zh-v1.5}"
REVISION="75c43b069aac4d136ba6bc1122f995fedcfd2781" # Xenova/bge-small-zh-v1.5, pinned
BASE="${HF_ENDPOINT:-https://huggingface.co}/Xenova/bge-small-zh-v1.5/resolve/$REVISION"
MODEL_SHA256="69a0b846f4f116b5e6aabf9546ea6754d02264f3211a13a1bd69b31b8040749a"
TOKENIZER_SHA256="48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26"
mkdir -p "$DIR"

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() { # <remote path> <local name> <expected sha256>
  if [ -s "$DIR/$2" ]; then echo "skip download $2 (exists)"
  else curl -fL --retry 3 -o "$DIR/$2.part" "$BASE/$1" && mv "$DIR/$2.part" "$DIR/$2"; fi
  local got; got="$(sha_of "$DIR/$2")"
  if [ "$got" != "$3" ]; then
    echo "SHA-256 mismatch for $2: expected $3, got $got" >&2
    echo "delete $DIR/$2 and re-run to download it again" >&2
    exit 1
  fi
  echo "ok $2 (sha256 verified)"
}

fetch tokenizer.json tokenizer.json "$TOKENIZER_SHA256"
fetch onnx/model.onnx model.onnx "$MODEL_SHA256"
