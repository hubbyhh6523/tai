#!/usr/bin/env bash
# Download bge-small-zh-v1.5 (ONNX, fp32) + tokenizer into ./models/bge-small-zh-v1.5 (idempotent).
# Behind a mirror: HF_ENDPOINT=https://hf-mirror.com ./fetch-model.sh
set -euo pipefail
DIR="${1:-$(cd "$(dirname "$0")" && pwd)/models/bge-small-zh-v1.5}"
BASE="${HF_ENDPOINT:-https://huggingface.co}/Xenova/bge-small-zh-v1.5/resolve/main"
mkdir -p "$DIR"
fetch() { # <remote path> <local name>
  if [ -s "$DIR/$2" ]; then echo "skip $2 (exists)"; return; fi
  curl -fL --retry 3 -o "$DIR/$2.part" "$BASE/$1" && mv "$DIR/$2.part" "$DIR/$2"
}
fetch tokenizer.json tokenizer.json
fetch onnx/model.onnx model.onnx
sha256sum "$DIR/model.onnx" "$DIR/tokenizer.json"
