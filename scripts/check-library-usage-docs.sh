#!/usr/bin/env bash
set -euo pipefail

# Build the crate so rustdoc can link against the compiled library artifact.
cargo build --quiet

DOCBERT_RLIB="$(find target/debug/deps -maxdepth 1 -type f -name 'libdocbert-*.rlib' | head -n 1)"
if [[ -z "${DOCBERT_RLIB}" ]]; then
  echo "error: could not find compiled libdocbert rlib in target/debug/deps" >&2
  exit 1
fi

rustdoc \
  --edition=2024 \
  --test docs/library-usage.md \
  -L dependency=target/debug/deps \
  --extern "docbert=${DOCBERT_RLIB}"
