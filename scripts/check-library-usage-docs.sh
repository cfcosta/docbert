#!/usr/bin/env bash
set -euo pipefail

# library-usage.md examples `use docbert_core::...`, so we link against
# the docbert-core library artifact (the docbert crate is binary-only).
cargo build -p docbert-core --quiet

DOCBERT_CORE_RLIB="$(find target/debug/deps -maxdepth 1 -type f -name 'libdocbert_core-*.rlib' | head -n 1)"
if [[ -z "${DOCBERT_CORE_RLIB}" ]]; then
  echo "error: could not find compiled libdocbert_core rlib in target/debug/deps" >&2
  exit 1
fi

rustdoc \
  --edition=2024 \
  --test docs/library-usage.md \
  -L dependency=target/debug/deps \
  --extern "docbert_core=${DOCBERT_CORE_RLIB}"
