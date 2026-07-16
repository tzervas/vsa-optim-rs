#!/usr/bin/env bash
# Local-parity gate for vsa-optim-rs (standalone crate).
# Usage: ./scripts/check.sh [--fix]
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

if [[ "$MODE" == "--fix" ]]; then
  cargo fmt
else
  cargo fmt --check
fi

cargo check --all-targets
cargo test --quiet

echo "OK: vsa-optim-rs checks passed"
