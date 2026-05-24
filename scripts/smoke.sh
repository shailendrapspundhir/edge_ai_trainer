#!/usr/bin/env bash
# End-to-end smoke run that does NOT need a GPU or HF token.
# Requires: redis up + at least one CPU worker.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! curl -s "http://${EAT_ORCHESTRATOR_HOST:-127.0.0.1}:${EAT_ORCHESTRATOR_PORT:-8765}/healthz" > /dev/null; then
  echo "orchestrator not running on ${EAT_ORCHESTRATOR_HOST:-127.0.0.1}:${EAT_ORCHESTRATOR_PORT:-8765}"
  echo "  start it with:  make orchestrator"
  exit 1
fi

uv run eat run --project _smoke --recipe smoke
