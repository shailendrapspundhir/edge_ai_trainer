#!/usr/bin/env bash
# Run the FastAPI orchestrator (foreground).
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run eat-orchestrator
