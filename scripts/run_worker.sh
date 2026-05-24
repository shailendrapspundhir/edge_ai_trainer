#!/usr/bin/env bash
# Run an RQ worker for a specific queue.  Usage: ./scripts/run_worker.sh [gpu|cpu|android|cloud]
set -euo pipefail
cd "$(dirname "$0")/.."

KIND="${1:-cpu}"
case "$KIND" in
  gpu)     export EAT_WORKER_QUEUES="${EAT_QUEUE_GPU:-eat.gpu}" ;;
  cpu)     export EAT_WORKER_QUEUES="${EAT_QUEUE_CPU:-eat.cpu}" ;;
  android) export EAT_WORKER_QUEUES="${EAT_QUEUE_ANDROID:-eat.android}" ;;
  cloud)   export EAT_WORKER_QUEUES="${EAT_QUEUE_CLOUD:-eat.cloud}" ;;
  all)     export EAT_WORKER_QUEUES="${EAT_QUEUE_GPU:-eat.gpu},${EAT_QUEUE_CPU:-eat.cpu},${EAT_QUEUE_ANDROID:-eat.android},${EAT_QUEUE_CLOUD:-eat.cloud}" ;;
  *)       echo "unknown kind: $KIND  (use gpu|cpu|android|cloud|all)" >&2; exit 2 ;;
esac

exec uv run eat-worker
