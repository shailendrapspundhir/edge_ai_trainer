"""RQ worker entry point — `eat-worker`.

Honours EAT_WORKER_QUEUES (comma-separated). If unset, listens on all four
default queues (gpu, cpu, android, cloud).
"""

from __future__ import annotations

import os
import signal
import sys

from rq import Worker

from eat.config import get_settings
from eat.logging_setup import get_logger, setup_logging
from eat.orchestrator import db, queue

log = get_logger("worker")


def main() -> int:
    setup_logging()
    db.init_db()
    s = get_settings()
    queue_names = queue.resolve_queues(os.environ.get("EAT_WORKER_QUEUES") or s.worker_queues)
    queues = [queue.get_queue(q) for q in queue_names]
    log.info("worker_start", queues=queue_names, redis=s.redis_url)
    worker = Worker(
        queues=queues,
        connection=queue.get_redis(),
        name=f"eat-{os.getpid()}",
    )

    def _term(*_args):
        log.info("worker_signal_term")
        worker.request_stop(*_args) if hasattr(worker, "request_stop") else None
        sys.exit(0)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    worker.work(with_scheduler=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
