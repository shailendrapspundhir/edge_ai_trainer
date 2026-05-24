"""Dataset loaders + the `prepare()` entry point invoked by the prepare_data job.

A dataset's `kind` + `source` decide which loader is used:

    source=local         -> read JSONL from the path under projects/<name>/data/
    source=hf            -> load a HF datasets repo, optionally sliced to splits
    source=synthetic     -> run the synthetic generator
    source=url           -> download once, cache under .cache/eat/datasets/<name>/

The default fallback writes an empty file so downstream jobs do not error out
when the project author hasn't built the dataset yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from eat.logging_setup import get_logger
from eat.paths import cache_dir, repo_root
from eat.registry import datasets as datasets_reg

log = get_logger("data.loaders")


def _resolve_local(location: str) -> Path:
    p = Path(location)
    if not p.is_absolute():
        p = repo_root() / p
    return p


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def prepare(project: str, dataset: str) -> dict[str, Any]:
    """Materialise a dataset into projects/<project>/data/processed/<dataset>.jsonl
    (or wherever the dataset config points). Returns simple stats.
    """
    entry = datasets_reg.get(dataset)
    out_path = _resolve_local(entry.location)
    stats: dict[str, Any] = {"name": dataset, "kind": entry.kind, "source": entry.source}

    if entry.source == "local":
        # Just verify + count.
        n = sum(1 for _ in iter_jsonl(out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            out_path.touch()
            log.warning("local_dataset_empty", dataset=dataset, path=str(out_path),
                        hint="run `eat data build` or hand-populate")
        stats["count"] = n
        stats["path"] = str(out_path)
        return stats

    if entry.source == "hf":
        from datasets import load_dataset  # type: ignore

        train_n = entry.splits.get("train") or 0
        eval_n = entry.splits.get("eval") or 0
        cache = cache_dir() / "datasets" / dataset
        cache.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(entry.location, cache_dir=str(cache))
        # Pick a sensible default split if "train" is missing.
        train_split = ds.get("train") or next(iter(ds.values()))
        if train_n:
            train_split = train_split.select(range(min(train_n, len(train_split))))
        items = []
        for ex in train_split:
            # Heuristic conversion to {messages} for instruction datasets like Alpaca.
            user = ex.get("instruction") or ex.get("prompt") or ex.get("input") or ""
            extra = ex.get("input") if ex.get("instruction") and ex.get("input") else ""
            if extra:
                user = f"{user}\n\n{extra}"
            assistant = ex.get("output") or ex.get("response") or ex.get("completion") or ""
            items.append({
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            })
        n = write_jsonl(out_path, items)
        stats["count"] = n
        stats["path"] = str(out_path)
        return stats

    if entry.source == "synthetic":
        from eat.data import synthetic
        n = synthetic.generate(project=project, dataset=dataset, out_path=out_path)
        stats["count"] = n
        stats["path"] = str(out_path)
        return stats

    raise NotImplementedError(f"unsupported dataset source: {entry.source}")
