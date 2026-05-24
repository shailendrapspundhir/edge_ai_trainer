"""Pick a training backend based on the recipe."""

from __future__ import annotations

from typing import IO, Any

from eat.logging_setup import get_logger
from eat.registry import recipes

log = get_logger("training.dispatch")


def train(run_id: str, project: str, recipe: str, log_file: IO[str]) -> dict[str, Any]:
    spec = recipes.get(recipe)
    if spec.backend == "hf_trl":
        from eat.training import hf_trl
        return hf_trl.train(run_id=run_id, project=project, recipe=spec, log_file=log_file)
    if spec.backend == "unsloth":
        from eat.training import unsloth_backend
        return unsloth_backend.train(run_id=run_id, project=project, recipe=spec, log_file=log_file)
    raise ValueError(f"unknown training backend: {spec.backend}")
