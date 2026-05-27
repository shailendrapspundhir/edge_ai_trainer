"""Async Successive Halving Algorithm (ASHA) scheduler.

ASHA promotes trials through rungs.  At each rung the budget (epochs) grows
by ``reduction_factor`` and only the top ``1/reduction_factor`` trials are
promoted.  Trials that are not promoted are early-stopped.

This implementation is *async*: promotions are evaluated whenever a trial
finishes its current rung — we do not wait for the full cohort to complete.

Reference:
    Li et al., "A System for Massively Parallel Hyperparameter Tuning", 2020.
"""

from __future__ import annotations

from typing import Any

from eat.sweep.types import (
    Direction,
    MultiObjective,
    TrialRecord,
    TrialStatus,
)


class ASHAScheduler:
    """Stateless ASHA logic — all state lives in the trial records."""

    def __init__(
        self,
        max_rung: int = 3,
        reduction_factor: int = 3,
        min_epochs: float = 1.0,
        objectives: MultiObjective | None = None,
    ):
        self.max_rung = max_rung
        self.reduction_factor = reduction_factor
        self.min_epochs = min_epochs
        self.objectives = objectives

    def epochs_for_rung(self, rung: int) -> float:
        """Compute the epoch budget for a given rung."""
        return self.min_epochs * (self.reduction_factor ** rung)

    def scalarise(self, metrics: dict[str, float]) -> float:
        """Combine multi-objective metrics into a single comparable score.

        Higher is always better in the returned scalar.
        """
        if not self.objectives or not self.objectives.objectives:
            return -metrics.get("train_loss", float("inf"))

        total = 0.0
        for obj in self.objectives.objectives:
            val = metrics.get(obj.metric, 0.0)
            sign = -1.0 if obj.direction == Direction.MINIMIZE else 1.0
            total += sign * val * obj.weight
        return total

    def should_promote(self, trial: TrialRecord, all_trials: list[TrialRecord]) -> bool:
        """Return True if *trial* should advance to the next rung.

        A trial is promoted if it is in the top ``1/reduction_factor`` of all
        trials that have completed the same rung.
        """
        if trial.rung >= self.max_rung:
            return False

        same_rung = [
            t for t in all_trials
            if t.rung == trial.rung
            and t.status in (TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED)
        ]
        if not same_rung:
            return False

        # need at least reduction_factor peers before we can decide
        if len(same_rung) < self.reduction_factor:
            return True  # promote optimistically when cohort is small

        scores = sorted(
            [(self.scalarise(t.metrics), t.id) for t in same_rung],
            reverse=True,
        )
        cutoff = max(1, len(scores) // self.reduction_factor)
        promoted_ids = {sid for _, sid in scores[:cutoff]}
        return trial.id in promoted_ids

    def decide(
        self,
        trial: TrialRecord,
        all_trials: list[TrialRecord],
    ) -> str:
        """Return 'promote', 'stop', or 'wait'.

        Called after a trial finishes its current rung's budget.
        """
        if trial.status == TrialStatus.FAILED:
            return "stop"
        if trial.rung >= self.max_rung:
            return "stop"
        if self.should_promote(trial, all_trials):
            return "promote"
        return "stop"
