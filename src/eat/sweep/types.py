"""Types for the hyperparameter sweep engine."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Search space primitives
# ---------------------------------------------------------------------------


class ParamType(str, Enum):
    UNIFORM = "uniform"         # continuous float [low, high]
    LOG_UNIFORM = "log_uniform" # log-scale float [low, high]
    INT = "int"                 # integer [low, high]
    CHOICE = "choice"           # categorical list


class ParamSpec(BaseModel):
    """One axis of the search space."""
    model_config = ConfigDict(extra="forbid")
    name: str
    type: ParamType
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None


class SearchSpace(BaseModel):
    """Parsed search_space.yaml: a list of tuneable parameters."""
    model_config = ConfigDict(extra="forbid")
    params: list[ParamSpec]


# ---------------------------------------------------------------------------
# Multi-objective spec
# ---------------------------------------------------------------------------


class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveSpec(BaseModel):
    """One optimisation objective."""
    metric: str            # e.g. "train_loss", "tokens_per_sec", "peak_mem_mb"
    direction: Direction
    weight: float = 1.0    # for scalarised comparison


class MultiObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objectives: list[ObjectiveSpec]


# ---------------------------------------------------------------------------
# Trial / Sweep records
# ---------------------------------------------------------------------------


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EARLY_STOPPED = "early_stopped"


class TrialRecord(BaseModel):
    """One HP trial — maps 1-to-1 to a planner run."""
    id: str
    sweep_id: str
    run_id: str | None = None          # back-ref to the planner's run_id
    rung: int = 0                       # ASHA rung (0 = initial)
    params: dict[str, Any]              # sampled HP values
    metrics: dict[str, float] = Field(default_factory=dict)
    status: TrialStatus = TrialStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    epochs_completed: float = 0.0


class SweepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SweepRecord(BaseModel):
    """Top-level sweep metadata."""
    id: str
    project: str
    recipe: str                          # base recipe to overlay HP samples on
    max_trials: int
    search_space: SearchSpace
    objectives: MultiObjective
    status: SweepStatus = SweepStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    trial_ids: list[str] = Field(default_factory=list)
    best_trial_id: str | None = None

    # ASHA config
    asha_max_rung: int = 3               # number of successive halving rungs
    asha_reduction_factor: int = 3        # keep 1/factor at each rung
    asha_min_epochs: float = 1.0          # epochs at rung 0
