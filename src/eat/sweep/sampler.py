"""Random search-space sampler for the sweep engine."""

from __future__ import annotations

import math
import random
from typing import Any

from eat.sweep.types import ParamSpec, ParamType, SearchSpace


def sample_one(spec: ParamSpec, rng: random.Random | None = None) -> Any:
    """Draw one value from a parameter spec."""
    r = rng or random.Random()
    if spec.type == ParamType.UNIFORM:
        return r.uniform(spec.low, spec.high)
    if spec.type == ParamType.LOG_UNIFORM:
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        return math.exp(r.uniform(log_low, log_high))
    if spec.type == ParamType.INT:
        return r.randint(int(spec.low), int(spec.high))
    if spec.type == ParamType.CHOICE:
        return r.choice(spec.choices)
    raise ValueError(f"unknown param type: {spec.type}")


def sample(space: SearchSpace, seed: int | None = None) -> dict[str, Any]:
    """Sample a full point from the search space."""
    rng = random.Random(seed)
    return {p.name: sample_one(p, rng) for p in space.params}
