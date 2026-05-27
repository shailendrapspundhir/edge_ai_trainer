"""Load sweep configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eat.sweep.types import (
    Direction,
    MultiObjective,
    ObjectiveSpec,
    ParamSpec,
    ParamType,
    SearchSpace,
)


def load_search_space(path: str | Path) -> SearchSpace:
    """Parse a search_space.yaml into a SearchSpace."""
    raw = yaml.safe_load(Path(path).read_text())
    params: list[ParamSpec] = []
    for p in raw["params"]:
        params.append(ParamSpec(
            name=p["name"],
            type=ParamType(p["type"]),
            low=p.get("low"),
            high=p.get("high"),
            choices=p.get("choices"),
        ))
    return SearchSpace(params=params)


def load_objectives(path: str | Path) -> MultiObjective:
    """Parse an objectives.yaml into a MultiObjective."""
    raw = yaml.safe_load(Path(path).read_text())
    objectives: list[ObjectiveSpec] = []
    for o in raw["objectives"]:
        objectives.append(ObjectiveSpec(
            metric=o["metric"],
            direction=Direction(o["direction"]),
            weight=o.get("weight", 1.0),
        ))
    return MultiObjective(objectives=objectives)


def load_search_space_from_dict(raw: dict[str, Any]) -> SearchSpace:
    """Build SearchSpace from an already-parsed dict."""
    params: list[ParamSpec] = []
    for p in raw["params"]:
        params.append(ParamSpec(
            name=p["name"],
            type=ParamType(p["type"]),
            low=p.get("low"),
            high=p.get("high"),
            choices=p.get("choices"),
        ))
    return SearchSpace(params=params)


def load_objectives_from_dict(raw: dict[str, Any]) -> MultiObjective:
    """Build MultiObjective from an already-parsed dict."""
    objectives: list[ObjectiveSpec] = []
    for o in raw["objectives"]:
        objectives.append(ObjectiveSpec(
            metric=o["metric"],
            direction=Direction(o["direction"]),
            weight=o.get("weight", 1.0),
        ))
    return MultiObjective(objectives=objectives)
