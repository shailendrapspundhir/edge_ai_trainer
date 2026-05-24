from __future__ import annotations

import json
from pathlib import Path

from eat.data.schemas import FoodCoachExample, FoodFacts, UserProfile


def test_user_profile_roundtrip():
    p = UserProfile(age=42, sex="m", conditions=["type 2 diabetes"], allergies=["peanut"])
    s = p.model_dump_json()
    again = UserProfile.model_validate_json(s)
    assert again.age == 42
    assert "peanut" in again.allergies


def test_food_facts_minimal():
    f = FoodFacts(name="banana")
    assert f.form == "text_query"


def test_food_coach_example_roundtrip():
    ex = FoodCoachExample(
        id="ev_001",
        user_profile=UserProfile(age=30, conditions=["type 2 diabetes"]),
        food=FoodFacts(name="mango", sugar_g=14.0),
        context="afternoon snack",
        expected_verdict="portion-limit",
    )
    s = ex.model_dump_json()
    again = FoodCoachExample.model_validate_json(s)
    assert again.expected_verdict == "portion-limit"


def test_eval_seed_file_parseable():
    p = Path("projects/food_health_coach/eval/seed_eval.jsonl")
    if not p.exists():
        return
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        n += 1
        json.loads(line)
    assert n > 0
