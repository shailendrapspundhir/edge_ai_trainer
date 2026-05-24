"""Synthetic data generator — uses a strong teacher API model to produce gold
outputs for (user_profile, food_facts, context) triples.

If no API key is present, falls back to a deterministic template generator so
the pipeline still produces parseable JSONL (useful for smoke runs / CI).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from eat.config import get_settings
from eat.data import knowledge
from eat.data.schemas import FoodCoachExample, FoodFacts, Message, UserProfile
from eat.logging_setup import get_logger
from eat.paths import projects_dir

log = get_logger("data.synthetic")


# ---------------------------------------------------------------------------
# profile + food combinatorics
# ---------------------------------------------------------------------------


_PROFILE_POOL: list[UserProfile] = [
    UserProfile(age=42, sex="m", weight_kg=86, height_cm=174,
                activity="sedentary", goals=["weight loss"],
                conditions=["type 2 diabetes"], medications=["metformin"]),
    UserProfile(age=29, sex="f", weight_kg=58, height_cm=162,
                activity="moderate", goals=["maintain"],
                conditions=["lactose intolerance"], allergies=["peanut"]),
    UserProfile(age=65, sex="m", weight_kg=78, height_cm=170,
                activity="light", goals=["lower bp"],
                conditions=["hypertension"], medications=["amlodipine"]),
    UserProfile(age=32, sex="f", weight_kg=72, height_cm=168,
                activity="light", goals=["pcos management"],
                conditions=["pcos", "insulin resistance"], diet=["vegetarian"]),
    UserProfile(age=8,  sex="f", weight_kg=26, height_cm=128,
                activity="active", goals=["healthy growth"], allergies=["egg"]),
    UserProfile(age=27, sex="f", weight_kg=64, height_cm=165,
                activity="moderate", goals=["pregnancy nutrition"],
                conditions=["pregnancy 2nd trimester"]),
    UserProfile(age=22, sex="m", weight_kg=70, height_cm=178,
                activity="very_active", goals=["muscle gain"]),
]


_CONTEXTS = [
    "after a long workday",
    "as a 4 pm snack",
    "before a 5 km run",
    "at a birthday party",
    "post-workout",
    "on a fasting day",
    "during pregnancy second trimester",
    "as iftar after a long fast",
    "as a midnight craving",
]


# ---------------------------------------------------------------------------
# templates (used when no teacher API is available)
# ---------------------------------------------------------------------------


def _heuristic_answer(profile: UserProfile, food: FoodFacts, context: str) -> tuple[str, str]:
    """Deterministic gold-ish answer when we have no LLM teacher."""
    cond = ",".join(profile.conditions).lower()
    sugar = food.sugar_g or 0
    sodium = food.sodium_mg or 0
    sat_fat = food.sat_fat_g or 0

    high_sugar = sugar >= 10
    high_sodium = sodium >= 500
    high_sat_fat = sat_fat >= 5

    verdict = "eat"
    reasons: list[str] = []
    portion_g = 100

    if "diabetes" in cond and high_sugar:
        verdict = "avoid"
        reasons.append(f"{food.name} has {sugar} g sugar per 100 g which spikes blood glucose")
        portion_g = 30
    elif "hypertension" in cond and high_sodium:
        verdict = "portion-limit"
        reasons.append(f"{sodium} mg sodium per 100 g is high for your BP goals")
        portion_g = 50
    elif "pcos" in cond and high_sugar:
        verdict = "portion-limit"
        reasons.append("added sugar worsens insulin resistance")
        portion_g = 40
    elif high_sat_fat and "weight loss" in profile.goals:
        verdict = "portion-limit"
        reasons.append(f"{sat_fat} g sat-fat per 100 g works against your weight-loss goal")
        portion_g = 60

    if profile.allergies and any(a.lower() in (food.name + " " + " ".join(food.ingredients)).lower()
                                 for a in profile.allergies):
        verdict = "avoid"
        reasons.append("contains a declared allergen")
        portion_g = 0

    if not reasons:
        reasons.append("fits within your daily targets when eaten in normal portions")

    answer = (
        f"Verdict: {verdict}.\n"
        f"Why: " + "; ".join(reasons) + ".\n"
        f"Portion: about {portion_g} g{' (avoid completely)' if portion_g == 0 else ''}.\n"
        f"Context: {context}.\n"
        "If you're unsure or symptoms change, please check with your clinician."
    )
    user = (
        f"User profile:\n{profile.model_dump_json(indent=2)}\n\n"
        f"Food facts (per 100 g):\n{food.model_dump_json(indent=2)}\n\n"
        f"Context: {context}\n\n"
        "Can I eat this? How much?"
    )
    return user, answer


# ---------------------------------------------------------------------------
# teacher API (Anthropic-only for v0; add OpenAI later)
# ---------------------------------------------------------------------------


def _teacher_answer(profile: UserProfile, food: FoodFacts, context: str,
                    system_prompt: str) -> str | None:
    s = get_settings()
    if s.teacher_provider != "anthropic" or not s.anthropic_api_key:
        return None
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError:
        log.warning("anthropic_sdk_missing")
        return None
    client = Anthropic(api_key=s.anthropic_api_key)
    user = (
        f"User profile (JSON):\n{profile.model_dump_json(indent=2)}\n\n"
        f"Food facts (JSON):\n{food.model_dump_json(indent=2)}\n\n"
        f"Context: {context}\n\n"
        "Produce a JSON response with fields: verdict, reasons (1-3), "
        "portion_g, follow_up (optional), and a short user-facing answer."
    )
    resp = client.messages.create(
        model=s.teacher_model,
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def _system_prompt_for(project: str) -> str:
    path = projects_dir() / project / "prompts" / "system.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "You are a careful, casual food / health coach. Always defer to clinicians."


def generate(project: str, dataset: str, out_path: Path, n: int = 200,
             seed: int = 7) -> int:
    """Generate `n` synthetic examples, write JSONL, return count.

    Each line conforms to the InstructionExample shape (`messages`) so the
    training loader can consume it without special-casing.
    """
    rng = random.Random(seed)
    sysp = _system_prompt_for(project)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    foods = list(knowledge.store())
    for i in range(n):
        profile = rng.choice(_PROFILE_POOL)
        rec = rng.choice(foods)
        food = knowledge.to_food_facts(rec, form="text_query")
        ctx = rng.choice(_CONTEXTS)

        teacher = _teacher_answer(profile, food, ctx, sysp)
        if teacher:
            user, assistant = (
                f"User profile (JSON):\n{profile.model_dump_json(indent=2)}\n\n"
                f"Food facts (JSON):\n{food.model_dump_json(indent=2)}\n\n"
                f"Context: {ctx}\nCan I eat this? How much?",
                teacher,
            )
        else:
            user, assistant = _heuristic_answer(profile, food, ctx)

        ex = {
            "id": f"syn_{i:05d}",
            "messages": [
                {"role": "system", "content": sysp},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"profile": profile.model_dump(), "food": food.model_dump(), "context": ctx},
        }
        items.append(ex)

    with out_path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    log.info("synthetic_generated", project=project, dataset=dataset, n=len(items), path=str(out_path))
    return len(items)
