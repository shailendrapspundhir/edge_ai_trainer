"""Pydantic schemas for training/eval records.

Every JSONL we produce/consume validates against one of these. Keep them
permissive on `extra` so projects can attach project-specific fields without
forking the schema.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    image_paths: list[str] = Field(default_factory=list)
    audio_paths: list[str] = Field(default_factory=list)


class InstructionExample(BaseModel):
    """Generic chat-style instruction example."""
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    messages: list[Message]
    meta: dict[str, Any] = Field(default_factory=dict)


class VisionPair(BaseModel):
    """(image, structured description) for image-grounded training."""
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    image_path: str
    caption: str
    structured: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None


class AudioPair(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    audio_path: str
    transcript: str
    source: str | None = None


# ---------------------------------------------------------------------------
# Food-coach specific (used in projects/food_health_coach but lives here so
# the synthetic generator and validator can both import it).
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int | None = None
    sex: Literal["m", "f", "other"] | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    activity: Literal["sedentary", "light", "moderate", "active", "very_active"] | None = None
    goals: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    diet: list[str] = Field(default_factory=list)


class FoodFacts(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    form: Literal["plate", "packaged_label", "ingredient_list", "text_query"] = "text_query"
    image_ref: str | None = None
    kcal_per_100g: float | None = None
    carbs_g: float | None = None
    sugar_g: float | None = None
    sodium_mg: float | None = None
    sat_fat_g: float | None = None
    protein_g: float | None = None
    fiber_g: float | None = None
    ingredients: list[str] = Field(default_factory=list)


class FoodCoachExample(BaseModel):
    """One end-to-end (input, gold output) record for food-coach SFT."""
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    user_profile: UserProfile
    food: FoodFacts
    context: str | None = None
    expected_verdict: Literal["eat", "portion-limit", "avoid"] | None = None
    expected_reasons_keywords: list[str] = Field(default_factory=list)
    expected_max_portion_g: float | None = None
    output: str | None = None  # gold answer; filled by teacher generator or human
    meta: dict[str, Any] = Field(default_factory=dict)
