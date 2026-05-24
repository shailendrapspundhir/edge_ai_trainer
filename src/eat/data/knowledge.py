"""Knowledge-base layer for food data.

A thin façade over the three sources we expect to ingest later:

    - USDA FoodData Central  (public domain)
    - Open Food Facts        (CC-BY-SA; packaged-product DB with label images)
    - IFCT 2017              (Indian Food Composition Tables; research-permissive)

For v0 the loader returns an empty in-memory store and a `lookup()` that does
a best-effort substring match on a tiny seed list — enough for tests and the
synthetic generator to wire end-to-end. Replace `_SEED` with real ingestion
once we have local copies of the source dumps under `projects/.../data/raw/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from eat.data.schemas import FoodFacts


@dataclass(frozen=True)
class FoodRecord:
    name: str
    kcal_per_100g: float | None = None
    carbs_g: float | None = None
    sugar_g: float | None = None
    sodium_mg: float | None = None
    sat_fat_g: float | None = None
    protein_g: float | None = None
    fiber_g: float | None = None
    source: str = "seed"


_SEED: list[FoodRecord] = [
    FoodRecord("white rice (cooked)", kcal_per_100g=130, carbs_g=28, sugar_g=0.1, sodium_mg=1, sat_fat_g=0.1, protein_g=2.7, fiber_g=0.4),
    FoodRecord("brown rice (cooked)", kcal_per_100g=112, carbs_g=24, sugar_g=0.2, sodium_mg=2, sat_fat_g=0.2, protein_g=2.6, fiber_g=1.8),
    FoodRecord("roti / chapati", kcal_per_100g=297, carbs_g=46, sugar_g=2, sodium_mg=300, sat_fat_g=1.4, protein_g=11, fiber_g=4.5),
    FoodRecord("dal (cooked, mixed)", kcal_per_100g=116, carbs_g=20, sugar_g=1.5, sodium_mg=320, sat_fat_g=0.4, protein_g=9, fiber_g=8),
    FoodRecord("paneer", kcal_per_100g=265, carbs_g=1.2, sugar_g=1.2, sodium_mg=18, sat_fat_g=12, protein_g=18, fiber_g=0),
    FoodRecord("banana", kcal_per_100g=89, carbs_g=23, sugar_g=12, sodium_mg=1, sat_fat_g=0.1, protein_g=1.1, fiber_g=2.6),
    FoodRecord("apple", kcal_per_100g=52, carbs_g=14, sugar_g=10, sodium_mg=1, sat_fat_g=0.0, protein_g=0.3, fiber_g=2.4),
    FoodRecord("mango", kcal_per_100g=60, carbs_g=15, sugar_g=14, sodium_mg=1, sat_fat_g=0.1, protein_g=0.8, fiber_g=1.6),
    FoodRecord("oats (dry)", kcal_per_100g=389, carbs_g=66, sugar_g=0.99, sodium_mg=2, sat_fat_g=1.2, protein_g=17, fiber_g=10.6),
    FoodRecord("jaggery", kcal_per_100g=383, carbs_g=98, sugar_g=97, sodium_mg=30, sat_fat_g=0.1, protein_g=0.4, fiber_g=0),
    FoodRecord("samosa (fried)", kcal_per_100g=308, carbs_g=32, sugar_g=2.4, sodium_mg=423, sat_fat_g=6.7, protein_g=6.5, fiber_g=3.1),
    FoodRecord("french fries", kcal_per_100g=312, carbs_g=41, sugar_g=0.3, sodium_mg=210, sat_fat_g=2.0, protein_g=3.4, fiber_g=3.8),
    FoodRecord("pizza margherita", kcal_per_100g=266, carbs_g=33, sugar_g=3.6, sodium_mg=598, sat_fat_g=4.5, protein_g=11, fiber_g=2.3),
    FoodRecord("dark chocolate 70%", kcal_per_100g=598, carbs_g=46, sugar_g=24, sodium_mg=20, sat_fat_g=24, protein_g=7.8, fiber_g=10.9),
    FoodRecord("sweetened yogurt", kcal_per_100g=99, carbs_g=14, sugar_g=14, sodium_mg=46, sat_fat_g=1.6, protein_g=4, fiber_g=0),
]


@lru_cache(maxsize=1)
def store() -> list[FoodRecord]:
    return list(_SEED)


def lookup(query: str) -> FoodRecord | None:
    q = query.lower().strip()
    items = store()
    for it in items:
        if q == it.name.lower():
            return it
    for it in items:
        if q in it.name.lower() or it.name.lower() in q:
            return it
    return None


def to_food_facts(record: FoodRecord, form: str = "text_query") -> FoodFacts:
    return FoodFacts(
        name=record.name,
        form=form,  # type: ignore[arg-type]
        kcal_per_100g=record.kcal_per_100g,
        carbs_g=record.carbs_g,
        sugar_g=record.sugar_g,
        sodium_mg=record.sodium_mg,
        sat_fat_g=record.sat_fat_g,
        protein_g=record.protein_g,
        fiber_g=record.fiber_g,
    )


def iter_records() -> Iterable[FoodRecord]:
    yield from store()


# ---------------------------------------------------------------------------
# Hooks for the real ingestion (no-ops for now)
# ---------------------------------------------------------------------------


def ingest_usda(json_dump_dir: Path) -> int:
    """TODO: parse FoodData Central JSON dump into the local store."""
    raise NotImplementedError(
        "USDA ingestion not implemented yet. Download FoodData Central JSON to "
        f"{json_dump_dir} and wire this function."
    )


def ingest_open_food_facts(jsonl_path: Path) -> int:
    """TODO: stream-parse OFF JSONL dump."""
    raise NotImplementedError("Open Food Facts ingestion not implemented yet.")


def ingest_ifct(csv_path: Path) -> int:
    raise NotImplementedError("IFCT 2017 ingestion not implemented yet.")
