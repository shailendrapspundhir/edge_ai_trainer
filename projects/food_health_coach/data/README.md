# food_health_coach / data

Storage layout for source data, processed training sets, and scratch.

## Layout

| Path | Contents | Tracked? |
|---|---|---|
| `raw/` | Source dumps from USDA / Open Food Facts / IFCT, and any teacher-model raw responses. Often 5 GB or more. | gitignored |
| `processed/` | JSONL files matching the dataset configs in `configs/datasets/`: `food_v1.jsonl`, `food_vision_v1.jsonl`. One valid JSON object per line; validated against the pydantic schemas in `src/eat/data/schemas.py`. | DVC |
| `cache/` | Scratch space for the data-build pipeline (tokeniser caches, intermediate parquet, image thumbnails). Safe to wipe. | gitignored |

## How to populate

```
eat data build --project food_health_coach --dataset food_v1
eat data build --project food_health_coach --dataset food_vision_v1
```

The build command:

1. Pulls / refreshes the source dumps into `raw/` (USDA bulk JSON, Open Food Facts CSV + label images, IFCT XLSX).
2. Builds the local food KB (SQLite + FTS5) from `raw/`.
3. Samples `(user_profile × food × context)` tuples and looks up the canonical macros.
4. Calls the configured teacher model with the system prompt + rubric to generate gold outputs.
5. Validates each record against the pydantic schema (rejects malformed entries with a report).
6. Writes JSONL into `processed/` and snapshots a DVC version.

## Licensing

| Source | License | Usage |
|---|---|---|
| USDA FoodData Central | Public domain | Free for training, redistribution, derived KB. |
| Open Food Facts | CC-BY-SA 3.0 | Attribute "Open Food Facts contributors"; derived KB must remain share-alike. |
| IFCT 2017 (NIN-ICMR) | Research-permissive (academic / non-commercial) | Fine for model training and internal eval; cite NIN-ICMR; clear use before any commercial redistribution. |

A detailed source manifest with retrieval dates, URLs, and per-field provenance lives in `SOURCES.md` (alongside this README — created during the first `eat data build` run).
