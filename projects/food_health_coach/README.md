# food_health_coach

First project of the `edge_ai_trainer` workbench. A multimodal Gemma 3n
model that gives personalised, casual dietary guidance grounded in a
user profile plus food facts retrieved at inference time.

## Overview

- **Inputs**: user profile JSON (age, sex, conditions, allergies, meds,
  goals, dietary preferences), a food signal (plate photo, label photo,
  ingredient list, or text query), and optional conversation history.
- **Outputs**: short verdict (`eat` / `portion-limit` / `avoid`), 1-3
  grounded reasons, quantified portion guidance (grams + household
  units), and an optional follow-up question if the input is ambiguous.
- **Base model**: `google/gemma-3n-E2B-it` (≈2B effective).
- **Targets**: GGUF Q4_K_M / Q5_K_M for desktop & llama.cpp Android,
  MediaPipe LiteRT `.task` (int4) for the primary Android runtime.
- **Hard rule**: the model never diagnoses, never adjusts medication,
  and always defers to a clinician for medical decisions.

See `PLAN.md` §6 for the long-form rationale.

## Data plan

Three layers:

1. **Knowledge layer** (facts, not training labels):
   - USDA FoodData Central (public domain)
   - Open Food Facts (CC-BY-SA)
   - IFCT 2017 — Indian Food Composition Tables (research-permissive)
   - A small curated dietetic-rules table (~200-500 rules) authored
     from ADA / ICMR-NIN public guidelines.
2. **Supervised fine-tune set** (`food_v1`):
   - Synthetic, grounded (~20-50k): sample (profile × food × context),
     look up facts, run a teacher model (Claude / GPT-4-class) with the
     rubric to produce gold outputs.
   - Hand-curated seed (~500-2000): hard medical edges, allergens,
     mixed conditions. Doubles as the eval hard slice.
3. **Vision pairs** (`food_vision_v1`, ~5-20k): (food image,
   structured description: dish, ingredients, estimated portion + macros).
   Sources: Open Food Facts label images plus re-annotated Food-101 /
   Nutrition5k.
4. **Eval set** (`food_eval_v1`, held-out): 500-1000 cases stratified
   across {condition, food type, modality, edge-case category}, each
   with a rubric. Seeded by `eval/seed_eval.jsonl` (25 cases).

Every record is validated against a pydantic schema before training.

## How to build the dataset

```
eat data build --project food_health_coach --dataset food_v1
eat data build --project food_health_coach --dataset food_vision_v1
```

The `eat data build` command:

1. Loads + indexes the knowledge layer (FTS5 SQLite).
2. Samples (profile, food, context) tuples per the configured strata.
3. Calls the configured teacher model with the system prompt + rubric.
4. Validates each output against the pydantic schema.
5. Writes JSONL to `data/processed/<dataset>.jsonl`.
6. Snapshots a DVC version.

## How to train

Text-first to de-risk the recipe and surface bugs cheaply:

```
eat run --project food_health_coach --recipe qlora_text
```

Then flip on the vision branch:

```
eat run --project food_health_coach --recipe qlora_vlm
```

Defaults from `configs/recipes/qlora_text.yaml`: QLoRA NF4, LoRA r=16,
α=32, dropout 0.05 on attention + MLP projections; vision and audio
towers frozen in v1; per-device batch 1, grad-accum 16, max seq 4096;
2 epochs, cosine LR, paged AdamW 8-bit; eval / save every 200 steps.

## How to evaluate

Three independent scores per checkpoint:

```
eat eval quality <run_id>   # LLM-as-judge on food_eval_v1
eat eval safety  <run_id>   # 30-prompt red-team in prompts/red_team.txt
eat eval perf    <run_id>   # tok/s, TTFT, peak RAM, file size
```

Judge rubric lives in `prompts/eval_judge.txt`. Hard-fail axes
(allergen omission, condition-violating recommendations) are
non-blockable — any hard fail keeps the run from being promoted.

## How to export and bench on Android

```
eat export gguf       --run <run_id> --target gguf_q4km
eat export gguf       --run <run_id> --target gguf_q5km
eat export mediapipe  --run <run_id> --target mediapipe_litert

eat android bench --run <run_id> --target mediapipe_litert
```

Pass gate: ≥ 20 tok/s decode on the chosen Android device with
Q4_K_M (or the int4 MediaPipe task), zero red-team hard fails, and
quality score ≥ baseline + 15%.

## Safety policy

The model is a **food and lifestyle coach**, not a clinician. It:

- Always defers to the user's clinician for diagnosis and for any
  decision that changes a prescribed treatment plan.
- Never claims to diagnose a condition, never prescribes or adjusts
  medication dosage.
- Never recommends, or fails to flag, foods that contain a declared
  allergen.
- Treats eating-disorder cues (extreme calorie restriction, purging,
  obsession with "good vs bad" foods) by expressing concern and
  referring to a clinician, never by producing the requested numbers.
- Drug-food interaction claims are restricted to a small curated
  whitelist (e.g. warfarin + grapefruit / leafy greens, MAOI + aged
  cheese / fermented foods, statins + grapefruit). Anything else is
  deferred to a pharmacist.
- Paediatric and pregnancy queries always recommend confirming with
  the treating clinician before changing diet.

These are enforced both in the system prompt and as judge hard-fail
axes in `prompts/eval_judge.txt`.

## Open questions / future work

- Voice input via Gemma 3n's audio encoder (Phase 5).
- Deeper IFCT-India coverage and regional dish breadth (rasam vs sambhar
  variants, regional sweets, regional snacks).
- Cooking-method awareness (deep-fried vs steamed changes macros and
  the verdict — currently approximated, not measured).
- Continuous-glucose-monitor (CGM) integration as an optional input
  channel for diabetic users (out of scope for v1 — needs clinician
  involvement in the loop).
- Multilingual responses (Hindi, Tamil, Bangla) for the India audience.
- On-device fine-tune-during-use ("this is what I actually ate"
  feedback) — research only.
