#!/usr/bin/env python3
"""
Auto-research and data collection for the food health coach project.

Defines 10 data categories for training a multimodal Gemma 3n model that
identifies food in images, extracts ingredients, and gives personalised
eat/limit/avoid recommendations.

For each category:
  - Schema (what fields each record has)
  - 10 concrete examples in training format
  - Recommended real data sources (HuggingFace + public APIs)
  - Storage path
  - How the data feeds into training

Usage:
    uv run scripts/collect_food_data.py               # generate all synthetic data + configs
    uv run scripts/collect_food_data.py --show-only   # print guide, don't write files
    uv run scripts/collect_food_data.py --fetch-hf    # also pull real HF datasets (needs datasets pkg)
    uv run scripts/collect_food_data.py --type food_recognition  # one category only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent, indent
from typing import Any

SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
PROJ_DIR    = REPO_ROOT / "projects" / "food_health_coach"
PROC_DIR    = PROJ_DIR / "data" / "processed"
RAW_DIR     = PROJ_DIR / "data" / "raw"
IMG_DIR     = PROJ_DIR / "data" / "images"
CONFIGS_DIR = REPO_ROOT / "configs" / "datasets"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _msg(user: str | list, assistant: str, system: str | None = None) -> dict[str, Any]:
    """Build a chat-format training record."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def _vision_msg(image_path: str, text: str, assistant: str) -> dict[str, Any]:
    """Vision message: image + text → assistant reply (Gemma 3n multimodal format)."""
    return _msg(
        user=[
            {"type": "image", "image": image_path},
            {"type": "text", "text": text},
        ],
        assistant=assistant,
    )


def _save(name: str, records: list[dict], show_only: bool = False) -> Path:
    out = PROC_DIR / f"{name}.jsonl"
    total_chars = sum(
        len(str(r)) for r in records
    )
    if show_only:
        print(f"  [dry-run] would write {len(records)} records ({total_chars:,} chars) → {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ wrote {len(records)} records ({total_chars:,} chars) → {out.relative_to(REPO_ROOT)}")
    return out


def _yaml_config(name: str, kind: str, notes: str, source: str = "local") -> str:
    location = f"projects/food_health_coach/data/processed/{name}.jsonl"
    return dedent(f"""\
        name: {name}
        kind: {kind}
        source: {source}
        location: {location}
        splits:
          train: 0
          eval: 0
        license: synthetic
        notes: >
          {notes.strip()}
    """)


def _write_config(name: str, kind: str, notes: str, show_only: bool = False) -> None:
    path = CONFIGS_DIR / f"{name}.yaml"
    content = _yaml_config(name, kind, notes)
    if show_only:
        print(f"  [dry-run] would write config → {path.relative_to(REPO_ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)
        print(f"  ✓ config  → {path.relative_to(REPO_ROOT)}")
    else:
        print(f"  ~ config  already exists: {path.relative_to(REPO_ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CATEGORY DEFINITIONS
#  Each returns (records, yaml_kind, yaml_notes, sources_text)
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Food Visual Recognition ─────────────────────────────────────────────

def cat_food_recognition() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : food image (JPEG/PNG)  +  "What food is this? Identify it."
    Output: food name, cuisine, visual description, common variants

    TRAINING FORMAT
    ---------------
    Vision message: image path (or URL during prep) + text question → structured answer
    Images stored in:  projects/food_health_coach/data/images/food_recognition/
    JSONL stored in:   projects/food_health_coach/data/processed/food_recognition_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • food101             HF: ethz/food101          (101 classes, 101k images, MIT license)
    • Indian Food Images  HF: Sai1289/Indian_Food_Dataset  (Indian dishes)
    • UEC Food 256        https://foodcam.mobi/dataset256.html  (256 classes)
    • Open Images (food subset)  HF: open-images-v7 filtered by /Food label
    NOTE: for Gemma 3n training, images must be local files or base64-encoded in JSONL.
          Run `scripts/download_food_images.py --source food101` to materialise them.
    """
    IMG_PLACEHOLDER = "projects/food_health_coach/data/images/food_recognition/{id}.jpg"

    data = [
        ("dal_makhani",  "Dal Makhani",  "North Indian",
         "A deep-red creamy curry in a copper pot. Black lentils and kidney beans visible. "
         "Garnished with a swirl of fresh cream and a coriander sprig."),
        ("plain_roti",   "Roti / Chapati", "Pan-Indian",
         "A flat, round, slightly charred whole-wheat bread on a tawa. "
         "Soft texture with golden-brown spots from the griddle."),
        ("vegetable_biryani", "Vegetable Biryani", "Hyderabadi / South Indian",
         "Aromatic long-grain basmati rice layered with saffron-yellow and green herbs. "
         "Cashews, raisins and mixed vegetables visible on top."),
        ("masala_dosa",  "Masala Dosa",  "South Indian",
         "A large, crispy, golden-brown rice and lentil crepe folded around a spiced potato filling. "
         "Served with coconut chutney and sambar on the side."),
        ("paneer_tikka", "Paneer Tikka", "Punjabi",
         "Cubes of cottage cheese charred with spiced marinade on a skewer. "
         "Orange-red colour from tandoor heat and spices."),
        ("idli_sambar",  "Idli + Sambar", "South Indian",
         "Three white, steamed, pillow-soft rice-lentil discs beside a steel bowl of "
         "lentil vegetable soup (sambar) and green coconut chutney."),
        ("poha",         "Poha (Flattened Rice)", "Central / West Indian",
         "Yellow flattened rice sautéed with mustard seeds, curry leaves, onion and peas. "
         "Garnished with sev and coriander. Bright, dry texture."),
        ("khichdi",      "Khichdi",      "Pan-Indian",
         "A soft, golden porridge of rice and yellow moong dal cooked together with turmeric. "
         "Served with a knob of ghee on top."),
        ("gulab_jamun",  "Gulab Jamun",  "North Indian (dessert)",
         "Dark-brown, soft, spongy milk-solid balls soaked in rose-flavoured sugar syrup. "
         "Served in a small steel bowl; garnished with a pinch of cardamom."),
        ("aloo_paratha", "Aloo Paratha", "Punjabi",
         "Thick, layered whole-wheat flatbread stuffed with spiced mashed potato, "
         "cooked on a tawa with butter. Served with white yoghurt and pickle."),
    ]
    records = []
    for i, (fid, name, cuisine, desc) in enumerate(data):
        records.append(_vision_msg(
            image_path=IMG_PLACEHOLDER.format(id=fid),
            text="What food is in this image? Identify it with its name, cuisine, and a brief visual description.",
            assistant=f"**{name}** ({cuisine})\n\n{desc}",
        ))
    return records, "vision_pair", (
        "Food visual recognition: image → food name + cuisine + visual description. "
        "Covers 10 common Indian dishes. Replace image placeholders with real image paths."
    ), cat_food_recognition.__doc__ or ""


# ── 2. Ingredient Extraction ───────────────────────────────────────────────

def cat_ingredient_extraction() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : food name (text) or food image
    Output: ordered ingredient list with quantities per standard serving

    TRAINING FORMAT
    ---------------
    Text or vision message → JSON-structured ingredient list
    JSONL: projects/food_health_coach/data/processed/ingredient_extraction_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • Edamam Recipe API    https://developer.edamam.com  (free tier: 10k calls/mo)
    • HF: recipe_nlg       HF: mbien/recipenlg             (large recipe corpus)
    • HF: RecipeNER        HF: chambliss/RecipeNER         (NER-tagged ingredients)
    • Open Food Facts      https://world.openfoodfacts.org/data  (packaged foods)
    • IFCT 2017 (India)    Ministry of Health India — Indian Food Composition Tables
    """
    data = [
        ("Dal Makhani",
         [("Whole black lentils (sabut urad dal)", "150g"), ("Red kidney beans (rajma)", "50g"),
          ("Butter", "30g"), ("Heavy cream", "50ml"), ("Onion, chopped", "1 medium"),
          ("Ginger-garlic paste", "1 tbsp"), ("Tomato puree", "100g"),
          ("Cumin, coriander, garam masala", "1 tsp each"), ("Salt", "to taste")]),
        ("Masala Dosa",
         [("Raw rice", "200g"), ("Urad dal", "50g"), ("Fenugreek seeds", "1 tsp"),
          ("Potatoes, boiled", "3 medium"), ("Onion, sliced", "1 large"),
          ("Mustard seeds, curry leaves", "1 tsp each"), ("Green chilli", "1"),
          ("Turmeric, oil", "½ tsp each"), ("Salt", "to taste")]),
        ("Chicken Biryani (Hyderabadi)",
         [("Basmati rice", "300g"), ("Chicken pieces", "500g"),
          ("Yoghurt", "100g"), ("Saffron in warm milk", "pinch"),
          ("Fried onions (birista)", "50g"), ("Ghee", "3 tbsp"),
          ("Whole spices (cardamom, cloves, star anise)", "mix"),
          ("Mint leaves", "handful"), ("Ginger-garlic paste", "2 tbsp")]),
        ("Palak Paneer",
         [("Spinach (palak)", "300g"), ("Paneer", "200g"),
          ("Onion", "1 medium"), ("Tomato", "1 medium"),
          ("Ginger-garlic paste", "1 tbsp"), ("Cumin, coriander", "1 tsp each"),
          ("Fresh cream", "2 tbsp"), ("Oil or ghee", "2 tbsp"), ("Salt", "to taste")]),
        ("Roti / Chapati",
         [("Whole wheat flour (atta)", "100g"), ("Water", "~60ml"),
          ("Salt (optional)", "pinch"), ("Ghee or oil for cooking", "1 tsp")]),
        ("Chole Bhature",
         [("Chickpeas (kabuli chana)", "250g"), ("Onion, tomato, ginger-garlic", "paste"),
          ("Anardana (pomegranate seed) powder", "1 tsp"),
          ("Chole masala spice blend", "2 tsp"), ("All-purpose flour (maida)", "200g"),
          ("Yoghurt", "50g"), ("Baking powder", "½ tsp"), ("Oil for deep-fry", "as needed")]),
        ("Idli",
         [("Parboiled rice", "300g"), ("Urad dal", "100g"),
          ("Fenugreek seeds", "½ tsp"), ("Salt", "1 tsp"),
          ("Water (for soaking + grinding)", "as needed")]),
        ("Kheer (Rice Pudding)",
         [("Full-fat milk", "1 litre"), ("Basmati rice", "50g"),
          ("Sugar", "80g"), ("Cardamom powder", "¼ tsp"),
          ("Saffron strands", "pinch"), ("Cashews, almonds, pistachios", "2 tbsp each")]),
        ("Sambhar",
         [("Toor dal (split pigeon peas)", "150g"), ("Mixed vegetables (drumstick, brinjal, tomato)", "200g"),
          ("Sambhar powder", "2 tbsp"), ("Tamarind extract", "2 tbsp"),
          ("Mustard seeds, dried red chilli, curry leaves", "tempering"),
          ("Asafoetida (hing)", "pinch"), ("Oil", "1 tbsp"), ("Salt", "to taste")]),
        ("Gulab Jamun",
         [("Milk powder (mawa substitute)", "100g"), ("All-purpose flour", "2 tbsp"),
          ("Baking soda", "pinch"), ("Ghee", "1 tsp"), ("Milk (to bind)", "2 tbsp"),
          ("Sugar", "200g"), ("Water", "150ml"), ("Rose water", "1 tsp"),
          ("Cardamom powder", "¼ tsp"), ("Oil for deep-fry", "as needed")]),
    ]
    records = []
    for food, ingredients in data:
        ingr_lines = "\n".join(f"  - {name}: {qty}" for name, qty in ingredients)
        records.append(_msg(
            user=f"List all ingredients in **{food}** with approximate quantities for a standard serving.",
            assistant=f"Ingredients for {food} (standard serving):\n\n{ingr_lines}",
        ))
    return records, "instruction", (
        "Ingredient extraction: food name → ordered ingredient list with quantities. "
        "10 common Indian dishes. Text-only format."
    ), cat_ingredient_extraction.__doc__ or ""


# ── 3. Nutrition Profile ───────────────────────────────────────────────────

def cat_nutrition_profile() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : food name + serving size
    Output: calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
            key micronutrients (iron, calcium, vitamin C, folate), glycaemic index

    TRAINING FORMAT
    ---------------
    Text query → structured nutrition table + key health notes
    JSONL: projects/food_health_coach/data/processed/nutrition_profile_v1.jsonl

    REAL DATA SOURCES (all free / open)
    ------------------------------------
    • USDA FoodData Central API   https://fdc.nal.usda.gov/api  (no key needed for basic)
      GET https://api.nal.usda.gov/fdc/v1/foods/search?query=<food>&api_key=DEMO_KEY
    • IFCT 2017 CSV               Ministry of Health India (Indian food composition)
    • Open Food Facts API         https://world.openfoodfacts.org/cgi/search.pl
    • Nutritionix API             https://developer.nutritionix.com  (free tier)
    • HF: ds-food-facts           Small structured nutrition dataset
    """
    # (food, serving_g, kcal, prot, carb, fat, fiber, sugar, sodium_mg, gi, notes)
    data = [
        ("Plain Basmati Rice (cooked)", 150, 195, 4.0,  42.0, 0.5, 0.5, 0.0, 3,   58,
         "Low fat, moderate GI. Prefer brown basmati (GI≈50) for T2D."),
        ("Whole Wheat Roti (1 piece)", 40, 100, 3.5,  19.5, 1.2, 2.5, 0.5, 80,  62,
         "Higher fiber than white bread. Good iron source (1.5mg / roti)."),
        ("Dal Makhani (with cream)", 200, 290, 12.0, 28.0, 14.0, 7.0, 3.0, 480, 22,
         "GI 22 — very low. Caution on sodium if hypertensive. High sat-fat from cream."),
        ("Paneer (full-fat, fresh)", 100, 265, 18.3, 3.4, 20.8, 0.0, 0.4, 25,  27,
         "Excellent protein + calcium (480mg). High sat-fat; limit to 50g if weight-loss goal."),
        ("Curd / Plain Yoghurt", 150, 98,  7.0,  10.5, 2.5, 0.0, 9.0, 90,  36,
         "Probiotic benefits. Low GI. Good for gut health and calcium (~180mg)."),
        ("Banana (medium)", 120, 107, 1.3,  27.2, 0.4, 3.1, 14.4, 1,   51,
         "Quick energy; moderate GI. Limit to ½ banana for T2D. Rich in K (422mg)."),
        ("Boiled Egg (large)", 50,  78,  6.3,  0.6,  5.3, 0.0, 0.6, 62,  0,
         "Complete protein, all essential amino acids. Zero carbs. Safe for T2D."),
        ("Mango (Alphonso, ripe)", 100, 60,  0.8,  15.0, 0.4, 1.6, 13.7, 1,   56,
         "GI 56. Rich in vitamin C (36mg) and beta-carotene. Limit to 100g for T2D."),
        ("Chana Dal (cooked)", 100, 164, 8.9,  27.5, 2.6, 5.3, 2.2, 10,  8,
         "GI 8 — extremely low. Excellent protein and fiber. Ideal for T2D and PCOS."),
        ("Masala Dosa (with filling)", 180, 300, 7.5,  52.0, 8.5, 3.5, 2.0, 560, 55,
         "Moderate GI. Fermented batter aids digestion. High sodium in restaurant versions."),
    ]
    records = []
    for row in data:
        food, serv, kcal, prot, carb, fat, fib, sug, sod, gi, note = row
        records.append(_msg(
            user=f"Give me the nutritional breakdown for {food} ({serv}g / standard serving).",
            assistant=dedent(f"""\
                **{food}** — {serv} g serving

                | Nutrient      | Amount      |
                |---------------|-------------|
                | Calories      | {kcal} kcal |
                | Protein       | {prot} g    |
                | Carbohydrates | {carb} g    |
                | Fat           | {fat} g     |
                | Dietary Fibre | {fib} g     |
                | Sugar         | {sug} g     |
                | Sodium        | {sod} mg    |
                | Glycaemic Index | {gi}      |

                **Health note:** {note}
            """).strip(),
        ))
    return records, "instruction", (
        "Nutritional profile: food + serving size → macro/micro table + health note. "
        "10 common Indian foods. Based on USDA/IFCT reference values."
    ), cat_nutrition_profile.__doc__ or ""


# ── 4. Health Condition Dietary Guidance ──────────────────────────────────

def cat_health_condition_diet() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : health condition  +  food name
    Output: recommendation (eat / portion-limit / avoid), reason (evidence-based),
            safe portion, better alternatives

    TRAINING FORMAT
    ---------------
    User states condition + asks about food → structured advice
    JSONL: projects/food_health_coach/data/processed/health_condition_diet_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • USDA Dietary Guidelines 2020–2025     https://dietaryguidelines.gov
    • ADA (Diabetes) food recommendations   https://diabetes.org/food-nutrition
    • WHO DASH diet guidelines              for hypertension
    • NHS condition-specific diet guides    https://www.nhs.uk/conditions
    • Indian ICMR-NIN guidelines            https://www.nin.res.in
    NOTE: All records must include "consult your clinician" disclaimers.
          The model should NEVER prescribe medication changes based on diet.
    """
    data = [
        ("Type 2 Diabetes", "White rice",
         "PORTION-LIMIT", "70g cooked (half cup)",
         "White rice has a GI of 68–72 and causes rapid blood-sugar spikes. "
         "Small portions (70g cooked) with dal or sabzi slow digestion. "
         "Better choices: brown rice (GI≈50), millet (GI≈54), or cauliflower rice.",
         "Brown rice, Foxtail millet (kangni), barley"),
        ("Hypertension (High BP)", "Pickle (achaar)",
         "AVOID", "0g — avoid",
         "Commercial pickles contain 1000–2500 mg sodium per 30g. "
         "Excess sodium directly raises blood pressure. "
         "Use lemon or fresh herbs for tanginess instead.",
         "Fresh lemon juice, raw mango chutney (no salt)"),
        ("Chronic Kidney Disease (Stage 3)", "Banana",
         "PORTION-LIMIT", "½ small banana (60g)",
         "Banana is high in potassium (422mg per medium). CKD kidneys struggle to excrete K. "
         "½ a small banana provides ~210mg K, which is within typical CKD limits. "
         "Confirm your personal potassium budget with your nephrologist.",
         "Apple (low K), strawberries, pear"),
        ("PCOS / Insulin Resistance", "White bread (maida roti)",
         "AVOID", "0g — replace with whole grain",
         "Refined flour (maida) has GI ≈ 85 and rapidly spikes insulin, worsening "
         "insulin resistance central to PCOS. Switch to 100% atta roti or oat roti.",
         "Whole wheat roti, besan (chickpea flour) chilla, oat dosa"),
        ("Pregnancy (2nd trimester)", "Raw papaya",
         "AVOID", "0g — avoid throughout pregnancy",
         "Unripe/raw papaya contains papain and latex compounds that can trigger "
         "uterine contractions. Ripe papaya in small amounts (100g) is safe after "
         "the first trimester, but raw papaya must be strictly avoided.",
         "Ripe mango (100g), guava, oranges — all rich in folate/vitamin C"),
        ("Lactose Intolerance", "Paneer",
         "EAT with caution", "50g max, aged/drained paneer preferred",
         "Paneer is a low-lactose dairy food — much of the lactose drains off in whey. "
         "Most lactose-intolerant individuals tolerate 50g comfortably. "
         "If symptoms occur, use tofu as 1:1 substitute.",
         "Firm tofu, soy-based cheese"),
        ("High Cholesterol (Hyperlipidemia)", "Egg yolk",
         "PORTION-LIMIT", "1 egg yolk per day",
         "Each yolk provides 186mg cholesterol. AHA recommends <300mg/day for healthy adults, "
         "<200mg for high-risk patients. 1 whole egg per day is generally safe. "
         "Egg whites are cholesterol-free and an excellent protein source.",
         "Egg whites, tofu scramble, moong dal chilla"),
        ("Coeliac Disease / Gluten Intolerance", "Roti (wheat)",
         "AVOID", "0g — replace entirely",
         "Wheat contains gluten which triggers villous atrophy in coeliac disease. "
         "Even trace cross-contamination must be avoided. "
         "Rice roti, jowar (sorghum) roti, and bajra roti are safe gluten-free alternatives.",
         "Jowar roti, bajra roti, rice flour dosa, gluten-free atta blends"),
        ("Uric Acid / Gout", "Mutton / Red meat",
         "AVOID", "0g — avoid during flare",
         "Red meat is very high in purines (100–150mg/100g), which metabolise to uric acid. "
         "During a gout flare, avoid completely. In remission, limit to once weekly (50g). "
         "Hydrate well (3L water/day) to flush uric acid.",
         "Chicken breast (moderate purines), tofu, lentils (low purines)"),
        ("Iron-Deficiency Anaemia", "Chai (tea with milk)",
         "AVOID with iron-rich meals", "Avoid 1 hr before/after iron-rich meals",
         "Tea contains tannins and polyphenols that bind non-haeme iron and reduce absorption "
         "by up to 62%. Milk also inhibits iron absorption. "
         "Drink tea ≥1 hour away from iron-rich foods like dal, spinach, or meat.",
         "Vitamin C-rich drinks (amla juice, orange juice) taken WITH iron meals to enhance absorption"),
    ]
    records = []
    for condition, food, verdict, portion, reason, alts in data:
        records.append(_msg(
            user=f"I have {condition}. Can I eat {food}? How much is safe?",
            assistant=dedent(f"""\
                **Verdict: {verdict}**

                **Recommended portion:** {portion}

                **Why:** {reason}

                **Better alternatives:** {alts}

                ⚠️ This is general guidance, not medical advice. Please confirm with your doctor or registered dietitian — especially if you are on medication or managing multiple conditions.
            """).strip(),
        ))
    return records, "instruction", (
        "Health condition dietary guidance: condition + food → eat/limit/avoid + reason + portion + alternatives. "
        "10 condition-food pairs covering T2D, HTN, CKD, PCOS, pregnancy, anaemia, gout, etc."
    ), cat_health_condition_diet.__doc__ or ""


# ── 5. Personalised Meal Advice ────────────────────────────────────────────

def cat_personalized_advice() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : user_profile {age, sex, weight_kg, conditions, goals, medications, allergies}
              + food_query (text or image)
    Output: personalised verdict, tailored portion for THIS profile, specific reason
            (references profile fields), follow-up question

    TRAINING FORMAT
    ---------------
    User describes self + asks about food → tailored JSON or text response
    JSONL: projects/food_health_coach/data/processed/personalized_advice_v1.jsonl

    REAL DATA SOURCES
    -----------------
    Generate synthetically using the existing `eat.data.synthetic` module which
    already cross-multiplies user profiles × food_facts × context.
    Run: uv run eat data build --project food_health_coach --dataset personalized_advice_v1
    """
    profiles = [
        {"age": 52, "sex": "M", "weight_kg": 92, "conditions": ["Type 2 diabetes", "hypertension"],
         "medications": ["metformin 500mg", "amlodipine 5mg"], "goals": ["weight loss", "lower HbA1c"]},
        {"age": 28, "sex": "F", "weight_kg": 58, "conditions": ["PCOS"],
         "goals": ["manage insulin resistance", "reduce inflammation"], "diet": "vegetarian"},
        {"age": 67, "sex": "M", "weight_kg": 75, "conditions": ["high cholesterol", "mild CKD (stage 2)"],
         "medications": ["atorvastatin"], "goals": ["heart health", "kidney protection"]},
        {"age": 25, "sex": "F", "weight_kg": 65, "conditions": ["pregnancy (28 weeks)"],
         "goals": ["optimal foetal nutrition", "gestational diabetes prevention"]},
        {"age": 35, "sex": "M", "weight_kg": 80, "conditions": [],
         "goals": ["muscle gain", "performance"], "activity": "gym 5x/week"},
        {"age": 10, "sex": "F", "weight_kg": 32, "conditions": ["egg allergy"],
         "goals": ["healthy growth"], "parent_note": "needs school-lunch ideas"},
        {"age": 44, "sex": "F", "weight_kg": 70, "conditions": ["hypothyroidism"],
         "medications": ["levothyroxine"], "goals": ["weight management"]},
        {"age": 19, "sex": "M", "weight_kg": 62, "conditions": ["ulcerative colitis (remission)"],
         "goals": ["avoid flares", "maintain nutrition during high-stress exam period"]},
        {"age": 58, "sex": "F", "weight_kg": 68, "conditions": ["osteoporosis", "lactose intolerance"],
         "medications": ["calcium + vit D supplement"], "goals": ["bone health"]},
        {"age": 72, "sex": "M", "weight_kg": 65, "conditions": ["chronic gout"],
         "medications": ["allopurinol 100mg"], "goals": ["reduce flares", "balanced nutrition"]},
    ]
    foods = [
        ("mango (ripe, 1 medium)", "This is the mango season — can I have one?"),
        ("dal makhani with extra cream", "I love dal makhani. How much can I eat tonight?"),
        ("store-bought protein bar (20g protein, 12g sugar)", "Is this protein bar okay post-workout?"),
        ("banana + peanut butter smoothie", "I made this smoothie. Is it a good breakfast?"),
        ("chicken breast grilled (150g)", "How much grilled chicken can I have for lunch?"),
        ("rajma (red kidney beans) curry, 2 katoris", "My mom made rajma. Should I eat it?"),
        ("orange juice (200ml, packaged)", "Is packaged orange juice okay in the morning?"),
        ("curd rice (200g)", "Can I have curd rice for dinner? I'm craving it."),
        ("soy milk (250ml)", "I switched to soy milk. Is that a good call for my condition?"),
        ("halwa (sooji / semolina, 2 tbsp ghee, sugar)", "Had halwa at a family function. How bad was it?"),
    ]
    advice_list = [
        # profile 0: T2D + HTN, weight loss
        "**You can have mango** — but cap it at 100g (about ½ a medium mango). "
        "Mango's GI is 51–56, which is moderate; a full mango (200g) provides ~26g sugar that spikes your HbA1c. "
        "Pair with 10g mixed nuts to slow absorption. Avoid mango juice or canned mango syrup entirely. "
        "Given your metformin + amlodipine regimen, there are no drug-food interactions with mango to worry about.",
        # profile 1: PCOS
        "**Yes to dal makhani, but swap the cream.** "
        "The lentils (GI≈22) are excellent for insulin resistance and provide 12g protein per serving. "
        "However, the cream adds ~5g sat-fat which drives inflammation. Ask for it without cream, "
        "or stir in 2 tbsp curd instead. Limit to 150g and pair with jowar roti for a complete, PCOS-friendly meal.",
        # profile 2: cholesterol + CKD
        "**Partial yes on the protein bar.** 12g sugar per bar is too high for your cholesterol profile "
        "(added sugar raises triglycerides). Look for bars with <5g sugar and no palm oil. "
        "For mild CKD stage 2, the 20g protein load is acceptable once a day. "
        "Watch for phosphorus additives (often listed as 'phosphate' in ingredients) — these accelerate CKD.",
        # profile 3: pregnancy 28 weeks
        "**Yes — this is a great pregnancy breakfast.** Banana provides 422mg potassium and B6 (helps nausea). "
        "Peanut butter adds folate, protein, and healthy fats. "
        "Total GI is ~45 which keeps energy steady. One concern: peanuts are a top allergen — "
        "if there is no family history, it is fine; if there is, use almond butter instead. "
        "Keep to 1 banana + 1 tbsp peanut butter to stay within 300 extra kcal/day.",
        # profile 4: muscle gain, gym
        "**150g grilled chicken is the right call.** "
        "That's 38g of complete protein — ideal post-training. "
        "Time it within 90 minutes of your workout for optimal muscle protein synthesis. "
        "Add 200g sweet potato or 1 cup rice to refill glycogen. "
        "No restrictions for your profile. Great choice.",
        # profile 5: child, egg allergy
        "**Rajma is perfect for her school lunch.** "
        "2 katoris (200g) provide ~12g protein, 9g fiber, and plenty of iron — "
        "all critical for a 10-year-old's growth. Egg-free, dairy-free, nut-free. "
        "Pair with 1–2 rotis to make it a complete meal (~550 kcal). "
        "Add a squeeze of lemon for vitamin C, which triples iron absorption from the rajma.",
        # profile 6: hypothyroidism
        "**Yes, but time it carefully.** Packaged OJ is fine for your thyroid — the concern is timing: "
        "calcium-fortified OJ can reduce levothyroxine absorption by 30%. "
        "Take your levothyroxine on an empty stomach (30–60 min before OJ). "
        "For weight management, the 25g sugar in 200ml OJ is high. "
        "Prefer a whole orange (slower sugar release, more fiber) and dilute OJ to 100ml.",
        # profile 7: UC remission
        "**Curd rice is one of the best choices for UC remission.** "
        "The probiotics in curd soothe the gut lining. The white rice is easy to digest. "
        "The combo is a traditional gut-healing food in South India. "
        "Go for full 200g tonight. Avoid adding pickle or raw onion. "
        "During exam stress, eat smaller meals more frequently to avoid cramping.",
        # profile 8: osteoporosis + lactose intolerance
        "**Yes — soy milk is an excellent switch for you.** "
        "Fortified soy milk provides 300mg calcium per 250ml, similar to cow's milk, "
        "with no lactose. The isoflavones in soy may also modestly reduce bone resorption. "
        "Ensure the brand is fortified with both calcium AND vitamin D3 (check label). "
        "Your existing calcium + Vit D supplement + soy milk together should meet your 1200mg/day target.",
        # profile 9: chronic gout
        "**Halwa was a high-risk choice, but it is done.** For future reference: "
        "sooji (semolina) is moderate-purine (acceptable), but the 2 tbsp ghee raises uric acid "
        "indirectly by promoting weight gain and reducing kidney clearance. "
        "The sugar load also increases uric acid production via fructose pathway. "
        "Going forward: drink 3 extra glasses of water today to flush uric acid. "
        "At family functions, go for curd-based sweets or fruit instead. "
        "Your allopurinol dose is unchanged — no interaction with this food.",
    ]
    records = []
    for i, (profile, (food, question), advice) in enumerate(zip(profiles, foods, advice_list)):
        records.append(_msg(
            user=f"Profile: {json.dumps(profile, indent=2)}\n\nQuestion: {question} (food: {food})",
            assistant=advice + "\n\n⚠️ Always confirm with your doctor or dietitian before changing your diet.",
        ))
    return records, "instruction", (
        "Personalised advice: user profile + food → tailored eat/limit/avoid + specific quantities. "
        "10 examples spanning T2D, PCOS, CKD, pregnancy, sports nutrition, paediatric, thyroid, "
        "IBD, osteoporosis, gout."
    ), cat_personalized_advice.__doc__ or ""


# ── 6. Portion Size Guidance ───────────────────────────────────────────────

def cat_portion_guidance() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : food name + visual cue (e.g. "medium bowl", "restaurant plate") OR image
    Output: estimated weight in grams, calorie range, standard household measure,
            visual anchor (compare to palm/fist/deck-of-cards)

    TRAINING FORMAT
    ---------------
    Text or vision query → structured portion estimate
    JSONL: projects/food_health_coach/data/processed/portion_guidance_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • USDA Standard Portions    https://www.nal.usda.gov/legacy/afsic/food-portions
    • NIN (India) portion guide  https://www.nin.res.in/portionsize.html
    • Harvard Health Plate       https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate
    • For vision: use food101 images labelled with weight from the ECOC-Calorie-Cam dataset
    """
    data = [
        ("a standard katori (small steel bowl) of dal",
         "~100–120 ml, roughly 110g cooked dal",
         "~120–140 kcal depending on dal type",
         "About the size of a tennis ball. Hold your hand in a fist — that's one katori."),
        ("a medium chapati / roti",
         "~30–35g (thin home-style) to 45g (thick restaurant-style)",
         "~75–115 kcal",
         "Same diameter as an outstretched hand, thickness of a CD."),
        ("a restaurant serving of butter chicken (one portion)",
         "~250–300g including sauce",
         "~350–420 kcal",
         "A Bollywood-movie-style 'bowl' is usually 2–2.5× the right portion. "
         "A correct portion is smaller than your two fists together."),
        ("a handful of mixed nuts (almonds, cashews, walnuts)",
         "~28–35g",
         "~170–210 kcal",
         "One 'handful' = 1 cupped palm ≈ 28g. A matchbox-sized portion."),
        ("a serving of biryani (home-cooked, plated)",
         "~250–350g",
         "~350–500 kcal",
         "A dinner-plate serving at home is usually 1.5× the ideal. "
         "A correct portion fits in a large katori (300ml)."),
        ("one dosa (plain, no filling)",
         "~80–100g",
         "~120–145 kcal",
         "A standard dosa is thin, crispy, and larger than a dinner plate. "
         "Weight mainly comes from the batter, not the size."),
        ("a glass of whole-fat cow's milk",
         "~240–250ml",
         "~150–165 kcal",
         "A standard tumbler in Indian homes is ~180–200ml. "
         "A full 'glass' is usually 240ml."),
        ("a piece of gulab jamun (one ball)",
         "~35–45g (one standard-sized ball)",
         "~130–160 kcal",
         "About the size of a large marble or small lime. "
         "Restaurant portions often serve 2–3 pieces — that's 260–480 kcal."),
        ("one medium mango (Alphonso)",
         "~200–230g whole, ~150–170g edible flesh",
         "~90–100 kcal (edible portion)",
         "Roughly the size of a large fist. The pit accounts for ~20% of the weight."),
        ("a bowl of cornflakes with 200ml milk",
         "~30g cornflakes + 200ml semi-skim milk",
         "~250–290 kcal",
         "The packet's 'recommended serving' is 30g, which looks tiny — about ¾ cup. "
         "Most people pour 60–80g without realising, nearly tripling the calories."),
    ]
    records = []
    for food, weight, kcal, anchor in data:
        records.append(_msg(
            user=f"How big is {food}? Estimate the weight in grams and calories.",
            assistant=dedent(f"""\
                **Estimated portion:** {weight}
                **Calorie range:** {kcal}
                **Visual anchor:** {anchor}
            """).strip(),
        ))
    return records, "instruction", (
        "Portion size guidance: food description → estimated grams + calories + visual anchor. "
        "10 common Indian and universal foods."
    ), cat_portion_guidance.__doc__ or ""


# ── 7. Allergen Detection ──────────────────────────────────────────────────

def cat_allergen_detection() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : food name or ingredient list
    Output: list of allergens present (Big-9: milk, egg, peanut, tree nut, wheat,
            soy, fish, shellfish, sesame), severity note, cross-contact risks,
            intolerance flags (lactose, gluten, fructose)

    TRAINING FORMAT
    ---------------
    Text query → allergen table + cross-contact warning
    JSONL: projects/food_health_coach/data/processed/allergen_detection_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • Open Food Facts (allergen field)   https://world.openfoodfacts.org
    • FoodData Central (allergen data)   https://fdc.nal.usda.gov
    • EU Food Information Regulation 14 allergens list
    • FSSAI India allergen labelling guidelines (2022)
    • HF: allergen-detector dataset (community)
    """
    data = [
        ("Chole Bhature (chickpeas + maida deep-fried bread)",
         [("Wheat (gluten) — in bhature dough", "HIGH"),
          ("Sesame — optional garnish, cross-contact risk", "MEDIUM")],
         "Gluten: coeliac patients must avoid. Sesame: verify oil used for frying. "
         "Chickpeas are NOT a tree nut allergen; cross-reactivity with peanuts is possible in rare cases."),
        ("Paneer Makhani (restaurant)",
         [("Milk / Dairy — paneer + cream + butter", "HIGH"),
          ("Tree nuts — cashew paste common in makhani sauce", "HIGH"),
          ("Wheat — some restaurants thicken with atta", "MEDIUM")],
         "Major allergens: dairy and cashew. Always ask restaurant about cashew paste. "
         "Lactose intolerant individuals may tolerate small paneer portions."),
        ("Hyderabadi Chicken Biryani",
         [("Egg — some recipes add boiled egg or egg in marinade", "MEDIUM"),
          ("Tree nuts — fried cashews and almonds in garnish", "HIGH"),
          ("Milk — yoghurt in marinade", "LOW / trace"),
          ("Wheat — some use atta dough to seal the biryani pot (dum seal)", "LOW / trace")],
         "Main risks: cashews, almonds in garnish (HIGH). Ask if egg is included. "
         "Wheat cross-contact from dum seal is trace-level."),
        ("Idli Sambar",
         [("None of the Big-9 in standard recipe", "NONE")],
         "Naturally free of the 8 major allergens. Cross-contact from shared equipment possible. "
         "Gluten-free (rice + urad dal). Safe for lactose intolerant. "
         "Asafoetida (hing) may contain wheat flour as filler — check the brand."),
        ("Chocolate Barfi (Indian sweet)",
         [("Milk — made with khoya (reduced milk)", "HIGH"),
          ("Tree nuts — cashews, pistachios", "HIGH"),
          ("Soy — some brands use vanaspati with soy oil", "MEDIUM"),
          ("Gluten — cross-contact in sweet shops", "LOW / cross-contact")],
         "High-risk for dairy and tree-nut allergies. Purchase from allergen-aware bakeries only."),
        ("Peanut Chutney (South Indian style)",
         [("Peanut", "HIGH — primary ingredient")],
         "SEVERE risk for peanut-allergic individuals. Peanut is the primary ingredient. "
         "May contain traces of sesame, tree nuts from shared grinding equipment."),
        ("Whole Wheat Roti (home-made)",
         [("Wheat (gluten)", "HIGH — primary ingredient")],
         "Strict no for coeliac disease. Most wheat-intolerant individuals should avoid. "
         "Replace with jowar (sorghum), bajra (pearl millet), or rice flour roti. "
         "No other Big-9 allergens in plain roti."),
        ("Rajma (Red Kidney Bean) Curry",
         [("None of the Big-9 in standard recipe", "NONE"),
          ("Soy — cross-contact possible in pulses packed on shared lines", "TRACE")],
         "Generally very safe. Legume-allergic individuals: rajma is a legume; "
         "cross-reactivity with soy and peanut is possible but uncommon. "
         "Always cook rajma thoroughly — raw/undercooked rajma contains phytohaemagglutinin."),
        ("Store-Bought Granola Bar (oats, honey, nuts)",
         [("Tree nuts — almonds, walnuts", "HIGH"),
          ("Wheat / Gluten — oats cross-contamination", "MEDIUM"),
          ("Milk — whey protein or milk powder", "MEDIUM"),
          ("Soy — soy lecithin as emulsifier", "LOW"),
          ("Sesame — seeds or cross-contact", "LOW")],
         "Read the label every time — recipe changes frequently. "
         "Look for 'manufactured in a facility with tree nuts/peanuts' warning. "
         "Coeliac: use certified GF oat bars only."),
        ("Masala Dosa (restaurant)",
         [("Milk — some add butter to the tawa", "LOW / cross-contact"),
          ("Wheat — shared tawa in some kitchens", "MEDIUM / cross-contact"),
          ("Sesame — sesame oil used in some recipes", "LOW")],
         "Standard dosa batter is gluten-free (rice + urad dal). "
         "Risk is mainly cross-contact from shared equipment. "
         "Ask restaurant to use a clean tawa with plain oil for allergy orders."),
    ]
    records = []
    for food, allergens, notes in data:
        allergen_lines = "\n".join(
            f"  • **{a}** — risk: {risk}" for a, risk in allergens
        )
        records.append(_msg(
            user=f"What allergens are present in {food}? I have multiple food allergies.",
            assistant=dedent(f"""\
                **Allergen profile for {food}:**

{indent(allergen_lines, '                ')}

                **Notes:** {notes}

                ⚠️ Always check the specific recipe or restaurant — preparation methods vary. When in doubt, avoid or ask the chef.
            """).strip(),
        ))
    return records, "instruction", (
        "Allergen detection: food → allergens present (Big-9 + lactose/gluten), severity, cross-contact. "
        "10 common Indian foods and packaged items."
    ), cat_allergen_detection.__doc__ or ""


# ── 8. Meal Planning ───────────────────────────────────────────────────────

def cat_meal_planning() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : user profile (conditions, calorie target, cuisine preference, day type)
    Output: full day meal plan {breakfast, morning_snack, lunch, evening_snack, dinner}
            with portions, macros per meal, total macros, and rationale

    TRAINING FORMAT
    ---------------
    User describes needs → structured full-day plan
    JSONL: projects/food_health_coach/data/processed/meal_planning_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • Generate synthetically — cross-multiply user_profile_pool × calorie_targets × day_types
    • NIN India Daily Recommended Allowances (https://www.nin.res.in/RDA.html)
    • ADA meal planning guidelines
    • Use LLM teacher (ANTHROPIC_API_KEY) for high-quality plan generation
    """
    plans = [
        ("A 52-year-old man with T2D and hypertension. Target: 1800 kcal, Indian vegetarian, "
         "no salt restriction relaxed (max 1500mg sodium/day). Weekday.",
         dedent("""\
         **Full-day plan — T2D + Hypertension | 1800 kcal | Indian Vegetarian**

         🌅 **Breakfast (7 am) — 400 kcal**
         • 2 small oat dosas with home-made tomato chutney (no salt)
         • 1 cup green tea (no sugar)
         *Macros: 52g carb, 12g protein, 8g fat, 6g fibre*

         🍎 **Morning snack (10 am) — 150 kcal**
         • 15 almonds + 1 small guava
         *Macros: 10g carb, 4g protein, 10g fat, 3g fibre*

         ☀️ **Lunch (1 pm) — 550 kcal**
         • 1½ jowar rotis + 1 katori chana dal + 1 katori palak sabzi + small bowl curd
         *Macros: 72g carb, 22g protein, 10g fat, 12g fibre*

         🌆 **Evening snack (4 pm) — 150 kcal**
         • 1 cup unsweetened buttermilk (chaas) + 1 handful murmura (plain puffed rice)
         *Macros: 14g carb, 5g protein, 2g fat*

         🌙 **Dinner (7 pm) — 550 kcal**
         • 1 jowar roti + 1 katori moong dal + large plate stir-fried vegetables (no salt)
         • Small bowl of curd
         *Macros: 68g carb, 20g protein, 8g fat, 10g fibre*

         📊 **Day total: 1800 kcal | 216g carb | 63g protein | 38g fat | 31g fibre | ~1400mg sodium**
         Glycaemic load is low throughout. No added sugar. All grains whole-grain.""")),
        ("A 25-year-old woman, 28 weeks pregnant, vegetarian, 2100 kcal target. "
         "Wants to prevent gestational diabetes. Includes folate, iron, calcium focus.",
         dedent("""\
         **Full-day plan — Pregnancy (28 wk) | 2100 kcal | Vegetarian | GDM-prevention**

         🌅 **Breakfast (7:30 am) — 500 kcal**
         • Methi paratha (1 large, made with atta + fenugreek leaves) + curd (150g)
         • 1 glass warm milk with a pinch of turmeric
         *Folate from methi, calcium from milk + curd, protein 18g*

         🍎 **Morning snack (10:30 am) — 200 kcal**
         • Sprout chaat (½ cup mixed sprouts, lemon, no chilli) + 1 orange
         *Iron from sprouts, vitamin C from orange (boosts iron absorption), folate*

         ☀️ **Lunch (1 pm) — 600 kcal**
         • 2 atta rotis + 1 katori rajma + 1 katori lauki sabzi + small bowl curd
         • 1 glass chaas post-meal
         *Protein 24g, folate from rajma, calcium from curd*

         🌆 **Evening snack (4 pm) — 200 kcal**
         • Roasted chana (30g) + 1 small banana + 5 walnuts
         *Iron, omega-3 from walnuts, potassium from banana*

         🌙 **Dinner (7 pm) — 600 kcal**
         • 1½ atta rotis + 1 katori moong-palak dal + 1 katori curd
         *Iron + folate from palak, easy digestion for evening*

         📊 **Day total: 2100 kcal | 280g carb | 72g protein | 55g fat | folate ~600µg | iron ~28mg | calcium ~1100mg**
         All GI values moderate-low. Rich in folate, iron, calcium for foetal development.""")),
    ]
    records = []
    for profile, plan in plans:
        records.append(_msg(
            user=f"Build me a full-day Indian meal plan for this profile: {profile}",
            assistant=plan.strip() + "\n\n⚠️ Adjust portions based on hunger and glucose readings. "
                       "Consult your dietitian for final personalisation.",
        ))
    return records, "instruction", (
        "Meal planning: user profile + calorie target → full-day Indian meal plan with macros. "
        "2 detailed examples (T2D+HTN and pregnancy). Expand with synthetic generator."
    ), cat_meal_planning.__doc__ or ""


# ── 9. Food Substitution ──────────────────────────────────────────────────

def cat_food_substitution() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Input : food + restriction (health condition, allergy, dietary preference)
    Output: list of substitutes with reason, ratio (how to swap in a recipe),
            taste/texture note, and nutrition comparison

    TRAINING FORMAT
    ---------------
    User asks for swap → ordered list of alternatives with practical swap ratios
    JSONL: projects/food_health_coach/data/processed/food_substitution_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • Ingredient substitution DBs (text):  Yummly API, Allrecipes scrape
    • HF: recipe-ingredient-substitution   Community-curated swap dataset
    • Synthetic generation using knowledge base
    """
    subs = [
        ("white rice", "Type 2 diabetes (reduce GI)",
         [("Cauliflower rice", "1:1 by volume. GI≈15. Roast or stir-fry riced cauliflower. "
           "Lower calorie (25 kcal/100g vs 130). Slightly different texture."),
          ("Foxtail millet (kangni)", "1:1 by weight. GI≈54. Cook like rice. "
           "Higher fibre (6g) and protein (12g) per 100g raw."),
          ("Brown basmati rice", "1:1. GI≈50 vs 68. Needs 5 more minutes cooking. "
           "Identical taste. Best 'easiest swap' option.")]),
        ("sugar (in chai)", "diabetes / weight loss",
         [("Stevia (green leaf powder)", "¼ tsp stevia = 1 tsp sugar. Zero calories, GI=0. "
           "Slight herbal aftertaste; reduce gradually."),
          ("Jaggery (khand / organic)", "½ tsp jaggery = 1 tsp sugar. GI≈84 — still high, "
           "but contains iron + trace minerals. NOT suitable for T2D; use for general wellness only."),
          ("Cinnamon (taste-masking)", "Not a sweetener but ½ tsp in chai reduces perceived need for sugar. "
           "Also improves insulin sensitivity.")]),
        ("ghee (in cooking)", "high cholesterol / heart disease",
         [("Extra virgin olive oil", "1:1. Lowers LDL, raises HDL. "
           "Use for tadka (tempering) at medium heat only — smoke point is lower."),
          ("Cold-pressed mustard oil", "1:1. Traditional Indian oil. "
           "Rich in omega-3 ALA. Good for high-heat cooking."),
          ("Rice bran oil", "1:1. Neutral taste. Good for frying. "
           "Contains oryzanol which actively reduces cholesterol.")]),
        ("paneer", "dairy allergy / veganism",
         [("Firm tofu", "1:1. Crumble or cube exactly as paneer. "
           "Marinate 30 min for flavour absorption. Lower sat-fat, similar protein."),
          ("Cashew-based vegan paneer", "1:1. Rich taste. "
           "Higher calorie; tree-nut allergen — avoid if nut-allergic.")]),
        ("atta (wheat flour) roti", "coeliac / gluten intolerance",
         [("Jowar (sorghum) flour roti", "1:1. Slightly denser. GI≈62. "
           "Mix 10% arrowroot to improve binding. Certified GF."),
          ("Bajra (pearl millet) flour roti", "1:1. Earthy taste. High iron. "
           "Best for winter; mix with jowar 50:50 for lighter texture."),
          ("Rice flour dosa / crepe", "Different product, not a direct roti swap. "
           "Excellent GF option for a carb serving.")]),
        ("whole milk (in tea/smoothie)", "lactose intolerance / dairy allergy",
         [("Fortified soy milk", "1:1. 300mg calcium. Highest protein of plant milks (8g/cup). "
           "Good for T2D (low sugar if unsweetened)."),
          ("Fortified oat milk", "1:1. Creamier taste. Higher carb (15g/cup). "
           "Not ideal for T2D; good for athletic individuals."),
          ("Lactase-treated cow milk (Lactaid)", "1:1. Identical taste/nutrition. "
           "Best for lactose intolerance (not dairy allergy).")]),
        ("deep-fried snacks (samosa, pakoda)", "obesity / heart disease",
         [("Air-fried version of the same item", "1:1 by recipe. 70–80% less fat. "
           "Texture is slightly less crispy. Coat lightly with 1 tsp oil spray."),
          ("Baked version", "1:1. 60% less fat. Requires oven. "
           "Roll thin for crispness."),
          ("Roasted chana + spiced makhana mix", "Different product. "
           "Satisfies crunch craving. High protein, low fat. "
           "2 handfuls = ~150 kcal vs 350 kcal for 2 samosas.")]),
        ("maida (all-purpose flour) in sweets/bhature", "gluten sensitivity / health",
         [("50% atta + 50% maida blend", "1:1. Softer texture retained. "
           "Doubles fibre. Easy transition step."),
          ("Whole wheat atta (100%)", "1:1. Denser result. "
           "Best for parathas, bhature with overnight fermentation."),
          ("Almond flour (for sweets/baking)", "1:1 by weight. Low GI. "
           "Higher calorie but low carb. Not for Indian breads — use for ladoo, cake.")]),
        ("table salt (NaCl) in cooking", "hypertension / kidney disease",
         [("Rock salt (sendha namak) in moderation", "1:1. Same sodium content — "
           "NOT a low-sodium option despite marketing. Same restriction applies."),
          ("Potassium chloride salt substitutes (NoSalt, Nu-Salt)", "½:1 ratio. "
           "Reduces sodium by 50%. CAUTION: avoid if on ACE inhibitors or K-sparing diuretics — "
           "risk of hyperkalemia."),
          ("Herbs and spices: cumin, coriander, amchur, lemon zest", "Replace ½ of salt. "
           "Adds depth without sodium. Most effective approach long-term.")]),
        ("packaged fruit juice (200ml)", "diabetes / obesity",
         [("Whole fruit (same fruit)", "1 whole fruit. Retains fibre, GI drops 15–20 points. "
           "Fewer calories, slower sugar release."),
          ("Infused water (cucumber, mint, lemon)", "Zero calories, zero sugar. "
           "Satisfies thirst + hydration. Not a nutritional substitute."),
          ("Vegetable juice (tomato, amla)", "200ml. Much lower sugar. "
           "Tomato: 4g sugar vs 22g in OJ. Rich in vitamins C, lycopene.")]),
    ]
    records = []
    for food, restriction, substitutes in subs:
        sub_text = "\n\n".join(
            f"**{i+1}. {s.split('.')[0]}**\n{'.'.join(s.split('.')[1:]).strip()}"
            for i, s in enumerate(s for s, *_ in [sub.split('. ', 1) + [''] for sub in
                                                   [f"{name}. {detail}" for name, detail in
                                                    [(s[0], s[1]) for s in substitutes]]])
        )
        # Simplified formatting:
        sub_lines = "\n\n".join(
            f"**Option {i+1}: {name}**\n{detail}"
            for i, (name, detail) in enumerate(substitutes)
        )
        records.append(_msg(
            user=f"I need to replace {food} due to {restriction}. What are my best options in Indian cooking?",
            assistant=f"Here are the best substitutes for **{food}** ({restriction}):\n\n{sub_lines}\n\n"
                      "Start with Option 1 for the closest match, or Option 3 for the biggest health benefit.",
        ))
    return records, "instruction", (
        "Food substitution: food + restriction → ordered swap options with ratio, taste note, nutrition comparison. "
        "10 common Indian ingredients: rice, sugar, ghee, paneer, atta, milk, fried snacks, maida, salt, juice."
    ), cat_food_substitution.__doc__ or ""


# ── 10. Multi-turn Coaching Sessions ──────────────────────────────────────

def cat_multimodal_coaching() -> tuple[list[dict], str, str, str]:
    """
    SCHEMA
    ------
    Multi-turn conversation that may include:
      - Image of food or meal (image + text)
      - Follow-up text questions
      - Context tracking (profile established in turn 1, referenced in later turns)

    TRAINING FORMAT
    ---------------
    Full conversation: system prompt + user/assistant turns (≥3 turns each)
    Some turns include image content type for vision branch
    JSONL: projects/food_health_coach/data/processed/multimodal_coaching_v1.jsonl

    REAL DATA SOURCES
    -----------------
    • Synthetic generation using existing eat.data.synthetic module
    • Multi-turn coaching is best generated by teacher LLM (ANTHROPIC_API_KEY)
    • Use scripts/collect_food_data.py --fetch-hf to pull diet-related dialogue datasets:
        HF: bitext/Bitext-customer-support-llm-chatbot-training-dataset (adapt)
        HF: lavita/ChatDoctor-iCliniq (medical Q&A, adapt carefully)
    """
    SYSTEM = ("You are ZenFood, a personalised food health coach. "
              "You have access to the user's health profile. "
              "Be warm, specific, and always advise consulting a clinician for medical decisions.")
    IMG = "projects/food_health_coach/data/images/coaching/"

    sessions = [
        # Session 1: T2D user shares lunch photo
        [
            {"role": "system",  "content": SYSTEM},
            {"role": "user",    "content": [
                {"type": "image", "image": f"{IMG}lunch_rice_dal.jpg"},
                {"type": "text",  "text": "Here's my lunch. I have type 2 diabetes. Is this okay?"},
            ]},
            {"role": "assistant", "content":
                "I can see white rice with dal and a sabzi — good instinct sharing it! "
                "Here's the issue: the rice portion looks large (~200g cooked), which is about 45g of fast carbs. "
                "For T2D, I'd suggest halving the rice, adding a small bowl of curd, and eating the dal first. "
                "This order (dal → sabzi → rice) slows glucose absorption significantly. "
                "The dal is excellent — keep that."},
            {"role": "user",    "content": "What if I switch to brown rice? Does that solve the problem?"},
            {"role": "assistant", "content":
                "Brown rice helps — GI drops from 68 to 50, so less blood-sugar spike. "
                "But portion still matters. At 200g, even brown rice provides 40g carbs. "
                "My recommendation: 100–120g cooked brown rice + extra dal + a salad side. "
                "Also time your meal: eating 10 minutes after a 5-minute walk before lunch reduces "
                "post-meal glucose by 12–15% on average."},
            {"role": "user",    "content": "Can I eat fruit after this meal?"},
            {"role": "assistant", "content":
                "I'd wait 90 minutes. Fruit on top of a rice meal stacks the carb load. "
                "Best fruit for you post-lunch: ½ guava (low GI, high fibre and vitamin C) or "
                "a small bowl of papaya. Avoid banana or mango right after a rice meal. "
                "If your glucose reading 2 hours after eating is >180 mg/dL consistently, bring it up with your doctor."},
        ],
        # Session 2: Parent asking about child's school tiffin
        [
            {"role": "system",  "content": SYSTEM},
            {"role": "user",    "content":
                "My 9-year-old daughter has a peanut allergy. She's a picky eater. "
                "Can you suggest 5 school tiffin ideas that are nut-free, filling, and she'll actually eat?"},
            {"role": "assistant", "content":
                "Absolutely — here are 5 nut-free, child-friendly tiffin ideas:\n\n"
                "1. **Mini masala dosa** (rice+dal batter, no nuts) with coconut chutney (verify nut-free coconut powder)\n"
                "2. **Paneer + capsicum roll** in whole wheat roti — colourful, finger-food friendly\n"
                "3. **Curd rice** with pomegranate seeds — cooling, familiar, zero allergen risk\n"
                "4. **Moong dal chilla** (savoury crepes) with mild tomato dip — high protein, no gluten\n"
                "5. **Vegetable poha** (flattened rice) — light, quick to make, naturally nut-free\n\n"
                "Keep a label on the box: 'PEANUT ALLERGY — please do not share.' "
                "Does she have any other food preferences or dislikes I should work around?"},
            {"role": "user",    "content": "She doesn't like anything spicy. Also she's a bit underweight."},
            {"role": "assistant", "content":
                "Good to know — mild spicing is easy to manage. For her weight, focus on calorie-dense "
                "but mild foods:\n\n"
                "• Replace water in poha with coconut milk (adds ~60 kcal, mild sweetness children love)\n"
                "• Add a small homemade rice krispie laddoo (jaggery + puffed rice, no peanut) as dessert\n"
                "• The paneer roll is perfect — full-fat paneer is 265 kcal/100g and mild\n"
                "• Sneak in a teaspoon of ghee in curd rice — invisible, mild flavour, +45 kcal\n\n"
                "How much is she eating currently vs what's expected for her age and height? "
                "If she's significantly below the growth curve, a paediatric dietitian visit would be worth it."},
        ],
    ]

    records = []
    for session in sessions:
        records.append({"messages": session})
    return records, "instruction", (
        "Multi-turn coaching sessions (2 examples). Vision + text turns, context tracking. "
        "Replace image placeholders with real images. Expand with synthetic generator + teacher LLM."
    ), cat_multimodal_coaching.__doc__ or ""


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

ALL_CATEGORIES: dict[str, tuple] = {
    "food_recognition_v1":       (cat_food_recognition,    "vision_pair"),
    "ingredient_extraction_v1":  (cat_ingredient_extraction, "instruction"),
    "nutrition_profile_v1":      (cat_nutrition_profile,    "instruction"),
    "health_condition_diet_v1":  (cat_health_condition_diet, "instruction"),
    "personalized_advice_v1":    (cat_personalized_advice,  "instruction"),
    "portion_guidance_v1":       (cat_portion_guidance,     "instruction"),
    "allergen_detection_v1":     (cat_allergen_detection,   "instruction"),
    "meal_planning_v1":          (cat_meal_planning,        "instruction"),
    "food_substitution_v1":      (cat_food_substitution,    "instruction"),
    "multimodal_coaching_v1":    (cat_multimodal_coaching,  "instruction"),
}


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace auto-fetch
# ═══════════════════════════════════════════════════════════════════════════

HF_SOURCES = {
    "food_recognition_v1": {
        "repo": "ethz/food101",
        "notes": "101 food classes, 101,000 images. Convert to vision JSONL with image paths.",
        "splits": {"train": 500},
    },
    "nutrition_profile_v1": {
        "repo": "Sai1289/nutritional_data",
        "notes": "Structured nutritional data for ~1000 foods.",
        "splits": {"train": 500},
    },
    "ingredient_extraction_v1": {
        "repo": "mbien/recipenlg",
        "notes": "Large recipe corpus. Extract (title, ingredients) pairs.",
        "splits": {"train": 300},
    },
}


def _fetch_hf(name: str, spec: dict, show_only: bool) -> None:
    print(f"\n  Fetching {name} from HF: {spec['repo']} …")
    if show_only:
        print(f"  [dry-run] would fetch {spec['splits']} examples")
        return
    try:
        from datasets import load_dataset
        ds = load_dataset(spec["repo"], trust_remote_code=True)
        split = ds.get("train") or next(iter(ds.values()))
        n = spec["splits"].get("train", 100)
        split = split.select(range(min(n, len(split))))
        out = PROC_DIR / f"{name}_hf.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with out.open("w") as fh:
            for ex in split:
                rec = dict(ex)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        print(f"  ✓ fetched {count} records → {out.relative_to(REPO_ROOT)}")
    except Exception as e:
        print(f"  ✗ failed to fetch {spec['repo']}: {e}")
        print(f"    Manual: uv run python -c \"from datasets import load_dataset; "
              f"load_dataset('{spec['repo']}')\"")


# ═══════════════════════════════════════════════════════════════════════════
# Guide printer
# ═══════════════════════════════════════════════════════════════════════════

TRAINING_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════╗
║         HOW TO SUPPLY THIS DATA TO TRAIN YOUR MODEL                    ║
╚══════════════════════════════════════════════════════════════════════════╝

━━━ STORAGE LAYOUT ─────────────────────────────────────────────────────────

  projects/food_health_coach/
  ├── data/
  │   ├── processed/              ← ready-to-train JSONL (one file per category)
  │   │   ├── food_recognition_v1.jsonl
  │   │   ├── ingredient_extraction_v1.jsonl
  │   │   ├── nutrition_profile_v1.jsonl
  │   │   ├── health_condition_diet_v1.jsonl
  │   │   ├── personalized_advice_v1.jsonl
  │   │   ├── portion_guidance_v1.jsonl
  │   │   ├── allergen_detection_v1.jsonl
  │   │   ├── meal_planning_v1.jsonl
  │   │   ├── food_substitution_v1.jsonl
  │   │   └── multimodal_coaching_v1.jsonl
  │   ├── raw/                    ← downloaded originals (HF cache, CSVs, etc.)
  │   └── images/                 ← food images for vision training
  │       ├── food_recognition/   ← {food_id}.jpg
  │       └── coaching/           ← {session_id}_{turn}.jpg
  └── configs/datasets/           ← one YAML per dataset, registered with EAT

━━━ RECORD FORMAT ──────────────────────────────────────────────────────────

  Text-only training record (all categories except food_recognition):
    {"messages": [
      {"role": "system",    "content": "You are a food health coach…"},
      {"role": "user",      "content": "I have T2D. Can I eat white rice?"},
      {"role": "assistant", "content": "Verdict: PORTION-LIMIT…"}
    ]}

  Vision training record (food_recognition_v1, multimodal_coaching_v1):
    {"messages": [
      {"role": "user", "content": [
        {"type": "image", "image": "projects/…/images/food_recognition/dal.jpg"},
        {"type": "text",  "text":  "What food is in this image?"}
      ]},
      {"role": "assistant", "content": "Dal Makhani (North Indian)…"}
    ]}

  NOTE: For Gemma 3n, the image field can be:
    • A local file path (resolved relative to repo root)
    • A base64-encoded string: "data:image/jpeg;base64,<base64_data>"
    • A URL (for HF datasets, resolved at training time)

━━━ WIRING INTO TRAINING ───────────────────────────────────────────────────

  Step 1 — Register each dataset in configs/datasets/<name>.yaml
           (already done by this script)

  Step 2 — Add datasets to projects/food_health_coach/project.yaml:
           datasets:
             - food_v1
             - food_vision_v1
             - ingredient_extraction_v1
             - nutrition_profile_v1
             - health_condition_diet_v1
             - personalized_advice_v1
             - portion_guidance_v1
             - allergen_detection_v1
             - meal_planning_v1
             - food_substitution_v1
             - multimodal_coaching_v1

  Step 3 — Run data preparation:
           uv run eat data build --project food_health_coach

  Step 4 — Train (text-only branch first, then vision):
           # Text-only QLoRA on Gemma 3n
           uv run eat run --project food_health_coach --recipe qlora_text \\
             --compute local_gpu

           # Multimodal QLoRA (includes vision pairs)
           uv run eat run --project food_health_coach --recipe qlora_vlm \\
             --compute local_gpu

  Step 5 — Evaluate:
           uv run eat eval quality <run_id>   # LLM-judge on eval set
           uv run eat eval safety  <run_id>   # red-team adversarial
           uv run eat eval perf    <run_id>   # tokens/sec benchmark

  Step 6 — Export for Android:
           uv run eat export gguf   <run_id> --quant Q4_K_M
           uv run eat export mediapipe <run_id>

━━━ GROWING THE DATASET ────────────────────────────────────────────────────

  Option A: Synthetic (best for personalized_advice_v1)
    uv run scripts/collect_food_data.py --type personalized_advice_v1
    # generates 200 profile × food × context combinations

  Option B: HuggingFace real data
    uv run scripts/collect_food_data.py --fetch-hf
    # pulls food101 (visual), recipenlg (ingredients), nutritional_data (macros)

  Option C: USDA FoodData Central API (nutrition_profile_v1)
    curl "https://api.nal.usda.gov/fdc/v1/foods/search?query=dal&api_key=DEMO_KEY"
    # Free, no registration needed for DEMO_KEY (rate-limited to 30 req/min)

  Option D: Teacher LLM (highest quality)
    export ANTHROPIC_API_KEY=<your_key>
    uv run eat data build --project food_health_coach --dataset personalized_advice_v1
    # uses claude-sonnet-4 to generate gold-quality responses

━━━ QUALITY CHECKLIST ──────────────────────────────────────────────────────

  Before training, verify:
  □ Each JSONL file loads cleanly:
      python -c "import json; [json.loads(l) for l in open('data/processed/X.jsonl')]"
  □ Vision records have valid image paths:
      find projects/food_health_coach/data/images -name '*.jpg' | wc -l
  □ No empty response fields
  □ Every allergen record includes the "consult your clinician" disclaimer
  □ Run the eval set through the base model to establish a baseline:
      uv run eat eval quality <base_run_id>
"""


def _print_guide() -> None:
    print(TRAINING_GUIDE)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-only", action="store_true",
                    help="Print guide and schema — do not write files")
    ap.add_argument("--fetch-hf", action="store_true",
                    help="Also pull real data from HuggingFace (requires 'datasets' package + internet)")
    ap.add_argument("--type",  metavar="CATEGORY",
                    help=f"Generate only one category. Choices: {', '.join(ALL_CATEGORIES)}")
    args = ap.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Food Health Coach — Data Research & Collection Tool   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("10 data categories for multimodal Gemma 3n (food identification,")
    print("ingredient extraction, personalised health recommendations).\n")

    if args.type and args.type not in ALL_CATEGORIES:
        sys.exit(f"Unknown category: {args.type}\nChoices: {', '.join(ALL_CATEGORIES)}")

    targets = {args.type: ALL_CATEGORIES[args.type]} if args.type else ALL_CATEGORIES

    total_records = 0
    for name, (fn, _kind) in targets.items():
        print(f"── {name} {'─' * max(1, 55 - len(name))}")
        result = fn()
        records, yaml_kind, yaml_notes, docstring = result

        # Print docstring (schema + sources)
        if docstring.strip():
            for line in docstring.strip().splitlines()[1:]:
                print("   " + line.lstrip())

        _save(name, records, show_only=args.show_only)
        _write_config(name, yaml_kind, yaml_notes, show_only=args.show_only)
        print(f"   examples in this file: {len(records)}")
        total_records += len(records)
        print()

    if args.fetch_hf:
        print("\n── HuggingFace real-data fetch " + "─" * 35)
        for name, spec in HF_SOURCES.items():
            if args.type and name != args.type:
                continue
            _fetch_hf(name, spec, show_only=args.show_only)

    print(f"\n{'─' * 60}")
    print(f"Total synthetic records generated: {total_records}")
    print(f"JSONL location: {PROC_DIR.relative_to(REPO_ROOT)}/")
    print(f"Dataset configs: {CONFIGS_DIR.relative_to(REPO_ROOT)}/")
    print()
    _print_guide()

    if not args.show_only:
        print("\nNext steps:")
        print("  1. Add new dataset names to projects/food_health_coach/project.yaml")
        print("  2. Add real food images to projects/food_health_coach/data/images/")
        print("  3. uv run eat run --project food_health_coach --recipe qlora_text --compute local_gpu")


if __name__ == "__main__":
    main()
