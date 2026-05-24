from __future__ import annotations

from eat.registry import datasets, models, projects, recipes, targets


def test_models_loaded():
    names = models.names()
    assert "gemma3n-e2b" in names
    assert "qwen2.5-0.5b" in names
    e = models.get("gemma3n-e2b")
    assert e.family == "gemma3n"
    assert "text" in e.multimodal


def test_datasets_loaded():
    names = datasets.names()
    assert "smoke_text" in names
    assert "food_eval_v1" in names


def test_recipes_loaded():
    r = recipes.get("qlora_text")
    assert r.method == "qlora"
    assert r.base_model in models.names()


def test_targets_loaded():
    t = targets.get("gguf_q4km")
    assert t.kind == "gguf"
    assert t.runtime == "llama.cpp"
    cm = targets.get("cloud_modal")
    assert (cm.platform or "").startswith("cloud:")


def test_food_project_present():
    p = projects.get("food_health_coach")
    assert p.base_model == "gemma3n-e2b"
    assert "qlora_text" in p.recipes
    assert "mediapipe_litert" in p.targets
