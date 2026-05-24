"""Project registry — each project is a folder with a `project.yaml`."""

from __future__ import annotations

from pathlib import Path

import yaml

from eat.paths import projects_dir
from eat.types import ProjectSpec


def _path(name: str) -> Path:
    return projects_dir() / name / "project.yaml"


def get(name: str) -> ProjectSpec:
    p = _path(name)
    if not p.exists():
        raise KeyError(f"project not found: {name!r} (expected {p})")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return ProjectSpec.model_validate(raw)


def all_specs() -> list[ProjectSpec]:
    out: list[ProjectSpec] = []
    if not projects_dir().exists():
        return out
    for child in sorted(projects_dir().iterdir()):
        if (child / "project.yaml").exists():
            try:
                out.append(get(child.name))
            except Exception:  # noqa: BLE001
                continue
    return out


def names() -> list[str]:
    return [p.name for p in all_specs()]


_DEFAULT_YAML = """\
name: {name}
description: ""
base_model: gemma3n-e2b
datasets: []
recipes: [qlora_text]
targets: [gguf_q4km, mediapipe_litert]
eval_set: null
system_prompt_file: prompts/system.txt
judge_prompt_file: prompts/eval_judge.txt
redteam_prompt_file: prompts/red_team.txt
notes: ""
"""


def scaffold(name: str) -> Path:
    """Create a new projects/<name>/ folder with empty placeholders."""
    root = projects_dir() / name
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "prompts", "eval"):
        (root / sub).mkdir(exist_ok=True)
        (root / sub / ".gitkeep").touch()
    yaml_path = root / "project.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(_DEFAULT_YAML.format(name=name), encoding="utf-8")
    sys_prompt = root / "prompts" / "system.txt"
    if not sys_prompt.exists():
        sys_prompt.write_text("You are a helpful assistant.\n", encoding="utf-8")
    return root
