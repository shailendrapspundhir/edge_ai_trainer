"""Generic YAML-folder registry — typed via a pydantic model."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Generic, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_ENV_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(obj):
    """Recursively expand ${VAR:-default} strings inside a parsed YAML object."""
    if isinstance(obj, str):
        return _ENV_VAR.sub(
            lambda m: os.environ.get(m.group(1), m.group(2) or ""),
            obj,
        )
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


class FolderRegistry(Generic[T]):
    """A registry backed by `*.yaml` files in a single directory.

    Each YAML file becomes one typed entry; the key is `entry.name`. Reads
    are cached after the first scan; call `reload()` to pick up changes.
    """

    def __init__(self, folder: Path, model_cls: type[T]) -> None:
        self.folder = folder
        self.model_cls = model_cls
        self._cache: dict[str, T] | None = None

    def reload(self) -> None:
        self._cache = None

    def _load(self) -> dict[str, T]:
        if self._cache is not None:
            return self._cache
        entries: dict[str, T] = {}
        if self.folder.exists():
            for path in sorted(self.folder.glob("*.yaml")):
                with path.open("r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}
                raw = _expand_env(raw)
                entry = self.model_cls.model_validate(raw)
                name = getattr(entry, "name", path.stem)
                entries[name] = entry
        self._cache = entries
        return entries

    def get(self, name: str) -> T:
        items = self._load()
        if name not in items:
            raise KeyError(f"{self.model_cls.__name__} not registered: {name!r}. "
                           f"Known: {sorted(items)}")
        return items[name]

    def all_entries(self) -> list[T]:
        return list(self._load().values())

    def names(self) -> list[str]:
        return list(self._load().keys())
