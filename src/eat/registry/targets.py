from __future__ import annotations

from eat.paths import configs_dir
from eat.registry.base import FolderRegistry
from eat.types import TargetEntry

_registry = FolderRegistry(configs_dir() / "targets", TargetEntry)


def get(name: str) -> TargetEntry:
    return _registry.get(name)


def all_entries() -> list[TargetEntry]:
    return _registry.all_entries()


def names() -> list[str]:
    return _registry.names()


def reload() -> None:
    _registry.reload()
