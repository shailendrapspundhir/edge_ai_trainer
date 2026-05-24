"""Edge AI Trainer (eat) — local-first workbench for training and deploying edge AI models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("edge-ai-trainer")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
