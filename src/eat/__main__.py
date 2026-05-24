"""Allow `python -m eat ...` as an alias for the `eat` console script."""

from eat.cli import app

if __name__ == "__main__":
    app()
