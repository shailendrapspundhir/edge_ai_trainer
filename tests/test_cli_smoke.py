"""Sanity-check that the CLI imports and exposes top-level commands."""

from __future__ import annotations

from typer.testing import CliRunner

from eat.cli import app


def test_version_runs():
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "edge-ai-trainer" in result.stdout


def test_registry_list_models():
    runner = CliRunner()
    result = runner.invoke(app, ["registry", "list", "models"])
    assert result.exit_code == 0
    assert "gemma3n-e2b" in result.stdout


def test_project_list():
    runner = CliRunner()
    result = runner.invoke(app, ["project", "list"])
    assert result.exit_code == 0
    assert "food_health_coach" in result.stdout
