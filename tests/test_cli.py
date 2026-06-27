"""Tests for CLI entry point."""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestCli:
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dsp_graph", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "dsp-graph" in result.stdout

    def test_no_command_exits(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dsp_graph"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_serve_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dsp_graph", "serve", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--host" in result.stdout
        assert "--experimental" in result.stdout


class TestServeExperimentalFlag:
    """``serve --experimental`` plumbs the flag through the environment."""

    def test_experimental_flag_sets_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import uvicorn

        from dsp_graph.cli import main
        from dsp_graph.config import EXPERIMENTAL_ENV, is_experimental

        monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)
        ran: dict[str, bool] = {}
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: ran.setdefault("ran", True))

        main(["serve", "--experimental"])

        assert ran.get("ran") is True
        assert is_experimental() is True

    def test_default_leaves_experimental_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import uvicorn

        from dsp_graph.cli import main
        from dsp_graph.config import EXPERIMENTAL_ENV, is_experimental

        monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

        main(["serve"])

        assert is_experimental() is False
