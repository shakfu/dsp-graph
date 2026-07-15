"""Tests for CLI entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_flag_env() -> Iterator[None]:
    """Snapshot/restore the feature-flag env vars around each test.

    ``main()`` mutates ``os.environ`` directly, and ``monkeypatch.delenv`` on an
    absent key registers no undo, so those direct assignments would otherwise
    leak into later tests (e.g. re-enabling build gating elsewhere).
    """
    from dsp_graph.config import DISABLE_BUILD_ENV, EXPERIMENTAL_ENV

    keys = (DISABLE_BUILD_ENV, EXPERIMENTAL_ENV)
    saved = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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


class TestServeDisableBuildFlag:
    """``serve --disable-build`` plumbs the flag through the environment."""

    def test_disable_build_flag_sets_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import uvicorn

        from dsp_graph.cli import main
        from dsp_graph.config import DISABLE_BUILD_ENV, is_build_enabled

        monkeypatch.delenv(DISABLE_BUILD_ENV, raising=False)
        ran: dict[str, bool] = {}
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: ran.setdefault("ran", True))

        main(["serve", "--disable-build"])

        assert ran.get("ran") is True
        assert is_build_enabled() is False

    def test_default_leaves_build_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import uvicorn

        from dsp_graph.cli import main
        from dsp_graph.config import DISABLE_BUILD_ENV, is_build_enabled

        monkeypatch.delenv(DISABLE_BUILD_ENV, raising=False)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

        main(["serve"])

        assert is_build_enabled() is True
