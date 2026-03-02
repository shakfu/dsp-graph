"""Tests for CLI entry point."""

from __future__ import annotations

import subprocess
import sys


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
