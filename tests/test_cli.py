"""Tests for the dsp-graph CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsp_graph.cli import main


@pytest.fixture
def graph_json(tmp_path: Path) -> Path:
    """Write a minimal valid graph JSON and return its path."""
    data = {
        "name": "test_graph",
        "inputs": [{"id": "in1"}],
        "outputs": [{"id": "out1", "source": "scaled"}],
        "params": [{"name": "gain", "min": 0.0, "max": 2.0, "default": 1.0}],
        "nodes": [{"id": "scaled", "op": "mul", "a": "in1", "b": "gain"}],
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def invalid_graph_json(tmp_path: Path) -> Path:
    """Write a graph JSON with validation errors and return its path."""
    data = {
        "name": "bad",
        "nodes": [{"id": "a", "op": "add", "a": "missing", "b": 0.0}],
        "outputs": [{"id": "out1", "source": "a"}],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    return p


class TestCompile:
    def test_compile_to_stdout(self, graph_json: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["compile", str(graph_json)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "TestGraphState" in out
        assert "test_graph_create" in out

    def test_compile_to_dir(self, graph_json: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "build"
        rc = main(["compile", str(graph_json), "-o", str(out_dir)])
        assert rc == 0
        cpp = out_dir / "test_graph.cpp"
        assert cpp.exists()
        assert "TestGraphState" in cpp.read_text()

    def test_compile_with_optimize(
        self, graph_json: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["compile", str(graph_json), "--optimize"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "test_graph_perform" in out

    def test_compile_gen_dsp(self, graph_json: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "gen_dsp_build"
        rc = main(["compile", str(graph_json), "--gen-dsp", "chuck", "-o", str(out_dir)])
        assert rc == 0
        assert (out_dir / "test_graph.cpp").exists()
        assert (out_dir / "_ext_chuck.cpp").exists()
        assert (out_dir / "manifest.json").exists()

    def test_compile_gen_dsp_requires_output(
        self,
        graph_json: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["compile", str(graph_json), "--gen-dsp", "chuck"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "--gen-dsp requires -o" in err


class TestValidate:
    def test_validate_valid(self, graph_json: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["validate", str(graph_json)])
        assert rc == 0
        assert "valid" in capsys.readouterr().out

    def test_validate_invalid_exits_1(
        self, invalid_graph_json: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["validate", str(invalid_graph_json)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "error:" in err


class TestDot:
    def test_dot_to_stdout(self, graph_json: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["dot", str(graph_json)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "digraph" in out

    def test_dot_to_dir(self, graph_json: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "dot_out"
        rc = main(["dot", str(graph_json), "-o", str(out_dir)])
        assert rc == 0
        dot_file = out_dir / "test_graph.dot"
        assert dot_file.exists()
        assert "digraph" in dot_file.read_text()


class TestErrorHandling:
    def test_no_command_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([])
        assert rc == 1

    def test_missing_file_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["compile", "/nonexistent/graph.json"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "error:" in err

    def test_malformed_json_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        rc = main(["compile", str(bad)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "invalid JSON" in err

    def test_invalid_schema_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad_schema.json"
        bad.write_text(json.dumps({"name": 123}))
        rc = main(["compile", str(bad)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "error:" in err
