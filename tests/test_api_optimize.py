"""Tests for /api/optimize endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from gen_dsp.graph.models import (
    AudioInput,
    AudioOutput,
    BinOp,
    Constant,
    Graph,
)


class TestOptimize:
    def test_optimize_constant_fold(self, client: TestClient) -> None:
        """Graph with constant arithmetic should be folded."""
        g = Graph(
            name="foldable",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="scaled")],
            params=[],
            nodes=[
                Constant(id="two", value=2.0),
                Constant(id="three", value=3.0),
                BinOp(id="six", op="mul", a="two", b="three"),
                BinOp(id="scaled", op="mul", a="in1", b="six"),
            ],
        )
        resp = client.post("/api/optimize", json={"graph": g.model_dump()})
        assert resp.status_code == 200
        data = resp.json()
        assert "original" in data
        assert "optimized" in data
        assert "stats" in data
        # Optimized should have fewer nodes
        orig_nodes = [n for n in data["original"]["nodes"] if n["type"] == "dsp_node"]
        opt_nodes = [n for n in data["optimized"]["nodes"] if n["type"] == "dsp_node"]
        assert len(opt_nodes) <= len(orig_nodes)

    def test_optimize_preserves_structure(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post("/api/optimize", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert data["optimized"]["name"] == "stereo_gain"

    def test_optimize_stats_fields(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post("/api/optimize", json={"graph": stereo_gain_json})
        data = resp.json()
        stats = data["stats"]
        assert "constants_folded" in stats
        assert "cse_merges" in stats
        assert "dead_nodes_removed" in stats
