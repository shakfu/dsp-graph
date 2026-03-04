"""Tests for /api/optimize and /api/optimize/pass endpoints."""

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


class TestOptimizePass:
    """Tests for POST /api/optimize/pass."""

    def _foldable_graph(self) -> dict[str, Any]:
        """Graph with Constant(2) * Constant(3) -> foldable."""
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
        return g.model_dump()

    def _dead_node_graph(self) -> dict[str, Any]:
        """Graph with an unreachable node."""
        g = Graph(
            name="dead",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="pass_through")],
            params=[],
            nodes=[
                BinOp(id="pass_through", op="add", a="in1", b=0.0),
                Constant(id="unused", value=99.0),
            ],
        )
        return g.model_dump()

    def test_pass_constant_fold(self, client: TestClient) -> None:
        resp = client.post(
            "/api/optimize/pass",
            json={"graph": self._foldable_graph(), "pass_name": "constant_fold"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # constant_fold replaces BinOp(two*three) with Constant(6), keeping
        # the same node count.  Verify the fold happened by checking that
        # "six" is now op=constant instead of op=mul.
        opt_ops = {
            n["id"]: n["data"].get("op")
            for n in data["optimized"]["nodes"]
            if n["type"] == "dsp_node"
        }
        assert opt_ops["six"] == "constant"

    def test_pass_eliminate_dead(self, client: TestClient) -> None:
        resp = client.post(
            "/api/optimize/pass",
            json={"graph": self._dead_node_graph(), "pass_name": "eliminate_dead_nodes"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["nodes_removed"] > 0

    def test_pass_invalid_name(self, client: TestClient) -> None:
        resp = client.post(
            "/api/optimize/pass",
            json={"graph": self._foldable_graph(), "pass_name": "bogus"},
        )
        assert resp.status_code == 422

    def test_pass_noop(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        """constant_fold on a graph with no constants to fold should be a noop."""
        resp = client.post(
            "/api/optimize/pass",
            json={"graph": stereo_gain_json, "pass_name": "constant_fold"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["nodes_removed"] == 0
        assert data["stats"]["nodes_before"] == data["stats"]["nodes_after"]
