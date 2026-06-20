"""Tests for the /api/layout endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
from gen_dsp.graph.models import Graph

from dsp_graph.convert import LAYER_X_SPACING, graph_to_reactflow
from dsp_graph.server import app


class TestLayout:
    def test_layout_repositions_by_type(self, client: TestClient, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain).model_dump()
        # Scramble positions so we can tell the endpoint recomputed them.
        for n in rf["nodes"]:
            n["position"] = {"x": 999.0, "y": 999.0}

        resp = client.post("/api/layout", json=rf)
        assert resp.status_code == 200
        nodes = {n["id"]: n for n in resp.json()["nodes"]}

        # Inputs and params anchored at x=0.
        for n in nodes.values():
            if n["type"] in ("input", "param"):
                assert n["position"]["x"] == 0

        # dsp nodes placed in middle layers (x > 0).
        dsp = [n for n in nodes.values() if n["type"] == "dsp_node"]
        assert dsp and all(n["position"]["x"] > 0 for n in dsp)

        # Outputs at the rightmost layer.
        outs = [n for n in nodes.values() if n["type"] == "output"]
        max_layer = len(dsp) + 1
        assert outs and all(n["position"]["x"] == max_layer * LAYER_X_SPACING for n in outs)

    def test_layout_preserves_node_count(self, client: TestClient, onepole: Graph) -> None:
        rf = graph_to_reactflow(onepole).model_dump()
        resp = client.post("/api/layout", json=rf)
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) == len(rf["nodes"])

    def test_layout_requires_token(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain).model_dump()
        resp = TestClient(app).post("/api/layout", json=rf)  # no token header
        assert resp.status_code == 403
