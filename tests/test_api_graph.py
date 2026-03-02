"""Tests for /api/graph/* endpoints."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


class TestLoadJson:
    def test_load_valid_graph(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        resp = client.post("/api/graph/load/json", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 7

    def test_load_invalid_graph(self, client: TestClient) -> None:
        resp = client.post(
            "/api/graph/load/json",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent"}]}},
        )
        assert resp.status_code == 422


class TestValidate:
    def test_validate_valid(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        resp = client.post("/api/graph/validate", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_invalid(self, client: TestClient) -> None:
        resp = client.post(
            "/api/graph/validate",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent"}]}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0


class TestExport:
    def test_export_roundtrip(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        # Load to get ReactFlow
        load_resp = client.post("/api/graph/load/json", json={"graph": stereo_gain_json})
        rf = load_resp.json()
        # Export back to Graph JSON
        export_resp = client.post("/api/graph/export/json", json=rf)
        assert export_resp.status_code == 200
        exported = export_resp.json()
        assert exported["name"] == "stereo_gain"
        assert len(exported["nodes"]) == 2
        assert len(exported["inputs"]) == 2
        assert len(exported["outputs"]) == 2


class TestNodeTypes:
    def test_node_types_catalog(self, client: TestClient) -> None:
        resp = client.get("/api/graph/node-types")
        assert resp.status_code == 200
        data = resp.json()
        assert "colors" in data
        assert "mul" in data["colors"]


class TestDot:
    def test_dot_output(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        resp = client.post("/api/graph/dot", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert "digraph" in data["dot"]
        assert "stereo_gain" in data["dot"]
