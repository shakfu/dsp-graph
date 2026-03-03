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
        # Catalog should be present
        assert "catalog" in data
        catalog = data["catalog"]
        # Phasor should have freq field
        assert "phasor" in catalog
        assert "freq" in catalog["phasor"]["fields"]
        assert catalog["phasor"]["fields"]["freq"]["required"] is True
        # BinOp ops like add should have a and b
        assert "add" in catalog
        assert "a" in catalog["add"]["fields"]
        assert "b" in catalog["add"]["fields"]
        # Each entry should have class, fields, color
        for op, info in catalog.items():
            assert "class" in info
            assert "fields" in info
            assert "color" in info


class TestLoadGdsp:
    def test_load_valid_gdsp(self, client: TestClient) -> None:
        source = (
            "graph simple {\n"
            "  param freq 1.0 .. 20000.0 = 440.0\n"
            "  ph = phasor(freq)\n"
            "  out out1 = ph\n"
            "}"
        )
        resp = client.post("/api/graph/load/gdsp", json={"source": source})
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0

    def test_load_invalid_gdsp_structured_error(self, client: TestClient) -> None:
        source = "graph bad {\n  broken syntax here\n}"
        resp = client.post("/api/graph/load/gdsp", json={"source": source})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "message" in detail
        assert "line" in detail
        assert "col" in detail
        assert isinstance(detail["line"], int)
        assert isinstance(detail["col"], int)


class TestDot:
    def test_dot_output(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        resp = client.post("/api/graph/dot", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert "digraph" in data["dot"]
        assert "stereo_gain" in data["dot"]
