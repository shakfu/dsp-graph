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

    def test_validate_invalid_parse(self, client: TestClient) -> None:
        resp = client.post(
            "/api/graph/validate",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent"}]}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
        err = data["errors"][0]
        assert "message" in err
        assert "kind" in err
        assert "severity" in err

    def test_validate_structural_error_returns_structured_details(
        self, client: TestClient
    ) -> None:
        """A graph with dangling output source returns structured error details."""
        graph = {
            "name": "dangling",
            "inputs": [],
            "outputs": [{"id": "out1", "source": "missing_node"}],
            "params": [],
            "nodes": [],
        }
        resp = client.post("/api/graph/validate", json={"graph": graph})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) >= 1
        err = data["errors"][0]
        assert err["kind"] == "bad_output_source"
        assert err["node_id"] == "out1"
        assert err["field_name"] == "source"
        assert err["severity"] == "error"
        assert "missing_node" in err["message"]


class TestExport:
    def test_export_roundtrip(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
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

    def test_export_gdsp_roundtrip(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        """Load JSON -> ReactFlow -> export gdsp -> reload -> verify node count."""
        # Load to get ReactFlow
        load_resp = client.post("/api/graph/load/json", json={"graph": stereo_gain_json})
        assert load_resp.status_code == 200
        rf = load_resp.json()
        # Export to .gdsp source
        gdsp_resp = client.post("/api/graph/export/gdsp", json=rf)
        assert gdsp_resp.status_code == 200
        source = gdsp_resp.json()["source"]
        assert "graph stereo_gain" in source
        # Re-parse the .gdsp to get ReactFlow again
        reload_resp = client.post("/api/graph/load/gdsp", json={"source": source})
        assert reload_resp.status_code == 200
        reloaded = reload_resp.json()
        # Original had 7 RF nodes (2 in + 2 out + 1 param + 2 mul)
        # Re-parsed should have same structure
        assert len(reloaded["nodes"]) == len(rf["nodes"])


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


class TestLoadGdspMultiGraph:
    SOURCE = (
        "graph gain_a {\n  in x\n  param g 0..2 = 0.5\n  y = x * g\n  out out1 = y\n}\n"
        "graph gain_b {\n  in x\n  param g 0..2 = 1.5\n  y = x * g\n  out out1 = y\n}\n"
    )

    def test_returns_all_graph_names(self, client: TestClient) -> None:
        resp = client.post("/api/graph/load/gdsp", json={"source": self.SOURCE})
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_names"] == ["gain_a", "gain_b"]

    def test_defaults_to_last_graph(self, client: TestClient) -> None:
        resp = client.post("/api/graph/load/gdsp", json={"source": self.SOURCE})
        assert resp.json()["name"] == "gain_b"

    def test_selects_requested_graph(self, client: TestClient) -> None:
        resp = client.post(
            "/api/graph/load/gdsp",
            json={"source": self.SOURCE, "graph_name": "gain_a"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "gain_a"

    def test_unknown_graph_name_falls_back_to_last(self, client: TestClient) -> None:
        resp = client.post(
            "/api/graph/load/gdsp",
            json={"source": self.SOURCE, "graph_name": "does_not_exist"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "gain_b"

    def test_single_graph_lists_one_name(self, client: TestClient) -> None:
        source = "graph only {\n  in x\n  out out1 = x\n}"
        resp = client.post("/api/graph/load/gdsp", json={"source": source})
        assert resp.json()["graph_names"] == ["only"]


class TestDot:
    def test_dot_output(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        resp = client.post("/api/graph/dot", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert "digraph" in data["dot"]
        assert "stereo_gain" in data["dot"]
