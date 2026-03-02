"""Tests for /api/simulate endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


class TestSimulate:
    def test_simulate_phasor(self, client: TestClient, phasor_json: dict[str, Any]) -> None:
        resp = client.post(
            "/api/simulate",
            json={"graph": phasor_json, "n_samples": 32, "sample_rate": 44100},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "outputs" in data
        assert "out1" in data["outputs"]
        assert len(data["outputs"]["out1"]) == 32

    def test_simulate_with_param_override(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/simulate",
            json={
                "graph": phasor_json,
                "n_samples": 16,
                "params": {"freq": 1000.0},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["outputs"]["out1"]) == 16

    def test_simulate_output_values_in_range(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/simulate",
            json={"graph": phasor_json, "n_samples": 64},
        )
        assert resp.status_code == 200
        data = resp.json()
        values = data["outputs"]["out1"]
        # Phasor output should be in [0, 1)
        assert all(0.0 <= v < 1.0 for v in values)

    def test_simulate_invalid_graph(self, client: TestClient) -> None:
        # Send something that's not even a valid JSON for Graph
        resp = client.post(
            "/api/simulate",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent_op"}]}, "n_samples": 10},
        )
        assert resp.status_code == 422
