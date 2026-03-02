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

    def test_simulate_stereo_gain_with_ones_input(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        """Graphs with AudioInputs produce non-zero output when given explicit inputs."""
        resp = client.post(
            "/api/simulate",
            json={
                "graph": stereo_gain_json,
                "n_samples": 16,
                "inputs": {"in1": "ones", "in2": "ones"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "out1" in data["outputs"]
        assert "out2" in data["outputs"]
        # With gain=1.0 (default) and ones input, output should be all 1.0
        assert all(v == 1.0 for v in data["outputs"]["out1"])
        assert all(v == 1.0 for v in data["outputs"]["out2"])

    def test_simulate_stereo_gain_default_sine(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        """Graphs with AudioInputs auto-fill sine when no inputs specified."""
        resp = client.post(
            "/api/simulate",
            json={"graph": stereo_gain_json, "n_samples": 64},
        )
        assert resp.status_code == 200
        data = resp.json()
        values = data["outputs"]["out1"]
        # Default sine at 440 Hz: sustained non-zero output (not all zeros)
        assert any(abs(v) > 0.01 for v in values[1:]), "Expected sustained non-zero output"

    def test_simulate_invalid_graph(self, client: TestClient) -> None:
        # Send something that's not even a valid JSON for Graph
        resp = client.post(
            "/api/simulate",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent_op"}]}, "n_samples": 10},
        )
        assert resp.status_code == 422
