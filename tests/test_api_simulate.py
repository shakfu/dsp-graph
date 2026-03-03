"""Tests for /api/simulate* endpoints."""

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
        assert "session_id" in data

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


class TestStatefulSimulation:
    def test_simulate_returns_session_id(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/simulate",
            json={"graph": phasor_json, "n_samples": 16},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_continue_produces_different_output(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        """Continuing a session produces different output than the initial run."""
        # Initial simulation
        resp1 = client.post(
            "/api/simulate",
            json={"graph": phasor_json, "n_samples": 16},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        session_id = data1["session_id"]
        out1 = data1["outputs"]["out1"]

        # Continue from same state
        resp2 = client.post(
            "/api/simulate/continue",
            json={"session_id": session_id, "n_samples": 16},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        out2 = data2["outputs"]["out1"]

        # Phasor is stateful: continued output should differ from initial
        assert out1 != out2

    def test_set_param_changes_behavior(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        # Create session
        resp = client.post(
            "/api/simulate",
            json={"graph": phasor_json, "n_samples": 16},
        )
        session_id = resp.json()["session_id"]

        # Set frequency to a very different value
        param_resp = client.post(
            "/api/simulate/param",
            json={"session_id": session_id, "name": "freq", "value": 10000.0},
        )
        assert param_resp.status_code == 200

        # Continue and verify the output changed
        resp_after = client.post(
            "/api/simulate/continue",
            json={"session_id": session_id, "n_samples": 16},
        )
        assert resp_after.status_code == 200

    def test_reset_returns_to_initial_state(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        # Run initial simulation
        resp1 = client.post(
            "/api/simulate",
            json={"graph": phasor_json, "n_samples": 16},
        )
        session_id = resp1.json()["session_id"]
        initial_out = resp1.json()["outputs"]["out1"]

        # Continue to advance state
        client.post(
            "/api/simulate/continue",
            json={"session_id": session_id, "n_samples": 64},
        )

        # Reset
        reset_resp = client.post(
            "/api/simulate/reset",
            json={"session_id": session_id},
        )
        assert reset_resp.status_code == 200

        # Continue again -- should match initial output
        resp_after = client.post(
            "/api/simulate/continue",
            json={"session_id": session_id, "n_samples": 16},
        )
        after_out = resp_after.json()["outputs"]["out1"]
        assert initial_out == after_out

    def test_invalid_session_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/simulate/continue",
            json={"session_id": "nonexistent", "n_samples": 16},
        )
        assert resp.status_code == 404

    def test_set_param_invalid_name(self, client: TestClient, phasor_json: dict[str, Any]) -> None:
        resp = client.post(
            "/api/simulate",
            json={"graph": phasor_json, "n_samples": 8},
        )
        session_id = resp.json()["session_id"]

        param_resp = client.post(
            "/api/simulate/param",
            json={"session_id": session_id, "name": "nonexistent_param", "value": 1.0},
        )
        assert param_resp.status_code == 404
