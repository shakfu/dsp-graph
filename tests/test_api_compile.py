"""Tests for /api/compile endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


class TestCompile:
    def test_compile_valid_graph(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post("/api/compile", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert "cpp_source" in data
        cpp = data["cpp_source"]
        # Should contain C++ markers
        assert "void" in cpp or "float" in cpp or "#include" in cpp

    def test_compile_invalid_graph(self, client: TestClient) -> None:
        resp = client.post(
            "/api/compile",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent"}]}},
        )
        assert resp.status_code == 422
