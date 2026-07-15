"""Tests for /api/genexpr endpoint and experimental-feature gating."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dsp_graph.config import EXPERIMENTAL_ENV


@pytest.fixture
def experimental_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable experimental features for the duration of a test."""
    monkeypatch.setenv(EXPERIMENTAL_ENV, "1")


@pytest.fixture
def experimental_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure experimental features are disabled (default)."""
    monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)


class TestGenExpr:
    """The transpiler endpoint, with experimental features enabled."""

    def test_genexpr_valid_graph(
        self, client: TestClient, stereo_gain_json: dict[str, Any], experimental_on: None
    ) -> None:
        resp = client.post("/api/genexpr", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert "genexpr_source" in data
        src = data["genexpr_source"]
        # gen~ codebox markers: Param declaration and the multiply body.
        assert "Param gain" in src
        assert "scaled1 = in1 * gain;" in src

    def test_genexpr_invalid_graph(self, client: TestClient, experimental_on: None) -> None:
        resp = client.post(
            "/api/genexpr",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent"}]}},
        )
        assert resp.status_code == 422

    def test_genexpr_unsupported_node(self, client: TestClient, experimental_on: None) -> None:
        """An approximation op (fastexp) has no gen~ equivalent -> 400."""
        graph = {
            "name": "unsupported",
            "inputs": [{"id": "in1"}],
            "outputs": [{"id": "out1", "source": "r"}],
            "nodes": [{"id": "r", "op": "fastexp", "a": "in1"}],
        }
        resp = client.post("/api/genexpr", json={"graph": graph})
        assert resp.status_code == 400
        assert "GenExpr" in resp.json()["detail"]


class TestExperimentalGating:
    """GenExpr is hidden behind the --experimental flag."""

    def test_genexpr_disabled_by_default(
        self, client: TestClient, stereo_gain_json: dict[str, Any], experimental_off: None
    ) -> None:
        """Without the flag the endpoint 404s, even for an otherwise-valid graph."""
        resp = client.post("/api/genexpr", json={"graph": stereo_gain_json})
        assert resp.status_code == 404
        assert "experimental" in resp.json()["detail"].lower()

    def test_config_reports_experimental_off(
        self, client: TestClient, experimental_off: None
    ) -> None:
        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["experimental"] is False

    def test_config_reports_experimental_on(
        self, client: TestClient, experimental_on: None
    ) -> None:
        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["experimental"] is True
