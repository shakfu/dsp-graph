"""Tests for security hardening: session token, path containment, input bounds."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dsp_graph.security import (
    SESSION_HEADER,
    SESSION_TOKEN,
    BodySizeLimitMiddleware,
)
from dsp_graph.server import app


class TestSessionToken:
    def test_session_endpoint_returns_token(self) -> None:
        client = TestClient(app)
        resp = client.get("/api/session")
        assert resp.status_code == 200
        assert resp.json() == {"token": SESSION_TOKEN}

    def test_post_without_token_rejected(self, stereo_gain_json: dict[str, Any]) -> None:
        client = TestClient(app)  # no default token header
        resp = client.post("/api/graph/validate", json={"graph": stereo_gain_json})
        assert resp.status_code == 403
        assert "token" in resp.json()["detail"].lower()

    def test_post_with_invalid_token_rejected(self, stereo_gain_json: dict[str, Any]) -> None:
        client = TestClient(app)
        resp = client.post(
            "/api/graph/validate",
            json={"graph": stereo_gain_json},
            headers={SESSION_HEADER: "not-the-token"},
        )
        assert resp.status_code == 403

    def test_post_with_valid_token_allowed(self, stereo_gain_json: dict[str, Any]) -> None:
        client = TestClient(app)
        resp = client.post(
            "/api/graph/validate",
            json={"graph": stereo_gain_json},
            headers={SESSION_HEADER: SESSION_TOKEN},
        )
        assert resp.status_code == 200

    def test_safe_get_allowed_without_token(self) -> None:
        client = TestClient(app)  # no token header
        resp = client.get("/api/graph/node-types")
        assert resp.status_code == 200

    def test_build_endpoint_protected_without_token(
        self, stereo_gain_json: dict[str, Any]
    ) -> None:
        """The high-risk native-build endpoint must reject untokened POSTs."""
        client = TestClient(app)
        resp = client.post("/api/build", json={"graph": stereo_gain_json, "platform": "max"})
        assert resp.status_code == 403


class TestStaticPathContainment:
    def test_legitimate_static_file_served(self) -> None:
        client = TestClient(app)
        resp = client.get("/index.html")
        assert resp.status_code == 200
        assert "<" in resp.text  # served HTML, not an error

    def test_traversal_does_not_escape_static_dir(self) -> None:
        """Encoded ``..`` must not reach files outside STATIC_DIR (e.g. server.py)."""
        client = TestClient(app)
        resp = client.get("/%2e%2e/server.py")
        # Either rejected outright or falls back to the SPA index; never the
        # actual Python source of the server module.
        assert "SessionTokenMiddleware" not in resp.text
        assert "def _spa_fallback" not in resp.text


class TestBodySizeLimit:
    def _app(self, max_bytes: int) -> FastAPI:
        mini = FastAPI()
        mini.add_middleware(BodySizeLimitMiddleware, max_bytes=max_bytes)

        @mini.post("/echo")
        async def echo(payload: dict[str, Any]) -> dict[str, Any]:
            return payload

        return mini

    def test_oversized_body_rejected(self) -> None:
        client = TestClient(self._app(max_bytes=10))
        resp = client.post("/echo", json={"key": "x" * 1000})
        assert resp.status_code == 413

    def test_small_body_allowed(self) -> None:
        client = TestClient(self._app(max_bytes=10_000))
        resp = client.post("/echo", json={"k": "v"})
        assert resp.status_code == 200


class TestInputBounds:
    def test_n_samples_over_limit_rejected(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        resp = client.post("/api/simulate", json={"graph": phasor_json, "n_samples": 10**10})
        assert resp.status_code == 422

    def test_n_samples_zero_rejected(
        self, client: TestClient, phasor_json: dict[str, Any]
    ) -> None:
        resp = client.post("/api/simulate", json={"graph": phasor_json, "n_samples": 0})
        assert resp.status_code == 422

    def test_batch_platforms_over_limit_rejected(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build/batch",
            json={"graph": stereo_gain_json, "platforms": ["max"] * 100},
        )
        assert resp.status_code == 422
