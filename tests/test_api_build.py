"""Tests for /api/build/* endpoints."""

from __future__ import annotations

import io
import zipfile
from typing import Any

from fastapi.testclient import TestClient


class TestBuild:
    def test_build_valid_graph_clap(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build",
            json={"graph": stereo_gain_json, "platform": "clap"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "clap"
        assert "dsp_cpp" in data
        assert "adapter_cpp" in data
        assert "manifest" in data
        assert len(data["dsp_cpp"]) > 0
        assert len(data["adapter_cpp"]) > 0
        assert len(data["manifest"]) > 0
        assert "clap" in data["supported_platforms"]

    def test_build_invalid_platform(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build",
            json={"graph": stereo_gain_json, "platform": "nonexistent"},
        )
        assert resp.status_code == 422
        assert "nonexistent" in resp.json()["detail"]

    def test_build_invalid_graph(self, client: TestClient) -> None:
        resp = client.post(
            "/api/build",
            json={
                "graph": {"name": "bad", "nodes": [{"op": "nonexistent_op"}]},
                "platform": "vst3",
            },
        )
        assert resp.status_code == 422


class TestBuildZip:
    def test_build_zip_valid(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        resp = client.post(
            "/api/build/zip",
            json={"graph": stereo_gain_json, "platform": "lv2"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        buf = io.BytesIO(resp.content)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
            assert any(n.endswith("_dsp.h") for n in names)
            assert any(n.endswith("_lv2.cpp") for n in names)
            assert "manifest.json" in names

    def test_build_zip_invalid_platform(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build/zip",
            json={"graph": stereo_gain_json, "platform": "bad_platform"},
        )
        assert resp.status_code == 422


class TestPlatforms:
    def test_platforms_list(self, client: TestClient) -> None:
        resp = client.get("/api/build/platforms")
        assert resp.status_code == 200
        data = resp.json()
        assert "platforms" in data
        assert "clap" in data["platforms"]
        assert "vst3" in data["platforms"]
        assert len(data["platforms"]) >= 11
