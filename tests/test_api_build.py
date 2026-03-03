"""Tests for /api/build/* endpoints."""

from __future__ import annotations

import io
import shutil
import zipfile
from typing import Any

import pytest
from fastapi.testclient import TestClient

_has_cmake = shutil.which("cmake") is not None


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
        # clap and vst3 are available on all OSes
        assert "clap" in data["platforms"]
        assert "vst3" in data["platforms"]
        # Filtered by host OS: macOS=9, Linux=11, Windows=3
        assert len(data["platforms"]) >= 3

    def test_platforms_os_filtered(self, client: TestClient) -> None:
        """Platforms list should exclude platforms unavailable on the host OS."""
        import platform as plat

        resp = client.get("/api/build/platforms")
        data = resp.json()
        platforms = data["platforms"]
        if plat.system() == "Darwin":
            # macOS-only: au, max present; linux-only: daisy, circle absent
            assert "au" in platforms
            assert "max" in platforms
            assert "daisy" not in platforms
            assert "circle" not in platforms
        elif plat.system() == "Linux":
            # linux-only: daisy, circle present; macOS-only: au, max absent
            assert "daisy" in platforms
            assert "circle" in platforms
            assert "au" not in platforms
            assert "max" not in platforms


class TestCompileBuild:
    def test_compile_build_invalid_platform(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build/compile",
            json={"graph": stereo_gain_json, "platform": "nonexistent"},
        )
        assert resp.status_code == 422
        assert "nonexistent" in resp.json()["detail"]

    def test_compile_build_invalid_graph(self, client: TestClient) -> None:
        resp = client.post(
            "/api/build/compile",
            json={
                "graph": {"name": "bad", "nodes": [{"op": "nonexistent_op"}]},
                "platform": "clap",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.skipif(not _has_cmake, reason="cmake not available")
    def test_compile_build_valid_graph(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build/compile",
            json={"graph": stereo_gain_json, "platform": "clap"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "clap"
        assert isinstance(data["success"], bool)
        assert isinstance(data["stdout"], str)
        assert isinstance(data["stderr"], str)
