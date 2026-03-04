"""Tests for /api/generate/* and /api/build/* endpoints."""

from __future__ import annotations

import io
import shutil
import zipfile
from typing import Any

import pytest
from fastapi.testclient import TestClient

_has_cmake = shutil.which("cmake") is not None


class TestGenerate:
    def test_generate_valid_graph_clap(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/generate",
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

    def test_generate_invalid_platform(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/generate",
            json={"graph": stereo_gain_json, "platform": "nonexistent"},
        )
        assert resp.status_code == 422
        assert "nonexistent" in resp.json()["detail"]

    def test_generate_invalid_graph(self, client: TestClient) -> None:
        resp = client.post(
            "/api/generate",
            json={
                "graph": {"name": "bad", "nodes": [{"op": "nonexistent_op"}]},
                "platform": "vst3",
            },
        )
        assert resp.status_code == 422


class TestGenerateZip:
    def test_generate_zip_valid(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/generate/zip",
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

    def test_generate_zip_invalid_platform(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/generate/zip",
            json={"graph": stereo_gain_json, "platform": "bad_platform"},
        )
        assert resp.status_code == 422


class TestPlatforms:
    def test_platforms_list(self, client: TestClient) -> None:
        resp = client.get("/api/generate/platforms")
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

        resp = client.get("/api/generate/platforms")
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


class TestBatchBuild:
    @pytest.mark.skipif(not _has_cmake, reason="cmake not available")
    def test_batch_build_multiple_platforms(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build/batch",
            json={"graph": stereo_gain_json, "platforms": ["clap", "vst3"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "batch_id" in data
        assert isinstance(data["batch_id"], str)
        assert len(data["batch_id"]) > 0
        assert "results" in data
        assert len(data["results"]) == 2
        platforms_returned = {r["platform"] for r in data["results"]}
        assert platforms_returned == {"clap", "vst3"}

    def test_batch_build_invalid_platform_included(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/build/batch",
            json={"graph": stereo_gain_json, "platforms": ["nonexistent"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "batch_id" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["success"] is False
        assert "nonexistent" in data["results"][0]["stderr"]

    def test_batch_build_invalid_graph(self, client: TestClient) -> None:
        resp = client.post(
            "/api/build/batch",
            json={
                "graph": {"name": "bad", "nodes": [{"op": "nonexistent_op"}]},
                "platforms": ["clap"],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.skipif(not _has_cmake, reason="cmake not available")
    def test_batch_build_download_zip(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        # First do a batch build
        resp = client.post(
            "/api/build/batch",
            json={"graph": stereo_gain_json, "platforms": ["clap", "vst3"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        batch_id = data["batch_id"]
        successful = [r for r in data["results"] if r["success"]]
        assert len(successful) > 0, "Need at least one successful build for zip test"

        # Download the zip
        zip_resp = client.get(f"/api/build/batch/{batch_id}/zip")
        assert zip_resp.status_code == 200
        assert zip_resp.headers["content-type"] == "application/zip"

        buf = io.BytesIO(zip_resp.content)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
            for r in successful:
                plat = r["platform"]
                matching = [n for n in names if n.startswith(f"{plat}/")]
                assert len(matching) > 0, f"Expected binary for {plat} in zip"

        # Second download should 404 (cache entry consumed)
        zip_resp2 = client.get(f"/api/build/batch/{batch_id}/zip")
        assert zip_resp2.status_code == 404

    def test_batch_build_zip_missing_batch_id(self, client: TestClient) -> None:
        resp = client.get("/api/build/batch/nonexistent_id_12345/zip")
        assert resp.status_code == 404


class TestBuildCacheEndpoints:
    def test_cache_info_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/build/cache/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "total_bytes" in data
        assert "cache_dir" in data
        assert isinstance(data["entries"], int)
        assert isinstance(data["total_bytes"], int)

    def test_cache_clear_endpoint(self, client: TestClient) -> None:
        resp = client.delete("/api/build/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert "removed" in data
        assert isinstance(data["removed"], int)

    @pytest.mark.skipif(not _has_cmake, reason="cmake not available")
    def test_build_cache_hit(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        """Building same graph+platform twice should return cached on second call."""
        # Clear cache first
        client.delete("/api/build/cache")

        resp1 = client.post(
            "/api/build",
            json={"graph": stereo_gain_json, "platform": "clap"},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["success"] is True
        assert data1["stdout"] != "(cached)"

        resp2 = client.post(
            "/api/build",
            json={"graph": stereo_gain_json, "platform": "clap"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["success"] is True
        assert data2["stdout"] == "(cached)"

    @pytest.mark.skipif(not _has_cmake, reason="cmake not available")
    def test_batch_build_uses_cache(
        self, client: TestClient, stereo_gain_json: dict[str, Any]
    ) -> None:
        """Batch build same graph twice -- second run should use cache."""
        client.delete("/api/build/cache")

        resp1 = client.post(
            "/api/build/batch",
            json={"graph": stereo_gain_json, "platforms": ["clap"]},
        )
        assert resp1.status_code == 200
        results1 = resp1.json()["results"]
        successful1 = [r for r in results1 if r["success"]]
        assert len(successful1) > 0
        assert successful1[0]["stdout"] != "(cached)"

        resp2 = client.post(
            "/api/build/batch",
            json={"graph": stereo_gain_json, "platforms": ["clap"]},
        )
        assert resp2.status_code == 200
        results2 = resp2.json()["results"]
        successful2 = [r for r in results2 if r["success"]]
        assert len(successful2) > 0
        assert successful2[0]["stdout"] == "(cached)"


class TestBuild:
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
                "platform": "clap",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.skipif(not _has_cmake, reason="cmake not available")
    def test_build_valid_graph(self, client: TestClient, stereo_gain_json: dict[str, Any]) -> None:
        resp = client.post(
            "/api/build",
            json={"graph": stereo_gain_json, "platform": "clap"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "clap"
        assert isinstance(data["success"], bool)
        assert isinstance(data["stdout"], str)
        assert isinstance(data["stderr"], str)
