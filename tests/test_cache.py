"""Unit tests for BuildCache (no cmake required)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from gen_dsp.graph.models import AudioInput, AudioOutput, BinOp, Graph, Param

from dsp_graph.cache import BuildCache, _default_cache_root, cache_key


@pytest.fixture
def simple_graph() -> Graph:
    return Graph(
        name="test",
        inputs=[AudioInput(id="in1")],
        outputs=[AudioOutput(id="out1", source="scaled")],
        params=[Param(name="gain", min=0.0, max=2.0, default=1.0)],
        nodes=[BinOp(id="scaled", op="mul", a="in1", b="gain")],
    )


@pytest.fixture
def build_cache(tmp_path: Path) -> BuildCache:
    return BuildCache(root=tmp_path, max_age=3600)


class TestCacheKey:
    def test_deterministic(self, simple_graph: Graph) -> None:
        """Same graph+platform always produces the same key."""
        k1 = cache_key(simple_graph, "clap")
        k2 = cache_key(simple_graph, "clap")
        assert k1 == k2
        assert len(k1) == 64  # sha256 hex

    def test_different_platform_different_key(self, simple_graph: Graph) -> None:
        k1 = cache_key(simple_graph, "clap")
        k2 = cache_key(simple_graph, "vst3")
        assert k1 != k2

    def test_different_graph_different_key(self) -> None:
        g1 = Graph(
            name="a",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="in1")],
        )
        g2 = Graph(
            name="b",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="in1")],
        )
        assert cache_key(g1, "clap") != cache_key(g2, "clap")

    def test_includes_version(self, simple_graph: Graph) -> None:
        """Different gen_dsp versions produce different keys."""
        k1 = cache_key(simple_graph, "clap")
        with patch("dsp_graph.cache.pkg_version", return_value="99.99.99"):
            k2 = cache_key(simple_graph, "clap")
        assert k1 != k2


class TestBuildCache:
    def test_put_and_get(self, build_cache: BuildCache) -> None:
        """Roundtrip: put data, get it back."""
        key = "a" * 64
        data = b"hello binary world"
        build_cache.put(key, data, "clap", "test.clap")

        result = build_cache.get(key)
        assert result is not None
        got_data, got_filename = result
        assert got_data == data
        assert got_filename == "test.clap"

    def test_get_miss(self, build_cache: BuildCache) -> None:
        assert build_cache.get("b" * 64) is None

    def test_put_idempotent(self, build_cache: BuildCache) -> None:
        """Double put doesn't error."""
        key = "c" * 64
        data = b"some data"
        build_cache.put(key, data, "clap", "out.clap")
        build_cache.put(key, data, "clap", "out.clap")

        result = build_cache.get(key)
        assert result is not None
        assert result[0] == data

    def test_eviction_by_age(self, tmp_path: Path) -> None:
        """Entries older than max_age are evicted."""
        bc = BuildCache(root=tmp_path, max_age=1)  # 1 second
        key = "d" * 64
        bc.put(key, b"old data", "clap", "old.clap")

        # Manipulate meta.json timestamp to the past
        shard = tmp_path / "builds" / key[:2] / key
        meta_path = shard / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["created"] = time.time() - 100
        meta_path.write_text(json.dumps(meta))

        # Should be expired on get
        assert bc.get(key) is None

    def test_clear(self, build_cache: BuildCache) -> None:
        for i in range(3):
            key = f"{i:0>64}"
            build_cache.put(key, f"data{i}".encode(), "clap", f"out{i}.clap")

        removed = build_cache.clear()
        assert removed == 3
        assert build_cache.size() == (0, 0)

    def test_size(self, build_cache: BuildCache) -> None:
        build_cache.put("e" * 64, b"12345", "clap", "a.clap")
        build_cache.put("f" * 64, b"67890abc", "vst3", "b.vst3")

        count, total = build_cache.size()
        assert count == 2
        # total includes meta.json files + artifact data
        assert total > len(b"12345") + len(b"67890abc")

    def test_size_empty(self, build_cache: BuildCache) -> None:
        assert build_cache.size() == (0, 0)

    def test_clear_empty(self, build_cache: BuildCache) -> None:
        assert build_cache.clear() == 0


class TestDefaultCacheRoot:
    def test_returns_path_containing_dsp_graph(self) -> None:
        root = _default_cache_root()
        assert "dsp-graph" in str(root)
        assert isinstance(root, Path)
