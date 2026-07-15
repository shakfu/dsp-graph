"""Concurrency tests for the event-loop-offload changes.

These verify that blocking work (native builds, numpy simulation, cache disk IO)
no longer stalls the async event loop, that native-build concurrency is bounded,
and that the BuildCache's disk bookkeeping is safe under parallel access.

The async scenarios are driven through httpx's ASGI transport (which runs the
real event loop and threadpool) via ``asyncio.run`` so no pytest-asyncio plugin
is required.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

import dsp_graph.api.build as build_mod
from dsp_graph.api.build import CompileBuildResponse, _CachedBuildResult
from dsp_graph.cache import BuildCache
from dsp_graph.security import SESSION_HEADER, SESSION_TOKEN
from dsp_graph.server import app


@asynccontextmanager
async def _async_client() -> Any:
    """An httpx client bound to the ASGI app, pre-authenticated with the token."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={SESSION_HEADER: SESSION_TOKEN},
    ) as client:
        yield client


def _fake_cached_result(platform: str) -> _CachedBuildResult:
    resp = CompileBuildResponse(
        success=True, platform=platform, stdout="ok", stderr="", output_file="out.bin"
    )
    return _CachedBuildResult(resp, b"binary-bytes", "out.bin")


def test_event_loop_not_blocked_during_build(
    monkeypatch: pytest.MonkeyPatch, stereo_gain_json: dict[str, Any]
) -> None:
    """A slow build must not stall other requests.

    The build stub blocks a worker thread until released. While it is blocked, a
    /validate request must still complete. On the pre-fix (inline) code the build
    would hold the event loop and this scenario would deadlock/time out.
    """
    build_entered = threading.Event()
    release = threading.Event()

    def fake_build(g: Any, platform: str) -> _CachedBuildResult:
        build_entered.set()
        # Block the worker thread (simulating a long native build).
        assert release.wait(timeout=5), "test never released the build"
        return _fake_cached_result(platform)

    monkeypatch.setattr(build_mod, "_compile_build_cached", fake_build)

    async def scenario() -> None:
        async with _async_client() as c:
            build_task = asyncio.create_task(
                c.post("/api/build", json={"graph": stereo_gain_json, "platform": "clap"})
            )
            # Wait until the build is actually executing in a worker thread.
            for _ in range(500):
                if build_entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert build_entered.is_set(), "build never started"

            # The loop must remain responsive while the build blocks a thread.
            v = await asyncio.wait_for(
                c.post("/api/graph/validate", json={"graph": stereo_gain_json}),
                timeout=5,
            )
            assert v.status_code == 200

            release.set()
            b = await asyncio.wait_for(build_task, timeout=5)
            assert b.status_code == 200
            assert b.json()["success"] is True

    asyncio.run(scenario())


def test_native_build_concurrency_is_bounded(
    monkeypatch: pytest.MonkeyPatch, stereo_gain_json: dict[str, Any]
) -> None:
    """No more than _MAX_CONCURRENT_BUILDS builds may run at once."""
    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def fake_build(g: Any, platform: str) -> _CachedBuildResult:
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.2)
        with lock:
            state["current"] -= 1
        return _fake_cached_result(platform)

    monkeypatch.setattr(build_mod, "_compile_build_cached", fake_build)

    n_requests = build_mod._MAX_CONCURRENT_BUILDS + 4

    async def scenario() -> list[httpx.Response]:
        async with _async_client() as c:
            tasks = [
                c.post("/api/build", json={"graph": stereo_gain_json, "platform": "clap"})
                for _ in range(n_requests)
            ]
            return await asyncio.gather(*tasks)

    resps = asyncio.run(scenario())
    assert all(r.status_code == 200 for r in resps)
    assert state["max"] <= build_mod._MAX_CONCURRENT_BUILDS
    # Sanity: we actually exercised concurrency (more than one ran together).
    assert state["max"] >= 2


def test_concurrent_distinct_sessions_stay_isolated(
    stereo_gain_json: dict[str, Any],
) -> None:
    """Parallel /simulate calls each get a distinct session and correct output."""
    n = 8

    async def scenario() -> list[httpx.Response]:
        async with _async_client() as c:
            tasks = [
                c.post(
                    "/api/simulate",
                    json={"graph": stereo_gain_json, "n_samples": 32},
                )
                for _ in range(n)
            ]
            return await asyncio.gather(*tasks)

    resps = asyncio.run(scenario())
    assert all(r.status_code == 200 for r in resps)
    session_ids = {r.json()["session_id"] for r in resps}
    assert len(session_ids) == n
    for r in resps:
        outputs = r.json()["outputs"]
        for arr in outputs.values():
            assert len(arr) == 32


def test_buildcache_concurrent_put_get_roundtrip(tmp_path: Any) -> None:
    """Concurrent puts/gets of distinct keys must all round-trip correctly."""
    bc = BuildCache(root=tmp_path / "cache")
    n = 64

    def work(i: int) -> None:
        key = f"{i:064d}"
        data = f"payload-{i}".encode()
        bc.put(key, data, "clap", "art.bin")
        hit = bc.get(key)
        assert hit is not None
        got, filename = hit
        assert got == data
        assert filename == "art.bin"

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, range(n)))


def test_buildcache_eviction_during_reads_never_raises(tmp_path: Any) -> None:
    """Eviction rmtree racing with reads must not raise (may yield misses)."""
    # max_age=0 makes every entry immediately expirable, so _evict_expired
    # actively removes shards while readers touch them -- the race the lock guards.
    bc = BuildCache(root=tmp_path / "cache", max_age=0)
    for i in range(32):
        bc.put(f"{i:064d}", f"d{i}".encode(), "clap", "art.bin")

    errors: list[Exception] = []

    def reader() -> None:
        try:
            for _ in range(4):
                for i in range(32):
                    bc.get(f"{i:064d}")
        except Exception as exc:  # noqa: BLE001 - the point is to catch any escape
            errors.append(exc)

    def evictor() -> None:
        try:
            for _ in range(16):
                bc._evict_expired()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads += [threading.Thread(target=evictor) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
