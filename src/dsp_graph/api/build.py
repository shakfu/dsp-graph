"""Build-to-binary endpoints (compilation via gen-dsp's Builder)."""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from gen_dsp.core.builder import Builder  # type: ignore[import-untyped]
from gen_dsp.core.project import ProjectConfig, ProjectGenerator  # type: ignore[import-untyped]
from gen_dsp.graph.adapter import SUPPORTED_PLATFORMS
from gen_dsp.graph.models import Graph
from pydantic import BaseModel, Field

from dsp_graph.api.generate import (
    GenerateRequest,
    _validate_generate_request,
    safe_filename,
    validate_graph_name,
)
from dsp_graph.cache import BuildCache, cache_key, get_cache
from dsp_graph.config import is_build_enabled

logger = logging.getLogger(__name__)


def _require_build_enabled() -> None:
    """Router dependency: 404 when the server was started with --disable-build.

    Native compilation is the highest-privilege api (it invokes a host toolchain
    and writes to disk). It is on by default but can be disabled for a hardened
    deployment. A 404 (rather than 403) keeps the endpoints indistinguishable
    from "not present" when disabled.
    """
    if not is_build_enabled():
        raise HTTPException(
            status_code=404,
            detail="Native build endpoints are disabled (the server was started "
            "with --disable-build).",
        )


# Every route on this router is gated: disabled when --disable-build is set.
router = APIRouter(dependencies=[Depends(_require_build_enabled)])

# Batch artifact cache: batch_id -> {created, artifacts: {platform: (cache_key, filename)}}.
# In-process and per-worker (single-worker assumption; see README "Security model").
# Accessed only on the event loop, so it needs no lock.
_batch_cache: dict[str, dict[str, Any]] = {}
_BATCH_TTL = 600  # 10 minutes

# Native builds are heavy (a full toolchain invocation each). Cap how many run
# concurrently so a burst cannot thrash CPU/IO or exhaust the threadpool. This is
# back-pressure: the (N+1)th build waits for a slot rather than failing. Acquired
# on the event loop before entering the worker thread, so it needs no lock.
_MAX_CONCURRENT_BUILDS = 2
_build_slot = asyncio.Semaphore(_MAX_CONCURRENT_BUILDS)


def _cleanup_expired_batches() -> None:
    now = time.monotonic()
    expired = [k for k, v in _batch_cache.items() if now - v["created"] > _BATCH_TTL]
    for k in expired:
        del _batch_cache[k]


class CompileBuildResponse(BaseModel):
    success: bool
    platform: str
    stdout: str
    stderr: str
    output_file: str | None


class _BuildResult:
    """Internal result from _compile_build, carrying response + filesystem details."""

    __slots__ = ("response", "tmpdir", "output_path")

    def __init__(
        self,
        response: CompileBuildResponse,
        tmpdir: Path | None,
        output_path: Path | None,
    ) -> None:
        self.response = response
        self.tmpdir = tmpdir
        self.output_path = output_path


def _compile_build(g: Graph, platform: str) -> _BuildResult:
    """Run the full compile+build pipeline in a temp directory.

    Uses gen-dsp's ProjectGenerator for complete project setup (source files,
    platform templates, build files, platform-specific extras) and Builder for
    compilation.

    Returns a _BuildResult with response, tmpdir (caller must clean up on success),
    and the actual output_path on disk.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="dsp_graph_build_")).resolve()
    try:
        config = ProjectConfig(name=g.name, platform=platform)
        gen = ProjectGenerator.from_graph(g, config)
        gen.generate(output_dir=tmpdir)
        result = Builder(tmpdir).build(target_platform=platform)
        output_file = result.output_file.name if result.output_file else None
        resp = CompileBuildResponse(
            success=result.success,
            platform=result.platform,
            stdout=result.stdout,
            stderr=result.stderr,
            output_file=output_file,
        )
        if result.success:
            return _BuildResult(resp, tmpdir, result.output_file)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return _BuildResult(resp, None, None)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def _zip_batch_artifacts(bc: BuildCache, artifacts: dict[str, tuple[str, str]]) -> bytes:
    """Read cached artifacts from disk and assemble a zip. Blocking; run in a thread."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for plat, (key, filename) in artifacts.items():
            hit = bc.get(key)
            if hit is not None:
                data, _ = hit
                zf.writestr(f"{plat}/{filename}", data)
    return buf.getvalue()


def _read_output_bytes(output_path: Path) -> bytes:
    """Read output artifact as bytes, handling both files and bundle directories."""
    if output_path.is_file():
        return output_path.read_bytes()
    if output_path.is_dir():
        # Bundle (e.g. .clap on macOS) -- zip the directory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for child in output_path.rglob("*"):
                if child.is_file():
                    zf.writestr(str(child.relative_to(output_path.parent)), child.read_bytes())
        return buf.getvalue()
    msg = f"Output path is neither file nor directory: {output_path}"
    raise FileNotFoundError(msg)


class _CachedBuildResult:
    """Result from _compile_build_cached."""

    __slots__ = ("response", "data", "filename")

    def __init__(
        self, response: CompileBuildResponse, data: bytes | None, filename: str | None
    ) -> None:
        self.response = response
        self.data = data
        self.filename = filename


def _compile_build_cached(
    g: Graph, platform: str, disk_cache: BuildCache | None = None
) -> _CachedBuildResult:
    """Compile with disk cache. On hit, returns cached bytes without recompiling."""
    bc = disk_cache or get_cache()
    key = cache_key(g, platform)

    # Check cache
    hit = bc.get(key)
    if hit is not None:
        data, filename = hit
        resp = CompileBuildResponse(
            success=True,
            platform=platform,
            stdout="(cached)",
            stderr="",
            output_file=filename,
        )
        logger.info("Build cache hit for %s/%s", platform, key[:12])
        return _CachedBuildResult(resp, data, filename)

    # Cache miss -- build
    br = _compile_build(g, platform)
    if not br.response.success or br.output_path is None:
        return _CachedBuildResult(br.response, None, None)

    try:
        data = _read_output_bytes(br.output_path)
    except FileNotFoundError:
        if br.tmpdir is not None:
            shutil.rmtree(br.tmpdir, ignore_errors=True)
        return _CachedBuildResult(br.response, None, None)

    filename = br.response.output_file or platform
    bc.put(key, data, platform, filename)

    if br.tmpdir is not None:
        shutil.rmtree(br.tmpdir, ignore_errors=True)

    return _CachedBuildResult(br.response, data, filename)


class BatchBuildRequest(BaseModel):
    graph: dict[str, Any]
    # Cap the list to the number of known platforms; each triggers a native build.
    platforms: list[str] = Field(min_length=1, max_length=len(SUPPORTED_PLATFORMS))


class BatchBuildResponse(BaseModel):
    batch_id: str
    results: list[CompileBuildResponse]


@router.post("/build", response_model=CompileBuildResponse)
async def compile_build(req: GenerateRequest) -> CompileBuildResponse:
    """Compile a Graph to a binary plugin via gen-dsp's Builder."""
    g = _validate_generate_request(req)

    try:
        async with _build_slot:
            cr = await run_in_threadpool(_compile_build_cached, g, req.platform)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return cr.response


@router.post("/build/binary")
async def download_binary(req: GenerateRequest) -> StreamingResponse:
    """Compile a Graph and return the binary plugin file."""
    g = _validate_generate_request(req)

    try:
        async with _build_slot:
            cr = await run_in_threadpool(_compile_build_cached, g, req.platform)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not cr.response.success or cr.data is None:
        raise HTTPException(
            status_code=400,
            detail=f"Build failed: {cr.response.stderr}",
        )

    filename = safe_filename(cr.filename or "output")
    return StreamingResponse(
        io.BytesIO(cr.data),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.post("/build/batch", response_model=BatchBuildResponse)
async def batch_build(req: BatchBuildRequest) -> BatchBuildResponse:
    """Build for multiple platforms at once, caching artifacts for later zip download."""
    _cleanup_expired_batches()

    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    validate_graph_name(g.name)

    batch_id = uuid.uuid4().hex
    artifacts: dict[str, tuple[str, str]] = {}  # platform -> (cache_key, filename)
    results: list[CompileBuildResponse] = []

    for plat in req.platforms:
        if plat not in SUPPORTED_PLATFORMS:
            results.append(
                CompileBuildResponse(
                    success=False,
                    platform=plat,
                    stdout="",
                    stderr=f"Unsupported platform: {plat!r}",
                    output_file=None,
                )
            )
            continue
        try:
            # Acquire per platform (not once for the whole batch) so a large
            # batch does not monopolize every slot and starve single builds.
            async with _build_slot:
                cr = await run_in_threadpool(_compile_build_cached, g, plat)
            if cr.response.success and cr.data is not None and cr.filename is not None:
                key = cache_key(g, plat)
                artifacts[plat] = (key, cr.filename)
            results.append(cr.response)
        except Exception as exc:
            results.append(
                CompileBuildResponse(
                    success=False,
                    platform=plat,
                    stdout="",
                    stderr=str(exc),
                    output_file=None,
                )
            )

    _batch_cache[batch_id] = {"created": time.monotonic(), "artifacts": artifacts}
    return BatchBuildResponse(batch_id=batch_id, results=results)


@router.get("/build/batch/{batch_id}/zip")
async def download_batch_zip(batch_id: str) -> StreamingResponse:
    """Download a zip of all cached binaries from a batch build."""
    _cleanup_expired_batches()

    entry = _batch_cache.get(batch_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Batch not found or expired")

    artifacts: dict[str, tuple[str, str]] = entry["artifacts"]
    if not artifacts:
        raise HTTPException(status_code=404, detail="No successful builds in batch")

    bc = get_cache()
    data = await run_in_threadpool(_zip_batch_artifacts, bc, artifacts)

    # Remove from cache after serving
    del _batch_cache[batch_id]

    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_build.zip"'},
    )


# -- Cache management endpoints --


class CacheClearResponse(BaseModel):
    removed: int


class CacheInfoResponse(BaseModel):
    entries: int
    total_bytes: int
    cache_dir: str


@router.delete("/build/cache", response_model=CacheClearResponse)
async def clear_build_cache() -> CacheClearResponse:
    """Clear all cached build artifacts."""
    n = await run_in_threadpool(get_cache().clear)
    return CacheClearResponse(removed=n)


@router.get("/build/cache/info", response_model=CacheInfoResponse)
async def build_cache_info() -> CacheInfoResponse:
    """Return cache statistics."""
    bc = get_cache()
    entries, total_bytes = await run_in_threadpool(bc.size)
    return CacheInfoResponse(
        entries=entries,
        total_bytes=total_bytes,
        cache_dir=str(bc.cache_dir),
    )
