"""Build-to-plugin-target endpoints."""

from __future__ import annotations

import io
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from gen_dsp.core.builder import Builder  # type: ignore[import-untyped]
from gen_dsp.core.project import ProjectConfig, ProjectGenerator  # type: ignore[import-untyped]
from gen_dsp.graph.adapter import (
    SUPPORTED_PLATFORMS,
    generate_adapter_cpp,
    generate_manifest,
)
from gen_dsp.graph.compile import compile_graph
from gen_dsp.graph.models import Graph
from pydantic import BaseModel

# OS availability per platform, from gen-dsp README.
# Keys: platform name -> set of sys.platform-compatible OS tags.
_PLATFORM_OS: dict[str, set[str]] = {
    "au": {"darwin"},
    "chuck": {"darwin", "linux"},
    "circle": {"linux"},
    "clap": {"darwin", "linux", "win32"},
    "daisy": {"linux"},
    "lv2": {"darwin", "linux"},
    "max": {"darwin"},
    "pd": {"darwin", "linux"},
    "sc": {"darwin", "linux", "win32"},
    "vcvrack": {"darwin", "linux"},
    "vst3": {"darwin", "linux", "win32"},
}

_HOST_OS = (
    "darwin"
    if platform.system() == "Darwin"
    else ("win32" if platform.system() == "Windows" else "linux")
)


def _host_platforms() -> list[str]:
    """Return platforms supported on the current OS, sorted alphabetically."""
    return sorted(p for p, oses in _PLATFORM_OS.items() if _HOST_OS in oses)


router = APIRouter()


class BuildRequest(BaseModel):
    graph: dict[str, Any]
    platform: str


class BuildResponse(BaseModel):
    dsp_cpp: str
    adapter_cpp: str
    manifest: str
    platform: str
    supported_platforms: list[str]


class CompileBuildResponse(BaseModel):
    success: bool
    platform: str
    stdout: str
    stderr: str
    output_file: str | None


def _validate_build_request(req: BuildRequest) -> Graph:
    """Validate platform and parse graph, raising HTTPException on failure."""
    if req.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported platform: {req.platform!r}. Valid: {sorted(SUPPORTED_PLATFORMS)}",
        )
    try:
        return Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/build", response_model=BuildResponse)
async def generate(req: BuildRequest) -> BuildResponse:
    """Generate plugin source files: DSP C++, adapter C++, and manifest."""
    g = _validate_build_request(req)

    try:
        dsp_cpp = compile_graph(g)
        adapter_cpp = generate_adapter_cpp(g, req.platform)
        manifest = generate_manifest(g)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BuildResponse(
        dsp_cpp=dsp_cpp,
        adapter_cpp=adapter_cpp,
        manifest=manifest,
        platform=req.platform,
        supported_platforms=sorted(SUPPORTED_PLATFORMS),
    )


@router.get("/build/platforms")
async def platforms() -> dict[str, list[str]]:
    """Return the list of build platforms available on the current OS."""
    return {"platforms": _host_platforms()}


@router.post("/build/zip")
async def build_zip(req: BuildRequest) -> StreamingResponse:
    """Build and return a zip archive with DSP C++, adapter C++, and manifest."""
    g = _validate_build_request(req)

    try:
        dsp_cpp = compile_graph(g)
        adapter_cpp = generate_adapter_cpp(g, req.platform)
        manifest = generate_manifest(g)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{g.name}_dsp.h", dsp_cpp)
        zf.writestr(f"{g.name}_{req.platform}.cpp", adapter_cpp)
        zf.writestr("manifest.json", manifest)
    buf.seek(0)

    filename = f"{g.name}_{req.platform}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _compile_build(g: Graph, platform: str) -> tuple[CompileBuildResponse, Path | None]:
    """Run the full compile+build pipeline in a temp directory.

    Uses gen-dsp's ProjectGenerator for complete project setup (source files,
    platform templates, build files, platform-specific extras) and Builder for
    compilation.

    Returns the response and the temp dir path (caller must clean up on success,
    or None if already cleaned up on failure).
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
            return resp, tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)
        return resp, None
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


class BatchBuildRequest(BaseModel):
    graph: dict[str, Any]
    platforms: list[str]


class BatchBuildResponse(BaseModel):
    results: list[CompileBuildResponse]


@router.post("/build/batch", response_model=BatchBuildResponse)
async def batch_build(req: BatchBuildRequest) -> BatchBuildResponse:
    """Build for multiple platforms at once."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
            resp, tmpdir = _compile_build(g, plat)
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
            results.append(resp)
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
    return BatchBuildResponse(results=results)


@router.post("/build/compile", response_model=CompileBuildResponse)
async def compile_build(req: BuildRequest) -> CompileBuildResponse:
    """Compile a Graph to a binary plugin via gen-dsp's Builder."""
    g = _validate_build_request(req)

    try:
        resp, tmpdir = _compile_build(g, req.platform)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # Always clean up for the non-download endpoint
        pass

    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return resp


@router.post("/build/binary")
async def download_binary(req: BuildRequest) -> StreamingResponse:
    """Compile a Graph and return the binary plugin file."""
    g = _validate_build_request(req)

    try:
        resp, tmpdir = _compile_build(g, req.platform)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not resp.success or tmpdir is None or resp.output_file is None:
        raise HTTPException(
            status_code=400,
            detail=f"Build failed: {resp.stderr}",
        )

    output_path = tmpdir / resp.output_file
    if not output_path.is_file():
        # Check in build subdirectories
        candidates = list(tmpdir.rglob(resp.output_file))
        if candidates:
            output_path = candidates[0]
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Build succeeded but output file not found: {resp.output_file}",
            )

    binary_data = output_path.read_bytes()
    shutil.rmtree(tmpdir, ignore_errors=True)

    return StreamingResponse(
        io.BytesIO(binary_data),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{resp.output_file}"',
        },
    )
