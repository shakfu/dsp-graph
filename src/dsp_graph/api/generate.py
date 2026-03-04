"""Code-generation endpoints (Graph -> C++ source, adapter, manifest)."""

from __future__ import annotations

import io
import platform
import zipfile
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


class GenerateRequest(BaseModel):
    graph: dict[str, Any]
    platform: str


class GenerateResponse(BaseModel):
    dsp_cpp: str
    adapter_cpp: str
    manifest: str
    platform: str
    supported_platforms: list[str]


def _validate_generate_request(req: GenerateRequest) -> Graph:
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


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate plugin source files: DSP C++, adapter C++, and manifest."""
    g = _validate_generate_request(req)

    try:
        dsp_cpp = compile_graph(g)
        adapter_cpp = generate_adapter_cpp(g, req.platform)
        manifest = generate_manifest(g)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GenerateResponse(
        dsp_cpp=dsp_cpp,
        adapter_cpp=adapter_cpp,
        manifest=manifest,
        platform=req.platform,
        supported_platforms=sorted(SUPPORTED_PLATFORMS),
    )


@router.get("/generate/platforms")
async def platforms() -> dict[str, list[str]]:
    """Return the list of build platforms available on the current OS."""
    return {"platforms": _host_platforms()}


@router.post("/generate/zip")
async def generate_zip(req: GenerateRequest) -> StreamingResponse:
    """Generate and return a zip archive with DSP C++, adapter C++, and manifest."""
    g = _validate_generate_request(req)

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
