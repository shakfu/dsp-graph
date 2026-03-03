"""Build-to-plugin-target endpoints."""

from __future__ import annotations

import io
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


@router.post("/build", response_model=BuildResponse)
async def build(req: BuildRequest) -> BuildResponse:
    """Compile a Graph to a plugin target: DSP C++, adapter C++, and manifest."""
    if req.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported platform: {req.platform!r}. Valid: {sorted(SUPPORTED_PLATFORMS)}",
        )
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
    """Return the list of supported build platforms."""
    return {"platforms": sorted(SUPPORTED_PLATFORMS)}


@router.post("/build/zip")
async def build_zip(req: BuildRequest) -> StreamingResponse:
    """Build and return a zip archive with DSP C++, adapter C++, and manifest."""
    if req.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported platform: {req.platform!r}. Valid: {sorted(SUPPORTED_PLATFORMS)}",
        )
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
