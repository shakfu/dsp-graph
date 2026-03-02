"""Simulation endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SimulateRequest(BaseModel):
    graph: dict[str, Any]
    n_samples: int = 64
    sample_rate: int = 44100
    params: dict[str, float] | None = None


class SimulateResponse(BaseModel):
    outputs: dict[str, list[float]]


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest) -> SimulateResponse:
    """Run a per-sample simulation of the graph."""
    try:
        from gen_dsp.graph.models import Graph
        from gen_dsp.graph.simulate import simulate as run_sim
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="Simulation requires numpy: pip install gen-dsp[sim]",
        ) from exc

    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        import numpy as np

        result = run_sim(
            g,
            n_samples=req.n_samples,
            params=req.params or None,
            sample_rate=float(req.sample_rate),
        )

        outputs: dict[str, list[float]] = {}
        for key, arr in result.outputs.items():
            if isinstance(arr, np.ndarray):
                outputs[key] = arr.tolist()
            else:
                outputs[key] = list(arr)

        return SimulateResponse(outputs=outputs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
