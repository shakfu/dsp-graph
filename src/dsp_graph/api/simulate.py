"""Simulation endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

SIGNAL_TYPES = ("impulse", "sine", "noise", "ones")


class SimulateRequest(BaseModel):
    graph: dict[str, Any]
    n_samples: int = 64
    sample_rate: int = 44100
    params: dict[str, float] | None = None
    inputs: dict[str, str] | None = None


class SimulateResponse(BaseModel):
    outputs: dict[str, list[float]]


def _generate_signal(
    signal_type: str, n_samples: int, sample_rate: float
) -> "numpy.ndarray[Any, numpy.dtype[numpy.float32]]":
    """Generate a synthetic input signal array."""
    import numpy as np

    if signal_type == "impulse":
        sig = np.zeros(n_samples, dtype=np.float32)
        sig[0] = 1.0
        return sig
    if signal_type == "sine":
        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    if signal_type == "noise":
        rng = np.random.default_rng(42)
        return rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)
    if signal_type == "ones":
        return np.ones(n_samples, dtype=np.float32)
    raise ValueError(f"Unknown signal type: {signal_type!r} (valid: {SIGNAL_TYPES})")


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

        # Build input signal arrays
        input_ids = [inp.id for inp in g.inputs]
        sim_inputs: dict[str, np.ndarray] | None = None
        if input_ids:
            input_spec = req.inputs or {}
            sr = float(req.sample_rate)
            sim_inputs = {}
            for iid in input_ids:
                sig_type = input_spec.get(iid, "sine")
                sim_inputs[iid] = _generate_signal(sig_type, req.n_samples, sr)

        result = run_sim(
            g,
            n_samples=req.n_samples,
            inputs=sim_inputs,
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
