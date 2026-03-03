"""Simulation endpoints with stateful session support."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

SIGNAL_TYPES = ("impulse", "sine", "noise", "ones")

# Server-side session storage: session_id -> (SimState, Graph, last_access_time)
_sessions: dict[str, tuple[Any, Any, float]] = {}
_MAX_SESSIONS = 10
_SESSION_TTL = 300.0  # 5 minutes


def _cleanup_sessions() -> None:
    """Evict expired sessions."""
    now = time.monotonic()
    expired = [sid for sid, (_, _, t) in _sessions.items() if now - t > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
    # If still over limit, evict oldest
    while len(_sessions) > _MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k][2])
        del _sessions[oldest]


class SimulateRequest(BaseModel):
    graph: dict[str, Any]
    n_samples: int = 64
    sample_rate: int = 44100
    params: dict[str, float] | None = None
    inputs: dict[str, str] | None = None
    session_id: str | None = None


class SimulateResponse(BaseModel):
    outputs: dict[str, list[float]]
    session_id: str


class ContinueRequest(BaseModel):
    session_id: str
    n_samples: int = 64
    inputs: dict[str, str] | None = None


class ParamRequest(BaseModel):
    session_id: str
    name: str
    value: float


class SessionRequest(BaseModel):
    session_id: str


class BufferGetRequest(BaseModel):
    session_id: str
    buffer_id: str


class BufferGetResponse(BaseModel):
    data: list[float]


class BufferSetRequest(BaseModel):
    session_id: str
    buffer_id: str
    data: list[float]


class PeekResponse(BaseModel):
    values: dict[str, float]


def _generate_signal(signal_type: str, n_samples: int, sample_rate: float) -> Any:
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


def _build_inputs(
    graph: Any,
    n_samples: int,
    sample_rate: float,
    input_spec: dict[str, str] | None,
) -> Any:
    """Build input signal arrays for a graph's AudioInputs."""
    input_ids = [inp.id for inp in graph.inputs]
    if not input_ids:
        return None
    spec = input_spec or {}
    sr = float(sample_rate)
    return {iid: _generate_signal(spec.get(iid, "sine"), n_samples, sr) for iid in input_ids}


def _get_session(session_id: str) -> tuple[Any, Any]:
    """Retrieve a session's SimState and Graph, or raise 404."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")
    state, graph, _ = _sessions[session_id]
    _sessions[session_id] = (state, graph, time.monotonic())
    return state, graph


def _outputs_to_dict(outputs: dict[str, Any]) -> dict[str, list[float]]:
    """Convert numpy arrays to plain lists."""
    import numpy as np

    result: dict[str, list[float]] = {}
    for key, arr in outputs.items():
        if isinstance(arr, np.ndarray):
            result[key] = arr.tolist()
        else:
            result[key] = list(arr)
    return result


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest) -> SimulateResponse:
    """Run a per-sample simulation of the graph, optionally resuming a session."""
    try:
        from gen_dsp.graph.models import Graph
        from gen_dsp.graph.simulate import SimState
        from gen_dsp.graph.simulate import simulate as run_sim
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="Simulation requires numpy: pip install dsp-graph",
        ) from exc

    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        sim_inputs = _build_inputs(g, req.n_samples, req.sample_rate, req.inputs)

        # Resume from existing session or start fresh
        state: SimState | None = None
        if req.session_id and req.session_id in _sessions:
            state, _, _ = _sessions[req.session_id]

        result = run_sim(
            g,
            n_samples=req.n_samples,
            inputs=sim_inputs,
            params=req.params or None,
            sample_rate=float(req.sample_rate),
            state=state,
        )

        # Store session
        session_id = req.session_id or str(uuid.uuid4())
        _cleanup_sessions()
        _sessions[session_id] = (result.state, g, time.monotonic())

        return SimulateResponse(
            outputs=_outputs_to_dict(result.outputs),
            session_id=session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/simulate/continue", response_model=SimulateResponse)
async def simulate_continue(req: ContinueRequest) -> SimulateResponse:
    """Continue simulation from an existing session state."""
    try:
        from gen_dsp.graph.simulate import simulate as run_sim
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="Simulation requires numpy: pip install dsp-graph",
        ) from exc

    state, graph = _get_session(req.session_id)

    try:
        sim_inputs = _build_inputs(graph, req.n_samples, graph.sample_rate, req.inputs)

        result = run_sim(
            graph,
            n_samples=req.n_samples,
            inputs=sim_inputs,
            state=state,
            sample_rate=float(graph.sample_rate),
        )

        _sessions[req.session_id] = (result.state, graph, time.monotonic())

        return SimulateResponse(
            outputs=_outputs_to_dict(result.outputs),
            session_id=req.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/simulate/param")
async def simulate_param(req: ParamRequest) -> dict[str, str]:
    """Set a parameter value on an existing simulation session."""
    state, _ = _get_session(req.session_id)
    try:
        state.set_param(req.name, req.value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok"}


@router.post("/simulate/peek", response_model=PeekResponse)
async def simulate_peek(req: SessionRequest) -> PeekResponse:
    """Return current peek node values from an existing session."""
    state, graph = _get_session(req.session_id)
    from gen_dsp.graph.models import Peek

    values: dict[str, float] = {}
    for node in graph.nodes:
        if isinstance(node, Peek):
            try:
                values[node.id] = state.get_peek(node.id)
            except KeyError:
                pass
    return PeekResponse(values=values)


@router.post("/simulate/reset")
async def simulate_reset(req: SessionRequest) -> dict[str, str]:
    """Reset simulation state to initial values."""
    state, _ = _get_session(req.session_id)
    state.reset()
    return {"status": "ok"}


@router.post("/simulate/buffer/get", response_model=BufferGetResponse)
async def buffer_get(req: BufferGetRequest) -> BufferGetResponse:
    """Get the contents of a buffer from a simulation session."""
    import numpy as np

    state, _ = _get_session(req.session_id)
    try:
        buf = state.get_buffer(req.buffer_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(buf, np.ndarray):
        return BufferGetResponse(data=buf.tolist())
    return BufferGetResponse(data=list(buf))


@router.post("/simulate/buffer/set")
async def buffer_set(req: BufferSetRequest) -> dict[str, str]:
    """Set the contents of a buffer in a simulation session."""
    import numpy as np

    state, _ = _get_session(req.session_id)
    try:
        state.set_buffer(req.buffer_id, np.array(req.data, dtype=np.float32))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok"}
