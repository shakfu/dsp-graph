"""Shared fixtures for dsp-graph tests."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from gen_dsp.graph.models import (
    AudioInput,
    AudioOutput,
    BinOp,
    Buffer,
    Graph,
    History,
    Param,
    Phasor,
)

from dsp_graph.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def stereo_gain() -> Graph:
    """Stateless stereo gain: in1 * gain -> out1, in2 * gain -> out2."""
    return Graph(
        name="stereo_gain",
        inputs=[AudioInput(id="in1"), AudioInput(id="in2")],
        outputs=[
            AudioOutput(id="out1", source="scaled1"),
            AudioOutput(id="out2", source="scaled2"),
        ],
        params=[Param(name="gain", min=0.0, max=2.0, default=1.0)],
        nodes=[
            BinOp(id="scaled1", op="mul", a="in1", b="gain"),
            BinOp(id="scaled2", op="mul", a="in2", b="gain"),
        ],
    )


@pytest.fixture
def onepole() -> Graph:
    """One-pole lowpass filter with feedback via History."""
    return Graph(
        name="onepole",
        inputs=[AudioInput(id="in1")],
        outputs=[AudioOutput(id="out1", source="result")],
        params=[Param(name="coeff", min=0.0, max=0.999, default=0.5)],
        nodes=[
            BinOp(id="inv_coeff", op="sub", a=1.0, b="coeff"),
            BinOp(id="dry", op="mul", a="in1", b="inv_coeff"),
            History(id="prev", init=0.0, input="result"),
            BinOp(id="wet", op="mul", a="prev", b="coeff"),
            BinOp(id="result", op="add", a="dry", b="wet"),
        ],
    )


@pytest.fixture
def phasor_graph() -> Graph:
    """Minimal phasor graph for simulation testing."""
    return Graph(
        name="phasor",
        inputs=[],
        outputs=[AudioOutput(id="out1", source="ph")],
        params=[Param(name="freq", min=1.0, max=20000.0, default=440.0)],
        nodes=[
            Phasor(id="ph", freq="freq"),
        ],
    )


@pytest.fixture
def buffer_graph() -> Graph:
    """Graph with a Buffer node for buffer endpoint testing."""
    return Graph(
        name="buffer_test",
        inputs=[AudioInput(id="in1")],
        outputs=[AudioOutput(id="out1", source="pass")],
        nodes=[
            Buffer(id="mybuf", size=16),
            BinOp(id="pass", op="add", a="in1", b=0.0),
        ],
    )


@pytest.fixture
def buffer_graph_json(buffer_graph: Graph) -> dict[str, Any]:
    return buffer_graph.model_dump()


@pytest.fixture
def stereo_gain_json(stereo_gain: Graph) -> dict[str, Any]:
    return stereo_gain.model_dump()


@pytest.fixture
def onepole_json(onepole: Graph) -> dict[str, Any]:
    return onepole.model_dump()


@pytest.fixture
def phasor_json(phasor_graph: Graph) -> dict[str, Any]:
    return phasor_graph.model_dump()
