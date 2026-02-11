from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# Type alias for node input references: either a node/input/param ID or a literal float.
Ref = Union[str, float]


# ---------------------------------------------------------------------------
# Param & I/O declarations
# ---------------------------------------------------------------------------


class Param(BaseModel):
    name: str
    min: float = 0.0
    max: float = 1.0
    default: float = 0.0


class AudioInput(BaseModel):
    id: str


class AudioOutput(BaseModel):
    id: str
    source: str  # node ID that feeds this output


# ---------------------------------------------------------------------------
# Node types (discriminated union on "op")
# ---------------------------------------------------------------------------


class BinOp(BaseModel):
    id: str
    op: Literal["add", "sub", "mul", "div", "min", "max", "mod", "pow"]
    a: Ref
    b: Ref


class UnaryOp(BaseModel):
    id: str
    op: Literal[
        "sin",
        "cos",
        "tanh",
        "exp",
        "log",
        "abs",
        "sqrt",
        "neg",
        "floor",
        "ceil",
        "round",
        "sign",
    ]
    a: Ref


class Clamp(BaseModel):
    id: str
    op: Literal["clamp"] = "clamp"
    a: Ref
    lo: Ref = 0.0
    hi: Ref = 1.0


class Constant(BaseModel):
    id: str
    op: Literal["constant"] = "constant"
    value: float


class History(BaseModel):
    id: str
    op: Literal["history"] = "history"
    init: float = 0.0
    input: str  # node ID whose value is stored for next sample


class DelayLine(BaseModel):
    id: str
    op: Literal["delay"] = "delay"
    max_samples: int = 48000


class DelayRead(BaseModel):
    id: str
    op: Literal["delay_read"] = "delay_read"
    delay: str  # delay line ID
    tap: Ref  # tap position: node ID or literal
    interp: Literal["none", "linear", "cubic"] = "none"


class DelayWrite(BaseModel):
    id: str
    op: Literal["delay_write"] = "delay_write"
    delay: str  # delay line ID
    value: Ref  # node ID or literal to write


class Phasor(BaseModel):
    id: str
    op: Literal["phasor"] = "phasor"
    freq: Ref


class Noise(BaseModel):
    id: str
    op: Literal["noise"] = "noise"


class Compare(BaseModel):
    id: str
    op: Literal["gt", "lt", "gte", "lte", "eq"]
    a: Ref
    b: Ref


class Select(BaseModel):
    id: str
    op: Literal["select"] = "select"
    cond: Ref
    a: Ref
    b: Ref


class Wrap(BaseModel):
    id: str
    op: Literal["wrap"] = "wrap"
    a: Ref
    lo: Ref = 0.0
    hi: Ref = 1.0


class Fold(BaseModel):
    id: str
    op: Literal["fold"] = "fold"
    a: Ref
    lo: Ref = 0.0
    hi: Ref = 1.0


class Mix(BaseModel):
    id: str
    op: Literal["mix"] = "mix"
    a: Ref
    b: Ref
    t: Ref


class Delta(BaseModel):
    id: str
    op: Literal["delta"] = "delta"
    a: Ref


class Change(BaseModel):
    id: str
    op: Literal["change"] = "change"
    a: Ref


# Discriminated union of all node types
Node = Annotated[
    Union[
        BinOp,
        UnaryOp,
        Clamp,
        Constant,
        History,
        DelayLine,
        DelayRead,
        DelayWrite,
        Phasor,
        Noise,
        Compare,
        Select,
        Wrap,
        Fold,
        Mix,
        Delta,
        Change,
    ],
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------------
# Top-level graph
# ---------------------------------------------------------------------------


class Graph(BaseModel):
    name: str
    sample_rate: float = 44100.0
    inputs: list[AudioInput] = []
    outputs: list[AudioOutput] = []
    params: list[Param] = []
    nodes: list[Node] = []
