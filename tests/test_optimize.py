"""Tests for optimization passes."""

from __future__ import annotations

import pytest

from dsp_graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    Change,
    Clamp,
    Compare,
    Constant,
    DelayLine,
    DelayRead,
    DelayWrite,
    Delta,
    Fold,
    Graph,
    History,
    Mix,
    Noise,
    Param,
    Phasor,
    Select,
    UnaryOp,
    Wrap,
    constant_fold,
    eliminate_dead_nodes,
    optimize_graph,
)

# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


class TestConstantFold:
    def test_binop_div_folded(self) -> None:
        """44100.0 / 1000.0 -> Constant(44.1)"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="sr", value=44100.0),
                Constant(id="k", value=1000.0),
                BinOp(id="r", op="div", a="sr", b="k"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(44.1)

    def test_chain_folds(self) -> None:
        """Chain of constants collapses: (2 + 3) * 4 -> 20"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="result")],
            nodes=[
                Constant(id="a", value=2.0),
                Constant(id="b", value=3.0),
                BinOp(id="sum", op="add", a="a", b="b"),
                Constant(id="c", value=4.0),
                BinOp(id="result", op="mul", a="sum", b="c"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["result"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(20.0)

    def test_non_constant_input_preserved(self) -> None:
        """Nodes with non-constant inputs are not folded."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="k", value=2.0),
                BinOp(id="r", op="mul", a="in1", b="k"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, BinOp)

    def test_stateful_never_folded(self) -> None:
        """Stateful nodes are never constant-folded."""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="p")],
            nodes=[
                Phasor(id="p", freq=440.0),
                Noise(id="n"),
                History(id="h", input="p"),
                Delta(id="d", a=0.0),
                Change(id="c", a=0.0),
            ],
        )
        folded = constant_fold(g)
        types = {n.id: type(n).__name__ for n in folded.nodes}
        assert types["p"] == "Phasor"
        assert types["n"] == "Noise"
        assert types["h"] == "History"
        assert types["d"] == "Delta"
        assert types["c"] == "Change"

    def test_unaryop_folded(self) -> None:
        """sin(0.0) -> 0.0"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="zero", value=0.0),
                UnaryOp(id="r", op="sin", a="zero"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(0.0)

    def test_compare_folded(self) -> None:
        """5.0 > 3.0 -> 1.0"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="a", value=5.0),
                Constant(id="b", value=3.0),
                Compare(id="r", op="gt", a="a", b="b"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(1.0)

    def test_select_folded(self) -> None:
        """select(1.0, 10.0, 20.0) -> 10.0"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="cond", value=1.0),
                Constant(id="a", value=10.0),
                Constant(id="b", value=20.0),
                Select(id="r", cond="cond", a="a", b="b"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(10.0)

    def test_clamp_folded(self) -> None:
        """clamp(5.0, 0.0, 1.0) -> 1.0"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="v", value=5.0),
                Clamp(id="r", a="v"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(1.0)

    def test_mix_folded(self) -> None:
        """mix(0.0, 10.0, 0.5) -> 5.0"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="a", value=0.0),
                Constant(id="b", value=10.0),
                Constant(id="t", value=0.5),
                Mix(id="r", a="a", b="b", t="t"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(5.0)

    def test_wrap_folded(self) -> None:
        """wrap(1.5, 0, 1) -> 0.5"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="v", value=1.5),
                Wrap(id="r", a="v"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(0.5)

    def test_fold_folded(self) -> None:
        """fold(1.7, 0, 1) -> 0.3"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="v", value=1.7),
                Fold(id="r", a="v"),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(0.3)

    def test_literal_ref_folded(self) -> None:
        """BinOp with literal float refs folds: 2.0 + 3.0 -> 5.0"""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                BinOp(id="r", op="add", a=2.0, b=3.0),
            ],
        )
        folded = constant_fold(g)
        r = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(r, Constant)
        assert r.value == pytest.approx(5.0)

    def test_extended_binops_fold(self) -> None:
        """min, max, mod, pow fold correctly."""
        for op, a, b, expected in [
            ("min", 3.0, 5.0, 3.0),
            ("max", 3.0, 5.0, 5.0),
            ("mod", 7.0, 3.0, 1.0),
            ("pow", 2.0, 3.0, 8.0),
        ]:
            g = Graph(
                name="test",
                outputs=[AudioOutput(id="out1", source="r")],
                nodes=[BinOp(id="r", op=op, a=a, b=b)],
            )
            folded = constant_fold(g)
            r = {n.id: n for n in folded.nodes}["r"]
            assert isinstance(r, Constant), f"{op} not folded"
            assert r.value == pytest.approx(expected), f"{op}: {r.value} != {expected}"


# ---------------------------------------------------------------------------
# Dead node elimination
# ---------------------------------------------------------------------------


class TestDeadNodeElimination:
    def test_unreachable_removed(self) -> None:
        """Nodes not reachable from outputs are removed."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="used")],
            nodes=[
                Constant(id="used", value=1.0),
                Constant(id="dead", value=999.0),
            ],
        )
        result = eliminate_dead_nodes(g)
        ids = {n.id for n in result.nodes}
        assert "used" in ids
        assert "dead" not in ids

    def test_reachable_preserved(self) -> None:
        """All nodes in the dependency chain are preserved."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="k", value=2.0),
                BinOp(id="r", op="mul", a="in1", b="k"),
            ],
        )
        result = eliminate_dead_nodes(g)
        ids = {n.id for n in result.nodes}
        assert ids == {"k", "r"}

    def test_feedback_edges_preserve_nodes(self) -> None:
        """Nodes reachable only through feedback edges are still preserved."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="result")],
            nodes=[
                History(id="prev", input="result", init=0.0),
                BinOp(id="result", op="add", a="in1", b="prev"),
            ],
        )
        result = eliminate_dead_nodes(g)
        ids = {n.id for n in result.nodes}
        # History references "result" via feedback, and "result" uses "prev"
        assert ids == {"prev", "result"}

    def test_delay_chain_preserved(self) -> None:
        """Delay line, read, and write are all preserved when output uses read."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="rd")],
            nodes=[
                DelayLine(id="dl"),
                DelayRead(id="rd", delay="dl", tap=100.0),
                DelayWrite(id="dw", delay="dl", value="in1"),
                Constant(id="dead", value=0.0),
            ],
        )
        result = eliminate_dead_nodes(g)
        ids = {n.id for n in result.nodes}
        assert "dl" in ids
        assert "rd" in ids
        # dw writes to the same delay line that rd reads -- side effect preserved
        assert "dw" in ids
        assert "dead" not in ids

    def test_delay_write_deps_preserved(self) -> None:
        """DelayWrite's input dependencies are also preserved."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="rd")],
            nodes=[
                DelayLine(id="dl"),
                DelayRead(id="rd", delay="dl", tap=100.0),
                BinOp(id="scaled", op="mul", a="in1", b=0.5),
                DelayWrite(id="dw", delay="dl", value="scaled"),
            ],
        )
        result = eliminate_dead_nodes(g)
        ids = {n.id for n in result.nodes}
        # scaled is only reachable through dw, which is kept as a side effect
        assert ids == {"dl", "rd", "scaled", "dw"}

    def test_unreferenced_delay_write_removed(self) -> None:
        """DelayWrite to a line with no reachable reader IS dead."""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="c")],
            nodes=[
                Constant(id="c", value=1.0),
                DelayLine(id="dl"),
                DelayWrite(id="dw", delay="dl", value=0.0),
            ],
        )
        result = eliminate_dead_nodes(g)
        ids = {n.id for n in result.nodes}
        assert "c" in ids
        assert "dw" not in ids
        assert "dl" not in ids


# ---------------------------------------------------------------------------
# Combined optimization
# ---------------------------------------------------------------------------


class TestOptimizeGraph:
    def test_fold_then_eliminate(self) -> None:
        """Constant folding + dead elimination applied together."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="sr", value=44100.0),
                Constant(id="k", value=1000.0),
                BinOp(id="ratio", op="div", a="sr", b="k"),
                BinOp(id="r", op="mul", a="in1", b="ratio"),
                Constant(id="dead", value=999.0),
            ],
        )
        result = optimize_graph(g)
        ids = {n.id for n in result.nodes}
        # ratio should be folded into a constant
        r_ratio = {n.id: n for n in result.nodes}.get("ratio")
        assert isinstance(r_ratio, Constant)
        assert r_ratio.value == pytest.approx(44.1)
        # dead should be eliminated
        assert "dead" not in ids
        # sr and k are now dead too (ratio is a constant, doesn't reference them)
        assert "sr" not in ids
        assert "k" not in ids

    def test_empty_graph(self) -> None:
        g = Graph(name="empty")
        result = optimize_graph(g)
        assert result.nodes == []

    def test_param_preserves_nodes(self) -> None:
        """Nodes depending on params are not folded."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            params=[Param(name="gain")],
            nodes=[
                BinOp(id="r", op="mul", a="in1", b="gain"),
            ],
        )
        result = optimize_graph(g)
        r = {n.id: n for n in result.nodes}["r"]
        assert isinstance(r, BinOp)
