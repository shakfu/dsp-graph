from __future__ import annotations

from dsp_graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    DelayLine,
    DelayRead,
    DelayWrite,
    Graph,
    History,
    Param,
    validate_graph,
)

# ---------------------------------------------------------------------------
# Valid graphs
# ---------------------------------------------------------------------------


class TestValidGraphs:
    def test_stereo_gain_valid(self, stereo_gain_graph: Graph) -> None:
        assert validate_graph(stereo_gain_graph) == []

    def test_onepole_valid(self, onepole_graph: Graph) -> None:
        assert validate_graph(onepole_graph) == []

    def test_fbdelay_valid(self, fbdelay_graph: Graph) -> None:
        assert validate_graph(fbdelay_graph) == []

    def test_empty_graph_valid(self) -> None:
        g = Graph(name="empty")
        assert validate_graph(g) == []


# ---------------------------------------------------------------------------
# Duplicate IDs
# ---------------------------------------------------------------------------


class TestDuplicateIds:
    def test_duplicate_node_id(self) -> None:
        g = Graph(
            name="test",
            nodes=[
                BinOp(id="x", op="add", a=1.0, b=2.0),
                BinOp(id="x", op="mul", a=3.0, b=4.0),
            ],
        )
        errors = validate_graph(g)
        assert any("Duplicate node ID" in e for e in errors)

    def test_node_id_collides_with_input(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            nodes=[BinOp(id="in1", op="add", a=1.0, b=2.0)],
        )
        errors = validate_graph(g)
        assert any("collides with audio input" in e for e in errors)

    def test_node_id_collides_with_param(self) -> None:
        g = Graph(
            name="test",
            params=[Param(name="gain")],
            nodes=[BinOp(id="gain", op="add", a=1.0, b=2.0)],
        )
        errors = validate_graph(g)
        assert any("collides with param" in e for e in errors)


# ---------------------------------------------------------------------------
# Dangling references
# ---------------------------------------------------------------------------


class TestDanglingReferences:
    def test_node_references_unknown_id(self) -> None:
        g = Graph(
            name="test",
            nodes=[BinOp(id="x", op="add", a="nonexistent", b=1.0)],
        )
        errors = validate_graph(g)
        assert any("unknown ID 'nonexistent'" in e for e in errors)

    def test_output_references_unknown_node(self) -> None:
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="nonexistent")],
        )
        errors = validate_graph(g)
        assert any("does not reference a node" in e for e in errors)

    def test_literal_float_not_flagged(self) -> None:
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="x")],
            nodes=[BinOp(id="x", op="add", a=1.0, b=2.0)],
        )
        assert validate_graph(g) == []


# ---------------------------------------------------------------------------
# Delay consistency
# ---------------------------------------------------------------------------


class TestDelayConsistency:
    def test_delay_read_references_nonexistent_line(self) -> None:
        g = Graph(
            name="test",
            nodes=[DelayRead(id="dr", delay="missing", tap=100.0)],
        )
        errors = validate_graph(g)
        assert any("non-existent delay line 'missing'" in e for e in errors)

    def test_delay_write_references_nonexistent_line(self) -> None:
        g = Graph(
            name="test",
            nodes=[DelayWrite(id="dw", delay="missing", value=1.0)],
        )
        errors = validate_graph(g)
        assert any("non-existent delay line 'missing'" in e for e in errors)

    def test_valid_delay_references(self) -> None:
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="dr")],
            nodes=[
                DelayLine(id="dl"),
                DelayRead(id="dr", delay="dl", tap=100.0),
                DelayWrite(id="dw", delay="dl", value=1.0),
            ],
        )
        assert validate_graph(g) == []


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_pure_cycle_detected(self) -> None:
        """add -> mul -> add with no history should be flagged."""
        g = Graph(
            name="test",
            nodes=[
                BinOp(id="a", op="add", a="b", b=1.0),
                BinOp(id="b", op="mul", a="a", b=2.0),
            ],
        )
        errors = validate_graph(g)
        assert any("cycle" in e.lower() for e in errors)

    def test_cycle_through_history_allowed(self) -> None:
        """result -> History -> result is a valid feedback loop."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="result")],
            nodes=[
                History(id="prev", input="result", init=0.0),
                BinOp(id="result", op="add", a="in1", b="prev"),
            ],
        )
        assert validate_graph(g) == []

    def test_cycle_through_delay_allowed(self) -> None:
        """Delay read/write loop is valid feedback."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="delayed")],
            nodes=[
                DelayLine(id="dl"),
                DelayRead(id="delayed", delay="dl", tap=100.0),
                BinOp(id="fb", op="mul", a="delayed", b=0.5),
                BinOp(id="sum", op="add", a="in1", b="fb"),
                DelayWrite(id="dw", delay="dl", value="sum"),
            ],
        )
        assert validate_graph(g) == []

    def test_three_node_cycle(self) -> None:
        """a -> b -> c -> a without feedback."""
        g = Graph(
            name="test",
            nodes=[
                BinOp(id="a", op="add", a="c", b=1.0),
                BinOp(id="b", op="mul", a="a", b=1.0),
                BinOp(id="c", op="sub", a="b", b=1.0),
            ],
        )
        errors = validate_graph(g)
        assert any("cycle" in e.lower() for e in errors)
