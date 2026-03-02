"""Tests for Graph <-> ReactFlow conversion."""

from __future__ import annotations

from gen_dsp.graph.models import Graph

from dsp_graph.convert import (
    INPUT_COLOR,
    OP_COLORS,
    OUTPUT_COLOR,
    PARAM_COLOR,
    graph_to_reactflow,
    reactflow_to_graph,
)


class TestGraphToReactflow:
    def test_node_count(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        # 2 inputs + 2 outputs + 1 param + 2 nodes = 7
        assert len(rf.nodes) == 7

    def test_edge_count(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        # scaled1 <- in1, scaled1 <- gain, scaled2 <- in2, scaled2 <- gain
        # out1 <- scaled1, out2 <- scaled2
        assert len(rf.edges) == 6

    def test_node_types(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        types = {n.id: n.type for n in rf.nodes}
        assert types["in1"] == "input"
        assert types["in2"] == "input"
        assert types["out1"] == "output"
        assert types["out2"] == "output"
        assert types["gain"] == "param"
        assert types["scaled1"] == "dsp_node"
        assert types["scaled2"] == "dsp_node"

    def test_node_colors(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        colors = {n.id: n.data.color for n in rf.nodes}
        assert colors["in1"] == INPUT_COLOR
        assert colors["out1"] == OUTPUT_COLOR
        assert colors["gain"] == PARAM_COLOR
        assert colors["scaled1"] == OP_COLORS["mul"]

    def test_feedback_edge_marking(self, onepole: Graph) -> None:
        rf = graph_to_reactflow(onepole)
        animated_edges = [e for e in rf.edges if e.animated]
        assert len(animated_edges) == 1
        # The feedback edge should be from result -> prev (History.input)
        fb = animated_edges[0]
        assert fb.source == "result"
        assert fb.target == "prev"

    def test_graph_metadata(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        assert rf.name == "stereo_gain"
        assert rf.sample_rate == 44100

    def test_positions_exist(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        for node in rf.nodes:
            assert "x" in node.position
            assert "y" in node.position

    def test_dsp_node_has_node_data(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        dsp_nodes = [n for n in rf.nodes if n.type == "dsp_node"]
        for n in dsp_nodes:
            assert n.data.node_data is not None
            assert "op" in n.data.node_data


class TestRoundtrip:
    def test_stereo_gain_roundtrip(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        g2 = reactflow_to_graph(rf)
        assert g2.name == stereo_gain.name
        assert len(g2.inputs) == len(stereo_gain.inputs)
        assert len(g2.outputs) == len(stereo_gain.outputs)
        assert len(g2.params) == len(stereo_gain.params)
        assert len(g2.nodes) == len(stereo_gain.nodes)
        assert {n.id for n in g2.nodes} == {n.id for n in stereo_gain.nodes}

    def test_onepole_roundtrip(self, onepole: Graph) -> None:
        rf = graph_to_reactflow(onepole)
        g2 = reactflow_to_graph(rf)
        assert g2.name == "onepole"
        assert len(g2.nodes) == len(onepole.nodes)
        assert g2.params[0].name == "coeff"

    def test_roundtrip_preserves_param_values(self, stereo_gain: Graph) -> None:
        rf = graph_to_reactflow(stereo_gain)
        g2 = reactflow_to_graph(rf)
        p = g2.params[0]
        assert p.name == "gain"
        assert p.min == 0.0
        assert p.max == 2.0
        assert p.default == 1.0


class TestColorMap:
    def test_common_ops_have_colors(self) -> None:
        common = ["add", "sub", "mul", "div", "history", "phasor", "sinosc", "constant"]
        for op in common:
            assert op in OP_COLORS, f"Missing color for op: {op}"

    def test_colors_are_hex(self) -> None:
        for op, color in OP_COLORS.items():
            assert color.startswith("#"), f"Color for {op} is not hex: {color}"
            assert len(color) == 7, f"Color for {op} has wrong length: {color}"
