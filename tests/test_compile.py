"""Tests for C++ code generation from DSP graphs."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

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
    compile_graph,
    compile_graph_to_file,
)


class TestStructure:
    """Verify structural elements of generated C++."""

    def test_includes(self, stereo_gain_graph: Graph) -> None:
        code = compile_graph(stereo_gain_graph)
        assert "#include <cmath>" in code
        assert "#include <cstdlib>" in code
        assert "#include <cstdint>" in code

    def test_struct_name_pascal_case(self, stereo_gain_graph: Graph) -> None:
        code = compile_graph(stereo_gain_graph)
        assert "struct StereoGainState {" in code

    def test_function_signatures(self, stereo_gain_graph: Graph) -> None:
        code = compile_graph(stereo_gain_graph)
        assert "StereoGainState* stereo_gain_create(float sr)" in code
        assert "void stereo_gain_destroy(StereoGainState* self)" in code
        assert "stereo_gain_perform(StereoGainState* self, float** ins" in code
        assert "int stereo_gain_num_inputs(void)" in code
        assert "int stereo_gain_num_outputs(void)" in code
        assert "int stereo_gain_num_params(void)" in code

    def test_num_inputs_outputs(self, stereo_gain_graph: Graph) -> None:
        code = compile_graph(stereo_gain_graph)
        assert "return 2; }" in code  # 2 inputs and 2 outputs

    def test_single_name(self) -> None:
        g = Graph(
            name="gain",
            nodes=[Constant(id="c", value=1.0)],
            outputs=[AudioOutput(id="out1", source="c")],
        )
        code = compile_graph(g)
        assert "struct GainState {" in code
        assert "GainState* gain_create(float sr)" in code


class TestNodeEmission:
    """Verify per-node C++ code generation."""

    def test_binop_mul(self, stereo_gain_graph: Graph) -> None:
        code = compile_graph(stereo_gain_graph)
        assert "float scaled1 = in1[i] * gain;" in code

    def test_binop_sub_with_literal(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "float inv_coeff = 1.0f - coeff;" in code

    def test_binop_add(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "float result = dry + wet;" in code

    def test_binop_min(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[BinOp(id="r", op="min", a="in1", b=1.0)],
        )
        code = compile_graph(g)
        assert "float r = fminf(in1[i], 1.0f);" in code

    def test_binop_max(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[BinOp(id="r", op="max", a="in1", b=0.0)],
        )
        code = compile_graph(g)
        assert "float r = fmaxf(in1[i], 0.0f);" in code

    def test_binop_mod(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[BinOp(id="r", op="mod", a="in1", b=2.0)],
        )
        code = compile_graph(g)
        assert "float r = fmodf(in1[i], 2.0f);" in code

    def test_binop_pow(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[BinOp(id="r", op="pow", a="in1", b=2.0)],
        )
        code = compile_graph(g)
        assert "float r = powf(in1[i], 2.0f);" in code

    def test_unaryop_sin(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="s")],
            nodes=[UnaryOp(id="s", op="sin", a="in1")],
        )
        code = compile_graph(g)
        assert "float s = sinf(in1[i]);" in code

    def test_unaryop_neg(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="n")],
            nodes=[UnaryOp(id="n", op="neg", a="in1")],
        )
        code = compile_graph(g)
        assert "float n = -in1[i];" in code

    def test_unaryop_floor(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[UnaryOp(id="r", op="floor", a="in1")],
        )
        code = compile_graph(g)
        assert "float r = floorf(in1[i]);" in code

    def test_unaryop_ceil(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[UnaryOp(id="r", op="ceil", a="in1")],
        )
        code = compile_graph(g)
        assert "float r = ceilf(in1[i]);" in code

    def test_unaryop_round(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[UnaryOp(id="r", op="round", a="in1")],
        )
        code = compile_graph(g)
        assert "float r = roundf(in1[i]);" in code

    def test_unaryop_sign(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[UnaryOp(id="r", op="sign", a="in1")],
        )
        code = compile_graph(g)
        assert "in1[i] > 0.0f ? 1.0f" in code
        assert "in1[i] < 0.0f ? -1.0f : 0.0f" in code

    def test_clamp(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="c")],
            nodes=[Clamp(id="c", a="in1")],
        )
        code = compile_graph(g)
        assert "fminf(fmaxf(in1[i], 0.0f), 1.0f)" in code

    def test_constant(self) -> None:
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="c")],
            nodes=[Constant(id="c", value=3.14)],
        )
        code = compile_graph(g)
        assert "float c = 3.14f;" in code

    def test_phasor(self) -> None:
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="p")],
            params=[Param(name="freq", min=0.0, max=20000.0, default=440.0)],
            nodes=[Phasor(id="p", freq="freq")],
        )
        code = compile_graph(g)
        assert "float p = p_phase;" in code
        assert "p_phase += freq / sr;" in code
        assert "m_p_phase" in code

    def test_noise(self) -> None:
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="n")],
            nodes=[Noise(id="n")],
        )
        code = compile_graph(g)
        assert "1664525u" in code
        assert "1013904223u" in code
        assert "m_n_seed" in code

    def test_compare(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="c")],
            nodes=[Compare(id="c", op="gt", a="in1", b=0.0)],
        )
        code = compile_graph(g)
        assert "float c = (float)(in1[i] > 0.0f);" in code

    def test_compare_all_ops(self) -> None:
        expected = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "=="}
        for op, sym in expected.items():
            g = Graph(
                name="test",
                inputs=[AudioInput(id="in1")],
                outputs=[AudioOutput(id="out1", source="c")],
                nodes=[Compare(id="c", op=op, a="in1", b=0.0)],
            )
            code = compile_graph(g)
            assert f"(float)(in1[i] {sym} 0.0f)" in code

    def test_select(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="s")],
            nodes=[
                Compare(id="cond", op="gt", a="in1", b=0.0),
                Select(id="s", cond="cond", a="in1", b=0.0),
            ],
        )
        code = compile_graph(g)
        assert "float s = cond > 0.0f ? in1[i] : 0.0f;" in code

    def test_wrap(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="w")],
            nodes=[Wrap(id="w", a="in1")],
        )
        code = compile_graph(g)
        assert "w_range" in code
        assert "fmodf" in code
        assert "w_raw" in code

    def test_fold(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="f")],
            nodes=[Fold(id="f", a="in1")],
        )
        code = compile_graph(g)
        assert "f_range" in code
        assert "f_t" in code
        assert "fmodf" in code

    def test_mix(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="m")],
            nodes=[
                Constant(id="one", value=1.0),
                Mix(id="m", a="in1", b="one", t=0.5),
            ],
        )
        code = compile_graph(g)
        assert "float m = in1[i] + (one - in1[i]) * 0.5f;" in code


class TestDeltaChangeState:
    """Verify Delta and Change state management."""

    def test_delta_struct_field(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="d")],
            nodes=[Delta(id="d", a="in1")],
        )
        code = compile_graph(g)
        assert "float m_d_prev;" in code

    def test_delta_init(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="d")],
            nodes=[Delta(id="d", a="in1")],
        )
        code = compile_graph(g)
        assert "self->m_d_prev = 0.0f;" in code

    def test_delta_load_save(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="d")],
            nodes=[Delta(id="d", a="in1")],
        )
        code = compile_graph(g)
        assert "float d_prev = self->m_d_prev;" in code
        assert "self->m_d_prev = d_prev;" in code

    def test_delta_compute(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="d")],
            nodes=[Delta(id="d", a="in1")],
        )
        code = compile_graph(g)
        assert "float d_cur = in1[i];" in code
        assert "float d = d_cur - d_prev;" in code
        assert "d_prev = d_cur;" in code

    def test_change_struct_field(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="c")],
            nodes=[Change(id="c", a="in1")],
        )
        code = compile_graph(g)
        assert "float m_c_prev;" in code

    def test_change_compute(self) -> None:
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="c")],
            nodes=[Change(id="c", a="in1")],
        )
        code = compile_graph(g)
        assert "float c_cur = in1[i];" in code
        assert "(c_cur != c_prev) ? 1.0f : 0.0f" in code
        assert "c_prev = c_cur;" in code


class TestInterpolatedDelayRead:
    """Verify interpolated delay read code generation."""

    def _make_delay_graph(self, interp: str) -> Graph:
        return Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="rd")],
            params=[Param(name="tap_pos", min=0.0, max=48000.0, default=100.0)],
            nodes=[
                DelayLine(id="dl", max_samples=48000),
                DelayRead(id="rd", delay="dl", tap="tap_pos", interp=interp),
                DelayWrite(id="dw", delay="dl", value="in1"),
            ],
        )

    def test_none_interp(self) -> None:
        code = compile_graph(self._make_delay_graph("none"))
        assert "rd_pos" in code
        assert "dl_buf[rd_pos]" in code
        # Should NOT have fractional tap
        assert "rd_ftap" not in code

    def test_linear_interp(self) -> None:
        code = compile_graph(self._make_delay_graph("linear"))
        assert "rd_ftap" in code
        assert "rd_itap" in code
        assert "rd_frac" in code
        assert "rd_i0" in code
        assert "rd_i1" in code
        # Linear interpolation formula
        assert "dl_buf[rd_i0]" in code
        assert "dl_buf[rd_i1]" in code

    def test_cubic_interp(self) -> None:
        code = compile_graph(self._make_delay_graph("cubic"))
        assert "rd_ftap" in code
        assert "rd_ym1" in code
        assert "rd_y0" in code
        assert "rd_y1" in code
        assert "rd_y2" in code
        assert "rd_c0" in code
        assert "rd_c1" in code
        assert "rd_c2" in code
        assert "rd_c3" in code


class TestHistoryState:
    """Verify History node state management."""

    def test_history_struct_field(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "float m_prev;" in code

    def test_history_preloop_load(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "float prev = self->m_prev;" in code

    def test_history_postloop_save(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "self->m_prev = prev;" in code

    def test_history_writeback_in_loop(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        # History write-back: prev = result; (inside loop, after node computations)
        assert "        prev = result;" in code


class TestDelayLine:
    """Verify delay line code generation."""

    def test_delay_struct_fields(self, fbdelay_graph: Graph) -> None:
        code = compile_graph(fbdelay_graph)
        assert "float* m_dline_buf;" in code
        assert "int m_dline_len;" in code
        assert "int m_dline_wr;" in code

    def test_delay_calloc(self, fbdelay_graph: Graph) -> None:
        code = compile_graph(fbdelay_graph)
        assert "calloc(48000, sizeof(float))" in code

    def test_delay_free(self, fbdelay_graph: Graph) -> None:
        code = compile_graph(fbdelay_graph)
        assert "free(self->m_dline_buf);" in code

    def test_delay_write_pointer(self, fbdelay_graph: Graph) -> None:
        code = compile_graph(fbdelay_graph)
        assert "dline_buf[dline_wr]" in code
        assert "dline_wr = (dline_wr + 1) % dline_len;" in code

    def test_delay_read(self, fbdelay_graph: Graph) -> None:
        code = compile_graph(fbdelay_graph)
        assert "dline_wr - (int)(tap)" in code
        assert "dline_buf[delayed_pos]" in code


class TestParamAPI:
    """Verify parameter introspection functions."""

    def test_param_name(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert 'return "coeff";' in code

    def test_param_min_max(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "return 0.0f;" in code  # min
        assert "return 0.999f;" in code  # max

    def test_set_param(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "self->p_coeff = value;" in code

    def test_get_param(self, onepole_graph: Graph) -> None:
        code = compile_graph(onepole_graph)
        assert "return self->p_coeff;" in code

    def test_multiple_params(self, fbdelay_graph: Graph) -> None:
        code = compile_graph(fbdelay_graph)
        assert 'return "delay_ms";' in code
        assert 'return "feedback";' in code
        assert 'return "mix";' in code


class TestEdgeCases:
    """Error conditions and edge cases."""

    def test_empty_graph_compiles(self) -> None:
        g = Graph(name="empty")
        code = compile_graph(g)
        assert "struct EmptyState {" in code
        assert "return 0; }" in code

    def test_invalid_graph_raises(self) -> None:
        g = Graph(
            name="bad",
            nodes=[BinOp(id="a", op="add", a="missing", b=0.0)],
            outputs=[AudioOutput(id="out1", source="a")],
        )
        with pytest.raises(ValueError, match="Invalid graph"):
            compile_graph(g)

    def test_invalid_c_identifier_raises(self) -> None:
        g = Graph(
            name="test",
            nodes=[Constant(id="my-node", value=1.0)],
            outputs=[AudioOutput(id="out1", source="my-node")],
        )
        with pytest.raises(ValueError, match="not a valid C identifier"):
            compile_graph(g)

    def test_invalid_param_name_raises(self) -> None:
        g = Graph(
            name="test",
            params=[Param(name="my param")],
            nodes=[Constant(id="c", value=1.0)],
            outputs=[AudioOutput(id="out1", source="c")],
        )
        with pytest.raises(ValueError, match="not a valid C identifier"):
            compile_graph(g)


class TestCompileToFile:
    """Verify compile_graph_to_file writes to disk."""

    def test_writes_file(self, stereo_gain_graph: Graph, tmp_path: Path) -> None:
        out = compile_graph_to_file(stereo_gain_graph, tmp_path / "build")
        assert out.exists()
        assert out.name == "stereo_gain.cpp"
        assert "StereoGainState" in out.read_text()

    def test_creates_directory(self, onepole_graph: Graph, tmp_path: Path) -> None:
        build_dir = tmp_path / "nested" / "build"
        assert not build_dir.exists()
        out = compile_graph_to_file(onepole_graph, build_dir)
        assert build_dir.is_dir()
        assert out == build_dir / "onepole.cpp"

    def test_returns_path(self, fbdelay_graph: Graph, tmp_path: Path) -> None:
        out = compile_graph_to_file(fbdelay_graph, tmp_path)
        assert isinstance(out, Path)
        assert out.parent == tmp_path


@pytest.mark.skipif(not shutil.which("g++"), reason="g++ not available")
class TestGccCompilation:
    """Integration: verify generated C++ compiles with g++."""

    def _compile_check(self, graph: Graph) -> None:
        code = compile_graph(graph)
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ["g++", "-std=c++17", "-c", "-o", "/dev/null", "-x", "c++", f.name],
                capture_output=True,
                text=True,
            )
            Path(f.name).unlink()
        assert result.returncode == 0, f"g++ failed:\n{result.stderr}"

    def test_stereo_gain_compiles(self, stereo_gain_graph: Graph) -> None:
        self._compile_check(stereo_gain_graph)

    def test_onepole_compiles(self, onepole_graph: Graph) -> None:
        self._compile_check(onepole_graph)

    def test_fbdelay_compiles(self, fbdelay_graph: Graph) -> None:
        self._compile_check(fbdelay_graph)

    def test_all_node_types_compile(self) -> None:
        """Graph exercising every node type compiles."""
        g = Graph(
            name="all_nodes",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="clamped")],
            params=[Param(name="freq", min=0.0, max=20000.0, default=440.0)],
            nodes=[
                Constant(id="half", value=0.5),
                BinOp(id="scaled", op="mul", a="in1", b="half"),
                BinOp(id="mn", op="min", a="in1", b="half"),
                BinOp(id="mx", op="max", a="in1", b="half"),
                BinOp(id="md", op="mod", a="in1", b="half"),
                BinOp(id="pw", op="pow", a="in1", b="half"),
                UnaryOp(id="shaped", op="tanh", a="scaled"),
                UnaryOp(id="fl", op="floor", a="in1"),
                UnaryOp(id="cl", op="ceil", a="in1"),
                UnaryOp(id="rn", op="round", a="in1"),
                UnaryOp(id="sg", op="sign", a="in1"),
                History(id="prev", input="clamped"),
                BinOp(id="mixed", op="add", a="shaped", b="prev"),
                Clamp(id="clamped", a="mixed"),
                Phasor(id="lfo", freq="freq"),
                Noise(id="noise"),
                DelayLine(id="dl", max_samples=4800),
                DelayRead(id="tap_out", delay="dl", tap="half"),
                DelayRead(id="tap_lin", delay="dl", tap="half", interp="linear"),
                DelayRead(id="tap_cub", delay="dl", tap="half", interp="cubic"),
                DelayWrite(id="dl_wr", delay="dl", value="scaled"),
                Compare(id="cmp", op="gt", a="in1", b=0.0),
                Select(id="sel", cond="cmp", a="in1", b=0.0),
                Wrap(id="wr", a="in1"),
                Fold(id="fo", a="in1"),
                Mix(id="mxn", a="in1", b="half", t=0.5),
                Delta(id="dt", a="in1"),
                Change(id="ch", a="in1"),
            ],
        )
        self._compile_check(g)
