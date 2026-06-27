"""Tests for the /api/graph/export/maxpat endpoint and the maxpat builder."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from gen_dsp.graph.models import Graph
from gen_dsp.graph.transpile import transpile_to_genexpr

from dsp_graph.config import EXPERIMENTAL_ENV
from dsp_graph.maxpat import DEFAULT_SOUND, graph_to_maxpat


@pytest.fixture
def experimental_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXPERIMENTAL_ENV, "1")


@pytest.fixture
def experimental_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)


def _boxes(patch: dict[str, Any]) -> list[dict[str, Any]]:
    return [b["box"] for b in patch["patcher"]["boxes"]]


def _gen_box(patch: dict[str, Any]) -> dict[str, Any]:
    return next(b for b in _boxes(patch) if b["maxclass"] == "gen.codebox~")


class TestMaxpatBuilder:
    """The graph_to_maxpat() builder (unit-level, no HTTP)."""

    def test_synth_patch_structure(self, stereo_gain: Graph) -> None:
        """A graph wraps its code in a self-contained gen.codebox~ feeding ezdac~."""
        patch = graph_to_maxpat(stereo_gain, transpile_to_genexpr(stereo_gain))
        gen = _gen_box(patch)
        # The transpiled code lives directly on the gen.codebox~ (no subpatcher).
        assert "patcher" not in gen
        assert "scaled1 = in1 * gain;" in gen["code"]
        # gen.codebox~ IO counts match the graph (2 in, 2 out).
        assert gen["numinlets"] == 2
        assert gen["numoutlets"] == 2
        # Exactly one bounded float param control per graph param.
        flonums = [b for b in _boxes(patch) if b["maxclass"] == "flonum"]
        assert len(flonums) == len(stereo_gain.params)
        names = {b["saved_attribute_attributes"]["valueof"]["parameter_longname"] for b in flonums}
        assert names == {"gain"}
        assert any(b["maxclass"] == "ezdac~" for b in _boxes(patch))

    def test_inputs_get_soundfile_player(self, onepole: Graph) -> None:
        """A graph with audio inputs gets an sfplay~ on a built-in sound."""
        patch = graph_to_maxpat(onepole, transpile_to_genexpr(onepole))
        texts = [b.get("text", "") for b in _boxes(patch)]
        assert any(t.startswith("sfplay~") for t in texts)
        assert any(f"open {DEFAULT_SOUND}" in t for t in texts)

    def test_no_inputs_no_player(self, phasor_graph: Graph) -> None:
        """A zero-input synth graph has no soundfile player."""
        patch = graph_to_maxpat(phasor_graph, transpile_to_genexpr(phasor_graph))
        texts = [b.get("text", "") for b in _boxes(patch)]
        assert not any(t.startswith("sfplay~") for t in texts)

    def test_param_messages_use_emitted_codebox_names(self, stereo_gain: Graph) -> None:
        """Param messages must match the (possibly auto-renamed) codebox Param.

        The transpiler renames params that collide with gen~ reserved words
        (e.g. ``mix`` -> ``mix_``); the ``name $1`` message must use the emitted
        name or gen~ reports 'doesn't understand'. The graph here has one param
        ``gain``; a hand-crafted codebox renames it to ``gain_`` to exercise the
        alignment independent of the installed gen-dsp version.
        """
        code = "Param gain_(1.0, min=0.0, max=2.0);\nout1 = in1 * gain_;\n"
        patch = graph_to_maxpat(stereo_gain, code)
        messages = [b["text"] for b in _boxes(patch) if b["maxclass"] == "message"]
        assert "gain_ $1" in messages
        assert "gain $1" not in messages


class TestMaxpatEndpoint:
    def test_export_maxpat_valid(
        self, client: TestClient, stereo_gain_json: dict[str, Any], experimental_on: None
    ) -> None:
        resp = client.post("/api/graph/export/maxpat", json={"graph": stereo_gain_json})
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "stereo_gain.maxpat"
        # maxpat_json is a valid Max patcher document containing the gen.codebox~.
        patch = json.loads(data["maxpat_json"])
        assert "patcher" in patch
        assert any(b["box"]["maxclass"] == "gen.codebox~" for b in patch["patcher"]["boxes"])

    def test_export_maxpat_disabled_by_default(
        self, client: TestClient, stereo_gain_json: dict[str, Any], experimental_off: None
    ) -> None:
        resp = client.post("/api/graph/export/maxpat", json={"graph": stereo_gain_json})
        assert resp.status_code == 404
        assert "experimental" in resp.json()["detail"].lower()

    def test_export_maxpat_invalid_graph(self, client: TestClient, experimental_on: None) -> None:
        resp = client.post(
            "/api/graph/export/maxpat",
            json={"graph": {"name": "bad", "nodes": [{"op": "nonexistent"}]}},
        )
        assert resp.status_code == 422

    def test_export_maxpat_untranspilable(self, client: TestClient, experimental_on: None) -> None:
        """A graph the transpiler rejects (fastexp) surfaces as 400."""
        graph = {
            "name": "unsupported",
            "inputs": [{"id": "in1"}],
            "outputs": [{"id": "out1", "source": "r"}],
            "nodes": [{"id": "r", "op": "fastexp", "a": "in1"}],
        }
        resp = client.post("/api/graph/export/maxpat", json={"graph": graph})
        assert resp.status_code == 400
