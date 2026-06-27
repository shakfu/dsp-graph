"""Build a Max/MSP ``.maxpat`` test patch wrapping a transpiled gen~ codebox.

Given a gen-dsp :class:`~gen_dsp.graph.models.Graph` and its GenExpr source (from
``gen_dsp.graph.transpile.transpile_to_genexpr``), produce a ready-to-open Max
patch that drops the code into a self-contained ``gen.codebox~`` object and wires
a minimal test rig around it:

- each graph **param** becomes a bounded float parameter box -> ``name $1``
  message into the ``gen.codebox~`` left inlet (which sets the named Param), using
  the param name as emitted in the codebox (auto-renamed reserved words included);
- each graph **audio input** is fed from an ``sfplay~`` playing a built-in Max
  sound (``vibes-a1.aif``), preloaded on ``loadbang`` and started by a ``toggle``;
- the ``gen.codebox~`` outputs drive an ``ezdac~`` (a mono graph is sent to both
  channels).

The patch is built with py2max (a zero-dependency .maxpat generator) and returned
as a plain dict ready to ``json.dumps`` into a ``.maxpat`` file.
"""

from __future__ import annotations

import json
import re
from typing import Any

from gen_dsp.graph.models import Graph
from py2max import Patcher
from py2max.core import Rect

#: Built-in Max sound used as the default audio source for graphs with inputs.
DEFAULT_SOUND = "vibes-a1.aif"

# Matches a gen~ `Param <name>(...)` declaration emitted by the transpiler.
_PARAM_DECL = re.compile(r"^\s*Param\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def _emitted_param_names(genexpr_source: str) -> list[str]:
    """Param names as they appear in the codebox, in declaration order.

    The transpiler auto-renames params that collide with gen~ reserved words
    (e.g. ``mix`` -> ``mix_``), so the message boxes that set them must use these
    emitted names, not the original graph param names. The transpiler emits one
    ``Param`` per graph param in graph order, so the lists align by index.
    """
    return _PARAM_DECL.findall(genexpr_source)


# Layout constants (Max patch coordinates).
_COL = 100
_PARAM_Y = 20
_PLAYER_X = 40
_GEN_Y = 240
_DAC_Y = 470


def graph_to_maxpat(graph: Graph, genexpr_source: str) -> dict[str, Any]:
    """Build a Max patch wrapping *genexpr_source* and return it as a dict.

    *graph* supplies the structure (params, audio in/out counts) used to wire the
    test rig; *genexpr_source* is the gen~ codebox body to embed.
    """
    n_in = len(graph.inputs)
    n_out = len(graph.outputs)

    p = Patcher()

    # The transpiled code goes in a self-contained gen.codebox~ object (the form
    # gen transpilers emit), not a gen~ + inner codebox~ subpatcher. IO counts are
    # set from the graph so the param/signal/output patchlines attach to the right
    # inlets/outlets; at least one inlet is kept for the Param-setting messages
    # even when the graph has no audio inputs.
    gen = p.add_gen_codebox(
        genexpr_source,
        numinlets=max(n_in, 1),
        numoutlets=max(n_out, 1),
        patching_rect=Rect(_PLAYER_X, _GEN_Y, 480, 200),
    )

    # Params: a bounded float parameter box -> "name $1" message -> gen~ left
    # inlet (gen~ sets the named Param from that message). Use the param name as
    # emitted in the codebox (which may be auto-renamed, e.g. mix -> mix_) so the
    # message matches the Param the gen~ actually declares.
    emitted_names = _emitted_param_names(genexpr_source)
    for i, prm in enumerate(graph.params):
        name = emitted_names[i] if i < len(emitted_names) else prm.name
        x = 20 + i * _COL
        fp = p.add_floatparam(
            longname=name,
            initial=prm.default,
            minimum=prm.min,
            maximum=prm.max,
            rect=Rect(x, _PARAM_Y, 50, 22),
        )
        msg = p.add_message(f"{name} $1", patching_rect=Rect(x, _PARAM_Y + 40, 90, 22))
        p.add_line(fp, msg)
        p.add_line(msg, gen, inlet=0)

    # Audio inputs: a single sfplay~ playing a built-in sound feeds the inputs.
    if n_in > 0:
        px = _PLAYER_X + max(len(graph.params), 1) * _COL + 40
        loadbang = p.add_textbox("loadbang", patching_rect=Rect(px, _PARAM_Y, 60, 22))
        openmsg = p.add_message(
            f"open {DEFAULT_SOUND}", patching_rect=Rect(px, _PARAM_Y + 30, 130, 22)
        )
        toggle = p.add_textbox("toggle", patching_rect=Rect(px + 140, _PARAM_Y + 30, 24, 24))
        sfplay = p.add_textbox(f"sfplay~ {n_in}", patching_rect=Rect(px, _PARAM_Y + 70, 120, 22))
        p.add_line(loadbang, openmsg)
        p.add_line(openmsg, sfplay)
        p.add_line(toggle, sfplay)
        for i in range(n_in):
            # sfplay~ signal outlet i -> gen~ signal inlet i.
            p.add_line(sfplay, gen, inlet=i, outlet=i)

    # Outputs: gen~ -> ezdac~ (mono graph sent to both channels).
    dac = p.add_textbox("ezdac~", patching_rect=Rect(_PLAYER_X, _DAC_Y, 45, 45))
    if n_out == 1:
        p.add_line(gen, dac, inlet=0, outlet=0)
        p.add_line(gen, dac, inlet=1, outlet=0)
    else:
        for j in range(n_out):
            p.add_line(gen, dac, inlet=min(j, 1), outlet=j)

    # to_json() renders the patcher (populates boxes/lines) before serializing;
    # to_dict() alone would not, so round-trip through to_json() for the dict.
    return json.loads(p.to_json())  # type: ignore[no-any-return]
