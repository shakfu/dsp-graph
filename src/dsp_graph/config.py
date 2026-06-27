"""Runtime feature configuration.

Read from the process environment at call time (not import time) so the server
process started by the CLI -- and uvicorn's ``--reload`` child processes, which
inherit the environment -- see the same flags, and so tests can toggle them with
``monkeypatch.setenv``.
"""

from __future__ import annotations

import os

#: Environment variable set by ``dsp-graph serve --experimental``.
EXPERIMENTAL_ENV = "DSP_GRAPH_EXPERIMENTAL"

_TRUTHY = {"1", "true", "yes", "on"}


def is_experimental() -> bool:
    """Whether experimental features (e.g. the GenExpr transpiler) are enabled.

    Experimental features are off by default and only exposed when the server is
    started with ``--experimental`` (which sets :data:`EXPERIMENTAL_ENV`).
    """
    return os.environ.get(EXPERIMENTAL_ENV, "").strip().lower() in _TRUTHY
