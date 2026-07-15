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

#: Environment variable set by ``dsp-graph serve --disable-build``.
DISABLE_BUILD_ENV = "DSP_GRAPH_DISABLE_BUILD"

_TRUTHY = {"1", "true", "yes", "on"}


def is_experimental() -> bool:
    """Whether experimental features (e.g. the GenExpr transpiler) are enabled.

    Experimental features are off by default and only exposed when the server is
    started with ``--experimental`` (which sets :data:`EXPERIMENTAL_ENV`).
    """
    return os.environ.get(EXPERIMENTAL_ENV, "").strip().lower() in _TRUTHY


def is_build_enabled() -> bool:
    """Whether the native build endpoints (``/api/build*``) are enabled.

    Native compilation invokes a host toolchain and writes to disk. It is the
    app's core author->build value chain, so it is **on by default**, but it can
    be turned off with ``--disable-build`` (which sets :data:`DISABLE_BUILD_ENV`)
    for a hardened, inspect-only deployment.
    """
    return os.environ.get(DISABLE_BUILD_ENV, "").strip().lower() not in _TRUTHY
