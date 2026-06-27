"""CLI entry point: ``dsp-graph serve``."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser


def _open_browser(url: str) -> None:
    """Open *url* in the default browser after a short delay."""
    import time

    time.sleep(1.5)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dsp-graph", description="dsp-graph visual editor")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the web server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    serve_parser.add_argument("--open", action="store_true", help="Open browser on start")
    serve_parser.add_argument(
        "--experimental",
        action="store_true",
        help="Enable experimental features (e.g. the gen~/GenExpr transpiler tab)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        import os

        import uvicorn

        from dsp_graph.config import EXPERIMENTAL_ENV

        # Plumb the flag through the environment so it reaches the app process
        # (and uvicorn --reload child processes, which inherit the environment).
        if args.experimental:
            os.environ[EXPERIMENTAL_ENV] = "1"

        if args.open:
            url = f"http://{args.host}:{args.port}"
            threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

        uvicorn.run(
            "dsp_graph.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
