"""Command-line interface for dsp-graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from dsp_graph.compile import compile_graph, compile_graph_to_file
from dsp_graph.gen_dsp_adapter import SUPPORTED_PLATFORMS, compile_for_gen_dsp
from dsp_graph.models import Graph
from dsp_graph.optimize import optimize_graph
from dsp_graph.validate import validate_graph
from dsp_graph.visualize import graph_to_dot, graph_to_dot_file


def _load_graph(path: str) -> Graph:
    """Load and parse a graph JSON file."""
    text = Path(path).read_text()
    data = json.loads(text)
    return Graph.model_validate(data)


def _cmd_compile(args: argparse.Namespace) -> int:
    graph = _load_graph(args.file)
    if args.optimize:
        graph = optimize_graph(graph)
    if args.gen_dsp:
        if not args.output:
            print("error: --gen-dsp requires -o/--output", file=sys.stderr)
            return 1
        compile_for_gen_dsp(graph, args.output, args.gen_dsp)
        return 0
    if args.output:
        compile_graph_to_file(graph, args.output)
    else:
        sys.stdout.write(compile_graph(graph))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    graph = _load_graph(args.file)
    errors = validate_graph(graph)
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    print("valid")
    return 0


def _cmd_dot(args: argparse.Namespace) -> int:
    graph = _load_graph(args.file)
    if args.output:
        graph_to_dot_file(graph, args.output)
    else:
        sys.stdout.write(graph_to_dot(graph))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the dsp-graph CLI."""
    parser = argparse.ArgumentParser(
        prog="dsp-graph",
        description="Compile, validate, and visualize DSP signal graphs.",
    )
    sub = parser.add_subparsers(dest="command")

    # compile
    p_compile = sub.add_parser("compile", help="Compile graph to C++")
    p_compile.add_argument("file", help="Graph JSON file")
    p_compile.add_argument("-o", "--output", help="Output directory")
    p_compile.add_argument("--optimize", action="store_true", help="Apply optimization passes")
    p_compile.add_argument(
        "--gen-dsp",
        metavar="PLATFORM",
        choices=sorted(SUPPORTED_PLATFORMS),
        help="Generate gen-dsp adapter for PLATFORM",
    )

    # validate
    p_validate = sub.add_parser("validate", help="Validate graph JSON")
    p_validate.add_argument("file", help="Graph JSON file")

    # dot
    p_dot = sub.add_parser("dot", help="Generate DOT visualization")
    p_dot.add_argument("file", help="Graph JSON file")
    p_dot.add_argument("-o", "--output", help="Output directory")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return 1

    try:
        if args.command == "compile":
            return _cmd_compile(args)
        elif args.command == "validate":
            return _cmd_validate(args)
        elif args.command == "dot":
            return _cmd_dot(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON: {e}", file=sys.stderr)
        return 1
    except ValidationError as e:
        print(f"error: invalid graph: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
