# dsp-graph

Visual graph editor, inspector, and debugger for [gen-dsp](https://github.com/shakfu/gen-dsp) signal-processing graphs.

## Overview

dsp-graph provides a web UI (React + React Flow) backed by a Python server (FastAPI) for loading, visualizing, simulating, optimizing, and inspecting DSP signal graphs defined with gen-dsp's graph DSL.

**Features:**

- Load graphs from JSON or `.gdsp` DSL source
- Interactive graph visualization with React Flow (pan, zoom, minimap)
- Node inspector showing all properties
- Per-sample simulation with output display
- Multi-pass graph optimization with before/after comparison
- C++ code generation preview
- Graphviz DOT export
- SVG export of the current canvas view
- Export/roundtrip back to graph JSON
- Auto fit-and-center on graph load

## Installation

```bash
pip install dsp-graph
```

For simulation support:

```bash
pip install dsp-graph[sim]
```

## Quick Start

```bash
dsp-graph serve
```

Opens the editor at `http://127.0.0.1:8765`. Load a gen-dsp graph JSON file via the toolbar.

### CLI Options

```
dsp-graph serve [--host HOST] [--port PORT] [--reload] [--open]
```

- `--host`: Bind address (default: 127.0.0.1)
- `--port`: Port (default: 8765)
- `--reload`: Auto-reload on code changes
- `--open`: Open browser on start

## Development

```bash
# Install dev dependencies
make install-dev

# Run backend tests
make test

# Install frontend dependencies
make frontend-install

# Run frontend dev server (hot reload, proxies /api to backend)
make frontend-dev

# Build frontend (outputs to src/dsp_graph/static/)
make frontend-build

# Run full QA (test + lint + typecheck + format)
make qa

# Start server with auto-reload
make dev
```

## Architecture

### Backend (Python)

- `convert.py` -- Core Graph <-> ReactFlow conversion layer
- `server.py` -- FastAPI app with SPA static serving
- `cli.py` -- CLI entry point (`dsp-graph serve`)
- `api/graph.py` -- Graph load/validate/export/catalog endpoints
- `api/simulate.py` -- Per-sample simulation endpoint
- `api/optimize.py` -- Multi-pass optimization endpoint
- `api/compile.py` -- C++ code generation endpoint
- `api/layout.py` -- Auto-layout endpoint

### Frontend (React + TypeScript)

- React Flow (xyflow) for graph visualization
- Zustand for state management
- Custom node types: input, output, param, dsp_node
- Vite build, outputs to `src/dsp_graph/static/`

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/graph/load/json` | POST | Load Graph JSON -> ReactFlow |
| `/api/graph/load/gdsp` | POST | Parse .gdsp source -> ReactFlow |
| `/api/graph/validate` | POST | Validate a Graph |
| `/api/graph/export/json` | POST | ReactFlow -> Graph JSON |
| `/api/graph/node-types` | GET | Node type catalog + colors |
| `/api/graph/dot` | POST | Graph -> Graphviz DOT |
| `/api/simulate` | POST | Run per-sample simulation |
| `/api/optimize` | POST | Optimize graph (before/after) |
| `/api/compile` | POST | Graph -> C++ source |
| `/api/layout` | POST | Auto-layout ReactFlow nodes |

## License

MIT
