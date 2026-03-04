# dsp-graph

Visual graph editor, inspector, and debugger for [gen-dsp](https://github.com/shakfu/gen-dsp) signal-processing graphs.

## Overview

dsp-graph provides a web UI (React + React Flow) backed by a Python server (FastAPI) for loading, visualizing, simulating, optimizing, building, and inspecting DSP signal graphs defined with gen-dsp's graph DSL.

**Features:**

- Load graphs from JSON or `.gdsp` DSL source (via toolbar or drag-and-drop)
- `.gdsp` editor with syntax highlighting, autocomplete, go-to-definition, and live preview
- Interactive graph visualization with React Flow (pan, zoom, minimap)
- Auto fit-to-view on graph load, topology change, and container resize
- Node inspector showing all properties
- Per-sample simulation with output display
- Multi-pass graph optimization with before/after comparison
- C++ code generation preview with copy/download
- Binary plugin build for multiple platforms (CLAP, VST3, AU, LV2, etc.)
- Batch multi-platform build with zip download
- Content-addressed build cache (avoids recompilation for unchanged graphs)
- Graphviz DOT export
- SVG export of the current canvas view
- Export/roundtrip back to graph JSON and `.gdsp`

## Installation

```bash
pip install dsp-graph
```

Or from source:

```bash
git clone https://github.com/shakfu/dsp-graph.git
cd dsp-graph
make install-dev
make frontend-build
```

## Quick Start

```bash
dsp-graph serve
```

Starts the server at `http://127.0.0.1:8765`. Open that URL in a browser, then load a gen-dsp graph JSON or `.gdsp` file via the toolbar or by dragging it onto the window. Use `--open` to auto-open the browser on start.

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
- `cache.py` -- Content-addressed disk cache for build artifacts
- `api/graph.py` -- Graph load/validate/export/catalog endpoints
- `api/simulate.py` -- Per-sample simulation endpoint
- `api/optimize.py` -- Multi-pass optimization endpoint
- `api/compile.py` -- C++ code generation endpoint
- `api/generate.py` -- Project generation (source files, zip download, platform listing)
- `api/build.py` -- Binary build, batch build, and build cache management
- `api/layout.py` -- Auto-layout endpoint

### Frontend (React + TypeScript)

- React Flow (xyflow) for graph visualization
- Zustand for state management
- CodeMirror editor with custom `.gdsp` language support
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
| `/api/generate` | POST | Generate project source files |
| `/api/generate/zip` | POST | Generate project as zip download |
| `/api/generate/platforms` | GET | List supported target platforms |
| `/api/build` | POST | Compile graph to binary plugin |
| `/api/build/binary` | POST | Compile and download binary |
| `/api/build/batch` | POST | Build for multiple platforms |
| `/api/build/batch/{id}/zip` | GET | Download batch build as zip |
| `/api/build/cache` | DELETE | Clear build cache |
| `/api/build/cache/info` | GET | Build cache statistics |

## License

MIT
