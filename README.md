# dsp-graph

Visual graph editor, inspector, and debugger for [gen-dsp](https://github.com/shakfu/gen-dsp) signal-processing graphs.

## Overview

dsp-graph provides a web UI (React + React Flow) backed by a Python server (FastAPI) for loading, visualizing, simulating, optimizing, building, and inspecting DSP signal graphs defined with gen-dsp's graph DSL.

**Features:**

- Load graphs from JSON or `.gdsp` DSL source (toolbar or drag-and-drop), with localStorage auto-save/restore
- `.gdsp` editor with syntax highlighting, autocomplete, snippets, go-to-definition, inline error markers, and debounced live preview
- Interactive React Flow canvas (pan, zoom, minimap) with ELK auto-layout and auto fit-to-view
- Canvas editing: add/delete/duplicate nodes and draw/replace/delete edges via per-input handles, with undo/redo
- Validation overlays: per-node error/warning borders and feedback-cycle highlighting
- Node inspector showing all properties
- Stateful, streaming simulation: live parameter sliders, "step N more samples", peek-value overlays, oscilloscope waveform and FFT spectrum views, with a configurable sample rate
- Multi-pass and per-pass graph optimization with before/after comparison
- C++ code generation preview with copy/download
- Binary plugin build for multiple platforms (CLAP, VST3, AU, LV2, etc.), with batch build + zip download
- Content-addressed build cache (avoids recompilation for unchanged graphs)
- Graphviz DOT and SVG export; round-trip export back to graph JSON and `.gdsp`

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
dsp-graph serve [--host HOST] [--port PORT] [--reload] [--open] [--experimental]
```

- `--host`: Bind address (default: 127.0.0.1)
- `--port`: Port (default: 8765)
- `--reload`: Auto-reload on code changes
- `--open`: Open browser on start
- `--experimental`: Enable experimental features (the gen~/GenExpr transpiler tab and its Max `.maxpat` test-patch export). Off by default; when omitted the GenExpr tab is hidden and `POST /api/genexpr` and `POST /api/graph/export/maxpat` return 404.

### Security model

dsp-graph is designed for **single-user, localhost** use and binds to `127.0.0.1` by default. The build endpoints generate and compile native code on the host, so the server should not be exposed to untrusted networks. As a safeguard against cross-origin requests from other browser tabs, all state-changing (`POST`/`PUT`/`PATCH`/`DELETE`) `/api` requests require a per-process session token (`GET /api/session`), which the bundled UI fetches automatically. The server also runs a single worker; its session and cache state are in-process.

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

# Run frontend tests (Vitest)
cd frontend && npm test

# Run full QA (test + lint + typecheck + format)
make qa

# Start server with auto-reload
make dev
```

CI (`.github/workflows/ci.yml`) runs the backend lint/format/type-check/tests (Python 3.10 and 3.13) and the frontend tests + build on every push and pull request.

## Architecture

### Backend (Python)

- `convert.py` -- Core Graph <-> ReactFlow conversion layer
- `server.py` -- FastAPI app with SPA static serving
- `security.py` -- Session-token (anti-CSRF) and body-size middleware
- `cli.py` -- CLI entry point (`dsp-graph serve`)
- `cache.py` -- Content-addressed disk cache for build artifacts
- `api/graph.py` -- Graph load/validate/export/catalog endpoints
- `api/simulate.py` -- Per-sample simulation endpoint
- `api/optimize.py` -- Multi-pass optimization endpoint
- `api/compile.py` -- C++ code generation endpoint
- `api/genexpr.py` -- gen~ codebox (GenExpr) transpile endpoint
- `api/maxpat.py` -- Max `.maxpat` test-patch export endpoint (wraps `maxpat.py`)
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

FastAPI also serves interactive docs at `/docs` (Swagger UI) and `/redoc`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/session` | GET | Per-process session token (required header for mutating requests) |
| `/api/graph/load/json` | POST | Load Graph JSON -> ReactFlow |
| `/api/graph/load/gdsp` | POST | Parse .gdsp source -> ReactFlow |
| `/api/graph/validate` | POST | Validate a Graph (structured errors) |
| `/api/graph/export/json` | POST | ReactFlow -> Graph JSON |
| `/api/graph/export/gdsp` | POST | ReactFlow -> .gdsp source |
| `/api/graph/node-types` | GET | Node type catalog + colors |
| `/api/graph/dot` | POST | Graph -> Graphviz DOT |
| `/api/simulate` | POST | Run per-sample simulation (starts a session) |
| `/api/simulate/continue` | POST | Step N more samples on an existing session |
| `/api/simulate/param` | POST | Set a parameter on a session |
| `/api/simulate/peek` | POST | Read peek-node values from a session |
| `/api/simulate/reset` | POST | Reset session state to initial |
| `/api/simulate/buffer/get` | POST | Read a buffer's contents |
| `/api/simulate/buffer/set` | POST | Write a buffer's contents |
| `/api/optimize` | POST | Optimize graph, all passes (before/after) |
| `/api/optimize/pass` | POST | Apply a single named optimization pass |
| `/api/compile` | POST | Graph -> C++ source |
| `/api/config` | GET | Runtime feature flags (e.g. `experimental`) |
| `/api/genexpr` | POST | Graph -> gen~ codebox (GenExpr) source (experimental; 404 unless `--experimental`) |
| `/api/graph/export/maxpat` | POST | Graph -> Max `.maxpat` test patch wrapping the gen~ codebox (experimental; 404 unless `--experimental`) |
| `/api/layout` | POST | Auto-layout ReactFlow nodes |
| `/api/generate` | POST | Generate project source files |
| `/api/generate/zip` | POST | Generate project as zip download |
| `/api/generate/platforms` | GET | List supported target platforms (host-filtered) |
| `/api/build` | POST | Compile graph to binary plugin |
| `/api/build/binary` | POST | Compile and download binary |
| `/api/build/batch` | POST | Build for multiple platforms |
| `/api/build/batch/{id}/zip` | GET | Download batch build as zip |
| `/api/build/cache` | DELETE | Clear build cache |
| `/api/build/cache/info` | GET | Build cache statistics |

## License

MIT
