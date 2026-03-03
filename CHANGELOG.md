# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **DSL editor with live graph rendering**: CodeMirror 6 editor pane with 300ms-debounced live preview. Type .gdsp, see the graph update in real-time. AbortController cancels stale requests; re-layout only triggers when node topology changes.
- **Custom .gdsp syntax highlighting**: StreamLanguage tokenizer with keyword, builtin, number, comment (`#`), string, and operator recognition. All ~130 node ops highlighted as builtins; `graph`, `param`, `in`, `out`, `buffer`, etc. as keywords.
- **C++ compile tab**: editor pane tab showing compiled C++ output with full syntax highlighting via `@codemirror/lang-cpp`. Compile and Copy buttons in the tab header.
- **Tabbed editor pane**: `.gdsp` and `C++` tabs in the left pane with dark theme matching One Dark.
- **Tabbed sidebar**: Tools, Inspect, and Catalog tabs replace the previous stacked panel layout, reducing visual clutter.
- **StatusBar**: bottom bar showing parse status (green/red), validation status, node/edge counts, and graph name.
- **Node type catalog**: browsable sidebar tab with all node ops grouped by category (Arithmetic, Oscillators, State/Memory, etc.), color swatches, field info, and text search/filter.
- **Node type catalog backend**: `/api/graph/node-types` now returns a `catalog` key alongside `colors`, with per-op class name, field types/required/defaults, and color. Built from `TypeAdapter(Node).json_schema()`.
- **Structured parse errors**: `/api/graph/load/gdsp` returns `{message, line, col}` in 422 detail for `GDSPSyntaxError`, enabling in-editor error positioning.
- **Compile endpoint tests**: `tests/test_api_compile.py` with valid/invalid graph compilation tests.
- **GDSP loading tests**: `test_load_valid_gdsp` and `test_load_invalid_gdsp_structured_error` in `tests/test_api_graph.py`.
- **Auto-validation**: successful live preview parses automatically trigger graph validation.
- **Editor toggle and live preview toggle**: toolbar buttons to show/hide the editor pane and enable/disable live preview (with manual Parse fallback).
- **Draggable divider**: resizable editor/graph split (200-800px range).
- Synthetic input generation for simulation: graphs with AudioInput nodes now receive generated input signals instead of silence. Supported signal types: impulse, sine (440 Hz), white noise, and DC (ones).
- Per-input signal type selector in the SimulationPanel: each AudioInput gets a dropdown to choose the signal type before running simulation.
- `inputs` field on the `/api/simulate` endpoint: callers can specify signal types per input ID (e.g. `{"in1": "sine", "in2": "ones"}`).
- Auto-layout on graph load: opening a graph now applies the ELK layout automatically so the configured direction takes effect immediately.
- SVG export: "Export SVG" toolbar button captures the current canvas viewport as a downloadable SVG file via html-to-image.
- Auto fit-view on graph load: newly loaded graphs are centered and scaled to fit the canvas.
- Client-side graph layout via elkjs: five layout algorithms (layered, stress, mrtree, radial, force) with configurable direction and node/layer spacing, all running in the browser with no server round-trip.
- LayoutPanel sidebar component with algorithm/direction dropdowns, spacing inputs, and "Apply Layout" button.
- `elkLayout()` async utility converting React Flow nodes/edges to ELK format and returning updated positions.
- Layout state and actions (`layoutOptions`, `setLayoutOptions`, `runLayout`) in the zustand store.

### Changed

- App layout redesigned: horizontal split with editor pane (left), graph canvas (center), tabbed sidebar (right), and status bar (bottom). Replaces the previous single-pane-with-stacked-sidebar layout.
- CompilePanel removed as standalone component; compile trigger moved to sidebar Tools tab, output moved to editor C++ tab.
- NodeCatalog always expanded when its tab is active (removed collapsible accordion).
- Default layout direction changed from RIGHT to DOWN.
- Default simulation input signal changed from impulse to sine (440 Hz) for more useful output visualization.

## [0.2.0]

### Added

- Web UI for visualizing, simulating, optimizing, and inspecting DSP graphs (React 19 + React Flow frontend, FastAPI backend).
- `convert.py` core module: bidirectional `graph_to_reactflow()` / `reactflow_to_graph()` conversion with topo-sort-based layout, color mapping, and feedback edge detection.
- REST API surface: graph load (JSON/.gdsp), validate, export, node-types catalog, DOT export, simulate, optimize, compile (C++ preview), and auto-layout endpoints.
- Custom React Flow node components: DspNode, InputNode, OutputNode, ParamNode.
- NodeInspector sidebar for examining selected node properties.
- SimulationPanel for running per-sample simulation and viewing output.
- OptimizePanel for running graph optimization with before/after stats.
- GraphToolbar with JSON import/export controls.
- `dsp-graph serve` CLI command with `--host`, `--port`, `--reload`, `--open` flags.
- SPA static file serving from built frontend assets.

### Changed

- Project restructured: graph modeling (models, algebra, validation, compilation, simulation, optimization) moved to gen-dsp. dsp-graph now imports from `gen_dsp.graph.*` and provides the visual layer only.
