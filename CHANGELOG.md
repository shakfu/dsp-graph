# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
