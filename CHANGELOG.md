# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.9]

### Added

- **Multi-graph .gdsp support**: the gdsp load endpoint now parses all graphs in a source via gen-dsp's `parse_multi` (previously only the last graph was kept). The response includes every graph's name, and a request may select which graph to return (`graph_name`, defaulting to the last). The toolbar shows a graph selector dropdown when a source defines more than one graph; selecting one re-renders it (live preview and explicit load both honor the selection).

- **Export GDSP button**: "Export GDSP" toolbar button downloads the current graph as a `.gdsp` file. The backend endpoint and store action already existed but had no UI trigger.

- **Bidirectional edge editing on canvas**: dsp nodes now expose one target handle per connectable input field (named after the field). Drawing, replacing, or deleting an edge writes the connection into the target node's `node_data`, so canvas edits persist through export, simulation, and compilation. Deleting a node also clears dangling references in surviving nodes. Each edge carries its target field (`target_handle`) so loaded graphs attach edges to the correct input.

- **Continuous integration**: `.github/workflows/ci.yml` runs backend lint/format-check/type-check/tests (Python 3.10 + 3.13) and frontend tests + build on push and pull request.

- **Frontend tests**: introduced Vitest with coverage of the graph store's edge-editing logic (`frontend/src/hooks/useGraph.test.ts`).

- **Configurable sample rate**: a "Rate" control in the Simulation panel sets the graph's sample rate (Hz), which now drives simulation (previously hardcoded to 44100), the spectrum display's frequency axis, and JSON/GDSP export. The rate is loaded from the graph and locked while a simulation session is active (reset to change it).

### Security

- **Per-session token (anti-CSRF)**: the server now mints a per-process token, exposes it via `GET /api/session`, and requires it in the `X-DSP-Session` header on all state-changing (`POST`/`PUT`/`PATCH`/`DELETE`) `/api` requests. This prevents a malicious page in another browser tab from issuing cross-origin requests to the localhost server (notably triggering native builds via `/api/build*`). The SPA fetches the token at startup; safe (`GET`) requests are unaffected.

- **Static path-traversal containment**: the SPA fallback handler now resolves the requested path and serves it only if it stays within the static directory, preventing `..`-style access to files outside the build output.

- **Request input bounds**: `n_samples` (simulate/continue), buffer `data` length, and the batch-build `platforms` list now have explicit limits, and a 16 MiB body-size cap is enforced, closing trivial memory-exhaustion vectors.

- **Graph-name sanitization**: graph names that reach filesystem paths (temp build dirs) or download headers are validated (path separators, parent refs, control chars, and quotes are rejected with 422), and `Content-Disposition` filenames are sanitized to a safe ASCII slug, preventing path traversal and header injection via `graph.name`.

### Fixed

- **Export preserves sample rate**: exporting JSON/GDSP no longer hardcodes `sample_rate: 44100` / `control_interval: 0`. The values are now carried from the loaded graph (including via live preview), so non-44.1 kHz graphs round-trip correctly.

- **Safari: canvas no longer blanks on node drag**: the store<->local React Flow sync re-applied its own echoes, which round-tripped on every node move and overran React's update-depth limit in Safari (error #185), unmounting the canvas. The sync now ignores echoes of its own writes and only applies genuinely external store changes. Also reduced redundant work: handle re-measurement runs only when the node set or layout direction changes (not on every render), and node components read the type catalog via context instead of each subscribing to the store.

- **Canvas input handles placement and routing**: per-field input handles now sit on the node's edge (spread along the axis matching the layout direction) instead of inside the node body, and the live-preview render path carries each edge's `target_handle` so edges attach to the correct input (e.g. `a` vs `b` of a `mul`) rather than all defaulting to the first handle.

## [0.1.8]

### Added

- **Editor: autocomplete**: keywords, builtins (~130 ops), named constants, and dynamic node IDs from the current graph. Completions are categorized by type (keyword, function, constant, variable) with priority boosting.

- **Editor: snippets**: template expansion for `graph {}`, `param`, `in`, `out`, `feedback` declarations with tab stops for quick authoring.

- **Editor: inline error markers**: parse errors from the store are pushed as CodeMirror diagnostics (squiggly underlines) at the reported line/col position.

- **Editor: bracket matching**: `{}` and `()` pairs highlighted via `@codemirror/language`.

- **Editor: go-to-definition**: Cmd+Click (Ctrl+Click) on a node ID in the editor selects the corresponding node on the canvas.

- **C++ download button**: "Download" button next to "Copy" in the C++ tab. Downloads the compiled source as `{graphName}.h`.

- **Drag-and-drop file loading**: drop `.gdsp` or `.json` files anywhere on the app to load them. Green dashed overlay shown during drag. `.gdsp` files load into the editor; `.json` files load via the graph JSON pipeline.

- **localStorage persistence**: `.gdsp` source auto-saves to `localStorage` with 500ms debounce and restores on page load. Survives browser refresh without explicit save.

- **selectNodeById store action**: programmatically select a node by ID and update React Flow selection state (used by go-to-definition).

- **Cycle detection visualization**: validation endpoint returns `cycle_node_ids` on cycle errors. Cycle nodes are highlighted with a dashed purple border on the canvas; edges between cycle nodes are animated with purple stroke.

- **Waveform display (oscilloscope)**: new `WaveformDisplay` canvas component for time-domain output visualization. Auto-scaling amplitude, center line, dB labels. Output data accumulates across `continue` calls for a full-session view.

- **FFT/spectrum view**: new `SpectrumDisplay` canvas component with client-side radix-2 FFT (no external deps). Magnitude spectrum plot with -80..0 dB range. Toggle between Time and Freq views in SimulationPanel.

- **Buffer get/set endpoints**: `POST /simulate/buffer/get` and `POST /simulate/buffer/set` for inspecting and modifying buffer contents in a simulation session. Buffer nodes show a "View" button in SimulationPanel that fetches and renders contents via WaveformDisplay.

- **Undo/redo stack**: snapshot-based undo/redo for all graph mutations (add/delete/duplicate nodes, add/delete edges). Capped at 50 entries. Toolbar buttons with disabled state. Keyboard shortcuts: Cmd+Z (undo), Cmd+Shift+Z (redo).

- **Batch build**: `POST /api/build/batch` accepts `{graph, platforms[]}` and returns per-platform build results. "Build All" button in BuildPanel with summary table (platform | status). Per-platform errors handled gracefully (no short-circuit).

- Tests: `TestBufferEndpoints` (get, set/get roundtrip, invalid session, invalid buffer ID), `TestBatchBuild` (multi-platform, invalid platform, invalid graph).

### Fixed

- **Canvas fitView no longer fires on unrelated store updates**: `fitView` previously re-triggered on every React re-render (e.g. simulation results, peek values) because `useReactFlow().fitView` has an unstable identity. Now uses a ref for the latest `fitView` and a topology key (sorted node IDs) so `fitView` only fires when nodes are actually added/removed/reloaded -- not on position changes from dragging or unrelated state updates.

- **Buffer node filter**: fixed `op === "buf"` to `op === "buffer"` matching the gen-dsp model.

- **Binary plugin compilation**: `POST /api/build` compiles a graph to a binary plugin (`.clap`, `.vst3`, `.component`, etc.) using gen-dsp's `ProjectGenerator` and `Builder`. Returns success/failure status, stdout/stderr build logs, and output filename. `POST /api/build/binary` returns the compiled binary as a download.

- **OS-filtered platform list**: `GET /api/generate/platforms` now returns only platforms available on the host OS (e.g. macOS omits daisy/circle; Linux omits au/max; Windows returns only clap/sc/vst3).

- **Build output UI in BuildPanel**: success/failure status line (green/red), collapsible stdout/stderr build log, and "Download Binary" button on successful builds.

- Tests: `TestCompileBuild` (invalid platform, invalid graph, valid build with cmake skip) and `test_platforms_os_filtered` (verifies OS-specific platform filtering).

### Changed

- **Simulation output auto-displays**: waveform (time-domain) view now defaults to visible after running simulation, instead of requiring a manual toggle.

- **BuildPanel renamed to Plugin Target**: section header "Build Plugin" -> "Plugin Target", "Build" button -> "Generate", status text "Built for" -> "Generated for". New "Build" button triggers actual binary compilation.

- **Plugin Target moved to top of Tools tab**: BuildPanel is now the first section in the sidebar Tools tab (above Simulation, Optimize, Layout).

- **Platform dropdown defaults to first available**: instead of hardcoding "clap", the dropdown defaults to the first platform returned by the OS-filtered backend endpoint.

- **numpy is now a core dependency**: moved from optional `[sim]` extra to main `dependencies`. Removed the `[sim]` install group and updated all documentation and error messages accordingly.

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

- App layout redesigned: horizontal split with editor pane (left), graph canvas (center), tabbed sidebar (right), and status bar (bottom). Replaces the previous single-pane-with-stacked-sidebar layout.

- CompilePanel removed as standalone component; compile trigger moved to sidebar Tools tab, output moved to editor C++ tab.

- NodeCatalog always expanded when its tab is active (removed collapsible accordion).

- Default layout direction changed from RIGHT to DOWN.

- Default simulation input signal changed from impulse to sine (440 Hz) for more useful output visualization.

## [0.1.7]

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
