# TODO

## P0 -- High Impact

### Structured validation errors with per-node highlighting
`GraphValidationError` exposes `.kind`, `.node_id`, `.field_name`, `.severity` but
`/api/graph/validate` only returns `str(error)`, discarding all structured metadata.
- [ ] Return structured error fields from the validate endpoint
- [ ] Highlight invalid nodes on the canvas (red/yellow border by severity)
- [ ] Validation panel with click-to-select-node navigation
- [ ] Cycle detection visualization: highlight feedback loops on the canvas

### Build to plugin targets (11 platforms)
`gen_dsp.graph.adapter` provides the full build pipeline from a graph (.gdsp or
JSON) to a deployable plugin project. Supported platforms: au, chuck, circle,
clap, daisy, lv2, max, pd, sc, vcvrack, vst3. This is the core value chain:
author a DSP graph in the editor, then build it into a real plugin.
- [x] Backend `/api/build` endpoint: generate DSP C++ + adapter C++ + manifest
- [x] Frontend "Plugin Target" panel: platform selector + "Generate" button
- [x] Show generated artifacts (DSP C++, adapter, manifest) in panel
- [x] Download zip of the complete plugin project
- [x] Backend `/api/build/compile` endpoint: binary compilation via `Builder`
- [x] Backend `/api/build/binary` endpoint: download compiled binary
- [x] Frontend build output UI: success/failure status, collapsible logs, download binary
- [ ] Batch build: generate for multiple platforms at once

### Stateful / streaming simulation
`SimState` persists across `simulate()` calls and supports `get_param()`/`set_param()`,
`get_buffer()`/`set_buffer()`, `get_peek()`, `reset()`. Currently state is discarded
after each call.
- [ ] Real-time parameter sliders that feed into live re-simulation
- [ ] Streaming mode: "step N more samples" without resetting state
- [ ] Buffer waveform display (inspect/load buffer contents)
- [ ] Peek node values shown as overlays on the canvas
- [ ] Waveform display (oscilloscope-style) for simulation output
- [ ] Frequency-domain view (FFT/spectrum) alongside time-domain output

### Graph editing on canvas
- [ ] Add/delete nodes (node palette or context menu, driven by node catalog)
- [ ] Draw edges between ports by dragging from source handle to target handle
- [ ] Delete edges via click or context menu
- [ ] Duplicate selected node(s)
- [ ] Undo/redo stack for all graph mutations
- [x] Bidirectional sync: canvas edits update .gdsp source in editor

---

## P1 -- Medium Impact

### Individual optimization passes
`constant_fold()`, `eliminate_cse()`, `eliminate_dead_nodes()`, `promote_control_rate()`
are available individually but only the all-in-one `optimize_graph()` is exposed.
- [ ] Backend endpoints for individual passes
- [ ] Step-through UI: apply one pass at a time, see before/after diff
- [ ] Side-by-side diff view for C++ output before/after optimization

### Subgraph support
`expand_subgraphs(graph)` recursively flattens all Subgraph nodes.
- [ ] "Expand subgraph" context menu action on Subgraph nodes
- [ ] Side-by-side nested vs flat graph view
- [ ] Node grouping / subgraph collapse for large graphs

### Multi-graph GDSP parsing
`parse_multi(source)` returns `dict[str, Graph]` from a single source file.
Currently `parse()` silently discards all but the last graph.
- [ ] Graph selector dropdown when a .gdsp defines multiple graphs
- [ ] Subgraph library browsing from a multi-graph file

### Control-rate node visualization
`Graph.control_nodes` lists IDs running at control rate. `promote_control_rate()`
auto-detects promotable nodes.
- [ ] Annotate nodes with audio-rate vs control-rate badge/style
- [ ] "Promote to control rate" action with visual feedback

### Import / export
- [x] Copy graph to clipboard as .gdsp source
- [ ] Download generated C++ as a file
- [ ] Drag-and-drop file loading (drop JSON/.gdsp onto canvas)
- [ ] PNG export (raster alternative to SVG)
- [ ] DOT/PDF export button (uses `graph_to_dot_file()` for Graphviz rendering)

> **Note:** `graph_to_gdsp()` is currently implemented in dsp-graph (`convert.py`) as a
> stopgap. Upstream to `gen_dsp.graph.dsl` when the interface stabilizes.

---

## P2 -- Nice to Have

### Forward dependency highlighting
`build_forward_deps(graph)` returns `dict[str, set[str]]`.
- [ ] Highlight all downstream dependents when a node is selected
- [ ] "Select all connected" for subgraph extraction

### Structured DSL compile errors
`GDSPCompileError` (semantic errors like "undefined function") is caught by the
generic `except Exception`, losing `.line` and `.col`. Should get same structured
treatment as `GDSPSyntaxError`.

### Unmapped subgraph param warnings
`validate_graph(graph, warn_unmapped_params=True)` is never called with the flag.
Low-cost addition.

### Block diagram algebra UI
`series()`, `parallel()`, `split()`, `merge()` compose two graphs. Could enable
drag-and-drop graph composition, but the interaction design is non-trivial.

### Persistence & collaboration
- [ ] Save/load graphs to browser localStorage or IndexedDB
- [ ] Shareable URL encoding (graph state in URL hash or query param)

### UX polish
- [ ] Keyboard shortcuts (delete node, fit view, export, undo)
- [ ] Search/filter nodes by name or op type on the canvas
- [ ] Dark mode theme for the full app (not just editor)
- [ ] Touch/trackpad gesture improvements

---

## Testing

- [ ] Frontend component tests (React Testing Library or Playwright)
- [ ] End-to-end tests: load graph, simulate, export, verify results
- [ ] Visual regression tests for SVG export

---

## Done

- [x] Syntax-highlighted C++ preview panel
- [x] Custom .gdsp syntax highlighting (keywords, builtins, comments, numbers)
- [x] Live .gdsp editor with debounced graph rendering
- [x] Node type catalog with search/filter
- [x] Structured parse errors (line/col) from GDSPSyntaxError
- [x] Tabbed editor pane (.gdsp / C++)
- [x] Tabbed sidebar (Tools / Inspect / Catalog)
- [x] Status bar with parse/validation status
- [x] Auto-validation after live preview parse
- [x] Compile endpoint + tests
