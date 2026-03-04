# TODO

## Design decision: editor-primary

The .gdsp text editor is the primary authoring surface. The canvas is a read-only
visualization (topology, layout, validation overlays, simulation results) with
click-to-select and go-to-definition bridging back to the editor. Bidirectional
canvas editing was evaluated and descoped -- the cost of source transforms
(especially expression rewriting for connect/disconnect) outweighs the benefit
given that .gdsp is a concise textual DSL.

---

## P0 -- High Impact

### Structured validation errors with per-node highlighting
`GraphValidationError` exposes `.kind`, `.node_id`, `.field_name`, `.severity` but
`/api/graph/validate` only returns `str(error)`, discarding all structured metadata.
- [x] Return structured error fields from the validate endpoint
- [x] Highlight invalid nodes on the canvas (red/yellow border by severity)
- [x] Validation panel with click-to-select-node navigation
- [x] Cycle detection visualization: highlight feedback loops on the canvas

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
- [x] Batch build: generate for multiple platforms at once

### Stateful / streaming simulation
`SimState` persists across `simulate()` calls and supports `get_param()`/`set_param()`,
`get_buffer()`/`set_buffer()`, `get_peek()`, `reset()`. Currently state is discarded
after each call.
- [x] Real-time parameter sliders that feed into live re-simulation
- [x] Streaming mode: "step N more samples" without resetting state
- [x] Buffer waveform display (inspect/load buffer contents)
- [x] Peek node values shown as overlays on the canvas
- [x] Waveform display (oscilloscope-style) for simulation output
- [x] Frequency-domain view (FFT/spectrum) alongside time-domain output

### Graph editing on canvas
Canvas editing works for direct store manipulation. With the editor-primary
decision, these remain functional but the .gdsp editor is the preferred
authoring path. Canvas edits do not round-trip back to .gdsp source.
- [x] Add/delete nodes (node palette or context menu, driven by node catalog)
- [x] Draw edges between ports by dragging from source handle to target handle
- [x] Delete edges via click or context menu
- [x] Duplicate selected node(s)
- [x] Undo/redo stack for all graph mutations

---

## P1 -- Medium Impact

### Persistence
Editor-primary workflow needs persistence -- losing source on refresh is painful.
- [x] Auto-save/restore .gdsp source to browser localStorage

### Import / export
- [x] Copy graph to clipboard as .gdsp source
- [x] Download generated C++ as a file
- [x] Drag-and-drop file loading (drop JSON/.gdsp onto editor or canvas)
- [ ] PNG export (raster alternative to SVG)
- [ ] DOT/PDF export button (uses `graph_to_dot_file()` for Graphviz rendering)

### Individual optimization passes
`constant_fold()`, `eliminate_cse()`, `eliminate_dead_nodes()`, `promote_control_rate()`
are available individually but only the all-in-one `optimize_graph()` is exposed.
- [ ] Backend endpoints for individual passes
- [ ] Step-through UI: apply one pass at a time, see before/after diff
- [ ] Side-by-side diff view for C++ output before/after optimization

### Structured DSL compile errors
`GDSPCompileError` (semantic errors like "undefined function") is caught by the
generic `except Exception`, losing `.line` and `.col`. Should get same structured
treatment as `GDSPSyntaxError`. Directly improves the editor experience.

### Multi-graph GDSP parsing and composition
`parse_multi(source)` returns `dict[str, Graph]` from a single source file.
Currently `parse()` silently discards all but the last graph. The .gdsp syntax
already supports block diagram algebra (`>>` series, `//` parallel) and subgraph
instantiation via function-call syntax. This section covers the UI support for
multi-graph workflows.
- [ ] Graph selector dropdown when a .gdsp defines multiple graphs
- [ ] Subgraph library browsing from a multi-graph file
- [ ] Autocomplete for graph names defined in the same file (for `>>`, `//`, call syntax)

---

## P2 -- Nice to Have

### Subgraph support
`expand_subgraphs(graph)` recursively flattens all Subgraph nodes.
- [ ] "Expand subgraph" context menu action on Subgraph nodes
- [ ] Side-by-side nested vs flat graph view
- [ ] Node grouping / subgraph collapse for large graphs

### Control-rate node visualization
`Graph.control_nodes` lists IDs running at control rate. `promote_control_rate()`
auto-detects promotable nodes.
- [ ] Annotate nodes with audio-rate vs control-rate badge/style
- [ ] "Promote to control rate" action with visual feedback

### Forward dependency highlighting
`build_forward_deps(graph)` returns `dict[str, set[str]]`.
- [ ] Highlight all downstream dependents when a node is selected
- [ ] "Select all connected" for subgraph extraction

### Unmapped subgraph param warnings
`validate_graph(graph, warn_unmapped_params=True)` is never called with the flag.
Low-cost addition.

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
- [x] Editor: autocomplete (keywords, builtins, named constants, node IDs)
- [x] Editor: inline error markers (squiggly underlines at parse error line/col)
- [x] Editor: bracket matching for `{}` and `()`
- [x] Editor: snippets (`graph {}`, `param`, `in`, `out`, `feedback`)
- [x] Editor: go-to-definition (Cmd+Click node ID to select on canvas)
