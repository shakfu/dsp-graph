## Design decision: editor-primary

The .gdsp text editor is the primary authoring surface. The canvas is a read-only visualization (topology, layout, validation overlays, simulation results) with click-to-select and go-to-definition bridging back to the editor. Bidirectional canvas editing was evaluated and descoped -- the cost of source transforms (especially expression rewriting for connect/disconnect) outweighs the benefit given that .gdsp is a concise textual DSL.

---

## P0 -- High Impact

### Structured validation errors with per-node highlighting

`GraphValidationError` exposes `.kind`, `.node_id`, `.field_name`, `.severity` but `/api/graph/validate` only returns `str(error)`, discarding all structured metadata.

### Build to plugin targets (11 platforms)

`gen_dsp.graph.adapter` provides the full build pipeline from a graph (.gdsp or JSON) to a deployable plugin project. Supported platforms: au, chuck, circle, clap, daisy, lv2, max, pd, sc, vcvrack, vst3. This is the core value chain: author a DSP graph in the editor, then build it into a real plugin.

### Stateful / streaming simulation

`SimState` persists across `simulate()` calls and supports `get_param()`/`set_param()`, `get_buffer()`/`set_buffer()`, `get_peek()`, `reset()`. Currently state is discarded after each call.

### Graph editing on canvas

Canvas editing works for direct store manipulation. With the editor-primary decision, these remain functional but the .gdsp editor is the preferred authoring path. Canvas edits do not round-trip back to .gdsp source.

---

## P1 -- Medium Impact

### Persistence

Editor-primary workflow needs persistence -- losing source on refresh is painful.

### Individual optimization passes

`constant_fold()`, `eliminate_cse()`, `eliminate_dead_nodes()`, `promote_control_rate()` are available individually but only the all-in-one `optimize_graph()` is exposed.

### Structured DSL compile errors

`GDSPCompileError` (semantic errors like "undefined function") was caught by the generic `except Exception`, losing `.line` and `.col`. Now gets the same structured treatment as `GDSPSyntaxError`. Directly improves the editor experience.

### Multi-graph GDSP parsing and composition

The gdsp load endpoint now uses `parse_multi(source)` (returns `dict[str, Graph]`) instead of `parse()`, so all graphs in a source are available. The .gdsp syntax also supports block diagram algebra (`>>` series, `//` parallel) and subgraph instantiation via function-call syntax. Remaining items cover deeper composition UX.

---

## P2 -- Nice to Have

### Subgraph support

`expand_subgraphs(graph)` recursively flattens all Subgraph nodes.

### Control-rate node visualization

`Graph.control_nodes` lists IDs running at control rate. `promote_control_rate()` auto-detects promotable nodes.

### Forward dependency highlighting

`build_forward_deps(graph)` returns `dict[str, set[str]]`.

### Unmapped subgraph param warnings

`validate_graph(graph, warn_unmapped_params=True)` is never called with the flag. Low-cost addition.

---

## Code review follow-ups

Remaining items from the architecture/code review (the security, input-bounds, edge-editing, Safari-loop, CI, sample-rate, and dead-code items are already done).

