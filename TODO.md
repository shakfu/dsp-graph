# TODO

## Graph Editing

- [ ] Add/delete nodes from the canvas (node palette or context menu)
- [ ] Draw edges between ports by dragging from source handle to target handle
- [ ] Delete edges via click or context menu
- [ ] Duplicate selected node(s)
- [ ] Undo/redo stack for all graph mutations

## Interactive Simulation

- [ ] Real-time parameter sliders in NodeInspector that feed into simulation
- [ ] Streaming simulation mode: continuously re-simulate as params change
- [ ] Waveform display (oscilloscope-style) for simulation output
- [ ] Frequency-domain view (FFT/spectrum) alongside time-domain output

## Validation & Diagnostics

- [ ] Live validation overlay: highlight invalid nodes/edges in red as the graph is edited
- [ ] Validation panel showing all errors/warnings with click-to-select-node navigation
- [ ] Cycle detection visualization: highlight feedback loops on the canvas

## Import / Export

- [ ] PNG export (raster alternative to SVG)
- [ ] Import from C++ gen-dsp target code (round-trip from compiled output)
- [ ] Copy graph snippet to clipboard as .gdsp DSL source
- [ ] Drag-and-drop file loading (drop JSON/.gdsp onto canvas)

## Collaboration & Persistence

- [ ] Save/load graphs to browser localStorage or IndexedDB
- [ ] Shareable URL encoding (graph state in URL hash or query param)
- [ ] Multi-tab awareness: warn if same graph is open in another tab

## Performance & UX

- [ ] Keyboard shortcuts (delete node, fit view, export, undo)
- [ ] Search/filter nodes by name or op type
- [ ] Node grouping / subgraph collapse for large graphs
- [ ] Dark mode theme
- [ ] Touch/trackpad gesture improvements for mobile/tablet use

## Compile & Code Generation

- [ ] Side-by-side diff view for C++ output before/after optimization
- [ ] Syntax-highlighted C++ preview panel (instead of plain text)
- [ ] Download generated C++ as a file

## Testing

- [ ] Frontend component tests (React Testing Library or Playwright)
- [ ] End-to-end tests: load graph, simulate, export, verify results
- [ ] Visual regression tests for SVG export
