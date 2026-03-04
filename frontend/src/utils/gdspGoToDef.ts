import { EditorView } from "@codemirror/view";

/**
 * Creates a CodeMirror extension that handles Cmd+Click (or Ctrl+Click)
 * on node IDs, selecting the corresponding node on the canvas.
 */
export function gdspGoToDef(
  getNodeIds: () => Set<string>,
  onSelectNode: (nodeId: string) => void,
) {
  return EditorView.domEventHandlers({
    click(event: MouseEvent, view: EditorView) {
      if (!(event.metaKey || event.ctrlKey)) return false;

      const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos === null) return false;

      const wordRange = view.state.wordAt(pos);
      if (!wordRange) return false;

      const word = view.state.sliceDoc(wordRange.from, wordRange.to);
      const nodeIds = getNodeIds();

      if (nodeIds.has(word)) {
        event.preventDefault();
        onSelectNode(word);
        return true;
      }

      return false;
    },
  });
}
