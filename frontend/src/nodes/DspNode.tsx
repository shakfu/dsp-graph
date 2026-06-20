import type { CSSProperties } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";
import { useHandlePositions, useCatalog } from "../hooks/useHandlePositions";
import { connectableFields } from "../utils/nodeFields";

type DspNodeProps = NodeProps<Node<RFNodeData>>;

export function DspNode({ data }: DspNodeProps) {
  const { source, target } = useHandlePositions();
  const catalog = useCatalog();
  const peekValue = data.peekValue as number | undefined;
  const op = data.op as string | undefined;
  const fields = connectableFields(catalog, op);
  // One target handle per connectable input field, evenly spread along the
  // target edge. The handle id is the field name, so onConnect can write the
  // connection into the right node_data field. Falls back to a single unnamed
  // handle when the catalog/fields are unavailable.
  //
  // Spread along the axis that matches the handle's side: vertically (top) when
  // handles sit on the left/right edge, horizontally (left) when they sit on
  // the top/bottom edge. Otherwise the offset would push handles into the node.
  const onVerticalEdge =
    target === Position.Left || target === Position.Right;
  const handleOffset = (i: number): CSSProperties => {
    const pct = `${((i + 1) / (fields.length + 1)) * 100}%`;
    return onVerticalEdge ? { top: pct } : { left: pct };
  };
  const multi = fields.length > 1;
  const minHeight = multi && onVerticalEdge ? fields.length * 18 + 12 : undefined;
  const minWidth = multi && !onVerticalEdge ? fields.length * 28 + 40 : 110;
  return (
    <div
      style={{
        background: (data.color as string) || "#fff3cd",
        border: "1px solid #ccc",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth,
        minHeight,
        fontSize: 12,
        position: "relative",
      }}
    >
      {fields.length > 0 ? (
        fields.map((field, i) => (
          <Handle
            key={field}
            id={field}
            type="target"
            position={target}
            style={handleOffset(i)}
            title={field}
          />
        ))
      ) : (
        <Handle type="target" position={target} />
      )}
      <div style={{ fontWeight: 600 }}>{data.label as string}</div>
      {op && <div style={{ fontSize: 10, color: "#666" }}>{op}</div>}
      {fields.length > 1 && (
        <div style={{ fontSize: 8, color: "#999", marginTop: 2 }}>
          {fields.join(" / ")}
        </div>
      )}
      {peekValue !== undefined && (
        <div
          style={{
            position: "absolute",
            top: -8,
            right: -8,
            background: "#333",
            color: "#fff",
            fontSize: 9,
            fontFamily: "monospace",
            padding: "1px 4px",
            borderRadius: 3,
            whiteSpace: "nowrap",
          }}
        >
          {peekValue.toFixed(4)}
        </div>
      )}
      <Handle type="source" position={source} />
    </div>
  );
}
