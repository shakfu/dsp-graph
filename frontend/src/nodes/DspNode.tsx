import { Handle } from "@xyflow/react";
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
  // One target handle per connectable input field, evenly stacked along the
  // target edge. The handle id is the field name, so onConnect can write the
  // connection into the right node_data field. Falls back to a single unnamed
  // handle when the catalog/fields are unavailable.
  const minHeight = fields.length > 1 ? fields.length * 18 + 12 : undefined;
  return (
    <div
      style={{
        background: (data.color as string) || "#fff3cd",
        border: "1px solid #ccc",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 110,
        minHeight,
        fontSize: 12,
        position: "relative",
      }}
    >
      {fields.length > 0 ? (
        fields.map((field, i) => {
          const top = `${((i + 1) / (fields.length + 1)) * 100}%`;
          return (
            <Handle
              key={field}
              id={field}
              type="target"
              position={target}
              style={{ top }}
              title={field}
            />
          );
        })
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
