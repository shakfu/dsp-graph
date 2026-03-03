import { Handle } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";
import { useHandlePositions } from "../hooks/useHandlePositions";

type DspNodeProps = NodeProps<Node<RFNodeData>>;

export function DspNode({ data }: DspNodeProps) {
  const { source, target } = useHandlePositions();
  const peekValue = data.peekValue as number | undefined;
  return (
    <div
      style={{
        background: (data.color as string) || "#fff3cd",
        border: "1px solid #ccc",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 100,
        fontSize: 12,
        position: "relative",
      }}
    >
      <Handle type="target" position={target} />
      <div style={{ fontWeight: 600 }}>{data.label as string}</div>
      {data.op && (
        <div style={{ fontSize: 10, color: "#666" }}>{data.op as string}</div>
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
