import { Handle } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";
import { useHandlePositions } from "../hooks/useHandlePositions";

type InputNodeProps = NodeProps<Node<RFNodeData>>;

export function InputNode({ data }: InputNodeProps) {
  const { source } = useHandlePositions();
  return (
    <div
      style={{
        background: (data.color as string) || "#d4edda",
        border: "1px solid #28a745",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 80,
        fontSize: 12,
        fontWeight: 600,
      }}
    >
      {data.label as string}
      <Handle type="source" position={source} />
    </div>
  );
}
