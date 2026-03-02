import { Handle } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";
import { useHandlePositions } from "../hooks/useHandlePositions";

type OutputNodeProps = NodeProps<Node<RFNodeData>>;

export function OutputNode({ data }: OutputNodeProps) {
  const { target } = useHandlePositions();
  return (
    <div
      style={{
        background: (data.color as string) || "#f8d7da",
        border: "1px solid #dc3545",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 80,
        fontSize: 12,
        fontWeight: 600,
      }}
    >
      <Handle type="target" position={target} />
      {data.label as string}
    </div>
  );
}
