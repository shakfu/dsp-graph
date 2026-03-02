import { Handle } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";
import { useHandlePositions } from "../hooks/useHandlePositions";

type ParamNodeProps = NodeProps<Node<RFNodeData>>;

export function ParamNode({ data }: ParamNodeProps) {
  const { source } = useHandlePositions();
  return (
    <div
      style={{
        background: (data.color as string) || "#cce5ff",
        border: "1px solid #007bff",
        borderRadius: 12,
        padding: "6px 10px",
        minWidth: 80,
        fontSize: 12,
      }}
    >
      <div style={{ fontWeight: 600 }}>{data.label as string}</div>
      <Handle type="source" position={source} />
    </div>
  );
}
