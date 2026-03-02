import type { Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";

interface Props {
  node: Node<RFNodeData>;
}

export function NodeInspector({ node }: Props) {
  const d = node.data;
  return (
    <div style={{ marginBottom: 16 }}>
      <h4 style={{ margin: "0 0 8px" }}>Node: {node.id}</h4>
      <table style={{ fontSize: 12, width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            <td style={{ padding: "2px 6px", fontWeight: 600 }}>Type</td>
            <td style={{ padding: "2px 6px" }}>{node.type}</td>
          </tr>
          {d.op && (
            <tr>
              <td style={{ padding: "2px 6px", fontWeight: 600 }}>Op</td>
              <td style={{ padding: "2px 6px" }}>{d.op as string}</td>
            </tr>
          )}
          <tr>
            <td style={{ padding: "2px 6px", fontWeight: 600 }}>Color</td>
            <td style={{ padding: "2px 6px" }}>
              <span
                style={{
                  display: "inline-block",
                  width: 12,
                  height: 12,
                  background: d.color as string,
                  border: "1px solid #ccc",
                  verticalAlign: "middle",
                  marginRight: 4,
                }}
              />
              {d.color as string}
            </td>
          </tr>
          {d.node_data &&
            Object.entries(d.node_data as Record<string, unknown>).map(([k, v]) => (
              <tr key={k}>
                <td style={{ padding: "2px 6px", fontWeight: 600 }}>{k}</td>
                <td style={{ padding: "2px 6px", wordBreak: "break-all" }}>
                  {typeof v === "object" ? JSON.stringify(v) : String(v)}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}
