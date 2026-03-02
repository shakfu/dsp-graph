export const NODE_COLORS: Record<string, string> = {
  input: "#d4edda",
  output: "#f8d7da",
  param: "#cce5ff",
  dsp_node: "#fff3cd",
};

export function getNodeColor(type: string): string {
  return NODE_COLORS[type] ?? "#ffffff";
}
