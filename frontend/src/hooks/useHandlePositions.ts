import { createContext, useContext } from "react";
import { Position } from "@xyflow/react";
import type { ElkDirection } from "../utils/elkLayout";

const HANDLE_POSITIONS: Record<ElkDirection, { source: Position; target: Position }> = {
  RIGHT: { source: Position.Right, target: Position.Left },
  LEFT: { source: Position.Left, target: Position.Right },
  DOWN: { source: Position.Bottom, target: Position.Top },
  UP: { source: Position.Top, target: Position.Bottom },
};

export const DirectionContext = createContext<ElkDirection>("RIGHT");

export function useHandlePositions() {
  const direction = useContext(DirectionContext);
  return HANDLE_POSITIONS[direction];
}
