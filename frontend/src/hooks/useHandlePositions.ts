import { createContext, useContext } from "react";
import { Position } from "@xyflow/react";
import type { ElkDirection } from "../utils/elkLayout";
import type { NodeTypeCatalog } from "../api/types";

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

// Catalog provided via context so node components don't each subscribe to the
// Zustand store. A per-node store subscription makes every store write (e.g.
// the store<->local node mirror during a drag) synchronously notify every node,
// which in Safari cascaded into a React update-depth loop (error #185).
export const CatalogContext = createContext<NodeTypeCatalog | null>(null);

export function useCatalog(): NodeTypeCatalog | null {
  return useContext(CatalogContext);
}
