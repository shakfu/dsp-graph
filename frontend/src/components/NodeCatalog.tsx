import { useState, useEffect, useMemo } from "react";
import { useGraph } from "../hooks/useGraph";
import type { NodeTypeCatalog } from "../api/types";

const COLOR_CATEGORIES: Record<string, string> = {
  "#fff3cd": "Arithmetic",
  "#e9ecef": "Constants",
  "#fde0c8": "State / Memory",
  "#e2d5f1": "Oscillators",
  "#d4edda": "Utility",
  "#cce5ff": "Subgraph",
};

function categorize(catalog: NodeTypeCatalog): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const [op, info] of Object.entries(catalog)) {
    const category = COLOR_CATEGORIES[info.color] ?? "Other";
    if (!groups[category]) groups[category] = [];
    groups[category].push(op);
  }
  for (const ops of Object.values(groups)) ops.sort();
  return groups;
}

export function NodeCatalog() {
  const catalog = useGraph((s) => s.nodeTypeCatalog);
  const fetchNodeTypes = useGraph((s) => s.fetchNodeTypes);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (!catalog) void fetchNodeTypes();
  }, [catalog, fetchNodeTypes]);

  const groups = useMemo(() => (catalog ? categorize(catalog) : {}), [catalog]);

  const filtered = useMemo(() => {
    if (!catalog || !filter.trim()) return groups;
    const q = filter.toLowerCase();
    const result: Record<string, string[]> = {};
    for (const [cat, ops] of Object.entries(groups)) {
      const matched = ops.filter((op) => {
        const info = catalog?.[op];
        if (!info) return false;
        return (
          op.includes(q) ||
          info.class.toLowerCase().includes(q) ||
          Object.keys(info.fields).some((f) => f.includes(q))
        );
      });
      if (matched.length > 0) result[cat] = matched;
    }
    return result;
  }, [groups, filter, catalog]);

  if (!catalog) {
    return (
      <div style={{ fontSize: 12, color: "#999", padding: 8 }}>
        Loading node catalog...
      </div>
    );
  }

  return (
    <div>
      <input
        type="text"
        placeholder="Filter nodes..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        style={{
          width: "100%",
          padding: "4px 8px",
          fontSize: 12,
          border: "1px solid #ccc",
          borderRadius: 4,
          marginBottom: 8,
          boxSizing: "border-box",
        }}
      />
      {Object.entries(filtered).map(([category, ops]) => (
        <div key={category} style={{ marginBottom: 10 }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: "#666",
              textTransform: "uppercase",
              marginBottom: 4,
            }}
          >
            {category}
          </div>
          {ops.map((op) => {
            const info = catalog![op];
            if (!info) return null;
            return (
              <div
                key={op}
                style={{
                  padding: "4px 8px",
                  marginBottom: 2,
                  borderRadius: 3,
                  background: "#fff",
                  border: "1px solid #eee",
                  fontSize: 12,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background: info.color,
                      border: "1px solid #ccc",
                      flexShrink: 0,
                    }}
                  />
                  <strong>{op}</strong>
                  <span style={{ fontSize: 10, color: "#999" }}>
                    {info.class}
                  </span>
                </div>
                {Object.keys(info.fields).length > 0 && (
                  <div
                    style={{
                      marginTop: 2,
                      fontSize: 10,
                      color: "#666",
                      paddingLeft: 16,
                    }}
                  >
                    {Object.entries(info.fields).map(([fname, finfo]) => (
                      <span key={fname} style={{ marginRight: 8 }}>
                        <code>{fname}</code>
                        {!finfo.required && (
                          <span style={{ color: "#999" }}>?</span>
                        )}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
