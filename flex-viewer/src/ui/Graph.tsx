import { useMemo } from "react";
import { Background, Controls, MiniMap, ReactFlow, type Edge } from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";

import type { FlexModuleHistory } from "../core/types";
import {
  CandidateNode,
  NODE_HEIGHT,
  NODE_WIDTH,
  type CandidateFlowNode,
} from "./CandidateNode";

const nodeTypes = { candidate: CandidateNode };

function layout(history: FlexModuleHistory): {
  nodes: CandidateFlowNode[];
  edges: Edge[];
} {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "TB", nodesep: 48, ranksep: 80 });
  g.setDefaultEdgeLabel(() => ({}));

  for (const node of history.nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of history.edges) {
    g.setEdge(edge.source, edge.target);
  }
  dagre.layout(g);

  const nodes: CandidateFlowNode[] = history.nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      id: node.id,
      type: "candidate",
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
      data: { node },
    };
  });

  const edges: Edge[] = history.edges.map((edge) => ({
    id: `${edge.source}->${edge.target}`,
    source: edge.source,
    target: edge.target,
    animated: edge.kind === "manifest",
    style: edge.kind === "manifest" ? { strokeDasharray: "5 4" } : undefined,
  }));

  return { nodes, edges };
}

export function Graph({
  history,
  selectedId,
  onSelect,
}: {
  history: FlexModuleHistory;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  // Re-layout only when the module's topology changes.
  const { nodes, edges } = useMemo(() => layout(history), [history]);

  const renderedNodes = useMemo(
    () => nodes.map((n) => ({ ...n, selected: n.id === selectedId })),
    [nodes, selectedId],
  );

  return (
    <div className="graph">
      <ReactFlow
        nodes={renderedNodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => onSelect(node.id)}
        nodesDraggable={false}
        nodesConnectable={false}
        fitView
        minZoom={0.1}
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable />
      </ReactFlow>
    </div>
  );
}
