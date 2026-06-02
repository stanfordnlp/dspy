import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import type { FlexNode } from "../core/types";
import { eventColor, eventLabel, formatScore, shortCid } from "./format";

export type CandidateNodeData = { node: FlexNode };
export type CandidateFlowNode = Node<CandidateNodeData, "candidate">;

export const NODE_WIDTH = 200;
export const NODE_HEIGHT = 70;

export function CandidateNode({ data, selected }: NodeProps<CandidateFlowNode>) {
  const { node } = data;
  const color = eventColor(node.firstSeenEvent);
  const accepted = node.isAccepted;
  const latestVersion = node.manifestVersions.at(-1);

  return (
    <div
      className="candidate-node"
      style={{
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        borderColor: selected ? "#0f172a" : color,
        borderWidth: selected ? 3 : 2,
        boxShadow: selected ? "0 0 0 3px rgba(15,23,42,0.15)" : undefined,
        opacity: node.scores.length === 0 && !accepted ? 0.85 : 1,
      }}
    >
      <Handle type="target" position={Position.Top} />
      <div className="candidate-node__header">
        <span className="candidate-node__dot" style={{ background: color }} />
        <span className="candidate-node__event">{eventLabel(node.firstSeenEvent)}</span>
        {accepted && (
          <span className="candidate-node__badge candidate-node__badge--accepted">
            ★ v{latestVersion?.id ?? "?"}
          </span>
        )}
      </div>
      <div className="candidate-node__cid">{shortCid(node.id)}</div>
      <div className="candidate-node__meta">
        {node.bestScore !== null && (
          <span className="candidate-node__badge candidate-node__badge--score">
            score {formatScore(node.bestScore)}
          </span>
        )}
        {node.scores.length > 0 && (
          <span className="candidate-node__evals">
            {node.scores.length} eval{node.scores.length === 1 ? "" : "s"}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
