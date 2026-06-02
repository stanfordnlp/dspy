/**
 * Type definitions for dspy.Flex `.flex/` artifacts and the normalized history
 * model the viewer renders.
 *
 * The RAW artifact shapes below mirror the Python writers — keep them in sync
 * with:
 *   - dspy/flex/exploration.py  (FlexEvent, ExplorationStore: exploration.jsonl + candidates/<cid>.json)
 *   - dspy/flex/manifest.py     (ManifestStore: manifest.json)
 *
 * This module is PURE TS (no DOM, no React) so it can be reused verbatim by a
 * future IDE-extension host.
 */

// ---------------------------------------------------------------------------
// Raw artifacts (on-disk JSON shapes)
// ---------------------------------------------------------------------------

/** Closed set of event names in `exploration.jsonl` (mirrors FlexEvent). */
export type FlexEventType =
  | "codegen"
  | "load"
  | "manual_edit"
  | "evaluate"
  | "propose"
  | "accept";

/** One row of `<flexId>/exploration.jsonl`. Extra keys vary by event. */
export interface RawEvent {
  event: FlexEventType | string;
  ts: string;
  candidate_id?: string;
  score?: number;
  parents?: string[];
  // Known extras: source_path, version_id, reason, n_examples, changed_component
  [key: string]: unknown;
}

/** One `<flexId>/candidates/<cid>.json` file. */
export interface RawCandidate {
  candidate_id: string;
  first_seen_event: string;
  first_seen_ts: string;
  predictors_src: string;
  forward_src: string;
  signature_hash: string;
}

/** One accepted version inside `manifest.json`. */
export interface ManifestVersion {
  id: number;
  candidate_id: string | null;
  src_path: string;
  signature_hash: string;
  score: number | null;
  parents: string[];
  ts: string;
  notes: string | null;
}

/** Top-level `manifest.json` shape. */
export interface Manifest {
  flex_modules: Record<string, { versions: ManifestVersion[] }>;
}

/** All raw artifacts for a single flex module, gathered from disk/host. */
export interface RawModuleArtifacts {
  flexId: string;
  events: RawEvent[];
  candidates: RawCandidate[];
  versions: ManifestVersion[];
}

// ---------------------------------------------------------------------------
// Normalized model (what the UI consumes)
// ---------------------------------------------------------------------------

export interface FlexEdge {
  source: string; // parent candidate_id
  target: string; // child candidate_id
  /** Where the edge was inferred from. */
  kind: "event" | "manifest";
}

export interface FlexNode {
  /** candidate_id (the node's stable id). */
  id: string;
  predictorsSrc?: string;
  forwardSrc?: string;
  signatureHash?: string;
  firstSeenEvent?: string;
  firstSeenTs?: string;
  /** Every event referencing this candidate, sorted ascending by ts. */
  events: RawEvent[];
  /** Scores from `evaluate` events, in chronological order. */
  scores: number[];
  /** Highest seen score, or null if never evaluated. */
  bestScore: number | null;
  /** Accepted manifest versions whose candidate_id === id. */
  manifestVersions: ManifestVersion[];
  /** True if this candidate appears as an accepted manifest version. */
  isAccepted: boolean;
  parents: string[];
  children: string[];
}

export interface FlexModuleHistory {
  flexId: string;
  nodes: FlexNode[];
  edges: FlexEdge[];
  /** Full event log for the module, sorted ascending by ts. */
  events: RawEvent[];
}

export interface FlexHistory {
  modules: FlexModuleHistory[];
}
