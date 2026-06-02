import { describe, expect, it } from "vitest";

import { parseExploration, parseManifest } from "./parse";
import { buildModuleHistory } from "./model";
import type { RawModuleArtifacts } from "./types";

// A synthetic GEPA-ish run: A (codegen) -> B (propose+optimized) plus a later
// hand edit C accepted as a manual_edit. The demo .flex dir only has one node,
// so this fixture is what exercises edges/scores/aggregation.
const fixture: RawModuleArtifacts = {
  flexId: "Demo",
  candidates: [
    {
      candidate_id: "A",
      first_seen_event: "codegen",
      first_seen_ts: "2026-01-01T00:00:01Z",
      predictors_src: "PREDICTORS = {}",
      forward_src: "def forward(self): ...",
      signature_hash: "sig",
    },
    {
      candidate_id: "B",
      first_seen_event: "propose",
      first_seen_ts: "2026-01-01T00:00:03Z",
      predictors_src: "PREDICTORS = {'x': 1}",
      forward_src: "def forward(self): return 1",
      signature_hash: "sig",
    },
    {
      candidate_id: "C",
      first_seen_event: "manual_edit",
      first_seen_ts: "2026-01-01T00:00:06Z",
      predictors_src: "PREDICTORS = {'x': 2}",
      forward_src: "def forward(self): return 2",
      signature_hash: "sig",
    },
  ],
  events: [
    { event: "codegen", ts: "2026-01-01T00:00:01Z", candidate_id: "A" },
    { event: "evaluate", ts: "2026-01-01T00:00:02Z", candidate_id: "A", score: 0.5, n_examples: 3 },
    {
      event: "propose",
      ts: "2026-01-01T00:00:03Z",
      candidate_id: "B",
      parents: ["A"],
      changed_component: "forward_src",
    },
    { event: "evaluate", ts: "2026-01-01T00:00:04Z", candidate_id: "B", score: 0.8, n_examples: 3 },
    { event: "evaluate", ts: "2026-01-01T00:00:05Z", candidate_id: "B", score: 0.6, n_examples: 3 },
    { event: "manual_edit", ts: "2026-01-01T00:00:06Z", candidate_id: "C" },
    { event: "accept", ts: "2026-01-01T00:00:07Z", candidate_id: "C", version_id: 1 },
  ],
  versions: [
    {
      id: 0,
      candidate_id: "B",
      parents: ["A"],
      score: 0.8,
      notes: "FlexGEPA optimized",
      signature_hash: "sig",
      src_path: "/x.py",
      ts: "2026-01-01T00:00:04Z",
    },
    {
      id: 1,
      candidate_id: "C",
      parents: ["B"],
      score: null,
      notes: "manual edit",
      signature_hash: "sig",
      src_path: "/x.py",
      ts: "2026-01-01T00:00:07Z",
    },
  ],
};

describe("buildModuleHistory", () => {
  const history = buildModuleHistory(fixture);
  const byId = (id: string) => history.nodes.find((n) => n.id === id)!;

  it("creates a node per candidate", () => {
    expect(history.nodes.map((n) => n.id).sort()).toEqual(["A", "B", "C"]);
  });

  it("aggregates evaluate scores and picks the best", () => {
    expect(byId("B").scores).toEqual([0.8, 0.6]);
    expect(byId("B").bestScore).toBe(0.8);
    expect(byId("A").bestScore).toBe(0.5);
    expect(byId("C").bestScore).toBeNull();
  });

  it("derives lineage edges from event and manifest parents", () => {
    expect(byId("B").parents).toContain("A");
    expect(byId("A").children).toContain("B");
    expect(byId("C").parents).toContain("B");
  });

  it("dedupes an edge inferred from both an event and a manifest version", () => {
    const aToB = history.edges.filter((e) => e.source === "A" && e.target === "B");
    expect(aToB).toHaveLength(1);
  });

  it("marks accepted candidates with their manifest versions", () => {
    expect(byId("B").isAccepted).toBe(true);
    expect(byId("B").manifestVersions[0].id).toBe(0);
    expect(byId("C").isAccepted).toBe(true);
    expect(byId("C").manifestVersions[0].notes).toBe("manual edit");
    expect(byId("A").isAccepted).toBe(false);
  });

  it("keeps the code snapshot for each node", () => {
    expect(byId("C").forwardSrc).toContain("return 2");
    expect(byId("C").firstSeenEvent).toBe("manual_edit");
  });
});

describe("parse helpers", () => {
  it("parses jsonl skipping blank/CRLF lines", () => {
    const text = '{"event":"codegen","ts":"t1"}\r\n\n{"event":"accept","ts":"t2"}\n';
    const events = parseExploration(text);
    expect(events.map((e) => e.event)).toEqual(["codegen", "accept"]);
  });

  it("defaults flex_modules when manifest is empty", () => {
    expect(parseManifest("{}").flex_modules).toEqual({});
  });
});
