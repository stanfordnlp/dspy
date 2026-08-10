/**
 * Mirrors of the pydantic models in `dspy_vibe/types.py`.
 *
 * The Python side is the authority: it validates before anything is printed,
 * so this side parses rather than re-validates. `parseBundle` still checks the
 * shape, because a broken interpreter or a stray print statement upstream
 * should surface as a clear message, not a runtime crash three calls later.
 */

export interface AcceptanceCriterion {
  statement: string;
  verification: string;
}

export interface OpenQuestion {
  question: string;
  why_it_matters: string;
  assumption_used: string;
}

export interface VibeSpec {
  title: string;
  slug: string;
  goal: string;
  context: string;
  scope: string[];
  non_goals: string[];
  constraints: string[];
  acceptance: AcceptanceCriterion[];
  risks: string[];
  open_questions: OpenQuestion[];
  source_instruction: string;
  confidence: "HIGH" | "MEDIUM" | "LOW";
}

export interface AgentArtifact {
  name: string;
  description: string;
  role: string;
  tools: string[];
  model: string;
  instructions: string[];
  guardrails: string[];
  success_criteria: string[];
  source_spec: string;
}

export interface SkillArtifact {
  name: string;
  description: string;
  triggers: string[];
  procedure: string[];
  inputs: string[];
  outputs: string[];
  checks: string[];
  limits: string[];
  source_spec: string;
}

export interface RenderedBundle {
  spec: string;
  agent: string;
  skill: string;
}

export interface VibeBundle {
  spec: VibeSpec;
  agent: AgentArtifact;
  skill: SkillArtifact;
  rendered: RenderedBundle;
}

export class BundleParseError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Parse the JSON printed by `dspy_vibe convert --stdout`. */
export function parseBundle(stdout: string): VibeBundle {
  let payload: unknown;
  try {
    payload = JSON.parse(stdout);
  } catch {
    // A non-JSON stdout almost always means the interpreter printed something
    // of its own — a warning, a traceback — so show the head of it verbatim.
    const head = stdout.trim().split("\n").slice(0, 5).join("\n");
    throw new BundleParseError(`converter did not return JSON:\n${head || "(no output)"}`);
  }
  if (!isRecord(payload)) {
    throw new BundleParseError("converter returned a non-object payload");
  }
  for (const key of ["spec", "agent", "skill", "rendered"]) {
    if (!isRecord(payload[key])) {
      throw new BundleParseError(`converter payload is missing "${key}"`);
    }
  }
  return payload as unknown as VibeBundle;
}

/** Questions with no assumption applied: work should not start until answered. */
export function blockingQuestions(spec: VibeSpec): OpenQuestion[] {
  return (spec.open_questions ?? []).filter((question) => !question.assumption_used?.trim());
}
