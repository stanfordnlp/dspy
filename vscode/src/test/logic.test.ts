/**
 * Unit tests for everything that does not need an editor host.
 *
 * Run with `npm test`. The one integration test shells out to the real Python
 * converter and skips itself when the interpreter is unavailable, so the suite
 * still passes on a machine without the package installed.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";

import { planArtifacts, writeArtifacts, type FileSystem, type PlannedArtifact } from "../artifacts";
import { detectContext, type WorkspaceFiles } from "../context";
import { buildArgs, ConverterError, interpretResult, runConverter } from "../runner";
import { blockingQuestions, BundleParseError, parseBundle, type VibeBundle } from "../types";

function bundleFixture(): VibeBundle {
  return {
    spec: {
      title: "Dark mode",
      slug: "dark-mode",
      goal: "The dashboard has a dark theme.",
      context: "",
      scope: ["Add a dark theme to the dashboard"],
      non_goals: ["do not touch the login screen"],
      constraints: [],
      acceptance: [{ statement: "Theme switches", verification: "Open the dashboard and toggle." }],
      risks: [],
      open_questions: [
        { question: "Which palette?", why_it_matters: "Defines the tokens.", assumption_used: "" },
        { question: "Persist it?", why_it_matters: "Storage design.", assumption_used: "Component state." },
      ],
      source_instruction: "dark mode for the dashboard",
      confidence: "LOW",
    },
    agent: {
      name: "dark-mode-agent",
      description: "Adds a dark theme.",
      role: "You implement the brief.",
      tools: ["Read"],
      model: "",
      instructions: ["Implement"],
      guardrails: ["Out of scope: login screen"],
      success_criteria: ["Theme switches"],
      source_spec: "dark-mode",
    },
    skill: {
      name: "theme-skill",
      description: "Adds a theme.",
      triggers: ["dark mode"],
      procedure: ["Locate", "Implement"],
      inputs: ["Target page"],
      outputs: ["The theme"],
      checks: ["Toggle it"],
      limits: ["No persistence"],
      source_spec: "dark-mode",
    },
    rendered: { spec: "# spec", agent: "# agent", skill: "# skill" },
  };
}

test("buildArgs passes only the options that were set", () => {
  assert.deepEqual(buildArgs({ pythonPath: "python", instruction: "do it" }), [
    "-m",
    "dspy_vibe",
    "convert",
    "do it",
    "--stdout",
  ]);
  const full = buildArgs({
    pythonPath: "python",
    instruction: "do it",
    context: " Next.js ",
    tools: ["Read", " ", "Edit"],
    model: "openai/gpt-4o-mini",
  });
  assert.deepEqual(full.slice(5), ["--context", "Next.js", "--tools", "Read,Edit", "--model", "openai/gpt-4o-mini"]);
});

test("a missing package produces an actionable message", () => {
  assert.throws(
    () =>
      interpretResult({
        code: 1,
        stdout: "",
        stderr: "ModuleNotFoundError: No module named 'dspy_vibe'",
        timedOut: false,
      }),
    (error: ConverterError) => /pip install -e \./.test(error.message),
  );
});

test("a timeout is reported as a timeout, not as an exit code", () => {
  assert.throws(
    () => interpretResult({ code: null, stdout: "", stderr: "", timedOut: true }),
    /timed out/,
  );
});

test("non-JSON output is quoted back instead of throwing a parse error", () => {
  assert.throws(
    () => parseBundle("Traceback (most recent call last):\n  File ...\nRuntimeError: boom"),
    (error: BundleParseError) => /RuntimeError: boom/.test(error.message),
  );
});

test("a payload missing a section is rejected", () => {
  assert.throws(() => parseBundle(JSON.stringify({ spec: {}, agent: {}, skill: {} })), /missing "rendered"/);
});

test("blockingQuestions returns only the ones with no assumption", () => {
  const blocking = blockingQuestions(bundleFixture().spec);
  assert.equal(blocking.length, 1);
  assert.equal(blocking[0].question, "Which palette?");
});

test("artifacts are planned into their configured directories", () => {
  const planned = planArtifacts(bundleFixture(), {
    agentDirectory: ".claude/agents",
    skillDirectory: ".claude/skills",
    specDirectory: "docs/briefs",
  });
  assert.deepEqual(
    planned.map((item) => item.path),
    ["docs/briefs/dark-mode.spec.md", ".claude/agents/dark-mode-agent.agent", ".claude/skills/theme-skill.skill"],
  );
});

function memoryFs(existing: string[] = []): FileSystem & { written: Map<string, string> } {
  const written = new Map<string, string>();
  return {
    written,
    async exists(path) {
      return existing.includes(path) || written.has(path);
    },
    async write(path, content) {
      written.set(path, content);
    },
  };
}

test("existing files are skipped rather than clobbered", async () => {
  const planned = planArtifacts(bundleFixture(), {
    agentDirectory: "agents",
    skillDirectory: "skills",
    specDirectory: "briefs",
  });
  const fs = memoryFs(["agents/dark-mode-agent.agent"]);
  const outcome = await writeArtifacts(planned, fs);
  assert.equal(outcome.written.length, 2);
  assert.deepEqual(
    outcome.skipped.map((item: PlannedArtifact) => item.path),
    ["agents/dark-mode-agent.agent"],
  );
});

test("overwrite writes the skipped files", async () => {
  const planned = planArtifacts(bundleFixture(), {
    agentDirectory: "agents",
    skillDirectory: "skills",
    specDirectory: "briefs",
  });
  const fs = memoryFs(["agents/dark-mode-agent.agent"]);
  const first = await writeArtifacts(planned, fs);
  const second = await writeArtifacts(first.skipped, fs, true);
  assert.equal(second.written.length, 1);
  assert.equal(fs.written.get("agents/dark-mode-agent.agent"), "# agent");
});

function fakeWorkspace(files: Record<string, string>): WorkspaceFiles {
  return { async read(path) { return files[path]; } };
}

test("context detection reads manifests", async () => {
  const context = await detectContext(
    fakeWorkspace({
      "package.json": JSON.stringify({
        dependencies: { react: "^18", tailwindcss: "^3" },
        packageManager: "pnpm@9.0.0",
      }),
      "tsconfig.json": "{}",
    }),
  );
  assert.match(context, /React/);
  assert.match(context, /Tailwind/);
  assert.match(context, /pnpm/);
  assert.match(context, /TypeScript/);
});

test("a malformed manifest does not break detection", async () => {
  const context = await detectContext(fakeWorkspace({ "package.json": "{not json", "go.mod": "module x" }));
  assert.equal(context, "Stack detected in the workspace: Go");
});

test("an empty workspace yields no context, not a guess", async () => {
  assert.equal(await detectContext(fakeWorkspace({})), "");
});

function pythonAvailable(): string | undefined {
  for (const candidate of ["python", "python3"]) {
    try {
      execFileSync(candidate, ["-c", "import dspy_vibe"], { stdio: "ignore" });
      return candidate;
    } catch {
      continue;
    }
  }
  return undefined;
}

test("integration: the real converter returns a usable bundle", async (t) => {
  const python = pythonAvailable();
  if (!python) {
    t.skip("dspy_vibe is not importable from any python on PATH");
    return;
  }
  const bundle = await runConverter({
    pythonPath: python,
    instruction: "add a dark mode toggle to the dashboard, don't touch the login screen",
    tools: ["Read", "Edit"],
    timeoutSeconds: 60,
  });
  assert.ok(bundle.spec.slug.length > 0);
  assert.ok(bundle.rendered.agent.startsWith("---"));
  assert.deepEqual(bundle.agent.tools, ["Edit", "Read"]);
  assert.ok(bundle.spec.non_goals.some((item) => /login/.test(item)));
});
