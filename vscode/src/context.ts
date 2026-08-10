/**
 * Workspace stack detection.
 *
 * Passed to the converter as `--context`, which does two things: it stops the
 * brief from asking "which stack?" when the answer is sitting in the manifest,
 * and it anchors the faithfulness metric, which treats context words as part of
 * the source.
 *
 * Detection reads manifests only — never source files. A wrong guess would
 * become a stated fact in the brief, and a missing guess merely becomes an
 * honest open question.
 */

export interface WorkspaceFiles {
  /** Returns file contents, or undefined when the file does not exist. */
  read(relativePath: string): Promise<string | undefined>;
}

interface Probe {
  file: string;
  extract(content: string): string[];
}

const DEPENDENCY_HINTS: Record<string, string> = {
  react: "React",
  next: "Next.js",
  vue: "Vue",
  svelte: "Svelte",
  "@angular/core": "Angular",
  tailwindcss: "Tailwind",
  express: "Express",
  fastify: "Fastify",
  prisma: "Prisma",
  vitest: "Vitest",
  jest: "Jest",
  playwright: "Playwright",
  django: "Django",
  flask: "Flask",
  fastapi: "FastAPI",
  sqlalchemy: "SQLAlchemy",
  pydantic: "pydantic",
  pytest: "pytest",
  dspy: "DSPy",
};

function hintsFrom(names: Iterable<string>): string[] {
  const found: string[] = [];
  for (const name of names) {
    const hint = DEPENDENCY_HINTS[name.toLowerCase()];
    if (hint && !found.includes(hint)) {
      found.push(hint);
    }
  }
  return found;
}

const PROBES: Probe[] = [
  {
    file: "package.json",
    extract(content) {
      const manifest = JSON.parse(content) as {
        dependencies?: Record<string, string>;
        devDependencies?: Record<string, string>;
        packageManager?: string;
      };
      const names = [
        ...Object.keys(manifest.dependencies ?? {}),
        ...Object.keys(manifest.devDependencies ?? {}),
      ];
      const found = ["Node.js", ...hintsFrom(names)];
      if (manifest.packageManager) {
        found.push(manifest.packageManager.split("@")[0]);
      }
      return found;
    },
  },
  {
    file: "pyproject.toml",
    extract(content) {
      // A tolerant scan, not a TOML parse: a dependency name anywhere in the
      // file is evidence enough for a context hint.
      const names = content.match(/[A-Za-z0-9_.-]+/g) ?? [];
      return ["Python", ...hintsFrom(names)];
    },
  },
  { file: "requirements.txt", extract: (content) => ["Python", ...hintsFrom(content.match(/[A-Za-z0-9_.-]+/g) ?? [])] },
  { file: "Cargo.toml", extract: () => ["Rust"] },
  { file: "go.mod", extract: () => ["Go"] },
  { file: "pom.xml", extract: () => ["Java", "Maven"] },
  { file: "Gemfile", extract: () => ["Ruby"] },
  { file: "tsconfig.json", extract: () => ["TypeScript"] },
  { file: "Dockerfile", extract: () => ["Docker"] },
];

/** Build a one-line `--context` string from workspace manifests. */
export async function detectContext(files: WorkspaceFiles): Promise<string> {
  const found: string[] = [];
  for (const probe of PROBES) {
    const content = await files.read(probe.file);
    if (content === undefined) {
      continue;
    }
    let hints: string[];
    try {
      hints = probe.extract(content);
    } catch {
      // A malformed manifest is not worth failing a conversion over.
      continue;
    }
    for (const hint of hints) {
      if (hint && !found.includes(hint)) {
        found.push(hint);
      }
    }
  }
  return found.length > 0 ? `Stack detected in the workspace: ${found.join(", ")}` : "";
}
