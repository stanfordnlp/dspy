/**
 * Where generated files go, and what happens when one already exists.
 *
 * Path planning is pure so it can be tested without a workspace; the actual
 * writes take a minimal filesystem interface for the same reason.
 */

import { posix as posixPath } from "node:path";

import type { VibeBundle } from "./types";

export interface ArtifactLayout {
  agentDirectory: string;
  skillDirectory: string;
  specDirectory: string;
}

export interface PlannedArtifact {
  kind: "spec" | "agent" | "skill";
  /** Workspace-relative, POSIX-separated path. */
  path: string;
  content: string;
}

export interface FileSystem {
  exists(relativePath: string): Promise<boolean>;
  write(relativePath: string, content: string): Promise<void>;
}

/** Decide the destination and content of every artifact. */
export function planArtifacts(bundle: VibeBundle, layout: ArtifactLayout): PlannedArtifact[] {
  const slug = bundle.spec.slug;
  return [
    {
      kind: "spec",
      path: posixPath.join(layout.specDirectory, `${slug}.spec.md`),
      content: bundle.rendered.spec,
    },
    {
      kind: "agent",
      path: posixPath.join(layout.agentDirectory, `${bundle.agent.name}.agent`),
      content: bundle.rendered.agent,
    },
    {
      kind: "skill",
      path: posixPath.join(layout.skillDirectory, `${bundle.skill.name}.skill`),
      content: bundle.rendered.skill,
    },
  ];
}

export interface WriteOutcome {
  written: PlannedArtifact[];
  skipped: PlannedArtifact[];
}

/**
 * Write the planned artifacts.
 *
 * Existing files are skipped rather than replaced unless `overwrite` is set: a
 * generated file the user has since edited by hand is worth more than a fresh
 * generation, and losing it silently would be the worst outcome here.
 */
export async function writeArtifacts(
  planned: PlannedArtifact[],
  fs: FileSystem,
  overwrite = false,
): Promise<WriteOutcome> {
  const written: PlannedArtifact[] = [];
  const skipped: PlannedArtifact[] = [];
  for (const artifact of planned) {
    if (!overwrite && (await fs.exists(artifact.path))) {
      skipped.push(artifact);
      continue;
    }
    await fs.write(artifact.path, artifact.content);
    written.push(artifact);
  }
  return { written, skipped };
}
