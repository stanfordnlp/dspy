/**
 * Parse raw `.flex/` file contents (strings) into typed artifacts.
 *
 * Pure string -> object functions only — no filesystem access — so they are
 * trivially testable and reusable from any host (Node exporter, IDE extension,
 * browser fetch).
 */
import type { Manifest, ManifestVersion, RawCandidate, RawEvent } from "./types";

/** Parse `manifest.json` contents. */
export function parseManifest(text: string): Manifest {
  const data = JSON.parse(text) as Partial<Manifest>;
  return { flex_modules: data.flex_modules ?? {} };
}

/** Parse `<flexId>/exploration.jsonl` contents (one JSON object per line). */
export function parseExploration(text: string): RawEvent[] {
  const events: RawEvent[] = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    events.push(JSON.parse(trimmed) as RawEvent);
  }
  return events;
}

/** Parse a single `candidates/<cid>.json` file. */
export function parseCandidate(text: string): RawCandidate {
  return JSON.parse(text) as RawCandidate;
}

/** Pull the version list for a given flex id out of a parsed manifest. */
export function versionsForFlex(manifest: Manifest, flexId: string): ManifestVersion[] {
  return manifest.flex_modules[flexId]?.versions ?? [];
}

/** All flex ids present in a parsed manifest. */
export function flexIdsInManifest(manifest: Manifest): string[] {
  return Object.keys(manifest.flex_modules);
}
