/**
 * Read a `.flex/` directory from disk into raw artifacts + the normalized
 * history model. Node-only (uses `fs`).
 *
 * This is the exact read logic a future IDE-extension host would run against a
 * workspace's `.flex` dir — keep it host-agnostic (no CLI/UI concerns here).
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

import { buildHistory } from "../core/model";
import {
  flexIdsInManifest,
  parseCandidate,
  parseExploration,
  parseManifest,
  versionsForFlex,
} from "../core/parse";
import type { FlexHistory, Manifest, RawCandidate, RawModuleArtifacts } from "../core/types";

function readManifest(root: string): Manifest {
  const p = join(root, "manifest.json");
  return existsSync(p) ? parseManifest(readFileSync(p, "utf-8")) : { flex_modules: {} };
}

/** Subdirectories of `root` that look like a flex module (have a log or candidates). */
function discoverModuleDirs(root: string): string[] {
  if (!existsSync(root)) return [];
  return readdirSync(root).filter((name) => {
    const dir = join(root, name);
    if (!statSync(dir).isDirectory()) return false;
    return existsSync(join(dir, "exploration.jsonl")) || existsSync(join(dir, "candidates"));
  });
}

function readCandidates(moduleDir: string): RawCandidate[] {
  const dir = join(moduleDir, "candidates");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => parseCandidate(readFileSync(join(dir, f), "utf-8")));
}

function readModule(root: string, manifest: Manifest, flexId: string): RawModuleArtifacts {
  const moduleDir = join(root, flexId);
  const logPath = join(moduleDir, "exploration.jsonl");
  return {
    flexId,
    events: existsSync(logPath) ? parseExploration(readFileSync(logPath, "utf-8")) : [],
    candidates: readCandidates(moduleDir),
    versions: versionsForFlex(manifest, flexId),
  };
}

/** Gather raw artifacts for every flex module under `root`. */
export function readFlexDir(root: string): RawModuleArtifacts[] {
  const manifest = readManifest(root);
  const ids = new Set<string>([...flexIdsInManifest(manifest), ...discoverModuleDirs(root)]);
  return [...ids].map((id) => readModule(root, manifest, id));
}

/** Read a `.flex/` dir and build the normalized history model. */
export function loadHistoryFromDir(root: string): FlexHistory {
  return buildHistory(readFlexDir(root));
}
