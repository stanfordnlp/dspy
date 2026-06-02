import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { loadHistoryFromDir, readFlexDir } from "./readFlexDir";

const here = dirname(fileURLToPath(import.meta.url));
// flex-viewer/src/bin -> repo root -> the checked-in demo artifacts.
const DEMO = resolve(here, "../../../tests/flex/demo/.flex");

describe("readFlexDir (real demo .flex)", () => {
  it("reads the MathWord module with code and an accepted version", () => {
    expect(existsSync(DEMO), `demo dir missing at ${DEMO}`).toBe(true);

    const history = loadHistoryFromDir(DEMO);
    const mathWord = history.modules.find((m) => m.flexId === "MathWord");
    expect(mathWord, "MathWord module").toBeTruthy();

    const accepted = mathWord!.nodes.filter((n) => n.isAccepted);
    expect(accepted.length).toBeGreaterThan(0);

    const coded = mathWord!.nodes.filter((n) => n.forwardSrc);
    expect(coded.length).toBeGreaterThan(0);
    expect(coded[0].forwardSrc).toContain("def forward");
  });

  it("produces JSON-serializable artifacts for every module", () => {
    const raw = readFlexDir(DEMO);
    expect(raw.length).toBeGreaterThan(0);
    // round-trips (no undefined/circular surprises)
    expect(() => JSON.stringify(loadHistoryFromDir(DEMO))).not.toThrow();
  });
});
