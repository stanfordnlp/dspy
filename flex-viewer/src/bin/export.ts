/**
 * CLI: read a `.flex/` directory and write a `history.json` the standalone app
 * fetches at runtime.
 *
 *   npm run export -- <flexDir> [outFile]
 *   npm run export:sample          # ../tests/flex/demo/.flex -> sample/history.json
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

import { loadHistoryFromDir } from "./readFlexDir";

function main(argv: string[]): void {
  const [flexDir, outFile = "sample/history.json"] = argv;
  if (!flexDir) {
    console.error("usage: export <flexDir> [outFile]");
    process.exit(2);
  }

  const root = resolve(flexDir);
  const history = loadHistoryFromDir(root);

  const out = resolve(outFile);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(history, null, 2) + "\n", "utf-8");

  const nodeCount = history.modules.reduce((n, m) => n + m.nodes.length, 0);
  console.log(
    `Wrote ${out}: ${history.modules.length} module(s), ${nodeCount} candidate node(s).`,
  );
}

main(process.argv.slice(2));
