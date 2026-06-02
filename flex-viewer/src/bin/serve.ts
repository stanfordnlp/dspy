/**
 * Live serve launcher: start the Vite dev server pointed at a `.flex` dir so the
 * viewer auto-refreshes as the directory changes.
 *
 *   npm run serve -- <flexDir>     # defaults to the demo dir
 *   npm run serve:sample
 *
 * Sets FLEX_DIR before Vite loads its config (see vite.config.ts), which wires
 * up the flexWatch plugin and flips the app to LiveDataSource.
 */
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { createServer } from "vite";

const flexDir = resolve(process.argv[2] ?? "../tests/flex/demo/.flex");
process.env.FLEX_DIR = flexDir;

if (!existsSync(flexDir)) {
  console.warn(`[serve] ${flexDir} does not exist yet — it will appear once a run writes it.`);
}

const server = await createServer();
await server.listen();
server.config.logger.info("");
server.printUrls();
server.bindCLIShortcuts({ print: true });
