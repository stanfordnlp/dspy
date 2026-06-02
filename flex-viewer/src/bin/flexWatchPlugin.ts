/**
 * Vite dev-server plugin for live serve mode.
 *
 * - Serves the freshly-read history at GET `/__flex/history`.
 * - Watches the `.flex` dir and pushes a `flex:update` message over Vite's HMR
 *   websocket whenever a file under it changes (e.g. during a FlexGEPA run).
 *
 * The browser side ({@link ../data/liveDataSource}) re-fetches on that ping.
 * `loadHistoryFromDir` here is the same reader the static exporter and a future
 * IDE-extension host use.
 */
import { resolve, sep } from "node:path";
import type { Plugin } from "vite";

import { FLEX_HISTORY_ENDPOINT, FLEX_UPDATE_EVENT } from "../shared/liveProtocol";
import { loadHistoryFromDir } from "./readFlexDir";

export function flexWatch({ flexDir }: { flexDir: string }): Plugin {
  const root = resolve(flexDir);
  const within = (file: string): boolean => {
    const p = resolve(file);
    return p === root || p.startsWith(root + sep);
  };

  return {
    name: "flex-watch",
    configureServer(server) {
      server.middlewares.use(FLEX_HISTORY_ENDPOINT, (_req, res) => {
        try {
          const history = loadHistoryFromDir(root);
          res.setHeader("content-type", "application/json");
          res.setHeader("cache-control", "no-store");
          res.end(JSON.stringify(history));
        } catch (e) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }));
        }
      });

      server.watcher.add(root);
      let timer: ReturnType<typeof setTimeout> | undefined;
      const notify = (file: string) => {
        if (!within(file)) return;
        clearTimeout(timer);
        // Debounce: a single optimization step can write many files at once.
        timer = setTimeout(() => {
          server.ws.send({ type: "custom", event: FLEX_UPDATE_EVENT });
        }, 120);
      };
      server.watcher.on("add", notify);
      server.watcher.on("change", notify);
      server.watcher.on("unlink", notify);

      server.config.logger.info(`  [flex-watch] serving + watching ${root}`);
    },
  };
}
