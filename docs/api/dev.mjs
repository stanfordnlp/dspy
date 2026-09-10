// Local dev server for the Ask AI backend: serves docs/api/chat.js on
// http://localhost:8787 (override with PORT) so `mkdocs serve` previews can
// use the widget. Keys come from the environment or docs/api/.env.local
// (gitignored, KEY=value lines). Needs Node 18+.
//
//     node docs/api/dev.mjs

import http from "node:http";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
try {
  for (const line of readFileSync(join(here, ".env.local"), "utf8").split("\n")) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/.exec(line);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
} catch {
  /* no .env.local; rely on the environment */
}

const { default: handler } = await import("./chat.js");
const port = Number(process.env.PORT || 8787);

http
  .createServer(async (req, res) => {
    const cors = {
      "Access-Control-Allow-Origin": req.headers.origin || "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };
    if (req.method === "OPTIONS") {
      res.writeHead(204, cors);
      return res.end();
    }
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const request = new Request(`http://localhost:${port}${req.url}`, {
      method: req.method,
      headers: req.headers,
      body: req.method === "POST" ? Buffer.concat(chunks) : undefined,
    });
    let response;
    try {
      response = await handler(request);
    } catch (e) {
      res.writeHead(500, cors);
      return res.end(`dev server error: ${e.message}`);
    }
    res.writeHead(response.status, { ...Object.fromEntries(response.headers), ...cors });
    if (!response.body) return res.end();
    for await (const chunk of response.body) res.write(chunk);
    res.end();
  })
  .listen(port, () => console.log(`Ask AI dev server on http://localhost:${port}  (MODE=${process.env.MODE || "pages"})`));
