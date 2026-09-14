// Vercel Edge Function behind the docs "Ask AI" widget: POST /api/chat.
//
// The docs are deployed by mirroring docs/ into the dspy-docs repo that the
// Vercel project builds, so this file ships with the site and is served on
// the same origin as the pages (no CORS, no separate host). For local
// previews run `node docs/api/dev.mjs` next to `mkdocs serve`.
//
// Two modes, both grounded in the dspy-docs and dspy-code Mixedbread stores:
//   pages  (default) toast-1 agentic store search picks the pages, the
//          function reads the full rendered Markdown of the top result pages
//          from the docs site, and ANSWERER_MODEL composes the answer. This
//          is the pipeline that won the docs-harness A/B.
//   toast  toast-1 itself answers with the hosted store_search/store_grep
//          tools and system_prompt.txt (fetched from the repo). Cheaper and
//          needs only MXBAI_API_KEY.
//
// Environment variables (Vercel project settings):
//   MXBAI_API_KEY        required
//   OPENROUTER_API_KEY   only when ANSWERER_MODEL is an OpenRouter model
//   MODE                 "pages" (default) or "toast"
//   STORES               default "dspy-docs,dspy-code"
//   DOCS_SITE            default "https://dspy.ai"
//   ANSWERER_MODEL       model that composes the answer in pages mode:
//                        "toast-1" (Mixedbread, no second key) or an
//                        OpenRouter id such as "anthropic/claude-sonnet-5"
//   PAGES                default 3
//   MAX_TOKENS           default 4000
//   SYSTEM_PROMPT_URL    MODE=toast only; default is docs/api/system_prompt.txt on main
//   ALLOWED_ORIGINS      extra browser origins allowed to call the endpoint,
//                        comma-separated ("*" disables the check). The site's
//                        own origin, DOCS_SITE and localhost previews always are.
//   RATE_LIMIT_IP        requests per minute per client IP (default 10; 0 = off)
//   RATE_LIMIT_GLOBAL    requests per hour across all clients (default 300; 0 = off)
//   UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN (or the KV_REST_API_URL +
//                        KV_REST_API_TOKEN pair the Vercel Marketplace sets):
//                        optional Redis REST endpoint that makes the rate limits
//                        global across edge regions. Without it every edge
//                        isolate counts in its own memory (best effort).

export const config = { runtime: "edge" };

const ANSWERER_SYSTEM = `You are the DSPy documentation assistant. DSPy is the Python framework for programming, rather than prompting, language models.

Answer the user's question using ONLY the retrieved documentation and source excerpts provided. Do not use outside knowledge. If the excerpts do not establish the answer, say plainly that the DSPy docs do not cover it instead of guessing.

Style: concise Markdown; lead with the direct answer; keep exact DSPy names, parameters, and defaults as written in the excerpts; a short code example only when it helps; cite the documentation page(s) you used by their path (for example \`diving-deeper/react\`). No preamble, no restating the question.`;

// Answerer prompt used when ANSWERER_MODEL is toast-1. Evolved with GEPA
// against the docs-QA judge (toast-1 as the student), memorized training
// facts removed; holdout 0.586 -> 0.664 versus the generic prompt above.
const ANSWERER_SYSTEM_TOAST = `You are the DSPy documentation assistant. DSPy is the Python framework for programming, rather than prompting, language models.

## Input format

You will receive:

- \`## Retrieved excerpts\`: DSPy documentation pages and/or source excerpts, each under a header such as \`### page: api/utils/load\` or \`### source: dspy/utils/saving.py\`.
- \`## Question\`: the user's documentation question.

## Core requirements

Answer using ONLY the retrieved excerpts supplied in the current input. Do not use outside knowledge or fill gaps by inference. If the supplied material does not establish a requested detail, say plainly that the provided DSPy docs do not cover it.

Lead with the direct answer. Use concise Markdown and exact DSPy names, parameters, field names, defaults, filenames, exception types, message text, and execution order as written in the excerpts. Do not restate the question or add a preamble.

Treat every clause of a multi-part question as a required item. Before answering, make an internal checklist of all requested details and verify that each is covered. Include relevant qualifiers that define the behavior, even if the question does not repeat them—for example:

- which saving/loading mode or configuration the behavior applies to;
- prerequisites or intended usage;
- exact files, fields, and return types;
- ordering and classification rules;
- exception types and meaningful details of their messages;
- fallback values and whether all output fields are populated;
- whether placeholders or prefixes are conditional rather than universal.

Distinguish clearly between what the excerpts state and what they do not state. Do not turn an implementation observation into a broader guarantee unless the supplied material supports it.

## Citations

Cite the documentation page path(s) actually used, using the exact identifiers from the excerpt headers, such as:

- \`api/utils/load\`
- \`api/adapters/JSONAdapter\`
- \`tutorials/gepa_papillon\`

Prefer the most directly relevant page. Do not replace a supplied documentation-path citation with an internal Python filename unless that source filename itself was supplied as an excerpt. Additional citations are acceptable only when they materially support the answer.

## Code

Include a short code excerpt only when it materially clarifies control flow, classification, ordering, or return behavior. Keep it focused and do not let code substitute for answering the question in prose.

## Final quality check

Before responding, verify:

1. Every part of the question is answered.
2. No unsupported facts were added.
3. Conditions, exceptions, ordering, and fallback behavior are explicit.
4. Exact DSPy terminology and literals are preserved.
5. Citations use supplied documentation page paths.
6. The answer is concise and contains no irrelevant background.`;

const DEFAULT_SYSTEM_PROMPT_URL =
  "https://raw.githubusercontent.com/stanfordnlp/dspy/main/docs/api/system_prompt.txt";
const DEFAULT_ANSWERER = "anthropic/claude-sonnet-5"; // set after the A/B, see README
const PAGE_CHARS = 30000;
const CHUNK_CHARS = 2500;

const env = (name, fallback) => {
  const v = typeof process !== "undefined" && process.env ? process.env[name] : undefined;
  return v === undefined || v === "" ? fallback : v;
};
const stores = () => env("STORES", "dspy-docs,dspy-code").split(",").map((s) => s.trim());

// ── abuse controls ───────────────────────────────────────────────────────
//
// The endpoint is public: the widget calls it from the docs pages with no
// credential, so what bounds spend on the paid providers behind it is
//   * the body limits in the handler (input size, history length, MAX_TOKENS),
//   * an origin check: a browser request must come from the docs site itself,
//     DOCS_SITE, a localhost preview, or ALLOWED_ORIGINS. Browsers set Origin
//     and a web page cannot forge it, so this stops other sites embedding the
//     widget. It is not authentication (a script can omit or spoof it), hence
//   * per-client and global rate limits keyed on the client IP that Vercel
//     sets (x-real-ip / x-forwarded-for are overwritten at its edge and cannot
//     be spoofed). Counters live in Redis when a REST endpoint is configured,
//     making the limits global; otherwise in the memory of each edge isolate,
//     which still caps what any one isolate can spend.

function corsHeaders(request) {
  // null: Origin present and not allowed. {}: no Origin (not a browser page).
  const origin = request.headers.get("Origin");
  if (!origin) return {};
  const allowed = env("ALLOWED_ORIGINS", "").split(",").map((s) => s.trim()).filter(Boolean);
  let ok = allowed.includes("*") || allowed.includes(origin);
  if (!ok) {
    let host = null;
    try {
      host = new URL(origin).host;
    } catch {
      return null; // "null" (sandboxed/file: pages) or malformed
    }
    const own = [new URL(request.url).host, request.headers.get("x-forwarded-host"), request.headers.get("host")];
    ok =
      own.includes(host) ||
      origin === env("DOCS_SITE", "https://dspy.ai").replace(/\/$/, "") ||
      /^(localhost|127\.0\.0\.1)(:\d+)?$/.test(host); // mkdocs serve previews
  }
  return ok ? { "Access-Control-Allow-Origin": origin, Vary: "Origin" } : null;
}

function clientIp(request) {
  const h = request.headers;
  return h.get("x-real-ip") || (h.get("x-forwarded-for") || "").split(",")[0].trim() || "unknown";
}

const limits = () => [
  { scope: "ip", n: Number(env("RATE_LIMIT_IP", 10)), window: 60 },
  { scope: "global", n: Number(env("RATE_LIMIT_GLOBAL", 300)), window: 3600 },
];

function redisConfig() {
  const url = env("UPSTASH_REDIS_REST_URL", env("KV_REST_API_URL"));
  const token = env("UPSTASH_REDIS_REST_TOKEN", env("KV_REST_API_TOKEN"));
  return url && token ? { url: url.replace(/\/$/, ""), token } : null;
}

// Upstash / Vercel KV REST pipeline: INCR the window-scoped key, expire it
// once the window is over. Returns the count after this request.
async function bumpRedis({ url, token }, key, windowSec) {
  const r = await fetch(`${url}/pipeline`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify([["INCR", key], ["EXPIRE", key, windowSec * 2]]),
  });
  if (!r.ok) throw new Error(`redis ${r.status}`);
  const [incr] = await r.json();
  if (!incr || incr.error) throw new Error(incr?.error || "redis: bad reply");
  return Number(incr.result);
}

// Per-isolate fallback. Keys carry the window index, so a stale entry is just
// garbage; prune it once the map grows instead of resetting live counts.
const buckets = new Map();
function bumpMemory(key, expiresAt, now) {
  if (buckets.size > 10_000) {
    for (const [k, v] of buckets) if (v.expiresAt <= now) buckets.delete(k);
  }
  const b = buckets.get(key) || { n: 0, expiresAt };
  b.n += 1;
  buckets.set(key, b);
  return b.n;
}

// Seconds the client should wait, or 0 when the request may proceed. Fixed
// windows; the per-IP limit is checked first so a blocked client does not
// eat into the global budget.
async function rateLimited(ip) {
  const now = Date.now();
  const redis = redisConfig();
  for (const limit of limits()) {
    if (!(limit.n > 0)) continue;
    const idx = Math.floor(now / 1000 / limit.window);
    const key = `rl:${limit.scope}:${limit.scope === "ip" ? ip : "all"}:${idx}`;
    const resetAt = (idx + 1) * limit.window * 1000;
    let n;
    if (redis) {
      try {
        n = await bumpRedis(redis, key, limit.window);
      } catch (e) {
        console.error(`rate limit store unavailable, counting in isolate memory: ${e.message}`);
      }
    }
    if (n === undefined) n = bumpMemory(key, resetAt, now);
    if (n > limit.n) return Math.max(1, Math.ceil((resetAt - now) / 1000));
  }
  return 0;
}

// ── mode: toast (toast-1 answers with the hosted store tools) ────────────

let systemPromptCache = null;
async function systemPrompt() {
  if (systemPromptCache) return systemPromptCache;
  const r = await fetch(env("SYSTEM_PROMPT_URL", DEFAULT_SYSTEM_PROMPT_URL));
  if (!r.ok) throw new Error(`system prompt fetch failed: ${r.status}`);
  systemPromptCache = await r.text();
  return systemPromptCache;
}

async function toastMode(messages) {
  const ids = stores();
  const system = await systemPrompt();
  const call = (storeIds) =>
    fetch("https://api.mixedbread.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env("MXBAI_API_KEY")}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "toast-1",
        stream: true,
        max_tokens: 2048,
        messages: [{ role: "system", content: system }, ...messages],
        tools: [
          { type: "store_search", store_identifiers: storeIds },
          { type: "store_grep", store_identifiers: storeIds },
        ],
      }),
    });
  let upstream = await call(ids);
  if (!upstream.ok && ids.length > 1) upstream = await call(ids.slice(0, 1));
  return upstream;
}

// ── mode: pages (toast-1 retrieval, whole-page reading, answerer model) ──

async function pagesMode(messages) {
  const question = messages[messages.length - 1].content;
  const site = env("DOCS_SITE", "https://dspy.ai").replace(/\/$/, "");
  const nPages = Number(env("PAGES", 3));

  const search = await fetch("https://api.mixedbread.com/v1/stores/search", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env("MXBAI_API_KEY")}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: question,
      store_identifiers: stores(),
      top_k: 10,
      search_options: { agentic: true, return_metadata: true },
    }),
  });
  if (!search.ok) return search;
  const chunks = ((await search.json()).data || []).map((c) => ({
    path: c.external_id || c.filename || "",
    text: c.text || "",
  }));

  const slugs = [];
  for (const c of chunks) {
    if (c.path.endsWith(".md")) {
      const slug = c.path.replace(/\/index\.md$/, "").replace(/\.md$/, "");
      if (!slugs.includes(slug)) slugs.push(slug);
    }
    if (slugs.length === nPages) break;
  }
  const pages = await Promise.all(
    slugs.map(async (slug) => {
      try {
        const r = await fetch(`${site}/${slug}/index.md`);
        const body = r.ok ? await r.text() : `(fetch failed: ${r.status})`;
        return `### page: ${slug}\n${body.slice(0, PAGE_CHARS)}`;
      } catch (e) {
        return `### page: ${slug}\n(fetch failed: ${e.message})`;
      }
    })
  );
  const rest = chunks
    .filter((c) => !(c.path.endsWith(".md") && slugs.includes(c.path.replace(/\/index\.md$/, ""))))
    .map((c) => `### ${c.path.endsWith(".md") ? "page" : "source"}: ${c.path}\n${c.text.slice(0, CHUNK_CHARS)}`);
  const context = [...pages, ...rest].join("\n\n") || "(no results)";

  const history = messages.slice(0, -1);
  const answerer = env("ANSWERER_MODEL", DEFAULT_ANSWERER);
  const chat = [
    { role: "system", content: answerer.startsWith("toast") ? ANSWERER_SYSTEM_TOAST : ANSWERER_SYSTEM },
    ...history,
    { role: "user", content: `## Retrieved excerpts\n\n${context}\n\n## Question\n\n${question}` },
  ];
  // toast-1 as the answerer: same Mixedbread key, no hosted tools (it only
  // sees the pages above); anything else goes to OpenRouter
  const upstream = answerer.startsWith("toast")
    ? await fetch("https://api.mixedbread.com/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env("MXBAI_API_KEY")}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: answerer,
          stream: true,
          max_tokens: Number(env("MAX_TOKENS", 4000)),
          temperature: 0,
          messages: chat,
        }),
      })
    : await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env("OPENROUTER_API_KEY")}`,
          "Content-Type": "application/json",
          "HTTP-Referer": site,
          "X-Title": "dspy-docs-chat",
        },
        body: JSON.stringify({
          model: answerer,
          stream: true,
          max_tokens: Number(env("MAX_TOKENS", 4000)),
          temperature: 0,
          messages: chat,
        }),
      });
  if (!upstream.ok) return upstream;

  // prepend a synthetic event so the widget can show what was searched/read
  const trace = `data: ${JSON.stringify({
    hosted_tool_calls: [
      { id: "search-1", type: "store_search_call", queries: [question] },
      ...slugs.map((s, i) => ({ id: `read-${i}`, type: "store_grep_call", pattern: `read ${s}` })),
    ],
    choices: [{ delta: {} }],
  })}\n\n`;
  const reader = upstream.body.getReader();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(trace));
    },
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) controller.close();
      else controller.enqueue(value);
    },
    cancel() {
      reader.cancel().catch(() => {});
    },
  });
  return new Response(stream, { status: 200 });
}

// ── handler ──────────────────────────────────────────────────────────────

export default async function handler(request) {
  const cors = corsHeaders(request);
  if (cors === null) return new Response("origin not allowed", { status: 403 });
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: { ...cors, "Access-Control-Allow-Methods": "POST, OPTIONS",
                 "Access-Control-Allow-Headers": "Content-Type" },
    });
  }
  if (request.method !== "POST") return new Response("POST only", { status: 405, headers: cors });
  if (!env("MXBAI_API_KEY")) return new Response("MXBAI_API_KEY is not configured", { status: 500 });

  let messages;
  try {
    const raw = await request.text();
    if (raw.length > 32_000) throw new Error(); // bound input-token spend
    ({ messages } = JSON.parse(raw));
    if (!Array.isArray(messages) || !messages.length) throw new Error();
    if (messages.some((m) => typeof m.content !== "string" || m.content.length > 4_000)) throw new Error();
  } catch {
    return new Response("body must be {messages: [...]} within size limits", { status: 400, headers: cors });
  }
  // keep history bounded; roles/content only, drop anything else
  messages = messages.slice(-12).map(({ role, content }) => ({ role, content }));

  const retryAfter = await rateLimited(clientIp(request));
  if (retryAfter) {
    return new Response("too many requests, try again shortly", {
      status: 429,
      headers: { ...cors, "Retry-After": String(retryAfter) },
    });
  }

  const mode = env("MODE", "pages").toLowerCase();
  let upstream;
  try {
    upstream = mode === "toast" ? await toastMode(messages) : await pagesMode(messages);
  } catch (e) {
    return new Response(`chat backend error: ${e.message}`, { status: 502, headers: cors });
  }
  if (!upstream.ok) {
    const detail = await upstream.text();
    return new Response(`upstream ${upstream.status}: ${detail.slice(0, 300)}`, { status: 502, headers: cors });
  }
  return new Response(upstream.body, {
    headers: { ...cors, "Content-Type": "text/event-stream", "Cache-Control": "no-store" },
  });
}
