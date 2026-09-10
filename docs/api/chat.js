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
//   OPENROUTER_API_KEY   required for MODE=pages (the answerer model)
//   MODE                 "pages" (default) or "toast"
//   STORES               default "dspy-docs,dspy-code"
//   DOCS_SITE            default "https://dspy.ai"
//   ANSWERER_MODEL       default "anthropic/claude-sonnet-5"
//   PAGES                default 3
//   MAX_TOKENS           default 4000
//   SYSTEM_PROMPT_URL    MODE=toast only; default is docs/api/system_prompt.txt on main

export const config = { runtime: "edge" };

const ANSWERER_SYSTEM = `You are the DSPy documentation assistant. DSPy is the Python framework for programming, rather than prompting, language models.

Answer the user's question using ONLY the retrieved documentation and source excerpts provided. Do not use outside knowledge. If the excerpts do not establish the answer, say plainly that the DSPy docs do not cover it instead of guessing.

Style: concise Markdown; lead with the direct answer; keep exact DSPy names, parameters, and defaults as written in the excerpts; a short code example only when it helps; cite the documentation page(s) you used by their path (for example \`diving-deeper/react\`). No preamble, no restating the question.`;

const DEFAULT_SYSTEM_PROMPT_URL =
  "https://raw.githubusercontent.com/stanfordnlp/dspy/main/docs/api/system_prompt.txt";
const PAGE_CHARS = 30000;
const CHUNK_CHARS = 2500;

const env = (name, fallback) => {
  const v = typeof process !== "undefined" && process.env ? process.env[name] : undefined;
  return v === undefined || v === "" ? fallback : v;
};
const stores = () => env("STORES", "dspy-docs,dspy-code").split(",").map((s) => s.trim());

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
  const upstream = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env("OPENROUTER_API_KEY")}`,
      "Content-Type": "application/json",
      "HTTP-Referer": site,
      "X-Title": "dspy-docs-chat",
    },
    body: JSON.stringify({
      model: env("ANSWERER_MODEL", "anthropic/claude-sonnet-5"),
      stream: true,
      max_tokens: Number(env("MAX_TOKENS", 4000)),
      temperature: 0,
      messages: [
        { role: "system", content: ANSWERER_SYSTEM },
        ...history,
        { role: "user", content: `## Retrieved excerpts\n\n${context}\n\n## Question\n\n${question}` },
      ],
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
  if (request.method === "OPTIONS") {
    return new Response(null, {
      headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS",
                 "Access-Control-Allow-Headers": "Content-Type" },
    });
  }
  if (request.method !== "POST") return new Response("POST only", { status: 405 });
  if (!env("MXBAI_API_KEY")) return new Response("MXBAI_API_KEY is not configured", { status: 500 });

  let messages;
  try {
    const raw = await request.text();
    if (raw.length > 32_000) throw new Error(); // bound input-token spend
    ({ messages } = JSON.parse(raw));
    if (!Array.isArray(messages) || !messages.length) throw new Error();
    if (messages.some((m) => typeof m.content !== "string" || m.content.length > 4_000)) throw new Error();
  } catch {
    return new Response("body must be {messages: [...]} within size limits", { status: 400 });
  }
  // keep history bounded; roles/content only, drop anything else
  messages = messages.slice(-12).map(({ role, content }) => ({ role, content }));

  const mode = env("MODE", "pages").toLowerCase();
  let upstream;
  try {
    upstream = mode === "toast" ? await toastMode(messages) : await pagesMode(messages);
  } catch (e) {
    return new Response(`chat backend error: ${e.message}`, { status: 502 });
  }
  if (!upstream.ok) {
    const detail = await upstream.text();
    return new Response(`upstream ${upstream.status}: ${detail.slice(0, 300)}`, { status: 502 });
  }
  return new Response(upstream.body, {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-store" },
  });
}
