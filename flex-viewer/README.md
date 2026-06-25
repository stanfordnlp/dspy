# flex-viewer

A visual viewer for `dspy.Flex` history. It reads the `.flex/` artifacts a Flex
module writes — `manifest.json`, `<id>/exploration.jsonl`, `<id>/candidates/*.json`
— and renders the **candidate lineage DAG**: every codegen / propose / manual-edit
candidate as a node, parent edges between them, and a detail panel (per node) with
the generated code, the event timeline, evaluation scores, and accepted-version info.

This is a standalone web app today, structured so it can later be wrapped as an
**IDE-plugin webview** with no rewrite of the parsing or UI layers (see
[Becoming an IDE plugin](#becoming-an-ide-plugin)).

## Quick start

```bash
cd flex-viewer
npm install

# Generate history.json from the checked-in demo .flex dir, then open the app:
npm run export:sample
npm run dev
```

Click a node to inspect its code, events, and scores. Pan/zoom with the mouse;
use the tabs at the top to switch between flex modules.

To view *your own* run's history:

```bash
npm run export -- /path/to/your/project/.flex sample/history.json
npm run dev
```

### Live serve mode

To watch a `.flex` dir update in real time (e.g. while a `dspy.GEPA` run is
writing candidates), use serve mode instead of a static export:

```bash
npm run serve:sample                       # the demo dir
npm run serve -- /path/to/project/.flex     # your own dir
```

The dev server reads the dir on each request (`/__flex/history`) and pushes a
refresh over Vite's HMR socket whenever a file under it changes, so the graph
re-renders automatically — no re-export, no manual reload. The directory need
not exist yet; the view fills in once a run creates it.

## Scripts

| script | what it does |
| --- | --- |
| `npm run dev` | Vite dev server; the app fetches `history.json` (served from `sample/`). |
| `npm run serve -- <flexDir>` | Live mode: watch a `.flex` dir and auto-refresh on change. |
| `npm run serve:sample` | Live mode pointed at `../tests/flex/demo/.flex`. |
| `npm run export -- <flexDir> [out]` | Read a `.flex` dir, write `history.json`. |
| `npm run export:sample` | Export `../tests/flex/demo/.flex` → `sample/history.json`. |
| `npm run build` | Type-check + production bundle into `dist/`. |
| `npm run test` | Run the `core/` + exporter vitest suite. |
| `npm run typecheck` | `tsc --noEmit`. |

## Architecture

```
src/
  core/    PURE TS — no DOM/React. Parses .flex artifacts → normalized FlexHistory
           (nodes per candidate, lineage edges, per-node event/score aggregation).
  data/    FlexDataSource interface + JsonDataSource (static history.json) and
           LiveDataSource (serve-mode endpoint + HMR refresh).
  shared/  liveProtocol.ts — endpoint/event constants shared by plugin + browser
           (dependency-free, so no Node code leaks into the bundle).
  ui/      React + React Flow (dagre layout) graph, custom nodes, detail panel.
  bin/     readFlexDir.ts (Node fs reader), export.ts (CLI → history.json),
           serve.ts + flexWatchPlugin.ts (live dev server).
```

The reuse boundary is `core/` plus the `FlexDataSource` interface. Everything
else (exporter, UI, host) depends only on those.

### Data model notes

- Lineage **edges are derived from `parents` on event rows** (`propose`, and
  `codegen` when seeded from a manual edit / signature change) **and on manifest
  versions** — candidate files themselves carry no parent pointer. Edges are
  deduped across both sources.
- A node is **accepted** when its `candidate_id` appears as a `manifest.json`
  version; `manual edit` and `GEPA optimized` show in the version `notes`.
- The raw artifact types in `src/core/types.ts` mirror the Python writers in
  `dspy/flex/exploration.py` and `dspy/flex/manifest.py` — keep them in sync.

## Becoming an IDE plugin

The eventual target is an IDE plugin, whose UI is a webview rendered with the
same web tech used here. To get there:

1. Add `src/data/messageDataSource.ts` implementing `FlexDataSource` — it
   receives a `FlexHistory` over the webview message channel instead of fetching
   a file, and calls `onChange` when the host re-reads the dir.
2. Build a thin extension host that imports `src/bin/readFlexDir.ts`
   (`loadHistoryFromDir`) to read the workspace `.flex` dir, posts the result to
   the webview, and re-posts on a file-system watch for live updates during an
   optimization run.

`core/` and all of `ui/` are reused unchanged; only the data source and host are
new. The host stays host-agnostic — the same `readFlexDir` logic works for VS
Code (Node host) or any other editor that can run a Node/TS process.
