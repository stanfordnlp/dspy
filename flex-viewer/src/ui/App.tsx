import { useEffect, useMemo, useState } from "react";

import type { FlexHistory } from "../core/types";
import type { FlexDataSource } from "../data/dataSource";
import { JsonDataSource } from "../data/jsonDataSource";
import { LiveDataSource } from "../data/liveDataSource";
import { DetailPanel } from "./DetailPanel";
import { Graph } from "./Graph";

// The only place that picks a data source. `__FLEX_LIVE__` is injected by Vite
// (true under `npm run serve`). Swap in a webview-message source here when the
// viewer is hosted inside an IDE extension.
const dataSource: FlexDataSource = __FLEX_LIVE__ ? new LiveDataSource() : new JsonDataSource();

export function App() {
  const [history, setHistory] = useState<FlexHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [moduleId, setModuleId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    dataSource
      .load()
      .then((h) => {
        if (!live) return;
        setHistory(h);
        setModuleId(h.modules[0]?.flexId ?? null);
      })
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    const unsubscribe = dataSource.subscribe?.((h) => {
      if (!live) return;
      setHistory(h);
    });
    return () => {
      live = false;
      unsubscribe?.();
    };
  }, []);

  const flexModule = useMemo(
    () => history?.modules.find((m) => m.flexId === moduleId) ?? null,
    [history, moduleId],
  );
  const selectedNode = useMemo(
    () => flexModule?.nodes.find((n) => n.id === selectedId) ?? null,
    [flexModule, selectedId],
  );

  if (error) return <div className="state">Failed to load history: {error}</div>;
  if (!history) return <div className="state">Loading…</div>;
  if (history.modules.length === 0) return <div className="state">No flex modules found.</div>;

  return (
    <div className="app">
      <header className="topbar">
        <span className="topbar__title">dspy.Flex history</span>
        <nav className="tabs">
          {history.modules.map((m) => (
            <button
              key={m.flexId}
              type="button"
              className={`tab ${m.flexId === moduleId ? "tab--active" : ""}`}
              onClick={() => {
                setModuleId(m.flexId);
                setSelectedId(null);
              }}
            >
              {m.flexId}
              <span className="tab__count">{m.nodes.length}</span>
            </button>
          ))}
        </nav>
      </header>
      <main className="main">
        {flexModule && (
          <Graph history={flexModule} selectedId={selectedId} onSelect={setSelectedId} />
        )}
        <DetailPanel node={selectedNode} onSelect={setSelectedId} />
      </main>
    </div>
  );
}
