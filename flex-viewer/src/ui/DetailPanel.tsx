import type { FlexNode } from "../core/types";
import { CodeView } from "./CodeView";
import { Timeline } from "./Timeline";
import { eventColor, eventLabel, formatScore, formatTs, shortCid } from "./format";

function Chips({
  label,
  ids,
  onSelect,
}: {
  label: string;
  ids: string[];
  onSelect: (id: string) => void;
}) {
  if (ids.length === 0) return null;
  return (
    <div className="chips">
      <span className="chips__label">{label}</span>
      {ids.map((id) => (
        <button key={id} className="chip" type="button" onClick={() => onSelect(id)}>
          {shortCid(id)}
        </button>
      ))}
    </div>
  );
}

export function DetailPanel({
  node,
  onSelect,
}: {
  node: FlexNode | null;
  onSelect: (id: string) => void;
}) {
  if (!node) {
    return (
      <aside className="detail detail--empty">
        <p className="muted">Select a candidate node to inspect its code, events, and scores.</p>
      </aside>
    );
  }

  return (
    <aside className="detail">
      <header className="detail__header">
        <span className="detail__dot" style={{ background: eventColor(node.firstSeenEvent) }} />
        <div>
          <h2 className="detail__cid">{shortCid(node.id)}</h2>
          <code className="detail__cid-full">{node.id}</code>
        </div>
      </header>

      <dl className="detail__facts">
        <dt>first seen</dt>
        <dd>
          {eventLabel(node.firstSeenEvent)} · {formatTs(node.firstSeenTs)}
        </dd>
        {node.signatureHash && (
          <>
            <dt>signature</dt>
            <dd>
              <code>{node.signatureHash.slice(0, 12)}</code>
            </dd>
          </>
        )}
        {node.bestScore !== null && (
          <>
            <dt>best score</dt>
            <dd>
              {formatScore(node.bestScore)}{" "}
              <span className="muted">({node.scores.length} eval(s))</span>
            </dd>
          </>
        )}
      </dl>

      {node.isAccepted && (
        <section className="detail__section">
          <h3>Accepted versions</h3>
          {node.manifestVersions.map((v) => (
            <div key={v.id} className="version">
              <span className="version__id">v{v.id}</span>
              {v.notes && <span className="version__notes">{v.notes}</span>}
              {v.score !== null && <span className="muted">score {formatScore(v.score)}</span>}
              <span className="muted version__ts">{formatTs(v.ts)}</span>
              {v.src_path && <code className="version__path">{v.src_path}</code>}
            </div>
          ))}
        </section>
      )}

      <section className="detail__section">
        <h3>Lineage</h3>
        <Chips label="parents" ids={node.parents} onSelect={onSelect} />
        <Chips label="children" ids={node.children} onSelect={onSelect} />
        {node.parents.length === 0 && node.children.length === 0 && (
          <p className="muted">No recorded lineage edges.</p>
        )}
      </section>

      <section className="detail__section">
        <h3>Events</h3>
        <Timeline events={node.events} />
      </section>

      <section className="detail__section">
        <h3>Code</h3>
        {node.predictorsSrc !== undefined ? (
          <CodeView title="PREDICTORS" code={node.predictorsSrc} />
        ) : (
          <p className="muted">No code snapshot for this candidate.</p>
        )}
        {node.forwardSrc !== undefined && <CodeView title="forward()" code={node.forwardSrc} />}
      </section>
    </aside>
  );
}
