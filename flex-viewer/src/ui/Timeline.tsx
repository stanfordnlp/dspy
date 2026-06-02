import type { RawEvent } from "../core/types";
import { eventColor, eventLabel, formatScore, formatTs } from "./format";

/** Extra fields worth surfacing per event row, in display order. */
const EXTRA_KEYS = ["score", "n_examples", "changed_component", "version_id", "reason", "source_path"];

function extras(ev: RawEvent): [string, string][] {
  const out: [string, string][] = [];
  for (const key of EXTRA_KEYS) {
    const value = ev[key];
    if (value === undefined || value === null) continue;
    out.push([key, key === "score" ? formatScore(value as number) : String(value)]);
  }
  return out;
}

export function Timeline({ events }: { events: RawEvent[] }) {
  if (events.length === 0) return <p className="muted">No events.</p>;
  return (
    <ol className="timeline">
      {events.map((ev, i) => (
        <li key={i} className="timeline__row">
          <span className="timeline__dot" style={{ background: eventColor(ev.event) }} />
          <div className="timeline__body">
            <div className="timeline__head">
              <span className="timeline__event">{eventLabel(ev.event)}</span>
              <span className="timeline__ts">{formatTs(ev.ts)}</span>
            </div>
            {extras(ev).length > 0 && (
              <div className="timeline__extras">
                {extras(ev).map(([k, v]) => (
                  <span key={k} className="timeline__extra">
                    <b>{k}</b> {v}
                  </span>
                ))}
              </div>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}
