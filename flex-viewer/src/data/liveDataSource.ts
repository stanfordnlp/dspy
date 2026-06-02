/**
 * Live data source for serve mode: fetches history from the dev server's
 * `/__flex/history` endpoint and re-fetches whenever the flexWatch plugin pings
 * over Vite's HMR socket.
 */
import type { FlexHistory } from "../core/types";
import { FLEX_HISTORY_ENDPOINT, FLEX_UPDATE_EVENT } from "../shared/liveProtocol";
import type { FlexDataSource } from "./dataSource";

export class LiveDataSource implements FlexDataSource {
  constructor(private readonly url: string = FLEX_HISTORY_ENDPOINT) {}

  async load(): Promise<FlexHistory> {
    const res = await fetch(this.url, { cache: "no-store" });
    if (!res.ok) throw new Error(`live history endpoint ${this.url} returned ${res.status}`);
    return (await res.json()) as FlexHistory;
  }

  subscribe(onChange: (history: FlexHistory) => void): () => void {
    const hot = import.meta.hot;
    if (!hot) return () => {};
    const handler = () => {
      this.load()
        .then(onChange)
        .catch(() => {
          /* transient read during a write — the next ping will retry */
        });
    };
    hot.on(FLEX_UPDATE_EVENT, handler);
    return () => hot.off(FLEX_UPDATE_EVENT, handler);
  }
}
