/**
 * The seam between "where history data comes from" and the UI.
 *
 * Today there is one implementation ({@link JsonDataSource}, which fetches a
 * pre-exported `history.json`). When this viewer becomes an IDE plugin, add a
 * `MessageDataSource` that receives a {@link FlexHistory} over the webview
 * message channel and calls `onChange` when the host re-reads the `.flex` dir.
 * The UI depends only on this interface, so nothing else changes.
 */
import type { FlexHistory } from "../core/types";

export interface FlexDataSource {
  /** Load the current history snapshot. */
  load(): Promise<FlexHistory>;
  /**
   * Optional live updates. Returns an unsubscribe function. Implementations
   * without live data (e.g. static JSON) may omit this.
   */
  subscribe?(onChange: (history: FlexHistory) => void): () => void;
}
