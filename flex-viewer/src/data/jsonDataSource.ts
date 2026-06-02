/**
 * Standalone data source: fetch a pre-exported `history.json`.
 *
 * Generate the file with `npm run export:sample` (or `npm run export <flexDir>`),
 * which writes it into the `sample/` dir that Vite serves as `public`.
 */
import type { FlexHistory } from "../core/types";
import type { FlexDataSource } from "./dataSource";

export class JsonDataSource implements FlexDataSource {
  constructor(private readonly url: string = "history.json") {}

  async load(): Promise<FlexHistory> {
    const res = await fetch(this.url);
    if (!res.ok) {
      throw new Error(
        `Could not load ${this.url} (${res.status}). Run \`npm run export:sample\` first.`,
      );
    }
    return (await res.json()) as FlexHistory;
  }
}
