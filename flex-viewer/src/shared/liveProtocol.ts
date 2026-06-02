/**
 * Constants shared between the dev-server watch plugin (Node) and the live data
 * source (browser). Kept dependency-free so importing it pulls no Node modules
 * into the browser bundle.
 */
export const FLEX_HISTORY_ENDPOINT = "/__flex/history";
export const FLEX_UPDATE_EVENT = "flex:update";
