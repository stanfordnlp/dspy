/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

import { flexWatch } from "./src/bin/flexWatchPlugin";

// FLEX_DIR is set by `npm run serve` (src/bin/serve.ts). When present we enable
// live mode: watch that .flex dir and flip the app to LiveDataSource. Otherwise
// the static app fetches `history.json` (generate it with `npm run export`).
const flexDir = process.env.FLEX_DIR;

export default defineConfig({
  plugins: [react(), ...(flexDir ? [flexWatch({ flexDir })] : [])],
  publicDir: "sample",
  define: {
    __FLEX_LIVE__: JSON.stringify(Boolean(flexDir)),
  },
  test: {
    // core/ is pure TS (no DOM); run tests in Node.
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
