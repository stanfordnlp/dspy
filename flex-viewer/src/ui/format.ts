import type { FlexEventType } from "../core/types";

/** Color per event type — used for node accents and timeline dots. */
export const EVENT_COLOR: Record<string, string> = {
  codegen: "#6366f1", // indigo
  propose: "#0ea5e9", // sky
  manual_edit: "#f59e0b", // amber
  evaluate: "#94a3b8", // slate
  accept: "#22c55e", // green
  load: "#64748b", // gray-slate
};

export function eventColor(event: string | undefined): string {
  return (event && EVENT_COLOR[event]) || "#94a3b8";
}

export function shortCid(id: string): string {
  return id.length > 8 ? id.slice(0, 8) : id;
}

export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return "—";
  return Number.isInteger(score) ? String(score) : score.toFixed(3);
}

export function formatTs(ts: string | undefined): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

export const EVENT_LABEL: Record<string, string> = {
  codegen: "codegen",
  propose: "propose",
  manual_edit: "manual edit",
  evaluate: "evaluate",
  accept: "accept",
  load: "load",
};

export function eventLabel(event: FlexEventType | string | undefined): string {
  if (!event) return "?";
  return EVENT_LABEL[event] ?? event;
}
