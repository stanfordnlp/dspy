/**
 * Runs the Python converter.
 *
 * Deliberately free of `vscode` imports so the argument building and output
 * handling can be unit tested with plain `node --test`.
 */

import { spawn } from "node:child_process";

import { BundleParseError, parseBundle, type VibeBundle } from "./types";

export interface RunOptions {
  pythonPath: string;
  instruction: string;
  context?: string;
  tools?: string[];
  model?: string;
  cwd?: string;
  timeoutSeconds?: number;
}

export class ConverterError extends Error {
  constructor(
    message: string,
    readonly detail = "",
  ) {
    super(message);
  }
}

/** Build the argv for `python -m dspy_vibe convert --stdout`. */
export function buildArgs(options: RunOptions): string[] {
  const args = ["-m", "dspy_vibe", "convert", options.instruction, "--stdout"];
  if (options.context?.trim()) {
    args.push("--context", options.context.trim());
  }
  const tools = (options.tools ?? []).map((tool) => tool.trim()).filter(Boolean);
  if (tools.length > 0) {
    args.push("--tools", tools.join(","));
  }
  if (options.model?.trim()) {
    args.push("--model", options.model.trim());
  }
  return args;
}

export interface ProcessResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

/** Interpret a finished process. Separated from spawning so it can be tested. */
export function interpretResult(result: ProcessResult): VibeBundle {
  if (result.timedOut) {
    throw new ConverterError("The converter timed out.", result.stderr);
  }
  if (result.code !== 0) {
    const detail = result.stderr.trim() || result.stdout.trim();
    if (/No module named ['"]?dspy_vibe/.test(detail)) {
      throw new ConverterError(
        "dspy_vibe is not installed in the selected interpreter. Run: pip install -e .",
        detail,
      );
    }
    throw new ConverterError(`The converter exited with code ${result.code}.`, detail);
  }
  try {
    return parseBundle(result.stdout);
  } catch (error) {
    if (error instanceof BundleParseError) {
      throw new ConverterError(error.message, result.stderr);
    }
    throw error;
  }
}

/** Spawn the converter and return the parsed bundle. */
export function runConverter(options: RunOptions): Promise<VibeBundle> {
  const timeoutMs = Math.max(1, options.timeoutSeconds ?? 120) * 1000;

  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawn(options.pythonPath, buildArgs(options), {
        cwd: options.cwd,
        // The converter prints UTF-8; Windows consoles otherwise mangle
        // accented characters, which is most of the Hungarian test corpus.
        env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
      });
    } catch (error) {
      reject(new ConverterError(`Could not start ${options.pythonPath}`, String(error)));
      return;
    }

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeoutMs);

    child.stdout.on("data", (chunk) => (stdout += chunk.toString("utf-8")));
    child.stderr.on("data", (chunk) => (stderr += chunk.toString("utf-8")));

    child.on("error", (error) => {
      clearTimeout(timer);
      reject(new ConverterError(`Could not run ${options.pythonPath}`, error.message));
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      try {
        resolve(interpretResult({ code, stdout, stderr, timedOut }));
      } catch (error) {
        reject(error);
      }
    });
  });
}
