/**
 * VS Code entry point.
 *
 * The only file that imports `vscode`. Everything it needs from the workspace
 * is adapted here and handed to the plain-TypeScript modules, which keeps the
 * logic testable without an editor host.
 */

import * as vscode from "vscode";

import { planArtifacts, writeArtifacts, type FileSystem } from "./artifacts";
import { detectContext, type WorkspaceFiles } from "./context";
import { ConverterError, runConverter } from "./runner";
import { blockingQuestions, type VibeBundle } from "./types";

const DIAGNOSTIC_SOURCE = "DSPy Vibe";

let lastBundle: VibeBundle | undefined;
let diagnostics: vscode.DiagnosticCollection;
let output: vscode.OutputChannel;

function config() {
  return vscode.workspace.getConfiguration("dspyVibe");
}

function workspaceRoot(): vscode.Uri | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri;
}

/**
 * Resolve the interpreter: explicit setting first, then whatever the Python
 * extension has selected, then PATH. Borrowing the Python extension's choice
 * matters — a user who selected a venv expects that venv, not a system Python
 * where dspy_vibe is not installed.
 */
async function resolvePython(): Promise<string> {
  const configured = config().get<string>("pythonPath")?.trim();
  if (configured) {
    return configured;
  }
  const pythonExtension = vscode.extensions.getExtension("ms-python.python");
  if (pythonExtension) {
    try {
      const api = pythonExtension.isActive ? pythonExtension.exports : await pythonExtension.activate();
      const resource = workspaceRoot();
      const details = await api?.environments?.resolveEnvironment?.(
        api.environments.getActiveEnvironmentPath(resource),
      );
      const executable = details?.executable?.uri?.fsPath ?? details?.path;
      if (executable) {
        return executable;
      }
    } catch {
      // The Python extension's API is optional; fall through to PATH.
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

function workspaceFiles(root: vscode.Uri): WorkspaceFiles {
  return {
    async read(relativePath) {
      try {
        const bytes = await vscode.workspace.fs.readFile(vscode.Uri.joinPath(root, relativePath));
        return Buffer.from(bytes).toString("utf-8");
      } catch {
        return undefined;
      }
    },
  };
}

function workspaceFileSystem(root: vscode.Uri): FileSystem {
  return {
    async exists(relativePath) {
      try {
        await vscode.workspace.fs.stat(vscode.Uri.joinPath(root, relativePath));
        return true;
      } catch {
        return false;
      }
    },
    async write(relativePath, content) {
      const target = vscode.Uri.joinPath(root, relativePath);
      await vscode.workspace.fs.createDirectory(vscode.Uri.joinPath(target, ".."));
      await vscode.workspace.fs.writeFile(target, Buffer.from(content, "utf-8"));
    },
  };
}

/**
 * Surface open questions where the user is already looking.
 *
 * Blocking questions are warnings; answered ones are hints. This is the part
 * that earns the extension: the gaps in a vibe instruction become visible in
 * the editor before any code gets written against a guess.
 */
function publishQuestions(bundle: VibeBundle, document: vscode.TextDocument, range: vscode.Range): void {
  const items = (bundle.spec.open_questions ?? []).map((question) => {
    const blocking = !question.assumption_used?.trim();
    const message = blocking
      ? `Unanswered: ${question.question} — ${question.why_it_matters}`
      : `Assumption: ${question.question} → ${question.assumption_used}`;
    const diagnostic = new vscode.Diagnostic(
      range,
      message,
      blocking ? vscode.DiagnosticSeverity.Warning : vscode.DiagnosticSeverity.Hint,
    );
    diagnostic.source = DIAGNOSTIC_SOURCE;
    return diagnostic;
  });
  diagnostics.set(document.uri, items);
}

async function convert(instruction: string, document?: vscode.TextDocument, range?: vscode.Range): Promise<void> {
  if (!instruction.trim()) {
    vscode.window.showWarningMessage("DSPy Vibe: nothing to convert.");
    return;
  }
  const root = workspaceRoot();
  const settings = config();
  const model = settings.get<string>("model")?.trim() ?? "";

  const context =
    settings.get<boolean>("autoContext") && root ? await detectContext(workspaceFiles(root)) : "";

  const bundle = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: model ? `DSPy Vibe: converting with ${model}…` : "DSPy Vibe: converting (offline)…",
      cancellable: false,
    },
    async () =>
      runConverter({
        pythonPath: await resolvePython(),
        instruction,
        context,
        tools: settings.get<string[]>("tools") ?? [],
        model,
        cwd: root?.fsPath,
        timeoutSeconds: settings.get<number>("timeoutSeconds") ?? 120,
      }),
  );

  lastBundle = bundle;

  const preview = await vscode.workspace.openTextDocument({
    content: bundle.rendered.spec,
    language: "markdown",
  });
  await vscode.window.showTextDocument(preview, { preview: true, viewColumn: vscode.ViewColumn.Beside });

  if (document && range) {
    publishQuestions(bundle, document, range);
  }

  const blocking = blockingQuestions(bundle.spec);
  const summary = `DSPy Vibe: brief ready (confidence ${bundle.spec.confidence}).`;
  const action = await vscode.window.showInformationMessage(
    blocking.length > 0 ? `${summary} ${blocking.length} question(s) need an answer first.` : summary,
    "Generate .agent and .skill",
  );
  if (action) {
    await vscode.commands.executeCommand("dspyVibe.writeArtifacts");
  }
}

async function writeLastBundle(): Promise<void> {
  if (!lastBundle) {
    vscode.window.showWarningMessage("DSPy Vibe: convert an instruction first.");
    return;
  }
  const root = workspaceRoot();
  if (!root) {
    vscode.window.showWarningMessage("DSPy Vibe: open a folder to write artifacts into.");
    return;
  }
  const settings = config();
  const planned = planArtifacts(lastBundle, {
    agentDirectory: settings.get<string>("agentDirectory") ?? ".claude/agents",
    skillDirectory: settings.get<string>("skillDirectory") ?? ".claude/skills",
    specDirectory: settings.get<string>("specDirectory") ?? "docs/briefs",
  });

  let outcome = await writeArtifacts(planned, workspaceFileSystem(root), false);
  if (outcome.skipped.length > 0) {
    const choice = await vscode.window.showWarningMessage(
      `DSPy Vibe: ${outcome.skipped.length} file(s) already exist. Overwrite?`,
      { modal: true, detail: outcome.skipped.map((item) => item.path).join("\n") },
      "Overwrite",
    );
    if (choice === "Overwrite") {
      outcome = await writeArtifacts(outcome.skipped, workspaceFileSystem(root), true);
    }
  }

  for (const artifact of outcome.written) {
    output.appendLine(`wrote ${artifact.path}`);
  }
  if (outcome.written.length > 0) {
    vscode.window.showInformationMessage(
      `DSPy Vibe: wrote ${outcome.written.map((item) => item.path).join(", ")}`,
    );
  }
}

async function checkEnvironment(): Promise<void> {
  const python = await resolvePython();
  output.show(true);
  output.appendLine(`interpreter: ${python}`);
  try {
    const bundle = await runConverter({ pythonPath: python, instruction: "smoke test", timeoutSeconds: 60 });
    output.appendLine(`dspy_vibe responded; sample slug: ${bundle.spec.slug}`);
    vscode.window.showInformationMessage("DSPy Vibe: the environment works.");
  } catch (error) {
    reportError(error);
  }
}

function reportError(error: unknown): void {
  if (error instanceof ConverterError) {
    output.appendLine(`${error.message}\n${error.detail}`);
    output.show(true);
    vscode.window.showErrorMessage(`DSPy Vibe: ${error.message}`, "Show details").then((choice) => {
      if (choice) {
        output.show(true);
      }
    });
    return;
  }
  output.appendLine(String(error));
  vscode.window.showErrorMessage(`DSPy Vibe: ${String(error)}`);
}

function guarded(handler: () => Promise<void>): () => Promise<void> {
  return async () => {
    try {
      await handler();
    } catch (error) {
      reportError(error);
    }
  };
}

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("DSPy Vibe");
  diagnostics = vscode.languages.createDiagnosticCollection("dspyVibe");
  context.subscriptions.push(output, diagnostics);

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "dspyVibe.convertSelection",
      guarded(async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) {
          vscode.window.showWarningMessage("DSPy Vibe: select the instruction text first.");
          return;
        }
        await convert(editor.document.getText(editor.selection), editor.document, editor.selection);
      }),
    ),
    vscode.commands.registerCommand(
      "dspyVibe.convertPrompt",
      guarded(async () => {
        const instruction = await vscode.window.showInputBox({
          prompt: "What do you want built?",
          placeHolder: "csinálj egy sötét témát a dashboardra, ne nyúlj a loginhoz",
        });
        if (instruction !== undefined) {
          await convert(instruction);
        }
      }),
    ),
    vscode.commands.registerCommand("dspyVibe.writeArtifacts", guarded(writeLastBundle)),
    vscode.commands.registerCommand("dspyVibe.checkEnvironment", guarded(checkEnvironment)),
  );
}

export function deactivate(): void {
  lastBundle = undefined;
}
