// Adapted from "Simon Willison's TILs" (https://til.simonwillison.net/deno/pyodide-sandbox)

import pyodideModule from "npm:pyodide@0.29.4/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

const JSON = Object.freeze(globalThis.JSON), console = Object.freeze(globalThis.console);
// =============================================================================
// Python Code Templates
// =============================================================================

// Setup code run before each user code execution.
// Captures stdout, defines SUBMIT for early termination, and
// provides a helper to extract exception args across the JS/Python boundary.
const PYTHON_SETUP_CODE = `
import sys, io, json
old_stdout, old_stderr = sys.stdout, sys.stderr
buf_stdout, buf_stderr = io.StringIO(), io.StringIO()
sys.stdout, sys.stderr = buf_stdout, buf_stderr

def last_exception_args():
    return json.dumps(sys.last_exc.args) if sys.last_exc else None

class _DSPyFinalOutput(BaseException):
    # Control-flow exception to signal completion (like StopIteration)
    pass

# Default SUBMIT for single-output signatures.
# Only define if not already registered with typed signatures.
if 'SUBMIT' not in dir():
    def SUBMIT(output):
        raise _DSPyFinalOutput({"output": output})
`;

// Generate a tool wrapper function with typed signature.
// Parameters is an array of {name, type?, default?} objects.
// Convert a JavaScript/JSON value to Python literal syntax
const toPythonLiteral = (value) => {
  const serialized = JSON.stringify(value);
  return `__import__("json").loads(${JSON.stringify(serialized)})`;
};

const toPythonType = (type) => type === "NoneType" ? "type(None)" : type;

const makeToolWrapper = (toolName, parameters = []) => {
  // Build signature parts: "query: str, limit: int = 10"
  const sigParts = parameters.map(p => {
    let part = p.name;
    if (p.type) part += `: ${toPythonType(p.type)}`;
    if (p.default !== undefined) part += ` = ${toPythonLiteral(p.default)}`;
    return part;
  });
  const signature = sigParts.join(', ');
  const argNames = parameters.map(p => p.name);
  const kwargParts = argNames.map(n => `"${n}": ${n}`).join(', ');

  return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(${signature}):
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"kwargs": {${kwargParts}}})))
    parsed = result.to_py() if isinstance(result, JsProxy) else result
    if isinstance(parsed, dict) and parsed.get("${TOOL_BRIDGE_ERROR_KEY}"):
        raise RuntimeError(parsed.get("message", "Tool bridge error"))
    return parsed
`;
};

// Generate SUBMIT function with output field signature.
// Outputs is an array of {name, type?} objects.
const makeSubmitWrapper = (outputs) => {
  if (!outputs || outputs.length === 0) {
    // Fallback to single-arg SUBMIT if no outputs defined
    return `
def SUBMIT(output):
    raise _DSPyFinalOutput({"output": output})
`;
  }

  const sigParts = outputs.map(o => {
    let part = o.name;
    if (o.type) part += `: ${toPythonType(o.type)}`;
    return part;
  });
  const dictParts = outputs.map(o => `"${o.name}": ${o.name}`);

  return `
def SUBMIT(${sigParts.join(', ')}):
    raise _DSPyFinalOutput({${dictParts.join(', ')}})
`;
};

// =============================================================================
// JSON-RPC 2.0 Helpers
// =============================================================================

// JSON-RPC 2.0 protocol errors (reserved range: -32700 to -32600)
const JSONRPC_PROTOCOL_ERRORS = {
  ParseError: -32700,
  InvalidRequest: -32600,
  MethodNotFound: -32601,
};

// Application errors (range: -32000 to -32099)
const JSONRPC_APP_ERRORS = {
  SyntaxError: -32000,
  NameError: -32001,
  TypeError: -32002,
  ValueError: -32003,
  AttributeError: -32004,
  IndexError: -32005,
  KeyError: -32006,
  RuntimeError: -32007,
  CodeInterpreterError: -32008,
  Unknown: -32099,
};

const jsonrpcRequest = (method, params, id) =>
  JSON.stringify({ __proto__: null, jsonrpc: "2.0", method, params: { __proto__: null, ...params }, id });

const jsonrpcNotification = (method, params = null) => {
  const msg = { jsonrpc: "2.0", method };
  if (params) msg.params = params;
  return JSON.stringify(msg);
};

const jsonrpcResult = (result, id) =>
  JSON.stringify({ jsonrpc: "2.0", result, id });

const jsonrpcError = (code, message, id, data = null) => {
  const err = { code, message };
  if (data) err.data = data;
  return JSON.stringify({ jsonrpc: "2.0", error: err, id });
};

// Global handler to prevent uncaught promise rejections from crashing Deno
// These can occur during async Python <-> JS interop
globalThis.addEventListener("unhandledrejection", (event) => {
  event.preventDefault();
  console.log(jsonrpcNotification("unhandled_error", {
    message: `Unhandled async error: ${event.reason?.message || event.reason}`,
  }));
});

// Archives built by dspy/primitives/sandbox_bundle.py: Pyodide's standard library with its bytecode, which Pyodide
// starts from instead of compiling the library on every start, the dspy bundle, installed before any code runs, and a
// memory snapshot of a sandbox that has imported dspy, which a sandbox restores instead of importing dspy itself.
const argValue = (name) => Deno.args.find((arg) => arg.startsWith(`${name}=`))?.slice(name.length + 1);
const stdlib = argValue("--dspy-stdlib");
const bundle = argValue("--dspy-bundle");
const snapshot = argValue("--dspy-snapshot");
const buildTarget = argValue("--dspy-build");
const buildSource = argValue("--dspy-build-from");
const snapshotTarget = argValue("--dspy-build-snapshot");
// The host starts the sandbox again without its snapshot when it exits with this.
const SNAPSHOT_REFUSED = 70;

// The bundle's files are written into Pyodide's file system before Python starts. A snapshot holds the memory of the
// bundle's compiled extension modules but not their code: they are loaded right after the C runtime starts and before
// Python does, in the snapshot build and in every restore alike, so they land where that memory expects them. `layout`
// records where they landed, and a restore that differs is refused.
let layout = null;
if (bundle) {
  await import("npm:pyodide@0.29.4/pyodide.asm.js"); // defines the module factory loadPyodide uses
  const create = globalThis._createPyodideModule;
  globalThis._createPyodideModule = (settings) => {
    let extensions = [];
    settings.preRun.push((Module) => {
      const files = writeArchive(Module.FS, Module.API.sitePackages, Deno.readFileSync(bundle));
      // Python's extension modules, not the libraries they link, like libssl.so, which they load by name from here.
      extensions = files.filter((path) => path.endsWith(".so") && !/\.libs\/|\/lib[^/]*$/.test(path)).sort();
      Module.ENV.LD_LIBRARY_PATH = Module.API.sitePackages;
      Module.ENV.LANG = "C.UTF-8"; // not the host's locale, which would change what the runtime allocates
    });
    if (snapshot || snapshotTarget) {
      settings.onRuntimeInitialized = () => {
        for (const path of extensions) {
          settings.loadDynamicLibrary(path, { global: true, nodelete: true, allowUndefined: true });
        }
        const zip = settings.FS.readdir("/lib").find((name) => name.endsWith(".zip")); // the standard library
        layout = JSON.stringify([
          settings.FS.stat(`/lib/${zip}`).size,
          settings.wasmTable.length,
          ...extensions.map((path) =>
            Object.values(settings.LDSO.loadedLibsByName[path].exports).find((value) => typeof value === "number")
          ),
        ]);
      };
    }
    return create(settings);
  };
}

// Writes the files of a stored (uncompressed) zip under `root`, much faster than unpacking it in Python.
function writeArchive(FS, root, data) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let end = data.length - 22;
  while (view.getUint32(end, true) !== 0x06054b50) end--;
  const decoder = new TextDecoder(), files = [];
  for (let i = 0, at = view.getUint32(end + 16, true); i < view.getUint16(end + 10, true); i++) {
    const nameLength = view.getUint16(at + 28, true);
    const name = decoder.decode(data.subarray(at + 46, at + 46 + nameLength));
    const size = view.getUint32(at + 24, true), local = view.getUint32(at + 42, true);
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    if (!name.endsWith("/")) {
      const path = `${root}/${name}`;
      FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
      FS.writeFile(path, data.subarray(start, start + size), { canOwn: true });
      files.push(path);
    }
    at += 46 + nameLength + view.getUint16(at + 30, true) + view.getUint16(at + 32, true);
  }
  return files;
}

let pyodide;
try {
  pyodide = await pyodideModule.loadPyodide({
    ...(stdlib ? { stdLibURL: stdlib } : {}),
    ...(snapshot ? { _loadSnapshot: Deno.readFileSync(snapshot) } : {}),
    ...(snapshotTarget ? { _makeSnapshot: true } : {}),
  });
} catch (e) {
  if (!snapshot) throw e;
  console.error(`The dspy snapshot could not be restored: ${e?.message ?? e}`);
  Deno.exit(SNAPSHOT_REFUSED);
}
const denoDir = argValue("--dspy-deno-dir");
if (denoDir) await Deno.permissions.revoke({ name: "read", path: denoDir });
if (stdlib) await Deno.permissions.revoke({ name: "read", path: stdlib });

// Build mode runs no sandboxed code: it byte-compiles an archive (by default the standard library), writes it, and
// exits.
if (buildTarget) {
  if (buildSource) pyodide.FS.writeFile("/tmp/source.zip", Deno.readFileSync(buildSource));
  pyodide.runPython(`
import compileall, os, py_compile, site, sys, zipfile

stdlib = next(path for path in sys.path if path.endswith(".zip"))
legacy = ${buildSource ? "False" : "True"}
zipfile.ZipFile(stdlib if legacy else "/tmp/source.zip").extractall("/tmp/build")
# The standard library's bytecode goes beside each module, where zipimport looks for it; a bundle's goes in
# __pycache__ of the site-packages it is unpacked into.
compileall.compile_dir("/tmp/build", quiet=2, legacy=legacy, ddir=stdlib if legacy else site.getsitepackages()[0],
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
with zipfile.ZipFile("/tmp/build.zip", "w") as archive:
    for folder, _, files in os.walk("/tmp/build"):
        for name in files:
            archive.write(os.path.join(folder, name), os.path.relpath(os.path.join(folder, name), "/tmp/build"))
`);
  Deno.writeFileSync(buildTarget, pyodide.FS.readFile("/tmp/build.zip"));
  Deno.exit(0);
}

if (bundle) {
  await Deno.permissions.revoke({ name: "read", path: bundle });
  if (snapshot) {
    await Deno.permissions.revoke({ name: "read", path: snapshot });
    // Reading a str runs no extension code, so this is safe even when they landed elsewhere.
    if (pyodide.runPython("import sys; getattr(sys, '__dspy_layout', None)") !== layout) {
      console.error("The dspy snapshot does not match where this sandbox loaded its extension modules.");
      Deno.exit(SNAPSHOT_REFUSED);
    }
    // Sandboxes restored from one snapshot would otherwise all draw the same random numbers.
    pyodide.runPython("import random; random.seed()");
  } else {
    // A snapshot has this already.
    pyodide.runPython(`
import concurrent.futures

class InlineExecutor(concurrent.futures.Executor):
    # Pyodide cannot start threads: run each submitted call before returning its future.
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:
            future.set_exception(e)
        return future

concurrent.futures.ThreadPoolExecutor = InlineExecutor
del InlineExecutor
`);
  }
  // Snapshot build mode runs no sandboxed code either: it imports dspy, writes the sandbox's memory, and exits.
  if (snapshotTarget) {
    pyodide.pyimport("dspy");
    pyodide.runPython(`import sys; sys.__dspy_layout = ${JSON.stringify(layout)}`);
    for (const key of Object.keys(pyodide.loadedPackages)) delete pyodide.loadedPackages[key]; // must be empty in it
    Deno.writeFileSync(snapshotTarget, pyodide.makeMemorySnapshot());
    Deno.exit(0);
  }
  // Installed now, so loadPackagesFromImports must not fetch them.
  const sitePackages = pyodide.runPython("import site; site.getsitepackages()[0]");
  for (const entry of pyodide.FS.readdir(sitePackages)) {
    const name = entry.endsWith(".dist-info") ? entry.split("-")[0].toLowerCase() : null;
    if (name) for (const key of [name.replaceAll("_", "-"), name.replaceAll("-", "_")]) pyodide.loadedPackages[key] = "dspy";
  }
}

// Tool call support: allows Python code to call host-side functions
// The stdin reader is shared so tool_call can read responses during execution
const stdinReader = readLines(Deno.stdin);
let requestIdCounter = 0;

const TOOL_BRIDGE_ERROR_KEY = "__dspy_tool_bridge_error__";

// This function is called from Python to invoke a host-side tool
async function toolCallBridge(name, argsJson) {
  const requestId = `tc_${Date.now()}_${++requestIdCounter}`;

  try {
    const parsedArgs = JSON.parse(argsJson);

    // Send tool call request to host using JSON-RPC
    console.log(jsonrpcRequest("tool_call", {
      name: name,
      kwargs: parsedArgs.kwargs || {}
    }, requestId));

    // Wait for response from host
    const { value: responseLine, done } = await stdinReader.next();
    if (done) {
      throw new Error("stdin closed while waiting for tool response");
    }

    const response = JSON.parse(responseLine);

    // Expect JSON-RPC result or error with matching id
    if (response.id !== requestId) {
      return {
        [TOOL_BRIDGE_ERROR_KEY]: true,
        message: `Tool bridge error for '${name}': Unexpected response: expected id ${requestId}, got ${response.id}`
      };
    }

    if (response.error) {
      const errorType = response.error.data?.type || "ToolError";
      const errorMessage = response.error.message || "Tool call failed";
      return {
        [TOOL_BRIDGE_ERROR_KEY]: true,
        message: `${errorType}: ${errorMessage}`
      };
    }

    // Deserialize result based on type
    const result = response.result;
    if (result.type === "json") {
      return JSON.parse(result.value);
    }
    return result.value;
  } catch (error) {
    // Return a structured error payload so Python can raise with full context
    // without triggering a top-level unhandled rejection in Deno.
    return {
      [TOOL_BRIDGE_ERROR_KEY]: true,
      message: `Tool bridge error for '${name}': ${error.message}`
    };
  }
}

// Expose the bridge to Python
pyodide.globals.set("_js_tool_call", toolCallBridge);

try {
  const env_vars = (Deno.args[0] ?? "").split(",").filter(Boolean);
  for (const key of env_vars) {
    const val = Deno.env.get(key);
    if (val !== undefined) {
      pyodide.runPython(`
import os
os.environ[${JSON.stringify(key)}] = ${JSON.stringify(val)}
      `);
    }
  }
} catch (e) {
  console.error("Error setting environment variables in Pyodide:", e);
}

// Main loop using shared stdin reader
while (true) {
  const { value: line, done } = await stdinReader.next();
  if (done) break;

  let input;
  try {
    input = JSON.parse(line);
  } catch (error) {
    // JSON-RPC parse error
    console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.ParseError, "Invalid JSON input: " + error.message, null));
    continue;
  }

  // Validate JSON-RPC format
  if (typeof input !== 'object' || input === null || input.jsonrpc !== "2.0") {
    console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.InvalidRequest, "Invalid Request: not a JSON-RPC 2.0 message", null));
    continue;
  }

  const method = input.method;
  const params = input.params || {};
  const requestId = input.id; // May be undefined for notifications

  // Handle notifications (no response expected)
  if (method === "sync_file") {
    try {
      const virtualPath = params.virtual_path;
      const hostPath = params.host_path || virtualPath;
      await Deno.writeFile(hostPath, pyodide.FS.readFile(virtualPath));
    } catch (e) { /* ignore sync errors */ }
    continue;
  }

  if (method === "shutdown") break;

  // Handle requests (expect response)
  if (method === "mount_file") {
    const hostPath = params.host_path;
    const virtualPath = params.virtual_path || hostPath;
    try {
      const contents = await Deno.readFile(hostPath);
      const dirs = virtualPath.split('/').slice(1, -1);
      let cur = '';
      for (const d of dirs) {
        cur += '/' + d;
        // Check if directory exists before creating
        try {
          pyodide.FS.stat(cur);
          // Directory exists, continue to next
        } catch {
          // Directory doesn't exist, create it
          pyodide.FS.mkdir(cur);
        }
      }
      pyodide.FS.writeFile(virtualPath, contents);
      console.log(jsonrpcResult({ mounted: virtualPath }, requestId));
    } catch (e) {
      console.log(jsonrpcError(JSONRPC_APP_ERRORS.RuntimeError, `Failed to mount file: ${e.message}`, requestId));
    }
    continue;
  }

  if (method === "register") {
    const toolNames = [];

    // Register tools with typed signatures
    if (params.tools) {
      for (const tool of params.tools) {
        // Support both old format (string) and new format (object with parameters)
        if (typeof tool === 'string') {
          pyodide.runPython(makeToolWrapper(tool, []));
          toolNames.push(tool);
        } else {
          pyodide.runPython(makeToolWrapper(tool.name, tool.parameters || []));
          toolNames.push(tool.name);
        }
      }
    }

    // Register SUBMIT with output signature
    if (params.outputs) {
      pyodide.runPython(makeSubmitWrapper(params.outputs));
    }

    console.log(jsonrpcResult({
      tools: toolNames,
      outputs: params.outputs ? params.outputs.map(o => o.name) : []
    }, requestId));
    continue;
  }

  if (method === "inject_var") {
    const { name, value } = params;
    try {
      try { pyodide.FS.mkdir('/tmp'); } catch (e) { /* exists */ }
      try { pyodide.FS.mkdir('/tmp/dspy_vars'); } catch (e) { /* exists */ }
      pyodide.FS.writeFile(`/tmp/dspy_vars/${name}.json`, new TextEncoder().encode(value));
      console.log(jsonrpcResult({ injected: name }, requestId));
    } catch (e) {
      console.log(jsonrpcError(JSONRPC_APP_ERRORS.RuntimeError, `Failed to inject var: ${e.message}`, requestId));
    }
    continue;
  }

  if (method === "execute") {
    const code = params.code || "";
    let setupCompleted = false;  // Track if PYTHON_SETUP_CODE ran successfully

    try {
      await pyodide.loadPackagesFromImports(code);
      pyodide.runPython(PYTHON_SETUP_CODE);
      setupCompleted = true;  // Mark setup as complete - old_stdout/old_stderr now exist

      // Run the user's code
      const result = await pyodide.runPythonAsync(code);
      const capturedStdout = pyodide.runPython("buf_stdout.getvalue()");

      // If result is None, output prints; otherwise output the result
      let output = (result === null || result === undefined) ? capturedStdout : (result.toJs?.() ?? result);
      console.log(jsonrpcResult({ output }, requestId));
    } catch (error) {
      // We have an error => check if it's a SyntaxError or something else
      // The Python error class name is stored in error.type: https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.ffi.PythonError
      const errorType = error.type || "Error";
      // error.message is mostly blank.
      const errorMessage = (error.message || "").trim();

      // Handle SUBMIT's control-flow exception as a success result, not an error
      if (errorType === "_DSPyFinalOutput") {
        const last_exception_args = pyodide.globals.get("last_exception_args");
        const errorArgs = JSON.parse(last_exception_args()) || [];
        const answer = errorArgs[0] || null;
        console.log(jsonrpcResult({ final: answer }, requestId));
        continue;
      }

      // Get error args for other exception types
      let errorArgs = [];
      if (errorType !== "SyntaxError") {
        // Only python exceptions have args.
        const last_exception_args = pyodide.globals.get("last_exception_args");
        // Regarding https://pyodide.org/en/stable/usage/type-conversions.html#type-translations-errors,
        // we do a additional `json.dumps` and `JSON.parse` on the values, to avoid the possible memory leak.
        errorArgs = JSON.parse(last_exception_args()) || [];
      }

      // Map error type to JSON-RPC error code
      const errorCode = JSONRPC_APP_ERRORS[errorType] || JSONRPC_APP_ERRORS.Unknown;
      console.log(jsonrpcError(errorCode, errorMessage, requestId, { type: errorType, args: errorArgs }));
    } finally {
      // Always restore stdout/stderr if setup completed, even after errors.
      // This prevents stream corruption where subsequent executions capture
      // StringIO buffers as old_stdout/old_stderr instead of real streams.
      if (setupCompleted) {
        try {
          pyodide.runPython("sys.stdout, sys.stderr = old_stdout, old_stderr");
        } catch (e) {
          // Ignore restoration errors to avoid masking the original error
        }
      }
    }
    continue;
  }

  // Unknown method
  console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.MethodNotFound, `Method not found: ${method}`, requestId));
}
