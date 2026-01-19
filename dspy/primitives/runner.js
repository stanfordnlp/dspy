// Adapted from "Simon Willison's TILs" (https://til.simonwillison.net/deno/pyodide-sandbox)

import pyodideModule from "npm:pyodide/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

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

class FinalOutput(BaseException):
    # Control-flow exception to signal completion (like StopIteration)
    pass

# Default SUBMIT for single-output signatures (e.g., Program of Thought).
# Only define if not already registered with typed signatures.
if 'SUBMIT' not in dir():
    def SUBMIT(output):
        raise FinalOutput({"output": output})
`;

// Generate a tool wrapper function with typed signature.
// Parameters is an array of {name, type?, default?} objects.
// Convert a JavaScript/JSON value to Python literal syntax
const toPythonLiteral = (value) => {
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  return JSON.stringify(value);  // Works for strings, numbers, arrays, objects
};

const makeToolWrapper = (toolName, parameters = []) => {
  // Build signature parts: "query: str, limit: int = 10"
  const sigParts = parameters.map(p => {
    let part = p.name;
    if (p.type) part += `: ${p.type}`;
    if (p.default !== undefined) part += ` = ${toPythonLiteral(p.default)}`;
    return part;
  });
  const signature = sigParts.join(', ');
  const argNames = parameters.map(p => p.name);

  // If no parameters, fall back to *args, **kwargs for flexibility
  if (parameters.length === 0) {
    return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(*args, **kwargs):
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"args": args, "kwargs": kwargs})))
    return result.to_py() if isinstance(result, JsProxy) else result
`;
  }

  return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(${signature}):
    _args = [${argNames.join(', ')}]
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"args": _args, "kwargs": {}})))
    return result.to_py() if isinstance(result, JsProxy) else result
`;
};

// Generate SUBMIT function with output field signature.
// Outputs is an array of {name, type?} objects.
const makeSubmitWrapper = (outputs) => {
  if (!outputs || outputs.length === 0) {
    // Fallback to single-arg SUBMIT if no outputs defined
    return `
def SUBMIT(output):
    raise FinalOutput({"output": output})
`;
  }

  const sigParts = outputs.map(o => {
    let part = o.name;
    if (o.type) part += `: ${o.type}`;
    return part;
  });
  const dictParts = outputs.map(o => `"${o.name}": ${o.name}`);

  return `
def SUBMIT(${sigParts.join(', ')}):
    raise FinalOutput({${dictParts.join(', ')}})
`;
};

// Global handler to prevent uncaught promise rejections from crashing Deno
// These can occur during async Python <-> JS interop
globalThis.addEventListener("unhandledrejection", (event) => {
  event.preventDefault();
  console.log(JSON.stringify({
    error: `Unhandled async error: ${event.reason?.message || event.reason}`,
    errorType: "UnhandledPromiseRejection"
  }));
});

const pyodide = await pyodideModule.loadPyodide();

// Tool call support: allows Python code to call host-side functions
// The stdin reader is shared so tool_call can read responses during execution
const stdinReader = readLines(Deno.stdin);
let requestIdCounter = 0;

// This function is called from Python to invoke a host-side tool
async function toolCallBridge(name, argsJson) {
  const requestId = `req_${Date.now()}_${++requestIdCounter}`;

  try {
    // Send tool call request to host
    console.log(JSON.stringify({
      type: "tool_call",
      id: requestId,
      name: name,
      args: JSON.parse(argsJson)
    }));

    // Wait for response from host
    const { value: responseLine, done } = await stdinReader.next();
    if (done) {
      throw new Error("stdin closed while waiting for tool response");
    }

    const response = JSON.parse(responseLine);
    if (response.type !== "tool_response" || response.id !== requestId) {
      throw new Error(`Unexpected response: expected tool_response with id ${requestId}`);
    }

    if (response.error) {
      throw new Error(response.error);
    }

    // Deserialize result based on type
    if (response.result_type === "json") {
      return JSON.parse(response.result);
    }
    return response.result;
  } catch (error) {
    // Re-throw with context so Python can catch it properly
    throw new Error(`Tool bridge error for '${name}': ${error.message}`);
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
    console.log(JSON.stringify({
      error: "Invalid JSON input: " + error.message,
      errorType: "ValueError"
    }));
    continue;
  }

  if (input.mount_file) {
    const virtualPath = input.virtual_path || input.mount_file;
    try {
      const contents = await Deno.readFile(input.mount_file);
      const dirs = virtualPath.split('/').slice(1, -1);
      let cur = '';
      for (const d of dirs) {
        cur += '/' + d;
        try {
          pyodide.FS.mkdir(cur);
        } catch (e) {
          if (!e.message?.includes('File exists')) {
            throw e;
          }
        }
      }
      pyodide.FS.writeFile(virtualPath, contents);
    } catch (e) {
      console.log(JSON.stringify({error: "Failed to mount file: " + e.message}));
    }
    continue;
  }

  if (input.sync_file) {
    try {
      await Deno.writeFile(input.host_file || input.sync_file, pyodide.FS.readFile(input.sync_file));
    } catch (e) { /* ignore sync errors */ }
    continue;
  }

  if (typeof input !== 'object' || input === null) {
    console.log(JSON.stringify({ error: "Input is not a JSON object", errorType: "ValueError" }));
    continue;
  }

  if (input.shutdown) break;

  // Register tools and/or output fields
  if (input.register_tools || input.register_outputs) {
    const toolNames = [];

    // Register tools with typed signatures
    if (input.register_tools) {
      for (const tool of input.register_tools) {
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
    if (input.register_outputs) {
      pyodide.runPython(makeSubmitWrapper(input.register_outputs));
    }

    console.log(JSON.stringify({
      tools_registered: toolNames,
      outputs_registered: input.register_outputs ? input.register_outputs.map(o => o.name) : []
    }));
    continue;
  }

  const code = input.code || "";

  try {
    await pyodide.loadPackagesFromImports(code);
    pyodide.runPython(PYTHON_SETUP_CODE);
    // Run the user's code
    const result = await pyodide.runPythonAsync(code);
    const capturedStdout = pyodide.runPython("buf_stdout.getvalue()");
    pyodide.runPython("sys.stdout, sys.stderr = old_stdout, old_stderr");

    // If result is None, output prints; otherwise output the result
    let output = (result === null || result === undefined) ? capturedStdout : (result.toJs?.() ?? result);
    console.log(JSON.stringify({ output }));
  } catch (error) {
    // We have an error => check if it's a SyntaxError or something else
    // The Python error class name is stored in error.type: https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.ffi.PythonError
    const errorType = error.type || "Error";
    // error.message is mostly blank.
    const errorMessage = (error.message || "").trim();
    // The arguments of the exception are stored in sys.last_exc.args,
    // which is always helpful but pyodide don't extract them for us.
    // Use a bridge function to get them.
    let errorArgs = [];
    if (errorType !== "SyntaxError") {
      // Only python exceptions have args.
      const last_exception_args = pyodide.globals.get("last_exception_args");
      // Regarding https://pyodide.org/en/stable/usage/type-conversions.html#type-translations-errors,
      // we do a additional `json.dumps` and `JSON.parse` on the values, to avoid the possible memory leak.
      errorArgs = JSON.parse(last_exception_args()) || [];
    }
    console.log(JSON.stringify({ error: errorMessage, errorArgs, errorType }));
  }
}
