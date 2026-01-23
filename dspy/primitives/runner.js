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
  JSON.stringify({ jsonrpc: "2.0", method, params, id });

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
  console.log(jsonrpcError(JSONRPC_APP_ERRORS.RuntimeError, `Unhandled async error: ${event.reason?.message || event.reason}`, null));
});

const pyodide = await pyodideModule.loadPyodide();

// Tool call support: allows Python code to call host-side functions
// The stdin reader is shared so tool_call can read responses during execution
const stdinReader = readLines(Deno.stdin);
let requestIdCounter = 0;

// This function is called from Python to invoke a host-side tool
async function toolCallBridge(name, argsJson) {
  const requestId = `tc_${Date.now()}_${++requestIdCounter}`;

  try {
    // Parse args to extract positional and keyword args
    const parsedArgs = JSON.parse(argsJson);

    // Send tool call request to host using JSON-RPC
    console.log(jsonrpcRequest("tool_call", {
      name: name,
      args: parsedArgs.args || [],
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
      throw new Error(`Unexpected response: expected id ${requestId}, got ${response.id}`);
    }

    if (response.error) {
      throw new Error(response.error.message || "Tool call failed");
    }

    // Deserialize result based on type
    const result = response.result;
    if (result.type === "json") {
      return JSON.parse(result.value);
    }
    return result.value;
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
        try {
          pyodide.FS.mkdir(cur);
        } catch (e) {
          if (!e.message?.includes('File exists')) {
            throw e;
          }
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

      // Handle FinalOutput as a success result, not an error
      if (errorType === "FinalOutput") {
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
