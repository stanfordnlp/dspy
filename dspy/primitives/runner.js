// Adapted from "Simon Willison's TILs" (https://til.simonwillison.net/deno/pyodide-sandbox)

import pyodideModule from "npm:pyodide/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

// =============================================================================
// Python Code Templates
// =============================================================================

// Model reconstruction code: executed in the sandbox when typed models are
// needed.  Builds lightweight proxy classes (no pydantic dependency) that
// provide attribute access, dict access, model_dump(), and proper type names.
const MODEL_RECONSTRUCTION_CODE = `
if "_REGISTERED_MODELS" not in dir():
    _REGISTERED_MODELS = {}

def _make_model_class(_name, _field_types):
    def __init__(self, **kwargs):
        object.__setattr__(self, '_data', {})
        for k, v in kwargs.items():
            ft = self._field_types.get(k)
            fc = _REGISTERED_MODELS.get(ft) if ft else None
            if isinstance(v, dict) and fc:
                v = fc(**v)
            elif isinstance(v, list) and fc:
                v = [fc(**i) if isinstance(i, dict) else i for i in v]
            self._data[k] = v
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, '_data')[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    def __repr__(self):
        items = ', '.join(f'{k}={v!r}' for k, v in self._data.items())
        return f'{type(self).__name__}({items})'
    def __getitem__(self, key):
        return self._data[key]
    def keys(self):
        return self._data.keys()
    def items(self):
        return self._data.items()
    def values(self):
        return self._data.values()
    def model_dump(self):
        out = {}
        for k, v in self._data.items():
            if hasattr(v, 'model_dump'):
                out[k] = v.model_dump()
            elif isinstance(v, list):
                out[k] = [i.model_dump() if hasattr(i, 'model_dump') else i for i in v]
            else:
                out[k] = v
        return out
    return type(_name, (), {
        '__init__': __init__, '__getattr__': __getattr__,
        '__repr__': __repr__, '__getitem__': __getitem__,
        'keys': keys, 'items': items, 'values': values,
        'model_dump': model_dump, '_field_types': _field_types,
    })

import json as _json
_defs = _json.loads(_model_defs_json)
_order = _json.loads(_model_order_json)
for _name in _order:
    _field_types = {}
    for _fname, _fspec in _defs[_name].items():
        _ft = _fspec["type"]
        if _ft in _REGISTERED_MODELS:
            _field_types[_fname] = _ft
        elif _ft.startswith("list[") and _ft.endswith("]") and _ft[5:-1] in _REGISTERED_MODELS:
            _field_types[_fname] = _ft[5:-1]
    _cls = _make_model_class(_name, _field_types)
    _REGISTERED_MODELS[_name] = _cls
    globals()[_name] = _cls
del _defs, _order, _model_defs_json, _model_order_json
`;

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

const JSON_SCHEMA_TYPE_TO_PYTHON = {
  string: "str",
  integer: "int",
  number: "float",
  boolean: "bool",
  array: "list",
  object: "dict",
  null: "None",
};

const TOOL_BRIDGE_ERROR_KEY = "__dspy_tool_bridge_error__";

const makeToolWrapper = (toolName, parameters = []) => {
  const hasModelParams = parameters.some(p => p.model_type);

  // Build signature parts: "query: str, limit: int = 10"
  const sigParts = parameters.map(p => {
    let part = p.name;
    // Prefer model_type (real pydantic class in sandbox), then simple type, then inferred from schema
    if (p.model_type) {
      part += `: ${p.model_type}`;
    } else {
      const inferredType = (!p.type && p.json_schema && typeof p.json_schema.type === "string")
        ? JSON_SCHEMA_TYPE_TO_PYTHON[p.json_schema.type]
        : null;
      const pythonType = p.type || inferredType;
      if (pythonType) part += `: ${pythonType}`;
    }
    if (p.default !== undefined) part += ` = ${toPythonLiteral(p.default)}`;
    return part;
  });
  const signature = sigParts.join(', ');
  const argNames = parameters.map(p => p.name);
  const kwargParts = argNames.map(n => `"${n}": ${n}`).join(', ');

  // When model-typed params exist, serialize BaseModel instances before the JSON bridge
  if (hasModelParams) {
    return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(${signature}):
    _kw = {${kwargParts}}
    _ser = {_k: _v.model_dump() if hasattr(_v, 'model_dump') else _v for _k, _v in _kw.items()}
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"kwargs": _ser})))
    parsed = result.to_py() if isinstance(result, JsProxy) else result
    if isinstance(parsed, dict) and parsed.get("${TOOL_BRIDGE_ERROR_KEY}"):
        raise RuntimeError(parsed.get("message", "Tool bridge error"))
    return parsed
`;
  }

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
// Outputs is an array of {name, type?, model_type?, json_schema?} objects.
const makeSubmitWrapper = (outputs) => {
  if (!outputs || outputs.length === 0) {
    // Fallback to single-arg SUBMIT if no outputs defined
    return `
def SUBMIT(output):
    raise FinalOutput({"output": output})
`;
  }

  const hasModelOutputs = outputs.some(o => o.json_schema);

  // SUBMIT type hints use inferred types (not model_type) since output
  // model classes may not be registered in the sandbox.
  const sigParts = outputs.map(o => {
    let part = o.name;
    const inferredType = (!o.type && o.json_schema && typeof o.json_schema.type === "string")
      ? JSON_SCHEMA_TYPE_TO_PYTHON[o.json_schema.type]
      : null;
    const pythonType = o.type || inferredType;
    if (pythonType) part += `: ${pythonType}`;
    return part;
  });
  const dictParts = outputs.map(o => `"${o.name}": ${o.name}`);

  // Build docstring with schema info for complex output fields
  const schemaLines = outputs
    .filter(o => o.json_schema)
    .map(o => `    ${o.name}: ${JSON.stringify(o.json_schema)}`);
  const docstring = schemaLines.length > 0
    ? `    """Expected output schemas:\\n${schemaLines.join('\\n')}\\n    """\n`
    : '';

  // When model outputs exist, serialize BaseModel instances before raising
  if (hasModelOutputs) {
    return `
def SUBMIT(${sigParts.join(', ')}):
${docstring}    _out = {${dictParts.join(', ')}}
    _ser = {_k: _v.model_dump() if hasattr(_v, 'model_dump') else _v for _k, _v in _out.items()}
    raise FinalOutput(_ser)
`;
  }

  return `
def SUBMIT(${sigParts.join(', ')}):
${docstring}    raise FinalOutput({${dictParts.join(', ')}})
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

  if (method === "register_models") {
    try {
      pyodide.globals.set("_model_defs_json", JSON.stringify(params.models));
      pyodide.globals.set("_model_order_json", JSON.stringify(params.model_order));
      pyodide.runPython(MODEL_RECONSTRUCTION_CODE);
      console.log(jsonrpcResult({ models: params.model_order }, requestId));
    } catch (e) {
      console.log(jsonrpcError(JSONRPC_APP_ERRORS.RuntimeError, `Failed to register models: ${e.message}`, requestId));
    }
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
