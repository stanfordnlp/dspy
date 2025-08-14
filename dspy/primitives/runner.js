// Adapted from "Simon Willison’s TILs" (https://til.simonwillison.net/deno/pyodide-sandbox)

import pyodideModule from "npm:pyodide/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

const pyodide = await pyodideModule.loadPyodide();

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

for await (const line of readLines(Deno.stdin)) {
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
      const hostPath = input.mount_file;
      const virtualPath = input.virtual_path || hostPath;
      try {
          const contents = await Deno.readFile(hostPath);
          const dirs = virtualPath.split('/').slice(1, -1);
          let cur = '';
          for (const d of dirs) {
              cur += '/' + d;
              try {
                  pyodide.FS.mkdir(cur);
              } catch (e) {
                  if (!(e && e.message && e.message.includes('File exists'))) {
                      console.log("[DEBUG] Error creating directory in Pyodide file system:", cur, "|", e.message);
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
      const virtualPath = input.sync_file;
      const hostPath = input.host_file || virtualPath;
      try {
          const contents = pyodide.FS.readFile(virtualPath);
          await Deno.writeFile(hostPath, contents);
      } catch (e) {
          console.log("[DEBUG] Failed to sync file:", hostPath, "|", e.message);
      }
      continue;
  }


  // Expecting an object like { "code": "...", ... }
  if (typeof input !== 'object' || input === null) {
    console.log(JSON.stringify({
      error: "Input is not a JSON object",
      errorType: "ValueError"
    }));
    continue;
  }

  // Check for shutdown
  if (input.shutdown) {
    break;
  }

  const code = input.code || "";

  // Wrap execution in a try/catch so we can handle syntax errors, etc.
  try {
    await pyodide.loadPackagesFromImports(code);
    // 1. Temporarily override stdout/stderr so we can capture prints.
    pyodide.runPython(`
import sys
import io

# Keep references to the old stdout/stderr so we can restore them later
old_stdout = sys.stdout
old_stderr = sys.stderr

# New "file-like" buffers
buf_stdout = io.StringIO()
buf_stderr = io.StringIO()

sys.stdout = buf_stdout
sys.stderr = buf_stderr
    `);

    // 2. Setup proper exception arguments extractor and FinalAnswer bridge
    // The idea is borrowed from `smolagents` that uses the exception to simulate non-local exit
    pyodide.runPython(`
import json

def last_exception_args():
    return json.dumps(sys.last_exc.args) if sys.last_exc else None 

class FinalAnswer(Exception):
    pass

def final_answer(*args):
    raise FinalAnswer(*args)
      `);

    // 3. Run the user's code asynchronously
    const result = await pyodide.runPythonAsync(code);

    // 4. Retrieve captured stdout/stderr
    const capturedStdout = pyodide.runPython("buf_stdout.getvalue()");
    const capturedStderr = pyodide.runPython("buf_stderr.getvalue()");

    // 5. Restore original stdout/stderr
    pyodide.runPython(`
sys.stdout = old_stdout
sys.stderr = old_stderr
    `);

    // 6. Build our output object according to the rules:
    //    - If result is None (or Python "None" => JS null), output all prints
    //    - Else output the result only
    // Note: `None` in Python becomes `null` in JS.
    let output;
    if (result === null || result === undefined) {
      // The final statement was None or no return => deliver printed output
      // If you want to combine capturedStderr as well, you can append it
      // But here we'll just do stdout for clarity
      output = capturedStdout;
      // If there's something in stderr, you might want to include that or log it
      // output += capturedStderr;
    } else {
      // If the code returned a real value, just return that
      try {
        output = result.toJs();
      } catch (e) {
        output = result;
      }
    }

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


    console.log(JSON.stringify({
      error: errorMessage,
      errorArgs: errorArgs,
      errorType: errorType
    }));
  }
}