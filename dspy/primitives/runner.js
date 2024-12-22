// Adapted from "Simon Willison’s TILs" (https://til.simonwillison.net/deno/pyodide-sandbox)

import pyodideModule from "npm:pyodide/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

const pyodide = await pyodideModule.loadPyodide();

for await (const line of readLines(Deno.stdin)) {
    let input;
    try {
        input = JSON.parse(line);
    } catch (error) {
        console.log(JSON.stringify({ error: "Invalid JSON input: " + error.message }));
        continue;
    }

    if (typeof input !== 'object' || input === null) {
        console.log(JSON.stringify({ error: "Input is not a JSON object" }));
        continue;
    }

    if (input.shutdown) {
        break;
    }

    let output;
    try {
        const result = await pyodide.runPythonAsync(input.code || "");
        output = JSON.stringify({ output: result });
    } catch (error) {
        output = JSON.stringify({ error: error.message.trim().split('\n').pop() || ''});
    }
    console.log(output);
}
