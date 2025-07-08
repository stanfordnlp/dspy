import inspect
import os
from contextlib import AsyncExitStack
from typing import Callable

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # load environment variables from .env


class PythonInterpreterClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

        import dspy

        class GetCleanFunctionDefinition(dspy.Signature):
            """Get the clean function definition code. Code unrelated to function definition, like main functions is
            removed.

            A few additional rules:
            1. If the function definition relies on imported modules, make sure the import statements are included in the
            output clean code.
            2. If the function definition relies on custom helper functions, make sure the helper function definitions are
            included in the output clean code.
            3. If the function definition relies on some global variables, make sure the global variable definitions are
            included in the output clean code.
            """

            dirty_code: str = dspy.InputField(
                description="The code containing the function definitions, which might be dirty."
            )
            function_names: list[str] = dspy.InputField(
                description=(
                    "The names of the functions that the clean code must be able to define. If it relies on "
                    "custom helper functions, imported modules or global variables, make sure the relevant code is "
                    "included in the output clean code."
                )
            )
            clean_code: str = dspy.OutputField(
                description="The code only contain the function definitions, without any other code."
            )

        self.code_cleaner = dspy.ChainOfThought(GetCleanFunctionDefinition)

    async def connect_to_server(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """

        server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),  # directory of client.py
            "./server.py",
        )
        server_path = os.path.abspath(server_path)
        server_params = StdioServerParameters(
            command="python",
            args=[server_path],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Call a tool"""
        response = await self.session.call_tool(tool_name, tool_args)
        return response.content

    def _get_source_code(self, funcs: list[Callable]):
        source_files = set()
        for func in funcs:
            original_func = inspect.unwrap(func)
            path = inspect.getsourcefile(original_func)
            if path is None:
                raise ValueError("Could not determine source file")
            source_files.add(path)

        source_code = ""
        for path in source_files:
            with open(path) as f:
                source_code += f.read()
                source_code += "\n\n"
        return source_code

    async def execute(self, code: str):
        """Execute Python code"""
        response = await self.session.call_tool("run_python_code", {"code": code})
        return response.content

    async def register_functions_by_code(self, code: str):
        """Register functions by code"""
        await self.session.call_tool("register_functions", {"code": code})

    async def register_functions_by_file(self, file_path: str):
        """Register functions by file path"""
        with open(file_path) as f:
            code = f.read()
        await self.session.call_tool("register_functions", {"code": code})

    async def register_functions(self, functions: list[dict]):
        """Register functions to the MCP server"""
        source_code = self._get_source_code(functions)

        import dspy

        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
            clean_code = self.code_cleaner(
                dirty_code=source_code,
                function_names=[func.__name__ for func in functions],
            ).clean_code
        if clean_code.startswith("```python"):
            clean_code = clean_code[len("```python") : -len("```")]
        await self.session.call_tool("register_functions", {"code": clean_code})

    async def shutdown(self):
        """Clean up resources"""
        await self.session.call_tool("cleanup", {})
        await self.exit_stack.aclose()
