import os
import asyncio
from typing import Any
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy


# Global variables for MCP client session
stdio_context: Any | None = None
session: ClientSession | None = None
_cleanup_lock: asyncio.Lock = asyncio.Lock()
exit_stack: AsyncExitStack = AsyncExitStack()

# Default MCP configuration
DEFAULT_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_MCP_COMMAND = "npx"
DEFAULT_MCP_ARGS = ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
DEFAULT_ENV_VARS = {}

react_agent: Any | None = None

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
   global react_agent, session, exit_stack
   # Configure DSPy with LLM
   api_key = os.getenv("GOOGLE_API_KEY")
   if not api_key:
      raise HTTPException(status_code=400, detail="GOOGLE_API_KEY environment variable not set")
   
   LLM = dspy.LM(DEFAULT_MODEL, api_key=api_key)
   dspy.configure(lm=LLM)
   
   # Initialize MCP server and tools
   session, tool_list = await initialize_stdio_client()
   
   # Create MCPTools instance
   mcp_tools = dspy.MCPTools(session=session, tools_list=tool_list)
   
   # Create ReAct agent in the same async context
   react_agent = dspy.ReAct("input->output", mcp_tools.get_tools())
   
   yield
   # Shutdown - clean up resources 
   # This will be executed in the same task context where resources were initialized
   await cleanup()

# Create FastAPI app with lifespan
app = FastAPI(
    title="DSPy MCP API",
    description="FastAPI server for DSPy Model Context Protocol interactions",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: Any
    status: str

# MCP initialization function
async def initialize_stdio_client(
        command: str = DEFAULT_MCP_COMMAND,
        command_args: list[str] = DEFAULT_MCP_ARGS,
        env: dict[str, str] | None = DEFAULT_ENV_VARS     
):
    global stdio_context, session, exit_stack, react_agent
    if stdio_context is not None:
        return session, stdio_context

    print(f"Initializing MCP server with command: {command} {' '.join(command_args)}")
    server_params = StdioServerParameters(
         command=command,
         args=command_args,
         env=env if env else None
    )
    try:
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        tools = await session.list_tools()
        stdio_context = tools.tools

        return session, tools.tools
    except Exception as e:
        print(f"Error initializing MCP server: {str(e)}")
        await cleanup()
        raise

async def cleanup() -> None:
    """Clean up server resources."""
    global stdio_context, session, exit_stack
    print("Cleaning up MCP server resources...")
    async with _cleanup_lock:
        if session is not None:
            await exit_stack.aclose()
            session = None
            stdio_context = None
    print("Cleanup complete.")

# FastAPI endpoints
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Run the agent
        react_result = await react_agent(input=request.query)
        
        return QueryResponse(
            result=react_result.output,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_dspy_mcp:app", host="0.0.0.0", port=8000, reload=True)