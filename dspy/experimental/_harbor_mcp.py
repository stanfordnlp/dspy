"""Sandbox-side MCP client, uploaded by the Harbor bridge (no DSPy dependency).

Requires Python 3.10+ and mcp>=1.26,<2 in the task image. A private Unix socket
keeps sessions alive between public BaseEnvironment.exec calls without exposing
a network port or executing task commands on the DSPy host.
"""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path


def request(directory, payload):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(str(directory / "socket"))
        connection.sendall(json.dumps(payload).encode())
        connection.shutdown(socket.SHUT_WR)
        with connection.makefile("rb") as response:
            return json.load(response)


async def serve(directory):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    pending = set()
    sessions = {}

    async def handle(reader, writer):
        pending.add(asyncio.current_task())
        try:
            try:
                message = json.loads(await reader.read())
                if message["method"] == "list":
                    result = []
                    for name, session in sessions.items():
                        cursor = None
                        while True:
                            page = await session.list_tools(cursor=cursor)
                            result.extend(
                                {"server": name, "tool": tool.model_dump(by_alias=True)} for tool in page.tools
                            )
                            cursor = page.nextCursor
                            if not cursor:
                                break
                elif message["method"] == "call":
                    result = await sessions[message["server"]].call_tool(message["name"], message["arguments"])
                    result = result.model_dump(by_alias=True, exclude_unset=True)
                else:
                    raise ValueError("Unknown MCP bridge method")
                response = {"result": result}
            except Exception as error:
                response = {"error": f"{type(error).__name__}: {error}"}
            writer.write(json.dumps(response).encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            pending.discard(asyncio.current_task())

    async with AsyncExitStack() as stack:
        for config in json.loads((directory / "servers.json").read_text()):
            if config["transport"] == "stdio":
                transport = stdio_client(
                    StdioServerParameters(command=config["command"], args=config["args"], env=dict(os.environ))
                )
            elif config["transport"] == "sse":
                transport = sse_client(config["url"])
            elif config["transport"] == "streamable-http":
                transport = streamablehttp_client(config["url"])
            else:
                raise ValueError(f"Unsupported MCP transport: {config['transport']}")
            streams = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
            await session.initialize()
            sessions[config["name"]] = session
        # After startup, SIGTERM requests an orderly shutdown in the SDK contexts'
        # owning task rather than interrupting their cleanup with cancellation.
        stop = asyncio.Event()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
        server = await asyncio.start_unix_server(handle, path=directory / "socket")
        try:
            (directory / "ready").touch()
            await stop.wait()
        finally:
            server.close()
            # wait_closed can wait for active connections; cancel handlers first.
            for task in list(pending):
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await server.wait_closed()


def main():
    action, path = sys.argv[1:3]
    directory = Path(path)
    if action == "serve":
        try:
            asyncio.run(serve(directory))
        except asyncio.CancelledError:
            pass
        finally:
            (directory / "closed").touch()
    elif action == "start":
        with (directory / "log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "serve", path],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        (directory / "pid").write_text(str(process.pid))
        # The host's Harbor agent timeout owns the startup deadline.
        while True:
            if (directory / "ready").exists():
                return
            if process.poll() is not None:
                raise RuntimeError(f"MCP client failed to start:\n{(directory / 'log').read_text()[-8000:]}")
            time.sleep(0.1)
    elif action == "stop":
        if not (directory / "pid").exists() or (directory / "closed").exists():
            return
        os.kill(int((directory / "pid").read_text()), signal.SIGTERM)
        for _ in range(200):
            if (directory / "closed").exists():
                return
            time.sleep(0.1)
        raise TimeoutError(f"MCP client did not close; inspect {directory / 'log'}")
    elif action == "request":
        path = Path(sys.argv[3])
        payload = json.loads(path.read_text())
        path.unlink()
        print(json.dumps(request(directory, payload)))
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
