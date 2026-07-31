"""In-process OpenAI Chat Completions test server."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest


@dataclass
class OpenAICompatServerState:
    requests: list[dict[str, Any]] = field(default_factory=list)
    status: int = 200
    response: dict[str, Any] = field(
        default_factory=lambda: {
            "id": "chatcmpl-local",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
    )

    stream_chunks: list[dict[str, Any]] | None = None

    def reply(self, status: int, response: dict[str, Any]) -> None:
        self.status = status
        self.response = response

    def reply_stream(self, chunks: list[dict[str, Any]]) -> None:
        """Serve requests with `stream: true` as SSE from these chunk dicts."""
        self.status = 200
        self.stream_chunks = chunks


@pytest.fixture
def openai_compat_server():
    """Yield `(base_url, state)`: a live local endpoint and its scriptable state."""
    state = OpenAICompatServerState()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length))
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            if body.get("stream") and state.stream_chunks is not None and state.status < 400:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in state.stream_chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            encoded = json.dumps(state.response).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
