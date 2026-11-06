"""A deterministic OpenAI-compatible HTTP server, shared once per pytest worker."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(self.server.log_path, "a") as log:
            log.write(json.dumps(request) + "\n")
        content = str(request.get("messages", [{}])[0].get("content", ""))
        for status, message in [
            (429, "Rate limit exceeded"),
            (504, "Request timed out!"),
            (400, "Bad request"),
            (401, "Authentication error"),
        ]:
            if str(status) in content:
                self._json({"error": {"message": message, "type": "test_error", "code": str(status)}}, status)
                return
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in [({"role": "assistant", "content": '{"output_text": "Hello!"}'}, None), ({}, "stop")]:
                chunk = {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": request["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            usage = {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": request["model"],
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            self._json(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            )

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="session")
def _openai_test_server(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("openai-server") / "requests.jsonl"
    log_path.touch()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.log_path = log_path
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", str(log_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def openai_test_server(_openai_test_server):
    url, log_path = _openai_test_server
    with open(log_path, "w"):
        pass
    return url, log_path


def read_test_server_request_logs(log_path):
    with open(log_path) as log:
        return [json.loads(line) for line in log]
