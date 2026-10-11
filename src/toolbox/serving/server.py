"""Serving: launch SGLang/vLLM servers, or expose any engine over a tiny OpenAI-style API."""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from toolbox.core.logging import get_logger
from toolbox.serving.base import InferenceEngine, SamplingParams

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
HTTP_OK, HTTP_BAD_REQUEST, HTTP_NOT_FOUND = 200, 400, 404

log = get_logger("serve")


@dataclass
class ServerLaunchSpec:
    """The shell command that starts an official inference server."""

    backend: str
    model: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    extra_args: Sequence[str] = field(default_factory=tuple)

    def command(self) -> list[str]:
        builders = {
            "sglang": lambda: [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                self.model,
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
            "vllm": lambda: [
                "vllm",
                "serve",
                self.model,
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
        }
        if self.backend not in builders:
            raise ValueError(f"backend must be one of {sorted(builders)}, got {self.backend!r}")
        return builders[self.backend]() + list(self.extra_args)

    def shell(self) -> str:
        return shlex.join(self.command())


def make_handler(engine: InferenceEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(HTTP_OK, {"status": "ok", "engine": engine.name})
            else:
                self._send(HTTP_NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/completions":
                self._send(HTTP_NOT_FOUND, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompts = request["prompt"]
                prompts = [prompts] if isinstance(prompts, str) else prompts
                params = SamplingParams(
                    max_new_tokens=int(request.get("max_tokens", 64)),
                    temperature=float(request.get("temperature", 1.0)),
                    top_p=float(request.get("top_p", 1.0)),
                    stop=tuple(request.get("stop") or ()),
                    n=int(request.get("n", 1)),
                )
            except (KeyError, ValueError, TypeError) as exc:
                self._send(HTTP_BAD_REQUEST, {"error": str(exc)})
                return
            completions = engine.generate(prompts, params)
            self._send(
                HTTP_OK,
                {
                    "object": "text_completion",
                    "model": engine.name,
                    "choices": [
                        {"index": i, "text": c.text, "finish_reason": c.finish_reason}
                        for i, c in enumerate(completions)
                    ],
                },
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug(fmt, *args)

    return Handler


def serve_engine(
    engine: InferenceEngine, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Create (not start) an HTTP server; call ``serve_forever()`` on the result."""
    return ThreadingHTTPServer((host, port), make_handler(engine))
