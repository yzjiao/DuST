#!/usr/bin/env python3
"""Minimal code execution sandbox HTTP server.

Accepts POST /execute requests, runs Python code in a subprocess with
resource limits (timeout + memory), and returns stdout/stderr/exit_code.

Environment variables:
    PORT  — Port to listen on (default: 8080)
"""

import json
import os
import resource
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1MB


def execute_code(code: str, stdin_input: str, timeout: int, memory_limit_mb: int) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        memory_bytes = memory_limit_mb * 1024 * 1024

        def set_limits():
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        proc = subprocess.run(
            [sys.executable, "-u", tmp_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=set_limits,
        )
        return {
            "stdout": proc.stdout[:MAX_OUTPUT_BYTES],
            "stderr": proc.stderr[:MAX_OUTPUT_BYTES],
            "exit_code": proc.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Execution timed out",
            "exit_code": -1,
            "timed_out": True,
        }
    except MemoryError:
        return {
            "stdout": "",
            "stderr": "Memory limit exceeded",
            "exit_code": -1,
            "timed_out": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "timed_out": False,
        }
    finally:
        os.unlink(tmp_path)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/execute":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        code = req.get("code", "")
        stdin_input = req.get("input", "")
        timeout = min(req.get("timeout", 10), 60)
        memory_limit_mb = min(req.get("memory_limit_mb", 512), 2048)

        result = execute_code(code, stdin_input, timeout, memory_limit_mb)

        resp = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        pass


def main():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Sandbox server listening on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
