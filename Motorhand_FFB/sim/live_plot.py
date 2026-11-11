"""Local-only live dashboard; recording remains the authoritative data source."""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading


class LivePlot:
    """A bounded, best-effort dashboard that never waits for chart consumers."""

    def __init__(self, motor_names, scene_metadata):
        self.samples = deque(maxlen=1200)
        self.lock = threading.Lock()
        self.motor_names = list(motor_names)
        self.scene_metadata = dict(scene_metadata)
        self.skipped = 0
        html = Path(__file__).with_name("dashboard.html").read_bytes()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    content, content_type = html, "text/html; charset=utf-8"
                elif self.path == "/data":
                    with owner.lock:
                        samples = list(owner.samples)
                    content = json.dumps({"samples": samples, "motor_names": owner.motor_names,
                                          "scene": owner.scene_metadata,
                                          "skipped": owner.skipped}, allow_nan=False).encode("utf-8")
                    content_type = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}/"
        self.thread = threading.Thread(target=self.server.serve_forever, name="sim-live-dashboard", daemon=True)
        self.thread.start()

    def append(self, sample):
        # A slow browser is never allowed to delay the feedback path.
        if not self.lock.acquire(blocking=False):
            self.skipped += 1
            return
        try:
            self.samples.append(sample)
        finally:
            self.lock.release()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
