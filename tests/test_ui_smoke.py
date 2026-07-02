from __future__ import annotations

import socket
import sys
import time
import unittest
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class UiStartupSmokeTest(unittest.TestCase):
    """Launches the real Gradio app briefly, checks it serves HTTP, then shuts it down.

    This does not exercise any button click (no SAM3, no training); it only proves
    the app can start and stop cleanly as a local web server.
    """

    def test_app_launches_and_serves_http(self) -> None:
        import app

        port = _free_port()
        app.demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
            show_error=True,
        )
        try:
            deadline = time.time() + 15
            last_exc = None
            body = b""
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as resp:
                        body = resp.read()
                        break
                except Exception as exc:  # server may not be ready yet
                    last_exc = exc
                    time.sleep(0.5)
            else:
                self.fail(f"UI did not respond within timeout: {last_exc!r}")
            self.assertIn(b"<html", body.lower())
        finally:
            app.demo.close()


if __name__ == "__main__":
    unittest.main()
