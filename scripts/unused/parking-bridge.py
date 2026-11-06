"""Local-only HTTP bridge between the camera detector and ParkBru's browser app."""

from __future__ import annotations

import json
import time
import ipaddress
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOST = "127.0.0.1"
PORT = 8765
STATE: dict[str, Any] = {"bays": [], "source": None, "updated_at": None}


class Handler(BaseHTTPRequestHandler):
    server_version = "ParkBruLocalBridge/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print("ParkBru bridge: " + format % args)

    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        hostname = urlsplit(origin).hostname
        local_origin = hostname in {"localhost", "127.0.0.1", "::1"}
        if hostname and not local_origin:
            try:
                local_origin = ipaddress.ip_address(hostname).is_private
            except ValueError:
                local_origin = False
        if origin.startswith("http://") and local_origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/api/state":
            self._send(404, {"error": "Not found"})
            return
        updated = STATE["updated_at"]
        camera_online = updated is not None and time.time() - updated < 5
        self._send(200, {"camera_online": camera_online, **STATE})

    def do_POST(self) -> None:
        if self.path != "/api/state":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64_000:
                raise ValueError("Invalid payload size")
            payload = json.loads(self.rfile.read(length))
            bays = payload.get("bays")
            if not isinstance(bays, list) or len(bays) > 100:
                raise ValueError("Expected a list of up to 100 bays")
            clean_bays = []
            for bay in bays:
                if not isinstance(bay, dict) or not isinstance(bay.get("occupied"), bool):
                    raise ValueError("Each bay needs an occupied boolean")
                number = bay.get("bay")
                if not isinstance(number, int) or number < 1:
                    raise ValueError("Each bay needs a positive bay number")
                plate = bay.get("plate")
                clean_bays.append({
                    "bay": number,
                    "occupied": bay["occupied"],
                    "plate": plate[:24] if isinstance(plate, str) else None,
                    "ocr_engine": bay.get("ocr_engine") if isinstance(bay.get("ocr_engine"), str) else None,
                })
            source = payload.get("source")
            STATE["bays"] = clean_bays
            STATE["source"] = source[:100] if isinstance(source, str) else None
            STATE["updated_at"] = time.time()
            self._send(200, {"ok": True})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send(400, {"error": str(error)})


def create_server(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    """Create the local detector-to-app API server."""
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    server = create_server()
    print(f"ParkBru bridge listening at http://{HOST}:{PORT}")
    print("Keep this window open while using the camera detector and app.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nParkBru bridge stopped.")
    finally:
        server.server_close()
