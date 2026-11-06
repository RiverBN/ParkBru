"""Run the parking detector from an Android IP camera stream.

Examples:
    python scripts/phone-parking-detect-android.py
    python scripts/phone-parking-detect-android.py http://192.168.1.25:8080/video

For the Android IP Webcam app, start its video server and use its video URL
(commonly http://PHONE_IP:8080/video). The stream stays on the local network.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


DEFAULT_URL = "http://PHONE_IP:8080/video"


def ask_for_stream_url() -> str | None:
    """Show a small URL dialog when the stream URL was not passed on CLI."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError:
        print("Tkinter is unavailable; pass the Android stream URL as an argument.")
        return None

    result: dict[str, str | None] = {"url": None}
    root = tk.Tk()
    root.title("Android IP camera")
    root.resizable(False, False)
    root.geometry("480x190")

    body = ttk.Frame(root, padding=18)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Android camera stream URL").pack(anchor="w")
    url_var = tk.StringVar(value=DEFAULT_URL)
    entry = ttk.Entry(body, textvariable=url_var, width=62)
    entry.pack(fill="x", pady=(6, 8))
    ttk.Label(
        body,
        text="In IP Webcam, start the server and enter its Video URL, usually ending in /video.",
        wraplength=440,
    ).pack(anchor="w")

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))

    def connect() -> None:
        url = url_var.get().strip()
        if not url.startswith(("http://", "https://", "rtsp://", "rtsps://")):
            messagebox.showerror("Invalid stream URL", "Use an http://, https://, rtsp://, or rtsps:// URL.", parent=root)
            entry.focus_set()
            return
        result["url"] = url
        root.destroy()

    ttk.Button(buttons, text="Connect", command=connect).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(0, 8))
    entry.bind("<Return>", lambda _event: connect())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    entry.focus_set()
    root.mainloop()
    return result["url"]


def load_detector_module():
    path = Path(__file__).with_name("phone-parking-detect.py")
    spec = importlib.util.spec_from_file_location("parkbru_phone_parking_detector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load detector module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ParkBru with an Android IP camera stream.")
    parser.add_argument("stream_url", nargs="?", help="Android camera URL, such as http://192.168.1.25:8080/video")
    parser.add_argument("--server-url", help="Optional local ParkBru bridge URL")
    args = parser.parse_args()

    stream_url = args.stream_url.strip() if args.stream_url else ask_for_stream_url()
    if not stream_url:
        print("Camera setup cancelled.")
        return 0
    if not stream_url.startswith(("http://", "https://", "rtsp://", "rtsps://")):
        parser.error("The stream URL should start with http://, https://, rtsp://, or rtsps://")

    detector = load_detector_module()
    # Reuse the detector implementation while keeping this Android workflow in
    # its own script and leaving the USB and browser-camera versions untouched.
    sys.argv = [str(Path(__file__).with_name("phone-parking-detect.py")), stream_url]
    if args.server_url:
        sys.argv.extend(["--server-url", args.server_url])
    return detector.main()


if __name__ == "__main__":
    raise SystemExit(main())
