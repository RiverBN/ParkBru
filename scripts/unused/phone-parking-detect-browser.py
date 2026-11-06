"""Run the parking detector using camera frames sent by a local browser page.

Usage:
    python scripts/phone-parking-detect-browser.py

The page is served only on 127.0.0.1. The browser asks for camera permission,
lets you select a camera and capture mode, then posts JPEG frames to this local
process. No camera frames are sent outside the laptop.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from urllib.parse import unquote
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


CAMERA_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParkBru local camera</title>
<style>
  :root { color-scheme: light; font: 15px/1.45 system-ui, sans-serif; color: #17212b; background: #f2f5f7; }
  body { max-width: 660px; margin: 40px auto; padding: 0 20px; }
  main { background: white; border: 1px solid #dce3e8; border-radius: 14px; padding: 24px; box-shadow: 0 8px 30px #16334b12; }
  h1 { margin: 0 0 5px; font-size: 23px; } p { color: #52616d; margin: 5px 0 18px; }
  label { display: block; margin: 14px 0 5px; font-weight: 600; }
  select, button { box-sizing: border-box; min-height: 42px; border-radius: 8px; font: inherit; }
  select { width: 100%; border: 1px solid #bdc9d1; padding: 8px 10px; background: white; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .buttons { display: flex; gap: 10px; margin-top: 20px; }
  button { border: 0; padding: 9px 16px; background: #176b57; color: white; font-weight: 650; cursor: pointer; }
  button.secondary { background: #e7edf0; color: #17212b; }
  button:disabled { opacity: .5; cursor: default; }
  #status { margin-top: 18px; padding: 12px; background: #f2f5f7; border-radius: 8px; min-height: 24px; }
  small { display: block; color: #52616d; margin-top: 14px; }
</style>
</head>
<body><main>
  <h1>ParkBru camera connection</h1>
  <p>Choose a camera and capture mode. The camera image is sent to the detector on this laptop only.</p>
  <button id="find">Allow camera access and find cameras</button>
  <label for="camera">Camera</label>
  <select id="camera" disabled><option>Allow access to list cameras</option></select>
  <div class="row">
    <div><label for="resolution">Resolution</label>
      <select id="resolution"><option value="default">Camera default</option><option value="640x480">640 × 480</option><option value="1280x720">1280 × 720</option><option value="1920x1080">1920 × 1080</option></select>
    </div>
    <div><label for="fps">Requested frame rate</label>
      <select id="fps"><option value="default">Camera default</option><option>15</option><option>24</option><option selected>30</option><option>60</option></select>
    </div>
  </div>
  <div class="buttons"><button id="start" disabled>Start camera stream</button><button id="stop" class="secondary" disabled>Stop</button></div>
  <div id="status" role="status">Waiting for camera permission.</div>
  <small>Keep this page open while the detector is running. Close the page or press Stop to end the stream.</small>
  <video id="video" autoplay muted playsinline style="position:fixed;left:-2px;top:-2px;width:1px;height:1px;opacity:0;pointer-events:none"></video>
  <canvas id="canvas" hidden></canvas>
</main>
<script>
const cameraSelect = document.querySelector('#camera');
const resolutionSelect = document.querySelector('#resolution');
const fpsSelect = document.querySelector('#fps');
const video = document.querySelector('#video');
const canvas = document.querySelector('#canvas');
const statusBox = document.querySelector('#status');
const startButton = document.querySelector('#start');
const stopButton = document.querySelector('#stop');
let stream = null;
let sending = false;
let sentCount = 0;
let rateStart = 0;
let sequence = 0;

async function loadCameras() {
  try {
    const permissionStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    permissionStream.getTracks().forEach(track => track.stop());
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'videoinput');
    cameraSelect.replaceChildren();
    devices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Camera ${index + 1}`;
      cameraSelect.append(option);
    });
    cameraSelect.disabled = devices.length === 0;
    startButton.disabled = devices.length === 0;
    statusBox.textContent = devices.length ? `Found ${devices.length} camera(s). Select one and start the stream.` : 'No cameras are available to this browser.';
  } catch (error) {
    statusBox.textContent = `Camera access failed: ${error.name}: ${error.message}`;
  }
}

async function stopStream(notifyServer = true) {
  sending = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  if (notifyServer) {
    try { await fetch('/stop', {method: 'POST'}); } catch (_) { }
  }
  statusBox.textContent = 'Camera stream stopped.';
}

async function sendFrames() {
  if (!sending || !stream) return;
  if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    canvas.getContext('2d', {alpha: false}).drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.86));
    if (blob && sending) {
      try {
        const response = await fetch('/frame', {
          method: 'POST',
          headers: {
            'Content-Type': 'image/jpeg',
            'X-Frame-Sequence': String(++sequence),
            'X-Camera-Name': encodeURIComponent(cameraSelect.selectedOptions[0]?.textContent || 'Camera'),
            'X-Camera-Index': String(cameraSelect.selectedIndex),
            'X-Reported-Fps': String(stream.getVideoTracks()[0].getSettings().frameRate || 0)
          },
          body: blob
        });
        if (!response.ok) throw new Error(`server returned ${response.status}`);
        sentCount++;
        const now = performance.now();
        if (!rateStart) rateStart = now;
        if (now - rateStart >= 1000) {
          const settings = stream.getVideoTracks()[0].getSettings();
          const elapsed = (now - rateStart) / 1000;
          statusBox.textContent = `Streaming ${settings.width} × ${settings.height}; browser reports ${Number(settings.frameRate || 0).toFixed(1)} FPS; sent ${ (sentCount / elapsed).toFixed(1)} FPS.`;
          sentCount = 0;
          rateStart = now;
        }
      } catch (error) {
        statusBox.textContent = `Local stream error: ${error.message}`;
        stopStream(false);
        return;
      }
    }
  }
  const targetFps = Number(fpsSelect.value) || 30;
  window.setTimeout(sendFrames, Math.max(0, 1000 / targetFps));
}

async function startStream() {
  try {
    const [width, height] = resolutionSelect.value === 'default' ? [undefined, undefined] : resolutionSelect.value.split('x').map(Number);
    const fps = Number(fpsSelect.value);
    const videoConstraints = {deviceId: {exact: cameraSelect.value}};
    if (width) videoConstraints.width = {ideal: width};
    if (height) videoConstraints.height = {ideal: height};
    if (fps) videoConstraints.frameRate = {ideal: fps};
    stream = await navigator.mediaDevices.getUserMedia({video: videoConstraints, audio: false});
    video.srcObject = stream;
    await video.play();
    const settings = stream.getVideoTracks()[0].getSettings();
    startButton.disabled = true;
    stopButton.disabled = false;
    statusBox.textContent = `Connected: ${settings.width} × ${settings.height}, browser reports ${Number(settings.frameRate || 0).toFixed(1)} FPS. Sending frames locally…`;
    sending = true;
    rateStart = 0;
    sentCount = 0;
    sendFrames();
  } catch (error) {
    statusBox.textContent = `Could not start camera: ${error.name}: ${error.message}`;
    if (stream) await stopStream(false);
  }
}

document.querySelector('#find').addEventListener('click', loadCameras);
startButton.addEventListener('click', startStream);
stopButton.addEventListener('click', () => stopStream(true));
window.addEventListener('pagehide', () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
  navigator.sendBeacon('/stop', '');
});
</script></body></html>"""


class BrowserCapture:
    """A VideoCapture-compatible reader fed by JPEGs from localhost."""

    def __init__(self, server: ThreadingHTTPServer):
        self.server = server
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.read_sequence = 0
        self.stopped = False
        self.released = False
        self.camera_name = "Browser camera"
        self.reported_fps = 0.0
        self.frame_rate = 0.0
        self.camera_index = 0

    def submit(self, frame: np.ndarray, sequence: int, camera_name: str, reported_fps: float) -> None:
        with self.condition:
            self.frame = frame
            self.sequence = max(self.sequence + 1, sequence)
            self.camera_name = camera_name
            self.reported_fps = reported_fps
            self.condition.notify_all()

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def wait_for_first_frame(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.frame is None and not self.stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return self.frame is not None

    def isOpened(self) -> bool:
        return not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.condition:
            deadline = time.monotonic() + 3.0
            while self.sequence <= self.read_sequence and not self.stopped and not self.released:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self.condition.wait(remaining)
            if self.stopped or self.released or self.frame is None:
                return False, None
            self.read_sequence = self.sequence
            return True, self.frame.copy()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.stop()
        try:
            self.server.shutdown()
        finally:
            self.server.server_close()


def make_handler(capture: BrowserCapture):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            payload = CAMERA_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.path == "/stop":
                capture.stop()
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path != "/frame":
                self.send_error(404)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 16_000_000:
                    self.send_error(413)
                    return
                encoded = self.rfile.read(content_length)
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    self.send_error(400, "Invalid JPEG frame")
                    return
                sequence = int(self.headers.get("X-Frame-Sequence", "0"))
                camera_name = unquote(self.headers.get("X-Camera-Name", "Browser camera"))
                capture.camera_index = int(self.headers.get("X-Camera-Index", "0"))
                reported_fps = float(self.headers.get("X-Reported-Fps", "0"))
                capture.submit(frame, sequence, camera_name, reported_fps)
                capture.frame_rate = capture.frame_rate * 0.9 + 0.1 * reported_fps if capture.frame_rate else reported_fps
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except (ValueError, OSError):
                self.send_error(400, "Invalid frame request")

    return Handler


def open_browser_camera(initial_index: int) -> tuple[BrowserCapture | None, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    capture = BrowserCapture(server)
    server.RequestHandlerClass = make_handler(capture)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="parkbru-browser-camera", daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Local camera page: {url}")
    print("Allow camera access, choose the camera and resolution, then click Start camera stream.")
    webbrowser.open(url)
    if not capture.wait_for_first_frame(timeout=180):
        capture.release()
        print("No browser camera frames arrived. Check the browser permission, or start again.")
        return None, initial_index
    print(f"Browser camera connected: {capture.camera_name}; browser-reported {capture.reported_fps:g} FPS")
    return capture, capture.camera_index


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
    detector = load_detector_module()
    detector.choose_camera_source_gui = lambda: ("usb", None)
    detector.configure_camera_gui = open_browser_camera
    return detector.main()


if __name__ == "__main__":
    raise SystemExit(main())
