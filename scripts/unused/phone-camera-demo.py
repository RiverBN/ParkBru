"""Preview an Android IP camera stream locally with OpenCV.

Usage:
    python scripts/phone-camera-demo.py "http://PHONE_IP:PORT/video/mjpeg"

Start the camera server in the Android app first, then pass the MJPEG stream URL
shown by that app. Press q or Esc in the preview window to exit.
"""

from __future__ import annotations

import sys
import time

import cv2


def main() -> int:
    if len(sys.argv) != 2:
        print(
            'Usage: python scripts/phone-camera-demo.py "MJPEG_STREAM_URL"\n'
            'Example: python scripts/phone-camera-demo.py '
            '"http://192.168.1.25:4444/video/mjpeg"'
        )
        return 2

    stream_url = sys.argv[1].strip()
    if not stream_url.startswith(("http://", "https://", "rtsp://")):
        print("The stream URL should start with http://, https://, or rtsp://.")
        return 2

    print("Connecting to phone camera stream...")
    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        print(
            "Could not open the stream. Check that the phone's camera server is "
            "running, both devices are on the same Wi-Fi, and you copied the "
            "video stream URL (not just the app's control-page URL)."
        )
        capture.release()
        return 1

    window_name = "ParkBru phone camera - press q or Esc to close"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_frame_time = time.monotonic()
    fps = 0.0

    print("Live video received. Press q or Esc in the preview window to close.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("The video stream stopped or returned an empty frame.")
                break

            now = time.monotonic()
            elapsed = now - last_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps
            last_frame_time = now

            height, width = frame.shape[:2]
            cv2.putText(
                frame,
                f"Phone camera connected  |  {width}x{height}  |  {fps:.1f} FPS",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 220, 150),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
