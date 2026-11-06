"""Detect occupied miniature parking bays from a fixed Android IP camera.

This prototype uses background difference to detect a car-sized change inside
a bay, then uses OCR to attempt a miniature plate read. It works best with a
fixed camera, steady lighting, white paper, pencil bay lines, and a car that
contrasts with the paper.

Usage:
    python scripts/phone-parking-detect.py "http://PHONE_IP:PORT/video/mjpeg"

Draw a box inside each bay when prompted. Automatic detection looks for
three-sided, pencil-drawn U shapes, including skewed perspective; it does not
use the car outline as a bay. Drag any yellow corner to fine-tune a bay. Keep
the bays empty and press C to capture the empty reference. Press N to add a bay,
click a bay to select it and press D to delete it, A to scan again, R to redraw
all bay boxes, L to try license-plate OCR on a selected car, or Q/Esc to quit.
"""

from __future__ import annotations

import sys
import time
import re
import os
import shutil
import argparse
import json
import urllib.request
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


PIXEL_DIFF_THRESHOLD = 24
OCCUPANCY_THRESHOLD = 0.035
OCCUPANCY_COMPONENT_THRESHOLD = 0.02
EMPTY_CHANGE_THRESHOLD = 0.018
FRAMES_TO_CONFIRM = 4
BayPolygon = np.ndarray
_PADDLE_READER = None


@dataclass(frozen=True)
class PlateReading:
    text: str | None
    box: tuple[int, int, int, int] | None
    engine: str


def put_text_stroked(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Draw readable outlined text over a live camera image."""
    cv2.putText(image, text, origin, font, scale, (0, 0, 0), thickness + 3, line_type)
    cv2.putText(image, text, origin, font, scale, color, thickness, line_type)


def select_bays(frame: np.ndarray) -> list[BayPolygon]:
    """Let the user draw rectangular bay areas, avoiding pencil lines."""
    preview = frame.copy()
    cv2.putText(
        preview,
        "Drag a box inside each bay; press ENTER when done",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 30, 40),
        2,
        cv2.LINE_AA,
    )
    rectangles = cv2.selectROIs(
        "Select bay interiors - Enter to finish, Esc to cancel",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Select bay interiors - Enter to finish, Esc to cancel")
    bays: list[BayPolygon] = []
    for x, y, w, h in rectangles:
        if w > 3 and h > 3:
            bays.append(
                np.array(
                    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    dtype=np.float32,
                )
            )
    return bays


def select_one_bay(frame: np.ndarray) -> BayPolygon | None:
    """Let the user add one bay by dragging a box; corners can then be skewed."""
    preview = frame.copy()
    cv2.putText(
        preview,
        "Drag a box for the new bay; press ENTER to add it, Esc to cancel",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 30, 40),
        2,
        cv2.LINE_AA,
    )
    x, y, w, h = cv2.selectROI(
        "Add one bay - Enter to add, Esc to cancel",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Add one bay - Enter to add, Esc to cancel")
    if w <= 3 or h <= 3:
        return None
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def detect_bays(frame: np.ndarray) -> list[BayPolygon]:
    """Detect skewed, three-sided pencil U shapes and infer their open side."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_dimension = min(frame.shape[:2])
    block_size = max(31, min(101, int(min_dimension * 0.08) | 1))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    pencil_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        5,
    )
    frame_area = frame.shape[0] * frame.shape[1]
    candidates: list[tuple[BayPolygon, float]] = []

    def add_polygon(points: np.ndarray) -> None:
        poly = np.asarray(points, dtype=np.float32).reshape(4, 2)
        if not cv2.isContourConvex(np.round(poly).astype(np.int32)):
            return
        area = abs(float(cv2.contourArea(poly)))
        if not frame_area * 0.002 <= area <= frame_area * 0.12:
            return
        x, y, w, h = cv2.boundingRect(poly)
        frame_h, frame_w = frame.shape[:2]
        if w < 35 or h < 35 or w > frame_w * 0.55 or h > frame_h * 0.75:
            return
        if not 0.3 <= w / h <= 3.0:
            return
        center = np.mean(poly, axis=0)
        # Pull the region slightly inside its pencil boundary.
        inset = center + (poly - center) * 0.94
        candidates.append((inset.astype(np.float32), area))

    # For an open U, infer the missing entrance edge from the two side strokes
    # and preserve the four perspective corners as a skewable polygon.
    minimum_line = max(35, int(min_dimension * 0.08))
    raw_lines = cv2.HoughLinesP(
        pencil_mask,
        1,
        np.pi / 180,
        threshold=max(18, int(min_dimension * 0.025)),
        minLineLength=minimum_line,
        maxLineGap=max(12, int(min_dimension * 0.025)),
    )
    lines: list[tuple[int, int, int, int, float, float]] = []
    if raw_lines is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        pencil_pixels = (saturation < 90) & (value >= 45) & (value <= 215)
        paper_pixels = (saturation < 90) & (value >= 175)
        support_size = max(13, int(min_dimension * 0.025) | 1)
        nearby_paper = cv2.dilate(
            paper_pixels.astype(np.uint8),
            np.ones((support_size, support_size), np.uint8),
            iterations=1,
        ) > 0

        for values in np.asarray(raw_lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, values)
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            sample_count = max(12, int(length / 8))
            sample_x = np.clip(np.linspace(x1, x2, sample_count).astype(np.int32), 0, frame.shape[1] - 1)
            sample_y = np.clip(np.linspace(y1, y2, sample_count).astype(np.int32), 0, frame.shape[0] - 1)
            # Keep graphite-like strokes surrounded by white paper. This rejects
            # most car edges, printed graphics, and desk clutter as bay lines.
            pencil_ratio = float(np.mean(pencil_pixels[sample_y, sample_x]))
            paper_support = float(np.mean(nearby_paper[sample_y, sample_x]))
            if pencil_ratio < 0.28 or paper_support < 0.4:
                continue
            angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            lines.append((x1, y1, x2, y2, length, angle))
    lines.sort(key=lambda item: item[4], reverse=True)
    lines = lines[:100]

    def angle_gap(a: float, b: float) -> float:
        gap = abs(a - b) % 180.0
        return min(gap, 180.0 - gap)

    def line_intersection(first: tuple[int, int, int, int, float, float], second: tuple[int, int, int, int, float, float]) -> tuple[float, float] | None:
        x1, y1, x2, y2 = first[:4]
        x3, y3, x4, y4 = second[:4]
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-6:
            return None
        d1, d2 = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
        return (
            (d1 * (x3 - x4) - (x1 - x2) * d2) / denominator,
            (d1 * (y3 - y4) - (y1 - y2) * d2) / denominator,
        )

    def lies_on_segment(point: tuple[float, float], line: tuple[int, int, int, int, float, float], margin: float) -> bool:
        x1, y1, x2, y2 = line[:4]
        dx, dy = x2 - x1, y2 - y1
        length = max(1.0, float(np.hypot(dx, dy)))
        along = ((point[0] - x1) * dx + (point[1] - y1) * dy) / length
        across = abs((point[0] - x1) * dy - (point[1] - y1) * dx) / length
        return -margin <= along <= length + margin and across <= margin

    for index, side_a in enumerate(lines):
        for side_b in lines[index + 1 :]:
            if angle_gap(side_a[5], side_b[5]) > 16:
                continue
            ux, uy = np.cos(np.radians(side_a[5])), np.sin(np.radians(side_a[5]))
            ca = ((side_a[0] + side_a[2]) / 2, (side_a[1] + side_a[3]) / 2)
            cb = ((side_b[0] + side_b[2]) / 2, (side_b[1] + side_b[3]) / 2)
            separation = abs((cb[0] - ca[0]) * -uy + (cb[1] - ca[1]) * ux)
            if not 18 <= separation <= min(frame.shape[:2]) * 0.5:
                continue

            for crossbar in lines:
                if crossbar is side_a or crossbar is side_b:
                    continue
                if abs(angle_gap(side_a[5], crossbar[5]) - 90.0) > 30:
                    continue
                far_a = line_intersection(side_a, crossbar)
                far_b = line_intersection(side_b, crossbar)
                if far_a is None or far_b is None:
                    continue
                margin = max(12.0, minimum_line * 0.45)
                if not (lies_on_segment(far_a, crossbar, margin) and lies_on_segment(far_b, crossbar, margin)):
                    continue
                if not (lies_on_segment(far_a, side_a, margin) and lies_on_segment(far_b, side_b, margin)):
                    continue

                end_a = max(((side_a[0], side_a[1]), (side_a[2], side_a[3])), key=lambda p: np.hypot(p[0] - far_a[0], p[1] - far_a[1]))
                end_b = max(((side_b[0], side_b[1]), (side_b[2], side_b[3])), key=lambda p: np.hypot(p[0] - far_b[0], p[1] - far_b[1]))
                depth_a = float(np.hypot(end_a[0] - far_a[0], end_a[1] - far_a[1]))
                depth_b = float(np.hypot(end_b[0] - far_b[0], end_b[1] - far_b[1]))
                bay_width = float(np.hypot(far_a[0] - far_b[0], far_a[1] - far_b[1]))
                minimum_depth = max(40.0, min_dimension * 0.14)
                if min(depth_a, depth_b, bay_width) < minimum_depth:
                    continue
                if max(depth_a, depth_b) / min(depth_a, depth_b) > 1.7:
                    continue
                add_polygon(np.array([far_a, far_b, end_b, end_a], dtype=np.float32))

    # Remove duplicate detections from inner/outer pencil edges while comparing
    # their actual polygons rather than upright bounding boxes.
    candidates.sort(key=lambda item: item[1], reverse=True)
    unique: list[BayPolygon] = []
    for polygon, _area in candidates:
        duplicate = False
        for kept in unique:
            intersection_area, _ = cv2.intersectConvexConvex(
                polygon.reshape(-1, 1, 2), kept.reshape(-1, 1, 2)
            )
            union_area = abs(cv2.contourArea(polygon)) + abs(cv2.contourArea(kept)) - intersection_area
            if union_area > 0 and intersection_area / union_area >= 0.42:
                duplicate = True
                break
        if not duplicate:
            unique.append(polygon)

    # Give stable left-to-right IDs within approximate rows.
    unique.sort(key=lambda p: (float(np.mean(p[:, 1])), float(np.mean(p[:, 0]))))
    if not unique:
        return []
    median_height = float(np.median([cv2.boundingRect(p)[3] for p in unique]))
    tolerance = max(12.0, median_height * 0.45)
    rows: list[list[BayPolygon]] = []
    for polygon in unique:
        center_y = float(np.mean(polygon[:, 1]))
        if not rows or abs(center_y - float(np.mean([np.mean(p[:, 1]) for p in rows[-1]]))) > tolerance:
            rows.append([polygon])
        else:
            rows[-1].append(polygon)
    return [p for row in rows for p in sorted(row, key=lambda p: float(np.mean(p[:, 0])))]


def crop_bay(frame: np.ndarray, polygon: BayPolygon) -> tuple[np.ndarray, np.ndarray]:
    """Return the polygon's bounding crop and a mask for only the bay interior."""
    height, width = frame.shape[:2]
    points = np.round(polygon).astype(np.int32)
    x, y, w, h = cv2.boundingRect(points)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    crop = frame[y0:y1, x0:x1].copy()
    local = points - np.array([x0, y0], dtype=np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, local, 255)
    return crop, mask


def detect_change(
    reference: np.ndarray, current: np.ndarray, bay_mask: np.ndarray
) -> tuple[float, float, tuple[int, int, int, int] | None]:
    """Find changed foreground pixels and their largest component in a bay."""
    ref = reference.astype(np.int16)
    cur = current.astype(np.int16)

    # Correct small whole-frame exposure/white-balance shifts using the median
    # per-channel change. A car occupies only part of a bay, so it should not
    # dominate this estimate.
    bay_pixels = bay_mask > 0
    shift = np.median((cur - ref)[bay_pixels], axis=0).astype(np.int16).reshape(1, 1, 3)
    corrected = np.clip(cur - shift, 0, 255).astype(np.uint8)
    diff = cv2.absdiff(reference, corrected)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Remove isolated JPEG/compression specks before measuring the changed area.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.bitwise_and(mask, mask, mask=bay_mask)
    valid_pixels = cv2.countNonZero(bay_mask)
    ratio = float(cv2.countNonZero(mask)) / float(max(1, valid_pixels))

    # A toy car should form a larger, connected foreground region than camera
    # compression noise. Return its location so the preview can mark it.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ratio, 0.0, None
    largest = max(contours, key=cv2.contourArea)
    component_ratio = float(cv2.contourArea(largest)) / float(max(1, valid_pixels))
    if cv2.contourArea(largest) < max(20.0, valid_pixels * 0.01):
        return ratio, component_ratio, None
    return ratio, component_ratio, cv2.boundingRect(largest)


def read_plate_text(car_image: np.ndarray) -> PlateReading:
    """Try OCR on the car crop and plate-shaped regions within it."""
    global _PADDLE_READER
    try:
        from paddleocr import PaddleOCR

        if _PADDLE_READER is None:
            print("Loading PaddleOCR PP-OCRv5 CPU models (first use downloads model files)...")
            _PADDLE_READER = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=False,
            )
        paddle_results = _PADDLE_READER.predict(car_image)
        paddle_candidates: list[tuple[float, str, tuple[int, int, int, int] | None]] = []
        for result in paddle_results:
            payload = result.json
            if callable(payload):
                payload = payload()
            if isinstance(payload, dict):
                details = payload.get("res", payload)
                texts = details.get("rec_texts", [])
                scores = details.get("rec_scores", [])
                boxes = details.get("rec_boxes", [])
                for text_index, (text, score) in enumerate(zip(texts, scores)):
                    cleaned = "".join(re.findall(r"[A-Z0-9]", str(text).upper()))
                    if 4 <= len(cleaned) <= 10:
                        box = None
                        if text_index < len(boxes):
                            x0, y0, x1, y1 = [int(value) for value in boxes[text_index]]
                            box = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
                        paddle_candidates.append((float(score), cleaned, box))
        if paddle_candidates:
            score, text, box = max(paddle_candidates, key=lambda item: (item[0], len(item[1])))
            return PlateReading(text, box, "PaddleOCR PP-OCRv5")
    except ImportError:
        pass
    except Exception as error:
        print(f"PaddleOCR unavailable; falling back to Tesseract ({error}).")

    try:
        import pytesseract
    except ImportError:
        raise RuntimeError(
            "Plate OCR needs the pytesseract Python package and the Tesseract OCR "
            "program. Install both, then restart this script."
        ) from None
    if not shutil.which("tesseract"):
        likely_paths = (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        )
        found = next((path for path in likely_paths if path.is_file()), None)
        if found is not None:
            pytesseract.pytesseract.tesseract_cmd = str(found)
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "The Tesseract OCR program was not found. Install it and make sure "
            "its folder is on PATH, then restart this script."
        ) from None

    gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        return PlateReading(None, None, "Tesseract")

    # Upscaling helps small toy plates; OCR the entire car and likely plate
    # rectangles because the miniature may not have a clean plate border.
    candidates: list[tuple[np.ndarray, int, int]] = [(gray, 0, 0)]
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(1, h)
        if w >= max(8, width * 0.08) and h >= 4 and 1.8 <= aspect <= 6.5:
            pad_x, pad_y = max(1, w // 12), max(1, h // 5)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            candidates.append((gray[y0:min(height, y + h + pad_y),
                                    x0:min(width, x + w + pad_x)], x0, y0))

    reads: list[tuple[float, str, tuple[int, int, int, int]]] = []
    for candidate, origin_x, origin_y in candidates:
        if candidate.size == 0:
            continue
        enlarged = cv2.resize(candidate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(enlarged)
        sharpened = cv2.addWeighted(processed, 1.7, cv2.GaussianBlur(processed, (0, 0), 2), -0.7, 0)
        for image in (
            processed,
            sharpened,
            cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        ):
            for page_mode in (6, 7, 8, 13):
                data = pytesseract.image_to_data(
                    image,
                    config=f"--psm {page_mode} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    output_type=pytesseract.Output.DICT,
                )
                text = "".join(re.findall(r"[A-Z0-9]", "".join(data["text"]).upper()))
                confidences = [float(value) for value in data["conf"] if str(value).replace(".", "", 1).isdigit()]
                # Reject very short fragments such as a single toy decal.
                if 4 <= len(text) <= 10 and confidences:
                    reads.append((
                        sum(confidences) / len(confidences),
                        text,
                        (origin_x, origin_y, candidate.shape[1], candidate.shape[0]),
                    ))
    if not reads:
        return PlateReading(None, None, "Tesseract")
    _score, text, box = max(reads, key=lambda item: (len(item[1]), item[0]))
    return PlateReading(text, box, "Tesseract")


def select_plate_crop(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Let the user tightly select a hard-to-localize miniature plate."""
    preview = frame.copy()
    cv2.putText(
        preview,
        "Draw a tight box around the rear plate; press ENTER (Esc to cancel)",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 30, 40),
        2,
        cv2.LINE_AA,
    )
    title = "Select rear plate - Enter to read, Esc to cancel"
    x, y, w, h = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w <= 3 or h <= 3:
        return None
    return frame[y:y + h, x:x + w].copy(), (x, y)


def send_bridge_state(server_url: str, bays: list[dict[str, object]]) -> None:
    body = json.dumps({"bays": bays}).encode("utf-8")
    request = urllib.request.Request(
        server_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect parking bays from an Android IP camera stream.")
    parser.add_argument("stream_url", help="Camera MJPEG/RTSP stream URL")
    parser.add_argument(
        "--server-url", default=None,
        help="Optional ParkBru local bridge URL (default: http://127.0.0.1:8765/api/state)",
    )
    args = parser.parse_args()
    stream_url = args.stream_url.strip()
    bridge_url = args.server_url
    if not stream_url.startswith(("http://", "https://", "rtsp://")):
        print("The stream URL should start with http://, https://, or rtsp://.")
        return 2

    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        print(
            "Could not open the stream. Start the camera server in the phone app, "
            "put both devices on the same Wi-Fi, and use its video stream URL."
        )
        capture.release()
        return 1

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        print("Connected, but no video frame arrived. Check the stream URL.")
        capture.release()
        return 1

    height, width = first_frame.shape[:2]
    print(f"Camera stream: {width}x{height}")
    if bridge_url:
        print(f"Sending bay and plate readings to the local ParkBru app at {bridge_url}")
    bays = detect_bays(first_frame)
    if bays:
        print(f"Automatically found {len(bays)} bay outline(s). Check the polygons in the preview.")
    else:
        print("Could not confidently find bay outlines; select the bay interiors manually.")
        bays = select_bays(first_frame)
    if not bays:
        print("No bays selected. Run the script again and draw at least one bay box.")
        capture.release()
        cv2.destroyAllWindows()
        return 1

    # Keep every corner inside the actual camera frame.
    for polygon in bays:
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    window_name = "ParkBru bays - N add | click + D delete | drag corners | A auto | Q quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    reference: list[tuple[np.ndarray, np.ndarray]] | None = None
    occupied = [False] * len(bays)
    pending = [0] * len(bays)
    selected_bay: int | None = None
    plate_results: list[str | None] = [None] * len(bays)
    plate_engines: list[str | None] = [None] * len(bays)
    plate_boxes: list[tuple[int, int, int, int] | None] = [None] * len(bays)
    plate_crop_origins: list[tuple[int, int] | None] = [None] * len(bays)
    occupied_since: list[float | None] = [None] * len(bays)
    plate_attempted = [False] * len(bays)
    plate_futures: list[Future[PlateReading] | None] = [None] * len(bays)
    ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parkbru-ocr")
    bridge_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parkbru-bridge")
    bridge_future: Future[None] | None = None
    bridge_last_send = 0.0
    bridge_last_error = 0.0
    drag_state: dict[str, object] = {"handle": None, "changed": False}

    def mouse_callback(event: int, mouse_x: int, mouse_y: int, _flags: int, _param: object) -> None:
        nonlocal reference, occupied, pending, selected_bay
        nonlocal plate_results, plate_engines, plate_boxes, plate_crop_origins

        if event == cv2.EVENT_LBUTTONDOWN:
            closest: tuple[float, int, int] | None = None
            for bay_index, polygon in enumerate(bays):
                for corner_index, (cx, cy) in enumerate(polygon):
                    distance = ((mouse_x - cx) ** 2 + (mouse_y - cy) ** 2) ** 0.5
                    if distance <= 22 and (closest is None or distance < closest[0]):
                        closest = (distance, bay_index, corner_index)

            if closest is not None:
                _, bay_index, corner_index = closest
                selected_bay = bay_index
                drag_state["handle"] = (bay_index, corner_index)
                drag_state["changed"] = False
            else:
                hits = [
                    (abs(cv2.contourArea(polygon)), bay_index)
                    for bay_index, polygon in enumerate(bays)
                    if cv2.pointPolygonTest(
                        polygon.astype(np.float32).reshape(-1, 1, 2),
                        (float(mouse_x), float(mouse_y)),
                        False,
                    ) >= 0
                ]
                selected_bay = min(hits)[1] if hits else None

        elif event == cv2.EVENT_MOUSEMOVE and drag_state["handle"] is not None:
            bay_index, corner_index = drag_state["handle"]  # type: ignore[misc]
            new_x = min(max(0, mouse_x), width - 1)
            new_y = min(max(0, mouse_y), height - 1)
            adjusted = bays[bay_index].copy()
            adjusted[corner_index] = (new_x, new_y)
            if cv2.isContourConvex(np.round(adjusted).astype(np.int32)) and abs(cv2.contourArea(adjusted)) >= 500:
                bays[bay_index] = adjusted
                drag_state["changed"] = True

        elif event == cv2.EVENT_LBUTTONUP and drag_state["handle"] is not None:
            changed = bool(drag_state["changed"])
            drag_state["handle"] = None
            drag_state["changed"] = False
            if changed:
                # The stored background crops no longer match the adjusted boxes.
                reference = None
                occupied = [False] * len(bays)
                pending = [0] * len(bays)
                plate_results = [None] * len(bays)
                plate_engines = [None] * len(bays)
                plate_boxes = [None] * len(bays)
                plate_crop_origins = [None] * len(bays)
                print("Bay box adjusted. Empty all bays and press C to calibrate again.")

    cv2.setMouseCallback(window_name, mouse_callback)
    last_frame_time = time.monotonic()
    fps = 0.0

    print("Move every car out of the selected bay areas, then press C to calibrate.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("The camera stream stopped or returned an empty frame.")
                break

            now = time.monotonic()
            elapsed = now - last_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps
            last_frame_time = now

            display = frame.copy()
            put_text_stroked(
                display,
                "Plate OCR: PaddleOCR PP-OCRv5 (preferred); Tesseract fallback",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (20, 30, 40),
                2,
                cv2.LINE_AA,
            )
            put_text_stroked(
                display,
                "Click a bay to select | drag yellow corners | N add bay | D delete selected",
                (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (20, 30, 40),
                2,
                cv2.LINE_AA,
            )
            if reference is None:
                message = "EMPTY ALL BAYS, THEN PRESS C TO CALIBRATE"
                color = (0, 170, 255)
                for index, polygon in enumerate(bays, start=1):
                    points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
                    bay_color = (255, 220, 20) if selected_bay == index - 1 else color
                    cv2.polylines(display, [points], True, bay_color, 3 if selected_bay == index - 1 else 2, cv2.LINE_AA)
                    for corner_x, corner_y in polygon.astype(np.int32):
                        cv2.rectangle(
                            display,
                            (corner_x - 6, corner_y - 6),
                            (corner_x + 6, corner_y + 6),
                            (0, 230, 255),
                            -1,
                        )
                    put_text_stroked(
                        display,
                        f"Bay {index} - waiting for empty reference",
                        (int(np.min(polygon[:, 0])), max(22, int(np.min(polygon[:, 1])) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        bay_color,
                        2,
                        cv2.LINE_AA,
                    )
            else:
                message = "OCR: PaddleOCR PP-OCRv5 preferred; Tesseract fallback | auto after 5s | L retry"
                for index, (polygon, (base, bay_mask)) in enumerate(zip(bays, reference)):
                    crop, _ = crop_bay(frame, polygon)
                    ratio, component_ratio, changed_box = detect_change(base, crop, bay_mask)
                    # Use a lower threshold to keep an occupied bay detected;
                    # a larger connected foreground region can also qualify
                    # even when it covers only a modest part of a large bay.
                    if occupied[index]:
                        candidate = (
                            ratio >= EMPTY_CHANGE_THRESHOLD
                            or component_ratio >= OCCUPANCY_COMPONENT_THRESHOLD * 0.65
                        )
                    else:
                        candidate = (
                            ratio >= OCCUPANCY_THRESHOLD
                            or component_ratio >= OCCUPANCY_COMPONENT_THRESHOLD
                        )

                    if candidate == occupied[index]:
                        pending[index] = 0
                    else:
                        pending[index] += 1
                        if pending[index] >= FRAMES_TO_CONFIRM:
                            occupied[index] = candidate
                            pending[index] = 0

                    future = plate_futures[index]
                    if future is not None and future.done():
                        try:
                            reading = future.result()
                            plate_results[index] = reading.text or ""
                            plate_engines[index] = reading.engine
                            if reading.box is not None and plate_crop_origins[index] is not None:
                                ox, oy = plate_crop_origins[index]
                                bx, by, bw, bh = reading.box
                                plate_boxes[index] = (ox + bx, oy + by, bw, bh)
                            if plate_results[index]:
                                print(f"Bay {index + 1}: {reading.engine} read {plate_results[index]}; verify the characters.")
                            else:
                                print(f"Bay {index + 1}: {reading.engine} could not read the plate; press L to try a closer crop.")
                        except Exception as error:
                            plate_results[index] = ""
                            print(f"Bay {index + 1}: automatic plate OCR failed: {error}")
                        plate_futures[index] = None
                        plate_crop_origins[index] = None

                    if occupied[index]:
                        if occupied_since[index] is None:
                            occupied_since[index] = now
                        if not plate_attempted[index] and now - occupied_since[index] >= 5.0:
                            if changed_box is not None:
                                box_x, box_y, box_w, box_h = changed_box
                                pad = max(2, int(max(box_w, box_h) * 0.08))
                                x0, y0 = max(0, box_x - pad), max(0, box_y - pad)
                                x1 = min(crop.shape[1], box_x + box_w + pad)
                                y1 = min(crop.shape[0], box_y + box_h + pad)
                                ocr_crop = crop[y0:y1, x0:x1].copy()
                            else:
                                x0, y0 = 0, 0
                                ocr_crop = crop.copy()
                            bay_x, bay_y, _, _ = cv2.boundingRect(np.round(polygon).astype(np.int32))
                            plate_crop_origins[index] = (bay_x + x0, bay_y + y0)
                            plate_futures[index] = ocr_executor.submit(read_plate_text, ocr_crop)
                            plate_attempted[index] = True
                            print(f"Bay {index + 1}: occupied for 5 seconds; starting automatic plate OCR.")
                    else:
                        occupied_since[index] = None
                        plate_attempted[index] = False
                        plate_results[index] = None
                        plate_engines[index] = None
                        plate_boxes[index] = None
                        plate_crop_origins[index] = None

                    color = (60, 190, 100) if occupied[index] else (80, 100, 225)
                    status = "OCCUPIED" if occupied[index] else "EMPTY"
                    points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
                    bay_color = (255, 220, 20) if selected_bay == index else color
                    cv2.polylines(display, [points], True, bay_color, 4 if selected_bay == index else 3, cv2.LINE_AA)
                    if occupied[index] and changed_box is not None:
                        box_x, box_y, box_w, box_h = changed_box
                        offset_x, offset_y, _, _ = cv2.boundingRect(np.round(polygon).astype(np.int32))
                        cv2.rectangle(
                            display,
                            (offset_x + box_x, offset_y + box_y),
                            (offset_x + box_x + box_w, offset_y + box_y + box_h),
                            (255, 80, 20),
                            2,
                        )
                        put_text_stroked(
                            display,
                            "CAR / OBJECT",
                            (offset_x + box_x, max(22, offset_y + box_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 80, 20),
                            2,
                            cv2.LINE_AA,
                        )
                    for corner_x, corner_y in polygon.astype(np.int32):
                        cv2.rectangle(
                            display,
                            (corner_x - 6, corner_y - 6),
                            (corner_x + 6, corner_y + 6),
                            (0, 230, 255),
                            -1,
                        )
                    put_text_stroked(
                        display,
                        f"Bay {index + 1}: {status}  ({ratio:.0%} changed)",
                        (int(np.min(polygon[:, 0])), max(22, int(np.min(polygon[:, 1])) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        bay_color,
                        2,
                        cv2.LINE_AA,
                    )
                    if plate_results[index] is not None:
                        put_text_stroked(
                            display,
                            f"{plate_engines[index] or 'OCR'}: {plate_results[index] or 'NO READ'}",
                            (int(np.min(polygon[:, 0])), min(height - 38, int(np.max(polygon[:, 1])) + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            bay_color,
                            2,
                            cv2.LINE_AA,
                        )
                        if plate_boxes[index] is not None:
                            px, py, pw, ph = plate_boxes[index]
                            cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 255, 0), 3)
                            put_text_stroked(
                                display,
                                "PLATE",
                                (px, max(20, py - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.62,
                                (0, 180, 0),
                                2,
                                cv2.LINE_AA,
                            )
                    elif occupied[index] and occupied_since[index] is not None:
                        remaining = max(0, int(5 - (now - occupied_since[index]) + 0.99))
                        put_text_stroked(
                            display,
                            f"Plate OCR starts in {remaining}s" if not plate_attempted[index] else "Plate OCR running...",
                            (int(np.min(polygon[:, 0])), min(height - 38, int(np.max(polygon[:, 1])) + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            bay_color,
                            2,
                            cv2.LINE_AA,
                        )

            put_text_stroked(
                display,
                f"{width}x{height}  |  {fps:.1f} FPS  |  {message}",
                (14, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (35, 45, 50),
                2,
                cv2.LINE_AA,
            )
            if bridge_url and reference is not None and now - bridge_last_send >= 1.0:
                if bridge_future is None or bridge_future.done():
                    if bridge_future is not None:
                        try:
                            bridge_future.result()
                        except Exception as error:
                            if now - bridge_last_error >= 10.0:
                                print(f"ParkBru app bridge unavailable: {error}")
                                bridge_last_error = now
                    snapshot = [
                        {
                            "bay": index + 1,
                            "occupied": bool(is_occupied),
                            "plate": plate_results[index] or None,
                            "ocr_engine": plate_engines[index],
                        }
                        for index, is_occupied in enumerate(occupied)
                    ]
                    bridge_future = bridge_executor.submit(send_bridge_state, bridge_url, snapshot)
                    bridge_last_send = now
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                if not bays:
                    print("Add or detect at least one bay before calibrating.")
                    continue
                # Grab a fresh frame so calibration uses the current empty scene.
                ok, calibration_frame = capture.read()
                if ok and calibration_frame is not None:
                    reference = [crop_bay(calibration_frame, polygon) for polygon in bays]
                    occupied = [False] * len(bays)
                    pending = [0] * len(bays)
                    plate_results = [None] * len(bays)
                    plate_engines = [None] * len(bays)
                    plate_boxes = [None] * len(bays)
                    plate_crop_origins = [None] * len(bays)
                    occupied_since = [None] * len(bays)
                    plate_attempted = [False] * len(bays)
                    plate_futures = [None] * len(bays)
                    print("Empty-bay reference captured. Place a car in a bay to test.")
            if key == ord("l"):
                if selected_bay is None:
                    print("Click a bay containing a detected car, then press L to try plate OCR.")
                elif reference is None:
                    print("Calibrate empty bays with C before reading a plate.")
                else:
                    bay_index = selected_bay
                    current_crop, _ = crop_bay(frame, bays[bay_index])
                    _, _, car_box = detect_change(reference[bay_index][0], current_crop, reference[bay_index][1])
                    reading: PlateReading | None = None
                    origin = (0, 0)
                    try:
                        if car_box is not None:
                            bx, by, bw, bh = car_box
                            pad = max(2, int(max(bw, bh) * 0.08))
                            x0, y0 = max(0, bx - pad), max(0, by - pad)
                            x1, y1 = min(current_crop.shape[1], bx + bw + pad), min(current_crop.shape[0], by + bh + pad)
                            bay_x, bay_y, _, _ = cv2.boundingRect(np.round(bays[bay_index]).astype(np.int32))
                            origin = (bay_x + x0, bay_y + y0)
                            reading = read_plate_text(current_crop[y0:y1, x0:x1])
                        if reading is None or reading.text is None:
                            print("Automatic reading missed the plate. Draw a tight box around it in the next window.")
                            plate_selection = select_plate_crop(frame)
                            if plate_selection is not None:
                                plate_crop, origin = plate_selection
                                reading = read_plate_text(plate_crop)
                    except RuntimeError as error:
                        print(error)
                    plate_results[bay_index] = reading.text if reading and reading.text else ""
                    plate_engines[bay_index] = reading.engine if reading else None
                    plate_boxes[bay_index] = None
                    if reading and reading.box is not None:
                        bx, by, bw, bh = reading.box
                        plate_boxes[bay_index] = (origin[0] + bx, origin[1] + by, bw, bh)
                    if reading and reading.text:
                        print(f"Bay {bay_index + 1}: {reading.engine} read {reading.text}. Check it against the plate; OCR can misread characters.")
                    elif plate_results[bay_index] == "":
                        print(f"Bay {bay_index + 1}: no readable plate. Try a close, sharp, front-facing crop with printed block characters.")
            if key == ord("n"):
                ok, add_frame = capture.read()
                if ok and add_frame is not None:
                    new_bay = select_one_bay(add_frame)
                    if new_bay is not None:
                        new_bay[:, 0] = np.clip(new_bay[:, 0], 0, width - 1)
                        new_bay[:, 1] = np.clip(new_bay[:, 1], 0, height - 1)
                        bays.append(new_bay)
                        plate_results.append(None)
                        plate_engines.append(None)
                        plate_boxes.append(None)
                        plate_crop_origins.append(None)
                        occupied_since.append(None)
                        plate_attempted.append(False)
                        plate_futures.append(None)
                        selected_bay = len(bays) - 1
                        occupied.append(False)
                        pending.append(0)
                        # Every bay needs a synchronized empty reference crop.
                        if reference is not None:
                            reference = None
                            occupied = [False] * len(bays)
                            pending = [0] * len(bays)
                        print(
                            f"Added Bay {len(bays)}. Drag its corners to match the "
                            "skew; empty all bays and press C to calibrate."
                        )
            if key in (ord("d"), ord("x"), 8, 127) and selected_bay is not None:
                removed_number = selected_bay + 1
                bays.pop(selected_bay)
                occupied.pop(selected_bay)
                pending.pop(selected_bay)
                plate_results.pop(selected_bay)
                plate_engines.pop(selected_bay)
                plate_boxes.pop(selected_bay)
                plate_crop_origins.pop(selected_bay)
                occupied_since.pop(selected_bay)
                plate_attempted.pop(selected_bay)
                plate_futures.pop(selected_bay)
                if reference is not None:
                    reference.pop(selected_bay)
                selected_bay = None
                print(f"Deleted Bay {removed_number}.")
            if key == ord("r"):
                ok, selection_frame = capture.read()
                if ok and selection_frame is not None:
                    new_bays = select_bays(selection_frame)
                    if new_bays:
                        bays = new_bays
                        for polygon in bays:
                            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                        reference = None
                        occupied = [False] * len(bays)
                        pending = [0] * len(bays)
                        plate_results = [None] * len(bays)
                        plate_engines = [None] * len(bays)
                        plate_boxes = [None] * len(bays)
                        plate_crop_origins = [None] * len(bays)
                        occupied_since = [None] * len(bays)
                        plate_attempted = [False] * len(bays)
                        plate_futures = [None] * len(bays)
                        selected_bay = None
                        print("Bays reselected. Empty them and press C to calibrate again.")
            if key == ord("a"):
                ok, detection_frame = capture.read()
                if ok and detection_frame is not None:
                    detected = detect_bays(detection_frame)
                    if detected:
                        bays = detected
                        reference = None
                        occupied = [False] * len(bays)
                        pending = [0] * len(bays)
                        plate_results = [None] * len(bays)
                        plate_engines = [None] * len(bays)
                        plate_boxes = [None] * len(bays)
                        plate_crop_origins = [None] * len(bays)
                        occupied_since = [None] * len(bays)
                        plate_attempted = [False] * len(bays)
                        plate_futures = [None] * len(bays)
                        selected_bay = None
                        print(
                            f"Automatically found {len(bays)} bay outline(s). "
                            "Check the boxes, empty all bays, and press C to calibrate."
                        )
                    else:
                        print(
                            "No clear three-sided bay outline found. Darken the pencil "
                            "lines or press R to mark bays manually."
                        )
    except KeyboardInterrupt:
        pass
    finally:
        ocr_executor.shutdown(wait=False, cancel_futures=True)
        bridge_executor.shutdown(wait=False, cancel_futures=True)
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
