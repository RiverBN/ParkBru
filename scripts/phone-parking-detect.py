"""Detect occupied miniature parking bays from a USB webcam or IP camera.

This prototype uses background difference to detect a car-sized change inside
a bay, then uses OCR to attempt a miniature plate read. It works best with a
fixed camera, steady lighting, white paper, pencil bay lines, and a car that
contrasts with the paper.

Usage:
    python scripts/phone-parking-detect.py                 # default camera (index 0)
    python scripts/phone-parking-detect.py --camera 1      # choose another camera
    python scripts/phone-parking-detect.py --list-cameras   # show available indices
    python scripts/phone-parking-detect.py "http://PHONE_IP:PORT/video/mjpeg"

For a local camera, a setup window opens first with a live preview, camera and
resolution selectors, and image controls. Individual controls depend on the
camera and its driver; unsupported settings are reported in the window.

Draw a box inside each bay when prompted. Automatic detection looks for
three-sided, pencil-drawn U shapes, including skewed perspective; it does not
use the car outline as a bay. Drag any yellow corner to fine-tune a bay. Keep
the bays empty and press C to capture the empty reference. Press N to add a bay,
click a bay to select it and press D to delete it, A to scan again, R to redraw
all bay boxes, L to try license-plate OCR on a selected car, or Q/Esc to quit.
"""

from __future__ import annotations

import time
import argparse
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from app_bridge import DEFAULT_BRIDGE_URL, ensure_local_bridge, send_bridge_state
from bay_vision import (
    BayPolygon, crop_bay, detect_bays, detect_change,
    EMPTY_CHANGE_THRESHOLD, FRAMES_TO_CONFIRM, OCCUPANCY_COMPONENT_THRESHOLD,
    OCCUPANCY_THRESHOLD,
)
from camera_setup import choose_camera_source_gui, configure_camera_gui, list_cameras
from detector_ui import (
    focus_cv_window, normalize_key, put_text_stroked, select_bays,
    select_one_bay, select_plate_crop,
)
from plate_reader import (
    PLATE_OCR_RETRIES, PLATE_OCR_RETRY_SECONDS, PlateReading, read_plate_text,
)


def main() -> int:
    """Parse options, open a camera, then run the live bay-monitoring loop."""
    # Read command-line options before opening camera or GUI resources.
    parser = argparse.ArgumentParser(description="Detect parking bays from a local USB/webcam camera or IP camera stream.")
    parser.add_argument("stream_url", nargs="?", help="Optional Android IP camera MJPEG/RTSP stream URL")
    parser.add_argument("--camera", type=int, default=None, help="Use a specific USB camera index and skip the source chooser")
    parser.add_argument("--list-cameras", action="store_true", help="Probe and list available local camera indices, then exit")
    parser.add_argument(
        "--server-url", default=DEFAULT_BRIDGE_URL,
        help=f"ParkBru API URL (default: {DEFAULT_BRIDGE_URL})",
    )
    parser.add_argument("--no-bridge", action="store_true", help="Do not send detector updates to the ParkBru app")
    args = parser.parse_args()
    stream_url = args.stream_url.strip() if args.stream_url else None
    bridge_url = None if args.no_bridge else args.server_url
    if args.list_cameras:
        cameras = list_cameras()
        if cameras:
            print("Available local camera indices: " + ", ".join(map(str, cameras)))
            print("Choose one with --camera N, for example: --camera " + str(cameras[0]))
        else:
            print("No local cameras found. Check the USB connection and camera permissions.")
        return 0
    camera_choice = args.camera if args.camera is not None else 0
    if camera_choice < 0:
        parser.error("--camera must be zero or greater")
    if stream_url and not stream_url.startswith(("http://", "https://", "rtsp://", "rtsps://")):
        parser.error("The stream URL should start with http://, https://, rtsp://, or rtsps://")

    # Choose a camera source, then apply USB controls or open the IP stream.
    if stream_url:
        camera_source = "android"
    elif args.camera is not None:
        camera_source = "usb"
    else:
        try:
            camera_source, stream_url = choose_camera_source_gui()
        except ImportError as error:
            print(f"Camera source chooser could not start ({error}).")
            return 1
        if camera_source is None:
            print("Camera setup cancelled.")
            return 0

    if stream_url:
        capture = cv2.VideoCapture(stream_url)
        camera_index = camera_choice
    else:
        try:
            capture, camera_index = configure_camera_gui(camera_choice)
        except ImportError as error:
            print(f"Camera setup window could not start ({error}). Install Python with Tk support.")
            return 1
        if capture is None:
            print("Camera setup cancelled.")
            return 0
    if not capture.isOpened():
        if stream_url:
            print("Could not open the IP camera stream. Check its URL and that the camera server is running.")
        else:
            print(f"Could not open local camera {camera_index}. Use --list-cameras to find available camera indices.")
        capture.release()
        return 1

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        source = "stream URL" if stream_url else f"camera {camera_index}"
        print(f"Opened {source}, but no video frame arrived. Check the camera connection and permissions.")
        capture.release()
        return 1

    # Find bay outlines automatically where possible; otherwise ask the user to draw them.
    height, width = first_frame.shape[:2]
    source_name = "Android IP camera" if camera_source == "android" else f"USB camera {camera_index}"
    print(f"Camera {source_name}: {width}x{height}")
    if bridge_url:
        print(f"Sending bay and plate readings to the local ParkBru app at {bridge_url}")
    bays = detect_bays(first_frame)
    if bays:
        print(f"Automatically found {len(bays)} bay outline(s). Check the polygons in the preview.")
    else:
        print("Could not confidently find bay outlines; select the bay interiors manually.")
        bays = select_bays(capture, first_frame)
    if not bays:
        print("No bays selected. Run the script again and draw at least one bay box.")
        capture.release()
        cv2.destroyAllWindows()
        return 1

    # Keep every corner inside the actual camera frame.
    for polygon in bays:
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    # Start the optional app bridge and initialize per-bay tracking state.
    local_bridge_server = ensure_local_bridge(bridge_url)
    window_name = "ParkBru bays - N add | click + D delete | drag corners | A auto | Q quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    focus_cv_window(window_name)
    reference: list[tuple[np.ndarray, np.ndarray]] | None = None
    occupied = [False] * len(bays)
    pending = [0] * len(bays)
    selected_bay: int | None = None
    plate_results: list[str | None] = [None] * len(bays)
    plate_engines: list[str | None] = [None] * len(bays)
    plate_boxes: list[tuple[int, int, int, int] | None] = [None] * len(bays)
    plate_crop_origins: list[tuple[int, int] | None] = [None] * len(bays)
    occupied_since: list[float | None] = [None] * len(bays)
    plate_attempt_count = [0] * len(bays)
    plate_retry_at: list[float | None] = [None] * len(bays)
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
    # Main loop: sample frames, update occupancy/OCR, draw the view, and handle shortcuts.
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
                "Plate OCR: PaddleOCR PP-OCRv5 only | format: 2-3 letters + 1-4 digits",
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
                message = "PaddleOCR only | retry every 60s up to 5 times | L manual scan"
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
                        reading = None
                        try:
                            reading = future.result()
                        except Exception as error:
                            print(f"Bay {index + 1}: PaddleOCR read failed: {error}")
                        if reading is not None and reading.text:
                            plate_results[index] = reading.text
                            plate_engines[index] = reading.engine
                            plate_retry_at[index] = None
                            if reading.box is not None and plate_crop_origins[index] is not None:
                                ox, oy = plate_crop_origins[index]
                                bx, by, bw, bh = reading.box
                                plate_boxes[index] = (ox + bx, oy + by, bw, bh)
                            print(f"Bay {index + 1}: {reading.engine} read valid plate {plate_results[index]}.")
                        elif plate_attempt_count[index] > PLATE_OCR_RETRIES:
                            plate_results[index] = "INVALID PLATE"
                            plate_engines[index] = "PaddleOCR PP-OCRv5"
                            plate_retry_at[index] = None
                            print(f"Bay {index + 1}: no valid plate after {PLATE_OCR_RETRIES} retries; marking INVALID PLATE.")
                        else:
                            plate_results[index] = None
                            plate_engines[index] = None
                            plate_retry_at[index] = now + PLATE_OCR_RETRY_SECONDS
                            print(f"Bay {index + 1}: no valid plate; retrying in 60 seconds ({plate_attempt_count[index]}/{PLATE_OCR_RETRIES} retries used).")
                        plate_futures[index] = None
                        plate_crop_origins[index] = None

                    if occupied[index]:
                        if occupied_since[index] is None:
                            occupied_since[index] = now
                        first_scan_due = plate_attempt_count[index] == 0 and now - occupied_since[index] >= 5.0
                        retry_due = plate_retry_at[index] is not None and now >= plate_retry_at[index]
                        if plate_results[index] is None and plate_futures[index] is None and (first_scan_due or retry_due):
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
                            plate_attempt_count[index] += 1
                            plate_retry_at[index] = None
                            print(f"Bay {index + 1}: starting PaddleOCR scan {plate_attempt_count[index]}.")
                    else:
                        occupied_since[index] = None
                        if plate_futures[index] is not None:
                            plate_futures[index].cancel()
                        plate_futures[index] = None
                        plate_attempt_count[index] = 0
                        plate_retry_at[index] = None
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
                        if plate_futures[index] is not None:
                            plate_status = "PaddleOCR reading plate..."
                        elif plate_attempt_count[index] == 0:
                            remaining = max(0, int(5 - (now - occupied_since[index]) + 0.99))
                            plate_status = f"Plate scan starts in {remaining}s"
                        elif plate_retry_at[index] is not None:
                            retry_remaining = max(0, int(plate_retry_at[index] - now + 0.99))
                            plate_status = f"No valid plate Â· retry in {retry_remaining}s"
                        else:
                            plate_status = "PaddleOCR waiting..."
                        put_text_stroked(
                            display,
                            plate_status,
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
                    bridge_future = bridge_executor.submit(send_bridge_state, bridge_url, snapshot, source_name)
                    bridge_last_send = now
            cv2.imshow(window_name, display)

            key = normalize_key(cv2.waitKeyEx(1) & 0xFF)
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
                    plate_attempt_count = [0] * len(bays)
                    plate_retry_at = [None] * len(bays)
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
                            plate_selection = select_plate_crop(capture, frame)
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
                    new_bay = select_one_bay(capture, add_frame)
                    if new_bay is not None:
                        new_bay[:, 0] = np.clip(new_bay[:, 0], 0, width - 1)
                        new_bay[:, 1] = np.clip(new_bay[:, 1], 0, height - 1)
                        bays.append(new_bay)
                        plate_results.append(None)
                        plate_engines.append(None)
                        plate_boxes.append(None)
                        plate_crop_origins.append(None)
                        occupied_since.append(None)
                        plate_attempt_count.append(0)
                        plate_retry_at.append(None)
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
                plate_attempt_count.pop(selected_bay)
                plate_retry_at.pop(selected_bay)
                plate_futures.pop(selected_bay)
                if reference is not None:
                    reference.pop(selected_bay)
                selected_bay = None
                print(f"Deleted Bay {removed_number}.")
            if key == ord("r"):
                ok, selection_frame = capture.read()
                if ok and selection_frame is not None:
                    new_bays = select_bays(capture, selection_frame)
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
                        plate_attempt_count = [0] * len(bays)
                        plate_retry_at = [None] * len(bays)
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
                        plate_attempt_count = [0] * len(bays)
                        plate_retry_at = [None] * len(bays)
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
    # Always release camera, workers, windows, and the embedded API on exit.
    finally:
        ocr_executor.shutdown(wait=False, cancel_futures=True)
        bridge_executor.shutdown(wait=False, cancel_futures=True)
        capture.release()
        cv2.destroyAllWindows()
        if local_bridge_server is not None:
            local_bridge_server.shutdown()
            local_bridge_server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
