"""OpenCV windows and mouse/keyboard helpers used by the detector."""

import cv2
import numpy as np

from bay_vision import BayPolygon

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

def focus_cv_window(title: str) -> None:
    """Keep an OpenCV window in normal z-order, not always on top."""
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        pass

def normalize_key(key: int) -> int:
    """Treat shifted letter shortcuts the same as lowercase letters."""
    return key + 32 if 65 <= key <= 90 else key

def select_rectangles(
    capture: cv2.VideoCapture,
    first_frame: np.ndarray,
    title: str,
    instruction: str,
    multiple: bool = True,
) -> list[tuple[int, int, int, int]]:
    """Draw one or more boxes over the live camera preview; Enter accepts."""
    rectangles: list[tuple[int, int, int, int]] = []
    drag_start: tuple[int, int] | None = None
    drag_end: tuple[int, int] | None = None

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal drag_start, drag_end
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            drag_end = (x, y)
            x0, x1 = sorted((drag_start[0], drag_end[0]))
            y0, y1 = sorted((drag_start[1], drag_end[1]))
            if x1 - x0 > 3 and y1 - y0 > 3:
                rectangles.append((x0, y0, x1 - x0, y1 - y0))
            drag_start = None

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    focus_cv_window(title)
    cv2.setMouseCallback(title, mouse_callback)
    frame = first_frame
    try:
        while True:
            ok, live_frame = capture.read()
            if ok and live_frame is not None:
                frame = live_frame
            display = frame.copy()
            cv2.putText(display, instruction, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 30, 40), 2, cv2.LINE_AA)
            help_text = "Drag box | Enter/Return: accept | Backspace: undo | Esc: cancel"
            if multiple:
                help_text = "Drag each box | Enter/Return: finish | Backspace: undo | Esc: cancel"
            cv2.putText(display, help_text, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 30, 40), 2, cv2.LINE_AA)
            for x, y, w, h in rectangles:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 255), 2, cv2.LINE_AA)
            if drag_start is not None and drag_end is not None:
                cv2.rectangle(display, drag_start, drag_end, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(title, display)
            key = normalize_key(cv2.waitKeyEx(15) & 0xFF)
            if key in (10, 13, 32):  # LF, CR, or Space
                if rectangles:
                    break
                if not multiple and drag_start is None:
                    print("Draw a box before accepting the bay area.")
            elif key in (8, 127):
                if rectangles:
                    rectangles.pop()
            elif key == 27:
                rectangles.clear()
                break
    finally:
        cv2.destroyWindow(title)
    return rectangles

def select_bays(capture: cv2.VideoCapture, frame: np.ndarray) -> list[BayPolygon]:
    """Let the user draw rectangular bay areas, avoiding pencil lines."""
    rectangles = select_rectangles(capture, frame, "Select bay interiors", "Drag a box inside each bay")
    return [
        np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        for x, y, w, h in rectangles
    ]

def select_one_bay(capture: cv2.VideoCapture, frame: np.ndarray) -> BayPolygon | None:
    """Let the user add one bay by dragging a box; corners can then be skewed."""
    rectangles = select_rectangles(capture, frame, "Add one bay", "Drag a box for the new bay", multiple=False)
    if not rectangles:
        return None
    x, y, w, h = rectangles[0]
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

def select_plate_crop(capture: cv2.VideoCapture, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Let the user tightly select a hard-to-localize miniature plate."""
    rectangles = select_rectangles(
        capture, frame, "Select rear plate", "Draw a tight box around the rear plate", multiple=False
    )
    if not rectangles:
        return None
    x, y, w, h = rectangles[0]
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame[y:y + h, x:x + w].copy(), (x, y)

