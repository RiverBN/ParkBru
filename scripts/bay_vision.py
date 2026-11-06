"""Parking-bay geometry and image-difference occupancy detection."""

import cv2
import numpy as np

BayPolygon = np.ndarray

PIXEL_DIFF_THRESHOLD = 24
OCCUPANCY_THRESHOLD = 0.035
OCCUPANCY_COMPONENT_THRESHOLD = 0.02
EMPTY_CHANGE_THRESHOLD = 0.018
FRAMES_TO_CONFIRM = 4

# Occupancy uses reduced-resolution bay crops, capped at 640 pixels on the
# longest side. Full-resolution source crops remain available for OCR/display.
OCCUPANCY_SCALE = 0.5

OCCUPANCY_MAX_DIMENSION = 640

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
    original_height, original_width = current.shape[:2]
    processing_scale = min(
        OCCUPANCY_SCALE,
        OCCUPANCY_MAX_DIMENSION / max(original_width, original_height),
    )
    small_width = max(1, int(original_width * processing_scale))
    small_height = max(1, int(original_height * processing_scale))
    target_size = (small_width, small_height)
    reference = cv2.resize(reference, target_size, interpolation=cv2.INTER_AREA)
    current = cv2.resize(current, target_size, interpolation=cv2.INTER_AREA)
    bay_mask = cv2.resize(bay_mask, target_size, interpolation=cv2.INTER_NEAREST)
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
    x, y, box_width, box_height = cv2.boundingRect(largest)
    scale_x = original_width / small_width
    scale_y = original_height / small_height
    full_resolution_box = (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(box_width * scale_x))),
        max(1, int(round(box_height * scale_y))),
    )
    return ratio, component_ratio, full_resolution_box

