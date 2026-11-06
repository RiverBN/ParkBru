"""PaddleOCR setup and validation for supported miniature plate formats."""

import re
from dataclasses import dataclass

import numpy as np

_PADDLE_READER = None

PLATE_PATTERN = re.compile(r"^[A-Z]{2,3}[0-9]{1,4}$")

PLATE_OCR_RETRIES = 5

PLATE_OCR_RETRY_SECONDS = 60.0

@dataclass(frozen=True)
class PlateReading:
    text: str | None
    box: tuple[int, int, int, int] | None
    engine: str

def read_plate_text(car_image: np.ndarray) -> PlateReading:
    """Read a plate with PaddleOCR and accept only the supported plate format."""
    global _PADDLE_READER
    try:
        import paddle
        from paddleocr import PaddleOCR
    except ImportError:
        print("PaddleOCR is unavailable; plate reads will remain pending until it is installed.")
        return PlateReading(None, None, "PaddleOCR PP-OCRv5")

    try:
        if _PADDLE_READER is None:
            use_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            ocr_device = "gpu:0" if use_gpu else "cpu"
            print(f"PaddleOCR device: {'NVIDIA GPU' if use_gpu else 'CPU'}")
            print("Loading PaddleOCR PP-OCRv5 models (first use downloads model files)...")
            _PADDLE_READER = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=ocr_device,
                enable_mkldnn=False,
            )
        paddle_results = _PADDLE_READER.predict(car_image)
        paddle_candidates: list[tuple[float, str, tuple[int, int, int, int] | None]] = []
        for result in paddle_results:
            payload = result.json
            if callable(payload):
                payload = payload()
            if not isinstance(payload, dict):
                continue
            details = payload.get("res", payload)
            texts = details.get("rec_texts", [])
            scores = details.get("rec_scores", [])
            boxes = details.get("rec_boxes", [])
            for text_index, (text, score) in enumerate(zip(texts, scores)):
                cleaned = "".join(re.findall(r"[A-Z0-9]", str(text).upper()))
                if not PLATE_PATTERN.fullmatch(cleaned):
                    continue
                box = None
                if text_index < len(boxes):
                    x0, y0, x1, y1 = [int(value) for value in boxes[text_index]]
                    box = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
                paddle_candidates.append((float(score), cleaned, box))
        if not paddle_candidates:
            return PlateReading(None, None, "PaddleOCR PP-OCRv5")
        _score, text, box = max(paddle_candidates, key=lambda item: item[0])
        return PlateReading(text, box, "PaddleOCR PP-OCRv5")
    except Exception as error:
        print(f"PaddleOCR plate read failed: {error}")
        return PlateReading(None, None, "PaddleOCR PP-OCRv5")

