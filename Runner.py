"""Real-time ASL detection from a webcam."""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from asl.config import CLASS_NAMES, BACKSPACE, CONFIDENCE_THRESHOLD, DEFAULT_MODEL_PATH
from asl.sentence_builder import SentenceBuilder

logger = logging.getLogger("runner")

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_HEIGHT = 180


def draw_detection(frame, box, class_name: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    color = (0, 165, 255) if class_name == BACKSPACE else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"{class_name} {confidence:.2f}"
    (width, height), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - height - 10), (x1 + width + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), FONT, 0.7, (255, 255, 255), 2)


def build_panel(width: int, text: str) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, width, 3), 40, dtype=np.uint8)
    if len(text) > 50:
        text = "..." + text[-47:]
    cv2.putText(panel, "ASL Sentence Builder", (10, 30), FONT, 0.8, (100, 200, 255), 2)
    cv2.rectangle(panel, (10, 45), (width - 10, 100), (60, 60, 60), -1)
    cv2.putText(panel, text, (20, 80), FONT, 1.0, (255, 255, 255), 2)
    cv2.putText(panel, "Q: Quit  |  C: Clear", (10, 130), FONT, 0.5, (150, 150, 150), 1)
    return panel


def run(model_path: Path, camera: int, confidence: float) -> None:
    from ultralytics import YOLO

    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}. Run `python train_model.py` first.")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot access camera {camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    builder = SentenceBuilder()
    fps, frames, fps_start = 0, 0, time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera returned no frame; stopping")
                break
            frame = cv2.flip(frame, 1)
            now = time.time()

            for result in model(frame, conf=confidence, verbose=False):
                for box in result.boxes:
                    xyxy = tuple(map(int, box.xyxy[0]))
                    score = float(box.conf[0])
                    name = CLASS_NAMES[int(box.cls[0])]
                    draw_detection(frame, xyxy, name, score)
                    builder.add_character(name, now)

            frames += 1
            if now - fps_start > 1:
                fps, frames, fps_start = frames, 0, now
            cv2.putText(frame, f"FPS: {fps}", (10, 35), FONT, 1.0, (0, 255, 0), 2)

            cv2.imshow("ASL Detection", np.vstack([frame, build_panel(frame.shape[1], builder.get_text())]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                builder.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info("Final text: %s", builder.get_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="confidence threshold")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.model, args.camera, args.conf)


if __name__ == "__main__":
    main()
