"""Export the trained weights to ONNX for the browser app (web/model/asl.onnx)."""

import argparse
import logging
import shutil
from pathlib import Path

from asl.config import DEFAULT_MODEL_PATH, ROOT_DIR

logger = logging.getLogger("export_onnx")
WEB_MODEL = ROOT_DIR / "web" / "model" / "asl.onnx"


def export(weights: Path, output: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}. Run `python train_model.py` first.")
    exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=12, simplify=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, output)
    logger.info("Wrote %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=WEB_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export(args.weights, args.output, args.imgsz)


if __name__ == "__main__":
    main()
