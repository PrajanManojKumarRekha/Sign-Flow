"""Train the YOLOv8 ASL detector on the merged dataset."""

import argparse
import logging
import shutil
from pathlib import Path

from asl.config import (
    BASE_MODEL,
    DEFAULT_MODEL_PATH,
    MERGED_DATASET_DIR,
    RUN_NAME,
    TRAINING_OUTPUT_DIR,
)

logger = logging.getLogger("train_model")


def train(epochs: int, batch: int, imgsz: int, patience: int, device: str | None) -> Path:
    import torch
    from ultralytics import YOLO

    data_yaml = MERGED_DATASET_DIR / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"{data_yaml} not found. Run `python merge_datasets.py` first.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        name=RUN_NAME,
        project=str(TRAINING_OUTPUT_DIR),
        device=device,
        plots=True,
    )

    metrics = model.val()
    logger.info(
        "mAP@0.5=%.3f mAP@0.5-95=%.3f precision=%.3f recall=%.3f",
        metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr,
    )

    best = Path(model.trainer.best)
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, DEFAULT_MODEL_PATH)
    logger.info("Best weights saved to %s", DEFAULT_MODEL_PATH)
    return DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16, help="reduce if out of GPU memory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=10, help="early-stopping patience")
    parser.add_argument("--device", default=None, help="e.g. cpu, 0 (default: auto)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(args.epochs, args.batch, args.imgsz, args.patience, args.device)


if __name__ == "__main__":
    main()
