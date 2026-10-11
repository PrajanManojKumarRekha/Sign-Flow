"""Turn raw webcam captures of punctuation gestures into a YOLO dataset.

The original collector burned a green guide box and an "Images: N" counter into
every frame. Left in, a model can learn those instead of the hand, so they are
inpainted away here. Boxes are weak labels: the guide rectangle the hand was
held in. Review the output (and tighten boxes in a labeling tool if you can)
before trusting the trained model.

Splits are contiguous blocks per class, not random, so near-identical
consecutive frames never leak between train and test.
"""

import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from asl.config import COLLECTOR_DATA_DIR, DATASET_PUNCTUATION, PUNCTUATION_CLASSES

logger = logging.getLogger("prepare_punctuation")

# Guide rectangle drawn by the original Datacollection.py, as (x1, y1, x2, y2).
GUIDE_BOX = (200, 100, 450, 350)
COUNTER_REGION = (0, 0, 320, 50)
SPLITS = (("train", 0.70), ("valid", 0.15), ("test", 0.15))


def remove_overlay(image: np.ndarray) -> np.ndarray:
    """Inpaint the green guide box and counter text baked into old captures."""
    b, g, r = (image[:, :, i].astype(int) for i in range(3))
    green = (g > 170) & (r < 120) & (b < 120)

    zone = np.zeros(green.shape, dtype=bool)
    x1, y1, x2, y2 = GUIDE_BOX
    pad = 4
    zone[y1 - pad:y2 + pad, x1 - pad:x2 + pad] = True
    zone[y1 + pad:y2 - pad, x1 + pad:x2 - pad] = False  # keep hand pixels inside the box
    cx1, cy1, cx2, cy2 = COUNTER_REGION
    zone[cy1:cy2, cx1:cx2] = True

    mask = (green & zone).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def yolo_label(class_id: int, box: tuple, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{class_id} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n"


def split_bounds(n: int) -> dict[str, range]:
    train_end = int(n * SPLITS[0][1])
    valid_end = train_end + int(n * SPLITS[1][1])
    return {"train": range(0, train_end), "valid": range(train_end, valid_end), "test": range(valid_end, n)}


def prepare(source: Path, output: Path, force: bool) -> dict:
    if output.exists():
        if not force:
            raise SystemExit(f"{output} already exists; pass --force to rebuild it.")
        shutil.rmtree(output)
    for split, _ in SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)

    counts = {split: 0 for split, _ in SPLITS}
    for class_id, name in enumerate(PUNCTUATION_CLASSES):
        files = sorted((source / name).glob("*.jpg"), key=lambda f: int(f.stem))
        if not files:
            logger.warning("No images for %s in %s", name, source)
            continue
        for split, idx_range in split_bounds(len(files)).items():
            for i in idx_range:
                image = cv2.imread(str(files[i]))
                if image is None:
                    logger.warning("Unreadable image %s", files[i])
                    continue
                height, width = image.shape[:2]
                stem = f"{name}_{files[i].stem}"
                cv2.imwrite(str(output / split / "images" / f"{stem}.jpg"), remove_overlay(image))
                (output / split / "labels" / f"{stem}.txt").write_text(
                    yolo_label(class_id, GUIDE_BOX, width, height))
                counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=COLLECTOR_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_PUNCTUATION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Prepared %s", prepare(args.source, args.output, args.force))


if __name__ == "__main__":
    main()
