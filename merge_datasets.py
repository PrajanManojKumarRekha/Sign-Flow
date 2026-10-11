"""Merge the A-Z and backspace datasets into a single 27-class YOLO dataset."""

import argparse
import logging
import shutil
from pathlib import Path

import yaml

from asl.config import (
    CLASS_NAMES,
    DATASET_A_Z,
    DATASET_BACKSPACE,
    IMAGE_EXTENSIONS,
    MERGED_DATASET_DIR,
    SPLITS,
)

logger = logging.getLogger("merge_datasets")


def offset_label_file(src: Path, dst: Path, class_offset: int) -> None:
    """Copy a YOLO label file, shifting every class id by ``class_offset``."""
    if class_offset == 0:
        shutil.copy2(src, dst)
        return
    lines = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if parts:
            parts[0] = str(int(parts[0]) + class_offset)
            lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def copy_dataset(source: Path, merged: Path, class_offset: int, prefix: str) -> dict:
    """Copy one dataset into ``merged``, prefixing filenames to avoid collisions.

    Image and label files share a stem, so both are renamed together and
    always stay paired.
    """
    stats = {split: 0 for split in SPLITS}
    for split in SPLITS:
        src_images = source / split / "images"
        src_labels = source / split / "labels"
        if not src_images.is_dir():
            logger.warning("%s not found, skipping", src_images)
            continue

        dst_images = merged / split / "images"
        dst_labels = merged / split / "labels"
        for image in sorted(src_images.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = image.stem if not (dst_images / image.name).exists() else f"{prefix}_{image.stem}"
            shutil.copy2(image, dst_images / f"{stem}{image.suffix}")

            label = src_labels / f"{image.stem}.txt"
            if label.is_file():
                offset_label_file(label, dst_labels / f"{stem}.txt", class_offset)
            else:
                logger.warning("No label for %s", image)
            stats[split] += 1
    return stats


def write_data_yaml(merged: Path) -> Path:
    config = {
        "path": str(merged.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = merged / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, default_flow_style=False))
    return yaml_path


def merge(output: Path, force: bool = False) -> None:
    if output.exists():
        if not force:
            raise SystemExit(f"{output} already exists; pass --force to rebuild it.")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)

    stats_az = copy_dataset(DATASET_A_Z, output, class_offset=0, prefix="AZ")
    stats_back = copy_dataset(DATASET_BACKSPACE, output, class_offset=26, prefix="backspace")
    for split in SPLITS:
        logger.info("%s: %d images", split, stats_az[split] + stats_back[split])

    logger.info("Wrote %s", write_data_yaml(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MERGED_DATASET_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite an existing merged dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    merge(args.output, args.force)


if __name__ == "__main__":
    main()
