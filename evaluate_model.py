"""Per-class evaluation of a trained model on a dataset split.

Use this to find weak letters before deciding what data to collect.
"""

import argparse
import logging
from pathlib import Path

from asl.config import DEFAULT_MODEL_PATH, MERGED_DATASET_DIR

logger = logging.getLogger("evaluate_model")


def evaluate(weights: Path, data_yaml: Path, split: str, device: str | None) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, device=device, plots=False, verbose=False)
    rows = []
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        rows.append({
            "class": model.names[int(class_idx)],
            "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"{'class':<18}{'precision':>10}{'recall':>10}{'mAP50':>10}{'mAP50-95':>10}")
    for row in sorted(rows, key=lambda r: r["mAP50"]):
        print(f"{row['class']:<18}{row['precision']:>10.2f}{row['recall']:>10.2f}"
              f"{row['mAP50']:>10.2f}{row['mAP50-95']:>10.2f}")
    mean = sum(r["mAP50"] for r in rows) / len(rows)
    print(f"\nclasses evaluated: {len(rows)}   mean mAP50: {mean:.3f}   (sorted worst first)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=MERGED_DATASET_DIR / "data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print_table(evaluate(args.weights, args.data, args.split, args.device))


if __name__ == "__main__":
    main()
