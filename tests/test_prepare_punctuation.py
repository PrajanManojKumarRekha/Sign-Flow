import cv2
import numpy as np

import prepare_punctuation as pp
from asl.config import PUNCTUATION_CLASSES


def test_split_bounds_are_contiguous_and_cover_everything():
    bounds = pp.split_bounds(100)
    assert [len(bounds[s]) for s in ("train", "valid", "test")] == [70, 15, 15]
    assert bounds["train"].stop == bounds["valid"].start
    assert bounds["valid"].stop == bounds["test"].start == 85


def test_yolo_label_is_normalised():
    assert pp.yolo_label(2, (0, 0, 320, 240), 640, 480) == "2 0.250000 0.250000 0.500000 0.500000\n"


def test_remove_overlay_erases_guide_box_but_keeps_hand_pixels():
    image = np.full((480, 640, 3), (120, 130, 140), np.uint8)
    x1, y1, x2, y2 = pp.GUIDE_BOX
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, "Images: 5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image[200:220, 300:320] = (0, 255, 0)  # green pixels inside the box: not overlay

    cleaned = pp.remove_overlay(image)

    green = (cleaned[:, :, 1] > 200) & (cleaned[:, :, 0] < 60) & (cleaned[:, :, 2] < 60)
    assert not green[:60, :].any() and not green[y1 - 4:y1 + 4, x1:x2].any()
    assert (cleaned[210, 310] == (0, 255, 0)).all()


def test_prepare_writes_paired_images_and_labels(tmp_path):
    src = tmp_path / "src"
    for name in PUNCTUATION_CLASSES:
        (src / name).mkdir(parents=True)
        for i in range(10):
            cv2.imwrite(str(src / name / f"{i}.jpg"), np.zeros((480, 640, 3), np.uint8))

    counts = pp.prepare(src, tmp_path / "out", force=False)

    assert counts == {"train": 35, "valid": 5, "test": 10}
    for split in ("train", "valid", "test"):
        images = {p.stem for p in (tmp_path / "out" / split / "images").iterdir()}
        labels = {p.stem for p in (tmp_path / "out" / split / "labels").iterdir()}
        assert images == labels
