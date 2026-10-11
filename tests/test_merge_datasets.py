import yaml

import merge_datasets as md
from asl.config import CLASS_NAMES


def _make_dataset(root, stem, cls):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "train" / "images" / f"{stem}.jpg").write_bytes(b"x")
    (root / "train" / "labels" / f"{stem}.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n")


def test_offset_label_file(tmp_path):
    src, dst = tmp_path / "a.txt", tmp_path / "b.txt"
    src.write_text("0 0.1 0.2 0.3 0.4\n\n")
    md.offset_label_file(src, dst, 26)
    assert dst.read_text() == "26 0.1 0.2 0.3 0.4\n"


def test_copy_dataset_keeps_pairs_on_collision(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _make_dataset(a, "img", 3)
    _make_dataset(b, "img", 0)
    for split in md.SPLITS:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)

    md.copy_dataset(a, out, 0, "AZ")
    md.copy_dataset(b, out, 26, "backspace")

    assert (out / "train/labels/img.txt").read_text().startswith("3 ")
    assert (out / "train/images/backspace_img.jpg").exists()
    assert (out / "train/labels/backspace_img.txt").read_text().startswith("26 ")


def test_data_yaml(tmp_path):
    cfg = yaml.safe_load(md.write_data_yaml(tmp_path).read_text())
    assert cfg["nc"] == len(CLASS_NAMES) == 27
    assert cfg["names"][-1] == "backspace"


def test_data_yaml_with_punctuation(tmp_path):
    from asl.config import ALL_CLASS_NAMES

    cfg = yaml.safe_load(md.write_data_yaml(tmp_path, ALL_CLASS_NAMES).read_text())
    assert cfg["nc"] == 32
    assert cfg["names"][26] == "backspace" and cfg["names"][27] == "space"
