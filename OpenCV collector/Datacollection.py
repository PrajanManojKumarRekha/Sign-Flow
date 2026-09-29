"""Capture webcam images for a gesture class. Press C to capture, Q to quit."""

import argparse
from pathlib import Path

import cv2


def collect(class_name: str, camera: int, output_root: Path) -> None:
    save_dir = output_root / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    # Continue numbering so re-runs never overwrite earlier captures.
    count = len(list(save_dir.glob("*.jpg")))

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot access camera {camera}.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            preview = frame.copy()
            cv2.rectangle(preview, (200, 100), (450, 350), (0, 255, 0), 2)
            cv2.putText(preview, f"{class_name}: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                cv2.imwrite(str(save_dir / f"{count}.jpg"), frame)
                count += 1
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("class_name", help="gesture label, e.g. space")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "custom_data")
    args = parser.parse_args()
    collect(args.class_name, args.camera, args.output)


if __name__ == "__main__":
    main()
