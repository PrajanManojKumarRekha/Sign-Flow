# Sign-Flow

Real-time American Sign Language (ASL) letter detection with YOLOv8. It recognises the 26 letters A–Z plus a `backspace` gesture and assembles them into text from a webcam feed.

## Project structure

```
Sign-Flow/
├── asl/                      # Shared package
│   ├── config.py             #   class names, paths, defaults
│   └── sentence_builder.py   #   gesture -> text with debouncing
├── OpenCV collector/         # Webcam capture tool for new gesture data
├── ASL.v1i.yolov8/           # Source dataset: letters A–Z
├── ASL-Custom-Gestures-1/    # Source dataset: backspace gesture
├── models/best_asl_27.pt     # Trained 27-class weights
├── tests/                    # pytest suite
├── merge_datasets.py         # Build ASL_Merged/ from the source datasets
├── train_model.py            # Train and export best weights
├── Runner.py                 # Real-time webcam inference
├── requirements.txt          # Runtime dependencies
└── requirements-dev.txt      # + pytest, ruff
```

`ASL_Merged/`, `training_results/` and base weights (`yolov8n.pt`) are generated and git-ignored.

## Setup

Requires Python 3.9+ and a webcam. An NVIDIA GPU is optional but speeds up training.

```bash
git clone https://github.com/PrajanManojKumarRekha/Sign-Flow.git
cd Sign-Flow
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Run detection** using the bundled model:

```bash
python Runner.py                      # defaults
python Runner.py --camera 1 --conf 0.7
```

Keys: `Q` quit, `C` clear the sentence. Hold each gesture steady; repeats of the same letter within 1.5 s are ignored.

**Retrain** (optional):

```bash
python merge_datasets.py              # add --force to rebuild
python train_model.py --epochs 100 --batch 16   # lower --batch if out of GPU memory
```

Best weights are copied to `models/best_asl_27.pt`.

**Collect new gesture images:**

```bash
python "OpenCV collector/Datacollection.py" space   # C = capture, Q = quit
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

CI runs both on every push and pull request.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Cannot access camera` | Close other apps using it, check OS permissions, or try `--camera 1`. |
| `data.yaml not found` | Run `python merge_datasets.py` first. |
| CUDA out of memory | Use `--batch 8` or lower. |
| Poor accuracy | Improve lighting, use a plain background, or add training data. |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), and the Roboflow community datasets used for training.
