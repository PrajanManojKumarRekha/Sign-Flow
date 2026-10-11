# Sign-Flow

Real-time American Sign Language (ASL) letter detection with YOLOv8. It recognises the 26 letters A–Z plus a `backspace` gesture and assembles them into text from a webcam feed.

## Project structure

```
Sign-Flow/
├── asl/                      # Shared package
│   ├── config.py             #   class names, paths, defaults
│   └── sentence_builder.py   #   gesture -> text with debouncing
├── web/                      # Browser app (runs the model client-side via ONNX)
│   ├── index.html, css/, js/ #   UI, detector, sentence composer, tests in tests/
│   ├── model/asl.onnx        #   exported model
│   └── vendor/               #   onnxruntime-web (wasm build)
├── OpenCV collector/         # Webcam capture tool for new gesture data
├── ASL.v1i.yolov8/           # Source dataset: letters A–Z
├── ASL-Custom-Gestures-1/    # Source dataset: backspace gesture
├── models/best_asl_27.pt     # Trained 27-class weights
├── tests/                    # pytest suite
├── merge_datasets.py         # Build ASL_Merged/ from the source datasets
├── prepare_punctuation.py    # Webcam captures -> labeled punctuation dataset
├── train_model.py            # Train and export best weights
├── evaluate_model.py         # Per-class accuracy report
├── export_onnx.py            # Export weights to web/model/asl.onnx
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

## Web app (translator MVP)

Runs entirely in the browser: the camera stream is processed on-device and never uploaded.

```bash
python -m http.server 8000 -d web     # then open http://localhost:8000
```

The camera needs `localhost` or HTTPS. Hold a letter steady until the ring fills to type it; lower your hand to repeat the same letter. Sentences are auto-capitalised, punctuation attaches to the previous word, and finished sentences can be spoken aloud. Space and punctuation currently use the on-screen buttons or keyboard (`Space`, `.`, `?`, `!`, `,`, `Enter`, `Backspace`, `Esc`) until those gestures are trained into the model. After retraining, refresh the browser model with `python export_onnx.py`.

## Desktop usage

**Run detection** using the bundled model:

```bash
python Runner.py                      # defaults
python Runner.py --camera 1 --conf 0.7
```

Keys: `Q` quit, `C` clear the sentence. Hold each gesture steady; repeats of the same letter within 1.5 s are ignored.

**Retrain** (optional; a GPU is strongly recommended):

```bash
python prepare_punctuation.py         # optional: adds space . ? ! newline classes (32 total)
python merge_datasets.py --force      # picks up ASL-Punctuation/ automatically
python train_model.py --epochs 100 --batch 16   # lower --batch if out of GPU memory
python evaluate_model.py              # per-class precision/recall, worst first
python export_onnx.py                 # refresh the browser model and its class list
```

Best weights are copied to `models/best_asl_27.pt`. The browser app and `Runner.py` read class names from the model, so a 32-class model needs no code changes: punctuation gestures type `. ? !`, spaces and line breaks directly.

Punctuation labels are weak (the capture guide box) and all come from one person and room, so treat that model as a prototype and collect varied data (more people, lighting and backgrounds) before relying on it.

**Collect new gesture images:**

```bash
python "OpenCV collector/Datacollection.py" space   # C = capture, Q = quit
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
node --test web/tests/*.test.js      # web app logic (Node 18+)
```

CI runs all three on every push and pull request.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Cannot access camera` | Close other apps using it, check OS permissions, or try `--camera 1`. |
| `data.yaml not found` | Run `python merge_datasets.py` first. |
| CUDA out of memory | Use `--batch 8` or lower. |
| Letter not recognised | The model is weaker on some letters (e.g. A, and motion letters J/Z). Adjust the Sensitivity slider, improve lighting, or add training data. |
| Poor accuracy | Improve lighting, use a plain background, or add training data. |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), and the Roboflow community datasets used for training.
