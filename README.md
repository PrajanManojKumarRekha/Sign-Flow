# Sign-Flow

On-device American Sign Language (ASL) fingerspelling to text and speech. Sign letters in front of your camera and Sign-Flow builds sentences you can show in a meeting or have read aloud.

**Privacy:** recognition runs entirely in your browser or on your machine. Camera video is never uploaded, recorded or stored.

> Status: early MVP. It recognises fingerspelled letters A–Z plus a backspace gesture; accuracy varies by letter (see [Roadmap](docs/ROADMAP.md)). It is an accessibility aid, not a full ASL translator.

## Repository layout

```
Sign-Flow/
├── apps/
│   └── web/                  # Browser app (UI, model, onnxruntime-web)
├── packages/
│   └── engine/               # @sign-flow/engine: recognition + text logic shared by all apps
├── ml/                       # Python: datasets, training, evaluation, export, desktop demo
│   ├── asl/                  #   shared config and sentence builder
│   ├── datasets/             #   source datasets and raw webcam captures
│   ├── models/               #   trained weights (best_asl_27.pt)
│   ├── tools/                #   data collection
│   └── tests/
├── scripts/                  # build-web.mjs, serve.mjs
├── docs/                     # Roadmap
└── .github/workflows/        # CI and GitHub Pages deploy
```

## Web app

Requires Node.js 20+ and a webcam. No `npm install` is needed.

```bash
npm run dev        # builds dist/web and serves it at http://localhost:8000
```

Click **Start camera**, then hold a letter steady until the ring fills. Lower your hand to repeat the same letter. Sentences are auto-capitalised and punctuation attaches to the previous word. Space and punctuation use the on-screen buttons or keyboard (`Space`, `.`, `?`, `!`, `,`, `Enter`, `Backspace`, `Esc`) until those gestures are trained into the model.

**In a meeting:** open **Captions view** and share that window or tab in Zoom, Google Meet or Teams. Use **Speak** or *Speak each sentence* to have messages read aloud.

Every push to `main` deploys the app to GitHub Pages (enable it once under *Settings → Pages → Source: GitHub Actions*).

## Training and model work (`ml/`)

Requires Python 3.10+; a CUDA GPU is strongly recommended for training.

```bash
cd ml
pip install -r requirements.txt

python prepare_punctuation.py         # optional: adds space . ? ! newline classes (32 total)
python merge_datasets.py --force      # builds datasets/merged
python train_model.py --epochs 100 --batch 16 --model yolov8s.pt
python evaluate_model.py              # per-class precision/recall, worst first
python export_onnx.py                 # refreshes apps/web/model/
```

The web app and the desktop demo read class names from the model, so a model with punctuation classes needs no code changes.

Other tools:

```bash
python run_webcam.py                  # desktop OpenCV demo (Q quit, C clear)
python tools/collect_gestures.py space   # capture new gesture images (C capture, Q quit)
```

Punctuation labels are weak (the capture guide box) and come from one person and room; collect varied data before relying on that model.

## Development

```bash
npm test                              # engine unit tests
cd ml && pip install -r requirements-dev.txt && ruff check . && pytest
```

CI runs both on every push to `main` and `develop` and on pull requests. Work happens on `develop`; `main` is what gets deployed.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Camera does not start | The page must be on `localhost` or HTTPS. Close other apps using the camera and check browser permissions. |
| Letter not recognised | Some letters are weaker (I, Q, Y, P, N, S, T; J and Z need motion). Adjust Sensitivity, improve lighting, use a plain background. |
| `data.yaml not found` | Run `python merge_datasets.py` in `ml/` first. |
| CUDA out of memory | Use `--batch 8` or lower. |

## License

MIT — see [LICENSE](LICENSE). Bundled onnxruntime-web is MIT licensed (see `apps/web/vendor/NOTICE.txt`).

## Acknowledgments

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [ONNX Runtime](https://onnxruntime.ai/), [OpenCV](https://opencv.org/), and the Roboflow community datasets used for training.
