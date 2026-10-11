# Roadmap

Goal: a Deaf or mute participant signs in front of their camera, and everyone in a Zoom, Google Meet or Teams meeting sees (and can hear) their message live. Only the signer installs anything, and video never leaves their device.

## Architecture

One recognition engine, several integration "surfaces":

```
camera → [ENGINE: hand landmarks → sign classifier → text decoder] → text events
                                                                          │
     ┌───────────────┬──────────────────┬─────────────────────┬─────────┘
  Web app        Browser extension    Caption bridges        Desktop app
 (GitHub Pages)  (Meet / Zoom web /   (Zoom 3rd-party CC,    (virtual camera +
                  Teams web)           Teams CART)            virtual mic)
```

The engine lives in `packages/engine` and is imported by every app, so recognition improvements reach all surfaces at once.

## Current state

| Area | Status |
| --- | --- |
| Browser app, on-device inference | Done (`apps/web`, ONNX Runtime Web) |
| Sentence composer, hold-to-type stabiliser | Done (`packages/engine`) |
| Text-to-speech, captions view for screen sharing | Done |
| Punctuation data pipeline | Done; model not yet trained (needs a GPU) |
| Per-class evaluation | Done; current model mean mAP50 0.61, weakest I, Q, Y, P, N, S, T |
| Monorepo layout, CI, GitHub Pages deploy | Done (Phase 0) |
| Hand-landmark recognition | Not started |
| J/Z motion, spelling correction | Not started |
| Browser extension, caption bridges, desktop app | Not started |

## Phases

| Phase | Work | Done when |
| --- | --- | --- |
| 0. Foundation | Monorepo layout, shared engine package, CI, Pages deploy | Public link works; CI green |
| 1. New engine | MediaPipe hand landmarks + small classifier; short-sequence model for J/Z; punctuation gestures; lexicon-based spelling correction | ≥95% letter accuracy on held-out signers; <100 ms per frame in browser |
| 2. Browser extension | Intercept the meeting's camera to burn captions into your video; optional text-to-speech as your microphone; settings popup | Full meeting works in Meet, Zoom web and Teams web with nobody else installing anything |
| 3. Caption bridges | Zoom third-party closed captions (host-provided API token); Teams CART captions | Captions appear natively for all participants |
| 4. Desktop app | Tauri app with virtual camera and microphone for native Zoom/Teams clients | Works on Windows and macOS native clients |
| 5. Beyond letters | Common whole-word signs, autocomplete, phrase shortcuts | Measurably faster than letter-only typing |

## Why landmarks (Phase 1)

The current model detects signs from raw pixels, so it is sensitive to background, lighting and skin tone, and runs at roughly 2–3 frames per second in the browser. Classifying the 21 hand keypoints from MediaPipe instead is far more robust, under 1 MB, and ~10–30 ms per frame. Public landmark datasets (e.g. Google's ASL fingerspelling data from Kaggle, ASL Citizen) give many signers to train on. The existing YOLO pipeline stays as a fallback.

## Principles

- **Deaf signers test early.** Usability testing with Deaf ASL users from Phase 1 onwards.
- **Honest scope.** Fingerspelling is a small part of ASL, and ASL is not English word-for-word. Present the product as fingerspelling-to-text until whole-word signs work.
- **Privacy.** On-device processing only; no recording; no analytics on video.
- **Accessible tool.** Large text, high contrast, keyboard control, adjustable hold time and sensitivity.
- **Measure what matters.** Letter accuracy across different signers, characters per minute, end-to-end latency; a fixed test set of signers never used in training.

## Before relying on a meeting integration

- Verify each platform's current integration terms and APIs (Zoom captions API, Teams CART, Meet add-ons) at build time; they change.
- Camera interception in the extension must be opt-in per site and clearly indicated.
