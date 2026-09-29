import { decode } from "./postprocess.js";

export const CLASS_NAMES = [
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  "backspace",
];

const SIZE = 640;

export class Detector {
  constructor() {
    this.session = null;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = SIZE;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  async load(modelUrl) {
    ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url).href;
    ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation
    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  #preprocess(video) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const scale = Math.min(SIZE / vw, SIZE / vh);
    const w = Math.round(vw * scale);
    const h = Math.round(vh * scale);
    const padX = Math.floor((SIZE - w) / 2);
    const padY = Math.floor((SIZE - h) / 2);

    this.ctx.fillStyle = "rgb(114,114,114)";
    this.ctx.fillRect(0, 0, SIZE, SIZE);
    this.ctx.drawImage(video, padX, padY, w, h);
    const { data } = this.ctx.getImageData(0, 0, SIZE, SIZE);

    const plane = SIZE * SIZE;
    const input = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      input[i] = data[i * 4] / 255;
      input[plane + i] = data[i * 4 + 1] / 255;
      input[2 * plane + i] = data[i * 4 + 2] / 255;
    }
    return { input, letterbox: { scale, padX, padY } };
  }

  // Returns detections in source-video pixel coordinates, best first.
  async detect(video, scoreThreshold) {
    const { input, letterbox } = this.#preprocess(video);
    const tensor = new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
    const outputs = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const out = outputs[this.session.outputNames[0]];
    const [, channels, numBoxes] = out.dims;
    return decode(out.data, numBoxes, channels - 4, letterbox, scoreThreshold).map((d) => ({
      ...d,
      label: CLASS_NAMES[d.classId],
    }));
  }
}
