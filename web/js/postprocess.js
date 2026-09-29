// YOLOv8 output decoding: raw [1, 4 + nc, N] tensor -> detections after NMS.
// Pure functions, unit tested in Node.

export function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

export function nms(detections, iouThreshold = 0.45) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const d of sorted) {
    if (kept.every((k) => iou(k, d) < iouThreshold)) kept.push(d);
  }
  return kept;
}

// data: Float32Array laid out as [4 + numClasses, numBoxes] (channel-major).
// letterbox: { scale, padX, padY } used during preprocessing.
export function decode(data, numBoxes, numClasses, letterbox, scoreThreshold) {
  const { scale, padX, padY } = letterbox;
  const out = [];
  for (let i = 0; i < numBoxes; i++) {
    let best = -1;
    let bestScore = scoreThreshold;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * numBoxes + i];
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (best < 0) continue;
    const cx = data[i];
    const cy = data[numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];
    out.push({
      x1: (cx - w / 2 - padX) / scale,
      y1: (cy - h / 2 - padY) / scale,
      x2: (cx + w / 2 - padX) / scale,
      y2: (cy + h / 2 - padY) / scale,
      score: bestScore,
      classId: best,
    });
  }
  return nms(out);
}
