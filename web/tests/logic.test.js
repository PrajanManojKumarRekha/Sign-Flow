import test from "node:test";
import assert from "node:assert/strict";
import { Composer } from "../js/sentence.js";
import { Stabilizer } from "../js/stabilizer.js";
import { decode, iou, nms } from "../js/postprocess.js";

const type = (c, s) => [...s].forEach((ch) => c.addLetter(ch));

test("capitalises the first letter and the start of each sentence", () => {
  const c = new Composer();
  type(c, "hi");
  c.addPunctuation(".");
  c.addSpace();
  type(c, "ok");
  assert.equal(c.text, "Hi. Ok");
});

test("punctuation removes preceding space and is not doubled", () => {
  const c = new Composer();
  type(c, "yes");
  c.addSpace();
  c.addPunctuation("?");
  c.addPunctuation("!");
  assert.equal(c.text, "Yes?");
  assert.ok(c.endsSentence());
});

test("no leading or repeated spaces, newline starts a new sentence", () => {
  const c = new Composer();
  c.addSpace();
  c.addPunctuation(".");
  assert.equal(c.text, "");
  type(c, "a");
  c.addSpace();
  c.addSpace();
  c.addNewline();
  type(c, "b");
  assert.equal(c.text, "A\nB");
});

test("backspace recomputes capitalisation", () => {
  const c = new Composer();
  type(c, "hi");
  c.addPunctuation(".");
  c.backspace();
  type(c, "!");
  assert.equal(c.text, "Hi!");
  assert.equal(c.currentSentence(), "Hi!");
});

test("stabilizer needs a steady hold and ignores flicker", () => {
  const s = new Stabilizer({ holdMs: 500, repeatMs: 1500, releaseMs: 300 });
  assert.equal(s.update("A", 0), null);
  assert.equal(s.update("B", 200), null); // switched, restart
  assert.equal(s.update("B", 600), null);
  assert.equal(s.update("B", 700), "B");
});

test("same sign repeats only after release or long hold", () => {
  const s = new Stabilizer({ holdMs: 500, repeatMs: 1500, releaseMs: 300 });
  s.update("L", 0);
  assert.equal(s.update("L", 500), "L");
  assert.equal(s.update("L", 1200), null); // still holding
  s.update(null, 1300);
  s.update(null, 1700); // released
  s.update("L", 1800);
  assert.equal(s.update("L", 2300), "L");
});

test("iou and nms", () => {
  const a = { x1: 0, y1: 0, x2: 10, y2: 10, score: 0.9 };
  const b = { x1: 1, y1: 1, x2: 11, y2: 11, score: 0.8 };
  const c = { x1: 50, y1: 50, x2: 60, y2: 60, score: 0.7 };
  assert.equal(iou(a, a), 1);
  assert.deepEqual(nms([b, c, a]).map((d) => d.score), [0.9, 0.7]);
});

test("decode maps letterboxed boxes back to source pixels", () => {
  // 1 box, 2 classes; channel-major layout [cx, cy, w, h, c0, c1]
  const data = Float32Array.from([320, 320, 100, 100, 0.1, 0.8]);
  const [d] = decode(data, 1, 2, { scale: 0.5, padX: 0, padY: 140 }, 0.5);
  assert.equal(d.classId, 1);
  assert.deepEqual([d.x1, d.y1, d.x2, d.y2], [540, 260, 740, 460]);
  assert.equal(decode(data, 1, 2, { scale: 1, padX: 0, padY: 0 }, 0.9).length, 0);
});
