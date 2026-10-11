import { Detector } from "./detector.js";
import { Composer } from "./sentence.js";
import { Stabilizer } from "./stabilizer.js";

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");

const detector = new Detector();
const composer = new Composer();
const stabilizer = new Stabilizer();
let confidence = 0.6;
let running = false;

const PUNCTUATION = new Set([".", "?", "!", ","]);

function setStatus(msg) {
  $("status").textContent = msg;
}

function render() {
  const out = $("output");
  out.textContent = composer.text;
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  out.append(cursor);
}

function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

function apply(action) {
  const before = composer.text;
  if (action === "space") composer.addSpace();
  else if (action === "newline") composer.addNewline();
  else if (action === "backspace") composer.backspace();
  else if (action === "clear") composer.clear();
  else if (PUNCTUATION.has(action)) composer.addPunctuation(action);
  else composer.addLetter(action);

  if (composer.text === before) return;
  render();
  if (PUNCTUATION.has(action) && composer.endsSentence() && $("autoSpeak").checked) {
    const sentences = composer.text.trim().split(/(?<=[.?!])\s+|\n/);
    speak(sentences[sentences.length - 1]);
  }
}

function drawDetections(dets) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (overlay.width !== w) { overlay.width = w; overlay.height = h; }
  octx.clearRect(0, 0, w, h);
  octx.lineWidth = Math.max(2, w / 300);
  for (const d of dets) {
    const color = d.label === "backspace" ? "#ffb020" : "#3ddc84";
    octx.strokeStyle = color;
    octx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
  }
}

function showHold(label, progress) {
  const box = $("hold");
  box.hidden = !label;
  if (!label) return;
  $("holdLabel").textContent = label;
  $("holdBar").style.width = `${Math.round(progress * 100)}%`;
}

async function loop() {
  if (!running) return;
  const t0 = performance.now();
  try {
    const dets = await detector.detect(video, confidence);
    drawDetections(dets);
    const top = dets[0] ?? null;
    const committed = stabilizer.update(top ? top.label : null, performance.now());
    showHold(top?.label, stabilizer.progress);
    if (committed) apply(committed);
    setStatus(top ? `Seeing ${top.label} (${Math.round(top.score * 100)}%) · ${Math.round(performance.now() - t0)} ms` : "No hand detected");
  } catch (err) {
    console.error(err);
    setStatus(`Detection error: ${err.message}`);
  }
  requestAnimationFrame(loop);
}

async function start() {
  const btn = $("startBtn");
  btn.disabled = true;
  try {
    setStatus("Loading model…");
    if (!detector.session) await detector.load(new URL("../model/asl.onnx", import.meta.url).href);
    setStatus("Requesting camera…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    running = true;
    btn.textContent = "Stop camera";
    btn.disabled = false;
    loop();
  } catch (err) {
    console.error(err);
    const denied = err.name === "NotAllowedError";
    setStatus(denied ? "Camera permission denied. Allow camera access and try again." : `Could not start: ${err.message}`);
    btn.disabled = false;
  }
}

function stop() {
  running = false;
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  showHold(null);
  stabilizer.reset();
  $("startBtn").textContent = "Start camera";
  setStatus("Camera stopped.");
}

$("startBtn").addEventListener("click", () => (running ? stop() : start()));

document.querySelectorAll("[data-act]").forEach((b) =>
  b.addEventListener("click", () => apply(b.dataset.act)));

$("confidence").addEventListener("input", (e) => {
  confidence = Number(e.target.value);
  $("confidenceOut").textContent = confidence.toFixed(2);
});
$("holdTime").addEventListener("input", (e) => {
  stabilizer.holdMs = Number(e.target.value);
  $("holdTimeOut").textContent = `${(stabilizer.holdMs / 1000).toFixed(1)}s`;
});

$("speakBtn").addEventListener("click", () => speak(composer.text));
$("copyBtn").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(composer.text);
    setStatus("Copied to clipboard.");
  } catch {
    setStatus("Copy failed. Select the text and copy manually.");
  }
});

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const map = { " ": "space", Enter: "newline", Backspace: "backspace", Escape: "clear" };
  const act = map[e.key] ?? (PUNCTUATION.has(e.key) ? e.key : null);
  if (act && document.activeElement?.tagName !== "BUTTON") {
    e.preventDefault();
    apply(act);
  }
});

render();
