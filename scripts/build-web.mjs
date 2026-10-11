// Assemble the deployable web app: apps/web plus the shared engine, into dist/web.
// The page imports "@sign-flow/engine", which its import map points at ./engine/.
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "dist", "web");

rmSync(out, { recursive: true, force: true });
cpSync(join(root, "apps", "web"), out, { recursive: true });
cpSync(join(root, "packages", "engine", "src"), join(out, "engine"), { recursive: true });

for (const required of ["index.html", "engine/index.js", "model/asl.onnx", "model/classes.json", "vendor/ort.wasm.min.js"]) {
  if (!existsSync(join(out, required))) {
    console.error(`Build incomplete: missing ${required}`);
    process.exit(1);
  }
}
console.log(`Built ${out}`);
