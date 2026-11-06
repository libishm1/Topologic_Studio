/**
 * An animated GIF from PNG frames already captured (for example by a
 * Playwright run of the app): every `--every`-th file of a folder between
 * `--from` and `--to`, cropped to `--crop x,y,w,h` (pixels of the PNG),
 * scaled to `--width`, `--delay` ms a frame. No encoder to install
 * (tools/gif.mjs).
 *
 *   node tools/frames-gif.mjs --dir frames --from 6 --to 65 --every 2
 *        --crop 0,56,1600,724 --width 720 --delay 140 --out fire.gif
 *
 * Files are taken in name order; --from and --to are positions in that order.
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { decodePng, downscale, encodeGif } from "./gif.mjs";

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const dir = argOf("--dir", ".");
const files = readdirSync(dir).filter((f) => /\.png$/i.test(f)).sort();
const from = Number(argOf("--from", 0));
const to = Number(argOf("--to", files.length - 1));
const every = Math.max(1, Number(argOf("--every", 1)));
const width = Number(argOf("--width", 720));
const delay = Number(argOf("--delay", 140));
const crop = argOf("--crop", null)?.split(",").map(Number);
const out = path.resolve(argOf("--out", "frames.gif"));

/** A rectangle of an RGBA image. */
function cropImage(img, [x, y, w, h]) {
  w = Math.min(w, img.width - x);
  h = Math.min(h, img.height - y);
  const data = new Uint8Array(w * h * 4);
  for (let r = 0; r < h; r += 1) data.set(img.data.subarray(((y + r) * img.width + x) * 4, ((y + r) * img.width + x + w) * 4), r * w * 4);
  return { width: w, height: h, data };
}

const picked = files.slice(from, to + 1).filter((_, i) => i % every === 0);
const frames = picked.map((f) => {
  const img = decodePng(readFileSync(path.join(dir, f)));
  return downscale(crop ? cropImage(img, crop) : img, width);
});
const gif = encodeGif(frames, delay);
writeFileSync(out, gif);
console.log(`${out}: ${frames.length} frames, ${frames[0].width} x ${frames[0].height}, ${(gif.length / 1e6).toFixed(2)} MB`);
