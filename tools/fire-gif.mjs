/**
 * Records the Flamescapes fire on the Duplex example as an animated GIF:
 * the 3D view at every fire step, from the running app, using only the
 * examples shipped with the site. No encoder to install: the PNG decoding,
 * palette (median cut) and GIF/LZW encoding are below.
 *
 *   node tools/fire-gif.mjs [--url http://localhost:5173/] [--out fire.gif]
 *        [--width 640] [--horizon 120] [--smoke-at 60] [--delay 120]
 *
 * --smoke-at switches the view to smoke at that step (off by default);
 * --delay is the GIF's time per frame in ms. Like readme-shots.mjs it aims
 * at graph nodes through window.__viewer, which a dev server exposes and a
 * production build exposes with ?debug.
 */
import { chromium } from "playwright-core";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import zlib from "node:zlib";

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const URL_BASE = argOf("--url", "http://localhost:5173/");
const URL_DEBUG = `${URL_BASE}${URL_BASE.includes("?") ? "&" : "?"}debug`;
const OUT = path.resolve(argOf("--out", "duplex-flamescapes.gif"));
const WIDTH = Number(argOf("--width", 640));
const HORIZON = argOf("--horizon", "120");
const SMOKE_AT = Number(argOf("--smoke-at", Infinity));
const DELAY = Number(argOf("--delay", 120));
const CHROME = argOf("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
const ORIGIN = args.includes("--origin") ? new RegExp(argOf("--origin", ""), "i") : null;
const ZOOM = Number(argOf("--zoom", 0.55));

// ------------------------------------------------------------------ PNG in

/** RGBA pixels of an 8-bit, non-interlaced PNG (what Chrome's screenshots are). */
function decodePng(buf) {
  let pos = 8, width = 0, height = 0, colourType = 0;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString("ascii", pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      if (data[8] !== 8 || data[12] !== 0) throw new Error("Only 8-bit non-interlaced PNGs.");
      colourType = data[9];
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    pos += 12 + len;
  }
  const channels = { 2: 3, 6: 4 }[colourType];
  if (!channels) throw new Error(`PNG colour type ${colourType} is not handled.`);
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = new Uint8Array(width * height * 4);
  const prev = new Uint8Array(stride), cur = new Uint8Array(stride);
  for (let y = 0; y < height; y += 1) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i += 1) {
      const a = i >= channels ? cur[i - channels] : 0, b = prev[i], c = i >= channels ? prev[i - channels] : 0;
      let v = line[i];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      cur[i] = v & 255;
    }
    for (let x = 0; x < width; x += 1) {
      const o = (y * width + x) * 4;
      out[o] = cur[x * channels];
      out[o + 1] = cur[x * channels + 1];
      out[o + 2] = cur[x * channels + 2];
      out[o + 3] = 255;
    }
    prev.set(cur);
  }
  return { width, height, data: out };
}

/** Box-filtered downscale to `width` (aspect kept). */
function downscale(img, width) {
  if (img.width <= width) return img;
  const s = img.width / width;
  const height = Math.round(img.height / s);
  const out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const y0 = Math.floor(y * s), y1 = Math.min(img.height, Math.floor((y + 1) * s));
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.floor(x * s), x1 = Math.min(img.width, Math.floor((x + 1) * s));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < y1; yy += 1) {
        for (let xx = x0; xx < x1; xx += 1) {
          const o = (yy * img.width + xx) * 4;
          r += img.data[o];
          g += img.data[o + 1];
          b += img.data[o + 2];
          n += 1;
        }
      }
      const o = (y * width + x) * 4;
      out[o] = r / n;
      out[o + 1] = g / n;
      out[o + 2] = b / n;
      out[o + 3] = 255;
    }
  }
  return { width, height, data: out };
}

// ------------------------------------------------------------------ palette

/** One palette for all frames: median cut over a 5-bit-per-channel histogram. */
function buildPalette(frames, size = 255) {
  const hist = new Uint32Array(32768);
  for (const f of frames) {
    for (let o = 0; o < f.data.length; o += 4) hist[((f.data[o] >> 3) << 10) | ((f.data[o + 1] >> 3) << 5) | (f.data[o + 2] >> 3)] += 1;
  }
  const bins = [];
  for (let k = 0; k < 32768; k += 1) if (hist[k]) bins.push(k);
  const ch = (k, c) => (k >> (10 - 5 * c)) & 31;
  let boxes = [bins];
  while (boxes.length < size) {
    // Split the box with the widest channel range (weighted by its pixels).
    let best = -1, bestScore = 0, bestC = 0;
    boxes.forEach((box, i) => {
      if (box.length < 2) return;
      for (let c = 0; c < 3; c += 1) {
        let lo = 31, hi = 0, count = 0;
        for (const k of box) {
          lo = Math.min(lo, ch(k, c));
          hi = Math.max(hi, ch(k, c));
          count += hist[k];
        }
        const score = (hi - lo) * Math.sqrt(count);
        if (score > bestScore) [best, bestScore, bestC] = [i, score, c];
      }
    });
    if (best < 0) break;
    const box = boxes[best].sort((a, b) => ch(a, bestC) - ch(b, bestC));
    let total = 0;
    for (const k of box) total += hist[k];
    let acc = 0, cut = 1;
    for (let i = 0; i < box.length - 1; i += 1) {
      acc += hist[box[i]];
      if (acc >= total / 2) {
        cut = i + 1;
        break;
      }
      cut = i + 1;
    }
    boxes.splice(best, 1, box.slice(0, cut), box.slice(cut));
  }
  const palette = boxes.map((box) => {
    let r = 0, g = 0, b = 0, n = 0;
    for (const k of box) {
      const w = hist[k];
      r += (ch(k, 0) * 8 + 4) * w;
      g += (ch(k, 1) * 8 + 4) * w;
      b += (ch(k, 2) * 8 + 4) * w;
      n += w;
    }
    return [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
  });
  // Nearest palette entry for every 15-bit colour.
  const lut = new Uint8Array(32768);
  for (let k = 0; k < 32768; k += 1) {
    const r = ch(k, 0) * 8 + 4, g = ch(k, 1) * 8 + 4, b = ch(k, 2) * 8 + 4;
    let bi = 0, bd = Infinity;
    palette.forEach(([pr, pg, pb], i) => {
      const d = 2 * (r - pr) ** 2 + 4 * (g - pg) ** 2 + 3 * (b - pb) ** 2;
      if (d < bd) [bi, bd] = [i, d];
    });
    lut[k] = bi;
  }
  return { palette, lut };
}

// ------------------------------------------------------------------ GIF out

/** GIF LZW of one frame's indices, as data sub-blocks. */
function lzw(indices, minCodeSize) {
  const out = [];
  let bitBuf = 0, bitLen = 0;
  const clear = 1 << minCodeSize, end = clear + 1;
  let codeSize = minCodeSize + 1, next = end + 1;
  let dict = new Map();
  const emit = (code) => {
    bitBuf |= code << bitLen;
    bitLen += codeSize;
    while (bitLen >= 8) {
      out.push(bitBuf & 255);
      bitBuf >>>= 8;
      bitLen -= 8;
    }
  };
  emit(clear);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i += 1) {
    const k = indices[i];
    const key = prefix * 256 + k;
    const found = dict.get(key);
    if (found !== undefined) {
      prefix = found;
      continue;
    }
    emit(prefix);
    if (next < 4096) {
      dict.set(key, next);
      if (next === 1 << codeSize && codeSize < 12) codeSize += 1;
      next += 1;
    } else {
      emit(clear);
      dict = new Map();
      codeSize = minCodeSize + 1;
      next = end + 1;
    }
    prefix = k;
  }
  emit(prefix);
  emit(end);
  if (bitLen > 0) out.push(bitBuf & 255);
  const blocks = [];
  for (let i = 0; i < out.length; i += 255) {
    const chunk = out.slice(i, i + 255);
    blocks.push(chunk.length, ...chunk);
  }
  blocks.push(0);
  return Buffer.from(blocks);
}

/** An animated GIF: one global palette, unchanged pixels transparent after the first frame. */
function encodeGif(frames, delayMs) {
  const { width, height } = frames[0];
  const { palette, lut } = buildPalette(frames);
  const TRANSPARENT = 255;
  // Header and per-frame bytes, then each frame's image data as its own buffer.
  const parts = [];
  let bytes = [];
  const flush = () => {
    parts.push(Buffer.from(bytes));
    bytes = [];
  };
  const u16 = (v) => bytes.push(v & 255, (v >> 8) & 255);
  bytes.push(...Buffer.from("GIF89a"));
  u16(width);
  u16(height);
  bytes.push(0xf7, 0, 0); // global table of 256, 8 bits a colour
  for (let i = 0; i < 256; i += 1) bytes.push(...(palette[i] || [0, 0, 0]));
  bytes.push(0x21, 0xff, 11, ...Buffer.from("NETSCAPE2.0"), 3, 1, 0, 0, 0); // loop for ever
  let previous = null;
  frames.forEach((f, n) => {
    const indices = new Uint8Array(width * height);
    for (let p = 0, o = 0; p < indices.length; p += 1, o += 4) {
      indices[p] = lut[((f.data[o] >> 3) << 10) | ((f.data[o + 1] >> 3) << 5) | (f.data[o + 2] >> 3)];
    }
    const shown = indices.slice();
    if (previous) for (let p = 0; p < indices.length; p += 1) if (indices[p] === previous[p]) indices[p] = TRANSPARENT;
    previous = shown;
    const last = n === frames.length - 1;
    const delay = Math.round((last ? 2000 : delayMs) / 10);
    // Graphic control: keep the frame (disposal 1), transparency after the first.
    bytes.push(0x21, 0xf9, 4, (1 << 2) | (n > 0 ? 1 : 0), delay & 255, delay >> 8, TRANSPARENT, 0);
    bytes.push(0x2c);
    u16(0);
    u16(0);
    u16(width);
    u16(height);
    bytes.push(0, 8);
    flush();
    parts.push(lzw(indices, 8));
  });
  bytes.push(0x3b);
  flush();
  return Buffer.concat(parts);
}

// ------------------------------------------------------------------ the app

async function pickAt(page, kind, fx, fy) {
  const picker = page.locator(".picker", { hasText: new RegExp(kind, "i") });
  await picker.getByRole("button").first().click();
  await page.waitForTimeout(200);
  const aim = await page.evaluate(([tx, ty]) => {
    const v = window.__viewer;
    const nodes = v?._graphNodes, cam = v?.world?.camera?.three, el = v?._container;
    if (!nodes || !cam || !el) return null;
    const r = el.getBoundingClientRect();
    const V = cam.matrixWorldInverse.elements, P = cam.projectionMatrix.elements;
    const mul = (m, x, y, z, w) => [0, 1, 2, 3].map((i) => m[i] * x + m[4 + i] * y + m[8 + i] * z + m[12 + i] * w);
    let best = null;
    for (let i = 0; i < nodes.length; i += 3) {
      const [cx, cy, , cw] = mul(P, ...mul(V, nodes[i], nodes[i + 1], nodes[i + 2], 1));
      if (cw <= 0) continue;
      const sx = ((cx / cw + 1) / 2) * r.width, sy = ((1 - cy / cw) / 2) * r.height;
      if (sx < 0 || sy < 0 || sx > r.width || sy > r.height) continue;
      const d = (sx - r.width * tx) ** 2 + (sy - r.height * ty) ** 2;
      if (!best || d < best.d) best = { d, x: r.left + sx, y: r.top + sy };
    }
    return best;
  }, [fx, fy]);
  if (!aim) throw new Error("No graph node to aim at: open the app on a dev server or with ?debug.");
  await page.mouse.click(aim.x, aim.y);
  await page.waitForTimeout(800);
}

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: !args.includes("--headed"),
  args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, colorScheme: "light" });
const page = await context.newPage();
page.on("pageerror", (e) => console.log(`  page error: ${e.message.slice(0, 160)}`));
await page.goto(URL_DEBUG, { waitUntil: "load" });
await page.waitForFunction(() => document.querySelector(".welcome .example"), null, { timeout: 120000 });
await page.waitForTimeout(1500);

console.log("Opening the Duplex example and building its graph");
await page.locator(".welcome").getByRole("button", { name: /^Duplex apartment/ }).click();
await page.waitForFunction(
  () => [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled),
  null,
  { timeout: 180000 },
);
await page.waitForTimeout(1500);
await page.getByRole("tab", { name: /^Graph/ }).click();
await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
await page.waitForFunction(() => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === "route", null, { timeout: 180000 });
await page.waitForTimeout(800);
await page.getByRole("button", { name: "Fit", exact: true }).click().catch(() => page.evaluate(() => window.__viewer?.setStandardView("fit")));
await page.waitForTimeout(1300);

// A route, so the playback shows it rerouting around the fire.
await page.getByRole("tab", { name: /^Route/ }).click();
for (const [fx, fy] of [[0.42, 0.62], [0.55, 0.62], [0.6, 0.45]]) {
  await pickAt(page, "Start", fx, fy);
  const nearest = page.getByRole("button", { name: "Use the nearest exit door" });
  if (await nearest.isVisible().catch(() => false)) await nearest.click();
  await page.getByRole("button", { name: /Find egress route/ }).click();
  await page.waitForFunction(() => /route found|no route/i.test(document.querySelector(".result__title")?.textContent || ""), null, { timeout: 60000 });
  if (/route found/i.test(await page.evaluate(() => document.querySelector(".result__title")?.textContent || ""))) break;
}
const box = await page.locator(".viewport__canvas").boundingBox();
await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.55);
await page.mouse.wheel(0, -120);
await page.waitForTimeout(1000);

console.log("Starting the Flamescapes fire at the most likely ignition point");
await page.getByRole("tab", { name: /^Simulate/ }).click();
await page.getByText("Flamescapes combustion solver (optional)", { exact: true }).click();
await page.getByText("Flamescapes: fire, smoke and embers", { exact: true }).click();
await page.getByText("Likely ignition points", { exact: true }).click();
await page.getByRole("button", { name: /^Find in this model$/ }).click();
await page.getByRole("button", { name: /^Start here$/ }).first().waitFor({ timeout: 60000 });
const candidates = await page.locator(".ignition-list__item").allInnerTexts();
candidates.forEach((text, i) => console.log(`  ${i + 1}. ${text.replace(/\s+/g, " ").replace(/ Start here$/, "")}`));
const chosen = ORIGIN ? Math.max(0, candidates.findIndex((text) => ORIGIN.test(text))) : 0;
console.log(`  starting at ${chosen + 1}`);
await page.locator(".ignition-list__item").nth(chosen).getByRole("button", { name: /^Start here$/ }).click();
await page.waitForTimeout(500);

// Closer, looking at the fire origin from the same direction.
await page.evaluate((factor) => {
  const v = window.__viewer;
  const controls = v?.world?.camera?.controls;
  const fire = v?.markers?.fire?.position;
  if (!controls?.setLookAt || !fire) return;
  const eye = controls.getPosition(new fire.constructor());
  const target = controls.getTarget(new fire.constructor());
  const d = eye.sub(target).multiplyScalar(factor);
  controls.setLookAt(fire.x + d.x, fire.y + d.y, fire.z + d.z, fire.x, fire.y, fire.z, false);
}, ZOOM);
await page.waitForTimeout(800);
await page.getByText("Flamescapes settings", { exact: true }).click();
await page.getByLabel("Simulated time (s)").fill(HORIZON);
await page.getByLabel("Simulated time (s)").blur();
await page.getByRole("button", { name: /^Start fire$/ }).scrollIntoViewIfNeeded();
// Toasts report timings over the model; the playback bar says what happens.
await page.addStyleTag({ content: ".toast { display: none !important; }" });
await page.getByRole("button", { name: /^Start fire$/ }).click();

// One capture per fire step, as the playback reaches it; decoded afterwards,
// so the capture keeps up with the playback.
const stepNow = () => page.evaluate(() => Number((/Fire step\s*(\d+)/.exec(document.body.innerText) || [])[1] ?? -1));
const running = () => page.evaluate(() => /fire step \d+/i.test(document.body.innerText) && [...document.querySelectorAll("button")].some((b) => /^Stop$/.test(b.textContent.trim())));
await page.waitForFunction(() => /fire step \d+/i.test(document.body.innerText), null, { timeout: 120000 });
const view = page.locator(".viewport");
const shots = [];
let last = -1, smoke = false;
const t0 = Date.now();
while (Date.now() - t0 < 900000) {
  const step = await stepNow();
  if (step > last) {
    if (!smoke && step >= SMOKE_AT) {
      await page.getByText("Smoke", { exact: true }).click().catch(() => {});
      smoke = true;
    }
    shots.push(await view.screenshot({ type: "png" }));
    if (step > last + 1) console.log(`  steps ${last + 1} to ${step - 1} played before they could be captured`);
    last = step;
    if (shots.length % 10 === 0) console.log(`  ${shots.length} frames (step ${step})`);
  }
  if (!(await running())) break;
  await page.waitForTimeout(40);
}
// The last step, after the run ends.
await page.waitForTimeout(800);
shots.push(await view.screenshot({ type: "png" }));
await browser.close();
const frames = shots.map((png) => downscale(decodePng(png), WIDTH));

const gif = encodeGif(frames, DELAY);
await writeFile(OUT, gif);
console.log(`wrote ${OUT}: ${frames.length} frames, ${frames[0].width}x${frames[0].height}, ${(gif.length / 1e6).toFixed(2)} MB`);
