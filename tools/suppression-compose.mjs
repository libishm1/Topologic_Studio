/**
 * GIFs and a video from tools/suppression-media.mjs's captures: every frame
 * is drawn on a canvas in Chrome (the 3D views, a label and the heat
 * release per panel, a heat release chart), read back as PNG and encoded
 * with tools/gif.mjs; the video is the same frames played on the canvas and
 * recorded by Chrome's MediaRecorder as H.264 MP4. Nothing to install.
 *
 *   node tools/suppression-compose.mjs --in DIR --out OUTDIR [--analysis analysis.json] [--width 900]
 *
 * Writes OUTDIR/suppression-workflow.gif (the Duplex with wetting: the heads
 * laid out, then the fire and the heads opening), suppression-compare.gif
 * (the same fire with no sprinklers, Evans's cap, the spray and wetting),
 * suppression-classes.gif (class A furniture and a heptane item, no
 * sprinklers against wetting), suppression-office.gif (the NIBS Office,
 * when captured) and suppression-analysis.mp4 (all of them with title and
 * results cards).
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { chromium } from "playwright-core";
import { decodePng, downscale, encodeGif } from "./gif.mjs";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const IN = flag("--in", null), OUT = flag("--out", null);
if (!IN || !OUT) throw new Error("--in DIR and --out OUTDIR are required");
const WIDTH = Number(flag("--width", 900));
const ANALYSIS = flag("--analysis", null);
const CHROME = flag("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
await mkdir(OUT, { recursive: true });

const MODE_LABEL = { none: "No sprinklers", evans: "Evans's cap", spray: "Cap and spray", wetting: "Wetting (default)", systems: "Clean agent and sprinklers" };
const MODE_COLOUR = { none: "#b91c1c", evans: "#d97706", spray: "#7c3aed", wetting: "#0284c7", systems: "#0d9488" };

/**
 * A captured run, by step number (its frames are NNN.png): entry k is step
 * k's frame and readings, or the last captured step before it when the
 * capture missed k (a busy machine), so two runs' entries k are the same
 * moment. `readings` are the captured steps alone, for the chart.
 */
async function run(caseName, mode) {
  const dir = path.join(IN, caseName, mode);
  if (!existsSync(dir) || !existsSync(path.join(IN, caseName, `${mode}.json`))) return null;
  const meta = JSON.parse(await readFile(path.join(IN, caseName, `${mode}.json`), "utf8"));
  const png = (k) => path.join(dir, `${String(k).padStart(3, "0")}.png`);
  const byStep = new Map(meta.steps.map((s) => [s.step, s]));
  const have = (k) => byStep.has(k) && existsSync(png(k));
  const last = Math.max(...byStep.keys());
  // From step 1: step 0 is on screen while the solver builds its grid,
  // before its first frame, and can still show the last run's heads.
  let at = [...Array(last + 1).keys()].find((k) => k >= FIRST_STEP && have(k));
  if (at === undefined) return null;
  const files = [], steps = [];
  for (let k = FIRST_STEP; k <= last; k += 1) {
    if (have(k)) at = k;
    files.push(png(at));
    steps.push(byStep.get(at));
  }
  const readings = [...byStep.values()].sort((a, b) => a.step - b.step);
  return { mode, files, steps, readings };
}
const FIRST_STEP = 1;
const b64 = async (file) => `data:image/png;base64,${(await readFile(file)).toString("base64")}`;

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
await page.setContent(`<canvas id="c"></canvas>`);
// The drawing code, in the page.
await page.evaluate(() => {
  const c = document.getElementById("c");
  const g = c.getContext("2d");
  const load = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
  const font = (px, weight = 600) => `${weight} ${px}px "Segoe UI", Arial, sans-serif`;
  const kw = (x) => (x == null ? "-" : x >= 1000 ? `${(x / 1000).toFixed(1)} MW` : `${Math.round(x)} kW`);
  window.compose = async ({ w, h, panels, title, subtitle, chart }) => {
    c.width = w;
    c.height = h;
    g.fillStyle = "#f8fafc";
    g.fillRect(0, 0, w, h);
    const head = title ? 54 : 0, foot = chart ? 150 : 0;
    if (title) {
      g.fillStyle = "#0f172a";
      g.font = font(24);
      g.fillText(title, 20, 34);
      if (subtitle) {
        g.fillStyle = "#475569";
        g.font = font(15, 400);
        g.fillText(subtitle, w - 20 - g.measureText(subtitle).width, 34);
      }
    }
    // Four panels two by two; up to three in one row.
    const cols = panels.length === 4 ? 2 : panels.length, rows = Math.ceil(panels.length / cols);
    const pw = w / cols, ph = (h - head - foot) / rows;
    for (let k = 0; k < panels.length; k += 1) {
      const p = panels[k];
      const x = (k % cols) * pw, y = head + Math.floor(k / cols) * ph;
      const img = await load(p.src);
      const s = Math.max(pw / img.width, ph / img.height);
      const dw = img.width * s, dh = img.height * s;
      g.save();
      g.beginPath();
      g.rect(x, y, pw, ph);
      g.clip();
      g.drawImage(img, x + (pw - dw) / 2, y + (ph - dh) / 2, dw, dh);
      g.restore();
      g.strokeStyle = "#e2e8f0";
      g.lineWidth = 2;
      g.strokeRect(x + 1, y + 1, pw - 2, ph - 2);
      // The panel's label and readings.
      g.fillStyle = "rgba(255,255,255,0.88)";
      g.fillRect(x + 10, y + 10, Math.min(pw - 20, 330), 58);
      g.fillStyle = p.colour || "#0f172a";
      g.fillRect(x + 10, y + 10, 6, 58);
      g.fillStyle = "#0f172a";
      g.font = font(18);
      g.fillText(p.label, x + 24, y + 33);
      g.font = font(14, 400);
      g.fillStyle = "#334155";
      if (p.hrr != null) g.fillText(`${p.time != null ? `${Math.round(p.time)} s   ` : ""}heat ${kw(p.hrr)}${p.open != null ? `   ${p.open} heads open` : ""}`, x + 24, y + 56);
    }
    if (chart) {
      const y0 = h - foot + 12, x0 = 60, cw = w - 90, ch = foot - 40;
      g.fillStyle = "#ffffff";
      g.fillRect(0, h - foot, w, foot);
      g.strokeStyle = "#cbd5e1";
      g.lineWidth = 1;
      g.strokeRect(x0, y0, cw, ch);
      const max = Math.max(1, ...chart.series.flatMap((s) => s.points.map((p) => p[1])));
      const tmax = chart.tmax;
      g.font = font(12, 400);
      g.fillStyle = "#475569";
      g.fillText(kw(max), 6, y0 + 10);
      g.fillText("0", 40, y0 + ch);
      g.fillText(`${tmax} s`, x0 + cw - 30, y0 + ch + 16);
      g.fillText("heat release", x0 + 6, y0 + 14);
      for (const s of chart.series) {
        g.strokeStyle = s.colour;
        g.lineWidth = 2.5;
        g.beginPath();
        s.points.forEach(([t, q], i) => {
          const px = x0 + (t / tmax) * cw, py = y0 + ch - (q / max) * ch;
          if (i) g.lineTo(px, py); else g.moveTo(px, py);
        });
        g.stroke();
      }
      if (chart.now != null) {
        g.strokeStyle = "#0f172a";
        g.setLineDash([4, 4]);
        g.beginPath();
        g.moveTo(x0 + (chart.now / tmax) * cw, y0);
        g.lineTo(x0 + (chart.now / tmax) * cw, y0 + ch);
        g.stroke();
        g.setLineDash([]);
      }
    }
    return c.toDataURL("image/png");
  };
  /** A card: a title, lines of text, and optional bars [{label, value, colour}] (percent change). */
  window.card = ({ w, h, title, lines = [], bars = null, note = null }) => {
    c.width = w;
    c.height = h;
    g.fillStyle = "#0f172a";
    g.fillRect(0, 0, w, h);
    g.fillStyle = "#f8fafc";
    g.font = font(34);
    g.fillText(title, 50, 80);
    g.font = font(19, 400);
    g.fillStyle = "#cbd5e1";
    lines.forEach((t, i) => g.fillText(t, 50, 130 + i * 32));
    if (bars) {
      // Labels in their own column; the bars diverge from zero in the middle
      // of the rest, each value written on the other side of zero.
      g.font = font(15, 400);
      const labelW = Math.max(...bars.map((b) => g.measureText(b.label).width));
      const left = 50 + labelW + 30, right = w - 50;
      const top = 130 + lines.length * 32 + 20, rowH = 34, mid = (left + right) / 2, scale = (mid - left) / 100;
      g.strokeStyle = "#475569";
      g.beginPath();
      g.moveTo(mid, top - 10);
      g.lineTo(mid, top + bars.length * rowH);
      g.stroke();
      bars.forEach((b, i) => {
        const y = top + i * rowH;
        g.fillStyle = "#cbd5e1";
        g.font = font(15, 400);
        g.fillText(b.label, 50, y + 18);
        g.fillStyle = b.colour;
        const v = Math.max(-100, Math.min(100, b.value));
        g.fillRect(v < 0 ? mid + v * scale : mid, y + 4, Math.abs(v) * scale, 20);
        g.fillStyle = "#f8fafc";
        const text = `${b.value > 0 ? "+" : ""}${Math.round(b.value)} %`;
        g.fillText(text, v < 0 ? mid + 8 : mid - 8 - g.measureText(text).width, y + 19);
      });
    }
    if (note) {
      g.fillStyle = "#94a3b8";
      g.font = font(15, 400);
      g.fillText(note, 50, h - 30);
    }
    return c.toDataURL("image/png");
  };
  // The video, recorded from the canvas by MediaRecorder, its frames sent
  // one at a time (all at once, a few hundred MB of PNGs, crash the page):
  // each is drawn when the one before has been on screen for its time.
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let rec = null, chunks = [], done = null, until = 0;
  window.recStart = (w, h) => {
    c.width = w;
    c.height = h;
    rec = new MediaRecorder(c.captureStream(30), { mimeType: "video/mp4;codecs=avc1.42E01E", videoBitsPerSecond: 6e6 });
    chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    done = new Promise((res) => (rec.onstop = res));
    rec.start(250);
    until = 0;
  };
  window.recFrame = async (src, ms) => {
    const img = await load(src);
    const wait = until - performance.now();
    if (wait > 0) await sleep(wait);
    g.drawImage(img, 0, 0, c.width, c.height);
    until = performance.now() + ms;
  };
  /** Ends the recording once the last frame has had its time; resolves with base64 MP4. */
  window.recStop = async () => {
    const wait = until - performance.now();
    if (wait > 0) await sleep(wait);
    rec.stop();
    await done;
    const buf = new Uint8Array(await new Blob(chunks, { type: "video/mp4" }).arrayBuffer());
    let s = "";
    for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    return btoa(s);
  };
});

const toPng = (dataUrl) => Buffer.from(dataUrl.split(",")[1], "base64");
const W = 1280, H = 820;
const gifOf = async (name, pngs, delay) => {
  const frames = pngs.map((p) => downscale(decodePng(p), WIDTH));
  const gif = encodeGif(frames, delay);
  await writeFile(path.join(OUT, name), gif);
  console.log(`wrote ${name}: ${frames.length} frames, ${(gif.length / 1e6).toFixed(2)} MB`);
};
const series = (r) => ({ colour: MODE_COLOUR[r.mode], points: r.readings.filter((s) => s.time != null && s.hrrKw != null).map((s) => [s.time, s.hrrKw]) });
const video = [];

// 1. The workflow: the heads laid out, then the fire with wetting.
const wet = await run("duplex-A", "wetting");
if (wet) {
  const frames = [];
  const setup = path.join(IN, "duplex-A", "setup.png");
  if (existsSync(setup)) {
    const f = toPng(await page.evaluate((o) => window.compose(o), { w: W, h: H, title: "Sprinklers on: heads placed per room by NFPA 13", subtitle: "Duplex, Topologic Studio", panels: [{ src: await b64(setup), label: "Heads laid out", colour: "#0284c7" }] }));
    for (let k = 0; k < 8; k += 1) frames.push(f);
  }
  const tmax = Math.max(...wet.steps.map((s) => s.time || 0));
  for (let i = 0; i < wet.files.length; i += 1) {
    const s = wet.steps[Math.min(i, wet.steps.length - 1)];
    frames.push(toPng(await page.evaluate((o) => window.compose(o), {
      w: W, h: H, title: "A fire, the heads opening and spraying (wetting)", subtitle: "Duplex living room, class A",
      panels: [{ src: await b64(wet.files[i]), label: "Wetting", colour: MODE_COLOUR.wetting, time: s.time, hrr: s.hrrKw, open: s.open }],
      chart: { tmax, now: s.time, series: [series(wet)] },
    })));
  }
  await gifOf("suppression-workflow.gif", frames, 110);
  video.push(...frames.map((p) => ({ png: p, ms: 110 })));
}

// 2. The comparison: four modes side by side, step by step.
const modes = [];
for (const m of ["none", "evans", "spray", "wetting"]) {
  const r = await run("duplex-A", m);
  if (r) modes.push(r);
}
if (modes.length > 1) {
  const n = Math.min(...modes.map((r) => r.files.length));
  const tmax = Math.max(...modes.flatMap((r) => r.steps.map((s) => s.time || 0)));
  const frames = [];
  for (let i = 0; i < n; i += 1) {
    frames.push(toPng(await page.evaluate((o) => window.compose(o), {
      w: W, h: H, title: "The same fire, four ways", subtitle: "Duplex living room, class A furniture",
      panels: await Promise.all(modes.map(async (r) => {
        const s = r.steps[Math.min(i, r.steps.length - 1)];
        return { src: await b64(r.files[i]), label: MODE_LABEL[r.mode], colour: MODE_COLOUR[r.mode], time: s.time, hrr: s.hrrKw, open: r.mode === "none" ? null : s.open };
      })),
      chart: { tmax, now: modes[0].steps[Math.min(i, modes[0].steps.length - 1)].time, series: modes.map(series) },
    })));
  }
  await gifOf("suppression-compare.gif", frames, 120);
  video.push(...frames.map((p) => ({ png: p, ms: 120 })));
}

// 3. Fire classes, 4. the NIBS Office: no sprinklers against wetting; and
// 5. a server room: no system, sprinklers, and the systems by room use.
for (const [caseName, gifName, title, subtitle, modeList, labels = {}] of [
  ["duplex-B", "suppression-classes.gif", "A heptane item: no sprinklers against wetting", "Duplex living room, class B first item", ["none", "wetting"]],
  ["office-A", "suppression-office.gif", "An office fire: no sprinklers against wetting", "NIBS Office, class A", ["none", "wetting"]],
  ["office-C", "suppression-systems.gif", "A server room fire: sprinklers, then a clean agent", "NIBS Office server room, class C (electrical equipment)", ["none", "wetting", "systems"], { none: "No system", wetting: "Sprinklers (wetting)" }],
]) {
  const pair = (await Promise.all(modeList.map((m) => run(caseName, m)))).filter(Boolean);
  if (pair.length < 2) continue;
  const n = Math.min(...pair.map((r) => r.files.length));
  const tmax = Math.max(...pair.flatMap((r) => r.steps.map((s) => s.time || 0)));
  const frames = [];
  for (let i = 0; i < n; i += 1) {
    frames.push(toPng(await page.evaluate((o) => window.compose(o), {
      w: W, h: H, title, subtitle,
      panels: await Promise.all(pair.map(async (r) => {
        const s = r.steps[Math.min(i, r.steps.length - 1)];
        return { src: await b64(r.files[i]), label: labels[r.mode] ?? MODE_LABEL[r.mode], colour: MODE_COLOUR[r.mode], time: s.time, hrr: s.hrrKw, open: r.mode === "none" ? null : s.open };
      })),
      chart: { tmax, now: pair[0].steps[Math.min(i, pair[0].steps.length - 1)].time, series: pair.map(series) },
    })));
  }
  await gifOf(gifName, frames, 120);
  video.push({ png: toPng(await page.evaluate((o) => window.card(o), { w: W, h: H, title, lines: [subtitle] })), ms: 2200 });
  video.push(...frames.map((p) => ({ png: p, ms: 120 })));
}

// The video: a title card, the clips, the results and the caveats.
if (video.length) {
  const a = ANALYSIS ? JSON.parse(await readFile(ANALYSIS, "utf8")) : null;
  const cards = [];
  cards.push({ png: toPng(await page.evaluate((o) => window.card(o), { w: W, h: H, title: "Fire suppression in Topologic Studio", lines: ["Sprinklers placed per room by NFPA 13, opened by the fire's heat;", "four ways of letting the water act, compared on the same fires;", "and a clean agent (HFC-227ea) flooding a server room.", "Recorded from the app in the browser (Flamescapes solver)."], note: "libishm1.github.io/Topologic_Studio" })), ms: 4000 });
  const tail = [];
  if (a) {
    for (const slide of a.slides) tail.push({ png: toPng(await page.evaluate((o) => window.card(o), { w: W, h: H, ...slide })), ms: slide.ms || 6000 });
  }
  const all = [...cards, ...video, ...tail];
  await page.evaluate(([w, h]) => window.recStart(w, h), [W, H]);
  for (const f of all) await page.evaluate(([src, ms]) => window.recFrame(src, ms), [`data:image/png;base64,${f.png.toString("base64")}`, f.ms]);
  const mp4 = await page.evaluate(() => window.recStop());
  await writeFile(path.join(OUT, "suppression-analysis.mp4"), Buffer.from(mp4, "base64"));
  console.log(`wrote suppression-analysis.mp4: ${all.length} frames, ${(all.reduce((s, f) => s + f.ms, 0) / 1000).toFixed(0)} s, ${(Buffer.from(mp4, "base64").length / 1e6).toFixed(1)} MB`);
}
await browser.close();
