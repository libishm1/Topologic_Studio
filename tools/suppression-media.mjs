/**
 * Captures the running app for the fire suppression GIFs and video: the
 * same fire run with no sprinklers and with each sprinkler mode (and with
 * the suppression systems by room use), on the site's examples, one PNG of
 * the 3D view per fire step and, per step, the heat release and the heads
 * open, read from the run card.
 *
 *   node tools/suppression-media.mjs --out DIR [--url http://localhost:5173/] [--horizon 90]
 *        [--cases duplex-A,duplex-B,office-A,office-C] [--modes none,wetting] [--backend auto|cpu] [--verbose]
 *
 * Capture from a production build (vite preview), not the dev server: a
 * file saved while it runs reloads the page. Each case is a model, a fire
 * class and the modes to run; the camera is set once per case, near the
 * fire, and kept for every mode, so the frames line up. A mode is "none",
 * a sprinkler mode ("evans", "spray", "wetting") or "systems" (wetting
 * sprinklers and the systems by room use). Writes DIR/<case>/<mode>/NNN.png
 * (NNN the fire step) and DIR/<case>/<mode>.json ({steps: [{step, time,
 * hrrKw, open}]}), and DIR/<case>/setup.png (the heads and systems laid
 * out, before the fire). A page that crashes or stalls is opened again and
 * the mode run once more. tools/suppression-compose.mjs turns them into
 * GIFs and a video.
 */
import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-core";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const OUT = flag("--out", null);
if (!OUT) throw new Error("--out DIR is required");
const URL_BASE = flag("--url", "http://localhost:5173/");
const URL_DEBUG = `${URL_BASE}${URL_BASE.includes("?") ? "&" : "?"}debug`;
const HORIZON = flag("--horizon", "90");
const CHROME = flag("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
const CASES = {
  "duplex-A": { example: /^Duplex apartment/, cls: "A", room: "Living Room", modes: ["none", "evans", "spray", "wetting"] },
  "duplex-B": { example: /^Duplex apartment/, cls: "B", room: "Living Room", modes: ["none", "wetting"] },
  "office-A": { example: /^Office building \(NIBS\)/, cls: "A", room: "OPEN OFFICE", modes: ["none", "wetting"] },
  "office-C": { example: /^Office building \(NIBS\)/, cls: "C", room: "SERVER RM", modes: ["none", "wetting", "systems"], setupAt: "systems" },
};
const chosen = flag("--cases", Object.keys(CASES).join(",")).split(",");
// --modes none,spray: only these modes of each case (default: the case's own).
const onlyModes = flag("--modes", null)?.split(",") ?? null;
// The solver's "Run on" setting; --verbose logs each step's readings and the page's console.
const BACKEND = flag("--backend", "auto");
const VERBOSE = args.includes("--verbose");

const browser = await chromium.launch({ executablePath: CHROME, headless: !args.includes("--headed") });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, colorScheme: "light" });

/**
 * The 3D view as a PNG: a clip of the page, which does not wait for the
 * view to hold still, taken after the view has drawn again (`settle` ms
 * more: a new run's first step, which the last run's spray must leave).
 */
async function shot(page, file, settle = 0) {
  if (settle) await page.waitForTimeout(settle);
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const box = await page.locator(".viewport").boundingBox();
  await page.screenshot({ path: file, clip: box, timeout: 60000 });
}

/** A page with the case's model, graph, fire and camera, the sprinkler and systems sections open. */
async function openCase(name, spec) {
  const page = await context.newPage();
  page.on("pageerror", (e) => console.log(`  page error: ${e.message.slice(0, 160)}`));
  page.on("console", (m) => {
    if (/CORS policy|ERR_FAILED/.test(m.text())) return;
    if (VERBOSE || m.type() === "error" || m.type() === "warning") console.log(`  [page ${m.type()}] ${m.text().slice(0, 200)}`);
  });
  page.on("crash", () => console.log("  PAGE CRASHED"));
  // The app keeps its settings in the browser: each case starts from the
  // defaults, or the switches clicked below would turn the last case's off.
  await page.addInitScript(() => {
    try {
      localStorage.clear();
    } catch {
      // Blocked storage starts from the defaults anyway.
    }
  });
  await page.goto(URL_DEBUG, { waitUntil: "load" });
  await page.waitForFunction(() => document.querySelector(".welcome .example"), null, { timeout: 120000 });
  // The composer draws its own labels: no toasts, hints or view overlays.
  await page.addStyleTag({ content: ".toast, .hint { display: none !important; } .viewport .overlay { opacity: 0 !important; pointer-events: none !important; }" });
  console.log(`\n${name}: opening the example`);
  await page.locator(".welcome").getByRole("button", { name: spec.example }).click();
  await page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled), null, { timeout: 300000 });
  await page.getByRole("tab", { name: /^Graph/ }).click();
  await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
  await page.waitForFunction(() => /nodes/.test(document.querySelector(".statusbar")?.innerText || ""), null, { timeout: 300000 });
  await page.getByRole("tab", { name: /^Model/ }).click();
  await page.getByRole("radio", { name: "Ghost" }).click().catch(() => page.getByText("Ghost", { exact: true }).first().click());
  await page.getByRole("tab", { name: /^Simulate/ }).click();
  await page.getByText("Flamescapes combustion solver (optional)", { exact: true }).click();
  await page.getByLabel("What burns first").selectOption(spec.cls).catch(() => {});
  // The fire: the middle of the room the case names (the lowest of its
  // namesakes), placed without the mouse; it snaps to the graph.
  const at = await page.evaluate(async (pattern) => {
    const v = window.__viewer;
    // The rooms' names, as the ignition ranking reads them (works in a production build).
    const { rooms: records } = await v.ignitionData();
    const ids = records.filter((r) => (r.name || "").toLowerCase().includes(pattern.toLowerCase())).map((r) => r.itemId);
    const pos = await v.positionsOf(ids);
    const list = [...pos.values()].sort((a, b) => a[1] - b[1]);
    return list[0] || null;
  }, spec.room);
  if (!at) throw new Error(`no room matching ${spec.room}`);
  console.log(`  fire at the middle of ${spec.room}: ${at.map((x) => x.toFixed(2)).join(", ")}`);
  await page.getByText("Place without the mouse", { exact: true }).first().click();
  await page.getByLabel("Point", { exact: true }).selectOption("fire").catch(() => {});
  for (const [axis, value] of [["x", at[0]], ["y", at[1]], ["z", at[2]]]) {
    await page.getByLabel(axis, { exact: true }).fill(value.toFixed(2));
    await page.getByLabel(axis, { exact: true }).blur();
  }
  await page.getByRole("button", { name: /^Place fire/ }).click();
  await page.getByText("Flamescapes settings", { exact: true }).click();
  await page.getByLabel("Simulated time (s)").fill(HORIZON);
  await page.getByLabel("Simulated time (s)").blur();
  await page.getByLabel("Run on").selectOption(BACKEND);
  // The camera: near the fire, looking down at it, the same for every mode.
  await page.evaluate(() => {
    const v = window.__viewer;
    const c = v?.world?.camera?.controls, f = v?.markers?.fire?.position;
    if (c?.setLookAt && f) c.setLookAt(f.x + 7, f.y + 8, f.z + 9, f.x, f.y, f.z, false);
  });
  await page.waitForTimeout(1200);
  for (const section of [/^Sprinklers$/, /^Systems by room use$/]) await page.locator("summary", { hasText: section }).first().click().catch(() => {});
  return page;
}

/** Switches a toggle by its label to `want`, if it is not already. */
async function toggle(page, label, want, state, key) {
  if (state[key] === want) return;
  await page.locator(".toggle__label", { hasText: label }).first().click();
  state[key] = want;
}

/** The page's switches for a mode: sprinklers and their mode, and the systems by room use. */
async function setMode(page, mode, state) {
  await toggle(page, /^Sprinklers$/, mode !== "none", state, "sprinklers");
  if (state.sprinklers) {
    await page.waitForFunction(() => /\d+ sprinkler heads/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 60000 });
    await page.getByLabel("An open head").selectOption(mode === "systems" ? "wetting" : mode);
  }
  await toggle(page, /^Systems by room use$/, mode === "systems", state, "systems");
  if (state.systems) await page.waitForFunction(() => /\d+ rooms: \d+ water/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 60000 });
  await page.waitForTimeout(600);
}

/** One run of the fire in a mode: a frame and the card's readings per step. */
async function capture(page, dir, mode, spec) {
  if (mode === (spec.setupAt ?? spec.modes[1])) await shot(page, path.join(dir, "setup.png"));
  const frames = path.join(dir, mode);
  await rm(frames, { recursive: true, force: true });
  await mkdir(frames, { recursive: true });
  console.log(`  ${mode}: running ${HORIZON} s`);
  await page.getByRole("button", { name: /^Start fire$/ }).scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: /^Start fire$/ }).click();
  await page.waitForFunction(() => /fire step \d+/i.test(document.body.innerText), null, { timeout: 180000 });
  const steps = [];
  let last = -1;
  const t0 = Date.now();
  const read = () =>
    page.evaluate(() => {
      const text = document.body.innerText;
      const step = Number((/Fire step\s*(\d+)/.exec(text) || [])[1] ?? -1);
      const card = document.querySelector('[aria-label="Flamescapes run"]')?.innerText || "";
      const hrr = /Heat release\s*([\d.,]+)\s*(kW|MW)/.exec(card);
      const time = /Simulated\s*([\d.]+)\s*\//.exec(card);
      const open = /(\d+) open/.exec(card);
      const running = [...document.querySelectorAll("button")].some((b) => /^Stop$/.test(b.textContent.trim()));
      return {
        heapMb: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null,
        step,
        time: time ? Number(time[1]) : null,
        hrrKw: hrr ? Number(hrr[1].replace(/,/g, "")) * (hrr[2] === "MW" ? 1000 : 1) : null,
        open: open ? Number(open[1]) : 0,
        running,
      };
    });
  while (Date.now() - t0 < 1200000) {
    let s = await read();
    // The run card can still show the last run's end for the first step:
    // wait briefly for this run's own, else record no reading.
    for (let tries = 0; s.step === 0 && last < 0 && (s.time == null || s.time > 10) && tries < 20; tries += 1) {
      await page.waitForTimeout(100);
      s = await read();
    }
    if (s.step === 0 && (s.time == null || s.time > 10)) Object.assign(s, { time: null, hrrKw: null, open: 0 });
    if (s.step > last) {
      await shot(page, path.join(frames, `${String(s.step).padStart(3, "0")}.png`), last < 0 ? 400 : 0);
      steps.push({ step: s.step, time: s.time, hrrKw: s.hrrKw, open: s.open });
      if (VERBOSE) console.log(`    step ${s.step}: ${s.time} s, ${s.hrrKw} kW, ${s.open} open, heap ${s.heapMb} MB`);
      last = s.step;
    }
    if (!s.running && s.step >= 0) break;
    await page.waitForTimeout(40);
  }
  await page.waitForTimeout(800);
  // The finished run (the step readout is gone by now): one frame after the last step.
  const end = await read();
  await shot(page, path.join(frames, `${String(last + 1).padStart(3, "0")}.png`));
  steps.push({ step: last + 1, time: end.time, hrrKw: end.hrrKw, open: end.open, final: true });
  const card = await page.locator('[aria-label="Flamescapes run"]').innerText().catch(() => "");
  await writeFile(path.join(dir, `${mode}.json`), JSON.stringify({ mode, steps, card }, null, 1));
  console.log(`    ${steps.length} frames; ${card.split("\n").filter((l) => /Heat release|heads|kW|clean agent|foam/.test(l)).slice(0, 4).join(" | ").slice(0, 260)}`);
}

for (const name of chosen) {
  const spec = CASES[name];
  if (!spec) throw new Error(`no case ${name}`);
  const dir = path.join(OUT, name);
  await mkdir(dir, { recursive: true });
  let page = await openCase(name, spec);
  let state = { sprinklers: false, systems: false };
  for (const mode of spec.modes.filter((m) => !onlyModes || onlyModes.includes(m))) {
    for (let attempt = 1; ; attempt += 1) {
      try {
        await setMode(page, mode, state);
        await capture(page, dir, mode, spec);
        break;
      } catch (error) {
        console.log(`  ${mode}: ${String(error.message).split("\n")[0].slice(0, 200)}`);
        if (attempt >= 2) throw error;
        console.log(`  ${mode}: once more, in a new page`);
        await page.close().catch(() => {});
        page = await openCase(name, spec);
        state = { sprinklers: false, systems: false };
      }
    }
  }
  await page.close();
}
await browser.close();
