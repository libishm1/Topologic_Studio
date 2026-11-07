/**
 * Captures the running app for the fire suppression GIFs and video: the
 * same fire run with no sprinklers and with each sprinkler mode, on the
 * site's examples, one PNG of the 3D view per fire step and, per step, the
 * heat release and the heads open, read from the run card.
 *
 *   node tools/suppression-media.mjs --out DIR [--url http://localhost:5173/] [--horizon 90]
 *        [--cases duplex-A,duplex-B,office-A]
 *
 * Each case is a model, a fire class and the modes to run; the camera is
 * set once per case, near the fire, and kept for every mode, so the frames
 * line up. Writes DIR/<case>/<mode>/NNN.png and DIR/<case>/<mode>.json
 * ({steps: [{step, time, hrrKw, open}]}), and DIR/<case>/setup.png (the
 * heads laid out, before the fire). tools/suppression-compose.mjs turns
 * them into GIFs and a video.
 */
import { mkdir, writeFile } from "node:fs/promises";
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
};
const chosen = flag("--cases", Object.keys(CASES).join(",")).split(",");

const browser = await chromium.launch({ executablePath: CHROME, headless: !args.includes("--headed") });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, colorScheme: "light" });

for (const name of chosen) {
  const spec = CASES[name];
  if (!spec) throw new Error(`no case ${name}`);
  const dir = path.join(OUT, name);
  await mkdir(dir, { recursive: true });
  const page = await context.newPage();
  page.on("pageerror", (e) => console.log(`  page error: ${e.message.slice(0, 160)}`));
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
    const { elementRecords } = await import("/src/viewer/categories.js");
    const records = await elementRecords(v.model, v.categoryIds.spaces || []);
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
  // The camera: near the fire, looking down at it, the same for every mode.
  await page.evaluate(() => {
    const v = window.__viewer;
    const c = v?.world?.camera?.controls, f = v?.markers?.fire?.position;
    if (c?.setLookAt && f) c.setLookAt(f.x + 7, f.y + 8, f.z + 9, f.x, f.y, f.z, false);
  });
  await page.waitForTimeout(1200);
  await page.locator("summary", { hasText: /^Sprinklers$/ }).first().click().catch(() => {});
  const sprinklerSwitch = page.locator(".toggle__label", { hasText: /^Sprinklers$/ }).first();
  let on = false;
  for (const mode of spec.modes) {
    const want = mode !== "none";
    if (want !== on) {
      await sprinklerSwitch.click();
      on = want;
      if (on) await page.waitForFunction(() => /\d+ sprinkler heads/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 60000 });
    }
    if (on) await page.getByLabel("An open head").selectOption(mode);
    await page.waitForTimeout(600);
    if (on && mode === spec.modes[1]) await page.locator(".viewport").screenshot({ path: path.join(dir, "setup.png") });
    const frames = path.join(dir, mode);
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
          step,
          time: time ? Number(time[1]) : null,
          hrrKw: hrr ? Number(hrr[1].replace(/,/g, "")) * (hrr[2] === "MW" ? 1000 : 1) : null,
          open: open ? Number(open[1]) : 0,
          running,
        };
      });
    while (Date.now() - t0 < 1200000) {
      const s = await read();
      if (s.step > last) {
        await page.locator(".viewport").screenshot({ path: path.join(frames, `${String(s.step).padStart(3, "0")}.png`) });
        steps.push({ step: s.step, time: s.time, hrrKw: s.hrrKw, open: s.open });
        last = s.step;
      }
      if (!s.running && s.step >= 0) break;
      await page.waitForTimeout(40);
    }
    await page.waitForTimeout(800);
    const end = await read();
    await page.locator(".viewport").screenshot({ path: path.join(frames, `${String(end.step + 1).padStart(3, "0")}.png`) });
    steps.push({ step: end.step + 1, time: end.time, hrrKw: end.hrrKw, open: end.open, final: true });
    const card = await page.locator('[aria-label="Flamescapes run"]').innerText().catch(() => "");
    await writeFile(path.join(dir, `${mode}.json`), JSON.stringify({ mode, steps, card }, null, 1));
    console.log(`    ${steps.length} frames; ${card.split("\n").filter((l) => /Heat release|heads|kW/.test(l)).slice(0, 3).join(" | ").slice(0, 200)}`);
  }
  await page.close();
}
await browser.close();
