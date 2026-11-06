/**
 * Records the Flamescapes fire on the Duplex example as an animated GIF:
 * the 3D view at every fire step, from the running app, using only the
 * examples shipped with the site. No encoder to install: the PNG decoding,
 * palette (median cut) and GIF/LZW encoding are in tools/gif.mjs.
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
import { decodePng, downscale, encodeGif } from "./gif.mjs";

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
const backend = await page.locator(".result__note", { hasText: "Running on" }).first().textContent({ timeout: 10000 }).catch(() => "");
console.log(`  ${backend.split(".")[0] || "backend not shown"}`);
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
