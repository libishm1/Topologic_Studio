/**
 * The model and the graph changing under work in progress, in real Chrome:
 * a graph build still running when another model is opened, the build key
 * held down, a fire left over a rebuilt graph, the markers of a model no
 * longer open, and the viewer switched away and back
 * (wiki/architecture/state-machines.md, area 3: SM-7).
 *
 *   node tools/model-race-test.mjs [--url http://localhost:4173/Topologic_Studio/] [--ifc path.ifc] [--headed]
 */
import { chromium } from "playwright-core";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const URL_BASE = argOf("--url", "http://localhost:4173/Topologic_Studio/");
const IFC = argOf("--ifc", path.resolve(HERE, "../../Ifc2x3_Duplex_Architecture.ifc"));
const CHROME = argOf("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");

const steps = [];
function record(ok, label, detail = "") {
  steps.push({ ok, label });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  — ${detail}` : ""}`);
}

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: !args.includes("--headed"),
  args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--ignore-gpu-blocklist"],
});

async function openApp() {
  const context = await browser.newContext({ viewport: { width: 1600, height: 950 } });
  const page = await context.newPage();
  // Graph builds sent to the sampler worker, counted.
  await page.addInitScript(() => {
    const Base = window.Worker;
    window.__builds = 0;
    window.Worker = class extends Base {
      postMessage(message, ...rest) {
        if (message?.payload?.floors) window.__builds += 1;
        return super.postMessage(message, ...rest);
      }
    };
  });
  await page.goto(`${URL_BASE}${URL_BASE.includes("?") ? "&" : "?"}debug`, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(() => document.querySelector(".app") && !document.body.innerText.includes("Starting the 3D engine"), null, { timeout: 90000 });
  return { context, page };
}

const buildButtonReady = (page) =>
  page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled), null, { timeout: 150000 });
async function openModel(page) {
  await page.locator('input[type="file"][accept=".ifc"]').first().setInputFiles(IFC);
  await buildButtonReady(page);
}
async function build(page) {
  await page.getByRole("tab", { name: /^Graph/ }).click();
  await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
  await page.waitForFunction(() => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === "route", null, { timeout: 150000 });
}
// A graph is offered for rebuilding on the Graph step (its panel is drawn only while open).
async function hasGraph(page) {
  await page.getByRole("tab", { name: /^Graph/ }).click().catch(() => {});
  await page.waitForTimeout(300);
  return page.evaluate(() => [...document.querySelectorAll("button")].some((b) => /Rebuild graph/.test(b.textContent)));
}
async function unfold(page, summary) {
  const s = page.locator("summary", { hasText: summary }).first();
  if (!(await s.evaluate((el) => el.parentElement.open).catch(() => true))) await s.click();
}
/** The exit at the first door; returns its coordinates. */
async function exitAtDoor(page) {
  await page.getByRole("tab", { name: /^Route/ }).click();
  await unfold(page, "Place without the mouse");
  await page.getByLabel("Exit at a door").selectOption({ index: 1 });
  return ((await page.locator(".picker", { hasText: "Exit" }).locator(".picker__coord").first().textContent()) || "").split(",").map(Number);
}
async function placeAt(page, kind, at) {
  await unfold(page, "Place without the mouse");
  if (kind !== "fire") await page.getByLabel("Point", { exact: true }).selectOption(kind);
  for (const [i, axis] of ["x", "y", "z"].entries()) {
    await page.getByLabel(axis, { exact: true }).fill(String(Math.round(at[i] * 100) / 100));
    await page.getByLabel(axis, { exact: true }).blur();
  }
  await page.getByRole("button", { name: kind === "fire" ? "Place fire origin" : `Place ${kind}` }).click();
}

console.log("[SM-7] The model and the graph changing under work");
{
  // A build still running when the same file is opened again (a new model):
  // it no longer lands on the new one. The "b" key pressed meanwhile starts
  // no second build. The sampler worker's script is held back, so the build waits.
  const { context, page } = await openApp();
  await openModel(page);
  let release;
  const held = new Promise((resolve) => (release = resolve));
  await page.route(/sampler\.worker/, async (route) => {
    await held;
    await route.continue();
  });
  await page.getByRole("tab", { name: /^Graph/ }).click();
  await page.getByRole("button", { name: /Build egress graph/ }).first().click();
  await page.waitForTimeout(300);
  await page.keyboard.press("b");
  await page.keyboard.press("b");
  await page.waitForTimeout(300);
  const builds = await page.evaluate(() => window.__builds);
  record(builds === 1, "the build key pressed during a build starts no second one", `${builds} build(s) sent`);
  // The new model loads (its Build button may stay disabled while the old build holds it busy).
  await page.locator('input[type="file"][accept=".ifc"]').first().setInputFiles(IFC);
  await page.waitForTimeout(8000);
  release();
  await page.waitForTimeout(4000);
  const landed = await hasGraph(page);
  record(!landed, "a build still running when another model opens does not land on it", landed ? "the old model's graph was set on the new one" : "dropped");
  await context.close();
}
{
  // A fire played, then the graph rebuilt: its step and burning nodes go,
  // and with them the export that wrote them onto the new graph.
  const { context, page } = await openApp();
  await openModel(page);
  await build(page);
  const exit = await exitAtDoor(page);
  await page.getByRole("tab", { name: /^Simulate/ }).click();
  await placeAt(page, "fire", [exit[0] + 2, exit[1], exit[2] + 2]);
  await page.getByRole("button", { name: /^Start fire$/ }).click();
  await page.waitForTimeout(1500);
  await page.getByRole("button", { name: /^Stop$/ }).click().catch(() => {});
  const exportBefore = await page.locator(".panel__scroll").getByRole("button", { name: "Export for TopologicPy (JSON)" }).count();
  await build(page);
  await page.getByRole("tab", { name: /^Simulate/ }).click();
  const exportAfter = await page.locator(".panel__scroll").getByRole("button", { name: "Export for TopologicPy (JSON)" }).count();
  record(exportBefore === 1 && exportAfter === 0, "a rebuilt graph drops the last fire and its export", `export offered ${exportBefore} before the rebuild, ${exportAfter} after`);
  // The points' markers go with the model.
  await page.getByRole("tab", { name: /^Route/ }).click();
  await placeAt(page, "start", [exit[0] + 2, exit[1], exit[2] + 2]);
  const shownBefore = await page.evaluate(() => window.__viewer?.markers?.start?.visible ?? null);
  await openModel(page);
  const shownAfter = await page.evaluate(() => window.__viewer?.markers?.start?.visible ?? null);
  record(shownBefore === true && shownAfter === false, "the start marker goes when another model is opened", `visible ${shownBefore} before, ${shownAfter} after`);
  await context.close();
}
{
  // Switched to the topology viewer and back: the model's viewer is new and
  // empty, so the model, graph and points go with the old one.
  const { context, page } = await openApp();
  await openModel(page);
  await build(page);
  const viewerMode = page.getByRole("group", { name: "Viewer mode" });
  await viewerMode.getByRole("button", { name: "Topology JSON" }).click();
  await page.waitForTimeout(800);
  await viewerMode.getByRole("button", { name: "IFC", exact: true }).click();
  await page.waitForFunction(() => !document.body.innerText.includes("Starting the 3D engine"), null, { timeout: 90000 }).catch(() => {});
  await page.waitForTimeout(800);
  const stale = await hasGraph(page);
  record(!stale, "back from the topology viewer, no graph is offered for a model no longer drawn", stale ? "the old graph is still offered" : "cleared");
  await context.close();
}

await browser.close();
const failed = steps.filter((s) => !s.ok).length;
console.log(`\n${steps.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
