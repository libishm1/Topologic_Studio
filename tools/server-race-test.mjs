/**
 * The backend's timing, in real Chrome: answers that come late, a server
 * that is waking, has lost the graph or is not there at all. Each check
 * stands between the page and the real local backend (page.route) and holds
 * back, refuses or passes on chosen requests, then reads what the page does
 * (wiki/architecture/state-machines.md, area 2: SM-4 to SM-6).
 *
 *   node tools/server-race-test.mjs [--url http://localhost:4173/Topologic_Studio/] [--ifc path.ifc] [--headed]
 *
 * Needs the backend on port 8000 with topologicpy, and the page on a port the
 * backend's CORS allows (4173 for vite preview).
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
const BACKEND = /^https?:\/\/(localhost|127\.0\.0\.1):8000\//;

const steps = [];
function record(ok, label, detail = "") {
  steps.push({ ok, label });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  — ${detail}` : ""}`);
}
const later = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: !args.includes("--headed"),
  args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--ignore-gpu-blocklist"],
});

/**
 * A fresh page on the app, with the backend's requests passed through `rule`:
 * rule(request, path) returns { delay, fulfill } or nothing (pass at once).
 * Counts the requests by path.
 */
async function openApp({ rule = () => null, init = null } = {}) {
  const context = await browser.newContext({ viewport: { width: 1600, height: 950 } });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  if (init) await page.addInitScript(init);
  const seen = {};
  await page.route(BACKEND, async (route) => {
    const request = route.request();
    const where = `${request.method()} ${new URL(request.url()).pathname}`;
    seen[where] = (seen[where] || 0) + 1;
    const plan = rule(request, where, seen[where]) || {};
    if (plan.delay) await later(plan.delay);
    if (plan.abort) return route.abort();
    if (plan.fulfill) return route.fulfill(plan.fulfill);
    return route.continue().catch(() => {});
  });
  await page.goto(`${URL_BASE}${URL_BASE.includes("?") ? "&" : "?"}debug`, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(() => document.querySelector(".app") && !document.body.innerText.includes("Starting the 3D engine"), null, { timeout: 90000 });
  return { context, page, seen, errors };
}

const count = (seen, where) => seen[where] || 0;
const backendUp = (page) => page.waitForFunction(() => /backend up/.test(document.body.innerText), null, { timeout: 60000 }).then(() => true).catch(() => false);

async function loadAndBuild(page) {
  await page.locator('input[type="file"][accept=".ifc"]').first().setInputFiles(IFC);
  await page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled), null, { timeout: 150000 });
  await page.getByRole("tab", { name: /^Graph/ }).click();
  await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
  await page.waitForFunction(() => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === "route", null, { timeout: 150000 });
}

/** Unfolds a closed disclosure by its summary. */
async function unfold(page, summary) {
  const s = page.locator("summary", { hasText: summary }).first();
  if (!(await s.evaluate((el) => el.parentElement.open).catch(() => true))) await s.click();
}

/** A start near an exit door, then the nearest exit door from it, by the keyboard path. */
async function placePoints(page, offset = 2) {
  await page.getByRole("tab", { name: /^Route/ }).click();
  await unfold(page, "Place without the mouse");
  await page.getByLabel("Exit at a door").selectOption({ index: 1 });
  const exit = ((await page.locator(".picker", { hasText: "Exit" }).locator(".picker__coord").first().textContent()) || "").split(",").map(Number);
  await placeStart(page, [exit[0] + offset, exit[1], exit[2] + offset]);
  await page.getByRole("button", { name: "Use the nearest exit door" }).click();
  await page.waitForTimeout(300);
}

async function placeStart(page, at) {
  await unfold(page, "Place without the mouse");
  await page.getByLabel("Point", { exact: true }).selectOption("start");
  for (const [i, axis] of ["x", "y", "z"].entries()) {
    await page.getByLabel(axis, { exact: true }).fill(String(Math.round(at[i] * 100) / 100));
    await page.getByLabel(axis, { exact: true }).blur();
  }
  await page.getByRole("button", { name: "Place start" }).click();
}

async function useTopologicPy(page) {
  await unfold(page, "Route options");
  await page.getByLabel("Engine", { exact: true }).selectOption("topologicpy");
}

const routeDrawn = (page) => page.evaluate(() => window.__viewer?.overlays?.has("path") ?? false);
const toastText = (page) => page.evaluate(() => [...document.querySelectorAll(".toast, [role='status'], [role='alert']")].map((t) => t.innerText).join(" | "));

// ---- SM-4: the graph's copy on the server ---------------------------------

console.log("[SM-4] The graph's copy on the server");
{
  // A slow upload's answer left the graph as it was: it used to make a new
  // graph object, and the code check made meanwhile vanished with it (B1).
  const { context, page } = await openApp({ rule: (_r, where) => (where === "POST /api/ifc/graph/import" ? { delay: 6000 } : null) });
  await backendUp(page);
  await loadAndBuild(page);
  await page.locator(".panel__scroll").getByText("Code check", { exact: true }).click();
  await page.getByRole("button", { name: "Check the egress" }).click();
  await page.waitForFunction(() => /Table 1017\.2/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(8000);
  const kept = /Table 1017\.2/.test(await page.locator(".panel__scroll").first().innerText());
  record(kept, "the code check stays while the graph's upload is still out", kept ? "still shown" : "gone when the upload answered");
  await context.close();
}
{
  // Built while the backend was waking: the graph goes up when a TopologicPy
  // feature first asks, rather than never (B2).
  const waking = (_r, where) => (where === "GET /api/capabilities" || where === "GET /health" ? { delay: 12000 } : null);
  const { context, page, seen } = await openApp({ rule: waking });
  await loadAndBuild(page);
  const upAfter = await backendUp(page);
  await placePoints(page);
  await useTopologicPy(page);
  await page.getByRole("button", { name: "Find egress route" }).click();
  await page.waitForFunction(() => window.__viewer?.overlays?.has("path"), null, { timeout: 60000 }).catch(() => {});
  const drawn = await routeDrawn(page);
  record(upAfter && drawn && count(seen, "POST /api/ifc/graph/import") === 1, "a graph built while the backend woke is sent when TopologicPy routes it", `${drawn ? "route drawn" : `no route (${await toastText(page)})`}; ${count(seen, "POST /api/ifc/graph/import")} upload(s)`);
  await context.close();
}
{
  // The server lost the graph (asleep, restarted, pushed out): sent again
  // once and the route asked again, rather than "build it again" (B3).
  const gone = (_r, where, n) =>
    where === "POST /api/ifc/path" && n === 1
      ? { fulfill: { status: 400, contentType: "application/json", body: JSON.stringify({ detail: "No navigation graph available. Build one first (POST /api/ifc/graph)." }) } }
      : null;
  const { context, page, seen } = await openApp({ rule: gone });
  await backendUp(page);
  await loadAndBuild(page);
  await placePoints(page);
  await useTopologicPy(page);
  await page.getByRole("button", { name: "Find egress route" }).click();
  await page.waitForFunction(() => window.__viewer?.overlays?.has("path"), null, { timeout: 60000 }).catch(() => {});
  const drawn = await routeDrawn(page);
  record(drawn && count(seen, "POST /api/ifc/path") === 2, "a graph the server lost is sent again and the route asked again", `${drawn ? "route drawn" : `no route (${await toastText(page)})`}; ${count(seen, "POST /api/ifc/graph/import")} upload(s), ${count(seen, "POST /api/ifc/path")} route request(s)`);
  await context.close();
}
{
  // No backend at all and TopologicPy saved from an earlier visit: the
  // browser's engine routes rather than refusing (B5).
  const init = () => {
    try {
      localStorage.setItem("ts.settings", JSON.stringify({ ...JSON.parse(localStorage.getItem("ts.settings") || "{}"), engine: "topologicpy" }));
    } catch {
      // A blocked store: the check below fails and says so.
    }
  };
  const { context, page } = await openApp({ rule: () => ({ abort: true }), init });
  await loadAndBuild(page);
  await placePoints(page);
  await page.getByRole("button", { name: "Find egress route" }).click();
  await page.waitForFunction(() => window.__viewer?.overlays?.has("path"), null, { timeout: 30000 }).catch(() => {});
  const drawn = await routeDrawn(page);
  record(drawn, "with no backend, a saved TopologicPy engine gives way to the browser's", drawn ? "route drawn" : `no route (${await toastText(page)})`);
  await context.close();
}

// ---- SM-5: answers for older inputs -----------------------------------------

console.log("\n[SM-5] Answers for older inputs");
{
  const slowRoute = (_r, where) => (where === "POST /api/ifc/path" ? { delay: 3000 } : null);
  const { context, page } = await openApp({ rule: slowRoute });
  await backendUp(page);
  await loadAndBuild(page);
  await placePoints(page);
  await useTopologicPy(page);
  // A route asked for, then the start moved before it came (B4).
  const exit = ((await page.locator(".picker", { hasText: "Exit" }).locator(".picker__coord").first().textContent()) || "").split(",").map(Number);
  await page.getByRole("button", { name: "Find egress route" }).click();
  await page.waitForTimeout(500);
  await placeStart(page, [exit[0] + 3, exit[1], exit[2] + 1]);
  await page.waitForTimeout(5000);
  const stale = await routeDrawn(page);
  record(!stale, "a route that comes after the start moved is not drawn", stale ? "the old start's route was drawn" : "dropped");
  // The engines compared, then the start moved: the comparison goes with the route (B16).
  await page.getByRole("button", { name: "Use the nearest exit door" }).click();
  await page.getByRole("button", { name: "Compare both engines" }).click();
  await page.waitForFunction(() => /Same route/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 30000 }).catch(() => {});
  const compared = /Same route/.test(await page.locator(".panel__scroll").first().innerText());
  await placeStart(page, [exit[0] + 2, exit[1], exit[2] + 2]);
  await page.waitForTimeout(500);
  const outlived = /Same route/.test(await page.locator(".panel__scroll").first().innerText());
  record(compared && !outlived, "the engines' comparison goes when the start moves", compared ? (outlived ? "it stayed" : "gone") : "no comparison shown");
  await context.close();
}

// ---- SM-6: the site and the backend's state ---------------------------------

console.log("\n[SM-6] The site and the backend's state");
{
  // A part of the site replaced by a newer deploy: the message says to reload (B9).
  const { context, page } = await openApp();
  await loadAndBuild(page);
  await page.route(/\/recorder(-[\w-]+)?\.js/, (route) => route.fulfill({ status: 404, body: "" }));
  await page.getByRole("button", { name: /^Record GIF$/ }).click();
  await page.waitForTimeout(1500);
  const said = await toastText(page);
  record(/Reload the page/.test(said), "a part of the site gone after a deploy says to reload", said.slice(0, 160) || "no message");
  await context.close();
}

await browser.close();
const failed = steps.filter((s) => !s.ok).length;
console.log(`\n${steps.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
