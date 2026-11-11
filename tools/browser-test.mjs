/**
 * Drives the real app in real Chrome and reports what breaks.
 *
 * Uses playwright-core against the installed Chrome (no browser download).
 * Everything Chrome complains about is captured: console errors, uncaught
 * exceptions, failed requests, non-2xx responses. Screenshots land in
 * tools/shots/ so the UI can be eyeballed afterwards.
 *
 *   node tools/browser-test.mjs [--url http://127.0.0.1:5173] [--ifc path.ifc] [--headed]
 *                              [--linked mep.ifc]   a second model of the --ifc building
 */
import { chromium } from "playwright-core";
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SHOTS = path.join(HERE, "shots");

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
// Vite binds to "localhost", which on this machine resolves to ::1 only;
// hard-coding 127.0.0.1 gets connection-refused.
const URL_BASE = argOf("--url", "http://localhost:5173");
const IFC = argOf("--ifc", path.resolve(HERE, "../../Ifc2x3_Duplex_Architecture.ifc"));
const HEADED = args.includes("--headed");
const CHROME =
  argOf("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");

const problems = [];
const steps = [];
let stepNo = 0;

function record(ok, label, detail = "") {
  steps.push({ ok, label, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  — ${detail}` : ""}`);
  if (!ok) problems.push(`${label}${detail ? `: ${detail}` : ""}`);
}

async function shot(page, name) {
  stepNo += 1;
  const file = path.join(SHOTS, `${String(stepNo).padStart(2, "0")}-${name}.png`);
  await page.screenshot({ path: file, fullPage: false });
  return file;
}

await mkdir(SHOTS, { recursive: true });

if (!existsSync(IFC)) {
  console.error(`No IFC file at ${IFC}. Pass --ifc <path>.`);
  process.exit(2);
}

console.log(`\nChrome : ${CHROME}`);
console.log(`App    : ${URL_BASE}`);
console.log(`Model  : ${IFC}\n`);

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: !HEADED,
  args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--ignore-gpu-blocklist"],
});

// --mobile: a phone (390 x 844, DPR 3, touch) with the CPU slowed 4x, about
// a mid-range handset, to measure what a phone user waits for.
const MOBILE = args.includes("--mobile");
const context = await browser.newContext(
  MOBILE
    ? { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true }
    : { viewport: { width: 1600, height: 950 }, deviceScaleFactor: 1 },
);
const page = await context.newPage();
// Every worker the page starts, by its script, and the kind of each job sent
// to one, so a check can count them (wiki/architecture/state-machines.md: a
// stopped fire starts none, a double click makes one recorder or one fire).
await page.addInitScript(() => {
  const Base = window.Worker;
  window.__workers = [];
  window.__jobs = [];
  window.Worker = class extends Base {
    constructor(url, opts) {
      super(url, opts);
      window.__workers.push(String(url));
    }
    postMessage(message, ...rest) {
      window.__jobs.push(message?.payload?.kind ?? null);
      return super.postMessage(message, ...rest);
    }
  };
});
const workersOf = (pattern) => page.evaluate((source) => window.__workers.filter((url) => new RegExp(source).test(url)).length, pattern.source);
const jobsOf = (kind) => page.evaluate((k) => window.__jobs.filter((j) => j === k).length, kind);
// A step's sections and disclosures close when it is left: these open one again if it is closed.
const openSection = async (title) => {
  const header = page.locator("button.section__header", { hasText: title }).first();
  if ((await header.getAttribute("aria-expanded").catch(() => "true")) !== "true") await header.click();
};
// Simulate's "Rerouting and timing" fields.
const openTiming = async () => {
  if (!(await page.getByLabel("Steps", { exact: true }).isVisible().catch(() => false))) await page.getByText("Rerouting and timing", { exact: true }).click();
};
if (MOBILE) {
  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  console.log("Mobile: 390 x 844, DPR 3, touch, CPU 4x slower\n");
}

// --block-backend: abort every request to the API, which is what a site
// served from GitHub Pages alone is. Shows what runs without a server.
const BLOCK_BACKEND = args.includes("--block-backend");
if (BLOCK_BACKEND) {
  await page.route(/onrender\.com|localhost:8000|127\.0\.0\.1:8000/, (route) => route.abort());
  console.log("Backend: blocked (GitHub-only run)\n");
}

// ---- capture everything Chrome reports -------------------------------------

const consoleErrors = [];
const pageErrors = [];
const failedRequests = [];
const badResponses = [];
// Cloudflare Turnstile's own frames (the human check on the Cloudflare
// describer, loaded once a site has DESCRIBE_URL and a site key) probe and log
// in a headless browser: a 401 token probe, CSP refusals and hidden logs
// inside its iframe. Counted apart, so the app's own errors stay visible.
const challengeNoise = [];
const fromChallenge = (url) => /^https:\/\/([a-z0-9-]+\.)?challenges\.cloudflare\.com\//.test(url || "");

page.on("console", (msg) => {
  const type = msg.type();
  if (type === "error" || type === "warning") {
    const text = msg.text();
    if (fromChallenge(msg.location()?.url)) {
      if (type === "error") challengeNoise.push(`console: ${text}`);
      return;
    }
    // Vite injects a websocket for HMR; its noise is not the app's problem.
    if (text.includes("[vite]") || text.includes("Download the React DevTools")) return;
    (type === "error" ? consoleErrors : []).push?.(text);
    if (type === "error") console.log(`    [console.error] ${text.slice(0, 200)}`);
  }
});
page.on("pageerror", (error) => {
  pageErrors.push(error.message);
  console.log(`    [pageerror] ${error.message.slice(0, 300)}`);
});
page.on("requestfailed", (req) => {
  const failure = req.failure()?.errorText || "";
  if (failure.includes("ERR_ABORTED")) return; // navigation churn
  if (fromChallenge(req.url())) {
    challengeNoise.push(`request failed: ${failure}`);
    return;
  }
  failedRequests.push(`${req.method()} ${req.url()} — ${failure}`);
  console.log(`    [requestfailed] ${req.url().slice(0, 140)} ${failure}`);
});
page.on("response", (res) => {
  if (res.status() >= 400) {
    if (fromChallenge(res.url())) {
      challengeNoise.push(`http ${res.status()}`);
      return;
    }
    badResponses.push(`${res.status()} ${res.url()}`);
    console.log(`    [http ${res.status()}] ${res.url().slice(0, 140)}`);
  }
});

// ---- timing -----------------------------------------------------------------
//
// Stage stopwatches (click to result, as a user sees it) plus the duration of
// every backend call. The fire stream is long-lived by design and left out.

const timings = [];
const timing = (label, ms) => timings.push({ label, ms: Math.round(ms) });
const apiCalls = [];
page.on("requestfinished", (req) => {
  const url = new URL(req.url());
  if (url.port !== "8000" || url.pathname.startsWith("/api/fire/stream")) return;
  const t = req.timing();
  if (t.responseEnd > 0) apiCalls.push({ call: `${req.method()} ${url.pathname}`, ms: t.responseEnd });
});

// ---- 1. boot ----------------------------------------------------------------

console.log("[1] Loading the app");
const tBoot = Date.now();
// ?debug exposes window.__viewer in a production build as well (Viewport.jsx),
// so the same picks work against a deployed site.
await page.goto(`${URL_BASE}${URL_BASE.includes("?") ? "&" : "?"}debug`, { waitUntil: "domcontentloaded", timeout: 60000 });

record(
  (await page.title()).length > 0,
  "page has a title",
  await page.title(),
);
await page.waitForSelector(".app", { timeout: 20000 }).catch(() => {});
record(await page.locator(".app").count() > 0, "app shell rendered");

// The engine is a lazy chunk; the placeholder flips once it is live.
const engineReady = await page
  .waitForFunction(
    () => !document.body.innerText.includes("Starting the 3D engine"),
    null,
    { timeout: 90000 },
  )
  .then(() => true)
  .catch(() => false);
record(engineReady, "3D engine finished booting");
timing("Open page to 3D engine ready", Date.now() - tBoot);

await page.waitForLoadState("load").catch(() => {});
const pageLoad = await page.evaluate(() => {
  const nav = performance.getEntriesByType("navigation")[0];
  const fcp = performance.getEntriesByName("first-contentful-paint")[0];
  const bytes = performance
    .getEntriesByType("resource")
    .reduce((sum, r) => sum + (r.transferSize || 0), nav?.transferSize || 0);
  return { fcp: fcp?.startTime, dcl: nav?.domContentLoadedEventEnd, load: nav?.loadEventEnd, bytes };
});
if (pageLoad.fcp) timing("First contentful paint", pageLoad.fcp);
if (pageLoad.load) timing("Page load event", pageLoad.load);
console.log(`    page transferred: ${(pageLoad.bytes / 1024 / 1024).toFixed(1)} MiB`);

const webglOk = await page.evaluate(() => {
  const canvas = document.querySelector(".viewport__canvas canvas");
  if (!canvas) return { canvas: false };
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  return { canvas: true, gl: Boolean(gl), w: canvas.width, h: canvas.height };
});
record(webglOk.canvas, "viewer canvas exists", JSON.stringify(webglOk));

record(
  !(await page.locator("text=The 3D viewer could not start").count()),
  "no viewer init error",
);

const backendUp = await page
  .waitForFunction(() => !document.body.innerText.includes("Backend offline"), null, {
    timeout: 15000,
  })
  .then(() => true)
  .catch(() => false);
record(BLOCK_BACKEND ? !backendUp : backendUp, BLOCK_BACKEND ? "backend reported offline (blocked)" : "backend reported online");

await shot(page, "boot");
// Baseline for the render check after the model loads.
const emptyShot = await page.locator(".viewport__canvas").screenshot();

// ---- 2. load the IFC --------------------------------------------------------

console.log("\n[2] Loading the IFC model");
const t0 = Date.now();
await page.locator('input[type="file"][accept=".ifc"]').first().setInputFiles(IFC);

const modelLoaded = await page
  .waitForFunction(
    () =>
      [...document.querySelectorAll("button")].some(
        (b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled,
      ),
    null,
    { timeout: 180000 },
  )
  .then(() => true)
  .catch(() => false);
record(modelLoaded, "IFC model loaded", `${((Date.now() - t0) / 1000).toFixed(1)}s`);
timing("Open IFC, first time (convert + cache)", Date.now() - t0);

if (modelLoaded) {
  // Opening a model moves on to the Graph step; the counts are on Model.
  await page.getByRole("tab", { name: /^Model/ }).click();
  await page.waitForTimeout(300);
  const stats = await page.evaluate(() => {
    const out = {};
    document.querySelectorAll(".stat").forEach((el) => {
      const k = el.querySelector(".stat__label")?.textContent?.trim();
      const v = el.querySelector(".stat__value")?.textContent?.trim();
      if (k) out[k] = v;
    });
    return out;
  });
  console.log(`    category counts: ${JSON.stringify(stats)}`);
  record(
    Number(stats.Floors?.replace(/,/g, "")) > 0,
    "floors were detected",
    `Floors=${stats.Floors} Stairs=${stats.Stairs} Doors=${stats.Doors} Walls=${stats.Walls}`,
  );
}

// Is the viewport actually drawing 3D? A WebGL canvas cannot be read back via
// drawImage without preserveDrawingBuffer, so instead move the camera between
// two standard views and compare the pixels. A blank canvas renders
// identically from every angle; a real model does not.
async function viewShot(which) {
  // A phone shows only Iso and Fit; set the others through the viewer.
  if (MOBILE && !["Iso", "Fit"].includes(which)) {
    await page.evaluate((view) => window.__viewer?.setStandardView(view), which.toLowerCase());
  } else {
    await page.getByRole("button", { name: which, exact: true }).click();
  }
  await page.waitForTimeout(1400);
  return page.locator(".viewport__canvas").screenshot();
}
const topView = await viewShot("Top");
const frontView = await viewShot("Front");
const identical = Buffer.compare(topView, frontView) === 0;
record(
  !identical,
  "viewport renders 3D geometry (top view differs from front view)",
  `top ${(topView.length / 1024).toFixed(0)}KB vs front ${(frontView.length / 1024).toFixed(0)}KB`,
);
await page.getByRole("button", { name: "Iso", exact: true }).click();
await page.waitForTimeout(1000);

await shot(page, "model-loaded");

// ---- 3. build the graph -----------------------------------------------------

console.log("\n[3] Building the navigation graph");
await page.getByRole("tab", { name: /^Graph/ }).click();
await page.waitForTimeout(200);
const buildBtn = page.getByRole("button", { name: /Build egress graph|Rebuild graph/ });
record(await buildBtn.count() > 0, "build button present");

if (await buildBtn.count()) {
  const t1 = Date.now();
  await buildBtn.first().click();
  // Done means the app has moved to the Route tab, its last step. Waiting for
  // "nodes" to appear raced it: the status bar shows the node count as soon as
  // the worker finishes, before the upload returns and the tab switches.
  const built = await page
    .waitForFunction(
      () => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === "route",
      null,
      { timeout: 180000 },
    )
    .then(() => true)
    .catch(() => false);
  record(built, "graph built", `${((Date.now() - t1) / 1000).toFixed(1)}s`);
  timing("Build egress graph (sample in browser + server)", Date.now() - t1);

  if (built) {
    // buildGraph() advances to the Route step; the graph stats live on the
    // Graph step, so come back before scraping them.
    await page.getByRole("tab", { name: /^Graph/ }).click();
    await page.waitForTimeout(400);
    const g = await page.evaluate(() => {
      const out = {};
      document.querySelectorAll(".stat").forEach((el) => {
        const k = el.querySelector(".stat__label")?.textContent?.trim();
        const v = el.querySelector(".stat__value")?.textContent?.trim();
        if (k) out[k] = v;
      });
      return out;
    });
    console.log(`    graph: ${JSON.stringify(g)}`);
    // A space outside the building that no door opens onto is not floor (core/outdoorSpaces.js, 2026-11-11): the Duplex's "Roof".
    const outdoorNote = await page.getByText(/^Not walked on: /).first().textContent({ timeout: 3000 }).catch(() => "");
    record(/outside the building/.test(outdoorNote || ""), "a roof drawn as a room is not walked on", (outdoorNote || "no note").slice(0, 140));
    record(Number(g.Nodes?.replace(/,/g, "")) > 100, "graph has a sane node count", `${g.Nodes}`);
    record(Number(g.Links?.replace(/,/g, "")) > 100, "graph has edges", `${g.Links}`);
  }
}
await shot(page, "graph-built");

// ---- 4. pick start and exit, then route -------------------------------------

console.log("\n[4] Picking points and routing");

// Frame the model so a click at a screen fraction actually lands on it,
// then switch to the Route tab, where the pickers live.
await page.getByRole("button", { name: "Fit", exact: true }).click();
await page.waitForTimeout(1300);
await page.getByRole("tab", { name: /^Route/ }).click();
await page.waitForTimeout(400);

const box = await page.locator(".viewport__canvas").boundingBox();
async function pickAt(kind, fx, fy) {
  const picker = page.locator(".picker", { hasText: new RegExp(kind, "i") });
  const count = await picker.count();
  if (!count) {
    const debug = await page.evaluate(() => ({
      tabs: [...document.querySelectorAll('[role="tab"]')].map(
        (t) => `${t.textContent}:${t.getAttribute("aria-selected")}`,
      ),
      panelText: document.querySelector(".panel__scroll")?.innerText?.slice(0, 160),
      pickers: document.querySelectorAll(".picker").length,
    }));
    console.log(`    [pick:${kind}] no .picker matched — ${JSON.stringify(debug)}`);
    return false;
  }
  const btn = picker.getByRole("button").first();
  const disabled = await btn.isDisabled();
  console.log(`    [pick:${kind}] pickers=${count} disabled=${disabled}`);
  if (disabled) return false;

  await btn.click();
  await page.waitForTimeout(200);
  const armed = await picker.locator(".picker__coord").first().textContent();
  console.log(`    [pick:${kind}] after arming: "${armed?.trim()}"`);

  // Picks land on the graph only, so aim at the graph node whose screen
  // position is nearest the target, as a user would. Needs the dev build's
  // window.__viewer; a production build falls back to the plain target.
  const aim = await page.evaluate(
    ([tx, ty]) => {
      const v = window.__viewer;
      const nodes = v?._graphNodes;
      const cam = v?.world?.camera?.three;
      const el = v?._container;
      if (!nodes || !cam || !el) return null;
      const r = el.getBoundingClientRect();
      const V = cam.matrixWorldInverse.elements;
      const P = cam.projectionMatrix.elements;
      const mul = (m, x, y, z, w) => [
        m[0] * x + m[4] * y + m[8] * z + m[12] * w,
        m[1] * x + m[5] * y + m[9] * z + m[13] * w,
        m[2] * x + m[6] * y + m[10] * z + m[14] * w,
        m[3] * x + m[7] * y + m[11] * z + m[15] * w,
      ];
      let best = null;
      for (let i = 0; i < nodes.length; i += 3) {
        const [cx, cy, cz, cw] = mul(P, ...mul(V, nodes[i], nodes[i + 1], nodes[i + 2], 1));
        if (cw <= 0) continue;
        const sx = ((cx / cw + 1) / 2) * r.width;
        const sy = ((1 - cy / cw) / 2) * r.height;
        if (sx < 0 || sy < 0 || sx > r.width || sy > r.height) continue;
        const d = (sx - r.width * tx) ** 2 + (sy - r.height * ty) ** 2;
        if (!best || d < best.d) best = { d, x: r.left + sx, y: r.top + sy, z: cz / cw };
      }
      return best;
    },
    [fx, fy],
  );
  // Move first, then click: the engine's raycaster tracks pointer position
  // from move events.
  const x = aim ? aim.x : box.x + box.width * fx;
  const y = aim ? aim.y : box.y + box.height * fy;
  await page.mouse.move(x, y);
  await page.waitForTimeout(120);
  await page.mouse.click(x, y);
  await page.waitForTimeout(900);

  const text = await picker.locator(".picker__coord").first().textContent();
  console.log(`    [pick:${kind}] after click(${fx},${fy}): "${text?.trim()}"`);
  return Boolean(text && !text.includes("not set") && !text.includes("click in"));
}

const gotStart = await pickAt("Start", 0.42, 0.62);
record(gotStart, "start point placed by clicking the model");

// The Duplex is two homes, so a click can land in the other one, where "no
// route" is the right answer. Try a few exits, as a user would, until the
// route is found; the first attempt is the one timed.
const routeBtn = page.getByRole("button", { name: /Find egress route/ });
const findRoute = async () => {
  await routeBtn.click();
  return page
    .waitForFunction(() => /route found|no route/i.test(document.querySelector(".result__title")?.textContent || ""), null, {
      timeout: 60000,
    })
    .then(() => true)
    .catch(() => false);
};
const routeFound = () => page.evaluate(() => /route found/i.test(document.querySelector(".result__title")?.textContent || ""));
let gotExit = false;
let routed = false;
let tRoute = 0;
search: for (const [sx, sy] of [[null, null], [0.6, 0.45], [0.55, 0.62], [0.38, 0.45]]) {
  if (sx !== null) await pickAt("Start", sx, sy);
  for (const [fx, fy] of [[0.6, 0.34], [0.5, 0.45], [0.36, 0.5], [0.62, 0.55], [0.48, 0.66], [0.3, 0.62]]) {
    gotExit = await pickAt("Exit", fx, fy);
    if (!gotExit || !gotStart) continue;
    const t = Date.now();
    routed = await findRoute();
    if (!tRoute && routed) tRoute = Date.now() - t;
    if (routed && (await routeFound())) break search;
  }
}
// On a phone-sized canvas every click can land in the other dwelling. Then
// keep the clicked start and take the exit at each door in turn (the
// keyboard path), as a user would pick an exit door.
if (gotStart && !(await routeFound())) {
  await page.getByText("Place without the mouse").first().click().catch(() => {});
  const doorSelect = page.getByLabel("Exit at a door");
  const doors = await doorSelect.locator("option").count().catch(() => 0);
  for (let i = 1; i < doors; i += 1) {
    await doorSelect.selectOption({ index: i });
    routed = await findRoute();
    if (routed && (await routeFound())) {
      console.log(`    exit taken at door ${i} after the clicked exits were in the other part`);
      gotExit = true;
      break;
    }
  }
}
record(gotExit, "exit point placed by clicking the model");
await shot(page, "points-picked");

if (gotStart && gotExit) {
  record(routed, "route computed");
  if (routed) timing("Find egress route (click to drawn)", tRoute);

  if (routed) {
    const r = await page.evaluate(() => {
      const card = document.querySelector(".result");
      const out = { Found: /route found/i.test(card?.querySelector(".result__title")?.textContent || "") ? "yes" : "no" };
      card?.querySelectorAll(".result__item").forEach((el) => {
        const k = el.querySelector(".result__label")?.textContent?.trim();
        const v = el.querySelector(".result__value")?.textContent?.trim();
        if (k) out[k] = v;
      });
      return out;
    });
    console.log(`    route: ${JSON.stringify(r)}`);
    record(r.Found === "yes", "a route was found", JSON.stringify(r));
    record(
      parseFloat(r.Length) > 3,
      "route spans a meaningful distance",
      r.Length,
    );
    // The egress code check (core/codeCheck.js): a verdict per storey, each with its clause.
    await page.locator(".panel__scroll").getByText("Code check", { exact: true }).click();
    await page.getByRole("button", { name: "Check the egress" }).click();
    const verdicts = await page
      .locator(".code-list")
      .innerText({ timeout: 30000 })
      .catch(() => "");
    const storeysChecked = (verdicts.match(/longest walk to an exit|reaches an exit/g) || []).length;
    record(storeysChecked > 0 && /Table 1017\.2/.test(verdicts), "the code check gives each storey's walk to an exit, with its clause", `${storeysChecked} storeys`);
    // A switch low in a panel once scrolled the whole page when clicked,
    // leaving a blank band under the app (fixed 2026-11-06). Flip one there
    // and back: the page itself must not move.
    const sprinklers = page.getByText("The building has automatic sprinklers", { exact: true });
    await sprinklers.click();
    await sprinklers.click();
    const pageScroll = await page.evaluate(() => document.scrollingElement.scrollTop + document.body.scrollTop + document.getElementById("root").scrollTop);
    record(pageScroll === 0, "a switch low in a panel leaves the page where it is", `${pageScroll} px scrolled`);
    // Straightened by default (src/core/straighten.js): never longer than
    // the lattice route it came from.
    record(
      r["On the grid"] !== undefined && parseFloat(r.Length) <= parseFloat(r["On the grid"]) + 0.05,
      "route is drawn straightened, no longer than on the grid",
      `${r.Length} vs ${r["On the grid"]}`,
    );
  }
}
await shot(page, "route");

// ---- 4b. alternative routes -------------------------------------------------
//
// Added 2026-10-27 with /api/ifc/path/alternatives.

const readStats = () =>
  page.evaluate(() => {
    const out = {};
    document.querySelectorAll(".stat").forEach((el) => {
      const k = el.querySelector(".stat__label")?.textContent?.trim();
      const v = el.querySelector(".stat__value")?.textContent?.trim();
      if (k) out[k] = v;
    });
    return out;
  });

console.log("\n[4b] Alternative routes");
const altBtn = page.getByRole("button", { name: /^Find alternative routes$/ });
if (gotStart && gotExit && (await altBtn.isEnabled())) {
  // Computed in the browser now, so read the result off the panel rather than
  // waiting for a request that no longer happens.
  const tAlt = Date.now();
  await altBtn.click();
  const listed = await page
    .waitForFunction(() => /route 1/i.test(document.body.innerText), null, { timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  if (listed) timing("Alternative routes (click to answer)", Date.now() - tAlt);
  record(listed, "alternative routes listed in the panel");
  if (listed) {
    const rows = await page.evaluate(() =>
      [...document.querySelectorAll(".kv .kv__v")]
        .map((el) => el.textContent.trim())
        .filter((t) => /^\d+(\.\d+)? m/.test(t)),
    );
    console.log(`    alternatives: ${rows.join(" | ")}`);
    const lengths = rows.map((t) => parseFloat(t));
    record(lengths.length > 0, "alternative routes returned", `${lengths.length} route(s)`);
    record(
      lengths.every((c, i) => i === 0 || c >= lengths[i - 1] - 0.05),
      "alternatives are ranked by length",
      lengths.join(" <= "),
    );
  }
  await page.locator(".kv", { hasText: /route 1/i }).scrollIntoViewIfNeeded().catch(() => {});
  await shot(page, "alternatives");
} else {
  record(false, "Find alternative routes was enabled", "needs start and exit");
}

// ---- 4b'. people leaving -----------------------------------------------------
// The egress animation (core/crowd.js, viewer/crowd.js, 2026-11-09): off by
// default; on, everyone walks out by the quickest exit (the nearest an
// option, 2026-11-10), drawn on the GPU.

console.log("\n[4b'] People leaving");
const peopleSection = page.locator("summary, button", { hasText: /^People leaving/ }).first();
if (await peopleSection.count()) {
  await peopleSection.scrollIntoViewIfNeeded();
  await peopleSection.click();
  record(!(await page.evaluate(() => window.__viewer?.overlays?.has("crowd") ?? false)), "no people are drawn until switched on");
  const tCrowd = Date.now();
  await page.locator(".toggle__label", { hasText: /^Show people leaving$/ }).first().click();
  const crowdLine = await page
    .waitForFunction(() => document.body.innerText.match(/\d[\d,]* people, one per [^\n]*the last is out at \d+ s[^\n]*/)?.[0], null, { timeout: 60000 })
    .then((h) => h.jsonValue())
    .catch(() => "");
  const drawn = await page.evaluate(() => {
    const o = window.__viewer?.overlays?.get("crowd");
    return o ? (o.geometry.instanceCount ?? o.geometry.attributes.position.count) : 0;
  });
  const counted = Number((crowdLine.match(/^([\d,]+) people/)?.[1] || "0").replace(/,/g, ""));
  record(Boolean(crowdLine) && drawn === counted && counted > 0, "everyone walks out to an exit, drawn as figures", `${crowdLine.slice(0, 160)} (${drawn} drawn, ${((Date.now() - tCrowd) / 1000).toFixed(1)}s)`);
  const runnerDrawn = await page.evaluate(() => window.__viewer?.overlays?.has("stairRunner") ?? false);
  record(runnerDrawn, "a runner loops down a stair", runnerDrawn ? "drawn" : "none");
  // The switch lasts for the visit only, so a later graph does not start them (2026-11-09).
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("ts.settings") || "{}").crowd);
  record(stored === undefined, "the people switch is not remembered for the next visit", stored === undefined ? "not stored" : `stored ${stored}`);
  await shot(page, "people-leaving");
  await page.getByLabel("Show them as").selectOption("dots").catch(() => {});
  await page.waitForTimeout(300);
  const dots = await page.evaluate(() => window.__viewer?.overlays?.get("crowd")?.type ?? null);
  record(dots === "Points", "the people can be drawn as dots", dots || "none");
  await page.getByLabel("Show them as").selectOption("figures").catch(() => {});
  // Full gathering places clear (core/crowd.js shelterArea), an option.
  await page.locator(".toggle__label", { hasText: /^Full gathering places clear$/ }).first().click().catch(() => {});
  await page.getByLabel("Each gathering place").selectOption("10").catch(() => {});
  const shelterLine = await page
    .waitForFunction(() => document.body.innerText.match(/Each gathering place holds \d+; \d+ leave a full one\./)?.[0], null, { timeout: 30000 })
    .then((h) => h.jsonValue())
    .catch(() => "");
  record(Boolean(shelterLine), "a full gathering place clears as newcomers arrive", shelterLine || "no line");
  await page.locator(".toggle__label", { hasText: /^Full gathering places clear$/ }).first().click().catch(() => {});
  // Egress with the fire, an option: with no fire yet they wait for one.
  await page.getByLabel("Egress").selectOption("fire").catch(() => {});
  const waits = await page.getByText(/They wait for the fire/).count();
  record(waits > 0, "egress with the fire waits for a fire", waits ? "waiting" : "no hint");
  await page.getByLabel("Egress").selectOption("none").catch(() => {});
  // Each person takes the quickest exit, queues counted, by default; the nearest is an option.
  const choiceField = page.getByLabel("Each person takes");
  const choiceDefault = await choiceField.inputValue().catch(() => "");
  const quickestText = await page
    .waitForFunction(() => /Each person takes the exit that gets them out soonest/.test(document.body.innerText), null, { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  record(choiceDefault === "quickest" && quickestText, "people take the quickest exit by default", `${choiceDefault || "no field"}, ${quickestText ? "said so" : "not said"}`);
  await choiceField.selectOption("nearest").catch(() => {});
  const nearestWalk = await page
    .waitForFunction(() => /Each person heads for the nearest exit by the shortest walk/.test(document.body.innerText), null, { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  record(nearestWalk, "the nearest exit is an option, walked again", nearestWalk ? "walked again" : "no change");
  await choiceField.selectOption("quickest").catch(() => {});
  // Randomness in how people move (wiki/roadmap/crowd-motion-randomness-plan-2026-11-11.md, stage 1):
  // speeds that wobble and floor-field steps, run over five seeds, the range reported.
  await page.locator(".toggle__label", { hasText: /^Walking speed varies over time$/ }).first().click().catch(() => {});
  await page.getByLabel("Each step").selectOption("wandering").catch(() => {});
  const rangeLine = await page
    .waitForFunction(() => document.body.innerText.match(/Over 5 of 5 runs: the last out between \d+ and \d+ s \(median \d+ s\)[^\n]*/)?.[0], null, { timeout: 60000 })
    .then((h) => h.jsonValue())
    .catch(() => "");
  record(Boolean(rangeLine), "people's motion varies: five seeds, the range of the last one out", rangeLine.slice(0, 140) || "no range");
  await page.getByLabel("Each step").selectOption("exact").catch(() => {});
  await page.locator(".toggle__label", { hasText: /^Walking speed varies over time$/ }).first().click().catch(() => {});
  const backToExact = await page
    .waitForFunction(() => !/Over \d+ of \d+ runs/.test(document.body.innerText), null, { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  record(backToExact, "with the randomness off, one exact walk again", backToExact ? "no range" : "range still shown");
  // Recording the view as a GIF (lib/recorder.js, workers/gif.worker.js).
  const recordButton = page.getByRole("button", { name: /^Record GIF$/ });
  if (await recordButton.count()) {
    // Clicked twice while the recorder's module loads (held here), it makes
    // one recording: the second used to start a recorder no Stop reached (SM-3).
    const recorderChunk = /\/recorder(-[\w-]+)?\.js/;
    await page.route(recorderChunk, async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 800));
      await route.continue();
    });
    const gifWorkers = await workersOf(/gif\.worker/);
    await recordButton.dblclick();
    await page.waitForTimeout(2500);
    await page.unroute(recorderChunk);
    const recorders = (await workersOf(/gif\.worker/)) - gifWorkers;
    record(recorders === 1, "a double click on Record GIF makes one recording", `${recorders} recorder(s)`);
    const [gif] = await Promise.all([
      page.waitForEvent("download", { timeout: 30000 }).catch(() => null),
      page.getByRole("button", { name: /^Stop \(/ }).click().catch(() => {}),
    ]);
    const gifName = gif?.suggestedFilename() || "";
    const gifPath = gif ? await gif.path().catch(() => null) : null;
    const gifBytes = gifPath ? statSync(gifPath).size : 0;
    record(/\.gif$/.test(gifName) && gifBytes > 1000, "the view records as a GIF", `${gifName || "no download"} (${(gifBytes / 1024).toFixed(0)} KB)`);
  } else {
    record(false, "the Record GIF button is offered");
  }
  await page.locator(".toggle__label", { hasText: /^Show people leaving$/ }).first().click();
  await page.waitForTimeout(300);
  record(!(await page.evaluate(() => window.__viewer?.overlays?.has("crowd") ?? false)), "switched off, the people go");
} else {
  record(false, "the People leaving section is offered in Route");
}

// ---- 4b''. slime mould -------------------------------------------------------
// Jones's Physarum agents on the walkable graph (core/slimeEgress.js,
// workers/slime.worker.js, 2026-11-11): grown when asked, drawn yellow, the
// start's route along the tubes.

console.log("\n[4b''] Slime mould");
const slimeSection = page.locator("summary, button", { hasText: /^Slime mould/ }).first();
if (await slimeSection.count()) {
  await slimeSection.scrollIntoViewIfNeeded();
  await slimeSection.click();
  const tSlime = Date.now();
  await page.getByRole("button", { name: /^Grow the slime mould$/ }).click();
  const slimeLine = await page
    .waitForFunction(() => document.body.innerText.match(/[\d,]+ agents for \d+ steps[^\n]*/)?.[0], null, { timeout: 60000 })
    .then((h) => h.jsonValue())
    .catch(() => "");
  const slimeRoute = await page.evaluate(() => window.__viewer?.overlays?.has("slimePath") ?? false);
  record(/reached an exit/.test(slimeLine) && /along its tubes/.test(slimeLine) && slimeRoute, "the slime mould grows to the exits and the start's route keeps to its tubes", `${slimeLine.slice(0, 180)} (${((Date.now() - tSlime) / 1000).toFixed(1)}s)`);
  await page
    .locator("section.section", { has: page.locator(".section__header", { hasText: /^Slime mould/ }) })
    .getByRole("button", { name: /^Clear$/ })
    .click()
    .catch(() => {});
  await page.waitForTimeout(300);
  record(!(await page.evaluate(() => window.__viewer?.overlays?.has("slimePath") ?? false)), "cleared, the slime route goes");
} else {
  record(false, "the Slime mould section is offered in Route");
}

// ---- 4c. TopologicPy Navigation cross-check ---------------------------------

console.log("\n[4c] TopologicPy Navigation cross-check");
await page.getByRole("tab", { name: /^Graph/ }).click();
await page.waitForTimeout(300);
await page.getByRole("button", { name: /Cross-check with TopologicPy/ }).click();
await page.waitForTimeout(300);
const navBtn = page.getByRole("button", { name: /^Run Navigation cross-check$/ });
if (await navBtn.isEnabled()) {
  // Playwright exposes no body for a multipart request holding a Blob, so
  // whether the coordination matrix was sent is read from the server's answer
  // (alignment === "coordination") instead of from the request.
  const navResponse = page.waitForResponse(
    (res) => res.url().includes("/api/ifc/navigation-check"),
    { timeout: 240000 },
  );
  const t1 = Date.now();
  await navBtn.click();
  const nav = await navResponse
    .then(async (res) => (res.ok() ? res.json() : { error: `${res.status()} ${await res.text()}` }))
    .catch((e) => ({ error: String(e) }));
  record(!nav.error, "Navigation cross-check returned", nav.error || `${((Date.now() - t1) / 1000).toFixed(1)}s`);
  if (!nav.error) {
    timing(
      `TopologicPy Navigation cross-check${nav.cached ? " (server cache)" : ""}`,
      Date.now() - t1,
    );
  }
  if (!nav.error) {
    console.log(
      `    navigation: ${nav.navigation_nodes} nodes, ${nav.navigation_components} parts; ` +
        `alignment ${nav.alignment} (median ${nav.residual_median} m); ` +
        `its covered by ours ${nav.navigation_covered}, ours covered by it ${nav.studio_covered}`,
    );
    nav.storeys.forEach((s) => console.log(`    storey ${s.name}: ${JSON.stringify(s)}`));
    if (nav.route) console.log(`    route: ${JSON.stringify({ ...nav.route, points: nav.route.points.length })}`);
    record(
      nav.alignment === "coordination",
      "viewer sent its coordination matrix; graphs aligned exactly, not fitted",
    );
    record(nav.navigation_covered > 0.8, "our graph covers most of Navigation's", String(nav.navigation_covered));
    const shown = await page
      .waitForFunction(() => /our nodes it covers/i.test(document.body.innerText), null, { timeout: 10000 })
      .then(() => true)
      .catch(() => false);
    record(shown, "cross-check stats rendered");
    await page.locator(".stat", { hasText: /our nodes it covers/i }).scrollIntoViewIfNeeded().catch(() => {});
  }
  await shot(page, "navigation-check");
} else {
  record(false, "Run Navigation cross-check was enabled", "needs a graph, the file and server support");
}

// ---- 5. fire simulation -----------------------------------------------------

console.log("\n[5] Fire simulation");
await page.getByRole("tab", { name: /^Simulate/ }).click();
await page.waitForTimeout(300);
const gotFire = await pickAt("Fire", 0.50, 0.50);
record(gotFire, "fire origin placed");

const startFire = page.getByRole("button", { name: /^Start fire$/ });
if (gotFire && (await startFire.isEnabled())) {
  const tFire = Date.now();
  await startFire.click();
  const ran = await page
    .waitForFunction(() => /fire step \d+/i.test(document.body.innerText), null, {
      timeout: 45000,
    })
    .then(() => true)
    .catch(() => false);
  record(ran, "fire simulation started streaming");
  if (ran) timing("Start fire (click to first step on screen)", Date.now() - tFire);
  await page.waitForTimeout(4000);
  await shot(page, "fire-running");
  await page.getByRole("button", { name: /^Stop$/ }).click().catch(() => {});
} else {
  record(false, "fire start button was enabled", "needs a fire origin");
}

// ---- 5b. reinforcement learning ---------------------------------------------
//
// Added 2026-10-27. RL used to seed the fire at the evacuee's own start point
// and never reached the exit on a real model; nothing exercised it here.

console.log("\n[5b] Reinforcement learning");
await page.getByRole("button", { name: /Train an agent against this fire/ }).click();
await page.waitForTimeout(300);
const trainBtn = page.getByRole("button", { name: /^Train policy$/ });
if (await trainBtn.isEnabled()) {
  // RL trains in the browser now. That it used the picked fire origin shows
  // as the "Escape possible" stat, which appears only when a fire was seeded.
  // A slower fire (0.25 links per move), so an escape usually exists and the
  // shield's guarantee is exercised, not met vacuously.
  await page.getByLabel("Fire speed (links per move)").fill("0.25").catch(() => {});
  const tRl = Date.now();
  await trainBtn.click();
  // innerText applies the stat labels' text-transform: uppercase, so match
  // case-insensitively.
  const stats = () =>
    page.evaluate(() => {
      const out = {};
      document.querySelectorAll(".stat").forEach((el) => {
        const k = el.querySelector(".stat__label")?.textContent?.trim();
        const v = el.querySelector(".stat__value")?.textContent?.trim();
        if (k) out[k] = v;
      });
      return out;
    });
  // The default, shielded learner (rlSafe.js).
  const trained = await page
    .waitForFunction(() => /certified earliest/i.test(document.body.innerText), null, {
      timeout: 120000,
    })
    .then(() => true)
    .catch(() => false);
  record(trained, "RL training returned and rendered its stats");
  if (trained) timing("Train RL policy (click to stats)", Date.now() - tRl);
  if (trained) {
    await page.locator(".stat", { hasText: /certified earliest/i }).scrollIntoViewIfNeeded().catch(() => {});
    const r = await stats();
    console.log(`    rl (shielded): ${JSON.stringify(r)}`);
    // Whether an escape exists depends on where the fire was clicked, so the
    // contract is that the UI states it, not that the agent always escapes.
    record(r["Escape possible"] !== undefined, "RL reports whether an escape exists", r["Escape possible"]);
    // The shield's guarantee: when an escape exists, the agent reaches an exit.
    const escapes = /^yes/.test(r["Escape possible"] || "");
    record(!escapes || r["Reached exit"] === "yes", "the shielded agent escapes whenever an escape exists", `${r["Escape possible"]}; reached ${r["Reached exit"]}; certified ${r["Certified earliest"]}`);
  }
  await shot(page, "rl-trained");
  // And the Q-learning it replaces, still offered for comparison.
  await page.getByRole("group", { name: "Learning method" }).getByRole("button", { name: "Q-learning" }).click();
  await trainBtn.click();
  const trainedQ = await page
    .waitForFunction(() => /late-training success/i.test(document.body.innerText), null, { timeout: 120000 })
    .then(() => true)
    .catch(() => false);
  const q = trainedQ ? await stats() : {};
  record(
    q["Return vs best within the step limit"] !== undefined,
    "Q-learning reports its return against the exact optimum (backward induction)",
    q["Return vs best within the step limit"],
  );
  await page.getByRole("group", { name: "Learning method" }).getByRole("button", { name: "Shielded (proved)" }).click();
  // Moving the fire while the agent trains drops the training and its
  // answer: still training, it drew a route for the old points when it ended
  // (SM-3). The fire is placed again where it is, by its coordinates, which
  // counts as a move.
  const fireAt = ((await page.locator(".picker", { hasText: "Fire origin" }).locator(".picker__coord").first().textContent().catch(() => "")) || "").split(",").map((v) => v.trim());
  if (fireAt.length === 3 && fireAt.every((v) => Number.isFinite(Number(v)))) {
    await page.getByText("Place without the mouse", { exact: true }).first().click();
    for (const [i, axis] of ["x", "y", "z"].entries()) await page.getByLabel(axis, { exact: true }).first().fill(fireAt[i]);
    await trainBtn.click();
    // Still training as the fire moves, or done already on a fast machine: either way the answer goes.
    const training = await page.evaluate(() => [...document.querySelectorAll("button")].some((b) => /Train policy/.test(b.textContent) && b.getAttribute("aria-busy") === "true"));
    await page.getByRole("button", { name: /^Place fire origin$/ }).click();
    await page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /^Train policy$/.test(b.textContent) && !b.disabled), null, { timeout: 120000 }).catch(() => {});
    await page.waitForTimeout(500);
    const stale = /certified earliest/i.test(await page.locator(".panel__scroll").first().innerText().catch(() => ""));
    record(!stale, "moving the fire while the agent trains drops its answer", `${stale ? "the old points' answer was shown" : "dropped"} (${training ? "moved while training" : "training had ended"})`);
    await page.getByText("Place without the mouse", { exact: true }).first().click();
  }
} else {
  record(false, "Train policy was enabled", "needs start and exit");
}

// ---- 5c. planning with the fire forecast -------------------------------------
//
// Added 2026-10-27: the forecast routers and the escape-deadline map
// (src/core/forecast.js, src/core/deadline.js) run in their own worker.

console.log("\n[5c] Plan with the forecast");
// The exit doors come from the IFC's own flags, read by the import worker
// (src/core/ifcSemantics.js); the Duplex flags four doors IsExternal. The
// forecast offers the exits out of the building among them.
const exitChoice = await page
  .getByText(/^All \d+ exits? out of the building$/)
  .first()
  .textContent({ timeout: 15000 })
  .catch(() => null);
record(Boolean(exitChoice), "exit doors read from the model's own flags", exitChoice || "none offered");
const planBtn = page.getByRole("button", { name: /^Plan with the forecast$/ });
if (await planBtn.isEnabled().catch(() => false)) {
  const tPlan = Date.now();
  await planBtn.click();
  const planned = await page
    .waitForFunction(
      () =>
        /safe route: out in|above the dose limit|no safe route within the forecast/i.test(
          document.querySelector('[aria-label="Plan with the forecast"]')?.innerText || "",
        ),
      null,
      { timeout: 120000 },
    )
    .then(() => true)
    .catch(() => false);
  record(planned, "planning with the forecast returned and rendered its result");
  if (planned) {
    timing("Plan with the forecast + deadline map (click to result)", Date.now() - tPlan);
    const card = await page.evaluate(() => {
      const el = document.querySelector('[aria-label="Plan with the forecast"]');
      const out = { title: el?.querySelector(".result__title")?.textContent?.trim() };
      el?.querySelectorAll(".result__item").forEach((item) => {
        out[item.querySelector(".result__label")?.textContent?.trim()] = item.querySelector(".result__value")?.textContent?.trim();
      });
      return out;
    });
    console.log(`    forecast: ${JSON.stringify(card)}`);
    record(Boolean(card["Leave within"]) && Boolean(card["Forecast"]), "the plan states how long one can wait and the forecast's length");
    const legendShown = await page.getByText("Escape deadline", { exact: true }).isVisible().catch(() => false);
    record(legendShown || MOBILE, "the escape-deadline map has a legend", MOBILE ? "legend folded on a phone" : undefined);
    await page.locator('[aria-label="Plan with the forecast"]').scrollIntoViewIfNeeded().catch(() => {});
    await shot(page, "forecast-plan");
    // The performance-based check (core/performanceCheck.js, 2026-11-07):
    // the plan's escape deadlines against detection, alarm and pre-movement,
    // a verdict per storey.
    await page.getByRole("tab", { name: /^Route/ }).click();
    await page.getByText("Performance check (with the fire)", { exact: true }).click();
    await page.getByRole("button", { name: "Check against the fire" }).click();
    // The check loads its modules on demand: wait for its verdict.
    await page
      .waitForFunction(() => /Occupants must be able to set off/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 20000 })
      .catch(() => {});
    const perfText = await page
      .locator(".panel__scroll")
      .first()
      .innerText()
      .catch(() => "");
    const perfStoreys = (perfText.match(/(Pass|Fail|Cannot tell) Storey \d/g) || []).length;
    // The office preset and the base-case criterion by default; a graph fire has no smoke, so detection is the time set.
    record(
      perfStoreys > 0 && /Occupants must be able to set off up to \d+ s/.test(perfText) && /only a Flamescapes run gives smoke/.test(perfText) && /Base case/.test(perfText),
      "the plan's deadlines are checked against detection, alarm and pre-movement, per storey",
      `${perfStoreys} storeys`,
    );
    const perfVerdict = perfText.match(/(Pass|Fail|Cannot tell)\. Occupants[^\n]*/)?.[0];
    if (perfVerdict) console.log(`    performance: ${perfVerdict.slice(0, 220)}`);
    // Queues at doors, the measured spread beside the design and the chart of
    // both, by default; the check joins the report (2026-11-07).
    const chart = await page.locator('svg[aria-label^="Share of occupants who have started"]').count();
    record(
      /Queues: \d+ occupants; the last to start waits up to \d+ s/.test(perfText) && /With the measured spread/.test(perfText) && chart > 0 && /Report: 1 fire\b/.test(perfText),
      "the check adds queues at doors and the measured pre-movement beside the design, and starts a report",
      (perfText.match(/Queues: [^.]*\./) || ["no queue line"])[0],
    );
    // More fires for the report, on every storey, each planned in the worker.
    const tFires = Date.now();
    await page.getByRole("button", { name: "Add fires on every storey" }).click();
    await page
      .waitForFunction(() => /Report: [2-9] fires/.test(document.querySelector(".panel__scroll")?.innerText || "") && !/Planning fire \d+ of/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, {
        timeout: 120000,
      })
      .catch(() => {});
    const reportText = await page.locator(".panel__scroll").first().innerText().catch(() => "");
    const fires = Number(reportText.match(/Report: (\d+) fires/)?.[1] || 0);
    record(fires >= 2, "fires on every storey join one report", `${fires} fires in ${((Date.now() - tFires) / 1000).toFixed(1)}s`);
    if (fires >= 2) timing("performance report, fires added", Date.now() - tFires);
    const [reportDownload] = await Promise.all([
      page.waitForEvent("download", { timeout: 30000 }).catch(() => null),
      page.getByRole("button", { name: "Download the report" }).click().catch(() => {}),
    ]);
    const reportName = reportDownload?.suggestedFilename() || "";
    const reportPath = reportDownload ? await reportDownload.path().catch(() => null) : null;
    const reportHtml = reportPath ? readFileSync(reportPath, "utf8") : "";
    record(
      /-performance-report\.html$/.test(reportName) && new RegExp(`Verdict over ${fires} fires`).test(reportHtml) && /Inputs and sources/.test(reportHtml),
      "the report downloads as one page with every fire, its inputs and sources",
      reportName || "no download",
    );
    await shot(page, "performance-report");
    // The report's Stop leaves a plan running beside it alone. They shared a
    // worker, and the plan stayed busy, its button spinning, for good (SM-2).
    // Longer fires (2000 steps) keep both at work while the steps change.
    const addFires = page.getByRole("button", { name: "Add fires on every storey" });
    if (await addFires.isEnabled().catch(() => false)) {
      await page.getByRole("tab", { name: /^Simulate/ }).click();
      await openTiming();
      const stepsField = page.getByLabel("Steps", { exact: true });
      const stepsBefore = await stepsField.inputValue();
      await stepsField.fill("2000");
      await stepsField.blur();
      await page.getByRole("tab", { name: /^Route/ }).click();
      await openSection("Performance check (with the fire)");
      await addFires.click();
      await page.getByRole("tab", { name: /^Simulate/ }).click();
      await planBtn.click();
      await page.getByRole("tab", { name: /^Route/ }).click();
      await openSection("Performance check (with the fire)");
      const stopped = await page
        .locator(".panel__scroll")
        .getByRole("button", { name: /^Stop$/ })
        .click({ timeout: 3000 })
        .then(() => true)
        .catch(() => false);
      await page.getByRole("tab", { name: /^Simulate/ }).click();
      const planFree = await page
        .waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /^Plan with the forecast$/.test(b.textContent) && !b.disabled), null, { timeout: 120000 })
        .then(() => true)
        .catch(() => false);
      if (stopped) record(planFree, "stopping the report's fires leaves a plan beside them to finish", planFree ? "plan usable" : "plan stuck busy");
      else console.log("    (the report's fires ended before Stop: not checked)");
      await openTiming();
      await stepsField.fill(stepsBefore);
      await stepsField.blur();
    }
    await page.getByRole("tab", { name: /^Simulate/ }).click();
  }
} else {
  record(false, "Plan with the forecast was enabled", "needs start, exit and fire origin");
}

// ---- 5d. the two-zone room model ---------------------------------------------
//
// Added 2026-10-28 (gap LG8; src/core/rooms.js, src/core/zone.js): offered only
// on a model with IfcSpace rooms (the Duplex has 21). The same routers plan on
// its forecast, the result card states the model's limits, and the live
// simulation plays its gas temperatures.

console.log("\n[5d] Two-zone room model");
const zoneChoice = page.getByText("Two-zone rooms (degrees C)", { exact: true });
const zoneOffered = await zoneChoice.isVisible().catch(() => false);
record(zoneOffered, "the two-zone model is offered on a model with IfcSpace rooms");
if (zoneOffered) {
  await zoneChoice.click();
  await page.waitForTimeout(200);
  if (await planBtn.isEnabled().catch(() => false)) {
    const tZone = Date.now();
    await planBtn.click();
    const zoned = await page
      .waitForFunction(() => /smoke reaches/i.test(document.querySelector('[aria-label="Plan with the forecast"]')?.innerText || ""), null, {
        timeout: 120000,
      })
      .then(() => true)
      .catch(() => false);
    record(zoned, "planning on the two-zone forecast returned and rendered its result");
    if (zoned) {
      timing("Plan with the two-zone forecast + deadline map (click to result)", Date.now() - tZone);
      const card = await page.evaluate(() => {
        const el = document.querySelector('[aria-label="Plan with the forecast"]');
        const out = { title: el?.querySelector(".result__title")?.textContent?.trim() };
        el?.querySelectorAll(".result__item").forEach((item) => {
          out[item.querySelector(".result__label")?.textContent?.trim()] = item.querySelector(".result__value")?.textContent?.trim();
        });
        return out;
      });
      console.log(`    zone: ${JSON.stringify(card)}`);
      const text = await page.locator('[aria-label="Plan with the forecast"]').innerText();
      record(Boolean(card["Flashover"]) && Boolean(card["Smoke reaches"]), "the card reports flashover and the rooms the smoke reaches", `${card["Flashover"]}; ${card["Smoke reaches"]}`);
      record(/not a substitute for CFAST, FDS or an engineer/i.test(text), "the card states the model's limits");
      await page.locator('[aria-label="Plan with the forecast"]').scrollIntoViewIfNeeded().catch(() => {});
      await shot(page, "zone-plan");
    }
  } else {
    record(false, "Plan with the forecast was enabled for the two-zone model");
  }
  if (await startFire.isEnabled().catch(() => false)) {
    await page.getByRole("button", { name: /^Start fire$/ }).scrollIntoViewIfNeeded().catch(() => {});
    await startFire.click();
    const played = await page
      .waitForFunction(() => /fire step \d+/i.test(document.body.innerText), null, { timeout: 45000 })
      .then(() => true)
      .catch(() => false);
    await page.waitForTimeout(3000);
    const legend = await page.getByText("Gas temperature (°C)", { exact: true }).isVisible().catch(() => false);
    record(played, "the live simulation plays the two-zone model", legend ? "gas-temperature legend shown" : "legend not visible");
    await shot(page, "zone-fire-running");
    await page.getByRole("button", { name: /^Stop$/ }).click().catch(() => {});
    // A double click on Start while the two-zone frames compute starts the
    // fire once. Each click computed the frames and started the fire, the
    // second from step 0 again (SM-2).
    const zoneJobs = await jobsOf("zoneFrames");
    await startFire.dblclick();
    await page.waitForFunction(() => /fire step \d+/i.test(document.body.innerText), null, { timeout: 45000 }).catch(() => {});
    await page.waitForTimeout(1500);
    const zoneStarts = (await jobsOf("zoneFrames")) - zoneJobs;
    record(zoneStarts === 1, "a double click on Start fire starts the two-zone fire once", `${zoneStarts} start(s)`);
    await page.getByRole("button", { name: /^Stop$/ }).click().catch(() => {});
  }
  // Back to heat diffusion for the steps that follow.
  await page.getByText("Heat diffusion", { exact: true }).click().catch(() => {});
}

// ---- 5d'. people leaving with a graph fire started again --------------------
//
// With no step delay, starting the fire again flipped the people's walk from
// "with the fire" to "no fire yet" and back within the walk's 30 ms pause:
// the panel was left on the dropped one, with nobody drawn (SM-3).

console.log("\n[5d'] People leaving with a fire started again");
if (await startFire.isEnabled().catch(() => false)) {
  await page.getByRole("tab", { name: /^Route/ }).click();
  const showPeople = page.locator(".toggle__label", { hasText: /^Show people leaving$/ }).first();
  const openPeople = () => openSection("People leaving");
  await openPeople();
  await showPeople.click();
  await page.getByLabel("Egress").selectOption("fire").catch(() => {});
  await page.getByRole("tab", { name: /^Simulate/ }).click();
  await openTiming();
  const delayField = page.getByLabel("Step delay (ms)", { exact: true });
  const stepsField = page.getByLabel("Steps", { exact: true });
  const [delayBefore, stepsBefore] = [await delayField.inputValue(), await stepsField.inputValue()];
  await delayField.fill("0");
  await delayField.blur();
  await stepsField.fill("30");
  await stepsField.blur();
  const fireEnded = () =>
    page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /^Start fire$/.test(b.textContent) && !b.disabled), null, { timeout: 60000 }).catch(() => {});
  await startFire.click();
  await page.waitForTimeout(500);
  await fireEnded();
  await page.waitForTimeout(1000);
  await startFire.click();
  await page.waitForTimeout(500);
  await fireEnded();
  await page.waitForTimeout(1500);
  const peopleDrawn = await page.evaluate(() => window.__viewer?.overlays?.has("crowd") ?? false);
  record(peopleDrawn, "people leaving with the fire are drawn after the fire is started again", peopleDrawn ? "drawn" : "nobody drawn");
  await delayField.fill(delayBefore);
  await delayField.blur();
  await stepsField.fill(stepsBefore);
  await stepsField.blur();
  await page.getByRole("tab", { name: /^Route/ }).click();
  await openPeople();
  await page.getByLabel("Egress").selectOption("none").catch(() => {});
  await showPeople.click().catch(() => {});
  await page.getByRole("tab", { name: /^Simulate/ }).click();
}

// ---- 5e. the optional Flamescapes solver --------------------------------------
//
// Added 2026-11-05: switched on, it adds two fire models run by the
// Flamescapes solver in its own worker (src/core/flamescapes.js). A short run
// must stream frames, finish, colour the graph and say what it ran on.

console.log("\n[5e] Flamescapes solver (optional)");
const fsToggle = page.getByText("Flamescapes combustion solver (optional)", { exact: true });
if (await fsToggle.isVisible().catch(() => false)) {
  await fsToggle.click();
  await page.getByText("Flamescapes: fire, smoke and embers", { exact: true }).click();
  await page.getByText("Flamescapes settings", { exact: true }).click();
  await page.getByLabel("Simulated time (s)").fill("30");
  await page.getByLabel("Simulated time (s)").blur();
  const tFs = Date.now();
  await page.getByRole("button", { name: /^Start fire$/ }).scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: /^Start fire$/ }).click();
  // The doors switch stays usable while the fire runs, and offers a restart
  // with the new doors (it was locked until the page was reloaded, 2026-11-06).
  // Set back at once, so this run carries on with the doors open.
  await page.waitForTimeout(800);
  const doorsGroup = page.getByRole("group", { name: "Doors", exact: true });
  await doorsGroup.getByRole("button", { name: "Closed" }).click({ timeout: 5000 }).catch(() => {});
  const offered = await page.getByRole("button", { name: /^Restart the fire with the doors closed$/ }).count();
  record(offered === 1, "the doors can be switched while the fire runs, with a restart offered", offered ? "offered" : "not offered");
  await doorsGroup.getByRole("button", { name: "Open" }).click({ timeout: 5000 }).catch(() => {});
  const fsDone = await page
    .waitForFunction(() => /Flamescapes run complete/.test(document.body.innerText), null, { timeout: 240000 })
    .then(() => true)
    .catch(() => false);
  record(fsDone, "a Flamescapes run streams its frames and completes", `${((Date.now() - tFs) / 1000).toFixed(1)}s`);
  if (fsDone) timing("Flamescapes, 30 s simulated (click to complete)", Date.now() - tFs);
  const card = await page.locator('[aria-label="Flamescapes run"]').innerText().catch(() => "");
  record(/Running on (the GPU|one CPU core)/.test(card), "the run says what it ran on", card.split("\n").find((l) => /Running on/.test(l)) || "no backend line");
  // The standard grid by default, the same on every device; a small CPU builds a lighter one and says so (lib/hardware.js, 2026-11-09).
  const gridNote = card.match(/The standard grid, the same on every device\.|This device's CPU builds a lighter grid \(at most [\d,]+ cells\)/)?.[0] || "";
  record(Boolean(gridNote), "the run is on the standard grid, and the card says so", gridNote || "no grid note");
  // This run asked for embers: the card says they are on (an "off" run says "off").
  const embersLine = card.split("\n").find((l) => /^on: \d+ thrown|^off$/.test(l.trim())) || "";
  record(/^on: \d+ thrown/.test(embersLine.trim()), "the run card says embers are on", embersLine.trim() || "no embers line");
  // The default openings: doors open, windows closed.
  const openings = card.split("\n").find((l) => /doors (open|closed), \d+ windows (open|closed)/.test(l)) || "";
  record(/doors open, \d+ windows closed/.test(openings), "the run card gives the openings", openings.trim() || "no openings line");
  // The smoke, gas and tenability measures (core/tenability.js, core/measures.js; 2026-11-06).
  const measuresLine = card.split("\n").find((l) => /^heat, smoke, CO/.test(l.trim())) || "";
  record(/CO₂, O₂/.test(measuresLine), "the run records soot, CO₂ and O₂ too", measuresLine.trim() || "no measures line");
  // The first item as a t-squared design fire by default (fireClasses.js designFireFor, 2026-11-08).
  const itemLine = card.split("\n").find((l) => /t-squared fire/.test(l)) || "";
  record(/^a medium t-squared fire \(0\.0117 kW\/s²\), 20 kg/.test(itemLine.trim()), "the first item grows as its class's design fire, and the card says so", itemLine.trim() || "no first-item line");
  // The burning item is drawn as flames from the first frame (flamescapesRun.js
  // withBurnerFlame, 2026-11-08): its burner makes no flame-front particles of
  // its own, so until then a design fire showed none before the contents caught.
  const flamePoints = await page
    .waitForFunction(() => (window.__viewer?.overlays?.get("flames")?.geometry?.drawRange?.count ?? 0) > 0, null, { timeout: 15000 })
    .then(() => page.evaluate(() => window.__viewer.overlays.get("flames").geometry.drawRange.count))
    .catch(() => 0);
  record(flamePoints > 0, "the burning first item is drawn as flames", `${flamePoints} flame points`);
  // Virtual smoke detectors under the ceiling round the fire (flamescapesRun.js ceilingDetectors, 2026-11-07).
  const detectorLine = card.split("\n").find((l) => /under the ceiling, 7 m from the fire/.test(l)) || "";
  record(/^\d+ under the ceiling/.test(detectorLine.trim()), "the run reads smoke at ceiling detectors round the fire", detectorLine.trim() || "no detector line");
  const showGroup = page.getByRole("group", { name: "Show on the graph", exact: true });
  await showGroup.getByRole("button", { name: "Tenability" }).click().catch(() => {});
  await page.waitForTimeout(400);
  const indexLegend = await page.getByText("Tenability index", { exact: true }).isVisible().catch(() => false);
  record(indexLegend || MOBILE, "the graph can be coloured by the tenability index", MOBILE ? "legend folded on a phone" : undefined);
  await showGroup.getByRole("button", { name: "Heat" }).click().catch(() => {});
  // The exports are offered once the playback has ended too.
  await page.waitForFunction(() => ![...document.querySelectorAll("button")].some((b) => /^Stop fire$/.test(b.textContent)), null, { timeout: 180000 }).catch(() => {});
  const tableButton = page.getByRole("button", { name: "Export the run as a table (CSV)" });
  const [tableDownload] = await Promise.all([page.waitForEvent("download", { timeout: 30000 }).catch(() => null), tableButton.click().catch(() => {})]);
  const tableName = tableDownload?.suggestedFilename() || "";
  record(/\.csv(\.gz)?$/.test(tableName), "the run exports as a table for analysis", tableName || "no download");
  const legendFs = await page.getByText("Gas temperature at head height (°C)", { exact: true }).isVisible().catch(() => false);
  record(legendFs || MOBILE, "the graph is coloured by gas temperature at head height", MOBILE ? "legend folded on a phone" : undefined);
  await shot(page, "flamescapes-run");
  // Kept runs (core/runCache.js, 2026-11-07): after another run, this one
  // asked for again plays from the browser's store, not from memory.
  const startFs = async () => {
    await page.getByRole("button", { name: /^Start fire$/ }).scrollIntoViewIfNeeded();
    await page.getByRole("button", { name: /^Start fire$/ }).click();
    await page.waitForFunction(() => /Flamescapes run complete/.test(document.querySelector('[aria-label="Flamescapes run"]')?.innerText || ""), null, { timeout: 240000 }).catch(() => {});
    await page.waitForFunction(() => ![...document.querySelectorAll("button")].some((b) => /^Stop fire$/.test(b.textContent)), null, { timeout: 180000 }).catch(() => {});
  };
  await page.waitForTimeout(6000);
  const burns = page.getByLabel("What burns first");
  const firstClass = await burns.inputValue().catch(() => "A");
  await burns.selectOption(firstClass === "B" ? "A" : "B").catch(() => {});
  await startFs();
  await burns.selectOption(firstClass).catch(() => {});
  const tReplay = Date.now();
  await startFs();
  const replayCard = await page.locator('[aria-label="Flamescapes run"]').innerText().catch(() => "");
  record(/Played from this browser/.test(replayCard), "the same run asked for again plays from the browser's kept runs", `${((Date.now() - tReplay) / 1000).toFixed(1)}s`);
  // Repeat runs (settings.flamescapesRepeats, 2026-11-09): on the GPU, after
  // the run, three more on its grid and the card gives the range; on the CPU none.
  const onGpu = /Running on the GPU/.test(card);
  await page.getByLabel("Repeat runs on the GPU").selectOption("4").catch(() => {});
  await burns.selectOption(firstClass === "B" ? "A" : "B").catch(() => {});
  const tRepeat = Date.now();
  await startFs();
  const repeated = onGpu
    ? await page
        .waitForFunction(() => /\n4 runs, the median and its range/.test(document.querySelector('[aria-label="Flamescapes run"]')?.innerText || ""), null, { timeout: 180000 })
        .then(() => true)
        .catch(() => false)
    : true;
  const repeatCard = await page.locator('[aria-label="Flamescapes run"]').innerText().catch(() => "");
  const repeatLine = repeatCard.split("\n").find((l) => /runs.*the median and its range/.test(l)) || "";
  record(
    onGpu ? repeated && /nodes untenable/.test(repeatLine) : !/Repeat runs/.test(repeatCard),
    onGpu ? "repeat runs on the GPU give the range on the card" : "no repeat runs on the CPU",
    onGpu ? `${repeatLine.trim().slice(0, 200) || "no range"} (${((Date.now() - tRepeat) / 1000).toFixed(1)}s)` : "CPU run, none made",
  );
  // Stop ends the repeat runs too: they ran on after Stop, holding the GPU
  // (SM-1). A new length makes a new run; the GPU computes it faster than it
  // plays, so its repeats start while it is still on screen.
  const simulated = page.getByLabel("Simulated time (s)");
  if (onGpu) {
    await page.getByLabel("Repeat runs on the GPU").selectOption("8").catch(() => {});
    await simulated.fill("33");
    await simulated.blur();
    await page.getByRole("button", { name: /^Start fire$/ }).click();
    const repeating = await page
      .waitForFunction(() => /of 8 runs so far/.test(document.querySelector('[aria-label="Flamescapes run"]')?.innerText || ""), null, { timeout: 180000 })
      .then(() => true)
      .catch(() => false);
    const stopped = await page
      .getByRole("button", { name: /^Stop$/ })
      .click({ timeout: 3000 })
      .then(() => true)
      .catch(() => false);
    const ended = await page
      .waitForFunction(() => /of 8 runs \(the rest stopped\)/.test(document.querySelector('[aria-label="Flamescapes run"]')?.innerText || ""), null, { timeout: 10000 })
      .then(() => true)
      .catch(() => false);
    if (repeating && stopped) record(ended, "Stop ends the repeat runs too", ended ? "the rest stopped" : "they ran on");
    else console.log(`    (repeat runs ${repeating ? "ran" : "did not start"}, Stop ${stopped ? "pressed" : "not offered"}: not checked)`);
  }
  await page.getByLabel("Repeat runs on the GPU").selectOption("1").catch(() => {});
  await burns.selectOption(firstClass).catch(() => {});
  // Start fire while a plan's run computes plays that run. It was thrown away
  // and computed again from the start (SM-1). A new length makes a new run.
  await simulated.fill("31");
  await simulated.blur();
  const solversBeforePlan = await workersOf(/flamescapes\.worker/);
  await planBtn.click();
  // The plan's run is computing: its solver has started.
  await page
    .waitForFunction((n) => window.__workers.filter((url) => /flamescapes\.worker/.test(url)).length > n, solversBeforePlan, { timeout: 60000 })
    .catch(() => {});
  await startFs();
  const planSolvers = (await workersOf(/flamescapes\.worker/)) - solversBeforePlan;
  record(planSolvers === 1, "Start fire while a plan's run computes plays that run", `${planSolvers} solver run(s) for one fire`);
  // Stopped while its kept run is looked up, a run starts nothing. It
  // started its solver once the lookup ended, and no Stop could end it
  // (SM-1). The browser's store of kept runs is held busy so the lookup waits.
  await simulated.fill("32");
  await simulated.blur();
  const held = await page.evaluate(
    () =>
      new Promise((resolve) => {
        const open = indexedDB.open("topologic-studio-runs");
        open.onerror = () => resolve(false);
        open.onsuccess = () => {
          try {
            const tx = open.result.transaction(["meta", "frames"], "readwrite");
            const store = tx.objectStore("meta");
            window.__holdRuns = true;
            const spin = () => {
              if (window.__holdRuns) store.get("hold").onsuccess = spin;
            };
            spin();
            resolve(true);
          } catch {
            resolve(false);
          }
        };
      }),
  );
  const solversBeforeStop = await workersOf(/flamescapes\.worker/);
  await page.getByRole("button", { name: /^Start fire$/ }).click();
  await page.waitForTimeout(400);
  await page.getByRole("button", { name: /^Stop$/ }).click().catch(() => {});
  await page.evaluate(() => {
    window.__holdRuns = false;
  });
  await page.waitForTimeout(3000);
  const afterStop = (await workersOf(/flamescapes\.worker/)) - solversBeforeStop;
  if (held) record(afterStop === 0, "a fire stopped while its kept run is looked up starts no solver", `${afterStop} solver run(s) after Stop`);
  else console.log("    (the kept runs could not be held: not checked)");
  await simulated.fill("30");
  await simulated.blur();
  // Sprinklers (core/sprinklers.js, core/suppression.js, 2026-11-07): off by
  // default; on, the Duplex's rooms get heads by NFPA 13 with a rule report,
  // and a run reports when they open.
  await page.locator("summary", { hasText: /^Sprinklers$/ }).first().click().catch(() => {});
  const sprinklerSwitch = page.locator(".toggle__label", { hasText: /^Sprinklers$/ });
  await sprinklerSwitch.first().click().catch(() => {});
  await page.waitForFunction(() => /\d+ sprinkler heads/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 30000 }).catch(() => {});
  const layoutText = await page.locator(".panel__scroll").first().innerText().catch(() => "");
  const layoutLine = layoutText.match(/\d+ sprinkler heads[^\n]*/)?.[0] || "";
  record(/\d+ sprinkler heads, placed by NFPA 13 in \d+ rooms/.test(layoutLine), "sprinklers are laid out per room by NFPA 13, with a rule report", layoutLine.slice(0, 160) || "no layout");
  // A medium design fire opens no head in 30 s: this run is 90 s with an
  // ultra-fast item (1 MW in 75 s), then the settings go back.
  const growthSelect = page.getByLabel("How it grows");
  await growthSelect.selectOption("ultra-fast").catch(() => {});
  await page.getByLabel("Simulated time (s)").fill("90");
  await page.getByLabel("Simulated time (s)").blur();
  const tSpr = Date.now();
  await startFs();
  const sprCard = await page.locator('[aria-label="Flamescapes run"]').innerText().catch(() => "");
  const sprLine = sprCard.split("\n").find((l) => /^\d+ heads placed per room/.test(l.trim())) || "";
  record(/the first opens at .* by the estimate/.test(sprLine), "a run with sprinklers says when the heads open by both readings", `${sprLine.trim().slice(0, 180) || "no sprinkler line"} (${((Date.now() - tSpr) / 1000).toFixed(1)}s)`);
  const sprItem = sprCard.split("\n").find((l) => /t-squared fire/.test(l)) || "";
  record(/sprinkler water on it from \d+ s/.test(sprItem), "the water over the first item decays it", sprItem.trim().slice(0, 180) || "no first-item line");
  await growthSelect.selectOption("class").catch(() => {});
  await page.getByLabel("Simulated time (s)").fill("30");
  await page.getByLabel("Simulated time (s)").blur();
  // Systems by room use (core/suppressionSystems.js, 2026-11-07): off by
  // default; on, every room gets water, clean agent, foam or none by its name
  // and the occupancy, with a report (the Duplex has no agent or foam rooms).
  await page.locator("summary", { hasText: /^Systems by room use$/ }).first().click().catch(() => {});
  const systemsSwitch = page.locator(".toggle__label", { hasText: /^Systems by room use$/ });
  await systemsSwitch.first().click().catch(() => {});
  await page.waitForFunction(() => /\d+ rooms: \d+ water/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 30000 }).catch(() => {});
  const systemsText = await page.locator(".panel__scroll").first().innerText().catch(() => "");
  const systemsLine = systemsText.match(/\d+ rooms: \d+ water[^\n]*/)?.[0] || "";
  record(/\d+ rooms: \d+ water, \d+ clean agent .*, \d+ foam, \d+ none/.test(systemsLine), "systems by room use give every room water, clean agent, foam or none", systemsLine.slice(0, 160) || "no plan");
  await systemsSwitch.first().click().catch(() => {});
  await sprinklerSwitch.first().click().catch(() => {});
  // Back to the default, so later steps see the page as before.
  await page.getByRole("button", { name: /^Stop$/ }).click().catch(() => {});
  await fsToggle.click();
} else {
  record(false, "the Flamescapes toggle is offered in the Simulate step");
}

// ---- 5f. extinguishers -------------------------------------------------------
//
// Off by default; on, the Duplex (no extinguishers in its IFC) gets units
// placed on the walk by the 22.86 m rule and checked, and one person's
// attempt on the fire on show reads put out, not put out or cannot tell.
console.log("\n[5f] Extinguishers");
await page.getByRole("tab", { name: /^Simulate/ }).click();
await page.locator("summary", { hasText: /^Extinguishers$/ }).first().click().catch(() => {});
const extSwitch = page.locator(".toggle__label", { hasText: /^Extinguishers$/ });
const tExt = Date.now();
await extSwitch.first().click().catch(() => {});
await page.waitForFunction(() => /\d+ extinguishers/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 60000 }).catch(() => {});
timing("Extinguishers placed and checked (switch to result)", Date.now() - tExt);
const extText = await page.locator(".panel__scroll").first().innerText().catch(() => "");
const extLine = extText.match(/\d+ extinguishers[^\n]*/)?.[0] || "";
record(/\d+ extinguishers, placed on the walk by the travel rule\. The check: pass/.test(extLine), "extinguishers are placed on the walk and pass the 22.86 m check", extLine.slice(0, 160) || "no layout");
await page.getByRole("button", { name: "Try to put the fire out" }).click().catch(() => {});
const attemptCard = (await page.locator(".linked-model", { hasText: "The attempt" }).innerText().catch(() => "")).replace(/\s+/g, " ");
record(/The attempt (Put out|Not put out|Cannot tell)/.test(attemptCard) && /starts at \d+ s/.test(attemptCard), "  one person's attempt says when they start and whether it is put out", attemptCard.slice(0, 160) || "no card");
await extSwitch.first().click().catch(() => {});

// ---- 6. theme, panels, topology mode ----------------------------------------

console.log("\n[6] UI surfaces");

await page.getByRole("button", { name: "Toggle colour theme" }).click();
await page.waitForTimeout(400);
const theme = await page.evaluate(() => document.documentElement.dataset.theme);
record(theme === "light" || theme === "dark", "theme toggle works", theme);
await shot(page, "theme-light");
await page.getByRole("button", { name: "Toggle colour theme" }).click();
await page.waitForTimeout(300);

for (const tab of ["Model", "Graph", "Route", "Simulate"]) {
  await page.getByRole("tab", { name: new RegExp(`^${tab}`) }).click();
  await page.waitForTimeout(200);
  const visible = await page.locator(".panel__scroll").isVisible();
  record(visible, `workflow step "${tab}" renders`);
  if (tab === "Model") {
    // What the IFC itself says about egress (EgressCheck, ifcSemantics.js).
    const row = await page.locator(".panel__scroll").getByText(/^\d+ of \d+ \(/).first().textContent({ timeout: 5000 }).catch(() => null);
    record(Boolean(row), "the Model step reports the file's exit doors", row || "not shown");
    // What the elements are (identify.js): rules only, no download. The shape
    // classifier loads its runtime from a CDN, so it is checked by hand.
    await page.locator(".panel__scroll").getByText("What the elements are", { exact: true }).click();
    await page.getByRole("button", { name: /^Identify elements$/ }).click();
    const named = await page
      .locator(".panel__scroll")
      .getByText(/generic:|Every element is typed by its class/)
      .first()
      .textContent({ timeout: 30000 })
      .catch(() => null);
    record(Boolean(named), "the Model step identifies the elements", named || "not shown");
    // By shape, when its module cannot load (offline, or a tab open across a
    // deploy): the button is usable again, not spinning for good (SM-3).
    const shapeButton = page.getByRole("button", { name: /^Identify [\d,]+ by shape/ });
    if (await shapeButton.count()) {
      const shapeChunk = /identifyGeometry/;
      await page.route(shapeChunk, (route) => route.abort("aborted"));
      const errorsBefore = consoleErrors.length;
      await shapeButton.click();
      const freed = await page
        .waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /^Identify [\d,]+ by shape/.test(b.textContent) && !b.disabled), null, { timeout: 8000 })
        .then(() => true)
        .catch(() => false);
      await page.unroute(shapeChunk);
      // The error it reports is this check's own.
      consoleErrors.splice(errorsBefore);
      record(freed, "identifying by shape ends cleanly when its module cannot load", freed ? "button usable again" : "button stuck");
    }
    // Describe this model: the facts it would send are the app's own, shown before anything is sent.
    await page.locator(".panel__scroll").getByText("Describe this model", { exact: true }).click();
    await page.locator(".panel__scroll").getByText("The facts it is given", { exact: true }).click();
    const facts = (await page.locator(".describe__facts").textContent({ timeout: 10000 }).catch(() => "")) || "";
    record(/Model: .*storeys/.test(facts) && /Doors: \d+/.test(facts), "Describe this model shows the facts it is given", facts.split("\n").slice(0, 2).join(" | ").slice(0, 160));
    // With the graph built, each storey's ways out and longest walk (describeFacts.js, codeCheck.js; 2026-11-08).
    const storeyLine = facts.split("\n").find((l) => /^- Storey .*(longest walk to one is [\d.]+ m|no place reaches a way out)/.test(l)) || "";
    record(Boolean(storeyLine), "the facts give each storey's ways out and longest walk", storeyLine.slice(0, 160) || "no storey line");
  }
}

// Arrow keys move between steps (the tab pattern).
await page.getByRole("tab", { name: /^Simulate/ }).focus();
await page.keyboard.press("ArrowLeft");
await page.waitForTimeout(150);
record(
  await page.evaluate(() => document.activeElement?.dataset.step === "route"),
  "arrow keys move between workflow steps",
);

// Methods: a modal page that Escape closes, returning focus. It loads lazily,
// with KaTeX, which typesets the proved statements.
await page.getByRole("button", { name: /^Methods$/ }).click();
const methodsShown = await page
  .getByRole("dialog", { name: "Methods and verification" })
  .waitFor({ timeout: 20000 })
  .then(() => true, () => false);
record(methodsShown, "Methods page opens");
record(
  await page.getByRole("dialog").getByText(/proved in Lean 4/).isVisible().catch(() => false),
  "Methods page lists the Lean-proved properties",
);
await page.waitForFunction(() => document.querySelectorAll(".equation .katex").length > 0, null, { timeout: 20000 }).catch(() => {});
await page.evaluate(() => document.fonts.ready);
const maths = await page.evaluate(() => {
  const eqs = [...document.querySelectorAll('[role="dialog"] .equation')];
  return {
    count: eqs.length,
    typeset: eqs.filter((e) => e.querySelector(".katex .katex-html") && e.querySelector(".katex-mathml math")).length,
    errors: document.querySelectorAll('[role="dialog"] .katex-error').length,
    fonts: document.fonts.check("16px KaTeX_Main") && document.fonts.check("16px KaTeX_Math"),
    wide: eqs.filter((e) => e.scrollWidth > e.clientWidth + 1).length,
    pageWide: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
  };
});
record(
  maths.count > 20 && maths.typeset === maths.count && maths.errors === 0,
  `every proved statement is typeset with KaTeX, with MathML for screen readers (${maths.typeset}/${maths.count}, ${maths.errors} errors)`,
);
record(maths.fonts, "KaTeX's fonts load from the site's own assets");
record(!maths.wide && !maths.pageWide, `no equation needs sideways scrolling (${maths.wide} wide)`);
await shot(page, "methods");
await page.keyboard.press("Escape");
await page.waitForTimeout(200);
record(await page.getByRole("dialog").count() === 0, "Escape closes the Methods page");

// The legacy Topology JSON mode is not offered on a phone.
if (!MOBILE) {
  await page.getByRole("button", { name: "Topology JSON" }).click();
  await page.waitForTimeout(1200);
  record(
    await page.locator("text=Topology JSON viewer").count() > 0,
    "topology mode renders its placeholder",
  );
  await shot(page, "topology-mode");
  await page.getByRole("button", { name: /^IFC$/ }).click();
  await page.waitForTimeout(500);
}

// ---- 7. reload and reopen: the fragment cache --------------------------------
//
// A returning visitor: same browser, same model. The first open converted the
// IFC and cached the fragments in IndexedDB; this one should skip the parse.

console.log("\n[7] Reload and reopen the same IFC");
const tReload = Date.now();
await page.reload({ waitUntil: "domcontentloaded" });
await page
  .waitForFunction(() => window.__viewer?.ready || !document.body.innerText.includes("Starting the 3D engine"), null, {
    timeout: 90000,
  })
  .catch(() => {});
timing("Reload page to 3D engine ready", Date.now() - tReload);
const tCached = Date.now();
await page.locator('input[type="file"][accept=".ifc"]').first().setInputFiles(IFC);
const fromCache = await page
  .waitForFunction(() => /loaded from cache|local fragment cache/i.test(document.body.innerText), null, {
    timeout: 120000,
  })
  .then(() => true)
  .catch(() => false);
record(fromCache, "second open used the fragment cache", `${((Date.now() - tCached) / 1000).toFixed(1)}s`);
if (fromCache) timing("Open IFC, second time (from cache)", Date.now() - tCached);

// ---- 8. example models shipped with the site ------------------------------
//
// public/examples, offered on the empty viewport. Each must open, build a
// graph and come out as one connected piece (the CC0 examples are designed
// that way; see tools/make_example_ifcs.py).

console.log("\n[8] Example models");
for (const name of ["Courtyard ring", "Office, two storeys"]) {
  await page.reload({ waitUntil: "domcontentloaded" });
  const button = page.locator(".welcome").getByRole("button", { name: new RegExp(`^${name}`) });
  const offered = await button
    .waitFor({ timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  record(offered, `example "${name}" is offered`);
  if (!offered) continue;
  const tEx = Date.now();
  await button.click();
  const opened = await page
    .waitForFunction(
      () =>
        [...document.querySelectorAll("button")].some(
          (b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled,
        ),
      null,
      { timeout: 120000 },
    )
    .then(() => true)
    .catch(() => false);
  record(opened, `example "${name}" opened`, `${((Date.now() - tEx) / 1000).toFixed(1)}s`);
  if (!opened) continue;
  await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
  await page
    .waitForFunction(
      () => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === "route",
      null,
      { timeout: 120000 },
    )
    .catch(() => {});
  await page.getByRole("tab", { name: /^Graph/ }).click();
  await page.waitForTimeout(300);
  const stats = await readStats();
  console.log(`    ${name}: ${JSON.stringify(stats)}`);
  record(stats.Reachable === "100%", `example "${name}" is one connected graph`, stats.Reachable);
  await shot(page, `example-${name.toLowerCase().replace(/[^a-z]+/g, "-")}`);
}

// Added 2026-11-06: the TU Wien escape-route model (CC BY 4.0), a real
// building in Vienna. It must open, read its exit doors from the file
// (32 FireExit, 2 IsExternal, 1 external space boundary) and build a graph
// whose largest part holds at least 85% of it: a few pockets have no door in
// the model, so it is not one piece. 90% on 2026-11-08, once its walls'
// layers were read (shafts closed, a refuse room with only an outside door)
// and links blocked by walls stopped counting as joined; 95% before both.
{
  await page.reload({ waitUntil: "domcontentloaded" });
  const button = page.locator(".welcome").getByRole("button", { name: /^Vienna apartment building/ });
  const offered = await button
    .waitFor({ timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  record(offered, 'example "Vienna apartment building (TU Wien)" is offered');
  if (offered) {
    const tEx = Date.now();
    await button.click();
    const opened = await page
      .waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled), null, { timeout: 180000 })
      .then(() => true)
      .catch(() => false);
    record(opened, 'example "Vienna apartment building (TU Wien)" opened', `${((Date.now() - tEx) / 1000).toFixed(1)}s`);
    if (opened) {
      await page.getByRole("tab", { name: /^Model/ }).click();
      const row = await page.locator(".panel__scroll").getByText(/^\d+ of \d+ \(/).first().textContent({ timeout: 10000 }).catch(() => null);
      record(/^35 of 81 \(/.test(row || ""), "  and reads its exit doors from the file", row || "not shown");
      await page.getByRole("tab", { name: /^Graph/ }).click();
      await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
      await page
        .waitForFunction((s) => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === s, "route", { timeout: 180000 })
        .catch(() => {});
      await page.getByRole("tab", { name: /^Graph/ }).click();
      await page.waitForTimeout(300);
      const stats = await readStats();
      console.log(`    Vienna apartment building: ${JSON.stringify(stats)}`);
      record(parseFloat(stats.Reachable) >= 85, "  and its graph's largest part holds at least 85% of it", stats.Reachable);
      await shot(page, "example-vienna");
      // Of its 35 exit doors, two lead out of the building: the street door
      // and the first-floor fire exit at the back; the flat doors flagged
      // FireExit do not. From the top floor the nearest is the fire exit.
      await page.getByRole("tab", { name: /^Route/ }).click();
      const hint = await page.locator(".panel__scroll").getByText(/exit doors?, from/).first().textContent({ timeout: 10000 }).catch(() => "");
      record(/one of the 2 that lead out of the building/.test(hint), "  and two of its exit doors lead out of the building", hint.replace(/\s+/g, " ").slice(0, 120));
      const rings = await page.evaluate(() => window.__viewer?.overlays?.get("exits")?.count ?? null);
      record(rings === 2, "  drawn as two rings", String(rings));
      await page.getByText("Place without the mouse", { exact: true }).click();
      await page.getByLabel("Point", { exact: true }).selectOption("start");
      for (const [axis, value] of [["x", -10.6], ["y", 12.93], ["z", 3.72]]) {
        await page.getByLabel(axis, { exact: true }).fill(String(value));
        await page.getByLabel(axis, { exact: true }).blur();
      }
      await page.getByRole("button", { name: "Place start" }).click();
      await page.getByRole("button", { name: "Use the nearest exit door" }).click();
      await page.waitForTimeout(300);
      const exitAt = await page.evaluate(() => window.__viewer?.markers?.exit?.position?.toArray().map((x) => Math.round(x * 100) / 100));
      record(Math.hypot(exitAt[0] + 3.57, exitAt[2] + 3.01) < 0.5, "  and the nearest from the top floor is the fire exit, not a flat door", JSON.stringify(exitAt));
      // Doors as built (core/doorStates.js, 2026-11-07): the 14 doors whose
      // EN 13501-2 class carries -C (EI2 30-C, EI2 30-C-Sm) close themselves.
      await page.getByRole("tab", { name: /^Simulate/ }).click();
      const solver = page.getByText("Flamescapes combustion solver (optional)", { exact: true });
      await solver.click();
      await page.getByRole("group", { name: "Doors", exact: true }).getByRole("button", { name: "As built" }).click();
      const asBuilt = await page.getByText(/close themselves are closed/).first().textContent({ timeout: 10000 }).catch(() => "");
      record(/14 of 81 doors here/.test(asBuilt), "  and its self-closing doors are read for an as-built run", (asBuilt.match(/\d+ of \d+ doors here/) || ["not shown"])[0]);
      await page.getByRole("group", { name: "Doors", exact: true }).getByRole("button", { name: "Open" }).click();
      await solver.click();
    }
  }
}

// The examples are converted when the site is built
// (scripts/prepare-examples.mjs): one opens from its fragments, with no IFC
// worker, and still knows its exit doors. Skipped against a build made with
// `vite build` alone, which has none.
{
  await page.reload({ waitUntil: "domcontentloaded" });
  const prepared = await page
    .evaluate(() => fetch(`${location.pathname.replace(/[^/]*$/, "")}examples/prepared/index.json`).then((r) => r.ok))
    .catch(() => false);
  const button = page.locator(".welcome").getByRole("button", { name: /^Duplex apartment/ });
  const offered = await button
    .waitFor({ timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  if (!prepared || !offered) {
    console.log("    (no prepared examples in this build; skipped)");
  } else {
    const workers = [];
    const onWorker = (w) => workers.push(w.url());
    page.on("worker", onWorker);
    const tEx = Date.now();
    await button.click();
    const opened = await page
      .waitForFunction(
        () => [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled),
        null,
        { timeout: 120000 },
      )
      .then(() => true)
      .catch(() => false);
    record(opened, 'example "Duplex apartment" opens from its prepared fragments', `${((Date.now() - tEx) / 1000).toFixed(1)}s`);
    page.off("worker", onWorker);
    const ifcWorkers = workers.filter((u) => /ifc-import|semantics/.test(u));
    record(ifcWorkers.length === 0, "  without starting an IFC worker", ifcWorkers.join(", ") || "none started");
    if (opened) {
      await page.getByRole("tab", { name: /^Model/ }).click();
      const row = await page.locator(".panel__scroll").getByText(/^\d+ of \d+ \(/).first().textContent({ timeout: 10000 }).catch(() => null);
      record(row === "4 of 14 (IsExternal)", "  and reports its exit doors", row || "not shown");
    }
  }
}

// ---- 9. a rooms (LOD1) model ---------------------------------------------------
//
// Added 2026-11-05: a GeoJSON plan (src/core/lod1.js) opens with no IFC, is
// drawn as rooms, exports back as a plan, and gives a graph joined across its
// two storeys by the stair.

console.log("\n[9] Rooms (LOD1) model");
{
  const rect = (x0, y0, x1, y1) => [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]];
  const room = (id, level, coords, name) => ({ type: "Feature", geometry: { type: "Polygon", coordinates: coords }, properties: { id, level, name } });
  const point = (x, y, props) => ({ type: "Feature", geometry: { type: "Point", coordinates: [x, y] }, properties: props });
  const plan = {
    type: "FeatureCollection",
    properties: { name: "two storeys", levels: [{ id: "G", ffl: 0, height: 3 }, { id: "1", ffl: 3, height: 3 }] },
    features: [
      room("hall", "G", rect(0, 0, 6, 5), "Hall"),
      room("st-g", "G", rect(6, 0, 9, 5), "Stair"),
      room("corr", "1", rect(0, 0, 6, 5), "Corridor"),
      room("st-1", "1", rect(6, 0, 9, 5), "Stair"),
      room("bed", "1", rect(0, 5, 6, 9), "Bedroom"),
      point(6, 2.5, { level: "G", kind: "door", width: 1 }),
      point(6, 2.5, { level: "1", kind: "door", width: 1 }),
      point(3, 5, { level: "1", kind: "door", width: 0.9, rooms: ["corr", "bed"] }),
      { type: "Feature", geometry: { type: "LineString", coordinates: [[0, 2], [0, 3.2]] }, properties: { level: "G", kind: "exit" } },
      point(7.5, 2.5, { level: "G", kind: "stair", toLevel: "1" }),
    ],
  };
  const planFile = { name: "studio-two-storey-plan.geojson", mimeType: "application/geo+json", buffer: Buffer.from(JSON.stringify(plan)) };
  // The step is passed in: the function runs in the page, where no closure reaches.
  const onStep = (step) => page.waitForFunction((s) => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === s, step, { timeout: 60000 });
  await page.getByRole("tab", { name: /^Model/ }).click();
  await page.getByText("Rooms models (LOD1)", { exact: true }).click();
  const tOpen = Date.now();
  await page.locator('input[type="file"][multiple]').setInputFiles(planFile);
  const opened = await onStep("graph").then(() => true).catch(() => false);
  const said = opened ? `${((Date.now() - tOpen) / 1000).toFixed(1)}s` : (await page.locator(".toast").allInnerTexts().catch(() => [])).join(" | ").replace(/\s+/g, " ").slice(0, 300);
  record(opened, "a GeoJSON rooms plan opens with no IFC", said);
  if (opened) timing("Open a GeoJSON rooms plan (choose to drawn)", Date.now() - tOpen);
  if (opened) {
    await page.getByRole("tab", { name: /^Model/ }).click();
    const text = await page.locator(".panel__scroll").first().innerText();
    record(/Rooms\s+5\b/.test(text) && /Storeys\s+2\b/.test(text) && /GeoJSON plan/.test(text), "  and the Model step lists its rooms and storeys");
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 10000 }).catch(() => null),
      page.getByRole("button", { name: "Export as a GeoJSON plan" }).click(),
    ]);
    const saved = download?.suggestedFilename() || "";
    record(/\.geojson$/.test(saved), "  and exports as a GeoJSON plan", saved || "no download");
    await page.getByRole("tab", { name: /^Graph/ }).click();
    await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
    const built = await onStep("route").then(() => true).catch(() => false);
    let stats = "";
    if (built) {
      await page.getByRole("tab", { name: /^Graph/ }).click();
      stats = await page.locator(".panel__scroll").first().innerText();
    }
    record(built && /Reachable\s+100%/.test(stats), "  its graph joins both storeys through the stair", (stats.match(/Nodes\s+[\d,]+/) || ["no graph"])[0].replace(/\s+/g, " "));
    await shot(page, "rooms-model");
  }
}

// ---- report -----------------------------------------------------------------

console.log("\n=== Timing (as a user sees it) ===");
for (const { label, ms } of timings) {
  console.log(`  ${label.padEnd(50)} ${ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms} ms`}`);
}
const byCall = new Map();
for (const { call, ms } of apiCalls) byCall.set(call, [...(byCall.get(call) || []), ms]);
console.log("\n=== Backend calls (request to last byte) ===");
for (const [call, list] of byCall) {
  const sorted = [...list].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  console.log(`  ${call.padEnd(42)} n=${String(list.length).padEnd(3)} median ${Math.round(median)} ms`);
}

// A guided run from the welcome screen (core/guided.js), in a fresh page:
// it takes every step itself and stops at the route.
{
  const guidedPage = await context.newPage();
  await guidedPage.goto(URL_BASE, { waitUntil: "load", timeout: 60000 });
  await guidedPage.waitForFunction(() => document.querySelector(".welcome .example"), null, { timeout: 120000 }).catch(() => {});
  const t0 = Date.now();
  await guidedPage.locator(".welcome").getByRole("button", { name: /^The way out/ }).click().catch(() => {});
  const caption = await guidedPage
    .waitForFunction(() => /^(Done|Stopped)/.test(document.querySelector(".guided__caption")?.textContent || "") && document.querySelector(".guided__caption").textContent, null, { timeout: 240000 })
    .then((h) => h.jsonValue())
    .catch(() => "");
  const result = (await guidedPage.locator(".result__title").first().textContent().catch(() => "")) || "";
  record(/^Done/.test(caption) && /route found/i.test(result), "a guided run takes every step and stops at the route", `${((Date.now() - t0) / 1000).toFixed(1)}s: ${caption || "no caption"}; ${result.trim() || "no result"}`);
  // It ends on its field note (core/fieldNotes.js), found.
  const note = (await guidedPage.locator(".guided__note").textContent().catch(() => "")) || "";
  record(/^Field note found: How far is the walk out\?/.test(note), "  and ends on the field note it found", note.slice(0, 80) || "no note");
  await guidedPage.close();
}

// Hints and field notes from a fresh browser (nothing stored): the count
// starts at 0, a hint leads past the model, and a dismissed hint stays gone.
{
  const fresh = await browser.newContext(MOBILE ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } : { viewport: { width: 1600, height: 950 } });
  const p = await fresh.newPage();
  const openDuplex = async () => {
    await p.goto(URL_BASE, { waitUntil: "load", timeout: 60000 });
    await p.locator(".welcome").getByRole("button", { name: /^Duplex apartment/ }).click({ timeout: 120000 });
    await p.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Build egress graph/.test(b.textContent) && !b.disabled), null, { timeout: 120000 });
  };
  await p.goto(URL_BASE, { waitUntil: "load", timeout: 60000 });
  const count = await p.getByRole("button", { name: /Field notes, \d+ of 12 found/ }).getAttribute("aria-label").catch(() => "");
  record(/0 of 12/.test(count || ""), "field notes start at 0 of 12 in a fresh browser", count || "no button");
  await openDuplex();
  const first = (await p.locator(".hint").innerText({ timeout: 10000 }).catch(() => "")).replace(/\s+/g, " ");
  record(/The model is only the start/.test(first), "a hint leads past the model step", first.slice(0, 90) || "no hint");
  await p.locator(".hint").getByRole("button", { name: "Dismiss hint" }).click().catch(() => {});
  await openDuplex();
  // Longer than the delay before a hint shows (App.jsx HINT_SETTLE_MS).
  await p.waitForTimeout(3500);
  const again = (await p.locator(".hint").innerText().catch(() => "")).replace(/\s+/g, " ");
  record(!/The model is only the start/.test(again), "  and once dismissed it does not come back", again.slice(0, 60) || "no hint shown");
  await fresh.close();
}

// A second model (core/federation.js), with --linked <file>: an MEP or
// structural IFC of the --ifc building, added beside it in the Model step,
// checked against it, and, when it has sprinklers, used for the heads.
const LINKED = argOf("--linked", null);
if (LINKED && existsSync(LINKED)) {
  console.log("\n[9] A second model");
  const fresh = await browser.newContext({ viewport: { width: 1600, height: 950 } });
  const p = await fresh.newPage();
  p.on("pageerror", (error) => pageErrors.push(error.message));
  await p.goto(URL_BASE, { waitUntil: "load", timeout: 60000 });
  await p.waitForFunction(() => !document.body.innerText.includes("Starting the 3D engine"), null, { timeout: 90000 }).catch(() => {});
  await p.locator('input[type="file"][accept=".ifc"]').first().setInputFiles(IFC);
  await p.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Build egress graph/.test(b.textContent) && !b.disabled), null, { timeout: 180000 }).catch(() => {});
  await p.getByRole("tab", { name: /^Model/ }).click();
  // A Section: a header button that opens its body.
  await p.getByRole("button", { name: /^Second model/ }).first().click().catch(() => {});
  const t0 = Date.now();
  await p.locator("section.section", { has: p.getByRole("button", { name: /^Second model/ }) }).locator('input[type="file"]').first().setInputFiles(LINKED).catch(() => {});
  await p.waitForSelector(".linked-model", { timeout: 300000 }).catch(() => {});
  timing("Add a second model (choose to checked)", Date.now() - t0);
  const card = (await p.locator(".linked-model").first().innerText().catch(() => "")).replace(/\s+/g, " ");
  record(/(Aligned|Check)/.test(card) && /Placed by/.test(card), "a second model is placed in the first one's frame and checked", card.slice(0, 200) || "no card");
  record(/plans overlap \(/.test(card) && /storeys? match by elevation/.test(card), "  its plan overlaps and its storeys match by elevation", card.match(/The plans overlap[^.]*|\d+ storeys match[^;.]*/g)?.join("; ") || card.slice(0, 120));
  const heads = Number(card.match(/Brings (\d+) sprinklers/)?.[1] || 0);
  if (heads) {
    await p.getByRole("tab", { name: /^Graph/ }).click();
    await p.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
    await p.waitForFunction(() => /nodes/.test(document.querySelector(".statusbar")?.innerText || ""), null, { timeout: 120000 }).catch(() => {});
    await p.getByRole("tab", { name: /^Simulate/ }).click();
    await p.getByText("Flamescapes combustion solver (optional)", { exact: true }).click().catch(() => {});
    await p.locator("summary", { hasText: /^Sprinklers$/ }).first().click().catch(() => {});
    await p.locator(".toggle__label", { hasText: /^Sprinklers$/ }).first().click().catch(() => {});
    await p.waitForFunction(() => /\d+ sprinkler heads/.test(document.querySelector(".panel__scroll")?.innerText || ""), null, { timeout: 60000 }).catch(() => {});
    const line = ((await p.locator(".panel__scroll").first().innerText().catch(() => "")).match(/\d+ sprinkler heads[^\n]*/)?.[0] || "").trim();
    const name = path.basename(LINKED);
    record(line.includes(name) && Number(line.match(/^(\d+)/)?.[1]) > 0, "  the heads come from the second model, in place", line.slice(0, 200) || "no layout");
  }
  await fresh.close();
}

console.log("\n=== Chrome diagnostics ===");
console.log(`  console errors : ${consoleErrors.length}`);
consoleErrors.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));
console.log(`  page errors    : ${pageErrors.length}`);
pageErrors.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));
console.log(`  failed requests: ${failedRequests.length}`);
failedRequests.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));
console.log(`  http >= 400    : ${badResponses.length}`);
badResponses.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));
console.log(`  Cloudflare Turnstile's own frames (not the app's): ${challengeNoise.length}`);
challengeNoise.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 160)}`));

await writeFile(
  path.join(SHOTS, "report.json"),
  JSON.stringify(
    { steps, timings, apiCalls, pageLoad, consoleErrors, pageErrors, failedRequests, badResponses, challengeNoise },
    null,
    2,
  ),
);

const failed = steps.filter((s) => !s.ok).length;
console.log(
  `\n${failed} failed step(s), ${pageErrors.length} uncaught error(s), ` +
    `${consoleErrors.length} console error(s). Screenshots in tools/shots/\n`,
);

await browser.close();
process.exit(failed || pageErrors.length ? 1 : 0);
