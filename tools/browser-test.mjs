/**
 * Drives the real app in real Chrome and reports what breaks.
 *
 * Uses playwright-core against the installed Chrome (no browser download).
 * Everything Chrome complains about is captured: console errors, uncaught
 * exceptions, failed requests, non-2xx responses. Screenshots land in
 * tools/shots/ so the UI can be eyeballed afterwards.
 *
 *   node tools/browser-test.mjs [--url http://127.0.0.1:5173] [--ifc path.ifc] [--headed]
 */
import { chromium } from "playwright-core";
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
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

const context = await browser.newContext({
  viewport: { width: 1600, height: 950 },
  deviceScaleFactor: 1,
});
const page = await context.newPage();

// ---- capture everything Chrome reports -------------------------------------

const consoleErrors = [];
const pageErrors = [];
const failedRequests = [];
const badResponses = [];

page.on("console", (msg) => {
  const type = msg.type();
  if (type === "error" || type === "warning") {
    const text = msg.text();
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
  failedRequests.push(`${req.method()} ${req.url()} — ${failure}`);
  console.log(`    [requestfailed] ${req.url().slice(0, 140)} ${failure}`);
});
page.on("response", (res) => {
  if (res.status() >= 400) {
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
await page.goto(URL_BASE, { waitUntil: "domcontentloaded", timeout: 60000 });

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
record(backendUp, "backend reported online");

await shot(page, "boot");
// Baseline for the render check after the model loads.
const emptyShot = await page.locator(".viewport__canvas").screenshot();

// ---- 2. load the IFC --------------------------------------------------------

console.log("\n[2] Loading the IFC model");
const t0 = Date.now();
await page.setInputFiles('input[type="file"][accept=".ifc"]', IFC);

const modelLoaded = await page
  .waitForFunction(
    () => {
      // innerText reflects CSS text-transform, so match case-insensitively.
      const text = document.body.innerText.toLowerCase();
      return text.includes("floors") && !text.includes("no model loaded");
    },
    null,
    { timeout: 180000 },
  )
  .then(() => true)
  .catch(() => false);
record(modelLoaded, "IFC model loaded", `${((Date.now() - t0) / 1000).toFixed(1)}s`);
timing("Open IFC, first time (convert + cache)", Date.now() - t0);

if (modelLoaded) {
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
  await page.getByRole("button", { name: which, exact: true }).click();
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
      () => document.querySelector('[role="tab"][aria-selected="true"]')?.textContent?.trim() === "Route",
      null,
      { timeout: 180000 },
    )
    .then(() => true)
    .catch(() => false);
  record(built, "graph built", `${((Date.now() - t1) / 1000).toFixed(1)}s`);
  timing("Build egress graph (sample in browser + server)", Date.now() - t1);

  if (built) {
    // buildGraph() advances the panel to the Route tab; the graph stats live
    // on the Model tab, so come back before scraping them.
    await page.getByRole("tab", { name: "Model" }).click();
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
    record(Number(g.Nodes?.replace(/,/g, "")) > 100, "graph has a sane node count", `${g.Nodes}`);
    record(Number(g.Edges?.replace(/,/g, "")) > 100, "graph has edges", `${g.Edges}`);
  }
}
await shot(page, "graph-built");

// ---- 4. pick start and exit, then route -------------------------------------

console.log("\n[4] Picking points and routing");

// Frame the model so a click at a screen fraction actually lands on it,
// then switch to the Route tab, where the pickers live.
await page.getByRole("button", { name: "Fit", exact: true }).click();
await page.waitForTimeout(1300);
await page.getByRole("tab", { name: "Route" }).click();
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

  // Move first, then click: the engine's raycaster tracks pointer position
  // from move events.
  const x = box.x + box.width * fx;
  const y = box.y + box.height * fy;
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

const gotExit = await pickAt("Exit", 0.60, 0.34);
record(gotExit, "exit point placed by clicking the model");

await shot(page, "points-picked");

if (gotStart && gotExit) {
  const routeBtn = page.getByRole("button", { name: /Find egress route/ });
  const tRoute = Date.now();
  await routeBtn.click();
  const routed = await page
    .waitForFunction(() => document.body.innerText.toLowerCase().includes("waypoints"), null, {
      timeout: 60000,
    })
    .then(() => true)
    .catch(() => false);
  record(routed, "route computed");
  if (routed) timing("Find egress route (click to drawn)", Date.now() - tRoute);

  if (routed) {
    const r = await page.evaluate(() => {
      const out = {};
      document.querySelectorAll(".stat").forEach((el) => {
        const k = el.querySelector(".stat__label")?.textContent?.trim();
        const v = el.querySelector(".stat__value")?.textContent?.trim();
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
  const altResponse = page.waitForResponse(
    (res) => res.url().includes("/api/ifc/path/alternatives"),
    { timeout: 60000 },
  );
  const tAlt = Date.now();
  await altBtn.click();
  const alt = await altResponse.then((res) => res.json()).catch(() => null);
  if (alt) timing("Alternative routes (click to answer)", Date.now() - tAlt);
  record(Boolean(alt?.routes?.length), "alternative routes returned", alt ? `${alt.routes.length} route(s), ${alt.ms} ms` : "no response");
  if (alt?.routes?.length) {
    console.log(
      `    alternatives: ${alt.routes.map((r) => `${r.length.toFixed(2)} m (${Math.round(r.overlap * 100)}% shared)`).join(", ")}`,
    );
    if (alt.note) console.log(`    note: ${alt.note}`);
    const listed = await page
      .waitForFunction(() => /route 1/i.test(document.body.innerText), null, { timeout: 10000 })
      .then(() => true)
      .catch(() => false);
    record(listed, "alternative routes listed in the panel");
    const costs = alt.routes.map((r) => r.cost);
    record(
      costs.every((c, i) => i === 0 || c >= costs[i - 1] - 1e-6),
      "alternatives are ranked by cost",
      costs.map((c) => c.toFixed(2)).join(" <= "),
    );
  }
  await page.locator(".kv", { hasText: /route 1/i }).scrollIntoViewIfNeeded().catch(() => {});
  await shot(page, "alternatives");
} else {
  record(false, "Find alternative routes was enabled", "needs start and exit");
}

// ---- 4c. TopologicPy Navigation cross-check ---------------------------------

console.log("\n[4c] TopologicPy Navigation cross-check");
await page.getByRole("tab", { name: "Model" }).click();
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
await page.getByRole("tab", { name: "Route" }).click();
await page.waitForTimeout(300);
const gotFire = await pickAt("Fire", 0.50, 0.50);
record(gotFire, "fire origin placed");

await page.getByRole("tab", { name: "Simulate" }).click();
await page.waitForTimeout(300);

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
await page.getByRole("button", { name: /Reinforcement learning/ }).click();
await page.waitForTimeout(300);
const trainBtn = page.getByRole("button", { name: /^Train policy$/ });
if (await trainBtn.isEnabled()) {
  const sent = page.waitForRequest((req) => req.url().includes("/api/rl/train"), {
    timeout: 30000,
  });
  const tRl = Date.now();
  await trainBtn.click();
  const body = await sent.then((req) => JSON.parse(req.postData() || "{}")).catch(() => ({}));
  record(
    Array.isArray(body.fire_point) && body.fire_point.length === 3,
    "RL request carries the picked fire origin",
    JSON.stringify(body.fire_point),
  );
  // innerText applies the stat labels' text-transform: uppercase, so match
  // case-insensitively.
  const trained = await page
    .waitForFunction(() => /late-training success/i.test(document.body.innerText), null, {
      timeout: 120000,
    })
    .then(() => true)
    .catch(() => false);
  record(trained, "RL training returned and rendered its stats");
  if (trained) timing("Train RL policy (click to stats)", Date.now() - tRl);
  if (trained) {
    await page
      .locator(".stat", { hasText: /late-training success/i })
      .scrollIntoViewIfNeeded()
      .catch(() => {});
    const r = await page.evaluate(() => {
      const out = {};
      document.querySelectorAll(".stat").forEach((el) => {
        const k = el.querySelector(".stat__label")?.textContent?.trim();
        const v = el.querySelector(".stat__value")?.textContent?.trim();
        if (k) out[k] = v;
      });
      return out;
    });
    console.log(`    rl: ${JSON.stringify(r)}`);
    // Whether an escape exists depends on where the fire was clicked, so the
    // contract is that the UI states it, not that the agent always escapes.
    record(
      r["Escape possible"] !== undefined,
      "RL reports whether an escape exists",
      r["Escape possible"],
    );
  }
  await shot(page, "rl-trained");
} else {
  record(false, "Train policy was enabled", "needs start and exit");
}

// ---- 6. theme, panels, topology mode ----------------------------------------

console.log("\n[6] UI surfaces");

await page.getByRole("button", { name: "Toggle colour theme" }).click();
await page.waitForTimeout(400);
const theme = await page.evaluate(() => document.documentElement.dataset.theme);
record(theme === "light" || theme === "dark", "theme toggle works", theme);
await shot(page, "theme-light");
await page.getByRole("button", { name: "Toggle colour theme" }).click();
await page.waitForTimeout(300);

for (const tab of ["Model", "Route", "Simulate", "About"]) {
  await page.getByRole("tab", { name: tab }).click();
  await page.waitForTimeout(200);
  const visible = await page.locator(".panel__scroll").isVisible();
  record(visible, `panel tab "${tab}" renders`);
}
await shot(page, "about-tab");

await page.getByRole("button", { name: "Topology JSON" }).click();
await page.waitForTimeout(1200);
record(
  await page.locator("text=Topology JSON viewer").count() > 0,
  "topology mode renders its placeholder",
);
await shot(page, "topology-mode");
await page.getByRole("button", { name: /^IFC$/ }).click();
await page.waitForTimeout(500);

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
await page.setInputFiles('input[type="file"][accept=".ifc"]', IFC);
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
  const button = page.getByRole("button", { name: new RegExp(`^${name}`) });
  const offered = await button
    .waitFor({ timeout: 60000 })
    .then(() => true)
    .catch(() => false);
  record(offered, `example "${name}" is offered`);
  if (!offered) continue;
  const tEx = Date.now();
  await button.click();
  const opened = await page
    .waitForFunction(() => /floors/i.test(document.body.innerText) && !/no model loaded/i.test(document.body.innerText), null, {
      timeout: 120000,
    })
    .then(() => true)
    .catch(() => false);
  record(opened, `example "${name}" opened`, `${((Date.now() - tEx) / 1000).toFixed(1)}s`);
  if (!opened) continue;
  await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
  await page
    .waitForFunction(
      () => document.querySelector('[role="tab"][aria-selected="true"]')?.textContent?.trim() === "Route",
      null,
      { timeout: 120000 },
    )
    .catch(() => {});
  await page.getByRole("tab", { name: "Model" }).click();
  await page.waitForTimeout(300);
  const stats = await readStats();
  console.log(`    ${name}: ${JSON.stringify(stats)}`);
  record(stats.Reachable === "100%", `example "${name}" is one connected graph`, stats.Reachable);
  await shot(page, `example-${name.toLowerCase().replace(/[^a-z]+/g, "-")}`);
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

console.log("\n=== Chrome diagnostics ===");
console.log(`  console errors : ${consoleErrors.length}`);
consoleErrors.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));
console.log(`  page errors    : ${pageErrors.length}`);
pageErrors.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));
console.log(`  failed requests: ${failedRequests.length}`);
failedRequests.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));
console.log(`  http >= 400    : ${badResponses.length}`);
badResponses.slice(0, 12).forEach((e) => console.log(`    - ${e.slice(0, 220)}`));

await writeFile(
  path.join(SHOTS, "report.json"),
  JSON.stringify(
    { steps, timings, apiCalls, pageLoad, consoleErrors, pageErrors, failedRequests, badResponses },
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
