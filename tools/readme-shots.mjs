/**
 * Takes the README's screenshots from the running app, using only the
 * examples shipped with the site (their credits and licences are in
 * topologicpy-web-frontend/public/examples/index.json).
 *
 *   node tools/readme-shots.mjs [--url http://localhost:5173/] [--only duplex,examples,phone,methods]
 *
 * Writes docs/images/*.jpg. Picks aim at graph nodes through window.__viewer,
 * which a dev server exposes and a production build exposes with ?debug.
 */
import { chromium } from "playwright-core";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "../docs/images");

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const URL_BASE = argOf("--url", "http://localhost:5173/");
const URL_DEBUG = `${URL_BASE}${URL_BASE.includes("?") ? "&" : "?"}debug`;
const ONLY = argOf("--only", "welcome,duplex,examples,phone,methods").split(",");
const CHROME = argOf("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch({
  executablePath: CHROME,
  headless: !args.includes("--headed"),
  args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--ignore-gpu-blocklist"],
});

const DESKTOP = { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, colorScheme: "light" };
const PHONE = { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: "light" };

async function newPage(options = DESKTOP) {
  const context = await browser.newContext(options);
  const page = await context.newPage();
  page.on("pageerror", (e) => console.log(`    page error: ${e.message.slice(0, 160)}`));
  await page.goto(URL_DEBUG, { waitUntil: "load" });
  await page.waitForFunction(() => document.querySelector(".welcome .example"), null, { timeout: 120000 });
  await page.waitForTimeout(1500);
  return page;
}

async function save(page, name, target = page) {
  // Toasts report timings and cover the model; the panels already say what happened.
  for (const close of await page.locator(".toast__close").all()) await close.click().catch(() => {});
  await page.waitForTimeout(500);
  // JPEG at this quality keeps text sharp and halves the 3D views against PNG.
  const file = path.join(OUT, `${name}.jpg`);
  await target.screenshot({ path: file, type: "jpeg", quality: 90 });
  console.log(`  saved ${path.relative(process.cwd(), file)}`);
}

const buildEnabled = () =>
  [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled);

async function openExample(page, name) {
  await page.locator(".welcome").getByRole("button", { name: new RegExp(`^${name}`) }).click();
  await page.waitForFunction(buildEnabled, null, { timeout: 180000 });
  await page.waitForTimeout(1500);
}

async function tab(page, name) {
  await page.getByRole("tab", { name: new RegExp(`^${name}`) }).click();
  await page.waitForTimeout(400);
}

async function view(page, which) {
  if (await page.getByRole("button", { name: which, exact: true }).isVisible().catch(() => false)) {
    await page.getByRole("button", { name: which, exact: true }).click();
  } else {
    await page.evaluate((v) => window.__viewer?.setStandardView(v), which.toLowerCase());
  }
  await page.waitForTimeout(1300);
}

async function buildGraph(page) {
  await tab(page, "Graph");
  await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
  await page.waitForFunction(
    () => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === "route",
    null,
    { timeout: 180000 },
  );
  await page.waitForTimeout(800);
}

/** Arms a picker and clicks the graph node drawn nearest a point of the viewport, given as fractions. */
async function pickAt(page, kind, fx, fy) {
  const picker = page.locator(".picker", { hasText: new RegExp(kind, "i") });
  await picker.getByRole("button").first().click();
  await page.waitForTimeout(200);
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
        const [cx, cy, , cw] = mul(P, ...mul(V, nodes[i], nodes[i + 1], nodes[i + 2], 1));
        if (cw <= 0) continue;
        const sx = ((cx / cw + 1) / 2) * r.width;
        const sy = ((1 - cy / cw) / 2) * r.height;
        if (sx < 0 || sy < 0 || sx > r.width || sy > r.height) continue;
        const d = (sx - r.width * tx) ** 2 + (sy - r.height * ty) ** 2;
        if (!best || d < best.d) best = { d, x: r.left + sx, y: r.top + sy };
      }
      return best;
    },
    [fx, fy],
  );
  if (!aim) throw new Error("No graph node to aim at: open the app on a dev server or with ?debug.");
  await page.mouse.move(aim.x, aim.y);
  await page.waitForTimeout(120);
  await page.mouse.click(aim.x, aim.y);
  await page.waitForTimeout(800);
}

const resultTitle = () => document.querySelector(".result__title")?.textContent || "";

/** Tries starts in turn, each with its nearest exit door, until a route is found. */
async function routeFrom(page, starts) {
  await tab(page, "Route");
  for (const [fx, fy] of starts) {
    await pickAt(page, "Start", fx, fy);
    const nearest = page.getByRole("button", { name: "Use the nearest exit door" });
    if (await nearest.isVisible().catch(() => false)) await nearest.click();
    else await pickAt(page, "Exit", 1 - fx, fy);
    await page.waitForTimeout(300);
    await page.getByRole("button", { name: /Find egress route/ }).click();
    await page.waitForFunction((re) => new RegExp(re, "i").test(document.querySelector(".result__title")?.textContent || ""), "route found|no route", {
      timeout: 60000,
    });
    if (/route found/i.test(await page.evaluate(resultTitle))) return true;
  }
  return false;
}

async function duplex() {
  console.log("\nDuplex apartment");
  const page = await newPage();
  await openExample(page, "Duplex apartment");
  await tab(page, "Model");
  await view(page, "Iso");
  await save(page, "duplex-model");

  await buildGraph(page);
  await tab(page, "Graph");
  await view(page, "Iso");
  await save(page, "duplex-graph");

  await view(page, "Fit");
  if (!(await routeFrom(page, [[0.42, 0.62], [0.55, 0.62], [0.6, 0.45], [0.38, 0.45]]))) console.log("  no route found");
  await page.locator(".result").first().scrollIntoViewIfNeeded();
  await save(page, "duplex-route");

  // One notch closer, so the heat reads at README size. A fire at the foot
  // of the stair, on the route, reroutes it near the end of the run.
  const box = await page.locator(".viewport__canvas").boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.55);
  await page.mouse.wheel(0, -120);
  await page.waitForTimeout(1000);
  await tab(page, "Simulate");
  await pickAt(page, "Fire", 0.5, 0.68);
  await page.getByRole("button", { name: /^Start fire$/ }).click();
  await page.waitForFunction(() => /fire step \d+/i.test(document.body.innerText), null, { timeout: 45000 });
  await page.waitForFunction(() => !/fire step \d+/i.test(document.body.innerText), null, { timeout: 120000 });
  await save(page, "duplex-fire");

  const plan = page.getByRole("button", { name: /^Plan with the forecast$/ });
  await plan.click();
  await page.waitForFunction(
    () =>
      /safe route: out in|above the dose limit|no safe route within the forecast/i.test(
        document.querySelector('[aria-label="Plan with the forecast"]')?.innerText || "",
      ),
    null,
    { timeout: 120000 },
  );
  await page.locator('[aria-label="Plan with the forecast"]').scrollIntoViewIfNeeded();
  await save(page, "duplex-forecast");

  // A fire on the dwelling's only stair leaves the agent no escape at all,
  // so it trains against a slower fire upstairs, away from the route.
  await page.locator(".picker", { hasText: /Fire/i }).scrollIntoViewIfNeeded();
  await pickAt(page, "Fire", 0.62, 0.42);
  await page.getByRole("button", { name: /Train an agent against this fire/ }).click();
  await page.getByLabel("Fire speed (links per move)").fill("0.25");
  await page.getByRole("button", { name: /^Train policy$/ }).click();
  await page.waitForFunction(() => /late-training success/i.test(document.body.innerText), null, { timeout: 180000 });
  await page.locator(".stat", { hasText: /late-training success/i }).scrollIntoViewIfNeeded();
  await save(page, "duplex-rl");

  // The two-zone model needs the model's rooms (IfcSpace), which the Duplex has.
  await page.getByText("Two-zone rooms (degrees C)", { exact: true }).click();
  await page.getByRole("button", { name: /^Start fire$/ }).scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: /^Start fire$/ }).click();
  await page.waitForFunction(() => /fire step \d+/i.test(document.body.innerText), null, { timeout: 45000 });
  await page.waitForTimeout(6000);
  await save(page, "duplex-zone");
  await page.getByRole("button", { name: /^Stop$/ }).click().catch(() => {});
  await page.context().close();
}

async function examples() {
  for (const [name, file, starts] of [
    ["Office, two storeys", "example-office", [[0.55, 0.45], [0.6, 0.5], [0.5, 0.42]]],
    ["Courtyard ring", "example-courtyard", [[0.35, 0.5], [0.5, 0.35], [0.6, 0.6]]],
    ["buildingSMART sample scene", "example-buildingsmart", null],
  ]) {
    console.log(`\n${name}`);
    const page = await newPage();
    await openExample(page, name);
    await buildGraph(page);
    await view(page, "Fit");
    if (starts) {
      if (!(await routeFrom(page, starts))) console.log("  no route found");
      if (name === "Courtyard ring") {
        await page.getByRole("button", { name: /^Find alternative routes$/ }).click();
        await page.waitForTimeout(2500);
      }
    } else {
      await tab(page, "Graph");
    }
    await save(page, file);
    await page.context().close();
  }
}

async function phone() {
  console.log("\nPhone");
  const page = await newPage(PHONE);
  await openExample(page, "Duplex apartment");
  await buildGraph(page);
  await view(page, "Fit");
  if (!(await routeFrom(page, [[0.42, 0.62], [0.55, 0.62], [0.6, 0.45], [0.38, 0.45]]))) console.log("  no route found");
  await page.locator(".viewport").scrollIntoViewIfNeeded().catch(() => {});
  await save(page, "phone-route");
  await page.context().close();
}

async function welcome() {
  console.log("\nWelcome");
  const page = await newPage();
  await save(page, "welcome");
  await page.context().close();
}

async function methods() {
  console.log("\nMethods");
  const page = await newPage();
  await page.getByRole("button", { name: /^Methods$/ }).click();
  await page.getByRole("dialog", { name: "Methods and verification" }).waitFor({ timeout: 30000 });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1500);
  await save(page, "methods");
  await page.context().close();
}

const jobs = { welcome, duplex, examples, phone, methods };
for (const name of ONLY) {
  if (!jobs[name]) throw new Error(`Unknown set "${name}"; use ${Object.keys(jobs).join(", ")}.`);
  await jobs[name]();
}
await browser.close();
