/**
 * Accessibility audit of the running app: axe-core (WCAG 2.1 A and AA) over
 * the main states in both themes, the keyboard tab order with focus
 * visibility, target sizes, and reflow at 200% zoom.
 *
 *   node tools/a11y-audit.mjs --axe <path/to/axe.min.js> [--url http://localhost:5173] [--json out.json]
 *
 * axe-core is not a project dependency; install it anywhere
 * (`npm install axe-core --prefix <dir>`) and pass its axe.min.js. The 3D
 * canvas is excluded: WebGL content is not in the DOM for axe to read.
 */
import { chromium } from "playwright-core";
import { readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const URL_BASE = argOf("--url", "http://localhost:5173");
const AXE = argOf("--axe", null);
const CHROME = argOf("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
if (!AXE) {
  console.error("usage: node tools/a11y-audit.mjs --axe <path/to/axe.min.js> [--url ...] [--json out.json]");
  process.exit(2);
}
const axeSource = readFileSync(AXE, "utf-8");

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();

const report = { states: {}, keyboard: [], targets: {}, reflow: {} };

async function scan(name) {
  await page.addScriptTag({ content: axeSource }).catch(() => {});
  const result = await page.evaluate(async () => {
    // eslint-disable-next-line no-undef
    const r = await axe.run(
      { exclude: [[".viewport__canvas"]] },
      { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } },
    );
    return r.violations.map((v) => ({
      id: v.id,
      impact: v.impact,
      help: v.help,
      nodes: v.nodes.slice(0, 6).map((n) => ({ target: n.target.join(" "), summary: n.failureSummary?.split("\n")[1]?.trim() })),
      count: v.nodes.length,
    }));
  });
  report.states[name] = result;
  console.log(`\n[${name}] ${result.length ? `${result.length} rule(s) violated` : "no axe violations"}`);
  for (const v of result) {
    console.log(`  ${v.impact?.padEnd(9)} ${v.id} (${v.count}): ${v.help}`);
    for (const n of v.nodes) console.log(`      ${n.target}${n.summary ? ` - ${n.summary}` : ""}`);
  }
}

async function targets(name) {
  const small = await page.evaluate(() => {
    const out = [];
    for (const el of document.querySelectorAll('button, a[href], input, select, summary, [role="tab"], label.btn')) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      const style = getComputedStyle(el);
      if (style.visibility === "hidden" || el.closest("[hidden]")) continue;
      // Inline links in running text are exempt (WCAG 2.5.8).
      if (el.tagName === "A" && style.display === "inline") continue;
      if (el.type === "file" || el.type === "radio" || el.type === "checkbox") continue;
      if (r.width < 24 || r.height < 24) {
        out.push(`${el.tagName.toLowerCase()} "${(el.getAttribute("aria-label") || el.textContent || "").trim().slice(0, 30)}" ${Math.round(r.width)}x${Math.round(r.height)}`);
      }
    }
    return out;
  });
  report.targets[name] = small;
  console.log(`  targets under 24 px (WCAG 2.5.8): ${small.length ? small.join("; ") : "none"}`);
}

// ---- welcome --------------------------------------------------------------
await page.goto(URL_BASE, { waitUntil: "domcontentloaded" });
await page.locator(".welcome").waitFor({ timeout: 90000 });
await scan("welcome, light");
await targets("welcome");

// Keyboard: the first stops from the top of the page.
await page.locator("body").click({ position: { x: 5, y: 5 } }).catch(() => {});
for (let i = 0; i < 18; i += 1) {
  await page.keyboard.press("Tab");
  const stop = await page.evaluate(() => {
    const el = document.activeElement;
    if (!el || el === document.body) return null;
    const style = getComputedStyle(el);
    const ring =
      (style.outlineStyle !== "none" && parseFloat(style.outlineWidth) > 0) || style.boxShadow !== "none";
    const label = el.getAttribute("aria-label") || el.textContent?.trim() || el.getAttribute("title") || el.type;
    return { role: el.getAttribute("role") || el.tagName.toLowerCase(), label: label?.slice(0, 40), ring };
  });
  if (stop) report.keyboard.push(stop);
}
console.log("\n[keyboard, from the top]");
report.keyboard.forEach((s, i) => console.log(`  ${String(i + 1).padStart(2)} ${s.role.padEnd(8)} ${s.ring ? "ring" : "NO RING"}  ${s.label}`));

// ---- reflow at 200% (1280 x 900 CSS px seen at 640 x 450) --------------------
await page.setViewportSize({ width: 640, height: 450 });
await page.waitForTimeout(300);
report.reflow.welcome = await page.evaluate(() => ({
  horizontalScroll: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
  width: document.documentElement.scrollWidth,
}));
console.log(`\n[reflow at 200%] welcome: ${report.reflow.welcome.horizontalScroll ? `scrolls sideways (${report.reflow.welcome.width} px)` : "fits"}`);
await page.setViewportSize({ width: 1440, height: 900 });

// ---- a model open, graph built --------------------------------------------
await page.locator(".welcome").getByRole("button", { name: /^Office, two storeys/ }).click();
await page.waitForFunction(
  () => [...document.querySelectorAll("button")].some((b) => /Build egress graph/.test(b.textContent) && !b.disabled),
  null,
  { timeout: 120000 },
);
await scan("graph step, model open");
await page.getByRole("button", { name: /Build egress graph/ }).first().click();
await page.waitForFunction(() => document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.step === "route", null, {
  timeout: 120000,
});
await scan("route step, graph built");
await targets("route step");

// Start and exit without a mouse: exit at a door, start by coordinates
// (the first door's own, moved a metre along x), then route.
await page.getByText("Place without the mouse").first().click();
const door = await page.getByLabel("Exit at a door").locator("option").nth(1).textContent();
await page.getByLabel("Exit at a door").selectOption({ index: 1 });
const [dx, dy, dz] = door.split(":")[1].split(",").map(Number);
await page.getByLabel("Point").selectOption("start");
await page.getByLabel("x", { exact: true }).fill(String(dx + 3));
await page.getByLabel("y", { exact: true }).fill(String(dy));
await page.getByLabel("z", { exact: true }).fill(String(dz + 3));
await page.getByRole("button", { name: /^Place start$/ }).click();
await page.getByRole("button", { name: /^Find egress route$/ }).click();
report.keyboardRoute = await page
  .waitForFunction(() => document.querySelector(".result__title")?.textContent || null, null, { timeout: 30000 })
  .then((h) => h.jsonValue())
  .catch(() => null);
console.log(`\n[placing without the mouse] ${door} as exit -> ${report.keyboardRoute ?? "no result"}`);
await scan("route step, route found");
for (const step of ["Model", "Graph", "Simulate"]) {
  await page.getByRole("tab", { name: new RegExp(`^${step}`) }).click();
  await page.waitForTimeout(200);
  await scan(`${step.toLowerCase()} step`);
}
await targets("simulate step");

await page.setViewportSize({ width: 640, height: 450 });
await page.waitForTimeout(300);
report.reflow.workflow = await page.evaluate(() => ({
  horizontalScroll: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
  width: document.documentElement.scrollWidth,
}));
console.log(`\n[reflow at 200%] workflow: ${report.reflow.workflow.horizontalScroll ? `scrolls sideways (${report.reflow.workflow.width} px)` : "fits"}`);
await page.setViewportSize({ width: 1440, height: 900 });

// ---- Methods ----------------------------------------------------------------
await page.getByRole("button", { name: /^Methods$/ }).click();
await page.getByRole("dialog").waitFor();
await scan("methods dialog");
report.methodsFocus = await page.evaluate(() => document.activeElement?.id || document.activeElement?.tagName);
console.log(`  focus on open: ${report.methodsFocus}`);
await page.keyboard.press("Escape");

// ---- dark theme --------------------------------------------------------------
await page.getByRole("button", { name: "Toggle colour theme" }).click();
await page.waitForTimeout(300);
await page.getByRole("tab", { name: /^Route/ }).click();
await scan("route step, dark");
await page.getByRole("button", { name: /^Methods$/ }).click();
await page.getByRole("dialog").waitFor();
await scan("methods dialog, dark");
await page.keyboard.press("Escape");

await browser.close();
const out = argOf("--json", null);
if (out) writeFileSync(out, JSON.stringify(report, null, 2));
const total = Object.values(report.states).reduce((n, v) => n + v.length, 0);
console.log(`\n${total} axe rule violation(s) across ${Object.keys(report.states).length} states.`);
