/**
 * The Cloudflare describer's waits, in real Chrome, against a build made with
 * VITE_DESCRIBE_URL=http://127.0.0.1:8787 and one of Cloudflare's Turnstile
 * test keys (wiki/architecture/state-machines.md, SM-5). The Worker is faked
 * here (page.route); Turnstile's test keys need its script from the internet.
 *
 *   --check model  a key that always passes (1x00000000000000000000AA): a
 *                  description belongs to its model (B7)
 *   --check stop   a key that always asks for a click (3x00000000000000000000FF):
 *                  Stop ends the wait for the human check (B6)
 *
 *   node tools/describer-test.mjs --url http://localhost:4175/Topologic_Studio/ --check model
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
const URL_BASE = argOf("--url", "http://localhost:4175/Topologic_Studio/");
const CHECK = argOf("--check", "model");
const IFC = argOf("--ifc", path.resolve(HERE, "../../Ifc2x3_Duplex_Architecture.ifc"));
const CHROME = argOf("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
const TEXT = "A two-storey duplex with two ways out.";

let failed = 0;
function record(ok, label, detail = "") {
  if (!ok) failed += 1;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  — ${detail}` : ""}`);
}

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: !args.includes("--headed"),
  args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--ignore-gpu-blocklist"],
});
const page = await (await browser.newContext({ viewport: { width: 1600, height: 950 } })).newPage();
const cors = { "access-control-allow-origin": "*", "access-control-allow-headers": "content-type", "access-control-allow-methods": "POST, OPTIONS" };
let answers = 0;
await page.route("http://127.0.0.1:8787/**", async (route) => {
  if (route.request().method() === "OPTIONS") return route.fulfill({ status: 204, headers: cors });
  answers += 1;
  // The model's answer takes three seconds.
  await new Promise((resolve) => setTimeout(resolve, 3000));
  return route.fulfill({ status: 200, headers: cors, contentType: "application/json", body: JSON.stringify({ text: TEXT, model: "test" }) });
});
await page.goto(URL_BASE, { waitUntil: "domcontentloaded", timeout: 60000 });
await page.waitForFunction(() => document.querySelector(".app") && !document.body.innerText.includes("Starting the 3D engine"), null, { timeout: 90000 });

const open = async () => {
  await page.locator('input[type="file"][accept=".ifc"]').first().setInputFiles(IFC);
  await page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Build egress graph|Rebuild graph/.test(b.textContent) && !b.disabled), null, { timeout: 150000 });
  await page.getByRole("tab", { name: /^Model/ }).click();
  const summary = page.locator("summary", { hasText: "Describe this model" }).first();
  if (!(await summary.evaluate((el) => el.parentElement.open))) await summary.click();
};
const describeButton = page.getByRole("button", { name: "Send the facts and describe" });
const busy = () => describeButton.getAttribute("aria-busy").then((v) => v === "true").catch(() => false);
const shown = () => page.evaluate((text) => document.body.innerText.includes(text), TEXT);

await open();
await describeButton.waitFor({ timeout: 30000 });
if (CHECK === "model") {
  // Written, then another model opened: the text goes with its model.
  await page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Send the facts and describe/.test(b.textContent) && !b.disabled), null, { timeout: 30000 });
  await describeButton.click();
  await page.waitForFunction((text) => document.body.innerText.includes(text), TEXT, { timeout: 30000 }).catch(() => {});
  const written = await shown();
  await open();
  await page.waitForTimeout(1000);
  record(written && !(await shown()), "a description goes when another model is opened", written ? ((await shown()) ? "it stayed under the new model" : "gone") : "no description was written");
  // Still being written when another model is opened: it does not land under it.
  await page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Send the facts and describe/.test(b.textContent) && !b.disabled), null, { timeout: 30000 });
  await describeButton.click();
  await page.waitForTimeout(1500);
  await open();
  await page.waitForTimeout(4000);
  record(!(await shown()), "a description still being written when another model opens does not land under it", (await shown()) ? "it landed" : "dropped");
} else {
  // The check asks for a click that never comes; Stop ends the wait.
  await page.waitForTimeout(3000);
  await describeButton.click();
  await page.waitForTimeout(1500);
  const waiting = await busy();
  await page.getByRole("button", { name: "Stop", exact: true }).click().catch(() => {});
  await page.waitForTimeout(1500);
  const still = await busy();
  if (waiting) record(!still, "Stop ends a wait for the human check", still ? "still waiting" : "ended");
  else record(false, "the describer waited for the human check", "it did not wait: is Turnstile's script reachable?");
}
console.log(`  (the faked describer answered ${answers} time(s))`);
await browser.close();
process.exit(failed ? 1 : 0);
