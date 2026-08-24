/**
 * Diagnostic: is the navigation graph a flat floor mesh or a space-frame truss?
 *
 * Loads the model, builds a graph, then reports the vertical distribution of
 * nodes and classifies every edge by how much height it spans and which kinds
 * of node it joins. Stairs are supposed to climb; floors are not.
 */
import { chromium } from "playwright-core";

const URL_BASE = process.argv[2] || "http://localhost:5173";
const IFC = process.argv[3] || "../Ifc2x3_Duplex_Architecture.ifc";
const KIND = ["floor", "stair", "door"];

const browser = await chromium.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: ["--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

let graph = null;
page.on("response", async (r) => {
  if (r.url().includes("/api/ifc/graph") && r.request().method() === "POST") {
    try {
      graph = await r.json();
    } catch {
      /* not json */
    }
  }
});

await page.goto(URL_BASE, { waitUntil: "domcontentloaded" });
await page.waitForFunction(
  () => !document.body.innerText.includes("Starting the 3D engine"),
  null,
  { timeout: 90000 },
);
await page.setInputFiles('input[type=file][accept=".ifc"]', IFC);
await page.waitForFunction(
  () => document.body.innerText.toLowerCase().includes("floors"),
  null,
  { timeout: 120000 },
);
await page.waitForTimeout(1500);
await page.getByRole("button", { name: /Build egress graph|Rebuild graph/ }).first().click();
await page.waitForFunction(
  () => document.body.innerText.toLowerCase().includes("nodes"),
  null,
  { timeout: 180000 },
);
await page.waitForTimeout(1000);
await browser.close();

if (!graph) {
  console.error("No graph response captured.");
  process.exit(1);
}

const view = (b64, Type) => {
  const buf = Buffer.from(b64, "base64");
  return new Type(buf.buffer, buf.byteOffset, buf.byteLength / Type.BYTES_PER_ELEMENT);
};

const pos = view(graph.nodes_b64, Float32Array);
const edges = view(graph.edges_b64, Uint32Array);
const kinds = view(graph.kinds_b64, Uint8Array);
const count = pos.length / 3;
const UP = 1; // the viewer works in y-up

// --- node heights -----------------------------------------------------------

const hist = new Map();
for (let i = 0; i < count; i += 1) {
  const y = Math.round(pos[i * 3 + UP] * 20) / 20;
  hist.set(y, (hist.get(y) || 0) + 1);
}
const rows = [...hist.entries()].sort((a, b) => a[0] - b[0]).filter(([, c]) => c > 5);

console.log(`\n  ${count} nodes, ${rows.length} height bands (5 cm buckets, count > 5)\n`);
for (const [y, c] of rows) {
  console.log(`    y=${y.toFixed(2).padStart(7)}  ${"#".repeat(Math.min(60, Math.round(c / 20)))} ${c}`);
}

// --- edges ------------------------------------------------------------------

let level = 0;
let sloped = 0;
let steep = 0;
const byKind = {};

for (let k = 0; k < edges.length; k += 2) {
  const a = edges[k];
  const b = edges[k + 1];
  const rise = Math.abs(pos[a * 3 + UP] - pos[b * 3 + UP]);
  const band = rise <= 0.05 ? "level" : rise <= 0.35 ? "sloped" : "steep";
  if (band === "level") level += 1;
  else if (band === "sloped") sloped += 1;
  else steep += 1;

  const pair = [KIND[kinds[a]] || "?", KIND[kinds[b]] || "?"].sort().join("-");
  byKind[pair] = byKind[pair] || { level: 0, sloped: 0, steep: 0 };
  byKind[pair][band] += 1;
}

console.log(
  `\n  ${edges.length / 2} edges | level (<=5cm) ${level} | sloped (5-35cm) ${sloped} | steep (>35cm) ${steep}`,
);
console.log("\n  by node kind:");
console.log("    pair              level   sloped    steep");
for (const [pair, v] of Object.entries(byKind).sort()) {
  console.log(
    `    ${pair.padEnd(16)} ${String(v.level).padStart(6)} ${String(v.sloped).padStart(8)} ${String(v.steep).padStart(8)}`,
  );
}

const nodeKinds = {};
for (let i = 0; i < count; i += 1) {
  const t = KIND[kinds[i]] || "?";
  nodeKinds[t] = (nodeKinds[t] || 0) + 1;
}
console.log("\n  nodes by kind:", JSON.stringify(nodeKinds));

const floorBraced = (byKind["floor-floor"]?.sloped || 0) + (byKind["floor-floor"]?.steep || 0);
console.log(
  `\n  VERDICT: ${floorBraced} non-level floor-to-floor edges ` +
    `(these are the space-frame bracing; want ~0)\n`,
);
