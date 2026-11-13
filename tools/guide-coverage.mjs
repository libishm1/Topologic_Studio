/**
 * Which of the panels' labels the guide (topologicpy-web-frontend/public/docs)
 * never names: every field label, section title and folded list's summary in
 * src/ui/Panels.jsx, looked for in the guide pages' text. The interface audit
 * (2026-11-11) asked that a control a user meets be one the guide explains.
 *
 *   node tools/guide-coverage.mjs [--max N]
 *
 * Prints the labels the guide does not name. With --max, exits 1 when more
 * than N are missing, so a new control cannot go in without its words in the
 * guide (a ratchet: lower N as the guide catches up; 0 on 2026-11-12).
 */
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.join(HERE, "..", "topologicpy-web-frontend");
const args = process.argv.slice(2);
const at = args.indexOf("--max");
const max = at >= 0 ? Number(args[at + 1]) : null;

const panels = await readFile(path.join(FRONTEND, "src", "ui", "Panels.jsx"), "utf8");
const labels = new Set();
for (const pattern of [/\blabel="([^"{}]+)"/g, /<Section title="([^"{}]+)"/g, /<Disclosure summary="([^"{}]+)"/g]) {
  for (const m of panels.matchAll(pattern)) labels.add(m[1]);
}

const docsDir = path.join(FRONTEND, "public", "docs");
const pages = (await readdir(docsDir)).filter((f) => f.endsWith(".html"));
const text = (await Promise.all(pages.map((f) => readFile(path.join(docsDir, f), "utf8"))))
  .join("\n")
  .replace(/<[^>]+>/g, " ")
  .replace(/&amp;/g, "&")
  .replace(/&#39;|&rsquo;|&apos;/g, "'")
  .replace(/\s+/g, " ")
  .toLowerCase();

const missing = [...labels].filter((label) => !text.includes(label.toLowerCase())).sort();
console.log(`${labels.size - missing.length} of ${labels.size} panel labels are named in the guide (${pages.length} pages).`);
if (missing.length) console.log(`Not named (${missing.length}):\n  ${missing.join("\n  ")}`);

// Each section's "In the guide" link (primitives.jsx Section guide="page#anchor")
// names a guide page that exists and an id on it.
const broken = [];
for (const [, page, anchor] of panels.matchAll(/\bguide="([^"#]+)(?:#([^"]+))?"/g)) {
  const html = pages.includes(page) ? await readFile(path.join(docsDir, page), "utf8") : null;
  if (!html || (anchor && !html.includes(`id="${anchor}"`))) broken.push(`${page}${anchor ? `#${anchor}` : ""}`);
}
if (broken.length) console.error(`\nGuide links that lead nowhere: ${broken.join(", ")}`);

if (max !== null && missing.length > max) console.error(`\n${missing.length} labels missing from the guide, more than the ${max} allowed.`);
if (broken.length || (max !== null && missing.length > max)) process.exit(1);
