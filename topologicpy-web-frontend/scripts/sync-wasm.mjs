// Copies the web-ifc WASM runtime out of node_modules into public/wasm/ so the
// app never depends on a third-party CDN and the wasm always matches the
// installed web-ifc version.
import { cp, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const src = join(root, "node_modules", "web-ifc");
const dest = join(root, "public", "wasm");

if (!existsSync(src)) {
  console.warn("[sync-wasm] web-ifc not installed yet - skipping.");
  process.exit(0);
}

await mkdir(dest, { recursive: true });
const entries = await readdir(src);
const wanted = entries.filter((f) => f.endsWith(".wasm") || f === "web-ifc-mt.worker.js");
for (const file of wanted) {
  await cp(join(src, file), join(dest, file));
}

const pkg = JSON.parse(await readFile(join(src, "package.json"), "utf8"));
await writeFile(
  join(dest, "VERSION.txt"),
  `web-ifc ${pkg.version}\ncopied by scripts/sync-wasm.mjs\n`,
  "utf8",
);
console.log(`[sync-wasm] copied ${wanted.length} file(s) from web-ifc@${pkg.version} -> public/wasm/`);
