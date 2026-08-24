/**
 * Copies the runtime assets the viewer needs out of node_modules into public/,
 * so the app never depends on a third-party CDN and the copies always match the
 * installed package versions.
 *
 * Two things get copied:
 *
 *   public/wasm/       the web-ifc WASM runtime
 *   public/fragments/  the @thatopen/fragments worker
 *
 * The worker matters as much as the WASM: `FragmentsManager.getWorker()` fetches
 * it from `unpkg.com/@thatopen/fragments@<version>/dist/worker/worker.mjs` at
 * runtime. That is a hard dependency on a third party for the app to boot at
 * all, it breaks offline and air-gapped deployment, and when the fetch fails the
 * failure surfaces as an unhelpful worker syntax error.
 *
 * Runs automatically on `npm install` (postinstall).
 */
import { cp, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const modules = join(root, "node_modules");

let copied = 0;

async function versionOf(pkg) {
  const manifest = JSON.parse(await readFile(join(modules, pkg, "package.json"), "utf8"));
  return manifest.version;
}

// ---------------------------------------------------------------- web-ifc

const webIfcDir = join(modules, "web-ifc");
if (!existsSync(webIfcDir)) {
  console.warn("[sync-assets] web-ifc not installed yet - skipping.");
} else {
  const dest = join(root, "public", "wasm");
  await mkdir(dest, { recursive: true });
  const wanted = (await readdir(webIfcDir)).filter(
    (f) => f.endsWith(".wasm") || f === "web-ifc-mt.worker.js",
  );
  for (const file of wanted) {
    await cp(join(webIfcDir, file), join(dest, file));
    copied += 1;
  }
  const version = await versionOf("web-ifc");
  await writeFile(
    join(dest, "VERSION.txt"),
    `web-ifc ${version}\ncopied by scripts/sync-assets.mjs\n`,
    "utf8",
  );
  console.log(`[sync-assets] web-ifc@${version}: ${wanted.length} file(s) -> public/wasm/`);
}

// -------------------------------------------------------- fragments worker

const fragmentsDir = join(modules, "@thatopen", "fragments");
if (!existsSync(fragmentsDir)) {
  console.warn("[sync-assets] @thatopen/fragments not installed yet - skipping.");
} else {
  const dest = join(root, "public", "fragments");
  await mkdir(dest, { recursive: true });

  // Prefer the minified build: same code, a third of the bytes.
  const candidates = [
    join(fragmentsDir, "dist", "Worker", "worker.min.mjs"),
    join(fragmentsDir, "dist", "Worker", "worker.mjs"),
    join(fragmentsDir, "dist", "worker", "worker.min.mjs"),
    join(fragmentsDir, "dist", "worker", "worker.mjs"),
  ];
  const source = candidates.find((p) => existsSync(p));

  if (!source) {
    console.error(
      "[sync-assets] Could not find the fragments worker in @thatopen/fragments/dist.\n" +
        "              The viewer will fall back to fetching it from unpkg.com.",
    );
  } else {
    await cp(source, join(dest, "worker.mjs"));
    copied += 1;
    const version = await versionOf("@thatopen/fragments");
    await writeFile(
      join(dest, "VERSION.txt"),
      `@thatopen/fragments ${version}\nfrom ${source.replace(root, ".")}\n` +
        `copied by scripts/sync-assets.mjs\n`,
      "utf8",
    );
    console.log(
      `[sync-assets] @thatopen/fragments@${version}: worker -> public/fragments/worker.mjs`,
    );
  }
}

console.log(`[sync-assets] ${copied} file(s) copied.`);
