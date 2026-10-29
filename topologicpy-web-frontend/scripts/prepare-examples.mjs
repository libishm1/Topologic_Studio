/**
 * Converts the example IFCs in public/examples to fragments, and reads their
 * egress semantics, at build time, with the settings the import worker uses
 * (src/core/importSettings.js).
 *
 * Opening an example then fetches its fragments (81 KB for the Duplex,
 * against a 487 KB gzipped IFC) and starts no web-ifc at all: no import
 * worker (645 KB gzipped), no WASM (477 KB), no conversion, which on a phone
 * was most of the wait.
 *
 *   public/examples/prepared/index.json      { converter, examples: { id: {...} } }
 *   public/examples/prepared/<id>.frag       the fragments
 *   public/examples/prepared/<id>.json       the semantics (core/ifcSemantics.js)
 *
 * Generated, not committed (.gitignore); runs before `vite build` in
 * `npm run build`. The page uses an example's files only while `converter`
 * matches its own, and otherwise opens the IFC as before. An example that
 * fails to convert is left out, with a warning, and opens from its IFC.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { DataSet, IfcImporter } from "@thatopen/fragments";
import * as WEBIFC from "web-ifc";

import { converterId, trimImporter, WEB_IFC_SETTINGS } from "../src/core/importSettings.js";
import { ifcSemantics } from "../src/core/ifcSemantics.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const examplesDir = join(root, "public", "examples");
const outDir = join(examplesDir, "prepared");
const wasmDir = join(root, "node_modules", "web-ifc") + sep;

const versionOf = async (pkg) =>
  JSON.parse(await readFile(join(root, "node_modules", pkg, "package.json"), "utf8")).version;
const converter = converterId(await versionOf("@thatopen/fragments"), await versionOf("web-ifc"));

// As the semantics worker read it: a small tape is enough for the relations.
const TAPE_SIZE = 4 * 1024 * 1024;

async function semanticsOf(bytes) {
  const api = new WEBIFC.IfcAPI();
  api.SetWasmPath(wasmDir, true);
  await api.Init();
  api.SetLogLevel(WEBIFC.LogLevel.LOG_LEVEL_OFF);
  const model = api.OpenModel(bytes, { TAPE_SIZE });
  try {
    return ifcSemantics(api, model, WEBIFC);
  } finally {
    api.CloseModel(model);
  }
}

async function fragmentsOf(bytes) {
  const importer = trimImporter(new IfcImporter(), WEBIFC, DataSet);
  importer.wasm = { path: wasmDir, absolute: true };
  importer.webIfcSettings = { ...WEB_IFC_SETTINGS };
  return importer.process({ bytes });
}

const list = JSON.parse(await readFile(join(examplesDir, "index.json"), "utf8"));
await rm(outDir, { recursive: true, force: true });
await mkdir(outDir, { recursive: true });

const examples = {};
for (const entry of list) {
  const source = join(examplesDir, entry.file);
  if (!existsSync(source)) {
    console.warn(`[prepare-examples] ${entry.file} is missing; skipped.`);
    continue;
  }
  const t0 = performance.now();
  try {
    const bytes = new Uint8Array(await readFile(source));
    const fragments = await fragmentsOf(bytes.slice());
    const semantics = await semanticsOf(bytes.slice());
    await writeFile(join(outDir, `${entry.id}.frag`), fragments);
    await writeFile(join(outDir, `${entry.id}.json`), JSON.stringify(semantics));
    examples[entry.id] = {
      fragments: `prepared/${entry.id}.frag`,
      semantics: `prepared/${entry.id}.json`,
      ifcSha256: createHash("sha256").update(bytes).digest("hex"),
      fragmentsBytes: fragments.byteLength,
    };
    console.log(
      `[prepare-examples] ${entry.file}: ${(bytes.byteLength / 1e6).toFixed(2)} MB IFC -> ` +
        `${(fragments.byteLength / 1e3).toFixed(0)} KB fragments in ${Math.round(performance.now() - t0)} ms`,
    );
  } catch (error) {
    console.warn(`[prepare-examples] ${entry.file} could not be converted; it opens from its IFC. ${error?.message || error}`);
  }
}

await writeFile(join(outDir, "index.json"), JSON.stringify({ converter, examples }, null, 2));
console.log(`[prepare-examples] ${Object.keys(examples).length} of ${list.length} examples -> public/examples/prepared/ (${converter})`);
