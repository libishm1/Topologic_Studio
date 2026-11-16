#!/usr/bin/env node
/**
 * Builds the gas radiation tables' file for physics item 8, second attempt
 * (wiki/research/fire-calibration-plan-2026-11-11.md, 16.43, point 11; 16.44,
 * decision 7; src/core/gasRadiation.js reads it).
 *
 *   node tools/gas-radiation-tables.mjs --source DIR --out FILE
 *
 * --source (or the environment variable FLAMESCAPES_RADLIB_DATA) is a folder
 * outside this repository that holds the two text files of the absorption
 * line blackbody distribution function at 1 atm (Pearson, Webb, Solovjov and
 * Ma 2014, doi:10.1016/j.jqsrt.2014.01.019) as the RadLib library ships them
 * (Stephens, Jensen, Wheeler and Lignell 2022, doi:10.1016/j.cpc.2021.108227;
 * github.com/BYUignite/radlib at commit 583e0fe, data/rcslw_data):
 *   co2_p1_0.txt   55 664 values, in the order [T_g][T_b][C]
 *   h2o_p1_0.txt  500 976 values, in the order [Y][T_g][T_b][C]
 * one value a line, the last index fastest. Their SHA-256 must be those the
 * plan lists; the script stops otherwise.
 *
 * --out (or FLAMESCAPES_GAS_TABLES) is the file to write: the 556 640 values
 * as float32, little-endian, CO2's then H2O's, 2 226 560 bytes. It must lie
 * outside this repository: neither the tables nor the built file are
 * committed, since whether they may be redistributed here is not settled
 * (plan 16.44, decision 7). The script refuses a path inside the repository.
 *
 * With --verify the file at --out is not written: it is read and compared
 * with the two text files, value for value after rounding to float32.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const SOURCES = Object.freeze([
  Object.freeze({ file: "co2_p1_0.txt", values: 28 * 28 * 71, sha256: "5bd9c9ecdd3449f5c145a9e6bf07a181486752cec22920ef1928e9422a17508b" }),
  Object.freeze({ file: "h2o_p1_0.txt", values: 9 * 28 * 28 * 71, sha256: "b3dd6c860866d36c443719c3cadb2c1c1e2b83c3198040ee9d3b0847a2972e53" }),
]);

/** A source file's values as float32, after its SHA-256 and its count are checked. */
export function readSource(dir, source) {
  const bytes = readFileSync(path.join(dir, source.file));
  const sha = createHash("sha256").update(bytes).digest("hex");
  if (sha !== source.sha256) throw new Error(`${source.file}: SHA-256 ${sha}, not ${source.sha256} (plan 16.43, point 11)`);
  const lines = bytes.toString("latin1").split(/\r?\n/).filter((line) => line.trim() !== "");
  if (lines.length !== source.values) throw new Error(`${source.file}: ${lines.length} values, not ${source.values}`);
  const out = new Float32Array(source.values);
  for (let q = 0; q < lines.length; q += 1) {
    const v = Number(lines[q]);
    if (!Number.isFinite(v)) throw new Error(`${source.file}: line ${q + 1} is not a number`);
    out[q] = v;
  }
  return out;
}

/** The file's bytes for the two tables: float32, little-endian, CO2's then H2O's. */
export function tableBytes(tables) {
  const count = tables.reduce((n, t) => n + t.length, 0);
  const bytes = Buffer.alloc(4 * count);
  let o = 0;
  for (const t of tables) for (let q = 0; q < t.length; q += 1, o += 4) bytes.writeFloatLE(t[q], o);
  return bytes;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const args = process.argv.slice(2);
  const flag = (name, fallback) => {
    const i = args.indexOf(name);
    return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
  };
  const source = flag("--source", process.env.FLAMESCAPES_RADLIB_DATA);
  const out = flag("--out", process.env.FLAMESCAPES_GAS_TABLES);
  if (!source || !out) {
    console.error("usage: node tools/gas-radiation-tables.mjs --source DIR --out FILE [--verify]\n  (or FLAMESCAPES_RADLIB_DATA and FLAMESCAPES_GAS_TABLES)");
    process.exit(2);
  }
  const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const inside = (p) => {
    const rel = path.relative(repository, path.resolve(p));
    return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
  };
  if (inside(out)) {
    console.error("--out lies inside this repository: the tables are not committed (plan 16.44, decision 7). Give a path outside it.");
    process.exit(2);
  }
  if (inside(source)) {
    console.error("--source lies inside this repository: the tables are not committed (plan 16.44, decision 7). Keep them outside it.");
    process.exit(2);
  }
  const tables = SOURCES.map((s) => readSource(source, s));
  const bytes = tableBytes(tables);
  if (args.includes("--verify")) {
    if (!existsSync(out)) throw new Error(`${out} does not exist`);
    const have = readFileSync(out);
    const same = have.length === bytes.length && have.equals(bytes);
    console.log(`${same ? "equal" : "DIFFERENT"}: ${have.length} bytes against ${bytes.length} from the two source files`);
    process.exit(same ? 0 : 1);
  }
  mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
  writeFileSync(out, bytes);
  console.log(`${SOURCES.map((s, q) => `${s.file}: ${tables[q].length} values, SHA-256 as the plan lists`).join("; ")}`);
  console.log(`wrote ${bytes.length} bytes, SHA-256 ${createHash("sha256").update(bytes).digest("hex")}`);
}
