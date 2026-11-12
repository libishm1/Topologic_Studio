/**
 * Gas temperature at head height (fire-calibration-plan-2026-11-11.md,
 * section 13.2 item 1) on stored runs: reads results files of
 * validation-suite.mjs whose runs keep each tree's cells around head height
 * (run.series.head; runs made before 2026-11-12 13:00 keep none), reduces
 * them with lib/head.mjs against the reduced data, and gives the verdicts as
 * the suite does (series and laboratories; unweighted and weighted; every run
 * and with the sealed-door rule).
 *
 *   node tools/calibration/head-temperature.mjs results.json [more.json ...] [--json out.json]
 *   node tools/calibration/head-temperature.mjs --check
 *
 * --check runs the reducer's check, which uses no model: each device's
 * pre-ignition reading against the test's ambient, and the reducer fed the
 * measured series as if it were the model (it must return model = measured
 * for every device, delta 1).
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as HD from "./lib/head.mjs";
import * as S from "./lib/stats.mjs";
import { bothWays, printTables, PLAN } from "./validation-suite.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : null;
};
const datasets = Object.keys(PLAN);
const data = Object.fromEntries(datasets.map((d) => [d, JSON.parse(readFileSync(path.join(HERE, "data", `${d}.json`), "utf8"))]));
const testOf = (d, id) => data[d]?.tests.find((T) => T.id === id);

if (args.includes("--check")) {
  // 1. Pre-ignition: the mean reading at t <= 0 less the test's ambient, against the device's peak rise.
  // 2. The reducer on the measured series as its own model: a 1 s series, each second the reading nearest it,
  //    in K; each pair must come back with model = measured.
  let worstPre = { ratio: 0 }, worstSelf = { err: 0 }, devices = 0, excluded = [];
  const offsets = [], selfPairs = {};
  for (const d of datasets) {
    for (const T of data[d].tests) {
      const M = T.measured;
      const head = {};
      for (const tree of T.trees) {
        if (!tree.head?.length) continue;
        head[tree.id] = { z: tree.head.map((h) => h.z), T: [] };
        for (const dev of tree.head) {
          devices += 1;
          const ser = M.head?.[tree.id]?.[dev.id];
          if (T.steady) {
            head[tree.id].T.push(Array.from({ length: 300 }, () => ser + 273.15));
            continue;
          }
          const pre = M.times.map((t, q) => (t <= 0 && ser[q] != null ? ser[q] : null)).filter((x) => x != null);
          const peak = Math.max(...ser.filter((x) => x != null)) - T.ambientC;
          if (pre.length) {
            const off = pre.reduce((a, b) => a + b, 0) / pre.length - T.ambientC;
            const ratio = Math.abs(off) / Math.max(peak, 1e-9);
            if (Math.abs(off) > 5) offsets.push(`${d}/${T.id} ${tree.id}/${dev.id} ${off.toFixed(1)} K`);
            if (ratio > worstPre.ratio) worstPre = { ratio: +ratio.toFixed(3), off: +off.toFixed(2), peak: +peak.toFixed(1), at: `${d}/${T.id} ${tree.id}/${dev.id}` };
          }
          // Each second takes the reading nearest it (a tie takes the earlier).
          const end = Math.max(...M.times);
          const oneSecond = [];
          let q = 0;
          for (let s = 1; s <= end; s += 1) {
            while (q + 1 < M.times.length && Math.abs(M.times[q + 1] - s) < Math.abs(M.times[q] - s)) q += 1;
            oneSecond.push((ser[q] ?? T.ambientC) + 273.15);
          }
          head[tree.id].T.push(oneSecond);
        }
      }
      // Cells as fine as 1 mm, so the device's own height is always the nearest cell.
      const run = { dataset: d, cell: 0.001, horizon: T.steady ? 300 : Math.max(...M.times), health: { ok: true }, series: { from: 1, step: 1, head } };
      const got = HD.headPairs(run, T, PLAN[d].steadyFrom ?? 200);
      excluded.push(...got.excluded.map((e) => `${d}/${T.id} ${e.device}: ${e.why}`));
      for (const p of got.pairs) {
        (selfPairs[d] ||= []).push([p.model, p.measured]);
        const err = Math.abs(p.model - p.measured) / Math.max(p.measured, 1e-9);
        if (err > worstSelf.err) worstSelf = { err: +err.toFixed(4), at: `${d}/${T.id} ${p.device}`, model: p.model, measured: p.measured };
      }
    }
  }
  console.log(`head-height devices: ${devices} (all tests); left out by the plume rule (the same in every run):`);
  for (const e of excluded) console.log(`  ${e}`);
  console.log(`pre-ignition offset, worst: ${JSON.stringify(worstPre)}; devices more than 5 K off at t <= 0: ${offsets.length}`);
  for (const o of offsets) console.log(`  ${o}`);
  console.log(`the reducer on the measured series as the model, worst |M/E - 1|: ${JSON.stringify(worstSelf)}`);
  for (const [d, ps] of Object.entries(selfPairs)) {
    const s = S.statistics(ps, 0);
    console.log(`  ${d}: ${ps.length} pairs, delta ${s.delta}, sigmaM ${s.sigmaM} (sigma_E 0; 1 and 0 when the reducer reads the data as they are)`);
  }
  process.exit(0);
}

const files = args.filter((a, q) => a.endsWith(".json") && args[q - 1] !== "--json");
const runs = [];
const sources = [];
let withSeries = 0, without = 0;
for (const f of files) {
  const r = JSON.parse(readFileSync(f, "utf8"));
  sources.push({ file: f.replace(/\\/g, "/").replace(/^.*benchmark\//, "benchmark/"), generated: r.generated, head: r.git?.head, solverDigests: r.git?.solverDigests, harnessDigests: r.git?.harnessDigests });
  for (const run of r.runs) {
    const T = testOf(run.dataset, run.test);
    if (!T) continue;
    if (!run.series?.head) {
      without += 1;
      continue;
    }
    withSeries += 1;
    const got = HD.headPairs(run, T, PLAN[run.dataset]?.steadyFrom ?? 200);
    runs.push({ id: run.id, dataset: run.dataset, test: run.test, cell: run.cell, horizon: run.horizon, cells: run.cells, plumeIndex: run.plumeIndex, wallSeconds: run.wallSeconds, health: run.health, region: run.region, door: run.door, grid: run.grid, trees: run.trees, pairs: got.pairs, head: { excluded: got.excluded, missing: got.missing } });
  }
}
console.log(`${withSeries} runs with head-height series, ${without} without (made before the series were kept)`);
if (!runs.length) process.exit(0);
const measures = {};
for (const r of runs) if (testOf(r.dataset, r.test).trees.some((tr) => tr.head?.length)) (measures[r.dataset] ||= new Set()).add("headTemperature");
const digest = (rel) => createHash("sha256").update(readFileSync(path.join(HERE, rel))).digest("hex").slice(0, 16);
const report = {
  generated: new Date().toISOString(),
  tool: "tools/calibration/head-temperature.mjs",
  plan: "wiki/research/fire-calibration-plan-2026-11-11.md, section 13.2 item 1",
  sources,
  reducer: { "head-temperature.mjs": digest("head-temperature.mjs"), "lib/head.mjs": digest("lib/head.mjs"), "lib/stats.mjs": digest("lib/stats.mjs"), ...Object.fromEntries(datasets.map((d) => [`data/${d}.json`, digest(`data/${d}.json`)])) },
  plumeRule: `r < D/2 + ${HD.PLUME_SPREAD} (z - z_burner), unless a floor lies between (lib/head.mjs)`,
  jobs: "-",
  wallSeconds: 0,
  settings: { measures: Object.fromEntries(Object.entries(measures).map(([d, q]) => [d, [...q]])), sigmaE: S.SIGMA_E.headTemperature },
  runs,
};
const tests = {};
for (const r of runs) tests[`${r.dataset}/${r.test}`] = testOf(r.dataset, r.test);
await bothWays(report, S, measures, tests);
printTables(report);
const out = flag("--json");
if (out) {
  mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
  writeFileSync(path.resolve(out), JSON.stringify(report, null, 1) + "\n");
  console.log(`\nwrote ${out}`);
}
