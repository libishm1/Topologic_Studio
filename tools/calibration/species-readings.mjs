/**
 * The species statistics three ways (plan sections 14.7 and 14.8). The ratio
 * of two changes has no meaning when the measured one is within the
 * analyser's uncertainty, so each O2 and CO2 statistic of a results file is
 * read:
 *   - over all pairs, as pre-registered;
 *   - with the floor: a pair whose measured change (the drop of O2, the rise
 *     of CO2, as the suite stores it) is below 0.005 by volume fraction is
 *     left out;
 *   - with the series' own figure: left out below the larger of the floor and
 *     twice the report's stated uncertainty. A relative figure is taken at
 *     the pair's own reading, rebuilt from the nominal ambient (O2 0.2095
 *     less the drop; CO2 0.0004 plus the rise).
 * Each reading is reduced by the suite's own summarise(), every run and with
 * the 2026-11-12 rule (the runs' stored representation). The strictest
 * verdict of the six decides, so leaving pairs out can never lift one. The
 * pairs left out are counted beside every statistic. Nothing is run: the
 * measured change is in each stored pair.
 *
 *   node tools/calibration/species-readings.mjs <results.json> [--json out.json]
 */
import { readFileSync, writeFileSync } from "node:fs";
import * as S from "./lib/stats.mjs";
import { summarise } from "./validation-suite.mjs";

const SPECIES = ["o2", "co2"];
const FLOOR = 0.005;
const AMBIENT = { o2: 0.2095, co2: 0.0004 };
// Each report's stated uncertainty (expanded, about 95 %): abs by volume fraction, rel of the reading. Plan
// 14.8 gives the sources; WTC's O2 is its Table 2-3 (Table 6-1 covers the analyser alone).
const STATED = {
  nistnrc: { o2: { abs: 0.01 }, co2: { abs: 0.0025 } },
  wtc: { o2: { rel: 0.05 }, co2: { rel: 0.02 } },
  fse: { o2: { rel: 0.12 }, co2: { rel: 0.12 } },
  fsri: { o2: { rel: 0.12 } },
  // The same laboratory's analysers (plan 15.4, added 2026-11-13; FSRI_CHAIRS_OPEN item 7 in validation-data.mjs).
  "fsri-chairs": { o2: { rel: 0.12 } },
};
const noFigure = new Set();
const own = (p) => {
  const u = STATED[p.dataset]?.[p.quantity];
  if (!u) {
    noFigure.add(`${p.dataset} ${p.quantity}`);
    return FLOOR;
  }
  const at = p.quantity === "o2" ? AMBIENT.o2 - p.measured : AMBIENT.co2 + p.measured;
  return Math.max(FLOOR, 2 * (u.abs ?? u.rel * at));
};
const READINGS = [
  ["all pairs", () => 0],
  ["floor 0.005", () => FLOOR],
  ["series' own figure", own],
];
const RANK = { calibrated: 2, indicative: 1, "not calibrated": 0 };

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args.splice(i, 2)[1] : fallback;
};
const jsonOut = flag("--json", null);
const file = args[0];
if (!file) {
  console.error("usage: node tools/calibration/species-readings.mjs <results.json> [--json out.json]");
  process.exit(2);
}
const report = JSON.parse(readFileSync(file, "utf8"));
const measures = {};
for (const [d, qs] of Object.entries(report.settings.measures)) measures[d] = new Set(qs);
if (report.runs.some((r) => !r.representation)) throw new Error("a run carries no representation: summarise the file first (validation-suite.mjs --summary)");

const f = (x, d = 2) => (x == null || !Number.isFinite(x) ? "-" : x.toFixed(d));
const out = {};
for (const [name, threshold] of READINGS) {
  const below = {}; // dataset quantity cell -> pairs left out
  const runs = report.runs.map((r) => ({
    ...r,
    pairs: r.pairs.filter((p) => {
      if (!SPECIES.includes(p.quantity) || !Number.isFinite(p.measured) || !(p.measured < threshold(p))) return true;
      const k = `${p.dataset} ${p.quantity} ${p.cell}`;
      below[k] = (below[k] || 0) + 1;
      return false;
    }),
  }));
  const ways = { every: summarise(runs, S, measures), rule: summarise(runs.filter((r) => r.representation.represented), S, measures) };
  for (const [way, stats] of Object.entries(ways)) {
    for (const q of SPECIES) {
      for (const [h, v] of Object.entries(stats.byQuantity[q] || {})) {
        const series = Object.fromEntries(v.datasets.map((d) => {
          const s = stats.byDataset[d][q][h];
          return [d, { n: s.n, belowInstrument: below[`${d} ${q} ${h}`] || 0, delta: s.delta, sigmaM: s.sigmaM, tier: s.tier, weighted: s.weighted, stopped: s.stopped }];
        }));
        (((out[q] ||= {})[h] ||= {})[name] ||= {})[way] = { series, pooled: v.pooled, weightedPooled: v.weighted.pooled, datasetsInBand: v.datasetsInBand, final: v.final };
      }
    }
  }
}

// The first reading is the file's own statistics.
for (const q of SPECIES) {
  for (const [h, v] of Object.entries(report.stats?.byQuantity?.[q] || {})) {
    const mine = out[q][h]["all pairs"].every;
    if (mine.final !== v.final || mine.pooled.delta !== v.pooled.delta || mine.pooled.sigmaM !== v.pooled.sigmaM) throw new Error(`${q} on ${h} m: the first reading (${mine.pooled.delta}, ${mine.pooled.sigmaM}, ${mine.final}) is not the file's (${v.pooled.delta}, ${v.pooled.sigmaM}, ${v.final})`);
  }
}

console.log(`${file}\nThe first reading reproduces the file's statistics.`);
for (const q of SPECIES) {
  for (const [h, byReading] of Object.entries(out[q] || {})) {
    console.log(`\n${q} on ${h} m`);
    console.log("reading             runs   series      n  below   delta  sigmaM  tier                       | weighted delta sigmaM");
    const finals = [];
    for (const [name, ways] of Object.entries(byReading)) {
      for (const [way, v] of Object.entries(ways)) {
        for (const [d, s] of Object.entries(v.series)) console.log(`${name.padEnd(19)} ${way.padEnd(6)} ${d.padEnd(9)} ${String(s.n).padStart(3)}  ${String(s.belowInstrument).padStart(5)}  ${f(s.delta).padStart(6)}  ${f(s.sigmaM).padStart(6)}  ${String(s.tier).padEnd(26)} | ${f(s.weighted.delta).padStart(14)} ${f(s.weighted.sigmaM).padStart(6)}${s.stopped ? `  (${s.stopped} stopped)` : ""}`);
        console.log(`${name.padEnd(19)} ${way.padEnd(6)} ${"pooled".padEnd(9)} ${String(v.pooled.n).padStart(3)}  ${"".padStart(5)}  ${f(v.pooled.delta).padStart(6)}  ${f(v.pooled.sigmaM).padStart(6)}  ${String(v.pooled.tier).padEnd(26)} | ${f(v.weightedPooled.delta).padStart(14)} ${f(v.weightedPooled.sigmaM).padStart(6)}  -> ${v.final}`);
        finals.push(v.final);
      }
    }
    const decides = finals.reduce((a, b) => (RANK[b] < RANK[a] ? b : a));
    byReading.decides = decides;
    console.log(`decides (the strictest of the ${finals.length}): ${decides}`);
  }
}
if (noFigure.size) console.log(`\nNo stated figure, so the floor stands in the third reading: ${[...noFigure].join(", ")}`);
if (jsonOut) {
  writeFileSync(jsonOut, `${JSON.stringify({ generated: new Date().toISOString(), tool: "tools/calibration/species-readings.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md, sections 14.7 and 14.8", source: file, floor: FLOOR, ambient: AMBIENT, stated: STATED, noFigure: [...noFigure], readings: out }, null, 1)}\n`);
  console.log(`Wrote ${jsonOut}`);
}
