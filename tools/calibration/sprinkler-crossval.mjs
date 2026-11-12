/**
 * Sprinkler activation, cross-validated (plan section 14.3). The walls'
 * convection coefficient was set on the four sprinkler series, so none of
 * them is held out. This reads runs of tools/sprinkler-validation.mjs at
 * several coefficients and leaves one laboratory out at a time (and, beside
 * it, one series at a time):
 *   - in each fold the coefficient is the candidate with the lowest
 *     (ln delta)^2 + sigma_M^2 over the pooled first-head pairs of the
 *     laboratories kept in;
 *   - the laboratory left out is then read at that coefficient: delta,
 *     sigma_M, the band (lib/stats.mjs tier) and its stopped runs.
 * The pairs are sprinkler-stats.mjs's: each test's earliest estimate against
 * its earliest measured opening, tabulated or measured fires only, sigma_E
 * taken as 0. "Cross-validated" needs every laboratory left out within the
 * calibrated band with no stopped run.
 *
 *   node tools/calibration/sprinkler-crossval.mjs <runs.json> [more.json]
 *        [--candidates 5,7,10,14,20] [--json out.json]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { statistics, tier } from "./lib/stats.mjs";

const LAB = {
  vettori_flat_1998: "NIST",
  vettori_sloped_2003: "NIST",
  bittern_2004: "Canterbury",
  ul_nfprf_1998_series_I: "UL",
};
const CALIBRATED = "within the calibrated band";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args.splice(i, 2)[1] : fallback;
};
const candidates = flag("--candidates", "5,7,10,14,20").split(",").map(Number);
const jsonOut = flag("--json", null);
const files = args;
if (!files.length) {
  console.error("usage: node tools/calibration/sprinkler-crossval.mjs <runs.json> [more.json] [--candidates 5,7,10,14,20] [--json out.json]");
  process.exit(2);
}

// runs[coefficient][series] = { pairs: [[estimate, measured]], runs, stopped }
const runs = new Map();
const seen = new Set();
let unchecked = 0; // runs from before the solver reported its health
for (const file of files) {
  for (const r of JSON.parse(readFileSync(file, "utf8")).results) {
    if (!(r.ramp === "tabulated" || r.ramp === "measured")) continue;
    if (!candidates.includes(r.convection)) continue;
    if (seen.has(r.id)) throw new Error(`run ${r.id} is in more than one file`);
    seen.add(r.id);
    if (!(r.dataset in LAB)) throw new Error(`no laboratory for ${r.dataset}`);
    if (!runs.has(r.convection)) runs.set(r.convection, new Map());
    const at = runs.get(r.convection);
    if (!at.has(r.dataset)) at.set(r.dataset, { pairs: [], runs: 0, stopped: 0 });
    const set = at.get(r.dataset);
    set.runs += 1;
    if (!r.health) unchecked += 1;
    else if (!r.health.ok) set.stopped += 1;
    const heads = r.heads.filter((h) => h.onGrid !== false);
    const m = Math.min(...heads.map((h) => (Number.isFinite(h.measured) ? h.measured : Infinity)));
    const e = Math.min(...heads.map((h) => (Number.isFinite(h.estimate) ? h.estimate : Infinity)));
    if (Number.isFinite(m) && Number.isFinite(e)) set.pairs.push([e, m]);
  }
}
const missing = candidates.filter((c) => !runs.has(c));
if (missing.length) throw new Error(`no runs at ${missing.join(", ")} W/m2K`);
const series = [...runs.get(candidates[0]).keys()];
for (const c of candidates) {
  for (const d of series) {
    const a = runs.get(c).get(d);
    const b = runs.get(candidates[0]).get(d);
    if (!a || a.runs !== b.runs) throw new Error(`${d}: ${a?.runs ?? 0} runs at ${c} W/m2K against ${b.runs} at ${candidates[0]}`);
  }
}

/** (ln delta)^2 + sigma_M^2 on pairs, sigma_E 0, unrounded. */
function score(pairs) {
  const logs = pairs.filter(([m, e]) => m > 0 && e > 0).map(([m, e]) => Math.log(m / e));
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length;
  const varLog = logs.reduce((a, b) => a + (b - mean) ** 2, 0) / (logs.length - 1);
  return (mean + varLog / 2) ** 2 + varLog;
}
const pooled = (c, names) => names.flatMap((d) => runs.get(c).get(d).pairs);

function folds(groupOf) {
  const groups = [...new Set(series.map(groupOf))];
  return groups.map((out) => {
    const train = series.filter((d) => groupOf(d) !== out);
    const held = series.filter((d) => groupOf(d) === out);
    const scores = candidates.map((c) => ({ coefficient: c, score: score(pooled(c, train)) }));
    const best = scores.reduce((a, b) => (b.score < a.score ? b : a));
    const tie = scores.some((s) => s !== best && Math.abs(s.score - best.score) < 1e-9);
    const stats = statistics(pooled(best.coefficient, held), 0);
    const stopped = held.reduce((a, d) => a + runs.get(best.coefficient).get(d).stopped, 0);
    const band = tier(stats);
    return { leftOut: out, series: held, trainedOn: train, chosen: best.coefficient, tie, scores: scores.map((s) => ({ ...s, score: +s.score.toFixed(4) })), tests: stats.n, delta: stats.delta, sigmaM: stats.sigmaM, band, stopped, passes: band === CALIBRATED && stopped === 0 };
  });
}

function report(title, list) {
  console.log(title);
  for (const f of list) {
    console.log(`  left out ${f.leftOut.padEnd(24)} chose ${String(f.chosen).padStart(3)} W/m2K${f.tie ? " (tie)" : ""}  tests ${String(f.tests).padStart(2)}  delta ${f.delta}  sigmaM ${f.sigmaM}  ${f.band}${f.stopped ? `, ${f.stopped} stopped` : ""}`);
    console.log(`    scores ${f.scores.map((s) => `${s.coefficient}: ${s.score}`).join("  ")}`);
  }
}

const byLaboratory = folds((d) => LAB[d]);
const bySeries = folds((d) => d);
report("Leaving one laboratory out (decides):", byLaboratory);
report("Leaving one series out (beside):", bySeries);
if (unchecked) console.log(`${unchecked} runs carry no health record (an earlier solver): no verdict is drawn from them.`);
const crossValidated = unchecked === 0 && byLaboratory.every((f) => f.passes);
const chosen = [...new Set(byLaboratory.map((f) => f.chosen))];
console.log(unchecked ? "Not a verdict." : crossValidated ? `Cross-validated on ${series.length} series from ${byLaboratory.length} laboratories.` : "Not cross-validated: a laboratory left out is outside the calibrated band, or a run stopped.");
console.log(chosen.length === 1 ? `Every laboratory fold chose ${chosen[0]} W/m2K.` : `The laboratory folds chose ${chosen.join(", ")} W/m2K: the app's value stays.`);
if (jsonOut) {
  writeFileSync(jsonOut, `${JSON.stringify({ generated: new Date().toISOString(), tool: "tools/calibration/sprinkler-crossval.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md, section 14.3", sources: files, candidates, series, laboratories: LAB, byLaboratory, bySeries, crossValidated, chosen }, null, 1)}\n`);
  console.log(`Wrote ${jsonOut}`);
}
