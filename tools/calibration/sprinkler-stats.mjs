/**
 * The first sprinkler's activation time against measurement, per data set,
 * as the FDS Validation Guide's statistics (lib/stats.mjs) on each test's
 * first head to open: the earliest estimate against the earliest measured
 * opening (estimate reading; tabulated or measured fires only; UL/NFPRF
 * runs in spray mode, and its first head opens before any water).
 * The experimental scatter sigma_E is taken as 0 (none is stated for these
 * tests), so sigma_M here holds the experiment's scatter too: it can only
 * overstate the model's.
 *   node tools/calibration/sprinkler-stats.mjs benchmark/results/<sprinkler>.json [more.json]
 */
import { readFileSync } from "node:fs";
import { statistics, tier } from "./lib/stats.mjs";

for (const file of process.argv.slice(2)) {
  const j = JSON.parse(readFileSync(file, "utf8"));
  const sets = new Map();
  for (const r of j.results) {
    if (!(r.ramp === "tabulated" || r.ramp === "measured")) continue;
    // The first head to open, whichever it is: the earliest estimate against the earliest measurement.
    const heads = r.heads.filter((h) => h.onGrid !== false);
    const m = Math.min(...heads.map((h) => (Number.isFinite(h.measured) ? h.measured : Infinity)));
    const e = Math.min(...heads.map((h) => (Number.isFinite(h.estimate) ? h.estimate : Infinity)));
    if (!Number.isFinite(m) || !Number.isFinite(e)) continue;
    if (!sets.has(r.dataset)) sets.set(r.dataset, []);
    sets.get(r.dataset).push([e, m]);
  }
  console.log(file);
  const all = [];
  for (const [d, pairs] of sets) {
    all.push(...pairs);
    const s = statistics(pairs, 0);
    console.log(`  ${d.padEnd(24)} tests ${String(pairs.length).padStart(2)}  delta ${s.delta}  sigmaM ${s.sigmaM}  ${tier(s)}`);
  }
  const s = statistics(all, 0);
  console.log(`  ${"pooled".padEnd(24)} tests ${String(all.length).padStart(2)}  delta ${s.delta}  sigmaM ${s.sigmaM}  ${tier(s)}`);
}
