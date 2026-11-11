/**
 * Proposal C of the fire calibration (wiki/research/fire-calibration-plan-2026-11-11.md):
 * the vorticity confinement's strength fitted on McCaffrey's plume and
 * Alpert's ceiling jet together (transport-checks.mjs, Stage 1a), on the
 * app's cells. The rule, fixed before the runs:
 *
 *   - candidates: vorticity 0, 1, 2, 3, 4, 6 and 10 (the app's value before);
 *   - each on the 0.27 and 0.34 m cells; the two best on 0.20 m too;
 *   - a candidate is out if any of its runs diverged or its plume's
 *     centreline flowed down in more than 5 % of the samples (off its axis);
 *   - score: the mean over cells and quantities of (ln delta)^2 +
 *     (sigma_M - sigma_E)^2, the quantities being McCaffrey's intermittent
 *     temperature, his plume region's temperature and speed, and Alpert's
 *     ceiling-jet temperature beyond r/H 0.18 (FDS Validation Guide
 *     statistics, as transport-checks.mjs computes them);
 *   - the fitted value is the lowest score on all three cells.
 *
 * Shorter runs than Stage 1a's to keep the CPU moderate: 90 s, averaged from
 * 50 s (the burner reaches its heat release at 30 s). Each run is a child
 * process of transport-checks.mjs.
 *
 *   node tools/calibration/fit-confinement.mjs [--candidates 0,1,2,3,4,6,10] [--out file.json]
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const candidates = flag("--candidates", "0,1,2,3,4,6,10").split(",").map(Number);
const out = flag("--out", null);
const tmp = mkdtempSync(path.join(os.tmpdir(), "confinement-fit-"));
const SIGMA_E = { temperature: 0.07, velocity: 0.08 };

function checks(vorticity, cells) {
  const file = path.join(tmp, `v${vorticity}-${cells.join("_")}.json`);
  execFileSync(process.execPath, [path.join(HERE, "transport-checks.mjs"), "--cases", "plume,ceiling", "--cells", cells.join(","), "--horizon", "90", "--average-from", "50", "--solver", `vorticity=${vorticity}`, "--json", file], { stdio: ["ignore", "ignore", "inherit"] });
  return JSON.parse(readFileSync(file, "utf8")).results;
}

/** The candidate's terms on some runs, and whether it is out. */
function terms(results) {
  const parts = [];
  let out = null;
  for (const r of results) {
    if (r.health.diverged) out = `diverged (${r.case}, ${r.cell} m, ${r.health.divergedAt} s)`;
    if (r.case === "plume" && r.health.offAxisShare > 0.05) out ||= `plume off its axis (${r.cell} m, ${Math.round(100 * r.health.offAxisShare)} %)`;
    const pick = r.case === "plume"
      ? [["intermittentTemperature", SIGMA_E.temperature], ["plumeTemperature", SIGMA_E.temperature], ["plumeVelocity", SIGMA_E.velocity]]
      : [["ceilingJetTemperature", SIGMA_E.temperature]];
    for (const [key, se] of pick) {
      const s = r.stats?.[key];
      if (!s || s.delta == null || s.sigmaM == null) {
        out ||= `no ${key} (${r.cell} m)`;
        continue;
      }
      parts.push({ cell: r.cell, key, delta: s.delta, sigmaM: s.sigmaM, term: Math.log(s.delta) ** 2 + (s.sigmaM - se) ** 2 });
    }
  }
  return { parts, out, score: parts.length ? parts.reduce((a, p) => a + p.term, 0) / parts.length : Infinity };
}

const rows = [];
for (const v of candidates) {
  const t0 = Date.now();
  const results = checks(v, [0.27, 0.34]);
  const t = terms(results);
  rows.push({ vorticity: v, coarse: { ...t, results }, seconds: (Date.now() - t0) / 1000 });
  console.log(`vorticity ${v}: score on 0.27 and 0.34 m ${t.score.toFixed(3)}${t.out ? `, OUT: ${t.out}` : ""} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
const shortlist = rows.filter((r) => !r.coarse.out).sort((a, b) => a.coarse.score - b.coarse.score).slice(0, 2);
for (const r of shortlist) {
  const t0 = Date.now();
  const fine = checks(r.vorticity, [0.2]);
  const all = terms([...r.coarse.results, ...fine]);
  r.fine = { results: fine };
  r.all = { parts: all.parts, out: all.out, score: all.score };
  console.log(`vorticity ${r.vorticity}: score on all three cells ${all.score.toFixed(3)}${all.out ? `, OUT: ${all.out}` : ""} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
const fitted = shortlist.filter((r) => !r.all.out).sort((a, b) => a.all.score - b.all.score)[0] || null;
console.log(fitted ? `fitted: vorticity ${fitted.vorticity}` : "no candidate qualified");
if (out) {
  writeFileSync(out, JSON.stringify({ generated: new Date().toISOString(), rule: "see the header of tools/calibration/fit-confinement.mjs", candidates, rows: rows.map((r) => ({ vorticity: r.vorticity, coarseScore: r.coarse.score, coarseOut: r.coarse.out, coarseParts: r.coarse.parts, allScore: r.all?.score ?? null, allOut: r.all?.out ?? null, allParts: r.all?.parts ?? null, seconds: r.seconds })), fitted: fitted ? fitted.vorticity : null }, null, 1));
  console.log(`wrote ${out}`);
}
