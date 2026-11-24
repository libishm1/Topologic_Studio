/**
 * The tenability quantities at head height on stored runs
 * (fire-calibration-plan-2026-11-11.md, section 16.54): the time to 60 C and
 * the time to a heat dose of 0.3, measured against modelled, from results
 * files of validation-suite.mjs (run.series.head). The reducer and the rule
 * are lib/crossing.mjs; its check is lib/crossing.test.mjs
 * (node --test tools/calibration/lib/crossing.test.mjs). No solver runs.
 *
 *   node tools/calibration/tenability-times.mjs --dir RESULTS_DIR
 *        label=file.json[+more.json] [label2=...] [--json out.json] [--rows rows.json] [--md out.md]
 *
 * A label's files are joined in memory (a later file's run replaces the same
 * run of an earlier one), each read as it stands. Each file is checked
 * against the stored head-height pairs: the tool's own series must give them.
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as C from "./lib/crossing.mjs";
import * as S from "./lib/stats.mjs";
import { PLAN } from "./validation-suite.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : null;
};
const dir = flag("--dir") || ".";
const jobs = args.filter((a) => a.includes("=") && !a.startsWith("--")).map((a) => ({ label: a.slice(0, a.indexOf("=")), files: a.slice(a.indexOf("=") + 1).split("+") }));
const data = Object.fromEntries(Object.keys(PLAN).map((d) => [d, JSON.parse(readFileSync(path.join(HERE, "data", `${d}.json`), "utf8"))]));
const testOf = (d, id) => data[d]?.tests.find((T) => T.id === id);
const digest = (rel) => createHash("sha256").update(readFileSync(path.join(HERE, rel))).digest("hex").slice(0, 16);
const quart = (xs) => {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  const at = (p) => (v.length ? +v[Math.min(v.length - 1, Math.floor(p * v.length))].toFixed(3) : null);
  return { n: v.length, q1: at(0.25), median: at(0.5), q3: at(0.75) };
};

const out = { generated: new Date().toISOString(), tool: "tools/calibration/tenability-times.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md, section 16.54", choices: "benchmark/results/2026-11-24-tenability-choices.md", levelC: C.LEVEL_C, doseLimit: C.DOSE_LIMIT, sigmaE: C.SIGMA_E_TIME, missLimit: C.MISS_LIMIT, results: {} };
const allRows = {};
for (const job of jobs) {
  const byId = new Map();
  const sources = [];
  for (const f of job.files) {
    const r = JSON.parse(readFileSync(path.join(dir, f), "utf8"));
    const was = r.git?.harnessDigests || null;
    // Whether the data and the head reducer are those the file was made with (a merged file records none).
    const differ = was ? Object.keys(was).filter((k) => (k.startsWith("data/") || k === "lib/head.mjs") && was[k] !== digest(k)) : null;
    sources.push({ file: f, generated: r.generated, tool: r.tool, solverRev: r.git?.solverRev ?? r.solverRev ?? null, head: r.git?.head ?? null, jobs: r.jobs, wallSeconds: r.wallSeconds != null ? +Number(r.wallSeconds).toFixed(0) : null, runs: r.runs.length, mergedFrom: r.sources ? r.sources.map((s) => s.file) : null, harnessDiffers: differ });
    for (const run of r.runs) byId.set(run.id, run);
  }
  const runs = [...byId.values()];
  const cells = [...new Set(runs.map((r) => r.cell))].sort((a, b) => a - b);
  for (const cell of cells) {
    const mine = runs.filter((r) => r.cell === cell);
    const read = C.readRuns(mine, testOf);
    // The self-check: the tool's peak rises against the file's stored head-height pairs.
    const stored = new Map();
    for (const r of mine) for (const p of r.pairs || []) if (p.quantity === "headTemperature") stored.set(`${r.dataset}/${p.test}/${p.device}`, p);
    let same = 0, differs = 0, absent = 0;
    const worst = [];
    for (const row of read.rows.filter((x) => x.status === "ok")) {
      const p = stored.get(`${row.dataset}/${row.test}/${row.device}`);
      if (!p) absent += 1;
      else if (p.model === row.peak.model && p.measured === row.peak.measured) same += 1;
      else {
        differs += 1;
        if (worst.length < 5) worst.push({ device: `${row.dataset}/${row.test}/${row.device}`, stored: [p.model, p.measured], tool: [row.peak.model, row.peak.measured] });
      }
    }
    const ok = read.rows.filter((x) => x.status === "ok");
    // Part B: the same devices' peak rise, and how the two quantities sit beside it.
    const peak = S.statistics(ok.map((r) => [r.peak.model, r.peak.measured]), S.SIGMA_E.headTemperature);
    const storedAll = S.statistics([...stored.values()].map((p) => [p.model, p.measured]), S.SIGMA_E.headTemperature);
    const B = { peakRise: { n: peak.n, delta: peak.delta, sigmaM: peak.sigmaM }, storedHeadTemperature: { n: storedAll.n, delta: storedAll.delta, sigmaM: storedAll.sigmaM }, peakCelsiusRatio: quart(ok.map((r) => (r.peak.model + r.ambientC) / (r.peak.measured + r.ambientC))) };
    for (const q of C.QUANTITIES) {
      const e = read.quantities[q].every;
      B[q] = { delta: e.unweighted.delta, sigmaM: e.unweighted.sigmaM, timeRatio: quart(ok.filter((r) => r[q].outcome === "pair").map((r) => r[q].model / r[q].measured)), earlyClosures: e.counts.early, misses: e.counts.miss };
    }
    B.endDoseRatio = quart(ok.filter((r) => r.headHeatDose.endModel > 0 && r.headHeatDose.endMeasured > 0).map((r) => r.headHeatDose.endModel / r.headHeatDose.endMeasured));
    const stopped = mine.filter((r) => !r.health?.ok).map((r) => ({ id: r.id, reason: r.health?.reason, time: r.health?.time, value: r.health?.value }));
    const key = cells.length > 1 ? `${job.label}@${cell}` : job.label;
    out.results[key] = { label: job.label, cell, sources, runs: mine.length, datasets: [...new Set(mine.map((r) => r.dataset))], stopped, hasRepresentation: read.hasRepresentation, notRepresented: read.notRepresented, selfCheck: { same, differs, absent, examples: worst }, quantities: read.quantities, B };
    allRows[key] = read.rows;
    const w = (q) => read.quantities[q];
    console.log(`${key}: ${mine.length} runs, ${ok.length} devices; self-check ${same} same, ${differs} differ, ${absent} absent; ` + C.QUANTITIES.map((q) => `${q} pairs ${w(q).every.counts.pair} delta ${w(q).every.unweighted.delta} sigmaM ${w(q).every.unweighted.sigmaM} misses ${w(q).every.counts.miss}/${w(q).every.crosses} -> ${w(q).strictest}`).join("; "));
  }
}

const f2 = (x) => (x == null || !Number.isFinite(x) ? "n/a" : x.toFixed(2));
const pct = (x) => (x == null ? "n/a" : `${(100 * x).toFixed(1)} %`);
const NAME = { headCrossing60: "Time to 60 °C at head height", headHeatDose: "Time to a heat dose of 0.3 at head height" };
let md = "# The tenability quantities on the saved results (plan 16.54)\n\nMade by `tools/calibration/tenability-times.mjs`. The choices where the plan is silent: `benchmark/results/2026-11-24-tenability-choices.md`. Each cell of a statistic is delta / sigma_M.\n";
for (const q of C.QUANTITIES) {
  md += `\n## ${NAME[q]} (\`${q}\`)\n\n| Result | Cell | Devices | Pairs | Unresolved | Early closures | Misses | Neither | Misses' share | Pooled | 10-bin weighted | Series alike | With the rule | Series in band | Word: pooled | Word: series alike | Word: rule | Strictest of six |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n`;
  for (const [key, r] of Object.entries(out.results)) {
    const v = r.quantities[q], e = v.every, c = e.counts;
    md += `| ${key} | ${r.cell} | ${e.devices} | ${c.pair} | ${c.unresolved} | ${c.early} | ${c.miss} | ${c.neither} | ${pct(e.missShare)} (${c.miss} of ${e.crosses}) | ${f2(e.unweighted.delta)} / ${f2(e.unweighted.sigmaM)} | ${f2(e.binWeighted.delta)} / ${f2(e.binWeighted.sigmaM)} | ${f2(e.seriesAlike.delta)} / ${f2(e.seriesAlike.sigmaM)} | ${f2(v.rule.unweighted.delta)} / ${f2(v.rule.unweighted.sigmaM)} (${v.rule.counts.pair}) | ${e.seriesInBand.length} of ${e.seriesWithPairs.length} | ${v.words.pooled} | ${v.words.seriesAlike} | ${v.words.rule} | ${v.strictest} |\n`;
  }
  md += "\n### By series: pairs, delta / sigma_M, and early closures, misses, neither\n\n";
  const ds = [...new Set(Object.values(out.results).flatMap((r) => Object.keys(r.quantities[q].every.series)))];
  md += `| Result | ${ds.join(" | ")} |\n| --- | ${ds.map(() => "---").join(" | ")} |\n`;
  for (const [key, r] of Object.entries(out.results)) {
    md += `| ${key} | ${ds.map((d) => {
      const s = r.quantities[q].every.series[d];
      return s ? `${s.n}: ${f2(s.delta)} / ${f2(s.sigmaM)}${s.inBand ? " **in**" : ""}; ${s.early}, ${s.miss}, ${s.neither}` : "";
    }).join(" | ")} |\n`;
  }
}
md += "\n## Part B: what the dose and the deadline inherit\n\n| Result | Peak rise, same devices | Model's peak over measured, in °C (median) | Time to 60 °C | Its time ratio (q1, median, q3) | Time to dose 0.3 | Its time ratio (q1, median, q3) | Summed dose at the end, model over measured (q1, median, q3; n) |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n";
for (const [key, r] of Object.entries(out.results)) {
  const b = r.B, t = (x) => `${f2(x.q1)}, ${f2(x.median)}, ${f2(x.q3)}`;
  md += `| ${key} | ${f2(b.peakRise.delta)} / ${f2(b.peakRise.sigmaM)} (${b.peakRise.n}) | ${f2(b.peakCelsiusRatio.median)} | ${f2(b.headCrossing60.delta)} / ${f2(b.headCrossing60.sigmaM)} | ${t(b.headCrossing60.timeRatio)} | ${f2(b.headHeatDose.delta)} / ${f2(b.headHeatDose.sigmaM)} | ${t(b.headHeatDose.timeRatio)} | ${t(b.endDoseRatio)}; ${b.endDoseRatio.n} |\n`;
}
md += "\n## The files, and the tool's check against each file's stored head-height pairs\n\n| Result | Files | Runs | Stopped | Not represented | Check: same, differ, absent |\n| --- | --- | --- | --- | --- | --- |\n";
for (const [key, r] of Object.entries(out.results)) md += `| ${key} | ${r.sources.map((s) => `\`${s.file}\``).join(" + ")} | ${r.runs} | ${r.stopped.map((s) => `${s.id} (${s.reason} at ${s.time} s)`).join("; ") || "none"} | ${r.notRepresented.length} | ${r.selfCheck.same}, ${r.selfCheck.differs}, ${r.selfCheck.absent} |\n`;

const write = (name, text) => {
  if (!name) return;
  mkdirSync(path.dirname(path.resolve(name)), { recursive: true });
  writeFileSync(path.resolve(name), text);
  console.log(`wrote ${name}`);
};
write(flag("--json"), JSON.stringify(out, null, 1) + "\n");
write(flag("--rows"), "{\n" + Object.entries(allRows).map(([k, rows]) => `${JSON.stringify(k)}: [\n${rows.map((r) => JSON.stringify(r)).join(",\n")}\n]`).join(",\n") + "\n}\n");
write(flag("--md"), md);
