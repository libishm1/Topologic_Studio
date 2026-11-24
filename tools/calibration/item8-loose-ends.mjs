/**
 * Item 8's loose ends from saved results (fire-calibration-plan-2026-11-11.md,
 * sections 16.50 point 10, 16.52 and 16.54 part D): each run's heat ledger at
 * its last row, item 8's tallies and the radRoundoff sums by series; the stop
 * of the FSE series' ISOHept26; and the head-height temperature series by
 * series. It reads results files of validation-suite.mjs as they stand and
 * runs no solver.
 *
 *   node tools/calibration/item8-loose-ends.mjs --dir RESULTS_DIR label=file.json [...]
 *        [--json out.json] [--md out.md]
 *
 * The ledger's remainder is released - (gas + walls + shell + leak +
 * radiatedOut), as plan 16.43 (k) writes it; a term a file does not hold
 * counts as 0. `radiated` is the stage's gross emission: it is re-absorbed
 * many times over and is no term of the balance.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import * as S from "./lib/stats.mjs";

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : null;
};
const dir = flag("--dir") || ".";
const jobs = args.filter((a) => a.includes("=") && !a.startsWith("--")).map((a) => ({ label: a.slice(0, a.indexOf("=")), file: a.slice(a.indexOf("=") + 1) }));
const med = (xs) => {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  return v.length ? v[v.length >> 1] : null;
};
const sum = (xs) => xs.filter(Number.isFinite).reduce((a, b) => a + b, 0);
const r4 = (x) => (x == null || !Number.isFinite(x) ? null : +x.toPrecision(4));
const z = (x) => (Number.isFinite(x) ? x : 0);
const TALLIES = ["radFloor", "radCapped", "radNegative", "radShareNegative", "surfaceFloor", "surfaceCapped", "flameSilenced", "flameCleared", "burnerRadiatedOut", "radiatedOut"];
const COUNTS = ["gasRays", "gasCrossings", "gasEmitters", "surfaceRays", "surfaceCrossings"];

const out = { generated: new Date().toISOString(), tool: "tools/calibration/item8-loose-ends.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md, sections 16.50 (point 10), 16.52 and 16.54 (part D)", results: {} };
const runsOf = {};
for (const job of jobs) {
  const J = JSON.parse(readFileSync(path.join(dir, job.file), "utf8"));
  const res = { file: job.file, jobs: J.jobs, wallSeconds: r4(J.wallSeconds), runs: J.runs.length, health: J.solver?.health ?? null, series: {}, stopped: [], head: {} };
  runsOf[job.label] = new Map(J.runs.map((r) => [r.id, { wallSeconds: r.wallSeconds, ok: !!r.health?.ok, dataset: r.dataset }]));
  for (const d of [...new Set(J.runs.map((r) => r.dataset))]) {
    const all = J.runs.filter((r) => r.dataset === d);
    const ok = all.filter((r) => r.health?.ok && r.ledger?.length);
    const last = ok.map((r) => r.ledger[r.ledger.length - 1]);
    const share = (f) => r4(med(last.map((l) => f(l) / l.released)));
    const rem = (l) => l.released - (z(l.gas) + z(l.walls) + z(l.shell) + z(l.leak) + z(l.radiatedOut));
    const round = (l, k) => (l.radRoundoff ? l.radRoundoff[k] : null);
    const roundAll = (l) => (l.radRoundoff ? z(l.radRoundoff.own) + z(l.radRoundoff.deposits) + z(l.radRoundoff.burner) : null);
    res.series[d] = {
      runs: all.length, stopped: all.length - all.filter((r) => r.health?.ok).length, read: ok.length,
      releasedMJ: r4(sum(last.map((l) => l.released))),
      medianShare: { gas: share((l) => z(l.gas)), walls: share((l) => z(l.walls)), shell: share((l) => z(l.shell)), leak: share((l) => z(l.leak)), radiatedOut: share((l) => z(l.radiatedOut)), burnerRadiatedOut: share((l) => z(l.burnerRadiatedOut)), remainder: share(rem), radiatedGross: share((l) => z(l.radiated)) },
      remainderRange: [r4(Math.min(...last.map((l) => rem(l) / l.released))), r4(Math.max(...last.map((l) => rem(l) / l.released)))],
      talliesMJ: Object.fromEntries(TALLIES.map((k) => [k, last.some((l) => l[k] != null) ? r4(sum(last.map((l) => l[k]))) : null])),
      tallyMaxShare: Object.fromEntries(TALLIES.map((k) => [k, last.some((l) => l[k] != null) ? r4(Math.max(...last.map((l) => z(l[k]) / l.released))) : null])),
      counts: Object.fromEntries(COUNTS.map((k) => [k, last.some((l) => l[k] != null) ? sum(last.map((l) => l[k])) : null])),
      radRoundoffMJ: last.some((l) => l.radRoundoff) ? { own: r4(sum(last.map((l) => round(l, "own")))), deposits: r4(sum(last.map((l) => round(l, "deposits")))), burner: last.some((l) => l.radRoundoff?.burner != null) ? r4(sum(last.map((l) => round(l, "burner")))) : null, medianOfReleased: r4(med(last.map((l) => roundAll(l) / l.released))), largestOfReleased: r4(last.map((l) => roundAll(l) / l.released).reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0)), largestOfRadiated: r4(last.map((l) => roundAll(l) / l.radiated).reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0)) } : null,
      wallSeconds: r4(sum(all.map((r) => r.wallSeconds))),
    };
  }
  for (const r of J.runs.filter((x) => !x.health?.ok)) res.stopped.push({ id: r.id, health: r.health });
  // ISOHept26, stopped or not: its health, its burner and each ledger row as shares of the heat released.
  const iso = J.runs.find((r) => r.dataset === "fse" && r.test === "ISOHept26");
  if (iso) {
    res.isoHept26 = {
      cell: iso.cell, grid: iso.grid, cells: iso.cells, openCells: iso.openCells, horizon: iso.horizon, peakKW: iso.peakKW, plumeIndex: iso.plumeIndex, health: iso.health, burner: iso.burner, fuel: iso.fuel, wallSeconds: iso.wallSeconds,
      pairs: iso.pairs.map((p) => ({ quantity: p.quantity, device: p.device, model: p.model, measured: p.measured })),
      ledger: iso.ledger.map((l) => ({ t: l.t, released: l.released, gas: r4(z(l.gas) / l.released), walls: r4(z(l.walls) / l.released), shell: r4(z(l.shell) / l.released), radiatedOut: l.radiatedOut != null ? r4(l.radiatedOut / l.released) : null, remainder: r4((l.released - (z(l.gas) + z(l.walls) + z(l.shell) + z(l.leak) + z(l.radiatedOut))) / l.released), radiatedGross: r4(l.radiated / l.released), flameSilenced: l.flameSilenced != null ? r4(l.flameSilenced / l.released) : null, flameCleared: l.flameCleared != null ? r4(l.flameCleared / l.released) : null, radCapped: l.radCapped ?? null, gasMJ: l.gas })),
    };
    res.fseOthers = J.runs.filter((r) => r.dataset === "fse" && r.test !== "ISOHept26").map((r) => {
      const l = r.ledger[r.ledger.length - 1];
      return { id: r.id, health: r.health, peakKW: r.peakKW, t: l?.t, flameSilenced: l?.flameSilenced != null ? r4(l.flameSilenced / l.released) : null, walls: l ? r4(l.walls / l.released) : null };
    });
  }
  // The head-height temperature, series by series: the file's own statistics, and the scatter of ln(M/E) split
  // into the part within the series and the part between them.
  for (const [name, stats] of [["every", J.stats], ["rule", J.statsRule]]) {
    const cellsOf = stats?.byQuantity?.headTemperature || {};
    for (const [cell, v] of Object.entries(cellsOf)) {
      const series = {};
      for (const [d, qs] of Object.entries(stats.byDataset)) {
        const s = qs.headTemperature?.[cell];
        if (s) series[d] = { n: s.n, delta: s.delta, sigmaM: s.sigmaM, tier: s.tier, inBand: s.inBand, stopped: s.stopped, weighted: { delta: s.weighted?.delta, sigmaM: s.weighted?.sigmaM, inBand: s.weighted?.inBand } };
      }
      ((res.head[name] ||= {})[cell] = { pooled: v.pooled, weightedPooled: v.weighted?.pooled, inBand: v.datasetsInBand, inBandWeighted: v.weighted?.datasetsInBand, verdict: v.verdict, claim: v.claim, weightedVerdict: v.weighted?.verdict, weightedClaim: v.weighted?.claim, final: v.final, series });
    }
  }
  const kept = name => J.runs.filter((r) => name === "every" || r.representation?.represented !== false);
  for (const name of ["every", "rule"]) {
    for (const cell of Object.keys(res.head[name] || {})) {
      const ps = kept(name).filter((r) => String(r.cell) === cell).flatMap((r) => r.pairs).filter((p) => p.quantity === "headTemperature" && p.model > 0 && p.measured > 0);
      const logs = ps.map((p) => ({ d: p.dataset, x: Math.log(p.model / p.measured) }));
      const mean = sum(logs.map((l) => l.x)) / logs.length;
      const by = {};
      for (const l of logs) (by[l.d] ||= []).push(l.x);
      let within = 0, between = 0;
      const means = {};
      for (const [d, xs] of Object.entries(by)) {
        const m = sum(xs) / xs.length;
        means[d] = r4(m);
        within += sum(xs.map((x) => (x - m) ** 2));
        between += xs.length * (m - mean) ** 2;
      }
      const total = within + between;
      // The pooled statistic with each series' own mean taken out: what the scatter would be were every series unbiased alike.
      const centred = S.statistics(logs.map((l) => [Math.exp(l.x - (sum(by[l.d]) / by[l.d].length) + mean), 1]), S.SIGMA_E.headTemperature);
      // The pooled statistic with one series left out in turn: which series carry the scatter (a diagnosis, no word).
      res.head[name][cell].without = Object.fromEntries(Object.keys(by).map((d) => {
        const st = S.statistics(ps.filter((p) => p.dataset !== d).map((p) => [p.model, p.measured]), S.SIGMA_E.headTemperature);
        const sw = S.weightedStatistics(ps.filter((p) => p.dataset !== d).map((p) => [p.model, p.measured]), S.SIGMA_E.headTemperature);
        return [d, { n: st.n, delta: st.delta, sigmaM: st.sigmaM, weighted: { delta: sw.delta, sigmaM: sw.sigmaM } }];
      }));
      res.head[name][cell].scatter = { pairs: logs.length, sdLog: r4(Math.sqrt(total / (logs.length - 1))), withinShare: r4(within / total), betweenShare: r4(between / total), seriesMeanLog: means, sigmaMSeriesCentred: centred.sigmaM };
    }
  }
  out.results[job.label] = res;
}
// Each run's wall seconds in an arm with item 8 beside the base arm (labels r8-<cell> and r8b-<cell>).
for (const [label, res] of Object.entries(out.results)) {
  const base = runsOf[label.replace(/^r8-/, "r8b-")];
  if (!label.startsWith("r8-") || !base) continue;
  const by = {};
  for (const [id, r] of runsOf[label]) {
    const b = base.get(id);
    if (b && r.ok && b.ok && b.wallSeconds > 0) (by[r.dataset] ||= []).push(r.wallSeconds / b.wallSeconds);
  }
  res.wallRatioToBase = Object.fromEntries(Object.entries(by).map(([d, xs]) => [d, { runs: xs.length, median: r4(med(xs)), least: r4(Math.min(...xs)), most: r4(Math.max(...xs)) }]));
}

const f = (x, d = 3) => (x == null || !Number.isFinite(x) ? "n/a" : x.toFixed(d));
const e = (x) => (x == null || !Number.isFinite(x) ? "n/a" : x === 0 ? "0" : x.toExponential(2));
let md = "# Item 8's loose ends, from the saved files (plan 16.52; 16.54 part D)\n\nMade by `tools/calibration/item8-loose-ends.mjs`. Nothing here enters a statistic or a word.\n";
md += "\n## The heat ledger at each run's last row: median share of the heat released, by series\n\nRuns the health check stopped are left out. The remainder is released less gas, walls, the grid's sides, leak and `radiatedOut`.\n\n| Result | Series | Runs read | Gas | Walls | Out of the grid's sides | Leak | radiatedOut | Burner's rays out | Remainder (least to most) | Gross emission |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
for (const [label, r] of Object.entries(out.results)) for (const [d, s] of Object.entries(r.series)) {
  const m = s.medianShare;
  md += `| ${label} | ${d} | ${s.read} of ${s.runs} | ${f(m.gas)} | ${f(m.walls)} | ${f(m.shell)} | ${f(m.leak)} | ${f(m.radiatedOut)} | ${f(m.burnerRadiatedOut)} | ${f(m.remainder)} (${f(s.remainderRange[0])} to ${f(s.remainderRange[1])}) | ${f(m.radiatedGross, 2)} |\n`;
}
md += "\n## Item 8's tallies and the radRoundoff sums, by series (MJ summed over the runs read)\n\n| Result | Series | Released | radFloor | radCapped | radNegative | radShareNegative | surfaceFloor | surfaceCapped | flameSilenced | flameCleared | gasRays | gasCrossings | radRoundoff own | deposits | burner | Round-off over released: median, largest | Largest over the gross emission |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
for (const [label, r] of Object.entries(out.results)) for (const [d, s] of Object.entries(r.series)) {
  const t = s.talliesMJ, c = s.counts, ro = s.radRoundoffMJ;
  if (!ro && t.radFloor == null && t.surfaceFloor == null) continue;
  md += `| ${label} | ${d} | ${f(s.releasedMJ, 0)} | ${e(t.radFloor)} | ${e(t.radCapped)} | ${e(t.radNegative)} | ${e(t.radShareNegative)} | ${e(t.surfaceFloor)} | ${e(t.surfaceCapped)} | ${e(t.flameSilenced)} | ${e(t.flameCleared)} | ${c.gasRays == null ? "n/a" : c.gasRays.toExponential(2)} | ${c.gasCrossings == null ? "n/a" : c.gasCrossings.toExponential(2)} | ${e(ro?.own)} | ${e(ro?.deposits)} | ${e(ro?.burner)} | ${e(ro?.medianOfReleased)}, ${e(ro?.largestOfReleased)} | ${e(ro?.largestOfRadiated)} |\n`;
}
md += "\n## Wall seconds, item 8 on over the base arm, run by run\n\n| Result | Series | Runs | Median | Least | Most |\n| --- | --- | --- | --- | --- | --- |\n";
for (const [label, r] of Object.entries(out.results)) for (const [d, s] of Object.entries(r.wallRatioToBase || {})) md += `| ${label} | ${d} | ${s.runs} | ${f(s.median, 1)} | ${f(s.least, 1)} | ${f(s.most, 1)} |\n`;
md += "\n## FSE ISOHept26\n\n| Result | Cell | Health | Burner cells (columns, layers) | Released of expected, MJ | Run's wall s |\n| --- | --- | --- | --- | --- | --- |\n";
for (const [label, r] of Object.entries(out.results)) if (r.isoHept26) {
  const i = r.isoHept26, h = i.health;
  md += `| ${label} | ${i.cell} | ${h.ok ? `ran to ${i.horizon} s; fastest gas ${h.maxSpeed} m/s, ${h.substeps} substeps at most` : `stopped: ${h.reason} at ${h.time} s, reading ${h.value} K against the limit of ${r.health?.maxT ?? 3000} K`} | ${i.burner.cells} (${i.burner.columns}, ${i.burner.layers}) | ${i.burner.released} of ${i.burner.expectedMJ} | ${i.wallSeconds} |\n`;
}
md += "\nIts ledger, each row as shares of the heat released to that time:\n\n| Result | t, s | Released, MJ | Gas, MJ | Walls | Grid's sides | radiatedOut | Remainder | Gross emission | flameSilenced | flameCleared |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
for (const [label, r] of Object.entries(out.results)) for (const l of r.isoHept26?.ledger || []) md += `| ${label} | ${l.t} | ${f(l.released, 1)} | ${f(l.gasMJ, 2)} | ${f(l.walls)} | ${f(l.shell)} | ${f(l.radiatedOut)} | ${f(l.remainder)} | ${f(l.radiatedGross, 2)} | ${f(l.flameSilenced, 2)} | ${f(l.flameCleared, 3)} |\n`;
md += "\nThe series' other runs, at their last row:\n\n| Result | Run | Health | Peak kW | flameSilenced over released | Walls' share |\n| --- | --- | --- | --- | --- | --- |\n";
for (const [label, r] of Object.entries(out.results)) for (const o of r.fseOthers || []) md += `| ${label} | ${o.id} | ${o.health.ok ? "ran" : `stopped: ${o.health.reason} at ${o.health.time} s`} | ${o.peakKW} | ${f(o.flameSilenced, 2)} | ${f(o.walls)} |\n`;
md += "\n## The head-height temperature, series by series (the files' own statistics)\n\nEach series: pairs, delta / sigma_M, **in** where it is in band; then the 10-bin weighted delta / sigma_M.\n";
for (const name of ["every", "rule"]) {
  md += `\n### ${name === "every" ? "Every run" : "With the 2026-11-12 rule"}\n\n`;
  const rows = [];
  const ds = new Set();
  for (const [label, r] of Object.entries(out.results)) for (const [cell, v] of Object.entries(r.head[name] || {})) {
    rows.push([label, cell, v]);
    Object.keys(v.series).forEach((d) => ds.add(d));
  }
  const D = [...ds];
  md += `| Result | Cell | Pooled | Weighted pooled | In band | Word | ${D.join(" | ")} | sd of ln(M/E) | Share between series | sigma_M with each series' mean taken out |\n| --- | --- | --- | --- | --- | --- | ${D.map(() => "---").join(" | ")} | --- | --- | --- |\n`;
  for (const [label, cell, v] of rows) md += `| ${label} | ${cell} | ${f(v.pooled.delta, 2)} / ${f(v.pooled.sigmaM, 2)} (${v.pooled.n}) | ${f(v.weightedPooled?.delta, 2)} / ${f(v.weightedPooled?.sigmaM, 2)} | ${v.inBand.length} of ${Object.keys(v.series).length}; weighted ${v.inBandWeighted?.length ?? "n/a"} | ${v.final} | ${D.map((d) => {
    const s = v.series[d];
    return s ? `${s.n}: ${f(s.delta, 2)} / ${f(s.sigmaM, 2)}${s.inBand ? " **in**" : ""}; w ${f(s.weighted.delta, 2)} / ${f(s.weighted.sigmaM, 2)}${s.weighted.inBand ? " **in**" : ""}` : "";
  }).join(" | ")} | ${f(v.scatter.sdLog)} | ${f(v.scatter.betweenShare, 2)} | ${f(v.scatter.sigmaMSeriesCentred, 2)} |\n`;
  md += `\nThe pooled delta / sigma_M with one series left out in turn, then the 10-bin weighted:\n\n| Result | Cell | ${D.map((d) => `without ${d}`).join(" | ")} |\n| --- | --- | ${D.map(() => "---").join(" | ")} |\n`;
  for (const [label, cell, v] of rows) md += `| ${label} | ${cell} | ${D.map((d) => (v.without[d] ? `${f(v.without[d].delta, 2)} / ${f(v.without[d].sigmaM, 2)} (${v.without[d].n}); w ${f(v.without[d].weighted.delta, 2)} / ${f(v.without[d].weighted.sigmaM, 2)}` : "")).join(" | ")} |\n`;
}
const write = (name, text) => {
  if (!name) return;
  mkdirSync(path.dirname(path.resolve(name)), { recursive: true });
  writeFileSync(path.resolve(name), text);
  console.log(`wrote ${name}`);
};
write(flag("--json"), JSON.stringify(out, null, 1) + "\n");
write(flag("--md"), md);
