/**
 * The tenability quantities at head height (fire-calibration-plan-2026-11-11.md,
 * section 16.54; written 2026-11-24, before any was computed on a result):
 *
 *   headCrossing60  the time at which the gas first reaches 60 C;
 *   headHeatDose    the time at which the app's heat dose reaches 0.3
 *                   (tenability.js heatRate: T^3.4 / 5e7 a minute, T in C).
 *
 * Devices, plume rule, cell, times and averaging are those of the head-height
 * temperature (lib/head.mjs): deviceSeries below repeats headPairs' selection
 * line for line, and peakRise lets a caller check it against headPairs on the
 * same run. The statistics are lib/stats.mjs's. The choices where the plan is
 * silent are in benchmark/results/2026-11-24-tenability-choices.md.
 */
import { plumeCheck } from "./head.mjs";
import * as S from "./stats.mjs";
import { heatRate, FED_CRIT } from "../../../topologicpy-web-frontend/src/core/tenability.js";

export const LEVEL_C = 60;
export const DOSE_LIMIT = FED_CRIT;
/** The guide gives no sigma_E for a time: the head-height temperature's is used (choice 8). */
export const SIGMA_E_TIME = S.SIGMA_E.headTemperature;
export const QUANTITIES = Object.freeze(["headCrossing60", "headHeatDose"]);
export const MISS_LIMIT = 0.05;
export const UNRESOLVED_INTERVALS = 3;

/**
 * The first time a series is at or above a level: a straight line between the
 * two samples that bracket it; the first sample's time if it is already there;
 * null if it never is. Returns {time, index (of the first sample at or above)}.
 */
export function firstCrossing(times, values, level) {
  for (let i = 0; i < times.length; i += 1) {
    if (!(values[i] >= level)) continue;
    if (i === 0) return { time: times[0], index: 0 };
    const a = values[i - 1], b = values[i];
    return { time: times[i - 1] + ((times[i] - times[i - 1]) * (level - a)) / (b - a), index: i };
  }
  return null;
}

/** The heat dose summed by the trapezium rule over the times (s), zero at the first; temperatures in C. */
export function heatDose(times, tempsC) {
  const dose = new Array(times.length).fill(0);
  for (let i = 1; i < times.length; i += 1) dose[i] = dose[i - 1] + (0.5 * (heatRate(tempsC[i - 1]) + heatRate(tempsC[i])) * (times[i] - times[i - 1])) / 60;
  return dose;
}

/**
 * Each head-height device of a run with its two series at the compared
 * times, as lib/head.mjs headPairs selects and averages them (absolute C).
 * Returns a list of {device, tree, status, why?, times, model, measured,
 * spacing}; status is "ok", "plume", "missing", "steady", "stopped" or
 * "no measurement".
 */
export function deviceSeries(run, T) {
  const out = [];
  const series = run.series?.head;
  const h = run.cell;
  const from = run.series?.from ?? 1, step = run.series?.step ?? 1;
  const M = T.measured;
  for (const tree of T.trees || []) {
    for (const dev of tree.head || []) {
      const device = `${tree.id}/${dev.id}`;
      const base = { device, tree: tree.id, z: dev.z };
      const plume = plumeCheck(T, tree, dev);
      if (plume.inPlume) {
        out.push({ ...base, status: "plume", why: `in the plume: r ${plume.r} m < ${plume.reach} m` });
        continue;
      }
      if (T.steady) {
        out.push({ ...base, status: "steady", why: "a steady test has no series" });
        continue;
      }
      if (!run.health?.ok) {
        out.push({ ...base, status: "stopped", why: `the health check stopped the run${run.health?.reason ? `: ${run.health.reason}` : ""}` });
        continue;
      }
      const s = series?.[tree.id];
      if (!s || !s.z.length) {
        out.push({ ...base, status: "missing", why: "no stored cells at head height" });
        continue;
      }
      if (s.end != null && dev.z > s.end) {
        out.push({ ...base, status: "missing", why: `above the tree's end at ${s.end} m (plan 13.22)` });
        continue;
      }
      let best = -1, dz = Infinity;
      s.z.forEach((z, q) => {
        if (Math.abs(z - dev.z) < dz) (dz = Math.abs(z - dev.z)), (best = q);
      });
      if (dz > h + 1e-6) {
        out.push({ ...base, status: "missing", why: `nearest open cell ${dz.toFixed(2)} m away` });
        continue;
      }
      const temps = s.T[best];
      const meas = M.head?.[tree.id]?.[dev.id];
      if (!meas) {
        out.push({ ...base, status: "no measurement", why: "the data hold no series for it" });
        continue;
      }
      const dts = M.times.slice(1).map((t, q) => t - M.times[q]).sort((a, b) => a - b);
      const spacing = dts[dts.length >> 1] || 1;
      const times = [], model = [], measured = [];
      M.times.forEach((t, iq) => {
        if (t < 1 || t > run.horizon || meas[iq] == null) return;
        const lo = t - spacing / 2, hi = t + spacing / 2;
        let sum = 0, n = 0;
        temps.forEach((v, q) => {
          const ts = from + q * step;
          if (ts > lo && ts <= hi) (sum += v), (n += 1);
        });
        if (!n) return;
        times.push(t);
        model.push(sum / n - 273.15);
        measured.push(meas[iq]);
      });
      if (!times.length) {
        out.push({ ...base, status: "missing", why: "no compared time" });
        continue;
      }
      out.push({ ...base, status: "ok", times, model, measured, spacing });
    }
  }
  return out;
}

/** The head-height temperature's pair from a device's series (peak rises), as headPairs rounds it. */
export function peakRise(d, T) {
  return { model: +(Math.max(...d.model) - T.ambientC).toFixed(4), measured: +(Math.max(...d.measured) - T.ambientC).toFixed(4) };
}

/**
 * One device and one quantity: the two crossing times and the outcome,
 * "pair", "unresolved" (both cross, the measured one within three measured
 * intervals of the start), "early" (the model alone), "miss" (the
 * measurement alone) or "neither".
 */
export function outcomeOf(d, quantity) {
  const level = quantity === "headHeatDose" ? DOSE_LIMIT : LEVEL_C;
  const m = quantity === "headHeatDose" ? heatDose(d.times, d.model) : d.model;
  const e = quantity === "headHeatDose" ? heatDose(d.times, d.measured) : d.measured;
  const cm = firstCrossing(d.times, m, level), ce = firstCrossing(d.times, e, level);
  const limit = d.times[0] + UNRESOLVED_INTERVALS * d.spacing;
  const early = ce != null && ce.time <= limit;
  const outcome = cm && ce ? (early ? "unresolved" : "pair") : cm ? "early" : ce ? "miss" : "neither";
  return { outcome, model: cm ? +cm.time.toFixed(3) : null, measured: ce ? +ce.time.toFixed(3) : null, measuredEarly: early, modelAtFirst: cm ? cm.index === 0 : false, endModel: +m[m.length - 1].toFixed(5), endMeasured: +e[e.length - 1].toFixed(5) };
}

/** Every device of every run: rows {dataset, test, cell, device, status, represented, peak, headCrossing60, headHeatDose}. */
export function rowsOf(runs, testOf) {
  const rows = [];
  for (const run of runs) {
    const T = testOf(run.dataset, run.test);
    if (!T) continue;
    for (const d of deviceSeries(run, T)) {
      const row = { dataset: run.dataset, test: T.id, cell: run.cell, device: d.device, z: d.z, status: d.status, ambientC: T.ambientC, represented: run.representation ? run.representation.represented !== false : true };
      if (d.why) row.why = d.why;
      if (d.status === "ok") {
        row.first = d.times[0];
        row.last = d.times[d.times.length - 1];
        row.spacing = d.spacing;
        row.samples = d.times.length;
        row.peak = peakRise(d, T);
        for (const q of QUANTITIES) row[q] = outcomeOf(d, q);
      }
      rows.push(row);
    }
  }
  return rows;
}

/**
 * The suite's weighted statistic (lib/stats.mjs weightedStatistics) with each
 * pair's weight n over the pairs in its series in place of its bin: every
 * series weighs alike (choice 9c). pairs: [model, measured, series].
 */
export function seriesAlikeStatistics(pairs, sigmaE) {
  const P = pairs.filter(([m, e]) => m > 0 && e > 0 && Number.isFinite(m) && Number.isFinite(e));
  const n = P.length;
  if (n < 2) return { n, delta: n ? +(P[0][0] / P[0][1]).toFixed(3) : null, sigmaM: null, seriesAlike: true };
  const count = {};
  for (const [, , s] of P) count[s] = (count[s] || 0) + 1;
  const w = P.map(([, , s]) => n / count[s]);
  const sw = w.reduce((a, b) => a + b, 0);
  const lE = P.reduce((a, [, e], q) => a + Math.log(e) * w[q], 0) / sw;
  const lM = P.reduce((a, [m], q) => a + Math.log(m) * w[q], 0) / sw;
  const u2 = sw - 1 > 0 ? P.reduce((a, [m, e], q) => a + w[q] * (Math.log(m) - Math.log(e) - (lM - lE)) ** 2, 0) / (sw - 1) : 0;
  const u = Math.sqrt(u2);
  const sE = Math.min(u / Math.SQRT2, sigmaE);
  const sM = Math.sqrt(Math.max(0, u2 - sE * sE));
  const delta = Math.exp(lM - lE + (sM * sM) / 2 - (sE * sE) / 2);
  return { n, delta: +delta.toFixed(3), sigmaM: +sM.toFixed(3), sigmaEUsed: +sE.toFixed(3), seriesAlike: true };
}

const IN_BAND = "within the calibrated band";
const RANK = { calibrated: 2, indicative: 1, "not calibrated": 0, "too few pairs": 0, "no crossing measured": 0 };
export const strictest = (words) => words.reduce((a, b) => (RANK[b] < RANK[a] ? b : a));

/**
 * One quantity on one set of rows (one cell): the outcomes' counts, the
 * misses' share, each series' statistic and the word by 16.54's rule under
 * each weighting. stoppedBy: {dataset: stopped runs on the cell}.
 */
export function readQuantity(rows, quantity, stoppedBy = {}) {
  const ok = rows.filter((r) => r.status === "ok");
  const counts = { pair: 0, unresolved: 0, early: 0, miss: 0, neither: 0 };
  for (const r of ok) counts[r[quantity].outcome] += 1;
  const status = {};
  for (const r of rows) status[r.status] = (status[r.status] || 0) + 1;
  const crosses = counts.pair + counts.unresolved + counts.miss;
  const missShare = crosses ? counts.miss / crosses : null;
  const earlyMisses = ok.filter((r) => r[quantity].outcome === "miss" && r[quantity].measuredEarly).length;
  const resolved = counts.pair + counts.miss - earlyMisses;
  const pairs = ok.filter((r) => r[quantity].outcome === "pair");
  const modelAtFirst = pairs.filter((r) => r[quantity].modelAtFirst).length;
  const asPair = (r) => [r[quantity].model, r[quantity].measured, r.dataset];
  const datasets = [...new Set(ok.map((r) => r.dataset))];
  const series = {};
  for (const d of datasets) {
    const mine = ok.filter((r) => r.dataset === d);
    const ps = pairs.filter((r) => r.dataset === d).map(asPair);
    const c = { pair: 0, unresolved: 0, early: 0, miss: 0, neither: 0 };
    for (const r of mine) c[r[quantity].outcome] += 1;
    const s = S.statistics(ps, SIGMA_E_TIME);
    const sw = S.weightedStatistics(ps, SIGMA_E_TIME);
    const stopped = stoppedBy[d] || 0;
    series[d] = { devices: mine.length, ...c, n: s.n, delta: s.delta, sigmaM: s.sigmaM, tier: S.tier(s), inBand: S.tier(s) === IN_BAND && stopped === 0, weighted: { delta: sw.delta, sigmaM: sw.sigmaM, tier: S.tier(sw), inBand: S.tier(sw) === IN_BAND && stopped === 0 }, stopped };
  }
  const withPairs = datasets.filter((d) => series[d].n >= 1);
  const all = pairs.map(asPair);
  const word = (pooled, inBand) => {
    if (!crosses) return "no crossing measured";
    if (pooled.n < 2) return "too few pairs";
    const pooledIn = S.tier(pooled) === IN_BAND, missesOk = missShare <= MISS_LIMIT;
    const most = inBand.length > withPairs.length / 2;
    return pooledIn && most && missesOk ? "calibrated" : pooledIn && missesOk ? "indicative" : "not calibrated";
  };
  const inU = withPairs.filter((d) => series[d].inBand), inW = withPairs.filter((d) => series[d].weighted.inBand);
  const pooled = S.statistics(all, SIGMA_E_TIME);
  const binned = S.weightedStatistics(all, SIGMA_E_TIME);
  const alike = seriesAlikeStatistics(all, SIGMA_E_TIME);
  const ratios = all.map(([m, e]) => m / e).sort((a, b) => a - b);
  return {
    devices: ok.length, status, counts, crosses, missShare: missShare == null ? null : +missShare.toFixed(4), missesWithin: missShare == null ? null : missShare <= MISS_LIMIT,
    earlyMisses, missShareResolved: resolved ? +((counts.miss - earlyMisses) / resolved).toFixed(4) : null, modelAtFirst,
    medianRatio: ratios.length ? +ratios[ratios.length >> 1].toFixed(3) : null,
    series, seriesWithPairs: withPairs, seriesInBand: inU, seriesOutOfBand: withPairs.filter((d) => !series[d].inBand), seriesInBandWeighted: inW,
    unweighted: { ...pooled, tier: S.tier(pooled), word: word(pooled, inU) },
    binWeighted: { ...binned, tier: S.tier(binned), word: word(binned, inW) },
    seriesAlike: { ...alike, tier: S.tier(alike), word: word(alike, inU) },
  };
}

/**
 * Both quantities on a results file's runs (one cell): every run and with
 * the 2026-11-12 rule (the runs the file marks as not represented left out),
 * the three readings of 16.54 and the strictest word over all six (choice 9).
 */
export function readRuns(runs, testOf) {
  const rows = rowsOf(runs, testOf);
  const stoppedBy = (list) => {
    const out = {};
    for (const r of list) if (!r.health?.ok) out[r.dataset] = (out[r.dataset] || 0) + 1;
    return out;
  };
  const kept = runs.filter((r) => !r.representation || r.representation.represented !== false);
  const out = { rows, hasRepresentation: runs.some((r) => r.representation), notRepresented: runs.filter((r) => r.representation && r.representation.represented === false).map((r) => r.id).sort(), quantities: {} };
  for (const q of QUANTITIES) {
    const every = readQuantity(rows, q, stoppedBy(runs));
    const rule = readQuantity(rows.filter((r) => r.represented), q, stoppedBy(kept));
    const words = { pooled: every.unweighted.word, seriesAlike: every.seriesAlike.word, rule: rule.unweighted.word };
    out.quantities[q] = { every, rule, words, binWeightedWord: every.binWeighted.word, strictest: strictest([every.unweighted.word, every.binWeighted.word, every.seriesAlike.word, rule.unweighted.word, rule.binWeighted.word, rule.seriesAlike.word]) };
  }
  return out;
}
