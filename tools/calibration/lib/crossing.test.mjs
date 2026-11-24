/**
 * The check plan 16.54 demands of the tenability tool before it reads any
 * result: made-up series with known crossings (a ramp, a step, a series that
 * never crosses, one that crosses between samples), the dose against a hand
 * value, the four outcomes, the exclusions, and the statistics against
 * lib/stats.mjs. Run: node --test tools/calibration/lib/crossing.test.mjs
 */
import test from "node:test";
import assert from "node:assert/strict";
import * as C from "./crossing.mjs";
import * as S from "./stats.mjs";
import { headPairs } from "./head.mjs";

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} is not ${b} within ${tol}`);
const times = Array.from({ length: 31 }, (_, i) => 10 * i);

test("a ramp crosses at its known time, on a sample and between samples", () => {
  near(C.firstCrossing(times, times.map((t) => 20 + 2 * t), 60).time, 20);
  near(C.firstCrossing(times, times.map((t) => 20 + 3 * t), 60).time, 40 / 3);
  near(C.firstCrossing(times, times.map((t) => 20 + 0.37 * t), 60).time, 40 / 0.37);
});

test("a step crosses on the straight line between its two samples", () => {
  const v = times.map((t) => (t >= 50 ? 100 : 20));
  // 20 at 40 s, 100 at 50 s: 60 is half way.
  near(C.firstCrossing(times, v, 60).time, 45);
  near(C.firstCrossing(times, times.map((t) => (t >= 50 ? 60 : 20)), 60).time, 50);
});

test("a series that never crosses gives null; one already there gives its first time; the first crossing is taken", () => {
  assert.equal(C.firstCrossing(times, times.map(() => 59.999), 60), null);
  assert.equal(C.firstCrossing(times, times.map((t) => 20 + 0.1 * t), 60), null);
  assert.deepEqual(C.firstCrossing(times, times.map(() => 80), 60), { time: 0, index: 0 });
  // Up at 100 to 110 s, down again, up for good later: the first is returned.
  const v = times.map((t) => (t === 110 ? 70 : t >= 200 ? 90 : 30));
  near(C.firstCrossing(times, v, 60).time, 100 + (10 * 30) / 40);
});

test("the dose of a constant temperature is the app's rate times the minutes, and crosses 0.3 at the hand value", () => {
  const rate = 100 ** 3.4 / 5e7; // a minute
  const dose = C.heatDose(times, times.map(() => 100));
  near(dose[0], 0);
  near(dose[6], rate, 1e-12);
  near(C.firstCrossing(times, dose, 0.3).time, (0.3 / rate) * 60, 1e-9);
  // Below 0 C the app's rate is 0.
  near(C.heatDose(times, times.map(() => -5))[30], 0);
});

test("the dose of a ramp is the trapezium sum, and its crossing lies on the line between the samples", () => {
  const T = times.map((t) => 20 + t);
  const r = (x) => x ** 3.4 / 5e7;
  let d = 0;
  const hand = [0];
  for (let i = 1; i < times.length; i += 1) hand.push((d += (0.5 * (r(T[i - 1]) + r(T[i])) * 10) / 60));
  const dose = C.heatDose(times, T);
  hand.forEach((x, i) => near(dose[i], x, 1e-12));
  const i = hand.findIndex((x) => x >= 0.3);
  near(C.firstCrossing(times, dose, 0.3).time, times[i - 1] + (10 * (0.3 - hand[i - 1])) / (hand[i] - hand[i - 1]), 1e-9);
});

// A made-up test and run: measured every 10 s from -20 to 300 s, the model every second in kelvin.
const AMB = 20;
const mt = Array.from({ length: 33 }, (_, i) => -20 + 10 * i);
const ramp = (b) => (t) => AMB + Math.max(0, b * t);
const mk = (devs, opts = {}) => {
  const T = {
    id: "made-up", ambientC: AMB, steady: opts.steady,
    burner: { centre: [0, 0], z0: 0, area: 0.25 },
    geometry: { obsts: [] },
    trees: [{ id: "far", x: 5, y: 5, head: devs.map((d, q) => ({ id: `d${q}`, z: 1.5 + 0.1 * q })) }, { id: "near", x: 0.2, y: 0, head: [{ id: "p", z: 1.8 }] }],
    measured: { times: mt, head: { far: Object.fromEntries(devs.map((d, q) => [`d${q}`, mt.map(d.measured)])), near: { p: mt.map(ramp(1)) } } },
  };
  const secs = Array.from({ length: 300 }, (_, i) => i + 1);
  const run = {
    dataset: "made", test: "made-up", id: "made/made-up@0.1", cell: 0.1, horizon: 300, health: opts.health || { ok: true },
    series: { from: 1, step: 1, head: { far: { z: devs.map((_, q) => 1.5 + 0.1 * q), T: devs.map((d) => secs.map((s) => d.model(s) + 273.15)) }, near: { z: [1.8], T: [secs.map(() => 293.15)] } } },
  };
  return { T, run };
};

test("the model's series is the mean of its 1 s readings within half the measured interval, and a device's crossing is the known time", () => {
  // Model 20 + 0.5 t each second: at a measured time t the mean over t - 4 .. t + 5 is 20 + 0.5 (t + 0.5).
  const { T, run } = mk([{ measured: ramp(0.25), model: ramp(0.5) }]);
  const [d] = C.deviceSeries(run, T).filter((x) => x.status === "ok");
  assert.equal(d.times[0], 10);
  assert.equal(d.times[d.times.length - 1], 300);
  assert.equal(d.spacing, 10);
  near(d.model[0], 20 + 0.5 * 10.5, 1e-9);
  const o = C.outcomeOf(d, "headCrossing60");
  assert.equal(o.outcome, "pair");
  near(o.model, 80 - 0.5, 1e-3); // 20 + 0.5 (t + 0.5) = 60
  near(o.measured, 160, 1e-3); // 20 + 0.25 t = 60
});

test("the four outcomes, the plume, and the unresolved start", () => {
  const { T, run } = mk([
    { measured: ramp(0.25), model: ramp(0.5) }, // both cross: a pair
    { measured: ramp(0.05), model: ramp(0.5) }, // the model alone: an early closure
    { measured: ramp(0.25), model: ramp(0.05) }, // the measurement alone: a miss
    { measured: ramp(0.05), model: ramp(0.05) }, // neither
    { measured: ramp(1.25), model: ramp(1) }, // measured at 32 s, within 10 + 3 x 10 s: unresolved
    { measured: ramp(0.999), model: ramp(1) }, // measured at 40.04 s, just past 40 s: a pair
    { measured: ramp(1.25), model: ramp(0.05) }, // measured early, the model never: a miss still (choice 6)
  ]);
  const ds = C.deviceSeries(run, T);
  assert.equal(ds.find((d) => d.tree === "near").status, "plume");
  const got = ds.filter((d) => d.status === "ok").map((d) => C.outcomeOf(d, "headCrossing60").outcome);
  assert.deepEqual(got, ["pair", "early", "miss", "neither", "unresolved", "pair", "miss"]);
  const read = C.readRuns([run], () => T).quantities.headCrossing60.every;
  assert.deepEqual(read.counts, { pair: 2, unresolved: 1, early: 1, miss: 2, neither: 1 });
  assert.equal(read.crosses, 5);
  near(read.missShare, 0.4);
  assert.equal(read.earlyMisses, 1);
  assert.equal(read.status.plume, 1);
});

test("the tool's series give the head-height reducer's own pairs", () => {
  const { T, run } = mk([{ measured: ramp(0.25), model: ramp(0.5) }, { measured: (t) => AMB + 40 * Math.exp(-(((t - 150) / 40) ** 2)), model: (t) => AMB + 55 * Math.exp(-(((t - 130) / 30) ** 2)) }]);
  const pairs = headPairs(run, T).pairs;
  const mine = C.deviceSeries(run, T).filter((d) => d.status === "ok");
  assert.equal(pairs.length, mine.length);
  mine.forEach((d) => {
    const p = pairs.find((x) => x.device === d.device);
    assert.deepEqual(C.peakRise(d, T), { model: p.model, measured: p.measured });
  });
});

test("steady tests and stopped runs give no outcome", () => {
  const a = mk([{ measured: ramp(0.25), model: ramp(0.5) }], { steady: true });
  assert.deepEqual(C.deviceSeries(a.run, a.T).map((d) => d.status), ["steady", "plume"]);
  const b = mk([{ measured: ramp(0.25), model: ramp(0.5) }], { health: { ok: false, reason: "temperature" } });
  assert.deepEqual(C.deviceSeries(b.run, b.T).map((d) => d.status), ["stopped", "plume"]);
  const read = C.readRuns([b.run], () => b.T).quantities.headCrossing60;
  assert.equal(read.every.devices, 0);
  assert.equal(read.every.unweighted.word, "no crossing measured");
});

test("the dose's time on a device is the hand value", () => {
  // Both constant from the first compared sample: measured 100 C, model 120 C.
  const { T, run } = mk([{ measured: () => 100, model: () => 120 }]);
  const [d] = C.deviceSeries(run, T).filter((x) => x.status === "ok");
  const o = C.outcomeOf(d, "headHeatDose");
  near(o.measured, 10 + (0.3 / (100 ** 3.4 / 5e7)) * 60, 1e-3);
  near(o.model, 10 + (0.3 / (120 ** 3.4 / 5e7)) * 60, 1e-2);
  assert.equal(o.outcome, "pair");
});

// Rows made by hand for the statistics and the word.
const row = (dataset, model, measured, outcome = "pair", represented = true) => ({ dataset, status: "ok", represented, headCrossing60: { outcome, model, measured, measuredEarly: false, modelAtFirst: false } });

test("the statistic is lib/stats.mjs's, and the word follows 16.54's rule", () => {
  const A = [[100, 100], [110, 100], [95, 100], [105, 110]].map(([m, e]) => row("a", m, e));
  const B = [[200, 190], [180, 200], [210, 200]].map(([m, e]) => row("b", m, e));
  const Cc = [[300, 150], [280, 150], [330, 160]].map(([m, e]) => row("c", m, e));
  const all = [...A, ...B, ...Cc];
  const r = C.readQuantity(all, "headCrossing60");
  const want = S.statistics(all.map((x) => [x.headCrossing60.model, x.headCrossing60.measured]), C.SIGMA_E_TIME);
  assert.equal(r.unweighted.delta, want.delta);
  assert.equal(r.unweighted.sigmaM, want.sigmaM);
  assert.equal(r.series.a.delta, S.statistics(A.map((x) => [x.headCrossing60.model, x.headCrossing60.measured]), C.SIGMA_E_TIME).delta);
  assert.deepEqual(r.seriesInBand, ["a", "b"]);
  // Pooled delta is out of band here (series c reads 2): not calibrated.
  assert.equal(r.unweighted.word, S.tier(want) === "within the calibrated band" ? "calibrated" : "not calibrated");
  // Two series in band of two, pooled in band, no miss: calibrated.
  const two = C.readQuantity([...A, ...B], "headCrossing60");
  assert.equal(two.unweighted.word, "calibrated");
  assert.equal(two.seriesAlike.word, "calibrated");
  // One miss among 8 crossing devices is 12.5 %: over 5 %, not calibrated whatever the statistic.
  const missed = C.readQuantity([...A, ...B, row("b", null, 120, "miss")], "headCrossing60");
  near(missed.missShare, 0.125);
  assert.equal(missed.unweighted.word, "not calibrated");
  // Early closures do not fail the rule by themselves.
  const early = C.readQuantity([...A, ...B, row("b", 50, null, "early"), row("b", 60, null, "early")], "headCrossing60");
  assert.equal(early.unweighted.word, "calibrated");
  // Pooled in band, but one series of three in band: indicative.
  const D = [[100, 70], [100, 140]].map(([m, e]) => row("d", m, e));
  const E = [[100, 72], [100, 139]].map(([m, e]) => row("e", m, e));
  const big = Array.from({ length: 40 }, (_, q) => row("a", 100 + (q % 5), 100));
  const ind = C.readQuantity([...big, ...D, ...E], "headCrossing60");
  assert.equal(ind.unweighted.tier, "within the calibrated band");
  assert.deepEqual(ind.seriesInBand, ["a"]);
  assert.equal(ind.unweighted.word, "indicative");
  // A stopped run puts its series out of band.
  const stop = C.readQuantity([...A, ...B], "headCrossing60", { b: 1 });
  assert.deepEqual(stop.seriesInBand, ["a"]);
  assert.equal(stop.unweighted.word, "indicative");
});

test("the series weighted alike: equal ratios give that ratio, and a large series does not outweigh a small one", () => {
  const same = [...Array(30).fill(["a"]), ...Array(3).fill(["b"])].map(([s], q) => [150 + q, (150 + q) / 1.5, s]);
  near(C.seriesAlikeStatistics(same, 0.07).delta, 1.5, 2e-3);
  const mixed = [...Array.from({ length: 50 }, (_, q) => [200 + q, 100 + q / 2, "a"]), [50, 100, "b"], [51, 102, "b"]];
  const alike = C.seriesAlikeStatistics(mixed, 0.07);
  const pooled = S.statistics(mixed.map(([m, e]) => [m, e]), 0.07);
  assert.ok(pooled.meanRatio > 1.8, "pooled follows the large series");
  // Geometric mean of 2 and 0.5 is 1: the mean log ratio is 0 when the series weigh alike.
  const lm = Math.log(alike.delta) - (alike.sigmaM ** 2 - alike.sigmaEUsed ** 2) / 2;
  near(lm, 0, 5e-3);
});

test("with the rule, the runs marked not represented are left out; the strictest word is taken", () => {
  const { T, run } = mk([{ measured: ramp(0.25), model: ramp(0.5) }]);
  const other = { ...run, id: "made/other@0.1", representation: { represented: false } };
  const got = C.readRuns([{ ...run, representation: { represented: true } }, other], () => T);
  assert.equal(got.quantities.headCrossing60.every.devices, 2);
  assert.equal(got.quantities.headCrossing60.rule.devices, 1);
  assert.deepEqual(got.notRepresented, ["made/other@0.1"]);
  assert.equal(C.strictest(["calibrated", "indicative", "calibrated"]), "indicative");
  assert.equal(C.strictest(["indicative", "not calibrated", "calibrated"]), "not calibrated");
});
