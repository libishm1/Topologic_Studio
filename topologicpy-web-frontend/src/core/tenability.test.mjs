/**
 * Smoke, gas and tenability measures (core/tenability.js):
 *   node --test src/core/tenability.test.mjs
 */
import test from "node:test";
import assert from "node:assert/strict";

import {
  CO_DOSE,
  DETECTOR_LENGTH,
  DETECTOR_MIN_SPEED,
  DOSE_ISO,
  DOSE_PURSER,
  FED_CRIT,
  PURSER_CO,
  asphyxiantRate,
  chamberStep,
  detectorChambers,
  extinction,
  heatRate,
  hyperventilation,
  limitingCriterion,
  obscuration,
  opticalDensity,
  purserRate,
  runIndex,
  tenabilityTracker,
  visibilityOf,
} from "./tenability.js";
import { visibility } from "./flamescapesRun.js";
import { forecastFromFrames, runSummary } from "./forecast.js";
import { topologicGraph } from "./topologicExport.js";

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`);

test("smoke optics follow Bouguer's law from the soot concentration, and agree with the solver's visibility", () => {
  // 34.5 mg/m^3 of soot: K = 8.7 m^2/g x 0.0345 g/m^3 = 0.3 /m, a 10 m sign.
  close(extinction(34.4827586), 0.3, 1e-6);
  close(visibilityOf(34.4827586), 10, 1e-6);
  close(obscuration(34.4827586), 100 * (1 - Math.exp(-0.3)), 1e-6);
  close(opticalDensity(34.4827586), 0.3 / Math.LN10, 1e-6);
  for (const mg of [0, 1, 10, 50, 300, 5000]) close(visibilityOf(mg), visibility(mg / 1e6), 1e-12);
  assert.equal(visibilityOf(0), 100);
});

test("the asphyxiant dose is ISO 13571's: CO over 35 000 ppm min, CO2's factor above 2 %, low O2 negligible in air", () => {
  close(asphyxiantRate(CO_DOSE, null, null), 1);
  close(asphyxiantRate(3500, 0.04, 20.9), 0.1 + 1 / Math.exp(8.13), 1e-12);
  assert.equal(hyperventilation(2), 1);
  close(hyperventilation(5), Math.E);
  close(asphyxiantRate(3500, 5, 20.9) - asphyxiantRate(0, 5, 20.9), 0.1 * Math.E);
  // Purser's low-oxygen term: about 1/3394 a minute in air, 1/60 or so at 10 %.
  assert.ok(asphyxiantRate(0, 0, 20.9) < 3e-4);
  close(asphyxiantRate(0, 0, 10), 1 / Math.exp(8.13 - 0.54 * 10.9));
});

test("the convected-heat dose rate is ISO 13571's Eq. 10, 1 / (5e7 T^-3.4) a minute", () => {
  close(heatRate(60), 1 / (5e7 * Math.pow(60, -3.4)));
  assert.equal(heatRate(0), 0);
  // At 120 C a full dose takes about 4 min.
  const minutes = 1 / heatRate(120);
  assert.ok(minutes > 3 && minutes < 5, `${minutes}`);
});

test("the index is the largest share of a limit reached, and reaches 1 exactly when a criterion does", () => {
  const n = 4;
  const tracker = tenabilityTracker(n, { ambient: 20, limitC: 60, visibilityLimit: 10 });
  const frame = {
    T: Float32Array.from([20, 60, 20, 40]),
    vis: Float32Array.from([100, 100, 5, 100]),
    co: Float32Array.from([0, 0, 0, 0]),
  };
  const ti = tracker.step(frame, 1 / 60);
  // Clear air reads 0: the visibility cap (100 m) is not a tenth of the way to the limit.
  assert.ok(ti[0] < 1e-3, `clear air: ${ti[0]}`);
  close(ti[1], 1, 1e-6); // at the temperature limit
  close(ti[2], 2, 1e-6); // visibility half the limit
  close(ti[3], 0.5, 1e-6); // half way to the temperature limit
  // A dose grows with time at a steady reading: 10 500 ppm of CO reaches 0.3 in one minute.
  const gas = tenabilityTracker(1);
  let last = 0;
  for (let k = 0; k < 6; k += 1) last = gas.step({ T: [20], vis: [100], co: [10500] }, 10 / 60)[0];
  close(gas.fedGas[0], 10500 / CO_DOSE, 1e-6);
  close(last, gas.fedGas[0] / FED_CRIT, 1e-6);
  assert.equal(limitingCriterion({ T: 20, vis: 100, fedGas: 0.3, fedHeat: 0.01 }), "gases");
  assert.equal(limitingCriterion({ T: 20, vis: 4, fedGas: 0.01, fedHeat: 0 }), "smoke");
});

test("the doses never fall, so the index's dose parts never fall while a fire burns", () => {
  const tracker = tenabilityTracker(1);
  let fed = 0;
  for (const co of [0, 500, 2000, 100, 0]) {
    tracker.step({ T: [25], vis: [100], co: [co] }, 1);
    assert.ok(tracker.fedGas[0] >= fed);
    fed = tracker.fedGas[0];
  }
});

/** A run's frames as the page keeps them: one node at room temperature, clear, with CO only. */
const framesOf = (count, co) => Array.from({ length: count }, (_, k) => ({ time: 10 * k, T: Float32Array.of(20), vis: Float32Array.of(100), co: Float32Array.of(co) }));

test("a run restarted with new frames starts its index again, without the abandoned frames' doses (state-machines.md SM-1)", () => {
  const opts = { limitC: 60, smoke: true };
  // The GPU run gets to frame 5, painted as it came: CO enough to dose.
  const run = { frames: framesOf(6, 20000) };
  for (let k = 0; k <= 5; k += 1) runIndex(run, k, opts);
  assert.ok(run.ti[5][0] > 0.5);
  // It restarts on the CPU: its frames are dropped, and the new run is past
  // frame 5 when the playback asks for frame 6, so there are more frames than
  // before. This run has no CO: its index is a room's heat dose, not the old CO dose.
  run.frames.length = 0;
  run.frames.push(...framesOf(7, 0));
  const fresh = { frames: framesOf(7, 0) };
  assert.deepEqual(Array.from(runIndex(run, 6, opts)), Array.from(runIndex(fresh, 6, opts)));
  assert.ok(runIndex(run, 6, opts)[0] < 0.01);
  // The same frames painted again keep their index; a change of limits starts it again.
  const kept = run.ti;
  runIndex(run, 6, opts);
  assert.equal(run.ti, kept);
  runIndex(run, 6, { ...opts, limitC: 80 });
  assert.notEqual(run.ti, kept);
});

// ---------------------------------------------------------------- the dose's form

/** The asphyxiant rate as it was before `doseForm` existed, kept here to compare against. */
const rateBefore = (co, co2, o2) => {
  const c = Math.max(0, co) / 35000;
  const nu = co2 == null ? 1 : co2 > 2 ? Math.exp(co2 / 5) : 1;
  const lowO2 = o2 == null ? 0 : 1 / Math.exp(8.13 - 0.54 * (20.9 - Math.min(20.9, o2)));
  return c * nu + lowO2;
};

/** Stacked frames (forecast.js) of K one-minute frames at steady readings, a node a row. */
const steady = (K, rows) => {
  const n = rows.length;
  const fill = (key) => Float32Array.from({ length: K * n }, (_, i) => rows[i % n][key]);
  return { n, K, frameSeconds: 60, T: fill("T"), vis: fill("vis"), co: fill("co"), co2: fill("co2"), o2: fill("o2"), hrr: new Float32Array(K).fill(100), stats: { released: 1e6 } };
};

test("with no dose form asked for, every dose is the number it was: ISO 13571's form is the default", () => {
  const gases = [];
  for (const co of [-5, 0, 1, 250, 3500, 35000, 120000]) {
    for (const co2 of [null, 0, 0.04, 1.99, 2, 2.01, 5, 12]) {
      for (const o2 of [null, 20.95, 20.9, 20, 19.99, 13, 6]) gases.push([co, co2, o2]);
    }
  }
  for (const [co, co2, o2] of gases) {
    const was = rateBefore(co, co2, o2);
    // Bit for bit, not within a tolerance.
    assert.equal(asphyxiantRate(co, co2, o2), was);
    assert.equal(asphyxiantRate(co, co2, o2, DOSE_ISO), was);
    assert.equal(asphyxiantRate(co, co2, o2, undefined, "heavy"), was, "the activity belongs to Purser's form only");
    assert.equal(asphyxiantRate(co, co2, o2, "something else"), was, "an unknown form reads as ISO's");
  }
  assert.equal(DOSE_ISO, "iso13571");
  // The tracker, the forecast and the run summary, with the option left out and with ISO's named.
  const rows = [
    { T: 25, vis: 100, co: 3500, co2: 5, o2: 15 },
    { T: 40, vis: 30, co: 800, co2: 1, o2: 19.5 },
    { T: 20, vis: 100, co: 0, co2: 0, o2: 20.9 },
  ];
  const frames = steady(6, rows);
  const frame = { T: rows.map((r) => r.T), vis: rows.map((r) => r.vis), co: rows.map((r) => r.co), co2: rows.map((r) => r.co2), o2: rows.map((r) => r.o2) };
  const plain = tenabilityTracker(3), named = tenabilityTracker(3, { doseForm: DOSE_ISO });
  for (let k = 0; k < 6; k += 1) assert.deepEqual(plain.step(frame, 1), named.step(frame, 1));
  assert.deepEqual(plain.fedGas, named.fedGas);
  for (let v = 0; v < 3; v += 1) {
    let sum = 0;
    for (let k = 0; k < 6; k += 1) sum += rateBefore(rows[v].co, rows[v].co2, rows[v].o2) * 1;
    assert.equal(plain.fedGas[v], sum);
  }
  const a = forecastFromFrames(frames, { threshold: 60 }), b = forecastFromFrames(frames, { threshold: 60, doseForm: DOSE_ISO });
  assert.deepEqual(a.rate, b.rate);
  assert.deepEqual(a.untenable, b.untenable);
  // The forecast's rate is the heat rate plus the rate as it was, a second (frames hold 32-bit readings).
  for (let v = 0; v < 3; v += 1) {
    const r = (heatRate(Math.fround(rows[v].T)) + rateBefore(Math.fround(rows[v].co), Math.fround(rows[v].co2), Math.fround(rows[v].o2))) / 60;
    assert.equal(a.rate[v], Math.fround(r));
  }
  assert.deepEqual(runSummary(frames, { threshold: 60 }), runSummary(frames, { threshold: 60, doseForm: DOSE_ISO }));
});

test("Purser's form gives FDS's expected FED for its case of O2, CO2 and CO alone (Verification Guide, FED_FIC)", () => {
  // The FDS Verification Guide's section "FED and FIC of gas mixtures" has
  // four cases at fixed concentrations (its table, lines 4397 to 4399 of
  // FDS_Verification_Guide.tex). Only the first, "O2, CO2, CO", has no gas
  // this solver lacks; "Asphyxiants" and "All Gases" add NO and HCN, and
  // "Irritants" has no CO. The guide plots the expected values (its figure
  // FED_FIC_plot) without printing them; they are the column
  // "FED (O2 CO2 CO)" of Verification/Species/FED_FIC.csv in the FDS
  // repository, and the concentrations are those of FED_FIC.fds, in mol/mol:
  // O2 0.09772709, CO2 0.03430594, CO 0.00324186. (The guide's table prints
  // the CO2 as 343066 ppm, ten times the input file's 34306 ppm; the
  // expected values follow the input file.)
  const co = 0.00324186 * 1e6, co2 = 0.03430594 * 100, o2 = 0.09772709 * 100;
  const expected = [[20, 0.11988], [40, 0.23976], [60, 0.35964], [80, 0.47952], [100, 0.5994]];
  const rate = asphyxiantRate(co, co2, o2, DOSE_PURSER);
  assert.equal(rate, purserRate(co, co2, o2), "light work is the default, as FDS's");
  // To the five figures FDS's file gives.
  for (const [seconds, fed] of expected) close((rate * seconds) / 60, fed, 5e-5);
  // The same through the tracker, in the file's 20 s steps.
  const tracker = tenabilityTracker(1, { doseForm: DOSE_PURSER });
  for (const [, fed] of expected) {
    tracker.step({ T: [20], vis: [100], co: [co], co2: [co2], o2: [o2] }, 20 / 60);
    close(tracker.fedGas[0], fed, 5e-5);
  }
  // And through the forecast: five 20 s frames; the dose passes 0.3 in the third (0.35964), which starts at 40 s.
  const K = 5, one = (x) => new Float32Array(K).fill(x);
  const frames = { n: 1, K, frameSeconds: 20, T: one(20), vis: one(100), co: one(co), co2: one(co2), o2: one(o2) };
  assert.equal(forecastFromFrames(frames, { threshold: 60, doseForm: DOSE_PURSER }).untenable[0], 40);
  // ISO's form on the same gases is another number (0.506 at 100 s; this app's own arithmetic), so the option is read.
  close((asphyxiantRate(co, co2, o2) * 100) / 60, 0.50639, 1e-4);
  // At a limit of 0.33 Purser's dose still closes the node in the third frame (0.35964); ISO's (0.304 by then) in the fourth.
  assert.equal(forecastFromFrames(frames, { threshold: 60, fedCrit: 0.33, doseForm: DOSE_PURSER }).untenable[0], 40);
  assert.equal(forecastFromFrames(frames, { threshold: 60, fedCrit: 0.33 }).untenable[0], 60);
});

test("Purser's terms: CO by activity, CO2's factor at any level, low oxygen only below 20 %", () => {
  // CO: 3.317e-5 ppm^1.036 RMV / D; light work (25 L/min, 30 %) is the guide's 2.764e-5.
  close(PURSER_CO.light, 2.764e-5, 1e-4);
  close(PURSER_CO.rest, 0.70486e-5, 1e-4);
  close(PURSER_CO.heavy, 8.2925e-5, 1e-4);
  close(purserRate(1000, null, null), 2.7641667e-5 * Math.pow(1000, 1.036), 1e-7);
  close(purserRate(1000, null, null, "rest") / purserRate(1000, null, null), 8.5 / 40 / (25 / 30), 1e-12);
  close(purserRate(1000, null, null, "heavy") / purserRate(1000, null, null), 50 / 20 / (25 / 30), 1e-12);
  assert.equal(purserRate(1000, null, null, "running"), purserRate(1000, null, null), "an unknown activity reads as light work");
  assert.equal(purserRate(0, 5, 20.9), 0);
  assert.equal(purserRate(-3, 5, 20.9), 0);
  // CO2: exp(0.1903 % + 2.0004) / 7.1, below ISO's 2 % too; none when the run kept no CO2.
  close(purserRate(1000, 1, null) / purserRate(1000, null, null), Math.exp(0.1903 + 2.0004) / 7.1, 1e-12);
  close(purserRate(1000, 5, null) / purserRate(1000, null, null), Math.exp(0.1903 * 5 + 2.0004) / 7.1, 1e-12);
  assert.ok(purserRate(1000, 1, null) > purserRate(1000, null, null));
  assert.equal(asphyxiantRate(1000, 1, null), asphyxiantRate(1000, null, null), "ISO's factor starts above 2 %");
  assert.equal(purserRate(1000, 0, null), purserRate(1000, null, null), "as FDS, no factor with no CO2 at all");
  // Low oxygen: nothing at 20 % and above (ISO's form counts 1/3394 a minute in air), Purser's term below.
  assert.equal(purserRate(0, 0, 20.9), 0);
  assert.equal(purserRate(0, 0, 20), 0);
  close(purserRate(0, 0, 19.99), 1 / Math.exp(8.13 - 0.54 * 0.91), 1e-12);
  close(purserRate(0, 0, 10), 1 / Math.exp(8.13 - 0.54 * 10.9), 1e-12);
  assert.equal(purserRate(0, 0, 10), asphyxiantRate(0, 0, 10), "the same low-oxygen term as ISO's form, once below 20 %");
  assert.equal(purserRate(0, 0, null), 0);
});

test("the export says which dose form it used, and is as it was with ISO's", () => {
  const graph = { points: Float32Array.from([0, 0, 0, 1, 0, 0]), edges: Uint32Array.from([0, 1]), kinds: Uint8Array.from([0, 0]), upAxis: "y", agentHeight: 0 };
  const frames = [
    { time: 30, T: [20, 20], vis: [100, 100], co: [3000, 0], soot: [0, 0], co2: [3, 0], o2: [12, 20.9], hrr: 10 },
    { time: 60, T: [20, 20], vis: [100, 100], co: [3000, 0], soot: [0, 0], co2: [3, 0], o2: [12, 20.9], hrr: 10 },
  ];
  const plain = topologicGraph({ graph, frames, tenability: { untenableC: 60 } });
  const iso = topologicGraph({ graph, frames, tenability: { untenableC: 60, doseForm: DOSE_ISO } });
  const purser = topologicGraph({ graph, frames, tenability: { untenableC: 60, doseForm: DOSE_PURSER } });
  assert.deepEqual(iso, plain);
  assert.equal("dose_form" in plain.properties, false);
  assert.equal(purser.properties.dose_form, "purser");
  assert.match(purser.properties.tenability_method, /Purser's form as FDS/);
  assert.match(plain.properties.tenability_method, /CO \/ 35000 per min/);
  // One minute in all at steady gases.
  close(plain.vertices.Vertex_0.fed_asphyxiant, rateBefore(3000, 3, 12), 1e-4);
  close(purser.vertices.Vertex_0.fed_asphyxiant, purserRate(3000, 3, 12), 1e-4);
  assert.notEqual(purser.vertices.Vertex_0.fed_asphyxiant, plain.vertices.Vertex_0.fed_asphyxiant);
});

// ---------------------------------------------------------- a detector's entry lag

test("a detector's chamber follows a step in smoke as the closed form Y_0 (1 - exp(-t u / L))", () => {
  // The FDS Verification Guide's section "Smoke Detector Model" (lines 4490
  // and 4494 of FDS_Verification_Guide.tex): dY_c/dt = (Y_e - Y_c) / (L / u)
  // and, for smoke Y_0 outside from t = 0, Y_c = Y_0 (1 - exp(-t u / L)).
  // Its case: 100 mg/kg of soot at 0.25 m/s past a detector of L = 1.8 m.
  assert.equal(DETECTOR_LENGTH, 1.8);
  const Y0 = 100, u = 0.25;
  const exact = (t) => Y0 * (1 - Math.exp((-t * u) / DETECTOR_LENGTH));
  // Whatever the steps: the app's 5 s frames, short ones, uneven ones, one long one.
  for (const steps of [[5, 5, 5, 5, 5, 5], [0.1, 0.1, 0.1, 0.1], [1, 7, 0.25, 30, 2], [100]]) {
    let chamber = 0, t = 0;
    for (const dt of steps) {
      chamber = chamberStep(chamber, Y0, Y0, u, dt);
      t += dt;
      close(chamber, exact(t), 1e-12);
    }
  }
  // FDS's own expected values for the case (Verification/Detectors/smoke_detector.csv in its
  // repository, soot mass fraction at 1, 2 and 3 s, to three figures): 1.30e-5, 2.43e-5, 3.41e-5.
  for (const [t, y] of [[1, 1.3e-5], [2, 2.43e-5], [3, 3.41e-5]]) assert.ok(Math.abs(chamberStep(0, 100e-6, 100e-6, u, t) / y - 1) < 5e-3);
  // One time constant L / u: 63.2 % of the way. Faster gas fills the chamber sooner.
  close(chamberStep(0, Y0, Y0, u, DETECTOR_LENGTH / u), Y0 * (1 - 1 / Math.E), 1e-12);
  assert.ok(chamberStep(0, Y0, Y0, 2, 5) > chamberStep(0, Y0, Y0, 0.25, 5));
  // It empties the same way, never passes the smoke outside, and no time is no change.
  close(chamberStep(Y0, 0, 0, u, 10), Y0 * Math.exp((-10 * u) / DETECTOR_LENGTH), 1e-12);
  assert.ok(chamberStep(0, Y0, Y0, 50, 1e4) <= Y0);
  assert.equal(chamberStep(37, 0, Y0, u, 0), 37);
  // Still gas: the speed's floor, 0.01 m/s, stands in (a time constant of 180 s).
  assert.equal(DETECTOR_MIN_SPEED, 0.01);
  for (const still of [0, 0.001, -1, Number.NaN, undefined]) close(chamberStep(0, Y0, Y0, still, 60), Y0 * (1 - Math.exp(-60 / 180)), 1e-12);
  // Another length, for a detector that is not FDS's default.
  close(chamberStep(0, Y0, Y0, 1, 3, { length: 3 }), Y0 * (1 - Math.exp(-1)), 1e-12);
});

test("between two readings the smoke outside goes in a straight line, and the chamber's equation is solved exactly", () => {
  // Against the equation stepped finely (midpoint rule, 1 ms): outside from 20 to 80 over 5 s at 0.6 m/s, the chamber from 10.
  const u = 0.6, seconds = 5, from = 20, to = 80, tau = DETECTOR_LENGTH / u;
  const outside = (t) => from + ((to - from) * t) / seconds;
  let y = 10;
  const h = 1e-3;
  for (let i = 0; i < Math.round(seconds / h); i += 1) {
    const k1 = (outside(i * h) - y) / tau;
    y += (h * (outside((i + 0.5) * h) - (y + 0.5 * h * k1))) / tau;
  }
  close(chamberStep(10, from, to, u, seconds), y, 1e-6);
  // Two half steps are the whole step.
  const half = chamberStep(10, from, 50, u, 2.5);
  close(chamberStep(half, 50, to, u, 2.5), chamberStep(10, from, to, u, seconds), 1e-12);
});

test("a run's detectors read their chambers: visibility from the chamber's soot, behind the smoke outside", () => {
  // Two detectors, 5 s frames, soot outside in mg/m^3; time constants of 1 s and 10 s.
  const chambers = detectorChambers(2);
  const speed = [1.8, 0.18];
  let before = [0, 0];
  const soot = [0, 0];
  for (const frame of [[0, 0], [100, 100], [100, 100], [100, 0]]) {
    const vis = chambers.step(frame, speed, 5);
    for (let q = 0; q < 2; q += 1) {
      soot[q] = chamberStep(soot[q], before[q], frame[q], speed[q], 5);
      close(chambers.soot[q], soot[q], 1e-12);
      close(vis[q], visibilityOf(soot[q]), 1e-6);
      // Never more smoke inside than there has been outside.
      assert.ok(chambers.soot[q] <= 100 + 1e-9);
    }
    before = frame;
  }
  // Clear air reads clear; the slow detector lags the fast one while the smoke rises.
  const fresh = detectorChambers(2);
  assert.deepEqual(Array.from(fresh.step([0, 0], speed, 5)), [100, 100]);
  const rising = fresh.step([100, 100], speed, 5);
  assert.ok(rising[0] < rising[1], `${rising[0]} m at the fast detector, ${rising[1]} m at the slow one`);
  assert.ok(rising[0] > visibilityOf(100), "and neither is yet at the smoke outside");
  // Into the caller's buffer when given (the run's own row).
  const row = new Float32Array(2);
  assert.equal(detectorChambers(2).step([50, 50], speed, 5, row), row);
});
