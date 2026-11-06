/**
 * Smoke, gas and tenability measures (core/tenability.js):
 *   node --test src/core/tenability.test.mjs
 */
import test from "node:test";
import assert from "node:assert/strict";

import {
  CO_DOSE,
  FED_CRIT,
  asphyxiantRate,
  extinction,
  heatRate,
  hyperventilation,
  limitingCriterion,
  obscuration,
  opticalDensity,
  tenabilityTracker,
  visibilityOf,
} from "./tenability.js";
import { visibility } from "./flamescapesRun.js";

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
