/** A Flamescapes run until the fire service arrives (fireService.js). */
import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_SERVICE_RESPONSE,
  NFPA_1710_RESPONSE,
  SERVICE_HORIZON_MAX,
  SOLVER_STEP_S,
  beyondTested,
  computeSeconds,
  durationText,
  gpuPace,
  runHorizon,
  serviceArrival,
} from "./fireService.js";
import { DEFAULTS } from "./flamescapes.js";
import { migrateSettings } from "./settingsMigration.js";

test("the default response is NFPA 1710 (2020)'s first engine: 15 + 64 + 80 + 240 s", () => {
  assert.deepEqual(
    NFPA_1710_RESPONSE.map((p) => [p.id, p.seconds, p.clause]),
    [
      ["answering", 15, "4.1.2.3.1"],
      ["processing", 64, "4.1.2.3.3"],
      ["turnout", 80, "4.1.2.1(2)"],
      ["travel", 240, "4.1.2.1(3)"],
    ],
  );
  assert.equal(DEFAULT_SERVICE_RESPONSE, 399);
  assert.equal(DEFAULT_SERVICE_RESPONSE / 60, 6.65);
});

test("the fire service arrives at detection plus the response; the run lasts until then", () => {
  // The performance check's default detection, 60 s.
  assert.deepEqual(serviceArrival(60), { detection: 60, response: 399, arrival: 459, horizon: 459, capped: false, beyondTested: false });
  assert.equal(serviceArrival(60, 480).horizon, 540);
  // Rounded to the second.
  assert.equal(serviceArrival(10.4, 400.3).horizon, 411);
  // Past the held-out tests' 900 s: said so.
  const late = serviceArrival(600, 400);
  assert.equal(late.horizon, 1000);
  assert.equal(late.beyondTested, true);
  assert.equal(beyondTested(900), false);
  assert.equal(beyondTested(901), true);
});

test("the run is kept within 30 s and an hour, and says when it ends before the arrival", () => {
  const long = serviceArrival(1800, 3600);
  assert.equal(long.arrival, 5400);
  assert.equal(long.horizon, SERVICE_HORIZON_MAX);
  assert.equal(long.capped, true);
  assert.equal(serviceArrival(0, 10).horizon, 30);
  // Missing or negative values: no detection delay, the default response.
  assert.equal(serviceArrival(undefined).horizon, 399);
  assert.equal(serviceArrival(-5, null).horizon, 399);
  assert.equal(serviceArrival(60, Number.NaN).horizon, 459);
});

test("off, the run lasts the time set, whatever the detection and response", () => {
  const settings = { flamescapesHorizon: 180, flamescapesUntilService: false, perfDetection: 60, flamescapesServiceResponse: 399 };
  assert.equal(runHorizon(settings), 180);
  assert.equal(runHorizon({ ...settings, perfDetection: 900 }), 180);
  // Settings saved before the switch existed.
  assert.equal(runHorizon({ flamescapesHorizon: 240 }), 240);
  assert.equal(runHorizon({ ...settings, flamescapesUntilService: true }), 459);
  assert.equal(runHorizon({ ...settings, flamescapesUntilService: true, perfDetection: 120, flamescapesServiceResponse: 600 }), 720);
});

test("compute time at a measured pace: the grid's setup once, the stepping in proportion", () => {
  // 180 s simulated in 65 s, 5 s of it building the grid: 459 s takes 5 + 2.55 x 60 s.
  assert.equal(computeSeconds(459, { simulated: 180, wallMs: 65000, setupMs: 5000 }), 158);
  assert.equal(computeSeconds(180, { simulated: 180, wallMs: 65000, setupMs: 5000 }), 65);
  assert.equal(computeSeconds(360, { simulated: 180, wallMs: 60000 }), 120);
  assert.equal(computeSeconds(360, null), null);
  assert.equal(computeSeconds(0, { simulated: 180, wallMs: 60000 }), null);
  // A GPU's kept speed, cells a millisecond a solver step: 30,000 cells at 3,000 a ms is 10 ms a 0.1 s step.
  assert.equal(computeSeconds(600, gpuPace(3000, 30000)), 60);
  assert.equal(gpuPace(0, 30000), null);
});

test("the solver step the estimate counts is the solver's own", () => {
  assert.equal(SOLVER_STEP_S, DEFAULTS.dt);
});

test("lengths of time read in seconds, minutes or hours", () => {
  assert.equal(durationText(45), "45 s");
  assert.equal(durationText(459), "7 min 39 s");
  assert.equal(durationText(420), "7 min");
  assert.equal(durationText(3600), "1 h");
  assert.equal(durationText(3900), "1 h 5 min");
  assert.equal(durationText(Number.NaN), "-");
});

test("stored settings: only a switch and a response time this version takes", () => {
  assert.deepEqual(migrateSettings({ flamescapesUntilService: true, flamescapesServiceResponse: 480 }), { flamescapesUntilService: true, flamescapesServiceResponse: 480 });
  assert.deepEqual(migrateSettings({ flamescapesUntilService: "yes", flamescapesServiceResponse: "399" }), {});
  assert.deepEqual(migrateSettings({ flamescapesServiceResponse: 10 }), {});
  assert.deepEqual(migrateSettings({ flamescapesServiceResponse: 99999 }), {});
  assert.deepEqual(migrateSettings({ flamescapesServiceResponse: null }), {});
});
