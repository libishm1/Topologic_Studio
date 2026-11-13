/** What a Flamescapes run colours the graph by (measures.js): the heat ramp, fixed or fitted to the run. */
import test from "node:test";
import assert from "node:assert/strict";
import { fittedHeatPeak, measureView } from "./measures.js";
import { migrateSettings } from "./settingsMigration.js";

test("the heat ramp is the fixed one unless fitting to the run is chosen", () => {
  assert.deepEqual(measureView({}, false), { kind: "heat", key: "heat", legend: null, fit: false });
  assert.deepEqual(measureView({ flamescapesShow: "heat", flamescapesHeatScale: "fixed" }, false), { kind: "heat", key: "heat", legend: null, fit: false });
  const fitted = measureView({ flamescapesShow: "heat", flamescapesHeatScale: "run" }, false);
  assert.equal(fitted.fit, true);
  // A new key, so the frame on screen is drawn again when the choice changes.
  assert.equal(fitted.key, "heat:fit");
  // The other views are as before.
  assert.equal(measureView({ flamescapesShow: "smoke", flamescapesHeatScale: "run" }, false).kind, "smoke");
});

test("a ramp fitted to the run tops at its hottest gas, rounded up to 10 °C, at least 10 °C above ambient", () => {
  // A 42 kW fire that warms head height to 34.2 °C shades the graph to 40 °C, not to 600.
  assert.equal(fittedHeatPeak(34.2), 40);
  assert.equal(fittedHeatPeak(310), 310);
  assert.equal(fittedHeatPeak(311), 320);
  // No heat yet, or none above ambient: a 20 to 30 °C ramp, never an empty one.
  assert.equal(fittedHeatPeak(null), 30);
  assert.equal(fittedHeatPeak(21), 30);
  assert.equal(fittedHeatPeak(Number.NaN), 30);
});

test("a stored heat ramp this version does not draw gives way to the fixed one", () => {
  assert.deepEqual(migrateSettings({ flamescapesHeatScale: "run" }), { flamescapesHeatScale: "run" });
  assert.deepEqual(migrateSettings({ flamescapesHeatScale: "log" }), {});
});
