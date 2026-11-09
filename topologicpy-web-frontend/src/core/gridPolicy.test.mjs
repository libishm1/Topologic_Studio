// The standard grid every device builds by default, and the range over a
// fire's repeat runs on the GPU (wiki/research/backend-benchmark-2026-11-09.md).
import test from "node:test";
import assert from "node:assert/strict";

import { STANDARD_CELLS, cpuBudget, standardCells } from "../lib/hardware.js";
import { migrateSettings } from "./settingsMigration.js";
import { forecastFromFrames, runRange, runSummary, stackFrames } from "./forecast.js";

test("the standard grid gives a GPU the desktop CPU's budget and a small CPU its own", () => {
  const desktop = { hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (Windows NT 10.0)" };
  const phone = { hardwareConcurrency: 8, deviceMemory: 4, userAgent: "Mozilla/5.0 (Linux; Android 14) Mobile" };
  assert.equal(standardCells({ kind: "gpu", cells: 400000 }), STANDARD_CELLS);
  assert.equal(standardCells({ kind: "gpu", cells: 30000 }), STANDARD_CELLS);
  // The point of the standard grid: a GPU builds what a desktop CPU builds.
  assert.equal(standardCells({ kind: "gpu", cells: 400000 }), standardCells({ kind: "cpu", cells: cpuBudget(desktop) }));
  // A phone's CPU keeps its smaller budget (its card says the grid differs).
  assert.equal(standardCells({ kind: "cpu", cells: cpuBudget(phone) }), 15000);
});

test("settings from before the standard grid: the old default gives way, a chosen finest grid stays", () => {
  assert.deepEqual(migrateSettings({ flamescapesGrid: "speed", horizon: 60 }), { horizon: 60 });
  assert.deepEqual(migrateSettings({ flamescapesGrid: "finest" }), { flamescapesGridPolicy: "finest" });
  // Saved by this version: the policy stands and the old key is dropped.
  assert.deepEqual(migrateSettings({ flamescapesGrid: "finest", flamescapesGridPolicy: "speed" }), { flamescapesGridPolicy: "speed" });
  assert.deepEqual(migrateSettings({ flamescapesGridPolicy: "standard" }), { flamescapesGridPolicy: "standard" });
  assert.deepEqual(migrateSettings(null), {});
});

/** Frames for three nodes as the page keeps them, one object a frame. */
function frameList({ gases = true } = {}) {
  const T = [
    [20, 20, 20],
    [40, 25, 20],
    [65, 30, 20],
    [80, 30, 20],
  ];
  const vis = [
    [100, 100, 100],
    [100, 8, 100],
    [50, 6, 100],
    [20, 5, 100],
  ];
  return T.map((row, k) => ({
    k,
    T: Float32Array.from(row),
    vis: Float32Array.from(vis[k]),
    co: new Float32Array(3),
    ...(gases ? { co2: new Float32Array(3), o2: Float32Array.from([20.9, 20.9, 20.9]) } : {}),
  }));
}

test("frames kept one by one stack as the forecast reads them, CO2 and O2 only when every frame has them", () => {
  const stacked = stackFrames(frameList(), 3);
  assert.equal(stacked.K, 4);
  assert.deepEqual(Array.from(stacked.T.subarray(6, 9)), [65, 30, 20]);
  assert.deepEqual(Array.from(stacked.vis.subarray(3, 6)), [100, 8, 100]);
  assert.ok(stacked.co2 && stacked.o2);
  const mixed = frameList();
  delete mixed[2].co2;
  assert.equal(stackFrames(mixed, 3).co2, undefined);
  assert.equal(stackFrames([], 3).K, 0);
});

test("a repeat run's summary counts untenable nodes by the plan's own rule", () => {
  const frames = { ...stackFrames(frameList(), 3), frameSeconds: 5, hrr: Float32Array.from([100, 400, 300, 200]), stats: { released: 5e6 } };
  const summary = runSummary(frames, { threshold: 60, smoke: true, visibilityLimit: 10 });
  // Node 1 loses visibility in frame 1 (5 s), node 0 reaches 60 C in frame 2 (10 s), node 2 never.
  assert.deepEqual(summary, { releasedMJ: 5, peakKw: 400, untenable: 2, firstUntenable: 5 });
  // The same nodes and times as the forecast the planners use.
  const forecast = forecastFromFrames(frames, { threshold: 60, smoke: true, visibilityLimit: 10 });
  assert.deepEqual(Array.from(forecast.untenable), [10, 5, Infinity]);
  // Without the smoke criterion only the hot node counts.
  assert.deepEqual(runSummary(frames, { threshold: 60, smoke: false }), { releasedMJ: 5, peakKw: 400, untenable: 1, firstUntenable: 10 });
  // Nothing reached: no first time.
  assert.equal(runSummary(frames, { threshold: 500, smoke: false }).firstUntenable, null);
});

test("the range over repeat runs: median, least and most, and runs where no node became untenable", () => {
  const run = (untenable, firstUntenable, peakKw, releasedMJ) => ({ untenable, firstUntenable, peakKw, releasedMJ });
  const range = runRange([run(380, 20, 812, 40.7), run(372, 25, 786, 40.6), run(0, null, 369, 22.7), run(358, 30, 790, 39.9)]);
  assert.equal(range.runs, 4);
  assert.deepEqual(range.untenable, { median: 365, min: 0, max: 380 });
  // Three runs had a first untenable time; the fourth had none.
  assert.deepEqual(range.firstUntenable, { median: 25, min: 20, max: 30 });
  assert.equal(range.noneUntenable, 1);
  assert.deepEqual(range.peakKw, { median: 788, min: 369, max: 812 });
  // One run: its own values.
  assert.deepEqual(runRange([run(12, 35, 373, 26.6)]).untenable, { median: 12, min: 12, max: 12 });
  assert.equal(runRange([run(0, null, 10, 1)]).firstUntenable, null);
});
