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

test("people leaving is never remembered between visits: not stored, and dropped if an earlier version stored it", async () => {
  const { SESSION_ONLY, withoutSessionOnly } = await import("./settingsMigration.js");
  assert.ok(SESSION_ONLY.includes("crowd"));
  assert.deepEqual(withoutSessionOnly({ crowd: true, crowdStyle: "dots", walkSpeed: 1.2 }), { crowdStyle: "dots", walkSpeed: 1.2 });
  assert.deepEqual(withoutSessionOnly(null), {});
});

test("the calibrated grid builds cells of exactly the validated size, clipped round the fire to the budget", async () => {
  const { buildScene } = await import("./flamescapesRun.js");
  const { CALIBRATED_CELL } = await import("./calibrationStatus.js");
  const { MATERIAL } = await import("./voxelize.js");
  const box = (x0, y0, z0, x1, y1, z1) => ({
    positions: Float32Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3]),
    material: MATERIAL.STONE,
  });
  // A 40 x 30 m storey: too large for 30,000 cells of 0.27 m, so the grid is clipped round the fire.
  const geometry = { walls: [box(-0.2, 0, -0.2, 40.2, 3, 0)], floors: [box(-0.2, -0.2, -0.2, 40.2, 0, 30.2)], roofs: [box(-0.2, 3, -0.2, 40.2, 3.2, 30.2)], stairs: [], obstacles: [], doors: [], windows: [] };
  const { fuelsFor } = await import("./fireClasses.js");
  const { contentsFuel, ignitionFuel } = fuelsFor("A");
  const scene = buildScene(geometry, { fire: [20, 0.75, 15], maxCells: 30000, minCell: CALIBRATED_CELL, maxCell: CALIBRATED_CELL, contentsFuel, ignitionFuel });
  assert.ok(Math.abs(scene.grid.h - CALIBRATED_CELL) < 1e-9, `cell ${scene.grid.h}`);
  assert.ok(scene.grid.cells <= 30000);
  assert.equal(scene.clipped, true);
});

test("a laptop's integrated graphics is told apart, so the card can say how to use its faster card", async () => {
  const { integratedGpu } = await import("../lib/hardware.js");
  assert.equal(integratedGpu("intel gen-12lp"), true);
  assert.equal(integratedGpu("nvidia turing"), false);
  assert.equal(integratedGpu("amd rdna-2"), false);
  // Intel's discrete Arc cards are not integrated.
  assert.equal(integratedGpu("intel gen-12hp"), false);
  assert.equal(integratedGpu("intel xe-2hpg"), false);
  assert.equal(integratedGpu("intel xe-lpg"), true);
  assert.equal(integratedGpu(""), false);
});

test("the GPU grid is sized by the solver's largest binding, cf, at its own bytes a cell", async () => {
  const { GPU_BINDING_BYTES_PER_CELL, detectGpu } = await import("../lib/hardware.js");
  const { CF_FIELDS } = await import("./flamescapesGpu/kernels.js");
  assert.equal(GPU_BINDING_BYTES_PER_CELL, CF_FIELDS * 4);
  // A device that binds 16 MiB (and makes 256 MiB buffers) holds 16 MiB / 80 B cells; a large one is capped at 400,000.
  const gpuWith = (limits) => ({ gpu: { requestAdapter: async () => ({ info: { vendor: "test" }, limits }) } });
  assert.equal((await detectGpu(gpuWith({ maxStorageBufferBindingSize: 16 * 2 ** 20, maxBufferSize: 256 * 2 ** 20 }))).cells, Math.floor((16 * 2 ** 20) / (CF_FIELDS * 4)));
  assert.equal((await detectGpu(gpuWith({ maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 }))).cells, 400000);
  // The smaller of the two limits counts.
  assert.equal((await detectGpu(gpuWith({ maxStorageBufferBindingSize: 2 ** 30, maxBufferSize: 8 * 2 ** 20 }))).cells, Math.max(30000, Math.floor((8 * 2 ** 20) / (CF_FIELDS * 4))));
});
