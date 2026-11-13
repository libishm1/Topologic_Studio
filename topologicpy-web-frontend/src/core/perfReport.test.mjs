/** Several fire scenarios in one report (perfReport.js), and the ceiling detectors that time detection (flamescapesRun.js, forecast.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { caseFigures, planCases, planDetection, reportHtml, reportStatus, worstStatus } from "./perfReport.js";
import { ceilingDetection } from "./forecast.js";
import { DETECTOR_RADIUS, ceilingDetectors } from "./flamescapesRun.js";

const BASE = { id: "base", label: "Base case", factor: 2, margin: 30 };
const SENSITIVITY = { id: "sensitivity", label: "Sensitivity case", factor: 1.2, margin: 30 };

test("one plan gives its base case, the exit-blocked case and the failed-detector case, each with its own inputs", () => {
  const calls = [];
  const check = (graph, deadlines, opts) => {
    calls.push({ deadlines, ...opts });
    return { status: deadlines === "blocked" ? "fail" : "pass", storeys: [{ worstMargin: deadlines === "blocked" ? -5 : 40, worstRatio: 1.5, share: 0.99, worstWait: 3 }], queues: {} };
  };
  const plan = { deadlines: "map", deadlinesBlocked: "blocked", exitsLeft: [7], detectionSlowest: 95 };
  const cases = planCases(check, {}, plan, { criterion: BASE, sensitivity: SENSITIVITY, fromCeiling: true, exits: [3, 7], detection: 40 });
  assert.deepEqual(cases.map((c) => c.id), ["base", "exit-blocked", "detector"]);
  assert.deepEqual([calls[0].factor, calls[1].factor, calls[2].factor], [2, 1.2, 1.2]);
  assert.deepEqual(calls[1].exits, [7]);
  assert.equal(calls[1].deadlines, "blocked");
  assert.equal(calls[2].detection, 95);
  assert.equal(cases[2].detection, 95);
  assert.equal(reportStatus([{ cases }]), "fail");
  assert.deepEqual(caseFigures(cases[1]), { worstMargin: -5, worstRatio: 1.5, share: 0.99, worstWait: 3, measured: null });
  // No second map and no ceiling detectors: the base case alone.
  assert.equal(planCases(check, {}, { deadlines: "map" }, { criterion: BASE, sensitivity: SENSITIVITY }).length, 1);
  // A detector that never responded leaves the case open, never passed.
  const open = planCases(check, {}, { deadlines: "map", detectionSlowest: null }, { criterion: BASE, sensitivity: SENSITIVITY, fromCeiling: true });
  assert.equal(open[1].result, null);
  assert.equal(reportStatus([{ cases: open }]), "cannot tell");
  assert.equal(worstStatus([]), "cannot tell");
  assert.equal(worstStatus(["pass", "pass"]), "pass");
});

test("detection for the check and the extinguisher: the run's when asked and found, else the time typed, saying which", () => {
  const typed = 60;
  const ceiling = { model: "flamescapes", detection: 42, detectionAt: "ceiling" };
  assert.deepEqual(planDetection(ceiling, { fromRun: true, typed }), { seconds: 42, from: "ceiling" });
  assert.deepEqual(planDetection({ ...ceiling, detectionAt: "head height" }, { fromRun: true, typed }), { seconds: 42, from: "run" });
  assert.deepEqual(planDetection(ceiling, { fromRun: false, typed }), { seconds: 60, from: "set" });
  assert.deepEqual(planDetection({ model: "flamescapes-embers", detection: null }, { fromRun: true, typed }), { seconds: 60, from: "not in the run" });
  assert.deepEqual(planDetection({ model: "zone", detection: null }, { fromRun: true, typed }), { seconds: 60, from: "no run" });
  assert.deepEqual(planDetection(null, { fromRun: true, typed }), { seconds: 60, from: "no plan" });
});

test("the report page names every fire and case, escapes text and gives the verdict over all of them", () => {
  const result = (status) => ({ status, start: 300, cue: 90, storeys: [{ name: "Storey 1", status, text: "every place passes", worstMargin: 12, worstRatio: 2.1, share: 0.995, worstWait: null, occupants: null, measured: null }] });
  const entries = [
    { label: "Fire on storey 1, next to an exit", model: "Fire model: heat diffusion", detectionFrom: "set", cases: [{ id: "base", label: "Base case", criterion: "Base <case>", detection: 60, result: result("pass") }] },
    { label: "Fire on storey 2", model: "Fire model: heat diffusion", detectionFrom: "set", cases: [{ id: "base", label: "Base case", criterion: "Base", detection: 60, result: result("fail") }] },
  ];
  const html = reportHtml(entries, { title: "Report", model: "Model: a & b.ifc", date: "today", inputs: [["Occupants", "Office", "PD 7974-6"]], limits: ["Not validated."] });
  assert.match(html, /^<!doctype html>/);
  assert.match(html, /Fire on storey 1, next to an exit/);
  assert.match(html, /Base &lt;case&gt;/);
  assert.match(html, /a &amp; b\.ifc/);
  assert.match(html, /Verdict over 2 fires: <span class="fail">Fail<\/span>/);
  assert.equal((html.match(/<tr><td>/g) || []).length, 2);
});

test("detection at the ceiling detectors: the first to reach the optical density, and the last", () => {
  // Two detectors, three 10 s frames; 13.4 m of visibility is C/VM2's 0.097 per metre.
  const frames = { K: 3, frameSeconds: 10, detectorCount: 2, detectors: Float32Array.from([100, 100, 13, 100, 13, 12]) };
  assert.deepEqual(ceilingDetection(frames), { first: 20, slowest: 30, count: 2, responded: 2 });
  const late = ceilingDetection({ ...frames, detectors: Float32Array.from([100, 100, 13, 100, 13, 50]) });
  assert.equal(late.first, 20);
  assert.equal(late.slowest, null, "a detector that never responds leaves the slowest open");
  assert.equal(ceilingDetection({ K: 3, frameSeconds: 10 }), null, "no detectors read: a run kept from before");
});

/** A box room on a 0.5 m grid: solid floor (j = 0) and ceiling (j = 7, 3.5 m up), walls round it. */
function room(nx, nz) {
  const ny = 9, h = 0.5;
  const solid = new Uint8Array(nx * ny * nz);
  const at = (i, j, k) => i + nx * (j + ny * k);
  for (let i = 0; i < nx; i += 1) {
    for (let k = 0; k < nz; k += 1) {
      for (let j = 0; j < ny; j += 1) {
        if (j === 0 || j >= 7 || i === 0 || k === 0 || i === nx - 1 || k === nz - 1) solid[at(i, j, k)] = 1;
      }
    }
  }
  const ci = Math.floor(nx / 2), ck = Math.floor(nz / 2);
  return { scene: { grid: { nx, ny, nz, h, origin: [0, 0, 0] }, solid, ignition: { cell: at(ci, 1, ck) } }, at, ny };
}

test("ceiling detectors sit under the ceiling, 7.0 m from the fire in a large room and at the walls in a small one", () => {
  const big = room(41, 41);
  const d = ceilingDetectors(big.scene);
  assert.equal(d.cells.length, 16);
  for (let q = 0; q < d.cells.length; q += 1) {
    const j = Math.floor(d.cells[q] / 41) % big.ny;
    assert.equal(j, 6, "the top open cell, under the ceiling");
    assert.equal(d.distance[q], DETECTOR_RADIUS);
  }
  // A 4 m room: every ray meets a wall first.
  const small = room(9, 9);
  const s = ceilingDetectors(small.scene);
  assert.ok(s.cells.length > 0);
  for (const r of s.distance) assert.ok(r < 2.5, `distance ${r}`);
  assert.equal(ceilingDetectors({ ...small.scene, ignition: null }).cells.length, 0);
});

test("a fire beside a wall puts no detector over the plume, and a door's lintel bounds the fire's ceiling space", () => {
  // The fire one cell from the wall at i = 0: the rays toward it stop at once, the others reach 7 m.
  const big = room(41, 41);
  const nearWall = { ...big.scene, ignition: { cell: big.at(1, 1, 20) } };
  const d = ceilingDetectors(nearWall);
  assert.ok(d.cells.length > 0);
  for (const r of d.distance) assert.equal(r, DETECTOR_RADIUS);
  // A small room round the fire (walls at i = 14 and 26, k = 14 and 26) whose door (k 19 to 21, up to
  // j = 4) opens under a lintel onto the large room: the ray out through the door would reach 7 m, but
  // the lintel ends the fire's ceiling space, so every detector stays in the small room.
  const walled = room(41, 41);
  for (let j = 1; j < 7; j += 1) {
    for (let x = 14; x <= 26; x += 1) {
      walled.scene.solid[walled.at(x, j, 14)] = 1;
      walled.scene.solid[walled.at(x, j, 26)] = 1;
      walled.scene.solid[walled.at(14, j, x)] = 1;
      if (!(x >= 19 && x <= 21 && j <= 4)) walled.scene.solid[walled.at(26, j, x)] = 1;
    }
  }
  const t = ceilingDetectors(walled.scene);
  assert.ok(t.cells.length > 0);
  for (let q = 0; q < t.cells.length; q += 1) {
    const i = t.cells[q] % 41, k = Math.floor(t.cells[q] / (41 * walled.ny));
    assert.ok(i > 14 && i < 26 && k > 14 && k < 26, `a detector outside the fire's room at i = ${i}, k = ${k}`);
    assert.equal(Math.floor(t.cells[q] / 41) % walled.ny, 6);
    assert.ok(t.distance[q] < 4);
  }
});
