/** What the view draws of the sprinklers and the systems' rooms (suppressionView.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { sprinklerDrawing, systemsDrawing } from "./suppressionView.js";

const plan = { heads: [{ position: [0, 2.4, 0] }, { position: [3, 2.4, 0] }] };
// A run made with sprinklers, and one of its frames with the second head open.
const run = { sprinklerPoints: [[0, 2.4, 0], [3, 2.4, 0]], sprinklerDrops: [2.4, 2.4] };
const frame = { sprinklers: { open: [1] } };

test("switched off, no heads are drawn, not even a run's own made with them (the user's report, 2026-11-12)", () => {
  assert.equal(sprinklerDrawing({ on: false, plan, run, frame }), null);
  assert.equal(sprinklerDrawing({ on: false, plan }), null);
});

test("switched on, a run's frame draws its heads open as they opened; else the plan's, shut", () => {
  const drawn = sprinklerDrawing({ on: true, plan, run, frame });
  assert.deepEqual(drawn.points, run.sprinklerPoints);
  assert.deepEqual([...drawn.open], [1]);
  assert.deepEqual(drawn.drops, [2.4, 2.4]);
  assert.deepEqual(sprinklerDrawing({ on: true, plan }), { points: [[0, 2.4, 0], [3, 2.4, 0]], open: null, drops: null });
  // A run made without them, on screen with the switch now on: the plan's heads, shut.
  assert.deepEqual(sprinklerDrawing({ on: true, plan, run: { sprinklerPoints: null }, frame }).open, null);
  assert.equal(sprinklerDrawing({ on: true, plan: { heads: [] } }), null);
});

test("the systems' rooms follow their switch the same way", () => {
  const systemsPlan = { special: [{ id: 1, system: "foam" }], units: [{ room: 1 }] };
  const systemsRun = { systemRooms: [{ id: 1, system: "foam" }], systemUnits: [{ room: 1 }] };
  const systemsFrame = { systems: { discharged: [1] } };
  assert.equal(systemsDrawing({ on: false, plan: systemsPlan, run: systemsRun, frame: systemsFrame }), null);
  assert.deepEqual([...systemsDrawing({ on: true, plan: systemsPlan, run: systemsRun, frame: systemsFrame }).discharged], [1]);
  assert.deepEqual(systemsDrawing({ on: true, plan: systemsPlan }), { rooms: systemsPlan.special, discharged: null, units: systemsPlan.units });
  assert.equal(systemsDrawing({ on: true, plan: { special: [] } }), null);
});
