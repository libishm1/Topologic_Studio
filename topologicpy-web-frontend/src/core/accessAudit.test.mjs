// Run with: node --test src/core/accessAudit.test.mjs
// The wheelchair audit on a corridor with three rooms: one a wheelchair
// reaches, one behind a door too narrow, one up a 15 cm step. Known answers:
// the issues, the act step's two changes, and the accessible ways out.
import test from "node:test";
import assert from "node:assert/strict";
import { storeyRaster } from "./floorRaster.js";
import { PROFILES, auditStorey, stairWidths } from "./accessAudit.js";

function box(x0, y0, z0, x1, y1, z1) {
  const positions = Float32Array.from([x0, y0, z0, x1, y0, z0, x1, y0, z1, x0, y0, z1, x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1]);
  const indices = Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]);
  return { positions, indices };
}

const rect = (name, x0, z0, x1, z1, floor = 0) => ({
  name,
  floor,
  area: (x1 - x0) * (z1 - z0),
  footprint: Float64Array.from([x0, z0, x1, z0, x1, z1, x0, z0, x1, z1, x0, z1]),
});

/** A corridor (z 0..2) and rooms A, B, C above it, doors 0.9, 0.7 and 0.9 m; C's floor 15 cm up. */
function floor() {
  const walls = [
    box(-0.2, 0, -0.2, 12.2, 3, 0),
    box(-0.2, 0, 8, 12.2, 3, 8.2),
    box(-0.2, 0, 0, 0, 3, 8),
    box(12, 0, 0, 12.2, 3, 8),
    // The corridor's north wall, with the three doorways.
    box(0, 0, 2, 1.55, 3, 2.2),
    box(2.45, 0, 2, 5.65, 3, 2.2),
    box(6.35, 0, 2, 9.55, 3, 2.2),
    box(10.45, 0, 2, 12, 3, 2.2),
    // Partitions between the rooms.
    box(4, 0, 2.2, 4.2, 3, 8),
    box(8, 0, 2.2, 8.2, 3, 8),
  ];
  const floors = [box(-0.5, -0.2, -0.5, 12.5, 0, 2.2), box(-0.5, -0.2, 2.2, 8.2, 0, 8.5), box(8.2, -0.2, 2.2, 12.5, 0.15, 8.5)];
  const rooms = [rect("Corridor", 0, 0, 12, 2), rect("Room A", 0, 2.2, 4, 8), rect("Room B", 4.2, 2.2, 8, 8), rect("Room C", 8.2, 2.2, 12, 8, 0.15)];
  const R = storeyRaster({ floors, walls }, { level: 0, seeds: [[0.5, 0.5], [11.5, 1.5], [2, 5], [6, 5], [10, 5], [0.5, 7.5], [11.5, 7.5]], rooms, cell: 0.05 });
  const doors = [
    { x: 2, y: 2.1, width: 0.9, label: "A" },
    { x: 6, y: 2.1, width: 0.7, label: "B" },
    { x: 10, y: 2.1, width: 0.9, label: "C" },
  ];
  return { R, rooms, doors };
}

test("a wheelchair reaches the corridor and room A; B's door is too narrow; C is up a step", () => {
  const { R, rooms, doors } = floor();
  const out = auditStorey(R, {
    profile: "icc",
    rooms,
    doors,
    entrances: [{ x: 0.3, y: 1, label: "Exit" }],
    ways: [{ x: 0.3, y: 1, kind: "exit", label: "Exit" }, { x: 11.7, y: 1, kind: "stair", width: 1.3, label: "Stair" }],
  });
  assert.equal(out.rooms.onFoot, 4);
  assert.equal(out.rooms.byChair, 2);
  const kinds = Object.fromEntries(out.issues.filter((i) => i.severity === "high").map((i) => [i.name, i.kind]));
  assert.equal(kinds["Room B"], "Door too narrow");
  assert.equal(kinds["Room C"], "Step or stairs only");
  // The act step: two changes, each letting the wheelchair into one more room.
  assert.equal(out.act.length, 2, JSON.stringify(out.act));
  assert.deepEqual(out.act.map((a) => a.gain), [1, 1]);
  assert.deepEqual(new Set(out.act.map((a) => a.kind)), new Set(["Widen door", "Ramp the step"]));
  // Accessible means of egress: from the corridor and room A, the exit and the stair (1.30 m wide) as a refuge.
  for (const e of out.egress) {
    assert.equal(e.count, 2, JSON.stringify(e));
    assert.ok(e.ok);
  }
});

test("the metric profile's 0.80 m door still stops at a 0.70 m doorway, and a 1.0 m stair is no refuge", () => {
  const { R, rooms, doors } = floor();
  const out = auditStorey(R, {
    profile: PROFILES.metric,
    rooms,
    doors,
    entrances: [{ x: 0.3, y: 1 }],
    ways: [{ x: 0.3, y: 1, kind: "exit" }, { x: 11.7, y: 1, kind: "stair", width: 1.0 }],
  });
  assert.ok(out.issues.some((i) => i.name === "Room B" && i.kind === "Door too narrow"));
  for (const e of out.egress) {
    assert.equal(e.refuges, 0);
    assert.equal(e.narrowStairs, 1);
    assert.ok(!e.ok, "one accessible way out where two are needed");
  }
});

test("a stair's clear width is measured between its walls", () => {
  const stairs = [{ ...box(2, 0.1, 0, 3.2, 1.5, 4) }];
  const walls = [box(1.8, 0, 0, 2, 3, 4), box(3.2, 0, 0, 3.4, 3, 4)];
  const R = storeyRaster({ floors: [box(-1, -0.2, -1, 6, 0, 6)], walls, stairs }, { level: 0, seeds: [[0, 0], [5, 5]], cell: 0.05 });
  const list = stairWidths(R);
  assert.equal(list.length, 1);
  assert.ok(Math.abs(list[0].width - 1.2) <= 0.06, `${list[0].width}`);
});
