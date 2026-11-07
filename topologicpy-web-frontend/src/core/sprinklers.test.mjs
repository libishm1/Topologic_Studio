// Run with: npm test
// Sprinklers, step 1: the rectangle rule (formal/fireegress/FireEgress/Sprinklers.lean),
// rooms from IfcSpace meshes, the room layout and its checks, the storey fallback,
// heads from the IFC, and where the codes require sprinklers.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  HAZARDS,
  SPRINKLER_DEFAULTS,
  flowFor,
  hazardOfRoom,
  layoutRectangle,
  layoutRooms,
  layoutStoreys,
  pressureOf,
  requirementReport,
  roomsFromSpaces,
  sprinklersFromIfc,
} from "./sprinklers.js";

const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
const HEAD_KEYS = [
  "id", "source", "position", "room", "roomName", "hazard", "response", "rti", "c",
  "activationC", "kFactor", "flowLpm", "coverage", "footprint", "provenance",
].sort();

// ------------------------------------------------------------------ geometry helpers

/**
 * A room's volume as a mesh in the viewer frame (y up): its floor polygon,
 * given as plan rectangles [x0, z0, x1, z1] in (x, z), triangulated with
 * downward faces at y0 and upward faces at y1. `flip` winds it inside out.
 */
function roomMesh(itemId, rects, { y0 = 0, y1 = 3, rotate = 0, flip = false } = {}) {
  const c = Math.cos(rotate), s = Math.sin(rotate);
  const positions = [], indices = [];
  const vertex = (x, y, z) => {
    positions.push(x * c - z * s, y, x * s + z * c);
    return positions.length / 3 - 1;
  };
  for (const [x0, z0, x1, z1] of rects) {
    for (const [y, down] of [[y0, true], [y1, false]]) {
      const a = vertex(x0, y, z0), b = vertex(x1, y, z0), d = vertex(x1, y, z1), e = vertex(x0, y, z1);
      const faceDown = down !== flip;
      if (faceDown) indices.push(a, b, d, a, d, e);
      else indices.push(a, d, b, a, e, d);
    }
  }
  return { itemId, positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/** Whether a plan point (a, b) lies in a room's footprint triangles. */
function inFootprint(room, a, b) {
  const T = room.footprint;
  for (let k = 0; k < T.length; k += 6) {
    const [ax, ay, bx, by, cx, cy] = T.subarray(k, k + 6);
    const d1 = (a - bx) * (ay - by) - (ax - bx) * (b - by);
    const d2 = (a - cx) * (by - cy) - (bx - cx) * (b - cy);
    const d3 = (a - ax) * (cy - ay) - (cx - ax) * (b - ay);
    const neg = d1 < -1e-9 || d2 < -1e-9 || d3 < -1e-9;
    const pos = d1 > 1e-9 || d2 > 1e-9 || d3 > 1e-9;
    if (!(neg && pos)) return true;
  }
  return false;
}

// ------------------------------------------------------------------ values

test("hazard classes carry the NFPA 13 values the codes report confirms, each with its clause", () => {
  assert.equal(HAZARDS.light.areaPerHead.value, 20.9);
  assert.equal(HAZARDS.oh1.areaPerHead.value, 12.1);
  assert.equal(HAZARDS.oh2.areaPerHead.value, 12.1);
  for (const H of Object.values(HAZARDS)) {
    assert.equal(H.maxSpacing.value, 4.6);
    assert.equal(H.wallMax.fraction, 0.5);
    assert.equal(H.wallMin.value, 0.102);
    assert.equal(H.minBetween.value, 1.8);
    assert.equal(H.deflector.min, 0.025);
    assert.equal(H.deflector.max, 0.305);
    for (const key of ["areaPerHead", "maxSpacing", "wallMax", "wallMin", "minBetween", "deflector", "density"]) {
      assert.match(H[key].clause, /NFPA 13/, `${H.id}.${key} names its clause`);
    }
  }
  assert.deepEqual([HAZARDS.light.density.value, HAZARDS.oh1.density.value, HAZARDS.oh2.density.value], [4.1, 6.1, 8.1]);
  assert.equal(HAZARDS.light.smallRoom.area, 74);
  assert.equal(HAZARDS.light.smallRoom.wall, 2.7);
  assert.equal(HAZARDS.oh1.smallRoom, null, "the small-room rule is light hazard only");
  assert.equal(SPRINKLER_DEFAULTS.response.quick.rti, 50);
  assert.equal(SPRINKLER_DEFAULTS.response.standard.rti, 100);
  assert.equal(SPRINKLER_DEFAULTS.c.value, 0.4);
  assert.equal(SPRINKLER_DEFAULTS.activationC.value, 68);
  assert.equal(SPRINKLER_DEFAULTS.kFactor.value, 80);
  // Flow: density times area, at least the minimum pressure's; the pressure follows.
  assert.ok(near(flowFor("light", 20.9), 4.1 * 20.9));
  assert.ok(near(flowFor("light", 5), 80 * Math.sqrt(0.5)), "a small head still runs at 0.5 bar");
  assert.ok(near(pressureOf({ flowLpm: 80 * Math.sqrt(1.2), kFactor: 80 }), 1.2));
});

test("hazard by room name: kitchens, plant, garages and storage are ordinary hazard group 1, the rest light", () => {
  for (const name of ["Kitchen", "Küche 1.04", "Plant Room", "Mechanical", "ELEC. CL. 1C11", "Switch room", "Technikraum", "Garage A_1_U1", "Car park", "Parking", "STORAGE 109", "Store room", "Abstellraum"]) {
    assert.equal(hazardOfRoom(name).hazard, "oh1", name);
  }
  for (const name of ["OPEN OFFICE 203", "Corridor", "Classroom 2", "Bedroom", "Toilet", "Meeting", "", null]) {
    assert.equal(hazardOfRoom(name).hazard, "light", String(name));
  }
  assert.match(hazardOfRoom("Storage").reason, /2\.4 m/, "storage states its height caveat");
  assert.match(hazardOfRoom("Kitchen").clause, /NFPA 13/);
});

// ------------------------------------------------------------------ the rectangle rule

/** The rules of Sprinklers.lean, checked on the heads themselves. */
function assertRectangleRules(L, W, hazard, layout) {
  const H = HAZARDS[hazard];
  const S = H.maxSpacing.value, A = H.areaPerHead.value;
  const { nx, ny, dx, dy, heads } = layout;
  assert.equal(nx, Math.ceil(L / S - 1e-9), `${L} × ${W}: nx = ⌈L/S⌉`);
  assert.equal(ny, Math.max(Math.ceil(W / S - 1e-9), Math.ceil((dx * W) / A - 1e-9), 1), `${L} × ${W}: ny`);
  assert.equal(heads.length, nx * ny);
  const xs = [...new Set(heads.map((h) => +h[0].toFixed(9)))].sort((a, b) => a - b);
  const ys = [...new Set(heads.map((h) => +h[1].toFixed(9)))].sort((a, b) => a - b);
  // Every head within S of its neighbours (spacing_le).
  for (let i = 1; i < xs.length; i += 1) assert.ok(xs[i] - xs[i - 1] <= S + 1e-9);
  for (let j = 1; j < ys.length; j += 1) assert.ok(ys[j] - ys[j - 1] <= S + 1e-9);
  // The walls within S/2 (wall_distance_le).
  assert.ok(xs[0] <= S / 2 + 1e-9 && L - xs[xs.length - 1] <= S / 2 + 1e-9);
  assert.ok(ys[0] <= S / 2 + 1e-9 && W - ys[ys.length - 1] <= S / 2 + 1e-9);
  // Area per head at most A (area_le).
  assert.ok(dx * dy <= A + 1e-9, `${L} × ${W}: ${dx * dy} m² per head`);
  // Coverage: every point within half a bay of a head along each axis (covered).
  for (let x = 0; x <= L + 1e-9; x += L / 37) {
    for (let y = 0; y <= W + 1e-9; y += W / 23) {
      assert.ok(heads.some((h) => Math.abs(h[0] - x) <= dx / 2 + 1e-9 && Math.abs(h[1] - y) <= dy / 2 + 1e-9), `(${x}, ${y}) uncovered`);
    }
  }
}

test("the rectangle rule meets spacing, walls, area and coverage on many room sizes", () => {
  const cases = [
    [3, 3, "light"],
    [4.6, 4.6, "light"],
    [9.2, 4.6, "light"],
    [10, 6, "light"],
    [30, 30, "light"],
    [49.93, 14.57, "light"],
    [20, 12.5, "oh1"],
    [7.3, 2.1, "oh2"],
    [0.5, 0.4, "light"],
  ];
  for (const [L, W, hazard] of cases) {
    const layout = layoutRectangle(L, W, hazard);
    assertRectangleRules(L, W, hazard, layout);
    for (const c of layout.checks) assert.notEqual(c.status, "fail", `${L} × ${W} ${hazard}: ${c.id} ${c.text}`);
    assert.match(layout.checks.find((c) => c.id === "area").clause, /SBC 801-CR-2018 903\.3\.1\.1 and IBC 2021 903\.3\.1\.1/);
  }
  // 9.2 m is two spacings exactly: no third bay to rounding.
  assert.equal(layoutRectangle(9.2, 4.6, "light").nx, 2);
  // Light hazard 10 × 6: 3 × 2 heads.
  assert.deepEqual([layoutRectangle(10, 6).nx, layoutRectangle(10, 6).ny], [3, 2]);
  // Extra bays keep every rule.
  const denser = layoutRectangle(10, 6, "light", { extra: [1, 1] });
  assert.deepEqual([denser.nx, denser.ny], [4, 3]);
  for (const c of denser.checks) assert.notEqual(c.status, "fail", c.id);
});

test("the rectangle rule reports what it cannot meet: walls too close in a tiny room, heads too close in a narrow OH room", () => {
  const tiny = layoutRectangle(0.15, 0.15, "light");
  assert.equal(tiny.checks.find((c) => c.id === "wall-min").status, "fail");
  // Ordinary hazard 3.85 × 3.46: the area rule needs two heads, 1.73 m apart across.
  const narrow = layoutRectangle(3.85, 3.46, "oh1");
  assert.equal(narrow.heads.length, 2);
  assert.equal(narrow.checks.find((c) => c.id === "between").status, "fail");
  assert.equal(narrow.status, "fail");
  assert.throws(() => layoutRectangle(0, 3), RangeError);
  assert.throws(() => layoutRectangle(3, 3, "extra"), RangeError);
});

// ------------------------------------------------------------------ rooms from IfcSpace

test("rooms from spaces: footprint, area, floor and ceiling, and the minimum-area rectangle of a turned room", () => {
  const turn = Math.PI / 6;
  const rooms = roomsFromSpaces([roomMesh(7, [[0, 0, 10, 6]], { y0: 1, y1: 4, rotate: turn })], { upAxis: "y", names: new Map([[7, "Office 1"]]) });
  assert.equal(rooms.length, 1);
  const [r] = rooms;
  assert.equal(r.name, "Office 1");
  assert.ok(near(r.area, 60, 1e-4));
  assert.ok(near(r.floor, 1) && near(r.ceiling, 4));
  assert.ok(near(r.rect.half[0], 5, 1e-4) && near(r.rect.half[1], 3, 1e-4), JSON.stringify(r.rect));
  // The long side runs along the room's turned x axis.
  assert.ok(near(Math.abs(r.rect.axis[0] * Math.cos(turn) + r.rect.axis[1] * Math.sin(turn)), 1, 1e-6));
  assert.equal(r.problem, null);
  // Wound inside out, the bottom faces point up: the same footprint, counted once.
  const flipped = roomsFromSpaces([roomMesh(8, [[0, 0, 10, 6]], { flip: true })], { names: { 8: "Flipped" } })[0];
  assert.ok(near(flipped.area, 60, 1e-4));
  assert.equal(flipped.name, "Flipped");
});

test("spaces with no usable footprint are reported, not dropped", () => {
  const wall = { itemId: 3, positions: Float32Array.from([0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 0]), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
  const shaft = roomMesh(4, [[0, 0, 5, 0.3]]);
  const tiny = roomMesh(5, [[0, 0, 0.5, 0.5]]);
  const rooms = roomsFromSpaces([wall, shaft, tiny]);
  assert.equal(rooms.length, 3);
  for (const r of rooms) {
    assert.equal(r.footprint, null);
    assert.ok(r.problem, `room ${r.id} says why`);
  }
  assert.match(rooms[1].problem, /0\.30 m wide/);
  const laid = layoutRooms(rooms);
  assert.equal(laid.heads.length, 0);
  assert.deepEqual(laid.rooms.map((r) => r.status), ["cannot tell", "cannot tell", "cannot tell"]);
});

// ------------------------------------------------------------------ layout per room

function assertPlacementPasses(result, label) {
  for (const room of result.rooms) {
    for (const c of room.checks) {
      if (c.id === "obstructions") assert.equal(c.status, "cannot tell");
      else assert.ok(c.status === "pass" || c.status === "n/a", `${label} ${room.name}: ${c.id} ${c.status}: ${c.text}`);
      assert.ok(c.clause, `${label}: ${c.id} has a clause`);
    }
  }
}

test("a rectangular room gets the rectangle rule's heads, the head record, and a report", () => {
  const rooms = roomsFromSpaces([roomMesh(11, [[0, 0, 10, 6]], { y1: 3 })], { names: new Map([[11, "Office"]]) });
  const result = layoutRooms(rooms);
  assert.equal(result.heads.length, 6);
  const room = result.rooms[0];
  assert.equal(room.hazard, "light");
  assert.equal(room.heads, 6);
  assert.equal(room.status, "cannot tell", "obstructions cannot be checked without a structural or MEP model");
  assert.equal(room.placement, "pass", "every rule the geometry settles passes");
  assertPlacementPasses(result, "rectangle");
  assert.match(room.checks.find((c) => c.id === "obstructions").text, /no structural or MEP model/);
  const h = result.heads[0];
  assert.deepEqual(Object.keys(h).sort(), HEAD_KEYS);
  assert.equal(h.source, "placed");
  assert.equal(h.room, 11);
  assert.equal(h.roomName, "Office");
  assert.equal(h.response, "quick");
  assert.equal(h.rti, 50);
  assert.equal(h.c, 0.4);
  assert.equal(h.activationC, 68);
  assert.equal(h.kFactor, 80);
  assert.ok(near(h.position[1], 3 - 0.13), "link 0.03 m below a deflector 0.10 m below the ceiling");
  assert.ok(near(h.coverage, (10 / 3) * 3, 1e-9));
  assert.ok(near(h.flowLpm, Math.max(4.1 * h.coverage, 80 * Math.sqrt(0.5))));
  assert.ok(near(h.footprint.half[0] * h.footprint.half[1] * 4, h.coverage));
  assert.deepEqual(new Set(Object.values(h.provenance)), new Set(["default"]));
  for (const head of result.heads) assert.ok(inFootprint(rooms[0], head.position[0], head.position[2]));
  assert.ok(result.notes.some((n) => /RTI/.test(n)));
});

test("an L-shaped room built from triangles: heads stay inside, the room is covered, every placement rule holds", () => {
  // 12 × 8 less a 5 × 3 corner, as two rectangles meeting at a T-junction.
  const rooms = roomsFromSpaces([roomMesh(21, [[0, 0, 12, 5], [0, 5, 7, 8]])], { names: { 21: "Office L" } });
  assert.ok(near(rooms[0].area, 12 * 8 - 5 * 3, 1e-4));
  const result = layoutRooms(rooms);
  assertPlacementPasses(result, "L");
  assert.ok(result.heads.length >= 5);
  for (const h of result.heads) assert.ok(inFootprint(rooms[0], h.position[0], h.position[2]), "head inside the room");
  assert.match(result.rooms[0].checks.find((c) => c.id === "coverage").text, /^all \d+ points/);
  // A corridor turning a corner, 2 m wide: no head on the inner corner's walls.
  const corridor = roomsFromSpaces([roomMesh(22, [[0, 0, 20, 2], [0, 2, 2, 20]])], { names: { 22: "Corridor" } });
  const turned = layoutRooms(corridor);
  assertPlacementPasses(turned, "corridor");
  for (const h of turned.heads) assert.ok(inFootprint(corridor[0], h.position[0], h.position[2]));
});

test("a turned room keeps its heads inside and its rules", () => {
  const rooms = roomsFromSpaces([roomMesh(31, [[0, 0, 14, 9]], { rotate: 0.4 })], { names: { 31: "Office" } });
  const result = layoutRooms(rooms);
  assertPlacementPasses(result, "turned");
  for (const h of result.heads) assert.ok(inFootprint(rooms[0], h.position[0], h.position[2]));
});

test("a small ordinary hazard room lays its two heads along the long side, 1.8 m apart", () => {
  const rooms = roomsFromSpaces([roomMesh(41, [[0, 0, 3.85, 3.46]])], { names: { 41: "Storage 109" } });
  const result = layoutRooms(rooms);
  assert.equal(result.rooms[0].hazard, "oh1");
  assert.equal(result.heads.length, 2);
  assertPlacementPasses(result, "storage");
  const [a, b] = result.heads;
  assert.ok(Math.hypot(a.position[0] - b.position[0], a.position[2] - b.position[2]) >= 1.8);
  assert.equal(a.response, "standard");
  assert.ok(near(a.flowLpm, Math.max(6.1 * a.coverage, 80 * Math.sqrt(0.5))));
});

test("the user's hazard class and response override the defaults, and the report follows them", () => {
  const rooms = roomsFromSpaces([roomMesh(51, [[0, 0, 10, 6]])], { names: { 51: "Office" } });
  const result = layoutRooms(rooms, { hazardFor: () => "oh2", response: "standard" });
  assert.equal(result.rooms[0].hazard, "oh2");
  assert.equal(result.rooms[0].hazardSource, "user");
  assert.equal(result.heads[0].provenance.hazard, "user");
  assert.equal(result.heads[0].provenance.response, "user");
  assert.equal(result.heads[0].rti, 100);
  assert.ok(result.heads.every((h) => h.coverage <= 12.1 + 1e-9));
  // Light hazard needs quick response: standard fails.
  const light = layoutRooms(rooms, { response: "standard" });
  assert.equal(light.rooms[0].checks.find((c) => c.id === "response").status, "fail");
  assert.equal(light.rooms[0].status, "fail");
});

test("rooms drawn inside rooms: a gross-area space is not laid out; parking bays in a garage are parts of it", () => {
  // A 20 × 10 space over two 10 × 10 rooms.
  const gross = roomsFromSpaces([roomMesh(61, [[0, 0, 20, 10]]), roomMesh(62, [[0, 0, 10, 10]]), roomMesh(63, [[10, 0, 20, 10]])], {
    names: { 61: "Level 1 GFA", 62: "Office A", 63: "Office B" },
  });
  const a = layoutRooms(gross);
  const byId = new Map(a.rooms.map((r) => [r.id, r]));
  assert.equal(byId.get(61).status, "n/a");
  assert.equal(byId.get(61).heads, 0);
  assert.ok(byId.get(62).heads > 0 && byId.get(63).heads > 0);
  // A 20 × 12 garage holding two 2.5 × 5 bays.
  const garage = roomsFromSpaces([roomMesh(71, [[0, 0, 20, 12]]), roomMesh(72, [[1, 1, 3.5, 6]]), roomMesh(73, [[4, 1, 6.5, 6]])], {
    names: { 71: "Garage", 72: "Stellplatz 1", 73: "Stellplatz 2" },
  });
  const b = layoutRooms(garage);
  const byId2 = new Map(b.rooms.map((r) => [r.id, r]));
  assert.equal(byId2.get(71).hazard, "oh1");
  assert.ok(byId2.get(71).heads > 0);
  assert.equal(byId2.get(72).status, "n/a");
  assert.equal(byId2.get(72).partOf, 71);
  assert.ok(b.heads.every((h) => h.room === 71));
});

// ------------------------------------------------------------------ fallback: no rooms

test("with no rooms, a square lattice per storey covers every walkable node; heights wait for the fire grid", () => {
  for (const angle of [0, 0.3]) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const nodes = [], kinds = [];
    const lift = 0.75;
    for (const level of [0, 3]) {
      for (let i = 0; i <= 40; i += 1) {
        for (let j = 0; j <= 20; j += 1) {
          const u = i * 0.5, v = j * 0.5;
          nodes.push(u * c - v * s, level + lift, u * s + v * c);
          kinds.push(0);
        }
      }
    }
    for (let k = 0; k < 6; k += 1) {
      nodes.push(25, k * 0.5 + lift, 2);
      kinds.push(1); // stair
    }
    const graph = { nodes: Float32Array.from(nodes), kinds: Uint8Array.from(kinds), upAxis: "y", agentHeight: lift, angle };
    const result = layoutStoreys(graph, [0, 3], "light");
    const spacing = Math.sqrt(20.9);
    assert.equal(result.storeys.length, 2);
    for (const h of result.heads) {
      assert.deepEqual(Object.keys(h).sort(), HEAD_KEYS);
      assert.equal(h.provenance.ceiling, "unknown");
      assert.equal(h.provenance.hazard, "user");
      assert.ok(h.position[1] === 0 || h.position[1] === 3, "the storey's floor");
      assert.ok(near(h.coverage, 20.9, 1e-9));
    }
    // Every walkable node lies in a head's square on its storey; stairs get none.
    const n = nodes.length / 3;
    for (let v = 0; v < n; v += 1) {
      const [x, y, z] = [nodes[v * 3], nodes[v * 3 + 1], nodes[v * 3 + 2]];
      const floor = y - lift >= 2.5 ? 3 : 0;
      const covered = result.heads.some((h) => {
        if (h.position[1] !== floor) return false;
        const du = (x - h.position[0]) * c + (z - h.position[2]) * s;
        const dv = -(x - h.position[0]) * s + (z - h.position[2]) * c;
        return Math.abs(du) <= spacing / 2 + 1e-6 && Math.abs(dv) <= spacing / 2 + 1e-6;
      });
      if (kinds[v] === 1) continue;
      assert.ok(covered, `node ${v} uncovered`);
    }
    assert.ok(result.heads.every((h) => Math.hypot(h.position[0] - 25, h.position[2] - 2) > spacing / 2), "no head for the stair alone");
    const checks = result.storeys[0].checks;
    assert.equal(checks.find((x) => x.id === "walls").status, "cannot tell");
    assert.match(checks.find((x) => x.id === "walls").text, /no rooms/);
    assert.equal(checks.find((x) => x.id === "deflector").status, "cannot tell");
    assert.equal(checks.find((x) => x.id === "spacing").status, "pass");
    assert.equal(checks.find((x) => x.id === "area").status, "pass");
    assert.equal(result.storeys[0].placement, "cannot tell", "no rooms, no wall rules");
  }
  const none = layoutStoreys({ nodes: new Float32Array(0) }, []);
  assert.deepEqual([none.heads.length, none.storeys.length], [0, 0]);
  assert.equal(layoutStoreys({ nodes: Float32Array.from([0, 0.75, 0]), agentHeight: 0.75 }, [0]).heads[0].provenance.hazard, "default");
});

// ------------------------------------------------------------------ heads from the IFC

test("heads from the IFC: the file's values where it has them, stated defaults where not, rooms where given", () => {
  const blank = { activationC: null, kFactor: null, response: null, coverage: null, flowLpm: null, pressureBar: null };
  const records = [
    { id: 1, name: "Sprinkler - Pendent - Hosted:1/2\" Pendent", typeName: "1/2\" Pendent", ...blank },
    { id: 2, name: "M_Sprinkler", typeName: "15 mm Pendent", ...blank, activationC: 57, kFactor: 81, flowLpm: 120, pressureBar: 0.483 },
    { id: 3, name: "pressure only", typeName: null, ...blank, pressureBar: 1.0 },
    { id: 4, name: "COBie, no geometry", typeName: null, ...blank },
  ];
  const positions = new Map([[1, [2, 2.5, 2]], [2, [8, 2.5, 3]], [3, [5, 2.6, 5]]]);
  const heads = sprinklersFromIfc(records, positions);
  const { unplaced, notes } = heads;
  assert.ok(Array.isArray(heads));
  assert.equal(heads.length, 3);
  assert.deepEqual(unplaced.map((u) => u.id), [4]);
  const [a, b, c] = heads;
  assert.deepEqual(Object.keys(a).sort(), HEAD_KEYS);
  assert.equal(a.source, "ifc");
  assert.deepEqual(a.position, [2, 2.5, 2]);
  // No file carries RTI or C; the rest of head 1 is defaults too.
  assert.deepEqual([a.rti, a.c, a.activationC, a.kFactor, a.coverage, a.response], [50, 0.4, 68, 80, 20.9, "quick"]);
  assert.ok(near(a.flowLpm, 4.1 * 20.9));
  assert.deepEqual(a.provenance, {
    position: "ifc", hazard: "default", response: "default", rti: "default", c: "default",
    activationC: "default", kFactor: "default", flowLpm: "default", coverage: "default",
  });
  assert.deepEqual([b.activationC, b.kFactor, b.flowLpm], [57, 81, 120]);
  assert.deepEqual([b.provenance.activationC, b.provenance.kFactor, b.provenance.flowLpm, b.provenance.rti], ["ifc", "ifc", "ifc", "default"]);
  assert.ok(near(c.flowLpm, 80 * Math.sqrt(1.0)), "the file's pressure gives the flow, Q = K √P");
  assert.equal(c.provenance.flowLpm, "ifc");
  assert.ok(notes.some((n) => /RTI and C are in no IFC/.test(n)));
  assert.ok(notes.some((n) => /K-factor 1/.test(n)));
  // With rooms, a head takes its room and the room's hazard class.
  const rooms = roomsFromSpaces([roomMesh(90, [[0, 0, 4, 4]], { y1: 2.7 }), roomMesh(91, [[6, 0, 12, 6]], { y1: 2.7 })], { names: { 90: "Kitchen", 91: "Office" } });
  const placed = sprinklersFromIfc(records, positions, { rooms });
  assert.deepEqual(placed.map((h) => h.room), [90, 91, null]);
  assert.deepEqual(placed.map((h) => h.hazard), ["oh1", "light", "light"]);
  assert.equal(placed[0].response, "standard");
  assert.equal(placed[0].coverage, 12.1);
  assert.equal(placed[0].roomName, "Kitchen");
  // The user's response class fills what the file leaves out, and says so; the file's own wins.
  const told = sprinklersFromIfc([{ ...records[0] }, { ...records[1], response: "quick" }], positions, { response: "standard" });
  assert.deepEqual(told.map((h) => [h.response, h.provenance.response, h.rti]), [["standard", "user", 100], ["quick", "ifc", 50]]);
});

// ------------------------------------------------------------------ where sprinklers are required

test("requirement: Group R anywhere needs sprinklers throughout", () => {
  const r = requirementReport({ occupancy: ["B", "R-2"], storeyElevations: [0, 3], occupantsPerStorey: [20, 20] });
  assert.equal(r.status, "required");
  assert.equal(r.triggers.find((t) => t.id === "group-r").status, "required");
  assert.match(r.triggers.find((t) => t.id === "group-r").clause, /SBC 801-CR-2018 903\.2\.8; IBC 2021 903\.2\.8/);
  assert.match(r.standard.clause, /903\.3\.1\.2/);
});

test("requirement: an office storey with 30 or more occupants 16.764 m up (IBC, stricter than SBC's 17 m)", () => {
  const levels = [0, 4.2, 8.4, 12.6, 16.8];
  const at = (top, occ = 50) => requirementReport({ occupancy: "B", storeyElevations: [...levels.slice(0, -1), top], occupantsPerStorey: [100, 50, 50, 50, occ] });
  const tall = at(16.8);
  assert.equal(tall.triggers.find((t) => t.id === "tall").status, "required");
  assert.equal(tall.status, "required");
  assert.match(tall.triggers.find((t) => t.id === "tall").clause, /SBC 801-CR-2018 903\.2\.11\.3.*17 m.*IBC 2021 903\.2\.11\.3.*16\.764 m/);
  assert.equal(at(16.7).triggers.find((t) => t.id === "tall").status, "pass");
  assert.equal(at(16.7).status, "pass");
  assert.equal(at(16.8, 20).triggers.find((t) => t.id === "tall").status, "pass", "fewer than 30 occupants");
  assert.equal(at(18, null).triggers.find((t) => t.id === "tall").status, "cannot tell");
  // Measured from the fire service access when given.
  assert.equal(requirementReport({ occupancy: "B", storeyElevations: [3, 19.5], occupantsPerStorey: [50, 50], accessElevation: 3 }).triggers.find((t) => t.id === "tall").status, "pass");
  assert.equal(requirementReport({ occupancy: "B", storeyElevations: [3, 19.5], occupantsPerStorey: [50, 50], accessElevation: 2 }).triggers.find((t) => t.id === "tall").status, "required");
});

test("requirement: a high-rise has an occupied floor more than 22.86 m up", () => {
  const levels = [0, 4, 8, 12, 16, 20, 24];
  const r = requirementReport({ occupancy: "B", storeyElevations: levels, occupantsPerStorey: levels.map(() => 40) });
  assert.equal(r.triggers.find((t) => t.id === "high-rise").status, "required");
  assert.match(r.triggers.find((t) => t.id === "high-rise").clause, /IBC 2021 403\.3/);
  const roofOnly = requirementReport({ occupancy: "B", storeyElevations: levels, occupantsPerStorey: [...levels.slice(0, -1).map(() => 20), 0] });
  assert.equal(roofOnly.triggers.find((t) => t.id === "high-rise").status, "pass", "an unoccupied roof does not count");
  const unknown = requirementReport({ occupancy: "B", storeyElevations: levels });
  assert.equal(unknown.triggers.find((t) => t.id === "high-rise").status, "cannot tell");
});

test("requirement: A-3 and E fire areas by area, occupants and level", () => {
  const a3 = (o) => requirementReport({ occupancy: ["university", "A-3"], storeyElevations: [0], occupantsPerStorey: [10], ...o }).triggers.find((t) => t.id === "group-a3");
  assert.equal(a3({ area: 1200, occupants: 100, offDischarge: false }).status, "required");
  assert.equal(a3({ area: 800, occupants: 300, offDischarge: false }).status, "required");
  assert.equal(a3({ area: 800, occupants: 100, offDischarge: true }).status, "required");
  assert.equal(a3({ area: 800, occupants: 200, offDischarge: false }).status, "pass");
  assert.equal(a3({ occupants: 200, offDischarge: false }).status, "cannot tell");
  const report = requirementReport({ occupancy: ["university", "A-3"], area: 800, occupants: 200, offDischarge: false, storeyElevations: [0], occupantsPerStorey: [200] });
  assert.ok(report.notes.some((n) => /mosques/.test(n)), "SBC's mosque exception is stated, and IBC's stricter rule kept");
  assert.ok(report.notes.some((n) => /Group B/.test(n)), "a university is Group B");
  const e = requirementReport({ occupancy: "E", area: 900, occupants: 300, offDischarge: false, storeyElevations: [0], occupantsPerStorey: [300] });
  assert.equal(e.triggers.find((t) => t.id === "group-e").status, "required");
  assert.equal(requirementReport({ occupancy: "B", storeyElevations: [0] }).triggers.find((t) => t.id === "group-e").status, "pass");
  for (const t of report.triggers) assert.match(t.clause, /SBC 801-CR-2018.*IBC 2021|IBC 2021.*SBC/, `${t.id} cites both codes`);
  assert.ok(report.notChecked.length > 0);
});

test("requirement: with nothing given, it cannot tell", () => {
  const r = requirementReport({});
  assert.equal(r.status, "cannot tell");
  assert.equal(r.triggers.find((t) => t.id === "group-r").status, "cannot tell");
  assert.equal(r.triggers.find((t) => t.id === "tall").status, "cannot tell");
});
