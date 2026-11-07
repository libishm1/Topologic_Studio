// Run with: npm test
// Sprinklers and obstructions: NFPA 13's beam rule, three times rule and wide-obstruction
// rule (wiki/research/sprinkler-codes-2026-11-07.md, section 3.4, rows 33 to 38), heads
// moved clear of obstructions and added beneath wide ones, obstructions from meshes, and
// the room layout unchanged when no obstruction is given.
import assert from "node:assert/strict";
import { test } from "node:test";

import { OBSTRUCTION_RULES, beamRuleRow, checkObstructions, layoutRooms, obstructionFromMesh, roomsFromSpaces } from "./sprinklers.js";

const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
const FT = 0.3048;

// ------------------------------------------------------------------ geometry helpers

/** A room's volume as a mesh in the viewer frame (y up), from plan rectangles [x0, z0, x1, z1]. */
function roomMesh(itemId, rects, { y0 = 0, y1 = 3 } = {}) {
  const positions = [], indices = [];
  const vertex = (x, y, z) => {
    positions.push(x, y, z);
    return positions.length / 3 - 1;
  };
  for (const [x0, z0, x1, z1] of rects) {
    for (const [y, down] of [[y0, true], [y1, false]]) {
      const a = vertex(x0, y, z0), b = vertex(x1, y, z0), d = vertex(x1, y, z1), e = vertex(x0, y, z1);
      if (down) indices.push(a, b, d, a, d, e);
      else indices.push(a, d, b, a, e, d);
    }
  }
  return { itemId, positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/** An obstruction record over the plan rectangle [x0, z0] to [x1, z1]. */
const box = (itemId, kind, [x0, z0, x1, z1], bottom, top, structural = kind === "beam") => ({
  itemId,
  kind,
  structural,
  footprint: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]],
  bottom,
  top,
});

/** A head record at plan point (x, z) with its link point at y (its deflector 0.03 m above). */
const head = (id, x, y, z, hazard = "light") => ({ id, position: [x, y, z], hazard, room: null });

/** The 6 × 4 m office, 3 m high, of the tests below: two heads at x 1.5 and 4.5, z 2. */
const office = () => roomsFromSpaces([roomMesh(11, [[0, 0, 6, 4]])], { names: { 11: "Office" } });
const obstructionCheck = (room) => room.checks.find((c) => c.id === "obstructions");

// ------------------------------------------------------------------ the beam rule

test("the beam rule's table: each row from its A up to, not including, the next; no interpolation; nothing from 7 ft 6 in", () => {
  const { rows, reach } = OBSTRUCTION_RULES.beam;
  assert.deepEqual(
    rows.map((r) => r.max),
    [0, 0.065, 0.09, 0.14, 0.19, 0.24, 0.3, 0.35, 0.415, 0.45, 0.5, 0.6, 0.75, 0.875],
  );
  assert.deepEqual(rows.map((r) => r.confirmed), [...Array(7).fill("yes"), ...Array(7).fill("partly")], "rows from 4 ft on rest on one source");
  assert.ok(near(reach, 2.286));
  rows.forEach((row, k) => {
    // Less than 1 ft, then 6 in rows from 1 ft.
    assert.ok(near(row.from, k ? ((k + 1) / 2) * FT : 0, 1e-12) && near(row.below, ((k + 2) / 2) * FT, 1e-12), `row ${k} in feet`);
    assert.equal(beamRuleRow(row.from), row, `${row.a}: from its start`);
    assert.equal(beamRuleRow((row.from + row.below) / 2), row, `${row.a}: the same B through the row, no interpolation`);
    assert.equal(beamRuleRow(row.below - 1e-9), row, `${row.a}: up to its end`);
    assert.equal(beamRuleRow(row.below), rows[k + 1] ?? null, `${row.a}: not including its end`);
  });
  assert.equal(beamRuleRow(1 * FT).b, "2½ in (65 mm)");
  assert.equal(beamRuleRow(3.5 * FT - 1e-9).b, "9½ in (240 mm)");
  assert.equal(beamRuleRow(10), null);
  assert.equal(beamRuleRow(-1), null);
  assert.match(OBSTRUCTION_RULES.beam.clause, /NFPA 13-2019 Table 10\.2\.7\.1\.2/);
});

test("the beam rule on a head: A to the obstruction's side against B, the deflector's height above its bottom", () => {
  // A beam whose near side runs along x = A; the head at the origin, its deflector at 2.9.
  const beamAt = (A, B, kind = "beam") => box(501, kind, [A, -5, A + 0.3, 5], 2.9 - B, 3.1);
  const verdict = (A, B, kind) => checkObstructions([head("h", 0, 2.87, 0)], [beamAt(A, B, kind)]).heads[0].beam;
  // At 1 ft exactly, 2½ in (65 mm) is allowed; just under 1 ft, nothing.
  assert.equal(verdict(FT, 0.065).status, "pass");
  assert.equal(verdict(FT, 0.066).status, "fail");
  assert.equal(verdict(FT - 1e-4, 0.065).status, "fail");
  assert.equal(verdict(FT - 1e-4, 0.065).maxB, 0);
  // 3 ft 6 in to 4 ft: 12 in (300 mm), at both ends of the row.
  assert.equal(verdict(3.5 * FT, 0.3).status, "pass");
  assert.equal(verdict(4 * FT - 1e-4, 0.3).status, "pass");
  assert.equal(verdict(4 * FT - 1e-4, 0.31).status, "fail");
  // The measured values come back.
  const v = verdict(0.5, 0.2);
  assert.equal(v.status, "fail");
  assert.ok(near(v.A, 0.5) && near(v.B, 0.2) && v.maxB === 0.09 && v.item === 501);
  assert.match(v.text, /1 ft 6 in to less than 2 ft/);
  // From 7 ft 6 in the table sets no limit; above the deflector it does not apply.
  assert.equal(verdict(7.5 * FT + 1e-3, 0.8).status, "pass");
  assert.equal(verdict(0.1, -0.05).status, "pass", "its bottom above the deflector");
  // A pipe takes the three times rule, not the beam rule.
  assert.equal(verdict(0.1, 0.2, "pipe").status, "pass");
  // No model: cannot tell.
  const none = checkObstructions([head("h", 0, 2.87, 0)], null);
  assert.equal(none.heads[0].status, "cannot tell");
  assert.equal(none.status, "cannot tell");
});

// ------------------------------------------------------------------ the three times rule

test("the three times rule: 3 × the largest dimension, never more than 610 mm; structural members only in light and ordinary hazard", () => {
  // A structural member entirely below the deflector (2.9), its top 0.1 m below it.
  const member = (A, size, structural = true) => box(701, "pipe", [A, -5, A + size, 5], 2.8 - size, 2.8, structural);
  const verdict = (A, size, { structural = true, hazard = "light" } = {}) =>
    checkObstructions([head("h", 0, 2.87, 0, hazard)], [member(A, size, structural)]).heads[0].threeTimes;
  // 0.1 m: at least 0.3 m away.
  assert.equal(verdict(0.3, 0.1).status, "pass");
  assert.equal(verdict(0.29, 0.1).status, "fail");
  const v = verdict(0.29, 0.1);
  assert.ok(near(v.distance, 0.29) && near(v.required, 0.3) && near(v.dimension, 0.1) && v.item === 701);
  // 0.4 m: 3 × 0.4 is 1.2 m, but never more than 610 mm is required.
  assert.ok(near(verdict(0.61, 0.4).required, 0.61));
  assert.equal(verdict(0.61, 0.4).status, "pass");
  assert.equal(verdict(0.6, 0.4).status, "fail");
  // Non-structural items: not subject in light and ordinary hazard; subject where the class is none of those.
  assert.equal(verdict(0.1, 0.1, { structural: false }).status, "pass");
  assert.equal(verdict(0.1, 0.1, { structural: false }).exempt, 1);
  assert.equal(verdict(0.1, 0.1, { structural: false, hazard: "oh2" }).status, "pass");
  assert.equal(verdict(0.1, 0.1, { structural: false, hazard: "extra" }).status, "fail");
  assert.match(verdict(0.1, 0.1).clause, /NFPA 13-2022 10\.2\.7\.3\.1\.3.*NFPA 13-2022 10\.2\.7\.3\.1\.4/);
  // More than 18 in (457 mm) below the deflector it is out of the spray's way.
  const low = box(702, "pipe", [0.05, -5, 0.15, 5], 2.3, 2.4, true);
  assert.equal(checkObstructions([head("h", 0, 2.87, 0)], [low]).heads[0].threeTimes.distance, null);
});

// ------------------------------------------------------------------ placement around obstructions

test("a 0.3 m deep beam across a 6 × 4 m room moves the head beside it, and every rule then holds", () => {
  const rooms = office();
  const plain = layoutRooms(rooms);
  assert.deepEqual(plain.heads.map((h) => h.position[0]), [1.5, 4.5]);
  // The beam runs across the room at x 1.9 to 2.1, 0.3 m deep under a 3 m ceiling.
  const beam = box(501, "beam", [1.9, -0.2, 2.1, 4.2], 2.7, 3.0);
  // The head at x 1.5 is 0.4 m from it with its deflector 0.2 m above its bottom: 65 mm allowed.
  const before = checkObstructions(plain.heads, [beam], { rooms });
  assert.equal(before.heads[0].beam.status, "fail");
  assert.ok(near(before.heads[0].beam.A, 0.4) && near(before.heads[0].beam.B, 0.2) && before.heads[0].beam.maxB === 0.065);
  assert.equal(before.heads[1].status, "pass", "x 4.5 is 2.4 m away, beyond 7 ft 6 in");

  const result = layoutRooms(rooms, { obstructions: [beam] });
  const [a, b] = result.heads;
  // 9½ in (240 mm) from 3 ft (0.914 m): the first 0.05 m step that far is x 0.95.
  assert.ok(near(a.position[0], 0.95, 1e-9) && near(a.position[2], 2) && near(a.position[1], 2.87));
  assert.equal(a.provenance.obstruction, "moved");
  assert.deepEqual(b.position, plain.heads[1].position);
  assert.equal(b.provenance.obstruction, undefined);
  // The protection rectangles stay; the head still reaches its own within S/2.
  assert.deepEqual(a.footprint, plain.heads[0].footprint);
  assert.ok(Math.abs(a.position[0] - a.footprint.centre[0]) + a.footprint.half[0] <= 2.3 + 1e-9);
  const room = result.rooms[0];
  const check = obstructionCheck(room);
  assert.equal(check.status, "pass");
  assert.equal(room.status, "pass");
  assert.equal(room.placement, "pass");
  assert.deepEqual(check.moved.map((m) => [m.head, m.items, m.rules]), [["11/1", [501], ["beam rule"]]]);
  assert.ok(near(check.moved[0].by, 0.55, 1e-9));
  assert.deepEqual(check.unresolved, []);
  for (const c of room.checks) assert.equal(c.status, "pass", `${c.id}: ${c.text}`);
  assert.match(room.checks.find((c) => c.id === "between").text, /3\.55 m apart/);
  assert.match(room.checks.find((c) => c.id === "deflector").clause, /NFPA 13-2019 10\.2\.6\.1\.2/, "obstructed construction under the beam");
  assert.match(check.clause, /Table 10\.2\.7\.1\.2.*10\.2\.7\.3\.1\.3.*9\.5\.5\.3/);
  assert.ok(result.notes.some((n) => /1 head moved clear/.test(n)));
  assert.equal(checkObstructions(result.heads, [beam], { rooms }).status, "pass");
  // The same input, the same layout.
  assert.deepEqual(layoutRooms(rooms, { obstructions: [beam] }), result);

  // A beam 1 m deep needs more than 7 ft 6 in, which no point of the head's rectangle has.
  const deep = box(502, "beam", [1.9, -0.2, 2.1, 4.2], 2.0, 3.0);
  const stuck = layoutRooms(rooms, { obstructions: [deep] });
  const left = obstructionCheck(stuck.rooms[0]);
  assert.equal(left.status, "fail");
  assert.equal(stuck.rooms[0].placement, "fail", "with a model the obstructions count in placement");
  assert.deepEqual(stuck.heads.map((h) => h.position[0]), [1.5, 4.5], "not moved");
  assert.equal(left.unresolved.length, 1);
  assert.match(left.unresolved[0], /^head 11\/1: .*beam rule for beam 502/);
  assert.match(left.text, /NFPA 13-2019 10\.2\.6\.1\.2/, "the other options under structural members");
});

test("a 1.5 m wide duct gets a head beneath it, its deflector 0.1 m below the duct", () => {
  const rooms = office();
  // Across the room at x 2.25 to 3.75, its top 0.5 m below the deflectors: out of their spray's way.
  const duct = box(601, "duct", [2.25, -0.5, 3.75, 4.5], 2.0, 2.4);
  const plain = layoutRooms(rooms);
  const before = checkObstructions(plain.heads, [duct], { rooms });
  assert.equal(before.status, "fail");
  assert.deepEqual(before.wide.map((w) => [w.item, w.status, w.beneath]), [[601, "fail", []]]);
  assert.ok(near(before.wide[0].width, 1.5));

  const result = layoutRooms(rooms, { obstructions: [duct] });
  assert.equal(result.heads.length, 3);
  const added = result.heads[2];
  assert.equal(added.id, "11/3");
  assert.equal(added.provenance.obstruction, "beneath");
  assert.ok(near(added.position[0], 3) && near(added.position[2], 2));
  assert.ok(near(added.position[1], 2.0 - 0.1 - 0.03), "link 0.03 m below a deflector 0.1 m below the duct");
  assert.ok(near(added.coverage, 1.5 * 4), "it protects the floor under the duct");
  assert.deepEqual(result.heads.slice(0, 2).map((h) => h.position), plain.heads.map((h) => h.position), "the ceiling's heads stay");
  const check = obstructionCheck(result.rooms[0]);
  assert.equal(check.status, "pass");
  assert.deepEqual(check.added.map((x) => [x.head, x.item]), [["11/3", 601]]);
  assert.equal(result.rooms[0].heads, 3);
  const after = checkObstructions(result.heads, [duct], { rooms });
  assert.deepEqual(after.wide.map((w) => [w.status, w.beneath]), [["pass", ["11/3"]]]);
  assert.match(after.wide[0].clause, /NFPA 13-2019 9\.5\.5\.3/);

  // Close under the ceiling, the duct also moves both heads beside it (B 0.35 m: 4 ft away).
  const high = box(602, "duct", [2.25, -0.5, 3.75, 4.5], 2.55, 2.95);
  const both = layoutRooms(rooms, { obstructions: [high] });
  assert.deepEqual(both.heads.map((h) => [+h.position[0].toFixed(6), +h.position[1].toFixed(6), h.provenance.obstruction]), [
    [1, 2.87, "moved"],
    [5, 2.87, "moved"],
    [3, 2.42, "beneath"],
  ]);
  assert.equal(obstructionCheck(both.rooms[0]).status, "pass");
  // 1.0 m wide: no head beneath; within 24 in (610 mm) of the floor: exempt if noncombustible, not told.
  assert.equal(layoutRooms(rooms, { obstructions: [box(603, "duct", [2.5, -0.5, 3.5, 4.5], 2.0, 2.4)] }).heads.length, 2);
  const lowDuct = layoutRooms(rooms, { obstructions: [box(604, "duct", [2.25, -0.5, 3.75, 4.5], 0.5, 0.9)] });
  assert.equal(lowDuct.heads.length, 2);
  assert.equal(obstructionCheck(lowDuct.rooms[0]).status, "cannot tell");
});

test("obstructions outside the room, above its ceiling or behind its wall are left out", () => {
  const rooms = office();
  const plain = layoutRooms(rooms);
  for (const o of [box(801, "beam", [7, -0.2, 7.2, 4.2], 2.7, 3.0), box(802, "beam", [1.4, -0.2, 1.6, 4.2], 3.2, 3.5)]) {
    const result = layoutRooms(rooms, { obstructions: [o] });
    assert.deepEqual(result.heads, plain.heads, `obstruction ${o.itemId}`);
    const check = obstructionCheck(result.rooms[0]);
    assert.equal(check.status, "pass");
    assert.deepEqual(check.obstructions, []);
    assert.match(check.text, /no beam, duct.*lies over the room/);
  }
  // Two rooms; a deep beam in the second, 1.75 m from the first room's head through the wall.
  const two = roomsFromSpaces([roomMesh(11, [[0, 0, 6, 4]]), roomMesh(12, [[6.2, 0, 12.2, 4]])], { names: { 11: "Office", 12: "Office 2" } });
  const beam = box(803, "beam", [6.25, -0.2, 6.45, 4.2], 2.2, 3.0);
  const laid = layoutRooms(two, { obstructions: [beam] });
  const first = laid.heads.filter((h) => h.room === 11);
  assert.deepEqual(first.map((h) => h.position[0]), [1.5, 4.5], "the first room's heads stay");
  assert.deepEqual(obstructionCheck(laid.rooms[0]).obstructions, []);
  assert.deepEqual(obstructionCheck(laid.rooms[1]).obstructions, [803]);
  // Without rooms, a check cannot see the wall.
  const plainHeads = layoutRooms(two).heads;
  assert.equal(checkObstructions(plainHeads, [beam]).heads[1].beam.status, "fail");
  assert.equal(checkObstructions(plainHeads, [beam], { rooms: two }).heads[1].beam.status, "pass");
});

// ------------------------------------------------------------------ obstructions from meshes

/** A box mesh, x from x0 to x1, y (up) from y0 to y1, z from z0 to z1, with a stray vertex no triangle uses. */
function boxMesh(itemId, [x0, y0, z0], [x1, y1, z1], { up = "y" } = {}) {
  const corners = [];
  for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) corners.push(up === "z" ? [x, z, y] : [x, y, z]);
  const positions = [...corners.flat(), 100, 100, 100];
  // Twelve triangles over the eight corners (index = 4 ix + 2 iy + iz).
  const quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
  const indices = quads.flatMap(([a, b, c, d]) => [a, b, c, a, c, d]);
  return { itemId, positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

const signedArea = (P) => P.reduce((s, [x, y], i) => s + x * P[(i + 1) % P.length][1] - P[(i + 1) % P.length][0] * y, 0) / 2;
const sameCorners = (P, Q, tol = 1e-6) => P.length === Q.length && Q.every(([x, y]) => P.some(([a, b]) => near(a, x, tol) && near(b, y, tol)));

test("obstructionFromMesh: a box mesh gives its plan hull, bottom and top", () => {
  const o = obstructionFromMesh(boxMesh(42, [1, 2.5, 0], [3, 2.8, 0.4]), { kind: "duct" });
  assert.equal(o.itemId, 42);
  assert.equal(o.kind, "duct");
  assert.equal(o.structural, false);
  assert.ok(sameCorners(o.footprint, [[1, 0], [3, 0], [3, 0.4], [1, 0.4]]), JSON.stringify(o.footprint));
  assert.ok(signedArea(o.footprint) > 0, "counter-clockwise");
  assert.ok(near(o.bottom, 2.5, 1e-6) && near(o.top, 2.8, 1e-6), "the stray vertex no triangle uses is left out");
  assert.equal(obstructionFromMesh(boxMesh(43, [0, 2.7, 0], [0.3, 3, 6]), { kind: "beam" }).structural, true);
  assert.equal(obstructionFromMesh(boxMesh(43, [0, 2.7, 0], [0.3, 3, 6]), { kind: "beam", structural: false }).structural, false);
  assert.equal(obstructionFromMesh(boxMesh(44, [0, 0, 0], [1, 1, 1]), { kind: "tank" }).kind, "other");
  assert.equal(obstructionFromMesh(boxMesh(44, [0, 0, 0], [1, 1, 1])).kind, "other");
  // The viewer's meshes carry their kind and whether they are structural.
  const tagged = obstructionFromMesh({ ...boxMesh(49, [0, 2.6, 0], [1.4, 2.9, 5]), kind: "cable", structural: false });
  assert.deepEqual([tagged.kind, tagged.structural], ["cable", false]);
  // z up: the plan is (x, y).
  const zUp = obstructionFromMesh(boxMesh(45, [1, 2.5, 0], [3, 2.8, 0.4], { up: "z" }), { kind: "pipe", upAxis: "z" });
  assert.ok(sameCorners(zUp.footprint, [[1, 0], [3, 0], [3, 0.4], [1, 0.4]]));
  assert.ok(near(zUp.bottom, 2.5, 1e-6) && near(zUp.top, 2.8, 1e-6));
  // Several meshes of one element pool; a mesh with no plan area is no obstruction.
  const pooled = obstructionFromMesh([boxMesh(46, [0, 2, 0], [1, 2.2, 1]), boxMesh(46, [4, 2.1, 0], [5, 2.4, 1])]);
  assert.ok(sameCorners(pooled.footprint, [[0, 0], [5, 0], [5, 1], [0, 1]]));
  assert.ok(near(pooled.bottom, 2, 1e-6) && near(pooled.top, 2.4, 1e-6));
  const wall = { itemId: 47, positions: Float32Array.from([0, 0, 0, 0, 3, 0, 0, 3, 5, 0, 0, 5]), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
  assert.equal(obstructionFromMesh(wall), null);
  assert.equal(obstructionFromMesh(null), null);
  // The record goes straight into the check: 0.2 m wide, 0.3 m deep under the ceiling, 0.4 m from the head at x 1.5.
  const beam = obstructionFromMesh(boxMesh(48, [1.9, 2.7, -0.2], [2.1, 3.0, 4.2]), { kind: "beam" });
  const checked = checkObstructions(layoutRooms(office()).heads, [beam], { rooms: office() });
  assert.equal(checked.heads[0].beam.status, "fail");
  assert.ok(near(checked.heads[0].beam.A, 0.4, 1e-6));
});

// ------------------------------------------------------------------ no obstructions, no change

/** layoutRooms' output for the 6 × 4 m office before obstructions were added (2026-11-07). */
const BEFORE = String.raw`{"heads":[{"id":"11/1","source":"placed","position":[1.5,2.87,2],"room":11,"roomName":"Office","hazard":"light","response":"quick","rti":50,"c":0.4,"activationC":68,"kFactor":80,"flowLpm":56.568542494923804,"coverage":12,"footprint":{"centre":[1.5,2],"axis":[1,0],"half":[1.5,2]},"provenance":{"position":"default","hazard":"default","response":"default","rti":"default","c":"default","activationC":"default","kFactor":"default","flowLpm":"default","coverage":"default"}},{"id":"11/2","source":"placed","position":[4.5,2.87,2],"room":11,"roomName":"Office","hazard":"light","response":"quick","rti":50,"c":0.4,"activationC":68,"kFactor":80,"flowLpm":56.568542494923804,"coverage":12,"footprint":{"centre":[4.5,2],"axis":[1,0],"half":[1.5,2]},"provenance":{"position":"default","hazard":"default","response":"default","rti":"default","c":"default","activationC":"default","kFactor":"default","flowLpm":"default","coverage":"default"}}],"rooms":[{"id":11,"name":"Office","hazard":"light","hazardReason":"light hazard: offices, dwellings, classrooms, corridors, toilets and meeting rooms are; the name names no ordinary hazard use","hazardSource":"name","area":24,"floor":0,"ceiling":3,"heads":2,"grid":{"nx":2,"ny":1,"dx":3,"dy":4,"L":6,"W":4,"axis":[1,0],"parts":0},"checks":[{"id":"spacing","status":"pass","text":"grid of 2 × 1 bays of 3.00 × 4.00 m over the room's 6.00 × 4.00 m rectangle; no point is more than 2.3 m from a head along either axis, so heads are at most 4.6 m apart and at most 2.3 m from a wall","clause":"SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021); NFPA 13-2019 Table 10.2.4.2.1(a): 15 ft (4.6 m); NFPA 13-2019 10.2.5.2: at most half the allowable spacing from a wall, measured perpendicular to it"},{"id":"area","status":"pass","text":"every head protects at most 20.9 m² (2 heads)","clause":"SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021); NFPA 13-2019 Table 10.2.4.2.1(a) (2016: Table 8.6.2.2.1(a)): light hazard, hydraulically calculated, 225 sq ft (20.9 m²) per sprinkler"},{"id":"coverage","status":"pass","text":"all 384 points sampled every 0.25 m lie in a head's protection rectangle","clause":"SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021); NFPA 13-2019 Table 10.2.4.2.1(a): 15 ft (4.6 m); NFPA 13-2019 10.2.5.2: at most half the allowable spacing from a wall, measured perpendicular to it"},{"id":"wall-min","status":"pass","text":"every head at least 0.102 m from the walls","clause":"SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021); NFPA 13-2019 10.2.5.3: at least 4 in (102 mm) from a wall"},{"id":"between","status":"pass","text":"heads 3.00 m apart at the closest; at least 1.8 m","clause":"SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021); NFPA 13-2019 10.2.5.4: at least 6 ft (1.8 m) apart, unless baffled"},{"id":"deflector","status":"pass","text":"deflector 0.1 m below the space's top, taken as a smooth ceiling (a suspended ceiling below it is not in an IfcSpace)","clause":"SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021); NFPA 13-2019 10.2.6.1.1.1 (2016: 8.6.4.1.1.1): deflector 1 to 12 in (25 to 305 mm) below an unobstructed ceiling"},{"id":"response","status":"pass","text":"quick response; light hazard needs quick response","clause":"SBC 801-CR-2018 903.3.2 and IBC 2021 903.3.2: quick-response or residential sprinklers in NFPA 13 light hazard occupancies and in dwelling and sleeping units; NFPA 13-2010 8.3.3.1 (one source, partly confirmed)"},{"id":"obstructions","status":"cannot tell","text":"no structural or MEP model: beams, ducts, lights and other obstructions were not checked","clause":"SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021); NFPA 13-2019 Table 10.2.7.1.2 (beam rule); NFPA 13-2022 10.2.7.3.1.3 (three times rule, at most 24 in (610 mm)); NFPA 13-2019 9.5.5.3 (beneath obstructions wider than 4 ft (1.2 m)); NFPA 13-2019 9.5.6 (18 in (457 mm) clear below the deflector)"}],"status":"cannot tell","placement":"pass"}],"notes":["RTI by response class: quick 50, standard 100 (m s)^1/2 (NFPA 13 definitions and ISO 6182-1: fast response, RTI 50 (m s)^1/2 or less; NFPA 13 definitions: standard response, RTI 80 (m s)^1/2 or more; 100 is FDS's default)","C 0.4 (m/s)^1/2 (conduction factor C, 0.4 (m/s)^1/2 from Bittern's tests (FDS Validation Guide); not in any IFC)","activation 68 °C (NFPA 13 temperature ratings (2019 Table 7.2.4.1, numbering not checked): ordinary 57 to 77 °C for ceilings up to 38 °C; 68 °C is the red bulb)","K 80 L/(min bar^1/2) (NFPA 13: Q = K √P; K5.6 gpm/psi^1/2 is K80.6 L/(min bar^1/2), the common K80 head)","flow: the hazard's design density times the head's protection area, but at least K √(0.5 bar); the pressure then follows, P = (Q/K)² (NFPA 13: no sprinkler below 7 psi (0.48 bar; NFPA's metric figure is 0.5 bar))","deflector 0.1 m below the ceiling, link 0.03 m below the deflector"]}`;

test("without obstructions the room layout is unchanged", () => {
  const rooms = office();
  const result = layoutRooms(rooms);
  assert.deepEqual(JSON.parse(JSON.stringify(result)), JSON.parse(BEFORE));
  assert.deepEqual(layoutRooms(rooms, { obstructions: [] }), result);
  assert.deepEqual(layoutRooms(rooms, { obstructions: null }), result);
  const check = obstructionCheck(result.rooms[0]);
  assert.equal(check.status, "cannot tell");
  assert.equal(result.rooms[0].placement, "pass", "placement leaves out the check no model can make");
});

// ------------------------------------------------------------------ speed

test("200 heads and 1000 obstructions: the check and the layout's share stay well under 100 ms", (t) => {
  // 22 rooms of 10 × 10 m, 9 heads each, and 1000 obstructions of every kind
  // strewn over them (a fixed pseudo-random sequence), far denser than a real model.
  const meshes = [];
  for (let k = 0; k < 22; k += 1) {
    const i = k % 5, j = Math.floor(k / 5);
    meshes.push(roomMesh(k + 1, [[i * 10.2, j * 10.2, i * 10.2 + 10, j * 10.2 + 10]]));
  }
  const rooms = roomsFromSpaces(meshes);
  let seed = 7;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const kinds = OBSTRUCTION_RULES.kinds;
  const obstructions = [];
  for (let n = 0; n < 1000; n += 1) {
    const kind = kinds[n % kinds.length];
    const x = rand() * 51, z = rand() * 51;
    const along = rand() < 0.5;
    const length = kind === "light" ? 0.6 : 1 + rand() * 8;
    const width = { beam: 0.3, duct: 0.3 + rand() * 1.5, pipe: 0.05 + rand() * 0.1, light: 0.6 }[kind] ?? 0.3;
    const depth = kind === "beam" ? 0.2 + rand() * 0.4 : kind === "light" ? 0.1 : 0.1 + rand() * 0.4;
    const top = 3 - rand() * 0.6;
    const [dx, dz] = along ? [length, width] : [width, length];
    obstructions.push(box(1000 + n, kind, [x, z, x + dx, z + dz], top - depth, top));
  }
  const best = (f) => {
    let ms = Infinity, out = null;
    for (let k = 0; k < 3; k += 1) {
      const t0 = performance.now();
      out = f();
      ms = Math.min(ms, performance.now() - t0);
    }
    return [ms, out];
  };
  const [plainMs, plain] = best(() => layoutRooms(rooms));
  const [layoutMs, laid] = best(() => layoutRooms(rooms, { obstructions }));
  const [checkMs, checked] = best(() => checkObstructions(plain.heads, obstructions, { rooms }));
  t.diagnostic(`${plain.heads.length} heads, 1000 obstructions: layout ${plainMs.toFixed(1)} ms without, ${layoutMs.toFixed(1)} ms with; checkObstructions ${checkMs.toFixed(1)} ms`);
  assert.equal(plain.heads.length, 198);
  assert.equal(checked.heads.length, 198);
  assert.ok(laid.heads.length >= plain.heads.length);
  assert.ok(checkMs < 100, `checkObstructions took ${checkMs.toFixed(1)} ms`);
  assert.ok(layoutMs - plainMs < 100, `the obstructions added ${(layoutMs - plainMs).toFixed(1)} ms to the layout`);
});
