// Run with: node --test src/core
import assert from "node:assert/strict";
import { test } from "node:test";

import { CORE_ESCALATOR, CORE_LIFT, CORE_STAIR, coreItems, coreKind, coreStacks, levelsOf, walkingLine } from "./cores.js";
import { KIND_LIFT, KIND_STAIR, buildLatticeGraph } from "./lattice.js";

/** A horizontal rectangle at height z, as two triangles (z up). */
function rect(x0, y0, x1, y1, z, itemId) {
  const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].flatMap(([x, y]) => [x, y, z]);
  return { positions: Float32Array.from(pts), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]), itemId };
}

/** A closed box. */
function box(x0, y0, z0, x1, y1, z1, itemId) {
  const corners = [];
  for (const z of [z0, z1]) for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) corners.push(x, y, z);
  const f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]];
  return { positions: Float32Array.from(corners), indices: Uint32Array.from(f.flat()), itemId };
}

/** Union-find over a graph's edges. */
function components(g) {
  const parent = Int32Array.from({ length: g.stats.nodes }, (_, i) => i);
  const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  for (let e = 0; e < g.edges.length; e += 2) parent[find(g.edges[e])] = find(g.edges[e + 1]);
  return find;
}

/** The node nearest (x, y, z). */
function nearest(g, x, y, z) {
  let best = -1, d = Infinity;
  for (let n = 0; n < g.stats.nodes; n += 1) {
    const e = Math.hypot(g.points[n * 3] - x, g.points[n * 3 + 1] - y, g.points[n * 3 + 2] - z);
    if (e < d) {
      d = e;
      best = n;
    }
  }
  return best;
}

const base = { upAxis: "z", spacing: 0.5, agentHeight: 0 };
// A wall rising past the upper floor, so it is not taken for a roof.
const tallWall = box(-0.2, 0, 0, 0, 10, 5.8);

test("cores are named in the corpus's languages, and lobbies are not cores", () => {
  const kind = (type, name, predefined) => coreKind({ type, name, predefined });
  assert.equal(kind("IFCSPACE", "1-17 Treppenhaus Ost - EG"), CORE_STAIR);
  assert.equal(kind("IFCSPACE", "1CS3 STAIR"), CORE_STAIR);
  assert.equal(kind("IFCSPACE", "300 Aufzug West"), CORE_LIFT);
  assert.equal(kind("IFCSPACE", "E1 ELEVATOR"), CORE_LIFT);
  assert.equal(kind("IFCBUILDINGELEMENTPROXY", "Aufzug-Kabine-Hydraulik:1600 x 2400"), CORE_LIFT);
  assert.equal(kind("IFCBUILDINGELEMENTPROXY", "M_Electric Lift-4 person:1000 x 880mm"), CORE_LIFT);
  assert.equal(kind("IFCSPACE", "Aufzugsvorraum"), 0);
  assert.equal(kind("IFCSPACE", "Lift lobby"), 0);
  assert.equal(kind("IFCSPACE", "Elevator machine room"), 0);
  assert.equal(kind("IFCSPACE", "301 Schacht West 1"), 0, "a shaft may carry ducts, not people");
  assert.equal(kind("IFCRAILING", "Stair railing"), 0, "part of a stair, not one");
  assert.equal(kind("IFCSTAIR", "ST-01"), CORE_STAIR);
  assert.equal(kind("IFCTRANSPORTELEMENT", "ELEV-01", ""), CORE_LIFT);
  assert.equal(kind("IFCTRANSPORTELEMENT", "E-1", "ESCALATOR"), CORE_ESCALATOR);
  assert.equal(kind("IFCTRANSPORTELEMENT", "Crane", "CRANEWAY"), 0);
  const items = coreItems([{ itemId: 5, type: "IFCSPACE", name: "A105 Stair" }, { itemId: 6, type: "IFCSPACE", name: "Kitchen" }]);
  assert.deepEqual(items.map((i) => [i.itemId, i.kind, i.source]), [[5, CORE_STAIR, "IfcSpace name"]]);
});

test("the assumed walking line is as long as a 175/280 stair and stays in its footprint", () => {
  const core = { u0: 6, u1: 10, v0: 0, v1: 3 };
  for (const [ha, hb] of [[0, 3], [0, 4.2], [3, 6.5]]) {
    const line = walkingLine(core, ha, hb, { step: 0.5 });
    assert.equal(line[0][2], ha);
    assert.ok(Math.abs(line[line.length - 1][2] - hb) < 1e-9);
    let plan = 0, length = 0;
    for (let i = 1; i < line.length; i += 1) {
      const [a, b] = [line[i - 1], line[i]];
      assert.ok(b[2] >= a[2] - 1e-12, "never goes down");
      const piece = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      assert.ok(piece <= 0.5 + 1e-9 || b[2] === a[2], `a flight piece of ${piece} m is longer than the step`);
      if (b[2] > a[2]) plan += Math.hypot(b[0] - a[0], b[1] - a[1]);
      length += piece;
    }
    for (const [u, v] of line) assert.ok(u >= core.u0 && u <= core.u1 && v >= core.v0 && v <= core.v1, `(${u}, ${v}) outside the core`);
    const rise = hb - ha;
    assert.ok(Math.abs(plan - (rise / 0.175) * 0.28) < 1e-9, `flights' plan length ${plan}`);
    assert.ok(length >= (rise * Math.hypot(0.28, 0.175)) / 0.175 - 1e-9, `walking length ${length} for a rise of ${rise}`);
  }
});

test("stacked rooms and lift cars of one kind merge into one core; side by side they do not", () => {
  const frame = { up: 2, h0: 0, h1: 1, cos: 1, sin: 0 };
  const meshes = [box(6, 0, 0, 10, 3, 2.8, 1), box(6, 0, 3, 10, 3, 5.8, 2), box(0, 0, 0, 2, 2, 2.4, 3), box(2.5, 0, 0, 4.5, 2, 2.4, 4)];
  const items = [
    { itemId: 1, kind: CORE_STAIR, name: "Stair 1 - EG" },
    { itemId: 2, kind: CORE_STAIR, name: "Stair 1 - OG" },
    { itemId: 3, kind: CORE_LIFT, name: "Lift A" },
    { itemId: 4, kind: CORE_LIFT, name: "Lift B" },
  ];
  const stacks = coreStacks(items, meshes, frame);
  assert.equal(stacks.length, 3);
  const stair = stacks.find((s) => s.kind === CORE_STAIR);
  assert.ok(stair.lo === 0 && Math.abs(stair.hi - 5.8) < 1e-6, `height range ${stair.lo}..${stair.hi}`);
  assert.deepEqual(stair.names, ["Stair 1 - EG", "Stair 1 - OG"]);
  assert.deepEqual(levelsOf([0, 0.1, 0, 3, 3.02, 3, 6.1]), [0, 3, 6.1], "each level at its median height");
});

test("a stair core with no treads joins its floors along the assumed line", () => {
  // Ground floor everywhere; upper floor at 3 m everywhere but the core, whose
  // upper room's floor covers it. The core (x 6..10, y 0..3) is two rooms
  // named as a stair, with no stair geometry at all.
  const floors = [rect(0, 0, 10, 10, 0), rect(0, 3, 10, 10, 3), rect(0, 0, 6, 3, 3)];
  const spaces = [box(6, 0, 0, 10, 3, 2.8, 101), box(6, 0, 3, 10, 3, 5.8, 102)];
  const cores = coreItems([
    { itemId: 101, type: "IFCSPACE", name: "Treppenhaus - EG" },
    { itemId: 102, type: "IFCSPACE", name: "Treppenhaus - OG" },
  ]);
  const bare = buildLatticeGraph({ ...base, floors, spaces, walls: [tallWall] });
  const findBare = components(bare);
  assert.notEqual(findBare(nearest(bare, 2, 5, 0)), findBare(nearest(bare, 2, 5, 3)), "without the core the floors are apart");

  const g = buildLatticeGraph({ ...base, floors, spaces, walls: [tallWall], cores });
  const find = components(g);
  assert.equal(find(nearest(g, 2, 5, 0)), find(nearest(g, 2, 5, 3)), "the core joins them");
  assert.equal(g.stats.coreStairsAssumed, 1);
  assert.equal(g.cores.length, 1);
  assert.equal(g.cores[0].linked, 1);
  // The line itself: links between stair nodes, about 1.9 times the rise plus
  // the turn at the landing, not the 3 m straight up.
  let length = 0;
  for (let e = 0; e < g.edges.length; e += 2) {
    const a = g.edges[e], b = g.edges[e + 1];
    if (g.kinds[a] !== KIND_STAIR || g.kinds[b] !== KIND_STAIR) continue;
    length += Math.hypot(g.points[b * 3] - g.points[a * 3], g.points[b * 3 + 1] - g.points[a * 3 + 1], g.points[b * 3 + 2] - g.points[a * 3 + 2]);
  }
  assert.ok(length > 2 * Math.hypot(2.4, 1.5) && length < 2 * Math.hypot(2.4, 1.5) + 1.6, `assumed stair ${length} m long`);
});

test("in a double-height stair hall the line turns to end at the landing", () => {
  // The hall (x 6..10, y 0..3) is one room two storeys high: no floor inside
  // it at 3 m. The upper floor is east of it only, so the line must end on
  // the east side, which the unreflected line does not.
  const floors = [rect(0, 0, 14, 10, 0), rect(10, 0, 14, 10, 3)];
  const spaces = [box(6, 0, 0, 10, 3, 5.8, 103)];
  const cores = coreItems([{ itemId: 103, type: "IFCSPACE", name: "Stair hall" }]);
  const g = buildLatticeGraph({ ...base, floors, spaces, walls: [tallWall], cores });
  const find = components(g);
  assert.equal(find(nearest(g, 2, 5, 0)), find(nearest(g, 12, 5, 3)));
  for (let e = 0; e < g.edges.length; e += 2) {
    const a = g.edges[e], b = g.edges[e + 1];
    if ((g.kinds[a] === KIND_STAIR) === (g.kinds[b] === KIND_STAIR)) continue;
    const d = Math.hypot(g.points[b * 3] - g.points[a * 3], g.points[b * 3 + 1] - g.points[a * 3 + 1]);
    assert.ok(d < 1.5, `a foot or head link ${d.toFixed(2)} m long`);
  }
});

test("a stair whose treads are modelled is left as it was", () => {
  // The flight of lattice.test.mjs, as one IfcStair (item 7).
  const stairs = [];
  for (let k = 1; k <= 15; k += 1) stairs.push({ ...rect(2.25 + (k - 1) * 0.25, 0, 2.25 + k * 0.25, 2, 0.1875 * k), itemId: 7 });
  const floors = [rect(0, 0, 2.25, 4, 0), rect(2.25, 2, 10, 4, 0), rect(6, 0, 10, 2, 3)];
  const walls = [box(6, -0.2, 3, 10, 0, 5.8)];
  const plain = buildLatticeGraph({ ...base, floors, stairs, walls });
  const cored = buildLatticeGraph({ ...base, floors, stairs, walls, cores: coreItems([{ itemId: 7, type: "IFCSTAIR", name: "ST-01" }]) });
  assert.deepEqual(Array.from(cored.edges), Array.from(plain.edges));
  assert.deepEqual(Array.from(cored.points), Array.from(plain.points));
  assert.equal(cored.cores[0].linked, 0);
  assert.ok(cored.cores[0].modelled >= 1);
  assert.equal(cored.cores[0].reason, "modelled stair");
});

test("a lift joins the floors beside it only when lifts are on", () => {
  // A 2 x 2 m shaft at x, y 4..6 through the upper floor; the car is drawn on
  // the ground floor only, as exporters often do.
  const floors = [rect(0, 0, 10, 10, 0), rect(0, 0, 4, 10, 3), rect(6, 0, 10, 10, 3), rect(4, 0, 6, 4, 3), rect(4, 6, 6, 10, 3)];
  const coreMeshes = [box(4.2, 4.2, 0, 5.8, 5.8, 2.4, 201)];
  const cores = coreItems([{ itemId: 201, type: "IFCTRANSPORTELEMENT", name: "ELEV-01" }]);
  const off = buildLatticeGraph({ ...base, floors, walls: [tallWall], cores, coreMeshes });
  const findOff = components(off);
  assert.notEqual(findOff(nearest(off, 1, 1, 0)), findOff(nearest(off, 1, 1, 3)));
  assert.equal(off.stats.liftNodes, 0);
  assert.equal(off.cores[0].reason, "lifts and escalators are off");
  assert.equal(off.cores[0].levels, 2);

  const on = buildLatticeGraph({ ...base, floors, walls: [tallWall], cores, coreMeshes, useLifts: true });
  const find = components(on);
  assert.equal(find(nearest(on, 1, 1, 0)), find(nearest(on, 1, 1, 3)));
  assert.equal(on.stats.liftNodes, 2);
  assert.equal(on.stats.liftLinks, 1);
  const stops = [...Array(on.stats.nodes).keys()].filter((n) => on.kinds[n] === KIND_LIFT);
  for (const n of stops) assert.ok(Math.abs(on.points[n * 3] - 5) < 1e-6 && Math.abs(on.points[n * 3 + 1] - 5) < 1e-6, "stops on the shaft's centre line");
});
