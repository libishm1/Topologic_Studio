// Run with: node --test src/core/egressMap.test.mjs
// The egress map on synthetic floors whose answers are known by hand: a
// corridor between two exits, a dead-end arm off it, rooms behind one door
// and a suite behind another.
import test from "node:test";
import assert from "node:assert/strict";
import { DEAD_END_LIMIT, MinHeap, commonPaths, contours, criticalDoors, deadEnds, egressMap, exitField, exitSweep, isCirculation, prepareEgress } from "./egressMap.js";

const S = 0.5;

/** A lattice graph on the floor cells (i, j) at S m, 8 neighbours without corner cutting; door cells KIND_DOOR. */
function gridGraph(cells, doors = new Set()) {
  const index = new Map();
  const nodes = [], kinds = [], edges = [];
  for (const key of cells) {
    const [i, j] = key.split(",").map(Number);
    index.set(key, nodes.length / 3);
    nodes.push(i * S, 0.75, j * S);
    kinds.push(doors.has(key) ? 2 : 0);
  }
  for (const [key, a] of index) {
    const [i, j] = key.split(",").map(Number);
    for (const [di, dj] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
      const b = index.get(`${i + di},${j + dj}`);
      if (b === undefined) continue;
      if (di && dj && (!index.has(`${i + di},${j}`) || !index.has(`${i},${j + dj}`))) continue;
      edges.push(a, b);
    }
  }
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), kinds: Uint8Array.from(kinds), upAxis: "y", agentHeight: 0.75, spacing: S, index };
}

const rect = (x0, z0, x1, z1) => {
  for (let i = Math.round(x0 / S); i <= Math.round(x1 / S); i += 1) for (let j = Math.round(z0 / S); j <= Math.round(z1 / S); j += 1) cellsOf.add(`${i},${j}`);
};
let cellsOf = new Set();

/** A room as roomsFromSpaces gives it: its footprint triangles in plan (x, z), floor, area and rectangle. */
function room(name, x0, z0, x1, z1) {
  const w = x1 - x0, d = z1 - z0;
  return {
    name,
    floor: 0,
    area: w * d,
    footprint: Float64Array.from([x0, z0, x1, z0, x1, z1, x0, z0, x1, z1, x0, z1]),
    rect: { centre: [(x0 + x1) / 2, (z0 + z1) / 2], axis: w >= d ? [1, 0] : [0, 1], half: [Math.max(w, d) / 2, Math.min(w, d) / 2] },
  };
}

/** A 20 m corridor 2 m wide along x, an exit door at each end. */
function corridor() {
  cellsOf = new Set();
  rect(0, 0, 20, 2);
  const doors = new Set(["0,2", "40,2"]);
  const graph = gridGraph(cellsOf, doors);
  const exits = [graph.index.get("0,2"), graph.index.get("40,2")];
  return { graph, exits, rooms: [room("Corridor", 0, 0, 20, 2)] };
}

test("the heap pops keys in order", () => {
  const h = new MinHeap(2);
  const keys = [5, 1, 4, 1.5, 9, 0, 3];
  keys.forEach((k, i) => h.push(k, i));
  const out = [];
  while (h.size) {
    h.pop();
    out.push(h.top);
  }
  assert.deepEqual(out, keys.slice().sort((a, b) => a - b));
});

test("a corridor between two exits: the farthest walk is half its length; closing one exit doubles it", () => {
  const { graph, exits, rooms } = corridor();
  const base = prepareEgress(graph, { exits, levels: [0], rooms });
  assert.equal(base.groups.filter((g) => g.kind === "door").length, 2);
  const field = exitField(base);
  const middle = graph.index.get("20,2");
  assert.ok(Math.abs(field.dist[middle] - 10) < 1e-9, `${field.dist[middle]}`);
  const sweep = exitSweep(base, { limit: 60 });
  assert.equal(sweep.single.length, 2);
  for (const s of sweep.single) assert.ok(Math.abs(s.farthest - Math.hypot(20, 1)) < 0.01 || Math.abs(s.farthest - 20) < 0.6, `${s.farthest}`);
  // Two exits: one round, after which one exit is left.
  assert.equal(sweep.cascade.length, 1);
  // The worst place is at the corridor's side half way: 9 m along and 1 m across, 9 + sqrt(2) m on the lattice.
  const map = egressMap(graph, { exits, levels: [0], rooms, limit: 10.6 });
  assert.equal(map.rooms.length, 1);
  assert.ok(Math.abs(map.rooms[0].dist - (9 + Math.SQRT2)) < 1e-6, `${map.rooms[0].dist}`);
  assert.ok(map.rooms[0].near, "10.41 m against 10.6 m is within 5 % of the limit");
  assert.ok(map.rooms[0].route.length > 10 && map.rooms[0].route.at(-1) === exits[0] || map.rooms[0].route.at(-1) === exits[1]);
  assert.ok(map.exits.every((e) => e.catchment.area > 15), JSON.stringify(map.exits.map((e) => e.catchment)));
});

test("contours cross the corridor at every 5 m of walk", () => {
  const { graph, exits, rooms } = corridor();
  const base = prepareEgress(graph, { exits, levels: [0], rooms });
  const dashes = contours(base, exitField(base).dist, { step: 5 });
  // Two levels (5 m, and 10 m in the middle) from each end: dashes at x near 5, 10 and 15.
  const xs = new Set();
  for (let k = 0; k < dashes.length; k += 6) xs.add(Math.round((dashes[k] + dashes[k + 3]) / 2));
  for (const x of [5, 10, 15]) assert.ok([...xs].some((v) => Math.abs(v - x) <= 1), `${[...xs]}`);
});

/** The corridor with an arm 1 m wide going 8 m off its middle, closed at its end. */
function tee() {
  cellsOf = new Set();
  rect(0, 0, 20, 2);
  rect(9.5, 2.5, 10.5, 10.5);
  const doors = new Set(["0,2", "40,2"]);
  const graph = gridGraph(cellsOf, doors);
  const exits = [graph.index.get("0,2"), graph.index.get("40,2")];
  return { graph, exits, rooms: [room("Corridor", 0, 0, 20, 2), room("Corridor B", 9.5, 2.5, 10.5, 10.5)] };
}

test("a dead-end arm: its length is the walk back to where two ways open, against IBC 1020.5", () => {
  const { graph, exits, rooms } = tee();
  const base = prepareEgress(graph, { exits, levels: [0], rooms });
  const common = commonPaths(base);
  const tip = graph.index.get("20,21");
  assert.ok(common[tip] > 8 && common[tip] < 10, `the tip's common path ${common[tip]}`);
  const width = new Float64Array(base.n).fill(1);
  const plain = deadEnds(base, common, { widthAt: width });
  assert.equal(plain.corridors.length, 1, JSON.stringify(plain.corridors.map((c) => c.length)));
  const arm = plain.corridors[0];
  assert.ok(arm.over && arm.length > DEAD_END_LIMIT.plain, `${arm.length}`);
  assert.equal(arm.width, 1);
  assert.ok(Math.abs(arm.tip[2] - 10.5) < 1e-6);
  const sprinklered = deadEnds(base, common, { sprinklered: true, widthAt: width });
  assert.ok(!sprinklered.corridors[0].over);
  // Four metres wide, the same arm is shorter than 2.5 times its width: exempt.
  const wide = deadEnds(base, common, { widthAt: new Float64Array(base.n).fill(4) });
  assert.ok(wide.corridors[0].exempt && !wide.corridors[0].over);
  // The main corridor between two exits has no dead end of its own.
  assert.ok(plain.oneWayShare > 0 && plain.oneWayShare < 0.5, `${plain.oneWayShare}`);
});

test("critical doors: a room behind one door, a suite behind another; a room with two doors has none", () => {
  cellsOf = new Set();
  rect(0, 0, 20, 2); // corridor
  rect(2, 3, 5, 6); // office 1, one door at (3.5, 2.5)
  rect(7, 3, 10, 6); // office 2, two doors at (8, 2.5) and (9.5, 2.5)
  rect(12, 3, 15, 6); // suite: outer room
  rect(12, 7, 15, 10); // inner room, through a door at (13.5, 6.5)
  const doors = new Set(["7,5", "16,5", "19,5", "27,5", "27,13", "0,2", "40,2"]);
  for (const d of doors) cellsOf.add(d);
  const graph = gridGraph(cellsOf, doors);
  const exits = [graph.index.get("0,2"), graph.index.get("40,2")];
  const rooms = [room("Corridor", 0, 0, 20, 2), room("Office 1", 2, 3, 5, 6), room("Office 2", 7, 3, 10, 6), room("Suite", 12, 3, 15, 6), room("Inner", 12, 7, 15, 10)];
  const base = prepareEgress(graph, { exits, levels: [0], rooms });
  const { doors: critical, suites } = criticalDoors(base);
  const behind = (name) => critical.find((d) => d.names.includes(name));
  assert.ok(behind("Office 1"), "Office 1's only door");
  assert.ok(!behind("Office 2"), "Office 2 has two doors");
  assert.equal(suites.length, 1);
  assert.deepEqual(suites[0].names.sort(), ["Inner", "Suite"]);
  assert.ok(critical.some((d) => d.names.length === 1 && d.names[0] === "Inner"));
});

test("circulation by name, by shape, or, for a space whose name says nothing, by Poux's rule", () => {
  assert.ok(isCirculation(room("2AC1 CORRIDOR", 0, 0, 3, 4)));
  assert.ok(isCirculation(room("Unnamed", 0, 0, 12, 1.8)));
  assert.ok(!isCirculation(room("Office", 0, 0, 4, 3)));
  assert.ok(!isCirculation(room("Assembly hall", 0, 0, 20, 15)));
  // Poux (2026): the storey's largest space, one an exit opens from, or one with four doors or more.
  assert.ok(isCirculation(room("A101", 0, 0, 8, 6), { doors: 4 }));
  assert.ok(isCirculation(room(null, 0, 0, 8, 6), { exit: true }));
  assert.ok(isCirculation(room("B-12", 0, 0, 8, 6), { largest: true }));
  assert.ok(!isCirculation(room("A101", 0, 0, 8, 6), { doors: 2 }));
  // A name that says what the room is wins over the rule.
  assert.ok(!isCirculation(room("Open office", 0, 0, 8, 6), { doors: 5, largest: true }));
});

test("a dead end round a corner: its depth is the walk back, not a difference of distances", () => {
  cellsOf = new Set();
  rect(0, 0, 20, 2);
  rect(9.5, 2.5, 10.5, 6.5); // up 4 m
  rect(10.5, 5.5, 14.5, 6.5); // then right 4 m
  const doors = new Set(["0,2", "40,2"]);
  const graph = gridGraph(cellsOf, doors);
  const exits = [graph.index.get("0,2"), graph.index.get("40,2")];
  const rooms = [room("Corridor", 0, 0, 20, 2), room("Corridor B", 9.5, 2.5, 10.5, 6.5), room("Corridor C", 10.5, 5.5, 14.5, 6.5)];
  const base = prepareEgress(graph, { exits, levels: [0], rooms });
  const common = commonPaths(base);
  const dead = deadEnds(base, common, { widthAt: new Float64Array(base.n).fill(1) });
  const tip = graph.index.get("29,12");
  // On a lattice the product can read short round a corner; the walk back never reads shorter by more than a link.
  assert.ok(dead.depth[tip] >= common[tip] - S, `the walk back ${dead.depth[tip]} against the product ${common[tip]}`);
  // Roughly 8 m of stub, less the half-metre lattice corners.
  assert.ok(dead.depth[tip] > 7 && dead.depth[tip] < 9.5, `${dead.depth[tip]}`);
  assert.ok(dead.corridors.length >= 1 && dead.corridors[0].length > 7);
});
