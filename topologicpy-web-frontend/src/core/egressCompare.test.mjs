import test from "node:test";
import assert from "node:assert/strict";
import { compareEgressSlime } from "./egressCompare.js";

/** A W x H lattice of nodes 0.5 m apart, 8-connected. */
function lattice(W, H) {
  const nodes = [], edges = [];
  const id = (i, j) => j * W + i;
  for (let j = 0; j < H; j += 1) for (let i = 0; i < W; i += 1) nodes.push(i * 0.5, 0.75, j * 0.5);
  for (let j = 0; j < H; j += 1) for (let i = 0; i < W; i += 1) {
    for (const [a, b] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
      const ii = i + a, jj = j + b;
      if (ii >= 0 && jj >= 0 && ii < W && jj < H) edges.push(id(i, j), id(ii, jj));
    }
  }
  return { graph: { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), spacing: 0.5 }, id };
}

test("a slime on the shortest walk reads 1; one 3 m away reads 0; the exits' split and the walls are compared", () => {
  const { graph, id } = lattice(21, 11);
  const n = 21 * 11;
  // One room's shortest walk along the bottom row, to an exit at its left end.
  const route = Array.from({ length: 21 }, (_, i) => id(20 - i, 0));
  const exitField = new Int32Array(n).fill(0);
  const egress = { rooms: [{ route }], exits: [{ label: "A", catchment: { area: 30 } }, { label: "B", catchment: { area: 10 } }], field: { exit: exitField } };
  exitField[id(20, 10)] = 1;
  const on = new Float32Array(n);
  for (const v of route) on[v] = 1;
  const same = compareEgressSlime(graph, egress, { weights: on, exits: [{ node: id(0, 0), flux: 75 }, { node: id(20, 10), flux: 25 }] });
  assert.equal(same.onShortest, 1);
  assert.equal(same.covered, 1);
  assert.equal(same.exitsApart, 0, "the slime leaves 75/25 as the floor splits 30/10");
  // The bottom row is the lattice's edge: every shortest node and all the slime's trail sit at a wall.
  assert.equal(same.atWalls.shortest, 1);
  assert.equal(same.atWalls.slime, 1);

  const off = new Float32Array(n);
  for (let i = 0; i < 21; i += 1) off[id(i, 6)] = 1; // 3 m from the walk
  const apart = compareEgressSlime(graph, egress, { weights: off, exits: [{ node: id(20, 10), flux: 1 }] });
  assert.equal(apart.onShortest, 0);
  assert.equal(apart.covered, 0);
  assert.equal(apart.exitsApart, 0.75, "all by B, which is nearest for a quarter of the floor");
  assert.ok(Math.abs(apart.atWalls.slime - 2 / 21) < 1e-9, "row 6 touches a wall only at its two ends");
  assert.equal(compareEgressSlime(graph, null, { weights: on }), null);
});

test("without routes (as runMap sends them), each room's walk is rebuilt from the walk field", () => {
  const { graph, id } = lattice(21, 11);
  const n = 21 * 11;
  // The walk field to an exit at the bottom-left corner: straight-line distance on the lattice.
  const dist = new Float32Array(n);
  for (let j = 0; j < 11; j += 1) for (let i = 0; i < 21; i += 1) {
    const [a, b] = [i * 0.5, j * 0.5];
    dist[id(i, j)] = Math.SQRT2 * Math.min(a, b) + Math.abs(a - b);
  }
  const egress = { rooms: [{ node: id(20, 0), dist: 10 }], exits: [{ label: "A", catchment: { area: 1 } }], field: { dist, exit: new Int32Array(n) } };
  const on = new Float32Array(n);
  for (let i = 0; i < 21; i += 1) on[id(i, 0)] = 1;
  const c = compareEgressSlime(graph, egress, { weights: on, exits: [] });
  assert.equal(c.onShortest, 1, "the rebuilt walk is the bottom row");
  assert.equal(c.covered, 1);
});
