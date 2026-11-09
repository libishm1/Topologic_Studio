/** The guided runs' choices on the graph (guided.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { GUIDED_RUNS, farthestFromExits, guidedOrigin, nodeAlong, stagesFor } from "./guided.js";

test("the start is the floor node most links from any exit, past blocked links", () => {
  // A path 0-1-2-3-4 with the exit at 0, and a door node 5 hanging off 4; node 6 joins 0 only by a blocked link.
  const graph = {
    nodes: new Float32Array(7 * 3),
    edges: Uint32Array.from([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 6]),
    blocked: Uint8Array.from([0, 0, 0, 0, 0, 1]),
    kinds: Uint8Array.from([0, 0, 0, 0, 0, 2, 0]),
  };
  assert.equal(farthestFromExits(graph, [0]), 4, "node 5 is a door, node 6 cannot be reached");
  assert.equal(farthestFromExits(graph, [4]), 0);
  assert.equal(farthestFromExits(graph, []), -1);
});

test("a node part-way along a route, and the stages each preset passes through", () => {
  assert.equal(nodeAlong([10, 11, 12, 13, 14], 0.45), 12);
  assert.equal(nodeAlong([10, 11], 1), 11);
  assert.equal(nodeAlong([], 0.5), -1);
  for (const run of GUIDED_RUNS) {
    const ids = stagesFor(run).map((s) => s.id);
    assert.equal(ids[0], "open");
    assert.equal(ids.at(-1), "done");
    assert.equal(ids.includes("fire"), Boolean(run.fire));
  }
});

test("a guided run lights its fire at the likeliest origin near the route, on the walkable graph", () => {
  // A corridor of 21 nodes 1 m apart along x (exit at 0), and an island of
  // two nodes at x = 5, z = 10 that reaches no exit.
  const nodes = [];
  for (let i = 0; i <= 20; i += 1) nodes.push(i, 0, 0);
  nodes.push(5, 0, 10, 6, 0, 10);
  const edges = [];
  for (let i = 0; i < 20; i += 1) edges.push(i, i + 1);
  edges.push(21, 22);
  const graph = { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: new Uint8Array(23), upAxis: "y" };
  const route = Array.from({ length: 21 }, (_, i) => 20 - i);
  const at = (x, z) => ({ position: [x, 0, z], scenario: { label: `at ${x}`, fireClass: "A" } });
  // The likeliest is on the island (no exit), the next beside the start
  // (too close), then two reachable ones: the nearer the route wins.
  const candidates = [at(5.2, 10), at(19.5, 0.2), at(12, 6), at(9, 1)];
  const pick = guidedOrigin(graph, { candidates, route, exits: [0], start: 20 });
  assert.equal(pick.candidate, candidates[3]);
  assert.equal(pick.node, 9);
  // Only beyond the top few, or none at all: a node part-way along the route.
  assert.deepEqual(guidedOrigin(graph, { candidates: [], route, exits: [0], start: 20 }), { node: nodeAlong(route, 0.45), candidate: null });
  assert.equal(guidedOrigin(graph, { candidates: [at(5.2, 10)], route, exits: [0], start: 20 }).candidate, null);
  // Never at the start or an exit.
  for (const c of [at(0, 0), at(20, 0)]) assert.equal(guidedOrigin(graph, { candidates: [c], route, exits: [0], start: 20 }).candidate, null);
});
