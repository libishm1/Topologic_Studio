/** The guided runs' choices on the graph (guided.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { GUIDED_RUNS, farthestFromExits, nodeAlong, stagesFor } from "./guided.js";

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
