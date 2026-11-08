// Run with: npm test
// The graph's statistics as the Graph step shows them (src/core/graph.js).
import assert from "node:assert/strict";
import { test } from "node:test";

import { KIND_FLOOR } from "./lattice.js";
import { components, graphStats, reachablePct } from "./graph.js";

test("a link blocked by a wall joins nothing, as in the route search", () => {
  // Two rooms of two nodes each, joined only by a link through a wall.
  const edges = Uint32Array.from([0, 1, 1, 2, 2, 3]);
  const blocked = Uint8Array.from([0, 1, 0]);
  assert.deepEqual(components(4, edges), { count: 1, largest: 4 });
  assert.deepEqual(components(4, edges, blocked), { count: 2, largest: 2 });

  const stats = graphStats({ points: new Float32Array(12), edges, kinds: new Uint8Array(4).fill(KIND_FLOOR), blocked });
  assert.equal(stats.components, 2);
  assert.equal(stats.blocked_edges, 1);
  assert.equal(reachablePct(stats), 50);
});
