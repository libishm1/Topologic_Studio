// Run with: npm test
// The deadline map's colours (src/viewer/heat.js), which ViewerManager draws.
import assert from "node:assert/strict";
import { test } from "node:test";
import { Color } from "three";

import { HEAT_STOPS, heatColor, paintDeadlines } from "../viewer/heat.js";

const none = new Color(0x52525b);

test("a deadline map is drawn only on the graph it was made for (review C1)", () => {
  // A path of 3 nodes; the map came from a rebuilt graph of 5.
  const edges = Uint32Array.from([0, 1, 1, 2]);
  const colors = new Float32Array(12).fill(-1);
  assert.equal(paintDeadlines(colors, edges, 3, new Float64Array(5).fill(10), { max: 300, none }), false);
  assert.equal(paintDeadlines(colors, edges, 3, new Float64Array(2).fill(10), { max: 300, none }), false);
  assert.equal(paintDeadlines(colors, edges, 3, null, { max: 300, none }), false);
  assert.ok(colors.every((c) => c === -1), "nothing is written for a map of the wrong length");
});

test("no departure, and a value that is not a number, are drawn as none", () => {
  const edges = Uint32Array.from([0, 1, 1, 2, 2, 3]);
  const colors = new Float32Array(18);
  assert.equal(paintDeadlines(colors, edges, 4, [0, -Infinity, NaN, undefined], { max: 300, none }), true);
  assert.ok(colors.every(Number.isFinite));
  const at = (offset) => [colors[offset], colors[offset + 1], colors[offset + 2]];
  // Node 0 (deadline 0: no time left) is the hot end; nodes 1 to 3 are none.
  assert.deepEqual(at(0), [HEAT_STOPS[3].r, HEAT_STOPS[3].g, HEAT_STOPS[3].b].map(Math.fround));
  for (const offset of [3, 6, 9, 12, 15]) assert.deepEqual(at(offset), [none.r, none.g, none.b].map(Math.fround));
});

test("the heat ramp reads NaN as its cool end instead of throwing", () => {
  const c = heatColor(new Color(), NaN, 0, 300);
  assert.deepEqual([c.r, c.g, c.b], [HEAT_STOPS[0].r, HEAT_STOPS[0].g, HEAT_STOPS[0].b]);
});
