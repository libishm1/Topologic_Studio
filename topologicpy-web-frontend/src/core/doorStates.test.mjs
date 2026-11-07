/** Door states for a Flamescapes run (doorStates.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { closedDoorIds, doorsLabel, seeded, selfClosingRating } from "./doorStates.js";

test("the self-closing mark of an EN 13501-2 class is read, and nothing else", () => {
  for (const r of ["EI2 30-C", "EI2 30-C-Sm", "EI2 60-C5", "E 30-Sa-C3", "EI 30-C, Sm"]) assert.ok(selfClosingRating(r), r);
  for (const r of ["EI 30", "EI 60", "R 90", "ND", "", null, "EI2 30-Cx", "C-30"]) assert.ok(!selfClosingRating(r), String(r));
});

test("as built closes the self-closing doors; open and closed close none and all", () => {
  const ids = [5, 3, 9, 3];
  assert.deepEqual([...closedDoorIds(ids, "open")], []);
  assert.deepEqual([...closedDoorIds(ids, "closed")], [3, 5, 9]);
  assert.deepEqual([...closedDoorIds(ids, "asbuilt", { selfClosing: new Set([9, 4]) })], [9]);
});

test("random doors: the same seed closes the same doors, whatever the order given; the share holds", () => {
  const ids = Array.from({ length: 2000 }, (_, i) => i * 7 + 1);
  const a = closedDoorIds(ids, "random", { seed: 42, share: 0.3 });
  const b = closedDoorIds([...ids].reverse(), "random", { seed: 42, share: 0.3 });
  assert.deepEqual([...a], [...b]);
  assert.notDeepEqual([...a], [...closedDoorIds(ids, "random", { seed: 43, share: 0.3 })]);
  assert.ok(Math.abs(a.size / ids.length - 0.3) < 0.04, `${a.size}`);
  assert.equal(closedDoorIds(ids, "random", { seed: 1, share: 0 }).size, 0);
  assert.equal(closedDoorIds(ids, "random", { seed: 1, share: 1 }).size, ids.length);
  const next = seeded(7);
  for (let i = 0; i < 1000; i += 1) {
    const x = next();
    assert.ok(x >= 0 && x < 1);
  }
  assert.equal(doorsLabel("random", { seed: 7, share: 0.25 }), "random (25% closed, seed 7)");
  assert.equal(doorsLabel("asbuilt"), "as built");
});
