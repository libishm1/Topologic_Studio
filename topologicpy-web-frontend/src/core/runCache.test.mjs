/** The run store's pure parts (runCache.js); IndexedDB itself is tested in the browser suite. */
import test from "node:test";
import assert from "node:assert/strict";
import { evictionPlan, fingerprint, runBytes } from "./runCache.js";

test("a fingerprint changes with any word it reads and is the same for the same data", () => {
  const a = Float32Array.from([1, 2, 3, 4]);
  const b = Uint32Array.from([7, 8]);
  assert.equal(fingerprint([a, b]), fingerprint([Float32Array.from(a), Uint32Array.from(b)]));
  assert.notEqual(fingerprint([a, b]), fingerprint([Float32Array.from([1, 2, 3, 5]), b]));
  // The length counts too: an array and the same array with a zero appended differ.
  assert.notEqual(fingerprint([Float32Array.from([1, 2])]), fingerprint([Float32Array.from([1, 2, 0])]));
  // A view counts only its own bytes.
  const whole = Float32Array.from([9, 1, 2, 3, 4]);
  assert.equal(fingerprint([whole.subarray(1)]), fingerprint([a]));
});

test("a run's bytes are its frames' typed arrays, nested ones included", () => {
  const frame = { time: 1, T: new Float32Array(10), particles: { positions: new Float32Array(6), kinds: new Uint8Array(2) }, smoke: null };
  assert.equal(runBytes([frame, frame]), 2 * (40 + 24 + 2));
  assert.equal(runBytes([]), 0);
});

test("making room drops the least recently used runs, and only as many as needed", () => {
  const metas = [
    { key: "old", bytes: 50, usedAt: 1 },
    { key: "mid", bytes: 50, usedAt: 2 },
    { key: "new", bytes: 50, usedAt: 3 },
  ];
  assert.deepEqual(evictionPlan(metas, 40, 200), []);
  assert.deepEqual(evictionPlan(metas, 60, 200), ["old"]);
  assert.deepEqual(evictionPlan(metas, 120, 200), ["old", "mid"]);
  assert.deepEqual(evictionPlan(metas, 300, 200), ["old", "mid", "new"]);
});
