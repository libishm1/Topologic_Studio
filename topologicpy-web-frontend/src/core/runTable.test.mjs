/**
 * A run as a table (core/runTable.js):
 *   node --test src/core/runTable.test.mjs
 */
import test from "node:test";
import assert from "node:assert/strict";

import { runTableChunks } from "./runTable.js";

const run = (extras) => ({
  n: 2,
  frameSeconds: 30,
  points: Float32Array.from([0, 0, 0, 1.5, 2, 3]),
  kinds: Uint8Array.from([0, 2]),
  frames: [1, 2].map((k) => ({
    time: k * 30,
    T: Float32Array.from([20, 20 + 10 * k]),
    vis: Float32Array.from([100, 20 / k]),
    co: Float32Array.from([0, 1000 * k]),
    ...(extras ? { soot: Float32Array.from([0, 20 * k]), co2: Float32Array.from([0.04, 1.5 * k]), o2: Float32Array.from([20.9, 20.9 - 1.5 * k]) } : {}),
  })),
});

test("one row per node per frame, with the extra measures when the run kept them", () => {
  const text = [...runTableChunks(run(true))].join("");
  const lines = text.trim().split("\n");
  assert.equal(lines.length, 1 + 2 * 2);
  const head = lines[0].split(",");
  assert.deepEqual(head.slice(0, 8), ["node", "x", "y", "z", "kind", "time_s", "gas_temperature_C", "visibility_m"]);
  assert.ok(head.includes("soot_mg_m3") && head.includes("o2_pct") && head.at(-1) === "limiting");
  const last = Object.fromEntries(head.map((h, i) => [h, lines.at(-1).split(",")[i]]));
  assert.equal(last.node, "1");
  assert.equal(last.kind, "door");
  assert.equal(last.time_s, "60");
  // Half a minute of 1000 ppm CO at 1.5 % CO2 (no factor below 2 %), then of
  // 2000 ppm at 3 % CO2 (factor exp(3/5)), plus Purser's low-O2 dose at 19.4 and 17.9 % O2.
  const lowO2 = (o2) => 1 / Math.exp(8.13 - 0.54 * (20.9 - o2));
  const expected = 0.5 * (1000 / 35000 + lowO2(19.4)) + 0.5 * ((2000 / 35000) * Math.exp(3 / 5) + lowO2(17.9));
  assert.equal(Number(last.fed_asphyxiant).toFixed(5), expected.toFixed(5));
  // Visibility 10 m at a 10 m limit: the index is at least 1.
  assert.ok(Number(last.tenability_index) >= 1);
  assert.equal(last.limiting, "smoke");
});

test("without the extra measures the table keeps the basic columns", () => {
  const head = [...runTableChunks(run(false))][0].trim().split(",");
  assert.ok(!head.includes("soot_mg_m3"));
  assert.ok(head.includes("co_ppm") && head.includes("tenability_index"));
});
