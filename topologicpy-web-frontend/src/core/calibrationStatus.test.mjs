import test from "node:test";
import assert from "node:assert/strict";
import { TESTED_CELLS, VALIDATION, nearestTestedCell, statusFor, validationLine } from "./calibrationStatus.js";

test("only the smoke layer's depth on 0.27 m cells is calibrated", () => {
  const calibrated = VALIDATION.flatMap((v) => Object.entries(v.status).filter(([, s]) => s === "calibrated").map(([cell]) => `${v.id}@${cell}`));
  assert.deepEqual(calibrated, ["layerDepth@0.27"]);
  for (const v of VALIDATION) for (const s of Object.values(v.status)) assert.ok(["calibrated", "indicative", "not calibrated"].includes(s), `${v.id}: ${s}`);
  assert.deepEqual(TESTED_CELLS, [0.2, 0.27, 0.34]);
});

test("a run's cells are matched to a tested size within 0.035 m, else not tested", () => {
  assert.equal(nearestTestedCell(0.27), 0.27);
  assert.equal(nearestTestedCell(0.3), 0.27);
  assert.equal(nearestTestedCell(0.37), 0.34);
  assert.equal(nearestTestedCell(0.233), 0.2);
  assert.equal(nearestTestedCell(0.4), null);
  assert.equal(nearestTestedCell(0), null);
  assert.ok(statusFor(0.45).every((r) => r.said === "not tested on cells this size"));
});

test("the run card's sentence says what holds on its cells, and the walls the tests had", () => {
  const on27 = validationLine(0.27);
  assert.match(on27, /smoke layer depth calibrated/);
  assert.match(on27, /smoke layer temperature and gas temperature under the ceiling indicative/);
  assert.match(on27, /smoke visibility not calibrated/);
  assert.match(on27, /insulating-board walls/);
  assert.match(validationLine(0.37), /nearest tested \(0\.34 m\)/);
  assert.doesNotMatch(validationLine(0.37), /depth calibrated/);
  assert.match(validationLine(0.45), /^Not compared with measured fires on 0\.45 m cells/);
});
