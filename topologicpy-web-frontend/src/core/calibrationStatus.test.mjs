import test from "node:test";
import assert from "node:assert/strict";
import { TESTED_CELLS, VALIDATION, nearestTestedCell, statusFor, validationLine, validationVerdict } from "./calibrationStatus.js";

test("only the smoke layer's depth on 0.27 m cells is calibrated", () => {
  const calibrated = VALIDATION.flatMap((v) => Object.entries(v.status).filter(([, s]) => s === "calibrated").map(([cell]) => `${v.id}@${cell}`));
  assert.deepEqual(calibrated, ["layerDepth@0.27"]);
  for (const v of VALIDATION) for (const s of Object.values(v.status)) assert.ok(["calibrated", "indicative", "not calibrated"].includes(s), `${v.id}: ${s}`);
  assert.deepEqual(TESTED_CELLS, [0.2, 0.27, 0.34]);
});

test("a verdict holds on the cells it was found on; near them it is at most indicative; further off, not tested", () => {
  assert.equal(nearestTestedCell(0.27), 0.27);
  assert.equal(nearestTestedCell(0.3), 0.27);
  assert.equal(nearestTestedCell(0.37), 0.34);
  assert.equal(nearestTestedCell(0.233), 0.2);
  assert.equal(nearestTestedCell(0.4), null);
  assert.equal(nearestTestedCell(0), null);
  assert.equal(statusFor(0.272).find((r) => r.id === "layerDepth").said, "calibrated");
  assert.equal(statusFor(0.3).find((r) => r.id === "layerDepth").said, "indicative");
  assert.ok(statusFor(0.45).every((r) => r.said === "not tested on cells this size"));
});

test("the run card's sentence: what decides untenability, what holds on its cells, and under what conditions", () => {
  const on27 = validationLine(0.27);
  assert.match(on27, /^This run's untenable places come from smoke visibility and gas temperature at head height/);
  assert.match(on27, /smoke layer depth calibrated \(bias 1\.08, scatter 0\.15/);
  assert.match(on27, /smoke layer temperature, gas temperature under the ceiling, .* and smoke visibility not calibrated/);
  assert.match(validationLine(0.34), /smoke layer depth, smoke layer temperature and gas temperature under the ceiling indicative/);
  assert.match(on27, /smoke visibility not calibrated/);
  assert.match(on27, /insulating-board walls/);
  assert.match(on27, /predicts its own heat release, which is not calibrated, and its walls are stone\.$/);
  assert.match(validationLine(0.3), /tested on 0\.27 m; a verdict holds only on the cells it was found on/);
  assert.doesNotMatch(validationLine(0.3), /depth calibrated/);
  assert.doesNotMatch(validationLine(0.37), /depth calibrated/);
  assert.match(validationLine(0.45), /Not compared with measured fires on 0\.45 m cells/);
  assert.match(validationLine(0.27, { peakKW: 5000 }), /outside the tested range\.$/);
  assert.match(validationLine(0.27, { horizon: 1200 }), /outside the tested range\.$/);
  assert.doesNotMatch(validationLine(0.27, { peakKW: 1000, horizon: 300 }), /outside/);
});

test("the run card leads with one line: what holds for what it draws, and the calibrated quantity it does not draw", () => {
  assert.equal(validationVerdict(0.27), "Smoke layer depth calibrated on these 0.27 m cells, not drawn; the visibility and temperature drawn are not calibrated.");
  assert.equal(validationVerdict(0.34), "On these 0.34 m cells the visibility drawn is not calibrated, the temperature indicative.");
  assert.equal(validationVerdict(0.2), "On these 0.20 m cells the visibility and temperature drawn are not calibrated.");
  assert.equal(validationVerdict(0.45), "Not compared with measured fires on 0.45 m cells: read it as a ranking, not a prediction.");
  // Past the tests' 900 s or 3.2 MW, it says so.
  assert.match(validationVerdict(0.27, { horizon: 1000 }), / Outside the tested range.$/);
  assert.match(validationVerdict(0.45, { peakKW: 5000 }), / Outside the tested range.$/);
  // Never "calibrated" for anything drawn.
  for (const c of [0.2, 0.27, 0.3, 0.34, 0.37]) assert.doesNotMatch(validationVerdict(c), /drawn (is|are) calibrated/);
});
