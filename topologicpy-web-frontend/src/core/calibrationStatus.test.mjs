import test from "node:test";
import assert from "node:assert/strict";
import { TESTED_CELLS, TESTED_GRID_CELL, VALIDATION, nearestTestedCell, statusFor, validationLine, validationVerdict } from "./calibrationStatus.js";

const words = (s) => VALIDATION.flatMap((v) => Object.entries(v.status).filter(([, w]) => w === s).map(([cell]) => `${v.id}@${cell}`));

test("13.B's words: the layer's depth on 0.20 and 0.34 m and the neutral plane on 0.20 m are calibrated, nothing else", () => {
  assert.deepEqual(words("calibrated"), ["layerDepth@0.2", "layerDepth@0.34", "neutralPlane@0.2"]);
  assert.deepEqual(words("indicative"), ["layerDepth@0.27", "layerTemperature@0.2", "ceilingJet@0.2", "co2@0.2", "co2@0.27", "neutralPlane@0.27"]);
  for (const v of VALIDATION) {
    // Every quantity has a word on every tested cell, and a calibrated one its figures.
    assert.deepEqual(Object.keys(v.status).map(Number), TESTED_CELLS, v.id);
    for (const [cell, s] of Object.entries(v.status)) {
      assert.ok(["calibrated", "indicative", "not calibrated"].includes(s), `${v.id}: ${s}`);
      if (s === "calibrated") assert.match(v.figures?.[cell] ?? "", /^bias \d\.\d\d, scatter \d\.\d\d, in band in \w+ of \w+ series$/, `${v.id}@${cell}`);
    }
    // "From one laboratory's tests" goes with a verdict, never with "not calibrated".
    for (const cell of v.oneLab ?? []) assert.notEqual(v.status[cell], "not calibrated", `${v.id}@${cell}`);
  }
  assert.deepEqual(TESTED_CELLS, [0.2, 0.27, 0.34]);
  assert.ok(TESTED_CELLS.includes(TESTED_GRID_CELL));
});

test("13.A's words stand beside 13.B's: four differ", () => {
  const differ = VALIDATION.flatMap((v) => Object.entries(v.before ?? {}).map(([cell, was]) => `${v.id}@${cell}: ${was} -> ${v.status[cell]}`));
  assert.deepEqual(differ, [
    "layerDepth@0.27: calibrated -> indicative",
    "headTemperature@0.2: indicative -> not calibrated",
    "neutralPlane@0.2: indicative -> calibrated",
    "neutralPlane@0.34: indicative -> not calibrated",
  ]);
});

test("a verdict holds on the cells it was found on; near them it is at most indicative; further off, not tested", () => {
  assert.equal(nearestTestedCell(0.27), 0.27);
  assert.equal(nearestTestedCell(0.3), 0.27);
  assert.equal(nearestTestedCell(0.37), 0.34);
  assert.equal(nearestTestedCell(0.233), 0.2);
  assert.equal(nearestTestedCell(0.4), null);
  assert.equal(nearestTestedCell(0), null);
  assert.equal(statusFor(0.342).find((r) => r.id === "layerDepth").said, "calibrated");
  assert.equal(statusFor(0.37).find((r) => r.id === "layerDepth").said, "indicative");
  assert.equal(statusFor(0.203).find((r) => r.id === "neutralPlane").said, "calibrated");
  assert.equal(statusFor(0.22).find((r) => r.id === "neutralPlane").said, "indicative");
  assert.equal(statusFor(0.272).find((r) => r.id === "layerDepth").said, "indicative");
  assert.ok(statusFor(0.45).every((r) => r.said === "not tested on cells this size"));
});

test("the run card's sentence: what decides untenability, what holds on its cells, and under what conditions", () => {
  const on20 = validationLine(0.2);
  const on27 = validationLine(0.27);
  const on34 = validationLine(0.34);
  assert.match(on27, /^This run's untenable places come from smoke visibility and gas temperature at head height/);
  // A calibrated quantity carries its own figures.
  assert.match(on20, /smoke layer depth \(bias 1\.00, scatter 0\.14, in band in six of seven series\) and neutral plane in a door \(bias 0\.90, scatter 0\.12, in band in three of four series\) calibrated;/);
  assert.match(on34, /: smoke layer depth \(bias 1\.11, scatter 0\.14, in band in five of seven series\) calibrated;/);
  assert.doesNotMatch(on27, / calibrated;/);
  // An indicative one says when every series in band is one laboratory's.
  assert.match(on20, /smoke layer temperature, gas temperature under the ceiling and carbon dioxide indicative, each from one laboratory's tests;/);
  assert.match(on27, /smoke layer depth, carbon dioxide \(from one laboratory's tests\) and neutral plane in a door indicative;/);
  assert.doesNotMatch(on34, /indicative/);
  // What the run draws is not calibrated on any tested cell.
  for (const line of [on20, on27, on34]) assert.match(line, /gas temperature at head height, .*smoke visibility not calibrated\./);
  assert.match(on34, /smoke layer temperature, gas temperature under the ceiling, gas temperature at head height, carbon dioxide, oxygen, flow through a door, neutral plane in a door and smoke visibility not calibrated\./);
  assert.match(on27, /each test's own walls in layers/);
  assert.match(on27, /predicts its own heat release, which is not calibrated, and its walls are stone\.$/);
  assert.match(validationLine(0.3), /tested on 0\.27 m; a verdict holds only on the cells it was found on/);
  assert.doesNotMatch(validationLine(0.37), / calibrated;/);
  assert.match(validationLine(0.37), /smoke layer depth indicative;/);
  assert.doesNotMatch(validationLine(0.22), / calibrated;/);
  assert.match(validationLine(0.45), /Not compared with measured fires on 0\.45 m cells/);
  assert.match(validationLine(0.27, { peakKW: 5000 }), /outside the tested range\.$/);
  assert.match(validationLine(0.27, { horizon: 1200 }), /outside the tested range\.$/);
  assert.doesNotMatch(validationLine(0.27, { peakKW: 1000, horizon: 300 }), /outside/);
});

test("the run card leads with one line: what holds for what it draws, and the calibrated quantities it does not draw", () => {
  assert.equal(validationVerdict(0.2), "Smoke layer depth and neutral plane in a door calibrated on these 0.20 m cells, not drawn; the visibility and temperature drawn are not calibrated.");
  assert.equal(validationVerdict(0.27), "On these 0.27 m cells the visibility and temperature drawn are not calibrated.");
  assert.equal(validationVerdict(0.34), "Smoke layer depth calibrated on these 0.34 m cells, not drawn; the visibility and temperature drawn are not calibrated.");
  assert.equal(validationVerdict(0.37), "On these 0.37 m cells the visibility and temperature drawn are not calibrated.");
  assert.equal(validationVerdict(0.45), "Not compared with measured fires on 0.45 m cells: read it as a ranking, not a prediction.");
  // Past the tests' 900 s or 3.2 MW, it says so.
  assert.match(validationVerdict(0.27, { horizon: 1000 }), / Outside the tested range.$/);
  assert.match(validationVerdict(0.45, { peakKW: 5000 }), / Outside the tested range.$/);
  // Never "calibrated" for anything drawn.
  for (const c of [0.2, 0.27, 0.3, 0.34, 0.37]) assert.doesNotMatch(validationVerdict(c), /drawn (is|are) calibrated/);
});
