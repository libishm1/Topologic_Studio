/**
 * Unit tests of lib/species.mjs (what --store-species keeps; plan 16.54):
 *   node --test tools/calibration/lib/species.test.mjs
 * Made-up cells and states with known answers; no solver and no measured data.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { APP_HEAD_HEIGHT, MOLAR_KG, UNITS, appCellIndex, decode, dryFraction, encode, headIndices, recorder, speciesSeries, speciesSites, volumeFraction } from "./species.mjs";

const centres = (h, n) => Array.from({ length: n }, (_, i) => (i + 0.5) * h);

test("the app's cell is the one that holds 1.8 m over the floor", () => {
  // 0.34 m: 1.8 / 0.34 = 5.29, cell 5 (1.70 to 2.04 m, centre 1.87 m).
  assert.equal(appCellIndex(centres(0.34, 8), 0.34), 5);
  // 0.27 m: 6.67, cell 6 (1.62 to 1.89 m). 0.20 m: cell 9 (1.8 to 2.0 m; 1.8 is its lower face).
  assert.equal(appCellIndex(centres(0.27, 9), 0.27), 6);
  assert.equal(appCellIndex(centres(0.2, 12), 0.2), 9);
  assert.equal(APP_HEAD_HEIGHT, 1.8);
});

test("a missing cell falls to the nearest kept one, below first, within two", () => {
  const z = centres(0.34, 8);
  const without = (drop) => z.filter((_, i) => !drop.includes(i));
  assert.equal(without([5])[appCellIndex(without([5]), 0.34)], z[4]);
  assert.equal(without([5, 4])[appCellIndex(without([5, 4]), 0.34)], z[6]);
  assert.equal(without([5, 4, 6])[appCellIndex(without([5, 4, 6]), 0.34)], z[3]);
  assert.equal(without([5, 4, 6, 3])[appCellIndex(without([5, 4, 6, 3]), 0.34)], z[7]);
  assert.equal(appCellIndex(without([3, 4, 5, 6, 7]), 0.34), -1);
  // A tree whose column starts above the floor (a pocket dropped below): heights still count from the floor.
  assert.equal(appCellIndex([1.87, 2.21], 0.34), 0);
});

test("the head-height cells are those series.head keeps", () => {
  const z = centres(0.34, 8);
  // 1.5 - 0.34 = 1.16 to 2.0 + 0.34 = 2.34: centres 1.19, 1.53, 1.87, 2.21.
  assert.deepEqual(headIndices(z, 0.34), [3, 4, 5, 6]);
  assert.deepEqual(headIndices(centres(0.2, 12), 0.2), [6, 7, 8, 9, 10]);
});

test("sites: trees, analysers and a meter's path share cells without repeating them", () => {
  const h = 0.34;
  const z = centres(h, 8);
  const trees = [{ id: "A", z, cells: z.map((_, i) => 100 + i) }, { id: "none", z: [], cells: [] }];
  const points = [
    { id: "O2", quantity: "o2", x: 1, y: 2, z: 1.84, cell: { c: 105, moved: 0 } },
    { id: "CJ", quantity: "ceilingJet", x: 1, y: 2, z: 2.4, cell: { c: 107, moved: 0 } },
    { id: "beam", quantity: "smoke", x: 0, y: 0, z: 3.6, path: [[0, 0, 3.6], [1, 0, 3.6]], cell: { c: 300, moved: 0 }, pathCells: [300, 301, 302] },
    { id: "CO", quantity: "co", x: 3, y: 3, z: 3.2, cell: { c: 400, moved: 1 }, stored: "for storing only" },
    { id: "lost", quantity: "co2", x: 9, y: 9, z: 9, cell: null },
  ];
  const { cells, sites } = speciesSites(trees, points, h);
  assert.deepEqual(cells, [103, 104, 105, 106, 300, 301, 302, 400]);
  assert.deepEqual(sites.trees.A.at, [0, 1, 2, 3]);
  assert.deepEqual(sites.trees.A.z, [1.19, 1.53, 1.87, 2.21]);
  assert.deepEqual(sites.trees.A.app, { height: 1.8, z: 1.87, at: 2, moved: 0 });
  assert.equal(sites.trees.none, undefined);
  assert.deepEqual(sites.points.O2.at, [2]);
  assert.deepEqual(sites.points.beam.at, [4, 5, 6]);
  assert.deepEqual(sites.points.beam.path, [[0, 0, 3.6], [1, 0, 3.6]]);
  assert.deepEqual(sites.points.CO.at, [7]);
  assert.equal(sites.points.CO.stored, "for storing only");
  assert.equal(sites.points.CO.moved, 1);
  assert.equal(sites.points.CJ, undefined);
  assert.equal(sites.points.lost, undefined);
});

test("the recorder reads the state's own units, and a volume fraction follows from them", () => {
  // The solver's species order and molar masses (flamescapes.js SPECIES, MOLAR).
  const SP = { FUEL: 0, O2: 1, CO2: 2, H2O: 3, SOOT: 4, CO: 5 }, NS = 6;
  const MOLAR = [0.03, 0.032, 0.044, 0.018, 0.012, 0.028];
  const N = 3;
  const state = { T: new Float32Array(N), rho: new Float32Array(N), Y: new Float32Array(N * NS) };
  // Cell 1: air. Cell 2: products with soot and CO.
  state.T.set([0, 293.15, 500]);
  state.rho.set([0, 1.2, 0.7]);
  state.Y[1 * NS + SP.O2] = 0.232;
  state.Y.set([0, 0.15, 0.06, 0.03, 2e-4, 1e-3], 2 * NS);
  const rec = recorder(state, [1, 2], { NS, SP, MOLAR });
  rec.read();
  state.T[2] = 510;
  rec.read();
  assert.deepEqual(rec.rows.T[1].map((x) => +x.toFixed(2)), [500, 510]);
  assert.ok(Math.abs(rec.rows.soot[1][0] - 0.7 * 2e-4) < 1e-9);
  assert.ok(Math.abs(rec.rows.co[1][0] - 1e-3) < 1e-9);
  assert.equal(rec.rows.soot[0][0], 0);
  // Air: X_O2 = (0.232 / 0.032) / (0.232 / 0.032 + 0.768 / 0.028014) = 0.20914.
  const xAir = volumeFraction(rec.rows.o2[0], rec.rows.mol[0], MOLAR_KG.o2)[0];
  assert.ok(Math.abs(xAir - 0.232 / 0.032 / (0.232 / 0.032 + 0.768 / 0.028014)) < 1e-6);
  // Products: mol by hand.
  const rest = 1 - (0.15 + 0.06 + 0.03 + 2e-4 + 1e-3);
  const mol = 0.15 / 0.032 + 0.06 / 0.044 + 0.03 / 0.018 + 2e-4 / 0.012 + 1e-3 / 0.028 + rest / 0.028014;
  assert.ok(Math.abs(rec.rows.mol[1][0] - mol) < 1e-4);
  assert.ok(Math.abs(volumeFraction(rec.rows.co[1], rec.rows.mol[1], MOLAR_KG.co)[0] - 1e-3 / 0.028 / mol) < 1e-7);
  // What a run keeps: every field of UNITS, one string a cell, and they decode to the readings.
  const kept = speciesSeries(rec, [1, 2], { trees: {}, points: {} });
  assert.deepEqual(Object.keys(UNITS), ["T", "soot", "co", "co2", "o2", "h2o", "mol"]);
  // Water, and the dry reading: X_dry = X / (1 - X_H2O).
  assert.ok(Math.abs(rec.rows.h2o[1][0] - 0.03) < 1e-8);
  const xW = volumeFraction(rec.rows.h2o[1], rec.rows.mol[1], MOLAR_KG.h2o);
  assert.ok(Math.abs(xW[0] - 0.03 / 0.018 / mol) < 1e-7);
  const xO2 = volumeFraction(rec.rows.o2[1], rec.rows.mol[1], MOLAR_KG.o2);
  assert.ok(Math.abs(dryFraction(xO2, xW)[0] - (0.15 / 0.032) / (mol - 0.03 / 0.018)) < 1e-6);
  assert.equal(dryFraction([0.2], [0])[0], 0.2);
  for (const f of Object.keys(UNITS)) assert.equal(kept[f].length, 2);
  assert.deepEqual(decode(kept.T[1]), [500, 510]);
  assert.ok(Math.abs(decode(kept.soot[1])[0] - 1.4e-4) < 1e-8);
  assert.deepEqual(kept.cells, [1, 2]);
});

test("encode and decode: rounding, empty series and readings that are not finite", () => {
  assert.equal(encode([293.154, 300], "T"), "293.15,300");
  assert.equal(encode([0.000123456789, 0], "soot"), "0.00012346,0");
  assert.equal(encode([1, NaN, 2], "co"), "1,,2");
  const back = decode("1,,2");
  assert.equal(back[0], 1);
  assert.ok(Number.isNaN(back[1]));
  assert.equal(back[2], 2);
  assert.deepEqual(decode(""), []);
  assert.deepEqual(decode(encode([], "T")), []);
});
