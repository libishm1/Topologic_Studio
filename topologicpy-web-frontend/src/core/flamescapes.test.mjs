/**
 * The Flamescapes port (flamescapes.js, voxelize.js, flamescapesRun.js,
 * fireClasses.js): its exact integrators, bookkeeping, voxelisation and a
 * small room fire with and without embers.
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  HEAT_PER_O2,
  NS,
  SPECIES,
  Y_O2_AIR,
  changeGas,
  createSolver,
  exchange,
  gasDensity,
  step,
  stoichiometry,
  survives,
} from "./flamescapes.js";
import { MATERIAL, gridFor, interiorCells, interiorFuel, occupancy, raysFor, volumeFractions } from "./voxelize.js";
import { CO_LIMIT, buildScene, coDoseRate, forecastFromFrames, runScene, visibility } from "./flamescapesRun.js";
import { FIRE_CLASSES, FIRE_CLASS_KEYS, fuelsFor } from "./fireClasses.js";

function box(x0, y0, z0, x1, y1, z1, material = MATERIAL.STONE) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material };
}

/** A 5 x 4 x 2.7 m room with a door in one wall, y up. */
function room(t = 0.2) {
  const W = 5, D = 4, H = 2.7;
  return {
    walls: [
      box(-t, 0, -t, W + t, H, 0),
      box(-t, 0, D, W + t, H, D + t),
      box(-t, 0, 0, 0, H, D),
      box(W, 0, 0, W + t, H, 1.5),
      box(W, 0, 2.4, W + t, H, D),
      box(W, 2.0, 1.5, W + t, H, 2.4),
    ],
    floors: [box(-t, -t, -t, W + t, 0, D + t)],
    roofs: [box(-t, H, -t, W + t, H + t, D + t)],
    stairs: [],
    obstacles: [],
  };
}

test("survives is the exact first-order decay: in [0, 1], and two steps compose", () => {
  for (const k of [0, 0.1, 3, 50]) {
    for (const dt of [0.01, 0.1, 1]) {
      const s = survives(k, dt);
      assert.ok(s >= 0 && s <= 1);
      assert.ok(Math.abs(survives(k, dt) * survives(k, dt) - survives(k, 2 * dt)) < 1e-12);
    }
  }
  // Where explicit Euler would go negative (k dt > 1), the exponential does not.
  assert.ok(1 - 50 * 0.1 < 0 && survives(50, 0.1) > 0);
});

test("exchange conserves energy, never overshoots, and has the sign of the difference", () => {
  const cases = [
    [1000, 300, 50, 2000, 10, 0.1],
    [300, 1000, 2000, 50, 300, 1],
    [500, 500, 10, 10, 5, 10],
    [900, 293, 1e3, 1e6, 1e4, 100],
  ];
  for (const [Ta, Tb, Ca, Cb, G, dt] of cases) {
    const q = exchange(Ta, Tb, Ca, Cb, G, dt);
    const a = Ta - q / Ca, b = Tb + q / Cb;
    assert.ok(Math.abs(Ca * Ta + Cb * Tb - (Ca * a + Cb * b)) < 1e-6 * (Ca * Ta + Cb * Tb));
    if (Ta >= Tb) assert.ok(a >= b - 1e-9 && q >= 0);
    else assert.ok(a <= b + 1e-9 && q <= 0);
  }
  // For a small step it is the explicit flux G (Ta - Tb) dt of Eq. 19.
  const q = exchange(800, 300, 1e4, 1e4, 20, 1e-3);
  assert.ok(Math.abs(q - 20 * 500 * 1e-3) / (20 * 500 * 1e-3) < 1e-3);
});

test("the equation of state gives air at about 1.2 kg/m3 and density falls with temperature", () => {
  const Y = new Float32Array(NS);
  Y[SPECIES.O2] = Y_O2_AIR;
  const rho = gasDensity(293.15, Y);
  assert.ok(rho > 1.17 && rho < 1.23, String(rho));
  assert.ok(gasDensity(600, Y) < rho && gasDensity(1200, Y) < gasDensity(600, Y));
});

test("stoichiometry closes the mass balance for every fire class", () => {
  for (const key of FIRE_CLASS_KEYS) {
    const st = stoichiometry(FIRE_CLASSES[key].fuel);
    const products = st.co2 + st.co + st.soot + st.h2o + st.ash;
    assert.ok(Math.abs(products - (1 + st.o2)) < 1e-9, key);
  }
  // Organic fuels consume 417 kJ / mol of O2 (Fire-X); a metal its own oxygen.
  assert.ok(Math.abs(stoichiometry(FIRE_CLASSES.A.fuel).o2 - FIRE_CLASSES.A.fuel.heatOfCombustion / HEAT_PER_O2) < 1e-12);
  assert.equal(stoichiometry(FIRE_CLASSES.D.fuel).o2, 0.658);
});

test("changeGas adds exactly the species deltas to a cell's mass", () => {
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 4000, minCell: 0.4, contentsFuel: FIRE_CLASSES.A.fuel });
  const sim = createSolver(scene, { contentsFuel: FIRE_CLASSES.A.fuel });
  const c = scene.contents.cells[0];
  const before = sim.rho[c] * sim.V;
  const delta = new Float64Array(NS);
  delta[SPECIES.CO2] = 0.002;
  delta[SPECIES.O2] = -0.0005;
  delta[SPECIES.SOOT] = 0.0001;
  changeGas(sim, c, delta);
  const after = sim.rho[c] * sim.V;
  // Densities are float32: within their precision.
  assert.ok(Math.abs(after - before - 0.0016) < 1e-6 * before);
  let sum = 0;
  for (let q = 0; q < NS; q += 1) sum += sim.Y[c * NS + q];
  assert.ok(sum <= 1 + 1e-6);
});

test("volume fractions of a box add up to its volume", () => {
  const mesh = box(0.1, 0.1, 0.1, 1.3, 0.9, 0.7);
  const grid = gridFor([mesh], { cell: 0.2, margin: 1 });
  const fraction = volumeFractions(grid, [mesh], { rays: raysFor(grid) });
  let volume = 0;
  for (let c = 0; c < grid.cells; c += 1) volume += fraction[c * 5 + MATERIAL.STONE] * grid.h ** 3;
  assert.ok(Math.abs(volume - 1.2 * 0.8 * 0.6) / (1.2 * 0.8 * 0.6) < 0.03, String(volume));
});

test("a slab thinner than a cell still seals: every column through it has a solid cell", () => {
  // 0.2 m slab on a 0.25 m grid placed to straddle two cells.
  const slab = box(0, 1.03, 0, 3, 1.23, 3);
  const grid = gridFor([slab], { cell: 0.25, margin: 1, maxCells: 1e6 });
  const fraction = volumeFractions(grid, [slab], { rays: raysFor(grid) });
  const solid = occupancy(grid, fraction);
  for (let k = 2; k < grid.nz - 2; k += 1) {
    for (let i = 2; i < grid.nx - 2; i += 1) {
      let any = 0;
      for (let j = 0; j < grid.ny; j += 1) any += solid[i + grid.nx * (j + grid.ny * k)];
      assert.ok(any > 0, `column ${i},${k} leaks`);
    }
  }
});

test("the room contents carry the fire load: M = q A / heat of combustion", () => {
  const geo = room();
  const meshes = [...geo.walls, ...geo.floors, ...geo.roofs];
  const grid = gridFor(meshes, { cell: 0.25, margin: 2, maxCells: 1e6 });
  const fraction = volumeFractions(grid, meshes, { rays: raysFor(grid) });
  const cells = interiorCells(grid, fraction, { layers: 2 });
  const fuel = interiorFuel(grid, cells, { fireLoad: 780e6, heatOfCombustion: 17.5e6, layers: 2 });
  // The floor of the room is 20 m2; the grid sees it to within a cell's rim.
  assert.ok(fuel.area > 16 && fuel.area < 24, String(fuel.area));
  assert.ok(Math.abs(fuel.total - fuel.mass) / fuel.mass < 0.05);
  assert.ok(Math.abs(fuel.mass - (780e6 * fuel.area) / 17.5e6) < 1e-6 * fuel.mass);
});

test("visibility falls with soot, and the CO dose reaches 1 at 35,000 ppm min", () => {
  assert.equal(visibility(0), 100);
  assert.ok(visibility(1e-4) > visibility(1e-3));
  assert.ok(Math.abs(coDoseRate(CO_LIMIT) * 60 - 1) < 1e-12);
});

test("a room fire with its first item alight heats and smokes the room; embers only when on", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const geo = room();
  const make = () => buildScene(geo, { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.3, maxCell: 0.3, contentsFuel, ignitionFuel, item });
  const nodes = Float32Array.from([2.5, 0.75, 2, 1, 0.75, 1, 4.5, 0.75, 3.5, 7, 0.75, 2]);
  const runs = {};
  for (const embers of [false, true]) {
    const scene = make();
    assert.ok(scene.ignition, "the first item is placed");
    const out = runScene(scene, { nodes, horizon: 40, frameSeconds: 5, embers, particles: 0 });
    for (const a of [out.T, out.vis, out.co]) for (const x of a) assert.ok(Number.isFinite(x));
    runs[embers] = out;
  }
  const off = runs[false];
  const K = off.K, n = off.n;
  assert.ok(Math.max(...off.hrr) > 50, `peak HRR ${Math.max(...off.hrr)} kW`);
  // Inside the room it gets hot or smoky at head height; outside the door less so.
  const last = (a, v) => a[(K - 1) * n + v];
  assert.ok(last(off.T, 1) > 30 || last(off.vis, 1) < 30, "the room is affected");
  assert.equal(off.stats.embersSpawned, 0);
  assert.ok(runs[true].stats.embersSpawned > 0, "embers are thrown when on");
  // The forecast built from the frames never lets a dose rate fall.
  const forecast = forecastFromFrames(off, { threshold: 60, smoke: true, visibilityLimit: 10 });
  for (let v = 0; v < n; v += 1) {
    for (let k = 1; k < forecast.K; k += 1) assert.ok(forecast.rate[k * n + v] >= forecast.rate[(k - 1) * n + v]);
  }
});

test("fire classes carry what the solver and the panel need", () => {
  for (const key of FIRE_CLASS_KEYS) {
    const c = FIRE_CLASSES[key];
    assert.ok(c.label && c.iso && c.examples);
    assert.ok(c.fuel.heatOfCombustion > 0 && c.fuel.yields);
    assert.ok(c.item.mass > 0 && c.item.area > 0);
    assert.ok(c.agents.suitable.length > 0);
  }
  assert.equal(fuelsFor("nope").ignitionFuel, FIRE_CLASSES.A.fuel);
  assert.equal(fuelsFor("K").contentsFuel, FIRE_CLASSES.A.fuel);
});
