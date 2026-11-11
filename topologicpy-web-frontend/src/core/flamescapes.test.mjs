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
  POROUS,
  SPECIES,
  Y_O2_AIR,
  changeGas,
  createSolver,
  ducts,
  poolCap,
  exchange,
  gasDensity,
  hashUnit,
  particleStep,
  pcgHash,
  step,
  stoichiometry,
  survives,
} from "./flamescapes.js";
import { MATERIAL, gridFor, interiorCells, interiorFuel, occupancy, raysFor, volumeFractions } from "./voxelize.js";
import { CO_LIMIT, atriumTop, buildScene, coDoseRate, forecastFromFrames, runScene, smokeCloud, visibility } from "./flamescapesRun.js";
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

test("a 52 mm ceiling split evenly between two cells still seals, in one cell of each column", () => {
  // On a 0.4 m grid each half holds 6.5 %, under THIN_FRACTION (8 %).
  for (const offset of [0, 0.1, 0.174, 0.2, 0.3]) {
    const probe = gridFor([box(0, 0, 0, 3, 3, 3)], { cell: 0.4, margin: 1, maxCells: 1e6 });
    // The plate's middle on a cell face (offset 0.174 puts it there for this grid), or elsewhere.
    const y0 = probe.origin[1] + 6 * 0.4 - 0.026 + offset;
    const plate = box(0, y0, 0, 3, y0 + 0.052, 3);
    const grid = gridFor([plate, box(0, 0, 0, 3, 0.01, 3)], { cell: 0.4, margin: 1, maxCells: 1e6, bounds: { lo: probe.origin, hi: [probe.origin[0] + 6, probe.origin[1] + 6, probe.origin[2] + 6] } });
    const fraction = volumeFractions(grid, [plate], { rays: raysFor(grid) });
    const solid = occupancy(grid, fraction);
    for (let k = 3; k < grid.nz - 3; k += 1) {
      for (let i = 3; i < grid.nx - 3; i += 1) {
        let n = 0;
        for (let j = 0; j < grid.ny; j += 1) n += solid[i + grid.nx * (j + grid.ny * k)];
        assert.equal(n, 1, `offset ${offset}: column ${i},${k} has ${n} solid cells`);
      }
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

test("a suspended ceiling 5 cm thick is the fire grid's ceiling under a higher slab; the walls' coefficient acts on walls only", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  // The room with its slab at 4 m and a ceiling at 2.6 m (the NIBS models' 52 to 57 mm).
  const geo = room();
  geo.walls = geo.walls.map((w) => box(...[0, 1, 2].map((a) => w.positions[a]), w.positions[18], 4.0, w.positions[20]));
  geo.roofs = [box(-0.2, 4.0, -0.2, 5.2, 4.2, 4.2)];
  const ceilings = [box(0, 2.6, 0, 5, 2.655, 4)];
  for (const h of [0.25, 0.37]) {
    const scene = buildScene({ ...geo, ceilings }, { fire: [2.5, 0.75, 2], maxCells: 1e5, minCell: h, maxCell: h, contentsFuel, ignitionFuel, item });
    const g = scene.grid;
    const firstSolidAbove = (x, z) => {
      const i = Math.floor((x - g.origin[0]) / g.h), k = Math.floor((z - g.origin[2]) / g.h);
      for (let j = Math.floor((1.0 - g.origin[1]) / g.h); j < g.ny; j += 1) if (scene.solid[i + g.nx * (j + g.ny * k)]) return g.origin[1] + j * g.h;
      return Infinity;
    };
    for (const [x, z] of [[1, 1], [2.5, 2], [4, 3]]) {
      const y = firstSolidAbove(x, z);
      assert.ok(Math.abs(y - 2.6) <= g.h, `cell ${h} m at (${x}, ${z}): the gas meets a solid at ${y.toFixed(2)} m, not the slab at 4 m`);
    }
  }
  // A sealed box of hot gas with no fuel: the walls take less of its heat at
  // a lower walls' coefficient; the fuel's coefficient changes nothing.
  const hot = (opts) => {
    const shell = [box(-0.3, -0.3, -0.3, 2.3, 0, 2.3), box(-0.3, 2, -0.3, 2.3, 2.3, 2.3), box(-0.3, 0, -0.3, 0, 2, 2.3), box(2, 0, -0.3, 2.3, 2, 2.3), box(0, 0, -0.3, 2, 2, 0), box(0, 0, 2, 2, 2, 2.3)];
    const grid = gridFor(shell, { cell: 0.25, margin: 1, maxCells: 1e5 });
    const fraction = volumeFractions(grid, shell, { rays: raysFor(grid) });
    const scene = { grid, fraction, solid: occupancy(grid, fraction), contents: { cells: [], masses: [] }, ignition: null, pilot: null, fuels: { contentsFuel, ignitionFuel } };
    const sim = createSolver(scene, { contentsFuel, ignitionFuel, ...opts });
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === 0) sim.T[c] = 400;
    for (let s = 0; s < 20; s += 1) step(sim);
    let e = 0;
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === 0) e += sim.T[c];
    return e;
  };
  const at200 = hot({ wallConvection: 200 }), at25 = hot({ wallConvection: 25 });
  assert.ok(at25 > at200, "the gas keeps more heat with the lower walls' coefficient");
  assert.equal(hot({ wallConvection: 25, convection: 50 }), at25, "the fuel's coefficient is not the walls'");
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

test("the smoke cloud: the opacity of its soot across a block, none before the fire, a cloud after it", () => {
  // One block of 2 x 2 x 2 cells of 0.5 m with soot density 1.2e-4 kg/m3:
  // 1 - exp(-8700 x 1 m x 1.2e-4) = 0.648; a solid cell's soot is not counted.
  const grid = { nx: 2, ny: 2, nz: 2, h: 0.5, origin: [10, 0, -3], axes: [0, 1, 2] };
  const rho = new Float32Array(8).fill(1.2);
  const Y = new Float32Array(8 * NS);
  for (let c = 0; c < 8; c += 1) Y[c * NS + SPECIES.SOOT] = 1e-4;
  const kind = new Uint8Array(8);
  kind[7] = 2;
  Y[7 * NS + SPECIES.SOOT] = 1;
  const one = smokeCloud(grid, rho, Y, kind);
  assert.equal(one.opacity.length, 1);
  assert.equal(one.opacity[0], Math.round((1 - Math.exp(-8700 * 1 * 1.2e-4)) * 255));
  assert.deepEqual([...one.positions], [10.5, 0.5, -2.5]);
  assert.equal(one.size, 1);

  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.3, maxCell: 0.3, contentsFuel, ignitionFuel, item });
  const nodes = Float32Array.from([2.5, 0.75, 2]);
  const clouds = [];
  runScene(scene, { nodes, horizon: 30, frameSeconds: 5, particles: 0, smoke: true, keep: false, onFrame: (f) => clouds.push(f.smoke) });
  const fresh = smokeCloud(scene.grid, new Float32Array(scene.grid.nx * scene.grid.ny * scene.grid.nz).fill(1.2), new Float32Array(scene.grid.nx * scene.grid.ny * scene.grid.nz * NS), new Uint8Array(scene.grid.nx * scene.grid.ny * scene.grid.nz));
  assert.equal(fresh.opacity.length, 0, "clean air draws nothing");
  const last = clouds.at(-1);
  assert.ok(last.opacity.length > 0, "the fire makes smoke");
  for (const x of last.positions) assert.ok(Number.isFinite(x));
  // Without the option a frame carries no cloud.
  const plain = [];
  runScene(buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.3, maxCell: 0.3, contentsFuel, ignitionFuel, item }), { nodes, horizon: 5, frameSeconds: 5, particles: 0, keep: false, onFrame: (f) => plain.push(f.smoke) });
  assert.equal(plain[0], null);
});

test("a closed door keeps the smoke in: outside the room it stays clearer than with the door open", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  // The room's doorway is x = 5, z 1.5 to 2.4, up to 2 m; its leaf fills it.
  const leaf = box(5, 0, 1.5, 5.2, 2.0, 2.4);
  const nodes = Float32Array.from([2.5, 0.75, 2]);
  const outside = {};
  for (const closed of [false, true]) {
    const geo = { ...room(), ...(closed ? { doors: [leaf] } : {}) };
    const scene = buildScene(geo, { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.3, maxCell: 0.3, contentsFuel, ignitionFuel, item });
    let cloud = null;
    const out = runScene(scene, { nodes, horizon: 40, frameSeconds: 10, particles: 0, smoke: true, onFrame: (f) => (cloud = f.smoke) });
    for (const a of [out.T, out.vis, out.co]) for (const x of a) assert.ok(Number.isFinite(x));
    // The smoke's opacity summed over the cloud's blocks beyond the doorway's wall.
    let sum = 0;
    for (let q = 0; q < cloud.opacity.length; q += 1) if (cloud.positions[q * 3] > 5.4) sum += cloud.opacity[q];
    outside[closed] = sum;
  }
  assert.ok(outside[true] < 0.5 * outside[false], `smoke outside: ${outside[true]} with the door closed, ${outside[false]} open`);
});

test("running ventilation: the ducts conserve each species and the heat, and carry smoke to a sealed room", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  // The fire room, and a sealed room beside it (x 6 to 9) joined to it only by the ducts.
  const sealed = [box(5.8, 0, -0.2, 9.2, 2.7, 0), box(5.8, 0, 4, 9.2, 2.7, 4.2), box(5.8, 0, 0, 6, 2.7, 4), box(9, 0, 0, 9.2, 2.7, 4)];
  const geometry = () => {
    const g = room();
    return { ...g, walls: [...g.walls, ...sealed], floors: [...g.floors, box(5.8, -0.2, -0.2, 9.2, 0, 4.2)], roofs: [...g.roofs, box(5.8, 2.7, -0.2, 9.2, 2.9, 4.2)] };
  };
  const terminals = [[2.5, 2.6, 2], [7.5, 2.6, 2]];
  const nodes = Float32Array.from([7.5, 0.75, 2]);
  const options = (ventilation) => ({ fire: [2.5, 0.75, 2], maxCells: 30000, minCell: 0.3, maxCell: 0.3, contentsFuel, ignitionFuel, item, ventilation });

  // One exchange conserves sum m Y_i and sum m T over the terminals' cells.
  const scene = buildScene(geometry(), options({ terminals, flow: 0.1 }));
  assert.equal(scene.ducts.cells.length, 2, "both terminals land on open cells");
  const sim = createSolver(scene, { contentsFuel, ignitionFuel });
  const [a, b] = scene.ducts.cells;
  sim.T[a] = 600;
  sim.Y[a * NS + SPECIES.SOOT] = 0.01;
  const totals = () => {
    let heat = 0;
    const y = new Array(NS).fill(0);
    for (const c of [a, b]) {
      const m = sim.rho[c] * sim.V;
      heat += m * sim.T[c];
      for (let i = 0; i < NS; i += 1) y[i] += m * sim.Y[c * NS + i];
    }
    return { heat, y };
  };
  const before = totals();
  ducts(sim, 0.1);
  const after = totals();
  assert.ok(Math.abs(after.heat - before.heat) <= 1e-5 * before.heat, `heat ${before.heat} -> ${after.heat}`);
  for (let i = 0; i < NS; i += 1) assert.ok(Math.abs(after.y[i] - before.y[i]) <= 1e-6 * Math.max(1e-12, Math.abs(before.y[i])) + 1e-12);
  assert.ok(sim.Y[b * NS + SPECIES.SOOT] > 0, "soot reached the other terminal");

  // Over a fire: the sealed room stays clear with the ventilation off, and is reached with it running.
  const least = {};
  for (const on of [false, true]) {
    const out = runScene(buildScene(geometry(), options(on ? { terminals, flow: 0.1 } : null)), { nodes, horizon: 60, frameSeconds: 10, particles: 0 });
    for (const x of out.vis) assert.ok(Number.isFinite(x));
    least[on] = Math.min(...out.vis);
  }
  assert.equal(least[false], 100, "no smoke in the sealed room without ventilation");
  assert.ok(least[true] < 100, `the ducts bring smoke: visibility ${least[true]} m`);
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

test("regressions found by the Lean proofs: exact fire load, and yields that would create mass", () => {
  // interiorFuel cut negative draws at zero and so held more than M; it is
  // rescaled to exactly the budget (FlamescapesPort.lean).
  const geo = room();
  const meshes = [...geo.walls, ...geo.floors, ...geo.roofs];
  const grid = gridFor(meshes, { cell: 0.25, margin: 2, maxCells: 1e6 });
  const fraction = volumeFractions(grid, meshes, { rays: raysFor(grid) });
  const cells = interiorCells(grid, fraction, { layers: 2 });
  const fuel = interiorFuel(grid, cells, { fireLoad: 780e6, heatOfCombustion: 18e6, spread: 0.9, layers: 2 });
  let sum = 0;
  for (const m of fuel.masses) sum += m;
  assert.ok(Math.abs(sum - fuel.mass) / fuel.mass < 1e-5);
  // Products heavier than the fuel and its oxygen are refused.
  assert.throws(() => stoichiometry({ heatOfCombustion: 13.03e6, yields: { co2: 3, co: 0, soot: 0 } }));
});

test("regressions from the architecture review: walls are not a source of air, and clipping keeps crossing faces", () => {
  // A sealed room (no door) with its gas at 10% oxygen: advection must not
  // draw fresh air out of the walls, floor or ceiling.
  const t = 0.3, W = 5, D = 4, H = 2.7;
  const sealed = {
    walls: [box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, D)],
    floors: [box(-t, -t, -t, W + t, 0, D + t)],
    roofs: [box(-t, H, -t, W + t, H + t, D + t)],
    stairs: [],
    obstacles: [],
  };
  const scene = buildScene(sealed, { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.25, maxCell: 0.25, contentsFuel: FIRE_CLASSES.A.fuel });
  const sim = createSolver({ ...scene, contents: { cells: [], masses: [] }, ignition: null, pilot: null }, { contentsFuel: FIRE_CLASSES.A.fuel });
  const inside = [];
  for (let c = 0; c < sim.N; c += 1) {
    const g = sim.grid, i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
    const [x, y, z] = [g.origin[0] + (i + 0.5) * g.h, g.origin[1] + (j + 0.5) * g.h, g.origin[2] + (k + 0.5) * g.h];
    if (sim.kind[c] === 0 && x > 0 && x < W && z > 0 && z < D && y > 0 && y < H) inside.push(c);
  }
  for (const c of inside) sim.Y[c * NS + SPECIES.O2] = 0.1;
  for (let s = 0; s < 50; s += 1) step(sim);
  let max = 0;
  for (const c of inside) max = Math.max(max, sim.Y[c * NS + SPECIES.O2]);
  assert.ok(max < 0.1 + 1e-4, `oxygen rose to ${max}`);

  // A 30 m single storey clipped round the fire still has its walls and slab.
  const L = 30;
  const big = {
    walls: [box(-t, 0, -t, L + t, 3, 0), box(-t, 0, L, L + t, 3, L + t), box(-t, 0, 0, 0, 3, L), box(L, 0, 0, L + t, 3, L)],
    floors: [box(-t, -t, -t, L + t, 0, L + t)],
    roofs: [box(-t, 3, -t, L + t, 3 + t, L + t)],
    stairs: [],
    obstacles: [],
  };
  const clipped = buildScene(big, { fire: [15, 0.75, 15], maxCells: 30000, contentsFuel: FIRE_CLASSES.A.fuel });
  assert.ok(clipped.clipped, "the 30 m storey is clipped at the CPU budget");
  let solid = 0;
  for (const v of clipped.solid) solid += v;
  assert.ok(solid > 1000 && clipped.ignition, `${solid} solid cells`);

  // Under an atrium (two voids stacked over the fire to 12 m, the upper
  // storeys' walls round it) the clipped box reaches its top; a fire away
  // from it keeps the box as it was.
  const tall = { ...big, walls: [...big.walls, box(9.8, 0, 10, 10, 12, 20)] };
  const voids = [box(10, 3, 10, 20, 7.5, 20), box(10, 7.5, 10, 20, 12, 20)];
  const top = (scene) => scene.grid.origin[1] + scene.grid.ny * scene.grid.h;
  const options = (fire) => ({ fire, maxCells: 30000, contentsFuel: FIRE_CLASSES.A.fuel });
  const atrium = buildScene({ ...tall, voids }, options([15, 0.75, 15]));
  const without = buildScene(tall, options([15, 0.75, 15]));
  assert.ok(atrium.clipped && top(atrium) >= 12, `the grid's top ${top(atrium).toFixed(2)} m`);
  assert.ok(top(without) < 10, `without the voids ${top(without).toFixed(2)} m (the box's 6.75 m and the grid's margin)`);
  assert.ok(atrium.grid.h > without.grid.h, "the cells coarsen to keep the count");
  assert.equal(top(buildScene({ ...tall, voids }, options([4, 0.75, 4]))), top(buildScene(tall, options([4, 0.75, 4]))), "a fire away from the atrium");
});

test("atriumTop: the voids stacked over a fire, each reached from the one under it", () => {
  const axes = [0, 1, 2];
  const voids = [box(0, 4.5, 0, 6, 9, 6), box(0, 9, 0, 6, 13.5, 6), box(0, 20, 0, 6, 24, 6)];
  assert.equal(atriumTop(voids, axes, [3, 0.75, 3], 6.75), 13.5, "the third void is not reached: there is a storey between");
  assert.equal(atriumTop(voids, axes, [9, 0.75, 9], 6.75), 6.75, "a fire beyond reach of the voids' plan");
  assert.equal(atriumTop(null, axes, [3, 0.75, 3], 6.75), 6.75);
});

test("regressions from the GPU port: only real combustion spawns flame particles; a massless contents cell exchanges nothing", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.3, maxCell: 0.3, contentsFuel, ignitionFuel, item });
  // The Arrhenius oxidation never stops: at ambient every contents cell
  // reacted a little and counted as burning, spawning six particles a step.
  const sim = createSolver(scene, { contentsFuel, ignitionFuel, seed: 1 });
  let contents = 0;
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === POROUS && sim.fuelMass[c] > 0) contents += 1;
  assert.ok(contents > 50, `${contents} contents cells`);
  for (let k = 0; k < 5; k += 1) step(sim);
  const burning = sim.burning.reduce((a, b) => a + b, 0);
  assert.ok(burning < contents / 5, `${burning} burning of ${contents} contents cells`);
  assert.ok(sim.parts.n <= 6 * 5 * Math.max(1, burning), `${sim.parts.n} particles`);
  // A contents cell drawn with no mass has no heat capacity: conduction into
  // it was 0 / 0 and left NaN in the solid temperature.
  const empty = createSolver(scene, { contentsFuel, ignitionFuel, seed: 1 });
  const { nx, ny } = empty.grid;
  let zeroed = 0;
  for (let c = 0; c < empty.N; c += 1) {
    if (empty.kind[c] !== POROUS) continue;
    const below = c - nx * ny, west = c - 1;
    const solidBefore = [below, west].some((n) => n >= 0 && empty.kind[n] !== 0);
    if (!solidBefore) continue;
    empty.fuelMass[c] = 0;
    empty.moisture[c] = 0;
    empty.inertC[c] = 0;
    zeroed += 1;
  }
  assert.ok(zeroed > 0);
  step(empty);
  for (let c = 0; c < empty.N; c += 1) assert.ok(Number.isFinite(empty.Ts[c]), `Ts[${c}] = ${empty.Ts[c]}`);
});

test("every fire class burns: liquid pools give off at most their burning rate", () => {
  // n-heptane, 1 m^2: m''_inf (1 - exp(-k beta D)) A dt with D = sqrt(4 / pi).
  const heptane = FIRE_CLASSES.B.fuel.liquid;
  const expected = 0.101 * (1 - Math.exp(-1.1 * Math.sqrt(4 / Math.PI))) * 1 * 0.1;
  assert.ok(Math.abs(poolCap(heptane, 1, 0.1) - expected) < 1e-12);
  assert.equal(poolCap({ boiling: 300, latent: 1e5 }, 1, 0.1), Infinity);
  assert.ok(Math.abs(poolCap({ mInf: 0.017 }, 2, 0.1) - 0.017 * 2 * 0.1) < 1e-12, "no k beta: no diameter dependence");
  // Without the cap the pilot boiled a liquid pool off in a few steps into
  // one cell, too rich to burn: classes B and K released nothing.
  for (const cls of FIRE_CLASS_KEYS) {
    const { contentsFuel, ignitionFuel, item } = fuelsFor(cls);
    const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.3, maxCell: 0.3, contentsFuel, ignitionFuel, item });
    const out = runScene(scene, { nodes: Float32Array.from([2.5, 0.75, 2]), horizon: 5, frameSeconds: 5, particles: 0 });
    assert.ok(out.stats.released > 1e5, `class ${cls}: released ${out.stats.released} J`);
  }
});

test("radiation emits from the step's starting temperatures: the heat radiated does not hang on the rays", () => {
  // A block of hot, sooty gas: rays from one cell are nearly all absorbed in
  // its neighbours (kappa Y_soot h = 500 x 0.02 x 0.4 = 4). Added as each ray
  // went, a neighbour later in the loop re-emitted some of it, by where the
  // random rays had gone (the CPU before 2026-11-09; the GPU never did).
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 4000, minCell: 0.4, contentsFuel: FIRE_CLASSES.A.fuel });
  const radiated = (seed) => {
    const sim = createSolver({ ...scene, contents: { cells: [], masses: [] }, ignition: null, pilot: null }, { contentsFuel: FIRE_CLASSES.A.fuel, maxParticles: 0, seed });
    const g = sim.grid;
    const ci = Math.floor((2.5 - g.origin[0]) / g.h), cj = Math.floor((1.5 - g.origin[1]) / g.h), ck = Math.floor((2 - g.origin[2]) / g.h);
    let hot = 0;
    for (let k = ck - 1; k <= ck + 1; k += 1) for (let j = cj - 1; j <= cj + 1; j += 1) for (let i = ci - 1; i <= ci + 1; i += 1) {
      const c = i + g.nx * (j + g.ny * k);
      if (sim.kind[c] !== 0) continue;
      sim.T[c] = 1300;
      sim.Y[c * NS + SPECIES.SOOT] = 0.02;
      sim.rho[c] = gasDensity(sim.T[c], sim.Y, c * NS);
      hot += 1;
    }
    assert.equal(hot, 27);
    step(sim);
    return sim.stats.radiated;
  };
  const a = radiated(1), b = radiated(2);
  assert.ok(a > 0);
  assert.ok(Math.abs(a - b) <= 1e-9 * a, `radiated ${a} J with one seed, ${b} J with another`);
});

test("pcgHash is the GPU's u32 PCG hash: checked against BigInt arithmetic", () => {
  const M = 2n ** 32n;
  const ref = (v) => {
    const st = (BigInt(v) * 747796405n + 2891336453n) % M;
    const word = (((st >> ((st >> 28n) + 4n)) ^ st) * 277803737n) % M;
    return Number((word >> 22n) ^ word);
  };
  for (const v of [0, 1, 2, 5, 1234567, 2 ** 31, 2 ** 32 - 1, 0xdeadbeef]) assert.equal(pcgHash(v), ref(v), `v = ${v}`);
  // hashUnit is the first draw of a cell's stream: in [0, 1) and spread evenly.
  let low = 0;
  for (let c = 0; c < 4000; c += 1) {
    const u = hashUnit(c, 17, 5, 1);
    assert.ok(u >= 0 && u < 1);
    if (u < 0.5) low += 1;
  }
  assert.ok(low > 1850 && low < 2150, `${low} of 4000 below 0.5`);
});

test("flame particles that would overfill their share spawn from burning cells spread over the room, not the first in order", () => {
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 4000, minCell: 0.4, contentsFuel: FIRE_CLASSES.A.fuel });
  const sim = createSolver({ ...scene, contents: { cells: [], masses: [] }, ignition: null, pilot: null }, { contentsFuel: FIRE_CLASSES.A.fuel, maxParticles: 400 });
  // Every open cell burning, labelled by its rank in cell order through its temperature.
  const burning = [];
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === 0) burning.push(c);
  burning.forEach((c, r) => {
    sim.burning[c] = 1;
    sim.T[c] = 400 + r;
  });
  particleStep(sim, 0.1);
  const space = Math.ceil(sim.P.flameShare * sim.P.maxParticles);
  assert.ok(6 * burning.length > 4 * space, `${burning.length} burning cells against room for ${space}`);
  assert.ok(sim.parts.n > 0.7 * space && sim.parts.n <= space, `${sim.parts.n} particles, room for ${space}`);
  let late = 0;
  for (let q = 0; q < sim.parts.n; q += 1) if (sim.parts.T[q] - 400 >= burning.length / 2) late += 1;
  // Refused in order, every particle came from the first cells; shared, about half from the later half.
  assert.ok(late > 0.3 * sim.parts.n && late < 0.7 * sim.parts.n, `${late} of ${sim.parts.n} from the later half of the cells`);
});
