/**
 * The prescribed burner of the CPU solver (flamescapes.js createBurner,
 * burnerStep; tools/sprinkler-validation.mjs): it releases its table's
 * integral, puts exactly the convective part into its cells' gas with the
 * products of the matching fuel, and the frame heat release that the
 * sprinklers' estimate reads (runScene) includes it.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { HEAT_PER_O2, METHANE, NS, SPECIES, burnerCells, burnerEnergy, burnerHeat, burnerStep, createBurner, createSolver, flameHeight, mixtureCp, plumeRise } from "./flamescapes.js";
import { MATERIAL, gridFor, occupancy, raysFor, volumeFractions } from "./voxelize.js";
import { runScene } from "./flamescapesRun.js";

function box(x0, y0, z0, x1, y1, z1, material = MATERIAL.STONE) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material };
}

const FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

/** A W x D x H room, y up, cells of h, walls one cell thick, a door in the x = W wall when `door`; the burner's cells over (bx, bz). */
function roomScene({ W = 4, D = 3, H = 2.4, h = 0.3, door = true, burnerAt = [2, 1.5], layers = 2, hrr, radiantFraction, fuel } = {}) {
  const t = h;
  const meshes = [
    box(-t, -t, -t, W + t, 0, D + t),
    box(-t, H, -t, W + t, H + t, D + t),
    box(-t, 0, -t, W + t, H, 0),
    box(-t, 0, D, W + t, H, D + t),
    box(-t, 0, 0, 0, H, D),
    ...(door ? [box(W, 0, 0, W + t, H, 1.0), box(W, 0, 2.0, W + t, H, D), box(W, 2.1, 1.0, W + t, H, 2.0)] : [box(W, 0, 0, W + t, H, D)]),
  ];
  const grid = gridFor(meshes, { upAxis: "y", cell: h, margin: 2, headroom: 0, maxCells: 1e6 });
  const fraction = volumeFractions(grid, meshes, { rays: raysFor(grid) });
  const solid = occupancy(grid, fraction);
  const at = (a, b, c) => Math.floor((a - grid.origin[0]) / h) + grid.nx * (Math.floor((b - grid.origin[1]) / h) + grid.ny * Math.floor((c - grid.origin[2]) / h));
  const cells = [];
  for (const dx of [-h / 2, h / 2]) for (const dz of [-h / 2, h / 2]) for (let l = 0; l < layers; l += 1) cells.push(at(burnerAt[0] + dx, (l + 0.5) * h, burnerAt[1] + dz));
  return {
    grid,
    fraction,
    solid,
    contents: { cells: [], masses: [] },
    ignition: null,
    pilot: null,
    burner: { cells, hrr, radiantFraction, fuel },
    fuels: { contentsFuel: FUEL, ignitionFuel: FUEL },
  };
}

test("burnerEnergy is the exact integral of the linear table, held at its ends", () => {
  const b = createBurner({ cells: [0], hrr: [[0, 0], [10, 1000], [20, 1000], [30, 0]] }, new Uint8Array(1), { nx: 1, ny: 1, nz: 1 });
  assert.equal(burnerEnergy(b, 0), 0);
  assert.equal(burnerEnergy(b, 10), 5000);
  assert.ok(Math.abs(burnerEnergy(b, 5) - 1250) < 1e-9); // 0.5 * 500 W * 5 s
  assert.equal(burnerEnergy(b, 20), 15000);
  assert.equal(burnerEnergy(b, 30), 20000);
  assert.equal(burnerEnergy(b, 40), 20000); // held at 0 W after the last row
  assert.ok(burnerEnergy(b, -5) === 0); // held at 0 W before the first (-0 counts)
  const held = createBurner({ cells: [0], hrr: [[0, 0], [10, 1000]] }, new Uint8Array(1), { nx: 1, ny: 1, nz: 1 });
  assert.equal(burnerEnergy(held, 20) - burnerEnergy(held, 10), 10000); // the last value holds
});

test("burnerHeat: Evans's decay once water reaches the item, never above its table, and nothing once clean agent puts it out; the flame's layers follow Heskestad", () => {
  // 100 W more each second: 150 kJ from 10 to 20 s by the table.
  const b = createBurner({ cells: [0], hrr: [[0, 0], [100, 10000]] }, new Uint8Array(1), { nx: 1, ny: 1, nz: 1, h: 1 });
  assert.ok(Math.abs(burnerHeat(b, 10, 20) - 15000) < 1e-9);
  b.suppression = { start: 10, tau: 5, q: 1000 };
  assert.ok(Math.abs(burnerHeat(b, 10, 20) - 1000 * 5 * (1 - Math.exp(-2))) < 1e-9, "q tau (1 - e^-2)");
  assert.ok(Math.abs(burnerHeat(b, 5, 15) - (burnerEnergy(b, 10) - burnerEnergy(b, 5)) - 1000 * 5 * (1 - Math.exp(-1))) < 1e-9, "the table up to the start");
  b.suppression = { start: 10, tau: 1e6, q: 1e6 };
  assert.ok(Math.abs(burnerHeat(b, 10, 20) - 15000) < 1e-6, "never above the table");
  b.suppression = null;
  b.stopAt = 15;
  assert.ok(Math.abs(burnerHeat(b, 10, 20) - (burnerEnergy(b, 15) - burnerEnergy(b, 10))) < 1e-9);
  assert.equal(burnerHeat(b, 15, 20), 0);
  // A design-fire item's layers (4 cells a layer, 3 layers, 0.5 m cells, a 1 m fire): as many as the flame reaches, at least one.
  const g = { nx: 4, ny: 3, nz: 1, h: 0.5 };
  const item = createBurner({ layers: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], hrr: [[0, 0]], diameter: 1 }, new Uint8Array(12), g);
  assert.deepEqual(Array.from(item.layerStart), [0, 4, 8, 12]);
  assert.equal(burnerCells(item, 10e3), 4, "a small flame: its base layer");
  const L = flameHeight(500e3, 1); // 0.235 * 500^0.4 - 1.02: 1.43 m
  assert.ok(Math.abs(L - (0.235 * 500 ** 0.4 - 1.02)) < 1e-12);
  assert.equal(burnerCells(item, 500e3), 12, `${L.toFixed(2)} m reaches all three`);
  assert.equal(burnerCells(b, 1e9), 1, "a plain burner heats all its cells");
});

test("plumeRise is McCaffrey's Table 1: 797 K in the flame, 62.9 / z* and 21.6 z*^-5/3 above, meeting at z* 0.079 and 0.201", () => {
  const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
  const Q = 100e3; // 100 kW: z = z* Q^(2/5), Q^(2/5) = 6.31
  const z = (zs) => zs * 100 ** 0.4;
  assert.equal(plumeRise(z(0.03), Q), 797);
  close(plumeRise(z(0.1), Q), 629, 0.5, "intermittent at z* 0.1");
  close(plumeRise(z(1), Q), 21.6, 1e-9, "plume at z* 1");
  for (const at of [62.9 / 797, (21.6 / 62.9) ** 1.5]) close(plumeRise(z(at * (1 - 1e-9)), Q), plumeRise(z(at * (1 + 1e-9)), Q), 1e-3, `continuous at z* ${at.toFixed(3)}`);
  close(62.9 / 797, 0.079, 0.0005, "flame to intermittent");
  close((21.6 / 62.9) ** 1.5, 0.201, 0.0005, "intermittent to plume");
  assert.equal(plumeRise(1, 0), 0, "no fire, no rise");
});

test("a burner rejects what it cannot release: no open cell, a bad table, a bad radiant fraction", () => {
  const g = { nx: 2, ny: 1, nz: 1 };
  const kind = Uint8Array.from([2, 0]); // cell 0 solid
  assert.throws(() => createBurner({ cells: [0], hrr: [[0, 1]] }, kind, g), /open cell/);
  assert.throws(() => createBurner({ cells: [1], hrr: [] }, kind, g), /table/);
  assert.throws(() => createBurner({ cells: [1], hrr: [[0, 1], [-1, 2]] }, kind, g), /non-decreasing/);
  assert.throws(() => createBurner({ cells: [1], hrr: [[0, -1]] }, kind, g), /non-negative/);
  assert.throws(() => createBurner({ cells: [1], hrr: [[0, 1]], radiantFraction: 1.5 }, kind, g), /radiant fraction/);
  // Solid cells are dropped, the open ones kept.
  assert.deepEqual(Array.from(createBurner({ cells: [0, 1], hrr: [[0, 1]] }, kind, g).cells), [1]);
});

test("the burner's cells gain exactly its convective heat, with methane's products at 417 kJ per mol O2; its radiant part reaches the walls", () => {
  const hrr = [[0, 0], [10, 20e3], [20, 20e3]];
  // A sealed room, so every ray ends in a wall, floor or ceiling.
  const scene = roomScene({ h: 0.5, W: 4, D: 3, door: false, hrr, radiantFraction: 0.3 });
  const sim = createSolver(scene, { contentsFuel: FUEL, ignitionFuel: FUEL });
  assert.ok(sim.burner && sim.burner.cells.length === 8);
  const cells = Array.from(sim.burner.cells);
  const Ta = sim.P.ambient;
  const excess = () => cells.reduce((a, c) => a + sim.rho[c] * sim.V * mixtureCp(sim.Y, c * NS) * (sim.T[c] - Ta), 0);
  const mass = (s) => cells.reduce((a, c) => a + sim.rho[c] * sim.V * sim.Y[c * NS + s], 0);
  const walls = () => {
    let sum = 0;
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === 2) sum += sim.inertC[c] * (sim.Ts[c] - Ta);
    return sum;
  };
  const o2 = mass(SPECIES.O2), co2 = mass(SPECIES.CO2), walls0 = walls();
  // The burner stage alone (advection, conduction and the rest would move the heat on).
  const dt = 0.5;
  for (let s = 0; s < 40; s += 1) {
    burnerStep(sim, dt);
    sim.time += dt;
  }
  const released = 0.5 * 20e3 * 10 + 20e3 * 10; // the table's integral over 20 s, J
  const convective = 0.7 * released;
  // The deposit is the convective part to well within 1 % (float32 fields).
  assert.ok(Math.abs(excess() - convective) < 0.01 * convective, `${excess()} J against ${convective} J`);
  // Methane makes no soot, so the gas absorbs none of the radiant part: the surfaces take it all.
  const radiant = 0.3 * released;
  assert.ok(Math.abs(walls() - walls0 - radiant) < 0.02 * radiant, `${walls() - walls0} J in the surfaces against ${radiant} J`);
  assert.ok(Math.abs(sim.burner.convective - convective) < 1e-6 * convective);
  assert.ok(Math.abs(sim.burner.released - released) < 1e-6 * released);
  assert.ok(Math.abs(sim.stats.released - released) < 1e-6 * released);
  // Oxygen consumed at Huggett's 13.03 MJ per kg; CO2 at methane's yield.
  const burnt = released / METHANE.heatOfCombustion;
  assert.ok(Math.abs(o2 - mass(SPECIES.O2) - released / HEAT_PER_O2) < 0.01 * (released / HEAT_PER_O2));
  assert.ok(Math.abs(mass(SPECIES.CO2) - co2 - burnt * METHANE.yields.co2) < 0.01 * burnt * METHANE.yields.co2);
});

test("in a full run the frame heat release is the burner's power, and the sprinklers' estimate reads its plume", () => {
  const hrr = [[0, 0], [20, 60e3], [40, 60e3]];
  const scene = roomScene({ h: 0.3, hrr, radiantFraction: 0.25 });
  // A head over the burner, so the estimate reads the burner's plume; its
  // link 0.01 m under the 2.4 m ceiling.
  const head = { id: "over", position: [2.0, 2.39, 1.5], ceilingAbove: 0.01, rti: 50, c: 0.5, activationC: 57, flowLpm: 0, coverage: 1 };
  const frames = [];
  const result = runScene(scene, {
    nodes: Float32Array.from([1, 0.75, 1]),
    horizon: 40,
    frameSeconds: 1,
    particles: 0,
    keep: false,
    sensorGas: true,
    sprinklers: { heads: [head], mode: "activation" },
    onFrame: (f) => frames.push({ time: f.time, hrr: f.hrr, gas: f.sprinklers.gas.T[0] }),
  });
  const b = createBurner({ cells: [0], hrr }, new Uint8Array(1), { nx: 1, ny: 1, nz: 1 });
  for (const f of frames) {
    const table = (burnerEnergy(b, f.time) - burnerEnergy(b, f.time - 1)) / 1000; // kW over the frame
    assert.ok(Math.abs(f.hrr - table) <= 0.01 * table + 1e-9, `frame ${f.time} s: ${f.hrr} kW against ${table} kW`);
  }
  const released = 0.5 * 60e3 * 20 + 60e3 * 20;
  assert.ok(Math.abs(result.burner.released - released) < 0.01 * released);
  assert.ok(Math.abs(result.burner.convective - 0.75 * released) < 0.01 * 0.75 * released);
  // The estimate's plume is the burner's: from its lowest layer to the ceiling.
  assert.ok(Math.abs(result.sprinklers.plumeHeight - (2.4 - 0.15)) < 0.31, `plume height ${result.sprinklers.plumeHeight} m`);
  const h = result.sprinklers.heads[0];
  assert.ok(h.estimate > 0 && h.estimate < 40, `the head over the burner opens (estimate ${h.estimate} s)`);
  // The heat release it read when it opened is the table's (3 kW/s), give or take a frame.
  assert.ok(Math.abs(h.hrrEstimate - Math.min(60, 3 * h.estimate)) < 6, `heat release read at activation ${h.hrrEstimate} kW`);
  // The sensor gas over the burner heats (on this 0.3 m grid by only 10 to
  // 20 K over a 60 kW fire: the coarse plume; see the validation note).
  assert.ok(Math.max(...frames.map((f) => f.gas)) > 293.15 + 8);
});

test("a burner's fuel burns only with oxygen: in a sealed room its heat stops at the room's oxygen, its gas stays below methane's flame temperature, and the unburnt heat waits", async () => {
  const { step } = await import("./flamescapes.js");
  // 200 kW in a sealed 2 x 1.5 x 1.5 m room: its 1.2 kg of oxygen burns about
  // 16 MJ, which the table passes in 80 s; it asks for 30 MJ over 150 s.
  const hrr = [[0, 0], [5, 200e3], [150, 200e3]];
  const scene = roomScene({ h: 0.25, W: 2, D: 1.5, H: 1.5, door: false, burnerAt: [1, 0.75], layers: 2, hrr, radiantFraction: 0.2 });
  const sim = createSolver(scene, { contentsFuel: FUEL, ignitionFuel: FUEL });
  let o2 = 0;
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== 2) o2 += sim.rho[c] * sim.V * sim.Y[c * NS + SPECIES.O2];
  let hottest = 0;
  while (sim.time < 150 - 1e-9) {
    step(sim, 0.05);
    for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== 2 && sim.T[c] > hottest) hottest = sim.T[c];
  }
  const asked = burnerEnergy(sim.burner, 150);
  // No more heat than the room's oxygen can give (Huggett), and less than asked.
  assert.ok(sim.burner.released <= o2 * HEAT_PER_O2 * 1.001, `${sim.burner.released} J released, ${o2 * HEAT_PER_O2} J of oxygen`);
  assert.ok(sim.burner.released < 0.8 * asked);
  // What it could not burn is held, not lost: released and pending make up the table.
  assert.ok(Math.abs(sim.burner.released + sim.burner.pending - asked) < 1e-6 * asked);
  // Methane's adiabatic flame temperature in air is about 2,230 K.
  assert.ok(hottest < 2300, `hottest gas ${hottest.toFixed(0)} K`);
});
