/** Sprinkler activation and suppression (suppression.js) on their own and in a room fire (flamescapes.js, flamescapesRun.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { EVANS_MIN_DENSITY, alpert, createSuppression, crossing, evansFactor, evansTau, linkStep, openHeads } from "./suppression.js";
import { waterResponse } from "./fireClasses.js";
import { MATERIAL } from "./voxelize.js";
import { buildScene, readNodes, runScene, sprinklerFrame, sprinklerRun } from "./flamescapesRun.js";
import { cellSpeed, createSolver, step } from "./flamescapes.js";
import { fuelsFor } from "./fireClasses.js";

const close = (a, b, tol, what = "") => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs ${b}`);

test("the link's exact step and crossing time give the report's worked activation times", () => {
  // T0 = 20 C, Ta = 68 C, Tg = 100 C, u = 1 m/s (sprinkler-physics-2026-11-07.md 3.3).
  const cases = [
    [50, 0, 45.8],
    [50, 0.4, 65.4],
    [100, 0, 91.6],
    [36, 0.4, 47.1],
  ];
  for (const [rti, c, want] of cases) {
    const s = linkStep(293.15, 373.15, 293.15, 1, rti, c, 1);
    close(crossing(293.15, s.Tinf, s.k, 341.15), want, 0.1, `RTI ${rti}, C ${c}`);
  }
  // Two exact steps are one (Sprinklers.lean exactStep_add), and the link never passes the gas.
  const one = linkStep(300, 500, 293.15, 2, 50, 0.4, 10).T;
  const half = linkStep(linkStep(300, 500, 293.15, 2, 50, 0.4, 5).T, 500, 293.15, 2, 50, 0.4, 5).T;
  close(one, half, 1e-9);
  assert.ok(linkStep(300, 500, 293.15, 2, 50, 0, 1e6).T <= 500 + 1e-9);
});

test("Alpert's branches meet, and Evans's decay time matches his table", () => {
  for (const [Q, H] of [[100, 2.4], [500, 3], [1000, 6]]) {
    const inner = alpert(Q, H, 0.18 * H - 1e-9).dT, outer = alpert(Q, H, 0.18 * H + 1e-9).dT;
    assert.ok(Math.abs(inner - outer) / inner < 0.002, `dT ${inner} vs ${outer}`);
    const u1 = alpert(Q, H, 0.15 * H - 1e-9).U, u2 = alpert(Q, H, 0.15 * H + 1e-9).U;
    assert.ok(Math.abs(u1 - u2) / u1 < 0.011, `U ${u1} vs ${u2}`);
  }
  // NISTIR 5254: 411 s at 0.07 mm/s, 212 s at 0.10, 59 s at 0.2.
  close(evansTau(0.07), 411, 1);
  close(evansTau(0.1), 212, 1);
  close(evansTau(0.2), 59, 1);
});

test("Evans's cap reads a cell's rate the step after its head opens, then holds it under the decay", () => {
  const sup = createSuppression(4, { mode: "evans" });
  sup.tau[1] = 100;
  sup.tAct[1] = 10;
  assert.equal(evansFactor(sup, 0, 1, 10, 0.1), 1, "an uncovered cell is untouched");
  assert.equal(evansFactor(sup, 1, 0.02, 10, 0.1), 1, "the first step reads the rate");
  close(sup.rate[1], 0.2, 1e-7);
  // 100 s later the cap is 0.2 e^-1 kg/s: a loss of 0.02 kg in 0.1 s is scaled to it.
  close(evansFactor(sup, 1, 0.02, 110, 0.1) * 0.02, 0.2 * Math.exp(-1) * 0.1, 1e-9);
  assert.equal(evansFactor(sup, 1, 1e-6, 110, 0.1), 1, "a loss under the cap is left alone");
});

test("opening heads adds their densities over shared cells and shares each head's water over its columns", () => {
  const head = (flowLpm, coverage, covered, columns) => ({ sensor: 0, flowLpm, coverage, covered: Int32Array.from(covered), columns: Int32Array.from(columns) });
  const setup = { heads: [head(85, 20.9, [5, 6], [100, 4, 101, 4]), head(85, 20.9, [6, 7], [101, 4, 102, 3]), head(10, 20.9, [8], [])] };
  const sup = createSuppression(16, { mode: "spray" });
  const first = openHeads(sup, setup, [{ head: 0, time: 30 }]);
  assert.deepEqual(first.armed, [5, 6]);
  const w = 85 / 60 / 20.9;
  close(sup.tau[5], evansTau(w), 1e-3);
  openHeads(sup, setup, [{ head: 1, time: 40 }, { head: 2, time: 40 }]);
  close(sup.density[6], 2 * w, 1e-7);
  close(sup.tau[6], evansTau(2 * w), 1e-2);
  assert.equal(sup.tAct[6], 30, "a cell keeps the time its first head opened");
  assert.equal(sup.tau[8], 0, `a spray under ${EVANS_MIN_DENSITY} mm/s leaves the fuel alone`);
  // Columns: one entry each, the shared one with both heads' water.
  assert.equal(sup.columns.length, 3);
  const flux = Object.fromEntries(sup.columns.map((c) => [c.top, c.flux]));
  close(flux[101], 85 / 60, 1e-9);
  close(flux[100] + flux[101] + flux[102], 2 * (85 / 60), 1e-9);
  // Opening a head twice changes nothing.
  assert.equal(openHeads(sup, setup, [{ head: 0, time: 50 }]).touched.length, 0);
});

function box(x0, y0, z0, x1, y1, z1, material = MATERIAL.STONE) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material };
}

/** One quick-response head over the fire, covering the room, and one in a corner. */
const head = (id, x, z) => ({ id, position: [x, 2.55, z], footprint: { centre: [x, z], axis: [1, 0], half: [2.3, 2.3] }, rti: 50, c: 0.4, activationC: 68, flowLpm: 85, coverage: 20.9 });
const HEADS = [head("over", 2.5, 2), head("corner", 0.6, 0.6)];

/** The solver tests' 5 x 4 x 2.7 m room with a door, y up. */
function room(t = 0.2) {
  const W = 5, D = 4, H = 2.7;
  return {
    walls: [box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, 1.5), box(W, 0, 2.4, W + t, H, D), box(W, 2.0, 1.5, W + t, H, 2.4)],
    floors: [box(-t, -t, -t, W + t, 0, D + t)],
    roofs: [box(-t, H, -t, W + t, H + t, D + t)],
    stairs: [],
    obstacles: [],
  };
}

test("in a room fire a sprinkler opens, the estimate no later than the late bound; Evans and the spray cut the fire, and the water balances", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const make = () => buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item });
  const nodes = Float32Array.from([2.5, 0.75, 2, 1, 0.75, 1, 4.5, 0.75, 3.5]);
  const run = (sprinklers) => runScene(make(), { nodes, horizon: 60, frameSeconds: 5, particles: 0, sprinklers });
  const none = run(null);
  const watch = run({ heads: HEADS, mode: "activation" });
  const evans = run({ heads: HEADS, mode: "evans" });
  const spray = run({ heads: HEADS, mode: "spray" });
  // Activation alone changes nothing in the fire.
  assert.deepEqual(Array.from(watch.T), Array.from(none.T));
  assert.deepEqual(Array.from(watch.hrr), Array.from(none.hrr));
  const over = watch.sprinklers.heads[0];
  assert.ok(over.onGrid && over.inSight);
  assert.ok(over.estimate !== null && over.estimate < 40, `the head over the fire opens (estimate ${over.estimate} s, late ${over.late} s)`);
  assert.ok(over.late === null || over.estimate <= over.late, `estimate ${over.estimate} s, late ${over.late} s`);
  // Evans caps the fuel's reaction: less fuel burnt and less heat than the free fire.
  assert.ok(evans.sprinklers.covered > 0, "fuel cells under the head are covered");
  for (const out of [evans, spray]) {
    assert.ok(out.stats.burnt < none.stats.burnt, `burnt ${out.stats.burnt} kg against ${none.stats.burnt} kg`);
    assert.ok(out.stats.released < none.stats.released, `released ${out.stats.released} J against ${none.stats.released} J`);
  }
  // The spray's water: emitted = evaporated + at the floor.
  const st = spray.sprinklers.stats;
  assert.ok(st.emitted > 0 && st.evaporated > 0, `emitted ${st.emitted} kg, evaporated ${st.evaporated} kg`);
  close(st.evaporated + st.floor, st.emitted, 1e-6 * st.emitted);
  for (const a of [spray.T, spray.vis, spray.co]) for (const x of a) assert.ok(Number.isFinite(x));
});

test("the estimate takes each head's own height over the fire and its distance from its exact position, not the grid's", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  for (const h of [0.25, 0.35, 0.45]) {
    const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 40000, minCell: h, maxCell: h, contentsFuel, ignitionFuel, item });
    const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1, maxParticles: 0 });
    // Heads 0.13 m under the 2.7 m ceiling (the default link offset), and one with its own.
    const heads = [{ ...head("a", 3.7, 2.9), position: [3.7, 2.57, 2.9] }, { ...head("b", 1.1, 1.3), position: [1.1, 2.69, 1.3], ceilingAbove: 0.01 }];
    const { setup } = sprinklerRun(scene, sim, { heads, mode: "activation" });
    const g = scene.grid, c = scene.ignition.cell;
    const pi = c % g.nx, pj = Math.floor(c / g.nx) % g.ny, pk = Math.floor(c / (g.nx * g.ny));
    const fire = [g.origin[0] + (pi + 0.5) * g.h, g.origin[1] + (pj + 0.5) * g.h, g.origin[2] + (pk + 0.5) * g.h];
    for (const hd of setup.heads) {
      close(hd.r, Math.hypot(hd.position[0] - fire[0], hd.position[2] - fire[2]), 1e-9, `cell ${h}: r`);
      close(hd.H, 2.7 - fire[1], 1e-6, `cell ${h}: H`);
    }
    // The grid's plume column snaps the same ceiling to a cell face.
    assert.ok(Math.abs(setup.plume.H - (2.7 - fire[1])) <= g.h, `the grid's plume height ${setup.plume.H}`);
  }
});

test("every covered fuel cell stays under Evans's cap in every step after its head opens", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item });
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1, maxParticles: 0 });
  const spr = sprinklerRun(scene, sim, { heads: HEADS, mode: "evans" });
  sim.suppression = spr.sup;
  const sup = spr.sup;
  let checked = 0;
  for (let frame = 0; frame < 9; frame += 1) {
    const end = (frame + 1) * 5;
    while (sim.time < end - 1e-9) {
      const t = sim.time;
      const armed = Float32Array.from(sup.rate);
      step(sim, Math.min(sim.P.dt, end - t));
      for (let c = 0; c < sim.N; c += 1) {
        if (!(sup.tau[c] > 0) || armed[c] < 0) continue;
        const allowed = sup.rate[c] * Math.exp(-(t - sup.tAct[c]) / sup.tau[c]) * sim.P.dt;
        assert.ok(sim.solidLoss[c] <= allowed * (1 + 1e-6) + 1e-12, `cell ${c} at ${t.toFixed(1)} s: ${sim.solidLoss[c]} > ${allowed}`);
        checked += 1;
      }
    }
    const T = Float64Array.from(readNodes(sim, spr.sensors).T, (x) => x + 273.15);
    const u = Float64Array.from(spr.sensors, (c) => (c >= 0 ? cellSpeed(sim, c) : 0));
    sprinklerFrame(spr, { t0: end - 5, t1: end, T, u, hrr: sim.stats.hrr / 1000 });
  }
  assert.ok(checked > 1000, `${checked} cell steps checked`);
});

test("wetting: water soaks into the fuel it reaches, the water balances, the fire burns less, and no cap is set", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const make = () => buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item });
  const nodes = Float32Array.from([2.5, 0.75, 2, 1, 0.75, 1, 4.5, 0.75, 3.5]);
  const run = (sprinklers) => runScene(make(), { nodes, horizon: 90, frameSeconds: 5, particles: 0, sprinklers });
  const none = run(null);
  const wet = run({ heads: HEADS, mode: "wetting" });
  assert.ok(wet.sprinklers.heads[0].estimate !== null, "the head over the fire opens");
  const st = wet.sprinklers.stats;
  assert.ok(st.wetted > 0, `water on the fuel: ${st.wetted} kg`);
  close(st.evaporated + st.wetted + st.floor, st.emitted, 1e-6 * st.emitted, "emitted = evaporated + wetted + floor:");
  assert.ok(wet.stats.burnt < none.stats.burnt, `burnt ${wet.stats.burnt} kg against ${none.stats.burnt} kg`);
  // No Evans cap in this mode: the suppression record's decay times stay zero.
  const sup = createSuppression(10, { mode: "wetting" });
  assert.ok(sup.wet > 0);
  const setup = { heads: [{ sensor: 0, covered: [1, 2], columns: [5, 3], flowLpm: 85, coverage: 20.9 }] };
  openHeads(sup, setup, [{ head: 0, time: 10 }]);
  assert.deepEqual(Array.from(sup.tau), new Array(10).fill(0));
  assert.equal(sup.columns.length, 1);
});

test("water by fire class: a cooking-oil, metal or pack first item is not wetted and takes no cap; class A contents are", () => {
  for (const cls of ["A", "B", "C", "D", "K", "L"]) {
    const w = waterResponse(fuelsFor(cls).ignitionFuel);
    const solid = cls === "A" || cls === "C";
    assert.equal(w.wets, solid, `${cls} wets`);
    assert.equal(w.evans, solid, `${cls} takes Evans's cap`);
    assert.equal(w.dry, !solid, `${cls} holds no water of its own`);
    if (!solid) assert.ok(w.note, `${cls} says why`);
  }
  // A cooking-oil item under an open head, wetting: the pan cell gets no
  // water, its own water is none, and it is never capped; the room's class
  // A contents under the head do get wetted.
  const { contentsFuel, ignitionFuel, item } = fuelsFor("K");
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item });
  const sim = createSolver(scene, { contentsFuel, ignitionFuel, seed: 1, maxParticles: 0 });
  const pan = scene.ignition.cell;
  assert.equal(sim.moisture[pan], 0, "the oil holds no water of its own");
  const spr = sprinklerRun(scene, sim, { heads: HEADS, mode: "wetting" });
  sim.suppression = spr.sup;
  openHeads(spr.sup, spr.setup, [{ head: 0, time: 0 }]);
  const before = Float32Array.from(sim.moisture);
  for (let k = 0; k < 40; k += 1) step(sim, sim.P.dt);
  assert.ok(sim.moisture[pan] <= before[pan] + 1e-12, `the pan took no water: ${sim.moisture[pan]}`);
  let wettedContents = 0;
  for (const c of spr.setup.heads[0].covered) if (c !== pan && sim.moisture[c] > before[c] + 1e-9) wettedContents += 1;
  assert.ok(wettedContents > 0, "class A contents under the head are wetted");
  assert.ok(spr.sup.stats.wetted > 0);
});
