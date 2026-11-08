/** Sprinkler activation and suppression (suppression.js) on their own and in a room fire (flamescapes.js, flamescapesRun.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { EVANS_MIN_DENSITY, alpert, createSuppression, crossing, evansFactor, evansTau, linkStep, openHeads } from "./suppression.js";
import { waterResponse } from "./fireClasses.js";
import { MATERIAL } from "./voxelize.js";
import { buildScene, readNodes, runScene, sprinklerFrame, sprinklerRun, withBurnerFlame } from "./flamescapesRun.js";
import { SOLID, cellSpeed, createSolver, flameHeight, step } from "./flamescapes.js";
import { designFireFor, fuelsFor } from "./fireClasses.js";

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

test("a design-fire first item follows its t-squared curve with the room's contents stepped aside, and the water over it decays it", () => {
  assert.deepEqual(designFireFor("A", "fast"), { alpha: 0.0469, growth: "fast" });
  assert.deepEqual(designFireFor("A"), { alpha: 0.0117, growth: "medium" }, "class A's own: medium");
  assert.equal(designFireFor("B"), null, "a pool keeps its burning rate");
  assert.equal(designFireFor("L"), null, "so does a battery pack");
  assert.equal(designFireFor("A", "solver"), null);
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const designFire = designFireFor("A", "fast");
  const make = () => buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item, designFire });
  const scene = make();
  assert.ok(scene.burner && scene.pilot === null && scene.ignition.designFire, "a burner and no pilot");
  const under = new Set(scene.burner.layers.flat());
  assert.ok(!scene.contents.cells.some((c) => under.has(c)), "no room contents under the item");
  const nodes = Float32Array.from([2.5, 0.75, 2, 1, 0.75, 1, 4.5, 0.75, 3.5]);
  const run = (sprinklers) => runScene(make(), { nodes, horizon: 60, frameSeconds: 5, particles: 0, sprinklers });
  const free = run(null);
  // While the item burns alone each frame releases the curve's mean over it, alpha (t1^3 - t0^3) / 3 / 5 s.
  for (let k = 0; k < free.K; k += 1) {
    const t0 = 5 * k, t1 = t0 + 5;
    const want = (0.0469 * (t1 ** 3 - t0 ** 3)) / 3 / 5;
    close(free.hrr[k], want, 0.02 * want + 0.3, `frame ${k} (kW)`);
  }
  const energy = (46.9 * 60 ** 3) / 3;
  close(free.designFire.released, energy, 0.01 * energy, "the item's heat over 60 s (J)");
  // Activation alone leaves it as it is; wetting decays it from the end of the frame its head opened in.
  assert.equal(run({ heads: HEADS, mode: "activation" }).designFire.released, free.designFire.released);
  const wet = run({ heads: HEADS, mode: "wetting" });
  const over = wet.sprinklers.heads[0];
  assert.ok(over.estimate > 0 && over.estimate < 60, `the head over the item opens (${over.estimate} s)`);
  assert.equal(wet.designFire.suppressed, "water");
  assert.ok(wet.designFire.suppressedAt >= over.estimate && wet.designFire.suppressedAt <= over.estimate + 5, `water from ${wet.designFire.suppressedAt} s`);
  assert.ok(wet.designFire.released < free.designFire.released, "the water decays it");
});

test("the design-fire item's flame is drawn over it from the first frame, and drawing it changes nothing in the run", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("C");
  const make = () => buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item, designFire: designFireFor("C") });
  const nodes = Float32Array.from([2.5, 0.75, 2, 1, 0.75, 1]);
  const frames = [];
  const drawn = runScene(make(), { nodes, horizon: 15, frameSeconds: 5, particles: 3000, onFrame: (f) => frames.push(f.particles) });
  const plain = runScene(make(), { nodes, horizon: 15, frameSeconds: 5, particles: 0 });
  assert.deepEqual(Array.from(drawn.hrr), Array.from(plain.hrr), "the same heat release");
  assert.deepEqual(Array.from(drawn.T), Array.from(plain.T), "the same temperatures");
  // A medium fire is 0.3 kW at 5 s and 2.6 kW at 15 s: the room's contents
  // have not caught, so every flame point is the item's, over its footprint.
  const scene = make();
  const g = scene.grid;
  const columns = new Set(scene.burner.layers[0].map((c) => `${c % g.nx},${Math.floor(c / (g.nx * g.ny))}`));
  frames.forEach((p, k) => {
    let over = 0;
    for (let q = 0; q < p.kinds.length; q += 1) {
      const [i, , kk] = [0, 1, 2].map((a) => Math.floor((p.positions[q * 3 + g.axes[a]] - g.origin[a]) / g.h));
      if (p.kinds[q] === 0 && columns.has(`${i},${kk}`)) over += 1;
    }
    assert.ok(over >= scene.burner.layers[0].length, `frame ${k}: ${over} flame points over the item`);
  });
});

test("the item's drawn flame narrows to its tip, flickers, reaches no higher than one and a half mean heights, and takes its cells' temperatures", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("C");
  const scene = buildScene(room(), { fire: [2.5, 0.75, 2], maxCells: 20000, minCell: 0.25, maxCell: 0.25, contentsFuel, ignitionFuel, item, designFire: designFireFor("C") });
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1, maxParticles: 0 });
  while (sim.time < 120 - 1e-9) step(sim, 0.1);
  const g = sim.grid, b = sim.burner;
  const none = { positions: new Float32Array(0), temps: new Float32Array(0), kinds: new Uint8Array(0) };
  const p = withBurnerFlame(sim, none);
  const n = p.temps.length;
  assert.ok(n > 6 * b.layerStart[1], `${n} points`);
  // In grid units: each point's cell, its height over the item's base, its distance from the base's centre.
  const base = Array.from(b.cells.subarray(0, b.layerStart[1]));
  const ci = base.reduce((s, c) => s + (c % g.nx) + 0.5, 0) / base.length;
  const ck = base.reduce((s, c) => s + Math.floor(c / (g.nx * g.ny)) + 0.5, 0) / base.length;
  const floor = Math.floor(base[0] / g.nx) % g.ny;
  const L = flameHeight(b.flameRate, b.diameter) / g.h;
  const low = [], high = [];
  for (let q = 0; q < n; q += 1) {
    const x = [0, 1, 2].map((a) => (p.positions[q * 3 + g.axes[a]] - g.origin[a]) / g.h);
    const cell = Math.floor(x[0]) + g.nx * (Math.floor(x[1]) + g.ny * Math.floor(x[2]));
    assert.notEqual(sim.kind[cell], SOLID, "never in a solid cell");
    assert.equal(p.temps[q], Math.fround(sim.T[cell]), "its cell's gas temperature");
    const z = x[1] - floor;
    assert.ok(z >= 0 && z <= 1.5 * L + 1e-6, `${z} cells over the base, mean height ${L}`);
    (z < 0.5 * L ? low : z > L ? high : []).push(Math.hypot(x[0] - ci, x[2] - ck));
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  assert.ok(low.length > high.length && high.length > 0, `${low.length} points under half its height, ${high.length} over it`);
  assert.ok(mean(high) < 0.8 * mean(low), `narrower at the top: ${mean(high)} against ${mean(low)} cells from the centre`);
  // The same step draws the same flame; the next step another.
  assert.deepEqual(withBurnerFlame(sim, none).positions, p.positions);
  step(sim, 0.1);
  assert.notDeepEqual(withBurnerFlame(sim, none).positions.subarray(0, 30), p.positions.subarray(0, 30));
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
