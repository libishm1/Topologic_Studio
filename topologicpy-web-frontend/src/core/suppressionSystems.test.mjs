// Suppression systems by room use (suppressionSystems.js): the room rules on
// the NIBS Office and Clinic names, the plan, the parameters, and the clean
// agent and the foam in the solver.
import test from "node:test";
import assert from "node:assert/strict";
import {
  CLEAN_AGENT, FOAM, STATE, agentAt, agentMass, blanketAt, foamTimes, planSystems, rectangleRoom, roomCells, roomDetectors, specialExtinguishers, systemForRoom, systemsSetup,
} from "./suppressionSystems.js";
import { MATERIAL } from "./voxelize.js";
import { buildScene, readNodes, runScene, systemsFrame, systemsRun } from "./flamescapesRun.js";
import { createSolver, step } from "./flamescapes.js";
import { designFireFor, fuelsFor } from "./fireClasses.js";

const close = (a, b, tol, what = "") => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} against ${b} (tolerance ${tol})`);

test("room rules: the NIBS Office and Clinic names get clean agent, foam or water; apartments get water only", () => {
  const commercial = { occupancy: "B" };
  const agent = ["SERVER RM", "A/V SERVER RM", "TELE/COMM RM", "COMM. ROOM", "ELEC RM", "ELEC. ROOM", "ELECTRICAL", "ELEC. CL.", "UPS/BATT RM", "NOC", "COMPUTER ROOM", "ADP EQUIP SPACE"];
  for (const name of agent) {
    const s = systemForRoom({ name }, commercial);
    assert.equal(s.system, "clean-agent", name);
    assert.equal(s.design.concentration, CLEAN_AGENT.design.C.value, `${name}: class C design`);
    assert.ok(s.sprinklers, `${name}: keeps its sprinklers in a sprinklered building`);
    assert.match(s.clause, /904\.10/, `${name}: cites the clean-agent clause`);
    assert.equal(s.extinguisher.unit, "co2-2", `${name}: CO2 at the door, no powder on electronics`);
  }
  assert.equal(systemForRoom({ name: "SERVER RM" }, { occupancy: "B", agentOnly: true }).sprinklers, false, "agent only: the official's approval");
  for (const name of ["GENERATOR RM", "GENSET", "DIESEL DAY TANK", "FUEL OIL STORAGE", "OIL STORE", "FLAMMABLE LIQUIDS", "FLAMABLE STORAGE", "HAZ. STOR.", "EMERGENCY POWER"]) {
    const s = systemForRoom({ name }, commercial);
    assert.equal(s.system, "foam", name);
    assert.equal(s.design.rate, FOAM.rate.value, `${name}: foam-water sprinklers' rate`);
    assert.match(s.clause, /904\.7/, `${name}: cites the foam clause`);
  }
  for (const name of ["MECH RM", "CENTRAL MECH. RM.", "MAIN MECHANICAL ROOM", "OPEN OFFICE", "CORRIDOR", "COMMON ROOM", "CMD CONFERENCE RM", "GEN. DENT. DTR #4", "DATA / ARCHIVE STORAGE", "W. TOILET", "BOILER ROOM", "ELEV EQ RM", "CL. UTL. / KITCHENET", "GAS STORAGE", "COMPUTER LAB", "Computer classroom", ""]) {
    assert.equal(systemForRoom({ name }, commercial).system, "water", JSON.stringify(name));
  }
  assert.equal(systemForRoom("MECH RM", commercial).use, "mechanical");
  // Apartments: water in every room, whatever its name.
  for (const occupancy of ["R-2", "R", "apartment", ["R-2", "B"]]) {
    for (const name of ["Living Room", "Kitchen", "ELEC CL", "Utility", "SERVER RM", "GENERATOR"]) {
      const s = systemForRoom({ name }, { occupancy });
      assert.equal(s.system, "water", `${name} in ${occupancy}`);
      assert.match(s.clause, /903\.2\.8/);
    }
  }
  // A space that is not a room gets none; a building without sprinklers keeps only the special systems.
  assert.equal(systemForRoom({ name: "ROOF", problem: "outside the building" }, commercial).system, "none");
  assert.equal(systemForRoom({ name: "OPEN OFFICE" }, { occupancy: "B", sprinklered: false }).system, "none");
  const unsprinklered = systemForRoom({ name: "SERVER RM" }, { occupancy: "B", sprinklered: false });
  assert.equal(unsprinklered.system, "clean-agent");
  assert.equal(unsprinklered.sprinklers, false);
});

test("planSystems: counts, clauses and the agent by NFPA 2001's flooding formula", () => {
  // W = (V / s) (C / (100 - C)), s = 0.1269 + 0.000513 t: 0.549 kg/m3 at 7.0 % and 20 C (NFPA's table 0.5483).
  close(agentMass(1, 7.0, 20), 0.5488, 1e-3, "kg per m3 at 7 %");
  close(agentMass(100, 7.0, 20), 54.88, 0.05, "a 100 m3 room");
  close(agentMass(1, 8.7, 20), 0.695, 1e-3, "kg per m3 at 8.7 %");
  assert.equal(agentMass(0, 7), 0);
  const rooms = [
    { id: 1, name: "SERVER RM", floor: 0, ceiling: 3, area: 20 },
    { id: 2, name: "OPEN OFFICE", floor: 0, ceiling: 3, area: 200 },
    { id: 3, name: "FLAMABLE STORAGE", floor: 0, ceiling: 3, area: 10 },
    { id: 4, name: "ROOF", problem: "outside" },
  ];
  const plan = planSystems(rooms, { occupancy: "B" });
  assert.deepEqual(plan.counts, { water: 1, "clean-agent": 1, foam: 1, none: 1 });
  close(plan.rooms[0].agentKg, agentMass(60, 7.0, 20), 1e-9, "the server room's agent");
  close(plan.agentKg, plan.rooms[0].agentKg, 1e-9);
  assert.ok(plan.clauses.some((c) => /904\.10/.test(c)) && plan.clauses.some((c) => /904\.7/.test(c)) && plan.clauses.some((c) => /903\.3\.1\.1/.test(c)));
  assert.ok(plan.lines.length >= 3);
  const flats = planSystems(rooms.slice(0, 3), { occupancy: "R-2" });
  assert.deepEqual(flats.counts, { water: 3, "clean-agent": 0, foam: 0, none: 0 });
  // Portable units of their own: CO2 at the server room, class B at the flammable store; the office has ABC powder on the routes.
  const placed = [rectangleRoom({ id: 1, name: "SERVER RM", x0: 0, z0: 0, x1: 4, z1: 5, floor: 3, ceiling: 6 }), rectangleRoom({ id: 2, name: "OPEN OFFICE", x0: 4, z0: 0, x1: 20, z1: 10, floor: 3, ceiling: 6 }), rectangleRoom({ id: 3, name: "FLAMABLE STORAGE", x0: 0, z0: 5, x1: 3, z1: 8, floor: 3, ceiling: 6 })];
  const units = specialExtinguishers(planSystems(placed, { occupancy: "B" }), placed);
  assert.deepEqual(units.map((u) => [u.room, u.unit]), [[1, "co2-2"], [3, "foam-9.5"]]);
  assert.deepEqual(units[0].position, [2, 4, 2.5], "at the room's centre, 1 m over its floor");
  assert.equal(specialExtinguishers(planSystems(placed, { occupancy: "R-2" }), placed).length, 0, "apartments: one unit per dwelling, not placed here");
});

test("parameters: foam times from the delivered densities, the agent's rise, hold and end, the blanket's fall", () => {
  const t65 = foamTimes(6.5), t41 = foamTimes(4.1);
  // The research note's central values: 10 s and 30 s at 6.5 L/(min m2), 15 s and 45 s at 4.1.
  close(t65.control, 10, 0.5, "control at 6.5");
  close(t65.extinguish, 30, 1, "out at 6.5");
  close(t41.control, 15, 0.5, "control at 4.1");
  close(t41.extinguish, 45, 2, "out at 4.1");
  close(foamTimes(6.5, { fluorineFree: true }).extinguish, 2 * t65.extinguish, 1e-9, "fluorine-free");
  // Design concentrations: below the NOAEL, above the MECs they protect.
  for (const d of Object.values(CLEAN_AGENT.design)) assert.ok(d.value < CLEAN_AGENT.noael.value && d.value >= CLEAN_AGENT.mecClassA.value);
  close(CLEAN_AGENT.design.C.value, 1.35 * CLEAN_AGENT.mecClassA.value, 0.05, "class C = 1.35 x class A MEC");
  close(CLEAN_AGENT.design.B.value, 1.3 * CLEAN_AGENT.mecHeptane.value, 0.05, "class B = 1.3 x heptane MEC");
  const rec = { t0: 40, design: 7, discharge: 10, hold: 600, retained: 1 };
  assert.equal(agentAt(rec, 39.9), 0);
  close(agentAt(rec, 45), 3.5, 1e-12);
  assert.equal(agentAt(rec, 50), 7);
  assert.equal(agentAt(rec, 649), 7);
  assert.equal(agentAt(rec, 650), 0, "the flood ends at the hold's end");
  close(agentAt({ ...rec, retained: 0.85 }, 650 - 1e-9), 7 * 0.85, 1e-6, "with leakage, 85 % at the hold's end");
  assert.equal(agentAt({ ...rec, t0: Infinity }, 1e6), 0, "never released");
  const foam = { t0: 20, control: 10, extinguish: 30 };
  assert.equal(blanketAt(foam, 19), 1);
  close(blanketAt(foam, 30), 0.1, 1e-12, "a tenth left at control");
  close(blanketAt(foam, 40), 0.05, 1e-12);
  assert.equal(blanketAt(foam, 50), 0);
  assert.equal(blanketAt(foam, 1e4), 0, "the blanket stays");
});

function box(x0, y0, z0, x1, y1, z1, material = MATERIAL.STONE) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material };
}

/** Two rooms side by side, y up: the solver tests' 5 x 4 x 2.7 m room, then a 4 x 4 m room beyond its door (x 5.2 to 9.2). */
function twoRooms(t = 0.2) {
  const W = 5, D = 4, H = 2.7, W2 = 9.2;
  return {
    walls: [
      box(-t, 0, -t, W2 + t, H, 0), box(-t, 0, D, W2 + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W2, 0, 0, W2 + t, H, D),
      box(W, 0, 0, W + t, H, 1.5), box(W, 0, 2.4, W + t, H, D), box(W, 2.0, 1.5, W + t, H, 2.4),
    ],
    floors: [box(-t, -t, -t, W2 + t, 0, D + t)],
    roofs: [box(-t, H, -t, W2 + t, H + t, D + t)],
    stairs: [],
    obstacles: [],
  };
}
const ROOM1 = rectangleRoom({ id: "r1", name: "SERVER RM", x0: 0, z0: 0, x1: 5, z1: 4, floor: 0, ceiling: 2.7 });
const ROOM2 = rectangleRoom({ id: "r2", name: "OPEN OFFICE", x0: 5.2, z0: 0, x1: 9.2, z1: 4, floor: 0, ceiling: 2.7 });
const sceneOf = (cls, fire = [2.5, 0.75, 2]) => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor(cls);
  return buildScene(twoRooms(), { fire, maxCells: 30000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item });
};
const nodes = Float32Array.from([2.5, 0.75, 2, 1, 0.75, 1, 4.5, 0.75, 3.5, 7, 0.75, 2]);

test("room cells and detectors: only the room's open cells, under its ceiling, two zones each with a detector", () => {
  const scene = sceneOf("A");
  const g = scene.grid;
  const { cells, columns } = roomCells(g, scene.solid, ROOM1);
  assert.ok(cells.length > 500, `${cells.length} cells`);
  for (const c of cells) {
    assert.equal(scene.solid[c], 0, "an open cell");
    const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
    const x = g.origin[0] + (i + 0.5) * g.h, y = g.origin[1] + (j + 0.5) * g.h, z = g.origin[2] + (k + 0.5) * g.h;
    assert.ok(x > 0 && x < 5 && z > 0 && z < 4 && y > 0 && y < 2.7, `cell ${c} at ${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)} is in the room`);
  }
  const det = roomDetectors(g, columns, ROOM1, { zones: 2 });
  assert.ok(det.cells.length >= 2 && det.zone.includes(0) && det.zone.includes(1), `${det.cells.length} detectors in zones ${Array.from(det.zone)}`);
  for (const c of det.cells) {
    // Each detector is the top open cell of its column: solid above it.
    assert.ok(cells.includes(c));
    assert.equal(scene.solid[c + g.nx], 1, "the ceiling is right above the detector");
  }
  // A cell in two rooms goes to the first; rooms of other systems are left out.
  const sys = systemsSetup(scene, { rooms: [{ ...ROOM1, system: "clean-agent" }, { ...ROOM1, id: "again", system: "foam" }, { ...ROOM2, system: "water" }] });
  assert.equal(sys.R, 2);
  assert.equal(sys.rooms[1].cells, 0, "the second copy of the room gets no cells");
  let marked = 0;
  for (let c = 0; c < sys.room.length; c += 1) if (sys.room[c]) marked += 1;
  assert.equal(marked, cells.length);
  assert.equal(systemsSetup(scene, { rooms: [{ ...ROOM2, system: "water" }] }), null, "water rooms are the sprinklers' business");
});

test("off: no systems, an empty list or water rooms only leave the run exactly as it was", () => {
  const run = (systems) => runScene(sceneOf("A"), { nodes, horizon: 30, frameSeconds: 5, particles: 0, ...(systems === undefined ? {} : { systems }) });
  const base = run(undefined);
  assert.equal(base.systems, undefined, "no systems field without systems");
  for (const systems of [null, { rooms: [] }, { rooms: [{ ...ROOM2, system: "water" }] }]) {
    const out = run(systems);
    assert.deepEqual(Array.from(out.hrr), Array.from(base.hrr));
    assert.deepEqual(Array.from(out.T), Array.from(base.T));
    assert.equal(out.stats.released, base.stats.released);
    assert.equal(out.systems, undefined);
  }
});

test("clean agent: after detection and the delay the room floods, its fire stops burning in the gas and on the fuel while the agent holds, and it can re-ignite after", () => {
  const scene = sceneOf("A");
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1 });
  const hold = 30;
  const sys = systemsRun(scene, sim, { rooms: [{ ...ROOM1, system: "clean-agent", params: { hold } }] });
  sim.systems = sys;
  const rec = sys.rooms[0];
  assert.equal(rec.zones, 2);
  const inRoom = (c) => sys.room[c] === 1;
  let flooded = 0, before = 0, during = 0, after = 0;
  for (let frame = 0; frame < 24; frame += 1) {
    const end = (frame + 1) * 5;
    while (sim.time < end - 1e-9) {
      const t = sim.time;
      const released = sim.stats.released;
      step(sim, Math.min(sim.P.dt, end - t));
      const agent = agentAt(rec, t);
      if (agent >= rec.extA && agent > 0) {
        // Held at or above the class A (and here the heptane) value: no cell of the room burns.
        for (let c = 0; c < sim.N; c += 1) if (inRoom(c)) assert.equal(sim.burning[c], 0, `cell ${c} burns at ${t.toFixed(1)} s with ${agent.toFixed(2)} % agent`);
        during += sim.stats.released - released;
        flooded += 1;
      } else if (rec.t0 < Infinity && t >= rec.t0 + rec.discharge + rec.hold) after += sim.stats.released - released;
      else before += sim.stats.released - released;
    }
    systemsFrame(sys, { t1: end, vis: readNodes(sim, sys.detectorCells).vis });
  }
  assert.ok(rec.detectedAt !== null && rec.detectedAt <= 30, `detected at ${rec.detectedAt} s`);
  assert.equal(rec.t0, rec.detectedAt + CLEAN_AGENT.delay.value, "discharge after the pre-discharge delay");
  assert.ok(before > 1e6, "the fire burned before the discharge");
  assert.ok(flooded > 200, `${flooded} steps under the agent`);
  // Under the agent only vapour that leaves the room (through the door, outside its cells) can burn: next to nothing.
  assert.ok(during < 1e-3 * before, `${during} J released under the agent against ${before} J before`);
  assert.ok(rec.stats.oxBlocked > 0 && rec.stats.flameSteps > 0, "the agent stopped oxidation and flames");
  // The hot fuel left after a 30 s hold lights again when the agent goes (NFPA 2001 A.7.4.1).
  assert.ok(after > 1e6, `${(after / 1e6).toFixed(2)} MJ released after the hold`);
});

test("clean agent: a design-fire first item stops when the agent reaches the class A extinguishing concentration", () => {
  const { contentsFuel, ignitionFuel, item } = fuelsFor("C");
  const designFire = designFireFor("C", "fast");
  const make = () => buildScene(twoRooms(), { fire: [2.5, 0.75, 2], maxCells: 30000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item, designFire });
  const free = runScene(make(), { nodes, horizon: 40, frameSeconds: 5, particles: 0 });
  const flooded = runScene(make(), { nodes, horizon: 40, frameSeconds: 5, particles: 0, systems: { rooms: [{ ...ROOM1, system: "clean-agent", params: { releaseAt: 5 } }] } });
  const rec = { t0: 5, discharge: CLEAN_AGENT.discharge.value, extA: CLEAN_AGENT.mecClassA.value, design: CLEAN_AGENT.design.C.value };
  const stop = rec.t0 + (rec.discharge * rec.extA) / rec.design;
  assert.ok(Math.abs(flooded.designFire.stoppedAt - stop) < 1e-9, `stopped at ${flooded.designFire.stoppedAt} s, ${stop} s expected`);
  // Its heat is the curve's up to the stop (the stop is set at the end of the frame the agent was released in), then none.
  const curve = (46.9 * stop ** 3) / 3;
  assert.ok(Math.abs(flooded.designFire.released - curve) < 0.01 * curve, `${flooded.designFire.released} J against ${curve} J`);
  assert.ok(free.designFire.released > 2 * flooded.designFire.released, "unflooded it burns on");
});

test("clean agent: the agent stays in its room; a fire next door burns as it would", () => {
  // The fire in the second room, the first room flooded from the start (manual release at 0 s).
  const fire2 = [7.2, 0.75, 2];
  const run = (systems) => runScene(sceneOf("A", fire2), { nodes, horizon: 60, frameSeconds: 5, particles: 0, systems });
  const none = run(null);
  const flood = run({ rooms: [{ ...ROOM1, system: "clean-agent", params: { releaseAt: 0 } }] });
  const r = flood.systems.rooms[0];
  assert.equal(r.dischargeAt, 0);
  assert.ok(r.manual);
  // Room 2's own fire is untouched by the agent: the heat released is the free fire's within a few per cent
  // (the difference is vapour drifting into room 1, where it may not burn).
  const ratio = flood.stats.released / none.stats.released;
  assert.ok(ratio > 0.9 && ratio < 1.1, `released ${(flood.stats.released / 1e6).toFixed(2)} MJ against ${(none.stats.released / 1e6).toFixed(2)} MJ`);
  // And the agent's cells are the room's: none beyond its footprint.
  const scene = sceneOf("A", fire2);
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel });
  const sys = systemsRun(scene, sim, { rooms: [{ ...ROOM1, system: "clean-agent", params: { releaseAt: 0 } }] });
  sys.update(20);
  const g = scene.grid;
  for (let c = 0; c < sim.N; c += 1) {
    const i = c % g.nx, x = g.origin[0] + (i + 0.5) * g.h;
    const agent = sys.state[sys.room[c] * STATE.stride + STATE.agent];
    if (x > 5) assert.equal(agent, 0, `no agent at x = ${x.toFixed(2)}`);
    if (sys.room[c]) assert.equal(agent, 7);
  }
});

test("foam on a heptane pool: the blanket stops the pool's evaporation by the extinguishment time and the fire goes out", () => {
  // The pool alone: no room contents, so only the heptane burns. A pool's own
  // flame keeps its surface boiling, which the coarse grid cannot resolve
  // (the pool goes out once the pilot stops), so the pilot holds the
  // surface hot for the whole run. A 0.1 m2 spill of 5 kg: a larger pool
  // fills the room with more vapour than burns (the grid's rich-plume
  // limit), and that reservoir keeps burning for minutes after the blanket.
  const bare = () => {
    const { contentsFuel, ignitionFuel } = fuelsFor("B");
    const scene = buildScene(twoRooms(), { fire: [2.5, 0.75, 2], maxCells: 30000, minCell: 0.35, maxCell: 0.35, contentsFuel, ignitionFuel, item: { mass: 5, area: 0.1 } });
    scene.contents = { cells: [], masses: [] };
    scene.pilot = { ...scene.pilot, duration: 1e9 };
    return scene;
  };
  const scene = bare();
  const pool = scene.ignition.cell;
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1 });
  const sys = systemsRun(scene, sim, { rooms: [{ ...ROOM1, system: "foam" }] });
  sim.systems = sys;
  const rec = sys.rooms[0];
  let poolAtOut = null;
  for (let frame = 0; frame < 24; frame += 1) {
    const end = (frame + 1) * 5;
    while (sim.time < end - 1e-9) {
      const t = sim.time;
      step(sim, Math.min(sim.P.dt, end - t));
      if (t >= rec.t0 + rec.extinguish) {
        assert.equal(sim.solidLoss[pool], 0, `the pool gives off no vapour at ${t.toFixed(1)} s`);
        poolAtOut ??= sim.fuelMass[pool];
        assert.equal(sim.fuelMass[pool], poolAtOut, "the pool keeps its fuel under the blanket");
      }
    }
    systemsFrame(sys, { t1: end, vis: readNodes(sim, sys.detectorCells).vis });
  }
  assert.ok(rec.detectedAt !== null, "the foam room's detector responds");
  assert.ok(rec.t0 + rec.extinguish < 90, `out at ${rec.t0 + rec.extinguish} s`);
  assert.ok(poolAtOut > 1, `${poolAtOut} kg of heptane left under the foam`);
  assert.ok(rec.stats.withheld > 0);
  // The same pool with no foam burns on; under the foam, once the vapour
  // already in the room has burnt, the fire is out.
  const free = runScene(bare(), { nodes, horizon: 150, frameSeconds: 5, particles: 0 });
  const foamed = runScene(bare(), { nodes, horizon: 150, frameSeconds: 5, particles: 0, systems: { rooms: [{ ...ROOM1, system: "foam" }] } });
  const r = foamed.systems.rooms[0];
  assert.ok(r.outAt < 100 && r.controlAt < r.outAt);
  assert.ok(foamed.stats.released < 0.5 * free.stats.released, `released ${(foamed.stats.released / 1e6).toFixed(1)} MJ against ${(free.stats.released / 1e6).toFixed(1)} MJ`);
  const lastFrames = (out) => out.hrr.slice(-4).reduce((a, b) => a + b, 0);
  assert.ok(lastFrames(free) > 200, `the free pool still burns at the end: ${lastFrames(free).toFixed(0)} kW over the last frames`);
  assert.ok(lastFrames(foamed) < 0.1 * lastFrames(free), `the foamed pool is out: ${lastFrames(foamed).toFixed(2)} kW over the last frames`);
});

test("foam on class A contents: the solution's water wets the fuel it lands on", () => {
  const scene = sceneOf("A");
  const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1 });
  const sys = systemsRun(scene, sim, { rooms: [{ ...ROOM1, system: "foam", params: { releaseAt: 0 } }] });
  sim.systems = sys;
  const before = Float32Array.from(sim.moisture);
  for (let k = 0; k < 20; k += 1) step(sim, sim.P.dt);
  let wetter = 0;
  for (const c of sys.wet) if (sim.moisture[c] > before[c] + 1e-9) wetter += 1;
  assert.ok(sys.wet.length > 50 && wetter > 0.5 * sys.wet.length, `${wetter} of ${sys.wet.length} wet cells took water`);
  // Two seconds at 6.5 L/(min m2) on the wet cells' columns, the wetted share of it.
  const expected = sys.wet.length * (6.5 / 60) * scene.grid.h ** 2 * 2 * sys.rooms[0].wet;
  close(sys.rooms[0].stats.wetted, expected, 0.2 * expected, "water on the fuel (kg)");
});
