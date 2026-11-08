/**
 * Why do the heads open late? For fires in a model's largest rooms, the CPU
 * solver runs with the heads only opening (the fire is the free fire) and
 * records, for the heads in the fire's room: what the grid gives the
 * estimate (the sensor cell under the ceiling, the cell below it, the layer
 * the estimate adds Alpert's jet to), whether each head is in sight of the
 * plume, its distance from the fire on the grid and in the model, and when
 * it opened by the estimate and by the late reading. Beside them, the same
 * link driven by Alpert's jet over ambient air for the run's own heat
 * release at the head's true distance and the room's true height
 * (DETACT-QS): a head under a closed ceiling sees at least that, so an
 * estimate later than it means the grid's layer adds nothing.
 *
 *   node tools/activation-diagnose.mjs [model.ifc ...] [--fires 3] [--horizon 120]
 *        [--cells 30000] [--wall 200,25] [--frame 1] [--class A] [--json out.json]
 *
 * Without models, the Duplex example and the NIBS Office and Clinic.
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { buildScene, readNodes, sprinklerFrame, sprinklerRun } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { cellSpeed, createSolver, step } from "../topologicpy-web-frontend/src/core/flamescapes.js";
import { alpert, linkStep, crossing, SPEED_FLOOR } from "../topologicpy-web-frontend/src/core/suppression.js";
import { layoutRooms, roomsFromSpaces } from "../topologicpy-web-frontend/src/core/sprinklers.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES = path.join(HERE, "../topologicpy-web-frontend/public/examples");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const horizon = Number(flag("--horizon", 120));
const maxCells = Number(flag("--cells", 30000));
const fireCount = Number(flag("--fires", 3));
const frame = Number(flag("--frame", 1));
const walls = String(flag("--wall", "200,25")).split(",").map(Number);
const fireClass = flag("--class", "A");
const jsonOut = flag("--json", null);
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc"));
if (!models.length) models = ["duplex-apartment.ifc", "nibs-office.ifc", "nibs-clinic.ifc"].map((f) => path.join(EXAMPLES, f));

const KELVIN = 273.15;
const { contentsFuel, ignitionFuel, item } = fuelsFor(fireClass);
const report = [];

/** Whether a plan point lies in a room's rectangle (rect: centre, axis, half). */
function inRect(rect, x, z) {
  const da = x - rect.centre[0], db = z - rect.centre[1];
  const along = da * rect.axis[0] + db * rect.axis[1], across = -da * rect.axis[1] + db * rect.axis[0];
  return Math.abs(along) <= rect.half[0] + 1e-6 && Math.abs(across) <= rect.half[1] + 1e-6;
}

for (const file of models) {
  const geo = await extract(file, { withDoorFlags: true });
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const nodes = built.points;
  const n = nodes.length / 3;
  const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
  const rooms = geo.spaces.length ? roomsFromSpaces(geo.spaces, { upAxis: "y", names, overhead: [...geo.floors, ...(geo.roofs || [])], ceilings: geo.ceilings, walls: geo.walls }) : [];
  const heads = rooms.length ? layoutRooms(rooms).heads : [];
  console.log(`\n${path.basename(file)}: ${heads.length} heads in ${rooms.filter((r) => !r.problem).length} rooms; class ${fireClass}; ${horizon} s; frames of ${frame} s`);
  if (!heads.length) continue;
  const sceneGeo = slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo;
  const nearestNode = (p) => {
    let best = 0, bd = Infinity;
    for (let v = 0; v < n; v += 1) {
      const d = (nodes[v * 3] - p[0]) ** 2 + 4 * (nodes[v * 3 + 1] - p[1]) ** 2 + (nodes[v * 3 + 2] - p[2]) ** 2;
      if (d < bd) { bd = d; best = v; }
    }
    return best;
  };
  // The largest rooms with heads, a storey at a time (as tools/sprinkler-check.mjs --fires).
  const withHeads = new Set(heads.map((h) => h.room));
  const byStorey = new Map();
  for (const r of rooms.filter((r) => r.rect && !r.problem && withHeads.has(r.id)).sort((a, b) => b.area - a.area)) {
    const key = Math.round(r.floor * 2) / 2;
    if (!byStorey.has(key)) byStorey.set(key, []);
    byStorey.get(key).push(r);
  }
  const queues = [...byStorey.values()];
  const fires = [];
  for (let i = 0; fires.length < fireCount && queues.some((q) => q.length); i += 1) {
    const r = queues[i % queues.length].shift();
    if (r) fires.push(r);
  }

  for (const room of fires) {
    const v = nearestNode([room.rect.centre[0], room.floor + 0.75, room.rect.centre[1]]);
    const fire = [nodes[v * 3], nodes[v * 3 + 1], nodes[v * 3 + 2]];
    const height = room.ceiling - room.floor;
    console.log(`  fire in ${room.name || room.id}: ${room.area.toFixed(0)} m², ${height.toFixed(2)} m high`);
    for (const wall of walls) {
      const scene = buildScene(sceneGeo, { upAxis: "y", fire, maxCells, contentsFuel, ignitionFuel, item });
      const sim = createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, dt: 0.1, seed: 1, wallConvection: wall });
      const spr = sprinklerRun(scene, sim, { heads, mode: "activation" }, "y");
      sim.suppression = spr.sup;
      const g = scene.grid;
      const setup = spr.setup;
      // The fire room's heads: their index in the run, true distance and the room's true height.
      const mine = setup.heads.map((hd, h) => ({ hd, h })).filter(({ hd }) => hd.room === room.id || (hd.room == null && inRect(room.rect, hd.position[0], hd.position[2])));
      const Htrue = Math.max(0.5, room.ceiling - fire[1]);
      for (const m of mine) m.rTrue = Math.hypot(m.hd.position[0] - fire[0], m.hd.position[2] - fire[2]);
      mine.sort((a, b) => a.rTrue - b.rTrue);
      // DETACT-QS on the run's own heat release, per head of the room.
      const detact = new Map(mine.map((m) => [m.h, { T: sim.P.ambient, opened: null }]));
      const series = [];
      const early = []; // [t, frame heat release kW], every frame
      let released = 0, prevQ = 0;
      const K = Math.round(horizon / frame);
      for (let k = 0; k < K; k += 1) {
        const t1 = (k + 1) * frame;
        while (sim.time < t1 - 1e-9) step(sim, Math.min(0.1, t1 - sim.time));
        const hrr = (sim.stats.released - released) / frame / 1000;
        released = sim.stats.released;
        const T = Float64Array.from(readNodes(sim, spr.sensors).T, (x) => x + KELVIN);
        const u = Float64Array.from(spr.sensors, (c) => (c >= 0 ? cellSpeed(sim, c) : 0));
        // The cell under each sensor (one row down: j is the grid's up index).
        const Tb = Float64Array.from(setup.heads, (hd) => {
          if (hd.sensor < 0 || Math.floor(hd.sensor / g.nx) % g.ny === 0) return NaN;
          const c = hd.sensor - g.nx;
          return sim.kind[c] !== 2 ? sim.T[c] : NaN;
        });
        sprinklerFrame(spr, { t0: t1 - frame, t1, T, u, hrr });
        const Q = 0.5 * (prevQ + Math.max(0, hrr));
        prevQ = Math.max(0, hrr);
        early.push([t1, hrr]);
        for (const m of mine) {
          const d = detact.get(m.h);
          if (d.opened !== null) continue;
          const jet = alpert(Q, Htrue, Math.max(m.rTrue, 0.01));
          const s = linkStep(d.T, sim.P.ambient + jet.dT, sim.P.ambient, Math.max(SPEED_FLOOR, jet.U), m.hd.rti, m.hd.c, frame);
          const Ta = m.hd.activationC + KELVIN;
          if (s.T >= Ta && d.T < Ta) d.opened = t1 - frame + Math.min(frame, Math.max(0, crossing(d.T, s.Tinf, s.k, Ta)));
          d.T = s.T;
        }
        if (k % Math.max(1, Math.round(10 / frame)) === Math.max(1, Math.round(10 / frame)) - 1) {
          const near = mine.slice(0, 3).map((m) => ({ top: T[m.h] - KELVIN, below: Tb[m.h] - KELVIN }));
          series.push({ t: t1, Q, near });
        }
      }
      const state = spr.state;
      const rowsOut = mine.slice(0, 4).map((m) => ({
        r: m.hd.r ?? null,
        rTrue: m.rTrue,
        H: m.hd.H ?? null,
        inSight: m.hd.inSight,
        estimate: Number.isNaN(state.opened.estimate[m.h]) ? null : state.opened.estimate[m.h],
        qEstimate: Number.isNaN(state.hrrAt.estimate[m.h]) ? null : state.hrrAt.estimate[m.h],
        late: Number.isNaN(state.opened.late[m.h]) ? null : state.opened.late[m.h],
        detact: detact.get(m.h).opened,
      }));
      const firstAny = Math.min(...setup.heads.map((_, h) => (Number.isNaN(state.opened.estimate[h]) ? Infinity : state.opened.estimate[h])));
      const res = {
        model: path.basename(file),
        room: room.name || room.id,
        area: room.area,
        height,
        wall,
        cell: g.h,
        cells: g.nx * g.ny * g.nz,
        clipped: Boolean(scene.clipped),
        plumeH: setup.plume?.H ?? null,
        Htrue,
        headsInRoom: mine.length,
        inSight: mine.filter((m) => m.hd.inSight).length,
        firstAny: Number.isFinite(firstAny) ? firstAny : null,
        heads: rowsOut,
        series,
        hrr: early,
        releasedMJ: sim.stats.released / 1e6,
      };
      report.push(res);
      const f = (x, d = 1) => (x === null || x === undefined ? "  -  " : x.toFixed(d));
      console.log(
        `    wall ${String(wall).padStart(3)}: cell ${g.h.toFixed(2)} m, ${res.cells} cells${res.clipped ? " (clipped)" : ""}; plume H ${f(res.plumeH, 2)} m (room ${Htrue.toFixed(2)}); ` +
          `${res.headsInRoom} heads in the room, ${res.inSight} in sight; first head anywhere ${f(res.firstAny)} s; released ${res.releasedMJ.toFixed(0)} MJ`,
      );
      for (const r of rowsOut) {
        console.log(
          `      r ${f(r.r, 2)} (true ${r.rTrue.toFixed(2)}) H ${f(r.H, 2)} ${r.inSight ? "in sight " : "not seen "} estimate ${f(r.estimate)} s at ${f(r.qEstimate, 0)} kW; late ${f(r.late)} s; Alpert alone ${f(r.detact)} s`,
        );
      }
      const at = (t) => early.find(([tt]) => tt >= t - 1e-9)?.[1] ?? NaN;
      console.log(`      growth: ${[2, 5, 10, 15, 20, 30].map((t) => `${t} s ${Math.round(at(t))} kW`).join(", ")}`);
      for (const s of series.filter((_, i) => i % 2 === 1)) {
        console.log(`      ${String(s.t).padStart(4)} s  Q ${Math.round(s.Q).toString().padStart(5)} kW  ` + s.near.map((x) => `top ${Math.round(x.top)} C below ${Math.round(x.below)} C`).join(" | "));
      }
    }
  }
}
if (jsonOut) await writeFile(jsonOut, JSON.stringify(report, null, 1));
