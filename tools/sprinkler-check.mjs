/**
 * Do the sprinklers work? The same fire on a model, run by the CPU solver
 * with no sprinklers and with them in each mode the Simulate step offers,
 * side by side: how many heads opened and when, the heat released and the
 * fuel burnt, the peak and final heat release, the head-height gas near the
 * fire, and the water's balance.
 *
 *   node tools/sprinkler-check.mjs [model.ifc ...] [--horizon 180] [--class A] [--cells 30000] [--fires N]
 *
 * Without models, the Duplex example. Heads are placed per room by NFPA 13
 * (core/sprinklers.js) on the model's IfcSpace rooms, as the app does.
 * What "working" means here: a head over the fire opens; with "activation"
 * the fire is exactly the free fire; with the cap, the spray or wetting the
 * covered fuel burns less and the fire's heat after the opening is lower.
 */
import { readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { buildScene, runScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { layoutRooms, roomsFromSpaces } from "../topologicpy-web-frontend/src/core/sprinklers.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const horizon = Number(flag("--horizon", 180));
const maxCells = Number(flag("--cells", 30000));
const fireClass = flag("--class", "A");
// --fires N: a fire in each of the N largest rooms with heads, a storey at a time.
const fireCount = Number(flag("--fires", 0));
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc"));
if (!models.length) models = [path.join(HERE, "../topologicpy-web-frontend/public/examples/duplex-apartment.ifc")];

/** The node nearest the middle of the lowest storey (as tools/fire-bench.mjs). */
function seedNode(nodes) {
  const n = nodes.length / 3;
  let lo = Infinity;
  for (let v = 0; v < n; v += 1) lo = Math.min(lo, nodes[v * 3 + 1]);
  let sx = 0, sz = 0, c = 0;
  for (let v = 0; v < n; v += 1) if (nodes[v * 3 + 1] < lo + 0.5) { sx += nodes[v * 3]; sz += nodes[v * 3 + 2]; c += 1; }
  let best = 0, bd = Infinity;
  for (let v = 0; v < n; v += 1) {
    const d = (nodes[v * 3] - sx / c) ** 2 + (nodes[v * 3 + 2] - sz / c) ** 2 + 100 * Math.abs(nodes[v * 3 + 1] - lo);
    if (d < bd) { bd = d; best = v; }
  }
  return best;
}

const { contentsFuel, ignitionFuel, item } = fuelsFor(fireClass);
for (const file of models) {
  const geo = await extract(file, { withDoorFlags: true });
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const nodes = built.points;
  const n = nodes.length / 3;
  const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
  const rooms = geo.spaces.length ? roomsFromSpaces(geo.spaces, { upAxis: "y", names, overhead: [...geo.floors, ...(geo.roofs || [])], ceilings: geo.ceilings, walls: geo.walls }) : [];
  const heads = rooms.length ? layoutRooms(rooms).heads : [];
  const left = rooms.filter((r) => r.problem);
  console.log(`\n${path.basename(file)}: ${n} nodes, ${heads.length} heads in ${rooms.length - left.length} rooms (${left.length} spaces left out); class ${fireClass}; ${horizon} s`);
  if (!heads.length) {
    console.log("  no IfcSpace rooms: no heads to place");
    continue;
  }
  // Where the fires are: the middle of the lowest storey, or the middles of
  // the largest rooms with heads, a storey at a time.
  const nearestNode = (p) => {
    let best = 0, bd = Infinity;
    for (let v = 0; v < n; v += 1) {
      const d = (nodes[v * 3] - p[0]) ** 2 + 4 * (nodes[v * 3 + 1] - p[1]) ** 2 + (nodes[v * 3 + 2] - p[2]) ** 2;
      if (d < bd) { bd = d; best = v; }
    }
    return best;
  };
  const fires = [];
  if (fireCount > 0) {
    const withHeads = new Set(heads.map((h) => h.room));
    const byStorey = new Map();
    for (const r of rooms.filter((r) => r.rect && !r.problem && withHeads.has(r.id)).sort((a, b) => b.area - a.area)) {
      const key = Math.round(r.floor * 2) / 2;
      if (!byStorey.has(key)) byStorey.set(key, []);
      byStorey.get(key).push(r);
    }
    const queues = [...byStorey.values()];
    for (let i = 0; fires.length < fireCount && queues.some((q) => q.length); i += 1) {
      const r = queues[i % queues.length].shift();
      if (r) fires.push({ node: nearestNode([r.rect.centre[0], r.floor + 0.75, r.rect.centre[1]]), where: `${r.name || r.id} (${r.area.toFixed(0)} m², floor at ${r.floor.toFixed(1)} m)` });
    }
  } else {
    fires.push({ node: seedNode(nodes), where: "the middle of the lowest storey" });
  }
  for (const { node: seed, where } of fires) {
  const fire = [nodes[seed * 3], nodes[seed * 3 + 1], nodes[seed * 3 + 2]];
  console.log(`  fire in ${where}`);
  // The nodes within 3 m of the fire on its storey: the gas there at head height.
  const near = [];
  for (let v = 0; v < n; v += 1) {
    if (Math.abs(nodes[v * 3 + 1] - fire[1]) < 0.5 && Math.hypot(nodes[v * 3] - fire[0], nodes[v * 3 + 2] - fire[2]) <= 3) near.push(v);
  }
  const rows = [];
  // "activation" first: when its first head opens is the time every run's
  // "after the opening" is read from, the free fire's included.
  let open = null;
  for (const mode of ["activation", null, "evans", "spray", "wetting"]) {
    const scene = buildScene(slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo, { upAxis: "y", fire, maxCells, contentsFuel, ignitionFuel, item });
    let peak = 0, peakAfter = 0, nearMax = 0, nearAfter = 0;
    const t0 = performance.now();
    const out = runScene(scene, {
      nodes,
      horizon,
      frameSeconds: 5,
      particles: 0,
      sprinklers: mode ? { heads, mode } : null,
      onFrame: (f) => {
        if (f.hrr > peak) peak = f.hrr;
      },
    });
    const ms = performance.now() - t0;
    const s = out.sprinklers;
    const first = s ? Math.min(...s.heads.map((h) => (h.estimate ?? Infinity))) : null;
    if (mode === "activation") open = first;
    for (let k = 0; k < out.K; k += 1) {
      const t = (k + 1) * out.frameSeconds;
      let hot = 0;
      for (const v of near) hot = Math.max(hot, out.T[k * n + v]);
      nearMax = Math.max(nearMax, hot);
      if (open !== null && Number.isFinite(open) && t > open) {
        peakAfter = Math.max(peakAfter, out.hrr[k]);
        nearAfter = Math.max(nearAfter, hot);
      }
    }
    rows.push({
      mode: mode || "none",
      opened: s ? s.heads.filter((h) => h.estimate !== null).length : 0,
      first: s ? first : null,
      releasedMJ: out.stats.released / 1e6,
      burntKg: out.stats.burnt,
      peakKw: peak,
      peakAfterKw: peakAfter,
      endKw: out.hrr[out.K - 1],
      nearMaxC: nearMax,
      nearAfterC: nearAfter,
      water: s?.stats ?? null,
      clipped: scene.clipped,
      seconds: ms / 1000,
    });
    const r = rows[rows.length - 1];
    console.log(
      `  ${r.mode.padEnd(10)} opened ${String(r.opened).padStart(2)}${r.first !== null && Number.isFinite(r.first) ? ` (first ${r.first.toFixed(1)} s)` : "            "}  ` +
        `released ${r.releasedMJ.toFixed(1).padStart(6)} MJ  burnt ${r.burntKg.toFixed(2).padStart(6)} kg  peak ${Math.round(r.peakKw).toString().padStart(5)} kW  ` +
        `after opening ${Math.round(r.peakAfterKw).toString().padStart(5)} kW  at the end ${Math.round(r.endKw).toString().padStart(5)} kW  ` +
        `gas within 3 m ${Math.round(r.nearMaxC)} C (after ${Math.round(r.nearAfterC)} C)` +
        (r.water ? `  water ${r.water.emitted.toFixed(0)} kg (evaporated ${r.water.evaporated.toFixed(1)}, on the fuel ${(r.water.wetted ?? 0).toFixed(1)})` : "") +
        `  [${r.seconds.toFixed(0)} s]`,
    );
  }
  const act = rows[0], none = rows[1];
  const same = Math.abs(none.releasedMJ - act.releasedMJ) < 1e-9 && Math.abs(none.burntKg - act.burntKg) < 1e-12;
  console.log(`  activation alone leaves the fire as it is: ${same ? "yes" : "NO"}`);
  for (const r of rows.slice(2)) {
    console.log(
      `  ${r.mode.padEnd(10)} against no sprinklers: fuel burnt ${((r.burntKg / none.burntKg - 1) * 100).toFixed(0)} %, heat released ${((r.releasedMJ / none.releasedMJ - 1) * 100).toFixed(0)} %, peak after the opening ${((r.peakAfterKw / Math.max(1e-9, none.peakAfterKw) - 1) * 100).toFixed(0)} %, heat at the end ${((r.endKw / Math.max(1e-9, none.endKw) - 1) * 100).toFixed(0)} %`,
    );
  }
  }
}
