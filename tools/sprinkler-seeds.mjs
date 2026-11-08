/**
 * How much of a sprinkler run's difference from the free fire is the fire's
 * own scatter? One fire, the room contents laid out with several seeds
 * (buildScene's `seed`), with no sprinklers and in one mode: the heat
 * released and the fuel burnt for each.
 *
 *   node tools/sprinkler-seeds.mjs model.ifc --room "Living Room" --mode spray --seeds 1,2,3 [--horizon 120]
 */
import path from "node:path";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { buildScene, runScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { layoutRooms, roomsFromSpaces } from "../topologicpy-web-frontend/src/core/sprinklers.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const file = args.find((a) => a.toLowerCase().endsWith(".ifc"));
const roomName = flag("--room", null);
const mode = flag("--mode", "spray");
const seeds = flag("--seeds", "1,2,3").split(",").map(Number);
const horizon = Number(flag("--horizon", 120));
const geo = await extract(file, { withDoorFlags: true });
const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
const nodes = built.points, n = nodes.length / 3;
const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
const rooms = roomsFromSpaces(geo.spaces, { upAxis: "y", names, overhead: [...geo.floors, ...(geo.roofs || [])], ceilings: geo.ceilings, walls: geo.walls });
const heads = layoutRooms(rooms).heads;
const room = rooms.filter((r) => r.rect && !r.problem && (!roomName || r.name === roomName)).sort((a, b) => b.area - a.area)[0];
let seedNode = 0, bd = Infinity;
for (let v = 0; v < n; v += 1) {
  const d = (nodes[v * 3] - room.rect.centre[0]) ** 2 + 4 * (nodes[v * 3 + 1] - room.floor - 0.75) ** 2 + (nodes[v * 3 + 2] - room.rect.centre[1]) ** 2;
  if (d < bd) { bd = d; seedNode = v; }
}
const fire = [nodes[seedNode * 3], nodes[seedNode * 3 + 1], nodes[seedNode * 3 + 2]];
const fireGeo = slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo;
const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
console.log(`${path.basename(file)}, fire in ${room.name}; ${mode} against none; seeds ${seeds.join(", ")}; ${horizon} s`);
for (const seed of seeds) {
  const row = [];
  for (const m of [null, mode]) {
    const scene = buildScene(fireGeo, { upAxis: "y", fire, contentsFuel, ignitionFuel, item, seed });
    const out = runScene(scene, { nodes, horizon, frameSeconds: 5, particles: 0, sprinklers: m ? { heads, mode: m } : null });
    row.push({ released: out.stats.released / 1e6, burnt: out.stats.burnt, peak: Math.max(...out.hrr), open: out.sprinklers ? out.sprinklers.heads.filter((h) => h.estimate !== null).length : 0 });
  }
  const [a, b] = row;
  console.log(`  seed ${seed}: none ${a.released.toFixed(1)} MJ ${a.burnt.toFixed(2)} kg peak ${Math.round(a.peak)} kW | ${mode} ${b.released.toFixed(1)} MJ ${b.burnt.toFixed(2)} kg peak ${Math.round(b.peak)} kW, ${b.open} open | change ${((b.released / a.released - 1) * 100).toFixed(0)} %`);
}
