/**
 * Runs the Flamescapes fire solver on an IFC model as the site does
 * (src/core/flamescapesRun.js) and prints what it gives the planners: the
 * grid, the heat release per frame, how many graph nodes are hot, smoky or
 * untenable, and the time per step.
 *
 *   node tools/flamescapes-check.mjs [model.ifc] [--embers] [--horizon 120] [--cells 30000] [--fire x,y,z]
 *
 * Without --fire the fire starts at the graph node nearest the middle of the
 * largest room's floor.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { buildScene, forecastFromFrames, runScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const ifc = args.find((a) => a.endsWith(".ifc")) || path.join(HERE, "../topologicpy-web-frontend/public/examples/duplex-apartment.ifc");
const embers = args.includes("--embers");
const horizon = Number(flag("--horizon", 120));
const maxCells = Number(flag("--cells", 30000));
const solverOptions = {};
if (flag("--h")) solverOptions.convection = Number(flag("--h"));
const itemMass = Number(flag("--item", 5));

// The fuels the app uses for the class (--class, A by default), from fireClasses.js.
const fuels = fuelsFor(flag("--class", "A"));

const geo = await extract(ifc);
const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
const graph = { nodes: built.points, edges: built.edges };
const n = graph.nodes.length / 3;
let fire;
if (flag("--fire")) fire = flag("--fire").split(",").map(Number);
else {
  // The node nearest the centroid of the graph's lowest storey.
  let lo = Infinity;
  for (let v = 0; v < n; v += 1) lo = Math.min(lo, graph.nodes[v * 3 + 1]);
  let sx = 0, sz = 0, c = 0;
  for (let v = 0; v < n; v += 1) if (graph.nodes[v * 3 + 1] < lo + 0.5) { sx += graph.nodes[v * 3]; sz += graph.nodes[v * 3 + 2]; c += 1; }
  let best = 0, bd = Infinity;
  for (let v = 0; v < n; v += 1) {
    const d = (graph.nodes[v * 3] - sx / c) ** 2 + (graph.nodes[v * 3 + 2] - sz / c) ** 2 + 100 * Math.abs(graph.nodes[v * 3 + 1] - lo);
    if (d < bd) { bd = d; best = v; }
  }
  fire = [graph.nodes[best * 3], graph.nodes[best * 3 + 1], graph.nodes[best * 3 + 2]];
}

const t0 = performance.now();
const scene = buildScene(geo, { upAxis: "y", fire, maxCells, contentsFuel: fuels.contentsFuel, ignitionFuel: fuels.ignitionFuel, item: { ...fuels.item, ...(flag("--item") ? { mass: itemMass } : {}) } });
const g = scene.grid;
console.log(`${path.basename(ifc)}: ${n} graph nodes; grid ${g.nx}x${g.ny}x${g.nz} = ${g.cells} cells of ${g.h.toFixed(2)} m${scene.clipped ? " (clipped round the fire)" : ""}; contents ${scene.contentsSummary.mass.toFixed(0)} kg on ${scene.contentsSummary.area.toFixed(0)} m2; scene ${(performance.now() - t0).toFixed(0)} ms`);
console.log(`fire at ${fire.map((x) => x.toFixed(2)).join(", ")}; embers ${embers ? "on" : "off"}`);
const t1 = performance.now();
const frames = runScene(scene, {
  nodes: graph.nodes,
  upAxis: "y",
  horizon,
  frameSeconds: 5,
  embers,
  solverOptions,
  onFrame: (f) => {
    let hot = 0, smoky = 0, maxT = -Infinity;
    for (let v = 0; v < f.T.length; v += 1) {
      if (f.T[v] >= 60) hot += 1;
      if (f.vis[v] < 10) smoky += 1;
      if (f.T[v] > maxT) maxT = f.T[v];
    }
    console.log(`t ${String(f.time).padStart(4)} s  HRR ${f.hrr.toFixed(0).padStart(6)} kW  hottest node ${maxT.toFixed(0).padStart(4)} C  nodes >=60 C ${String(hot).padStart(5)}  vis<10 m ${String(smoky).padStart(5)}  embers ${f.embers} spots ${f.spots}  ${((performance.now() - t1) / f.time / 10).toFixed(1)} ms/step`);
  },
});
const forecast = forecastFromFrames(frames, { threshold: 60, smoke: true, visibilityLimit: 10 });
let closed = 0;
for (let v = 0; v < n; v += 1) if (forecast.untenable[v] < forecast.horizon) closed += 1;
console.log(`untenable within ${forecast.horizon} s: ${closed} of ${n} nodes; nodes in the grid ${frames.inGrid}; total ${((performance.now() - t1) / 1000).toFixed(1)} s`);
