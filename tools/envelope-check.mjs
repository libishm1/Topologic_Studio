/**
 * Does smoke leave the building when every door and window is closed?
 *
 * The Flamescapes solver treats a closed door or window as a solid: the
 * door leaf fills its opening and the window stays in the obstacles
 * (flamescapesRun.js buildScene). Whether the closed building is sealed
 * also depends on the voxels: a wall or leaf thinner than a cell is solid
 * only where enough of the cell is covered. This tool builds the scene with
 * the doors open and with them all closed (windows closed both times) and
 * reports, for each:
 *
 *   - whether a path of gas cells joins the fire's cell to the edge of the
 *     grid, the outside air (a leak in the closed envelope, if so);
 *   - after `--horizon` s of the CPU solver, the share of the soot in the
 *     outside air (gas cells joined to the grid's edge without passing the
 *     building's interior).
 *
 *   node tools/envelope-check.mjs [model.ifc ...] [--horizon 60] [--cells 30000]
 */
import { readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { buildScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { createSolver, step, SOLID, SPECIES, NS } from "../topologicpy-web-frontend/src/core/flamescapes.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const horizon = Number(flag("--horizon", 60));
const maxCells = Number(flag("--cells", 30000));
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc"));
if (!models.length) {
  const dir = path.join(HERE, "../topologicpy-web-frontend/public/examples");
  models = (await readdir(dir)).filter((f) => /duplex|office|vienna/i.test(f)).map((f) => path.join(dir, f));
}

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

/** Gas cells joined to the grid's edge: the outside air, by a flood fill through non-solid cells. */
function outsideAir(sim) {
  const { nx, ny, nz } = sim.grid;
  const seen = new Uint8Array(sim.N);
  const queue = [];
  const push = (i, j, k) => {
    const c = i + nx * (j + ny * k);
    if (seen[c] || sim.kind[c] === SOLID) return;
    seen[c] = 1;
    queue.push(c);
  };
  for (let i = 0; i < nx; i += 1) for (let j = 0; j < ny; j += 1) { push(i, j, 0); push(i, j, nz - 1); }
  for (let i = 0; i < nx; i += 1) for (let k = 0; k < nz; k += 1) { push(i, 0, k); push(i, ny - 1, k); }
  for (let j = 0; j < ny; j += 1) for (let k = 0; k < nz; k += 1) { push(0, j, k); push(nx - 1, j, k); }
  while (queue.length) {
    const c = queue.pop();
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    if (i > 0) push(i - 1, j, k);
    if (i < nx - 1) push(i + 1, j, k);
    if (j > 0) push(i, j - 1, k);
    if (j < ny - 1) push(i, j + 1, k);
    if (k > 0) push(i, j, k - 1);
    if (k < nz - 1) push(i, j, k + 1);
  }
  return seen;
}

const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
for (const file of models) {
  const geo = await extract(file, { withDoorFlags: true });
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const seed = seedNode(built.points);
  const fire = [built.points[seed * 3], built.points[seed * 3 + 1], built.points[seed * 3 + 2]];
  console.log(`\n${path.basename(file)}: ${(geo.doors || []).length} door meshes, ${(geo.obstacles || []).length} obstacle meshes (windows among them)`);
  const sceneFor = (doors) =>
    buildScene({ walls: geo.walls, floors: geo.floors, roofs: geo.roofs || [], stairs: geo.stairs, obstacles: geo.obstacles, ...(doors === "closed" ? { doors: geo.doors } : {}) }, { upAxis: "y", fire, maxCells, contentsFuel, ignitionFuel, item });
  // The outside air, from the sealed building: gas joined to the grid's edge
  // with every door and window shut. Used for both runs (same grid).
  const closed = sceneFor("closed");
  const air = outsideAir(createSolver(closed, { contentsFuel, ignitionFuel }));
  const fireCell = closed.ignition?.cell ?? -1;
  console.log(`  grid ${closed.grid.nx}x${closed.grid.ny}x${closed.grid.nz} (${closed.grid.h.toFixed(2)} m)${closed.clipped ? ", clipped round the fire: its sides cut the building and are open to ambient air" : ", the whole building"}; ` +
    `the closed building's fire cell joined to the outside air: ${fireCell < 0 ? "no fire cell" : air[fireCell] || air[fireCell + closed.grid.nx] ? "yes (a leak)" : "no"}`);
  for (const doors of ["open", "closed"]) {
    const sim = createSolver(doors === "closed" ? closed : sceneFor("open"), { contentsFuel, ignitionFuel });
    const t0 = performance.now();
    while (sim.time < horizon - 1e-9) step(sim, Math.min(sim.P.dt, horizon - sim.time));
    let inside = 0, outside = 0;
    for (let c = 0; c < sim.N; c += 1) {
      if (sim.kind[c] === SOLID) continue;
      const m = sim.Y[c * NS + SPECIES.SOOT] * sim.rho[c] * sim.V;
      if (air[c]) outside += m;
      else inside += m;
    }
    console.log(
      `  doors ${doors.padEnd(6)} (windows closed): soot after ${horizon} s ${(inside * 1000).toFixed(1)} g inside, ${(outside * 1000).toFixed(1)} g in the outside air (${((outside / Math.max(inside + outside, 1e-12)) * 100).toFixed(1)}%), not counting what left the grid  [${((performance.now() - t0) / 1000).toFixed(1)} s]`,
    );
  }
}
