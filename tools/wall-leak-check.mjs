/**
 * Does smoke cross walls, and where? A closed 5 x 4 x 2.7 m room (no door),
 * its walls `--thick` m thick and turned `--angles` degrees to the grid, on
 * the app's cells, with a fast design fire in the middle and no other
 * contents (about 170 kW at 60 s). Three checks
 * tell the causes apart:
 *
 *   gaps     the grid's open cells inside the room joined to the outside
 *            through the walls (a 6-connected flood from the room's middle);
 *            none means the walls are sealed on the grid
 *   solver   heat and soot outside the room's walls after `--horizon` s,
 *            against the room's own (the advection's interpolation or a gap
 *            can carry them across)
 *   drawing  the smoke cloud the viewer draws (flamescapesRun.js smokeCloud):
 *            blocks drawn outside the room's walls, against the soot the
 *            solver holds there
 *
 *   node tools/wall-leak-check.mjs [--cells 0.20,0.27,0.34] [--thick 0.1,0.2,0.3]
 *        [--angles 0,15,30,45] [--horizon 60] [--json out.json]
 */
import { writeFile } from "node:fs/promises";
import { buildScene, smokeCloud } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { DEFAULTS, NS, SOLID, SPECIES, createSolver, step } from "../topologicpy-web-frontend/src/core/flamescapes.js";
import { MATERIAL } from "../topologicpy-web-frontend/src/core/voxelize.js";
import { designFireFor, fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const nums = (name, fallback) => flag(name, fallback).split(",").map(Number);
const cells = nums("--cells", "0.20,0.27,0.34");
const thicks = nums("--thick", "0.1,0.2,0.3");
const angles = nums("--angles", "0,15,30,45");
const horizon = Number(flag("--horizon", 60));
const jsonOut = flag("--json", null);
const W = 5, D = 4, H = 2.7, CX = 2.5, CZ = 2;

/** An axis-aligned box turned `deg` about the room's vertical axis. */
function box(x0, y0, z0, x1, y1, z1, deg) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const a = (deg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a);
  for (let q = 0; q < p.length; q += 3) {
    const x = p[q] - CX, z = p[q + 2] - CZ;
    p[q] = CX + cos * x - sin * z;
    p[q + 2] = CZ + sin * x + cos * z;
  }
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: MATERIAL.STONE };
}

function closedRoom(t, deg) {
  return {
    walls: [box(-t, 0, -t, W + t, H, 0, deg), box(-t, 0, D, W + t, H, D + t, deg), box(-t, 0, 0, 0, H, D, deg), box(W, 0, 0, W + t, H, D, deg)],
    floors: [box(-t, -t, -t, W + t, 0, D + t, deg)],
    roofs: [box(-t, H, -t, W + t, H + t, D + t, deg)],
    stairs: [],
    obstacles: [],
  };
}

/** Where a point lies, in the room's own frame: inside, in the walls, or outside them. */
function where(x, y, z, t, deg) {
  const a = (-deg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a);
  const u = CX + cos * (x - CX) - sin * (z - CZ), v = CZ + sin * (x - CX) + cos * (z - CZ);
  if (u > 0 && u < W && v > 0 && v < D && y > 0 && y < H) return "inside";
  if (u > -t && u < W + t && v > -t && v < D + t && y > -t && y < H + t) return "wall";
  return "outside";
}

const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
const rows = [];
for (const h of cells) {
  for (const t of thicks) {
    for (const deg of angles) {
      const scene = buildScene(closedRoom(t, deg), {
        upAxis: "y", fire: [CX, 0.75, CZ], minCell: h, maxCell: h - 1e-3, reach: 9, maxCells: 5e6,
        contentsFuel, ignitionFuel, item, designFire: designFireFor("A", "fast"), fireLoad: 0,
      });
      const g = scene.grid;
      const centreOf = (c) => {
        const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
        return [g.origin[0] + (i + 0.5) * g.h, g.origin[1] + (j + 0.5) * g.h, g.origin[2] + (k + 0.5) * g.h];
      };
      const zone = new Uint8Array(g.cells); // 0 inside, 1 wall, 2 outside
      for (let c = 0; c < g.cells; c += 1) {
        const [x, y, z] = centreOf(c);
        const w = where(x, y, z, t, deg);
        zone[c] = w === "inside" ? 0 : w === "wall" ? 1 : 2;
      }
      const sim = createSolver(scene, { contentsFuel, ignitionFuel, dt: DEFAULTS.dt, seed: 1 });

      // Gaps: a 6-connected flood of open cells from the room's middle.
      const start = Math.floor((CX - g.origin[0]) / g.h) + g.nx * (Math.floor((1.2 - g.origin[1]) / g.h) + g.ny * Math.floor((CZ - g.origin[2]) / g.h));
      const seen = new Uint8Array(g.cells);
      const stack = [start];
      seen[start] = 1;
      let reachedOutside = 0;
      while (stack.length) {
        const c = stack.pop();
        if (zone[c] === 2) reachedOutside += 1;
        const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
        for (const [di, dj, dk] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
          const a = i + di, b = j + dj, e = k + dk;
          if (a < 0 || b < 0 || e < 0 || a >= g.nx || b >= g.ny || e >= g.nz) continue;
          const n = a + g.nx * (b + g.ny * e);
          if (seen[n] || sim.kind[n] === SOLID) continue;
          seen[n] = 1;
          stack.push(n);
        }
      }

      // Solver: run the fire, then heat and soot inside against outside.
      while (sim.time < horizon - 1e-9) step(sim, Math.min(DEFAULTS.dt, horizon - sim.time));
      const Ta = sim.P.ambient;
      let insideSoot = 0, insideRise = 0, insideN = 0, outsideSootMax = 0, outsideRiseMax = 0, outsideSootSum = 0;
      for (let c = 0; c < g.cells; c += 1) {
        if (sim.kind[c] === SOLID) continue;
        const soot = sim.rho[c] * sim.Y[c * NS + SPECIES.SOOT];
        const rise = sim.T[c] - Ta;
        if (zone[c] === 0) {
          insideSoot += soot;
          insideRise += rise;
          insideN += 1;
        } else if (zone[c] === 2) {
          outsideSootMax = Math.max(outsideSootMax, soot);
          outsideRiseMax = Math.max(outsideRiseMax, rise);
          outsideSootSum += soot;
        }
      }

      // Drawing: smoke-cloud blocks drawn outside the walls.
      const cloud = smokeCloud(g, sim.rho, sim.Y, sim.kind);
      let drawnOutside = 0, drawnTotal = cloud.opacity.length, drawnOutsideOpacity = 0;
      for (let q = 0; q < drawnTotal; q += 1) {
        const p = [0, 0, 0];
        for (let a = 0; a < 3; a += 1) p[a] = cloud.positions[q * 3 + g.axes[a]];
        if (where(p[0], p[1], p[2], t, deg) === "outside") {
          drawnOutside += 1;
          drawnOutsideOpacity = Math.max(drawnOutsideOpacity, cloud.opacity[q]);
        }
      }
      const row = {
        cell: h, thick: t, angle: deg, grid: [g.nx, g.ny, g.nz],
        gapCellsOutside: reachedOutside,
        insideSootMean: insideN ? insideSoot / insideN : 0, insideRiseMean: insideN ? +(insideRise / insideN).toFixed(2) : 0,
        outsideSootMax, outsideSootShare: insideSoot > 0 ? outsideSootSum / insideSoot : 0, outsideRiseMax: +outsideRiseMax.toFixed(2),
        drawnTotal, drawnOutside, drawnOutsideOpacity,
      };
      rows.push(row);
      console.log(
        `cell ${h.toFixed(2)} m, wall ${t} m at ${String(deg).padStart(2)}°: gaps ${reachedOutside ? `${reachedOutside} outside cells reached` : "sealed"}; ` +
          `inside +${row.insideRiseMean} K; outside max +${row.outsideRiseMax} K, soot outside ${(100 * row.outsideSootShare).toFixed(2)} % of inside; ` +
          `drawn ${drawnOutside} of ${drawnTotal} smoke blocks outside the walls (max opacity ${drawnOutsideOpacity}/255)`,
      );
    }
  }
}
if (jsonOut) {
  await writeFile(jsonOut, JSON.stringify({ generated: new Date().toISOString(), horizon, room: { W, D, H }, rows }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
