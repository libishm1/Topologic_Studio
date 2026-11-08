/**
 * Stage 0 of the calibration plan (wiki/research/flamescapes-calibration-plan-2026-11-08.md,
 * section 4): the same fire on several cells of the app's ladder, on the CPU,
 * with the first item as a design fire (transport alone: the heat release is
 * prescribed) and burnt by the solver. Every run covers the same box round
 * the fire (`--reach` m square), so only the cell changes.
 *
 * Recorded per run: the heat release curve and the energy by the horizon;
 * three thermocouple trees in the fire's room, from 0.3 m to under the
 * ceiling, as the rise above ambient; the layer's temperature rise and depth
 * by Janssens and Tran's (1992) reduction, as the FDS Validation Guide
 * reduces measured trees; the lowest visibility at 1.8 m; the hottest tree
 * point; D* / dx at the peak heat release; and the drawn flame points by band
 * of the viewer's glow scale (plan P8: the fire's colouring is kept).
 *
 *   node tools/grid-study.mjs [--scenes room,duplex] [--cells 0.20,0.27,0.34]
 *        [--modes design,solver] [--seeds 1] [--horizon 120] [--reach 9] [--json out.json]
 *
 * Each quantity is then compared with its value on the finest cell, against
 * FDS's model scatter for it (FDS Validation Guide 6.11.1, Table 16.1).
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildScene, runScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { DEFAULTS } from "../topologicpy-web-frontend/src/core/flamescapes.js";
import { MATERIAL } from "../topologicpy-web-frontend/src/core/voxelize.js";
import { roomsFromSpaces } from "../topologicpy-web-frontend/src/core/sprinklers.js";
import { designFireFor, fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES = path.join(HERE, "../topologicpy-web-frontend/public/examples");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const list = (name, fallback) => flag(name, fallback).split(",").map((s) => s.trim()).filter(Boolean);
const scenes = list("--scenes", "room,duplex");
const cells = list("--cells", "0.20,0.27,0.34").map(Number);
const modes = list("--modes", "design,solver");
const seeds = list("--seeds", "1").map(Number);
const horizon = Number(flag("--horizon", 120));
const reach = Number(flag("--reach", 9));
const jsonOut = flag("--json", null);
const FRAME = 5;
const AMBIENT_C = DEFAULTS.ambient - 273.15;

/** FDS's model relative standard deviation for each compared quantity (Validation Guide 6.11.1, Table 16.1). */
const FDS_SCATTER = { layerRise: 0.10, layerDepth: 0.14, hottestRise: 0.13, visibility: 0.26, energy: 0.28, peakHrr: 0.28 };

function box(x0, y0, z0, x1, y1, z1) {
  const p = [x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: MATERIAL.STONE };
}

/** The solver tests' 5 x 4 x 2.7 m room with a door (src/core/suppression.test.mjs), y up. */
function testRoom(t = 0.2) {
  const W = 5, D = 4, H = 2.7;
  return {
    geometry: {
      walls: [box(-t, 0, -t, W + t, H, 0), box(-t, 0, D, W + t, H, D + t), box(-t, 0, 0, 0, H, D), box(W, 0, 0, W + t, H, 1.5), box(W, 0, 2.4, W + t, H, D), box(W, 2.0, 1.5, W + t, H, 2.4)],
      floors: [box(-t, -t, -t, W + t, 0, D + t)],
      roofs: [box(-t, H, -t, W + t, H + t, D + t)],
      stairs: [],
      obstacles: [],
    },
    name: "the 5 x 4 x 2.7 m test room",
    floor: 0,
    ceiling: H,
    fire: [2.5, 0.75, 2],
    trees: [[1.5, 2], [3.5, 1], [2.5, 3.3]],
  };
}

/** The Duplex's living room, its fire at the room's middle, trees at a third of the way to three sides. */
async function duplex() {
  const geo = await extract(path.join(EXAMPLES, "duplex-apartment.ifc"), { withDoorFlags: true });
  const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
  const rooms = roomsFromSpaces(geo.spaces, { upAxis: "y", names, overhead: [...geo.floors, ...(geo.roofs || [])], ceilings: geo.ceilings, walls: geo.walls });
  const room = rooms.filter((r) => r.rect && !r.problem && /living/i.test(r.name || "")).sort((a, b) => a.floor - b.floor || b.area - a.area)[0];
  if (!room) throw new Error("No living room found in the Duplex.");
  const { centre: [a, b], axis: [ua, ub], half: [hu, hv] } = room.rect;
  const at = (s, t) => [a + s * hu * ua - t * hv * ub, b + s * hu * ub + t * hv * ua];
  const geometry = slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo;
  return {
    geometry,
    name: `the Duplex's ${room.name} (${room.area.toFixed(0)} m², ${(room.ceiling - room.floor).toFixed(2)} m high)`,
    floor: room.floor,
    ceiling: room.ceiling,
    fire: [a, room.floor + 0.75, b],
    trees: [at(0.33, 0), at(-0.33, 0), at(0, 0.33)],
  };
}

/** Tree points from 0.3 m over the floor to 0.1 m under the ceiling, every 0.2 m, in the viewer frame. */
function treePoints(sc) {
  const heights = [];
  for (let z = 0.3; z <= sc.ceiling - sc.floor - 0.1 + 1e-9; z += 0.2) heights.push(+z.toFixed(2));
  const points = [];
  for (const [x, zz] of sc.trees) for (const h of heights) points.push(x, sc.floor + h, zz);
  return { heights, points: Float32Array.from(points) };
}

/**
 * Janssens and Tran's (1992) layer reduction of one tree: heights (m over the
 * floor) and temperatures (K), the ceiling at H. The lower layer takes the
 * lowest point's temperature; z_int = T_l (I1 I2 - H^2) / (I1 + I2 T_l^2 -
 * 2 T_l H) with I1 the integral of T and I2 of 1/T from the floor to H; the
 * upper layer is T's mean above z_int.
 */
export function janssensTran(heights, T, H) {
  const z = [0, ...heights, H];
  const t = [T[0], ...T, T[T.length - 1]];
  let I1 = 0, I2 = 0;
  for (let i = 1; i < z.length; i += 1) {
    const dz = z[i] - z[i - 1];
    I1 += 0.5 * (t[i] + t[i - 1]) * dz;
    I2 += 0.5 * (1 / t[i] + 1 / t[i - 1]) * dz;
  }
  const Tl = t[0];
  const denom = I1 + I2 * Tl * Tl - 2 * Tl * H;
  let zi = Math.abs(denom) > 1e-12 ? (Tl * (I1 * I2 - H * H)) / denom : H;
  zi = Math.min(H, Math.max(0, zi));
  let upper = 0, span = 0;
  for (let i = 1; i < z.length; i += 1) {
    const lo = Math.max(z[i - 1], zi), hi = z[i];
    if (hi <= lo) continue;
    const f = (x) => t[i - 1] + ((t[i] - t[i - 1]) * (x - z[i - 1])) / (z[i] - z[i - 1]);
    upper += 0.5 * (f(lo) + f(hi)) * (hi - lo);
    span += hi - lo;
  }
  return { height: zi, depth: H - zi, upper: span > 0 ? upper / span : Tl };
}

/** The viewer's glow bands (ViewerManager.js glowColor: dull red at 800 K to yellow-white at 1800 K). */
const BANDS = [800, 1050, 1300, 1550, 1800];
function bandShares(particles) {
  const counts = new Array(BANDS.length + 1).fill(0);
  let n = 0;
  for (let q = 0; q < particles.kinds.length; q += 1) {
    if (particles.kinds[q] !== 0) continue;
    const T = particles.temps[q];
    let b = 0;
    while (b < BANDS.length && T >= BANDS[b]) b += 1;
    counts[b] += 1;
    n += 1;
  }
  return { points: n, shares: counts.map((c) => (n ? +(c / n).toFixed(3) : 0)) };
}

/** D* / dx at a heat release Q (kW) on cells of dx (m), with D* = (Q / (rho cp T sqrt(g)))^(2/5). */
const plumeIndex = (Q, dx) => (Q > 0 ? (Q / (1.204 * 1.005 * 293.15 * Math.sqrt(9.81))) ** 0.4 / dx : 0);

const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
const rows = [];
for (const name of scenes) {
  const sc = name === "room" ? testRoom() : name === "duplex" ? await duplex() : null;
  if (!sc) throw new Error(`Unknown scene ${name}`);
  const { heights, points } = treePoints(sc);
  const perTree = heights.length;
  console.log(`\n${name}: ${sc.name}; trees of ${perTree} points; box ${reach} m; ${horizon} s`);
  for (const mode of modes) {
    for (const seed of seeds) {
      for (const h of cells) {
        const t0 = performance.now();
        // maxCell just under the cell forces the clip to the same box at every cell; maxCells never coarsens it.
        const scene = buildScene(sc.geometry, {
          upAxis: "y", fire: sc.fire, minCell: h, maxCell: h - 1e-3, reach, maxCells: 5e6,
          contentsFuel, ignitionFuel, item, seed, designFire: mode === "design" ? designFireFor("A") : null,
        });
        const g = scene.grid;
        const colours = {};
        const out = runScene(scene, {
          nodes: points, agentHeight: 1.8, headHeight: 1.8, horizon, frameSeconds: FRAME, particles: 3000,
          onFrame: (f) => {
            if (f.particles && (Math.abs(f.time - 60) < 1e-6 || Math.abs(f.time - horizon) < 1e-6)) colours[f.time] = bandShares(f.particles);
          },
        });
        const ms = performance.now() - t0;
        const K = out.K, n = points.length / 3;
        const energy = out.hrr.reduce((s, q) => s + q * FRAME, 0) / 1000; // MJ
        const peak = Math.max(...out.hrr);
        const at = (time) => {
          const k = Math.min(K - 1, Math.round(time / FRAME) - 1);
          const T = out.T.subarray(k * n, (k + 1) * n);
          const vis = out.vis.subarray(k * n, (k + 1) * n);
          const layers = sc.trees.map((_, i) => janssensTran(heights, Array.from(T.subarray(i * perTree, (i + 1) * perTree), (c) => c + 273.15), sc.ceiling - sc.floor));
          const head = heights.reduce((best, z, j) => (Math.abs(z - 1.8) < Math.abs(heights[best] - 1.8) ? j : best), 0);
          return {
            layerRise: +(layers.reduce((s, l) => s + l.upper - 273.15 - AMBIENT_C, 0) / layers.length).toFixed(2),
            layerDepth: +(layers.reduce((s, l) => s + l.depth, 0) / layers.length).toFixed(3),
            hottestRise: +(Math.max(...T) - AMBIENT_C).toFixed(2),
            visibility: +Math.min(...sc.trees.map((_, i) => vis[i * perTree + head])).toFixed(2),
          };
        };
        const row = {
          scene: name, mode, seed, cell: +h.toFixed(3), grid: [g.nx, g.ny, g.nz], cells: g.cells, wallSeconds: +(ms / 1000).toFixed(1),
          energyMJ: +energy.toFixed(2), peakHrrKw: +peak.toFixed(1), plumeIndexAtPeak: +plumeIndex(peak, h).toFixed(2),
          hrrKw: Array.from(out.hrr, (q) => +q.toFixed(1)),
          at60: at(60), atEnd: at(horizon), colours,
        };
        rows.push(row);
        console.log(
          `  ${mode} seed ${seed} cell ${h.toFixed(2)} m (${g.nx}x${g.ny}x${g.nz}, ${(ms / 1000).toFixed(0)} s): ` +
            `${energy.toFixed(1)} MJ, peak ${peak.toFixed(0)} kW (D*/dx ${row.plumeIndexAtPeak}); at ${horizon} s layer +${row.atEnd.layerRise} K, ` +
            `depth ${row.atEnd.layerDepth} m, hottest +${row.atEnd.hottestRise} K, visibility ${row.atEnd.visibility} m; ` +
            `flame points ${colours[horizon]?.points ?? 0}`,
        );
      }
    }
  }
}

// Each quantity against its value on the finest cell, as a share, beside FDS's model scatter for it.
const compare = [];
for (const name of scenes) {
  for (const mode of modes) {
    for (const seed of seeds) {
      const runs = rows.filter((r) => r.scene === name && r.mode === mode && r.seed === seed).sort((a, b) => a.cell - b.cell);
      if (runs.length < 2) continue;
      const fine = runs[0];
      for (const r of runs.slice(1)) {
        const change = {};
        // A share of a tiny value means nothing: a rise is compared from 5 K, the layer's depth only
        // under a layer that warm, visibility only between 1 and 30 m (it is capped at 100 m).
        const comparable = {
          energy: (a) => a > 0.1,
          peakHrr: (a) => a > 5,
          layerRise: (a) => a >= 5,
          layerDepth: () => fine.atEnd.layerRise >= 5,
          hottestRise: (a) => a >= 5,
          visibility: (a) => a >= 1 && a <= 30,
        };
        for (const [key, value] of [["energy", (x) => x.energyMJ], ["peakHrr", (x) => x.peakHrrKw], ["layerRise", (x) => x.atEnd.layerRise], ["layerDepth", (x) => x.atEnd.layerDepth], ["hottestRise", (x) => x.atEnd.hottestRise], ["visibility", (x) => x.atEnd.visibility]]) {
          const a = value(fine), b = value(r);
          const ok = comparable[key](a);
          change[key] = { fine: a, coarse: b, change: ok ? +((b - a) / Math.abs(a)).toFixed(3) : null, fdsScatter: FDS_SCATTER[key], within: ok ? Math.abs((b - a) / a) <= FDS_SCATTER[key] : null };
        }
        compare.push({ scene: name, mode, seed, fine: fine.cell, coarse: r.cell, change });
        console.log(`\n${name} ${mode} seed ${seed}: ${r.cell} m against ${fine.cell} m`);
        for (const [key, c] of Object.entries(change)) console.log(`  ${key.padEnd(12)} ${String(c.fine).padStart(9)} -> ${String(c.coarse).padStart(9)}  ${c.change == null ? "n/a" : `${(100 * c.change).toFixed(1)} %`}  (FDS scatter ${(100 * c.fdsScatter).toFixed(0)} %)${c.within === false ? "  OUT" : ""}`);
      }
    }
  }
}

if (jsonOut) {
  await writeFile(jsonOut, JSON.stringify({ generated: new Date().toISOString(), plan: "wiki/research/flamescapes-calibration-plan-2026-11-08.md, section 4", horizon, reach, frameSeconds: FRAME, fdsScatter: FDS_SCATTER, rows, compare }, null, 1));
  console.log(`\nwrote ${jsonOut}`);
}
