/**
 * Straighten many routes on a real model and check them independently.
 *
 *   node tools/straighten-check.mjs <file.ifc> [--routes 40] [--seed 1] [--dump routes.json]
 *
 * Routes join random pairs of nodes in the largest part (walls on). For each,
 * reports the lattice and straight lengths and the time taken, and checks
 * every straight segment against the wall sections with a finer test than
 * straighten.js uses (crossing at 2 cm steps, plus the distance to the nearest
 * outline), so a segment that slips through a wall shows up here.
 */
import { writeFileSync } from "node:fs";
import path from "node:path";

import { buildLatticeGraph, SegmentIndex } from "../topologicpy-web-frontend/src/core/lattice.js";
import { astar, componentLabels } from "../topologicpy-web-frontend/src/core/route.js";
import { straightenRoute } from "../topologicpy-web-frontend/src/core/straighten.js";
import { extract } from "./ifc-extract.mjs";

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--") && !/^\d/.test(a));
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
if (!file) {
  console.error("usage: node tools/straighten-check.mjs <file.ifc> [--routes 40] [--seed 1] [--dump routes.json]");
  process.exit(2);
}

const geo = await extract(path.resolve(file));
const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
const g = { ...built, nodes: built.points, upAxis: "y", agentHeight: 0.75 };
const n = g.nodes.length / 3;

let seed = Number(opt("--seed", 1)) >>> 0;
const rand = () => {
  seed = (seed + 0x6d2b79f5) >>> 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const { labels, sizes } = componentLabels(g);
const big = sizes.indexOf(Math.max(...sizes));
const pool = [];
for (let v = 0; v < n; v += 1) if (labels[v] === big) pool.push(v);

const sections = new Map(g.sections.map(({ key, segs }) => [key, new SegmentIndex(segs)]));
const sectionFor = (floor) => {
  const k = Math.round((floor + g.sectionHeight) * 10);
  return sections.get(k) || sections.get(k - 1) || sections.get(k + 1);
};

// Door links cross the wall on purpose where the opening is not cut out of
// it (lattice.js), so a segment that is itself a lattice link is not counted.
const linked = new Set();
for (let e = 0; e < g.edges.length / 2; e += 1) {
  const a = g.edges[e * 2], b = g.edges[e * 2 + 1];
  linked.add(Math.min(a, b) * 4294967296 + Math.max(a, b));
}

function check(points, ids) {
  let crossings = 0;
  let clearance = Infinity;
  for (let i = 1; i < points.length; i += 1) {
    const [ax, ay, az] = points[i - 1], [bx, by, bz] = points[i];
    if (Math.abs(ay - by) > 0.35) continue; // a stair link, drawn as found
    if (linked.has(Math.min(ids[i - 1], ids[i]) * 4294967296 + Math.max(ids[i - 1], ids[i]))) continue;
    const index = sectionFor(Math.min(ay, by) - 0.75);
    if (!index) continue;
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, bz - az) / 0.02));
    for (let k = 0; k < steps; k += 1) {
      const x0 = ax + ((bx - ax) * k) / steps, z0 = az + ((bz - az) * k) / steps;
      const x1 = ax + ((bx - ax) * (k + 1)) / steps, z1 = az + ((bz - az) * (k + 1)) / steps;
      if (index.crosses(x0, z0, x1, z1)) crossings += 1;
      const g2 = index.segs;
      index.near(x0 - 0.5, z0 - 0.5, x0 + 0.5, z0 + 0.5, (s) => {
        const cx = g2[s * 4], cz = g2[s * 4 + 1], dx = g2[s * 4 + 2] - cx, dz = g2[s * 4 + 3] - cz;
        const len2 = dx * dx + dz * dz;
        const t = len2 > 0 ? Math.max(0, Math.min(1, ((x0 - cx) * dx + (z0 - cz) * dz) / len2)) : 0;
        clearance = Math.min(clearance, Math.hypot(x0 - cx - t * dx, z0 - cz - t * dz));
      });
    }
  }
  return { crossings, clearance };
}

const want = Number(opt("--routes", 40));
const rows = [];
const dump = [];
for (let tries = 0; rows.length < want && tries < want * 5; tries += 1) {
  const a = pool[Math.floor(rand() * pool.length)], b = pool[Math.floor(rand() * pool.length)];
  if (a === b) continue;
  const r = astar(g, a, b);
  if (!r || r.nodes.length < 3) continue;
  const grid = r.nodes.map((i) => [g.nodes[i * 3], g.nodes[i * 3 + 1], g.nodes[i * 3 + 2]]);
  let gridLength = 0;
  for (let i = 1; i < grid.length; i += 1) gridLength += Math.hypot(...grid[i].map((v, k) => v - grid[i - 1][k]));
  const t0 = performance.now();
  const s = straightenRoute(g, r.nodes);
  const ms = performance.now() - t0;
  const { crossings, clearance } = check(s.points, s.node_ids);
  rows.push({ gridLength, length: s.length, vertices: s.points.length, gridVertices: grid.length, ms, crossings, clearance });
  dump.push({ grid, straight: s.points });
}

const sum = (f) => rows.reduce((acc, r) => acc + f(r), 0);
const sorted = (f) => rows.map(f).sort((x, y) => x - y);
const pct = (xs, q) => xs[Math.min(xs.length - 1, Math.floor(q * xs.length))];
const report = {
  file: path.basename(file),
  nodes: n,
  routes: rows.length,
  meanGridLength: sum((r) => r.gridLength) / rows.length,
  meanStraightLength: sum((r) => r.length) / rows.length,
  meanShortening: 1 - sum((r) => r.length) / sum((r) => r.gridLength),
  verticesGridToStraight: `${(sum((r) => r.gridVertices) / rows.length).toFixed(1)} -> ${(sum((r) => r.vertices) / rows.length).toFixed(1)}`,
  msMedian: pct(sorted((r) => r.ms), 0.5),
  msP95: pct(sorted((r) => r.ms), 0.95),
  msMax: Math.max(...rows.map((r) => r.ms)),
  routesCrossingAWall: rows.filter((r) => r.crossings > 0).length,
  minClearanceM: Math.min(...rows.map((r) => r.clearance)),
  longerThanGrid: rows.filter((r) => r.length > r.gridLength + 1e-9).length,
};
console.log(report);
const dumpAt = args.indexOf("--dump");
if (dumpAt >= 0) {
  writeFileSync(
    args[dumpAt + 1],
    JSON.stringify({
      file: report.file,
      points: Array.from(g.nodes, (v) => Math.round(v * 1000) / 1000),
      edges: Array.from(g.edges),
      blocked: Array.from(g.blocked),
      kinds: Array.from(g.kinds),
      walls: g.walls.map((w) => ({ segment: w.segment, thickness: w.thickness, up_min: w.up_min, up_max: w.up_max })),
      doors: g.doorReport,
      // Wall sections at walking height, per level, as drawn by plot_lattice.py.
      sections: g.sections.map(({ key, segs }) => ({ height: key / 10, segs: Array.from(segs, (v) => Math.round(v * 1000) / 1000) })),
      routes: dump.slice(0, Number(opt("--plot", 6))),
    }),
  );
}
