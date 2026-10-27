/**
 * Build the lattice navigation graph from an IFC file in Node, without the
 * browser, and report how it hangs together.
 *
 *   node tools/lattice-check.mjs <file.ifc> [--spacing 0.5] [--json]
 *
 * Uses web-ifc's Node build for the geometry and src/core/lattice.js for the
 * graph, so it exercises exactly the code the site runs. Reports components
 * with and without walls, which separates "the walls cut the floor" from "the
 * floor itself is in pieces".
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildLatticeGraph, wallSegments } from "../topologicpy-web-frontend/src/core/lattice.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const WebIFC = require(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/web-ifc-api-node.js"));

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--"));
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
if (!file) {
  console.error("usage: node tools/lattice-check.mjs <file.ifc> [--spacing 0.5] [--json]");
  process.exit(2);
}

// Coverings are floors only when typed FLOORING: Office A's ceilings, typed
// CEILING, were otherwise the only "floors" at 2.55 m above each storey.
const GROUPS = {
  floors: ["IFCSLAB", "IFCPLATE", "IFCCOVERING"],
  stairs: ["IFCSTAIR", "IFCSTAIRFLIGHT", "IFCRAMP", "IFCRAMPFLIGHT"],
  doors: ["IFCDOOR"],
  walls: ["IFCWALL", "IFCWALLSTANDARDCASE"],
  spaces: ["IFCSPACE"],
  // Solid at walking height besides walls: cut with them (see lattice.js).
  obstacles: ["IFCWINDOW", "IFCRAILING", "IFCCOLUMN", "IFCCURTAINWALL", "IFCMEMBER", "IFCPLATE"],
};
const isWalkableCovering = (api, model, id) => {
  const value = api.GetLine(model, id)?.PredefinedType?.value;
  return value === "FLOORING";
};

export async function extract(ifcPath) {
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/"), true);
  await api.Init();
  const model = api.OpenModel(new Uint8Array(readFileSync(ifcPath)), { COORDINATE_TO_ORIGIN: true });
  const out = {};
  for (const [group, names] of Object.entries(GROUPS)) {
    const types = names.map((n) => WebIFC[n]).filter((t) => typeof t === "number");
    const meshes = [];
    api.StreamAllMeshesWithTypes(model, types, (mesh) => {
      if (
        api.GetLineType(model, mesh.expressID) === WebIFC.IFCCOVERING &&
        !isWalkableCovering(api, model, mesh.expressID)
      ) {
        return;
      }
      const placed = mesh.geometries;
      for (let g = 0; g < placed.size(); g += 1) {
        const pg = placed.get(g);
        const geom = api.GetGeometry(model, pg.geometryExpressID);
        const verts = api.GetVertexArray(geom.GetVertexData(), geom.GetVertexDataSize());
        const index = api.GetIndexArray(geom.GetIndexData(), geom.GetIndexDataSize());
        const m = pg.flatTransformation;
        const n = verts.length / 6;
        const positions = new Float32Array(n * 3);
        for (let v = 0; v < n; v += 1) {
          const x = verts[v * 6], y = verts[v * 6 + 1], z = verts[v * 6 + 2];
          positions[v * 3] = m[0] * x + m[4] * y + m[8] * z + m[12];
          positions[v * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
          positions[v * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        }
        const type = api.GetNameFromTypeCode(api.GetLineType(model, mesh.expressID));
        meshes.push({ positions, indices: Uint32Array.from(index), itemId: mesh.expressID, type });
        geom.delete?.();
      }
    });
    out[group] = meshes;
  }
  api.CloseModel(model);
  return out;
}

function componentSummary(g) {
  const n = g.stats.nodes;
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    if (!g.blocked[e]) parent[find(g.edges[e * 2])] = find(g.edges[e * 2 + 1]);
  }
  const groups = new Map();
  for (let i = 0; i < n; i += 1) {
    const r = find(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(i);
  }
  const parts = [...groups.values()].sort((a, b) => b.length - a.length);
  return {
    parts: parts.length,
    largestShare: parts.length ? parts[0].length / n : 0,
    top: parts.slice(0, 6).map((p) => {
      const ys = p.map((i) => g.points[i * 3 + 1]);
      return { nodes: p.length, height: [Math.min(...ys).toFixed(2), Math.max(...ys).toFixed(2)] };
    }),
  };
}

const t0 = Date.now();
const geo = await extract(path.resolve(file));
const loadMs = Date.now() - t0;
const spacing = opt("--spacing", 0.5);
const common = { ...geo, upAxis: "y", spacing, agentHeight: 0.75 };
const g = buildLatticeGraph(common);
const bare = buildLatticeGraph({ ...common, walls: [], doors: [] });
const walls = wallSegments(geo.walls, "y");
const thick = walls.map((w) => w.thickness).sort((a, b) => a - b);
const pct = (q) => (thick.length ? thick[Math.min(thick.length - 1, Math.floor(q * thick.length))].toFixed(2) : "-");

const report = {
  file: path.basename(file),
  meshes: Object.fromEntries(Object.entries(geo).map(([k, v]) => [k, v.length])),
  extractMs: loadMs,
  stats: g.stats,
  withWalls: componentSummary(g),
  withoutWalls: componentSummary(bare),
  wallThickness: { p50: pct(0.5), p90: pct(0.9), max: pct(1) },
  thickWalls: walls
    .filter((w) => w.thickness > 0.6)
    .map((w) => ({ thickness: +w.thickness.toFixed(2), length: +Math.hypot(w.segment[1][0] - w.segment[0][0], w.segment[1][1] - w.segment[0][1]).toFixed(2) })),
};
// Which component each door's node sits in, and how many nodes each side links.
{
  const n = g.stats.nodes;
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    if (!g.blocked[e]) parent[find(g.edges[e * 2])] = find(g.edges[e * 2 + 1]);
  }
  const size = new Map();
  for (let i = 0; i < n; i += 1) size.set(find(i), (size.get(find(i)) || 0) + 1);
  report.doors = g.doorReport.map((d) => ({
    at: [+d.x.toFixed(2), +d.y.toFixed(2)],
    sill: +d.sill.toFixed(2),
    width: +d.width.toFixed(2),
    added: d.added,
    sides: d.sides,
    partSize: d.node >= 0 ? size.get(find(d.node)) : 0,
  }));
  // Where the parts lie in plan.
  report.parts = [...size.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([root, count]) => {
      let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      for (let i = 0; i < n; i += 1) {
        if (find(i) !== root) continue;
        x0 = Math.min(x0, g.points[i * 3]);
        x1 = Math.max(x1, g.points[i * 3]);
        z0 = Math.min(z0, g.points[i * 3 + 2]);
        z1 = Math.max(z1, g.points[i * 3 + 2]);
      }
      return { nodes: count, x: [+x0.toFixed(1), +x1.toFixed(1)], z: [+z0.toFixed(1), +z1.toFixed(1)] };
    });
}
// --floors: per floor-group mesh, its IFC type, height range and how much of
// its triangle area faces up versus down. Finds slabs lost to winding.
if (args.includes("--floors")) {
  const summary = new Map();
  for (const mesh of geo.floors) {
    const p = mesh.positions, ix = mesh.indices;
    let upArea = 0, downArea = 0, lo = Infinity, hi = -Infinity;
    for (let t = 0; t + 2 < ix.length; t += 3) {
      const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
      const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
      const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
      const ny = e1[2] * e2[0] - e1[0] * e2[2];
      const area = 0.5 * Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], ny, e1[0] * e2[1] - e1[1] * e2[0]);
      if (area > 0 && ny / (2 * area) > 0.98) upArea += area;
      if (area > 0 && ny / (2 * area) < -0.98) downArea += area;
      for (const o of [a, b, c]) {
        lo = Math.min(lo, p[o + 1]);
        hi = Math.max(hi, p[o + 1]);
      }
    }
    const k = `${mesh.type} ${lo.toFixed(1)}..${hi.toFixed(1)}`;
    const s = summary.get(k) || { meshes: 0, upArea: 0, downArea: 0 };
    s.meshes += 1;
    s.upArea += upArea;
    s.downArea += downArea;
    summary.set(k, s);
  }
  report.floorMeshes = [...summary.entries()]
    .sort((x, y) => parseFloat(x[0].split(" ")[1]) - parseFloat(y[0].split(" ")[1]))
    .map(([k, s]) => `${k}  x${s.meshes}  up ${s.upArea.toFixed(0)} m2  down ${s.downArea.toFixed(0)} m2`);
}

// --dump <file.json>: the graph itself, for plotting (tools/plot_lattice.py).
const dumpAt = args.indexOf("--dump");
if (dumpAt >= 0) {
  const { writeFileSync } = await import("node:fs");
  writeFileSync(
    args[dumpAt + 1],
    JSON.stringify({
      file: report.file,
      points: Array.from(g.points),
      edges: Array.from(g.edges),
      blocked: Array.from(g.blocked),
      kinds: Array.from(g.kinds),
      walls: walls.map((w) => ({ segment: w.segment, thickness: w.thickness, up_min: w.up_min, up_max: w.up_max })),
      doors: g.doorReport,
    }),
  );
}
if (args.includes("--json")) console.log(JSON.stringify(report));
else console.dir(report, { depth: 5 });
