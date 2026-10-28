/**
 * Do vertical cores join the storeys, and would they if the treads were not
 * modelled? (src/core/cores.js)
 *
 *   node tools/core-check.mjs model.ifc [...]
 *
 * For each model, the graph is built five ways: as before cores (stairs by
 * IFC type only, no cores); as the site builds it (stair parts and cores,
 * lifts off); with lifts on; with every stair mesh removed and no cores; and
 * with every stair mesh removed but the rooms named as stairs kept as cores.
 * For each it reports how many storey levels (floorLevels) the best-joined
 * part of the graph spans, and the share of floor nodes in parts spanning two
 * or more levels. For the last it also compares routes: from sampled floor
 * nodes, the shortest route to a floor node on another level, as the mean
 * ratio of assumed-stair route to modelled-stair route. Then each core that
 * was not left alone for its modelled treads, with its outcome.
 */
import path from "node:path";

import { buildLatticeGraph, floorLevels } from "../topologicpy-web-frontend/src/core/lattice.js";
import { CORE_STAIR } from "../topologicpy-web-frontend/src/core/cores.js";
import { distancesFrom } from "../topologicpy-web-frontend/src/core/route.js";
import { extract } from "./ifc-extract.mjs";

const STAIR_TYPES = new Set(["IFCSTAIR", "IFCSTAIRFLIGHT", "IFCRAMP", "IFCRAMPFLIGHT"]);
const AGENT = 0.75;

/** Levels spanned by the best-joined part, and the share of floor in parts spanning two or more. */
function spread(g, levels) {
  const n = g.points.length / 3;
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  for (let e = 0; e < g.edges.length; e += 2) if (!g.blocked[e / 2]) parent[find(g.edges[e])] = find(g.edges[e + 1]);
  const parts = new Map();
  for (let i = 0; i < n; i += 1) {
    if (g.kinds[i] !== 0) continue;
    const z = g.points[i * 3 + 1] - AGENT;
    const level = levels.findIndex((h) => Math.abs(z - h) < 0.4);
    const r = find(i);
    let part = parts.get(r);
    if (!part) parts.set(r, (part = { count: 0, levels: new Map() }));
    part.count += 1;
    if (level >= 0) part.levels.set(level, (part.levels.get(level) || 0) + 1);
  }
  let floor = 0, multi = 0, most = 0;
  for (const part of parts.values()) {
    floor += part.count;
    const spanned = [...part.levels.values()].filter((c) => c >= 20).length;
    if (spanned >= 2) multi += part.count;
    most = Math.max(most, spanned);
  }
  return { levels: most, multiPct: floor ? Math.round((1000 * multi) / floor) / 10 : 0 };
}

/** From sampled floor nodes, the shortest route (walls on) to a floor node more than 2 m higher or lower. */
function toOtherLevel(g, starts = 12) {
  const graph = { nodes: g.points, edges: g.edges, blocked: g.blocked };
  const floor = [];
  for (let i = 0; i < g.points.length / 3; i += 1) if (g.kinds[i] === 0) floor.push(i);
  const out = [];
  for (let s = 0; s < starts && floor.length; s += 1) {
    const src = floor[Math.floor(((s + 0.5) / starts) * floor.length)];
    const dist = distancesFrom(graph, src, { blocked: g.blocked });
    const h = g.points[src * 3 + 1];
    let best = Infinity;
    for (const m of floor) if (Math.abs(g.points[m * 3 + 1] - h) > 2.0 && dist[m] < best) best = dist[m];
    out.push(best);
  }
  return out;
}

for (const file of process.argv.slice(2).filter((a) => a.toLowerCase().endsWith(".ifc"))) {
  let geo;
  try {
    geo = await extract(path.resolve(file), { withDoorFlags: false });
  } catch (error) {
    console.log(`${path.basename(file)}: could not read (${error.message})`);
    continue;
  }
  const levels = floorLevels(geo.floors, "y");
  const common = { upAxis: "y", spacing: 0.5, agentHeight: AGENT };
  const typed = geo.stairs.filter((m) => STAIR_TYPES.has(String(m.type).toUpperCase()));
  const rooms = geo.cores.filter((c) => c.kind === CORE_STAIR && c.source === "IfcSpace name");
  const before = buildLatticeGraph({ ...geo, ...common, stairs: typed, cores: [], coreMeshes: [] });
  const site = buildLatticeGraph({ ...geo, ...common });
  const lifts = buildLatticeGraph({ ...geo, ...common, useLifts: true });
  const bare = buildLatticeGraph({ ...geo, ...common, stairs: [], cores: [], coreMeshes: [] });
  const assumed = buildLatticeGraph({ ...geo, ...common, stairs: [], cores: rooms, coreMeshes: [] });
  const [b, s, l, n, a] = [before, site, lifts, bare, assumed].map((g) => spread(g, levels));
  const real = toOtherLevel(site), guess = toOtherLevel(assumed);
  const pairs = real.map((x, i) => [x, guess[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const ratio = pairs.length ? (pairs.reduce((sum, [x, y]) => sum + y / x, 0) / pairs.length).toFixed(2) : "-";
  console.log(
    `${path.basename(file)}: ${levels.length} levels; joined ${b.levels} before cores, ${s.levels} now, ${l.levels} with lifts; ` +
      `floor in multi-level parts ${b.multiPct}% / ${s.multiPct}% / ${l.multiPct}%; ${geo.stairs.length - typed.length} stair-part meshes`,
  );
  console.log(
    `    treads removed: ${n.levels} levels joined (${n.multiPct}%); with ${rooms.length} stair rooms as cores ${a.levels} (${a.multiPct}%), ` +
      `routes ${ratio} times the modelled stair's over ${pairs.length} starts`,
  );
  for (const core of lifts.cores) {
    const off = site.cores.find((c) => c.itemIds[0] === core.itemIds[0]);
    if (off?.reason === "modelled stair") continue;
    const outcome = core.kind === "stair" ? `${off?.linked ?? 0} storeys joined` : `${core.linked} storeys joined when on`;
    console.log(`    ${core.kind} "${core.names[0] || ""}" (${core.sources.join(", ")}): ${core.levels} levels beside it, ${outcome}${off?.reason ? `; ${off.reason}` : ""}`);
  }
  const left = site.cores.filter((c) => c.reason === "modelled stair").length;
  if (left) console.log(`    ${left} stair cores left alone: their treads are modelled`);
}
