/**
 * The egress map and the accessibility audit on IFC models, headless
 * (topologicpy-web-frontend/src/core/egressRun.js, as the Route step runs them):
 * the graph's readings, then each storey's measuring grid, the 16-neighbour
 * walk against the graph's and against Fast Marching, and the wheelchair audit.
 *
 *   node tools/egress-map-check.mjs [model.ifc ...] [--cell 0.05] [--profile icc|metric] [--out file]
 *
 * Without models, the site examples. Results go to benchmark/results/<date>-egress-map.json.
 */
import { mkdir, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph, floorLevels } from "../topologicpy-web-frontend/src/core/lattice.js";
import { exitDoors } from "../topologicpy-web-frontend/src/core/ifcSemantics.js";
import { outdoorSpaces } from "../topologicpy-web-frontend/src/core/outdoorSpaces.js";
import { EGRESS_B } from "../topologicpy-web-frontend/src/core/codeCheck.js";
import { rooms, runFine, runMap } from "../topologicpy-web-frontend/src/core/egressRun.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc"));
if (!models.length) {
  const dir = path.join(HERE, "../topologicpy-web-frontend/public/examples");
  models = (await readdir(dir)).filter((f) => f.endsWith(".ifc")).map((f) => path.join(dir, f));
}
const cell = Number(flag("--cell", 0.05));
const profile = flag("--profile", "icc");
const r1 = (x) => (Number.isFinite(x) ? Math.round(x * 10) / 10 : x);

const results = [];
for (const file of models) {
  const name = path.basename(file);
  const geo = await extract(file, { withDoorFlags: true });
  // As App.jsx: roofs no door opens onto, and voids, are not floor (core/outdoorSpaces.js).
  const spaceNames = new Map((geo.semantics?.spaces || []).map((s) => [s.id, s.longName || s.name]));
  const outside = new Set(outdoorSpaces(geo.spaces, { names: spaceNames, overhead: [...(geo.floors || []), ...(geo.roofs || [])], ceilings: geo.ceilings || [], walls: geo.walls, doors: geo.doors, upAxis: "y" }).map((s) => s.id));
  const built = buildLatticeGraph({ ...geo, spaces: geo.spaces.filter((m) => !outside.has(m.itemId)), upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const graph = { nodes: built.points, edges: built.edges, blocked: built.blocked, kinds: built.kinds, doors: built.doorReport, spacing: 0.5, agentHeight: 0.75, upAxis: "y" };
  const n = graph.nodes.length / 3;
  const exits = exitDoors(built.doorReport, geo.semantics, graph);
  const exitNodes = exits.final?.length ? exits.final : exits.nodes;
  // As App.jsx: the slabs' levels, or, when they give one or none, the IFC's
  // storeys, shifted so the lowest meets the lowest walkable floor.
  // (Headless there is no coordination shift to place the IFC's storeys with:
  // the graph's own floor heights stand in, each storey the commonest height
  // in a band 1.5 m deep holding at least 50 floor nodes.)
  let levels = floorLevels(geo.floors, "y");
  if (levels.length <= 1) {
    const counts = new Map();
    for (let v = 0; v < n; v += 1) if (graph.kinds[v] === 0) {
      const h = Math.round((graph.nodes[v * 3 + 1] - 0.75) * 10) / 10;
      counts.set(h, (counts.get(h) || 0) + 1);
    }
    const heights = [...counts].sort((a, b) => a[0] - b[0]);
    const groups = [];
    for (const [h, c] of heights) {
      const g = groups.at(-1);
      if (g && h - g.lo < 1.5) {
        g.n += c;
        if (c > g.bestCount) Object.assign(g, { best: h, bestCount: c });
      } else groups.push({ lo: h, n: c, best: h, bestCount: c });
    }
    levels = groups.filter((g) => g.n >= 50).map((g) => g.best);
  }
  const names = new Map((geo.semantics?.spaces || []).map((s) => [s.id, s.longName || s.name]));
  const declared = new Map((geo.semantics?.doors || []).map((d) => [d.id, d.widthM]));
  const doorWidth = new Float32Array(n);
  for (const d of graph.doors || []) if (d.node >= 0) doorWidth[d.node] = declared.get(d.itemId) > 0 ? declared.get(d.itemId) : d.width || 0;
  const opts = { exits: exitNodes, levels, useWalls: true, limit: EGRESS_B.travel.unsprinklered, sprinklered: false, occupantArea: EGRESS_B.occupantArea.value, doorWidth, profile, cell, fastMarching: true };
  console.log(`\n${name}: ${n} nodes, ${exitNodes.length} exit nodes, ${levels.length} storeys, ${geo.spaces.length} spaces`);
  const t0 = performance.now();
  const roomList = rooms(geo, names, "y");
  const map = runMap(graph, opts, roomList);
  const mapMs = performance.now() - t0;
  const worst = map.rooms.find((r) => Number.isFinite(r.dist));
  console.log(`  map ${Math.round(mapMs)} ms: ${map.exits.filter((e) => e.live).length} exits, farthest ${r1(map.summary.farthest)} m, rooms ${map.rooms.length}, worst ${worst ? `${r1(worst.dist)} m (${worst.name || "unnamed"})` : "none"}`);
  console.log(`  sweep: ${map.sweep.single.map((s) => `${s.label} ${r1(s.farthest)} m/${s.roomsRerouted}`).join("; ")}`);
  console.log(`  cascade: ${map.sweep.cascade.map((c) => `-${c.label} ${r1(c.farthest)} m${c.unreachedArea ? ` (${Math.round(c.unreachedArea)} m² cut off)` : ""}`).join(" ")}`);
  if (map.deadEnds) console.log(`  one-way ${Math.round(map.deadEnds.oneWayShare * 100)} %; dead-end corridors ${map.deadEnds.corridors.length} (longest ${r1(map.deadEnds.corridors[0]?.length ?? 0)} m, over ${map.deadEnds.corridors.filter((c) => c.over).length}); rooms' common path longest ${r1(map.deadEnds.rooms[0]?.length ?? 0)} m, over ${map.deadEnds.rooms.filter((c) => c.over).length}`);
  console.log(`  critical doors ${map.critical.doors.length}, suites ${map.critical.suites.length}${map.critical.suites[0] ? ` (largest: ${map.critical.suites[0].rooms.length} rooms)` : ""}`);
  const t1 = performance.now();
  const fine = runFine(graph, opts, geo, roomList, names);
  const fineMs = performance.now() - t1;
  console.log(`  fine ${Math.round(fineMs)} ms:`);
  for (const s of fine.storeys) {
    const a = s.audit;
    const fm = s.fastMarching ? `, FM ${(s.fastMarching.mean * 100).toFixed(1)} % (max ${(s.fastMarching.max * 100).toFixed(1)} %)` : "";
    console.log(`    storey ${s.storey} at ${r1(s.level)} m, ${s.cell} m cells: walk ${Math.round(s.walkArea)} m² (body ${Math.round(s.bodyArea)}), farthest ${r1(s.farthest)} m against the graph's ${r1(s.graphFarthest)} m${fm}; no way out ${r1(s.noWayOut)} m²`);
    console.log(`      wheelchair: ${a.rooms.byChair} of ${a.rooms.onFoot} rooms reached on foot (${a.rooms.turnable} with turning space); issues ${a.issues.length} (${a.issues.filter((i) => i.severity === "high").length} high); act: ${a.act.map((x) => `${x.kind} ${x.label} +${x.gain}`).join(", ") || "none"}; accessible egress ok in ${a.egress.filter((e) => e.ok).length} of ${a.egress.length} rooms`);
  }
  results.push({
    model: name,
    nodes: n,
    storeys: levels.length,
    mapMs: Math.round(mapMs),
    fineMs: Math.round(fineMs),
    farthest: r1(map.summary.farthest),
    exits: map.exits.filter((e) => e.live).length,
    sweep: map.sweep.single.map((s) => ({ exit: s.label, farthest: r1(s.farthest), rerouted: s.roomsRerouted })),
    cascade: map.sweep.cascade.map((c) => ({ closed: c.label, farthest: r1(c.farthest), cutOff: Math.round(c.unreachedArea) })),
    deadEnds: map.deadEnds && { oneWay: map.deadEnds.oneWayShare, corridors: map.deadEnds.corridors.map((c) => ({ length: r1(c.length), width: c.width && r1(c.width), over: c.over, exempt: c.exempt, name: c.name })).slice(0, 10), roomsOver: map.deadEnds.rooms.filter((r) => r.over).length },
    critical: { doors: map.critical.doors.length, suites: map.critical.suites.length },
    fine: fine.storeys.map((s) => ({ storey: s.storey, cell: s.cell, walk: Math.round(s.walkArea), farthest: r1(s.farthest), graph: r1(s.graphFarthest), fastMarching: s.fastMarching, noWayOut: r1(s.noWayOut), audit: { ...s.audit.rooms, issues: s.audit.issues.length, act: s.audit.act.map((a) => `${a.kind} +${a.gain}`), egressOk: s.audit.egress.filter((e) => e.ok).length, egressRooms: s.audit.egress.length } })),
  });
}
const out = flag("--out", path.join(HERE, `../benchmark/results/${new Date().toISOString().slice(0, 10)}-egress-map.json`));
await mkdir(path.dirname(out), { recursive: true });
await writeFile(out, JSON.stringify({ date: new Date().toISOString(), cell, profile, results }, null, 1));
console.log(`\nwritten ${out}`);
