/**
 * LOD1 models through the egress pipeline, headless (src/core/lod1.js):
 *
 *   1. parse the model (a TopologicPy CellComplex with its graphs, a GeoJSON
 *      plan, or an IFC's spaces, doors and stairs: the IfcSpace fast path);
 *   2. convert it to egress geometry (lod1ToEgressGeometry);
 *   3. build the lattice graph from it (lattice.js, as tools/fire-bench.mjs);
 *   4. report rooms, levels, doors, links, nodes, parts (components with
 *      walls on), whether every storey is in the main part, the exits found
 *      (ifcSemantics.js exitDoors, as the site picks them), and the route from
 *      the top floor's guest room furthest from an exit to the nearest exit,
 *      with its length;
 *   5. time each stage;
 *   6. build the Flamescapes scene (flamescapesRun.js buildScene) from the
 *      geometry, the fire in that guest room, to check it voxelises; with
 *      --horizon, run the solver that long and count non-finite values
 *      (doors open, or with --closed shut, openings left open);
 *   7. build the two-zone model's room graph (rooms.js buildRoomGraph);
 *   8. write the model as GeoJSON, read it back and run it through the
 *      pipeline again: the same rooms, doors, links, area, parts and route.
 *
 * For an IFC the full geometry's graph is built too (tools/ifc-extract.mjs),
 * with the route from the same room, for comparison.
 *
 *   node tools/lod1-check.mjs [input ...] [--spacing 0.5] [--cells 30000]
 *                             [--horizon 0] [--closed] [--out file] [--no-out]
 *
 * An input is a folder holding lod1_building.json (graphs/circulation_graph.json
 * and graphs/adjacency_graph.json beside it are read too), that file, a GeoJSON
 * plan (.geojson, or a .json FeatureCollection), or an .ifc. Without inputs:
 * every folder holding a lod1_building.json under the folder named by
 * $LOD1_OUTPUTS (the LOD1 project's outputs, kept outside the repository).
 *
 * The results file (benchmark/results/<date>-lod1.json) holds aggregates
 * only: TopologicPy and GeoJSON models are labelled "LOD1 model A", "B" ...,
 * with no names, ids or coordinates, since the lab's buildings are private.
 * IFC inputs keep their file names.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { KIND_DOOR, KIND_STAIR, buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { Heap, astar, componentLabels, csr, distancesFrom, edgeLengths } from "../topologicpy-web-frontend/src/core/route.js";
import { exitDoors, ifcSemantics } from "../topologicpy-web-frontend/src/core/ifcSemantics.js";
import { buildScene, runScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { buildRoomGraph } from "../topologicpy-web-frontend/src/core/rooms.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import {
  lod1FromIfcSpaces,
  lod1ToEgressGeometry,
  lod1ToGeoJSON,
  parseGeoJSONPlan,
  parseTopologicLOD1,
  pointInRoom,
  roomPoint,
} from "../topologicpy-web-frontend/src/core/lod1.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const valued = new Set(["--spacing", "--cells", "--horizon", "--out"]);
const inputs = args.filter((a, i) => !a.startsWith("--") && !valued.has(args[i - 1]));
const spacing = Number(flag("--spacing", 0.5));
const maxCells = Number(flag("--cells", 30000));
const horizon = Number(flag("--horizon", 0));
const closed = args.includes("--closed");
const AGENT = 0.75;

const ms = (t0) => performance.now() - t0;
const rss = () => Math.round(process.memoryUsage().rss / 2 ** 20);

/** Folders holding a lod1_building.json under $LOD1_OUTPUTS, largest model first. */
function defaultInputs() {
  const root = process.env.LOD1_OUTPUTS;
  if (!root || !existsSync(root)) return [];
  return readdirSync(root)
    .map((d) => path.join(root, d))
    .filter((d) => existsSync(path.join(d, "lod1_building.json")))
    .sort((a, b) => statSync(path.join(b, "lod1_building.json")).size - statSync(path.join(a, "lod1_building.json")).size);
}

// ------------------------------------------------------------- loading

const json = (file) => JSON.parse(readFileSync(file, "utf8"));

/** A TopologicPy export: the CellComplex and, if beside it, its graphs. */
function loadTopologic(input) {
  const dir = input.endsWith(".json") ? path.dirname(input) : input;
  const file = input.endsWith(".json") ? input : path.join(dir, "lod1_building.json");
  const graph = (name) => {
    const p = path.join(dir, "graphs", name);
    return existsSync(p) ? json(p) : null;
  };
  const t0 = performance.now();
  const records = json(file), circulation = graph("circulation_graph.json"), adjacency = graph("adjacency_graph.json");
  const readMs = ms(t0);
  const t1 = performance.now();
  const model = parseTopologicLOD1(records, { circulation, adjacency });
  return { model, timings: { readMs, parseMs: ms(t1) } };
}

/** An IFC's spaces, doors and stairs and its semantics, without walls, slabs or anything else. */
async function extractSpacesAndDoors(file) {
  const require = createRequire(import.meta.url);
  const WebIFC = require(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/web-ifc-api-node.js"));
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/"), true);
  await api.Init();
  const model = api.OpenModel(new Uint8Array(readFileSync(file)), { COORDINATE_TO_ORIGIN: true });
  const meshes = (types) => {
    const out = [];
    api.StreamAllMeshesWithTypes(model, types.map((t) => WebIFC[t]).filter((t) => typeof t === "number"), (mesh) => {
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
        out.push({ positions, indices: Uint32Array.from(index), itemId: mesh.expressID });
        geom.delete?.();
      }
    });
    return out;
  };
  const out = {
    spaces: meshes(["IFCSPACE"]),
    doors: meshes(["IFCDOOR"]),
    stairs: meshes(["IFCSTAIR", "IFCSTAIRFLIGHT", "IFCRAMP", "IFCRAMPFLIGHT"]),
    semantics: ifcSemantics(api, model, WebIFC),
  };
  api.CloseModel(model);
  return out;
}

async function loadIfc(file) {
  const t0 = performance.now();
  const inputsOf = await extractSpacesAndDoors(file);
  const readMs = ms(t0);
  const t1 = performance.now();
  const model = lod1FromIfcSpaces({ ...inputsOf, upAxis: "y" });
  return { model, timings: { readMs, parseMs: ms(t1) }, ifcSpaces: inputsOf.spaces.length, spaces: inputsOf.spaces };
}

// ------------------------------------------------------------- analysis

/** Distance to the nearest of `sources` from every node, over open links. */
function distancesToAny(graph, sources) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const w = edgeLengths(graph);
  const dist = new Float64Array(n).fill(Infinity);
  const heap = new Heap();
  for (const s of sources) {
    dist[s] = 0;
    heap.push(0, s);
  }
  const done = new Uint8Array(n);
  while (heap.size) {
    const u = heap.pop();
    if (done[u]) continue;
    done[u] = 1;
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
      if (graph.blocked?.[edgeIds[k]]) continue;
      const v = neighbours[k];
      const nd = dist[u] + w[edgeIds[k]];
      if (nd < dist[v]) {
        dist[v] = nd;
        heap.push(nd, v);
      }
    }
  }
  return dist;
}

/** The floor node standing in a room nearest its middle (viewer frame, y up), or -1. */
function nodeInRoom(graph, kinds, room) {
  const P = graph.nodes, n = P.length / 3;
  const [cx, cy] = roomPoint(room);
  let best = -1, bestD = Infinity;
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] !== 0) continue;
    const x = P[v * 3], y = -P[v * 3 + 2], h = P[v * 3 + 1] - AGENT;
    if (Math.abs(h - room.ffl) > 0.3) continue;
    const d = (x - cx) ** 2 + (y - cy) ** 2;
    if (d < bestD && pointInRoom(x, y, room)) [best, bestD] = [v, d];
  }
  return best;
}

/** The route from `start` to its nearest exit node: length, rise and what it passes. */
function routeFrom(graph, kinds, start, exits) {
  if (start < 0 || !exits.length) return { found: false };
  const t0 = performance.now();
  const dist = distancesFrom(graph, start, { blocked: graph.blocked });
  let end = -1;
  for (const e of exits) if (dist[e] < (end < 0 ? Infinity : dist[end])) end = e;
  if (end < 0 || !Number.isFinite(dist[end])) return { found: false, ms: ms(t0) };
  const path = astar(graph, start, end, { useWalls: true });
  const P = graph.nodes;
  let stairNodes = 0, doorNodes = 0;
  for (const v of path.nodes) {
    if (kinds[v] === KIND_STAIR) stairNodes += 1;
    if (kinds[v] === KIND_DOOR) doorNodes += 1;
  }
  return {
    found: true,
    length: path.cost,
    descent: P[start * 3 + 1] - P[end * 3 + 1],
    plan: Math.hypot(P[start * 3] - P[end * 3], P[start * 3 + 2] - P[end * 3 + 2]),
    nodes: path.nodes.length,
    stairNodes,
    doorNodes,
    ms: ms(t0),
  };
}

/** Parts, storeys in the main part, exits, and the route from the top floor's worst-placed guest room. */
function analyse(model, geo, built) {
  const t0 = performance.now();
  const graph = { nodes: built.points, edges: built.edges, blocked: built.blocked };
  const kinds = built.kinds;
  const n = graph.nodes.length / 3;
  const { labels, sizes } = componentLabels(graph);
  let main = 0;
  sizes.forEach((s, i) => {
    if (s > sizes[main]) main = i;
  });
  // Each floor node's storey: the highest level at or below its floor.
  const levels = model.levels;
  const storeys = levels.map(() => ({ nodes: 0, inMain: 0 }));
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] !== 0) continue;
    const h = graph.nodes[v * 3 + 1] - AGENT;
    let k = -1;
    for (let i = 0; i < levels.length; i += 1) if (levels[i].ffl <= h + 0.3) k = i;
    if (k < 0) continue;
    storeys[k].nodes += 1;
    if (labels[v] === main) storeys[k].inMain += 1;
  }
  // A storey counts when people walk on it (one of shafts alone does not),
  // and is joined when 90% of its floor nodes are in the main part.
  const walked = levels.map((l) => model.rooms.some((r) => r.level === l.id && r.walkable !== false));
  const joinedStoreys = storeys.filter((s, k) => walked[k] && s.nodes && s.inMain / s.nodes >= 0.9).length;
  const partsMs = ms(t0);

  // Exits as the site picks them, from the geometry's semantics. The route
  // goes to the flagged ones (the street exits of a LOD1 model), or to the
  // site's fallback when none is flagged.
  const exits = exitDoors(built.doorReport, geo.semantics, graph);
  const flaggedIds = new Set((geo.semantics?.doors || []).filter((d) => d.exit).map((d) => d.id));
  const flagged = [...new Set(built.doorReport.filter((d) => d.node >= 0 && flaggedIds.has(d.itemId)).map((d) => d.node))];
  const exitNodes = flagged.length ? flagged : exits.nodes;

  // The top floor's guest rooms (class Unit, with a door, 6 m2 or more: not
  // a closet), and the one furthest from an exit.
  const t1 = performance.now();
  const toExit = exitNodes.length ? distancesToAny(graph, exitNodes) : null;
  const doored = new Set(model.doors.flatMap((d) => d.rooms));
  const guest = model.rooms.filter((r) => r.walkable !== false && /^unit$/i.test(r.class) && doored.has(r.id) && r.area >= 6);
  const top = guest.length ? Math.max(...guest.map((r) => r.ffl)) : null;
  let start = -1, startRoom = null, worst = -1;
  for (const room of guest.filter((r) => r.ffl === top)) {
    const v = nodeInRoom(graph, kinds, room);
    if (v < 0) continue;
    const d = toExit ? toExit[v] : 0;
    const score = Number.isFinite(d) ? d : 1e9;
    if (score > worst) [worst, start, startRoom] = [score, v, room];
  }
  const route = routeFrom(graph, kinds, start, exitNodes);
  route.ms = ms(t1);
  const reachable = toExit ? Array.from({ length: n }, (_, v) => v).filter((v) => kinds[v] === 0 && Number.isFinite(toExit[v])).length : 0;
  let floorNodes = 0;
  for (let v = 0; v < n; v += 1) if (kinds[v] === 0) floorNodes += 1;
  return {
    graph,
    start,
    startRoom,
    summary: {
      nodes: n,
      edges: graph.edges.length / 2,
      blockedEdges: built.stats.blockedEdges,
      spacing: built.spacing,
      stairNodes: built.stats.stairNodes,
      doorNodes: built.stats.doorNodes,
      coreStairsAssumed: built.stats.coreStairsAssumed,
      cores: built.cores.length,
      coresLinked: built.cores.filter((c) => c.linked).length,
      parts: sizes.length,
      partsOver20: sizes.filter((s) => s >= 20).length,
      mainShare: n ? sizes[main] / n : 0,
      storeys: storeys.map((s, k) => ({ storey: k, nodes: s.nodes, mainShare: s.nodes ? s.inMain / s.nodes : null })),
      everyStoreyJoined: joinedStoreys === walked.filter(Boolean).length,
      storeysJoined: joinedStoreys,
      storeysWalked: walked.filter(Boolean).length,
      exits: exits.nodes.length,
      exitSource: exits.source,
      flaggedExits: flagged.length,
      floorNodesReachingAnExit: floorNodes ? reachable / floorNodes : 0,
      route,
      partsMs,
    },
  };
}

/** The Flamescapes scene with the fire at `start`; with --horizon, a short run. */
function scene(geo, graph, start) {
  const out = {};
  try {
    const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
    const fire = start >= 0 ? [graph.nodes[start * 3], graph.nodes[start * 3 + 1] - AGENT, graph.nodes[start * 3 + 2]] : null;
    const geometry = {
      walls: geo.walls,
      floors: geo.floors,
      roofs: geo.roofMeshes,
      stairs: geo.stairs,
      obstacles: geo.obstacles,
      // Closed doors are solids; openings have no leaf to close.
      ...(closed ? { doors: geo.doors.filter((d) => d.kind !== "opening") } : {}),
    };
    const t0 = performance.now();
    const s = buildScene(geometry, { upAxis: "y", fire, maxCells, contentsFuel, ignitionFuel, item });
    out.setupMs = ms(t0);
    const g = s.grid;
    let solid = 0;
    for (const v of s.solid) solid += v ? 1 : 0;
    Object.assign(out, {
      grid: [g.nx, g.ny, g.nz],
      cell: g.h,
      cells: g.cells,
      clipped: s.clipped,
      solidShare: solid / g.cells,
      contentsCells: s.contentsSummary.cells,
      contentsAreaM2: s.contentsSummary.area,
      ignition: Boolean(s.ignition),
      doors: closed ? "closed" : "open",
    });
    if (horizon > 0 && s.ignition) {
      const t1 = performance.now();
      let peak = 0;
      const frames = runScene(s, { nodes: graph.nodes, horizon, frameSeconds: 5, embers: false, particles: 0, solverOptions: {}, onFrame: (f) => { if (f.hrr > peak) peak = f.hrr; } });
      let nonFinite = 0;
      for (const a of [frames.T, frames.vis, frames.co]) for (const x of a) if (!Number.isFinite(x)) nonFinite += 1;
      Object.assign(out, { horizon, runMs: ms(t1), peakHrrKw: peak, nonFinite });
    }
  } catch (error) {
    out.error = error.message;
  }
  return out;
}

/**
 * The full IFC geometry's graph, with the route from the same room, for
 * comparison. web-ifc's COORDINATE_TO_ORIGIN takes its offset from the first
 * mesh it streams, slabs here and spaces in the fast path, so the room is
 * moved by the offset between the two, read off a space both extracted.
 */
async function fullIfc(file, startRoom, spaces) {
  const t0 = performance.now();
  const geo = await extract(file, { withDoorFlags: true });
  const extractMs = ms(t0);
  const t1 = performance.now();
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing, agentHeight: AGENT });
  const latticeMs = ms(t1);
  const graph = { nodes: built.points, edges: built.edges, blocked: built.blocked };
  const { sizes } = componentLabels(graph);
  const n = graph.nodes.length / 3;
  const exits = exitDoors(built.doorReport, geo.semantics, graph);
  const light = spaces.find((m) => geo.spaces.some((g) => g.itemId === m.itemId));
  const full = light && geo.spaces.find((g) => g.itemId === light.itemId);
  let room = startRoom;
  if (room && full) {
    // Viewer frame (x, y up, z) to IFC plan: x, -z; height: y.
    const [dx, dy, dz] = [0, 1, 2].map((k) => full.positions[k] - light.positions[k]);
    const move = (ring) => ring.map(([x, y]) => [x + dx, y - dz]);
    room = { ...room, footprint: move(room.footprint), holes: (room.holes || []).map(move), ffl: room.ffl + dy };
  }
  const start = room ? nodeInRoom(graph, built.kinds, room) : -1;
  return {
    extractMs,
    latticeMs,
    nodes: n,
    parts: sizes.length,
    partsOver20: sizes.filter((s) => s >= 20).length,
    mainShare: n ? Math.max(...sizes) / n : 0,
    exits: exits.nodes.length,
    exitSource: exits.source,
    route: routeFrom(graph, built.kinds, start, exits.nodes),
  };
}

// ----------------------------------------------------------------- run

const list = inputs.length ? inputs : defaultInputs();
if (!list.length) {
  console.log("No inputs: pass a LOD1 outputs folder, a lod1_building.json, a GeoJSON plan or an IFC, or set LOD1_OUTPUTS.");
  process.exit(1);
}
const results = [];
let anonymous = 0;
for (const input of list) {
  const lower = input.toLowerCase();
  const isIfc = lower.endsWith(".ifc");
  const isGeo = lower.endsWith(".geojson") || (lower.endsWith(".json") && !lower.endsWith("lod1_building.json") && json(input)?.type === "FeatureCollection");
  const label = isIfc ? path.basename(input) : `LOD1 model ${String.fromCharCode(65 + anonymous++)}`;
  console.log(`\n${label}`);
  let loaded;
  if (isIfc) loaded = await loadIfc(input);
  else if (isGeo) {
    const t0 = performance.now();
    const text = json(input);
    const readMs = ms(t0);
    const t1 = performance.now();
    loaded = { model: parseGeoJSONPlan(text), timings: { readMs, parseMs: ms(t1) } };
  } else loaded = loadTopologic(input);
  const { model, timings } = loaded;

  const t2 = performance.now();
  const geo = lod1ToEgressGeometry(model, { upAxis: "y" });
  timings.convertMs = ms(t2);
  const t3 = performance.now();
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing, agentHeight: AGENT });
  timings.latticeMs = ms(t3);
  const { graph, start, startRoom, summary } = analyse(model, geo, built);
  timings.partsMs = summary.partsMs;
  timings.routeMs = summary.route.ms;
  delete summary.partsMs;
  const fire = scene(geo, graph, start);
  timings.sceneMs = fire.setupMs ?? null;

  // The two-zone model's room graph (rooms.js), as tools/fire-bench.mjs builds it.
  const t5 = performance.now();
  const doorsWithAxes = built.doorReport.map((d, i) => ({ ...d, ux: built.doorPoints[i][3], uy: built.doorPoints[i][4] }));
  const roomGraph = buildRoomGraph(graph, { spaces: geo.spaces, doors: doorsWithAxes, declaredDoors: geo.semantics.doors, upAxis: "y", agentHeight: AGENT });
  timings.zoneRoomsMs = ms(t5);
  const zone = { ...roomGraph.stats, inFootprintShare: roomGraph.stats.inFootprint / roomGraph.stats.nodes };
  delete zone.ms;

  // The model through GeoJSON and back: the same rooms, doors, links and floor area.
  const t4 = performance.now();
  const fc = lod1ToGeoJSON(model);
  const back = parseGeoJSONPlan(JSON.parse(JSON.stringify(fc)));
  const areaOf = (m) => m.rooms.reduce((s, r) => s + r.area, 0);
  // And through the pipeline again: its graph's parts and route.
  const again = analyse(back, lod1ToEgressGeometry(back, { upAxis: "y" }), buildLatticeGraph({ ...lod1ToEgressGeometry(back, { upAxis: "y" }), upAxis: "y", spacing, agentHeight: AGENT })).summary;
  timings.geojsonMs = ms(t4);
  const geojson = {
    features: fc.features.length,
    sameCounts: back.rooms.length === model.rooms.length && back.doors.length === model.doors.length && back.links.length === model.links.length,
    areaDiffM2: Math.abs(areaOf(back) - areaOf(model)),
    warnings: back.meta.warnings.length,
    parts: again.parts,
    routeLength: again.route.found ? again.route.length : null,
  };

  const counts = model.meta.counts;
  const row = {
    label,
    source: model.meta.source,
    levels: counts.levels,
    rooms: counts.rooms,
    walkableRooms: model.rooms.filter((r) => r.walkable !== false).length,
    doors: counts.doors,
    doorKinds: counts.kinds,
    links: counts.links,
    warnings: model.meta.warnings.length,
    ...(model.meta.adjacency ? { adjacency: model.meta.adjacency } : {}),
    ...(loaded.ifcSpaces !== undefined ? { ifcSpaceMeshes: loaded.ifcSpaces } : {}),
    geometry: { walls: geo.walls.length, floors: geo.floors.length, ceilings: geo.roofMeshes.length, doors: geo.doors.length, cores: geo.cores.length },
    graph: summary,
    flamescapes: fire,
    zone,
    geojson,
    timings,
    rssMB: rss(),
  };
  if (isIfc) row.fullGeometry = await fullIfc(input, startRoom, loaded.spaces);
  results.push(row);

  // The console report (names stay out of it too).
  const r = summary.route;
  console.log(`  ${counts.rooms} rooms (${row.walkableRooms} walked in) on ${counts.levels} levels, ${counts.doors} doors ${JSON.stringify(counts.kinds)}, ${counts.links} links, ${row.warnings} warnings`);
  if (model.meta.adjacency) console.log(`  adjacency: ${model.meta.adjacency.wallPairsFound} of ${model.meta.adjacency.wallPairs} wall pairs share an outline`);
  console.log(`  geometry: ${row.geometry.walls} walls, ${row.geometry.floors} slabs, ${row.geometry.ceilings} ceilings, ${row.geometry.doors} doors, ${row.geometry.cores} cores`);
  console.log(`  graph: ${summary.nodes} nodes (${summary.spacing.toFixed(2)} m), ${summary.edges} edges, ${summary.parts} parts (${summary.partsOver20} of 20+ nodes), main part ${(summary.mainShare * 100).toFixed(1)}%`);
  console.log(`  cores: ${summary.coresLinked} of ${summary.cores} linked, ${summary.coreStairsAssumed} assumed flights; ${summary.stairNodes} stair nodes, ${summary.doorNodes} door nodes`);
  console.log(`  storeys: ${summary.everyStoreyJoined ? "every storey joined" : "NOT every storey joined"}, ${summary.storeysJoined} of ${summary.storeysWalked} walked storeys (${summary.storeys.map((s) => (s.mainShare === null ? "-" : Math.round(s.mainShare * 100))).join(" ")} % in the main part)`);
  console.log(`  exits: ${summary.exits} (${summary.exitSource}), ${summary.flaggedExits} flagged; ${(summary.floorNodesReachingAnExit * 100).toFixed(1)}% of floor nodes reach one the route may use`);
  console.log(
    r.found
      ? `  route: top-floor guest room to the nearest exit, ${r.length.toFixed(1)} m (${r.descent.toFixed(1)} m down, ${r.plan.toFixed(1)} m apart in plan), ${r.stairNodes} stair and ${r.doorNodes} door nodes`
      : "  route: none found",
  );
  console.log(
    fire.error
      ? `  flamescapes: error ${fire.error}`
      : `  flamescapes: grid ${fire.grid.join("x")} at ${fire.cell.toFixed(2)} m${fire.clipped ? " (clipped round the fire)" : ""}, ${(fire.solidShare * 100).toFixed(1)}% solid, ${fire.contentsCells} contents cells, ignition ${fire.ignition ? "placed" : "NOT placed"}${fire.runMs ? `, ${horizon} s in ${(fire.runMs / 1000).toFixed(1)} s, non-finite ${fire.nonFinite}` : ""}`,
  );
  console.log(`  two-zone rooms: ${zone.rooms} of ${zone.spaces} spaces, ${zone.joins} doors joining two rooms, ${zone.vents} venting outside, ${zone.unplaced} unplaced; ${(zone.inFootprintShare * 100).toFixed(1)}% of nodes in a room's footprint`);
  console.log(`  GeoJSON: ${geojson.features} features; back again ${geojson.sameCounts ? "the same" : "NOT the same"} rooms, doors and links, floor area within ${geojson.areaDiffM2.toFixed(3)} m2, ${geojson.warnings} warnings; ${geojson.parts} parts, route ${geojson.routeLength === null ? "none" : `${geojson.routeLength.toFixed(1)} m`}`);
  console.log(`  ms: ${Object.entries(timings).map(([k, v]) => `${k.replace(/Ms$/, "")} ${v === null ? "-" : v.toFixed(0)}`).join(", ")}`);
  if (row.fullGeometry) {
    const f = row.fullGeometry;
    console.log(`  full IFC: ${f.nodes} nodes, ${f.parts} parts (${f.partsOver20} of 20+), main ${(f.mainShare * 100).toFixed(1)}%, ${f.exits} exits, route ${f.route.found ? `${f.route.length.toFixed(1)} m` : "none"}; extract ${f.extractMs.toFixed(0)} ms, lattice ${f.latticeMs.toFixed(0)} ms`);
  }
}

if (!args.includes("--no-out")) {
  const out = flag("--out", path.join(HERE, `../benchmark/results/${new Date().toISOString().slice(0, 10)}-lod1.json`));
  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, JSON.stringify({ machine: { cpu: os.cpus()[0]?.model, cores: os.cpus().length, node: process.version }, spacing, maxCells, horizon, doors: closed ? "closed" : "open", results }, null, 1));
  console.log(`\nwritten ${path.relative(process.cwd(), out)}`);
}
