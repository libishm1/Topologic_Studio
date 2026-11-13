/**
 * The shortest walks against the slime mould's desire lines on the example
 * buildings, and what makes them part (wiki/research/egress-vs-slime-2026-11-12.md):
 * the egress map (egressRun.js runMap) and the slime (slimeEgress.js) on the
 * same graph and exits, compared by core/egressCompare.js; then the slime again
 * with one of its differences switched off at a time, so each one's share of
 * the parting shows.
 *
 *   node tools/egress-vs-slime.mjs [model.ifc ...] [--seeds 3] [--out file]
 *
 * Without models: the Duplex, the NIBS Office and Clinic. No fire.
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildLatticeGraph, floorLevels } from "../topologicpy-web-frontend/src/core/lattice.js";
import { exitDoors } from "../topologicpy-web-frontend/src/core/ifcSemantics.js";
import { outdoorSpaces } from "../topologicpy-web-frontend/src/core/outdoorSpaces.js";
import { EGRESS_B } from "../topologicpy-web-frontend/src/core/codeCheck.js";
import { rooms, runMap } from "../topologicpy-web-frontend/src/core/egressRun.js";
import { slimeEgress } from "../topologicpy-web-frontend/src/core/slimeEgress.js";
import { compareEgressSlime } from "../topologicpy-web-frontend/src/core/egressCompare.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc"));
if (!models.length) {
  const dir = path.join(HERE, "../topologicpy-web-frontend/public/examples");
  models = ["duplex-apartment.ifc", "nibs-office.ifc", "nibs-clinic.ifc"].map((f) => path.join(dir, f));
}
const SEEDS = Number(flag("--seeds", 3));
const today = new Date().toISOString().slice(0, 10);
const OUT = flag("--out", path.join(HERE, `../benchmark/results/${today}-egress-vs-slime.json`));
// The slime as the app runs it, then with one difference off at a time.
const VARIANTS = [
  ["as the app runs it", {}],
  ["no trail in what they follow (trailWeight 0)", { trailWeight: 0 }],
  ["sensors one link out (so 1)", { so: 1 }],
  ["a tenth of the agents (200): little crowding", { agents: 200 }],
  ["all three off", { trailWeight: 0, so: 1, agents: 200 }],
];
const r3 = (x) => (Number.isFinite(x) ? Math.round(x * 1000) / 1000 : x);
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

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
  // The storeys as tools/egress-map-check.mjs finds them.
  let levels = floorLevels(geo.floors, "y");
  if (levels.length <= 1) {
    const counts = new Map();
    for (let v = 0; v < n; v += 1) if (graph.kinds[v] === 0) {
      const h = Math.round((graph.nodes[v * 3 + 1] - 0.75) * 10) / 10;
      counts.set(h, (counts.get(h) || 0) + 1);
    }
    const groups = [];
    for (const [h, c] of [...counts].sort((a, b) => a[0] - b[0])) {
      const g = groups.at(-1);
      if (g && h - g.lo < 1.5) {
        g.n += c;
        if (c > g.bestCount) Object.assign(g, { best: h, bestCount: c });
      } else groups.push({ lo: h, n: c, best: h, bestCount: c });
    }
    levels = groups.filter((g) => g.n >= 50).map((g) => g.best);
  }
  const names = new Map((geo.semantics?.spaces || []).map((s) => [s.id, s.longName || s.name]));
  const opts = { exits: exitNodes, levels, useWalls: true, limit: EGRESS_B.travel.unsprinklered, sprinklered: false, occupantArea: EGRESS_B.occupantArea.value };
  const map = runMap(graph, opts, rooms(geo, names, "y"));
  console.log(`\n${name}: ${n} nodes, ${map.exits.filter((e) => e.live).length} exits, ${map.rooms.length} rooms`);
  const rows = [];
  for (const [label, extra] of VARIANTS) {
    const runs = [];
    for (let s = 1; s <= SEEDS; s += 1) runs.push(compareEgressSlime(graph, map, slimeEgress(graph, { exits: exitNodes, seed: s, ...extra })));
    const row = {
      variant: label,
      onShortest: r3(mean(runs.map((c) => c.onShortest))),
      covered: r3(mean(runs.map((c) => c.covered))),
      exitsApart: r3(mean(runs.map((c) => c.exitsApart ?? NaN))),
      atWallsSlime: r3(mean(runs.map((c) => c.atWalls.slime))),
      atWallsShortest: r3(runs[0].atWalls.shortest),
    };
    rows.push(row);
    console.log(`  ${label.padEnd(46)} on shortest ${row.onShortest}  covered ${row.covered}  exits apart ${row.exitsApart}  at walls ${row.atWallsSlime} (shortest ${row.atWallsShortest})`);
  }
  results.push({ model: name, nodes: n, exits: map.exits.filter((e) => e.live).length, rooms: map.rooms.length, seeds: SEEDS, rows });
}
await writeFile(OUT, JSON.stringify({ generated: new Date().toISOString(), tool: "tools/egress-vs-slime.mjs", near: 1, lineMin: 0.25, variants: VARIANTS.map(([label, options]) => ({ label, options })), results }, null, 1));
console.log(`\nwrote ${OUT}`);
