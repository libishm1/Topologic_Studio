/**
 * Where the time goes between an IFC file and its walkable graph:
 *
 *   node tools/profile-graph.mjs model.ifc [more.ifc ...]
 *
 * Times the web-ifc extraction (tools/ifc-extract.mjs, as the cross-model
 * checks read files) apart from buildLatticeGraph (src/core/lattice.js, what
 * the site's worker runs), and prints the graph's size. Run with
 * `node --cpu-prof` to see the hot functions.
 */
import path from "node:path";

import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { extract } from "./ifc-extract.mjs";

for (const file of process.argv.slice(2).filter((a) => a.toLowerCase().endsWith(".ifc"))) {
  const t0 = performance.now();
  const geo = await extract(path.resolve(file), { withDoorFlags: false });
  const t1 = performance.now();
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const t2 = performance.now();
  const count = (list) => (list || []).reduce((s, m) => s + (m.indices?.length || 0) / 3, 0);
  console.log(
    `${path.basename(file).slice(0, 40).padEnd(40)} extract ${Math.round(t1 - t0)} ms, lattice ${Math.round(t2 - t1)} ms; ` +
      `nodes ${built.points.length / 3}, edges ${built.edges.length / 2}; triangles floors ${count(geo.floors)}, walls ${count(geo.walls)}, ` +
      `stairs ${count(geo.stairs)}, doors ${geo.doors?.length ?? 0}, obstacles ${count(geo.obstacles)}`,
  );
  if (built.stats) console.log(`    stats ${JSON.stringify(built.stats).slice(0, 400)}`);
}
