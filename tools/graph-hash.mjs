/**
 * A fingerprint of the walkable graph each model produces, to show that a
 * speed-up changed nothing:
 *
 *   node tools/graph-hash.mjs model.ifc [...] > before.txt
 *   (change lattice.js)
 *   node tools/graph-hash.mjs model.ifc [...] > after.txt   # must match
 *
 * Hashes the node, edge, kind and blocked arrays and the wall sections, and
 * prints the build time.
 */
import { createHash } from "node:crypto";
import path from "node:path";

import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { extract } from "./ifc-extract.mjs";

const bytes = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);

for (const file of process.argv.slice(2).filter((a) => a.toLowerCase().endsWith(".ifc"))) {
  const geo = await extract(path.resolve(file), { withDoorFlags: false });
  const t0 = performance.now();
  const g = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const ms = performance.now() - t0;
  const h = createHash("sha256");
  for (const a of [g.points, g.edges, g.kinds, g.blocked]) h.update(bytes(a));
  for (const s of g.sections || []) h.update(bytes(s.segs));
  console.log(`${h.digest("hex").slice(0, 20)}  ${path.basename(file)}  (${Math.round(ms)} ms)`);
}
