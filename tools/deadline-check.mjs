/**
 * Cross-model check of the escape-deadline method (src/core/deadline.js).
 *
 *   node tools/deadline-check.mjs [model.ifc ...] [--json out.json]
 *
 * For each IFC model: build the walkable graph as the site does, find the
 * exits (doors flagged IsExternal or FireExit in Pset_DoorCommon; failing
 * that, doors with walkable floor on one side only), start a fire front in the
 * largest part of the building, compute every node's escape deadline, and
 * verify a sample of them with an independent forward search (leaving just
 * before the deadline must reach an exit, just after must not).
 *
 * The fire and walking speeds are illustrative (0.1 m/s front, 1.2 m/s walk):
 * this checks that the method runs and is consistent on each model, not what
 * a fire engineer should conclude about it.
 */
import { readdirSync, existsSync, writeFileSync } from "node:fs";
import path from "node:path";

import { buildLatticeGraph } from "../topologicpy-web-frontend/src/core/lattice.js";
import { componentLabels } from "../topologicpy-web-frontend/src/core/route.js";
import { departureSafe, escapeDeadlines, frontArrival } from "../topologicpy-web-frontend/src/core/deadline.js";
import { findExits } from "./exits.mjs";
import { extract } from "./ifc-extract.mjs";

const args = process.argv.slice(2);
const jsonAt = args.indexOf("--json");
const jsonOut = jsonAt >= 0 ? args[jsonAt + 1] : null;
let files = args.filter((a, i) => a.toLowerCase().endsWith(".ifc") && i !== jsonAt + 1);
if (!files.length) {
  const dirs = ["C:/Users/lmurugesan/ifc-samples", "topologicpy-web-frontend/public/examples", "C:/Users/lmurugesan/Downloads"];
  for (const d of dirs) {
    if (!existsSync(d)) continue;
    for (const f of readdirSync(d)) if (f.toLowerCase().endsWith(".ifc")) files.push(path.join(d, f));
  }
  // The Downloads copy of the Duplex is the same file as the shipped example.
  files = files.filter((f) => !/Ifc2x3_Duplex_Architecture\.ifc$/i.test(f));
}

const WALK = 1.2;
const FRONT = 0.1;
const rows = [];

for (const file of files) {
  const row = { model: path.basename(file) };
  try {
    const t0 = performance.now();
    const geo = await extract(path.resolve(file), { withDoorFlags: true });
    const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
    const g = { ...built, nodes: built.points };
    const n = g.nodes.length / 3;
    row.nodes = n;
    row.doors = built.doorReport.length;
    if (!n) {
      row.status = "no walkable graph";
      rows.push(row);
      continue;
    }

    // Exits.
    const found = findExits(geo, built);
    const exits = found.exits;
    row.doorsFlagged = found.flagged;
    row.exitSource = found.source ?? "none found";
    row.exits = new Set(exits).size;
    if (!exits.length) {
      row.status = "no exits";
      rows.push(row);
      continue;
    }

    // Without fire: who can reach an exit at all.
    const free = escapeDeadlines(g, { exits, speed: WALK, blocked: g.blocked });
    row.reachExit = +(free.filter((x) => x !== -Infinity).length / n).toFixed(3);

    // With a fire front from the middle of the largest part.
    const { labels, sizes } = componentLabels(g);
    const big = sizes.indexOf(Math.max(...sizes));
    const members = [];
    for (let v = 0; v < n; v += 1) if (labels[v] === big) members.push(v);
    const origin = members[Math.floor(members.length / 2)];
    const untenable = frontArrival(g, origin, { speed: FRONT, blocked: g.blocked });
    const t1 = performance.now();
    const opts = { exits, untenable, speed: WALK, blocked: g.blocked };
    const L = escapeDeadlines(g, opts);
    row.deadlineMs = +(performance.now() - t1).toFixed(1);
    const finite = [];
    let escapeNow = 0;
    for (let v = 0; v < n; v += 1) {
      if (L[v] > 0) escapeNow += 1;
      if (Number.isFinite(L[v]) && L[v] > 0) finite.push(L[v]);
    }
    finite.sort((a, b) => a - b);
    row.canEscapeAtStart = +(escapeNow / n).toFixed(3);
    row.medianDeadlineS = finite.length ? Math.round(finite[Math.floor(finite.length / 2)]) : null;

    // Independent forward check on a deterministic sample.
    const candidates = [];
    for (let v = 0; v < n; v += 1) if (Number.isFinite(L[v])) candidates.push(v);
    const stride = Math.max(1, Math.floor(candidates.length / 40));
    let checked = 0, agreed = 0;
    for (let k = 0; k < candidates.length; k += stride) {
      const v = candidates[k];
      const eps = 1e-6 * Math.max(1, Math.abs(L[v]));
      const before = departureSafe(g, v, L[v] - eps, opts);
      const after = departureSafe(g, v, L[v] + eps, opts);
      checked += 1;
      if (before && !after) agreed += 1;
    }
    row.forwardCheck = `${agreed}/${checked}`;
    row.totalMs = Math.round(performance.now() - t0);
    row.status = agreed === checked ? "ok" : "MISMATCH";
  } catch (error) {
    row.status = `error: ${error.message}`;
  }
  rows.push(row);
  console.log(
    `${row.model.slice(0, 34).padEnd(34)} ${String(row.status).padEnd(18)} nodes ${String(row.nodes ?? "-").padStart(6)}  ` +
      `exits ${String(row.exits ?? "-").padStart(3)} (${row.exitSource ?? "-"})  reach ${row.reachExit ?? "-"}  ` +
      `escape@0 ${row.canEscapeAtStart ?? "-"}  median ${row.medianDeadlineS ?? "-"} s  check ${row.forwardCheck ?? "-"}  ${row.deadlineMs ?? "-"} ms`,
  );
}

if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ walkSpeed: WALK, frontSpeed: FRONT, rows }, null, 1));
