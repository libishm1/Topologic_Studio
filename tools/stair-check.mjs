/**
 * Do the stairs join the floors, and where?
 *
 *   node tools/stair-check.mjs model.ifc [...]
 *
 * For each model: build the walkable graph as the site does, group the stair
 * nodes into staircases (connected stair nodes), and for each staircase report
 * the floor levels it touches, whether it joins two levels, whether those
 * floors are connected through the whole graph, and how far each floor-to-
 * stair link is from the staircase's centre line (in plan, across the run).
 */
import path from "node:path";

import { buildLatticeGraph, floorLevels } from "../topologicpy-web-frontend/src/core/lattice.js";
import { extract } from "./ifc-extract.mjs";

const KIND_STAIR = 1;

for (const file of process.argv.slice(2).filter((a) => a.toLowerCase().endsWith(".ifc"))) {
  const geo = await extract(path.resolve(file), { withDoorFlags: false });
  const g = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  const P = g.points, n = P.length / 3, E = g.edges, m = E.length / 2;
  const levels = floorLevels(geo.floors, "y");
  const adj = Array.from({ length: n }, () => []);
  for (let e = 0; e < m; e += 1) {
    if (g.blocked?.[e]) continue;
    adj[E[2 * e]].push(E[2 * e + 1]);
    adj[E[2 * e + 1]].push(E[2 * e]);
  }
  // Whole-graph components.
  const comp = new Int32Array(n).fill(-1);
  let nc = 0;
  for (let s = 0; s < n; s += 1) {
    if (comp[s] >= 0) continue;
    const stack = [s];
    comp[s] = nc;
    while (stack.length) {
      const v = stack.pop();
      for (const w of adj[v]) if (comp[w] < 0) { comp[w] = nc; stack.push(w); }
    }
    nc += 1;
  }
  // Staircases: components of the stair-node subgraph.
  const sc = new Int32Array(n).fill(-1);
  const stairs = [];
  for (let s = 0; s < n; s += 1) {
    if (g.kinds[s] !== KIND_STAIR || sc[s] >= 0) continue;
    const members = [s];
    sc[s] = stairs.length;
    for (let i = 0; i < members.length; i += 1) for (const w of adj[members[i]]) if (g.kinds[w] === KIND_STAIR && sc[w] < 0) { sc[w] = stairs.length; members.push(w); }
    stairs.push(members);
  }
  const level = (y) => {
    let best = -1, gap = Infinity;
    levels.forEach((l, i) => { const d = Math.abs(y - 0.75 - l); if (d < gap) { gap = d; best = i; } });
    return gap <= 0.6 ? best : -1;
  };
  const rows = [];
  for (const members of stairs) {
    if (members.length < 6) continue;
    // Centre line: principal axis of the stair nodes in plan; the run is the
    // direction along which height changes most.
    let cx = 0, cz = 0;
    for (const v of members) { cx += P[3 * v]; cz += P[3 * v + 2]; }
    cx /= members.length; cz /= members.length;
    let sxx = 0, szz = 0, sxz = 0;
    for (const v of members) { const dx = P[3 * v] - cx, dz = P[3 * v + 2] - cz; sxx += dx * dx; szz += dz * dz; sxz += dx * dz; }
    const ang = 0.5 * Math.atan2(2 * sxz, sxx - szz);
    const ux = Math.cos(ang), uz = Math.sin(ang);
    const across = (v) => -(P[3 * v] - cx) * uz + (P[3 * v + 2] - cz) * ux;
    let wmin = Infinity, wmax = -Infinity, ylo = Infinity, yhi = -Infinity;
    for (const v of members) { const w = across(v); wmin = Math.min(wmin, w); wmax = Math.max(wmax, w); ylo = Math.min(ylo, P[3 * v + 1]); yhi = Math.max(yhi, P[3 * v + 1]); }
    const wc = (wmin + wmax) / 2;
    const touched = new Map(); // level -> offsets of floor links from the centre line
    for (const v of members) {
      for (const w of adj[v]) {
        if (g.kinds[w] === KIND_STAIR) continue;
        const L = level(P[3 * w + 1]);
        if (!touched.has(L)) touched.set(L, []);
        touched.get(L).push(Math.abs(across(v) - wc));
      }
    }
    const lv = [...touched.keys()].filter((l) => l >= 0).sort((a, b) => a - b);
    rows.push({
      nodes: members.length,
      width: +(wmax - wmin).toFixed(2),
      rise: +(yhi - ylo).toFixed(2),
      levels: lv.map((l) => +levels[l].toFixed(2)),
      joinsTwo: lv.length >= 2,
      floorLinks: [...touched.values()].reduce((s, a) => s + a.length, 0),
      offCentre: +Math.max(0, ...[...touched.values()].flat()).toFixed(2),
    });
  }
  // Are the floor levels connected through the whole graph?
  const levelComps = levels.map((l, i) => {
    const set = new Map();
    for (let v = 0; v < n; v += 1) if (g.kinds[v] !== KIND_STAIR && level(P[3 * v + 1]) === i) set.set(comp[v], (set.get(comp[v]) || 0) + 1);
    const top = [...set.entries()].sort((a, b) => b[1] - a[1])[0];
    return top ? top[0] : -1;
  });
  const sameMain = levelComps.length > 1 && levelComps.every((c) => c === levelComps[0]);
  console.log(`${path.basename(file).slice(0, 34).padEnd(34)} levels ${JSON.stringify(levels.map((l) => +l.toFixed(2)))} main floor parts connected: ${sameMain}`);
  for (const r of rows) console.log(`    stair ${String(r.nodes).padStart(4)} nodes, ${r.width} m wide, rise ${r.rise} m, floors ${JSON.stringify(r.levels)}${r.joinsTwo ? "" : "  (does NOT join two floors)"}, ${r.floorLinks} floor links, up to ${r.offCentre} m off centre`);
}
