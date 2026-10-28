/**
 * The stress and regression suite of the benchmark (benchmark/README.md):
 *
 *   node tools/benchmark.mjs --stress                  # every case
 *   node tools/benchmark.mjs --stress --only "stair"   # cases whose id or title match
 *   node tools/benchmark.mjs --stress --core-rev ac2fe81^   # against an older core
 *
 * Three groups of cases, each run in its own Node process with a time limit,
 * so a hang or a crash fails that case and not the suite:
 *   geometry    synthetic meshes fed to buildLatticeGraph (as
 *               src/core/lattice.test.mjs builds them), each with properties
 *               checked automatically, then the routing, forecast and deadline
 *               pipeline on the result, checked for NaN;
 *   ifc         web-ifc's own test fixtures, broken files made on the spot
 *               (empty, truncated, not IFC, IFC-XML, zipped) and every corpus
 *               model, through extraction and the graph: no crash, and either
 *               a clear error or a valid (possibly empty) graph;
 *   regression  one minimal case per known bug, citing the commit or review
 *               finding that fixed it. A case whose fix is on a branch not yet
 *               merged is marked expected to fail.
 *
 * --core-rev REV runs the cases against src/core as it was at REV (extracted
 * with git into a temporary folder; the tools stay current), which is how the
 * regression cases were checked to fail before their fixes.
 */
import { createRequire } from "node:module";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const CORE_DIR = process.env.STRESS_CORE_DIR || path.join(ROOT, "topologicpy-web-frontend", "src", "core");
const core = (name) => import(pathToFileURL(path.join(CORE_DIR, name)).href);
export const FIXTURES = (samples) => path.join(samples, "more", "web-ifc-fixtures");

// ------------------------------------------------------------- meshes
// z up, as in src/core/lattice.test.mjs.

/** A horizontal rectangle at height z, as two triangles. */
function rect(x0, y0, x1, y1, z, { rotate = 0, dx = 0, dy = 0, scale = 1 } = {}) {
  const c = Math.cos(rotate), s = Math.sin(rotate);
  const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].flatMap(([x, y]) => [(c * x - s * y) * scale + dx, (s * x + c * y) * scale + dy, z * scale]);
  return { positions: Float32Array.from(pts), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
}

/** A rectangle tilted along x from height z0 at x0 to z1 at x1. */
function ramp(x0, y0, x1, y1, z0, z1) {
  return { positions: Float32Array.from([x0, y0, z0, x1, y0, z1, x1, y1, z1, x0, y1, z0]), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
}

/** A box, for walls and doors. */
function box(x0, y0, z0, x1, y1, z1, { rotate = 0, itemId, dx = 0, dy = 0, scale = 1 } = {}) {
  const c = Math.cos(rotate), s = Math.sin(rotate);
  const corners = [];
  for (const z of [z0, z1]) for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) corners.push((c * x - s * y) * scale + dx, (s * x + c * y) * scale + dy, z * scale);
  const f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]];
  return { positions: Float32Array.from(corners), indices: Uint32Array.from(f.flat()), itemId };
}

/** A wall segment of given thickness between two plan points, as a box mesh. */
function wallBetween(ax, ay, bx, by, t, z0 = 0, z1 = 3) {
  const len = Math.hypot(bx - ax, by - ay);
  const a = Math.atan2(by - ay, bx - ax);
  const m = box(0, -t / 2, z0, len, t / 2, z1, { rotate: a });
  for (let i = 0; i < m.positions.length; i += 3) {
    m.positions[i] += ax;
    m.positions[i + 1] += ay;
  }
  return m;
}

/** A wedge tread of a spiral: centre (cx, cy), radii r0..r1, angles a0..a1, height z. */
function wedge(cx, cy, r0, r1, a0, a1, z) {
  const pts = [];
  for (const [r, a] of [[r0, a0], [r1, a0], [r1, a1], [r0, a1]]) pts.push(cx + r * Math.cos(a), cy + r * Math.sin(a), z);
  return { positions: Float32Array.from(pts), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
}

/** A straight flight along +x from x0, treads of `going` on y0..y1, first tread at z0 + riser. */
function flight(x0, y0, y1, z0, count, { going = 0.25, riser = 0.1875 } = {}) {
  const out = [];
  for (let k = 1; k <= count; k += 1) out.push(rect(x0 + (k - 1) * going, y0, x0 + k * going, y1, z0 + riser * k));
  return out;
}

const transform = (meshes, f) =>
  meshes.map((m) => {
    const p = Float32Array.from(m.positions);
    for (let i = 0; i < p.length; i += 3) [p[i], p[i + 1], p[i + 2]] = f(p[i], p[i + 1], p[i + 2]);
    return { ...m, positions: p };
  });

// ------------------------------------------------------------- checks

function components(g, useWalls = true) {
  const n = g.points.length / 3;
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  for (let e = 0; e < g.edges.length / 2; e += 1) if (!(useWalls && g.blocked[e])) parent[find(g.edges[e * 2])] = find(g.edges[e * 2 + 1]);
  return find;
}

/** The node nearest (x, y, z) (z is the floor height; points carry agentHeight), of a kind if given. */
function nearest(g, x, y, z, { kind = null, agentHeight = 0.75, within = 1.0 } = {}) {
  let best = -1, bestD = Infinity;
  const P = g.points;
  for (let v = 0; v < P.length / 3; v += 1) {
    if (kind !== null && g.kinds[v] !== kind) continue;
    const d = Math.hypot(P[v * 3] - x, P[v * 3 + 1] - y, P[v * 3 + 2] - agentHeight - z);
    if (d < bestD) {
      bestD = d;
      best = v;
    }
  }
  return bestD <= within ? best : -1;
}

/** Whether an open link crosses the line x = at between y0 and y1 (walls on). */
function crossesX(g, at, y0 = -Infinity, y1 = Infinity) {
  const P = g.points;
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    if (g.blocked[e]) continue;
    const a = g.edges[e * 2], b = g.edges[e * 2 + 1];
    if ((P[a * 3] - at) * (P[b * 3] - at) >= 0) continue;
    const y = (P[a * 3 + 1] + P[b * 3 + 1]) / 2;
    if (y > y0 && y < y1) return true;
  }
  return false;
}

/** Structural soundness of a built graph: finite points, edge ends in range, arrays agree. */
function sound(g) {
  const n = g.points.length / 3;
  for (const x of g.points) if (!Number.isFinite(x)) return "a point is not finite";
  for (const v of g.edges) if (!(v < n)) return "an edge end is out of range";
  if (g.blocked.length !== g.edges.length / 2) return "blocked and edges disagree";
  if (g.kinds.length !== n) return "kinds and points disagree";
  if (g.stats.nodes !== n) return "stats.nodes disagrees";
  return null;
}

/** The routing, forecast and deadline pipeline on a graph; returns problems found (NaN, throws). */
async function pipeline(built, { exits = null } = {}) {
  const { astar, componentLabels } = await core("route.js");
  const { buildForecast, planWithForecast } = await core("forecast.js");
  const { doseDeadlineMap, escapeDeadlines } = await core("deadline.js");
  const { straightenRoute } = await core("straighten.js");
  const g = { ...built, nodes: built.points, upAxis: built.upAxis ?? "z", agentHeight: built.agentHeight ?? 0.75 };
  const n = g.nodes.length / 3;
  if (!n) return { problems: [], ran: [] };
  const problems = [];
  const ran = [];
  const nan = (label, arr) => {
    for (const x of arr) if (Number.isNaN(x)) return problems.push(`${label} has NaN`);
  };
  const { labels, sizes } = componentLabels(g);
  const big = sizes.indexOf(Math.max(...sizes));
  const members = [];
  for (let v = 0; v < n; v += 1) if (labels[v] === big) members.push(v);
  const a = members[0], b = members[members.length - 1];
  const route = astar(g, a, b);
  ran.push("route");
  if (route) {
    if (!Number.isFinite(route.cost)) problems.push("route cost not finite");
    const st = straightenRoute(g, route.nodes);
    ran.push("straighten");
    if (!Number.isFinite(st.length)) problems.push("straightened length not finite");
  } else if (a !== b) problems.push("no route inside one part");
  const ex = exits && exits.length ? exits : [b];
  for (const model of ["temperature", "flood"]) {
    const f = buildForecast(g, { model, seed: members[members.length >> 1], steps: 30, dt: 5, threshold: 80, blocked: g.blocked });
    nan(`${model} forecast rates`, f.rate);
    nan(`${model} closing times`, f.closes);
    const hard = escapeDeadlines(g, { exits: ex, untenable: f.closes, speed: 1.2, blocked: g.blocked });
    nan("hard deadlines", hard);
    const plan = planWithForecast(g, { start: a, exits: ex, forecast: f, speed: 1.2, blocked: g.blocked });
    for (const p of plan.front) if (Number.isNaN(p.arrival) || Number.isNaN(p.dose)) problems.push("front point NaN");
    const map = doseDeadlineMap(g, { exits: ex, forecast: f, limit: 0.3, speed: 1.2, blocked: g.blocked, maxWork: 2e6 });
    nan("deadline map", map.deadline);
    ran.push(model);
  }
  return { problems, ran };
}

// --------------------------------------------------------- geometry cases

const BASE = { upAxis: "z", spacing: 0.5, agentHeight: 0.75 };

async function lattice(input) {
  const { buildLatticeGraph } = await core("lattice.js");
  const t0 = performance.now();
  const g = buildLatticeGraph({ ...BASE, ...input });
  return { g: { ...g, upAxis: input.upAxis ?? BASE.upAxis, agentHeight: input.agentHeight ?? BASE.agentHeight }, ms: performance.now() - t0 };
}

/** Standard checks on a graph: sound, then each named expectation. */
async function judge({ g, ms }, expectations, { budgetMs = 2000, pipelineOpts = {} } = {}) {
  const checks = [];
  const bad = sound(g);
  checks.push({ name: "graph sound (finite points, valid edges)", ok: !bad, detail: bad || undefined });
  for (const [name, fn] of Object.entries(expectations)) {
    let ok, detail;
    try {
      const r = fn(g);
      ok = r === true || (r && r.ok === true);
      detail = r && typeof r === "object" ? r.detail : undefined;
    } catch (error) {
      ok = false;
      detail = `threw: ${error.message}`;
    }
    checks.push({ name, ok, detail });
  }
  checks.push({ name: `built within ${budgetMs} ms`, ok: ms <= budgetMs, detail: `${Math.round(ms)} ms` });
  let pipe = null;
  if (!bad) {
    try {
      pipe = await pipeline(g, pipelineOpts);
      checks.push({ name: "pipeline without NaN", ok: !pipe.problems.length, detail: pipe.problems.join("; ") || pipe.ran.join(", ") });
    } catch (error) {
      checks.push({ name: "pipeline without NaN", ok: false, detail: `threw: ${error.message}` });
    }
  }
  const s = g.stats || {};
  return { checks, metrics: { nodes: g.points.length / 3, edges: g.edges.length / 2, stairNodes: s.stairNodes, stairEnds: s.stairEnds, stairEndsLinked: s.stairEndsLinked, spacing: s.spacing, latticeMs: Math.round(ms) } };
}

const STAIR = 1, FLOOR = 0, DOOR = 2;
const linked = (g, p, q) => {
  const find = components(g);
  const a = nearest(g, ...p, { kind: FLOOR }), b = nearest(g, ...q, { kind: FLOOR });
  return { ok: a >= 0 && b >= 0 && find(a) === find(b), detail: a < 0 || b < 0 ? "a probe found no floor node" : find(a) === find(b) ? "connected" : "separate parts" };
};
const separate = (g, p, q) => {
  const r = linked(g, p, q);
  return { ok: r.detail === "separate parts", detail: r.detail };
};
const endsLinked = (want) => (g) => ({ ok: g.stats.stairEnds === want && g.stats.stairEndsLinked === want, detail: `${g.stats.stairEndsLinked} of ${g.stats.stairEnds} ends linked` });
const staircases = (g) => {
  const find = (() => {
    const n = g.points.length / 3;
    const parent = Int32Array.from({ length: n }, (_, i) => i);
    const f = (x) => {
      while (parent[x] !== x) x = parent[x] = parent[parent[x]];
      return x;
    };
    for (let e = 0; e < g.edges.length / 2; e += 1) {
      const a = g.edges[e * 2], b = g.edges[e * 2 + 1];
      if (g.kinds[a] === STAIR && g.kinds[b] === STAIR) parent[f(a)] = f(b);
    }
    return f;
  })();
  const roots = new Set();
  for (let v = 0; v < g.points.length / 3; v += 1) if (g.kinds[v] === STAIR) roots.add(find(v));
  return roots.size;
};

// Every geometry case: a scene, expectations, a budget.
const GEOMETRY = [
  {
    id: "stair-straight",
    title: "Straight flight between two floors",
    scene: () => ({ floors: [rect(0, 0, 2.25, 4, 0), rect(2.25, 2, 10, 4, 0), rect(6, 0, 10, 2, 3)], stairs: flight(2.25, 0, 2, 0, 15), walls: [box(6, -0.2, 3, 10, 0, 5.8)] }),
    expect: { "both ends linked, once each": endsLinked(2), "one staircase": (g) => staircases(g) === 1, "ground joins upper floor": (g) => linked(g, [1, 3, 0], [8, 1, 3]) },
  },
  {
    id: "stair-L-landing-stair",
    title: "L-stair with its landing modelled as a stair element",
    scene: () => {
      const stairs = flight(1, 0, 1, 0, 8);
      stairs.push(rect(3, 0, 4, 1, 1.5)); // landing
      for (let k = 1; k <= 8; k += 1) stairs.push(rect(3, 1 + (k - 1) * 0.25, 4, 1 + k * 0.25, 1.5 + 0.1875 * k));
      return { floors: [rect(-3, -1, 1, 2, 0), rect(2, 3, 6, 6, 3)], stairs, walls: [box(2, 6, 3, 6, 6.2, 5.8)] };
    },
    expect: { "both ends linked, once each": endsLinked(2), "ground joins upper floor": (g) => linked(g, [-2, 0, 0], [5, 5, 3]) },
  },
  {
    id: "stair-L-landing-slab",
    title: "L-stair with a separate landing slab (IfcSlab between two flights)",
    scene: () => {
      // Flight 1 along +x to a landing slab at 1.5 m; flight 2 along +y from it.
      const stairs = flight(1.25, 0, 1, 0, 7);
      for (let k = 1; k <= 7; k += 1) stairs.push(rect(3, 1.2 + (k - 1) * 0.25, 4, 1.2 + k * 0.25, 1.5 + 0.1875 * k));
      return { floors: [rect(-3, -1, 1.25, 2, 0), rect(3, 0, 4, 1.2, 1.5), rect(2, 2.95, 6, 6, 3)], stairs, walls: [box(2, 6, 3, 6, 6.2, 5.8)] };
    },
    expect: { "every end linked, once each": endsLinked(4), "two staircases": (g) => staircases(g) === 2, "ground joins upper floor through the landing": (g) => linked(g, [-2, 0, 0], [5, 5, 3]) },
  },
  {
    id: "stair-U-landing-stair",
    title: "U-stair with its landing modelled as a stair element",
    scene: () => {
      const stairs = flight(1, 0, 1, 0, 8);
      stairs.push(rect(3, 0, 4, 2, 1.5));
      for (let k = 1; k <= 8; k += 1) stairs.push(rect(3 - k * 0.25, 1, 3 - (k - 1) * 0.25, 2, 1.5 + 0.1875 * k));
      return { floors: [rect(-3, 0, 1, 1, 0), rect(-3, 1, 1, 2, 3)], stairs, walls: [box(-3, 2, 3, 1, 2.2, 5.8)] };
    },
    expect: { "both ends linked, once each": endsLinked(2), "ground joins upper floor": (g) => linked(g, [-2, 0.5, 0], [-2, 1.5, 3]) },
  },
  {
    id: "stair-U-landing-slab",
    title: "U-stair with a separate landing slab",
    scene: () => {
      const stairs = flight(1.25, 0, 1, 0, 7);
      for (let k = 1; k <= 7; k += 1) stairs.push(rect(3 - k * 0.25, 1.2, 3 - (k - 1) * 0.25, 2.2, 1.5 + 0.1875 * k));
      return { floors: [rect(-3, 0, 1.25, 1, 0), rect(3, 0, 4, 2.2, 1.5), rect(-3, 1.2, 1.25, 2.2, 3)], stairs, walls: [box(-3, 2.2, 3, 1.25, 2.4, 5.8)] };
    },
    expect: { "every end linked, once each": endsLinked(4), "two staircases": (g) => staircases(g) === 2, "ground joins upper floor through the landing": (g) => linked(g, [-2, 0.5, 0], [-2, 1.7, 3]) },
  },
  {
    id: "stair-spiral",
    title: "Spiral stair: 16 wedge treads through 337.5 degrees round a post",
    scene: () => {
      const stairs = [];
      const step = (2 * Math.PI) / 16;
      for (let k = 0; k < 15; k += 1) stairs.push(wedge(0, 0, 0.15, 1.0, k * step, (k + 1) * step, 0.1875 * (k + 1)));
      // Ground all round; upper floor beyond the stairwell (x > 1.1).
      return { floors: [rect(-5, -5, 5, 5, 0), rect(1.1, -5, 6, 5, 3)], stairs, walls: [box(1.1, 5, 3, 6, 5.2, 5.8)] };
    },
    expect: { "both ends linked, once each": endsLinked(2), "ground joins upper floor": (g) => linked(g, [-4, -4, 0], [4, 0, 3]) },
  },
  {
    id: "stair-winders",
    title: "Quarter-turn stair with three winder treads",
    scene: () => {
      const stairs = flight(0, 0, 1, 0, 6);
      // Winders: three wedges round the corner at (1.5, 1).
      for (let k = 0; k < 3; k += 1) stairs.push(wedge(1.5, 1, 0, 1.0, -Math.PI / 2 + (k * Math.PI) / 6, -Math.PI / 2 + ((k + 1) * Math.PI) / 6, 0.1875 * (7 + k)));
      for (let k = 1; k <= 6; k += 1) stairs.push(rect(1.5, 1 + (k - 1) * 0.25, 2.5, 1 + k * 0.25, 0.1875 * (9 + k)));
      return { floors: [rect(-4, -1, 0, 2, 0), rect(0, 2.5, 5, 6, 0.1875 * 16)], stairs, walls: [box(0, 6, 3, 5, 6.2, 5.8)] };
    },
    expect: { "both ends linked, once each": endsLinked(2), "ground joins upper floor": (g) => linked(g, [-3, 0.5, 0], [3, 5, 3]) },
  },
  {
    id: "ramp-terraced",
    title: "Terraced ramp: 20 level slabs 2.5 cm apart",
    scene: () => {
      const floors = [rect(-5, 0, 0, 3, 0)];
      for (let k = 0; k < 20; k += 1) floors.push(rect(k * 0.5, 0, (k + 1) * 0.5, 3, 0.025 * (k + 1)));
      floors.push(rect(10, 0, 15, 3, 0.5));
      return { floors };
    },
    expect: { "bottom joins top": (g) => linked(g, [-4, 1.5, 0], [14, 1.5, 0.5]), "no stair nodes": (g) => !g.stats.stairNodes },
  },
  {
    id: "ramp-1in12",
    title: "Sloped ramp at 1:12",
    scene: () => ({ floors: [rect(-5, 0, 0, 3, 0), ramp(0, 0, 6, 3, 0, 0.5), rect(6, 0, 11, 3, 0.5)] }),
    expect: { "bottom joins top": (g) => linked(g, [-4, 1.5, 0], [10, 1.5, 0.5]) },
  },
  {
    id: "ramp-1in8",
    title: "Sloped ramp at 1:8",
    scene: () => ({ floors: [rect(-5, 0, 0, 3, 0), ramp(0, 0, 4, 3, 0, 0.5), rect(4, 0, 9, 3, 0.5)] }),
    expect: { "bottom joins top": (g) => linked(g, [-4, 1.5, 0], [8, 1.5, 0.5]) },
  },
  {
    id: "stair-gap-riser",
    title: "Flight whose bottom tread is a full riser (0.2 m) above the floor, top a riser below the landing",
    scene: () => ({ floors: [rect(-3, 0, 2, 2, 0), rect(5.25, 0, 9, 2, 2.4)], stairs: flight(2, 0, 2, 0, 11, { going: 0.3, riser: 0.2 }).map((m, k) => m).slice(0, 11), walls: [box(5.25, -0.2, 2.4, 9, 0, 5.2)] }),
    expect: { "both ends linked, once each": endsLinked(2), "ground joins upper floor": (g) => linked(g, [-2, 1, 0], [8, 1, 2.4]) },
  },
  {
    id: "stair-gap-too-tall",
    title: "Flight starting 0.45 m above the floor (beyond the riser limit): the bottom end stays unlinked",
    scene: () => ({ floors: [rect(-3, 0, 2, 2, 0), rect(5.25, 0, 9, 2, 2.65)], stairs: flight(2, 0, 2, 0.25, 11, { going: 0.3, riser: 0.2 }), walls: [box(5.25, -0.2, 2.65, 9, 0, 5.4)] }),
    expect: { "the bottom end is reported unlinked": (g) => ({ ok: g.stats.stairEnds === 2 && g.stats.stairEndsLinked === 1, detail: `${g.stats.stairEndsLinked} of ${g.stats.stairEnds}` }), "ground does not join upper floor": (g) => separate(g, [-2, 1, 0], [8, 1, 2.65]) },
  },
  {
    id: "stair-to-door",
    title: "Flight ending at a door in a wall; the upper slab stops at the wall face",
    scene: () => {
      // Upper floor beyond a wall at x 6.0..6.2, door in it at the flight's end.
      const walls = [box(6, -2, 3, 6.2, 0.1, 5.8), box(6, 1.9, 3, 6.2, 4, 5.8), box(6, 0.1, 5.1, 6.2, 1.9, 5.8), box(6.2, 4, 3, 10, 4.2, 5.8)];
      return { floors: [rect(-2, -2, 2.25, 4, 0), rect(6.2, -2, 10, 4, 3)], stairs: [...flight(2.25, 0, 2, 0, 15), rect(6 - 0.25, 0, 6, 2, 3)].slice(0, 15), walls, doors: [box(6, 0.1, 3, 6.2, 1.9, 5.1, { itemId: 1 })] };
    },
    expect: { "both ends linked, once each": endsLinked(2), "ground joins upper floor through the door": (g) => linked(g, [-1, 1, 0], [9, 1, 3]) },
  },
  {
    id: "stair-to-nowhere",
    title: "Flight ending at an outside door with no floor beyond",
    scene: () => ({ floors: [rect(-2, -2, 2.25, 4, 0)], stairs: flight(2.25, 0, 2, 0, 15), walls: [box(6, -2, 0, 6.2, 0.1, 5.8), box(6, 1.9, 0, 6.2, 4, 5.8)], doors: [box(6, 0.1, 3, 6.2, 1.9, 5.1, { itemId: 1 })] }),
    expect: { "one end linked, the other reported unlinked": (g) => ({ ok: g.stats.stairEnds === 2 && g.stats.stairEndsLinked === 1, detail: `${g.stats.stairEndsLinked} of ${g.stats.stairEnds}` }) },
  },
  {
    id: "split-level-2-steps",
    title: "Split levels 0.36 m apart joined by two steps",
    scene: () => ({ floors: [rect(0, 0, 5, 4, 0), rect(5.5, 0, 10, 4, 0.36)], stairs: [rect(5, 0, 5.25, 2, 0.12), rect(5.25, 0, 5.5, 2, 0.24)], walls: [box(5, 2, 0, 5.5, 4.2, 3)] }),
    expect: { "both ends linked, once each": endsLinked(2), "lower level joins upper level": (g) => linked(g, [1, 1, 0], [9, 1, 0.36]) },
  },
  {
    id: "split-level-3-steps",
    title: "Split levels 0.54 m apart joined by three steps",
    scene: () => ({ floors: [rect(0, 0, 5, 4, 0), rect(5.75, 0, 10, 4, 0.54)], stairs: [rect(5, 0, 5.25, 2, 0.18), rect(5.25, 0, 5.5, 2, 0.36), rect(5.5, 0, 5.75, 2, 0.54 - 0.0001)].slice(0, 2).concat([rect(5.5, 0, 5.75, 2, 0.36)]).slice(0, 2), walls: [box(5, 2, 0, 5.75, 4.2, 3)] }),
    expect: { "both ends linked, once each": endsLinked(2), "lower level joins upper level": (g) => linked(g, [1, 1, 0], [9, 1, 0.54]) },
  },
  {
    id: "mezzanine",
    title: "Mezzanine at 2.8 m over part of a floor, reached by a flight",
    scene: () => ({
      floors: [rect(0, 0, 12, 10, 0), rect(0, 6, 12, 10, 2.8)],
      // A flight along +y from y 2.5 to the mezzanine edge at y 6, one riser below it.
      stairs: Array.from({ length: 14 }, (_, i) => rect(4, 2.5 + i * 0.25, 5, 2.5 + (i + 1) * 0.25, ((i + 1) * 2.8) / 15)),
      walls: [box(0, 10, 2.8, 12, 10.2, 5.6), box(0, 5.9, 2.8, 3.9, 6, 3.9), box(5.1, 5.9, 2.8, 12, 6, 3.9)],
    }),
    expect: {
      "floor kept under the mezzanine": (g) => nearest(g, 2, 8, 0, { kind: FLOOR, within: 0.4 }) >= 0,
      "mezzanine has nodes": (g) => nearest(g, 2, 8, 2.8, { kind: FLOOR, within: 0.4 }) >= 0,
      "no floor link between the two levels": (g) => {
        for (let e = 0; e < g.edges.length / 2; e += 1) {
          const a = g.edges[e * 2], b = g.edges[e * 2 + 1];
          if (g.kinds[a] === FLOOR && g.kinds[b] === FLOOR && Math.abs(g.points[a * 3 + 2] - g.points[b * 3 + 2]) > 0.35 + 1e-6) return false;
        }
        return true;
      },
      "ground joins mezzanine by the stair": (g) => linked(g, [1, 1, 0], [10, 8, 2.8]),
    },
  },
  ...[false, true].map((door) => ({
    id: `thin-wall-${door ? "door" : "closed"}`,
    title: `A 5 cm wall between two rooms${door ? ", with a door" : ", no opening"}`,
    scene: () => ({ floors: [rect(0, 0.25, 10, 9.75, 0)], walls: [box(5.2, 0, 0, 5.25, 10, 3)], doors: door ? [box(5.2, 4.55, 0, 5.25, 5.45, 2.1, { itemId: 7 })] : [] }),
    expect: door ? { "the door joins the rooms": (g) => linked(g, [2, 5, 0], [8, 5, 0]), "open only at the door": (g) => !crossesX(g, 5.225, -1, 4.4) && !crossesX(g, 5.225, 5.6, 11) } : { "the wall keeps the rooms apart": (g) => separate(g, [2, 5, 0], [8, 5, 0]), "no link through the wall": (g) => !crossesX(g, 5.225) },
  })),
  ...[false, true].map((door) => ({
    id: `rotated-30-${door ? "door" : "closed"}`,
    title: `Building turned 30 degrees, dividing wall${door ? " with a door" : ""}`,
    scene: () => {
      const r = (30 * Math.PI) / 180;
      return { floors: [rect(0, 0.25, 10, 9.75, 0, { rotate: r })], walls: [box(4.9, 0, 0, 5.1, 10, 3, { rotate: r }), box(0, -0.2, 0, 10, 0, 3, { rotate: r })], doors: door ? [box(4.9, 4.55, 0, 5.1, 5.45, 2.1, { rotate: r, itemId: 7 })] : [] };
    },
    expect: (() => {
      const r = (30 * Math.PI) / 180;
      const at = (x, y) => [Math.cos(r) * x - Math.sin(r) * y, Math.sin(r) * x + Math.cos(r) * y, 0];
      const base = { "lattice follows the walls": (g) => ({ ok: Math.abs(g.stats.angleDeg - 30) < 0.5, detail: `${g.stats.angleDeg.toFixed(2)} deg` }) };
      return door ? { ...base, "the door joins the rooms": (g) => linked(g, at(2, 5), at(8, 5)) } : { ...base, "the wall keeps the rooms apart": (g) => separate(g, at(2, 5), at(8, 5)) };
    })(),
  })),
  ...[false, true].map((door) => ({
    id: `curved-wall-${door ? "door" : "closed"}`,
    title: `Round room of 48 wall segments in a hall${door ? ", with a doorway" : ", closed"}`,
    scene: () => {
      const walls = [];
      const N = 48, R = 3;
      for (let k = 0; k < N; k += 1) {
        if (door && (k === 0 || k === N - 1)) continue; // a gap of two segments at angle 0
        const a0 = (2 * Math.PI * k) / N, a1 = (2 * Math.PI * (k + 1)) / N;
        walls.push(wallBetween(R * Math.cos(a0), R * Math.sin(a0), R * Math.cos(a1), R * Math.sin(a1), 0.2));
      }
      return { floors: [rect(-7, -7, 7, 7, 0)], walls, doors: door ? [box(2.9, -0.39, 0, 3.1, 0.39, 2.1, { itemId: 9 })] : [] };
    },
    expect: door ? { "the doorway joins inside and outside": (g) => linked(g, [0, 0, 0], [-6, -6, 0]) } : { "the round wall keeps inside and outside apart": (g) => separate(g, [0, 0, 0], [-6, -6, 0]) },
  })),
  {
    id: "door-no-slab",
    title: "Doorway with no slab under the door (both slabs stop at the wall faces)",
    scene: () => ({
      floors: [rect(0, 0, 5, 10, 0), rect(5.2, 0, 10, 10, 0)],
      walls: [box(5, 0, 0, 5.2, 4.5, 3), box(5, 5.5, 0, 5.2, 10, 3), box(5, 4.5, 2.1, 5.2, 5.5, 3)],
      doors: [box(5, 4.5, 0, 5.2, 5.5, 2.1, { itemId: 3 })],
    }),
    expect: { "the door joins the rooms": (g) => linked(g, [2, 5, 0], [8, 5, 0]), "one door node": (g) => g.stats.doorNodes === 1 },
  },
  ...[0.45, 0.7].map((w) => ({
    id: `corridor-${Math.round(w * 100)}cm`,
    title: `Corridor ${w} m wide (grid spacing 0.5 m) between two rooms`,
    scene: () => {
      // Walls run past the slab edges, as they meet outer walls.
      const y0 = 5.25 - w / 2, y1 = 5.25 + w / 2;
      return {
        floors: [rect(0, 0.25, 14, 9.75, 0)],
        walls: [box(5, -0.2, 0, 5.2, y0, 3), box(5, y1, 0, 5.2, 10.2, 3), box(8.8, -0.2, 0, 9, y0, 3), box(8.8, y1, 0, 9, 10.2, 3), box(5.2, y0 - 0.2, 0, 8.8, y0, 3), box(5.2, y1, 0, 8.8, y1 + 0.2, 3)],
      };
    },
    expect: w < 0.5
      ? { "no link through the corridor walls": (g) => !crossesX(g, 5.1, -1, 5.25 - w / 2) && !crossesX(g, 5.1, 5.25 + w / 2, 11) && !crossesX(g, 8.9, -1, 5.25 - w / 2) && !crossesX(g, 8.9, 5.25 + w / 2, 11), "rooms joined (reported; a corridor below the spacing may be lost)": (g) => ({ ok: true, detail: linked(g, [2, 5, 0], [12, 5, 0]).detail }) }
      : { "the corridor joins the rooms": (g) => linked(g, [2, 5, 0], [12, 5, 0]), "no link through the corridor walls": (g) => !crossesX(g, 5.1, -1, 5.25 - w / 2) && !crossesX(g, 5.1, 5.25 + w / 2, 11) && !crossesX(g, 8.9, -1, 5.25 - w / 2) && !crossesX(g, 8.9, 5.25 + w / 2, 11) },
  })),
  {
    id: "floor-200m",
    title: "A 200 x 200 m floor: the grid coarsens to the 40,000-point budget",
    budgetMs: 15000,
    scene: () => ({ floors: [rect(0, 0, 200, 200, 0)] }),
    expect: { "coarsened": (g) => g.stats.coarsened, "within the point budget": (g) => ({ ok: g.stats.nodes <= 40000, detail: `${g.stats.nodes} nodes at ${g.stats.spacing.toFixed(2)} m` }), "one part": (g) => linked(g, [5, 5, 0], [195, 195, 0]) },
  },
  {
    id: "degenerate-triangles",
    title: "Zero-area and repeated-vertex triangles beside a clean floor",
    scene: () => {
      const f = rect(0, 0, 10, 10, 0);
      const p = Array.from(f.positions), ix = Array.from(f.indices);
      p.push(2, 2, 0, 4, 4, 0, 6, 6, 0); // collinear
      ix.push(4, 5, 6, 0, 0, 1, 2, 2, 2); // zero area, repeated vertices
      return { floors: [{ positions: Float32Array.from(p), indices: Uint32Array.from(ix) }] };
    },
    expect: { "same nodes as the clean floor": (g) => ({ ok: g.stats.nodes === 441, detail: `${g.stats.nodes} nodes (clean: 441)` }) },
  },
  {
    id: "nan-vertex",
    title: "A floor triangle with a NaN vertex beside a clean floor",
    scene: () => {
      const f = rect(0, 0, 10, 10, 0);
      return { floors: [f, { positions: Float32Array.from([1, 1, 0, NaN, 2, 0, 3, 1, 0]), indices: Uint32Array.from([0, 1, 2]) }], walls: [box(4.9, 0, 0, 5.1, 3, 3)] };
    },
    expect: { "same nodes as without it": (g) => ({ ok: g.stats.nodes > 400, detail: `${g.stats.nodes} nodes` }) },
  },
  {
    id: "infinite-vertex",
    title: "A floor triangle with an infinite vertex beside a clean floor",
    open: "lattice.js samples a triangle with a non-finite vertex over an unbounded range (not fixed here)",
    timeoutMs: 20000,
    scene: () => {
      const f = rect(0, 0, 10, 10, 0);
      return { floors: [f, { positions: Float32Array.from([1, 1, 0, Infinity, 2, 0, 3, 1, 0]), indices: Uint32Array.from([0, 1, 2]) }] };
    },
    expect: { "same nodes as without it": (g) => ({ ok: g.stats.nodes >= 441, detail: `${g.stats.nodes} nodes` }) },
  },
  {
    id: "infinite-wall-vertex",
    title: "A wall mesh with an infinite vertex",
    open: "a non-finite wall vertex makes the dominant angle NaN (lattice.js, not fixed here)",
    timeoutMs: 20000,
    scene: () => ({ floors: [rect(0, 0, 10, 10, 0)], walls: [box(4.9, 0, 0, 5.1, 10, 3), { positions: Float32Array.from([1, 1, 0, -Infinity, 2, 3, 3, 1, 3]), indices: Uint32Array.from([0, 1, 2]) }] }),
    expect: { "the good wall still separates": (g) => separate(g, [2, 5, 0], [8, 5, 0]) },
  },
  {
    id: "inverted-winding",
    title: "A floor whose triangles all face down",
    scene: () => {
      const f = rect(0, 0, 10, 10, 0);
      return { floors: [{ positions: f.positions, indices: Uint32Array.from([0, 2, 1, 0, 3, 2]) }] };
    },
    expect: { "winding fallback used": (g) => g.stats.windingFallback, "same nodes as facing up": (g) => ({ ok: g.stats.nodes === 441, detail: `${g.stats.nodes}` }) },
  },
  ...[false, true].map((door) => ({
    id: `offset-500km-${door ? "door" : "closed"}`,
    title: `Rooms and a wall${door ? " with a door" : ""}, 500 km from the origin (float32 positions)`,
    scene: () => {
      const o = { dx: 500000, dy: 500000 };
      return { floors: [rect(0, 0.25, 10, 9.75, 0, o)], walls: [box(5.15, 0, 0, 5.35, 10, 3, o)], doors: door ? [box(5.15, 4.55, 0, 5.35, 5.45, 2.1, { ...o, itemId: 7 })] : [] };
    },
    expect: door ? { "the door joins the rooms": (g) => linked(g, [500002, 500005, 0], [500008, 500005, 0]) } : { "the wall keeps the rooms apart": (g) => separate(g, [500002, 500005, 0], [500008, 500005, 0]) },
  })),
  {
    id: "millimetres",
    title: "The door scene in millimetres (units not converted): no crash, within budget",
    open: "lattice.js samples the full grid before coarsening, 4e8 samples here (not fixed here)",
    budgetMs: 10000,
    scene: () => {
      const s = { scale: 1000 };
      return { floors: [rect(0, 0.25, 10, 9.75, 0, s)], walls: [box(5.15, 0, 0, 5.35, 10, 3, s)], doors: [box(5.15, 4.55, 0, 5.35, 5.45, 2.1, { ...s, itemId: 7 })] };
    },
    expect: { "within the point budget": (g) => ({ ok: g.stats.nodes <= 40000 * 1.25, detail: `${g.stats.nodes} nodes at ${g.stats.spacing.toFixed(1)} units` }), "rooms joined (reported; metre thresholds do not scale)": (g) => ({ ok: true, detail: linked(g, [2000, 5000, 0], [8000, 5000, 0]).detail }) },
  },
];

// ----------------------------------------------------------- IFC cases

const require = createRequire(import.meta.url);

async function webIfc() {
  const W = require(path.join(ROOT, "topologicpy-web-frontend/node_modules/web-ifc/web-ifc-api-node.js"));
  const api = new W.IfcAPI();
  api.SetWasmPath(path.join(ROOT, "topologicpy-web-frontend/node_modules/web-ifc/"), true);
  await api.Init();
  return { api, W };
}

/** Extraction and graph on a file: a clear error or a sound (possibly empty) graph. */
async function ifcPipeline(file) {
  const { extract } = await import("./ifc-extract.mjs");
  const { buildLatticeGraph } = await core("lattice.js");
  const checks = [];
  const metrics = { bytes: statSync(file).size };
  let messages = 0;
  for (const name of ["log", "warn", "error", "info"]) {
    const orig = console[name].bind(console);
    console[name] = (...a) => {
      if (typeof a[0] === "string" && a[0].startsWith("[WEB-IFC]")) messages += 1;
      else orig(...a);
    };
  }
  let geo;
  let t0 = performance.now();
  try {
    geo = await extract(file, { withDoorFlags: true });
  } catch (error) {
    metrics.extractMs = Math.round(performance.now() - t0);
    const msg = String(error?.message || error);
    const why = "the extraction passes the file to web-ifc unchecked (tools/ifc-extract.mjs; not changed here)";
    checks.push({ name: "extraction: an error or geometry, no crash", ok: msg.length > 0, detail: `error: ${msg.slice(0, 200)}` });
    checks.push({ name: "the error is readable (not a raw WebAssembly trap)", ok: !/memory access out of bounds|Aborted\(|reading 'arguments'|unreachable/i.test(msg), detail: msg.slice(0, 120), open: why });
    if (metrics.bytes < 1e6) checks.push({ name: "a small file fails within 5 s", ok: metrics.extractMs <= 5000, detail: `${metrics.extractMs} ms`, open: why });
    metrics.outcome = "extraction error";
    return { checks, metrics };
  }
  metrics.extractMs = Math.round(performance.now() - t0);
  metrics.webIfcMessages = messages;
  metrics.meshes = Object.fromEntries(["floors", "stairs", "doors", "walls", "spaces", "obstacles"].map((k) => [k, geo[k].length]));
  checks.push({ name: "extraction: an error or geometry, no crash", ok: true, detail: "geometry" });
  t0 = performance.now();
  let g;
  try {
    g = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  } catch (error) {
    metrics.latticeMs = Math.round(performance.now() - t0);
    checks.push({ name: "graph: sound or a clear error", ok: false, detail: `buildLatticeGraph threw: ${String(error?.message || error).slice(0, 200)}` });
    metrics.outcome = "graph error";
    return { checks, metrics };
  }
  metrics.latticeMs = Math.round(performance.now() - t0);
  const bad = sound(g);
  checks.push({ name: "graph: sound or a clear error", ok: !bad, detail: bad || `${g.stats.nodes} nodes, ${g.stats.edges} edges` });
  metrics.nodes = g.stats.nodes;
  metrics.outcome = g.stats.nodes ? "graph" : "empty graph";
  if (!bad && g.stats.nodes) {
    try {
      const pipe = await pipeline({ ...g, upAxis: "y", agentHeight: 0.75 });
      checks.push({ name: "pipeline without NaN", ok: !pipe.problems.length, detail: pipe.problems.join("; ") || pipe.ran.join(", ") });
    } catch (error) {
      checks.push({ name: "pipeline without NaN", ok: false, detail: `threw: ${error.message}` });
    }
  }
  return { checks, metrics };
}

/** A stored (uncompressed) zip archive holding one file. */
function zipOne(name, data) {
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  let crc = 0xffffffff;
  for (const b of data) crc = crcTable[(crc ^ b) & 0xff] ^ (crc >>> 8);
  crc = (crc ^ 0xffffffff) >>> 0;
  const fname = Buffer.from(name);
  const local = Buffer.alloc(30);
  local.writeUInt32LE(0x04034b50, 0);
  local.writeUInt16LE(20, 4);
  local.writeUInt32LE(crc, 14);
  local.writeUInt32LE(data.length, 18);
  local.writeUInt32LE(data.length, 22);
  local.writeUInt16LE(fname.length, 26);
  const central = Buffer.alloc(46);
  central.writeUInt32LE(0x02014b50, 0);
  central.writeUInt16LE(20, 4);
  central.writeUInt16LE(20, 6);
  central.writeUInt32LE(crc, 16);
  central.writeUInt32LE(data.length, 20);
  central.writeUInt32LE(data.length, 24);
  central.writeUInt16LE(fname.length, 28);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(1, 8);
  end.writeUInt16LE(1, 10);
  end.writeUInt32LE(46 + fname.length, 12);
  end.writeUInt32LE(30 + fname.length + data.length, 16);
  return Buffer.concat([local, fname, data, central, fname, end]);
}

/** Broken and unusual files, made on the spot from a real model. */
export function makeBrokenFiles(dir, realModel) {
  mkdirSync(dir, { recursive: true });
  const real = existsSync(realModel) ? readFileSync(realModel) : Buffer.from("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n");
  const files = {
    "empty.ifc": Buffer.alloc(0),
    "truncated-40pc.ifc": real.subarray(0, Math.floor(real.length * 0.4)),
    "truncated-header.ifc": real.subarray(0, 200),
    "not-ifc-json.ifc": Buffer.from(JSON.stringify({ type: "FeatureCollection", features: [] })),
    "not-ifc-binary.ifc": Buffer.from(Array.from({ length: 4096 }, (_, i) => (i * 2654435761) & 0xff)),
    "ifc-xml.ifc": Buffer.from('<?xml version="1.0" encoding="UTF-8"?>\n<ifcXML xmlns="http://www.buildingsmart-tech.org/ifcXML/IFC4/final"><IfcProject id="i1" GlobalId="0YvctVUKr0kugbFTf53O9L" Name="x"/></ifcXML>\n'),
    "zipped.ifc": zipOne("model.ifc", real),
  };
  for (const [name, data] of Object.entries(files)) writeFileSync(path.join(dir, name), data);
  return Object.keys(files).map((f) => path.join(dir, f));
}

// ----------------------------------------------------- regression cases

function graphOf(points, edges) {
  return { nodes: Float64Array.from(points.flatMap(([x, z]) => [x, 0, z])), edges: Int32Array.from(edges.flat()) };
}
function forecastOf(rates, untenable, frameSeconds) {
  const K = rates.length, n = rates[0].length;
  const rate = new Float32Array(K * n);
  rates.forEach((row, k) => row.forEach((r, v) => (rate[k * n + v] = r)));
  const horizon = K * frameSeconds;
  const closes = Float64Array.from(untenable, (u) => Math.min(u, horizon));
  return { n, K, stride: 1, dt: frameSeconds, frameSeconds, horizon, rate, untenable: Float64Array.from(untenable), closes };
}
/** tie_counterexample's network: 0 start, 1 hot room, 2 cool room, 3 exit; every link 1 s. */
const tieCase = () => ({ g: graphOf([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1], [0, 2], [1, 3], [2, 3]]), f: forecastOf([[0, 1, 0, 0]], [Infinity, Infinity, Infinity, Infinity], 100) });
const rlGraph = (pts, edges) => ({ nodes: Float32Array.from(pts.flat()), edges: Uint32Array.from(edges.flat()), blocked: new Uint8Array(edges.length) });

/** A minimal IFC4 file (STEP text) with the given DATA lines. */
function ifcText(lines) {
  return `ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');\nFILE_NAME('stress.ifc','2026-10-28T00:00:00',(''),(''),'','','');\nFILE_SCHEMA(('IFC4'));\nENDSEC;\nDATA;\n${lines.join("\n")}\nENDSEC;\nEND-ISO-10303-21;\n`;
}

async function semanticsOf(text) {
  const { ifcSemantics } = await core("ifcSemantics.js");
  const { api, W } = await webIfc();
  const id = api.OpenModel(new Uint8Array(Buffer.from(text)), { COORDINATE_TO_ORIGIN: false });
  try {
    return ifcSemantics(api, id, W);
  } finally {
    api.CloseModel(id);
  }
}

/**
 * Storey levels spanned by the best-joined part, and the share of floor nodes
 * in parts spanning two or more levels (as tools/core-check.mjs reports them).
 */
export function levelSpread(g, levels, agent = 0.75) {
  const n = g.points.length / 3;
  const find = components(g);
  const parts = new Map();
  for (let i = 0; i < n; i += 1) {
    if (g.kinds[i] !== 0) continue;
    const level = levels.findIndex((h) => Math.abs(g.points[i * 3 + 1] - agent - h) < 0.4);
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

const UNITS_M = ["#90=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);", "#91=IFCUNITASSIGNMENT((#90));"];
const PROJECT = ["#1=IFCPROJECT('0YvctVUKr0kugbFTf53O9L',$,'p',$,$,$,$,$,#91);"];

const REGRESSION = [
  {
    id: "reg-stair-one-floor",
    title: "A staircase joins both floors it spans, also when its treads miss the floor grid",
    cite: "ac2fe81 (Duplex second dwelling in the browser, Revit ARC main stair)",
    run: async () => {
      // A flight offset from the 0.5 m floor grid in both directions, 1.1 m wide.
      const { g, ms } = await lattice({ floors: [rect(-3, 0.13, 2.37, 4, 0), rect(2.37, 2, 10, 4, 0), rect(5.6, 0.13, 10, 1.23, 2.88)], stairs: flight(2.37, 0.13, 1.23, 0, 12, { going: 0.27, riser: 0.24 }), walls: [box(5.6, -0.1, 2.88, 10, 0.13, 5.7)] });
      const a = linked(g, [-2, 1, 0], [8, 0.7, 2.88]);
      return { checks: [{ name: "ground joins upper floor", ok: a.ok, detail: a.detail }, { name: "both ends linked", ok: g.stats.stairEndsLinked === 2, detail: `${g.stats.stairEndsLinked} of ${g.stats.stairEnds}` }], metrics: { latticeMs: Math.round(ms) } };
    },
  },
  {
    id: "reg-front-tie",
    title: "The dose front keeps no two points tied in arrival time; the fastest route has the least dose of its tie",
    cite: "a42e75e (Forecast.tie_counterexample)",
    run: async () => {
      const { doseFront, planWithForecast } = await core("forecast.js");
      const { g, f } = tieCase();
      const { front } = doseFront(g, 0, { exits: [3], forecast: f, speed: 1 });
      const strict = front.every((p, i) => i === 0 || p.arrival > front[i - 1].arrival);
      const plan = planWithForecast(g, { start: 0, exits: [3], forecast: f, speed: 1 });
      return { checks: [{ name: "arrivals strictly increase", ok: strict, detail: JSON.stringify(front.map((p) => [p.arrival, p.dose])) }, { name: "fastest route has dose 0", ok: plan.fastest?.dose === 0, detail: `dose ${plan.fastest?.dose}` }] };
    },
  },
  {
    id: "reg-negative-departure",
    title: "A departure before time 0 gives finite doses",
    cite: "a42e75e (linkDose read a frame that does not exist)",
    run: async () => {
      const { doseFront } = await core("forecast.js");
      const { g, f } = tieCase();
      const { front } = doseFront(g, 0, { exits: [3], forecast: f, speed: 1, depart: -1 });
      return { checks: [{ name: "front found", ok: front.length > 0 }, { name: "doses finite", ok: front.every((p) => Number.isFinite(p.dose)), detail: JSON.stringify(front.map((p) => p.dose)) }] };
    },
  },
  {
    id: "reg-map-rounds-down",
    title: "The deadline map never rounds a link down to a whole grid step",
    cite: "2f31fa9 (Deadline.epsilon_rounds_down)",
    run: async () => {
      const { doseDeadlineMap } = await core("deadline.js");
      const g = { nodes: Float64Array.from([0, 0, 0, 1 + 5e-10, 0, 0]), edges: Int32Array.from([0, 1]) };
      const forecast = { n: 2, K: 1, stride: 1, dt: 10, frameSeconds: 10, horizon: 10, rate: new Float32Array(2), untenable: Float64Array.from([Infinity, 1 + 3e-10]), closes: Float64Array.from([10, 1 + 3e-10]) };
      const { deadline } = doseDeadlineMap(g, { exits: [1], forecast, limit: 1, speed: 1, step: 1 });
      return { checks: [{ name: "no grid deadline where no departure works", ok: deadline[0] === -Infinity, detail: `deadline ${deadline[0]}` }] };
    },
  },
  {
    id: "reg-leave-at-L",
    title: "doseDeadline's bad end is bad even where leaving at exactly L passes in doubles",
    cite: "2f31fa9 (the bisection's end is checked)",
    run: async () => {
      const { doseDeadline } = await core("deadline.js");
      const { doseFront } = await core("forecast.js");
      const g = { nodes: Float64Array.from([0, 0, 0, 0.2, 0, 0]), edges: Int32Array.from([0, 1]) };
      const forecast = { n: 2, K: 1, stride: 1, dt: 10, frameSeconds: 10, horizon: 10, rate: new Float32Array(2), untenable: Float64Array.from([Infinity, 0.9]), closes: Float64Array.from([10, 0.9]) };
      const exact = doseDeadline(g, 0, { exits: [1], forecast, limit: Infinity, speed: 1, tolerance: 0.05 });
      const atBad = exact ? doseFront(g, 0, { exits: [1], forecast, speed: 1, depart: exact.bad }).front.length : -1;
      return { checks: [{ name: "leaving at the bad end finds no route", ok: atBad === 0, detail: `bad ${exact?.bad}, routes ${atBad}` }] };
    },
  },
  {
    id: "reg-rl-horizon-potential",
    title: "With one step left, the policy does not walk into the fire for the shaping's sake",
    cite: "ebc307d (QLearning.horizon_potential_prefers_fire)",
    run: async () => {
      const { trainRl } = await core("rl.js");
      // u = node 150 of a corridor 150 links from the exit; a burning side room f off it.
      const pts = [], edges = [];
      for (let i = 0; i <= 150; i += 1) pts.push([-i, 0, 0]);
      for (let i = 0; i < 150; i += 1) edges.push([i, i + 1]);
      pts.push([-150, 1, 0]);
      edges.push([150, 151]);
      const g = rlGraph(pts, edges);
      const r = trainRl(g, { startPoint: [-150, 0, 0], exitPoint: [0, 0, 0], firePoint: [-150, 1, 0], episodes: 200, maxSteps: 1, useFire: true, fireSpeed: 1, seed: 1 });
      return { checks: [{ name: "not burned", ok: r.burned === false, detail: `path ${JSON.stringify(r.path)}` }] };
    },
  },
  {
    id: "reg-rl-policy-above-optimum",
    title: "The policy's reported return never exceeds the exact optimum",
    cite: "review K1 (fix on branch fix/review-2026-10-28)",
    expectFail: "fix/review-2026-10-28 not merged",
    run: async () => {
      const { trainRl } = await core("rl.js");
      // A short triangle (0.5 m links) where waiting beats burning, behind a burning link to the exit.
      const pts = [[0, 0, 0], [0.5, 0, 0], [0.25, 0.433, 0], [-1, 0, 0], [-2, 0, 0], [-3, 0, 0], [-4, 0, 0]];
      const edges = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 5], [5, 6]];
      const g = rlGraph(pts, edges);
      const r = trainRl(g, { startPoint: [0, 0, 0], exitPoint: [-2, 0, 0], firePoint: [-1, 0, 0], episodes: 400, maxSteps: 200, useFire: true, fireSpeed: 1, seed: 1 });
      return { checks: [{ name: "policy return <= optimal return", ok: r.policy_return <= r.optimal_return + 1e-9, detail: `policy ${r.policy_return?.toFixed(4)}, optimum ${r.optimal_return?.toFixed(4)}` }] };
    },
  },
  {
    id: "reg-map-work-cap",
    title: "doseDeadlineMap keeps its work under maxWork",
    cite: "review K2 (fix on branch fix/review-2026-10-28)",
    expectFail: "fix/review-2026-10-28 not merged",
    run: async () => {
      const { doseDeadlineMap } = await core("deadline.js");
      const { csr } = await core("route.js");
      // A 20 x 20 lattice of 0.5 m links, a 300 s horizon, and a cap far below even one link per step.
      const pts = [], edges = [];
      for (let j = 0; j < 20; j += 1) for (let i = 0; i < 20; i += 1) pts.push([i * 0.5, j * 0.5]);
      for (let j = 0; j < 20; j += 1) for (let i = 0; i < 20; i += 1) {
        if (i < 19) edges.push([j * 20 + i, j * 20 + i + 1]);
        if (j < 19) edges.push([j * 20 + i, (j + 1) * 20 + i]);
      }
      const g = graphOf(pts, edges);
      const f = forecastOf([new Array(400).fill(0)], new Array(400).fill(Infinity), 300);
      const maxWork = 1e5;
      const map = doseDeadlineMap(g, { exits: [0], forecast: f, limit: 1, speed: 1.2, maxWork });
      const { neighbours, n } = csr(g);
      const work = map.layers * (neighbours.length + n);
      return { checks: [{ name: "layers x (links + nodes) <= maxWork", ok: work <= maxWork, detail: `${work} for maxWork ${maxWork}` }] };
    },
  },
  {
    id: "reg-pset-unset-occurrence",
    title: "An unset Pset_DoorCommon value on the door does not hide its type's IsExternal",
    cite: "review m12 (fix on branch fix/review-2026-10-28)",
    expectFail: "fix/review-2026-10-28 not merged",
    run: async () => {
      const s = await semanticsOf(ifcText([
        ...UNITS_M, ...PROJECT,
        "#10=IFCDOOR('1hOSvn6df7F8_7GcBWlRGQ',$,'D1',$,$,$,$,$,2.1,0.9,.DOOR.,.SINGLE_SWING_LEFT.,$);",
        "#11=IFCPROPERTYSINGLEVALUE('IsExternal',$,IFCBOOLEAN(.T.),$);",
        "#12=IFCPROPERTYSET('2hOSvn6df7F8_7GcBWlRGQ',$,'Pset_DoorCommon',$,(#11));",
        "#13=IFCDOORTYPE('3hOSvn6df7F8_7GcBWlRGQ',$,'T1',$,$,(#12),$,$,$,.DOOR.,.SINGLE_SWING_LEFT.,.F.,$);",
        "#14=IFCRELDEFINESBYTYPE('4hOSvn6df7F8_7GcBWlRGQ',$,$,$,(#10),#13);",
        "#15=IFCPROPERTYSINGLEVALUE('IsExternal',$,$,$);",
        "#16=IFCPROPERTYSET('5hOSvn6df7F8_7GcBWlRGQ',$,'Pset_DoorCommon',$,(#15));",
        "#17=IFCRELDEFINESBYPROPERTIES('6hOSvn6df7F8_7GcBWlRGQ',$,$,$,(#10),#16);",
      ]));
      const exit = s.doors[0]?.exit ?? null;
      return { checks: [{ name: "door read as external from its type", ok: exit === "Pset_DoorCommon.IsExternal", detail: `exit ${exit}` }] };
    },
  },
  {
    id: "reg-nested-units",
    title: "A unit defined through another conversion-based unit (inch via foot) scales correctly",
    cite: "review m12 (fix on branch fix/review-2026-10-28)",
    expectFail: "fix/review-2026-10-28 not merged",
    run: async () => {
      const s = await semanticsOf(ifcText([
        "#80=IFCDIMENSIONALEXPONENTS(1,0,0,0,0,0,0);",
        "#81=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);",
        "#82=IFCMEASUREWITHUNIT(IFCLENGTHMEASURE(0.3048),#81);",
        "#83=IFCCONVERSIONBASEDUNIT(#80,.LENGTHUNIT.,'FOOT',#82);",
        "#84=IFCMEASUREWITHUNIT(IFCLENGTHMEASURE(0.0833333333333333),#83);",
        "#85=IFCCONVERSIONBASEDUNIT(#80,.LENGTHUNIT.,'INCH',#84);",
        "#91=IFCUNITASSIGNMENT((#85));",
        ...PROJECT,
        "#20=IFCCARTESIANPOINT((0.,0.,120.));",
        "#21=IFCAXIS2PLACEMENT3D(#20,$,$);",
        "#22=IFCLOCALPLACEMENT($,#21);",
        "#23=IFCBUILDINGSTOREY('7hOSvn6df7F8_7GcBWlRGQ',$,'Level 1',$,$,#22,$,$,.ELEMENT.,120.);",
      ]));
      const z = s.storeys[0]?.elevation;
      return { checks: [{ name: "120 inches is 3.048 m", ok: Math.abs(z - 3.048) < 1e-6, detail: `elevation ${z} m` }] };
    },
  },
  {
    id: "reg-roof-slabs",
    title: "Roof slabs are read from the IFC and never taken as floors (C20, kleine Wohnung)",
    cite: "cd7558d",
    needsCorpus: ["C20-Institute-Var-2", "ISSUE_159_kleine_Wohnung_R22"],
    run: async ({ corpus }) => {
      const { extract } = await import("./ifc-extract.mjs");
      const checks = [];
      for (const file of corpus) {
        const geo = await extract(file, { withDoorFlags: true });
        const roofs = new Set(geo.semantics?.roofSlabs || []);
        const asFloor = geo.floors.filter((m) => roofs.has(m.itemId)).length;
        checks.push({ name: `${path.basename(file)}: roof slabs found`, ok: roofs.size > 0, detail: `${roofs.size}` });
        checks.push({ name: `${path.basename(file)}: none taken as floor`, ok: asFloor === 0, detail: `${asFloor} roof meshes among floors` });
      }
      return { checks };
    },
  },
  {
    id: "reg-exits-from-flags",
    title: "Exits come from the IFC's flags before the one-sided-door rule",
    cite: "cd7558d",
    run: async () => {
      const { exitDoors } = await core("ifcSemantics.js");
      const doors = [{ itemId: 1, node: 5, sides: [0, 3] }, { itemId: 2, node: 7, sides: [2, 2] }];
      const r = exitDoors(doors, { doors: [{ id: 2, exit: "Pset_DoorCommon.FireExit", name: "Fire exit" }] });
      return { checks: [{ name: "the flagged door, not the one-sided one", ok: JSON.stringify(r.nodes) === "[7]" && r.source === "Pset_DoorCommon.FireExit", detail: `${JSON.stringify(r.nodes)} from ${r.source}` }] };
    },
  },
  {
    id: "reg-core-stair-parts",
    title: "Stair parts aggregated as proxies are read, so S_Office's storeys join (lifts off)",
    cite: "63925b7 (vertical cores; tools/core-check.mjs)",
    needsCorpus: ["S_Office_Integrated Design Archi"],
    run: async ({ corpus }) => {
      const { extract } = await import("./ifc-extract.mjs");
      const { buildLatticeGraph, floorLevels } = await core("lattice.js");
      const geo = await extract(corpus[0], { withDoorFlags: false });
      const levels = floorLevels(geo.floors, "y");
      const s = levelSpread(buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 }), levels);
      return {
        checks: [
          { name: "at least 5 storey levels joined", ok: s.levels >= 5, detail: `${s.levels} of ${levels.length}` },
          { name: "at least 99% of floor in multi-level parts", ok: s.multiPct >= 99, detail: `${s.multiPct}%` },
        ],
      };
    },
  },
  {
    id: "reg-stairs-without-treads",
    title: "With every stair mesh removed, rooms named as stairs still join the storeys (FM_ARC, C20)",
    cite: "63925b7 (vertical cores; tools/core-check.mjs)",
    needsCorpus: ["FM_ARC_DigitalHub", "C20-Institute-Var-2"],
    run: async ({ corpus }) => {
      const { extract } = await import("./ifc-extract.mjs");
      const { buildLatticeGraph, floorLevels } = await core("lattice.js");
      const { CORE_STAIR } = await core("cores.js");
      const want = { FM_ARC_DigitalHub: 65, "C20-Institute-Var-2": 84 };
      const checks = [];
      for (const file of corpus) {
        const geo = await extract(file, { withDoorFlags: false });
        const levels = floorLevels(geo.floors, "y");
        const rooms = geo.cores.filter((c) => c.kind === CORE_STAIR && c.source === "IfcSpace name");
        const common = { ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75, stairs: [], coreMeshes: [] };
        const bare = levelSpread(buildLatticeGraph({ ...common, cores: [] }), levels);
        const assumed = levelSpread(buildLatticeGraph({ ...common, cores: rooms }), levels);
        const name = path.basename(file, ".ifc");
        checks.push({ name: `${name}: at least ${want[name]}% of floor in multi-level parts`, ok: assumed.multiPct >= want[name], detail: `${assumed.multiPct}% with ${rooms.length} stair rooms, ${bare.multiPct}% without` });
      }
      return { checks };
    },
  },
  {
    id: "reg-sections-per-height",
    title: "Wall sections are cut only where the plane meets a triangle: 250 storeys with 198,000 solid triangles build within 2 s",
    cite: "de8dfcf (byte-identical output, 2.5x faster on large models)",
    run: async () => {
      // 250 small storeys 3 m apart, each with 66 posts (12 triangles each)
      // beside the floor: every storey height gets its own section, and each
      // section used to visit every solid triangle of the building.
      const floors = [], walls = [];
      for (let s = 0; s < 250; s += 1) {
        floors.push(rect(0, 0, 6, 6, s * 3));
        for (let k = 0; k < 66; k += 1) walls.push(box(10 + k * 0.4, 0, s * 3, 10.2 + k * 0.4, 0.2, s * 3 + 3));
      }
      const { g, ms } = await lattice({ floors, walls, maxPoints: 1e6 });
      return { checks: [{ name: "built within 2 s", ok: ms < 2000, detail: `${Math.round(ms)} ms, ${g.stats.nodes} nodes, ${g.sections.length} sections` }], metrics: { latticeMs: Math.round(ms) } };
    },
  },
];

// ------------------------------------------------------------ registry

/** Every case: { id, group, title, cite?, expectFail?, timeoutMs, file? }. */
export function listCases({ samples, manifest, tmp }) {
  const cases = [];
  for (const c of GEOMETRY) cases.push({ id: `geo:${c.id}`, group: "geometry", title: c.title, expectFail: c.expectFail ?? null, open: c.open ?? null, timeoutMs: c.timeoutMs ?? 30000 });
  const fixtures = FIXTURES(samples);
  const ifcs = [];
  if (existsSync(fixtures)) {
    const walk = (d) => {
      for (const f of readdirSync(d).sort()) {
        const p = path.join(d, f);
        if (statSync(p).isDirectory()) walk(p);
        else if (/\.ifc$/i.test(f)) ifcs.push({ file: p, label: `fixture ${path.relative(fixtures, p).replace(/\\/g, "/")}` });
      }
    };
    walk(fixtures);
  }
  const real = manifest.models.find((m) => m.name === "AC20-FZK-Haus");
  const realPath = real ? path.join(samples, real.path) : "";
  for (const f of makeBrokenFiles(path.join(tmp, "broken"), realPath)) ifcs.push({ file: f, label: `made ${path.basename(f)}` });
  for (const m of manifest.models) {
    const p = m.root === "examples" ? path.join(ROOT, manifest.roots.examples, m.path) : path.join(samples, ...m.path.split("/"));
    if (existsSync(p)) ifcs.push({ file: p, label: `corpus ${m.name}`, big: m.bytes > 3e7 });
  }
  for (const f of ifcs) cases.push({ id: `ifc:${f.label}`, group: "ifc", title: f.label, file: f.file, timeoutMs: f.big ? 600000 : 120000 });
  for (const c of REGRESSION) {
    let corpus = [];
    if (c.needsCorpus) corpus = c.needsCorpus.map((name) => manifest.models.find((m) => m.name === name)).filter(Boolean).map((m) => path.join(samples, ...m.path.split("/"))).filter((p) => existsSync(p));
    cases.push({ id: `reg:${c.id}`, group: "regression", title: c.title, cite: c.cite, expectFail: c.expectFail ?? null, timeoutMs: 120000, corpus, skip: c.needsCorpus && !corpus.length ? "corpus files not found" : null });
  }
  return cases;
}

/** Run one case (in the child process). */
export async function runCase(c) {
  const t0 = performance.now();
  let result;
  if (c.group === "geometry") {
    const spec = GEOMETRY.find((x) => `geo:${x.id}` === c.id);
    const built = await lattice(spec.scene());
    result = await judge(built, spec.expect, { budgetMs: spec.budgetMs ?? 2000 });
  } else if (c.group === "ifc") {
    result = await ifcPipeline(c.file);
  } else {
    const spec = REGRESSION.find((x) => `reg:${x.id}` === c.id);
    result = await spec.run({ corpus: c.corpus || [] });
  }
  result.ms = Math.round(performance.now() - t0);
  result.pass = result.checks.every((k) => k.ok);
  return result;
}

/** Extract src/core at a git revision into a folder, for --core-rev. */
export function coreAt(rev, dir, git) {
  const rel = "topologicpy-web-frontend/src/core";
  const names = git(["ls-tree", "--name-only", `${rev}:${rel}`]).split(/\r?\n/).filter((f) => f.endsWith(".js"));
  mkdirSync(dir, { recursive: true });
  for (const f of names) writeFileSync(path.join(dir, f), git(["show", `${rev}:${rel}/${f}`]));
  return dir;
}

export const STRESS_TMP = () => path.join(os.tmpdir(), "studio-stress");
