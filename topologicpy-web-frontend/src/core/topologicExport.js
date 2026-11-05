/**
 * The walkable graph and what a simulation found on it, as the JSON that
 * TopologicPy reads back with Graph.ByJSONFile / Graph.ByJSONDictionary
 * (the layout Graph.ExportToJSON writes, checked against topologicpy 0.9.86):
 *
 *   { properties: {...},
 *     vertices: { "Vertex_0": { x, y, z, ...attributes }, ... },
 *     edges: { "Edge_0": { source: "Vertex_0", target: "Vertex_1", src, dst, ...attributes }, ... } }
 *
 * Coordinates are put back into the IFC's own world frame (metres, Z up),
 * the frame TopologicPy builds its graphs in, at floor level: the viewer
 * turns IFC points by (x, y, z) -> (x, z, -y), moves them by the model's
 * coordination matrix and lifts the graph by the agent height
 * (topologicpy-web-backend navigation_check.py, formal/fireegress Frames.lean),
 * and viewerToIfc undoes the three. Without the matrix only the turn and the
 * lift are undone, and the properties say so.
 */

const KIND_NAMES = ["floor", "stair", "door", "lift"];
const TURNS = {
  y: [
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
  ],
  z: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

const round = (v, digits = 3) => {
  const s = 10 ** digits;
  return Math.round(v * s) / s;
};

function invert3(a) {
  const [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]] = a;
  const c00 = a11 * a22 - a12 * a21, c01 = a12 * a20 - a10 * a22, c02 = a10 * a21 - a11 * a20;
  const det = a00 * c00 + a01 * c01 + a02 * c02;
  if (!(Math.abs(det) > 1e-12)) throw new Error("The coordination matrix cannot be inverted.");
  return [
    [c00 / det, (a02 * a21 - a01 * a22) / det, (a01 * a12 - a02 * a11) / det],
    [c01 / det, (a00 * a22 - a02 * a20) / det, (a02 * a10 - a00 * a12) / det],
    [c02 / det, (a01 * a20 - a00 * a21) / det, (a00 * a11 - a01 * a10) / det],
  ];
}

/**
 * The viewer's frame change, as the backend applies it: viewer = A ifc + b,
 * A the matrix's 3x3 block times the turn, b its translation plus the lift.
 *
 * @param {number[]|null} coordination  16 numbers, column-major (three.js)
 */
export function frameChange(coordination, upAxis = "y", lift = 0) {
  const turn = TURNS[upAxis];
  if (!turn) throw new Error(`Cannot export a graph whose up axis is ${upAxis}.`);
  const m = coordination && coordination.length === 16 && coordination.every(Number.isFinite) ? coordination : null;
  const block = m
    ? [
        [m[0], m[4], m[8]],
        [m[1], m[5], m[9]],
        [m[2], m[6], m[10]],
      ]
    : TURNS.z;
  const A = block.map((row) => [0, 1, 2].map((c) => row[0] * turn[0][c] + row[1] * turn[1][c] + row[2] * turn[2][c]));
  const b = m ? [m[12], m[13], m[14]] : [0, 0, 0];
  b["xyz".indexOf(upAxis)] += lift;
  return { A, b, exact: Boolean(m) };
}

/** Viewer-frame points (x, y, z triples) back to IFC world coordinates. */
export function viewerToIfc(points, change) {
  const inv = invert3(change.A);
  const out = new Float64Array(points.length);
  for (let i = 0; i < points.length; i += 3) {
    const d = [points[i] - change.b[0], points[i + 1] - change.b[1], points[i + 2] - change.b[2]];
    for (let r = 0; r < 3; r += 1) out[i + r] = inv[r][0] * d[0] + inv[r][1] * d[1] + inv[r][2] * d[2];
  }
  return out;
}

/**
 * The graph, and what a run found on it, as TopologicPy graph JSON.
 *
 * @param {object} args
 *   graph         the walkable graph: points (viewer frame, lifted), edges (pairs), kinds, upAxis, agentHeight
 *   coordination  the model's coordination matrix (ViewerManager.getCoordinationMatrix), or null
 *   route         node indices of the route on screen, start to exit, or null
 *   frames        a Flamescapes run's frames ({time, T, vis, co, hrr}), or null
 *   burning       the graph fire models' burning nodes at the end of the run, or null
 *   tenability    {untenableC, smoke, visibilityLimit}: when a node counts as untenable
 *   properties    anything else the graph should carry (file, fire model, settings)
 */
export function topologicGraph({ graph, coordination = null, route = null, frames = null, burning = null, tenability = {}, properties = {} }) {
  const lift = graph.agentHeight || 0;
  const change = frameChange(coordination, graph.upAxis || "y", lift);
  const ifc = viewerToIfc(graph.points, change);
  const n = graph.points.length / 3;
  const onRoute = new Map((route || []).map((v, i) => [v, i]));
  const { untenableC = 60, smoke = true, visibilityLimit = 10 } = tenability;
  const K = frames?.length || 0;

  const vertices = {};
  for (let v = 0; v < n; v += 1) {
    const entry = {
      x: round(ifc[v * 3], 4),
      y: round(ifc[v * 3 + 1], 4),
      z: round(ifc[v * 3 + 2], 4),
      index: v,
      kind: KIND_NAMES[graph.kinds?.[v] ?? 0] || "floor",
      route_order: onRoute.has(v) ? onRoute.get(v) : -1,
    };
    if (burning) entry.burning_at_end = burning.has(v) ? 1 : 0;
    if (K) {
      const T = [], vis = [], co = [];
      let untenable = -1;
      for (const f of frames) {
        T.push(round(f.T[v], 1));
        vis.push(round(f.vis[v], 2));
        co.push(round(f.co[v], 1));
        if (untenable < 0 && (f.T[v] > untenableC || (smoke && f.vis[v] < visibilityLimit))) untenable = round(f.time, 2);
      }
      entry.gas_T_max_C = Math.max(...T);
      entry.visibility_min_m = Math.min(...vis);
      entry.co_max_ppm = Math.max(...co);
      entry.untenable_s = untenable;
      entry.gas_T_C = T;
      entry.visibility_m = vis;
      entry.co_ppm = co;
    }
    vertices[`Vertex_${v}`] = entry;
  }

  const edges = {};
  const E = graph.edges.length / 2;
  for (let e = 0; e < E; e += 1) {
    const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1];
    const ka = KIND_NAMES[graph.kinds?.[a] ?? 0], kb = KIND_NAMES[graph.kinds?.[b] ?? 0];
    const length = Math.hypot(ifc[a * 3] - ifc[b * 3], ifc[a * 3 + 1] - ifc[b * 3 + 1], ifc[a * 3 + 2] - ifc[b * 3 + 2]);
    edges[`Edge_${e}`] = {
      source: `Vertex_${a}`,
      target: `Vertex_${b}`,
      src: a,
      dst: b,
      kind: ka === kb ? ka : ka === "floor" ? kb : ka,
      length_m: round(length, 4),
      on_route: onRoute.has(a) && onRoute.has(b) && Math.abs(onRoute.get(a) - onRoute.get(b)) === 1 ? 1 : 0,
    };
  }

  return {
    properties: {
      generator: "Topologic Studio",
      frame: change.exact
        ? "IFC world coordinates, metres, Z up, at floor level (the viewer's turn, the model's coordination matrix and the agent lift undone)"
        : "IFC axes, metres, Z up, at floor level; the model's coordination matrix was not available, so positions are relative to web-ifc's origin, not the IFC world",
      agent_height_m: round(lift, 3),
      vertex_count: n,
      edge_count: E,
      ...(K
        ? {
            frame_times_s: frames.map((f) => round(f.time, 2)),
            hrr_kW: frames.map((f) => round(f.hrr, 1)),
            untenable_C: untenableC,
            smoke_untenable: smoke ? 1 : 0,
            visibility_limit_m: visibilityLimit,
          }
        : {}),
      ...properties,
    },
    vertices,
    edges,
  };
}
