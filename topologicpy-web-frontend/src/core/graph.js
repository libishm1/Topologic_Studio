/**
 * Graph statistics computed in the browser, in the shape the server returns
 * (GraphStats), so the UI reads them the same way with or without a backend.
 */
import { KIND_DOOR, KIND_FLOOR, KIND_LIFT, KIND_STAIR } from "./lattice.js";

/**
 * Connected components by union-find: count, and the largest one's size.
 * Links blocked by walls join nothing, as in the route search
 * (route.js componentLabels); counting them put Vienna at 95 % reachable
 * when its routes reached 90 %.
 */
export function components(nodeCount, edges, blocked = null) {
  const parent = new Int32Array(nodeCount);
  for (let i = 0; i < nodeCount; i += 1) parent[i] = i;
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (let e = 0; e + 1 < edges.length; e += 2) {
    if (blocked?.[e / 2]) continue;
    const a = find(edges[e]), b = find(edges[e + 1]);
    if (a !== b) parent[a] = b;
  }
  const sizes = new Map();
  for (let i = 0; i < nodeCount; i += 1) {
    const r = find(i);
    sizes.set(r, (sizes.get(r) || 0) + 1);
  }
  let largest = 0;
  for (const s of sizes.values()) if (s > largest) largest = s;
  return { count: sizes.size, largest };
}

export function graphStats({ points, edges, kinds, blocked, stats = {} }) {
  const nodes = points.length / 3;
  const { count, largest } = components(nodes, edges, blocked);
  let floor = 0, stair = 0, door = 0, lift = 0, blockedEdges = 0;
  for (const k of kinds) {
    if (k === KIND_FLOOR) floor += 1;
    else if (k === KIND_STAIR) stair += 1;
    else if (k === KIND_DOOR) door += 1;
    else if (k === KIND_LIFT) lift += 1;
  }
  for (const b of blocked || []) blockedEdges += b;
  return {
    nodes,
    edges: edges.length / 2,
    floor_nodes: floor,
    stair_nodes: stair,
    door_nodes: door,
    lift_nodes: lift,
    wall_segments: stats.walls ?? 0,
    blocked_edges: blockedEdges,
    components: count,
    largest_component: largest,
  };
}

/** Share of nodes, in percent, in the largest part. */
export const reachablePct = (stats) =>
  stats?.nodes ? Math.round((stats.largest_component / stats.nodes) * 100) : 0;

/**
 * Whether a graph's split into parts is worth a warning: more than one part
 * and the largest under 98%. A few stray nodes past an outside wall are not.
 */
export const graphIsSplit = (stats) => Boolean(stats && stats.components > 1 && reachablePct(stats) < 98);
