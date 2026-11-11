/**
 * The egress map and the accessibility audit as one run each, for the worker
 * (workers/egress.worker.js) and the headless check (tools/egress-map-check.mjs):
 * core/egressMap.js on the walkable graph, core/floorRaster.js and
 * core/accessAudit.js on each storey's measuring grid, and what the viewer
 * draws from them.
 */
import { egressMap, prepareEgress, exitField } from "./egressMap.js";
import { compareFields, fastMarching, gridField, storeyRaster, widthClasses, WIDTH_CLASSES, cellsNear } from "./floorRaster.js";
import { auditStorey, stairWidths } from "./accessAudit.js";
import { roomsFromSpaces } from "./sprinklers.js";
import { storeysOf } from "./codeCheck.js";
const KIND_DOOR = 2;
const KIND_LIFT = 3;

/** Segments (viewer frame, six numbers each) along routes given as node lists. */
function routeSegments(P, routes) {
  const out = [];
  for (const route of routes) for (let i = 1; i < route.length; i += 1) out.push(P[route[i - 1] * 3], P[route[i - 1] * 3 + 1], P[route[i - 1] * 3 + 2], P[route[i] * 3], P[route[i] * 3 + 1], P[route[i] * 3 + 2]);
  return Float32Array.from(out);
}

/** Segments of the graph's links between nodes of a set. */
function setSegments(graph, nodes) {
  const inSet = new Uint8Array(graph.nodes.length / 3);
  for (const v of nodes) inSet[v] = 1;
  const P = graph.nodes, E = graph.edges;
  const out = [];
  for (let e = 0; e < E.length / 2; e += 1) {
    const a = E[e * 2], b = E[e * 2 + 1];
    if (inSet[a] && inSet[b]) out.push(P[a * 3], P[a * 3 + 1], P[a * 3 + 2], P[b * 3], P[b * 3 + 1], P[b * 3 + 2]);
  }
  return Float32Array.from(out);
}

export function rooms(geometry, names, upAxis) {
  if (!geometry?.spaces?.length) return [];
  const overhead = [...(geometry.floors || []), ...(geometry.roofMeshes || [])];
  const found = roomsFromSpaces(geometry.spaces, { upAxis, names, overhead, ceilings: geometry.ceilings, walls: geometry.walls });
  return (found?.rooms ? found.rooms : found) || [];
}

/** The map, with what the viewer draws made here. */
export function runMap(graph, opts, roomList, widthAt = null) {
  const result = egressMap(graph, { ...opts, rooms: roomList, widthAt });
  const P = graph.nodes;
  result.routeLinks = routeSegments(P, result.rooms.filter((r) => r.route.length > 1).map((r) => r.route));
  result.deadLinks = result.deadEnds ? setSegments(graph, result.deadEnds.corridors.flatMap((c) => c.nodes)) : new Float32Array(0);
  if (result.deadEnds) for (const c of result.deadEnds.corridors) delete c.nodes;
  for (const r of result.rooms) delete r.route;
  return result;
}

/** RGBA for a storey: the walk to the nearest exit in 5 m bands up to the limit, red beyond, grey where it reaches none. */
function distanceImage(R, dist, limit) {
  const N = R.w * R.h;
  const out = new Uint8ClampedArray(N * 4);
  const top = Number.isFinite(limit) ? limit : 60;
  // A sequential ramp, light near an exit to dark far from it.
  const ramp = [[254, 240, 138], [163, 230, 53], [52, 211, 153], [34, 211, 238], [59, 130, 246], [99, 102, 241], [124, 58, 237], [76, 29, 149]];
  for (let k = 0; k < N; k += 1) {
    if (!R.walk[k]) continue;
    const d = dist[k];
    let c;
    if (!Number.isFinite(d)) c = [148, 163, 184];
    else if (d > top) c = [220, 38, 38];
    else c = ramp[Math.min(ramp.length - 1, Math.floor((d / top) * ramp.length))];
    // A contour every 5 m.
    const edge = Number.isFinite(d) && d % 5 < R.cell * 1.2 && d > R.cell;
    out[k * 4] = edge ? 30 : c[0];
    out[k * 4 + 1] = edge ? 41 : c[1];
    out[k * 4 + 2] = edge ? 59 : c[2];
    out[k * 4 + 3] = 210;
  }
  return out;
}

/** RGBA for the audit: green where a wheelchair goes, darker where it can turn; amber on foot only; red reached by neither. */
function accessImage(R, audit) {
  const N = R.w * R.h;
  const out = new Uint8ClampedArray(N * 4);
  const chair = audit.fields.chair.dist, foot = audit.fields.foot.dist;
  for (let k = 0; k < N; k += 1) {
    if (!R.walk[k]) continue;
    let c;
    if (Number.isFinite(chair[k])) c = audit.turn[k] ? [21, 128, 61] : [74, 222, 128];
    else if (Number.isFinite(foot[k])) c = [245, 158, 11];
    else if (R.body[k]) c = [220, 38, 38];
    else c = [203, 213, 225];
    out[k * 4] = c[0];
    out[k * 4 + 1] = c[1];
    out[k * 4 + 2] = c[2];
    out[k * 4 + 3] = 200;
  }
  return out;
}

export function runFine(graph, opts, geometry, roomList, names) {
  const t0 = performance.now();
  const P = graph.nodes;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const [ha, hb] = [0, 1, 2].filter((c) => c !== up);
  const n = P.length / 3;
  const kinds = graph.kinds || new Uint8Array(n);
  const storeyOf = storeysOf(graph, opts.levels);
  const base = prepareEgress(graph, { exits: opts.exits, levels: opts.levels, useWalls: opts.useWalls, rooms: roomList });
  const graphField = exitField(base);
  const floors = base.floors;
  const declared = opts.doorWidth || new Float32Array(n);
  const storeys = [];
  const nodeWidth = new Float32Array(n);
  const images = [];
  for (let s = 0; s < floors.length; s += 1) {
    const level = floors[s];
    const seeds = [];
    for (let v = 0; v < n; v += 1) if (storeyOf[v] === s && kinds[v] !== KIND_DOOR) seeds.push([P[v * 3 + ha], P[v * 3 + hb]]);
    if (!seeds.length) continue;
    const R = storeyRaster(geometry, { level, upAxis: graph.upAxis || "y", seeds, rooms: roomList, spaceNames: names, cell: opts.cell || 0.05, maxCells: opts.maxCells || 2.5e6 });
    if (!R) continue;
    // Sources: this storey's exit entrances (exit doors, stair entrances), each from the plan point of its nodes.
    const ways = [];
    base.groupOf[s].forEach((g, v) => ways.push({ x: P[v * 3 + ha], y: P[v * 3 + hb], g, kind: base.groups[g].kind === "door" ? "exit" : "stair" }));
    const sources = ways.map((w) => ({ ...cellsNear(R, w.x, w.y, 0.75, R.body), id: w.g }));
    const field = gridField(R, sources, { passable: R.body, levelChange: 0.35 });
    let far = 0, farAt = -1, sum = 0, count = 0, none = 0;
    for (let k = 0; k < field.dist.length; k += 1) {
      if (!R.body[k]) continue;
      const d = field.dist[k];
      if (!Number.isFinite(d)) {
        none += 1;
        continue;
      }
      if (d > far) {
        far = d;
        farAt = k;
      }
      sum += d;
      count += 1;
    }
    // The graph's farthest walk on this storey, for the comparison.
    let graphFar = 0;
    for (let v = 0; v < n; v += 1) if (storeyOf[v] === s && kinds[v] !== KIND_DOOR && Number.isFinite(graphField.dist[v])) graphFar = Math.max(graphFar, graphField.dist[v]);
    const check = opts.fastMarching ? compareFields(field.dist, fastMarching(R, sources, { passable: R.body }), R.body) : null;
    // Each graph node's clear width on the grid, for the dead-end exception.
    for (let v = 0; v < n; v += 1) {
      if (storeyOf[v] !== s) continue;
      const i = Math.floor((P[v * 3 + ha] - R.a0) / R.cell), j = Math.floor((P[v * 3 + hb] - R.b0) / R.cell);
      if (i < 0 || j < 0 || i >= R.w || j >= R.h) continue;
      let best = 0;
      for (const k of cellsNear(R, P[v * 3 + ha], P[v * 3 + hb], 0.3).cells) best = Math.max(best, R.clear[k]);
      if (best > 0) nodeWidth[v] = 2 * best + R.cell / 2;
    }
    // The audit.
    const doors = [];
    for (let v = 0; v < n; v += 1) if (storeyOf[v] === s && kinds[v] === KIND_DOOR) doors.push({ x: P[v * 3 + ha], y: P[v * 3 + hb], width: declared[v] > 0 ? declared[v] : null, label: opts.doorNames?.[v] || null, node: v });
    const stairList = stairWidths(R);
    const lifts = [];
    for (let v = 0; v < n; v += 1) if (storeyOf[v] === s && kinds[v] === KIND_LIFT) lifts.push({ x: P[v * 3 + ha], y: P[v * 3 + hb], kind: "lift" });
    const discharge = base.discharge.has(s);
    const entrances = discharge ? ways.filter((w) => w.kind === "exit") : lifts;
    const wayList = [
      ...ways.filter((w) => w.kind === "exit"),
      ...ways.filter((w) => w.kind === "stair").map((w) => {
        let width = null, best = Infinity;
        for (const st of stairList) {
          const d = Math.hypot(st.x - w.x, st.y - w.y);
          if (d < best && d < 6) {
            best = d;
            width = st.width;
          }
        }
        return { ...w, width };
      }),
      ...lifts,
    ];
    const occupants = Math.ceil(R.bodyArea / (opts.occupantArea || 9));
    const audit = auditStorey(R, { profile: opts.profile || "icc", rooms: roomList, doors, entrances, ways: wayList, required: occupants > 49 ? 2 : 1 });
    const widths = widthClasses(R);
    let narrowArea = 0;
    const passageClass = WIDTH_CLASSES.indexOf(audit.profile.passage);
    for (let k = 0; k < widths.length; k += 1) if (R.walk[k] && widths[k] < passageClass) narrowArea += R.cell * R.cell;
    const at = (k) => {
      const p = [0, 0, 0];
      p[ha] = R.a0 + ((k % R.w) + 0.5) * R.cell;
      p[hb] = R.b0 + (Math.floor(k / R.w) + 0.5) * R.cell;
      p[up] = level;
      return p;
    };
    const lift3 = (xy) => {
      if (!xy) return null;
      const p = [0, 0, 0];
      p[ha] = xy[0];
      p[hb] = xy[1];
      p[up] = level;
      return p;
    };
    storeys.push({
      storey: s,
      level,
      cell: R.cell,
      cells: R.w * R.h,
      walkArea: R.walkArea,
      bodyArea: R.bodyArea,
      farthest: far,
      farthestAt: farAt >= 0 ? at(farAt) : null,
      graphFarthest: graphFar,
      mean: count ? sum / count : 0,
      noWayOut: none * R.cell * R.cell,
      fastMarching: check,
      narrowArea,
      stairs: stairList.map((st) => ({ width: st.width, at: lift3([st.x, st.y]) })),
      audit: {
        profile: audit.profile.label,
        clause: audit.profile.clause,
        rooms: audit.rooms,
        issues: audit.issues.map((i) => ({ ...i, at: lift3(i.at) })),
        act: audit.act.map((a) => ({ ...a, at: lift3(a.at) })),
        egress: audit.egress,
        required: occupants > 49 ? 2 : 1,
      },
    });
    images.push({ storey: s, level, a0: R.a0, b0: R.b0, cell: R.cell, w: R.w, h: R.h, distance: distanceImage(R, field.dist, opts.limit), access: accessImage(R, audit) });
  }
  const map = runMap(graph, opts, roomList, nodeWidth);
  return { storeys, images, map, ms: performance.now() - t0 };
}

