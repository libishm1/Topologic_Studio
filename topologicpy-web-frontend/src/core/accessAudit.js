/**
 * The wheelchair accessibility audit and accessible means of egress, on a
 * storey's measuring grid (floorRaster.js; wiki/roadmap/evacuation-map-and-accessibility-audit-plan-2026-11-11.md).
 *
 * - Reach: where a wheelchair's centre can go from the building's accessible
 *   ways in (exit doors on the storey with them, lifts elsewhere): the floor
 *   at least half the passage width clear, a door's own clear width at its
 *   doorway, no stairs, no change of level beyond a lip or a 1 in 12 slope.
 *   Rooms reached by a wheelchair against rooms reached on foot.
 * - Each room reached on foot but not by a wheelchair is diagnosed by
 *   relaxing one rule at a time: the doors (a door too narrow), the changes of
 *   level (a step or stairs only), the passage width (a passage too narrow,
 *   with the narrowest place on the way).
 * - Turning space: the floor the profile's turning circle sweeps, an opening
 *   by a disc (Matheron 1975; Serra 1982).
 * - The act step (after Poux 2026): the change that lets a wheelchair into the
 *   most rooms (a door widened, a ramp at a step), taken first, then the
 *   next, greedily.
 * - Accessible means of egress (IBC 2021 1009): from each room a wheelchair
 *   reaches, the exits, stairs (as areas of refuge) and lifts a wheelchair can
 *   get to; two where the storey needs two ways out. A stair counts as an area
 *   of refuge for assisted rescue only if 48 in (1.219 m) wide between
 *   handrails (1009.3.2); its enclosure and an elevator's standby power are
 *   not read.
 */
import { cellsNear, gridField, opening } from "./floorRaster.js";
import { edt } from "./floorRaster.js";

/**
 * Wheelchair profiles. The ICC A117.1-2017 / 2010 ADA values: accessible
 * route 36 in (403.5.1), door clear width 32 in (404.2.3), turning circle
 * 60 in (304.3.1), changes in level up to 1/2 in (303), ramps up to 1:12
 * (405.2); to check against the texts. The metric set is the one Poux's
 * audit uses; neither is SBC 201, which is to be read before any claim.
 */
export const PROFILES = Object.freeze({
  icc: { label: "Wheelchair, ICC A117.1-2017 and 2010 ADA", passage: 0.915, door: 0.815, turn: 1.525, levelChange: 0.013, slope: 1 / 12, clause: "ICC A117.1-2017 / 2010 ADA 403.5.1, 404.2.3, 304.3.1, 303, 405.2 (to check against the texts)" },
  metric: { label: "Wheelchair, 0.90 m passage, 0.80 m door, 1.50 m turning circle", passage: 0.9, door: 0.8, turn: 1.5, levelChange: 0.015, slope: 1 / 12, clause: "values as Poux's 2026 audit uses them; check the code that applies" },
});

/** IBC 2021 1009.3.2: an exit stair for assisted rescue, 48 in (1.219 m) clear between handrails. */
export const REFUGE_STAIR_WIDTH = 1.219;
/** How far round a door's position (m) its doorway's own clear width governs. */
const DOOR_REACH = 0.6;
/** The least clearance (m) a doorway cell must still have, however wide the door is declared. */
const DOOR_MIN_CLEAR = 0.3;
/** A walking body's radius (m), for the on-foot reach. */
const BODY = 0.25;

/** The cells within DOOR_REACH of each door, and each door's clear width: declared, else measured. */
function doorCells(R, doors) {
  const owner = new Int32Array(R.w * R.h).fill(-1);
  const widths = doors.map((d, i) => {
    const { cells } = cellsNear(R, d.x, d.y, DOOR_REACH);
    let widest = 0;
    for (const k of cells) {
      if (owner[k] < 0) owner[k] = i;
      if (Math.hypot(...cellOffset(R, k, d.x, d.y)) <= 0.3) widest = Math.max(widest, R.clear[k]);
    }
    const measured = widest > 0 ? 2 * widest + R.cell / 2 : null;
    return { declared: d.width > 0 ? d.width : null, measured, width: d.width > 0 ? d.width : measured };
  });
  return { owner, widths };
}

const cellOffset = (R, k, x, y) => [R.a0 + ((k % R.w) + 0.5) * R.cell - x, R.b0 + (Math.floor(k / R.w) + 0.5) * R.cell - y];

/** Where a wheelchair may be: the passage rule away from doors, the door's own width at them, never on a stair. */
function chairMask(R, profile, owner, doorOK, { passage = profile.passage } = {}) {
  const N = R.w * R.h;
  const out = new Uint8Array(N);
  for (let k = 0; k < N; k += 1) {
    if (!R.walk[k] || R.stair[k]) continue;
    const d = owner[k];
    out[k] = d >= 0 ? (doorOK[d] && R.clear[k] >= DOOR_MIN_CLEAR - 1e-9 ? 1 : R.clear[k] >= passage / 2 - 1e-9 ? 1 : 0) : R.clear[k] >= passage / 2 - 1e-9 ? 1 : 0;
  }
  return out;
}

/** Sources at plan points: the passable cells within `radius` of each, with their step to it. */
const sourcesAt = (R, points, mask, radius = 0.75) => points.map((p, i) => ({ ...cellsNear(R, p.x, p.y, radius, mask), id: i }));

/** Each room's cells reached in a field (finite distance). */
function roomsReached(R, dist) {
  const out = new Set();
  for (let k = 0; k < dist.length; k += 1) if (R.room[k] >= 0 && Number.isFinite(dist[k])) out.add(R.room[k]);
  return out;
}

/** Connected pieces of a mask (8 neighbours, the level rule), labelled from 0; -1 outside it. */
function components(R, mask, { levelChange = Infinity, slope = 0 } = {}) {
  const { w, h, height, cell } = R;
  const label = new Int32Array(w * h).fill(-1);
  let next = 0;
  const stack = [];
  for (let s = 0; s < w * h; s += 1) {
    if (!mask[s] || label[s] >= 0) continue;
    label[s] = next;
    stack.push(s);
    while (stack.length) {
      const c = stack.pop();
      const i = c % w, j = (c - i) / w;
      for (let dj = -1; dj <= 1; dj += 1) {
        for (let di = -1; di <= 1; di += 1) {
          if (!di && !dj) continue;
          const ni = i + di, nj = j + dj;
          if (ni < 0 || nj < 0 || ni >= w || nj >= h) continue;
          const nc = nj * w + ni;
          if (!mask[nc] || label[nc] >= 0) continue;
          if (di && dj && (!mask[j * w + ni] || !mask[nj * w + i])) continue;
          const len = di && dj ? Math.SQRT2 : 1;
          if (Math.abs(height[nc] - height[c]) > Math.max(levelChange, slope * len * cell) + 1e-9) continue;
          label[nc] = next;
          stack.push(nc);
        }
      }
    }
    next += 1;
  }
  return { label, count: next };
}

/** The narrowest place on a field's route from a cell back to its source: [cell, clearance]. */
function narrowestOnRoute(R, field, start) {
  let worst = -1, clear = Infinity;
  for (let c = start, guard = 0; c >= 0 && guard < R.w * R.h; c = field.parent[c], guard += 1) {
    if (R.clear[c] < clear) {
      clear = R.clear[c];
      worst = c;
    }
  }
  return [worst, clear];
}

/** Where a field's route from a cell crosses a change of level above the rule: plan points. */
function stepsOnRoute(R, field, start, profile) {
  const out = [];
  for (let c = start, guard = 0; c >= 0 && field.parent[c] >= 0 && guard < R.w * R.h; c = field.parent[c], guard += 1) {
    const p = field.parent[c];
    const len = Math.hypot((c % R.w) - (p % R.w), Math.floor(c / R.w) - Math.floor(p / R.w)) * R.cell;
    if (Math.abs(R.height[c] - R.height[p]) > Math.max(profile.levelChange, profile.slope * len) + 1e-9) out.push(c);
  }
  return out;
}

/**
 * Audit one storey.
 *
 * @param {object} R  storeyRaster's grid
 * @param {object} opts  profile (PROFILES key or object); rooms (roomsFromSpaces, for names and areas);
 *   doors [{x, y, width (declared, m, or null), label}]; entrances [{x, y, label}] (accessible ways in:
 *   exit doors on this storey, else lifts); ways [{x, y, label, kind: "exit"|"stair"|"lift"}] (every way
 *   out a person on foot uses: exits and stair entrances); required (accessible means of egress needed, 1 or 2)
 */
export function auditStorey(R, { profile = "icc", rooms = [], doors = [], entrances = [], ways = [], required = 2, actRounds = 4 } = {}) {
  const P = typeof profile === "string" ? PROFILES[profile] : profile;
  const { owner, widths } = doorCells(R, doors);
  const doorOK = widths.map((d) => d.width !== null && d.width >= P.door - 1e-9);
  const mask = chairMask(R, P, owner, doorOK);
  const rule = { levelChange: P.levelChange, slope: P.slope };
  const chair = gridField(R, sourcesAt(R, entrances, mask), { passable: mask, ...rule });
  const foot = gridField(R, sourcesAt(R, [...entrances, ...ways], R.body), { passable: R.body, levelChange: 0.35 });
  const byChair = roomsReached(R, chair.dist);
  const onFoot = roomsReached(R, foot.dist);
  // Turning space a wheelchair can reach.
  const turn = opening(R, R.clear, P.turn / 2, mask);
  const turnable = new Set();
  for (let k = 0; k < turn.length; k += 1) if (turn[k] && Number.isFinite(chair.dist[k]) && R.room[k] >= 0) turnable.add(R.room[k]);
  // Diagnose: one rule relaxed at a time.
  const allDoors = doors.map(() => true);
  const relaxed = {
    door: gridField(R, sourcesAt(R, entrances, chairMask(R, P, owner, allDoors)), { passable: chairMask(R, P, owner, allDoors), ...rule }),
    level: gridField(R, sourcesAt(R, entrances, mask), { passable: mask, levelChange: 0.35 }),
    passage: gridField(R, sourcesAt(R, entrances, chairMask(R, P, owner, allDoors, { passage: 2 * BODY })), { passable: chairMask(R, P, owner, allDoors, { passage: 2 * BODY }), ...rule }),
  };
  const reachedBy = Object.fromEntries(Object.entries(relaxed).map(([k, f]) => [k, roomsReached(R, f.dist)]));
  const nearestCell = (field, room) => {
    let best = -1;
    for (let k = 0; k < field.dist.length; k += 1) if (R.room[k] === room && Number.isFinite(field.dist[k]) && (best < 0 || field.dist[k] < field.dist[best])) best = k;
    return best;
  };
  const centre = (k) => [R.a0 + ((k % R.w) + 0.5) * R.cell, R.b0 + (Math.floor(k / R.w) + 0.5) * R.cell];
  const name = (r) => rooms[r]?.name || `Room ${r}`;
  const issues = [];
  const stepPoints = [];
  for (const r of onFoot) {
    if (byChair.has(r)) {
      if (!turnable.has(r)) issues.push({ kind: "No turning space", severity: "medium", room: r, name: name(r), measured: `no ${P.turn.toFixed(2)} m circle`, required: `${P.turn.toFixed(2)} m circle` });
      continue;
    }
    if (reachedBy.door.has(r)) {
      const narrow = doors.map((d, i) => ({ d, i })).filter(({ i }) => !doorOK[i]);
      // The narrow doors on the way: those whose doorway the relaxed route passes.
      const k = nearestCell(relaxed.door, r);
      const on = new Set();
      for (let c = k; c >= 0; c = relaxed.door.parent[c]) if (owner[c] >= 0 && !doorOK[owner[c]]) on.add(owner[c]);
      for (const i of on.size ? on : new Set(narrow.map((x) => x.i).slice(0, 1))) {
        const wd = widths[i];
        issues.push({ kind: "Door too narrow", severity: "high", room: r, name: name(r), door: i, at: [doors[i].x, doors[i].y], measured: wd.width !== null ? `${wd.width.toFixed(3)} m${wd.declared ? "" : " (measured)"}` : "not measured", required: `${P.door.toFixed(3)} m` });
      }
    } else if (reachedBy.level.has(r)) {
      const k = nearestCell(relaxed.level, r);
      const steps = stepsOnRoute(R, relaxed.level, k, P);
      if (steps.length) stepPoints.push(...steps.slice(0, 1).map((c) => ({ cell: c, room: r })));
      issues.push({ kind: "Step or stairs only", severity: "high", room: r, name: name(r), at: steps.length ? centre(steps[0]) : null, measured: "no level way in", required: "a step-free way" });
    } else if (reachedBy.passage.has(r)) {
      const k = nearestCell(relaxed.passage, r);
      const [pinch, clear] = narrowestOnRoute(R, relaxed.passage, k);
      issues.push({ kind: "Passage too narrow", severity: "high", room: r, name: name(r), at: pinch >= 0 ? centre(pinch) : null, measured: `${Math.max(0, 2 * clear).toFixed(2)} m`, required: `${P.passage.toFixed(3)} m` });
    } else {
      issues.push({ kind: "No step-free way in on this storey", severity: "high", room: r, name: name(r), measured: "reached by stairs only", required: "a lift or a level way" });
    }
  }
  widths.forEach((wd, i) => {
    if (wd.width === null) issues.push({ kind: "Door not measured", severity: "low", door: i, at: [doors[i].x, doors[i].y], measured: "no width in the model, none measurable", required: "check on site" });
  });
  // The act step: widen a door, or ramp a step; the change that reaches the most rooms first.
  const candidates = [
    ...doors.map((d, i) => ({ kind: "Widen door", door: i, at: [d.x, d.y], label: d.label || `door ${i + 1}` })).filter((c) => !doorOK[c.door]),
    ...stepPoints.map((s, i) => ({ kind: "Ramp the step", cell: s.cell, at: centre(s.cell), label: `step ${i + 1} (to ${name(s.room)})` })),
  ];
  const act = [];
  const applied = { doors: doorOK.slice(), ramps: [] };
  const reachWith = (state) => {
    const m = chairMask(R, P, owner, state.doors);
    // A ramp at a step: the level rule waived within a metre of it (the ramp itself is the designer's).
    let waive = null;
    if (state.ramps.length) {
      waive = new Uint8Array(R.w * R.h);
      for (const c of state.ramps) for (const k of cellsNear(R, ...centre(c), 1.0, R.walk).cells) waive[k] = 1;
    }
    const field = gridField(R, sourcesAt(R, entrances, m), { passable: m, ...rule, waive });
    return roomsReached(R, field.dist);
  };
  let base = byChair.size;
  for (let round = 0; round < actRounds && candidates.length; round += 1) {
    let best = null;
    for (const c of candidates) {
      if (c.done) continue;
      const state = { doors: applied.doors.slice(), ramps: applied.ramps.slice() };
      if (c.kind === "Widen door") state.doors[c.door] = true;
      else state.ramps.push(c.cell);
      const n = reachWith(state).size;
      if (!best || n > best.n) best = { c, n, state };
    }
    if (!best || best.n <= base) break;
    best.c.done = true;
    applied.doors = best.state.doors;
    applied.ramps = best.state.ramps;
    act.push({ kind: best.c.kind, label: best.c.label, at: best.c.at, rooms: best.n, gain: best.n - base });
    base = best.n;
  }
  // Accessible means of egress: per room a wheelchair reaches, the ways out it can get to.
  const { label } = components(R, mask, rule);
  const reachable = ways.map((wy) => {
    const set = new Set();
    for (const k of cellsNear(R, wy.x, wy.y, 0.75, mask).cells) set.add(label[k]);
    return set;
  });
  const meansOf = new Map();
  for (const r of byChair) {
    const comps = new Set();
    for (let k = 0; k < label.length; k += 1) if (R.room[k] === r && label[k] >= 0 && Number.isFinite(chair.dist[k])) comps.add(label[k]);
    const list = ways.map((wy, i) => ({ wy, i })).filter(({ i }) => [...reachable[i]].some((c) => comps.has(c)));
    meansOf.set(r, list.map(({ wy }) => wy));
  }
  const egress = [...meansOf].map(([r, list]) => {
    const exits = list.filter((x) => x.kind === "exit").length;
    const stairs = list.filter((x) => x.kind === "stair");
    const refuge = stairs.filter((x) => x.width === undefined || x.width === null || x.width >= REFUGE_STAIR_WIDTH).length;
    const lifts = list.filter((x) => x.kind === "lift").length;
    const count = exits + refuge + lifts;
    return { room: r, name: name(r), exits, refuges: refuge, narrowStairs: stairs.length - refuge, lifts, count, ok: count >= required };
  });
  return {
    profile: P,
    rooms: { total: new Set([...onFoot, ...byChair]).size, onFoot: onFoot.size, byChair: byChair.size, turnable: turnable.size },
    reached: { chair: byChair, foot: onFoot },
    issues,
    act,
    egress,
    doors: widths,
    fields: { chair, foot },
    turn,
    mask,
  };
}

/**
 * Each stair's clear width on this storey's grid: twice the largest
 * clearance inside its footprint (the walls and handrails cut over it),
 * against REFUGE_STAIR_WIDTH. Returns [{x, y, width}] for the stair pieces.
 */
export function stairWidths(R) {
  const N = R.w * R.h;
  const inside = new Uint8Array(N);
  for (let k = 0; k < N; k += 1) inside[k] = R.stair[k] && Number.isFinite(R.height[k]) ? 1 : 0;
  const sq = edt(inside, R.w, R.h);
  const { label, count } = components({ ...R, height: new Float32Array(N) }, inside);
  const out = Array.from({ length: count }, () => ({ best: 0, at: -1, cells: 0 }));
  for (let k = 0; k < N; k += 1) {
    if (label[k] < 0) continue;
    const s = out[label[k]];
    s.cells += 1;
    const c = Math.sqrt(sq[k]) * R.cell - R.cell / 2;
    if (c > s.best) {
      s.best = c;
      s.at = k;
    }
  }
  return out
    .filter((s) => s.cells * R.cell * R.cell >= 1 && s.at >= 0)
    .map((s) => ({ x: R.a0 + ((s.at % R.w) + 0.5) * R.cell, y: R.b0 + (Math.floor(s.at / R.w) + 0.5) * R.cell, width: 2 * s.best + R.cell / 2 }));
}
