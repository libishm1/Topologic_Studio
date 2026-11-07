/**
 * Sprinklers, step 1 of the suppression work: heads placed on a model by the
 * codes, heads read from an IFC that already has them, and a rule report. No
 * simulation (wiki/research/fire-suppression-sprinklers-2026-11-07.md,
 * sections 2 to 4 and 10).
 *
 * The codes, as the egress check reads them (core/codeCheck.js): it has to
 * pass SBC 801-CR-2018, with IBC 2021 where that is stricter (the user's
 * rule, 2026-11-07). Both require installation to NFPA 13 (903.3.1.1), so the
 * placement values are NFPA 13's. They come from secondary sources, and only
 * the values two or more of them agree on are used
 * (wiki/research/sprinkler-codes-2026-11-07.md, section 0 rows 22 to 53 and
 * section 3). Each clause is cited in the edition its source cites, because
 * NFPA renumbered the standard in 2019. The licensed SBC 801-CR-2018 copy is
 * cited by clause number and value only.
 *
 * Plan frame. Positions are in the viewer frame. "Plan" means its two
 * horizontal axes in index order (x and z when y is up), so a plan point
 * [a, b] is (position[h0], position[h1]).
 *
 * The head record, the same for placed heads and IFC heads (the solver's
 * input in step 2):
 *   { id, source: "placed" | "ifc",
 *     position: [x, y, z]   viewer frame, the link point,
 *     room, roomName        the room's id and name, or null,
 *     hazard                "light" | "oh1" | "oh2",
 *     response              "quick" | "standard",
 *     rti                   (m s)^1/2,
 *     c                     conduction factor, (m/s)^1/2,
 *     activationC           °C,
 *     kFactor               L/(min bar^1/2),
 *     flowLpm               L/min (the pressure follows: P = (Q/K)², pressureOf),
 *     coverage              m², the head's protection area,
 *     footprint: { centre: [a, b], axis: [ua, ub], half: [hu, hv] }
 *                           the protection rectangle in plan: half[0] along
 *                           axis, half[1] along axis turned a quarter left,
 *                           [-ub, ua],
 *     provenance: { field: "ifc" | "default" | "user" } }
 * The storey fallback's heads (layoutStoreys) add provenance.ceiling =
 * "unknown": their up coordinate is the storey's floor, and the fire grid
 * gives the ceiling later. Heads placed around obstructions add
 * provenance.obstruction = "moved" | "beneath".
 *
 * Obstructions (beams, ducts, pipes, cable carriers, lights) from a
 * structural or MEP model are checked by NFPA 13's beam rule, three times
 * rule and wide-obstruction rule, and the room layout moves heads clear of
 * them and adds heads beneath wide ones (checkObstructions, layoutRooms'
 * `obstructions`; codes report section 3.4). Without them the obstructions
 * check cannot tell.
 */

import { EGRESS_B, storeysOf } from "./codeCheck.js";

/** Both codes require NFPA 13; every placement check carries this. */
const CODES = "SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: installed to NFPA 13 (editions by reference: 2013 through IFC 2015, 2019 in IBC 2021)";
const withCodes = (...rules) => [CODES, ...rules].join("; ");

const WALL_MAX = Object.freeze({
  fraction: 0.5,
  clause: "NFPA 13-2019 10.2.5.2: at most half the allowable spacing from a wall, measured perpendicular to it",
});
const WALL_MIN = Object.freeze({ value: 0.102, clause: "NFPA 13-2019 10.2.5.3: at least 4 in (102 mm) from a wall" });
const BETWEEN = Object.freeze({ value: 1.8, clause: "NFPA 13-2019 10.2.5.4: at least 6 ft (1.8 m) apart, unless baffled" });
const DEFLECTOR = Object.freeze({
  min: 0.025,
  max: 0.305,
  clause: "NFPA 13-2019 10.2.6.1.1.1 (2016: 8.6.4.1.1.1): deflector 1 to 12 in (25 to 305 mm) below an unobstructed ceiling",
});
const DESIGN_AREA = Object.freeze({
  value: 139,
  clause: "NFPA 13-2022 Table 19.2.3.1.1: 1,500 sq ft (139 m²), the smallest design area on the density curve (two sources; partly confirmed)",
});
const OH_AREA = "NFPA 13-2019 Table 10.2.4.2.1(b) (2016: Table 8.6.2.2.1(b)): ordinary hazard, 130 sq ft (12.1 m²) per sprinkler";
const OH_SPACING = "NFPA 13-2019 Table 10.2.4.2.1(b): 15 ft (4.6 m)";

/**
 * NFPA 13 standard-spray values per hazard class, each with its clause
 * (codes report rows 22, 26, 29 to 33, 44 and 45; all confirmed by two or
 * more sources except the design area).
 */
export const HAZARDS = Object.freeze({
  light: Object.freeze({
    id: "light",
    label: "Light hazard",
    areaPerHead: { value: 20.9, clause: "NFPA 13-2019 Table 10.2.4.2.1(a) (2016: Table 8.6.2.2.1(a)): light hazard, hydraulically calculated, 225 sq ft (20.9 m²) per sprinkler" },
    maxSpacing: { value: 4.6, clause: "NFPA 13-2019 Table 10.2.4.2.1(a): 15 ft (4.6 m)" },
    wallMax: WALL_MAX,
    wallMin: WALL_MIN,
    smallRoom: {
      area: 74,
      wall: 2.7,
      clause: "NFPA 13-2019 10.2.5.2.3: in a light hazard room of 800 sq ft (74 m²) or less with unobstructed construction, up to 9 ft (2.7 m) from one wall",
    },
    minBetween: BETWEEN,
    deflector: DEFLECTOR,
    density: { value: 4.1, clause: "NFPA 13-2022 Table 19.2.3.1.1 (2016: Figure 11.2.3.1.1): light hazard 0.10 gpm/sq ft (4.1 mm/min)" },
    designArea: DESIGN_AREA,
  }),
  oh1: Object.freeze({
    id: "oh1",
    label: "Ordinary hazard group 1",
    areaPerHead: { value: 12.1, clause: OH_AREA },
    maxSpacing: { value: 4.6, clause: OH_SPACING },
    wallMax: WALL_MAX,
    wallMin: WALL_MIN,
    smallRoom: null,
    minBetween: BETWEEN,
    deflector: DEFLECTOR,
    density: { value: 6.1, clause: "NFPA 13-2022 Table 19.2.3.1.1 (2016: Figure 11.2.3.1.1): ordinary hazard group 1 0.15 gpm/sq ft (6.1 mm/min)" },
    designArea: DESIGN_AREA,
  }),
  oh2: Object.freeze({
    id: "oh2",
    label: "Ordinary hazard group 2",
    areaPerHead: { value: 12.1, clause: OH_AREA },
    maxSpacing: { value: 4.6, clause: OH_SPACING },
    wallMax: WALL_MAX,
    wallMin: WALL_MIN,
    smallRoom: null,
    minBetween: BETWEEN,
    deflector: DEFLECTOR,
    density: { value: 8.1, clause: "NFPA 13-2022 Table 19.2.3.1.1 (2016: Figure 11.2.3.1.1): ordinary hazard group 2 0.20 gpm/sq ft (8.1 mm/min)" },
    designArea: DESIGN_AREA,
  }),
});

const QUICK_REQUIRED =
  "SBC 801-CR-2018 903.3.2 and IBC 2021 903.3.2: quick-response or residential sprinklers in NFPA 13 light hazard occupancies and in dwelling and sleeping units; NFPA 13-2010 8.3.3.1 (one source, partly confirmed)";

/**
 * What a head is given when nothing says otherwise. No IFC carries RTI or C
 * (MEP report section 5.5), so every head states them (physics report
 * section 6.1).
 */
export const SPRINKLER_DEFAULTS = Object.freeze({
  response: Object.freeze({
    quick: { rti: 50, clause: "NFPA 13 definitions and ISO 6182-1: fast response, RTI 50 (m s)^1/2 or less" },
    standard: { rti: 100, clause: "NFPA 13 definitions: standard response, RTI 80 (m s)^1/2 or more; 100 is FDS's default" },
  }),
  /** Quick response where light hazard requires it, standard elsewhere. */
  responseFor: (hazard) => (hazard === "light" ? "quick" : "standard"),
  responseClause: QUICK_REQUIRED,
  c: { value: 0.4, clause: "conduction factor C, 0.4 (m/s)^1/2 from Bittern's tests (FDS Validation Guide); not in any IFC" },
  activationC: {
    value: 68,
    clause: "NFPA 13 temperature ratings (2019 Table 7.2.4.1, numbering not checked): ordinary 57 to 77 °C for ceilings up to 38 °C; 68 °C is the red bulb",
  },
  kFactor: { value: 80, clause: "NFPA 13: Q = K √P; K5.6 gpm/psi^1/2 is K80.6 L/(min bar^1/2), the common K80 head" },
  minPressure: { value: 0.5, clause: "NFPA 13: no sprinkler below 7 psi (0.48 bar; NFPA's metric figure is 0.5 bar)" },
  deflectorBelowCeiling: { value: 0.1, clause: DEFLECTOR.clause },
  linkBelowDeflector: { value: 0.03, clause: "the link point is the deflector point minus about 0.03 m (sprinkler-physics report section 6.1)" },
  flow: "the hazard's design density times the head's protection area, but at least K √(0.5 bar); the pressure then follows, P = (Q/K)²",
});

/** Default flow of a head (L/min): density × area, at least the minimum pressure's flow. */
export function flowFor(hazard, coverage, kFactor = SPRINKLER_DEFAULTS.kFactor.value) {
  const H = hazardOf(hazard);
  return Math.max(H.density.value * coverage, kFactor * Math.sqrt(SPRINKLER_DEFAULTS.minPressure.value));
}

/** A head's pressure (bar) from its flow and K-factor, P = (Q/K)². */
export function pressureOf(head) {
  return head?.kFactor > 0 ? (head.flowLpm / head.kFactor) ** 2 : null;
}

function hazardOf(hazard) {
  const H = typeof hazard === "string" ? HAZARDS[hazard] : hazard;
  if (!H?.areaPerHead) throw new RangeError(`Unknown hazard class "${hazard}"; use one of ${Object.keys(HAZARDS).join(", ")}.`);
  return H;
}

// ------------------------------------------------------------------ hazard by name

const HAZARD_CLAUSE = "NFPA 13-2022 4.3.2 to 4.3.3.2 and annex A.4.3: examples by use; the designer classifies";

const HAZARD_NAMES = [
  {
    hazard: "oh1",
    pattern: /kitchen|k(ü|ue)che|مطبخ/i,
    reason: "a kitchen: ordinary hazard group 1 (commercial kitchens and restaurant service areas)",
  },
  {
    hazard: "oh1",
    pattern: /plant|mechanical|\bmech\b|electric|\belec\b|elektro|switch ?(room|gear)|transformer|generator|boiler|pump ?room|hvac|\bahu\b|machine ?room|motor ?room|technik|heiz|l(ü|ue)ftung|كهرباء|ميكانيك|مولد/i,
    reason: "a plant, mechanical or electrical room: ordinary hazard group 1",
  },
  {
    hazard: "oh1",
    pattern: /garage|parking|car ?park|carport|stellpl|موقف|مواقف|كراج|جراج/i,
    reason: "a garage or car park: ordinary hazard group 1",
  },
  {
    hazard: "oh1",
    pattern: /storage|store ?room|\bstores?\b|lager|abstell|مخزن|مستودع/i,
    reason:
      "storage: ordinary hazard group 1 for storage up to 2.4 m (8 ft) high; group 2 up to 3.7 m (12 ft) or for more combustible contents, and storage above 3.7 m is outside the occupancy classes: set the class by hand",
  },
];

/**
 * The hazard class of a room from its name: kitchens, plant, mechanical and
 * electrical rooms, garages and car parks are ordinary hazard group 1, and so
 * is storage, with its height caveat; anything else is light hazard
 * (offices, dwellings, classrooms, corridors, toilets, meeting rooms; codes
 * report section 3.6). The user can override it per room (layoutRooms'
 * hazardFor).
 *
 * @param {string|null} name
 * @returns {{hazard: string, reason: string, clause: string}}
 */
export function hazardOfRoom(name) {
  const text = String(name ?? "");
  for (const rule of HAZARD_NAMES) {
    if (rule.pattern.test(text)) return { hazard: rule.hazard, reason: rule.reason, clause: HAZARD_CLAUSE };
  }
  return {
    hazard: "light",
    reason: text.trim()
      ? "light hazard: offices, dwellings, classrooms, corridors, toilets and meeting rooms are; the name names no ordinary hazard use"
      : "light hazard: the room has no name",
    clause: HAZARD_CLAUSE,
  };
}

// ------------------------------------------------------------------ the rectangle

/** Floating-point slack on the rule comparisons (m, m²). */
const EPS = 1e-6;

/**
 * The plan's layout of a rectangular ceiling of L × W for a hazard class,
 * with S its largest spacing and A its area per head:
 *
 *   nx = ⌈L / S⌉, dx = L / nx            Lean `bays`; dx ≤ S is `spacing_le`
 *   ny = max(⌈W / S⌉, ⌈dx W / A⌉), dy = W / ny
 *                                        dy ≤ S is `spacing_le` on W (ny ≥ ⌈W / S⌉
 *                                        only shortens dy); dx dy ≤ A is `area_le`
 *   heads at ((i + 1/2) dx, (j + 1/2) dy)
 *                                        Lean `centre`; dx/2, dy/2 ≤ S/2 from the
 *                                        walls is `wall_distance_le`; every point
 *                                        within (dx/2, dy/2) of a head is `covered`,
 *                                        once per axis
 *
 * (formal/fireegress/FireEgress/Sprinklers.lean). The ceilings take 1e-9 off
 * L/S so a length that is a whole number of spacings does not gain a bay to
 * rounding; the comparisons allow 1e-6. Extra bays (`extra`) keep every one
 * of these: more bays along only shorten dx, which lowers the bound ny must
 * meet, and more across only shorten dy.
 *
 * @param {number} L  length along the first axis (m)
 * @param {number} W  width (m)
 * @param {string} hazard  "light" | "oh1" | "oh2"
 * @param {{extra?: number[]}} opts  bays to add along and across, [0, 0] by default
 * @returns {{hazard, L, W, S, A, nx, ny, dx, dy, heads: number[][], checks: object[], status: string}}
 *          heads: [x, y] from the rectangle's corner, x along L
 */
export function layoutRectangle(L, W, hazard = "light", { extra = [0, 0] } = {}) {
  const H = hazardOf(hazard);
  const S = H.maxSpacing.value;
  const A = H.areaPerHead.value;
  if (!(L > 0 && W > 0)) throw new RangeError(`A rectangle needs a positive length and width, not ${L} × ${W}.`);
  const nx = Math.max(1, Math.ceil(L / S - 1e-9)) + Math.max(0, extra[0] | 0);
  const dx = L / nx;
  const ny = Math.max(1, Math.ceil(W / S - 1e-9), Math.ceil((dx * W) / A - 1e-9)) + Math.max(0, extra[1] | 0);
  const dy = W / ny;
  const heads = [];
  for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) heads.push([(i + 0.5) * dx, (j + 0.5) * dy]);
  const between = nx > 1 && ny > 1 ? Math.min(dx, dy) : nx > 1 ? dx : ny > 1 ? dy : null;
  const checks = [
    {
      id: "spacing",
      status: dx <= S + EPS && dy <= S + EPS ? "pass" : "fail",
      text: `${nx} × ${ny} heads, ${dx.toFixed(2)} m apart along and ${dy.toFixed(2)} m across; at most ${S} m`,
      clause: withCodes(H.maxSpacing.clause),
    },
    {
      id: "wall-max",
      status: dx / 2 <= S / 2 + EPS && dy / 2 <= S / 2 + EPS ? "pass" : "fail",
      text: `${(dx / 2).toFixed(2)} and ${(dy / 2).toFixed(2)} m from the walls; at most ${S / 2} m`,
      clause: withCodes(H.wallMax.clause),
    },
    {
      id: "wall-min",
      status: Math.min(dx, dy) / 2 >= H.wallMin.value - EPS ? "pass" : "fail",
      text: `${(Math.min(dx, dy) / 2).toFixed(2)} m from the nearest wall; at least ${H.wallMin.value} m`,
      clause: withCodes(H.wallMin.clause),
    },
    between === null
      ? { id: "between", status: "n/a", text: "one head", clause: withCodes(H.minBetween.clause) }
      : {
          id: "between",
          status: between >= H.minBetween.value - EPS ? "pass" : "fail",
          text: `heads ${between.toFixed(2)} m apart at the closest; at least ${H.minBetween.value} m`,
          clause: withCodes(H.minBetween.clause),
        },
    {
      id: "area",
      status: dx * dy <= A + EPS ? "pass" : "fail",
      text: `${(dx * dy).toFixed(1)} m² per head; at most ${A} m²`,
      clause: withCodes(H.areaPerHead.clause),
    },
    {
      id: "coverage",
      status: "pass",
      text: "every point within half a bay of a head along each axis",
      clause: withCodes(H.maxSpacing.clause, H.wallMax.clause),
    },
  ];
  return { hazard: H.id, L, W, S, A, nx, ny, dx, dy, heads, checks, status: worst(checks) };
}

const RANK = { "n/a": 0, pass: 1, "cannot tell": 2, fail: 3 };
/** The worst status of a list of checks: fail, then cannot tell, then pass. */
function worst(checks) {
  let w = "n/a";
  for (const c of checks) if (RANK[c.status] > RANK[w]) w = c.status;
  return w;
}
/**
 * The worst status of the checks the geometry settles: all but the
 * obstructions check made with no structural or MEP model (OBSTRUCTIONS);
 * one made with a model counts.
 */
const placementOf = (checks) => worst(checks.filter((c) => c !== OBSTRUCTIONS));

// ------------------------------------------------------------------ rooms from IfcSpace

const AXIS = { x: 0, y: 1, z: 2 };
const upIndex = (upAxis) => AXIS[upAxis] ?? 1;
const planAxes = (upAxis) => [0, 1, 2].filter((i) => i !== upIndex(upAxis));
/** Faces within 10 degrees of horizontal, at most 5 cm above the space's lowest point, as rooms.js. */
const MIN_UP = Math.cos((10 * Math.PI) / 180);
const BOTTOM_BAND = 0.05;
/** A footprint smaller than this (m²), or narrower than this (m), is not a room. */
const MIN_ROOM_AREA = 0.5;
const MIN_ROOM_WIDTH = 0.6;

/**
 * Rooms from IfcSpace meshes, one per item (the meshes of one space are
 * pooled). The floor and ceiling are the space's lowest and highest points
 * along the up axis; its top is taken as the ceiling the heads hang from.
 * The footprint is its bottom faces projected to plan: the horizontal faces
 * at the lowest level that face down, or, on a mesh wound inside out where
 * none does, those that face up (lattice.js takes either for the same
 * reason; taking one sign keeps a double-sided bottom from counting twice).
 * Each footprint gets its minimum-area bounding rectangle (rotating calipers
 * over the convex hull), long side first.
 *
 * Rooms with no usable footprint (no bottom faces, less than 0.5 m² of them,
 * or narrower than 0.6 m: a shaft or a riser closet, which the room rules do
 * not cover) stay in the list with footprint null and a `problem`, and
 * layoutRooms reports them.
 *
 * @param {Array<{itemId, positions: Float32Array, indices: Uint32Array}>} spaceMeshes  viewer frame
 * @param {{upAxis?: string, names?: Map|object}} opts  names: itemId -> room name
 * @returns {Array<{id, name, upAxis, floor, ceiling, footprint: Float64Array|null, area: number,
 *           rect: {centre: number[], axis: number[], half: number[]}|null, problem: string|null}>}
 *          footprint: plan triangles, six numbers each
 */
export function roomsFromSpaces(spaceMeshes, { upAxis = "y", names = null } = {}) {
  const up = upIndex(upAxis);
  const [h0, h1] = planAxes(upAxis);
  const nameOf = (id) => (names instanceof Map ? names.get(id) : names?.[id]) ?? null;
  const groups = new Map();
  (spaceMeshes || []).forEach((mesh, i) => {
    if (!mesh) return;
    const k = mesh.itemId ?? `mesh:${i}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(mesh);
  });
  const rooms = [];
  for (const [id, meshes] of groups) {
    const room = { id, name: nameOf(id), upAxis, floor: null, ceiling: null, footprint: null, area: 0, rect: null, problem: null };
    rooms.push(room);
    let lo = Infinity, hi = -Infinity;
    for (const { positions: p } of meshes) {
      if (!p) continue;
      for (let o = up; o < p.length; o += 3) {
        if (p[o] < lo) lo = p[o];
        if (p[o] > hi) hi = p[o];
      }
    }
    if (!(hi > lo)) {
      room.problem = "the space has no geometry, or no height";
      continue;
    }
    room.floor = lo;
    room.ceiling = hi;
    const down = [], upward = [];
    for (const { positions: p, indices: ix } of meshes) {
      if (!p || !ix) continue;
      for (let t = 0; t + 2 < ix.length; t += 3) {
        const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
        if (a + 2 >= p.length || b + 2 >= p.length || c + 2 >= p.length) continue;
        if (Math.max(p[a + up], p[b + up], p[c + up]) > lo + BOTTOM_BAND) continue;
        const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
        const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const mag = Math.hypot(n[0], n[1], n[2]);
        if (mag < 1e-12 || Math.abs(n[up] / mag) < MIN_UP) continue;
        (n[up] < 0 ? down : upward).push(p[a + h0], p[a + h1], p[b + h0], p[b + h1], p[c + h0], p[c + h1]);
      }
    }
    const tris = down.length ? down : upward;
    let area = 0;
    for (let k = 0; k < tris.length; k += 6) area += triArea(tris, k);
    if (area < MIN_ROOM_AREA) {
      room.problem = area > 0
        ? `its floor faces cover ${area.toFixed(2)} m², less than ${MIN_ROOM_AREA} m²: a shaft or a sliver, not a room`
        : "no horizontal faces at the bottom of the space";
      continue;
    }
    const rect = minAreaRect(tris);
    if (!rect) {
      room.problem = "the floor faces are a line, not an area";
      continue;
    }
    if (2 * rect.half[1] < MIN_ROOM_WIDTH) {
      room.problem = `${(2 * rect.half[1]).toFixed(2)} m wide, less than ${MIN_ROOM_WIDTH} m: a shaft or a sliver, which the room rules do not cover`;
      continue;
    }
    room.footprint = Float64Array.from(tris);
    room.area = area;
    room.rect = rect;
  }
  return rooms;
}

const triArea = (T, k) => Math.abs((T[k + 2] - T[k]) * (T[k + 5] - T[k + 1]) - (T[k + 4] - T[k]) * (T[k + 3] - T[k + 1])) / 2;

/** Convex hull of plan points (Andrew's monotone chain), counter-clockwise. */
function convexHull(points) {
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 1e-12) lower.pop();
    lower.push(p);
  }
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 1e-12) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/**
 * The minimum-area rectangle around plan triangles: one of its sides lies
 * along an edge of the convex hull. Returned long side first (half[0] ≥
 * half[1]), the axis pointing into the half plane a > 0 (or along +b).
 */
function minAreaRect(tris) {
  const points = [];
  for (let k = 0; k < tris.length; k += 2) points.push([tris[k], tris[k + 1]]);
  const hull = convexHull(points);
  if (hull.length < 3) return null;
  let best = null;
  for (let i = 0; i < hull.length; i += 1) {
    const p = hull[i], q = hull[(i + 1) % hull.length];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (len < 1e-9) continue;
    const ux = (q[0] - p[0]) / len, uy = (q[1] - p[1]) / len;
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const r of hull) {
      const u = r[0] * ux + r[1] * uy, v = -r[0] * uy + r[1] * ux;
      if (u < u0) u0 = u;
      if (u > u1) u1 = u;
      if (v < v0) v0 = v;
      if (v > v1) v1 = v;
    }
    const area = (u1 - u0) * (v1 - v0);
    if (!best || area < best.area * (1 - 1e-9)) best = { area, ux, uy, u0, u1, v0, v1 };
  }
  if (!best || !(best.area > 0)) return null;
  const { ux, uy, u0, u1, v0, v1 } = best;
  const cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
  const centre = [cu * ux - cv * uy, cu * uy + cv * ux];
  let axis = [ux, uy];
  let half = [(u1 - u0) / 2, (v1 - v0) / 2];
  if (half[1] > half[0]) {
    axis = [-uy, ux];
    half = [half[1], half[0]];
  }
  if (axis[0] < -1e-12 || (Math.abs(axis[0]) <= 1e-12 && axis[1] < 0)) axis = [-axis[0], -axis[1]];
  return { centre, axis: [axis[0] + 0, axis[1] + 0], half };
}

/** Whether (x, y) lies in triangle k of T (six numbers each), boundary included. */
function inTriangle(T, k, x, y) {
  const ax = T[k], ay = T[k + 1], bx = T[k + 2], by = T[k + 3], cx = T[k + 4], cy = T[k + 5];
  const d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by);
  const d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy);
  const d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay);
  const eps = 1e-9;
  return !((d1 < -eps || d2 < -eps || d3 < -eps) && (d1 > eps || d2 > eps || d3 > eps));
}

/** A point-in-footprint test over a bucket grid of the triangles. */
function footprintTest(T) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let k = 0; k < T.length; k += 2) {
    x0 = Math.min(x0, T[k]);
    x1 = Math.max(x1, T[k]);
    y0 = Math.min(y0, T[k + 1]);
    y1 = Math.max(y1, T[k + 1]);
  }
  const cell = Math.max(1, Math.sqrt(((x1 - x0) * (y1 - y0)) / 4096));
  const cells = new Map();
  const key = (i, j) => i * 65536 + j;
  for (let k = 0; k < T.length; k += 6) {
    const i0 = Math.floor((Math.min(T[k], T[k + 2], T[k + 4]) - x0) / cell);
    const i1 = Math.floor((Math.max(T[k], T[k + 2], T[k + 4]) - x0) / cell);
    const j0 = Math.floor((Math.min(T[k + 1], T[k + 3], T[k + 5]) - y0) / cell);
    const j1 = Math.floor((Math.max(T[k + 1], T[k + 3], T[k + 5]) - y0) / cell);
    for (let i = i0; i <= i1; i += 1) {
      for (let j = j0; j <= j1; j += 1) {
        const c = key(i, j);
        if (!cells.has(c)) cells.set(c, []);
        cells.get(c).push(k);
      }
    }
  }
  return (x, y) => {
    if (x < x0 - 1e-9 || x > x1 + 1e-9 || y < y0 - 1e-9 || y > y1 + 1e-9) return false;
    const list = cells.get(key(Math.floor((x - x0) / cell), Math.floor((y - y0) / cell)));
    if (!list) return false;
    for (const k of list) if (inTriangle(T, k, x, y)) return true;
    return false;
  };
}

/**
 * The footprint's walls: triangle edges used once, less those with floor on
 * both sides (a T-junction in the tessellation splits an edge on one side
 * only, so its pieces also count once).
 */
function wallEdges(T, inside) {
  const vkey = (x, y) => `${Math.round(x * 1000)},${Math.round(y * 1000)}`;
  const edges = new Map();
  for (let k = 0; k < T.length; k += 6) {
    for (const [p, q] of [[0, 2], [2, 4], [4, 0]]) {
      const ax = T[k + p], ay = T[k + p + 1], bx = T[k + q], by = T[k + q + 1];
      const ka = vkey(ax, ay), kb = vkey(bx, by);
      if (ka === kb) continue;
      const id = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      const e = edges.get(id);
      if (e) e.count += 1;
      else edges.set(id, { count: 1, seg: [ax, ay, bx, by] });
    }
  }
  const walls = [];
  for (const { count, seg } of edges.values()) {
    if (count % 2 === 0) continue;
    const [ax, ay, bx, by] = seg;
    const len = Math.hypot(bx - ax, by - ay);
    const nx = -(by - ay) / len, ny = (bx - ax) / len;
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    if (inside(mx + 0.01 * nx, my + 0.01 * ny) && inside(mx - 0.01 * nx, my - 0.01 * ny)) continue;
    walls.push(seg);
  }
  return walls;
}

function distanceToSegment(px, py, [x0, y0, x1, y1]) {
  const vx = x1 - x0, vy = y1 - y0;
  const len2 = vx * vx + vy * vy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * vx + (py - y0) * vy) / len2)) : 0;
  return Math.hypot(px - (x0 + t * vx), py - (y0 + t * vy));
}

// ------------------------------------------------------------------ layout per room

/** The coverage check samples the footprint this often (m). */
export const SAMPLE_STEP = 0.25;
const DEFLECTOR_BELOW = SPRINKLER_DEFAULTS.deflectorBelowCeiling.value;
const LINK_BELOW = SPRINKLER_DEFAULTS.linkBelowDeflector.value;

/** The obstructions check with no structural or MEP model (checkObstructions makes the real one). */
const OBSTRUCTIONS = {
  id: "obstructions",
  status: "cannot tell",
  text: "no structural or MEP model: beams, ducts, lights and other obstructions were not checked",
  clause: withCodes(
    "NFPA 13-2019 Table 10.2.7.1.2 (beam rule)",
    "NFPA 13-2022 10.2.7.3.1.3 (three times rule, at most 24 in (610 mm))",
    "NFPA 13-2019 9.5.5.3 (beneath obstructions wider than 4 ft (1.2 m))",
    "NFPA 13-2019 9.5.6 (18 in (457 mm) clear below the deflector)",
  ),
};

/**
 * Place heads in rooms and report the rules for each.
 *
 * Per room, the rectangle rule (layoutRectangle) is laid over the room's
 * minimum-area bounding rectangle, and the footprint is sampled every 0.25 m.
 * Each bay of the grid keeps its head at the bay centre (the Lean grid) when
 * that point lies in the footprint at least 0.102 m from its walls; the head
 * then protects the whole bay. Otherwise the head is dropped, and each
 * connected part of the room inside that bay is protected another way:
 *   - by a neighbouring head whose protection rectangle can take it in
 *     while that head still reaches all of it within S/2 along each axis and
 *     protects at most A;
 *   - failing that, by a head of its own protecting the part's bounding box,
 *     at the box's centre or a sampled point of the part, chosen clear of
 *     the walls, then reaching the box within S/2, then 1.8 m from the other
 *     heads (a part no point reaches is halved, up to 4 times);
 *   - two heads still closer than 1.8 m are then made one where a single
 *     point meets the same rules for both rectangles.
 * The grid runs along the room's long side and along its short side, and the
 * one with fewer failing checks, then fewer heads, is kept.
 * Then the checks, on the heads as placed:
 *   spacing    every head reaches its whole protection rectangle within S/2
 *              along each axis, so no point of the room, walls included, is
 *              more than S/2 from a head (grid heads: `spacing_le`,
 *              `wall_distance_le`; the others are measured);
 *   area       each protection rectangle is at most A (`area_le`; the others
 *              are measured);
 *   coverage   every footprint sample lies in some head's rectangle
 *              (`covered` for the grid);
 *   wall-min, between, deflector, response: measured;
 *   obstructions: cannot tell, with no structural or MEP model.
 * Rooms drawn inside other rooms (overlaps): a space whose rooms cover 75%
 * of it is a gross-area or zone space and is not laid out; rooms covering
 * less are parts of it (parking bays in a garage) and are not laid out, the
 * room around them is. A room's status is its worst check, so it reads
 * "cannot tell" until obstructions can be checked; `placement` is the worst
 * of the other checks, the ones the geometry settles.
 *
 * Given obstructions (checkObstructions' records), each room takes those
 * over it (plan overlap with its floor, bottom between its floor and
 * ceiling; obstructions are indexed on a grid first), and its heads are
 * cleared of them (clearObstructions): a head failing the beam rule or the
 * three times rule moves within its protection rectangle to the nearest
 * point, in 0.05 m steps, that meets every rule; an obstruction wider than
 * 1.2 m with no head beneath gets heads beneath it, after the room's
 * others. The obstructions check is then a verdict on the heads as they
 * stand ({id, status, text, clause, rules: beam, three-times, wide; moved,
 * added, unresolved}) and counts in `placement`; heads moved or added say so
 * in provenance.obstruction ("moved" | "beneath"). Under structural members
 * that hang below the deflector the deflector check is obstructed
 * construction's. With no obstructions given (null or empty) the output is
 * as without the option.
 *
 * @param {object[]} rooms  roomsFromSpaces' rooms
 * @param {{hazardFor?: (room) => string|null|undefined, response?: "quick"|"standard", obstructions?: object[]}} opts
 *        hazardFor: the user's class for a room, or nothing for the name's
 *        (hazardOfRoom); response: for every room, else quick in light
 *        hazard and standard elsewhere; obstructions: the structural and MEP
 *        model's (checkObstructions)
 * @returns {{heads: object[], rooms: object[], notes: string[]}}
 *          rooms: {id, name, hazard, hazardReason, hazardSource, area, floor,
 *          ceiling, heads (count), grid, checks: [{id, status, text, clause}],
 *          status, placement}; rooms not laid out carry partOf (the room
 *          they lie in) or a footprint check that says why
 */
export function layoutRooms(rooms, { hazardFor = null, response = null, obstructions = null } = {}) {
  const heads = [];
  const out = [];
  const usable = (rooms || []).filter((r) => r?.footprint && r.rect);
  const { gross, partOf } = overlaps(usable);
  const model = Array.isArray(obstructions) && obstructions.length ? prepareObstructions(obstructions) : null;
  const query = model ? boxIndex(model.list.map((o) => o.box)) : null;
  const tally = { over: new Set(), moved: 0, added: 0, unresolved: 0 };
  for (const room of rooms || []) {
    if (!room?.footprint || !room.rect) {
      out.push({
        id: room?.id ?? null,
        name: room?.name ?? null,
        hazard: null,
        area: 0,
        heads: 0,
        checks: [{ id: "footprint", status: "cannot tell", text: room?.problem || "no usable footprint", clause: CODES }],
        status: "cannot tell",
      });
      continue;
    }
    const chosen = hazardFor ? hazardFor(room) : null;
    const byName = hazardOfRoom(room.name);
    const user = typeof chosen === "string" && HAZARDS[chosen];
    const hazard = user ? chosen : byName.hazard;
    const base = {
      id: room.id,
      name: room.name,
      hazard,
      hazardReason: user ? "set by the user" : byName.reason,
      hazardSource: user ? "user" : "name",
      area: room.area,
      floor: room.floor,
      ceiling: room.ceiling,
    };
    const held = gross.get(room);
    const host = partOf.get(room);
    if (held || host) {
      out.push({
        ...base,
        heads: 0,
        grid: null,
        checks: [
          {
            id: "footprint",
            status: "n/a",
            text: held
              ? `it holds ${held} other room${held === 1 ? "" : "s"} covering most of it: a gross-area or zone space; its rooms are laid out instead`
              : `it lies inside ${host.name ?? `room ${host.id}`}, which is laid out with it`,
            clause: CODES,
          },
        ],
        status: "n/a",
        partOf: host?.id ?? null,
      });
      continue;
    }
    const resp = response === "quick" || response === "standard" ? response : SPRINKLER_DEFAULTS.responseFor(hazard);
    const placed = layoutRoom(room, hazardOf(hazard), resp);
    if (model) {
      const obs = obstructionsOver(room, model.list, query);
      const cleared = clearObstructions(room, hazardOf(hazard), placed, obs);
      for (const o of obs) tally.over.add(o);
      tally.moved += cleared.moved.length;
      tally.added += cleared.added.length;
      tally.unresolved += cleared.unresolved.length;
      placed.heads = cleared.heads;
      placed.checks = placed.checks.map((c) =>
        c === OBSTRUCTIONS ? cleared.check : c.id === "deflector" && cleared.deflector ? cleared.deflector : c.id === "between" && cleared.between ? cleared.between : c,
      );
    }
    const provenance = {
      position: "default",
      hazard: user ? "user" : "default",
      response: resp === response ? "user" : "default",
      rti: "default",
      c: "default",
      activationC: "default",
      kFactor: "default",
      flowLpm: "default",
      coverage: "default",
    };
    placed.heads.forEach((h, k) => {
      heads.push(
        headRecord({
          id: `${room.id}/${k + 1}`,
          source: "placed",
          position: h.position,
          room,
          hazard,
          response: resp,
          coverage: h.coverage,
          footprint: h.footprint,
          provenance: h.obstruction ? { ...provenance, obstruction: h.obstruction } : { ...provenance },
        }),
      );
    });
    out.push({ ...base, heads: placed.heads.length, grid: placed.grid, checks: placed.checks, status: worst(placed.checks), placement: placementOf(placed.checks) });
  }
  const notes = defaultNotes();
  if (model) {
    const { list, dropped } = model;
    notes.push(
      `obstructions: ${list.length} given (${list.filter((o) => o.structural).length} structural), ${tally.over.size} over a room laid out` +
        `${dropped.length ? `; ${dropped.length} left out with no plan area or heights` : ""}; ` +
        `${tally.moved} head${tally.moved === 1 ? "" : "s"} moved clear of them, ${tally.added} added beneath obstructions wider than ${WIDE.width} m, ` +
        `${tally.unresolved} not fixed (each room's obstructions check says which)`,
      "obstructions: the beam rule's rows from 4 ft on rest on one source; pipes take the three times rule only, and in light and ordinary hazard it takes structural members only",
    );
  }
  return { heads, rooms: out, notes };
}

function defaultNotes() {
  const D = SPRINKLER_DEFAULTS;
  return [
    `RTI by response class: quick ${D.response.quick.rti}, standard ${D.response.standard.rti} (m s)^1/2 (${D.response.quick.clause}; ${D.response.standard.clause})`,
    `C ${D.c.value} (m/s)^1/2 (${D.c.clause})`,
    `activation ${D.activationC.value} °C (${D.activationC.clause})`,
    `K ${D.kFactor.value} L/(min bar^1/2) (${D.kFactor.clause})`,
    `flow: ${D.flow} (${D.minPressure.clause})`,
    `deflector ${D.deflectorBelowCeiling.value} m below the ceiling, link ${D.linkBelowDeflector.value} m below the deflector`,
  ];
}

/** Contained rooms covering at least this share of a space make it a gross-area or zone space. */
const GROSS_SHARE = 0.75;

/**
 * Rooms drawn inside other rooms: q is inside r when q is smaller, on r's
 * floor (within 0.5 m), and a point of q's largest floor triangle lies in r's
 * footprint. Where the rooms inside r cover at least GROSS_SHARE of it, r is a
 * gross-area or zone space over its rooms; otherwise they are parts of r
 * (Vienna's garage holds its eight parking bays, half its area).
 *
 * @returns {{gross: Map<object, number>, partOf: Map<object, object>}}
 *          gross: space -> rooms it holds; partOf: room -> the room it lies in
 */
function overlaps(rooms) {
  const gross = new Map();
  const partOf = new Map();
  rooms.forEach((r, i) => {
    const held = containers(rooms, i);
    if (!held.length) return;
    const share = held.reduce((sum, q) => sum + q.area, 0) / r.area;
    if (share >= GROSS_SHARE) gross.set(r, held.length);
    else for (const q of held) if (!partOf.has(q) || partOf.get(q).area > r.area) partOf.set(q, r);
  });
  // A room in a gross-area space is a room, not a part of it; a part of a
  // part belongs to the room that is laid out.
  for (const [q, r] of partOf) if (gross.has(r)) partOf.delete(q);
  for (const [q, r] of partOf) {
    let host = r;
    for (let guard = 0; partOf.has(host) && guard < 8; guard += 1) host = partOf.get(host);
    partOf.set(q, host);
  }
  return { gross, partOf };
}

/** The rooms that lie inside room i (see overlaps). */
function containers(rooms, i) {
  const r = rooms[i];
  const T = r.footprint;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let k = 0; k < T.length; k += 2) {
    x0 = Math.min(x0, T[k]);
    x1 = Math.max(x1, T[k]);
    y0 = Math.min(y0, T[k + 1]);
    y1 = Math.max(y1, T[k + 1]);
  }
  return rooms.filter((q, j) => {
    if (i === j || q.area >= r.area || Math.abs(q.floor - r.floor) > 0.5) return false;
    const [x, y] = representative(q);
    if (x < x0 || x > x1 || y < y0 || y > y1) return false;
    for (let k = 0; k < T.length; k += 6) if (inTriangle(T, k, x, y)) return true;
    return false;
  });
}

const reps = new WeakMap();
/** A point surely in a room: the centroid of its largest floor triangle. */
function representative(room) {
  if (reps.has(room)) return reps.get(room);
  const T = room.footprint;
  let best = -1, at = 0;
  for (let k = 0; k < T.length; k += 6) {
    const a = triArea(T, k);
    if (a > best) (best = a), (at = k);
  }
  const p = [(T[at] + T[at + 2] + T[at + 4]) / 3, (T[at + 1] + T[at + 3] + T[at + 5]) / 3];
  reps.set(room, p);
  return p;
}

/**
 * One room's heads (positions in the viewer frame) and checks. The grid is
 * tried along the room's long side and along its short side (the Lean rule
 * holds either way), and, where both fail a check, with a bay more along,
 * across, or both; the layout with fewer failing checks, then fewer heads,
 * is kept. In a small ordinary hazard room the area rule can need two heads
 * that only the long side keeps 1.8 m apart; a strip of a room left at a
 * wall jog can need a denser grid.
 */
function layoutRoom(room, H, resp) {
  const { centre, axis: [ux, uy], half: [hu, hv] } = room.rect;
  const frames = [
    { centre, axis: [ux, uy], half: [hu, hv] },
    { centre, axis: [-uy, ux], half: [hv, hu] },
  ];
  const fails = (r) => r.checks.filter((c) => c.status === "fail").length;
  let best = null;
  for (const extra of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    if (best && !fails(best)) break;
    for (const rect of frames) {
      const r = layoutFrame(room, rect, H, resp, extra);
      if (!best || fails(r) < fails(best) || (fails(r) === fails(best) && r.heads.length < best.heads.length)) best = r;
    }
  }
  return best;
}

/** The layout of one room in one orientation of its rectangle (see layoutRooms). */
function layoutFrame(room, rect, H, resp, extra = [0, 0]) {
  const S = H.maxSpacing.value;
  const A = H.areaPerHead.value;
  const reach = S / 2;
  const wallMin = H.wallMin.value;
  const { centre: [ca, cb], axis: [ux, uy], half: [hu, hv] } = rect;
  const vx = -uy, vy = ux;
  const L = 2 * hu, W = 2 * hv;
  const grid = layoutRectangle(L, W, H.id, { extra });
  const { nx, ny, dx, dy } = grid;

  // The footprint in the rectangle's frame: x along the axis from 0 to L, y from 0 to W.
  const P = room.footprint;
  const F = new Float64Array(P.length);
  for (let k = 0; k < P.length; k += 2) {
    const a = P[k] - ca, b = P[k + 1] - cb;
    F[k] = a * ux + b * uy + hu;
    F[k + 1] = a * vx + b * vy + hv;
  }
  const inside = footprintTest(F);
  const walls = wallEdges(F, inside);
  const wallDistance = (x, y) => {
    let d = Infinity;
    for (const w of walls) d = Math.min(d, distanceToSegment(x, y, w));
    return d;
  };
  const clear = (x, y) => inside(x, y) && wallDistance(x, y) >= wallMin - 1e-9;

  // Samples, and the connected parts of the room inside each bay.
  const step = SAMPLE_STEP;
  const ni = Math.max(1, Math.ceil(L / step)), nj = Math.max(1, Math.ceil(W / step));
  const cellOf = new Int32Array(ni * nj).fill(-1);
  const sx = [], sy = [], si = [], sj = [], bayOf = [];
  const bayIndex = (x, y) => Math.min(ny - 1, Math.floor(y / dy)) * nx + Math.min(nx - 1, Math.floor(x / dx));
  for (let j = 0; j < nj; j += 1) {
    for (let i = 0; i < ni; i += 1) {
      const x = (i + 0.5) * step, y = (j + 0.5) * step;
      if (!inside(x, y)) continue;
      cellOf[j * ni + i] = sx.length;
      sx.push(x);
      sy.push(y);
      si.push(i);
      sj.push(j);
      bayOf.push(bayIndex(x, y));
    }
  }
  const count = sx.length;
  const neighbours = (s) => {
    const out = [];
    const i = si[s], j = sj[s];
    if (i > 0) out.push(cellOf[j * ni + i - 1]);
    if (i + 1 < ni) out.push(cellOf[j * ni + i + 1]);
    if (j > 0) out.push(cellOf[(j - 1) * ni + i]);
    if (j + 1 < nj) out.push(cellOf[(j + 1) * ni + i]);
    return out.filter((t) => t >= 0);
  };
  // Connected pieces of a set of samples, 4-neighbour.
  const mark = new Int32Array(count).fill(-1);
  let stamp = 0;
  const components = (list) => {
    stamp += 1;
    const tag = stamp;
    const member = new Set(list);
    const out = [];
    for (const s of list) {
      if (mark[s] === tag) continue;
      const piece = [];
      const stack = [s];
      mark[s] = tag;
      while (stack.length) {
        const t = stack.pop();
        piece.push(t);
        for (const w of neighbours(t)) {
          if (mark[w] !== tag && member.has(w)) {
            mark[w] = tag;
            stack.push(w);
          }
        }
      }
      out.push(piece);
    }
    return out;
  };
  const byBay = new Map();
  for (let s = 0; s < count; s += 1) {
    if (!byBay.has(bayOf[s])) byBay.set(bayOf[s], []);
    byBay.get(bayOf[s]).push(s);
  }
  const bayRect = (b) => {
    const bi = b % nx, bj = Math.floor(b / nx);
    return [bi * dx, (bi + 1) * dx, bj * dy, (bj + 1) * dy];
  };
  // A set of samples' box: their extent plus half a step, within the bay.
  const boxOf = (list, b) => {
    const [bx0, bx1, by0, by1] = bayRect(b);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of list) {
      x0 = Math.min(x0, sx[s]);
      x1 = Math.max(x1, sx[s]);
      y0 = Math.min(y0, sy[s]);
      y1 = Math.max(y1, sy[s]);
    }
    const h = step / 2;
    return [Math.max(bx0, x0 - h), Math.min(bx1, x1 + h), Math.max(by0, y0 - h), Math.min(by1, y1 + h)];
  };
  const reachOf = (x, y, r) => Math.max(x - r[0], r[1] - x, y - r[2], r[3] - y);

  const placed = []; // {x, y, rect: [x0, x1, y0, y1], kind}
  const headOf = new Int32Array(count).fill(-1);
  const assign = (list, h) => {
    for (const s of list) headOf[s] = h;
  };
  const pieces = []; // {bay, samples}
  // Grid heads: the bay centre, where it is clear of the walls, for the piece that holds it.
  for (const [b, list] of byBay) {
    const parts = components(list);
    const bi = b % nx, bj = Math.floor(b / nx);
    const cx = (bi + 0.5) * dx, cy = (bj + 0.5) * dy;
    let served = -1;
    if (clear(cx, cy)) {
      const ci = Math.floor(cx / step), cj = Math.floor(cy / step);
      let s = ci < ni && cj < nj ? cellOf[cj * ni + ci] : -1;
      if (s < 0 || bayOf[s] !== b) {
        let best = Infinity;
        for (const t of list) {
          const d = Math.hypot(sx[t] - cx, sy[t] - cy);
          if (d < best) (best = d), (s = t);
        }
      }
      served = parts.findIndex((p) => p.includes(s));
      placed.push({ x: cx, y: cy, rect: bayRect(b), kind: "grid" });
      assign(parts[served], placed.length - 1);
    }
    parts.forEach((p, k) => {
      if (k !== served) pieces.push({ bay: b, samples: p });
    });
  }
  // A piece with no head gets one of its own, protecting the piece's box, at
  // the best of the box's centre (if it is in the room) and the piece's
  // samples: clear of the walls first, then reaching the whole box within
  // S/2, then at least 1.8 m from the heads already placed, then the
  // shortest reach. A piece no clear point reaches within S/2 is halved, up
  // to 4 times.
  const bestPoint = (list, r, skip = []) => {
    const apart = (x, y) => placed.every((h, k) => !h || skip.includes(k) || Math.hypot(h.x - x, h.y - y) >= H.minBetween.value - 1e-9);
    let best = null;
    const consider = (x, y, isClear) => {
      const d = reachOf(x, y, r);
      const key = [isClear ? 0 : 1, d <= reach + EPS ? 0 : 1, apart(x, y) ? 0 : 1, d];
      const k = best ? key.findIndex((v, i) => v !== best.key[i]) : -1;
      if (!best || (k >= 0 && key[k] < best.key[k])) best = { x, y, key };
    };
    const cx = (r[0] + r[1]) / 2, cy = (r[2] + r[3]) / 2;
    if (inside(cx, cy)) consider(cx, cy, clear(cx, cy));
    for (const s of list) consider(sx[s], sy[s], wallDistance(sx[s], sy[s]) >= wallMin - 1e-9);
    return best;
  };
  // Samples taken in by a head whose rectangle can grow to hold them, still
  // reaching it all within S/2 and protecting at most A: a neighbouring
  // head, or, for a sliver along a wall with no point clear of it, any head
  // of the room.
  const absorb = (list, r) => {
    let best = null;
    const seen = new Set();
    const tryHead = (h) => {
      if (h < 0 || seen.has(h)) return;
      seen.add(h);
      const head = placed[h];
      const u = [Math.min(head.rect[0], r[0]), Math.max(head.rect[1], r[1]), Math.min(head.rect[2], r[2]), Math.max(head.rect[3], r[3])];
      const area = (u[1] - u[0]) * (u[3] - u[2]);
      if (reachOf(head.x, head.y, u) <= reach + EPS && area <= A + EPS && (!best || area < best.area)) best = { h, u, area };
    };
    for (const s of list) for (const t of neighbours(s)) tryHead(headOf[t]);
    if (!best && !list.some((s) => wallDistance(sx[s], sy[s]) >= wallMin - 1e-9)) {
      for (let h = 0; h < placed.length; h += 1) tryHead(h);
    }
    if (!best) return false;
    placed[best.h].rect = best.u;
    assign(list, best.h);
    return true;
  };
  const placePart = (list, b, depth) => {
    const r = boxOf(list, b);
    if (depth > 0 && absorb(list, r)) return;
    const best = bestPoint(list, r);
    if (best.key[1] && depth < 4 && list.length > 1) {
      const wide = r[1] - r[0] >= r[3] - r[2];
      const mid = wide ? (r[0] + r[1]) / 2 : (r[2] + r[3]) / 2;
      const lo = list.filter((s) => (wide ? sx[s] : sy[s]) < mid), hi = list.filter((s) => (wide ? sx[s] : sy[s]) >= mid);
      if (lo.length && hi.length) {
        for (const half of [lo, hi]) for (const p of components(half)) placePart(p, b, depth + 1);
        return;
      }
    }
    placed.push({ x: best.x, y: best.y, rect: r, kind: "part" });
    assign(list, placed.length - 1);
  };
  // The pieces, largest first: taken in by a head, or given heads of their own.
  pieces.sort((p, q) => q.samples.length - p.samples.length);
  for (const piece of pieces) {
    if (!absorb(piece.samples, boxOf(piece.samples, piece.bay))) placePart(piece.samples, piece.bay, 0);
  }
  // Two heads closer than 1.8 m (parts of one alcove split by a bay line)
  // become one, where a point clear of the walls reaches both rectangles
  // within S/2, protects at most A, and keeps 1.8 m from the other heads.
  for (let again = true; again; ) {
    again = false;
    for (let i = 0; i < placed.length && !again; i += 1) {
      for (let j = i + 1; j < placed.length && !again; j += 1) {
        const p = placed[i], q = placed[j];
        if (!p || !q || Math.hypot(p.x - q.x, p.y - q.y) >= H.minBetween.value - 1e-9) continue;
        const u = [Math.min(p.rect[0], q.rect[0]), Math.max(p.rect[1], q.rect[1]), Math.min(p.rect[2], q.rect[2]), Math.max(p.rect[3], q.rect[3])];
        if ((u[1] - u[0]) * (u[3] - u[2]) > A + EPS) continue;
        const list = [];
        for (let s = 0; s < count; s += 1) if (headOf[s] === i || headOf[s] === j) list.push(s);
        const best = bestPoint(list, u, [i, j]);
        if (!best || best.key[0] || best.key[1] || best.key[2]) continue;
        placed[i] = { x: best.x, y: best.y, rect: u, kind: "part" };
        placed[j] = null;
        assign(list, i);
        again = true;
      }
    }
  }
  if (placed.some((h) => !h)) {
    const index = new Int32Array(placed.length).fill(-1);
    const kept = [];
    placed.forEach((h, k) => {
      if (h) (index[k] = kept.length), kept.push(h);
    });
    for (let s = 0; s < count; s += 1) if (headOf[s] >= 0) headOf[s] = index[headOf[s]];
    placed.length = 0;
    placed.push(...kept);
  }

  // The checks, on the heads as placed.
  let overReach = 0, overArea = 0;
  for (const h of placed) {
    if (reachOf(h.x, h.y, h.rect) > reach + EPS) overReach += 1;
    if ((h.rect[1] - h.rect[0]) * (h.rect[3] - h.rect[2]) > A + EPS) overArea += 1;
  }
  const holds = (h, x, y) => x >= h.rect[0] - 1e-9 && x <= h.rect[1] + 1e-9 && y >= h.rect[2] - 1e-9 && y <= h.rect[3] + 1e-9;
  let uncovered = 0;
  for (let s = 0; s < count; s += 1) {
    if (headOf[s] >= 0 && holds(placed[headOf[s]], sx[s], sy[s])) continue;
    if (!placed.some((h) => holds(h, sx[s], sy[s]))) uncovered += 1;
  }
  let nearWall = 0, closest = Infinity, closePairs = 0;
  for (const h of placed) if (wallDistance(h.x, h.y) < wallMin - 1e-9) nearWall += 1;
  for (let i = 0; i < placed.length; i += 1) {
    for (let j = i + 1; j < placed.length; j += 1) {
      const d = Math.hypot(placed[i].x - placed[j].x, placed[i].y - placed[j].y);
      closest = Math.min(closest, d);
      if (d < H.minBetween.value - 1e-9) closePairs += 1;
    }
  }
  const parts = placed.filter((h) => h.kind === "part").length;
  const n = placed.length;
  const plural = (k, word) => `${k} ${word}${k === 1 ? "" : "s"}`;
  const checks = [
    {
      id: "spacing",
      status: !n ? "cannot tell" : overReach ? "fail" : "pass",
      text: !n
        ? "no head placed"
        : `grid of ${nx} × ${ny} bays of ${dx.toFixed(2)} × ${dy.toFixed(2)} m over the room's ${L.toFixed(2)} × ${W.toFixed(2)} m rectangle` +
          `${parts ? `, ${plural(parts, "head")} for parts of bays whose centre is not clear of the walls` : ""}; ` +
          (overReach
            ? `${plural(overReach, "head")} more than ${reach} m from part of what ${overReach === 1 ? "it protects" : "they protect"}`
            : `no point is more than ${reach} m from a head along either axis, so heads are at most ${S} m apart and at most ${reach} m from a wall`),
      clause: withCodes(H.maxSpacing.clause, H.wallMax.clause),
    },
    {
      id: "area",
      status: !n ? "cannot tell" : overArea ? "fail" : "pass",
      text: !n ? "no head placed" : overArea ? `${overArea} of ${n} heads protect more than ${A} m²` : `every head protects at most ${A} m² (${plural(n, "head")})`,
      clause: withCodes(H.areaPerHead.clause),
    },
    {
      id: "coverage",
      status: !count ? "cannot tell" : uncovered ? "fail" : "pass",
      text: !count
        ? `no point of the footprint on the ${step} m sample`
        : uncovered
          ? `${uncovered} of ${count} points sampled every ${step} m lie in no head's protection rectangle (about ${(uncovered * step * step).toFixed(1)} m²)`
          : `all ${count} points sampled every ${step} m lie in a head's protection rectangle`,
      clause: withCodes(H.maxSpacing.clause, H.wallMax.clause),
    },
    {
      id: "wall-min",
      status: !n ? "cannot tell" : nearWall ? "fail" : "pass",
      text: !n ? "no head placed" : nearWall ? `${nearWall} of ${n} heads closer than ${wallMin} m to a wall` : `every head at least ${wallMin} m from the walls`,
      clause: withCodes(H.wallMin.clause),
    },
    n < 2
      ? { id: "between", status: n ? "n/a" : "cannot tell", text: n ? "one head" : "no head placed", clause: withCodes(H.minBetween.clause) }
      : {
          id: "between",
          status: closePairs ? "fail" : "pass",
          text: closePairs
            ? `${plural(closePairs, "pair")} of heads closer than ${H.minBetween.value} m (closest ${closest.toFixed(2)} m)`
            : `heads ${closest.toFixed(2)} m apart at the closest; at least ${H.minBetween.value} m`,
          clause: withCodes(H.minBetween.clause),
        },
    {
      id: "deflector",
      status: DEFLECTOR_BELOW >= H.deflector.min && DEFLECTOR_BELOW <= H.deflector.max ? "pass" : "fail",
      text: `deflector ${DEFLECTOR_BELOW} m below the space's top, taken as a smooth ceiling (a suspended ceiling below it is not in an IfcSpace)`,
      clause: withCodes(H.deflector.clause),
    },
    {
      id: "response",
      status: H.id === "light" && resp !== "quick" ? "fail" : "pass",
      text: H.id === "light" ? `${resp} response; light hazard needs quick response` : `${resp} response`,
      clause: QUICK_REQUIRED,
    },
    OBSTRUCTIONS,
  ];

  // Back to plan and the viewer frame.
  const up = upIndex(room.upAxis);
  const [h0, h1] = planAxes(room.upAxis);
  const toPlan = (x, y) => [ca + (x - hu) * ux + (y - hv) * vx, cb + (x - hu) * uy + (y - hv) * vy];
  const z = room.ceiling - DEFLECTOR_BELOW - LINK_BELOW;
  const heads = placed.map((h) => {
    const [a, b] = toPlan(h.x, h.y);
    const position = [0, 0, 0];
    position[h0] = a;
    position[h1] = b;
    position[up] = z;
    const [x0, x1, y0, y1] = h.rect;
    return {
      position,
      coverage: (x1 - x0) * (y1 - y0),
      footprint: { centre: toPlan((x0 + x1) / 2, (y0 + y1) / 2), axis: [ux, uy], half: [(x1 - x0) / 2, (y1 - y0) / 2] },
    };
  });
  return { heads, checks, grid: { nx, ny, dx, dy, L, W, axis: [ux, uy], parts } };
}

/** The head record (see the header), with the defaults for what is not given. */
function headRecord({ id, source, position, room = null, roomName = null, hazard, response, activationC, kFactor, flowLpm, coverage, footprint, provenance }) {
  const D = SPRINKLER_DEFAULTS;
  const K = kFactor ?? D.kFactor.value;
  return {
    id,
    source,
    position,
    room: room?.id ?? null,
    roomName: room?.name ?? roomName ?? null,
    hazard,
    response,
    rti: D.response[response].rti,
    c: D.c.value,
    activationC: activationC ?? D.activationC.value,
    kFactor: K,
    flowLpm: flowLpm ?? flowFor(hazard, coverage, K),
    coverage,
    footprint,
    provenance,
  };
}

// ------------------------------------------------------------------ obstructions

const FT = 0.3048;

/**
 * NFPA 13's obstruction rules for standard spray heads, as the codes report
 * gives them (wiki/research/sprinkler-codes-2026-11-07.md, section 3.4 and
 * rows 33 to 38). Each clause is cited in the edition its source cites.
 *
 * beam        NFPA 13-2019 Table 10.2.7.1.2 (row 37): for the horizontal
 *             distance A from the sprinkler to the side of an obstruction,
 *             the largest height B of the deflector above the obstruction's
 *             bottom. No interpolation: a row runs from its A up to, not
 *             including, the next row's. A is in feet, converted exactly
 *             (1 ft = 0.3048 m); B is the report's metric figure. Beyond
 *             7 ft 6 in (2.286 m) the table sets no limit. The rows from 4 ft
 *             on rest on one source (partly confirmed).
 * threeTimes  NFPA 13-2022 10.2.7.3.1.3 (row 35): an obstruction entirely
 *             below the deflector, within 18 in (457 mm) of it, at least 3 ×
 *             its largest dimension away, never more than 24 in (610 mm)
 *             required; 10.2.7.3.1.4 (row 36, one source): in light and
 *             ordinary hazard only structural members are subject to it.
 * wide        NFPA 13-2019 9.5.5.3 (row 38): sprinklers beneath fixed
 *             obstructions wider than 4 ft (1.2 m), 1 to 12 in below them;
 *             noncombustible ones whose bottom is 24 in (610 mm) or less above
 *             the floor are exempt (section 3.4).
 * obstructed  NFPA 13-2019 10.2.6.1.2 (row 34): under structural members,
 *             the deflector at or above their bottom at most 22 in (559 mm)
 *             below the deck, with the beam rule; or 1 to 6 in (25 to
 *             152 mm) below the members; or one sprinkler in each bay.
 * kinds       the obstruction kinds; `beamKinds` are those the beam rule
 *             takes. Pipes take the three times rule only (the report's
 *             examples for it: columns, pipes, light fixtures, truss webs),
 *             so a branch line beside its own head does not fail the beam
 *             rule.
 */
export const OBSTRUCTION_RULES = Object.freeze({
  beam: Object.freeze({
    rows: Object.freeze(
      [
        [0, 1, 0, "less than 1 ft (0.3 m)", "0 in (0 mm)", "yes"],
        [1, 1.5, 0.065, "1 ft to less than 1 ft 6 in", "2½ in (65 mm)", "yes"],
        [1.5, 2, 0.09, "1 ft 6 in to less than 2 ft", "3½ in (90 mm)", "yes"],
        [2, 2.5, 0.14, "2 ft to less than 2 ft 6 in", "5½ in (140 mm)", "yes"],
        [2.5, 3, 0.19, "2 ft 6 in to less than 3 ft", "7½ in (190 mm)", "yes"],
        [3, 3.5, 0.24, "3 ft to less than 3 ft 6 in", "9½ in (240 mm)", "yes"],
        [3.5, 4, 0.3, "3 ft 6 in to less than 4 ft", "12 in (300 mm)", "yes"],
        [4, 4.5, 0.35, "4 ft to less than 4 ft 6 in", "14 in (350 mm)", "partly"],
        [4.5, 5, 0.415, "4 ft 6 in to less than 5 ft", "16½ in (415 mm)", "partly"],
        [5, 5.5, 0.45, "5 ft to less than 5 ft 6 in", "18 in (450 mm)", "partly"],
        [5.5, 6, 0.5, "5 ft 6 in to less than 6 ft", "20 in (500 mm)", "partly"],
        [6, 6.5, 0.6, "6 ft to less than 6 ft 6 in", "24 in (600 mm)", "partly"],
        [6.5, 7, 0.75, "6 ft 6 in to less than 7 ft", "30 in (750 mm)", "partly"],
        [7, 7.5, 0.875, "7 ft to less than 7 ft 6 in", "35 in (875 mm)", "partly"],
      ].map(([from, below, max, a, b, confirmed]) => Object.freeze({ from: from * FT, below: below * FT, max, a, b, confirmed })),
    ),
    reach: 7.5 * FT,
    clause:
      "NFPA 13-2019 Table 10.2.7.1.2 (2016: Table 8.6.5.1.2): beam rule, for the horizontal distance A from the sprinkler to the side of an obstruction a maximum height B of the deflector above the obstruction's bottom, no interpolation (rows from 4 ft on one source, partly confirmed)",
  }),
  threeTimes: Object.freeze({
    factor: 3,
    cap: 0.61,
    depth: 0.457,
    clause:
      "NFPA 13-2022 10.2.7.3.1.3: three times rule, for obstructions entirely below the deflector within 18 in (457 mm) of it, at least 3 × the obstruction's largest dimension away, never more than 24 in (610 mm) required",
    structural: "NFPA 13-2022 10.2.7.3.1.4: in light and ordinary hazard only structural members are subject to the three times rule (one source, partly confirmed)",
  }),
  wide: Object.freeze({
    width: 1.2,
    floor: 0.61,
    clause:
      "NFPA 13-2019 9.5.5.3: sprinklers beneath fixed obstructions wider than 4 ft (1.2 m), such as ducts and decks, 1 to 12 in below them; noncombustible ones whose bottom is 24 in (610 mm) or less above the floor are exempt",
  }),
  obstructed: Object.freeze({
    deck: 0.559,
    clause:
      "NFPA 13-2019 10.2.6.1.2: obstructed construction, the deflector at or above the members' bottom at most 22 in (559 mm) below the deck, with the beam rule; or 1 to 6 in (25 to 152 mm) below the members; or one sprinkler in each bay",
  }),
  kinds: Object.freeze(["beam", "duct", "pipe", "cable", "light", "other"]),
  beamKinds: Object.freeze(["beam", "duct", "cable", "light", "other"]),
});

const BEAM = OBSTRUCTION_RULES.beam;
const THREE = OBSTRUCTION_RULES.threeTimes;
const WIDE = OBSTRUCTION_RULES.wide;
const KINDS = new Set(OBSTRUCTION_RULES.kinds);
const BEAM_KINDS = new Set(OBSTRUCTION_RULES.beamKinds);
const KIND_NAME = { beam: "beam", duct: "duct", pipe: "pipe", cable: "cable carrier", light: "light", other: "obstruction" };
/** How far from a head an obstruction can matter (m): the beam rule's last row. */
const REACH = Math.max(BEAM.reach, THREE.cap);
/** A head failing an obstruction rule is moved in steps of this much along each axis of its rectangle (m). */
const MOVE_STEP = 0.05;

/** The beam rule's row for a horizontal distance A (m), or null at 7 ft 6 in (2.286 m) and beyond. */
export function beamRuleRow(A) {
  if (!(A >= 0)) return null;
  for (const row of BEAM.rows) if (A < row.below) return row;
  return null;
}

/**
 * An obstruction record from an element's mesh, or its meshes, in the viewer
 * frame: the convex hull of its vertices projected to plan, and its lowest
 * and highest points along the up axis. Where indices are given, only the
 * vertices the triangles use count. The viewer makes these from IfcBeam
 * (structural) and from an MEP model's IfcDuctSegment, IfcDuctFitting,
 * IfcPipeSegment and IfcCableCarrierSegment (IFC2x3: IfcFlowSegment,
 * IfcFlowFitting). A bent fitting's hull also takes in the inside of its
 * bend.
 *
 * @param {{itemId, positions: Float32Array, indices?: Uint32Array, kind?, structural?}|object[]} mesh
 *        one mesh, or several of one element
 * @param {{kind?: string, structural?: boolean, upAxis?: string}} opts
 *        kind: one of OBSTRUCTION_RULES.kinds ("other" if not), else the
 *        first mesh's; structural: else the first mesh's, else true for a
 *        beam only
 * @returns {{itemId, kind, structural: boolean, footprint: number[][], bottom: number, top: number}|null}
 *          footprint: [[a, b], ...] counter-clockwise in plan; null when the
 *          plan projection has no area
 */
export function obstructionFromMesh(mesh, { kind = null, structural = null, upAxis = "y" } = {}) {
  const meshes = (Array.isArray(mesh) ? mesh : [mesh]).filter((m) => m?.positions);
  if (!meshes.length) return null;
  kind = kind ?? meshes[0].kind ?? "other";
  structural = structural ?? meshes[0].structural ?? null;
  const up = upIndex(upAxis);
  const [h0, h1] = planAxes(upAxis);
  const points = [];
  let bottom = Infinity, top = -Infinity;
  for (const { positions: p, indices: ix } of meshes) {
    const n = Math.floor(p.length / 3);
    let used = null;
    if (ix?.length) {
      used = new Uint8Array(n);
      for (let t = 0; t < ix.length; t += 1) if (ix[t] < n) used[ix[t]] = 1;
    }
    for (let v = 0; v < n; v += 1) {
      if (used && !used[v]) continue;
      const a = p[v * 3 + h0], b = p[v * 3 + h1], y = p[v * 3 + up];
      if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(y)) continue;
      points.push([a, b]);
      if (y < bottom) bottom = y;
      if (y > top) top = y;
    }
  }
  const hull = convexHull(points);
  if (hull.length < 3) return null;
  const k = KINDS.has(kind) ? kind : "other";
  return {
    itemId: meshes[0].itemId ?? null,
    kind: k,
    structural: structural == null ? k === "beam" : Boolean(structural),
    footprint: hull.map(([a, b]) => [a, b]),
    bottom,
    top,
  };
}

/** Obstruction records checked and measured once: hull (counter-clockwise), box, plan width, depth. */
function prepareObstructions(obstructions) {
  const list = [], dropped = [];
  (obstructions || []).forEach((o, i) => {
    const points = Array.isArray(o?.footprint) ? o.footprint.filter((p) => Number.isFinite(p?.[0]) && Number.isFinite(p?.[1])) : [];
    const hull = points.length >= 3 ? convexHull(points) : [];
    if (hull.length < 3 || !Number.isFinite(o.bottom) || !Number.isFinite(o.top) || o.top < o.bottom) {
      dropped.push(o?.itemId ?? i);
      return;
    }
    const box = [Infinity, -Infinity, Infinity, -Infinity];
    for (const [a, b] of hull) {
      box[0] = Math.min(box[0], a);
      box[1] = Math.max(box[1], a);
      box[2] = Math.min(box[2], b);
      box[3] = Math.max(box[3], b);
    }
    const kind = KINDS.has(o.kind) ? o.kind : "other";
    list.push({
      item: o.itemId ?? `obstruction:${i}`,
      kind,
      structural: o.structural == null ? kind === "beam" : Boolean(o.structural),
      hull,
      box,
      width: convexWidth(hull),
      depth: o.top - o.bottom,
      bottom: o.bottom,
      top: o.top,
    });
  });
  return { list, dropped };
}

/**
 * A bucket grid over boxes [x0, x1, y0, y1]: query(box) gives the indices of
 * the boxes that overlap it, each once, in index order. The grid has at most
 * about 256 cells a side; a box over more than 4096 cells is always checked.
 */
function boxIndex(boxes) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const b of boxes) {
    x0 = Math.min(x0, b[0]);
    x1 = Math.max(x1, b[1]);
    y0 = Math.min(y0, b[2]);
    y1 = Math.max(y1, b[3]);
  }
  const cell = Math.max(2, (x1 - x0) / 256, (y1 - y0) / 256);
  const span = (lo, hi, o) => [Math.max(0, Math.floor((lo - o) / cell)), Math.max(0, Math.floor((hi - o) / cell))];
  const cells = new Map(), big = [];
  const key = (i, j) => i * 65536 + j;
  boxes.forEach((b, k) => {
    const [i0, i1] = span(b[0], b[1], x0), [j0, j1] = span(b[2], b[3], y0);
    if ((i1 - i0 + 1) * (j1 - j0 + 1) > 4096) {
      big.push(k);
      return;
    }
    for (let i = i0; i <= i1; i += 1) {
      for (let j = j0; j <= j1; j += 1) {
        const c = key(i, j);
        if (!cells.has(c)) cells.set(c, []);
        cells.get(c).push(k);
      }
    }
  });
  const seen = new Int32Array(boxes.length).fill(-1);
  let stamp = 0;
  const overlaps = (b, q) => b[0] <= q[1] && b[1] >= q[0] && b[2] <= q[3] && b[3] >= q[2];
  return (q) => {
    stamp += 1;
    const out = [];
    const take = (k) => {
      if (seen[k] === stamp) return;
      seen[k] = stamp;
      if (overlaps(boxes[k], q)) out.push(k);
    };
    if (boxes.length && q[1] >= x0 && q[0] <= x1 && q[3] >= y0 && q[2] <= y1) {
      const [i0, i1] = span(Math.max(q[0], x0), Math.min(q[1], x1), x0), [j0, j1] = span(Math.max(q[2], y0), Math.min(q[3], y1), y0);
      for (let i = i0; i <= i1; i += 1) for (let j = j0; j <= j1; j += 1) for (const k of cells.get(key(i, j)) || []) take(k);
    }
    for (const k of big) take(k);
    return out.sort((a, b) => a - b);
  };
}

/** Whether plan point (x, y) lies in a counter-clockwise convex polygon, boundary included. */
function inConvex(P, x, y) {
  for (let i = 0; i < P.length; i += 1) {
    const [ax, ay] = P[i], [bx, by] = P[(i + 1) % P.length];
    if ((bx - ax) * (y - ay) - (by - ay) * (x - ax) < -1e-9) return false;
  }
  return true;
}

/** The plan distance from (x, y) to a convex polygon, 0 inside it. */
function distanceToConvex(P, x, y) {
  if (inConvex(P, x, y)) return 0;
  let d = Infinity;
  for (let i = 0; i < P.length; i += 1) {
    const [ax, ay] = P[i], [bx, by] = P[(i + 1) % P.length];
    d = Math.min(d, distanceToSegment(x, y, [ax, ay, bx, by]));
  }
  return d;
}

/** A convex polygon's width: the least distance between two parallel lines that hold it. */
function convexWidth(P) {
  let w = Infinity;
  for (let i = 0; i < P.length; i += 1) {
    const [ax, ay] = P[i], [bx, by] = P[(i + 1) % P.length];
    const len = Math.hypot(bx - ax, by - ay);
    if (len < 1e-12) continue;
    let far = 0;
    for (const [x, y] of P) far = Math.max(far, Math.abs((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / len);
    w = Math.min(w, far);
  }
  return Number.isFinite(w) ? w : 0;
}

/** Whether two convex polygons overlap by more than a sliver (separating axes). */
function convexOverlap(P, Q) {
  for (const S of [P, Q]) {
    for (let i = 0; i < S.length; i += 1) {
      const [ax, ay] = S[i], [bx, by] = S[(i + 1) % S.length];
      const len = Math.hypot(bx - ax, by - ay);
      if (len < 1e-12) continue;
      const nx = (ay - by) / len, ny = (bx - ax) / len;
      let p0 = Infinity, p1 = -Infinity, q0 = Infinity, q1 = -Infinity;
      for (const [x, y] of P) {
        const t = x * nx + y * ny;
        p0 = Math.min(p0, t);
        p1 = Math.max(p1, t);
      }
      for (const [x, y] of Q) {
        const t = x * nx + y * ny;
        q0 = Math.min(q0, t);
        q1 = Math.max(q1, t);
      }
      if (Math.min(p1, q1) - Math.max(p0, q0) <= 1e-6) return false;
    }
  }
  return true;
}

/**
 * The obstructions over a room: their plan overlaps its floor triangles,
 * and their bottom lies between its floor and its ceiling.
 */
function obstructionsOver(room, list, query) {
  const T = room.footprint;
  const box = [Infinity, -Infinity, Infinity, -Infinity];
  for (let k = 0; k < T.length; k += 2) {
    box[0] = Math.min(box[0], T[k]);
    box[1] = Math.max(box[1], T[k]);
    box[2] = Math.min(box[2], T[k + 1]);
    box[3] = Math.max(box[3], T[k + 1]);
  }
  const out = [];
  for (const i of query(box)) {
    const o = list[i];
    if (o.bottom < room.floor - EPS || o.bottom > room.ceiling + EPS) continue;
    for (let k = 0; k < T.length; k += 6) {
      if (Math.max(T[k], T[k + 2], T[k + 4]) < o.box[0] || Math.min(T[k], T[k + 2], T[k + 4]) > o.box[1]) continue;
      if (Math.max(T[k + 1], T[k + 3], T[k + 5]) < o.box[2] || Math.min(T[k + 1], T[k + 3], T[k + 5]) > o.box[3]) continue;
      if (convexOverlap(o.hull, [[T[k], T[k + 1]], [T[k + 2], T[k + 3]], [T[k + 4], T[k + 5]]])) {
        out.push(o);
        break;
      }
    }
  }
  return out;
}

/**
 * The obstructions that govern one head, at plan point (a, b) with its
 * deflector at height d: for the beam rule, the one with the least margin
 * B allowed less B measured; for the three times rule, the one with the
 * least margin distance less distance required. An obstruction counts when
 * its bottom is below the deflector and its top at most 18 in (457 mm)
 * below it (one lower is clear of the spray's pattern; the wide rule may
 * still need heads beneath it). The beam rule takes it within 7 ft 6 in;
 * the three times rule when it lies entirely below the deflector, and, in
 * light and ordinary hazard, only when it is structural (`exempt` counts
 * the others).
 */
function governing(a, b, d, hazard, obs) {
  const structuralOnly = Boolean(HAZARDS[hazard]);
  let beam = null, three = null, exempt = 0;
  for (const o of obs) {
    if (a < o.box[0] - REACH || a > o.box[1] + REACH || b < o.box[2] - REACH || b > o.box[3] + REACH) continue;
    const below = d - o.top;
    const B = d - o.bottom;
    if (below > THREE.depth + EPS || B <= EPS) continue;
    const A = distanceToConvex(o.hull, a, b);
    if (A >= REACH) continue;
    if (BEAM_KINDS.has(o.kind) && A < BEAM.reach) {
      const row = beamRuleRow(A);
      const margin = row.max - B;
      if (!beam || margin < beam.margin) beam = { o, A, B, row, margin };
    }
    if (below > EPS) {
      if (structuralOnly && !o.structural) {
        exempt += 1;
        continue;
      }
      const dimension = Math.max(o.depth, o.width);
      const required = Math.min(THREE.factor * dimension, THREE.cap);
      const margin = A - required;
      if (!three || margin < three.margin) three = { o, distance: A, required, dimension, below, margin };
    }
  }
  return { beam, three, exempt };
}

const meets = (g) => (!g.beam || g.beam.margin >= -EPS) && (!g.three || g.three.margin >= -EPS);
const nameOf = (o) => `${KIND_NAME[o.kind]} ${o.item}`;

/**
 * The plan distance a head with its deflector at d keeps from obstruction o
 * to meet both rules there, as `governing` judges them (0: no limit). The
 * beam table's B grows row by row, so the beam rule's limit is the A of the
 * first row that allows the head's B, or 7 ft 6 in where none does.
 */
function clearance(o, d, structuralOnly) {
  const below = d - o.top, B = d - o.bottom;
  if (below > THREE.depth + EPS || B <= EPS) return 0;
  let need = 0;
  if (BEAM_KINDS.has(o.kind)) {
    const row = BEAM.rows.find((r) => r.max >= B - EPS);
    need = row ? row.from : BEAM.reach;
  }
  if (below > EPS && !(structuralOnly && !o.structural)) need = Math.max(need, Math.min(THREE.factor * Math.max(o.depth, o.width), THREE.cap) - EPS);
  return need;
}

/** One head's verdicts from `governing`, in the module's check style. */
function verdicts(g, hazard) {
  const structuralOnly = Boolean(HAZARDS[hazard]);
  const beamClause = withCodes(BEAM.clause);
  const threeClause = withCodes(THREE.clause, ...(structuralOnly ? [THREE.structural] : []));
  const beam = g.beam
    ? {
        status: g.beam.margin >= -EPS ? "pass" : "fail",
        text: `${nameOf(g.beam.o)}: ${g.beam.A.toFixed(2)} m to its side (${g.beam.row.a}), the deflector ${g.beam.B.toFixed(3)} m above its bottom; at most ${g.beam.row.b}`,
        clause: beamClause,
        A: g.beam.A,
        B: g.beam.B,
        maxB: g.beam.row.max,
        item: g.beam.o.item,
      }
    : {
        status: "pass",
        text: `no beam, duct, cable carrier, light or other obstruction within ${BEAM.reach.toFixed(3)} m (7 ft 6 in) with its bottom below the deflector`,
        clause: beamClause,
        A: null,
        B: null,
        maxB: null,
        item: null,
      };
  const t = g.three;
  const exempt = g.exempt ? `; ${g.exempt} non-structural item${g.exempt === 1 ? "" : "s"} below the deflector not subject to it` : "";
  const threeTimes = t
    ? {
        status: t.margin >= -EPS ? "pass" : "fail",
        text: `${nameOf(t.o)}, ${t.dimension.toFixed(3)} m at its largest, its top ${t.below.toFixed(3)} m below the deflector: ${t.distance.toFixed(2)} m away; at least ${t.required.toFixed(3)} m${exempt}`,
        clause: threeClause,
        distance: t.distance,
        required: t.required,
        dimension: t.dimension,
        item: t.o.item,
        exempt: g.exempt,
      }
    : {
        status: "pass",
        text: `no ${structuralOnly ? "structural member" : "obstruction"} entirely below the deflector within 18 in (457 mm) of it, within ${REACH.toFixed(3)} m${exempt}`,
        clause: threeClause,
        distance: null,
        required: null,
        dimension: null,
        item: null,
        exempt: g.exempt,
      };
  return { status: worst([beam, threeTimes]), beam, threeTimes };
}

/**
 * The wide rule over one room, or over no room: each obstruction wider than
 * 1.2 m (its plan width, across its run) whose bottom is below the deflector
 * of a head of the room needs a head beneath it, one whose plan point lies
 * in its footprint with the deflector at or below its bottom. Without a
 * room the floor beneath is not known: a wide obstruction with a head within
 * 12 in (305 mm) below it passes, any other cannot be told.
 *
 * @param {object[]} obs  prepared obstructions
 * @param {{id, a, b, d}[]} hs  heads: plan point and deflector height
 * @param {object|null} room
 */
function wideVerdicts(obs, hs, room) {
  const out = [];
  const clause = withCodes(WIDE.clause);
  for (const o of obs) {
    if (o.width <= WIDE.width + EPS) continue;
    if (room && !hs.some((h) => h.d > o.bottom + EPS)) continue;
    const beneath = hs.filter((h) => h.d <= o.bottom + EPS && (room || h.d >= o.bottom - DEFLECTOR.max - EPS) && inConvex(o.hull, h.a, h.b));
    const what = `${nameOf(o)}, ${o.width.toFixed(2)} m wide`;
    const base = { item: o.item, room: room?.id ?? null, width: o.width, bottom: o.bottom, beneath: beneath.map((h) => h.id), clause };
    if (beneath.length) out.push({ ...base, status: "pass", text: `${what}: ${beneath.length} head${beneath.length === 1 ? "" : "s"} beneath it` });
    else if (!room) out.push({ ...base, status: "cannot tell", text: `${what}: no head within 12 in (305 mm) below it, and no room to tell whether it is over a covered floor` });
    else if (o.bottom - room.floor <= WIDE.floor + EPS) {
      out.push({ ...base, status: "cannot tell", text: `${what}, its bottom ${(o.bottom - room.floor).toFixed(2)} m above the floor: exempt if noncombustible` });
    } else out.push({ ...base, status: "fail", text: `${what}: no head beneath it` });
  }
  return out;
}

/**
 * NFPA 13's obstruction rules for heads (see OBSTRUCTION_RULES): per head
 * the beam rule and the three times rule against the obstructions near it,
 * and the wide rule per room.
 *
 * The obstruction record, in the viewer frame (obstructionFromMesh makes it
 * from a mesh):
 *   { itemId,
 *     kind: "beam" | "duct" | "pipe" | "cable" | "light" | "other",
 *     structural: boolean   IfcBeam true, MEP items false (by default true
 *                           for a beam only),
 *     footprint: [[a, b], ...]  its plan polygon, convex, in the heads' plan
 *                           frame (x and z when y is up),
 *     bottom, top           its lowest and highest points along the up axis }
 * A head's deflector is linkBelowDeflector (0.03 m) above its position, the
 * link point (SPRINKLER_DEFAULTS); pass 0 for records whose position is the
 * deflector. A is the plan distance from the head to the obstruction's
 * footprint (0 under it), B the deflector's height above its bottom; the
 * three times rule's largest dimension is the larger of its depth and its
 * plan width (the narrowest extent of its footprint, across its run). A head
 * whose hazard is none of HAZARDS takes every obstruction under the three
 * times rule, not only structural members.
 *
 * Given rooms (roomsFromSpaces), a head whose `room` is among them is
 * checked against the obstructions over that room only (plan overlap with
 * its floor, bottom between its floor and ceiling), not those behind its
 * walls, and the wide rule is checked per room on that room's heads;
 * without rooms, each head against every obstruction within 7 ft 6 in, and
 * the wide rule cannot tell what lies over a covered floor.
 *
 * @param {object[]} heads  head records (id, position, room, hazard)
 * @param {object[]|null} obstructions  obstruction records; none given: every verdict "cannot tell"
 * @param {{rooms?: object[], upAxis?: string, linkBelowDeflector?: number}} opts
 * @returns {{heads: object[], wide: object[], status: string, notes: string[]}}
 *          heads: {id, room, status, beam: {status, text, clause, A, B, maxB, item},
 *          threeTimes: {status, text, clause, distance, required, dimension, item, exempt}};
 *          wide: {item, room, width, bottom, beneath (head ids), status, text, clause};
 *          statuses "pass" | "fail" | "cannot tell", as the room checks
 */
export function checkObstructions(heads, obstructions, { rooms = null, upAxis = "y", linkBelowDeflector = LINK_BELOW } = {}) {
  const up = upIndex(upAxis);
  const [h0, h1] = planAxes(upAxis);
  const given = Array.isArray(obstructions) && obstructions.length > 0;
  const { list, dropped } = prepareObstructions(given ? obstructions : []);
  const query = boxIndex(list.map((o) => o.box));
  const scopes = new Map();
  for (const r of rooms || []) if (r?.footprint && r.rect && !scopes.has(r.id)) scopes.set(r.id, { room: r, obs: null, heads: [] });
  const over = (s) => s.obs ?? (s.obs = obstructionsOver(s.room, list, query));
  const loose = [];
  const out = [];
  for (const head of heads || []) {
    const p = head?.position;
    if (!given || !p || !Number.isFinite(p[h0]) || !Number.isFinite(p[h1]) || !Number.isFinite(p[up])) {
      const text = given ? "the head has no position" : "no structural or MEP model: obstructions were not checked";
      const none = { status: "cannot tell", text, clause: OBSTRUCTIONS.clause };
      out.push({ id: head?.id ?? null, room: head?.room ?? null, status: "cannot tell", beam: { ...none, A: null, B: null, maxB: null, item: null }, threeTimes: { ...none, distance: null, required: null, dimension: null, item: null, exempt: 0 } });
      continue;
    }
    const a = p[h0], b = p[h1], d = p[up] + linkBelowDeflector;
    const scope = scopes.get(head.room);
    const obs = scope ? over(scope) : query([a - REACH, a + REACH, b - REACH, b + REACH]).map((k) => list[k]);
    out.push({ id: head.id, room: head.room ?? null, ...verdicts(governing(a, b, d, head.hazard, obs), head.hazard) });
    (scope ? scope.heads : loose).push({ id: head.id, a, b, d });
  }
  const wide = [];
  if (given) {
    if (scopes.size) for (const s of scopes.values()) if (s.heads.length) wide.push(...wideVerdicts(over(s), s.heads, s.room));
    if (!scopes.size) wide.push(...wideVerdicts(list, loose, null));
  }
  const notes = [];
  if (given) {
    notes.push(`${list.length} obstruction${list.length === 1 ? "" : "s"} (${list.filter((o) => o.structural).length} structural)${dropped.length ? `; ${dropped.length} left out with no plan area or heights` : ""}`);
    if (scopes.size && loose.length) notes.push(`${loose.length} head${loose.length === 1 ? "" : "s"} in no given room: the wide rule was not checked for ${loose.length === 1 ? "it" : "them"}`);
  }
  return { heads: out, wide, status: worst([...out, ...wide]), notes };
}

/**
 * A room's heads cleared of its obstructions (see layoutRooms). A head that
 * fails the beam rule or the three times rule moves within its protection
 * rectangle, in steps of 0.05 m along both of its axes, to the nearest point
 * that meets both while it still reaches the whole rectangle within S/2,
 * stays in the room at least 0.102 m from the walls, and keeps 1.8 m from
 * the other heads of its level; the rectangles do not change, so spacing,
 * wall distances, area and coverage hold as before. A wide obstruction with
 * no head beneath gets the rectangle rule's heads under the part of it over
 * the room, their deflectors 0.1 m below its bottom; a head beneath is
 * shielded by it from the ceiling's heads, so the 1.8 m rule is kept within
 * each level only.
 *
 * @returns {{heads: object[], check: object, deflector: object|null, between: object|null,
 *           moved: object[], added: object[], unresolved: string[]}}
 *          heads: {position, coverage, footprint, obstruction: null | "moved" | "beneath"};
 *          moved: {head, items, rules, by (m), from, to (plan points)};
 *          added: {head, item, position}; deflector and between replace the
 *          room's checks of those ids where not null
 */
function clearObstructions(room, H, placed, obs) {
  const up = upIndex(room.upAxis);
  const [h0, h1] = planAxes(room.upAxis);
  const reach = H.maxSpacing.value / 2;
  const wallMin = H.wallMin.value;
  const ceilingDeflector = room.ceiling - DEFLECTOR_BELOW;
  const inside = footprintTest(room.footprint);
  const walls = wallEdges(room.footprint, inside);
  const clear = (a, b) => {
    if (!inside(a, b)) return false;
    for (const w of walls) if (distanceToSegment(a, b, w) < wallMin - 1e-9) return false;
    return true;
  };
  const heads = placed.heads.map((h) => ({ ...h, a: h.position[h0], b: h.position[h1], d: ceilingDeflector, level: "ceiling", under: null, obstruction: null }));
  const idOf = (k) => `${room.id}/${k + 1}`;
  const moved = [], added = [], unresolved = [];
  const apart = (k, a, b) => heads.every((g, j) => j === k || g.level !== heads[k].level || Math.hypot(g.a - a, g.b - b) >= H.minBetween.value - 1e-9);

  // Move head k to the nearest point of its rectangle that meets every rule.
  const move = (k) => {
    const h = heads[k];
    const { centre: [c0, c1], axis: [ux, uy], half: [hu, hv] } = h.footprint;
    const vx = -uy, vy = ux;
    const u0 = (h.a - c0) * ux + (h.b - c1) * uy, v0 = (h.a - c0) * vx + (h.b - c1) * vy;
    const steps = (x0, half) => {
      const lo = Math.max(-half, half - reach), hi = Math.min(half, reach - half);
      if (lo > hi + 1e-9 || x0 < lo - 1e-9 || x0 > hi + 1e-9) return [x0];
      const out = [x0];
      for (let i = 1; x0 - i * MOVE_STEP >= lo - 1e-9 || x0 + i * MOVE_STEP <= hi + 1e-9; i += 1) {
        if (x0 - i * MOVE_STEP >= lo - 1e-9) out.push(x0 - i * MOVE_STEP);
        if (x0 + i * MOVE_STEP <= hi + 1e-9) out.push(x0 + i * MOVE_STEP);
      }
      for (const end of [lo, hi]) if (!out.some((x) => Math.abs(x - end) < 1e-9)) out.push(end);
      return out;
    };
    const us = steps(u0, hu), vs = steps(v0, hv);
    const at = (u, v) => [c0 + u * ux + v * vx, c1 + u * uy + v * vy];
    const corners = [at(Math.min(...us), Math.min(...vs)), at(Math.max(...us), Math.min(...vs)), at(Math.max(...us), Math.max(...vs)), at(Math.min(...us), Math.max(...vs))];
    const region = [Math.min(...corners.map((p) => p[0])), Math.max(...corners.map((p) => p[0])), Math.min(...corners.map((p) => p[1])), Math.max(...corners.map((p) => p[1]))];
    // The distance each obstruction asks of this head, and the box beyond
    // which it asks nothing; where one blocks all four corners of the
    // region, it blocks all of it (the points nearer a convex polygon than
    // a distance are a convex set), and no point will do.
    const needs = [];
    for (const o of obs) {
      const need = clearance(o, h.d, Boolean(HAZARDS[H.id]));
      if (!(need > 0)) continue;
      const box = [o.box[0] - need, o.box[1] + need, o.box[2] - need, o.box[3] + need];
      if (box[1] < region[0] || box[0] > region[1] || box[3] < region[2] || box[2] > region[3]) continue;
      if (corners.every(([a, b]) => distanceToConvex(o.hull, a, b) < need)) return null;
      needs.push({ o, need, box });
    }
    const blocked = (a, b) => {
      for (let i = 0; i < needs.length; i += 1) {
        const n = needs[i];
        if (a < n.box[0] || a > n.box[1] || b < n.box[2] || b > n.box[3] || distanceToConvex(n.o.hull, a, b) >= n.need) continue;
        // The last obstruction to block is tried first next time.
        if (i) needs.unshift(...needs.splice(i, 1));
        return true;
      }
      return false;
    };
    const near = needs.map((n) => n.o);
    const tries = [];
    for (const u of us) for (const v of vs) if (u !== u0 || v !== v0) tries.push([Math.hypot(u - u0, v - v0), u, v]);
    tries.sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
    for (const [by, u, v] of tries) {
      const [a, b] = at(u, v);
      if (blocked(a, b) || !clear(a, b) || !apart(k, a, b) || (h.under && !inConvex(h.under.hull, a, b))) continue;
      if (!meets(governing(a, b, h.d, H.id, near))) continue;
      const from = [h.a, h.b];
      h.a = a;
      h.b = b;
      h.position = h.position.slice();
      h.position[h0] = a;
      h.position[h1] = b;
      h.obstruction = h.obstruction || "moved";
      return { by, from, to: [a, b] };
    }
    return null;
  };
  let underMembers = false;
  const fix = (k) => {
    const g = governing(heads[k].a, heads[k].b, heads[k].d, H.id, obs);
    if (meets(g)) return;
    const failing = [g.beam && g.beam.margin < -EPS ? ["beam rule", g.beam.o] : null, g.three && g.three.margin < -EPS ? ["three times rule", g.three.o] : null].filter(Boolean);
    const done = move(k);
    if (done) {
      moved.push({ head: idOf(k), items: failing.map(([, o]) => o.item), rules: failing.map(([rule]) => rule), by: done.by, from: done.from, to: done.to });
      return;
    }
    if (failing.some(([, o]) => o.structural)) underMembers = true;
    unresolved.push(`head ${idOf(k)}: no point of its protection rectangle meets ${failing.map(([rule, o]) => `the ${rule} for ${nameOf(o)}`).join(" and ")}`);
  };
  for (let k = 0; k < heads.length; k += 1) fix(k);

  // Heads beneath wide obstructions (the exempt ones near the floor aside).
  for (const o of obs) {
    if (o.width <= WIDE.width + EPS || o.bottom >= ceilingDeflector - EPS || o.bottom - room.floor <= WIDE.floor + EPS) continue;
    if (heads.some((h) => h.d <= o.bottom + EPS && inConvex(o.hull, h.a, h.b))) continue;
    const under = headsBeneath(o, H, inside, clear);
    if (!under.heads.length) {
      unresolved.push(`${nameOf(o)}: wider than ${WIDE.width} m, but no point of the room's floor sampled every ${SAMPLE_STEP} m lies beneath it`);
      continue;
    }
    const d = o.bottom - DEFLECTOR_BELOW;
    for (const u of under.heads) {
      const position = [0, 0, 0];
      position[h0] = u.a;
      position[h1] = u.b;
      position[up] = d - LINK_BELOW;
      heads.push({ position, coverage: u.coverage, footprint: u.footprint, a: u.a, b: u.b, d, level: `beneath ${o.item}`, under: o, obstruction: "beneath" });
      added.push({ head: idOf(heads.length - 1), item: o.item, position });
    }
    if (under.status === "fail") unresolved.push(`${nameOf(o)}: the heads beneath it fail the rectangle rule (${under.failing.join(", ")})`);
  }
  for (let k = heads.length - added.length; k < heads.length; k += 1) fix(k);

  // The check, on the heads as they now stand.
  const per = heads.map((h, k) => ({ id: idOf(k), ...verdicts(governing(h.a, h.b, h.d, H.id, obs), H.id) }));
  const wide = wideVerdicts(obs, heads.map((h, k) => ({ id: idOf(k), a: h.a, b: h.b, d: h.d })), room);
  const check = obstructionCheck(obs, per, wide, moved, added, unresolved, underMembers);

  // Under structural members hanging below the deflector the construction
  // is obstructed: the deflector rule is the second option of 10.2.6.1.2.
  const members = obs.filter((o) => o.structural && o.bottom < ceilingDeflector - EPS && o.top >= ceilingDeflector - THREE.depth - EPS);
  const deflector = members.length
    ? {
        id: "deflector",
        status: DEFLECTOR_BELOW <= OBSTRUCTION_RULES.obstructed.deck + EPS ? "pass" : "fail",
        text: `deflector ${DEFLECTOR_BELOW} m below the space's top, taken as the deck; ${members.length} structural member${members.length === 1 ? " reaches" : "s reach"} below it: obstructed construction, the deflector above their bottom at most ${OBSTRUCTION_RULES.obstructed.deck} m below the deck, with the beam rule`,
        clause: withCodes(OBSTRUCTION_RULES.obstructed.clause),
      }
    : null;
  // The closest pair of ceiling heads, after the moves.
  let between = null;
  if (moved.length) {
    const ceiling = heads.filter((h) => h.level === "ceiling");
    let closest = Infinity, close = 0;
    for (let i = 0; i < ceiling.length; i += 1) {
      for (let j = i + 1; j < ceiling.length; j += 1) {
        const dd = Math.hypot(ceiling[i].a - ceiling[j].a, ceiling[i].b - ceiling[j].b);
        closest = Math.min(closest, dd);
        if (dd < H.minBetween.value - 1e-9) close += 1;
      }
    }
    if (ceiling.length > 1) {
      between = {
        id: "between",
        status: close ? "fail" : "pass",
        text: close
          ? `${close} pair${close === 1 ? "" : "s"} of heads closer than ${H.minBetween.value} m (closest ${closest.toFixed(2)} m)`
          : `heads ${closest.toFixed(2)} m apart at the closest; at least ${H.minBetween.value} m`,
        clause: withCodes(H.minBetween.clause),
      };
    }
  }
  return {
    heads: heads.map((h) => ({ position: h.position, coverage: h.coverage, footprint: h.footprint, obstruction: h.obstruction })),
    check,
    deflector,
    between,
    moved,
    added,
    unresolved,
  };
}

/**
 * The rectangle rule's heads under the part of a wide obstruction over the
 * room: its minimum-area rectangle sampled every SAMPLE_STEP, the samples
 * under it and in the room bounding the part, and each bay's head at its
 * centre, or, where that is not clear of the walls or not under it, at the
 * clear sample of the bay nearest its centre.
 */
function headsBeneath(o, H, inside, clear) {
  const rect = minAreaRect(o.hull.flat());
  if (!rect) return { heads: [], status: "n/a", failing: [] };
  const { centre: [c0, c1], axis: [ux, uy], half: [hu, hv] } = rect;
  const vx = -uy, vy = ux;
  const ni = Math.max(1, Math.ceil((2 * hu) / SAMPLE_STEP)), nj = Math.max(1, Math.ceil((2 * hv) / SAMPLE_STEP));
  const su = (2 * hu) / ni, sv = (2 * hv) / nj;
  const at = (u, v) => [c0 + u * ux + v * vx, c1 + u * uy + v * vy];
  const samples = [];
  for (let j = 0; j < nj; j += 1) {
    for (let i = 0; i < ni; i += 1) {
      const u = -hu + (i + 0.5) * su, v = -hv + (j + 0.5) * sv;
      const [a, b] = at(u, v);
      if (inConvex(o.hull, a, b) && inside(a, b)) samples.push({ u, v, a, b, clear: clear(a, b) });
    }
  }
  if (!samples.length) return { heads: [], status: "n/a", failing: [] };
  const u0 = Math.max(-hu, Math.min(...samples.map((s) => s.u)) - su / 2), u1 = Math.min(hu, Math.max(...samples.map((s) => s.u)) + su / 2);
  const v0 = Math.max(-hv, Math.min(...samples.map((s) => s.v)) - sv / 2), v1 = Math.min(hv, Math.max(...samples.map((s) => s.v)) + sv / 2);
  const grid = layoutRectangle(u1 - u0, v1 - v0, H.id);
  const { nx, dx, dy } = grid;
  const heads = [];
  grid.heads.forEach(([x, y], k) => {
    const bi = k % nx, bj = Math.floor(k / nx);
    const bay = [u0 + bi * dx, u0 + (bi + 1) * dx, v0 + bj * dy, v0 + (bj + 1) * dy];
    let [a, b] = at(u0 + x, v0 + y);
    if (!(inConvex(o.hull, a, b) && clear(a, b))) {
      const mine = samples.filter((s) => s.u >= bay[0] - 1e-9 && s.u <= bay[1] + 1e-9 && s.v >= bay[2] - 1e-9 && s.v <= bay[3] + 1e-9);
      if (!mine.length) return;
      const pool = mine.some((s) => s.clear) ? mine.filter((s) => s.clear) : mine;
      let best = pool[0], bestD = Infinity;
      for (const s of pool) {
        const dd = Math.hypot(s.u - (u0 + x), s.v - (v0 + y));
        if (dd < bestD - 1e-12) (best = s), (bestD = dd);
      }
      [a, b] = [best.a, best.b];
    }
    heads.push({ a, b, coverage: dx * dy, footprint: { centre: at(u0 + x, v0 + y), axis: [ux, uy], half: [dx / 2, dy / 2] } });
  });
  return { heads, status: grid.status, failing: grid.checks.filter((c) => c.status === "fail").map((c) => c.id) };
}

/** The room's obstruction check from the heads' verdicts and the wide rule's (see layoutRooms). */
function obstructionCheck(obs, per, wide, moved, added, unresolved, underMembers) {
  const plural = (k, word) => `${k} ${word}${k === 1 ? "" : "s"}`;
  const clause = withCodes(BEAM.clause, THREE.clause, THREE.structural, WIDE.clause);
  const summary = (key, label) => {
    const list = per.map((h) => ({ id: h.id, ...h[key] }));
    const failing = list.filter((v) => v.status === "fail");
    const margin = (v) => (key === "beam" ? (v.A == null ? Infinity : v.maxB - v.B) : v.distance == null ? Infinity : v.distance - v.required);
    let tight = null;
    for (const v of failing.length ? failing : list) if (margin(v) < Infinity && (!tight || margin(v) < margin(tight))) tight = v;
    const status = worst(list);
    const text = failing.length
      ? `${failing.length} of ${list.length} heads do not meet the ${label}; worst, head ${tight.id}: ${tight.text}`
      : tight
        ? `every head meets the ${label}; closest, head ${tight.id}: ${tight.text}`
        : `every head meets the ${label}: ${list[0]?.text ?? "no head"}`;
    return { id: key === "beam" ? "beam" : "three-times", status, text, clause: list[0]?.clause ?? clause, ...(tight ? { head: tight.id, item: tight.item } : {}), measured: tight ? (key === "beam" ? { A: tight.A, B: tight.B, maxB: tight.maxB } : { distance: tight.distance, required: tight.required, dimension: tight.dimension }) : null };
  };
  const rules = [summary("beam", "beam rule"), summary("threeTimes", "three times rule")];
  const wideFail = wide.filter((w) => w.status !== "pass");
  rules.push({
    id: "wide",
    status: wide.length ? worst(wide) : "pass",
    text: wide.length
      ? wideFail.length
        ? `${wideFail.length} of ${wide.length} obstructions wider than ${WIDE.width} m without a head beneath: ${wideFail.map((w) => w.text).join("; ")}`
        : `${plural(wide.length, "obstruction")} wider than ${WIDE.width} m, each with heads beneath${added.length ? ` (${plural(added.length, "head")} added)` : ""}`
      : `no obstruction wider than ${WIDE.width} m below the deflectors`,
    clause: withCodes(WIDE.clause, DEFLECTOR.clause),
    obstructions: wide,
  });
  const status = obs.length ? worst(rules) : "pass";
  const structural = obs.filter((o) => o.structural).length;
  const actions = [
    moved.length ? `${plural(moved.length, "head")} moved clear` : null,
    added.length ? `${plural(added.length, "head")} added beneath` : null,
    unresolved.length ? `${unresolved.length} not fixed: ${unresolved.slice(0, 3).join("; ")}${unresolved.length > 3 ? `; and ${unresolved.length - 3} more` : ""}` : null,
    underMembers
      ? "under structural members, heads 1 to 6 in (25 to 152 mm) below them at most 22 in (559 mm) below the deck, or one in each bay between them, are the other options of NFPA 13-2019 10.2.6.1.2"
      : null,
  ].filter(Boolean);
  const text = obs.length
    ? `${plural(obs.length, "obstruction")} over the room (${structural} structural): ` +
      rules.map((r) => (r.status === "pass" ? `${r.id === "wide" ? "wide rule" : r.id === "beam" ? "beam rule" : "three times rule"} met` : r.text)).join("; ") +
      (actions.length ? `; ${actions.join("; ")}` : "")
    : "no beam, duct, pipe, cable carrier, light or other obstruction of the model lies over the room";
  return { id: "obstructions", status, text, clause, rules, moved, added, unresolved, obstructions: obs.map((o) => o.item) };
}

// ------------------------------------------------------------------ fallback: no rooms

/**
 * Heads for a model with no IfcSpace: per storey, a square lattice at the
 * hazard's spacing √A (at most S), turned with the walkable graph's own grid
 * (graph.angle), kept where a node of that storey lies within half a spacing
 * of the lattice point along each axis, so every walkable place is in a
 * head's square. Storeys are codeCheck's (storeysOf); stair nodes belong to
 * none and get no heads. The heads stand at their storey's floor: the
 * ceiling is not known, and comes later from the fire grid (provenance
 * ceiling "unknown"). The wall rules cannot be checked without rooms.
 *
 * @param {object} graph   nodes (viewer frame, lifted by agentHeight), kinds, upAxis, agentHeight, angle
 * @param {number[]} levels  storey floor heights in the viewer frame (lattice.floorLevels)
 * @param {string} [hazard]  one class for the whole model, the user's; light hazard if not given
 * @param {{response?: "quick"|"standard"}} opts
 * @returns {{heads: object[], storeys: object[], notes: string[]}}
 *          storeys: {index, elevation, hazard, places, heads (count), checks, status}
 */
export function layoutStoreys(graph, levels = [], hazard = null, { response = null } = {}) {
  const H = hazardOf(hazard ?? "light");
  const S = H.maxSpacing.value;
  const A = H.areaPerHead.value;
  const s = Math.min(Math.sqrt(A), S);
  const resp = response === "quick" || response === "standard" ? response : SPRINKLER_DEFAULTS.responseFor(H.id);
  const n = (graph?.nodes?.length || 0) / 3;
  const upAxis = graph?.upAxis || "y";
  const up = upIndex(upAxis);
  const [h0, h1] = planAxes(upAxis);
  const lift = graph?.agentHeight || 0;
  const angle = Number.isFinite(graph?.angle) ? graph.angle : 0;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const storeyOf = n ? storeysOf(graph, levels) : new Int32Array(0);
  const floors = levels.length ? levels.slice().sort((a, b) => a - b) : null;
  const per = new Map(); // storey -> {u0, v0, floor, places}
  for (let v = 0; v < n; v += 1) {
    const st = storeyOf[v];
    if (st < 0) continue;
    const a = graph.nodes[v * 3 + h0], b = graph.nodes[v * 3 + h1];
    const u = a * cos + b * sin, w = -a * sin + b * cos;
    const e = per.get(st) || { u0: Infinity, v0: Infinity, floor: Infinity, places: 0, cells: new Map() };
    e.u0 = Math.min(e.u0, u);
    e.v0 = Math.min(e.v0, w);
    e.floor = Math.min(e.floor, graph.nodes[v * 3 + up] - lift);
    e.places += 1;
    per.set(st, e);
  }
  for (let v = 0; v < n; v += 1) {
    const e = per.get(storeyOf[v]);
    if (!e) continue;
    const a = graph.nodes[v * 3 + h0], b = graph.nodes[v * 3 + h1];
    const i = Math.floor((a * cos + b * sin - e.u0) / s), j = Math.floor((-a * sin + b * cos - e.v0) / s);
    e.cells.set(`${i},${j}`, [i, j]);
  }
  const heads = [];
  const storeys = [];
  for (const st of [...per.keys()].sort((x, y) => x - y)) {
    const e = per.get(st);
    const floor = floors ? floors[st] : e.floor;
    let k = 0;
    for (const [i, j] of [...e.cells.values()].sort((p, q) => p[1] - q[1] || p[0] - q[0])) {
      const u = e.u0 + (i + 0.5) * s, w = e.v0 + (j + 0.5) * s;
      const a = u * cos - w * sin, b = u * sin + w * cos;
      const position = [0, 0, 0];
      position[h0] = a;
      position[h1] = b;
      position[up] = floor;
      k += 1;
      heads.push(
        headRecord({
          id: `storey${st + 1}/${k}`,
          source: "placed",
          position,
          hazard: H.id,
          response: resp,
          coverage: s * s,
          footprint: { centre: [a, b], axis: [cos, sin], half: [s / 2, s / 2] },
          provenance: {
            position: "default",
            ceiling: "unknown",
            hazard: hazard == null ? "default" : "user",
            response: resp === response ? "user" : "default",
            rti: "default",
            c: "default",
            activationC: "default",
            kFactor: "default",
            flowLpm: "default",
            coverage: "default",
          },
        }),
      );
    }
    const checks = [
      {
        id: "spacing",
        status: s <= S + EPS ? "pass" : "fail",
        text: `a square lattice of ${s.toFixed(2)} m (√${A} m², at most ${S} m)`,
        clause: withCodes(H.maxSpacing.clause),
      },
      { id: "area", status: s * s <= A + EPS ? "pass" : "fail", text: `${(s * s).toFixed(1)} m² per head; at most ${A} m²`, clause: withCodes(H.areaPerHead.clause) },
      {
        id: "coverage",
        status: "pass",
        text: `every one of the storey's ${e.places} walkable places lies in a head's square (the graph's nodes; stairs left out)`,
        clause: withCodes(H.maxSpacing.clause, H.wallMax.clause),
      },
      { id: "between", status: k < 2 ? "n/a" : s >= H.minBetween.value - EPS ? "pass" : "fail", text: k < 2 ? "one head" : `heads ${s.toFixed(2)} m apart`, clause: withCodes(H.minBetween.clause) },
      {
        id: "walls",
        status: "cannot tell",
        text: `no rooms: the distances to walls (at most ${S / 2} m, at least ${H.wallMin.value} m) were not checked`,
        clause: withCodes(H.wallMax.clause, H.wallMin.clause),
      },
      {
        id: "deflector",
        status: "cannot tell",
        text: "ceiling unknown: the heads stand at the storey's floor until the fire grid gives the ceiling",
        clause: withCodes(H.deflector.clause),
      },
      {
        id: "response",
        status: H.id === "light" && resp !== "quick" ? "fail" : "pass",
        text: H.id === "light" ? `${resp} response; light hazard needs quick response` : `${resp} response`,
        clause: QUICK_REQUIRED,
      },
      OBSTRUCTIONS,
    ];
    storeys.push({ index: st, elevation: floor, hazard: H.id, places: e.places, heads: k, checks, status: worst(checks), placement: placementOf(checks) });
  }
  return {
    heads,
    storeys,
    notes: [`one hazard class for the whole model (${H.label}): no rooms to classify`, ...defaultNotes()],
  };
}

// ------------------------------------------------------------------ heads from the IFC

/**
 * The heads an IFC already has, as head records: ifcSemantics' sprinkler
 * records with their positions (the viewer's positionsOf: the centroid of
 * each head's geometry, viewer frame, taken as the link point). What the file
 * lacks takes the defaults, and provenance says which: no file carries RTI
 * or C; a head with no response class takes quick in light hazard and
 * standard elsewhere; no coverage takes the hazard's largest area per head;
 * no flow takes K √P from the file's pressure, else the default flow.
 *
 * Given rooms (roomsFromSpaces), each head takes the smallest room whose
 * footprint holds it with its height between the room's floor and 0.5 m
 * above its top, and that room's hazard class. The user's `response` applies
 * to the heads whose file says nothing (provenance "user").
 *
 * @param {object[]} records  ifcSemantics(...).sprinklers
 * @param {Map<number, number[]>} positions  itemId -> [x, y, z], viewer frame
 * @param {{hazard?: string, rooms?: object[], hazardFor?: Function, response?: string, upAxis?: string}} opts
 * @returns {object[]} the head records, as an array that also carries
 *          `unplaced` ({id, name, reason} for heads with no position) and
 *          `notes` (what came from the file and what took defaults)
 */
export function sprinklersFromIfc(records, positions, { hazard = "light", rooms = null, hazardFor = null, response: userResponse = null, upAxis = "y" } = {}) {
  const at = (id) => (positions instanceof Map ? positions.get(id) : positions?.[id]) ?? null;
  const up = upIndex(upAxis);
  const [h0, h1] = planAxes(upAxis);
  const usable = (rooms || []).filter((r) => r?.footprint);
  const roomAt = (p) => {
    let best = null;
    for (const r of usable) {
      if (p[up] < r.floor - 0.05 || p[up] > r.ceiling + 0.5) continue;
      if (best && r.area >= best.area) continue;
      for (let k = 0; k < r.footprint.length; k += 6) {
        if (inTriangle(r.footprint, k, p[h0], p[h1])) {
          best = r;
          break;
        }
      }
    }
    return best;
  };
  const heads = [];
  const unplaced = [];
  const fromFile = { activationC: 0, kFactor: 0, response: 0, coverage: 0, flowLpm: 0 };
  for (const r of records || []) {
    const p = at(r.id);
    if (!p || !p.every?.(Number.isFinite)) {
      unplaced.push({ id: r.id, name: r.name ?? null, reason: "no geometry to place it by" });
      continue;
    }
    const room = usable.length ? roomAt(p) : null;
    const chosen = room && hazardFor ? hazardFor(room) : null;
    const user = typeof chosen === "string" && HAZARDS[chosen];
    const H = hazardOf(user ? chosen : room ? hazardOfRoom(room.name).hazard : hazard);
    const fileResponse = r.response === "quick" || r.response === "standard" ? r.response : null;
    const userSet = !fileResponse && (userResponse === "quick" || userResponse === "standard");
    const response = fileResponse ?? (userSet ? userResponse : SPRINKLER_DEFAULTS.responseFor(H.id));
    const coverage = r.coverage > 0 ? r.coverage : H.areaPerHead.value;
    const kFactor = r.kFactor > 0 ? r.kFactor : null;
    const K = kFactor ?? SPRINKLER_DEFAULTS.kFactor.value;
    const flowLpm = r.flowLpm > 0 ? r.flowLpm : r.pressureBar > 0 ? K * Math.sqrt(r.pressureBar) : null;
    const activationC = r.activationC > 0 ? r.activationC : null;
    const side = Math.sqrt(coverage) / 2;
    const of = (v) => (v == null ? "default" : "ifc");
    const provenance = {
      position: "ifc",
      hazard: user ? "user" : "default",
      response: fileResponse ? "ifc" : userSet ? "user" : "default",
      rti: "default",
      c: "default",
      activationC: of(activationC),
      kFactor: of(kFactor),
      flowLpm: of(flowLpm),
      coverage: r.coverage > 0 ? "ifc" : "default",
    };
    for (const key of Object.keys(fromFile)) if (provenance[key] === "ifc") fromFile[key] += 1;
    heads.push(
      headRecord({
        id: r.id,
        source: "ifc",
        position: [p[0], p[1], p[2]],
        room,
        hazard: H.id,
        response,
        activationC,
        kFactor,
        flowLpm,
        coverage,
        footprint: { centre: [p[h0], p[h1]], axis: room?.rect?.axis ?? [1, 0], half: [side, side] },
        provenance,
      }),
    );
  }
  const n = heads.length;
  const D = SPRINKLER_DEFAULTS;
  const notes = [
    `${n} head${n === 1 ? "" : "s"} from the IFC${unplaced.length ? `; ${unplaced.length} without geometry left out` : ""}`,
    `from the file: activation temperature ${fromFile.activationC}, K-factor ${fromFile.kFactor}, response ${fromFile.response}, coverage ${fromFile.coverage}, flow or pressure ${fromFile.flowLpm} of ${n}; the rest take the defaults`,
    `RTI and C are in no IFC: quick ${D.response.quick.rti}, standard ${D.response.standard.rti} (m s)^1/2, C ${D.c.value} (m/s)^1/2`,
    `defaults: activation ${D.activationC.value} °C, K ${D.kFactor.value} L/(min bar^1/2), coverage the hazard's largest area per head, flow ${D.flow}`,
  ];
  return Object.assign(heads, { unplaced, notes });
}

// ------------------------------------------------------------------ where sprinklers are required

const TALL = Object.freeze({
  sbc: 17,
  ibc: 16.764,
  occupants: 30,
  clause:
    "SBC 801-CR-2018 903.2.11.3: a storey with 30 or more occupants 17 m or more above the lowest Civil Defense vehicle access; IBC 2021 903.2.11.3: 55 ft (16.764 m) or more above the lowest fire department vehicle access (open parking structures and Group F-2 excepted)",
});
const GROUP_R = "SBC 801-CR-2018 903.2.8; IBC 2021 903.2.8: sprinklers throughout every building with a Group R fire area";
const STANDARD_R =
  "SBC 801-CR-2018 903.3.1.2: NFPA 13R for Group R up to four storeys and 18 m; IBC 2021 903.3.1.2: up to 4 stories and 60 ft (18.288 m) above grade plane, with the highest story's floor at most 30 ft (9.144 m) above the lowest fire department vehicle access";
const HIGH_RISE = `${EGRESS_B.highRise.clause}; IBC 2021 403.3: sprinklers throughout a high-rise building (SBC 201 not read)`;
const A3 =
  "SBC 801-CR-2018 903.2.1.3: Group A-3 fire areas over 1,115 m², with 300 or more occupants, or on a floor other than the level of exit discharge (exception: detached mosques of Type I or II construction); IBC 2021 903.2.1.3: the same, without that exception";
const GROUP_E =
  "SBC 801-CR-2018 903.2.3: Group E fire areas over 1,115 m²; IBC 2021 903.2.3: over 12,000 sq ft (1,115 m²), with 300 or more occupants, or on a floor other than the level of exit discharge";
const UNIVERSITY = "IBC 2021 304.1: education above the 12th grade is Group B; a lecture hall may be A-3 (303.4)";

const NOT_CHECKED = Object.freeze([
  { text: "storeys and basements without qualifying openings, over 139 m²", clause: "SBC 801-CR-2018 903.2.11.1; IBC 2021 903.2.11.1" },
  {
    text: "openings on one side only, and basements more than 23 m (SBC) or 22.86 m (IBC) from the openings",
    clause: "SBC 801-CR-2018 903.2.11.1.2 and 903.2.11.1.3; IBC 2021 903.2.11.1.3",
  },
  { text: "parking garages (enclosed over 1,115 m², open over 4,459 m², or beneath another use)", clause: "IBC 2021 903.2.10" },
  { text: "higher-education laboratory suites: sprinklers throughout", clause: "IBC 2021 428" },
  { text: "ambulatory care facilities in Group B", clause: "IBC 2021 903.2.2" },
  { text: "atriums and covered malls", clause: "IBC 2021 404.3, 402.5" },
  { text: "kiosks and equipment wider than 1,200 mm need sprinklers beneath", clause: "SBC 801-CR-2018 903.3.3" },
]);

/**
 * Whether the codes require automatic sprinklers, trigger by trigger
 * (wiki/research/fire-suppression-sprinklers-2026-11-07.md, section 2), each
 * the stricter of SBC 801-CR-2018 and IBC 2021:
 *   - Group R anywhere in the building: sprinklers throughout;
 *   - a storey with 30 or more occupants 16.764 m or more above the fire
 *     service access (IBC's 55 ft, stricter than SBC's 17 m), any group;
 *   - a high-rise building (an occupied floor more than 22.86 m up);
 *   - Group A-3 and Group E fire areas over 1,115 m², with 300 or more
 *     occupants, or off the level of exit discharge.
 * Heights are measured from `accessElevation`, by default the lowest storey's
 * floor taken as the lowest fire service vehicle access. Each trigger is
 * "required", "pass" (does not apply) or "cannot tell" (data missing). The
 * triggers not checked are listed with their clauses.
 *
 * @param {object} opts
 *   occupancy           a group or groups: "B", "R" ("R-1" to "R-4"), "A-3", "E", or "university" (Group B)
 *   storeyElevations    storey floor heights (m), the same frame as accessElevation
 *   occupantsPerStorey  occupants per storey, in the same order (null: unknown)
 *   area                the A-3 or E fire area (m²)
 *   occupants           the A-3 or E fire area's occupant load
 *   offDischarge        the A-3 or E fire area is on a floor other than the level of exit discharge
 *   accessElevation     the lowest fire service vehicle access
 * @returns {{status: string, text: string, triggers: object[], standard: object|null, notChecked: object[], notes: string[]}}
 */
export function requirementReport({
  occupancy = [],
  storeyElevations = [],
  occupantsPerStorey = [],
  area = null,
  occupants = null,
  offDischarge = null,
  accessElevation = null,
} = {}) {
  const groups = (Array.isArray(occupancy) ? occupancy : [occupancy]).filter((g) => g != null && String(g).trim()).map((g) => String(g).trim());
  const notes = [];
  const has = (test) => groups.some(test);
  const isR = (g) => /^R(-?[1-4])?$/i.test(g);
  const isA3 = (g) => /^A-?3$/i.test(g);
  const isE = (g) => /^E$/i.test(g);
  if (has((g) => /univ/i.test(g))) notes.push(`${UNIVERSITY}: give a lecture hall as A-3 with its area and occupants to check 903.2.1.3`);
  const known = (g) => isR(g) || isA3(g) || isE(g) || /^B$/i.test(g) || /univ/i.test(g);
  const others = groups.filter((g) => !known(g));
  if (others.length) notes.push(`not covered by this report: ${others.join(", ")}`);
  const noGroup = !groups.length;
  const triggers = [];

  triggers.push({
    id: "group-r",
    status: noGroup ? "cannot tell" : has(isR) ? "required" : "pass",
    text: noGroup ? "no occupancy given" : has(isR) ? "a Group R fire area: sprinklers throughout the building" : "no Group R fire area",
    clause: GROUP_R,
  });

  const elevations = (storeyElevations || []).map(Number).filter(Number.isFinite);
  const base = Number.isFinite(accessElevation) ? accessElevation : elevations.length ? Math.min(...elevations) : 0;
  const baseText = Number.isFinite(accessElevation) ? "above the fire service access" : "above the lowest storey, taken as the fire service access";
  const tall = Math.min(TALL.sbc, TALL.ibc);
  if (!elevations.length) {
    triggers.push({ id: "tall", status: "cannot tell", text: "no storey heights given", clause: TALL.clause });
    triggers.push({ id: "high-rise", status: "cannot tell", text: "no storey heights given", clause: HIGH_RISE });
  } else {
    const rows = (storeyElevations || []).map((e, k) => ({ k, h: Number(e) - base, occ: occupantsPerStorey?.[k] ?? null })).filter((r) => Number.isFinite(r.h));
    const high = rows.filter((r) => r.h >= tall - 1e-9);
    const hits = high.filter((r) => r.occ != null && r.occ >= TALL.occupants);
    const unknown = high.filter((r) => r.occ == null);
    triggers.push({
      id: "tall",
      status: hits.length ? "required" : unknown.length ? "cannot tell" : "pass",
      text: hits.length
        ? `${hits.length} storey${hits.length === 1 ? "" : "s"} with ${TALL.occupants} or more occupants ${tall} m or more ${baseText} (the highest at ${Math.max(...hits.map((r) => r.h)).toFixed(2)} m): sprinklers throughout`
        : unknown.length
          ? `${unknown.length} storey${unknown.length === 1 ? "" : "s"} ${tall} m or more ${baseText} with no occupant count`
          : high.length
            ? `the storeys ${tall} m or more up have fewer than ${TALL.occupants} occupants`
            : `no storey ${tall} m or more ${baseText} (IBC's ${TALL.ibc} m, stricter than SBC's ${TALL.sbc} m)`,
      clause: TALL.clause,
    });
    const limit = EGRESS_B.highRise.value;
    const above = rows.filter((r) => r.h > limit + 1e-9);
    const occupied = above.filter((r) => r.occ != null && r.occ > 0);
    const maybe = above.filter((r) => r.occ == null);
    triggers.push({
      id: "high-rise",
      status: occupied.length ? "required" : maybe.length ? "cannot tell" : "pass",
      text: occupied.length
        ? `an occupied floor ${Math.max(...occupied.map((r) => r.h)).toFixed(2)} m ${baseText}, more than ${limit} m: a high-rise building, sprinklers throughout`
        : maybe.length
          ? `${maybe.length} floor${maybe.length === 1 ? "" : "s"} more than ${limit} m up with no occupant count: occupied or not decides`
          : `no occupied floor more than ${limit} m ${baseText}`,
      clause: HIGH_RISE,
    });
  }

  const fireArea = (present, id, clause, label, extra = []) => {
    if (!present) {
      triggers.push({ id, status: noGroup ? "cannot tell" : "pass", text: noGroup ? "no occupancy given" : `no ${label} fire area`, clause });
      return;
    }
    const reasons = [], missing = [];
    if (area == null) missing.push("its area");
    else if (area > 1115) reasons.push(`${area.toFixed(0)} m², over 1,115 m²`);
    if (occupants == null) missing.push("its occupant load");
    else if (occupants >= 300) reasons.push(`${occupants} occupants, 300 or more`);
    if (offDischarge == null) missing.push("whether it is on the level of exit discharge");
    else if (offDischarge) reasons.push("on a floor other than the level of exit discharge");
    triggers.push({
      id,
      status: reasons.length ? "required" : missing.length ? "cannot tell" : "pass",
      text: reasons.length
        ? `the ${label} fire area: ${reasons.join("; ")}`
        : missing.length
          ? `the ${label} fire area: not given ${missing.join(", ")}`
          : `the ${label} fire area: ${area.toFixed(0)} m², ${occupants} occupants, on the level of exit discharge`,
      clause,
    });
    notes.push(...extra);
  };
  fireArea(has(isA3), "group-a3", A3, "Group A-3", [
    "SBC 801-CR-2018 exempts detached mosques of Type I or II construction from 903.2.1.3; IBC 2021 has no such exception, so the stricter rule applies",
  ]);
  fireArea(has(isE), "group-e", GROUP_E, "Group E");

  const required = triggers.filter((t) => t.status === "required");
  const unsure = triggers.filter((t) => t.status === "cannot tell");
  const status = required.length ? "required" : unsure.length ? "cannot tell" : "pass";
  return {
    status,
    text:
      status === "required"
        ? `sprinklers are required throughout the building (${required.map((t) => t.id).join(", ")})`
        : status === "cannot tell"
          ? `not required by the triggers checked so far; ${unsure.map((t) => t.id).join(", ")} cannot be told`
          : "not required by the triggers checked",
    triggers,
    standard: has(isR)
      ? {
          text: "NFPA 13 throughout; NFPA 13R only for Group R up to 4 storeys and 18 m above grade plane (SBC 18 m, IBC 18.288 m), with the highest storey's floor at most 9.144 m above the fire service access (IBC)",
          clause: STANDARD_R,
        }
      : { text: "NFPA 13 throughout", clause: CODES },
    notChecked: NOT_CHECKED.map((x) => ({ ...x })),
    notes,
  };
}
