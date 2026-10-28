/**
 * Vertical cores the treads do not show: lifts, escalators, and stairs known
 * only by name or by a mass with no steps in it.
 *
 * Many models join their floors without modelled treads. Of the 30
 * multi-storey files in the test corpus, five name their stair cores as rooms
 * ("Treppenhaus Ost", "1CS3 STAIR", "A105 Stair"), six carry lifts as
 * IfcTransportElement or as proxies and spaces named "Aufzug", "Electric
 * Lift" or "ELEVATOR", and S_Office's IfcStairs have no geometry of their own
 * (their geometry sits on aggregated proxies). This module turns those
 * elements into cores: a plan footprint and a height range, from which the
 * lattice (lattice.js) links the floors beside them.
 *
 * A core stair with no treads is walked along an assumed dog-leg walking line
 * inside its footprint, on flights of 175 mm risers and 280 mm goings, so a
 * route through it is as long as the stair it stands for (about 1.9 times the
 * rise, plus the landing) and every router, which measures links by their
 * length, sees it so. A core is only linked between two floors when no
 * modelled stair already stands in it between them.
 */

export const CORE_STAIR = 1;
export const CORE_LIFT = 2;
export const CORE_ESCALATOR = 3;
export const CORE_NAMES = { [CORE_STAIR]: "stair", [CORE_LIFT]: "lift", [CORE_ESCALATOR]: "escalator" };

// Names in the languages of the corpus (English, German, French, Spanish,
// Italian, Dutch, Scandinavian). Lobbies, machine rooms and plant rooms beside
// a core are not the core.
const LIFT = /(aufzug|fahrstuhl|elevator|ascenseur|ascensore|ascensor|\blifts?\b|\bheis\b|\bhiss\b)/i;
const ESCALATOR = /(escalator|rolltreppe|fahrtreppe|moving ?walk|fahrsteig|travelator)/i;
const STAIR = /(stair|treppe|escalier|escalera|\bscala\b|\btrap\b|\btrappe\b|\btrappa\b|stiege)/i;
const NOT_CORE = /(lobby|vorraum|vorplatz|foyer|machine|maschinen|technik|plant|motor|control|steuer|pit\b|grube)/i;

/**
 * What kind of core an element is, or 0: IfcTransportElement by its
 * PredefinedType (IFC4) or OperationType (IFC2x3), anything else by name.
 *
 * @param {{type: string, name?: string, predefined?: string}} item
 */
export function coreKind({ type = "", name = "", predefined = "" }) {
  const t = String(type).toUpperCase();
  const p = String(predefined || "").toUpperCase();
  const n = String(name || "");
  // Every staircase is a candidate; where its treads are modelled the lattice
  // finds them in its footprint and leaves it alone.
  if (t === "IFCSTAIR" || t === "IFCSTAIRFLIGHT") return CORE_STAIR;
  if (t === "IFCTRANSPORTELEMENT" || t === "IFCTRANSPORTELEMENTTYPE") {
    if (/ESCALATOR|MOVINGWALKWAY/.test(p) || ESCALATOR.test(n)) return CORE_ESCALATOR;
    if (/LIFTINGGEAR|CRANEWAY|CONVEYOR/.test(p)) return 0;
    return CORE_LIFT;
  }
  if (!n || NOT_CORE.test(n)) return 0;
  if (ESCALATOR.test(n)) return CORE_ESCALATOR;
  if (LIFT.test(n)) return CORE_LIFT;
  // Stairs by name only for rooms and generic objects: a stair-named railing
  // or member is part of a stair, not one.
  if (STAIR.test(n) && /^IFC(SPACE|BUILDINGELEMENTPROXY)$/.test(t)) return CORE_STAIR;
  return 0;
}

/**
 * Core items from element records {itemId, type, name, predefined, parts}:
 * the records that are cores, with their kind and where they came from.
 * `parts` are the ids aggregated into the element (IfcRelAggregates), whose
 * meshes count as its own.
 */
const SOURCE = {
  IFCSTAIR: "IfcStair",
  IFCSTAIRFLIGHT: "IfcStairFlight",
  IFCTRANSPORTELEMENT: "IfcTransportElement",
  IFCSPACE: "IfcSpace name",
  IFCBUILDINGELEMENTPROXY: "IfcBuildingElementProxy name",
};

export function coreItems(records) {
  const out = [];
  for (const r of records) {
    const kind = coreKind(r);
    if (!kind) continue;
    const t = String(r.type || "").toUpperCase();
    out.push({ itemId: r.itemId, kind, name: String(r.name || "").trim().slice(0, 60), source: SOURCE[t] || t, parts: r.parts || [] });
  }
  return out;
}

/**
 * Cores from their elements' meshes: each element's footprint in the lattice
 * frame (u along the dominant wall direction, v across it) and height range,
 * then elements of one kind stacked over each other (plan overlap at least
 * half the smaller footprint, at most 1.5 m apart in height) merged into one
 * core. A lift drawn as one car per storey, or a stair core drawn as one room
 * per storey, is one core.
 *
 * @param {Array<{itemId: number, kind: number, name?: string, source?: string}>} items
 * @param {Array<{itemId: number, positions: Float32Array}>} meshes
 * @param {{up: number, h0: number, h1: number, cos: number, sin: number}} frame
 */
export function coreStacks(items, meshes, { up, h0, h1, cos, sin }) {
  const byItem = new Map();
  for (const mesh of meshes) {
    let list = byItem.get(mesh.itemId);
    if (!list) byItem.set(mesh.itemId, (list = []));
    list.push(mesh);
  }
  const boxes = [];
  for (const item of items) {
    const list = [item.itemId, ...(item.parts || [])].flatMap((id) => byItem.get(id) || []);
    if (!list.length || !item.kind) continue;
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity, lo = Infinity, hi = -Infinity;
    for (const { positions: p } of list) {
      for (let i = 0; i < p.length; i += 3) {
        const x = p[i + h0], y = p[i + h1], z = p[i + up];
        const u = x * cos + y * sin, v = -x * sin + y * cos;
        if (u < u0) u0 = u;
        if (u > u1) u1 = u;
        if (v < v0) v0 = v;
        if (v > v1) v1 = v;
        if (z < lo) lo = z;
        if (z > hi) hi = z;
      }
    }
    if (!Number.isFinite(u0)) continue;
    boxes.push({ ...item, u0, u1, v0, v1, lo, hi });
  }
  const parent = boxes.map((_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  const area = (b) => Math.max(b.u1 - b.u0, 0) * Math.max(b.v1 - b.v0, 0);
  for (let a = 0; a < boxes.length; a += 1) {
    for (let b = a + 1; b < boxes.length; b += 1) {
      const A = boxes[a], B = boxes[b];
      if (A.kind !== B.kind) continue;
      const du = Math.min(A.u1, B.u1) - Math.max(A.u0, B.u0);
      const dv = Math.min(A.v1, B.v1) - Math.max(A.v0, B.v0);
      if (du <= 0 || dv <= 0) continue;
      if (du * dv < 0.5 * Math.min(area(A), area(B))) continue;
      const gap = Math.max(A.lo, B.lo) - Math.min(A.hi, B.hi);
      if (gap > 1.5) continue;
      parent[find(a)] = find(b);
    }
  }
  const groups = new Map();
  boxes.forEach((box, i) => {
    const r = find(i);
    let core = groups.get(r);
    if (!core) {
      core = { kind: box.kind, names: [], sources: [], itemIds: [], u0: Infinity, u1: -Infinity, v0: Infinity, v1: -Infinity, lo: Infinity, hi: -Infinity };
      groups.set(r, core);
    }
    core.itemIds.push(box.itemId);
    if (box.name && !core.names.includes(box.name)) core.names.push(box.name);
    if (box.source && !core.sources.includes(box.source)) core.sources.push(box.source);
    core.u0 = Math.min(core.u0, box.u0);
    core.u1 = Math.max(core.u1, box.u1);
    core.v0 = Math.min(core.v0, box.v0);
    core.v1 = Math.max(core.v1, box.v1);
    core.lo = Math.min(core.lo, box.lo);
    core.hi = Math.max(core.hi, box.hi);
  });
  return [...groups.values()];
}

/**
 * An assumed walking line up a core stair from height ha to hb, as points
 * [u, v, h] in the lattice frame: flights of `riser`/`going` along the longer
 * side of the footprint, turning at a landing between them (two lanes a
 * quarter of the width in from each side, or one lane on the centre line when
 * the core is narrower than 1.2 m). Each flight is split into pieces no
 * longer than `step`, so fire spread that moves one link per step crosses it
 * at the lattice's pace. The flights' plan length is the stair's going in
 * total, (hb - ha) / riser * going, shared among as many flights as the
 * footprint needs (at least two). `mirrorAlong` and `mirrorAcross` reflect
 * the line in the footprint, so the caller can start and end it on the sides
 * where the floors are.
 */
export function walkingLine(box, ha, hb, { riser = 0.175, going = 0.28, step = 0.5, margin = 0.4, mirrorAlong = false, mirrorAcross = false } = {}) {
  const alongU = box.u1 - box.u0 >= box.v1 - box.v0;
  const [a0, a1] = alongU ? [box.u0, box.u1] : [box.v0, box.v1];
  const [c0, c1] = alongU ? [box.v0, box.v1] : [box.u0, box.u1];
  const rise = hb - ha;
  const total = (rise / riser) * going;
  const room = Math.max(a1 - a0 - 2 * margin, 0.5 * (a1 - a0), 0.3);
  const flights = Math.max(2, Math.ceil(total / room - 1e-9));
  const run = total / flights;
  const width = c1 - c0;
  const lanes = width >= 1.2 ? [c0 + width / 4, c1 - width / 4] : [(c0 + c1) / 2, (c0 + c1) / 2];
  const start = a0 + (a1 - a0 - Math.min(run, a1 - a0)) / 2;
  const out = [];
  const push = (a, c, h) => {
    if (mirrorAlong) a = a0 + a1 - a;
    if (mirrorAcross) c = c0 + c1 - c;
    const last = out[out.length - 1];
    const p = alongU ? [a, c, h] : [c, a, h];
    if (last && Math.hypot(last[0] - p[0], last[1] - p[1], last[2] - p[2]) < 1e-6) return;
    out.push(p);
  };
  for (let k = 0; k < flights; k += 1) {
    const lane = lanes[k % 2];
    const from = k % 2 === 0 ? start : start + run;
    const to = k % 2 === 0 ? start + run : start;
    const hA = ha + (rise * k) / flights;
    const hB = ha + (rise * (k + 1)) / flights;
    push(from, lane, hA);
    const pieces = Math.max(1, Math.ceil(Math.hypot(run, hB - hA) / step));
    for (let i = 1; i <= pieces; i += 1) push(from + ((to - from) * i) / pieces, lane, hA + ((hB - hA) * i) / pieces);
  }
  return out;
}

/** Floor levels among heights: sorted, split where consecutive heights differ by more than `gap`. */
export function levelsOf(heights, gap = 1.0) {
  const sorted = Float64Array.from(heights).sort();
  const levels = [];
  let begin = 0;
  for (let i = 1; i <= sorted.length; i += 1) {
    if (i === sorted.length || sorted[i] - sorted[i - 1] > gap) {
      if (i > begin) levels.push(sorted[begin + ((i - begin) >> 1)]);
      begin = i;
    }
  }
  return levels;
}
