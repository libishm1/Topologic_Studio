/**
 * IfcSpaces that are not floors to walk on, for the walkable graph
 * (lattice.js takes room bottoms as floor, for models with rooms but no
 * slabs).
 *
 * A space outside the building that no door opens onto: the sprinkler rule
 * (sprinklers.js roomsFromSpaces `outside`: nothing over it, and either no
 * wall round it rising 1.5 m over its floor, so a parapet does not enclose
 * it, or a name for an outdoor area) and no door whose sill is within
 * DOOR_LEVEL of its floor stands within DOOR_REACH of it in plan. The NIBS
 * Clinic draws five of its six "ROOF" spaces so, on roofs no stair or door
 * reaches: 7,940 of its places reached no exit, 817 without them. A
 * balcony or terrace with a door stays walkable: without its floor its door
 * would open onto nothing and be taken for a fire exit to the open air
 * (ifcSemantics.js exitDoors), as two of Vienna's balcony doors were.
 *
 * And a space named as a void ("open to below"): its bottom is the floor
 * of nothing, in the air over the atrium or stair it opens onto (the
 * Clinic's 3AC1).
 */
import { roomsFromSpaces } from "./sprinklers.js";

/** A name for a void drawn as a space: its bottom is no floor. */
export const VOID_NAME = /\b(open to below|void|luftraum)\b/i;

/** Metres between a door's sill and a space's floor for the door to open onto it. */
export const DOOR_LEVEL = 0.5;

/** Metres in plan a door may stand off a space's box and still open onto it. */
export const DOOR_REACH = 0.6;

const UP = { x: 0, y: 1, z: 2 };

/** The axis-aligned box of meshes grouped by itemId. */
function boxes(meshes) {
  const out = new Map();
  for (const m of meshes || []) {
    const p = m.positions;
    if (!p) continue;
    let b = out.get(m.itemId);
    if (!b) out.set(m.itemId, (b = { lo: [Infinity, Infinity, Infinity], hi: [-Infinity, -Infinity, -Infinity] }));
    for (let i = 0; i + 2 < p.length; i += 3) {
      for (let c = 0; c < 3; c += 1) {
        if (p[i + c] < b.lo[c]) b.lo[c] = p[i + c];
        if (p[i + c] > b.hi[c]) b.hi[c] = p[i + c];
      }
    }
  }
  return out;
}

/**
 * The spaces to leave out of the walkable floors.
 *
 * @param {Array<{itemId, positions, indices}>} spaces  the IfcSpace meshes
 * @param {object} opts  names: Map itemId -> name; overhead: slab and roof
 *                       meshes; ceilings, walls, doors: meshes; upAxis
 * @returns {Array<{id, name: string|null, why: string}>}
 */
export function outdoorSpaces(spaces, { names = null, overhead = [], ceilings = [], walls = [], doors = [], upAxis = "y" } = {}) {
  if (!spaces?.length) return [];
  const up = UP[upAxis] ?? 1;
  const [h0, h1] = [0, 1, 2].filter((c) => c !== up);
  const nameOf = (id) => (names instanceof Map ? names.get(id) : names?.[id]) ?? null;
  const out = [];
  const voids = new Set();
  for (const id of new Set(spaces.map((m) => m.itemId))) {
    const name = nameOf(id);
    if (name && VOID_NAME.test(name)) {
      voids.add(id);
      out.push({ id, name, why: "named as a void: its bottom is no floor" });
    }
  }
  const found = roomsFromSpaces(spaces, { upAxis, names, overhead, ceilings, walls });
  const rooms = (found?.rooms ? found.rooms : found) || [];
  const outside = rooms.filter((r) => r.outside && !voids.has(r.id));
  if (!outside.length) return out;
  const spaceBox = boxes(spaces);
  const doorBoxes = [...boxes(doors).values()];
  for (const room of outside) {
    const s = spaceBox.get(room.id);
    if (!s) continue;
    const reached = doorBoxes.some(
      (d) =>
        Math.abs(d.lo[up] - s.lo[up]) < DOOR_LEVEL &&
        d.hi[h0] >= s.lo[h0] - DOOR_REACH &&
        d.lo[h0] <= s.hi[h0] + DOOR_REACH &&
        d.hi[h1] >= s.lo[h1] - DOOR_REACH &&
        d.lo[h1] <= s.hi[h1] + DOOR_REACH,
    );
    if (!reached) out.push({ id: room.id, name: room.name ?? nameOf(room.id), why: "outside the building with no door onto it" });
  }
  return out;
}
