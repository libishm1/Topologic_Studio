/**
 * Which doors a Flamescapes run closes: none, all, the ones the model says
 * close themselves ("as built"), or a share drawn at random with a seed so a
 * run can be repeated and compared (App.jsx combustionInputs).
 *
 * "As built" reads Pset_DoorCommon.SelfClosing, and the self-closing mark of
 * an EN 13501-2 fire resistance class: "-C", with an optional durability
 * class C0 to C5 (EI2 30-C, EI2 30-C5-Sm). A rating without it (EI 30) says
 * nothing about closing, so such a door stays open.
 */

export const DOOR_MODES = [
  { value: "open", label: "Open" },
  { value: "closed", label: "Closed" },
  { value: "asbuilt", label: "As built" },
  { value: "random", label: "Random" },
];

/** Whether a fire rating carries EN 13501-2's self-closing mark. */
export function selfClosingRating(rating) {
  return typeof rating === "string" && /-C[0-5]?(?=$|[-\s,;/)])/.test(rating);
}

/** mulberry32: a small seeded generator, so the same seed closes the same doors. */
export function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The door items to close.
 *
 * @param {number[]} doorIds  the model's door leaves (IFC item ids)
 * @param {string} mode       open | closed | asbuilt | random
 * @param {object} opts       selfClosing: Set of ids; seed; share (0 to 1)
 * @returns {Set<number>}
 */
export function closedDoorIds(doorIds, mode, { selfClosing = new Set(), seed = 1, share = 0.5 } = {}) {
  const ids = [...new Set(doorIds)].sort((a, b) => a - b);
  if (mode === "closed") return new Set(ids);
  if (mode === "asbuilt") return new Set(ids.filter((id) => selfClosing.has(id)));
  if (mode === "random") {
    const next = seeded(seed);
    return new Set(ids.filter(() => next() < share));
  }
  return new Set();
}

/** The door setting as words, for the restart notice. */
export function doorsLabel(mode, { seed = 1, share = 0.5 } = {}) {
  if (mode === "asbuilt") return "as built";
  if (mode === "random") return `random (${Math.round(share * 100)}% closed, seed ${seed})`;
  return mode;
}
