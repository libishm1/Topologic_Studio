/**
 * Exit doors for the cross-model checks, in order of trust:
 *   1. doors flagged FireExit or IsExternal in Pset_DoorCommon (the door's own
 *      value over its type's), or with an EXTERNAL space boundary;
 *   2. failing that, doors with walkable floor on one side only;
 *   3. failing that, doors with a room (IfcSpace) on exactly one side. Weaker
 *      than rule 2: in a block of flats it picks flat entrances that open onto
 *      an unroomed stair hall.
 *
 * Each rule is exported on its own (flaggedDoors, oneSidedDoors,
 * roomOneSideDoors) so the benchmark (tools/benchmark.mjs) can score rules 2
 * and 3 against rule 1 on the models that have flags.
 */

/**
 * Rule 1: the IFC's own exit signal. The site's reading first (occurrence over
 * type, then EXTERNAL space boundaries; src/core/ifcSemantics.js), then
 * Revit's IsFireExit.
 *
 * @returns {{ids: Set<number>, source: string|null}}  door item ids, and which
 *          flag they came from (null when none).
 */
export function flaggedDoors(geo) {
  const flagged = new Set();
  let flagSource = null;
  for (const d of geo.semantics?.doors || []) {
    if (!d.exit) continue;
    flagged.add(d.id);
    flagSource = flagSource && flagSource !== d.exit.replace(/^Pset_DoorCommon\./, "") ? "mixed flags" : d.exit.replace(/^Pset_DoorCommon\./, "");
  }
  for (const [id, f] of geo.doorFlags || []) {
    if (f.FireExit) {
      flagged.add(id);
      flagSource = "FireExit";
    } else if (f.IsExternal) {
      flagged.add(id);
      flagSource = flagSource || "IsExternal";
    }
  }
  return { ids: flagged, source: flagSource };
}

/** Rule 2: graph doors (lattice.js doorReport entries) with walkable floor on one side only. */
export function oneSidedDoors(built) {
  return built.doorReport.filter((d) => d.node >= 0 && (d.sides[0] === 0 || d.sides[1] === 0));
}

/** Rule 3: graph doors with a room (IfcSpace) on exactly one side, 0.6 m either side of the opening. */
export function roomOneSideDoors(geo, built) {
  if (!geo.spaces?.length) return [];
  const tris = [];
  for (const m of geo.spaces) {
    const P = m.positions, I = m.indices;
    for (let t = 0; t + 2 < I.length; t += 3) {
      const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
      tris.push([P[a], P[a + 2], P[b], P[b + 2], P[c], P[c + 2], Math.min(P[a + 1], P[b + 1], P[c + 1]), Math.max(P[a + 1], P[b + 1], P[c + 1])]);
    }
  }
  const inside = (x, z, sill) =>
    tris.some(([ax, az, bx, bz, cx, cz, lo, hi]) => {
      if (sill < lo - 0.5 || sill > hi + 0.5) return false;
      const d1 = (x - bx) * (az - bz) - (ax - bx) * (z - bz);
      const d2 = (x - cx) * (bz - cz) - (bx - cx) * (z - cz);
      const d3 = (x - ax) * (cz - az) - (cx - ax) * (z - az);
      const neg = d1 < 0 || d2 < 0 || d3 < 0, pos = d1 > 0 || d2 > 0 || d3 > 0;
      return !(neg && pos);
    });
  const doorAt = new Map(built.doorPoints.map((d) => [d[6], d]));
  return built.doorReport.filter((d) => {
    const w = doorAt.get(d.itemId);
    if (d.node < 0 || !w) return false;
    const [dx, dz, sill, ux, uz] = w;
    const nx = -uz, nz = ux;
    return inside(dx + 0.6 * nx, dz + 0.6 * nz, sill) !== inside(dx - 0.6 * nx, dz - 0.6 * nz, sill);
  });
}

/**
 * @param {object} geo    tools/ifc-extract.mjs output (doorFlags, spaces)
 * @param {object} built  buildLatticeGraph output (doorReport, doorPoints)
 * @returns {{exits: number[], source: string|null, flagged: number}}
 */
export function findExits(geo, built) {
  const { ids: flagged, source: flagSource } = flaggedDoors(geo);
  let exits = built.doorReport.filter((d) => d.node >= 0 && flagged.has(d.itemId)).map((d) => d.node);
  if (exits.length) return { exits: [...new Set(exits)], source: /^(FireExit|IsExternal)$/.test(flagSource) ? `Pset_DoorCommon.${flagSource}` : flagSource, flagged: flagged.size };

  exits = oneSidedDoors(built).map((d) => d.node);
  if (exits.length) return { exits: [...new Set(exits)], source: "one-sided door (fallback)", flagged: flagged.size };

  exits = roomOneSideDoors(geo, built).map((d) => d.node);
  if (exits.length) return { exits: [...new Set(exits)], source: "room on one side (IfcSpace)", flagged: flagged.size };
  return { exits: [], source: null, flagged: flagged.size };
}
