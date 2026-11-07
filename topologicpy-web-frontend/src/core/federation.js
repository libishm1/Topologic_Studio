/**
 * A second IFC model beside the architecture: an MEP model's sprinklers and
 * ducts, or a structural model's beams (wiki/research/sprinkler-ifc-mep-2026-11-07.md
 * section 6). Discipline models are meant to share one frame, and usually
 * do; what goes wrong in practice is a different coordinate base at export,
 * a unit mismatch, TrueNorth written two ways, storeys that drift or are
 * named differently, and default georeferences (section 6.3). This module
 * works out how to place the second model in the first one's frame and
 * checks the pair as section 6.4 suggests:
 *
 *   1. length units, normalised to metres (the importer reads both in metres);
 *   2. the site's world placement, origin and rotation;
 *   3. TrueNorth, as metadata only: geometry follows placements;
 *   4. an IfcMapConversion in each: when both have one, the second model is
 *      aligned through them (the buildingSMART Simple-Scene files only line
 *      up this way); when only one has it, by placements, with a warning;
 *   5. storeys matched by elevation within a tolerance, not by name;
 *   6. how much of the second model's plan lies over the first's.
 *
 * Matrices are 4 x 4, column-major (three.js `elements`), as the viewer's
 * coordination matrices are. The IFC world frame is in metres with z up;
 * the viewer frame turns it to y up (x, z, -y), then applies the model's
 * coordination matrix (core/topologicExport.js frameChange).
 */

/** The identity, column-major. */
export const IDENTITY = Object.freeze([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** IFC (z up) to the viewer's y up: (x, y, z) to (x, z, -y). */
const TURN = [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1];
const UNTURN = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1];

/** a times b, both column-major. */
export function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      let s = 0;
      for (let k = 0; k < 4; k += 1) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

/** The inverse of an affine matrix (a linear block and a translation), column-major. */
export function invertAffine(m) {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) throw new Error("A coordination matrix that cannot be inverted.");
  const inv = [
    (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
  const t = [m[12], m[13], m[14]];
  const out = [inv[0], inv[3], inv[6], 0, inv[1], inv[4], inv[7], 0, inv[2], inv[5], inv[8], 0, 0, 0, 0, 1];
  for (let r = 0; r < 3; r += 1) out[12 + r] = -(inv[r * 3] * t[0] + inv[r * 3 + 1] * t[1] + inv[r * 3 + 2] * t[2]);
  return out;
}

/** A point through a column-major matrix. */
export function apply(m, [x, y, z]) {
  return [0, 1, 2].map((r) => m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]);
}

/**
 * IfcMapConversion as a matrix from the IFC world frame in metres to the
 * projected CRS: E = Eastings + s (a x - b y), N = Northings + s (b x + a y),
 * H = OrthogonalHeight + s z, with (a, b) the unit x axis and s the Scale per
 * file length unit, here per metre (Scale / the file's metres per unit).
 */
export function mapMatrix(map, lengthScale = 1) {
  const n = Math.hypot(map.xAxisAbscissa, map.xAxisOrdinate) || 1;
  const a = map.xAxisAbscissa / n, b = map.xAxisOrdinate / n;
  const s = (map.scale || 1) / (lengthScale || 1);
  return [s * a, s * b, 0, 0, -s * b, s * a, 0, 0, 0, 0, s, 0, map.eastings, map.northings, map.height, 1];
}

/**
 * How the second model goes into the first one's IFC frame: through the two
 * map conversions when both files have one, else as they are (the placements
 * already share a frame when the models were exported alike).
 *
 * @returns {{how: "map conversions" | "placements", relative: number[]}}
 *          relative maps the second model's IFC world (metres, z up) to the first's
 */
export function alignment(main, linked) {
  const a = main?.frame?.map, b = linked?.frame?.map;
  if (a && b) {
    const relative = multiply(invertAffine(mapMatrix(a, main.lengthScale)), mapMatrix(b, linked.lengthScale));
    return { how: "map conversions", relative };
  }
  return { how: "placements", relative: [...IDENTITY] };
}

/**
 * The matrix to give the second model's object in the viewer, so that it
 * sits in the first model's frame: C_main T R T^-1 C_linked^-1, where C are
 * the models' coordination matrices, T the turn to y up and R the IFC-frame
 * alignment. With R the identity this is what That Open's own
 * coordination does for a translation (FragmentsModels autoCoordinate).
 */
export function linkMatrix(coordinationMain, coordinationLinked, relative = IDENTITY) {
  const main = coordinationMain?.length === 16 ? coordinationMain : IDENTITY;
  const linked = coordinationLinked?.length === 16 ? coordinationLinked : IDENTITY;
  return multiply(multiply(multiply(multiply(main, TURN), relative), UNTURN), invertAffine(linked));
}

const degrees = (v) => (Math.atan2(v[1], v[0]) * 180) / Math.PI;
const angleBetween = (u, v) => {
  let d = degrees(u) - degrees(v);
  while (d > 180) d -= 360;
  while (d <= -180) d += 360;
  return d;
};
const fixed = (v, n = 2) => Number(v).toFixed(n);
const UNIT_NAMES = [[1, "metres"], [0.001, "millimetres"], [0.01, "centimetres"], [0.3048, "feet"], [0.0254, "inches"]];
const unitName = (scale) => UNIT_NAMES.find(([s]) => Math.abs(s - scale) < 1e-9)?.[1] ?? `${scale} m units`;

/**
 * Storeys paired by elevation (section 6.3 items 4 and 5): each storey of the
 * second model with the nearest of the first, within `tolerance` metres.
 * A storey with no placement reads its declared Elevation; when a file's
 * declared elevations fit the first model better unscaled (the NIBS MEP
 * exports give metres in a millimetre file), they are read so, and said so.
 */
export function matchStoreys(mainStoreys = [], linkedStoreys = [], { tolerance = 0.3, lengthScale = 1 } = {}) {
  const levels = (list) => {
    const placed = list.map((s) => s.elevation);
    const spread = Math.max(...placed) - Math.min(...placed);
    return list.length > 1 && !(spread > 0.5) && list.some((s) => Number.isFinite(s.declared)) ? list.map((s) => s.declared ?? s.elevation) : placed;
  };
  const main = levels(mainStoreys);
  const pair = (values) =>
    linkedStoreys.map((s, k) => {
      let best = -1, gap = Infinity;
      main.forEach((z, m) => {
        if (Math.abs(values[k] - z) < gap) [best, gap] = [m, Math.abs(values[k] - z)];
      });
      return { linked: s.name, main: best >= 0 ? mainStoreys[best].name : null, elevation: values[k], delta: best >= 0 ? values[k] - main[best] : null, matched: gap <= tolerance };
    });
  const own = levels(linkedStoreys);
  let pairs = pair(own);
  let rescaled = false;
  if (lengthScale !== 1 && linkedStoreys.some((s) => Number.isFinite(s.declared))) {
    const unscaled = linkedStoreys.map((s, k) => (Number.isFinite(s.declared) && own[k] === s.declared ? s.declared / lengthScale : own[k]));
    const other = pair(unscaled);
    // Scored by the first model's storeys reached: millimetres read as metres
    // collapse every storey onto the lowest one, which all "match".
    const reached = (list) => new Set(list.filter((p) => p.matched).map((p) => p.main)).size;
    if (reached(other) > reached(pairs)) {
      pairs = other;
      rescaled = true;
    }
  }
  return { pairs, rescaled };
}

/**
 * How much the two plans (x, z in the viewer) share, as a share of the
 * smaller: a site-wide model holding the building (a service model with the
 * site's terrain, or one drawn larger) overlaps it fully. And the offset
 * between their centres.
 */
export function planOverlap(a, b) {
  if (!a || !b) return null;
  const w = Math.max(0, Math.min(a.max[0], b.max[0]) - Math.max(a.min[0], b.min[0]));
  const d = Math.max(0, Math.min(a.max[2], b.max[2]) - Math.max(a.min[2], b.min[2]));
  const plan = (box) => (box.max[0] - box.min[0]) * (box.max[2] - box.min[2]);
  const area = Math.max(1e-9, Math.min(plan(a), plan(b)));
  const centre = (box) => [0, 1, 2].map((k) => (box.min[k] + box.max[k]) / 2);
  const ca = centre(a), cb = centre(b);
  return { share: Math.min(1, (w * d) / area), offset: [ca[0] - cb[0], a.min[1] - b.min[1], ca[2] - cb[2]] };
}

/**
 * The checks of section 6.4 on a pair, after the second model has been
 * placed (its box is in the first one's viewer frame).
 *
 * @param main    {name, semantics, box}   the architecture
 * @param linked  {name, semantics, box}   the second model
 * @returns {{status: "aligned" | "check" | "offset" | "no geometry", how, checks: Array<{id, status, text}>, storeys, offset}}
 *          status "offset" when the plans share less than half of the smaller
 */
export function federationCheck(main, linked, { how = "placements", storeyTolerance = 0.3, placementTolerance = 0.05, angleTolerance = 0.5 } = {}) {
  const A = main.semantics || {}, B = linked.semantics || {};
  const checks = [];
  checks.push({
    id: "units",
    status: "ok",
    text: `Lengths in ${unitName(A.lengthScale ?? 1)} and ${unitName(B.lengthScale ?? 1)}; both read in metres`,
  });

  const sa = A.frame?.site, sb = B.frame?.site;
  if (how === "map conversions") {
    checks.push({ id: "placement", status: "ok", text: "Placed through the two map conversions, not the site placements" });
  } else if (sa && sb) {
    const shift = sa.origin.map((v, k) => sb.origin[k] - v);
    const turn = angleBetween(sb.xAxis, sa.xAxis);
    const same = Math.hypot(...shift) <= placementTolerance && Math.abs(turn) <= angleTolerance;
    checks.push({
      id: "placement",
      status: same ? "ok" : "warning",
      text: same
        ? "The sites are placed alike"
        : `The second site is placed ${fixed(shift[0])}, ${fixed(shift[1])}, ${fixed(shift[2])} m and ${fixed(turn, 1)} degrees from the first; its geometry is shown where its file puts it`,
    });
  } else {
    checks.push({ id: "placement", status: "cannot tell", text: `${sa ? linked.name : main.name} places no site: compared by the geometry below` });
  }

  const na = A.frame?.trueNorth, nb = B.frame?.trueNorth;
  if (na && nb) {
    const d = angleBetween(nb, na);
    checks.push({
      id: "north",
      status: Math.abs(d) <= angleTolerance ? "ok" : "note",
      text: Math.abs(d) <= angleTolerance ? "TrueNorth agrees" : `TrueNorth differs by ${fixed(d, 1)} degrees: metadata only, the geometry follows the placements`,
    });
  } else {
    checks.push({ id: "north", status: "note", text: na || nb ? "Only one file gives TrueNorth (metadata only)" : "Neither file gives TrueNorth" });
  }

  const ma = A.frame?.map, mb = B.frame?.map;
  checks.push(
    ma && mb
      ? { id: "map", status: "ok", text: `Both have a map conversion (${ma.crs || "no CRS name"}; ${mb.crs || "no CRS name"}): aligned through them` }
      : ma || mb
        ? { id: "map", status: "warning", text: `Only ${ma ? main.name : linked.name} has a map conversion: aligned by placements instead` }
        : { id: "map", status: "note", text: "Neither has a map conversion: aligned by placements" },
  );

  const { pairs, rescaled } = matchStoreys(A.storeys || [], B.storeys || [], { tolerance: storeyTolerance, lengthScale: B.lengthScale ?? 1 });
  if (pairs.length) {
    // A storey below or above all of the first model's (a structural
    // model's foundation level) is expected; one between them is not.
    const levels = (A.storeys || []).map((s) => s.elevation);
    const inside = (p) => levels.length && p.elevation > Math.min(...levels) - storeyTolerance && p.elevation < Math.max(...levels) + storeyTolerance;
    const missed = pairs.filter((p) => !p.matched && inside(p));
    const outside = pairs.filter((p) => !p.matched && !inside(p));
    const drift = Math.max(...pairs.filter((p) => p.matched).map((p) => Math.abs(p.delta)), 0);
    const renamed = pairs.filter((p) => p.matched && p.main !== p.linked).length;
    checks.push({
      id: "storeys",
      status: missed.length ? "warning" : "ok",
      text:
        (missed.length
          ? `${missed.length} of ${pairs.length} storeys match none within ${storeyTolerance} m (${missed.map((p) => p.linked).join(", ")})`
          : `${pairs.length - outside.length} storeys match by elevation, within ${fixed(drift)} m`) +
        (outside.length ? `; ${outside.map((p) => p.linked).join(", ")} lies outside the first model's storeys` : "") +
        (renamed ? `; ${renamed} named differently` : "") +
        (rescaled ? "; its storey elevations are in metres although its unit is not" : ""),
    });
  } else {
    checks.push({ id: "storeys", status: "cannot tell", text: "The second model has no storeys to match" });
  }

  const overlap = planOverlap(main.box, linked.box);
  if (overlap) {
    const ok = overlap.share >= 0.5;
    checks.push({
      id: "overlap",
      status: ok ? "ok" : "warning",
      text: ok
        ? `The plans overlap (${Math.round(overlap.share * 100)}% of the smaller)`
        : `The plans overlap by only ${Math.round(overlap.share * 100)}% of the smaller: the centres are ${fixed(Math.hypot(overlap.offset[0], overlap.offset[2]), 1)} m apart (a different coordinate base at export?)`,
    });
  } else {
    checks.push({
      id: "overlap",
      status: linked.box ? "cannot tell" : "warning",
      text: linked.box ? "No geometry to compare" : "The file has no geometry (a properties-only export): nothing in it can be placed or drawn",
    });
  }

  const status = !linked.box ? "no geometry" : overlap && overlap.share < 0.5 ? "offset" : checks.some((c) => c.status === "warning") ? "check" : "aligned";
  return { status, how, checks, storeys: pairs, offset: overlap && overlap.share < 0.5 ? overlap.offset : null };
}
