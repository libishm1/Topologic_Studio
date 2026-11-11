/**
 * Portable fire extinguishers on the walking graph: the codes' placement
 * rules, a layout where the model has none, and one person's attempt to put
 * the fire out (wiki/research/extinguishers-2026-11-07.md, sections 1, 5.3
 * and 6). Pure functions; the solver coupling stays outside: capFor and
 * extinguisherCap give a suppression record (suppression.js) the per-cell
 * decay that evansFactor already applies, for the run to wire in.
 *
 * Sources. SBC 801-CR-2018 Section 906 and Tables 906.3(1) and 906.3(2) were
 * read in the user's licensed copy and are cited by clause and value only.
 * NFPA 10 was read in NFPA's public Second Draft Report for the A2025 cycle,
 * so its clause numbers are the next edition's and may differ by one
 * revision from the 2022 edition. UL 711 in its 7th edition (2004, revised
 * 2013). Each limit is the stricter of SBC and NFPA, as in the egress check
 * (codeCheck.js; the user's rule of 2026-11-07).
 *
 * Placement (note 6.1). Travel is NFPA 10 E.3.3's actual walking distance,
 * which is what the graph measures: d(v) is the shortest walk on v's storey,
 * over open links (graph.blocked, as the app's routing), to the nearest
 * extinguisher on it, by one multi-source Dijkstra per storey. Adding a unit
 * never lengthens a walk (a least over more sources; Lean travel_mono_sites),
 * and every walk is at least the straight line, each edge being its straight
 * length (graph_ge_euclid), so a pass on the graph is also a pass of a radius
 * check, not the converse. The graph samples the floor every s metres, so a
 * floor point lies up to delta = s / sqrt 2 from a node along a straight
 * walk: a node within D - delta passes for every point near it
 * (sample_margin); one between D - delta and D is near the limit and cannot
 * be told; one beyond D fails.
 *
 * The attempt (note 6.3). One responder, one extinguisher, one attempt: walk
 * to the nearest extinguisher whose agent fireClasses.js lists as suitable,
 * take it (t_take), walk to the attack place, and discharge at
 *   t_s = t_go + T(start, ext) + t_take + T(ext, attack),
 * T the graph's walking times. The fire's heat release Q(t_s) (or, for
 * liquids, its burning area) is set against the unit's capacity at the low
 * and high ends of its parameters, so the outcome is three-valued: success,
 * cannot tell, failure. On failure the fire is unchanged; on success the run
 * may cap the reached cells with tau_x = t_d / ln(1/epsilon) (extinguisherCap).
 * Not modelled: tenability on the way (the run's index), the line of sight
 * to the fire (the solver grid's `clear` test), a carrying speed (none was
 * found; the walking speed is used), the agent's gas-phase effect, powder's
 * loss of visibility, re-ignition, several responders or a second unit.
 */

import { FIRE_CLASSES } from "./fireClasses.js";
import { KIND_DOOR, KIND_FLOOR, KIND_STAIR } from "./lattice.js";
import { storeysOf } from "./codeCheck.js";
import { walkToExit } from "./performanceCheck.js";
import { Heap, csr, distancesFrom, edgeLengths } from "./route.js";

// ------------------------------------------------------------------ the code values

/**
 * Travel distance to the nearest extinguisher, by fire class (note section 1
 * and 6.1 item 2). Class B by hazard: [rating, metres] pairs, the smaller
 * unit within the shorter distance.
 */
export const TRAVEL = Object.freeze({
  A: Object.freeze({
    value: 22.86,
    clause:
      "SBC 801-CR-2018 906.3.1, Table 906.3(1): 23 m for class A; NFPA 10 Table 6.2.1.1: 75 ft (22.86 m), the stricter; NFPA 10 E.3.3: the actual walking distance to the nearest extinguisher",
  }),
  B: Object.freeze({
    light: Object.freeze([[5, 9], [10, 15]]),
    ordinary: Object.freeze([[10, 9], [20, 15]]),
    extra: Object.freeze([[40, 9], [80, 15]]),
    clause:
      "SBC 801-CR-2018 906.3.2, Table 906.3(2): class B spill fires (liquids at most 6 mm deep), light 5-B within 9 m or 10-B within 15 m, ordinary 10-B / 9 m or 20-B / 15 m, extra 40-B / 9 m or 80-B / 15 m; NFPA 10 Table 6.3.1.1: the same ratings within 30 ft (9.14 m) and 50 ft (15.25 m); SBC's distances are the stricter",
  }),
  K: Object.freeze({
    value: 9,
    clause: "SBC 801-CR-2018 904.12.5: class K within 9 m of travel of commercial cooking equipment; NFPA 10 6.6.2: 30 ft (9.1 m); SBC's 9 m is the stricter",
  }),
  D: Object.freeze({ value: 22.86, clause: "NFPA 10 6.5.2: class D within 75 ft (22.86 m) of travel" }),
});

const AREA_PER_UNIT_CLAUSE =
  "SBC 801-CR-2018 Table 906.3(1): 280 / 139 / 93 m² of floor per unit of A (light / ordinary / extra); NFPA 10 Table 6.2.1.1: 3000 / 1500 / 1000 ft² (279 / 139 / 92.9 m²); the stricter";
const MIN_RATING_CLAUSE = "SBC 801-CR-2018 Table 906.3(1); NFPA 10 Table 6.2.1.1: smallest single extinguisher 2-A (light, ordinary) or 4-A (extra)";

/**
 * Class A provision by hazard (NFPA 10 5.4.1.1 to 5.4.1.3 and annex
 * A.5.4.1: light for offices, classrooms and assembly halls; ordinary for
 * dining areas, mercantile, workshops and research operations, which takes in
 * university laboratories; extra for woodworking, vehicle repair, parking
 * garages and cooking areas). Not the sprinklers' hazard classes: garages and
 * kitchens differ (note 1.3).
 */
export const EXTINGUISHER_HAZARDS = Object.freeze({
  light: Object.freeze({
    id: "light",
    label: "Light hazard",
    areaPerUnit: { value: 279, clause: AREA_PER_UNIT_CLAUSE },
    minRating: { value: 2, clause: MIN_RATING_CLAUSE },
  }),
  ordinary: Object.freeze({
    id: "ordinary",
    label: "Ordinary hazard",
    areaPerUnit: { value: 139, clause: AREA_PER_UNIT_CLAUSE },
    minRating: { value: 2, clause: MIN_RATING_CLAUSE },
  }),
  extra: Object.freeze({
    id: "extra",
    label: "Extra hazard",
    areaPerUnit: { value: 92.9, clause: AREA_PER_UNIT_CLAUSE },
    minRating: { value: 4, clause: MIN_RATING_CLAUSE },
  }),
});

/** At most this floor area per extinguisher, whatever its rating (m²). */
export const AREA_PER_EXTINGUISHER = Object.freeze({
  value: 1045,
  clause:
    "SBC 801-CR-2018 Table 906.3(1); NFPA 10 Table 6.2.1.1: at most 1045 m² (11,250 ft²) per extinguisher; NFPA 10 E.3.4: the square inscribed in a 22.9 m circle",
});

const COUNT_CLAUSE =
  "NFPA 10 6.2.1.2.1 to 6.2.1.2.3: units per floor = floor area / area per extinguisher, then more until no point is beyond the travel distance; 6.2.1.5, E.3.2: a smaller floor still needs one of the minimum size";

/**
 * The graph's sampling margin (m): a floor point lies at most half a cell's
 * diagonal, s / sqrt 2, from its node (0.35 m on the 0.5 m lattice; note 6.1
 * item 3, geometry of the grid).
 */
export const samplingMargin = (spacing) => spacing / Math.SQRT2;

// ------------------------------------------------------------------ the units

/**
 * Portable units, with their rating (UL: A and B units as printed; EN: the
 * EN numbers, converted by ulA), discharge time t_d (s), reach (m: the lower
 * end of NFPA 10 Table H.2's range, as note 6.3 step 4 takes it) and agent.
 */
export const UNITS = Object.freeze({
  "water-9.5": Object.freeze({
    id: "water-9.5",
    label: "Water, 9.5 L (2.5 gal), stored pressure",
    agent: "water",
    rating: Object.freeze({ system: "UL", A: 2, B: 0, C: false, K: false }),
    tD: 60,
    reach: 9.1,
    source: "NFPA 10 Table H.2: 2-A, 1 min, 9.1 to 12.2 m (Amerex 240: 55 s); UL 711 Table 1: 9.0 L of water is 2-A without a fire test",
  }),
  "water-6": Object.freeze({
    id: "water-6",
    label: "Water, 6 L (1.6 gal), stored pressure",
    agent: "water",
    rating: Object.freeze({ system: "UL", A: 1, B: 0, C: false, K: false }),
    tD: 40,
    reach: 9.1,
    source: "NFPA 10 Table H.2: 1-A, 40 s, 9.1 to 12.2 m",
  }),
  "foam-9.5": Object.freeze({
    id: "foam-9.5",
    label: "Foam, 9.5 L",
    agent: "foam",
    rating: Object.freeze({ system: "UL", A: 3, B: 20, C: false, K: false }),
    tD: 50,
    reach: 6.1,
    source: "NFPA 10 Table H.2: 3-A:20 to 40-B, 50 s, 6.1 to 7.6 m (the lower B rating taken)",
  }),
  "abc-2.3": Object.freeze({
    id: "abc-2.3",
    label: "ABC dry chemical, 2.3 kg (5 lb), 2-A:10-B:C",
    agent: "abc",
    rating: Object.freeze({ system: "UL", A: 2, B: 10, C: true, K: false }),
    tD: 14,
    reach: 1.5,
    source:
      "Amerex B500: 2-A:10-B:C, 14 s, 3.7 to 5.5 m (B402: 3-A:40-B:C); reach 1.5 m, the lower end of NFPA 10 Table H.2's range for hand dry chemical units, as note 6.3 step 4 takes it for units up to 2.3 kg",
  }),
  "abc-4.5": Object.freeze({
    id: "abc-4.5",
    label: "ABC dry chemical, 4.5 kg (10 lb), 4-A:80-B:C",
    agent: "abc",
    rating: Object.freeze({ system: "UL", A: 4, B: 80, C: true, K: false }),
    tD: 20,
    reach: 4.6,
    source: "Amerex B456: 4-A:80-B:C, 20 s, 4.6 to 6.4 m",
  }),
  "abc-6-en": Object.freeze({
    id: "abc-6-en",
    label: "ABC powder, 6 kg (EN 3), 34A:183B:C",
    agent: "abc",
    rating: Object.freeze({ system: "EN", A: 34, B: 183, C: true, K: false }),
    tD: 16,
    tDRange: Object.freeze([16, 23]),
    reach: 4.6,
    source:
      "GLORIA 2026 catalogue: 34A to 55A, 183B to 233B (the lower ends taken), about 16 to 23 s, about 6 m (columns interleaved: medium confidence); reach 4.6 m as note 6.3 step 4 for ABC units above 2.3 kg",
  }),
  "co2-2": Object.freeze({
    id: "co2-2",
    label: "Carbon dioxide, 2 kg (EN 3), 34B",
    agent: "co2",
    rating: Object.freeze({ system: "EN", A: 0, B: 34, C: true, K: false }),
    tD: 8.5,
    reach: 0.9,
    source: "GLORIA 2026 catalogue: 34B, 8.5 s, about 3 m; reach 0.9 m, the lower end of NFPA 10 Table H.2's CO2 range (0.9 to 2.4 m)",
  }),
  "wet-chemical-6": Object.freeze({
    id: "wet-chemical-6",
    label: "Wet chemical (class K), 6 L",
    agent: "wet-chemical",
    rating: Object.freeze({ system: "UL", A: 0, B: 0, C: false, K: true }),
    tD: 40,
    reach: 2.4,
    source:
      "SBC 801-CR-2018 904.12.5.2 and NFPA 10 6.7: a 6 L class K unit; its discharge time and reach are not in the note (NFPA 10 Table H.2's wet chemical row was not read): 40 s and 2.4 m are stand-ins",
    provisional: Object.freeze(["tD", "reach"]),
  }),
});

/** The unit taken where none is given: the smallest 2-A unit, and suited to classes A, B and C. */
export const DEFAULT_UNIT = "abc-2.3";

/**
 * The capacity model's parameters (note 6.3 step 5 and its table), with
 * their ranges; the outcome uses the low and high ends.
 */
export const CAPACITY = Object.freeze({
  qA: Object.freeze({
    value: 300,
    low: 200,
    high: 450,
    unit: "kW per UL A unit at attack",
    source: "derived: UL 711 Table 2's crib wood (about 0.052 m³ per A unit), NIST FCD Test13_woodCrib_r1 (396 kW peak) and crib correlations; no rated test fire's heat release was measured (note 2.3)",
  }),
  eta: Object.freeze({ value: 0.4, low: 0.2, high: 1.0, source: "assumption: an ordinary operator's share of the rated fire, by analogy with UL 711 5.4's 40 % for class B (note 6.3 step 5)" }),
  areaPerB: Object.freeze({
    ul: 0.0929,
    ulExpert: 0.0929 / 0.4,
    source: "UL 711 5.4: the B number is 40 % of the pan area (ft²) an expert consistently extinguishes, so 0.0929 m² per unit, and 0.232 m² per unit for the expert's pan",
  }),
  areaPerBEn: Object.freeze({
    ordinary: (0.4 * Math.PI) / 100,
    tray: Math.PI / 100,
    source: "EN 3-7 trays of pi B / 100 m² (IS 15683:2006 Table 9, an EN-type standard), taken at UL's 40 % for an ordinary operator (note 6.3 step 5)",
  }),
  enToUlA: Object.freeze({ value: 2 / 13, source: "13A (EN) is about 2-A (UL) by the 9 L water bridge (UL 711 Table 1; BS 5306-8:2023, secondary); for powder the bridge is off by about 2 (note 2.4)" }),
  epsilon: Object.freeze({ value: 0.01, low: 0.001, high: 0.1, source: "constructed: the rating tests' criterion that the flames are out when the unit is empty (note 3.4, 6.3 step 7)" }),
});

/** Time from "go" to discharge with the unit at hand (s). */
export const T_TAKE = Object.freeze({
  value: 13.4,
  instructed: 7.2,
  source: "Poole et al. 2012 (WPI and EKU): 276 untrained adults, mean 13.4 s on the first try, 7.2 s after brief instruction; one study, a trainer, the unit beside the person",
});

/** The least distance from the fire at which discharge starts (m). */
export const STANDOFF = Object.freeze({ value: 1.8, source: "UL 711 7.2.6: the class A attack starts from 1.8 m (6 ft) or more" });

/** Nodes within this height (m) of the fire's node count as its floor for the attack place. */
const SAME_FLOOR = 1.5;
/** Layout candidates: within this walk of a door (m; NFPA 10 F.6.2.2 near exits, note 6.2). */
const DOOR_REACH = 2;
/** ... or of an egress route (m; SBC 801-CR-2018 906.5 on the routes people normally walk, note 6.1 item 5). */
const ROUTE_REACH = 1;
/** Layout candidates are thinned to one per this distance (m). */
const THIN = 1;

// ------------------------------------------------------------------ helpers

const AXIS = { x: 0, y: 1, z: 2 };
const upIndex = (upAxis) => AXIS[upAxis] ?? 1;
const planAxes = (upAxis) => [0, 1, 2].filter((i) => i !== upIndex(upAxis));

function hazardOf(hazard) {
  const H = EXTINGUISHER_HAZARDS[hazard];
  if (!H) throw new RangeError(`Unknown extinguisher hazard "${hazard}"; use one of ${Object.keys(EXTINGUISHER_HAZARDS).join(", ")}.`);
  return H;
}

/** The hazard of storey s: one for the model, or one per storey (light where none is given). */
const hazardAt = (hazard, s) => hazardOf(Array.isArray(hazard) ? (hazard[s] ?? "light") : (hazard ?? "light"));

/** A unit record from its key, or the record itself. */
export function unitOf(unit = DEFAULT_UNIT) {
  const u = typeof unit === "string" ? UNITS[unit] : unit;
  if (!u || !(u.tD >= 0) || !u.rating) throw new RangeError(`Unknown extinguisher unit "${unit}"; use one of ${Object.keys(UNITS).join(", ")}, or a record of the same shape.`);
  return u;
}

/** A unit's class A rating in UL A units (EN numbers times 2/13). */
export const ulA = (unit) => {
  const { rating } = unitOf(unit);
  return (rating.A || 0) * (rating.system === "EN" ? CAPACITY.enToUlA.value : 1);
};

const RANK = { "n/a": 0, pass: 1, "cannot tell": 2, fail: 3 };
/** The worst status of a list: fail, then cannot tell, then pass. */
function worst(statuses) {
  let w = "n/a";
  for (const s of statuses) if (RANK[s] > RANK[w]) w = s;
  return w;
}

/**
 * Shortest walks from several sources at once (all at 0) over open links,
 * on the sources' own storeys when `storeyOf` is given, up to `limit` metres.
 * `from` is the index in `sources` of each node's nearest source (-1: none).
 */
function walkFrom(graph, sources, { blocked = null, storeyOf = null, limit = Infinity } = {}) {
  const { indptr, neighbours, edgeIds, n } = csr(graph);
  const w = edgeLengths(graph);
  const dist = new Float64Array(n).fill(Infinity);
  const from = new Int32Array(n).fill(-1);
  const heap = new Heap();
  sources.forEach((v, i) => {
    if (!(v >= 0 && v < n) || dist[v] === 0) return;
    dist[v] = 0;
    from[v] = i;
    heap.push(0, v);
  });
  const done = new Uint8Array(n);
  while (heap.size) {
    const u = heap.pop();
    if (done[u]) continue;
    done[u] = 1;
    const du = dist[u];
    for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const v = neighbours[k];
      if (storeyOf && storeyOf[v] !== storeyOf[u]) continue;
      const d = du + w[edgeIds[k]];
      if (d < dist[v] && d <= limit) {
        dist[v] = d;
        from[v] = from[u];
        heap.push(d, v);
      }
    }
  }
  return { dist, from };
}

/** Sites as {index, node, rating}: node indices, or records with `node`. */
function siteList(sites, n) {
  return (sites || [])
    .map((s, index) => {
      const node = typeof s === "number" ? s : s?.node;
      const rating = typeof s === "object" && s && Number.isFinite(s.rating) ? s.rating : null;
      return { index, node, rating, site: s };
    })
    .filter((s) => Number.isInteger(s.node) && s.node >= 0 && s.node < n);
}

// ------------------------------------------------------------------ the rules

/**
 * The travel limit (m) for a fire class: A, K and D fixed; B by hazard and
 * the unit's B rating (null when the unit is below the smallest the hazard
 * allows).
 *
 * @returns {{value: number|null, clause: string, text: string}}
 */
export function travelLimit(fireClass = "A", { hazard = "light", rating = null } = {}) {
  if (fireClass === "B") {
    const rows = TRAVEL.B[hazardOf(hazard).id];
    const nB = rating ?? rows[0][0];
    const row = rows.slice().reverse().find(([r]) => nB >= r);
    return row
      ? { value: row[1], clause: TRAVEL.B.clause, text: `${nB}-B: within ${row[1]} m` }
      : { value: null, clause: TRAVEL.B.clause, text: `${nB}-B is below the smallest for ${hazard} hazard (${rows[0][0]}-B)` };
  }
  const T = TRAVEL[fireClass];
  if (!T?.value) throw new RangeError(`No travel rule for class "${fireClass}"; use A, B, K or D (class C follows the A or B hazard, NFPA 10 6.4).`);
  return { value: T.value, clause: T.clause, text: `within ${T.value} m` };
}

/**
 * The fewest class A units a floor needs (NFPA 10 6.2.1.2.1): ceil(A /
 * min(n_A a_h, 1045 m²)), at least one on any floor with area (6.2.1.5).
 * It never rises with a higher rating, nor, for a given rating, with a
 * lighter hazard (count_rule_antitone); with each hazard's least rating, as
 * when none is given, it can: 1000 m² needs 4 units of 2-A at ordinary
 * hazard but 3 of 4-A at extra (count_default_rating_counterexample).
 * Infinity for a unit with no class A rating.
 */
export function requiredCount(area, hazard = "light", rating = null) {
  const H = hazardOf(hazard);
  if (!(area > 0)) return 0;
  const nA = rating ?? H.minRating.value;
  if (!(nA > 0)) return Infinity;
  const per = Math.min(nA * H.areaPerUnit.value, AREA_PER_EXTINGUISHER.value);
  return Math.max(1, Math.ceil(area / per - 1e-9));
}

const REQUIRED_CLAUSE =
  "SBC 801-CR-2018 906.1 item 1: new and existing Groups A, B, E, F, H, I, M, R-1, R-2, R-4 and S (in R-2, only the places of items 2 to 6 where each dwelling unit has a unit of at least 1-A:10-B:C)";
const NOT_CHECKED_906 = [
  "SBC 801-CR-2018 906.1 items 2 to 6 not checked: within 9 m of commercial cooking equipment (904.12.5, class K), where flammable or combustible liquids are stored, used or dispensed (class B, Table 906.3(2)), each floor under construction, the uses of Table 906.1, and special-hazard areas such as laboratories, computer rooms and generator rooms where the official requires",
  "not checked: mounting heights (SBC 801-CR-2018 906.9.1 to 906.9.3), visibility and obstruction (906.6), cabinets unlocked (906.8)",
];

/**
 * Whether SBC 801-CR-2018 906.1 item 1 requires extinguishers for the
 * occupancy: true, false or null (not given, or not a group it names).
 */
function requiredFor(occupancy) {
  const groups = (Array.isArray(occupancy) ? occupancy : [occupancy]).filter((g) => g != null && String(g).trim()).map((g) => String(g).trim());
  if (!groups.length) return { required: null, groups, text: "no occupancy given" };
  const yes = [], no = [], unsure = [];
  for (const g of groups) {
    if (/univ/i.test(g)) yes.push(`${g} (Group B; a lecture hall may be A-3)`);
    else if (/^R-?3$/i.test(g) || /^U$/i.test(g)) no.push(g);
    else if (/^R$/i.test(g)) unsure.push(`${g} (give R-1 to R-4: R-3 is not listed)`);
    else if (/^(A(-?[1-5])?|B|E|F(-?[12])?|H(-?[1-5])?|I(-?[1-4])?|M|R-?[124]|S(-?[12])?)$/i.test(g)) yes.push(g);
    else unsure.push(`${g} (not a group 906.1 names)`);
  }
  if (yes.length) return { required: true, groups, text: `required for ${yes.join(", ")}` };
  if (unsure.length) return { required: null, groups, text: `cannot tell for ${unsure.join(", ")}` };
  return { required: false, groups, text: `not required by item 1 for ${no.join(", ")}` };
}

/**
 * The placement check (note 6.1): per storey, the travel rule with the
 * sampling margin, the places no extinguisher reaches, the count rule and the
 * least rating; for the building, where 906.1 requires extinguishers. Each a
 * rule report {id, status: "pass" | "fail" | "cannot tell" | "n/a", text,
 * clause}. One multi-source Dijkstra over the storeys' own links: O(E log V).
 *
 * Storeys are codeCheck's (storeysOf): stair nodes belong to none, and a
 * site on a stair serves none. Door nodes are walked through but are not
 * places, as in the egress checks. The floor area is the walkable area
 * (nodes times s²), less than the gross area the code divides, so the count
 * is a lower bound unless `areas` gives the storeys' gross areas.
 *
 * @param {object} graph   nodes (viewer frame, lifted by agentHeight), edges, kinds, blocked, upAxis, agentHeight, spacing
 * @param {Array<number|{node: number, rating?: number}>} sites  extinguisher nodes; `rating` the unit's (A units, or B units for class B)
 * @param {object} opts
 *   levels, storeyNames  as the code check
 *   hazard       "light" | "ordinary" | "extra", or one per storey (default light)
 *   occupancy    a group or groups ("B", "A-3", "R-2", "university" ...) for 906.1
 *   rating       the units' rating where a site gives none (default: not known)
 *   spacing      the graph's lattice spacing (default graph.spacing, else 0.5 m)
 *   fireClass    "A" (default), "B", "K" or "D": the travel limit; the count rule is class A's
 *   areas        gross floor area per storey (m²), optional
 *   useWalls     respect links blocked by walls (default true)
 * @returns {{status, fireClass, limit, margin, passWithin, required, storeys, distance: Float64Array, nearest: Int32Array, notes}}
 *   distance: each node's walk to its nearest extinguisher (m); nearest: that
 *   site's index in `sites` (-1: none)
 */
export function placementCheck(graph, sites = [], opts = {}) {
  const {
    levels = [],
    storeyNames = [],
    hazard = "light",
    occupancy = null,
    rating = null,
    spacing = graph.spacing || 0.5,
    fireClass = "A",
    areas = null,
    useWalls = true,
  } = opts;
  const n = graph.nodes.length / 3;
  const kinds = graph.kinds || new Uint8Array(n);
  const storeyOf = storeysOf(graph, levels);
  const floors = levels.length ? levels.length : 1;
  const delta = samplingMargin(spacing);
  const listed = siteList(sites, n);
  const usable = listed.filter((s) => storeyOf[s.node] >= 0);
  const { dist, from } = walkFrom(graph, usable.map((s) => s.node), { blocked: useWalls ? graph.blocked : null, storeyOf });
  const nearest = new Int32Array(n).fill(-1);
  for (let v = 0; v < n; v += 1) if (from[v] >= 0) nearest[v] = usable[from[v]].index;

  // Per storey: places, the farthest walk, and the sites with their ratings.
  const per = Array.from({ length: floors }, () => ({ places: 0, unreached: 0, max: -Infinity, worst: -1, near: 0, over: 0, sites: 0, least: Infinity, unrated: 0 }));
  for (const s of usable) {
    const p = per[storeyOf[s.node]];
    p.sites += 1;
    const r = s.rating ?? rating;
    if (r == null) p.unrated += 1;
    else p.least = Math.min(p.least, r);
  }
  const limits = per.map((p, s) => travelLimit(fireClass, { hazard: hazardAt(hazard, s).id, rating: Number.isFinite(p.least) ? p.least : rating }));
  for (let v = 0; v < n; v += 1) {
    const s = storeyOf[v];
    if (s < 0 || kinds[v] === KIND_DOOR) continue;
    const p = per[s];
    p.places += 1;
    const d = dist[v];
    if (d === Infinity) {
      p.unreached += 1;
      continue;
    }
    if (d > p.max) {
      p.max = d;
      p.worst = v;
    }
    const D = limits[s].value;
    if (D == null) continue;
    if (d > D + 1e-9) p.over += 1;
    else if (d > D - delta + 1e-9) p.near += 1;
  }

  // Where 906.1 requires them.
  const req = requiredFor(occupancy);
  const bare = per.map((p, s) => (p.places && !p.sites ? s : -1)).filter((s) => s >= 0);
  const name = (s) => storeyNames[s] || `Storey ${s + 1}`;
  const required = {
    id: "required",
    status: req.required === null ? "cannot tell" : req.required && bare.length ? "fail" : "pass",
    text:
      req.required === null
        ? `${req.text}: whether extinguishers are required is not known`
        : req.required
          ? `${req.text}${bare.length ? `; none on ${bare.map(name).join(", ")}` : "; every storey has at least one"}`
          : req.text,
    clause: REQUIRED_CLAUSE,
  };
  const notRequired = req.required === false;

  const fmt = (x) => x.toFixed(2);
  const storeys = per
    .map((p, s) => {
      if (!p.places) return null;
      const H = hazardAt(hazard, s);
      const L = limits[s];
      const judged = p.places - p.unreached;
      const checks = [];
      // Travel.
      if (!p.sites) {
        checks.push({
          id: "travel",
          status: notRequired ? "n/a" : "fail",
          text: notRequired ? "no extinguisher on this storey; not required" : "no extinguisher on this storey",
          clause: L.clause,
        });
      } else if (L.value == null) {
        checks.push({ id: "travel", status: "fail", text: L.text, clause: L.clause });
      } else if (!judged) {
        checks.push({ id: "travel", status: "cannot tell", text: "no place on this storey reaches an extinguisher on the graph", clause: L.clause });
      } else {
        const D = L.value;
        const far = `the farthest place ${fmt(p.max)} m from one (node ${p.worst})`;
        checks.push(
          p.over
            ? { id: "travel", status: "fail", text: `${p.over} of ${judged} places more than ${D} m from an extinguisher; ${far}`, clause: L.clause }
            : p.near
              ? {
                  id: "travel",
                  status: "cannot tell",
                  text: `every place within ${D} m, but ${p.near} near the limit (beyond ${fmt(D - delta)} m: the floor between nodes, up to ${fmt(delta)} m off, may be farther); ${far}`,
                  clause: L.clause,
                }
              : {
                  id: "travel",
                  status: "pass",
                  text: `every place within ${fmt(D - delta)} m (${D} m less the ${fmt(delta)} m sampling margin); ${far}`,
                  clause: L.clause,
                },
        );
      }
      if (p.sites && p.unreached && judged) {
        checks.push({
          id: "unreached",
          status: "cannot tell",
          text: `${p.unreached} of ${p.places} places reach no extinguisher on this storey on the graph: often a door the model lacks, or a pocket of floor`,
          clause: L.clause,
        });
      }
      // Count and least rating: class A only.
      const area = Number.isFinite(areas?.[s]) ? areas[s] : p.places * spacing * spacing;
      let needed = null;
      if (fireClass === "A") {
        // Units without a rating are taken at the hazard's least (or `rating`),
        // which asks for the most units; `best` is the fewest any rating allows.
        const assumed = rating ?? H.minRating.value;
        const known = p.sites > 0 && !p.unrated;
        const nA = !p.sites ? assumed : p.unrated ? Math.min(p.least, assumed) : p.least;
        needed = requiredCount(area, H.id, nA);
        const best = requiredCount(area, H.id, Infinity);
        const perUnit = Math.min(nA * H.areaPerUnit.value, AREA_PER_EXTINGUISHER.value);
        const areaText = `${area.toFixed(0)} m² ${Number.isFinite(areas?.[s]) ? "gross" : "walkable"}`;
        checks.push(
          !p.sites && notRequired
            ? { id: "count", status: "n/a", text: "no extinguisher on this storey; not required", clause: `${COUNT_CLAUSE}; ${AREA_PER_UNIT_CLAUSE}` }
            : {
                id: "count",
                status: p.sites >= needed ? "pass" : known || p.sites < best ? "fail" : "cannot tell",
                text:
                  `${p.sites} extinguisher${p.sites === 1 ? "" : "s"} for ${areaText}; ${needed} needed at ${perUnit.toFixed(0)} m² each (${nA}-A, ${H.label.toLowerCase()})` +
                  (known || !p.sites ? "" : `: ratings not given, taken as ${nA}-A; larger units need as few as ${best}`),
                clause: `${COUNT_CLAUSE}; ${AREA_PER_UNIT_CLAUSE}; ${AREA_PER_EXTINGUISHER.clause}`,
              },
        );
        if (p.sites) {
          const least = Number.isFinite(p.least) ? p.least : null;
          checks.push({
            id: "rating",
            status: least != null && least < H.minRating.value ? "fail" : p.unrated ? "cannot tell" : "pass",
            text:
              least != null && least < H.minRating.value
                ? `a ${least}-A unit, below the ${H.minRating.value}-A the ${H.label.toLowerCase()} needs`
                : p.unrated
                  ? `${p.unrated} of ${p.sites} units without a rating; at least ${H.minRating.value}-A needed`
                  : `every unit at least ${H.minRating.value}-A (the least ${least}-A)`,
            clause: MIN_RATING_CLAUSE,
          });
        }
      }
      return {
        index: s,
        name: name(s),
        hazard: H.id,
        places: p.places,
        judged,
        unreached: p.unreached,
        area,
        sites: p.sites,
        needed,
        maxTravel: p.max > -Infinity ? p.max : null,
        worst: p.worst >= 0 ? { node: p.worst, distance: p.max, site: nearest[p.worst] } : null,
        near: p.near,
        over: p.over,
        checks,
        status: worst(checks.map((c) => c.status)),
      };
    })
    .filter(Boolean);

  const notes = [
    `sampling margin ${fmt(delta)} m (spacing ${spacing} m over √2): a place passes within ${passText(fireClass, delta)}`,
    ...(Number.isFinite(areas?.[0]) ? [] : ["floor areas are the graph's walkable areas, less than the gross areas the code divides: the count is a lower bound"]),
    ...(listed.length > usable.length ? [`${listed.length - usable.length} site${listed.length - usable.length === 1 ? "" : "s"} on a stair serve no storey`] : []),
    ...(req.groups.some((g) => /^A-?5$/i.test(g)) ? ["SBC 801-CR-2018 906.2: the travel distance is not applied to Group A-5 spectator seating"] : []),
    ...NOT_CHECKED_906,
  ];
  const limit = fireClass === "B" ? null : (TRAVEL[fireClass]?.value ?? null);
  return {
    status: worst([required.status, ...storeys.map((s) => s.status)]),
    fireClass,
    limit,
    margin: delta,
    passWithin: limit == null ? null : limit - delta,
    required,
    storeys,
    distance: dist,
    nearest,
    notes,
  };
}

/** The pass distance in words, for the notes. */
function passText(fireClass, delta) {
  if (fireClass === "B") return `the class B distance less ${delta.toFixed(2)} m`;
  const D = TRAVEL[fireClass].value;
  return `${(D - delta).toFixed(2)} m of ${D} m`;
}

// ------------------------------------------------------------------ the layout

/**
 * A greedy layout (note 6.2), per storey. Candidates are walkable floor
 * nodes beside a wall (fewer open links than the lattice's full count: at
 * most a cell from it, about 0.5 m on the 0.5 m lattice against the note's
 * 0.6 m) and within 2 m of walk of a door or 1 m of an egress route node,
 * thinned to one per 1 m. Each covers the places within D_A - delta of walk
 * on its storey. Greedy set cover takes the candidate covering the most
 * uncovered places (ties: the nearest an exit) until all are covered (lazy:
 * a candidate's count only falls, so a recounted one that still beats the
 * next bound is the greedy choice); places no candidate reaches get units of
 * their own from among themselves (reason "fallback"). Then units are added
 * at the place farthest from the others until the count rule holds (reason
 * "count"). Greedy is within 1 + ln m of the fewest units for m places
 * (Johnson 1974; Chvátal 1979). The result is checked by placementCheck, so
 * the layout never has to be trusted, only the check.
 *
 * @param {object} graph  as placementCheck
 * @param {object} opts
 *   levels, storeyNames, hazard, occupancy, spacing, useWalls, areas  as placementCheck
 *   exits       exit nodes, for ties (nearest an exit first)
 *   routeNodes  nodes on egress routes (optional; doors alone without them)
 *   rating      the units' class A rating (default the hazard's least: 2-A, or 4-A in extra)
 * @returns {{sites: Array<{node, position, storey, rating, reason}>, storeys: object[], check: object, notes: string[]}}
 *   position: the node's, in the graph's frame (lifted by agentHeight)
 */
export function layoutExtinguishers(graph, opts = {}) {
  const {
    levels = [],
    storeyNames = [],
    hazard = "light",
    exits = [],
    routeNodes = [],
    occupancy = null,
    rating = null,
    spacing = graph.spacing || 0.5,
    useWalls = true,
    areas = null,
  } = opts;
  const n = graph.nodes.length / 3;
  const P = graph.nodes;
  const kinds = graph.kinds || new Uint8Array(n);
  const blocked = useWalls ? graph.blocked : null;
  const storeyOf = storeysOf(graph, levels);
  const floors = levels.length ? levels.length : 1;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const w = edgeLengths(graph);
  const [h0, h1] = planAxes(graph.upAxis || "y");
  const R = TRAVEL.A.value - samplingMargin(spacing);
  const exitDist = exits.length ? walkToExit(graph, exits, { speed: 1, blocked }) : new Float64Array(n).fill(Infinity);
  const isPlace = new Uint8Array(n);
  for (let v = 0; v < n; v += 1) if (storeyOf[v] >= 0 && kinds[v] !== KIND_DOOR) isPlace[v] = 1;

  // Beside a wall: fewer open links than the fullest floor node has.
  const open = new Uint8Array(n);
  let full = 0;
  for (let v = 0; v < n; v += 1) {
    let k = 0;
    for (let e = indptr[v]; e < indptr[v + 1]; e += 1) if (!blocked?.[edgeIds[e]]) k += 1;
    open[v] = Math.min(255, k);
    if (kinds[v] === KIND_FLOOR && k > full) full = k;
  }
  // Near a door or an egress route, by walk on the storey.
  const doorNodes = [];
  for (let v = 0; v < n; v += 1) if (kinds[v] === KIND_DOOR) doorNodes.push(v);
  for (const d of graph.doors || []) if (d?.node >= 0 && kinds[d.node] !== KIND_DOOR) doorNodes.push(d.node);
  const nearDoor = doorNodes.length ? walkFrom(graph, doorNodes, { blocked, storeyOf, limit: DOOR_REACH }).dist : null;
  const nearRoute = routeNodes.length ? walkFrom(graph, routeNodes, { blocked, storeyOf, limit: ROUTE_REACH }).dist : null;
  const guided = new Uint8Array(floors);
  for (const v of doorNodes) if (storeyOf[v] >= 0) guided[storeyOf[v]] = 1;
  for (const v of routeNodes) if (storeyOf[v] >= 0) guided[storeyOf[v]] = 1;

  // One per THIN metres, in the plan, per storey.
  const thin = (list) => {
    const cells = new Map();
    const kept = [];
    for (const v of list) {
      const s = storeyOf[v];
      const a = P[v * 3 + h0], b = P[v * 3 + h1];
      const i = Math.floor(a / THIN), j = Math.floor(b / THIN);
      let close = false;
      for (let di = -1; di <= 1 && !close; di += 1) {
        for (let dj = -1; dj <= 1 && !close; dj += 1) {
          for (const u of cells.get(`${s}:${i + di}:${j + dj}`) || []) {
            if (Math.hypot(P[u * 3 + h0] - a, P[u * 3 + h1] - b) < THIN - 1e-9) {
              close = true;
              break;
            }
          }
        }
      }
      if (close) continue;
      kept.push(v);
      const key = `${s}:${i}:${j}`;
      if (!cells.has(key)) cells.set(key, []);
      cells.get(key).push(v);
    }
    return kept;
  };

  // The places within R of a node on its storey (a bounded Dijkstra; the
  // work arrays are reset after each ball).
  const dist = new Float64Array(n).fill(Infinity);
  const done = new Uint8Array(n);
  const touched = [];
  const heap = new Heap();
  const ball = (src, visit) => {
    const s = storeyOf[src];
    dist[src] = 0;
    touched.push(src);
    heap.push(0, src);
    while (heap.size) {
      const u = heap.pop();
      if (done[u]) continue;
      done[u] = 1;
      visit(u);
      for (let k = indptr[u]; k < indptr[u + 1]; k += 1) {
        if (blocked?.[edgeIds[k]]) continue;
        const v = neighbours[k];
        if (done[v] || storeyOf[v] !== s) continue;
        const d = dist[u] + w[edgeIds[k]];
        if (d <= R && d < dist[v]) {
          if (dist[v] === Infinity) touched.push(v);
          dist[v] = d;
          heap.push(d, v);
        }
      }
    }
    for (const v of touched) {
      dist[v] = Infinity;
      done[v] = 0;
    }
    touched.length = 0;
  };

  const covered = new Uint8Array(n);
  const sites = [];
  const storeys = [];
  const byExit = (a, b) => exitDist[a] - exitDist[b] || a - b;
  for (let s = 0; s < floors; s += 1) {
    const places = [];
    for (let v = 0; v < n; v += 1) if (storeyOf[v] === s && isPlace[v]) places.push(v);
    if (!places.length) continue;
    const H = hazardAt(hazard, s);
    const nA = rating ?? H.minRating.value;
    const raw = places.filter(
      (v) => kinds[v] === KIND_FLOOR && open[v] < full && (!guided[s] || nearDoor?.[v] <= DOOR_REACH || nearRoute?.[v] <= ROUTE_REACH),
    );
    const candidates = thin(raw);
    let uncovered = places.length;
    const chosen = [];
    const take = (c, reason) => {
      chosen.push(c);
      sites.push({ node: c, position: [P[c * 3], P[c * 3 + 1], P[c * 3 + 2]], storey: s, rating: nA, reason });
      ball(c, (v) => {
        if (isPlace[v] && !covered[v]) {
          covered[v] = 1;
          uncovered -= 1;
        }
      });
    };
    // Lazy greedy over a candidate list: priority -gain M + rank, rank by exit.
    const greedy = (list, reason) => {
      const order = list.slice().sort(byExit);
      const M = order.length + 1;
      const gain = (c) => {
        let g = 0;
        ball(c, (v) => {
          if (isPlace[v] && !covered[v]) g += 1;
        });
        return g;
      };
      const queue = new Heap();
      order.forEach((c, rank) => {
        const g = gain(c);
        if (g > 0) queue.push(-g * M + rank, rank);
      });
      while (uncovered > 0 && queue.size) {
        const rank = queue.pop();
        const g = gain(order[rank]);
        if (!g) continue;
        const p = -g * M + rank;
        if (queue.size && p > queue.p[0]) {
          queue.push(p, rank);
          continue;
        }
        take(order[rank], reason);
      }
    };
    greedy(candidates, "cover");
    const fromCover = chosen.length;
    let extra = [];
    if (uncovered > 0) {
      extra = thin(places.filter((v) => !covered[v]));
      greedy(extra, "fallback");
    }
    const fallback = chosen.length - fromCover;
    // The count rule: add at the place farthest from the units placed.
    const area = Number.isFinite(areas?.[s]) ? areas[s] : places.length * spacing * spacing;
    const needed = requiredCount(area, H.id, nA);
    const pool = [...candidates, ...extra];
    let forCount = 0;
    while (chosen.length < needed && Number.isFinite(needed)) {
      const d = walkFrom(graph, chosen, { blocked, storeyOf }).dist;
      const far = (list) => list.reduce((b, v) => (d[v] < Infinity && (b < 0 || d[v] > d[b] || (d[v] === d[b] && byExit(v, b) < 0)) ? v : b), -1);
      let c = far(pool.filter((v) => d[v] > 0));
      if (c < 0) c = far(places.filter((v) => d[v] > 0));
      if (c < 0) break;
      take(c, "count");
      forCount += 1;
    }
    storeys.push({
      index: s,
      name: storeyNames[s] || `Storey ${s + 1}`,
      hazard: H.id,
      places: places.length,
      candidates: candidates.length,
      units: chosen.length,
      cover: fromCover,
      fallback,
      count: forCount,
      needed,
    });
  }
  const check = placementCheck(graph, sites, { levels, storeyNames, hazard, occupancy, spacing, useWalls, areas });
  const notes = [
    `each unit covers the places within ${R.toFixed(2)} m of walk (${TRAVEL.A.value} m less the sampling margin)`,
    "candidates: floor nodes beside a wall, within 2 m of walk of a door or 1 m of an egress route (NFPA 10 F.6.2.2; SBC 801-CR-2018 906.5), one per metre",
    ...(routeNodes.length ? [] : ["no egress route given: candidates near doors only"]),
    ...(storeys.some((st) => st.fallback) ? ["places no candidate reaches got units of their own, off the routes (fallback): often a room whose door the model lacks"] : []),
  ];
  return { sites, storeys, check, notes };
}

// ------------------------------------------------------------------ IFC records

/** An extinguisher or extinguisher cabinet by name, in the corpus's languages (note 5.2, 5.3 rule 1). */
const EXTINGUISHER_NAME =
  /extinguisher|fire[\s_-]?ext\b|feuerl(ö|oe|o)scher|extincteur|brann[\s_-]?sl[uo]kk|h(å|aa)ndslukker|brand[\s_-]?slukk|ildslukk|brand[\s_-]?sl(ä|a|ae)ckare|brand[\s_-]?blus|blusapparaat|blustoestel|sammutin|طفاي/i;
/** "FEC" counts only as a whole token beside a cabinet (note 5.2: every other FEC was a false positive). */
const FEC = /(^|[^a-z0-9])FEC([^a-z0-9]|$)/i;
const CABINET = /cabinet|schrank|armoire|skap\b|kaappi|kabinet|kast\b/i;
/**
 * Hose reels, hydrants, breeching inlets and sprinklers are never
 * extinguishers (note 5.3 rule 1; ifcSemantics.js). "Hose" as a whole word:
 * Revit's "Wall Hosted" families are not hoses.
 */
const NOT_EXTINGUISHER = /\bhose\b|hose[\s_-]?reel|hydrant|breeching|sprinkler|schlauch|\bslange|slangrulle|brandpost/i;
const OTHER_TERMINALS = new Set(["HOSEREEL", "FIREHYDRANT", "BREECHINGINLET", "SPRINKLER", "SPRINKLERDEFLECTOR", "FIREMONITOR"]);
/** The product classes an extinguisher was found as, or may be written as (note 5.1 to 5.3). */
const CANDIDATE_CLASSES = new Set([
  "IFCBUILDINGELEMENTPROXY",
  "IFCFLOWTERMINAL",
  "IFCFIRESUPPRESSIONTERMINAL",
  "IFCSANITARYTERMINAL",
  "IFCFURNISHINGELEMENT",
  "IFCFURNITURE",
  "IFCSYSTEMFURNITUREELEMENT",
  "IFCDISTRIBUTIONCONTROLELEMENT",
  "IFCDISTRIBUTIONELEMENT",
]);

/** Agents by name, the specific before the general (note 5.3 rule 2). */
const AGENT_NAMES = [
  ["wet-chemical", /wet[\s_-]?chem|class[\s_-]?k\b|fettbrand/i],
  ["class-d", /class[\s_-]?d\b|metallbrand/i],
  ["co2", /\bc[o0][\s_-]?2|co₂|carbon[\s_-]?dioxide|kohlendioxid|koldioxid|kooldioxide|hiilidioksidi|karbondioksid|kullsyre|kolsyre/i],
  ["water-mist", /water[\s_-]?mist|wassernebel/i],
  ["foam", /foam|schaum|mousse|\bskum|schuim|vaahto|\bafff\b/i],
  ["abc", /\babc\b|multi[\s_-]?purpose|glutbrand|polyvalent/i],
  ["powder", /powder|dry[\s_-]?chem|pulver|poudre|poeder|jauhe|\bdcp\b/i],
  ["water", /water|wasser|\beau\b|\bvann\b|\bvatten\b|vesi/i],
];

/** A rating from a name: UL "2-A:10-B:C", EN "34A 183B C"; null when none is written. */
function ratingFrom(text) {
  const A = /\b(\d{1,2})\s?-\s?A\b/i.exec(text);
  const B = /\b(\d{1,3})\s?-\s?B\b/i.exec(text);
  if (A || B) {
    return {
      system: "UL",
      A: A ? Number(A[1]) : 0,
      B: B ? Number(B[1]) : 0,
      C: /-\s?[AB]\s?:\s?(\d{1,3}\s?-\s?B\s?:\s?)?C\b/i.test(text),
      K: /:\s?K\b/i.test(text),
    };
  }
  const a = /\b(\d{1,2})\s?A\b/.exec(text);
  const b = /\b(\d{1,3})\s?B\b/.exec(text);
  if (a || b) return { system: "EN", A: a ? Number(a[1]) : 0, B: b ? Number(b[1]) : 0, C: /\d\s?B\s?[,:/\s]\s?C\b/.test(text), K: false };
  return null;
}

/** The charge from a name: kg, or litres (gallons and pounds converted). */
function chargeFrom(text) {
  const num = (m) => Number(m[1].replace(",", "."));
  const kg = /(\d+(?:[.,]\d+)?)\s?kg\b/i.exec(text);
  const lb = /(\d+(?:[.,]\d+)?)\s?lbs?\b/i.exec(text);
  const l = /(\d+(?:[.,]\d+)?)\s?(?:l|lt|ltr|litre|liter)\b/i.exec(text);
  const gal = /(\d+(?:[.,]\d+)?)\s?gal/i.exec(text);
  return { kg: kg ? num(kg) : lb ? num(lb) * 0.45359237 : null, litres: l ? num(l) : gal ? num(gal) * 3.785411784 : null };
}

/** The catalogue unit a name points to, when its agent and charge do. */
function unitGuess(agent, { kg, litres }) {
  if (agent === "water" || agent === "water-mist") return litres == null ? null : litres >= 8 ? "water-9.5" : "water-6";
  if (agent === "foam") return litres != null && litres >= 8 ? "foam-9.5" : null;
  if (agent === "abc") return kg == null ? null : kg <= 3 ? "abc-2.3" : kg <= 5.5 ? "abc-4.5" : "abc-6-en";
  if (agent === "co2") return kg != null && kg <= 3 ? "co2-2" : null;
  if (agent === "wet-chemical") return "wet-chemical-6";
  return null;
}

/**
 * The extinguishers among IFC element records (note 5.3): a product of a
 * candidate class (any class when none is given) whose name, object type or
 * type name names an extinguisher or an extinguisher cabinet ("Fire
 * Extinguisher", Revit's "Extinguisher - C02: Red", "Feuerlöscher",
 * "brannslukker", "brandsläckare", "brandblusser", "sammutin" ...; "FEC" only
 * beside "cabinet"). Anything naming or typed as a hose reel, hydrant,
 * breeching inlet or sprinkler is left out, a combined cabinet included (a
 * hose station is not an extinguisher, NFPA 10 6.2.1.4; leaving one out errs
 * on the safe side). A cabinet counts as one unit. The agent, rating and
 * catalogue unit are read from the names where written, else "unknown" and
 * null: the attempt then uses the default unit and says so.
 *
 * @param {Array<{itemId, name?, objectType?, typeName?, ifcClass?, predefinedType?}>} elements
 * @returns {Array<{itemId, name, agent, rating, unit, cabinet}>} with `excluded`:
 *   [{itemId, name, reason}] for records that named an extinguisher but were left out
 */
export function extinguisherRecords(elements) {
  const out = [];
  const excluded = [];
  for (const el of elements || []) {
    if (!el) continue;
    const texts = [el.name, el.objectType, el.typeName].filter((t) => t != null && String(t).trim()).map(String);
    const text = texts.join(" | ").normalize("NFC");
    const named = EXTINGUISHER_NAME.test(text) || (FEC.test(text) && CABINET.test(text));
    if (!named) continue;
    const name = el.name ?? el.typeName ?? el.objectType ?? null;
    const cls = el.ifcClass ? String(el.ifcClass).toUpperCase() : null;
    const kind = el.predefinedType ? String(el.predefinedType).toUpperCase() : null;
    if (cls && !CANDIDATE_CLASSES.has(cls)) {
      excluded.push({ itemId: el.itemId, name, reason: `${el.ifcClass} is not a product an extinguisher is written as` });
      continue;
    }
    if ((kind && OTHER_TERMINALS.has(kind)) || NOT_EXTINGUISHER.test(text)) {
      excluded.push({ itemId: el.itemId, name, reason: "names or is typed as a hose reel, hydrant, breeching inlet or sprinkler: not an extinguisher (a hose station, NFPA 10 6.2.1.4)" });
      continue;
    }
    let agent = AGENT_NAMES.find(([, re]) => re.test(text))?.[0] ?? "unknown";
    if (agent === "powder" && /\bBC\b/.test(text)) agent = "bc";
    out.push({
      itemId: el.itemId,
      name,
      agent,
      rating: ratingFrom(text),
      unit: unitGuess(agent, chargeFrom(text)),
      cabinet: CABINET.test(text) || FEC.test(text),
    });
  }
  return Object.assign(out, { excluded });
}

// ------------------------------------------------------------------ suitability

/** How each agent appears in fireClasses.js's agent lists. */
const AGENT_MATCH = {
  water: (s) => /\bwater\b/i.test(s),
  "water-mist": (s) => /\bwater\b/i.test(s),
  foam: (s) => /\bfoam\b/i.test(s),
  abc: (s) => /multipurpose/i.test(s) || /^dry chemical$/i.test(s),
  bc: (s) => /^dry chemical$/i.test(s),
  co2: (s) => /carbon dioxide/i.test(s),
  "wet-chemical": (s) => /wet chemical/i.test(s),
  "class-d": (s) => /class d dry powder/i.test(s),
  "clean-agent": (s) => /clean agent/i.test(s),
};

/**
 * Whether an agent suits a fire class, from fireClasses.js (NFPA 10): false
 * with the list's reason when it is listed as unsuitable, true when listed as
 * suitable, false when not listed at all, null when the agent is not known
 * (and for a powder of unknown kind on class A: ABC powder suits it, BC does
 * not). Class L is never fought with a portable unit: the list says leave.
 *
 * @returns {{suitable: boolean|null, text: string}}
 */
export function suitability(agent, fireClass = "A") {
  const C = FIRE_CLASSES[fireClass];
  if (!C) throw new RangeError(`Unknown fire class "${fireClass}"; use one of ${Object.keys(FIRE_CLASSES).join(", ")}.`);
  if (fireClass === "L") return { suitable: false, text: `class L: ${C.agents.suitable[0].toLowerCase()} (fireClasses.js)` };
  if (agent === "powder" && fireClass === "A") return { suitable: null, text: "a powder of unknown kind: ABC powder suits class A, BC powder does not" };
  const match = AGENT_MATCH[agent === "powder" ? "bc" : agent];
  if (!match) return { suitable: null, text: `agent ${agent || "unknown"}: suitability for class ${fireClass} cannot be told` };
  const bad = C.agents.unsuitable.find((u) => match(u.agent));
  if (bad) return { suitable: false, text: `${bad.agent}: ${bad.reason}` };
  const good = C.agents.suitable.find(match);
  if (good) return { suitable: true, text: `${good} suits class ${fireClass}` };
  return { suitable: false, text: `not among the agents listed for class ${fireClass}: ${C.agents.suitable.join("; ")}` };
}

// ------------------------------------------------------------------ the attempt

/** Seconds to 1 MW of each growth class, keyed as the two-zone setting is (zone.js GROWTH). */
export const GROWTH_TIME = Object.freeze({ slow: 600, medium: 300, fast: 150, "ultra-fast": 75 });

/**
 * The attempt's fire when there is no Flamescapes run: the two-zone
 * settings' growth and peak as a t-squared fire, 1 MW at the class's time,
 * capped at peakKw. An unknown growth is medium. Returns {growth, tGrowth,
 * hrrAt, text}, text naming it for the result line.
 */
export function designFire(growth, peakKw) {
  const name = GROWTH_TIME[growth] ? growth : "medium";
  const tGrowth = GROWTH_TIME[name];
  return {
    growth: name,
    tGrowth,
    hrrAt: (t) => Math.min(peakKw, 1000 * (t / tGrowth) ** 2),
    text: `${name === "ultra-fast" ? "an" : "a"} ${name} t-squared design fire (1 MW in ${tGrowth} s, at most ${peakKw} kW)`,
  };
}

/** tau_x = t_d / ln(1/epsilon) (s): the cap is epsilon of its start when the unit is empty (decay_at_discharge_end). */
export function capFor(unit = DEFAULT_UNIT, epsilon = CAPACITY.epsilon.value) {
  if (!(epsilon > 0 && epsilon < 1)) throw new RangeError(`epsilon must lie between 0 and 1, not ${epsilon}.`);
  return unitOf(unit).tD / Math.log(1 / epsilon);
}

/** Burning points as [[x, y, z], ...] from an array of points or a flat array. */
function pointsOf(raw) {
  if (!raw) return [];
  if (raw.length && typeof raw[0] === "number") {
    const out = [];
    for (let k = 0; k + 2 < raw.length; k += 3) out.push([raw[k], raw[k + 1], raw[k + 2]]);
    return out;
  }
  return Array.from(raw);
}

/**
 * The responder's attempt (note 6.3 steps 1 to 8), with the fire given as
 * functions of time so the solver stays outside.
 *
 * The extinguisher is the nearest by walk from the start whose agent suits
 * the fire's class; failing that one of unknown agent; failing that the
 * nearest of any (an unsuitable agent is the wrong outcome, with its reason).
 * The attack place is the walkable node nearest the extinguisher by walk,
 * on the fire's floor, from which every burning point (the fire's node, or
 * `burningAt` when the responder sets off with the unit) lies within the
 * unit's reach r in plan and none closer than the standoff: UL 711's 1.8 m,
 * or for a unit whose reach is shorter (2 kg CO2, 2.3 kg ABC at Table H.2's
 * lower end) its reach, since the agent must reach from where discharge
 * starts. A node stands for the floor within the sampling margin delta of
 * it, so the band is widened by delta on both sides (on a 0.5 m lattice no
 * node need lie between 0.55 and 0.9 m of the fire, CO2's band). At t_s the
 * burning points (burningAt(t_s)) must still be within r + delta.
 *
 * Size, at t_s:
 *   A, C   Q(t_s) (hrrAt, kW) against eta n_A q_A: success at most 0.2 x 200
 *          kW per UL A unit (40 kW), failure above 1.0 x 450 kW (450 kW);
 *          a 2-A unit: success up to 80 kW, failure above 0.9 MW
 *   B      the burning area (m²) against 0.0929 n_B (UL's own figure) for
 *          success and the expert's pan, 0.232 n_B, for failure; EN: 0.4 and
 *          1 times pi B / 100
 *   K      wet chemical only, and only while the fire is within the cooking
 *          item (its area, FIRE_CLASSES.K.item, or `itemArea`)
 *   D      a suitable unit cannot be sized: cannot tell
 *   L      no attempt (the list says leave)
 * The outcome: failure if any reason fails (unsuitable, out of reach, too
 * big, no extinguisher reached), else cannot tell if any is unsure, else
 * success. For a growing fire a later arrival never turns failure into
 * success (success_antitone).
 *
 * Time lost against walking straight out, with exits:
 *   (t_s + t_d + T(attack, exit)) - (t_go + T(start, exit)) >= t_take + t_d >= 0
 * by the triangle inequality of the walking times (time_lost_nonneg).
 *
 * @param {object} p
 *   graph, fireNode, startNode  the graph and node indices
 *   sites       extinguisher nodes, or {node, unit?, agent?} (unit: a UNITS key or record)
 *   tGo         when the responder sets off (s from ignition: cue plus response)
 *   speed       walking speed (m/s; default 1.2), also while carrying
 *   unit        the unit where a site names none (default DEFAULT_UNIT)
 *   tTake       readying time (s; default 13.4, untrained)
 *   hrrAt       t => the fire's heat release (kW)
 *   burningArea t => (or a number) the burning liquid's area (m²), classes B and K
 *   fireClass   "A" | "B" | "C" | "D" | "K" | "L" (fireClasses.js)
 *   exits       exit nodes, for the time lost (optional)
 *   burningAt   t => burning points [[x, y, z]] or flat, the graph's frame (optional)
 *   standoff, spacing, itemArea, useWalls
 * @returns {object} {outcome, reasons, attempted, site, extinguisher, attack, unit, unitSource,
 *   agent, suitability, times, hrr, burningArea, capacity, tau, timeLost, notes}
 */
export function attempt({
  graph,
  fireNode,
  startNode,
  sites = [],
  tGo = 0,
  speed = 1.2,
  unit = DEFAULT_UNIT,
  tTake = T_TAKE.value,
  hrrAt = null,
  burningArea = null,
  fireClass = "A",
  exits = [],
  burningAt = null,
  standoff = STANDOFF.value,
  spacing = null,
  itemArea = null,
  useWalls = true,
} = {}) {
  const n = graph.nodes.length / 3;
  const P = graph.nodes;
  const kinds = graph.kinds || new Uint8Array(n);
  const blocked = useWalls ? graph.blocked : null;
  const up = upIndex(graph.upAxis || "y");
  const [h0, h1] = planAxes(graph.upAxis || "y");
  const delta = samplingMargin(spacing ?? graph.spacing ?? 0.5);
  const cls = FIRE_CLASSES[fireClass];
  if (!cls) throw new RangeError(`Unknown fire class "${fireClass}"; use one of ${Object.keys(FIRE_CLASSES).join(", ")}.`);
  const base = {
    outcome: "failure",
    reasons: [],
    attempted: false,
    site: null,
    extinguisher: null,
    attack: null,
    unit: null,
    unitSource: null,
    agent: null,
    suitability: null,
    times: { tGo, toExtinguisher: null, take: tTake, toAttack: null, tS: null, tD: null, end: null },
    hrr: null,
    burningArea: null,
    capacity: null,
    tau: null,
    timeLost: null,
    notes: [],
  };
  const exitTimes = exits.length ? walkToExit(graph, exits, { speed, blocked }) : null;
  const fail = (id, text) => ({ ...base, reasons: [{ id, status: "failure", text }], timeLost: exitTimes ? 0 : null });
  if (fireClass === "L") return fail("leave", suitability(null, "L").text);

  // The extinguisher: the nearest suitable, else unknown, else any.
  const fromStart = distancesFrom(graph, startNode, { blocked });
  const choices = (sites || [])
    .map((s, index) => {
      const node = typeof s === "number" ? s : s?.node;
      if (!(Number.isInteger(node) && node >= 0 && node < n) || fromStart[node] === Infinity) return null;
      const own = typeof s === "object" && s?.unit ? s.unit : null;
      const u = unitOf(own ?? unit);
      const ag = typeof s === "object" && s?.agent && s.agent !== "unknown" ? s.agent : u.agent;
      const suit = suitability(ag, fireClass);
      return { index, node, unit: u, unitSource: own ? "site" : "default", agent: ag, suit, walk: fromStart[node] / speed };
    })
    .filter(Boolean);
  const pick = (want) => choices.filter((c) => c.suit.suitable === want).reduce((b, c) => (!b || c.walk < b.walk ? c : b), null);
  const chosen = pick(true) ?? pick(null) ?? pick(false);
  if (!chosen) return fail("no-extinguisher", "no extinguisher is reached on the graph from the start");
  const u = chosen.unit;
  const r = u.reach;
  const tLeave = tGo + chosen.walk + tTake;

  // The attack place, by the fire seen on setting off with the unit.
  const fire = [P[fireNode * 3], P[fireNode * 3 + 1], P[fireNode * 3 + 2]];
  const firstPoints = burningAt ? pointsOf(burningAt(tLeave)) : [];
  const points0 = firstPoints.length ? firstPoints : [fire];
  const near = Math.max(0, Math.min(standoff, r) - delta);
  const ring = [];
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] === KIND_STAIR || Math.abs(P[v * 3 + up] - fire[up]) > SAME_FLOOR) continue;
    let lo = Infinity, hi = 0;
    for (const q of points0) {
      const d = Math.hypot(P[v * 3 + h0] - q[h0], P[v * 3 + h1] - q[h1]);
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    if (lo >= near && hi <= r + delta) ring.push(v);
  }
  const toRing = ring.length ? walkFrom(graph, ring, { blocked }) : null;
  const common = {
    ...base,
    attempted: true,
    site: chosen.index,
    extinguisher: chosen.node,
    unit: u,
    unitSource: chosen.unitSource,
    agent: chosen.agent,
    suitability: chosen.suit,
    tau: capFor(u),
    notes: [
      ...(chosen.unitSource === "default" ? [`the site names no unit: ${u.label} taken`] : []),
      ...(u.provisional ? [`${u.label}: ${u.provisional.join(" and ")} are stand-ins, not from the research note`] : []),
    ],
  };
  if (!toRing || toRing.dist[chosen.node] === Infinity) {
    return {
      ...common,
      times: { ...base.times, toExtinguisher: chosen.walk },
      reasons: [{ id: "out-of-reach", status: "failure", text: `no walkable place on the fire's floor reaches the fire within the unit's ${r} m from at least ${near.toFixed(2)} m` }],
    };
  }
  const attack = ring[toRing.from[chosen.node]];
  const toAttack = toRing.dist[chosen.node] / speed;
  const tS = tLeave + toAttack;
  const times = { tGo, toExtinguisher: chosen.walk, take: tTake, toAttack, tS, tD: u.tD, end: tS + u.tD };
  const reasons = [];

  // The agent.
  if (chosen.suit.suitable === false) reasons.push({ id: "unsuitable", status: "failure", text: chosen.suit.text });
  else if (chosen.suit.suitable === null) reasons.push({ id: "agent", status: "cannot tell", text: chosen.suit.text });

  // Reach at t_s.
  const pointsS = burningAt ? pointsOf(burningAt(tS)) : points0;
  let farthest = 0;
  for (const q of pointsS.length ? pointsS : points0) farthest = Math.max(farthest, Math.hypot(P[attack * 3 + h0] - q[h0], P[attack * 3 + h1] - q[h1]));
  if (farthest > r + delta + 1e-9) {
    reasons.push({ id: "out-of-reach", status: "failure", text: `a burning point ${farthest.toFixed(2)} m from the attack place, beyond the unit's ${r} m (and the ${delta.toFixed(2)} m sampling margin)` });
  }

  // The size of the fire at t_s against the unit's capacity.
  const hrr = typeof hrrAt === "function" ? hrrAt(tS) : null;
  const area = typeof burningArea === "function" ? burningArea(tS) : Number.isFinite(burningArea) ? burningArea : null;
  let capacity = null;
  const sizeClass = fireClass === "C" ? cls.fuelClass || "A" : fireClass;
  const judge = (value, low, high, what, units) => {
    if (value == null) reasons.push({ id: "size", status: "cannot tell", text: `no ${what} given` });
    else if (value > high + 1e-9) reasons.push({ id: "too-big", status: "failure", text: `${what} ${value.toFixed(units === "kW" ? 0 : 2)} ${units} at t_s, above ${high.toFixed(units === "kW" ? 0 : 3)} ${units} even at the high end` });
    else if (value > low + 1e-9) reasons.push({ id: "size", status: "cannot tell", text: `${what} ${value.toFixed(units === "kW" ? 0 : 2)} ${units} at t_s, between ${low.toFixed(units === "kW" ? 0 : 3)} and ${high.toFixed(units === "kW" ? 0 : 3)} ${units}` });
  };
  if (sizeClass === "A") {
    const nA = ulA(u);
    const { qA, eta } = CAPACITY;
    capacity = { criterion: "hrr", units: "kW", nA, low: eta.low * qA.low * nA, central: eta.value * qA.value * nA, high: eta.high * qA.high * nA };
    judge(hrr, capacity.low, capacity.high, "heat release", "kW");
  } else if (sizeClass === "B") {
    const nB = u.rating.B || 0;
    capacity =
      u.rating.system === "EN"
        ? { criterion: "area", units: "m²", nB, low: CAPACITY.areaPerBEn.ordinary * nB, central: CAPACITY.areaPerBEn.ordinary * nB, high: CAPACITY.areaPerBEn.tray * nB }
        : { criterion: "area", units: "m²", nB, low: CAPACITY.areaPerB.ul * nB, central: CAPACITY.areaPerB.ul * nB, high: CAPACITY.areaPerB.ulExpert * nB };
    judge(area, capacity.low, capacity.high, "burning area", "m²");
  } else if (sizeClass === "K") {
    const item = itemArea ?? FIRE_CLASSES.K.item.area;
    capacity = { criterion: "item", units: "m²", low: item, central: item, high: item };
    judge(area, item, item, "burning area", "m²");
  } else {
    capacity = { criterion: "none", units: null, low: null, central: null, high: null };
    if (chosen.suit.suitable !== false) reasons.push({ id: "size", status: "cannot tell", text: `class ${fireClass}: no capacity model for a suitable unit` });
  }

  const outcome = reasons.some((x) => x.status === "failure") ? "failure" : reasons.some((x) => x.status === "cannot tell") ? "cannot tell" : "success";
  const out = exitTimes ? [exitTimes[attack], exitTimes[startNode]] : null;
  const timeLost = out && Number.isFinite(out[0]) && Number.isFinite(out[1]) ? tS + u.tD + out[0] - (tGo + out[1]) : null;
  return {
    ...common,
    outcome,
    reasons,
    attack: { node: attack, position: [P[attack * 3], P[attack * 3 + 1], P[attack * 3 + 2]], reach: r, standoff: Math.min(standoff, r) },
    times,
    hrr,
    burningArea: area,
    capacity,
    timeLost,
  };
}

/**
 * Cap the reached cells from t_s (note 6.3 step 7): each burning cell takes
 * evansFactor's cap with tau_x and t_act = t_s, so its reactions are at most
 * r_c(t_s) exp(-(t - t_s)/tau_x) from then on, and the heat it releases after
 * t_s at most r_c (tau_x + dt) (Suppression.lean evansCap_sum). Cells not
 * burning at t_s are left alone (powder and CO2 leave no protection), unlike
 * the sprinkler cap. With `rates` (kg/s per cell at t_s) a cell burns when its
 * rate exceeds `threshold`, and its rate is set; without, every listed cell is
 * taken as burning and armed, its rate read in the step after t_s, as a
 * sprinkler's. A cell already under a sprinkler's cap keeps whichever decays
 * faster, starting from no more than the sprinkler's cap at t_s, so the cap
 * never rises.
 *
 * @param {object} sup  a suppression record (suppression.js createSuppression): tau, tAct, rate
 * @param {Iterable<number>} cells  the fuel cells within reach of the attack place
 * @param {{tS: number, tau: number, rates?: ArrayLike<number>, threshold?: number}} opts
 * @returns {{capped: number[], kept: number[], idle: number}} capped: cells now
 *   under the extinguisher's cap (for a backend to copy); kept: under a
 *   faster sprinkler cap; idle: not burning
 */
export function extinguisherCap(sup, cells, { tS, tau, rates = null, threshold = 0 } = {}) {
  if (!(tau > 0) || !Number.isFinite(tS)) throw new RangeError("extinguisherCap needs tS and a positive tau (capFor).");
  const capped = [], kept = [];
  let idle = 0;
  for (const c of cells) {
    const rate = rates ? rates[c] : null;
    if (rates && !(rate > threshold)) {
      idle += 1;
      continue;
    }
    let start = rate ?? -1;
    if (sup.tau[c] > 0) {
      if (sup.tau[c] <= tau) {
        kept.push(c);
        continue;
      }
      if (sup.rate[c] >= 0) {
        const envelope = sup.rate[c] * Math.exp(-(tS - sup.tAct[c]) / sup.tau[c]);
        start = rate == null ? envelope : Math.min(rate, envelope);
      }
    }
    sup.tau[c] = tau;
    sup.tAct[c] = tS;
    sup.rate[c] = start;
    capped.push(c);
  }
  return { capped, kept, idle };
}
