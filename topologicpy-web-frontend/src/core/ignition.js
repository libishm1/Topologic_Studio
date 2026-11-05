/**
 * Likely ignition points: where in this model a fire most plausibly starts,
 * from the elements and rooms the IFC names, ranked by the ignition-source
 * review (wiki/research/ignition-sources-2026-11-05.md; catalogue in
 * ignitionCatalog.js).
 *
 * Matching, per element and scenario (the review's match confidences):
 *   1.0  the scenario's entity with its predefined type (IfcElectricAppliance
 *        ELECTRICCOOKER, IfcFurniture SOFA)
 *   0.9  the scenario's entity, named as the scenario's item
 *   0.7  any candidate element named as the item (exporters write cookers and
 *        fridges as IfcBuildingElementProxy or IfcFlowTerminal: the review's
 *        corpus check found no typed appliance in 45 models)
 *   0.4  no element, but a room named as the scenario's room (a kitchen)
 * times the entity entry's own weight. A name on the scenario's exclusion
 * list (a range hood is not a range) never matches.
 *
 * Ranking: score = w x confidence / N, w the scenario's share of fire deaths
 * (residential, US 2019-2023; non-residential uses that table's share, or
 * half the residential one, flagged), N the candidates matched to the same
 * scenario, so one sofa in a flat outranks one of forty desks.
 */

import { IGNITION_SCENARIOS } from "./ignitionCatalog.js";

export const CONFIDENCE = { typed: 1, entityAndName: 0.9, name: 0.7, room: 0.4 };
/** Weight for a scenario with no death share in the review (flagged in the result). */
export const UNKNOWN_WEIGHT = 0.005;

const compiled = new Map();
function pattern(source) {
  if (!source) return null;
  if (!compiled.has(source)) compiled.set(source, new RegExp(source, "i"));
  return compiled.get(source);
}

/** The ranking weight of a scenario for an occupancy, and whether it was estimated. */
export function scenarioWeight(s, occupancy = "residential") {
  if (occupancy === "non-residential") {
    if (s.nonResidentialShare != null) return { weight: s.nonResidentialShare, estimated: false };
    if (s.deathShare != null) return { weight: 0.5 * s.deathShare, estimated: true };
    return { weight: UNKNOWN_WEIGHT, estimated: true };
  }
  if (s.deathShare != null && s.deathShare > 0) return { weight: s.deathShare, estimated: false };
  return { weight: UNKNOWN_WEIGHT, estimated: true };
}

/** Confidence that one element is one scenario's ignition source (0 when not). */
export function matchElement(s, record) {
  const name = record.name || "";
  const type = (record.type || "").toUpperCase();
  const predefined = (record.predefined || "").toUpperCase();
  const exclude = pattern(s.nameExclude);
  if (exclude && name && exclude.test(name)) return 0;
  const named = Boolean(name && pattern(s.nameRegex)?.test(name));
  let best = 0;
  for (const e of s.entities) {
    if (e.entity !== type) continue;
    if (e.predefinedType && predefined === e.predefinedType) best = Math.max(best, CONFIDENCE.typed * e.weight);
    else if (named) best = Math.max(best, CONFIDENCE.entityAndName * e.weight);
  }
  if (!best && named) best = CONFIDENCE.name;
  return best;
}

/**
 * Rank candidate ignition points.
 *
 * @param {Array} records  elements: {itemId, type (IFC class), name, predefined, position: [x, y, z]}
 * @param {Array} rooms    IfcSpace rooms: {itemId, name, position}
 * @param {object} opts    occupancy: "residential" | "non-residential"; limit; scenarios (for tests)
 * @returns {Array} [{scenario, itemId, name, position, confidence, weight, estimated, score, byRoom}]
 */
export function rankIgnitionPoints(records, rooms = [], { occupancy = "residential", limit = 8, scenarios = IGNITION_SCENARIOS, skip = ["intentional-accelerant"] } = {}) {
  const found = [];
  for (const s of scenarios) {
    if (skip.includes(s.id)) continue;
    const matched = [];
    for (const record of records) {
      const confidence = matchElement(s, record);
      if (confidence > 0) matched.push({ record, confidence, byRoom: false });
    }
    // No element: a room named for the scenario stands in for it.
    if (!matched.length && s.spaceRegex) {
      const space = pattern(s.spaceRegex);
      for (const room of rooms) {
        if (room.name && space.test(room.name)) matched.push({ record: room, confidence: CONFIDENCE.room, byRoom: true });
      }
    }
    if (!matched.length) continue;
    const { weight, estimated } = scenarioWeight(s, occupancy);
    for (const m of matched) {
      found.push({
        scenario: s,
        itemId: m.record.itemId,
        name: m.record.name || "",
        position: m.record.position,
        confidence: m.confidence,
        weight,
        estimated,
        byRoom: m.byRoom,
        score: (weight * m.confidence) / matched.length,
      });
    }
  }
  found.sort((a, b) => b.score - a.score || b.confidence - a.confidence);
  // One entry per element: its best scenario.
  const seen = new Set();
  const out = [];
  for (const c of found) {
    const key = `${c.byRoom ? "room" : "item"}:${c.itemId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(c);
    if (out.length >= limit) break;
  }
  return out;
}

/** The IFC classes worth reading for candidates (the union of the catalogue's entities). */
export const CANDIDATE_CLASSES = [...new Set(IGNITION_SCENARIOS.flatMap((s) => s.entities.map((e) => e.entity)))];

/** Centroid of a set of meshes (positions in the viewer frame), or null. */
export function centroidOf(meshes) {
  let x = 0, y = 0, z = 0, n = 0;
  for (const mesh of meshes || []) {
    const p = mesh.positions;
    for (let v = 0; v < p.length; v += 3) {
      x += p[v];
      y += p[v + 1];
      z += p[v + 2];
      n += 1;
    }
  }
  return n ? [x / n, y / n, z / n] : null;
}
