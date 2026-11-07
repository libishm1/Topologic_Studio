/**
 * Likely ignition points (ignition.js) against the review's catalogue: names
 * as exporters write them, typed entities, exclusions, rooms standing in for
 * missing elements, and the ranking rule.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { CONFIDENCE, IGNITION_CONTEXTS, matchElement, rankIgnitionPoints, scenariosFor, scenarioWeight, centroidOf } from "./ignition.js";
import { IGNITION_SCENARIOS } from "./ignitionCatalog.js";
import { FIRE_CLASSES } from "./fireClasses.js";

const byId = (id) => IGNITION_SCENARIOS.find((s) => s.id === id);
const at = [0, 0, 0];

// The review lists IfcBuildingElementProxy by name for cooking, so a named
// proxy is its entity and name (0.9), not a name on any element (0.7).
test("a Revit cooktop exported as a proxy matches cooking, by entity and name", () => {
  const record = { itemId: 1, type: "IFCBUILDINGELEMENTPROXY", name: "M_Cook Top-4 Unit:Standard:123", predefined: "", position: at };
  assert.equal(matchElement(byId("cooking-range"), record), CONFIDENCE.entityAndName);
  const [top] = rankIgnitionPoints([record]);
  assert.equal(top.scenario.id, "cooking-range");
  assert.equal(top.scenario.fireClass, "K");
});

test("a range hood is not a range", () => {
  const record = { itemId: 2, type: "IFCBUILDINGELEMENTPROXY", name: "Range Hood 900", predefined: "", position: at };
  assert.equal(matchElement(byId("cooking-range"), record), 0);
});

test("a typed sofa matches upholstery at full confidence; a dining chair does not", () => {
  const sofa = { itemId: 3, type: "IFCFURNITURE", name: "Sofa 3 seat", predefined: "SOFA", position: at };
  assert.equal(matchElement(byId("smoking-upholstery"), sofa), CONFIDENCE.typed);
  const chair = { itemId: 4, type: "IFCFURNITURE", name: "Dining Chair", predefined: "CHAIR", position: at };
  assert.equal(matchElement(byId("smoking-upholstery"), chair), 0);
});

test("a kitchen stands in when no cooking appliance is modelled", () => {
  const rooms = [{ itemId: 9, name: "Kitchen", position: [1, 1, 1] }];
  const ranked = rankIgnitionPoints([], rooms);
  const cooking = ranked.find((c) => c.scenario.id === "cooking-range");
  assert.ok(cooking && cooking.byRoom && cooking.confidence === CONFIDENCE.room);
});

test("the ranking divides a scenario's weight among its candidates", () => {
  const desks = Array.from({ length: 4 }, (_, i) => ({ itemId: 100 + i, type: "IFCFURNITURE", name: "Desk", predefined: "DESK", position: at }));
  const sofa = { itemId: 200, type: "IFCFURNITURE", name: "Sofa", predefined: "SOFA", position: at };
  const ranked = rankIgnitionPoints([...desks, sofa], [], { limit: 10 });
  const sofaEntry = ranked.find((c) => c.itemId === 200);
  const s = byId("smoking-upholstery");
  assert.ok(Math.abs(sofaEntry.score - scenarioWeight(s).weight * CONFIDENCE.typed) < 1e-12);
  const desk = ranked.find((c) => c.itemId === 100);
  if (desk) assert.ok(desk.score <= desk.weight * desk.confidence / 4 + 1e-12);
  // Scores fall down the list.
  for (let i = 1; i < ranked.length; i += 1) assert.ok(ranked[i - 1].score >= ranked[i].score);
});

test("non-residential weights fall back to half the residential share, flagged", () => {
  const s = byId("smoking-upholstery");
  const w = scenarioWeight(s, "non-residential");
  if (s.nonResidentialShare == null && s.deathShare != null) {
    assert.equal(w.estimated, true);
    assert.ok(Math.abs(w.weight - 0.5 * s.deathShare) < 1e-12);
  }
});

test("every scenario's fire class is one the solver offers, and its patterns compile", () => {
  for (const s of IGNITION_SCENARIOS) {
    assert.ok(FIRE_CLASSES[s.fireClass], `${s.id}: class ${s.fireClass}`);
    for (const r of [s.nameRegex, s.nameExclude, s.spaceRegex]) if (r) new RegExp(r, "i");
  }
  // Lithium-ion scenarios are class L (ISO 3941:2026), not C.
  const lithium = IGNITION_SCENARIOS.filter((s) => /^(li-ion|battery-storage)/.test(s.id));
  assert.ok(lithium.length > 0);
  for (const s of lithium) assert.equal(s.fireClass, "L", s.id);
  assert.deepEqual(centroidOf([{ positions: Float32Array.from([0, 0, 0, 2, 4, 6]) }]), [1, 2, 3]);
  assert.equal(centroidOf([]), null);
});

test("each cause names real scenarios, and every scenario has a cause", () => {
  const ids = new Set(IGNITION_SCENARIOS.map((s) => s.id));
  const covered = new Set();
  for (const c of IGNITION_CONTEXTS) for (const id of c.scenarios || []) {
    assert.ok(ids.has(id), `${c.id}: ${id}`);
    covered.add(id);
  }
  for (const id of ids) assert.ok(covered.has(id), `no cause lists ${id}`);
  assert.equal(scenariosFor("any").length, IGNITION_SCENARIOS.length);
});

test("a gas fire at the cooktop starts as class B; the deliberate cause brings in the accelerant", () => {
  const cooktop = { itemId: 1, type: "IFCBUILDINGELEMENTPROXY", name: "M_Cook Top-4 Unit:Standard:123", predefined: "", position: at };
  const [gas] = rankIgnitionPoints([cooktop], [], { context: "gas" });
  assert.equal(gas.scenario.id, "cooking-range");
  assert.equal(gas.scenario.fireClass, "B");
  assert.equal(gas.scenario.label, "Gas hob or oven");
  // The catalogue itself is untouched.
  assert.equal(byId("cooking-range").fireClass, "K");
  assert.deepEqual(rankIgnitionPoints([cooktop], [], { context: "electrical" }), []);
  const lobby = { itemId: 9, name: "Eingang", position: at };
  assert.equal(rankIgnitionPoints([], [lobby]).length, 0, "left out of the ranked list");
  const [arson] = rankIgnitionPoints([], [lobby], { context: "deliberate" });
  assert.equal(arson.scenario.id, "intentional-accelerant");
  assert.equal(arson.byRoom, true);
});
