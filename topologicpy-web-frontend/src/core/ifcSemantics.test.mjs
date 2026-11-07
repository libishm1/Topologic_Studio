// Run with: npm test
// ifcSemantics against a mock of web-ifc's line API (review m12).
import assert from "node:assert/strict";
import { test } from "node:test";

import { exitDoors, exitsPerPart, ifcSemantics } from "./ifcSemantics.js";

test("parts no flagged exit reaches get the one-sided doors in them, small pockets do not", () => {
  // Three chains: A (nodes 0..59) and B (60..119) of 60 nodes, C (120..124) of 5.
  const n = 125;
  const edges = [];
  for (const [lo, hi] of [[0, 59], [60, 119], [120, 124]]) for (let i = lo; i < hi; i += 1) edges.push(i, i + 1);
  const graph = { nodes: new Float32Array(n * 3), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
  assert.deepEqual(exitsPerPart(graph, [10], [20, 70, 122]), [70]);
  assert.deepEqual(exitsPerPart(null, [10], [20, 70, 122]), [], "without the graph nothing is added");
  // Through exitDoors: door 1 is flagged (in A); doors 2 and 3 are one-sided, in A and B.
  const doors = [
    { itemId: 1, node: 10, sides: [2, 2] },
    { itemId: 2, node: 20, sides: [0, 3] },
    { itemId: 3, node: 70, sides: [3, 0] },
  ];
  const semantics = { doors: [{ id: 1, exit: "Pset_DoorCommon.FireExit" }] };
  assert.deepEqual(exitDoors(doors, semantics).nodes, [10]);
  const withGraph = exitDoors(doors, semantics, graph);
  assert.deepEqual(withGraph.nodes, [10, 70]);
  assert.match(withGraph.source, /parts with no flagged exit/);
});

test("the final exits are street-level external doors and fire escapes to the open air, not flat doors or balconies", () => {
  // A stair 0..9 rising 1 m a node (y up), and a separate part of 60 nodes
  // at street level whose only exit is a FireExit door.
  const nodes = [], edges = [];
  for (let i = 0; i < 10; i += 1) nodes.push(0, i, 0);
  for (let i = 0; i < 60; i += 1) nodes.push(20 + i, 0, 0);
  for (let i = 0; i < 9; i += 1) edges.push(i, i + 1);
  for (let i = 10; i < 69; i += 1) edges.push(i, i + 1);
  const graph = { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
  const doors = [
    { itemId: 1, node: 0, sides: [3, 0] }, // the street door
    { itemId: 2, node: 1, sides: [3, 0] }, // a side door, 1 m up the slope
    { itemId: 3, node: 5, sides: [3, 3] }, // a balcony door, flagged FireExit
    { itemId: 6, node: 6, sides: [3, 0] }, // a fire escape to the open air, 6 m up
    { itemId: 7, node: 7, sides: [3, 0] }, // an external door to nowhere, not a fire exit
    { itemId: 4, node: 9, sides: [3, 3] }, // a flat's front door, flagged
    { itemId: 5, node: 40, sides: [3, 3] }, // the other part's only exit
  ];
  const semantics = {
    doors: [
      { id: 1, exit: "Pset_DoorCommon.FireExit", external: true },
      { id: 2, exit: "Pset_DoorCommon.IsExternal", external: true },
      { id: 3, exit: "Pset_DoorCommon.FireExit", external: true },
      { id: 6, exit: "Pset_DoorCommon.FireExit", external: true },
      { id: 7, exit: "IfcRelSpaceBoundary.EXTERNAL", external: true },
      { id: 4, exit: "Pset_DoorCommon.FireExit", external: false },
      { id: 5, exit: "Pset_DoorCommon.FireExit", external: false },
    ],
  };
  const exits = exitDoors(doors, semantics, graph);
  assert.deepEqual(exits.nodes, [0, 1, 5, 6, 7, 9, 40]);
  assert.deepEqual(exits.final, [0, 1, 6, 40]);
  assert.match(exits.finalSource, /street level/);
  // Without the graph, or without external doors, they are the exits.
  assert.deepEqual(exitDoors(doors, semantics).final, [0, 1, 5, 6, 7, 9, 40]);
  const flagsOnly = { doors: semantics.doors.map((d) => ({ ...d, external: false })) };
  assert.deepEqual(exitDoors(doors, flagsOnly, graph).final, [0, 1, 5, 6, 7, 9, 40]);
  assert.equal(exitDoors(doors, flagsOnly, graph).finalSource, null);
});

const W = { IFCRELDEFINESBYPROPERTIES: 1, IFCRELDEFINESBYTYPE: 2, IFCRELSPACEBOUNDARY: 3, IFCDOOR: 4, IFCSLAB: 5, IFCUNITASSIGNMENT: 6, IFCBUILDINGSTOREY: 7 };
const ref = (id) => ({ type: 5, value: id });
const E = (v) => ({ type: 3, value: v });

function mockApi(lines, byType) {
  return {
    GetLineIDsWithType: (_m, t) => {
      const a = byType[t] || [];
      return { size: () => a.length, get: (i) => a[i] };
    },
    GetLine: (_m, id) => lines[id],
    GetModelSchema: () => "IFC4",
  };
}

test("a door's own unset Pset value does not hide its type's value", () => {
  for (const unset of [null, { type: 1, value: null }]) {
    const lines = {
      10: { Name: { value: "D1" } },
      20: { RelatedObjects: [ref(10)], RelatingPropertyDefinition: ref(21) },
      21: { Name: { value: "Pset_DoorCommon" }, HasProperties: [ref(22)] },
      22: { Name: { value: "FireExit" }, NominalValue: unset },
      30: { RelatedObjects: [ref(10)], RelatingType: ref(31) },
      31: { HasPropertySets: [ref(32)] },
      32: { Name: { value: "Pset_DoorCommon" }, HasProperties: [ref(33)] },
      33: { Name: { value: "FireExit" }, NominalValue: { type: 1, value: "T" } },
    };
    const s = ifcSemantics(mockApi(lines, { 1: [20], 2: [30], 4: [10] }), 0, W);
    assert.equal(s.doors[0].exit, "Pset_DoorCommon.FireExit");
  }
});

test("a door's own FALSE still overrides its type's TRUE", () => {
  const lines = {
    10: { Name: { value: "D1" } },
    20: { RelatedObjects: [ref(10)], RelatingPropertyDefinition: ref(21) },
    21: { Name: { value: "Pset_DoorCommon" }, HasProperties: [ref(22)] },
    22: { Name: { value: "FireExit" }, NominalValue: { type: 1, value: "F" } },
    30: { RelatedObjects: [ref(10)], RelatingType: ref(31) },
    31: { HasPropertySets: [ref(32)] },
    32: { Name: { value: "Pset_DoorCommon" }, HasProperties: [ref(33)] },
    33: { Name: { value: "FireExit" }, NominalValue: { type: 1, value: "T" } },
  };
  const s = ifcSemantics(mockApi(lines, { 1: [20], 2: [30], 4: [10] }), 0, W);
  assert.equal(s.doors[0].exit, null);
});

test("a door flagged FireExit still says it is external, from IsExternal or an EXTERNAL boundary", () => {
  const lines = {
    10: { Name: { value: "Street" } },
    11: { Name: { value: "Garden" } },
    12: { Name: { value: "Flat" } },
    20: { RelatedObjects: [ref(10), ref(12)], RelatingPropertyDefinition: ref(21) },
    21: { Name: { value: "Pset_DoorCommon" }, HasProperties: [ref(22), ref(23)] },
    22: { Name: { value: "FireExit" }, NominalValue: { type: 1, value: "T" } },
    23: { Name: { value: "IsExternal" }, NominalValue: { type: 1, value: "F" } },
    40: { RelatedBuildingElement: ref(11), InternalOrExternalBoundary: E("EXTERNAL") },
    41: { RelatedBuildingElement: ref(10), InternalOrExternalBoundary: E("EXTERNAL") },
  };
  const s = ifcSemantics(mockApi(lines, { 1: [20], 3: [40, 41], 4: [10, 11, 12] }), 0, W);
  const byName = Object.fromEntries(s.doors.map((d) => [d.name, d]));
  assert.equal(byName.Street.exit, "Pset_DoorCommon.FireExit");
  assert.equal(byName.Street.external, true, "its boundary says so, whatever its IsExternal");
  assert.equal(byName.Garden.exit, "IfcRelSpaceBoundary.EXTERNAL");
  assert.equal(byName.Garden.external, true);
  assert.equal(byName.Flat.exit, "Pset_DoorCommon.FireExit");
  assert.equal(byName.Flat.external, false);
});

// Sprinklers (wiki/research/sprinkler-ifc-mep-2026-11-07.md, sections 5.3, 5.5, 5.9).
const WS = { ...W, IFCFLOWTERMINAL: 8, IFCFIRESUPPRESSIONTERMINAL: 9, IFCFIRESUPPRESSIONTERMINALTYPE: 10 };
const AIR_TERMINAL_TYPE = 11;
const real = (v) => ({ type: 4, name: "IFCREAL", value: v });
const int = (v) => ({ type: 10, name: "IFCINTEGER", value: v });
const measure = (name, v) => ({ type: 4, name, value: v });
const label = (v) => ({ type: 1, name: "IFCLABEL", value: v });
const prop = (name, nominal) => ({ Name: label(name), NominalValue: nominal });

test("sprinklers: Revit's NOTDEFINED heads with Revit units; hose reels, air terminals and lights left out", () => {
  const lines = {
    // IFC2x3 NIBS pattern: IfcFlowTerminal typed by a NOTDEFINED IfcFireSuppressionTerminalType,
    // the Revit sets on the occurrence (Temperature Rating in kelvin, Flow in ft³/s, Pressure Drop in kg/(ft s²)).
    100: { type: 8, Name: label("M_Sprinkler - Pendent - Hosted:15 mm Pendent:15 mm Pendent:623904"), ObjectType: label("15 mm Pendent") },
    200: { type: 10, Name: label("15 mm Pendent"), ElementType: label("15 mm Pendent"), PredefinedType: E("NOTDEFINED"), HasPropertySets: null },
    300: { Name: label("PSet_Revit_Type_Fire Protection"), HasProperties: [ref(301), ref(302), ref(303), ref(304)] },
    301: prop("Temperature Rating", real(330.15)),
    302: prop("K-Factor", real(81)),
    303: prop("Response", int(0)),
    304: prop("Coverage", int(0)),
    310: { Name: label("PSet_Revit_Fire Protection"), HasProperties: [ref(311), ref(312)] },
    311: prop("Flow", real(0.07062933344297717)),
    312: prop("Pressure Drop", real(14721.84)),
    320: { Name: label("PSet_Revit_Other"), HasProperties: [ref(321)] },
    321: prop("K-Factor", real(5)),
    // A hose reel by its type's enumeration, one by name, and a Revit-hosted head whose name says "Hosted".
    101: { type: 8, Name: label("Hose reel 1") },
    201: { type: 10, Name: label("Reel"), PredefinedType: E("HOSEREEL") },
    102: { type: 8, Name: label("Cabinet") },
    202: { type: 10, Name: label("Sprinkler and Hose Reel cabinet"), PredefinedType: E("NOTDEFINED") },
    // An air terminal named like a sprinkler, an untyped pendant light, an untyped head.
    103: { type: 8, Name: label("Sprinkler-looking diffuser") },
    203: { type: AIR_TERMINAL_TYPE, Name: label("Pendent diffuser") },
    104: { type: 8, Name: label("Pendant light") },
    105: { type: 8, Name: label("Sprinkler head"), ObjectType: label("Upright") },
    // Relations.
    400: { RelatedObjects: [ref(100)], RelatingType: ref(200) },
    401: { RelatedObjects: [ref(101)], RelatingType: ref(201) },
    402: { RelatedObjects: [ref(102)], RelatingType: ref(202) },
    403: { RelatedObjects: [ref(103)], RelatingType: ref(203) },
    500: { RelatedObjects: [ref(100)], RelatingPropertyDefinition: ref(300) },
    501: { RelatedObjects: [ref(100)], RelatingPropertyDefinition: ref(310) },
    502: { RelatedObjects: [ref(100)], RelatingPropertyDefinition: ref(320) },
  };
  const s = ifcSemantics(mockApi(lines, { 1: [500, 501, 502], 2: [400, 401, 402, 403], 8: [100, 101, 102, 103, 104, 105] }), 0, WS);
  assert.deepEqual(s.sprinklers.map((h) => h.id), [100, 105]);
  const [nibs, untyped] = s.sprinklers;
  assert.deepEqual(Object.keys(nibs).sort(), ["activationC", "coverage", "flowLpm", "id", "kFactor", "name", "pressureBar", "response", "typeName"]);
  assert.equal(nibs.typeName, "15 mm Pendent");
  assert.ok(Math.abs(nibs.activationC - 57) < 1e-9, "330.15 K is 57 °C");
  assert.equal(nibs.kFactor, 81, "from the fire protection set, not PSet_Revit_Other");
  assert.equal(nibs.response, null, "Revit's Response integer code is not documented");
  assert.equal(nibs.coverage, null, "Revit's Coverage integer is an option code, not an area");
  assert.ok(Math.abs(nibs.flowLpm - 120) < 1e-6, "0.0706 ft³/s is 120 L/min");
  assert.ok(Math.abs(nibs.pressureBar - 0.483) < 1e-6, "14721.84 kg/(ft s²) is 0.483 bar (7 psi)");
  assert.equal(untyped.name, "Sprinkler head");
  assert.deepEqual([untyped.activationC, untyped.kFactor, untyped.flowLpm], [null, null, null]);
});

test("sprinklers: IFC4 IfcFireSuppressionTerminal, the standard Pset in project units, the occurrence over its type", () => {
  const lines = {
    // web-ifc's inherited query lists an IfcFireSuppressionTerminal among the flow terminals too.
    106: { type: 9, Name: label("Upright 1"), PredefinedType: null },
    204: { type: 10, Name: label("Upright"), PredefinedType: E("SPRINKLER"), HasPropertySets: [ref(330)] },
    330: { Name: label("Pset_FireSuppressionTerminalTypeSprinkler"), HasProperties: [ref(331), ref(332), ref(333), ref(334), ref(335), ref(336)] },
    331: prop("ActivationTemperature", measure("IFCTHERMODYNAMICTEMPERATUREMEASURE", 341.15)),
    332: prop("DischargeCoefficient", real(5.6)),
    333: { Name: label("Response"), EnumerationValues: [E("QUICK")] },
    334: prop("CoverageArea", measure("IFCAREAMEASURE", 12)),
    335: prop("DischargeFlowRate", measure("IFCVOLUMETRICFLOWRATEMEASURE", 0.001)),
    336: prop("ResidualFlowingPressure", measure("IFCPRESSUREMEASURE", 50000)),
    340: { Name: label("Pset_FireSuppressionTerminalTypeSprinkler"), HasProperties: [ref(341)] },
    341: prop("ActivationTemperature", measure("IFCTHERMODYNAMICTEMPERATUREMEASURE", 366.15)),
    // Untyped, NOTDEFINED, named as a sidewall head, with only a bulb colour.
    107: { type: 9, Name: label("Sidewall"), PredefinedType: E("NOTDEFINED") },
    350: { Name: label("Pset_FireSuppressionTerminalTypeSprinkler"), HasProperties: [ref(351)] },
    351: { Name: label("BulbLiquidColour"), EnumerationValues: [E("YELLOW")] },
    // Untyped, a breeching inlet.
    108: { type: 9, Name: label("Inlet"), PredefinedType: E("BREECHINGINLET") },
    400: { RelatedObjects: [ref(106)], RelatingType: ref(204) },
    500: { RelatedObjects: [ref(106)], RelatingPropertyDefinition: ref(340) },
    501: { RelatedObjects: [ref(107)], RelatingPropertyDefinition: ref(350) },
  };
  const s = ifcSemantics(mockApi(lines, { 1: [500, 501], 2: [400], 8: [106, 107, 108], 9: [106, 107, 108] }), 0, WS);
  assert.deepEqual(s.sprinklers.map((h) => h.id), [106, 107], "each head once, the breeching inlet left out");
  const [upright, sidewall] = s.sprinklers;
  assert.ok(Math.abs(upright.activationC - 93) < 1e-9, "the occurrence's 366.15 K overrides the type's 341.15 K");
  assert.ok(Math.abs(upright.kFactor - 5.6 * 14.416) < 0.01, "K5.6 in gpm/psi^1/2 read as about K80.7");
  assert.equal(upright.response, "quick");
  assert.equal(upright.coverage, 12);
  assert.ok(Math.abs(upright.flowLpm - 60) < 1e-9, "0.001 m³/s is 60 L/min");
  assert.ok(Math.abs(upright.pressureBar - 0.5) < 1e-9);
  assert.equal(sidewall.activationC, 79, "a yellow bulb is 79 °C");
  assert.equal(sidewall.typeName, null);
});

test("sprinklers: the project's units, and none where a model has no fire suppression terminals", () => {
  const lines = {
    1: { Units: [ref(2), ref(3), ref(5)] },
    2: { UnitType: E("THERMODYNAMICTEMPERATUREUNIT"), Name: E("DEGREE_CELSIUS"), Prefix: null },
    3: { UnitType: E("AREAUNIT"), Name: label("SQUARE FOOT"), ConversionFactor: ref(4) },
    4: { ValueComponent: { type: 4, value: 0.09290304 }, UnitComponent: ref(6) },
    5: { UnitType: E("LENGTHUNIT"), Name: E("METRE"), Prefix: E("MILLI") },
    6: { UnitType: E("AREAUNIT"), Name: E("SQUARE_METRE"), Prefix: null },
    106: { type: 9, Name: label("Pendent"), PredefinedType: E("SPRINKLER") },
    500: { RelatedObjects: [ref(106)], RelatingPropertyDefinition: ref(330) },
    330: { Name: label("Pset_FireSuppressionTerminalTypeSprinkler"), HasProperties: [ref(331), ref(332)] },
    331: prop("ActivationTemperature", measure("IFCTHERMODYNAMICTEMPERATUREMEASURE", 68)),
    332: prop("CoverageArea", measure("IFCAREAMEASURE", 130)),
  };
  const s = ifcSemantics(mockApi(lines, { 1: [500], 6: [1], 9: [106] }), 0, WS);
  assert.equal(s.sprinklers[0].activationC, 68, "a project in degrees Celsius");
  assert.ok(Math.abs(s.sprinklers[0].coverage - 130 * 0.09290304) < 1e-9, "130 sq ft");
  assert.ok(Math.abs(s.lengthScale - 0.001) < 1e-12, "the length unit is read as before");
  // A model without the classes, as the tests above.
  assert.deepEqual(ifcSemantics(mockApi({}, {}), 0, W).sprinklers, []);
});

test("storey heights follow nested conversion-based units (an inch defined through the foot)", () => {
  const storeyAt = (z, units) => ({
    ...units,
    7: { Name: { value: "L1" }, ObjectPlacement: ref(8) },
    8: { RelativePlacement: ref(9) },
    9: { Location: ref(11) },
    11: { Coordinates: [{ value: 0 }, { value: 0 }, { value: z }] },
  });
  const metre = { 6: { UnitType: E("LENGTHUNIT"), Name: E("METRE"), Prefix: null } };
  const foot = { 4: { UnitType: E("LENGTHUNIT"), Name: { value: "FOOT" }, ConversionFactor: ref(5) }, 5: { ValueComponent: { type: 4, value: 0.3048 }, UnitComponent: ref(6) } };
  const inch = { 2: { UnitType: E("LENGTHUNIT"), Name: { value: "INCH" }, ConversionFactor: ref(3) }, 3: { ValueComponent: { type: 4, value: 1 / 12 }, UnitComponent: ref(4) } };
  const elevation = (lines) => ifcSemantics(mockApi(lines, { 6: [1], 7: [7] }), 0, W).storeys[0].elevation;
  assert.ok(Math.abs(elevation(storeyAt(120, { 1: { Units: [ref(2)] }, ...inch, ...foot, ...metre })) - 3.048) < 1e-9);
  assert.ok(Math.abs(elevation(storeyAt(10, { 1: { Units: [ref(4)] }, ...foot, ...metre })) - 3.048) < 1e-9);
  const mm = { 1: { Units: [ref(6)] }, 6: { UnitType: E("LENGTHUNIT"), Name: E("METRE"), Prefix: E("MILLI") } };
  assert.ok(Math.abs(elevation(storeyAt(3048, mm)) - 3.048) < 1e-9);
});
