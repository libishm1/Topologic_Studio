// Run with: npm test
// ifcSemantics against a mock of web-ifc's line API (review m12).
import assert from "node:assert/strict";
import { test } from "node:test";

import { ifcSemantics } from "./ifcSemantics.js";

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
