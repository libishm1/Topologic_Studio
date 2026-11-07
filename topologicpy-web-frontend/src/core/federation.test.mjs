import { test } from "node:test";
import assert from "node:assert/strict";

import { IDENTITY, alignment, apply, federationCheck, invertAffine, linkMatrix, mapMatrix, matchStoreys, multiply, planOverlap } from "./federation.js";

const near = (a, b, tol = 1e-6) => a.every((v, k) => Math.abs(v - b[k]) <= tol);
const translate = (x, y, z) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
const turnZ = (deg) => {
  const c = Math.cos((deg * Math.PI) / 180), s = Math.sin((deg * Math.PI) / 180);
  return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
};

test("an affine matrix times its inverse is the identity", () => {
  const m = multiply(translate(3, -2, 7), turnZ(30));
  assert.ok(near(multiply(m, invertAffine(m)), IDENTITY));
  assert.ok(near(apply(invertAffine(m), apply(m, [1, 2, 3])), [1, 2, 3]));
});

// The buildingSMART Simple-Scene files (MEP report section 6.1): the building
// and the plumbing have different map conversions, both in millimetres.
const ARCH_MAP = { eastings: 729013.3488297004, northings: 9063992.684697364, height: 1.3, xAxisAbscissa: 0.5, xAxisOrdinate: 0.8660254037844387, scale: 0.001 };
const PLUMBING_MAP = { eastings: 729011.2258823584, northings: 9063960.607644705, height: 0, xAxisAbscissa: 1, xAxisOrdinate: 0, scale: 0.001 };

test("a map conversion places the origin at its eastings and turns by its x axis", () => {
  const m = mapMatrix(ARCH_MAP, 0.001);
  assert.ok(near(apply(m, [0, 0, 0]), [ARCH_MAP.eastings, ARCH_MAP.northings, 1.3]));
  // One metre along the file's x goes 60 degrees east of north... of east.
  const p = apply(m, [1, 0, 0]);
  assert.ok(near([p[0] - ARCH_MAP.eastings, p[1] - ARCH_MAP.northings], [0.5, 0.8660254037844387]));
});

test("two map conversions align the second model through the CRS", () => {
  const { how, relative } = alignment({ frame: { map: ARCH_MAP }, lengthScale: 0.001 }, { frame: { map: PLUMBING_MAP }, lengthScale: 0.001 });
  assert.equal(how, "map conversions");
  // The building's origin, written in the plumbing's local frame, lands on the building's origin.
  const inPlumbing = [ARCH_MAP.eastings - PLUMBING_MAP.eastings, ARCH_MAP.northings - PLUMBING_MAP.northings, 1.3];
  assert.ok(near(apply(relative, inPlumbing), [0, 0, 0], 1e-6));
  // Without both, the files are taken as they are.
  assert.deepEqual(alignment({ frame: { map: ARCH_MAP } }, { frame: { map: null } }).relative, [...IDENTITY]);
});

test("the link matrix is That Open's coordination for a translation, and the identity for one frame", () => {
  assert.ok(near(linkMatrix(translate(1, 2, 3), translate(1, 2, 3)), IDENTITY));
  assert.ok(near(linkMatrix(translate(10, 0, -4), translate(4, 5, 6)), translate(6, -5, -10)));
  // An IFC-frame shift of 2 m along x moves the second model 2 m along the viewer's x.
  const m = linkMatrix(IDENTITY, IDENTITY, translate(2, 0, 0));
  assert.ok(near(apply(m, [0, 0, 0]), [2, 0, 0]));
  // And 1 m up the IFC's z is 1 m up the viewer's y.
  assert.ok(near(apply(linkMatrix(IDENTITY, IDENTITY, translate(0, 0, 1)), [0, 0, 0]), [0, 1, 0]));
});

test("the Simple-Scene plumbing lands where its map conversion says, in the building's viewer frame", () => {
  // The viewer draws a model at web-ifc's world frame (x, z, -y) plus its
  // coordination translation (measured: the building's walls and the HVAC
  // model's duct both sit exactly so). These are the two files' own.
  const mainC = translate(-6.66, 0.55, 3.8), plumbingC = translate(-25.77, 0.02, 35.26);
  const { relative } = alignment({ frame: { map: ARCH_MAP }, lengthScale: 0.001 }, { frame: { map: PLUMBING_MAP }, lengthScale: 0.001 });
  const link = linkMatrix(mainC, plumbingC, relative);
  // The plumbing's origin marker, centred at web-ifc (-0.04, 0.05, 0).
  const world = [-0.04, 0.05, 0];
  const drawn = [world[0] - 25.77, world[1] + 0.02, world[2] + 35.26];
  // By hand: IFC (x, -z, y) to the CRS by the plumbing's conversion, back by the building's.
  const [X, Y, Z] = [world[0], -world[2], world[1]];
  const dE = PLUMBING_MAP.eastings + X - ARCH_MAP.eastings, dN = PLUMBING_MAP.northings + Y - ARCH_MAP.northings;
  const a = ARCH_MAP.xAxisAbscissa, b = ARCH_MAP.xAxisOrdinate;
  const [x, y, z] = [a * dE + b * dN, -b * dE + a * dN, Z - ARCH_MAP.height];
  assert.ok(near(apply(link, drawn), [x - 6.66, z + 0.55, -y + 3.8], 1e-6));
});

test("storeys match by elevation, and metres written in a millimetre file are read as metres", () => {
  const main = [
    { name: "Level 1", elevation: 0 },
    { name: "Level 2", elevation: 4.27 },
    { name: "Roof", elevation: 7.92 },
  ];
  // The NIBS MEP export: no storey placements, Elevation in metres, unit the millimetre.
  const mep = [
    { name: "Level 1", elevation: 0, declared: 0 },
    { name: "Roof", elevation: 0, declared: 7.924820899963379 * 0.001 },
    { name: "Level 2", elevation: 0, declared: 4.267220973968506 * 0.001 },
  ];
  const { pairs, rescaled } = matchStoreys(main, mep, { lengthScale: 0.001 });
  assert.equal(rescaled, true);
  assert.deepEqual(pairs.map((p) => [p.linked, p.main, p.matched]), [["Level 1", "Level 1", true], ["Roof", "Roof", true], ["Level 2", "Level 2", true]]);
  // Names may differ while elevations agree (the NIBS Clinic).
  const renamed = matchStoreys(main, [{ name: "First Floor", elevation: 0.02 }]);
  assert.deepEqual([renamed.pairs[0].main, renamed.pairs[0].matched], ["Level 1", true]);
  assert.equal(matchStoreys(main, [{ name: "Mezzanine", elevation: 2 }]).pairs[0].matched, false);
});

test("the plan overlap is the share of the smaller plan the two have in common", () => {
  const a = { min: [0, 0, 0], max: [10, 6, 10] };
  assert.equal(planOverlap(a, { min: [0, 0, 0], max: [10, 3, 10] }).share, 1);
  const half = planOverlap(a, { min: [5, 0, 0], max: [15, 3, 10] });
  assert.equal(half.share, 0.5);
  assert.deepEqual(half.offset, [-5, 0, 0]);
  // A site-wide model holding the building overlaps it fully (the Simple-Scene HVAC).
  assert.equal(planOverlap(a, { min: [-30, 0, -10], max: [20, 5, 20] }).share, 1);
});

test("the check says aligned for a matching pair, and offset when the plans miss", () => {
  const storeys = [{ name: "L1", elevation: 0 }, { name: "L2", elevation: 3.5 }];
  const site = { origin: [0, 0, 0], xAxis: [1, 0, 0], zAxis: [0, 0, 1] };
  const main = { name: "Arch.ifc", semantics: { lengthScale: 1, storeys, frame: { site, trueNorth: [0, 1], map: null } }, box: { min: [0, 0, 0], max: [20, 7, 12] } };
  const mep = { name: "MEP.ifc", semantics: { lengthScale: 0.3048, storeys, frame: { site, trueNorth: [0, 1], map: null } }, box: { min: [1, 2, 1], max: [19, 7, 11] } };
  const ok = federationCheck(main, mep);
  assert.equal(ok.status, "aligned");
  assert.match(ok.checks.find((c) => c.id === "units").text, /metres and feet/);
  // Exported from a different base: 12.3 m off in both directions (HHS, section 6.3 item 1).
  const shifted = { ...mep, semantics: { ...mep.semantics, frame: { ...mep.semantics.frame, site: { ...site, origin: [12.3, 12.3, 0] } } }, box: { min: [13.3, 2, 13.3], max: [31.3, 7, 23.3] } };
  const off = federationCheck(main, shifted);
  assert.equal(off.status, "offset");
  assert.equal(off.checks.find((c) => c.id === "placement").status, "warning");
  assert.ok(off.offset);
  // TrueNorth written differently is a note, not a warning (section 6.3 item 3).
  const north = federationCheck(main, { ...mep, semantics: { ...mep.semantics, frame: { ...mep.semantics.frame, trueNorth: [Math.sin(0.0873), Math.cos(0.0873)] } } });
  assert.equal(north.status, "aligned");
  assert.equal(north.checks.find((c) => c.id === "north").status, "note");
});
