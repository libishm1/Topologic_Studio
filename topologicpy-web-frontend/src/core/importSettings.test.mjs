import test from "node:test";
import assert from "node:assert/strict";

import { DataSet, IfcImporter } from "@thatopen/fragments";
import * as WEBIFC from "web-ifc";

import { IMPORT_SETTINGS, converterId, trimImporter } from "./importSettings.js";

test("importSettings: the converter id names both versions and the settings", () => {
  assert.equal(converterId("3.4.7", "0.0.77"), `fragments 3.4.7; web-ifc 0.0.77; settings ${IMPORT_SETTINGS}`);
});

test("importSettings: property, type and material relations go; the spatial tree's stay", () => {
  const importer = trimImporter(new IfcImporter(), WEBIFC, DataSet);
  for (const gone of [WEBIFC.IFCRELDEFINESBYPROPERTIES, WEBIFC.IFCRELDEFINESBYTYPE, WEBIFC.IFCRELASSOCIATESMATERIAL]) {
    assert.equal(importer.relations.has(gone), false);
  }
  // getItemsChildren (stair parts, viewer/categories.js) walks the tree these build.
  for (const kept of [
    WEBIFC.IFCRELAGGREGATES,
    WEBIFC.IFCRELCONTAINEDINSPATIALSTRUCTURE,
    WEBIFC.IFCRELREFERENCEDINSPATIALSTRUCTURE,
    WEBIFC.IFCRELNESTS,
  ]) {
    assert.equal(importer.relations.has(kept), true);
  }
});

test("importSettings: abstract classes are the spatial ones; elements, spaces included, are untouched", () => {
  const before = new IfcImporter();
  const importer = trimImporter(new IfcImporter(), WEBIFC, DataSet);
  assert.ok(importer.classes.abstract.has(WEBIFC.IFCBUILDINGSTOREY));
  for (const gone of [WEBIFC.IFCPROPERTYSET, WEBIFC.IFCMATERIAL, WEBIFC.IFCDOORTYPE]) {
    assert.equal(before.classes.abstract.has(gone), true, "the default reads it");
    assert.equal(importer.classes.abstract.has(gone), false);
  }
  assert.deepEqual([...importer.classes.elements].sort(), [...before.classes.elements].sort());
  assert.ok(importer.classes.elements.has(WEBIFC.IFCSPACE));
});
