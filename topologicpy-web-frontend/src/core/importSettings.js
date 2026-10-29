/**
 * How an IFC becomes a fragments model, shared by the import worker
 * (workers/ifc-import.worker.js), the fragment cache's key
 * (lib/fragmentCache.js) and the examples converted at build time
 * (scripts/prepare-examples.mjs), so the three cannot drift apart.
 *
 * No web-ifc import here: the page bundle reads IMPORT_SETTINGS, and the
 * callers pass the web-ifc and fragments modules they already have.
 */

/**
 * Bump when the importer settings below or in ViewerManager, or what
 * core/ifcSemantics.js reads, change: it is part of every cache key, so models
 * converted the old way are converted again. 2: property sets, materials and
 * types left out of the fragments; semantics stored with them.
 */
export const IMPORT_SETTINGS = 2;

/** What produced a fragments buffer; cache keys and prepared examples carry it. */
export function converterId(fragmentsVersion, webIfcVersion) {
  return `fragments ${fragmentsVersion}; web-ifc ${webIfcVersion}; settings ${IMPORT_SETTINGS}`;
}

/** web-ifc settings for the conversion, as ViewerManager gives the IfcLoader. */
export const WEB_IFC_SETTINGS = { COORDINATE_TO_ORIGIN: true };

/**
 * Leave out of the fragments what the page never reads from them.
 *
 * The page reads element attributes (Name, LongName, ObjectType,
 * PredefinedType, OperationType; viewer/categories.js) and the spatial tree
 * that getItemsChildren walks, which the importer builds from
 * IfcRelAggregates, IfcRelContainedInSpatialStructure,
 * IfcRelReferencedInSpatialStructure and IfcRelNests. Property sets,
 * quantities, materials and type objects, and the relations to them, are read
 * from the IFC by ifcSemantics instead. Leaving them out takes 40% off the
 * Duplex's conversion and 26% off Office A's at 4x CPU, and the fragments are
 * a third the size.
 *
 * @param {object} importer  a @thatopen/fragments IfcImporter
 * @param {object} WEBIFC    the web-ifc module
 * @param {Function} DataSet the @thatopen/fragments DataSet class
 */
export function trimImporter(importer, WEBIFC, DataSet) {
  for (const relation of [
    WEBIFC.IFCRELDEFINESBYPROPERTIES,
    WEBIFC.IFCRELDEFINESBYTYPE,
    WEBIFC.IFCRELASSOCIATESMATERIAL,
  ]) {
    importer.relations.delete(relation);
  }
  importer.classes.abstract = new DataSet([
    WEBIFC.IFCPROJECT,
    WEBIFC.IFCSITE,
    WEBIFC.IFCBUILDING,
    WEBIFC.IFCBUILDINGSTOREY,
    WEBIFC.IFCALIGNMENT,
    WEBIFC.IFCALIGNMENTHORIZONTAL,
    WEBIFC.IFCALIGNMENTVERTICAL,
    WEBIFC.IFCALIGNMENTCANT,
    WEBIFC.IFCALIGNMENTSEGMENT,
  ]);
  return importer;
}
