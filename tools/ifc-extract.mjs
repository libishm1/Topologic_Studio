/**
 * IFC geometry for the Node tools, grouped as the site groups it
 * (src/viewer/categories.js), from web-ifc's Node build.
 */
import { ifcSemantics } from "../topologicpy-web-frontend/src/core/ifcSemantics.js";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const WebIFC = require(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/web-ifc-api-node.js"));

// Coverings are floors only when typed FLOORING: Office A's ceilings, typed
// CEILING, were otherwise the only "floors" at 2.55 m above each storey.
const GROUPS = {
  floors: ["IFCSLAB", "IFCPLATE", "IFCCOVERING"],
  stairs: ["IFCSTAIR", "IFCSTAIRFLIGHT", "IFCRAMP", "IFCRAMPFLIGHT"],
  doors: ["IFCDOOR"],
  walls: ["IFCWALL", "IFCWALLSTANDARDCASE"],
  spaces: ["IFCSPACE"],
  // Solid at walking height besides walls: cut with them (see lattice.js).
  obstacles: ["IFCWINDOW", "IFCRAILING", "IFCCOLUMN", "IFCCURTAINWALL", "IFCMEMBER", "IFCPLATE"],
};
const isWalkableCovering = (api, model, id) => {
  const value = api.GetLine(model, id)?.PredefinedType?.value;
  return value === "FLOORING";
};

/**
 * Exit flags per door, from Pset_DoorCommon (IsExternal, FireExit), on the
 * door or its type. Exporters fill these inconsistently, so both are read and
 * a missing value stays undefined. Pset_DoorCommon is PSET_TYPEDRIVENOVERRIDE:
 * a value on the door itself overrides its type's (FM_ARC types 8 doors FALSE
 * and sets them TRUE on the door), so the type is read only for what the door
 * leaves unset. The browser reads the same rule (src/core/ifcSemantics.js).
 */
async function doorFlags(api, model) {
  const flags = new Map();
  const yes = (v) => v === true || v === "T" || v === ".T." || v === "TRUE" || v === 1;
  const read = (sets, entry) => {
    const own = {};
    for (const set of sets || []) {
      for (const prop of set?.HasProperties || []) {
        const name = prop?.Name?.value;
        const value = prop?.NominalValue?.value;
        // An unset value ($) says nothing: it must not hide the type's.
        if (value === undefined || value === null) continue;
        // Pset_DoorCommon's IsExternal and FireExit, and Revit's own IsFireExit.
        if (name === "IsExternal" && /DoorCommon/i.test(set.Name?.value || "")) own.IsExternal = (own.IsExternal ?? false) || yes(value);
        if (name === "FireExit" || name === "IsFireExit") own.FireExit = (own.FireExit ?? false) || yes(value);
      }
    }
    for (const key of ["IsExternal", "FireExit"]) if (entry[key] === undefined && own[key] !== undefined) entry[key] = own[key];
  };
  const ids = api.GetLineIDsWithType(model, WebIFC.IFCDOOR);
  for (let i = 0; i < ids.size(); i += 1) {
    const id = ids.get(i);
    const entry = {};
    try {
      // Instance sets. (Passing includeTypeProperties here makes web-ifc
      // 0.0.77 return the type objects instead, so types are read apart.)
      read(await api.properties.getPropertySets(model, id, true), entry);
      for (const type of (await api.properties.getTypeProperties(model, id, true)) || []) read(type?.HasPropertySets, entry);
    } catch {
      // A malformed property set leaves the door unflagged.
    }
    flags.set(id, entry);
  }
  return flags;
}

export async function extract(ifcPath, { withDoorFlags = false } = {}) {
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/"), true);
  await api.Init();
  const model = api.OpenModel(new Uint8Array(readFileSync(ifcPath)), { COORDINATE_TO_ORIGIN: true });
  const out = {};
  for (const [group, names] of Object.entries(GROUPS)) {
    const types = names.map((n) => WebIFC[n]).filter((t) => typeof t === "number");
    const meshes = [];
    api.StreamAllMeshesWithTypes(model, types, (mesh) => {
      // Slabs typed ROOF are not floors (as the site does, ViewerManager.js).
      if (
        group === "floors" &&
        api.GetLineType(model, mesh.expressID) === WebIFC.IFCSLAB &&
        api.GetLine(model, mesh.expressID)?.PredefinedType?.value === "ROOF"
      ) {
        return;
      }
      if (
        api.GetLineType(model, mesh.expressID) === WebIFC.IFCCOVERING &&
        !isWalkableCovering(api, model, mesh.expressID)
      ) {
        return;
      }
      const placed = mesh.geometries;
      for (let g = 0; g < placed.size(); g += 1) {
        const pg = placed.get(g);
        const geom = api.GetGeometry(model, pg.geometryExpressID);
        const verts = api.GetVertexArray(geom.GetVertexData(), geom.GetVertexDataSize());
        const index = api.GetIndexArray(geom.GetIndexData(), geom.GetIndexDataSize());
        const m = pg.flatTransformation;
        const n = verts.length / 6;
        const positions = new Float32Array(n * 3);
        for (let v = 0; v < n; v += 1) {
          const x = verts[v * 6], y = verts[v * 6 + 1], z = verts[v * 6 + 2];
          positions[v * 3] = m[0] * x + m[4] * y + m[8] * z + m[12];
          positions[v * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
          positions[v * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        }
        const type = api.GetNameFromTypeCode(api.GetLineType(model, mesh.expressID));
        meshes.push({ positions, indices: Uint32Array.from(index), itemId: mesh.expressID, type });
        geom.delete?.();
      }
    });
    out[group] = meshes;
  }
  if (withDoorFlags) {
    out.doorFlags = await doorFlags(api, model);
    // Exit doors as the site reads them, space boundaries included.
    out.semantics = ifcSemantics(api, model, WebIFC);
  }
  api.CloseModel(model);
  return out;
}

