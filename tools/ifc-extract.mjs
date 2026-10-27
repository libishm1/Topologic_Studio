/**
 * IFC geometry for the Node tools, grouped as the site groups it
 * (src/viewer/categories.js), from web-ifc's Node build.
 */
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

export async function extract(ifcPath) {
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/"), true);
  await api.Init();
  const model = api.OpenModel(new Uint8Array(readFileSync(ifcPath)), { COORDINATE_TO_ORIGIN: true });
  const out = {};
  for (const [group, names] of Object.entries(GROUPS)) {
    const types = names.map((n) => WebIFC[n]).filter((t) => typeof t === "number");
    const meshes = [];
    api.StreamAllMeshesWithTypes(model, types, (mesh) => {
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
  api.CloseModel(model);
  return out;
}

