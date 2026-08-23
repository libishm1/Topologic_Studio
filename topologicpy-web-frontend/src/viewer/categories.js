/**
 * IFC category extraction, straight from the fragments model.
 *
 * Classic called `ifcLoader.readIfcFile(bytes)` a second time after the model
 * was already on screen, purely to enumerate IFCSLAB/IFCSTAIR/IFCDOOR/IFCWALL
 * express IDs - a second full parse of the same file, every session.
 *
 * `FragmentsModel.getItemsOfCategories()` answers the same question from the
 * fragments already in memory, so the second parse is gone entirely.
 */

export const CATEGORY_PATTERNS = {
  floors: /^IFC(SLAB|SLABSTANDARDCASE|SLABELEMENTEDCASE|COVERING|PLATE)$/i,
  stairs: /^IFC(STAIR|STAIRFLIGHT|RAMP|RAMPFLIGHT)$/i,
  doors: /^IFC(DOOR|DOORSTANDARDCASE)$/i,
  walls: /^IFC(WALL|WALLSTANDARDCASE|WALLELEMENTEDCASE)$/i,
  spaces: /^IFCSPACE$/i,
  storeys: /^IFCBUILDINGSTOREY$/i,
};

/**
 * Resolve the local ids for every category we care about, in one round trip
 * to the fragments worker.
 */
export async function collectCategoryIds(model) {
  if (!model?.getItemsOfCategories) {
    throw new Error("This fragments model does not expose category queries.");
  }

  const keys = Object.keys(CATEGORY_PATTERNS);
  const found = await model.getItemsOfCategories(keys.map((k) => CATEGORY_PATTERNS[k]));

  const result = {};
  for (const key of keys) result[key] = [];

  // The result is keyed by the concrete IFC category name, so map each one
  // back to the bucket whose pattern matched it.
  for (const [category, ids] of Object.entries(found || {})) {
    for (const key of keys) {
      if (CATEGORY_PATTERNS[key].test(category)) {
        result[key].push(...(ids || []));
        break;
      }
    }
  }

  for (const key of keys) {
    result[key] = Array.from(new Set(result[key]));
  }
  return result;
}

/**
 * Pull raw triangle data for a set of local ids and flatten it into plain
 * transferable buffers the sampling worker can consume.
 *
 * The per-mesh transform is baked in here because it is cheap in bulk and
 * means the worker never needs a matrix library.
 */
export async function extractMeshes(model, localIds, worldMatrix = null) {
  if (!localIds?.length || !model?.getItemsGeometry) return [];

  const geometrySets = await model.getItemsGeometry(localIds);
  const meshes = [];

  for (const set of geometrySets || []) {
    for (const mesh of set || []) {
      if (!mesh?.positions?.length || !mesh?.indices?.length) continue;

      const source = mesh.positions;
      const positions = new Float32Array(source.length);
      const m = mesh.transform?.elements || null;

      if (m) {
        for (let i = 0; i < source.length; i += 3) {
          const x = source[i];
          const y = source[i + 1];
          const z = source[i + 2];
          positions[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
          positions[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
          positions[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        }
      } else {
        positions.set(source);
      }

      if (worldMatrix) {
        const w = worldMatrix.elements;
        for (let i = 0; i < positions.length; i += 3) {
          const x = positions[i];
          const y = positions[i + 1];
          const z = positions[i + 2];
          positions[i] = w[0] * x + w[4] * y + w[8] * z + w[12];
          positions[i + 1] = w[1] * x + w[5] * y + w[9] * z + w[13];
          positions[i + 2] = w[2] * x + w[6] * y + w[10] * z + w[14];
        }
      }

      meshes.push({
        positions,
        indices:
          mesh.indices instanceof Uint32Array
            ? mesh.indices
            : new Uint32Array(mesh.indices),
      });
    }
  }

  return meshes;
}

/** Buffers to hand to `postMessage` as transferables. */
export function transferablesOf(groups) {
  const list = [];
  for (const meshes of groups) {
    for (const mesh of meshes) {
      list.push(mesh.positions.buffer, mesh.indices.buffer);
    }
  }
  return list;
}
