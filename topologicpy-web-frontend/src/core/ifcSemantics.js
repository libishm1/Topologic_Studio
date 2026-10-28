/**
 * What an IFC file says about egress that its geometry does not
 * (wiki/research/ifc-handling-survey-2026-10-27.md, items 1 and 2): which
 * doors are exits, and which slabs are roofs. One pass over the relations, no
 * geometry; at most 184 ms on the survey's 18 files.
 *
 * Exit doors, by the survey's measurements on 18 models:
 *   - Pset_DoorCommon.FireExit or .IsExternal TRUE: 5 of 18 files. The set is
 *     PSET_TYPEDRIVENOVERRIDE, so the occurrence's own value overrides its
 *     type's (FM_ARC has 8 doors typed FALSE but set TRUE on the occurrence).
 *   - else a door whose IfcRelSpaceBoundary is EXTERNAL: 8 of 18 files, and on
 *     the files with both signals every such door is also flagged.
 * FALSE is not evidence: ISSUE_159 sets IsExternal FALSE on all 47 doors.
 *
 * @param api      a web-ifc IfcAPI with the model open
 * @param modelID  its model id
 * @param W        the web-ifc module (for the type constants)
 */
const val = (a) => (a && typeof a === "object" && "value" in a ? a.value : a);
const truthy = (v) => v === true || v === "T" || v === ".T." || v === "TRUE" || v === 1;

export function ifcSemantics(api, modelID, W) {
  const ids = (type) => {
    const vec = api.GetLineIDsWithType(modelID, type, true);
    const out = new Array(vec.size());
    for (let i = 0; i < out.length; i += 1) out[i] = vec.get(i);
    return out;
  };
  const line = (id) => api.GetLine(modelID, id);

  const ownSets = new Map();
  for (const r of ids(W.IFCRELDEFINESBYPROPERTIES)) {
    const rel = line(r);
    for (const o of rel.RelatedObjects || []) {
      const k = val(o);
      if (!ownSets.has(k)) ownSets.set(k, []);
      ownSets.get(k).push(val(rel.RelatingPropertyDefinition));
    }
  }
  const typeOf = new Map();
  for (const r of ids(W.IFCRELDEFINESBYTYPE)) {
    const rel = line(r);
    for (const o of rel.RelatedObjects || []) typeOf.set(val(o), val(rel.RelatingType));
  }
  const prop = (setIds, name) => {
    for (const p of setIds) {
      const set = line(p);
      if (!/DoorCommon/i.test(String(val(set?.Name) ?? ""))) continue;
      for (const h of set.HasProperties || []) {
        const property = line(val(h));
        if (val(property?.Name) === name) return val(property.NominalValue);
      }
    }
    return undefined;
  };
  const doorProp = (door, name) => {
    const own = prop(ownSets.get(door) || [], name);
    if (own !== undefined) return own;
    const type = typeOf.get(door);
    return type ? prop((line(type)?.HasPropertySets || []).map(val), name) : undefined;
  };

  const external = new Set();
  for (const r of ids(W.IFCRELSPACEBOUNDARY)) {
    const rel = line(r);
    const element = val(rel.RelatedBuildingElement);
    if (element && /EXTERNAL/.test(String(val(rel.InternalOrExternalBoundary) ?? ""))) external.add(element);
  }

  const doors = ids(W.IFCDOOR).map((id) => {
    const door = line(id);
    let exit = null;
    if (truthy(doorProp(id, "FireExit"))) exit = "Pset_DoorCommon.FireExit";
    else if (truthy(doorProp(id, "IsExternal"))) exit = "Pset_DoorCommon.IsExternal";
    else if (external.has(id)) exit = "IfcRelSpaceBoundary.EXTERNAL";
    return {
      id,
      name: val(door.Name) ?? null,
      operation: val(door.OperationType) ?? null,
      width: val(door.OverallWidth) ?? null,
      exit,
    };
  });
  const roofSlabs = ids(W.IFCSLAB).filter((s) => val(line(s).PredefinedType) === "ROOF");
  return { schema: api.GetModelSchema(modelID), doors, roofSlabs };
}

/**
 * The model's exit doors as graph nodes, in order of trust: the doors the IFC
 * marks as exits; failing that, doors with walkable floor on one side only
 * (73% recall and 69% precision against the flags on five models).
 *
 * @param graphDoors  the graph's doors, [{itemId, node, sides}] (lattice.js doorReport)
 * @param semantics   ifcSemantics output, or null
 * @returns {{nodes: number[], doors: object[], source: string|null}}
 */
export function exitDoors(graphDoors, semantics) {
  const doors = graphDoors || [];
  const marked = new Map((semantics?.doors || []).filter((d) => d.exit).map((d) => [d.id, d]));
  let chosen = doors.filter((d) => d.node >= 0 && marked.has(d.itemId));
  let source = null;
  if (chosen.length) {
    const sources = new Set(chosen.map((d) => marked.get(d.itemId).exit));
    source = sources.size === 1 ? [...sources][0] : "the model's exit flags";
  } else {
    chosen = doors.filter((d) => d.node >= 0 && (d.sides?.[0] === 0 || d.sides?.[1] === 0));
    if (chosen.length) source = "doors with floor on one side";
  }
  const seen = new Set();
  const unique = chosen.filter((d) => !seen.has(d.node) && seen.add(d.node));
  return {
    nodes: unique.map((d) => d.node),
    doors: unique.map((d) => ({ itemId: d.itemId, node: d.node, name: marked.get(d.itemId)?.name ?? null })),
    source,
  };
}
