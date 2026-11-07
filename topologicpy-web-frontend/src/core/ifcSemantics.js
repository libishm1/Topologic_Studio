/**
 * What an IFC file says about egress that its geometry does not
 * (wiki/research/ifc-handling-survey-2026-10-27.md, items 1 and 2): which
 * doors are exits, and which slabs are roofs. One pass over the relations, no
 * geometry; at most 184 ms on the survey's 18 files. Also each door's declared
 * size and each IfcSpace's name, for the two-zone fire model (src/core/rooms.js),
 * and the sprinkler heads with their properties (sprinklerHeads below; their
 * positions come from the viewer, src/core/sprinklers.js).
 *
 * Exit doors, by the survey's measurements on 18 models:
 *   - Pset_DoorCommon.FireExit or .IsExternal TRUE: 5 of 18 files. The set is
 *     PSET_TYPEDRIVENOVERRIDE, so the occurrence's own value overrides its
 *     type's (FM_ARC has 8 doors typed FALSE but set TRUE on the occurrence).
 *   - else a door whose IfcRelSpaceBoundary is EXTERNAL: 8 of 18 files, and on
 *     the files with both signals every such door is also flagged.
 * FALSE is not evidence: ISSUE_159 sets IsExternal FALSE on all 47 doors.
 * Each door also says whether it is external (IsExternal TRUE or an EXTERNAL
 * boundary) whatever its FireExit, so the doors out of the building can be
 * told from fire exits inside it (exitDoors' `final`).
 *
 * @param api      a web-ifc IfcAPI with the model open
 * @param modelID  its model id
 * @param W        the web-ifc module (for the type constants)
 */
import { componentLabels } from "./route.js";
import { selfClosingRating } from "./doorStates.js";

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
  // An unset value ($, null) says nothing, so it does not hide the type's
  // value: a door whose own FireExit was unset over a type set TRUE read as
  // no exit (review m12).
  const prop = (setIds, name) => {
    for (const p of setIds) {
      const set = line(p);
      if (!/DoorCommon/i.test(String(val(set?.Name) ?? ""))) continue;
      for (const h of set.HasProperties || []) {
        const property = line(val(h));
        if (val(property?.Name) !== name) continue;
        const value = val(property.NominalValue);
        if (value != null) return value;
      }
    }
    return undefined;
  };
  const doorProp = (door, name) => {
    const own = prop(ownSets.get(door) || [], name);
    if (own != null) return own;
    const type = typeOf.get(door);
    return type ? prop((line(type)?.HasPropertySets || []).map(val), name) : undefined;
  };

  const external = new Set();
  for (const r of ids(W.IFCRELSPACEBOUNDARY)) {
    const rel = line(r);
    const element = val(rel.RelatedBuildingElement);
    if (element && /EXTERNAL/.test(String(val(rel.InternalOrExternalBoundary) ?? ""))) external.add(element);
  }

  // OverallWidth and OverallHeight are in the file's length unit; widthM and
  // heightM are in metres (null when not given), for the two-zone fire model's
  // door flows (src/core/rooms.js).
  const scale = lengthScale(api, modelID, W, ids, line);
  const metres = (v) => (typeof v === "number" && v > 0 ? v * scale : null);
  const rating = (id) => {
    const v = doorProp(id, "FireRating");
    return v == null || String(v).trim() === "" ? null : String(v).trim();
  };
  const doors = ids(W.IFCDOOR).map((id) => {
    const door = line(id);
    let exit = null;
    if (truthy(doorProp(id, "FireExit"))) exit = "Pset_DoorCommon.FireExit";
    else if (truthy(doorProp(id, "IsExternal"))) exit = "Pset_DoorCommon.IsExternal";
    else if (external.has(id)) exit = "IfcRelSpaceBoundary.EXTERNAL";
    const width = val(door.OverallWidth) ?? null;
    const height = val(door.OverallHeight) ?? null;
    return {
      id,
      name: val(door.Name) ?? null,
      operation: val(door.OperationType) ?? null,
      width,
      height,
      widthM: metres(width),
      heightM: metres(height),
      exit,
      external: truthy(doorProp(id, "IsExternal")) || external.has(id),
      // For the "as built" doors of a Flamescapes run (core/doorStates.js).
      fireRating: rating(id),
      selfClosing: truthy(doorProp(id, "SelfClosing")) || selfClosingRating(rating(id)),
    };
  });
  const roofSlabs = ids(W.IFCSLAB).filter((s) => val(line(s).PredefinedType) === "ROOF");
  // Room names, to say which room the two-zone model's fire is in.
  const spaces = ids(W.IFCSPACE).map((id) => {
    const space = line(id);
    return { id, name: val(space.Name) ?? null, longName: val(space.LongName) ?? null };
  });
  return {
    schema: api.GetModelSchema(modelID),
    doors,
    roofSlabs,
    storeys: storeys(api, modelID, W, ids, line),
    spaces,
    sprinklers: sprinklerHeads(api, modelID, W, { ids, line, ownSets, typeOf }),
    lengthScale: scale,
    frame: modelFrame(W, ids, line, scale),
  };
}

/**
 * Where the file puts its building, for federating a second model with it
 * (core/federation.js; MEP report section 6.4): the world placement of the
 * site and the building (origin in metres, x and z axes), the model
 * context's TrueNorth, and an IFC4 IfcMapConversion with its target CRS.
 * Null where the file has none; a failed read is null too.
 */
function modelFrame(W, ids, line, scale) {
  const point = (id) => (id ? (line(id)?.Coordinates || []).map((c) => Number(val(c)) || 0) : null);
  const direction = (id) => (id ? (line(id)?.DirectionRatios || []).map((c) => Number(val(c)) || 0) : null);
  const unit = (v) => {
    const n = Math.hypot(...v);
    return n > 0 ? v.map((c) => c / n) : v;
  };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  // An IfcLocalPlacement chain composed into origin and axes (columns x, y, z).
  const world = (placementId) => {
    let o = [0, 0, 0], x = [1, 0, 0], y = [0, 1, 0], z = [0, 0, 1];
    const chain = [];
    for (let p = placementId, guard = 0; p && guard < 32; guard += 1) {
      const local = line(p);
      chain.push(local?.RelativePlacement ? line(val(local.RelativePlacement)) : null);
      p = val(local?.PlacementRelTo);
    }
    for (const axis of chain.reverse()) {
      if (!axis) continue;
      const loc = point(val(axis.Location)) || [0, 0, 0];
      const az = unit(direction(val(axis.Axis)) || [0, 0, 1]);
      const ref = direction(val(axis.RefDirection)) || [1, 0, 0];
      const ax = unit(cross(cross(az, ref), az));
      const ay = cross(az, ax);
      const apply = (v) => [0, 1, 2].map((r) => x[r] * v[0] + y[r] * v[1] + z[r] * v[2]);
      o = [0, 1, 2].map((r) => o[r] + apply([loc[0] || 0, loc[1] || 0, loc[2] || 0])[r]);
      [x, y, z] = [apply(ax), apply(ay), apply(az)];
    }
    return { origin: o.map((c) => c * scale), xAxis: x, zAxis: z };
  };
  const placed = (type) => {
    try {
      const id = ids(type)[0];
      const placement = id ? val(line(id).ObjectPlacement) : null;
      return placement ? world(placement) : null;
    } catch {
      return null;
    }
  };
  let trueNorth = null;
  try {
    for (const c of ids(W.IFCGEOMETRICREPRESENTATIONCONTEXT)) {
      const context = line(c);
      if (val(context.ContextType) && !/model/i.test(String(val(context.ContextType)))) continue;
      const north = direction(val(context.TrueNorth));
      if (north) {
        trueNorth = unit(north.slice(0, 2));
        break;
      }
    }
  } catch {
    trueNorth = null;
  }
  let map = null;
  try {
    const conversion = W.IFCMAPCONVERSION ? ids(W.IFCMAPCONVERSION)[0] : undefined;
    if (conversion) {
      const m = line(conversion);
      const crs = val(m.TargetCRS) ? line(val(m.TargetCRS)) : null;
      map = {
        eastings: Number(val(m.Eastings)) || 0,
        northings: Number(val(m.Northings)) || 0,
        height: Number(val(m.OrthogonalHeight)) || 0,
        xAxisAbscissa: val(m.XAxisAbscissa) == null ? 1 : Number(val(m.XAxisAbscissa)),
        xAxisOrdinate: val(m.XAxisOrdinate) == null ? 0 : Number(val(m.XAxisOrdinate)),
        scale: val(m.Scale) == null ? 1 : Number(val(m.Scale)),
        crs: crs ? String(val(crs.Name) ?? "") : null,
      };
    }
  } catch {
    map = null;
  }
  return { site: placed(W.IFCSITE), building: placed(W.IFCBUILDING), trueNorth, map };
}

const PREFIXES = { EXA: 1e18, PETA: 1e15, TERA: 1e12, GIGA: 1e9, MEGA: 1e6, KILO: 1e3, HECTO: 1e2, DECA: 10, DECI: 1e-1, CENTI: 1e-2, MILLI: 1e-3, MICRO: 1e-6, NANO: 1e-9 };

/**
 * SI units per one `unit`: an SI unit by its prefix (squared or cubed for
 * SQUARE_ and CUBIC_ units), a conversion-based unit by its factor times the
 * scale of the unit the factor is given in, which can itself be
 * conversion-based (an inch defined as 1/12 foot read as 1/12 m, review
 * m12), a derived unit by the product of its elements' scales.
 */
function unitScale(unit, line, depth = 0) {
  if (!unit || depth > 8) return 1;
  const prefix = val(unit.Prefix);
  if (prefix !== undefined && prefix !== null) {
    const name = String(val(unit.Name) ?? "");
    return (PREFIXES[prefix] ?? 1) ** (/^SQUARE/.test(name) ? 2 : /^CUBIC/.test(name) ? 3 : 1);
  }
  const factor = unit.ConversionFactor ? line(val(unit.ConversionFactor)) : null;
  const value = val(factor?.ValueComponent);
  if (typeof value === "number" && value > 0) {
    return value * (factor.UnitComponent ? unitScale(line(val(factor.UnitComponent)), line, depth + 1) : 1);
  }
  if (Array.isArray(unit.Elements)) {
    let scale = 1;
    for (const ref of unit.Elements) {
      const element = line(val(ref));
      scale *= unitScale(line(val(element?.Unit)), line, depth + 1) ** (Number(val(element?.Exponent)) || 1);
    }
    return scale;
  }
  return 1;
}

/** The project's units (IfcUnitAssignment) by UnitType, as lines. */
function projectUnits(W, ids, line) {
  const units = new Map();
  for (const a of ids(W.IFCUNITASSIGNMENT)) {
    for (const ref of line(a).Units || []) {
      const unit = line(val(ref));
      const type = val(unit?.UnitType);
      if (type && !units.has(type)) units.set(type, unit);
    }
  }
  return units;
}

/** Metres per length unit of the file (IfcUnitAssignment): SI prefixes and
 *  conversion-based units such as the foot. */
export function lengthScale(api, modelID, W, ids, line) {
  return unitScale(projectUnits(W, ids, line).get("LENGTHUNIT"), line);
}

/** A sprinkler head's name, on its type or the occurrence (MEP report section 5.3, step 4). */
export const SPRINKLER_NAME = /sprinkler|pendent|pendant|upright|side ?wall|concealed/i;
/** Fire suppression terminals that are not heads. Revit's "Sprinkler - Pendent - Hosted" is not a hose. */
const NOT_A_HEAD = /hose ?reel|hydrant|breeching/i;
const OTHER_TERMINALS = new Set(["HOSEREEL", "FIREHYDRANT", "BREECHINGINLET", "SPRINKLERDEFLECTOR", "FIREMONITOR"]);
const SPRINKLER_PSET = /^Pset_FireSuppressionTerminalTypeSprinkler$/i;
const REVIT_PSET = /revit.*fire ?protection/i;
/** Bulb colours (NFPA 13 temperature-rating table; codes report rows 50 and 51), °C. */
const BULB_C = { ORANGE: 57, RED: 68, YELLOW: 79, GREEN: 93, BLUE: 141 };
/** L/(min bar^1/2) per gpm/psi^1/2: 3.785411784 L per US gallon over √0.0689475729 bar per psi. */
const K_US = 3.785411784 / Math.sqrt(0.0689475729);
/** Revit's internal units, in which it writes plain IfcReal properties: feet, kelvin. */
const REVIT = { flowLpm: 0.028316846592 * 1000 * 60, pascal: 1 / 0.3048, squareMetre: 0.09290304 };

/**
 * The sprinkler heads (wiki/research/sprinkler-ifc-mep-2026-11-07.md,
 * sections 5.3, 5.5 and 5.9). Candidates are IfcFlowTerminal (IFC2x3) and
 * IfcFireSuppressionTerminal (IFC4 on). A head is one typed by an
 * IfcFireSuppressionTerminalType, or an IfcFireSuppressionTerminal, whose
 * PredefinedType is SPRINKLER; or, since every real export checked writes
 * NOTDEFINED, NOTDEFINED or USERDEFINED with the type's Name or ElementType
 * or the occurrence's Name or ObjectType naming a head (SPRINKLER_NAME) and
 * none naming a hose reel, hydrant or breeching inlet. Hose reels, hydrants,
 * breeching inlets, deflectors and monitors by enumeration are not heads. An
 * untyped IfcFlowTerminal counts only if its own name says "sprinkler":
 * "pendant" there may be a light.
 *
 * Properties, the occurrence's before its type's (PSET_TYPEDRIVENOVERRIDE),
 * the standard set before Revit's at each:
 *   activationC  Pset_FireSuppressionTerminalTypeSprinkler.ActivationTemperature
 *                (the project's temperature unit, kelvin by default), else its
 *                BulbLiquidColour/BulbLiquidColor; Revit "Temperature Rating" (kelvin)
 *   kFactor      DischargeCoefficient; Revit "K-Factor". Below 40 it is read as
 *                gpm/psi^1/2 (K5.6) and converted: metric K-factors start near 57
 *   response     Response QUICK or STANDARD; Revit "Response" only as text (its
 *                integer code is not documented)
 *   coverage     CoverageArea (m²); Revit "Coverage" only as an area (its integer
 *                code is an option, not an area)
 *   flowLpm      DischargeFlowRate; Revit "Flow" (ft³/s)
 *   pressureBar  ResidualFlowingPressure; Revit "Pressure Drop" (Revit's internal
 *                kg/(ft s²)), the pressure across the head
 * A plain IfcReal in a Revit set is in Revit's internal units; a typed measure
 * anywhere is in the project's. Each is null where the file says nothing.
 *
 * @returns {Array<{id, name, typeName, activationC, kFactor, response, coverage, flowLpm, pressureBar}>}
 */
function sprinklerHeads(api, modelID, W, { ids, line, ownSets, typeOf }) {
  const FST = W.IFCFIRESUPPRESSIONTERMINAL;
  const FST_TYPE = W.IFCFIRESUPPRESSIONTERMINALTYPE;
  if (W.IFCFLOWTERMINAL === undefined && FST === undefined) return [];
  // Exact IfcFlowTerminal: in IFC4 the inherited query would bring every light,
  // outlet and sanitary terminal.
  const exact = (type) => {
    if (type === undefined) return [];
    const vec = api.GetLineIDsWithType(modelID, type, false);
    const out = new Array(vec.size());
    for (let i = 0; i < out.length; i += 1) out[i] = vec.get(i);
    return out;
  };
  const candidates = new Set([...exact(W.IFCFLOWTERMINAL), ...(FST === undefined ? [] : ids(FST))]);
  if (!candidates.size) return [];

  const units = projectUnits(W, ids, line);
  const scaleOf = (type) => unitScale(units.get(type), line);
  const temperatureUnit = (() => {
    const unit = units.get("THERMODYNAMICTEMPERATUREUNIT");
    const name = String(val(unit?.Name) ?? "");
    return /CELSIUS/i.test(name) ? "C" : /FAHRENHEIT/i.test(name) ? "F" : "K";
  })();
  const scales = {
    area: scaleOf("AREAUNIT"),
    flow: scaleOf("VOLUMETRICFLOWRATEUNIT"),
    pressure: scaleOf("PRESSUREUNIT"),
  };

  const kindOf = (nominal) => String(nominal?.name ?? nominal?.label ?? "").toUpperCase();
  const numberOf = (nominal) => {
    const v = val(nominal);
    const x = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
    return Number.isFinite(x) ? x : null;
  };
  const celsius = (v, unit) => {
    const c = unit === "C" ? v : unit === "F" ? ((v - 32) * 5) / 9 : v >= 200 ? v - 273.15 : v;
    return c >= 30 && c <= 400 ? c : null;
  };
  // Each field: the standard property names, the Revit ones, and the reading.
  // `revit` is true for a Revit set, where a plain number is in internal units.
  const FIELDS = {
    activationC: {
      std: ["ActivationTemperature"],
      revit: ["Temperature Rating"],
      read: (nominal, revit) => {
        const v = numberOf(nominal);
        if (v == null) return null;
        const typed = /TEMPERATURE/.test(kindOf(nominal));
        return celsius(v, revit && !typed ? "K" : temperatureUnit);
      },
    },
    bulb: {
      std: ["BulbLiquidColour", "BulbLiquidColor"],
      revit: [],
      read: (nominal) => BULB_C[String(val(nominal) ?? "").toUpperCase()] ?? null,
    },
    kFactor: {
      std: ["DischargeCoefficient"],
      revit: ["K-Factor"],
      read: (nominal) => {
        const v = numberOf(nominal);
        return v > 0 ? (v < 40 ? v * K_US : v) : null;
      },
    },
    response: {
      std: ["Response"],
      revit: ["Response"],
      read: (nominal) => {
        const v = val(nominal);
        if (typeof v !== "string") return null;
        return /quick|fast/i.test(v) ? "quick" : /standard/i.test(v) ? "standard" : null;
      },
    },
    coverage: {
      std: ["CoverageArea"],
      revit: ["Coverage"],
      read: (nominal, revit) => {
        const kind = kindOf(nominal);
        if (/INTEGER/.test(kind)) return null;
        const v = numberOf(nominal);
        if (!(v > 0)) return null;
        return revit && !/MEASURE/.test(kind) ? v * REVIT.squareMetre : v * scales.area;
      },
    },
    flowLpm: {
      std: ["DischargeFlowRate"],
      revit: ["Flow"],
      read: (nominal, revit) => {
        const v = numberOf(nominal);
        if (!(v > 0)) return null;
        return revit && !/MEASURE/.test(kindOf(nominal)) ? v * REVIT.flowLpm : v * scales.flow * 60000;
      },
    },
    pressureBar: {
      std: ["ResidualFlowingPressure"],
      revit: ["Pressure Drop"],
      read: (nominal, revit) => {
        const v = numberOf(nominal);
        if (!(v > 0)) return null;
        return (revit && !/MEASURE/.test(kindOf(nominal)) ? v * REVIT.pascal : v * scales.pressure) / 1e5;
      },
    },
  };

  // Property sets, read once each: {revit, props: Map name -> nominal value}, or null if not one of ours.
  const setCache = new Map();
  const readSet = (setId) => {
    if (setCache.has(setId)) return setCache.get(setId);
    const set = line(setId);
    const name = String(val(set?.Name) ?? "");
    let entry = null;
    if (SPRINKLER_PSET.test(name) || REVIT_PSET.test(name)) {
      const props = new Map();
      for (const h of set.HasProperties || []) {
        const p = line(val(h));
        const key = val(p?.Name);
        if (key != null && !props.has(key)) props.set(key, p.NominalValue ?? p.EnumerationValues?.[0] ?? null);
      }
      entry = { revit: !SPRINKLER_PSET.test(name), props };
    }
    setCache.set(setId, entry);
    return entry;
  };
  const enumOf = (x) => String(val(x?.PredefinedType) ?? "").toUpperCase();
  const typeCache = new Map();
  const typeLine = (id) => {
    if (!typeCache.has(id)) typeCache.set(id, line(id));
    return typeCache.get(id);
  };
  const named = (...texts) => {
    const names = texts.map((t) => String(val(t) ?? "")).filter(Boolean);
    return names.some((t) => SPRINKLER_NAME.test(t)) && !names.some((t) => NOT_A_HEAD.test(t));
  };

  const heads = [];
  for (const id of candidates) {
    const typeId = typeOf.get(id);
    const type = typeId ? typeLine(typeId) : null;
    if (type && FST_TYPE !== undefined && type.type !== FST_TYPE) continue; // an air terminal, a light ...
    const occ = line(id);
    const fstClass = FST !== undefined && occ?.type === FST;
    let head;
    if (type || fstClass) {
      const own = enumOf(type);
      const kind = own && own !== "NOTDEFINED" && own !== "USERDEFINED" ? own : enumOf(occ);
      if (kind === "SPRINKLER") head = true;
      else if (OTHER_TERMINALS.has(kind)) head = false;
      else head = named(type?.Name, type?.ElementType, occ?.Name, occ?.ObjectType);
    } else {
      head = /sprinkler/i.test(String(val(occ?.Name) ?? "") + " " + String(val(occ?.ObjectType) ?? "")) && named(occ?.Name, occ?.ObjectType);
    }
    if (!head) continue;

    // The occurrence's sets first, then its type's; at each, the standard set before Revit's.
    const own = (ownSets.get(id) || []).map(readSet).filter(Boolean);
    const typed = typeId ? [...(type?.HasPropertySets || []).map(val), ...(ownSets.get(typeId) || [])].map(readSet).filter(Boolean) : [];
    const ordered = [own, typed].flatMap((group) => [...group.filter((e) => !e.revit), ...group.filter((e) => e.revit)]);
    const field = (name) => {
      const spec = FIELDS[name];
      for (const entry of ordered) {
        for (const key of entry.revit ? spec.revit : spec.std) {
          if (!entry.props.has(key)) continue;
          const v = spec.read(entry.props.get(key), entry.revit);
          if (v != null) return v;
        }
      }
      return null;
    };
    heads.push({
      id,
      name: val(occ?.Name) ?? null,
      typeName: val(type?.Name) ?? val(occ?.ObjectType) ?? null,
      activationC: field("activationC") ?? field("bulb"),
      kFactor: field("kFactor"),
      response: field("response"),
      coverage: field("coverage"),
      flowLpm: field("flowLpm"),
      pressureBar: field("pressureBar"),
    });
  }
  return heads;
}

/**
 * Storeys with their absolute height in metres: the z of the storey's local
 * placement chain (no tilt assumed), in the file's IFC frame (strategy 5 of
 * the survey; the viewer adds its coordination matrix's vertical shift). Floor
 * heights read from slab geometry collapse to one level on the dental clinic,
 * Office A and example.ifc, which declare 4, 3 and 2 storeys.
 */
function storeys(api, modelID, W, ids, line) {
  const scale = lengthScale(api, modelID, W, ids, line);
  return ids(W.IFCBUILDINGSTOREY).map((id) => {
    const storey = line(id);
    let z = 0;
    let placement = val(storey.ObjectPlacement);
    for (let guard = 0; placement && guard < 32; guard += 1) {
      const local = line(placement);
      const axis = local?.RelativePlacement ? line(val(local.RelativePlacement)) : null;
      const location = axis?.Location ? line(val(axis.Location)) : null;
      const c = location?.Coordinates || [];
      z += Number(val(c[2]) ?? 0) || 0;
      placement = val(local?.PlacementRelTo);
    }
    // The Elevation attribute too, for files that place no storey (the NIBS
    // MEP exports): federation matches storeys by it (core/federation.js).
    const declared = Number(val(storey.Elevation));
    return { id, name: val(storey.Name) ?? null, elevation: z * scale, declared: Number.isFinite(declared) ? declared * scale : null };
  });
}

/**
 * Fallback exits for the parts of the graph the flagged exits do not reach.
 * A model may flag only some of its exits: AdvancedProject flags one, in a
 * part holding 0.5% of the graph, so the rest had no way out. Each part
 * (connected with walls on) of at least 50 nodes and 2% of the graph that
 * holds no flagged exit gets the fallback doors standing in it.
 *
 * @param graph     {nodes, edges, blocked}, or null to skip
 * @param flagged   exit nodes from the flags
 * @param fallback  candidate nodes (doors with floor on one side)
 * @returns {number[]} the fallback nodes added
 */
export function exitsPerPart(graph, flagged, fallback) {
  if (!graph?.nodes || !fallback.length) return [];
  const { labels, sizes } = componentLabels(graph);
  const min = Math.max(50, 0.02 * labels.length);
  const served = new Set(flagged.map((n) => labels[n]));
  return fallback.filter((n) => !served.has(labels[n]) && sizes[labels[n]] >= min);
}

/**
 * The model's exit doors as graph nodes, in order of trust: the doors the IFC
 * marks as exits, plus doors with floor on one side in large parts of the
 * graph no flagged exit reaches (exitsPerPart) when the graph is given;
 * failing any flag, doors with walkable floor on one side only (73% recall
 * and 69% precision against the flags on five models).
 *
 * `final` holds the doors out of the building, for routing. A model may flag
 * the doors onto an escape route as fire exits too: the Vienna model flags 32
 * of its 81 doors, a flat's front door among them, so the nearest exit from a
 * flat was its own front door. Where the model says which doors are external,
 * the final exits are those at the level of the lowest (within STREET_BAND),
 * and the fire exits at any level that are external and open onto no
 * walkable floor (a fire escape to the outside, such as the Vienna model's
 * first-floor door at the back). An external door above street level with
 * floor beyond it opens onto a balcony or terrace and is left out. Large parts
 * of the graph none of these reaches keep their exits; a model that says
 * nothing about external doors keeps the exits above.
 *
 * @param graphDoors  the graph's doors, [{itemId, node, sides}] (lattice.js doorReport)
 * @param semantics   ifcSemantics output, or null
 * @returns {{nodes: number[], doors: object[], source: string|null, final: number[], finalSource: string|null}}
 */
export function exitDoors(graphDoors, semantics, graph = null) {
  const doors = graphDoors || [];
  const marked = new Map((semantics?.doors || []).filter((d) => d.exit).map((d) => [d.id, d]));
  const oneSided = doors.filter((d) => d.node >= 0 && (d.sides?.[0] === 0 || d.sides?.[1] === 0));
  let chosen = doors.filter((d) => d.node >= 0 && marked.has(d.itemId));
  let source = null;
  if (chosen.length) {
    const sources = new Set(chosen.map((d) => marked.get(d.itemId).exit));
    source = sources.size === 1 ? [...sources][0] : "the model's exit flags";
    const added = exitsPerPart(graph, chosen.map((d) => d.node), oneSided.map((d) => d.node));
    if (added.length) {
      const extra = new Set(added);
      chosen = [...chosen, ...oneSided.filter((d) => extra.has(d.node))];
      source = `${source}, and doors with floor on one side in parts with no flagged exit`;
    }
  } else {
    chosen = oneSided;
    if (chosen.length) source = "doors with floor on one side";
  }
  const seen = new Set();
  const unique = chosen.filter((d) => !seen.has(d.node) && seen.add(d.node));
  const nodes = unique.map((d) => d.node);
  let final = nodes;
  let finalSource = null;
  const outside = new Set((semantics?.doors || []).filter((d) => d.external).map((d) => d.id));
  const external = doors.filter((d) => d.node >= 0 && outside.has(d.itemId));
  if (external.length && graph?.nodes) {
    const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"];
    const height = (v) => graph.nodes[v * 3 + up];
    const ground = Math.min(...external.map((d) => height(d.node)));
    const openAir = (d) => d.sides?.[0] === 0 || d.sides?.[1] === 0;
    const out = external.filter(
      (d) => height(d.node) <= ground + STREET_BAND || (openAir(d) && /FireExit/.test(String(marked.get(d.itemId)?.exit))),
    );
    const leave = [...new Set(out.map((d) => d.node))];
    final = [...leave, ...exitsPerPart(graph, leave, nodes)];
    finalSource = "external doors at street level and fire exits to the open air";
  }
  return {
    nodes,
    doors: unique.map((d) => ({ itemId: d.itemId, node: d.node, name: marked.get(d.itemId)?.name ?? null })),
    source,
    final,
    finalSource,
  };
}

/** How far above the lowest external door (m) another still opens at street level. */
export const STREET_BAND = 1.5;
