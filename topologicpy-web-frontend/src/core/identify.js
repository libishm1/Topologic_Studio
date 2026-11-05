/**
 * What an IFC element is, by rules: its class when the exporter typed it,
 * else its PredefinedType, its name (Name and ObjectType, in English,
 * German, Norwegian, Dutch, French, Spanish, Italian and a few Arabic
 * words), and last its bounding box (boxOf). Pure JavaScript, no model,
 * about 6 microseconds an element in Chrome.
 *
 * The benchmark (wiki/research/ifc-identification-benchmark-2026-11-05.md)
 * compared this with IFCNet's geometry classifier, a sentence-embedding model
 * and small Qwen LLMs in the browser, on the local corpus; see it for the
 * numbers. Exporters rarely type furnishings and appliances (the ignition
 * review's corpus check: no typed appliance in 45 models; cookers arrive as
 * IfcBuildingElementProxy "M_Cook Top-4 Unit"), so names carry most of what
 * can be known, and these rules read them.
 *
 * Labels are what egress and fire need (LABELS). English tokens are preceded
 * by (?:^|[^a-z]) rather than \b, since Revit names start "M_" and "_" is a
 * word character; German and Norwegian tokens have no leading boundary so
 * compounds match (Innenwand, Kühlschrank), as in the ignition review.
 */
import { IGNITION_SCENARIOS } from "./ignitionCatalog.js";

export const LABELS = [
  "wall", "slab", "roof", "door", "window", "stair", "railing", "column", "beam", "covering",
  "furniture", "appliance", "sanitary", "heater", "lighting", "mep", "space", "other",
];

const MEP_CLASSES = [
  "DUCTSEGMENT", "DUCTFITTING", "DUCTSILENCER", "PIPESEGMENT", "PIPEFITTING", "CABLECARRIERSEGMENT", "CABLECARRIERFITTING",
  "CABLESEGMENT", "CABLEFITTING", "AIRTERMINAL", "AIRTERMINALBOX", "VALVE", "DAMPER", "OUTLET", "SWITCHINGDEVICE",
  "PROTECTIVEDEVICE", "PROTECTIVEDEVICETRIPPINGUNIT", "ELECTRICDISTRIBUTIONBOARD", "ELECTRICDISTRIBUTIONPOINT", "JUNCTIONBOX",
  "FIRESUPPRESSIONTERMINAL", "SENSOR", "ALARM", "ACTUATOR", "CONTROLLER", "FLOWINSTRUMENT", "FLOWMETER", "PUMP", "FAN",
  "BOILER", "CHILLER", "TANK", "UNITARYEQUIPMENT", "AIRTOAIRHEATRECOVERY", "COIL", "COOLEDBEAM", "HEATEXCHANGER",
  "HUMIDIFIER", "FILTER", "COMPRESSOR", "CONDENSER", "COOLINGTOWER", "EVAPORATOR", "EVAPORATIVECOOLER", "TRANSFORMER",
  "ELECTRICGENERATOR", "ELECTRICMOTOR", "MOTORCONNECTION", "SOLARDEVICE", "BURNER", "ENGINE", "FLOWSEGMENT",
  "FLOWFITTING", "FLOWCONTROLLER", "FLOWMOVINGDEVICE", "FLOWSTORAGEDEVICE", "FLOWTREATMENTDEVICE",
  "ENERGYCONVERSIONDEVICE", "DISTRIBUTIONCONTROLELEMENT", "DISTRIBUTIONCHAMBERELEMENT", "STACKTERMINAL", "WASTETERMINAL",
  "INTERCEPTOR", "ELECTRICFLOWSTORAGEDEVICE", "ELECTRICTIMECONTROL", "UNITARYCONTROLELEMENT",
];

/** IFC classes (upper case, without "IFC") and their labels. */
const CLASS_LABELS = {
  WALL: "wall", WALLSTANDARDCASE: "wall", WALLELEMENTEDCASE: "wall", CURTAINWALL: "wall",
  SLAB: "slab", SLABSTANDARDCASE: "slab", SLABELEMENTEDCASE: "slab",
  ROOF: "roof",
  DOOR: "door", DOORSTANDARDCASE: "door",
  WINDOW: "window", WINDOWSTANDARDCASE: "window",
  STAIR: "stair", STAIRFLIGHT: "stair", RAMP: "stair", RAMPFLIGHT: "stair",
  RAILING: "railing",
  COLUMN: "column", COLUMNSTANDARDCASE: "column",
  BEAM: "beam", BEAMSTANDARDCASE: "beam",
  COVERING: "covering",
  FURNITURE: "furniture", FURNISHINGELEMENT: "furniture", SYSTEMFURNITUREELEMENT: "furniture",
  ELECTRICAPPLIANCE: "appliance", AUDIOVISUALAPPLIANCE: "appliance", COMMUNICATIONSAPPLIANCE: "appliance",
  SANITARYTERMINAL: "sanitary",
  SPACEHEATER: "heater",
  LIGHTFIXTURE: "lighting", LAMP: "lighting",
  SPACE: "space",
  ...Object.fromEntries(MEP_CLASSES.map((c) => [c, "mep"])),
};

/** Classes that say nothing about what the element is: the rules decide. */
export const GENERIC_CLASSES = new Set([
  "IFCBUILDINGELEMENTPROXY", "IFCFLOWTERMINAL", "IFCDISTRIBUTIONELEMENT", "IFCDISTRIBUTIONFLOWELEMENT",
  "IFCBUILDINGELEMENT", "IFCELEMENT", "IFCPROXY",
]);

/**
 * The label of a typed element: its class, refined by its PredefinedType
 * (a slab typed ROOF is a roof, a LANDING belongs to its stair). Null for
 * generic classes (GENERIC_CLASSES); "other" for any other element class.
 */
export function labelForClass(ifcClass, predefined = "") {
  const cls = String(ifcClass || "").toUpperCase();
  if (!cls || GENERIC_CLASSES.has(cls)) return null;
  const key = cls.replace(/^IFC/, "");
  const p = String(predefined || "").toUpperCase();
  if (key.startsWith("SLAB") && p === "ROOF") return "roof";
  if (key.startsWith("SLAB") && p === "LANDING") return "stair";
  return CLASS_LABELS[key] || "other";
}

// ---------------------------------------------------------------- PredefinedType

const PREDEFINED = [
  ["space", /^(SPACE|PARKING|GFA|INTERNAL|EXTERNAL)$/],
  ["roof", /^(ROOF|FLAT_ROOF|SHED_ROOF|GABLE_ROOF|HIP_ROOF|HIPPED_GABLE_ROOF|GAMBREL_ROOF|MANSARD_ROOF|BARREL_ROOF|RAINBOW_ROOF|BUTTERFLY_ROOF|PAVILION_ROOF|DOME_ROOF|FREEFORM)$/],
  ["stair", /STAIR$|RAMP$|^LANDING$|^STRAIGHT|^SPIRAL$|^WINDER$|^QUARTER_|^HALF_|^TWO_|^THREE_|^CURVED$/],
  ["door", /^(DOOR|GATE|TRAPDOOR|BOOM_BARRIER|TURNSTILE)$|SWING|FOLDING|SLIDING|REVOLVING|ROLLINGUP|SWING_FIXED/],
  ["window", /^(WINDOW|SKYLIGHT|LIGHTDOME)$/],
  ["railing", /^(HANDRAIL|GUARDRAIL|BALUSTRADE|FENCE)$/],
  ["column", /^(COLUMN|PILASTER|PIERSTEM|STANDCOLUMN)$/],
  ["beam", /^(BEAM|JOIST|HOLLOWCORE|LINTEL|SPANDREL|T_BEAM|GIRDER_SEGMENT|EDGEBEAM|PIERCAP|CORNICE|DIAPHRAGM)$/],
  ["wall", /^(SOLIDWALL|PARTITIONING|SHEAR|STANDARD|POLYGONAL|PLUMBINGWALL|MOVABLE|PARAPET|ELEMENTEDWALL|RETAININGWALL|WAVEWALL)$/],
  ["covering", /^(CEILING|FLOORING|CLADDING|ROOFING|MOLDING|SKIRTINGBOARD|INSULATION|MEMBRANE|SLEEVING|WRAPPING|COPING|TOPPING)$/],
  ["slab", /^(FLOOR|BASESLAB|APPROACH_SLAB|PAVING|WEARING|SIDEWALK|TRACKSLAB)$/],
  ["furniture", /^(CHAIR|TABLE|DESK|BED|FILECABINET|SHELF|SOFA|TECHNICALCABINET)$/],
  ["appliance", /^(DISHWASHER|ELECTRICCOOKER|FREESTANDINGELECTRICHEATER|FREESTANDINGFAN|FREESTANDINGWATERHEATER|FREESTANDINGWATERCOOLER|FREEZER|FRIDGE_FREEZER|HANDDRYER|KITCHENMACHINE|MICROWAVE|PHOTOCOPIER|REFRIGERATOR|TUMBLEDRYER|VENDINGMACHINE|WASHINGMACHINE|TV|COMPUTER|DISPLAY|PRINTER|PROJECTOR|SCANNER|SPEAKER|TELEPHONE|AUDIOVISUALAPPLIANCE)$/],
  ["sanitary", /^(BATH|BIDET|CISTERN|SHOWER|SINK|SANITARYFOUNTAIN|TOILETPAN|URINAL|WASHHANDBASIN|WCSEAT)$/],
  ["heater", /^(RADIATOR|CONVECTOR)$/],
  ["lighting", /^(POINTSOURCE|DIRECTIONSOURCE|SECURITYLIGHTING|COMPACTFLUORESCENT|FLUORESCENT|HALOGEN|HIGHPRESSUREMERCURY|HIGHPRESSURESODIUM|LED|METALHALIDE|OLED|TUNGSTENFILAMENT)$/],
  ["mep", /^(BEND|JUNCTION|RIGIDSEGMENT|FLEXIBLESEGMENT|CULVERT|GUTTER|SPOOL|DIFFUSER|GRILLE|REGISTER|LOUVRE|CONNECTOR|ENTRY|EXIT|TRANSITION|OBSTRUCTION|CABLELADDERSEGMENT|CABLETRAYSEGMENT|CABLETRUNKINGSEGMENT|CONDUITSEGMENT|SPRINKLER|HOSEREEL|FIREHYDRANT|BREECHINGINLET|SOCKETOUTLET|POWEROUTLET|DATAOUTLET|TELEPHONEOUTLET|AUDIOVISUALOUTLET|COMMUNICATIONSOUTLET|SWITCH|TOGGLESWITCH|CONTACTOR|DIMMERSWITCH|EMERGENCYSTOP|KEYPAD|MOMENTARYSWITCH|SELECTORSWITCH|STARTER|SWITCHDISCONNECTOR|CIRCUITBREAKER|FUSEDISCONNECTOR|RESIDUALCURRENTCIRCUITBREAKER|DISTRIBUTIONBOARD|CONSUMERUNIT|SWITCHBOARD|MOTORCONTROLCENTRE|FLOORTRAP|FLOORWASTE|GULLYSUMP|GULLYTRAP|ROOFDRAIN|WASTEDISPOSALUNIT|WASTETRAP|AIRVALVE|BALLVALVE|CHECK|GASCOCK|GATEVALVE|ISOLATING|MIXING|PRESSUREREDUCING|SAFETYCUTOFF|STOPCOCK|ISOLATOR|PRESSURERELIEF|SMOKEDETECTOR|HEATSENSOR|SMOKESENSOR|TEMPERATURESENSOR|THERMOSTAT|CENTRIFUGALFAN|AXIALFAN|PROPELLORFAN|CIRCULATOR|SUBMERSIBLEPUMP|SUMPPUMP|VERTICALINLINE|VERTICALTURBINE|PREFORMED|SECTIONAL|EXPANSION|STORAGE|PRESSUREVESSEL|VESSEL|WATER|STEAM|AIRHANDLER|AIRCONDITIONINGUNIT|SPLITSYSTEM|ROOFTOPUNIT)$/],
  ["other", /^(PROVISIONFORVOID|PROVISIONFORSPACE|FOOTING_BEAM|PAD_FOOTING|PILE_CAP|STRIP_FOOTING|CAISSON_FOUNDATION|MULLION|PLATE|STUD|STRINGER|BRACE|CHORD|COLLAR|MEMBER|PURLIN|RAFTER|STRUT|ANCHORPLATE|CURTAIN_PANEL|SHEET|SPLICEPLATE|BORED|DRIVEN|JETGROUTING|COHESION|FRICTION|SUPPORT|SHUTTER|LOUVER|AWNING|ELEVATOR|ESCALATOR|MOVINGWALKWAY|CRANEWAY|LIFTINGGEAR)$/],
];

function fromPredefined(predefined) {
  const p = String(predefined || "").toUpperCase().trim();
  if (!p || p === "NOTDEFINED" || p === "USERDEFINED" || p === "NULL" || p === "ELEMENT" || p === "COMPLEX" || p === "PARTIAL") return null;
  for (const [label, re] of PREDEFINED) if (re.test(p)) return label;
  return null;
}

// ---------------------------------------------------------------- names

const B = "(?:^|[^a-z])";
/**
 * Ordered: the first label whose pattern matches wins. Parts named after
 * their whole come first (a stair's stringer, a curtain wall's mullion), then
 * the specific before the generic: a "Floor Lamp" is lighting, a "Base
 * Cabinet-Double Door" furniture, a "Wall Cabinet" furniture, a "Basic Wall:
 * Partition (92mm Stud)" a wall. Spanish and Italian words for the building
 * fabric too (Revit exports keep the family names of the template's language).
 */
const NAME_RULES = [
  ["other", new RegExp(`${B}(mullions?|transoms?|stringers?|curtain ?panels?|system ?panels?)(?![a-z])`)],
  ["window", new RegExp(`${B}(windows?|skylight|roof ?light|casement|double ?hung|glazing unit|fenêtre|fenetre|raam|ramen|ventanas?|finestr[ae]|lucernario)(?![a-z])|fenster|vindu|lichtkuppel|oberlicht|نافذة|شباك`)],
  // Sanitary fittings before services: a "Water Closet - Flush Valve" is a WC, not a valve.
  ["sanitary", new RegExp(`${B}(toilets?|wc|water ?closets?|urinals?|lavatory|lavatories|wash ?basins?|bath ?tubs?|bidet)(?![a-z])|waschbecken|waschtisch|badewanne|klosett|toalett`)],
  ["mep", new RegExp(`${B}(panel ?board|distribution ?board|switch ?board|consumer ?unit|fuse ?box|ducts?|ductwork|pipes?|piping|valves?|sprinklers?|cable ?tray|conduit|trunking|busbar|diffusers?|grilles?|air ?terminal|air ?outlet|dampers?|ahu|vav|fcu|fan ?coil|pumps?|boilers?|chiller|water ?heater|calorifier|cylinder|tank|drains?|gully|gutter|downpipe|rain ?water|hydrant|extinguisher|hose ?reel|detector|smoke ?alarm|fire ?alarm|sensor|thermostat|socket|receptacle|power ?outlet|data ?outlet|switch(?:es)?|junction ?box|meter|elbow|bend|tee|reducer|flange|coupling|taps?|faucets?|mixer|blandebatteri)(?![a-z])|rohr|leitung|lüftung|lueftung|luftauslass|luftkanal|ventil(?![a-z])|steckdose|schalter|sicherungskasten|verteiler|heizungsrohr|abwasser|sprinkler|melder|armatur|(?:^|[^a-z])rør|stikkontakt|bryter|tuyau|gaine|vanne|interrupteur|leiding|wcd(?![a-z])|schakelaar`)],
  ["lighting", new RegExp(`${B}(lights?|lighting|lamps?|luminaires?|downlights?|spot ?lights?|pendant|sconce|chandelier|led ?panel|flood ?light|exit ?sign|emergency ?light|lys|belysning|éclairage|eclairage|verlichting|lampara|lampada)(?![a-z])|leuchte|lampe|beleuchtung|strahler|انارة|إنارة|مصباح`)],
  ["appliance", new RegExp(`${B}(cook ?tops?|cookers?|hob|stoves?|ovens?|range(?! ?hood)|induction|microwave|fryer|komfyr|refrigerators?|fridges?|freezers?|dish ?washers?|washing ?machines?|washers?|tumble ?dryers?|dryers?|driers?|tv|television|monitors?|computers?|printers?|copier|appliances?|range ?hood|cooker ?hood|extractor ?hood|kettle|coffee ?machine|vending|ovn|fryser|vaskemaskin|oppvaskmaskin|tørketrommel|kjøleskap)(?![a-z])|herd(?![a-z])|kochfeld|backofen|mikrowelle|kühlschrank|kuehlschrank|gefrier|waschmaschine|trockner|geschirrsp|spülmaschine|dunstabzug|fernseh|bildschirm|فرن|ثلاجة|غسالة`)],
  ["furniture", new RegExp(`${B}(cabinets?|cupboards?|wardrobes?|drawers?|casework|lockers?|shel(?:f|ves|ving)|bookcases?)(?![a-z])|schrank|kommode`)],
  ["door", new RegExp(`${B}(doors?|doorway|gate|hatch|single[- ]?flush|double[- ]?flush|flush ?door|bi[- ]?fold|porte|deur|deuren|portail|puertas?|porta|porte)(?![a-z])|tür|tuer|türe|haustür|(?:^|[^a-z])tor(?![a-z])|dør|باب`)],
  ["stair", new RegExp(`${B}(stairs?|stairway|staircase|steps?|flight|landing|ramps?|escalier|trap|trappen|escaleras?|scal[ae]|rampa)(?![a-z])|treppe|treppen|podest|rampe|trapp|درج|سلم`)],
  ["railing", new RegExp(`${B}(railings?|rails?|handrails?|hand ?rail|guard ?rails?|balustrades?|balusters?|banisters?|håndløper|garde-corps|rambarde|leuning|reling|barandillas?|barandas?|pasamanos|ringhiera|corrimano|parapetto)(?![a-z])|geländer|gelaender|handlauf|rekkverk|درابزين`)],
  ["sanitary", new RegExp(`${B}(toilets?|wc|w\\.c\\.|water ?closet|urinals?|basins?|wash ?basin|lavatory|lavatories|sinks?|bath ?tubs?|baths?|showers?|shower ?tray|bidet|cistern|plumbing ?fixture|toalett|servant|vask|dusj|badekar|lavabo|évier|evier|baignoire|douche|wastafel|toiletpot|spoelbak|douchebak|inodoro|lavabo|ducha|bañera|wc)(?![a-z])|waschbecken|waschtisch|badewanne|dusche|spüle|spuele|klosett|urinal|مرحاض|مغسلة`)],
  ["heater", new RegExp(`${B}(radiators?|space ?heaters?|heaters?|convectors?|panel ?heater|radiateur|convecteur|panelovn|varmeovn|radiador|termosifone)(?![a-z])|heizkörper|heizkoerper|heizung|konvektor|heizlüfter|مدفأة`)],
  ["other", new RegExp(`${B}(footings?|foundations?|piles?|pile ?cap|rebar|reinforc[a-z]*|anchor|bolt|kerb|curb|entourage|people|person|cars?|vehicle|trees?|plants?|shrub|topography|terrain|site|openings?|void|lift|elevator|escalator|fundament|heis|ascenseur)(?![a-z])|aufzug|öffnung|oeffnung|durchbruch|aussparung`)],
  ["furniture", new RegExp(`${B}(chairs?|tables?|desks?|sofas?|couch|settee|beds?|bunk|crib|cot|benches?|stools?|dresser|counters?|counter ?top|worktop|furniture|furnishings?|ottoman|arm ?chair|recliner|nightstand|bedside|sideboard|credenza|workstation|reception|seating|planter|piano|stol|stoler|bord|skap|hylle|reol|sofa|chaise|fauteuil|armoire|étagère|lit|meuble|stoel|tafel|kast|bureau|kruk|silla|mesa|sedia|tavolo|armadio)(?![a-z])|stuhl|tisch|regal|bett(?![a-z])|seng(?![a-z])|sessel|möbel|moebel|garderobe|كرسي|طاولة|سرير|خزانة|أريكة`)],
  ["roof", new RegExp(`${B}(roofs?|roofing|eaves?|tak|toit|toiture|dak|cubierta|tejado|tetto|copertura)(?![a-z])|dach(?!rinne|fenster)|سقف`)],
  ["covering", new RegExp(`${B}(ceilings?|soffit|cladding|covering|flooring|floor ?finish|tiles?|tiling|skirting|plaster|suspended|acoustic|carpet|parquet|screed|insulation|lining|isolasjon|plafond|faux[- ]plafond|revêtement|bekleding|falso ?techo|controsoffitto|pavimento)(?![a-z])|unterdecke|abhangdecke|abgehängte|bekleidung|himling|kledning|belag|estrich|dämmung|daemmung|parkett|putz`)],
  ["column", new RegExp(`${B}(columns?|pillars?|posts?|pilasters?|piers?|poteau|poteaux|colonne|kolom|pilar|columna|pilastro)(?![a-z])|stütze|stuetze|säule|saeule|søyle|عمود`)],
  ["beam", new RegExp(`${B}(beams?|joists?|girders?|lintels?|drager|poutre|linteau|balk|ligger|vigas?|trave)(?![a-z])|träger|traeger|unterzug|sturz|balken|bjelke|كمرة|جسر`)],
  ["wall", new RegExp(`${B}(walls?|partitions?|parapet|mur|murs|muur|cloison|muro|pared|parete|tabique|tramezzo)(?![a-z])|wand|wände|waende|mauer|brüstung|vegg|جدار|حائط`)],
  ["slab", new RegExp(`${B}(floors?|slabs?|deck|decking|plancher|dalle|vloer|bodem|losa|forjado|solaio|soletta)(?![a-z])|decke|bodenplatte|geschossdecke|dekke|بلاطة|أرضية`)],
  ["other", new RegExp(`${B}(panels?|plates?|members?|brac(?:e|ing)|truss|purlins?|rafters?|studs?|frames?)(?![a-z])|sparren|pfette|pfosten|riegel`)],
];

/** The label a name gives, or null. */
export function labelFromName(text) {
  const s = String(text || "").toLowerCase();
  if (!s.trim()) return null;
  for (const [label, re] of NAME_RULES) if (re.test(s)) return label;
  return null;
}

// ---------------------------------------------------------------- geometry

/**
 * The label a bounding box gives (metres), or null. dims: [length, width,
 * height] with length >= width the horizontal extents; above: the bottom's
 * height above the storey's floor.
 */
export function labelFromBox(dims, above) {
  if (!dims) return null;
  const [l, w, h] = dims;
  if (!(l > 0) && !(h > 0)) return null;
  const z = Number.isFinite(above) ? above : 0;
  // Room volumes: wide, storey-high, from the floor.
  if (w >= 1.2 && h >= 2.0 && h <= 8 && Math.abs(z) < 0.5) return "space";
  // Flat and large: a ceiling high up, else a floor slab.
  if (h <= 0.6 && l >= 2 && w >= 1) return z >= 2.0 && h <= 0.15 ? "covering" : "slab";
  // Door-sized and thin, standing on the floor.
  if (w <= 0.35 && l >= 0.5 && l <= 2.6 && h >= 1.9 && h <= 3.0 && Math.abs(z) < 0.2) return "door";
  // A thin panel off the floor (as deep as its wall).
  if (w <= 0.6 && l >= 0.3 && l <= 4 && h >= 0.3 && h <= 2.6 && z >= 0.3 && z <= 2.5) return "window";
  // Thin, tall and long.
  if (w <= 0.6 && h >= 2.0 && l >= 1.0) return "wall";
  // Slender and upright.
  if (l <= 1.2 && h >= 2.0 && h >= 2.5 * l) return "column";
  // Long and slender high up.
  if (l >= 2 && w <= 0.8 && h <= 1.2 && z >= 1.8) return "beam";
  // Hand-height and thin.
  if (w <= 0.3 && l >= 0.8 && h >= 0.8 && h <= 1.3 && Math.abs(z) < 0.3) return "railing";
  // Small things high up: lights.
  if (l <= 1.5 && h <= 0.6 && z >= 2.0) return "lighting";
  // Small things on the floor: furniture.
  if (l <= 3 && h <= 2.4 && Math.abs(z) < 0.3) return "furniture";
  return null;
}

/**
 * An element's box from its meshes (viewer frame, Y up): the horizontal
 * extents along its own principal axes (a wall at 30 degrees is still thin),
 * the height, and the bottom and centre.
 *
 * @returns {{dims: [number, number, number], bottom: number, centre: [number, number, number]}|null}
 */
export function boxOf(meshes) {
  let n = 0, mx = 0, mz = 0, lo = Infinity, hi = -Infinity;
  for (const m of meshes || []) {
    const p = m.positions;
    for (let i = 0; i + 2 < p.length; i += 3) {
      mx += p[i];
      mz += p[i + 2];
      lo = Math.min(lo, p[i + 1]);
      hi = Math.max(hi, p[i + 1]);
      n += 1;
    }
  }
  if (!n) return null;
  mx /= n;
  mz /= n;
  let sxx = 0, sxz = 0, szz = 0;
  for (const m of meshes) {
    const p = m.positions;
    for (let i = 0; i + 2 < p.length; i += 3) {
      const dx = p[i] - mx, dz = p[i + 2] - mz;
      sxx += dx * dx;
      sxz += dx * dz;
      szz += dz * dz;
    }
  }
  const angle = 0.5 * Math.atan2(2 * sxz, sxx - szz);
  const c = Math.cos(angle), s = Math.sin(angle);
  let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
  for (const m of meshes) {
    const p = m.positions;
    for (let i = 0; i + 2 < p.length; i += 3) {
      const dx = p[i] - mx, dz = p[i + 2] - mz;
      const a = dx * c + dz * s, b = -dx * s + dz * c;
      a0 = Math.min(a0, a);
      a1 = Math.max(a1, a);
      b0 = Math.min(b0, b);
      b1 = Math.max(b1, b);
    }
  }
  const l = Math.max(a1 - a0, b1 - b0), w = Math.min(a1 - a0, b1 - b0);
  const am = (a0 + a1) / 2, bm = (b0 + b1) / 2;
  return { dims: [l, w, hi - lo], bottom: lo, centre: [mx + am * c - bm * s, (lo + hi) / 2, mz + am * s + bm * c] };
}

// ---------------------------------------------------------------- one element

/** Ignition sub-type of an appliance, furnishing or heater, from the review's name patterns. */
const SUBTYPED = new Set(["appliance", "furniture", "heater", "mep", "other"]);
const compiled = new Map();
const re = (source) => {
  if (!source) return null;
  if (!compiled.has(source)) compiled.set(source, new RegExp(source, "i"));
  return compiled.get(source);
};
export function subtypeOf(text) {
  const s = String(text || "");
  if (!s) return null;
  for (const scenario of IGNITION_SCENARIOS) {
    const exclude = re(scenario.nameExclude);
    if (exclude && exclude.test(s)) continue;
    if (re(scenario.nameRegex)?.test(s)) return scenario.id;
  }
  return null;
}

/**
 * Identify one element.
 *
 * @param {object} record  {type?, name, objectType?, longName?, predefined?, dims?, above?}
 *   type is the IFC class when the caller wants it used (omit it to judge
 *   from the rest, as the benchmark does).
 * @returns {{label: string, confidence: number, by: "class"|"predefined"|"name"|"box"|"none", subtype?: string}}
 */
export function identify(record) {
  const text = [record.name, record.objectType].filter(Boolean).join(" ");
  const withSubtype = (out) => {
    if (SUBTYPED.has(out.label)) {
      const subtype = subtypeOf(text);
      if (subtype) out.subtype = subtype;
    }
    return out;
  };
  const byClass = record.type ? labelForClass(record.type, record.predefined) : null;
  if (byClass) return withSubtype({ label: byClass, confidence: 1, by: "class" });
  // Only spatial elements (rooms, storeys) carry a LongName.
  if (record.longName && String(record.longName).trim()) return { label: "space", confidence: 0.9, by: "predefined" };
  const byPredefined = fromPredefined(record.predefined);
  if (byPredefined) return withSubtype({ label: byPredefined, confidence: 0.9, by: "predefined" });
  const byName = labelFromName(text);
  if (byName) return withSubtype({ label: byName, confidence: 0.8, by: "name" });
  const byBox = labelFromBox(record.dims, record.above);
  if (byBox) return withSubtype({ label: byBox, confidence: 0.4, by: "box" });
  return { label: "other", confidence: 0.1, by: "none" };
}

/** Counts per label, for the Model step (typed elements by class, the rest by rules). */
export function countLabels(records) {
  const counts = Object.fromEntries(LABELS.map((l) => [l, 0]));
  for (const r of records) counts[identify(r).label] += 1;
  return counts;
}
