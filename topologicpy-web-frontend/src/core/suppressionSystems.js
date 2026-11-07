/**
 * Suppression systems by room use (wiki/research/suppression-systems-by-room-2026-11-07.md):
 * which system each room gets, the parameters of a clean-agent total flood
 * (HFC-227ea, FM-200) and of a foam system, and the room on the fire grid
 * that the Flamescapes solver floods or blankets (flamescapes.js reads
 * `sim.systems`; flamescapesRun.js and flamescapesGpu.js build it).
 *
 * Rules (the note's section 4). Apartments (Group R) get water only:
 * sprinklers throughout (SBC 801-CR-2018 903.2.8, NFPA 13R where 903.3.1.2
 * allows) and an extinguisher in each dwelling (906.1 item 1 and its R-2
 * exception). In commercial buildings, rooms are read by name:
 *   - information technology, telecommunications, electrical and UPS or
 *     battery rooms get HFC-227ea at the class C design concentration,
 *     with the sprinklers kept (NFPA 75-2024 9.1.1 and 9.4.1: the agent is
 *     a supplement in a sprinklered building; SBC 801-CR-2018 903.3.1.1.1
 *     does not omit sprinklers because a room holds electrical equipment);
 *     "agent only" needs the official's approval (903.3.1.1.1 items 1, 2;
 *     904.2), an option here;
 *   - generator, fuel, oil and flammable-liquid rooms get foam-water
 *     sprinklers (NFPA 11-2021 6.5.2.2.1; SBC 801-CR-2018 904.7; NFPA 37
 *     11.4.1.1 allows NFPA 11 among others for engine rooms);
 *   - mechanical and plant rooms, and everything else, get water.
 *
 * In the solver:
 *   - Clean agent. After the room's detection (two cross-zoned detectors
 *     must see smoke) and the pre-discharge delay, the agent's concentration
 *     over the room's cells rises linearly to the design concentration over
 *     the discharge time, holds for the hold time, and is gone after it.
 *     Where it is at or above the minimum extinguishing concentration of the
 *     burning fuel (the heptane cup-burner value for a liquid's vapour, the
 *     class A value otherwise), gas-phase combustion and the oxidation of
 *     solid fuel stop; pyrolysis and a liquid's evaporation go on, so hot
 *     fuel keeps giving off vapour, and a fire whose fuel is still hot when
 *     the agent goes can re-ignite (NFPA 2001 annex A.7.4.1). The agent is
 *     not carried by the flow: the room is taken as a closed, well-mixed
 *     enclosure (NFPA 2001 requires its integrity and interlocks close
 *     doors and dampers, SBC 801-CR-2018 904.3.3), leakage is ignored
 *     within the hold, and the agent reaches no cell outside the room's
 *     cells. The gas that leaves the room through an open door in the
 *     model takes no agent with it and can burn outside.
 *   - Foam. After detection (one detector) and a delay, the blanket spreads
 *     over the room's liquid fuel: its evaporation falls linearly to a tenth
 *     at the control time and to nothing at the extinguishment time, and
 *     stays off (the blanket is kept for the run). While foam is applied,
 *     the solution's water wets the class A fuel it lands on as sprinkler
 *     water does ("wetting" mode, WETTED_FRACTION of what reaches the fuel's
 *     top cell in each column).
 *
 * Nothing here runs, and no field exists, unless a run asks for systems.
 */
import { WETTED_FRACTION } from "./suppression.js";
import { DETECTOR_VISIBILITY } from "./forecast.js";

// ------------------------------------------------------------------ parameters

/** HFC-227ea (FM-200) total flooding, NFPA 2001 (note section 1). */
export const CLEAN_AGENT = Object.freeze({
  agent: "HFC-227ea (FM-200)",
  mecHeptane: Object.freeze({
    value: 6.62,
    unit: "% v/v",
    source: "NFPA 2001-2022 Table A.7.2.2.2.1 (Errata 2001-22-1): n-heptane cup-burner minimum extinguishing concentration; Zegers et al. 2000 measured 6.4 % (doi:10.1016/S0010-2180(99)00162-5)",
  }),
  mecClassA: Object.freeze({ value: 5.2, unit: "% v/v", source: "NFPA 2001-2022 Table A.7.2.2.3(b) (First Draft): class A minimum extinguishing concentration from the listing tests (UL 2166)" }),
  design: Object.freeze({
    A: Object.freeze({ value: 6.7, unit: "% v/v", source: "NFPA 2001-2022 7.2.2.1.2: the greater of 1.2 times the class A MEC (6.24 %) and the heptane MEC" }),
    B: Object.freeze({ value: 8.6, unit: "% v/v", source: "NFPA 2001-2022 7.2.2.2.3: 1.3 times the class B extinguishing concentration (6.62 %)" }),
    C: Object.freeze({ value: 7.0, unit: "% v/v", source: "NFPA 2001-2022 7.2.2.4.1: 1.35 times the class A MEC (energised electrical equipment)" }),
  }),
  discharge: Object.freeze({ value: 10, unit: "s", source: "NFPA 2001-2004 5.7.1.2.1: 95 % of the design concentration within 10 s for halocarbon agents (taken here as a linear rise to the design value at 10 s)" }),
  hold: Object.freeze({
    value: 600,
    unit: "s",
    source: "NFPA 2001-2022 7.4.1: at least 85 % of the design concentration at the highest protected content for 10 min, or long enough for trained personnel to respond",
  }),
  retained: Object.freeze({ value: 0.85, unit: "of the design concentration", source: "NFPA 2001-2022 7.4.1 (the least the hold may keep); the solver holds the full design value unless a run sets `retained`" }),
  delay: Object.freeze({
    value: 30,
    unit: "s",
    source: "specification practice (University of Houston and Metrolinx 21 22 00): 30 s from the confirming detector; NFPA 2001-2025 9.7 requires a delay long enough to evacuate and sets no number",
  }),
  zones: Object.freeze({ value: 2, source: "cross-zoned release: a detector in each of two zones must respond (the same specifications); NFPA 2001-2004 4.3.1.2 (automatic actuation)" }),
  noael: Object.freeze({ value: 9.0, unit: "% v/v", source: "NFPA 2001-2025 Table 4.3.2.4(a)" }),
  loael: Object.freeze({ value: 10.5, unit: "% v/v", source: "NFPA 2001-2025 Table 4.3.2.4(a)" }),
  k1: Object.freeze({ value: 0.1269, unit: "m³/kg", source: "NFPA 2001-2004 5.5.1 and Table A.5.5.1(l): specific volume s = k1 + k2 t" }),
  k2: Object.freeze({ value: 0.000513, unit: "m³/(kg °C)", source: "NFPA 2001-2004 Table A.5.5.1(l) (its footnote rounds k2 to 0.0005; the tabled volumes follow 0.000513)" }),
});

/** Foam-water sprinklers and spill foam, NFPA 11 (note section 3). */
export const FOAM = Object.freeze({
  rate: Object.freeze({
    value: 6.5,
    unit: "L/(min m²)",
    source: "NFPA 11-2021 6.5.2.2.1: foam-water sprinklers at least 0.16 gpm/ft² (6.5 mm/min); NFPA 16 was merged into NFPA 11 in 2020",
  }),
  spillRate: Object.freeze({ value: 4.1, unit: "L/(min m²)", source: "NFPA 11-2021 Table 5.8.1.2: AFFF and FFFP on non-diked spills, 0.10 gpm/ft², 15 min (protein and fluoroprotein 0.16 gpm/ft², 6.5)" }),
  controlDensity: Object.freeze({
    value: 1.05,
    unit: "L/m²",
    source: "derived (note 3.6): AR-AFFF on a 4.65 m² heptane pan was controlled (90 % less intensity) after 0.9 to 1.2 L/m² of solution (Back and Farley 2020, FPRF); control time = density / rate",
  }),
  extinguishDensity: Object.freeze({
    value: 3.2,
    unit: "L/m²",
    source: "derived (note 3.6): put out after 1.8 to 4.1 L/m² (Back and Farley 2020); at 6.5 L/(min m²) about 30 s, at 4.1 about 45 s, counted from foam reaching the pool",
  }),
  fluorineFree: Object.freeze({ value: 2, unit: "times the AFFF densities", source: "Back 2023 (doi:10.1007/s10694-020-01051-4): fluorine-free foams needed 1.5 to 3 times the rate and 2 to 7 times the density of AFFF" }),
  duration: Object.freeze({ value: 600, unit: "s", source: "10 min of foam from foam-water sprinklers (NFPA 16 7.3.7.1, secondary; no NFPA 11-2021 clause found)" }),
  delay: Object.freeze({ value: 30, unit: "s", source: "assumed: from detection to foam at the outlets (valve, proportioner, pipe fill); no source read gives a typical value" }),
  zones: Object.freeze({ value: 1, source: "one detector releases a deluge or preaction foam-water system (assumed; closed heads would open on their own links)" }),
});

/** Smoke detectors, each zone at NFPA 72's nominal spacing. */
export const DETECTOR_SPACING = Object.freeze({
  value: 9.1,
  unit: "m",
  source: "NFPA 72-2019 17.7.3.2.3.1: 30 ft (9.1 m) nominal spacing for spot smoke detectors on smooth ceilings (not read; widely cited); each zone of a cross-zoned pair is laid out at it",
});

/** Specific volume of HFC-227ea vapour (m³/kg) at t (°C). */
export const agentVolume = (t = 20) => CLEAN_AGENT.k1.value + CLEAN_AGENT.k2.value * t;

/**
 * The agent for a total flood: W = (V / s) (C / (100 - C)) kg (NFPA 2001
 * 5.5.1), V the room's net volume (m³), C the design concentration (%), s
 * at the room's lowest design temperature t (°C).
 */
export function agentMass(volume, concentration = CLEAN_AGENT.design.C.value, t = 20) {
  if (!(volume > 0) || !(concentration > 0 && concentration < 100)) return 0;
  return (volume / agentVolume(t)) * (concentration / (100 - concentration));
}

/** Foam control and extinguishment times (s) at a rate (L/(min m²)), from the delivered densities. */
export function foamTimes(rate = FOAM.rate.value, { fluorineFree = false } = {}) {
  const k = fluorineFree ? FOAM.fluorineFree.value : 1;
  const r = Math.max(rate, 1e-9) / 60; // L/(m² s)
  return { control: (k * FOAM.controlDensity.value) / r, extinguish: (k * FOAM.extinguishDensity.value) / r };
}

// ------------------------------------------------------------------ rules by room

/** Portable units from extinguishers.js UNITS, by where they stand. */
const UNIT = Object.freeze({
  powder: { unit: "abc-2.3", text: "ABC dry powder (2-A:10-B:C or larger) on the floor's walking routes, within 22.86 m of travel", clause: "SBC 801-CR-2018 906.1, Table 906.3(1); NFPA 10 Table 6.2.1.1" },
  co2: {
    unit: "co2-2",
    text: "carbon dioxide (B:C) at the room's door; no dry powder on electronics; a 2-A water unit for paper nearby",
    clause: "NFPA 75-2020 9.3.1, 9.3.2; NFPA 10 (A2025) 5.5.4.6.2; NFPA 76-2024 8.6.3.1.3",
  },
  classB: { unit: "foam-9.5", text: "class B (foam, or B:C powder) at the room's door: 10-B within 9 m or 20-B within 15 m; 40-B or 80-B for extra hazard", clause: "SBC 801-CR-2018 Table 906.3(2); NFPA 10 6.3.2" },
  kitchen: { unit: "wet-chemical-6", text: "class K wet chemical within 9 m of the cooking equipment", clause: "SBC 801-CR-2018 904.12.5" },
  dwelling: { unit: "abc-2.3", text: "one extinguisher of at least 1-A:10-B:C in each dwelling unit, else in the common areas within 22.86 m", clause: "SBC 801-CR-2018 906.1 item 1 and its R-2 exception" },
});

const SPRINKLERS_KEPT = "NFPA 75-2024 9.1.1 and 9.4.1: in a sprinklered building the room keeps its sprinklers and the agent is a supplement; SBC 801-CR-2018 903.3.1.1.1: sprinklers are not omitted because a room holds electrical equipment";
const AGENT_CODES = "SBC 801-CR-2018 904.10 and IFC 2021 904.10: clean-agent systems to NFPA 2001; 904.3.2 to 904.3.4: automatic with a manual release, interlocks, alarms";
const FOAM_CODES = "SBC 801-CR-2018 904.7 and IFC 2021 904.7: foam systems to NFPA 11 (and NFPA 16); NFPA 11-2021 6.5.2.2.1: foam-water sprinklers at 6.5 mm/min";
const WATER_CODES = "SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1: sprinklers to NFPA 13";

/**
 * Room uses by name, first match wins (case-insensitive, whole words), EN
 * names as in the NIBS Office and Clinic (SERVER RM, TELE/COMM RM, ELEC.
 * CL., UPS/BATT RM, HAZ. STOR. ...) and some German and Arabic ones.
 */
const USES = [
  {
    use: "fuel",
    system: "foam",
    // "FLAMABLE STORAGE" is the NIBS Clinic's spelling.
    pattern: /generat|\bgen ?sets?\b|\bdiesel|\bfuel|\boil\b|day ?tank|\bflam+able|\bhaz(ardous|\.)? ?(mat(erials?)?\.? ?)?stor|\bpaint ?(stor|store)|\bsolvent|notstrom|emergency power|\bengine ?(room|rm)|مولد|وقود/i,
    reason: "a generator, fuel, oil or flammable-liquid room: a liquid pool fire, which water does not put out (water note 5); foam-water sprinklers",
    clause: `${FOAM_CODES}; NFPA 37-2021 11.4.1.1 (engine rooms: NFPA 11 among the permitted systems); SBC 801-CR-2018 903.3.1.1.1 item 3 (generator rooms with 2 h separation and detection may omit sprinklers), 5704.3.7.5.1 (foam-water in liquid storage rooms where approved)`,
    extinguisher: UNIT.classB,
  },
  {
    use: "it",
    system: "clean-agent",
    // A computer lab or classroom is a room of desks, not an IT equipment room (NFPA 75).
    pattern: /\bservers?\b|serverraum|\bcomputer\b(?!\s*(lab|class|training))|\bdata ?(cent(er|re)|processing|room)\b|\bADP\b|\bNOC\b|network op|\bIT ?(room|rm)\b|\b[MIB]DF\b/i,
    reason: "an information technology room (server, computer, data processing, network operations): HFC-227ea at the class C design concentration",
    clause: `NFPA 75-2024 9.4.1, 9.4.2: gaseous total flooding where data and return to service are critical, to NFPA 2001; ${AGENT_CODES}`,
    extinguisher: UNIT.co2,
  },
  {
    use: "telecom",
    system: "clean-agent",
    pattern: /\btele\b|\btelecom(m|munications?)?\b|\bcomm\b|\bcomms\b|\bcommunications?\b|fernmelde/i,
    reason: "a telecommunications room: HFC-227ea at the class C design concentration",
    clause: `NFPA 76-2020 A.8.6.2.2.1 (sprinklers not recommended in equipment areas; a telecom-only space with smoke detection and 1 h barriers needs none, SBC 801-CR-2018 903.2 exception); ${AGENT_CODES}`,
    extinguisher: UNIT.co2,
  },
  {
    use: "battery",
    system: "clean-agent",
    pattern: /\bUPS\b|\bbatt(ery|eries|\.)?\b|batterie|\bESS\b/i,
    reason: "a UPS or battery room: HFC-227ea at the class C design concentration puts out the flames, not a lithium-ion cell's runaway, so the sprinklers stay",
    clause: `IFC 2021 1207.5.5 (lithium-ion rooms: sprinklers at 12.2 mm/min); SBC 801-CR-2018 608.1, 608.9 (smoke detection); ${AGENT_CODES}`,
    extinguisher: UNIT.co2,
  },
  {
    use: "electrical",
    system: "clean-agent",
    pattern: /\belec(tric(al)?)?\b|elektro|switch ?(gear|room|rm)|\bswgr\b|\btransformer|\btrafo\b|\bsubstation|كهرباء/i,
    reason: "an electrical room or closet: HFC-227ea at the class C design concentration (above 480 V kept live, by hazard analysis, NFPA 2001-2022 7.2.2.4.2)",
    clause: `NFPA 2001-2022 7.2.2.4; SBC 801-CR-2018 903.3.1.1.1 item 3 (transformer rooms with 2 h separation and detection may omit sprinklers); ${AGENT_CODES}`,
    extinguisher: UNIT.co2,
  },
  {
    use: "elevator",
    system: "water",
    pattern: /\belev(ator)?\.? ?(eq(uip(ment)?)?\.?|mach(ine)?\.?) ?(rm|room)?|aufzug ?maschin/i,
    reason: "an elevator machine or equipment room: sprinklers, unless it serves a fire service access or occupant evacuation elevator and has detection",
    clause: `${WATER_CODES}; SBC 801-CR-2018 903.3.1.1.1 items 5 and 6`,
    extinguisher: UNIT.co2,
  },
  {
    use: "kitchen",
    system: "water",
    // Whole word: the Clinic's "KITCHENET" is a kitchenette, not commercial cooking.
    pattern: /\bkitchens?\b|k(ü|ue)che|مطبخ/i,
    reason: "a kitchen: sprinklers (ordinary hazard group 1); a commercial cooking hood takes its own wet-chemical system",
    clause: `${WATER_CODES}; SBC 801-CR-2018 904.2.2 and 904.12 (commercial cooking systems)`,
    extinguisher: UNIT.kitchen,
  },
  {
    use: "mechanical",
    system: "water",
    pattern: /\bmech(anical|\.)?\b|plant|\bAHU\b|boiler|pump ?(room|rm)|chiller|hvac|machine ?(room|rm)|technik|ميكانيك/i,
    reason: "a mechanical or plant room: sprinklers (ordinary hazard group 1)",
    clause: `${WATER_CODES}; NFPA 13-2022 annex A.4.3 (mechanical rooms ordinary hazard group 1)`,
    extinguisher: UNIT.powder,
  },
];

/** Group R occupancies (apartments, dwellings): water only. */
export const isResidential = (occupancy) => [].concat(occupancy ?? []).some((g) => /^R(-?[1-4])?$|resid|apartment|dwelling|\bflats?\b|villa|wohn/i.test(String(g).trim()));

/**
 * The suppression system for a room, from its name (or use) and the
 * building's occupancy.
 *
 * @param {object|string} room  a room record (sprinklers.js roomsFromSpaces: name, and `use` if known) or a name
 * @param {object} opts  occupancy: "R", "R-2", "apartment" ... (water only) or "B", "commercial" ...;
 *                       sprinklered (default true): the building has sprinklers;
 *                       agentOnly: the clean-agent rooms drop their sprinklers (the official's approval)
 * @returns {{system: "water"|"clean-agent"|"foam"|"none", reason: string, clause: string, use: string,
 *            sprinklers: boolean, design: object|null, extinguisher: object}}
 */
export function systemForRoom(room, { occupancy = null, sprinklered = true, agentOnly = false } = {}) {
  const name = typeof room === "string" ? room : `${room?.name ?? ""} ${room?.use ?? ""}`;
  const text = name.trim();
  if (room && typeof room === "object" && room.problem) {
    return { system: "none", use: "none", reason: `no room for a system: ${room.problem}`, clause: "n/a", sprinklers: false, design: null, extinguisher: null };
  }
  if (isResidential(occupancy)) {
    return {
      system: sprinklered ? "water" : "none",
      use: "dwelling",
      reason: sprinklered
        ? "an apartment building (Group R): water only, residential or quick-response sprinklers in dwelling and sleeping units"
        : "an apartment building without sprinklers: none (Group R requires them throughout, 903.2.8)",
      clause: "SBC 801-CR-2018 903.2.8 (sprinklers throughout a building with a Group R fire area), 903.3.1.2 (NFPA 13R up to four storeys and 18 m), 903.3.2 item 3 (quick-response or residential sprinklers in dwelling units)",
      sprinklers: sprinklered,
      design: null,
      extinguisher: UNIT.dwelling,
    };
  }
  for (const rule of USES) {
    if (!rule.pattern.test(text)) continue;
    const out = { system: rule.system, use: rule.use, reason: rule.reason, clause: rule.clause, sprinklers: rule.system === "water" ? sprinklered : false, design: null, extinguisher: rule.extinguisher };
    if (rule.system === "clean-agent") {
      out.sprinklers = sprinklered && !agentOnly;
      out.design = { agent: CLEAN_AGENT.agent, concentration: CLEAN_AGENT.design.C.value, hazard: "C", source: CLEAN_AGENT.design.C.source };
      out.reason += out.sprinklers ? "; the room keeps its sprinklers" : agentOnly ? "; agent only (sprinklers omitted with the official's approval)" : "; the building has no sprinklers";
      if (out.sprinklers) out.clause += `; ${SPRINKLERS_KEPT}`;
    } else if (rule.system === "foam") {
      const t = foamTimes(FOAM.rate.value);
      out.design = { rate: FOAM.rate.value, control: t.control, extinguish: t.extinguish, source: FOAM.rate.source };
    } else if (!sprinklered) {
      out.system = "none";
      out.reason += "; the building has no sprinklers";
    }
    return out;
  }
  return {
    system: sprinklered ? "water" : "none",
    use: "general",
    reason: text
      ? `${sprinklered ? "sprinklers" : "no automatic system (the building has no sprinklers)"}: the name names no special hazard (offices, corridors, clinic rooms, storage)`
      : `${sprinklered ? "sprinklers" : "no automatic system"}: the room has no name`,
    clause: WATER_CODES,
    sprinklers: sprinklered,
    design: null,
    extinguisher: UNIT.powder,
  };
}

/**
 * A system for every room, with a short report.
 *
 * @param {Array} rooms  sprinklers.js room records (name, floor, ceiling, area; problem when not a room)
 * @param {object} opts  occupancy, sprinklered, agentOnly (systemForRoom); temperature (°C, for the agent's mass)
 * @returns {{rooms: object[], counts: object, clauses: string[], agentKg: number, lines: string[]}}
 */
export function planSystems(rooms, { occupancy = null, sprinklered = true, agentOnly = false, temperature = 20 } = {}) {
  const out = [];
  const counts = { water: 0, "clean-agent": 0, foam: 0, none: 0 };
  const clauses = new Set();
  let agentKg = 0;
  for (const room of rooms || []) {
    const s = systemForRoom(room, { occupancy, sprinklered, agentOnly });
    const height = room?.ceiling > room?.floor ? room.ceiling - room.floor : null;
    const volume = height && room.area > 0 ? room.area * height : null;
    const entry = { id: room?.id ?? null, name: room?.name ?? null, ...s, volume };
    if (s.system === "clean-agent" && volume) {
      entry.agentKg = agentMass(volume, s.design.concentration, temperature);
      agentKg += entry.agentKg;
    }
    counts[s.system] += 1;
    if (s.clause && s.clause !== "n/a") clauses.add(s.clause);
    out.push(entry);
  }
  const lines = [
    `${out.length} rooms: ${counts.water} water, ${counts["clean-agent"]} clean agent (${CLEAN_AGENT.agent}), ${counts.foam} foam, ${counts.none} none`,
  ];
  if (isResidential(occupancy)) lines.push("apartments: water only (SBC 801-CR-2018 903.2.8); an extinguisher of at least 1-A:10-B:C in each dwelling (906.1 item 1)");
  const agentRooms = out.filter((r) => r.system === "clean-agent");
  if (agentRooms.length) {
    lines.push(
      `clean agent: ${CLEAN_AGENT.design.C.value} % (class C) in ${agentRooms.length} room${agentRooms.length === 1 ? "" : "s"}, ${agentKg.toFixed(0)} kg in all at ${temperature} °C; ` +
        `below the NOAEL of ${CLEAN_AGENT.noael.value} %; cross-zoned detection, ${CLEAN_AGENT.delay.value} s delay, ${CLEAN_AGENT.discharge.value} s discharge, ${CLEAN_AGENT.hold.value / 60} min hold`,
    );
  }
  const foamRooms = out.filter((r) => r.system === "foam");
  if (foamRooms.length) {
    const t = foamTimes();
    lines.push(`foam: foam-water sprinklers at ${FOAM.rate.value} L/(min m²) in ${foamRooms.length} room${foamRooms.length === 1 ? "" : "s"}; a pool controlled about ${t.control.toFixed(0)} s and out about ${t.extinguish.toFixed(0)} s after foam reaches it (derived)`);
  }
  const special = out.filter((r) => r.extinguisher && r.extinguisher !== UNIT.powder && r.extinguisher !== UNIT.dwelling);
  if (special.length) lines.push(`extinguishers: ${special.length} room${special.length === 1 ? "" : "s"} need a unit of their own at the door (CO2 at IT, telecom and electrical rooms; class B at fuel rooms; class K at kitchens), besides ABC powder on the routes`);
  return { rooms: out, counts, clauses: [...clauses], agentKg, lines };
}

/**
 * The portable units the special rooms need of their own, beside the ABC
 * powder that extinguishers.js layoutExtinguishers spreads over the walking
 * routes (22.86 m of travel, SBC 801-CR-2018 Table 906.3(1)): CO2 at IT,
 * telecom, electrical, battery and elevator machine rooms; class B at fuel
 * and flammable-liquid rooms; class K at kitchens. One per room, at the room
 * (its centre at 1 m, viewer frame; the interface snaps it to the room's
 * door on the walk). Apartments get one unit per dwelling instead (906.1
 * item 1), which this list does not place.
 *
 * @param {object} plan   planSystems' result
 * @param {Array} rooms   the same rooms, in the same order
 * @returns {Array<{room, name, unit: string, text: string, clause: string, position: number[]|null}>}
 */
export function specialExtinguishers(plan, rooms, { upAxis = "y" } = {}) {
  const [h0, h1] = planAxes(upAxis);
  const up = AXIS[upAxis] ?? 1;
  const out = [];
  plan.rooms.forEach((p, i) => {
    const x = p.extinguisher;
    if (!x || x === UNIT.powder || x === UNIT.dwelling) return;
    const r = rooms[i];
    let position = null;
    if (r?.rect && Number.isFinite(r.floor)) {
      position = [0, 0, 0];
      position[h0] = r.rect.centre[0];
      position[h1] = r.rect.centre[1];
      position[up] = r.floor + 1;
    }
    out.push({ room: p.id, name: p.name, unit: x.unit, text: x.text, clause: x.clause, position });
  });
  return out;
}

/**
 * A rectangular room record, as roomsFromSpaces makes them, for synthetic
 * rooms (a generator room the models lack, tests): plan corners in the
 * viewer's plan axes (x and z for y up), floor and ceiling heights.
 */
export function rectangleRoom({ id, name, x0, z0, x1, z1, floor, ceiling, upAxis = "y" }) {
  const footprint = Float64Array.from([x0, z0, x1, z0, x1, z1, x0, z0, x1, z1, x0, z1]);
  const a = Math.abs(x1 - x0), b = Math.abs(z1 - z0);
  const long = a >= b;
  return {
    id, name, upAxis, floor, ceiling, footprint, area: a * b, problem: null,
    rect: { centre: [(x0 + x1) / 2, (z0 + z1) / 2], axis: long ? [1, 0] : [0, 1], half: long ? [a / 2, b / 2] : [b / 2, a / 2] },
  };
}

// ------------------------------------------------------------------ on the fire grid

const AXIS = { x: 0, y: 1, z: 2 };
const planAxes = (upAxis) => [0, 1, 2].filter((i) => i !== (AXIS[upAxis] ?? 1));

function inTriangle(T, k, x, y) {
  const d1 = (x - T[k + 2]) * (T[k + 1] - T[k + 3]) - (T[k] - T[k + 2]) * (y - T[k + 3]);
  const d2 = (x - T[k + 4]) * (T[k + 3] - T[k + 5]) - (T[k + 2] - T[k + 4]) * (y - T[k + 5]);
  const d3 = (x - T[k]) * (T[k + 5] - T[k + 1]) - (T[k + 4] - T[k]) * (y - T[k + 1]);
  const neg = d1 < 0 || d2 < 0 || d3 < 0;
  const pos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(neg && pos);
}

/**
 * A room's cells on the grid: the open cells whose centres are inside its
 * footprint (plan triangles) and between its floor and ceiling; with the
 * columns they stand in (top cell under the ceiling first). `room.cells`
 * (grid cells) is taken as given.
 *
 * @returns {{cells: Int32Array, columns: Array<{i: number, k: number, cells: number[]}>}}
 */
export function roomCells(grid, solid, room, upAxis = "y") {
  const g = grid;
  const at = (i, j, k) => i + g.nx * (j + g.ny * k);
  const columns = new Map();
  const add = (c) => {
    const i = c % g.nx, k = Math.floor(c / (g.nx * g.ny));
    const key = i + g.nx * k;
    if (!columns.has(key)) columns.set(key, { i, k, cells: [] });
    columns.get(key).cells.push(c);
  };
  if (room.cells) {
    for (const c of room.cells) if (c >= 0 && c < g.nx * g.ny * g.nz && !solid[c]) add(c);
  } else if (room.footprint?.length >= 6 && Number.isFinite(room.floor) && Number.isFinite(room.ceiling)) {
    const [h0, h1] = planAxes(upAxis);
    const T = room.footprint;
    // The footprint's box in the grid's local plan axes (0 and 2).
    const local = (a, b) => {
      const w = [0, 0, 0];
      w[h0] = a;
      w[h1] = b;
      return [w[g.axes[0]], w[g.axes[2]]];
    };
    let i0 = Infinity, i1 = -Infinity, k0 = Infinity, k1 = -Infinity;
    for (let q = 0; q < T.length; q += 2) {
      const [a, c] = local(T[q], T[q + 1]);
      i0 = Math.min(i0, (a - g.origin[0]) / g.h);
      i1 = Math.max(i1, (a - g.origin[0]) / g.h);
      k0 = Math.min(k0, (c - g.origin[2]) / g.h);
      k1 = Math.max(k1, (c - g.origin[2]) / g.h);
    }
    const jLo = Math.max(0, Math.ceil((room.floor - g.origin[1]) / g.h - 0.5 + 1e-9));
    const jHi = Math.min(g.ny - 1, Math.floor((room.ceiling - g.origin[1]) / g.h - 0.5 - 1e-9));
    const world = [0, 0, 0];
    for (let k = Math.max(0, Math.floor(k0)); k <= Math.min(g.nz - 1, Math.floor(k1)); k += 1) {
      for (let i = Math.max(0, Math.floor(i0)); i <= Math.min(g.nx - 1, Math.floor(i1)); i += 1) {
        world[g.axes[0]] = g.origin[0] + (i + 0.5) * g.h;
        world[g.axes[2]] = g.origin[2] + (k + 0.5) * g.h;
        const x = world[h0], y = world[h1];
        let inside = false;
        for (let t = 0; t < T.length && !inside; t += 6) inside = inTriangle(T, t, x, y);
        if (!inside) continue;
        // Top down, so each column's first cell is the one under the ceiling.
        for (let j = jHi; j >= jLo; j -= 1) {
          const c = at(i, j, k);
          if (!solid[c]) add(c);
        }
      }
    }
  }
  const list = [...columns.values()];
  // Each column's cells top down.
  for (const col of list) col.cells.sort((a, b) => b - a);
  return { cells: Int32Array.from(list.flatMap((col) => col.cells)), columns: list };
}

/**
 * Smoke detectors under a room's ceiling, in `zones` interleaved zones: the
 * centres of an na x nb split of the room's rectangle (or its columns' box),
 * at most DETECTOR_SPACING / 2 apart, alternating zone by zone like a
 * chessboard, so each zone alone is within the nominal spacing; each snapped
 * to the nearest of the room's columns, read at that column's top cell.
 *
 * @returns {{cells: Int32Array, zone: Uint8Array}}
 */
export function roomDetectors(grid, columns, room, { zones = 2, spacing = DETECTOR_SPACING.value, upAxis = "y" } = {}) {
  const g = grid;
  if (!columns.length) return { cells: new Int32Array(0), zone: new Uint8Array(0) };
  // The rectangle in the grid's local plan (i, k as reals).
  let centre, ax, half;
  if (room?.rect && room.footprint) {
    const [h0, h1] = planAxes(upAxis);
    const toLocal = (a, b) => {
      const w = [0, 0, 0];
      w[h0] = a;
      w[h1] = b;
      return [(w[g.axes[0]] - g.origin[0]) / g.h, (w[g.axes[2]] - g.origin[2]) / g.h];
    };
    const c = toLocal(room.rect.centre[0], room.rect.centre[1]);
    const e = toLocal(room.rect.centre[0] + room.rect.axis[0], room.rect.centre[1] + room.rect.axis[1]);
    centre = c;
    ax = [e[0] - c[0], e[1] - c[1]];
    const n = Math.hypot(ax[0], ax[1]) || 1;
    ax = [ax[0] / n, ax[1] / n];
    half = [room.rect.half[0] / g.h, room.rect.half[1] / g.h];
  } else {
    let i0 = Infinity, i1 = -Infinity, k0 = Infinity, k1 = -Infinity;
    for (const col of columns) {
      i0 = Math.min(i0, col.i);
      i1 = Math.max(i1, col.i + 1);
      k0 = Math.min(k0, col.k);
      k1 = Math.max(k1, col.k + 1);
    }
    const long = i1 - i0 >= k1 - k0;
    centre = [(i0 + i1) / 2, (k0 + k1) / 2];
    ax = long ? [1, 0] : [0, 1];
    half = long ? [(i1 - i0) / 2, (k1 - k0) / 2] : [(k1 - k0) / 2, (i1 - i0) / 2];
  }
  const step = Math.max(spacing / 2, 0.5) / g.h;
  let na = Math.max(1, Math.ceil((2 * half[0]) / step - 1e-9));
  const nb = Math.max(1, Math.ceil((2 * half[1]) / step - 1e-9));
  if (zones > 1 && na * nb < zones) na = zones;
  const perp = [-ax[1], ax[0]];
  const picked = new Map();
  for (let b = 0; b < nb; b += 1) {
    for (let a = 0; a < na; a += 1) {
      const u = ((a + 0.5) / na - 0.5) * 2 * half[0];
      const v = ((b + 0.5) / nb - 0.5) * 2 * half[1];
      const pi = centre[0] + u * ax[0] + v * perp[0];
      const pk = centre[1] + u * ax[1] + v * perp[1];
      let best = null, bd = Infinity;
      for (const col of columns) {
        const d = (col.i + 0.5 - pi) ** 2 + (col.k + 0.5 - pk) ** 2;
        if (d < bd) [best, bd] = [col, d];
      }
      const cell = best.cells[0];
      const zone = zones > 1 ? (a + b) % zones : 0;
      if (!picked.has(cell)) picked.set(cell, zone);
    }
  }
  const cells = [...picked.keys()];
  const zone = [...picked.values()];
  // A zone left with no detector (a tiny room) reads the first one too.
  for (let z = 0; z < zones; z += 1) {
    if (!zone.includes(z)) {
      cells.push(cells[0]);
      zone.push(z);
    }
  }
  return { cells: Int32Array.from(cells), zone: Uint8Array.from(zone) };
}

/** Floats per room in the state the solver reads: agent %, class A and B extinguishing %, foam's share of evaporation left, foam's wetting share, then padding. */
export const STATE = Object.freeze({ agent: 0, extA: 1, extB: 2, blanket: 3, wet: 4, stride: 8 });
/** A concentration no fire reaches (Infinity is avoided: WGSL may assume finite floats). */
const NEVER = 3e38;

/**
 * The rooms of a run on the fire grid: each room's cells (a cell in two
 * rooms goes to the first), its detectors and zones, its parameters, and
 * the arrays the solver reads (`room`, `state`, the foam's wet list).
 *
 * @param {object} scene  buildScene's (grid, solid)
 * @param {{rooms: Array}} systems  rooms: {id, name, system: "clean-agent"|"foam", footprint+floor+ceiling or cells,
 *          params?: {design, extA, extB, discharge, hold, retained, delay, zones, releaseAt, rate, control,
 *          extinguish, duration, fluorineFree, wet}}; rooms of other systems are left out
 * @param {object} opts  upAxis; fuelMass (the solver's, for the foam's wet cells)
 */
export function systemsSetup(scene, systems, { upAxis = "y", fuelMass = null } = {}) {
  const g = scene.grid;
  const N = g.nx * g.ny * g.nz;
  const roomOf = new Uint16Array(N);
  const records = [];
  const detectorCells = [];
  const wet = [], wetRoom = [], wetFlux = [];
  const wetSeen = new Set();
  for (const room of systems?.rooms || []) {
    if (room?.system !== "clean-agent" && room?.system !== "foam") continue;
    if (records.length >= 65534) break;
    const r = records.length + 1;
    const { columns } = roomCells(g, scene.solid, room, upAxis);
    let count = 0;
    const kept = [];
    for (const col of columns) {
      const own = col.cells.filter((c) => roomOf[c] === 0);
      for (const c of own) roomOf[c] = r;
      count += own.length;
      if (own.length) kept.push({ ...col, cells: own });
    }
    const p = room.params || {};
    const agent = room.system === "clean-agent";
    const zones = p.zones ?? (agent ? CLEAN_AGENT.zones.value : FOAM.zones.value);
    const det = roomDetectors(g, kept, room, { zones, upAxis });
    const rec = {
      id: room.id ?? null,
      name: room.name ?? null,
      system: room.system,
      r,
      cells: count,
      volume: count * g.h ** 3,
      detectors: det,
      detectorFrom: detectorCells.length,
      zones,
      delay: p.delay ?? (agent ? CLEAN_AGENT.delay.value : FOAM.delay.value),
      releaseAt: Number.isFinite(p.releaseAt) ? p.releaseAt : null,
      detectedAt: null,
      t0: Number.isFinite(p.releaseAt) ? p.releaseAt : Infinity,
      stats: { oxBlocked: 0, flameSteps: 0, withheld: 0, wetted: 0 },
    };
    if (agent) {
      rec.design = p.design ?? CLEAN_AGENT.design.C.value;
      rec.extA = p.extA ?? CLEAN_AGENT.mecClassA.value;
      rec.extB = p.extB ?? CLEAN_AGENT.mecHeptane.value;
      rec.discharge = Math.max(1e-6, p.discharge ?? CLEAN_AGENT.discharge.value);
      rec.hold = p.hold ?? CLEAN_AGENT.hold.value;
      rec.retained = p.retained ?? 1;
      rec.agentKg = agentMass(rec.volume, rec.design, p.temperature ?? 20);
    } else {
      const times = foamTimes(p.rate ?? FOAM.rate.value, { fluorineFree: p.fluorineFree });
      rec.rate = p.rate ?? FOAM.rate.value;
      rec.control = Math.max(1e-6, p.control ?? times.control);
      rec.extinguish = Math.max(rec.control, p.extinguish ?? times.extinguish);
      rec.duration = p.duration ?? FOAM.duration.value;
      rec.wet = p.wet ?? WETTED_FRACTION;
      // The solution lands on the first fuel cell down each column (the
      // spray's columns end on the fuel's top surface the same way).
      const flux = ((rec.rate / 60) * g.h * g.h); // kg/s per column (1 L a kg)
      let wetCells = 0;
      for (const col of kept) {
        const c = fuelMass ? col.cells.find((q) => fuelMass[q] > 0) : undefined;
        if (c === undefined || wetSeen.has(c)) continue;
        wetSeen.add(c);
        wet.push(c);
        wetRoom.push(r);
        wetFlux.push(flux);
        wetCells += 1;
      }
      rec.wetCells = wetCells;
    }
    for (const c of det.cells) detectorCells.push(c);
    records.push(rec);
  }
  if (!records.length) return null;
  const R = records.length;
  const state = new Float32Array((R + 1) * STATE.stride);
  state[STATE.extA] = NEVER;
  state[STATE.extB] = NEVER;
  state[STATE.blanket] = 1;
  return {
    R,
    rooms: records,
    room: roomOf,
    state,
    wet: Int32Array.from(wet),
    wetRoom: Uint16Array.from(wetRoom),
    wetFlux: Float32Array.from(wetFlux),
    detectorCells: Int32Array.from(detectorCells),
    update: (t) => systemsState(records, state, t),
  };
}

/** A room's agent concentration (%) at time t: a linear rise over the discharge, the hold, then none. */
export function agentAt(rec, t) {
  if (!(t >= rec.t0)) return 0;
  const full = rec.t0 + rec.discharge;
  if (t < full) return (rec.design * (t - rec.t0)) / rec.discharge;
  if (t >= full + rec.hold) return 0;
  // Leakage ignored by default (retained 1); otherwise a well-mixed decay to `retained` at the hold's end.
  return rec.retained < 1 ? rec.design * rec.retained ** ((t - full) / Math.max(rec.hold, 1e-9)) : rec.design;
}

/** A room's share of its liquid fuel's evaporation left under the foam at time t: 1, to 0.1 at control, to 0 when out. */
export function blanketAt(rec, t) {
  if (!(t >= rec.t0)) return 1;
  const s = t - rec.t0;
  if (s < rec.control) return 1 - (0.9 * s) / rec.control;
  if (s < rec.extinguish) return 0.1 * (1 - (s - rec.control) / Math.max(rec.extinguish - rec.control, 1e-9));
  return 0;
}

/** The per-room state at time t (the start of a step), into `state`. */
export function systemsState(records, state, t) {
  for (const rec of records) {
    const o = rec.r * STATE.stride;
    if (rec.system === "clean-agent") {
      state[o + STATE.agent] = agentAt(rec, t);
      state[o + STATE.extA] = rec.extA;
      state[o + STATE.extB] = rec.extB;
      state[o + STATE.blanket] = 1;
      state[o + STATE.wet] = 0;
    } else {
      state[o + STATE.agent] = 0;
      state[o + STATE.extA] = NEVER;
      state[o + STATE.extB] = NEVER;
      state[o + STATE.blanket] = blanketAt(rec, t);
      state[o + STATE.wet] = t >= rec.t0 && t < rec.t0 + rec.duration ? rec.wet : 0;
    }
  }
  return state;
}

/**
 * One frame of detection: the visibility (m) at every room's detectors at
 * the frame's end. A room whose every zone has a detector at or below the
 * C/VM2 response (DETECTOR_VISIBILITY, as the run's ceiling detectors) is
 * detected at the frame's end and discharges after its delay (a manual
 * release time, when set, stands). Returns the rooms detected this frame.
 */
export function systemsDetect(sys, { t1, vis, limit = DETECTOR_VISIBILITY }) {
  const detected = [];
  for (const rec of sys.rooms) {
    if (rec.detectedAt !== null) continue;
    const seen = new Uint8Array(rec.zones);
    const { cells, zone } = rec.detectors;
    for (let q = 0; q < cells.length; q += 1) if (vis[rec.detectorFrom + q] <= limit) seen[zone[q]] = 1;
    if (!cells.length || seen.some((s) => !s)) continue;
    rec.detectedAt = t1;
    if (rec.releaseAt === null) rec.t0 = t1 + rec.delay;
    detected.push({ id: rec.id, name: rec.name, system: rec.system, detectedAt: t1, dischargeAt: rec.t0 });
  }
  return detected;
}

/**
 * What each room's system did, for the run's card: when it was detected and
 * discharged, and for the agent when it flooded the room, reached each
 * extinguishing concentration on its rise and ended; for the foam when the
 * pool was controlled and out. `stats` (CPU only; `cpu` false: the GPU keeps
 * no per-room tallies): oxBlocked, the solid oxidation the agent stopped,
 * kg summed step by step (each step's would-be amount, so it can exceed the
 * fuel there); flameSteps, cell-steps whose flame the agent stopped;
 * withheld, the pool evaporation the foam stopped, kg summed step by step;
 * wetted, kg of the foam solution's water taken up by class A fuel.
 */
export function systemsSummary(sys, { cpu = true } = {}) {
  return sys.rooms.map((rec) => {
    const fired = Number.isFinite(rec.t0);
    const base = {
      id: rec.id,
      name: rec.name,
      system: rec.system,
      cells: rec.cells,
      volume: rec.volume,
      detectors: rec.detectors.cells.length,
      zones: rec.zones,
      detectedAt: rec.detectedAt,
      dischargeAt: fired ? rec.t0 : null,
      manual: rec.releaseAt !== null,
      stats: cpu ? { ...rec.stats } : null,
    };
    if (rec.system === "clean-agent") {
      return {
        ...base,
        agent: CLEAN_AGENT.agent,
        design: rec.design,
        extinguishing: { A: rec.extA, B: rec.extB },
        agentKg: rec.agentKg,
        floodedAt: fired ? rec.t0 + rec.discharge : null,
        holdEnds: fired ? rec.t0 + rec.discharge + rec.hold : null,
        // When the agent first reached each extinguishing concentration on its rise.
        reachesA: fired && rec.design >= rec.extA ? rec.t0 + (rec.discharge * rec.extA) / rec.design : null,
        reachesB: fired && rec.design >= rec.extB ? rec.t0 + (rec.discharge * rec.extB) / rec.design : null,
      };
    }
    return {
      ...base,
      rate: rec.rate,
      controlAt: fired ? rec.t0 + rec.control : null,
      outAt: fired ? rec.t0 + rec.extinguish : null,
      foamEnds: fired ? rec.t0 + rec.duration : null,
      wetCells: rec.wetCells,
    };
  });
}
