/**
 * Fire classes for the first item to burn, and what each means for the
 * Flamescapes solver (src/core/flamescapes.js) and for the occupant. Every
 * value carries its source key; the keys resolve in SOURCES below and in
 * wiki/research/ignition-sources-2026-11-05.json (the review, with every DOI
 * checked against Crossref on 2026-11-05).
 *
 * The classes are NFPA 10's (A ordinary combustibles, B flammable liquids
 * and gases, C energised electrical equipment, D combustible metals, K
 * cooking oils and fats), which the Saudi Fire Code (SBC 801, after the
 * International Fire Code) uses, plus ISO 3941:2026's class L, lithium-ion
 * cells and batteries. `iso` gives the ISO 3941 / EN 2 class (A solids, B
 * liquids, C gases, D metals, F cooking media, L lithium-ion; electrical is
 * not a class there). Class C names the hazard, not a fuel: once the
 * equipment burns, its fuel is `fuelClass`.
 *
 * Per class, the solver's fuel:
 *   heatOfCombustion  J per kg burnt (effective)
 *   yields            kg of CO2, CO and soot per kg burnt, well ventilated.
 *                     CO2 is the carbon balance of a representative fuel:
 *                     44/12 (w_C - 12/28 y_CO - y_soot), w_C its carbon mass
 *                     fraction (cellulose 0.444, n-heptane 0.839, polypropylene
 *                     0.857, triolein 0.774); the sources give CO and soot.
 *   o2, ash           a metal's own oxygen and solid oxide (kg per kg), when
 *                     not 417 kJ / mol O2 (Fire-X Eq. 4; Huggett 1980)
 *   heatCapacity      J / (kg K) of the fuel, where it is not wood-like (metals)
 *   liquid            {boiling (K), latent (J/kg)}: evaporates past `boiling`
 *   embers            whether the burning item throws embers
 *   ignitionTemperature  degrees C, for spot ignition by embers
 *   growth            the t-squared class the sources give (for the panel and
 *                     the two-zone model; the solver grows the fire itself)
 *   item              the first item: mass (kg) and exposed area (m^2), ours
 *   water             what sprinkler water does to the item (waterResponse;
 *                     wiki/research/water-by-fire-class-2026-11-07.md): whether
 *                     it soaks in and cools the surface ("wets"), whether
 *                     Evans's cap, measured on wetted furniture and cribs,
 *                     applies ("evans"), and whether the item holds no water
 *                     of its own ("dry"); a note for the panel
 *
 * What the solver does not model: CO rising to 0.2 kg/kg under-ventilated
 * (gannbryner), hydrogen halides from appliances (hietaniemi2001vtt) and HF
 * from lithium-ion packs (larsson2017), jet flames, and re-ignition.
 */

/** CO2 per kg burnt from a fuel's carbon mass fraction, after CO and soot. */
export function co2FromCarbon(carbon, co, soot) {
  return (44 / 12) * Math.max(0, carbon - (12 / 28) * co - soot);
}

/**
 * What sprinkler water does to a burning item (water-by-fire-class-2026-11-07.md):
 * sprinkler drops are too large to put a flame out in the gas (Rasbash 1986);
 * water works by cooling the burning surface below its fire point, so only
 * where it reaches the surface and the fire point is above the water's
 * temperature. Solids (A, and C once the circuit is dead) are wetted and
 * take Evans's cap; a liquid lighter than water with a fire point below it
 * (heptane), cooking oil (it erupts), metals (water feeds them hydrogen) and
 * lithium-ion packs (water does not reach the cells) are neither, and hold
 * none of wood's water.
 */
const NOT_WETTED = { wets: false, evans: false, dry: true };
export function waterResponse(fuel) {
  const w = fuel?.water || {};
  return { wets: w.wets !== false, evans: w.evans !== false, dry: w.dry === true, note: w.note || null };
}

const fuel = (heatMJ, carbon, co, soot, extra = {}) => ({
  heatOfCombustion: heatMJ * 1e6,
  yields: { co2: co2FromCarbon(carbon, co, soot), co, soot },
  ...extra,
});

export const FIRE_CLASSES = {
  A: {
    label: "Class A: ordinary combustibles",
    short: "A",
    iso: "A",
    examples: "Upholstered furniture, mattresses and bedding, wood, paper and cardboard, clothing and curtains, plastic casings",
    fuel: fuel(18.0, 0.444, 0.01, 0.015, { embers: true, ignitionTemperature: 390 }),
    growth: { class: "medium", alpha: 0.0117, note: "dwellings and offices; upholstered furniture alone is fast" },
    item: { mass: 20, area: 2 },
    agents: {
      suitable: ["Water", "Foam rated for class A", "Multipurpose (ABC) dry chemical", "Wet chemical"],
      unsuitable: [{ agent: "Carbon dioxide", reason: "Listed for classes B and C only; short range and no cooling of deep-seated embers." }],
    },
    sources: {
      heatOfCombustion: ["babrauskas1982nbsir", "lawson1984nbsir"],
      co: ["gannbryner", "mckenna2016"],
      soot: ["klotemilke2002", "nureg1805s1"],
      ignitionTemperature: ["nureg1805s1"],
      growth: ["en1991-1-2", "babrauskas1982nbsir"],
      embers: ["manzello2020"],
      agents: ["nfpa10"],
    },
  },
  B: {
    label: "Class B: flammable liquids and gases",
    short: "B",
    iso: "B (liquids), C (gases)",
    examples: "Petrol or another accelerant, solvents and paints, heating oil, gas at burners and boilers",
    // n-heptane, the standard test liquid: boils at 98.4 C, 0.317 MJ/kg to evaporate,
    // and burns at most m''_inf (1 - exp(-k beta D)) per m^2 of pool, with
    // m''_inf 0.101 kg/(m^2 s) and k beta 1.1 /m (3.2 MW for this 1 m^2 pool).
    fuel: fuel(44.1, 0.839, 0.022, 0.072, {
      liquid: { boiling: 371.6, latent: 3.17e5, mInf: 0.101, kBeta: 1.1 },
      embers: false,
      ignitionTemperature: 250,
      water: {
        ...NOT_WETTED,
        note: "Heptane's fire point is below the water's temperature and it is lighter than water: the water sinks through the pool, which keeps burning. Sprinklers protect what is round it; putting it out takes foam (Rasbash 1986).",
      },
    }),
    growth: { class: "ultra-fast", alpha: 0.1876, note: "a spill or pool; gasoline burns at 0.055 kg/m2s in large pools" },
    item: { mass: 4, area: 1 },
    agents: {
      suitable: ["Foam (floats on the fuel and prevents re-ignition)", "Carbon dioxide", "Dry chemical"],
      unsuitable: [{ agent: "Water jet", reason: "Splashes and spreads the burning liquid; drops sinking into hot liquid boil and throw it out." }],
    },
    sources: {
      heatOfCombustion: ["hu2021tn2133"],
      co: ["hu2021tn2133", "gannbryner"],
      soot: ["hu2021tn2133"],
      liquid: ["nistwebbookHeptane", "babrauskas1983pool", "nureg1805s1"],
      growth: ["babrauskas1983pool", "nureg1805s1"],
      agents: ["nfpa10"],
    },
    // Flash and autoignition temperatures were not read; 250 C is ours, for embers only.
    provisional: ["ignitionTemperature"],
  },
  C: {
    label: "Class C: energised electrical equipment",
    short: "C",
    iso: "none (electrical is not an ISO 3941 class; the burning plastics are A)",
    examples: "Refrigerator-freezers, dishwashers, washing machines and dryers, televisions and electronics, distribution boards",
    fuel: fuel(30.3, 0.857, 0.056, 0.056, {
      embers: false,
      ignitionTemperature: 390,
      water: {
        note: "What burns is the equipment's plastic, wetted and capped here as class A; while the circuit is live, water puts the people using it at risk of a shock, and the item can re-ignite until it is cut off.",
      },
    }),
    fuelClass: "A",
    growth: { class: "medium", alpha: 0.0117, note: "refrigerators slow, televisions fast, after minutes of smouldering" },
    item: { mass: 10, area: 1 },
    agents: {
      suitable: ["De-energise the circuit first", "Carbon dioxide", "Clean agent", "Dry chemical"],
      unsuitable: [{ agent: "Water or foam on live equipment", reason: "Conducts electricity to the person holding it." }],
    },
    sources: {
      heatOfCombustion: ["hietaniemi2001vtt"],
      co: ["hietaniemi2001vtt"],
      soot: ["hietaniemi2001vtt"],
      growth: ["hietaniemi2001vtt"],
      agents: ["nfpa10"],
    },
    // Ignition is by overheating inside the equipment; 390 C is class A's, for embers only.
    provisional: ["ignitionTemperature"],
  },
  D: {
    label: "Class D: combustible metals",
    short: "D",
    iso: "D",
    examples: "Magnesium parts and swarf, sodium and potassium in laboratories, lithium metal primary cells (not lithium-ion)",
    // Mg + 1/2 O2 -> MgO: 601.6 kJ / mol over 24.305 g / mol; the oxide stays as ash.
    // Magnesium's own specific heat, 24.87 J/(mol K) over 24.305 g/mol.
    fuel: {
      heatOfCombustion: 24.75e6,
      o2: 0.658,
      ash: 1.658,
      heatCapacity: 1023,
      yields: { co2: 0, co: 0, soot: 0 },
      embers: false,
      ignitionTemperature: 473,
      water: {
        ...NOT_WETTED,
        note: "Water on burning magnesium gives hydrogen and heat (17.5 MJ and 0.11 kg of hydrogen per kg of water reacted): it feeds the fire and risks an explosion. The run credits the water nothing here; the reaction is not modelled.",
      },
    },
    growth: { class: null, alpha: null, note: "no t-squared characterisation found" },
    item: { mass: 2, area: 0.2 },
    agents: {
      suitable: ["Class D dry powder", "Dry sand, applied gently so the metal does not scatter"],
      unsuitable: [
        { agent: "Water", reason: "Releases hydrogen from the burning metal; the reaction can be explosive." },
        { agent: "Foam", reason: "Water-based: the same reaction." },
        { agent: "Carbon dioxide", reason: "Burning magnesium keeps reacting with it." },
      ],
    },
    sources: { heatOfCombustion: ["nistwebbookMgO"], heatCapacity: ["nistwebbookMg"], ignitionTemperature: ["cameo", "zhao2022"], agents: ["nfpa10", "cameo"] },
  },
  K: {
    label: "Class K: cooking oils and fats",
    short: "K",
    iso: "F",
    examples: "Oil or fat in a pan on a hob, a deep-fat fryer, grease in an oven or range hood",
    // Oil gives off vapour and autoignites near 390 C (canola 385, butter 429 C).
    // No pool burning rate for cooking oil was read: transformer oil's, the
    // nearest row of the SFPE table (0.039 kg/(m^2 s), k beta 0.7 /m), stands
    // in. For this pan it gives about 19 kW, the low end of the 20 to 110 kW
    // that pan fires reach (growth, below).
    fuel: fuel(37.4, 0.774, 0.021, 0.069, {
      liquid: { boiling: 663, latent: 2.5e5, mInf: 0.039, kBeta: 0.7 },
      embers: false,
      ignitionTemperature: 390,
      water: {
        ...NOT_WETTED,
        note: "Water on oil above about 210 C turns to steam under it and throws burning oil out (a NIST residential sprinkler on a cooktop fire put out one of two, spreading oil both times). The run credits the water nothing on the pan; the eruption is not modelled.",
      },
    }),
    growth: { class: "ultra-fast", alpha: 0.1876, note: "a pan fire peaks at 20 to 110 kW within 40 s and spreads to cabinets ultra-fast" },
    item: { mass: 2, area: 0.07 },
    agents: {
      suitable: ["Wet chemical (class K / F)", "Fire blanket", "Turn off the heat; lid on the pan"],
      unsuitable: [
        { agent: "Water", reason: "Sinks into the hot oil, boils and throws burning oil out." },
        { agent: "Carbon dioxide", reason: "Knocks the flame down but does not cool the oil, which re-ignites." },
      ],
    },
    sources: {
      heatOfCombustion: ["hu2021tn2133"],
      co: ["hu2021tn2133"],
      soot: ["hu2021tn2133", "hu2021jfs"],
      ignitionTemperature: ["hu2021tn2133", "hostikka2023"],
      liquid: ["nureg1805s1"],
      growth: ["hamins2018", "hostikka2023"],
      agents: ["nfpa10", "liu2005fss"],
    },
    // The latent heat of the oil's vapour release is ours; its burning rate is transformer oil's.
    provisional: ["liquid.latent", "liquid.mInf", "liquid.kBeta"],
  },
  L: {
    label: "Class L: lithium-ion batteries",
    short: "L",
    iso: "L (ISO 3941:2026)",
    examples: "E-bike and e-scooter packs, power banks, phones and laptops, battery storage",
    // Not a single fuel: heat scales with stored energy; burnt as appliance plastics here.
    fuel: fuel(30.3, 0.857, 0.056, 0.056, {
      embers: false,
      ignitionTemperature: 300,
      water: {
        ...NOT_WETTED,
        note: "Ceiling water does not reach the cells inside a pack: it cools what is round it and slows spread, but does not stop the burning pack, which can re-ignite minutes to days later (FM Global; Xu et al. 2022).",
      },
    }),
    growth: { class: "ultra-fast", alpha: 0.1876, note: "an e-scooter in a closed bedroom reached flashover within 30 s of visible smoke" },
    item: { mass: 3, area: 0.3 },
    agents: {
      suitable: ["Leave and call the fire service", "Sustained cooling with water once safe; packs re-ignite"],
      unsuitable: [],
    },
    sources: {
      ignitionTemperature: ["liu2025ebike"],
      growth: ["fleischmann2025", "liu2025ebike"],
      toxic: ["larsson2017"],
      class: ["iso3941-2026"],
    },
    // No peer-reviewed comparison of agents was read; the fuel stands in for the pack's plastics.
    provisional: ["fuel", "agents"],
  },
};

export const FIRE_CLASS_KEYS = Object.keys(FIRE_CLASSES);

/** The solver's fuel and first item for a class (the room contents are always class A). */
export function fuelsFor(cls = "A") {
  const first = FIRE_CLASSES[cls] || FIRE_CLASSES.A;
  return { contentsFuel: FIRE_CLASSES.A.fuel, ignitionFuel: first.fuel, item: first.item };
}

/** The t-squared growth classes, alpha in kW/s^2: 1055 kW at 600, 300, 150 and 75 s (NFPA 72 Annex B; zone.js GROWTH). */
export const GROWTH_CLASSES = Object.freeze({ slow: 0.00293, medium: 0.0117, fast: 0.0469, "ultra-fast": 0.188 });

/**
 * The first item's design fire for a growth choice (2026-11-08): "class",
 * the default, takes the fire class's own (`growth` above: medium for A and
 * C, after EN 1991-1-2 for dwellings and offices); a class name takes that
 * class; "solver" none, the solver burning the item itself as before (it
 * reached about 500 kW within 15 s, faster than any design fire). Only solids
 * that water wets have one (classes A and C): a pool keeps its burning rate,
 * and metal and battery packs their own. Returns {alpha (kW/s^2), growth}
 * or null.
 */
export function designFireFor(cls = "A", growth = "class") {
  const first = FIRE_CLASSES[cls] || FIRE_CLASSES.A;
  // A pool (classes B and K) is a design fire too since 2026-11-12, its
  // growth capped at its steady large-pool burning rate (itemBurner): on the
  // coarse grid its own flame cannot hold its surface at boiling, so a pan
  // lit by its pilot went out when the pilot stopped. Other fuels water does
  // not wet (metals, batteries) keep the solver's burning.
  if (growth === "solver" || (!first.fuel.liquid && !waterResponse(first.fuel).wets)) return null;
  const name = growth === "class" ? first.growth?.class : growth;
  const alpha = GROWTH_CLASSES[name];
  return alpha > 0 ? { alpha, growth: name } : null;
}

/**
 * The sources behind the values above (full records, with how each was read,
 * in wiki/research/ignition-sources-2026-11-05.json). DOIs were resolved at
 * Crossref on 2026-11-05.
 */
export const SOURCES = {
  nfpa10: "NFPA 10, Standard for Portable Fire Extinguishers, 2022 edition (standard).",
  "iso3941-2026": "ISO 3941:2026, Classification of fires, adding class L (standard). doi:10.3403/30482407",
  babrauskas1982nbsir: "Babrauskas, Lawson, Walton, Twilley (1982). Upholstered furniture heat release rates measured with a furniture calorimeter. NBSIR 82-2604. doi:10.6028/NBS.IR.82-2604",
  lawson1984nbsir: "Lawson, Walton, Twilley (1984). Fire performance of furnishings as measured in the NBS furniture calorimeter, part I. NBSIR 83-2787. doi:10.6028/NBS.IR.83-2787",
  babrauskas1983pool: "Babrauskas (1983). Estimating large pool fire burning rates. Fire Technology 19(4), 251-261. doi:10.1007/BF02380810",
  gannbryner: "Gann, Bryner. Combustion products and their effects on life safety. Fire Protection Handbook, NFPA (handbook).",
  mckenna2016: "McKenna, Hull (2016). The fire toxicity of polyurethane foams. Fire Science Reviews 5, 3. doi:10.1186/s40038-016-0012-3",
  klotemilke2002: "Klote, Milke (2002). Principles of Smoke Management. ASHRAE, p. 35, reproducing Tewarson's soot yields (handbook, secondary).",
  nureg1805s1: "US NRC (2013). NUREG-1805 Supplement 1, Fire Dynamics Tools (report, secondary for SFPE Handbook data).",
  "en1991-1-2": "EN 1991-1-2, Eurocode 1, actions on structures exposed to fire, Annex E (standard).",
  manzello2020: "Manzello, Suzuki, Gollner, Fernandez-Pello (2020). Role of firebrand combustion in large outdoor fire spread. Progress in Energy and Combustion Science 76, 100801. doi:10.1016/j.pecs.2019.100801",
  hu2021tn2133: "Hu et al. (2021). The character of residential cooktop fires. NIST Technical Note 2133. doi:10.6028/NIST.TN.2133",
  hu2021jfs: "Hu et al. (2021). The character of residential cooktop fires. Journal of Fire Sciences 39(2), 142-163. doi:10.1177/0734904121990392",
  hamins2018: "Hamins, Kim, Madrzykowski (2018). Characterization of stovetop cooking oil fires. Journal of Fire Sciences 36(3), 224-239. doi:10.1177/0734904118761884",
  hostikka2023: "Hostikka et al. (2023). Exploring the thermal characteristics, ignitions and heat release of oils and solid items at electric cooktops. Fire Safety Journal 140, 103889. doi:10.1016/j.firesaf.2023.103889",
  liu2005fss: "Liu, Carpenter, Kim (2005). Application of water mist to extinguish large oil pool fires for industrial oil cooker protection. Fire Safety Science 8, 741-752. doi:10.3801/IAFSS.FSS.8-741",
  hietaniemi2001vtt: "Hietaniemi, Mangs, Hakkarainen (2001). Burning of electrical household appliances: an experimental study. VTT Research Notes 2084 (report).",
  nistwebbookMgO: "NIST Chemistry WebBook, SRD 69: magnesium oxide, enthalpy of formation (CODATA) (reference data).",
  nistwebbookHeptane: "NIST Chemistry WebBook, SRD 69: n-heptane, boiling point and enthalpy of vaporisation (reference data).",
  nistwebbookMg: "NIST Chemistry WebBook, SRD 69: magnesium, condensed-phase heat capacity at 298 K (reference data).",
  cameo: "NOAA CAMEO Chemicals datasheets: magnesium, sodium, lithium (government database).",
  zhao2022: "Zhao et al. (2022). The relationship between ignition and oxidation of molten magnesium alloys during the cooling process. Frontiers in Chemistry 10, 980860. doi:10.3389/fchem.2022.980860",
  larsson2017: "Larsson, Andersson, Blomqvist, Mellander (2017). Toxic fluoride gas emissions from lithium-ion battery fires. Scientific Reports 7, 10018. doi:10.1038/s41598-017-09784-z",
  fleischmann2025: "Fleischmann, Weinschenk, Madrzykowski, Schraiber, Gaudet (2025). Quantifying the fire hazard from Li-ion battery fires caused by thermal runaway in e-scooters. Fire Technology 61(5), 2865-2887. doi:10.1007/s10694-025-01707-z",
  liu2025ebike: "Liu et al. (2025). Experimental investigation of the thermal runaway and fire characteristics of overcapacity lithium-ion batteries for electric bicycles in charging cabinets. Frontiers in Energy Research 13, 1681969. doi:10.3389/fenrg.2025.1681969",
};
