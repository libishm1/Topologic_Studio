/**
 * Fire classes for the first item to burn, and what each means for the
 * Flamescapes solver (src/core/flamescapes.js) and for the occupant.
 *
 * The classes are NFPA 10's (A ordinary combustibles, B flammable liquids
 * and gases, C energised electrical equipment, D combustible metals, K
 * cooking oils and fats), which the Saudi Fire Code (SBC 801, after the
 * International Fire Code) uses; `iso` gives the ISO 3941 / EN 2 class (A
 * solids, B liquids, C gases, D metals, F cooking media; electrical is not a
 * class there). Class C names the hazard, not a fuel: once the equipment
 * burns, its fuel is `fuel` (plastics, so class A by ISO 3941).
 *
 * Per class, the solver's fuel:
 *   heatOfCombustion  J per kg burnt
 *   yields            kg of CO2, CO and soot per kg burnt (the soot is the
 *                     paper's "residual" species, the smoke)
 *   o2                kg of oxygen per kg burnt, when not 417 kJ / mol O2
 *                     (Fire-X Eq. 4; Huggett 1980), as for metals
 *   ash               kg of solid oxide left per kg burnt (stays out of the gas)
 *   liquid            {boiling (K), latent (J/kg)}: evaporates when heated
 *                     past `boiling` with the heat above it (Fire-X's
 *                     energy-limited evaporation), instead of pyrolysing
 *   embers            whether the burning item throws embers (charring solids)
 *   ignitionTemperature  degrees C, for spot ignition by embers
 *   item              the first item: mass (kg) and exposed area (m^2)
 *
 * Sources (wiki/research/ignition-sources-2026-11-05.md has the review):
 * class definitions NFPA 10 (2022) and ISO 3941:2007; class A heat of
 * combustion from Flamescapes Table 3 (2.80 MJ/mol over wood's 0.16214
 * kg/mol); magnesium from the heat of formation of MgO (601.6 kJ/mol).
 * Values marked `provisional` are representative handbook figures still to
 * be checked against the review's sources.
 */

export const FIRE_CLASSES = {
  A: {
    label: "Class A: ordinary combustibles",
    short: "A",
    iso: "A",
    examples: "Upholstered furniture, bedding, wood, paper, cloth, waste, most plastics",
    fuel: {
      heatOfCombustion: 17.27e6,
      yields: { co2: 1.27, co: 0.004, soot: 0.015 },
      embers: true,
      ignitionTemperature: 300,
    },
    item: { mass: 20, area: 2 },
    agents: {
      suitable: ["Water", "Foam", "Multipurpose (ABC) dry powder"],
      unsuitable: [],
    },
  },
  B: {
    label: "Class B: flammable liquids and gases",
    short: "B",
    iso: "B (liquids), C (gases)",
    examples: "Petrol, solvents, paint thinners, alcohol, lamp oil",
    fuel: {
      heatOfCombustion: 43.4e6,
      yields: { co2: 2.85, co: 0.01, soot: 0.037 },
      liquid: { boiling: 371, latent: 3.2e5 },
      embers: false,
      ignitionTemperature: 250,
      provisional: true,
    },
    item: { mass: 4, area: 1 },
    agents: {
      suitable: ["Foam", "Carbon dioxide", "Dry powder"],
      unsuitable: [{ agent: "Water jet", reason: "Spreads the burning liquid; it floats on water." }],
    },
  },
  C: {
    label: "Class C: energised electrical equipment",
    short: "C",
    iso: "none (electrical is not an ISO 3941 class; the burning plastics are A)",
    examples: "Televisions, chargers and power banks, extension leads, distribution boards, appliances",
    fuel: {
      heatOfCombustion: 25e6,
      yields: { co2: 2.0, co: 0.06, soot: 0.1 },
      embers: false,
      ignitionTemperature: 350,
      provisional: true,
    },
    fuelClass: "A",
    item: { mass: 10, area: 1 },
    agents: {
      suitable: ["Carbon dioxide", "Clean agent", "Dry powder", "Isolate the power first"],
      unsuitable: [{ agent: "Water or foam while live", reason: "Conducts electricity to the person." }],
    },
  },
  D: {
    label: "Class D: combustible metals",
    short: "D",
    iso: "D",
    examples: "Magnesium, titanium, sodium, lithium metal (rare in buildings; workshops and labs)",
    fuel: {
      // Mg + 1/2 O2 -> MgO: 601.6 kJ / mol over 24.305 g / mol.
      heatOfCombustion: 24.75e6,
      o2: 0.658,
      ash: 1.658,
      yields: { co2: 0, co: 0, soot: 0 },
      embers: false,
      ignitionTemperature: 650,
    },
    item: { mass: 2, area: 0.2 },
    agents: {
      suitable: ["Class D powder", "Dry sand"],
      unsuitable: [
        { agent: "Water", reason: "Reacts with burning metal, releasing hydrogen and spattering." },
        { agent: "Carbon dioxide", reason: "Burning magnesium reduces it and keeps burning." },
      ],
    },
  },
  K: {
    label: "Class K: cooking oils and fats",
    short: "K",
    iso: "F",
    examples: "A pan or deep fryer of vegetable oil or fat on a hob",
    fuel: {
      heatOfCombustion: 37e6,
      yields: { co2: 2.8, co: 0.006, soot: 0.02 },
      // Oil does not boil before it ignites: it gives off vapour from its
      // smoke point and autoignites near 360 to 400 C.
      liquid: { boiling: 633, latent: 2.5e5 },
      embers: false,
      ignitionTemperature: 360,
      provisional: true,
    },
    item: { mass: 2, area: 0.07 },
    agents: {
      suitable: ["Wet chemical (class K / F)", "Fire blanket", "Turn off the heat; lid on the pan"],
      unsuitable: [{ agent: "Water", reason: "Turns to steam under the oil and throws burning oil out (a flare-up)." }],
    },
  },
};

export const FIRE_CLASS_KEYS = Object.keys(FIRE_CLASSES);

/** The solver's fuel and first item for a class (the room contents are always class A). */
export function fuelsFor(cls = "A") {
  const first = FIRE_CLASSES[cls] || FIRE_CLASSES.A;
  return { contentsFuel: FIRE_CLASSES.A.fuel, ignitionFuel: first.fuel, item: first.item };
}
