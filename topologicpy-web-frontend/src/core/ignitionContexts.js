/**
 * Causes for "Likely ignition points" (ignition.js), kept apart from the
 * catalogue so the panel can list them without loading it.
 *
 * Each narrows the list so a cooking, electrical or gas fire is one
 * choice. `scenarios` are catalogue ids; `fireClass` sets the class the fire
 * starts with (a gas hob burns gas, class B, not cooking oil), and `labels`
 * renames a scenario to match. "deliberate" brings in the accelerant
 * scenario, which the ranked list leaves out.
 */
export const IGNITION_CONTEXTS = [
  { id: "any", label: "Any cause (ranked)" },
  { id: "cooking", label: "Cooking", scenarios: ["cooking-range"] },
  {
    id: "electrical",
    label: "Electrical fault",
    scenarios: [
      "electrical-distribution",
      "extension-lead",
      "tv-electronics",
      "office-electrical-equipment",
      "refrigerator-freezer",
      "dishwasher",
      "clothes-dryer",
      "washing-machine",
    ],
  },
  { id: "gas", label: "Gas or liquid fuel", scenarios: ["cooking-range", "boiler-plant"], fireClass: "B", labels: { "cooking-range": "Gas hob or oven" } },
  { id: "heating", label: "Heating", scenarios: ["space-heater", "heating-stove-fireplace"] },
  { id: "smoking", label: "Smoking materials or candles", scenarios: ["smoking-upholstery", "smoking-bedding", "candle"] },
  { id: "batteries", label: "Lithium-ion batteries", scenarios: ["li-ion-micromobility", "li-ion-small-device", "battery-storage-ups"] },
  { id: "waste", label: "Waste, paper or decorations", scenarios: ["waste-bin-workstation", "christmas-tree"] },
  { id: "deliberate", label: "Deliberate, with an accelerant", scenarios: ["intentional-accelerant"] },
];

