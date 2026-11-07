/**
 * Alarm and pre-movement times for the performance-based check
 * (performanceCheck.js), by occupancy (wiki/research/rset-components-2026-11-07.md).
 * Seconds; pre-movement as its 1st and 99th percentiles, counted from the
 * alarm. Detection is not part of a preset: it is set, or taken from the
 * fire run (forecast.js smokeDetection).
 *
 * The alarm delay is 30 s for an immediate general alarm (NZ C/VM2 2nd
 * edition, 2025, 3.3.1.1; ABCB FSVM data sheets 1.1.3) and 120 s for a
 * staged one (Singapore SFEG 2025 4.2). PD 7974-6:2019's percentiles were
 * read in a secondary source (CFPA-E Guideline No 19, 2023); ABCB's and
 * C/VM2's in full.
 *
 * Each preset also carries, for the queues at doors and beside the design
 * values:
 * - perfNoResponse: the share of occupants, in per cent, who never respond
 *   (ABCB FSVM data sheets Tables 14 to 18: 0 for offices and assembly, 1 %
 *   on an apartment fire floor and in hotels with ordinary management);
 * - perfOccupantArea: floor area per occupant, m², for the occupant load
 *   (the code check's stricter of SBC 801-CR-2018 and IBC 2021 for offices;
 *   IBC 2021 Table 1004.5 for the others, the SBC values for those uses not
 *   having been read);
 * - perfMeasured: the measured pre-movement cluster shown beside the design
 *   log-normal (MEASURED_PRE).
 */
export const RSET_PRESETS = [
  {
    id: "office",
    label: "Office: awake and familiar",
    values: { perfAlarm: 30, perfPre1: 60, perfPre99: 180, perfNoResponse: 0, perfOccupantArea: 9, perfMeasured: "business-1" },
    load: "9 m² per occupant: the stricter of SBC 801-CR-2018 (business, 9 m² gross) and IBC 2021 (13.94 m²), as the code check.",
    source:
      "PD 7974-6:2019 category A (awake, familiar), management M2, simple layout, automatic detection and alarm: 1 and 3 min. C/VM2's single values: 30 s in the room of origin, 60 s elsewhere.",
  },
  {
    id: "classroom",
    label: "University classroom (as an office)",
    values: { perfAlarm: 30, perfPre1: 60, perfPre99: 180, perfNoResponse: 0, perfOccupantArea: 1.86, perfMeasured: "lecture" },
    load: "IBC 2021 Table 1004.5: classroom area, 20 sq ft (1.86 m²) net.",
    source: "PD 7974-6 has no educational row; BSI DD 240 groups colleges and universities with offices, so the office values (1 and 3 min).",
  },
  {
    id: "apartment",
    label: "Apartments: the other flats",
    values: { perfAlarm: 30, perfPre1: 85, perfPre99: 1056, perfNoResponse: 1, perfOccupantArea: 18.58, perfMeasured: "residential-1" },
    load: "IBC 2021 Table 1004.5: residential, 200 sq ft (18.58 m²) gross.",
    source:
      "ABCB FSVM Table 16, fire floor: 150 s for 9 %, 300 s for 80 %, 600 s for 10 %, 1 % never responding; a log-normal through 150 and 600 s as the 10th and 90th percentiles (median 300 s). C/VM2: 300 s away from the flat of origin.",
  },
  {
    id: "apartment-origin",
    label: "Apartments: the flat of fire origin",
    values: { perfAlarm: 30, perfPre1: 300, perfPre99: 600, perfNoResponse: 1, perfOccupantArea: 18.58, perfMeasured: "residential-1" },
    load: "IBC 2021 Table 1004.5: residential, 200 sq ft (18.58 m²) gross.",
    source: "PD 7974-6:2019 category Ci (dwellings), management M2: 5 and 10 min (marked as more uncertain). C/VM2: 60 s in the flat of origin.",
  },
  {
    id: "hotel",
    label: "Hotel: asleep and unfamiliar",
    values: { perfAlarm: 120, perfPre1: 1200, perfPre99: 2400, perfNoResponse: 1, perfOccupantArea: 18.58, perfMeasured: "hotel-1" },
    load: "IBC 2021 Table 1004.5: residential, 200 sq ft (18.58 m²) gross.",
    source:
      "ISO/TR 16738:2009's example for category Ciii (hotels), ordinary management, staged alarm: 20 and 40 min (as PD 7974-6:2004), with 120 s for staff to confirm the alarm (SFEG 2025 4.2). C/VM2: 600 s away from the room of origin with a standard alarm, 300 s with a voice alarm.",
  },
  {
    id: "assembly",
    label: "Assembly: awake and unfamiliar",
    values: { perfAlarm: 30, perfPre1: 60, perfPre99: 240, perfNoResponse: 0, perfOccupantArea: 1.39, perfMeasured: "theatre" },
    load: "IBC 2021 Table 1004.5: assembly without fixed seats, unconcentrated (tables and chairs), 15 sq ft (1.39 m²) net.",
    source: "PD 7974-6:2019 category B (awake, unfamiliar), management M2, simple layout: 1 and 4 min; add 0.5 to 1 min for a complex building.",
  },
  {
    id: "custom",
    label: "Custom (set the times below)",
    values: null,
    load: "The area per occupant below, as set.",
    source: "The times below, as set.",
  },
];

export const presetOf = (id) => RSET_PRESETS.find((p) => p.id === id) || RSET_PRESETS[RSET_PRESETS.length - 1];

/**
 * Measured pre-movement times: Weibull fits to the clusters of Lovreglio,
 * Kuligowski, Gwynne and Boyce (2019), "A pre-evacuation database for use in
 * egress simulations", Fire Safety Journal (2019),
 * doi:10.1016/j.firesaf.2018.12.009 (the author accepted manuscript's
 * tables; the paper's log-normal columns are inconsistent and not used).
 * Scale in seconds. Shown beside the design log-normal, not judged: the
 * design values are what the codes agree on, and the data are far wider
 * (wiki/research/rset-components-2026-11-07.md 2.3, 3.6).
 */
export const MEASURED_PRE = [
  { id: "business-1", label: "Offices (business cluster 1)", scale: 139.3, shape: 1.195, source: "Lovreglio et al. 2019, business cluster 1: 13 case studies, 2,597 people; median 102 s, 99th percentile 500 s." },
  { id: "lecture", label: "Lecture halls (educational cluster 5)", scale: 28.9, shape: 1.127, source: "Lovreglio et al. 2019, educational cluster 5 (lecture halls): median 21 s, 99th percentile 112 s." },
  { id: "library", label: "Libraries (educational cluster 3)", scale: 84.4, shape: 1.798, source: "Lovreglio et al. 2019, educational cluster 3 (libraries): median 69 s, 99th percentile 197 s." },
  {
    id: "residential-1",
    label: "Homes, drills with a working alarm (residential cluster 1)",
    scale: 102.5,
    shape: 0.767,
    source: "Lovreglio et al. 2019, residential cluster 1 (drills, a working alarm, mostly awake): median 64 s, 99th percentile 750 s.",
  },
  {
    id: "residential-2",
    label: "Homes, real fires and poor alarms (residential cluster 2)",
    scale: 724.6,
    shape: 0.978,
    source: "Lovreglio et al. 2019, residential cluster 2 (fires and poorly performing alarms): median 498 s, 99th percentile 3,454 s.",
  },
  {
    id: "hotel-1",
    label: "Hotels (hotel cluster 1)",
    scale: 90.5,
    shape: 1.79,
    source: "Lovreglio et al. 2019, hotel cluster 1 (Dutch drills on one floor, alarm by telephone, awake): median 74 s, 99th percentile 212 s; far quicker than sleeping guests.",
  },
  { id: "theatre", label: "Theatres (assembly cluster 4)", scale: 61.0, shape: 2.579, source: "Lovreglio et al. 2019, assembly cluster 4 (a UK theatre): median 53 s, 99th percentile 110 s." },
];

export const measuredOf = (id) => MEASURED_PRE.find((m) => m.id === id) || null;

/** Where the share that never responds comes from. */
export const NO_RESPONSE_SOURCE =
  "ABCB FSVM data sheets Tables 14 to 18: none for offices and assembly, 1 % on an apartment fire floor (4 % on the other floors) and in hotels with ordinary management, 10 % when management fails (a robustness case). They never start, so they are never counted as out in time; staff or the fire service must reach them.";

/**
 * What a place must show. With RSET = detection + alarm + the 99th
 * percentile of pre-movement + the walk with no fire, and ASET its deadline
 * plus the same walk, a place passes when ASET / RSET is at least `factor`
 * and ASET - RSET at least `margin` seconds. No Saudi document read sets
 * numbers (SBC 801-CR-2018 104.9 admits alternative methods the fire code
 * official approves), so these are the cited documents' own.
 */
export const RSET_CRITERIA = [
  {
    id: "base",
    label: "Base case: ASET at least twice RSET",
    factor: 2,
    margin: 30,
    source:
      "ASET / RSET at least 2 for the base case (Singapore SFEG 2025 Table 10.4), the same as the Italian fire code's margin of at least 100 % of RSET, and never under 30 s (M.3.2.2).",
  },
  {
    id: "sensitivity",
    label: "Sensitivity case: at least 1.2 times RSET",
    factor: 1.2,
    margin: 30,
    source: "ASET / RSET at least 1.2 for a sensitivity case (SFEG 2025 Table 10.4: a faster fire, a blocked exit, a failed detector), and never under 30 s (Italian fire code M.3.2.2).",
  },
  {
    id: "percentile",
    label: "The 99th percentile starts in time",
    factor: 1,
    margin: 0,
    source: "ASET at least RSET at the 99th percentile of pre-movement, with no margin on top: PD 7974-6's case for a sparse population (walking, not queuing). The least a design can show.",
  },
];

export const criterionOf = (id) => RSET_CRITERIA.find((c) => c.id === id) || RSET_CRITERIA[0];
