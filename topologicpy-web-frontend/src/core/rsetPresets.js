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
 */
export const RSET_PRESETS = [
  {
    id: "office",
    label: "Office: awake and familiar",
    values: { perfAlarm: 30, perfPre1: 60, perfPre99: 180 },
    source:
      "PD 7974-6:2019 category A (awake, familiar), management M2, simple layout, automatic detection and alarm: 1 and 3 min. C/VM2's single values: 30 s in the room of origin, 60 s elsewhere.",
  },
  {
    id: "classroom",
    label: "University classroom (as an office)",
    values: { perfAlarm: 30, perfPre1: 60, perfPre99: 180 },
    source: "PD 7974-6 has no educational row; BSI DD 240 groups colleges and universities with offices, so the office values (1 and 3 min).",
  },
  {
    id: "apartment",
    label: "Apartments: the other flats",
    values: { perfAlarm: 30, perfPre1: 85, perfPre99: 1056 },
    source:
      "ABCB FSVM Table 16, fire floor: 150 s for 9 %, 300 s for 80 %, 600 s for 10 %, 1 % never responding; a log-normal through 150 and 600 s as the 10th and 90th percentiles (median 300 s). C/VM2: 300 s away from the flat of origin.",
  },
  {
    id: "apartment-origin",
    label: "Apartments: the flat of fire origin",
    values: { perfAlarm: 30, perfPre1: 300, perfPre99: 600 },
    source: "PD 7974-6:2019 category Ci (dwellings), management M2: 5 and 10 min (marked as more uncertain). C/VM2: 60 s in the flat of origin.",
  },
  {
    id: "hotel",
    label: "Hotel: asleep and unfamiliar",
    values: { perfAlarm: 120, perfPre1: 1200, perfPre99: 2400 },
    source:
      "ISO/TR 16738:2009's example for category Ciii (hotels), ordinary management, staged alarm: 20 and 40 min (as PD 7974-6:2004), with 120 s for staff to confirm the alarm (SFEG 2025 4.2). C/VM2: 600 s away from the room of origin with a standard alarm, 300 s with a voice alarm.",
  },
  {
    id: "assembly",
    label: "Assembly: awake and unfamiliar",
    values: { perfAlarm: 30, perfPre1: 60, perfPre99: 240 },
    source: "PD 7974-6:2019 category B (awake, unfamiliar), management M2, simple layout: 1 and 4 min; add 0.5 to 1 min for a complex building.",
  },
  {
    id: "custom",
    label: "Custom (set the times below)",
    values: null,
    source: "The times below, as set.",
  },
];

export const presetOf = (id) => RSET_PRESETS.find((p) => p.id === id) || RSET_PRESETS[RSET_PRESETS.length - 1];

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
