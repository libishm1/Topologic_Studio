/**
 * A Flamescapes run that lasts until the fire service arrives
 * (settings.flamescapesUntilService): detection, the time the performance
 * check sets (settings.perfDetection, "Detection (s)"), plus the fire
 * service's response. Off by default; off, the run lasts the simulated time
 * set, as before.
 *
 * The response is chosen from presets (SERVICE_PRESETS, below: NFPA 1710's
 * first engine, NFPA 1720's urban, suburban and rural zones) or set by hand.
 * It defaults to NFPA 1710, Standard for the Organization and
 * Deployment of Fire Suppression Operations, Emergency Medical Operations,
 * and Special Operations to the Public by Career Fire Departments, 2020
 * edition: its objectives for the first engine company at a fire
 * suppression incident, from the alarm reaching the communication centre to
 * the engine on scene. Answering and processing as revised by TIA 20-1
 * (issued 2022-08-12, the same values); turnout and travel to be met in at
 * least 90 % of responses (4.1.2.4). They are a career department's design
 * objectives, not a prediction for any building or city.
 */
import { COVERED } from "./calibrationStatus.js";

/** NFPA 1710 (2020) response objectives for the first engine company, s, with their clauses. */
export const NFPA_1710_RESPONSE = [
  { id: "answering", label: "alarm answering", seconds: 15, clause: "4.1.2.3.1", share: "at least 95 % of alarms" },
  { id: "processing", label: "alarm processing", seconds: 64, clause: "4.1.2.3.3", share: "at least 90 % of alarms" },
  { id: "turnout", label: "turnout", seconds: 80, clause: "4.1.2.1(2)", share: "fire and special operations" },
  { id: "travel", label: "travel", seconds: 240, clause: "4.1.2.1(3)", share: "the first engine company" },
];

/** Their sum, the default response: 399 s, 6.65 min. */
export const DEFAULT_SERVICE_RESPONSE = NFPA_1710_RESPONSE.reduce((s, part) => s + part.seconds, 0);

/** NFPA 1710's alarm handling, answering and processing, s: added to the NFPA 1720 times, which run from dispatch. */
export const ALARM_HANDLING_S = NFPA_1710_RESPONSE.filter((p) => p.id === "answering" || p.id === "processing").reduce((s, p) => s + p.seconds, 0);

/**
 * The fire service's response, chosen (settings.flamescapesServicePreset):
 * NFPA 1710's first engine, the default; NFPA 1720's demand zones; or the
 * user's own time ("custom", settings.flamescapesServiceResponse).
 *
 * NFPA 1720, Standard for the Organization and Deployment of Fire Suppression
 * Operations, Emergency Medical Operations, and Special Operations to the
 * Public by Volunteer Fire Departments, 2020 edition, Table 4.3.2, for a
 * department mostly of volunteers: by the zone's population per square mile,
 * the staff to assemble on scene and the minutes to do it in, met the share
 * of the time given. Its times run from dispatch, call handling not included,
 * so each preset adds the 79 s of NFPA 1710's alarm answering and processing
 * that the 1710 preset counts.
 *
 * The standard is not open to read, so the table's figures were read from
 * secondary sources: Fire Engineering, A. Zielinski, "From Arrival to Action:
 * The Impact of Assembly Time" (2026-05-29), its table "NFPA 1720 Staff and
 * Time Requirements" (urban, suburban, rural and remote, with no edition
 * named); and Firehouse, N. Sohyda (chair of the NFPA 1720 committee), "NFPA
 * Standards: NFPA 1720: An Update on the Volunteer Deployment Standard"
 * (2021-10-18), on the 2020 edition and its times running from dispatch with
 * call handling not included. The 1720 times are a volunteer department's
 * objectives, as 1710's are a career one's: not what any town achieves.
 */
export const SERVICE_PRESETS = [
  {
    id: "nfpa1710",
    label: "NFPA 1710, first engine (6.65 min)",
    seconds: DEFAULT_SERVICE_RESPONSE,
    source: `NFPA 1710 (2020 edition) for the first engine company, ${NFPA_1710_RESPONSE.map((p) => `${p.seconds} s ${p.label}`).join(", ")}`,
  },
  {
    id: "nfpa1720urban",
    label: "NFPA 1720 urban (9 min from dispatch)",
    seconds: ALARM_HANDLING_S + 9 * 60,
    source: "NFPA 1720 (2020 edition) Table 4.3.2, urban (over 1,000 people a square mile): 15 staff in 9 min, 90 % of the time, from dispatch, plus 79 s of alarm handling",
  },
  {
    id: "nfpa1720suburban",
    label: "NFPA 1720 suburban (10 min from dispatch)",
    seconds: ALARM_HANDLING_S + 10 * 60,
    source: "NFPA 1720 (2020 edition) Table 4.3.2, suburban (500 to 1,000 people a square mile): 10 staff in 10 min, 80 % of the time, from dispatch, plus 79 s of alarm handling",
  },
  {
    id: "nfpa1720rural",
    label: "NFPA 1720 rural (14 min from dispatch)",
    seconds: ALARM_HANDLING_S + 14 * 60,
    source: "NFPA 1720 (2020 edition) Table 4.3.2, rural (under 500 people a square mile): 6 staff in 14 min, 80 % of the time, from dispatch, plus 79 s of alarm handling",
  },
  { id: "custom", label: "Custom", seconds: null, source: null },
];

/** The presets' ids, for the stored settings' allow-list (core/settingsMigration.js). */
export const SERVICE_PRESET_IDS = SERVICE_PRESETS.map((p) => p.id);
export const DEFAULT_SERVICE_PRESET = "nfpa1710";

/**
 * The response the settings ask for, s: a preset's, or the user's own with
 * "custom". Settings stored before the presets (no preset) mean the response
 * they hold, as their migration reads them; an unknown preset, the default.
 */
export function serviceResponse(settings) {
  const id = settings?.flamescapesServicePreset;
  const preset = SERVICE_PRESETS.find((p) => p.id === id);
  if (preset?.seconds) return preset.seconds;
  if (id === "custom" || id === undefined) return settings?.flamescapesServiceResponse ?? DEFAULT_SERVICE_RESPONSE;
  return DEFAULT_SERVICE_RESPONSE;
}

/** A response time the field takes, s (0.5 to 60 min). */
export const SERVICE_RESPONSE_MIN = 30;
export const SERVICE_RESPONSE_MAX = 3600;

/** The run's length until the fire service arrives: at least the shortest run, at most an hour. */
export const SERVICE_HORIZON_MIN = 30;
export const SERVICE_HORIZON_MAX = 3600;

/** The solver's step, s (core/flamescapes.js DEFAULTS.dt; a unit test holds the two together). */
export const SOLVER_STEP_S = 0.1;

const seconds = (x, fallback) => (Number.isFinite(x) && x >= 0 ? x : fallback);

/**
 * When the fire service arrives, s after ignition, and how long the run
 * lasts: the arrival, rounded to the second and kept within
 * SERVICE_HORIZON_MIN to MAX. `capped` when the run ends before the arrival;
 * `beyondTested` when it runs past the held-out tests' 900 s
 * (core/calibrationStatus.js COVERED.horizon).
 */
export function serviceArrival(detection, response = DEFAULT_SERVICE_RESPONSE) {
  const d = seconds(detection, 0);
  const r = seconds(response, DEFAULT_SERVICE_RESPONSE);
  const arrival = d + r;
  const horizon = Math.min(SERVICE_HORIZON_MAX, Math.max(SERVICE_HORIZON_MIN, Math.round(arrival)));
  return { detection: d, response: r, arrival, horizon, capped: arrival > SERVICE_HORIZON_MAX, beyondTested: beyondTested(horizon) };
}

/** Whether a run of `horizon` s lies outside the held-out tests' times. */
export const beyondTested = (horizon) => horizon > COVERED.horizon;

/** The simulated time a Flamescapes run lasts under these settings, s: the time set, or until the fire service arrives. */
export function runHorizon(settings) {
  if (!settings?.flamescapesUntilService) return settings?.flamescapesHorizon;
  return serviceArrival(settings.perfDetection, serviceResponse(settings)).horizon;
}

/**
 * Roughly how long a run of `horizon` simulated seconds takes to compute, s,
 * at a measured pace: `wallMs` of computing for `simulated` seconds, of which
 * `setupMs` built the grid (once a run). Null with nothing measured. Linear
 * in the simulated time, so a fire that grows faster than the measured one
 * (more solver substeps) takes longer.
 */
export function computeSeconds(horizon, pace) {
  if (!(horizon > 0) || !(pace?.simulated > 0) || !(pace.wallMs > 0)) return null;
  const setup = Math.min(pace.wallMs, Math.max(0, pace.setupMs || 0));
  return (setup + (horizon / pace.simulated) * (pace.wallMs - setup)) / 1000;
}

/** A GPU's kept speed (cells a millisecond a solver step) on a grid of `cells`, as a pace. */
export function gpuPace(rate, cells) {
  return rate > 0 && cells > 0 ? { simulated: SOLVER_STEP_S, wallMs: cells / rate, setupMs: 0 } : null;
}

/** A length of time to read: "45 s", "7 min 39 s", "1 h 5 min". */
export function durationText(s) {
  if (!Number.isFinite(s) || s < 0) return "-";
  const whole = Math.round(s);
  if (whole < 90) return `${whole} s`;
  if (whole < 3600) return `${Math.floor(whole / 60)} min${whole % 60 ? ` ${whole % 60} s` : ""}`;
  const minutes = Math.round(whole / 60);
  return `${Math.floor(minutes / 60)} h${minutes % 60 ? ` ${minutes % 60} min` : ""}`;
}
