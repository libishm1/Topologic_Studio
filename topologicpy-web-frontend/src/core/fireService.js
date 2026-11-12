/**
 * A Flamescapes run that lasts until the fire service arrives
 * (settings.flamescapesUntilService): detection, the time the performance
 * check sets (settings.perfDetection, "Detection (s)"), plus the fire
 * service's response. Off by default; off, the run lasts the simulated time
 * set, as before.
 *
 * The response defaults to NFPA 1710, Standard for the Organization and
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
  return serviceArrival(settings.perfDetection, settings.flamescapesServiceResponse).horizon;
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
