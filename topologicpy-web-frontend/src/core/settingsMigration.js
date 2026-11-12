/**
 * Settings kept in the browser by an older version, as this one reads them
 * (state/useStudio.js loadSettings). Here so that it can be tested without
 * the app's modules.
 *
 * The fire's grid was flamescapesGrid until 2026-11-09, "speed" by default
 * or "finest"; it is now flamescapesGridPolicy, "standard" by default. A
 * chosen "finest" is kept; the old default gives way to the standard grid.
 */
import { SERVICE_RESPONSE_MAX, SERVICE_RESPONSE_MIN } from "./fireService.js";

export function migrateSettings(stored) {
  const { flamescapesGrid, ...rest } = stored || {};
  if (rest.flamescapesGridPolicy === undefined && flamescapesGrid === "finest") rest.flamescapesGridPolicy = "finest";
  // Only the values this version offers: a grid it does not know ("calibrated",
  // stored by a build of 2026-11-12 before it had its own key) becomes the
  // standard grid, with the 0.27 m grid's own key for "calibrated".
  if (rest.flamescapesGridPolicy === "calibrated") rest.flamescapesCalibratedGrid = true;
  if (rest.flamescapesGridPolicy !== undefined && !GRID_POLICIES.includes(rest.flamescapesGridPolicy)) rest.flamescapesGridPolicy = "standard";
  if (rest.flamescapesSolver !== undefined && !SOLVER_NAMES.includes(rest.flamescapesSolver)) delete rest.flamescapesSolver;
  // A run until the fire service arrives (core/fireService.js): a switch, and a
  // response in the field's range, s; anything else gives way to the default.
  if (rest.flamescapesUntilService !== undefined && typeof rest.flamescapesUntilService !== "boolean") delete rest.flamescapesUntilService;
  const response = rest.flamescapesServiceResponse;
  if (response !== undefined && !(Number.isFinite(response) && response >= SERVICE_RESPONSE_MIN && response <= SERVICE_RESPONSE_MAX)) delete rest.flamescapesServiceResponse;
  return rest;
}

/** The fire grid policies and solvers this version offers (a stored other value is dropped). */
export const GRID_POLICIES = ["standard", "speed", "finest"];
export const SOLVER_NAMES = ["validated", "previous"];

/**
 * Settings that last for the visit only: never written to the browser, and
 * dropped when read back (one an earlier version wrote, too). People leaving
 * (crowd) starts only when switched on for the graph on screen.
 */
export const SESSION_ONLY = ["crowd"];

/** The settings without the visit-only ones, to store or to read back. */
export function withoutSessionOnly(settings) {
  const out = { ...(settings || {}) };
  for (const key of SESSION_ONLY) delete out[key];
  return out;
}
