/**
 * Settings kept in the browser by an older version, as this one reads them
 * (state/useStudio.js loadSettings). Here so that it can be tested without
 * the app's modules.
 *
 * The fire's grid was flamescapesGrid until 2026-11-09, "speed" by default
 * or "finest"; it is now flamescapesGridPolicy, "standard" by default. A
 * chosen "finest" is kept; the old default gives way to the standard grid.
 */
export function migrateSettings(stored) {
  const { flamescapesGrid, ...rest } = stored || {};
  if (rest.flamescapesGridPolicy === undefined && flamescapesGrid === "finest") rest.flamescapesGridPolicy = "finest";
  return rest;
}

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
