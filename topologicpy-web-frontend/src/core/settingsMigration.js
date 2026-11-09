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
