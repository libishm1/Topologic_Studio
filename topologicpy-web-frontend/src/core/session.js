/**
 * What this visit can use: the route engine the server offers now, and the
 * site's own code once a newer deploy has replaced it
 * (wiki/architecture/state-machines.md, SM-4 and SM-6).
 */

/**
 * The route engine to run: the saved choice when the server lists it, the
 * server's first otherwise. With no list (the server asleep, unreachable or
 * absent) only the browser's engine can run: a saved TopologicPy choice used
 * to stay, and Find route refused on a site whose routing runs in the browser.
 * The choice itself stays saved and applies again when the server lists it.
 *
 * @param {string} saved            settings.engine
 * @param {string[] | null} engines the server's capabilities.engines
 */
export function effectiveEngine(saved, engines) {
  if (!engines?.length) return saved === "topologicpy" ? "fast" : saved;
  return engines.includes(saved) ? saved : engines[0];
}

/**
 * A part of the site this page has not loaded yet failed to load because a
 * newer deploy replaced it (its file names carry a hash). Chrome, Firefox and
 * Safari word it differently.
 */
export const isStaleSite = (error) =>
  /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Unable to preload CSS/i.test(
    String(error?.message ?? error),
  );

export const STALE_SITE_MESSAGE = "The site was updated after this page was opened. Reload the page to continue; your model will need opening again.";
