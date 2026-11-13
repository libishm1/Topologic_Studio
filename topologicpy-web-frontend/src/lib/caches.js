/**
 * What Studio keeps in this browser that can grow, and clearing it (the
 * notice on running out of memory or storage, App.jsx; core/memoryNotice.js):
 *
 * - kept Flamescapes runs (core/runCache.js, IndexedDB "topologic-studio-runs",
 *   at most 256 MB, a run at most 96 MB);
 * - converted models' fragments and semantics (lib/fragmentCache.js,
 *   IndexedDB "topologic-studio", at most 512 MB and 24 models).
 *
 * Not cleared: the settings, theme, field notes, a guided run's saved
 * settings and the GPU's measured speed, all small entries in localStorage.
 * The site uses no Cache Storage and no service worker; the on-device
 * describer's model is Chrome's own (the Prompt API), not the site's.
 */
import { clearRuns, storedRuns } from "../core/runCache.js";
import { cacheSize, clearCache } from "./fragmentCache.js";

/** The kept runs and converted models (count, bytes), their total, and the browser's estimate of the site's storage when it gives one. */
export async function cachedSizes() {
  const [runs, models] = await Promise.all([storedRuns(), cacheSize()]);
  let estimate = null;
  try {
    estimate = (await globalThis.navigator?.storage?.estimate?.()) ?? null;
  } catch {
    // No estimate in this browser.
  }
  return { runs, models, bytes: runs.bytes + models.bytes, estimate };
}

/** Empty the run and model caches, never the settings; returns the sizes before and after. */
export async function clearCaches() {
  const before = await cachedSizes();
  await Promise.all([clearRuns(), clearCache()]);
  const after = await cachedSizes();
  return { before, after };
}
