/**
 * Running out of memory or storage, and the one notice that offers to clear
 * the caches (App.jsx MemoryNotice, lib/caches.js clearCaches): the user's
 * request of 2026-11-12, "if browser runs out of memory prompt to clear cache
 * button". Pure, so node can test it; the browser's storage and memory are
 * read only through the functions given them.
 *
 * Kinds:
 *   quota    a cache write refused: the site's storage is full
 *            (QuotaExceededError from IndexedDB);
 *   memory   an allocation failed (RangeError "Array buffer allocation
 *            failed", "Invalid array length") loading a model, building the
 *            graph or running a fire, in the page or a worker, or a worker
 *            that stopped (most often for the same reason);
 *   gpu      the GPU ran out of memory for the fire's grid (WebGPU's
 *            out-of-memory error): the run went on on the CPU;
 *   storage  a warning: after a run or a model was kept, the site's storage
 *            is over STORAGE_WARN of its quota;
 *   heap     a warning: the page's JavaScript heap is over HEAP_WARN of its
 *            limit (Chrome's performance.memory), once a visit.
 */

export const STORAGE_WARN = 0.8;
export const HEAP_WARN = 0.85;
export const MEMORY_KINDS = ["quota", "memory", "gpu", "storage", "heap"];
/** The warnings: shown once a visit. The errors are shown again each time they happen. */
export const WARNING_KINDS = ["storage", "heap"];

const ALLOCATION = /array buffer allocation failed|invalid array length|invalid typed array length|cannot enlarge memory|ran out of memory|out of memory/i;

/** Which kind of memory problem an error is, or null for any other error. */
export function classifyMemoryError(error) {
  if (!error) return null;
  // A worker's typed answer (workers/*.worker.js memoryKind), kept on the error by lib/workerJob.js.
  if (MEMORY_KINDS.includes(error.kind)) return error.kind;
  const name = error.name || "";
  const message = String(error.message ?? error);
  if (name === "QuotaExceededError" || error.code === 22 || /QuotaExceeded|quota (has been )?exceeded/i.test(message)) return "quota";
  if (name === "GPUOutOfMemoryError" || /GPUOutOfMemory/i.test(message)) return "gpu";
  if ((name === "RangeError" || name === "InternalError" || name === "Error" || !name) && ALLOCATION.test(message)) return "memory";
  return null;
}

/** For a worker: its error's kind, to post back with it. */
export const memoryKind = (error) => classifyMemoryError(error);

/** Whether the site's storage is over `share` of its quota; false when unknown. */
export function storageNearlyFull(estimate, share = STORAGE_WARN) {
  const usage = estimate?.usage, quota = estimate?.quota;
  return Number.isFinite(usage) && Number.isFinite(quota) && quota > 0 && usage > share * quota;
}

/** Whether the JavaScript heap is over `share` of its limit; false when unknown. */
export function heapNearlyFull(memory, share = HEAP_WARN) {
  const used = memory?.usedJSHeapSize, limit = memory?.jsHeapSizeLimit;
  return Number.isFinite(used) && Number.isFinite(limit) && limit > 0 && used > share * limit;
}

/**
 * Whether to show a notice of `kind` now: never while one of that kind is on
 * screen; a warning once a visit; an error again whenever it happens again.
 */
export function shouldShow(kind, { shown = new Set(), showing = null } = {}) {
  if (!MEMORY_KINDS.includes(kind) || showing === kind) return false;
  return !(WARNING_KINDS.includes(kind) && shown.has(kind));
}

/** Bytes to read: "640 KB", "12.4 MB", "1.2 GB". */
export function bytesText(bytes) {
  if (!(bytes > 0)) return "0 MB";
  if (bytes < 2 ** 20) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 2 ** 30) return `${(bytes / 2 ** 20).toFixed(1)} MB`;
  return `${(bytes / 2 ** 30).toFixed(1)} GB`;
}

/**
 * The notice's one sentence: what happened and how much the caches hold.
 * `held` is lib/caches.js cachedSizes(): kept runs and converted models
 * (count, bytes) and the browser's estimate of the site's storage, when it
 * gives one; `heap` the page's memory for the heap warning.
 */
export function noticeText(kind, { held = null, heap = null } = {}) {
  const runs = held?.runs?.count ?? 0, models = held?.models?.count ?? 0;
  const cached = held
    ? `Cached runs and models hold ${bytesText(held.bytes)} (${runs} run${runs === 1 ? "" : "s"}, ${models} model${models === 1 ? "" : "s"})`
    : "Cached runs and models are kept in this browser";
  const quota = held?.estimate?.quota > 0 ? ` of the ${bytesText(held.estimate.quota)} it allows this site` : "";
  switch (kind) {
    case "quota":
      return `The browser's storage for this site is full, so a run or a model could not be kept. ${cached}${quota}.`;
    case "memory":
      return `The browser ran out of memory. ${cached}; clearing them frees its storage, and reloading the page frees its memory.`;
    case "gpu":
      return `The GPU ran out of memory for this fire's grid, so the run went on on the CPU: choose the standard grid, or Run on the CPU, for the next. ${cached}.`;
    case "storage": {
      const pct = held?.estimate?.quota > 0 ? Math.round((held.estimate.usage / held.estimate.quota) * 100) : null;
      return `This site's storage is ${pct === null ? "nearly" : `${pct} %`} full. ${cached}${quota}.`;
    }
    case "heap": {
      const pct = heap?.jsHeapSizeLimit > 0 ? Math.round((heap.usedJSHeapSize / heap.jsHeapSizeLimit) * 100) : null;
      return `This page is using ${pct === null ? "most" : `${pct} %`} of the memory the browser allows it; reloading the page frees it. ${cached}.`;
    }
    default:
      return cached;
  }
}

/** After clearing: how much was freed, and of what. */
export function freedText(before, after) {
  const freed = Math.max(0, (before?.bytes ?? 0) - (after?.bytes ?? 0));
  const runs = Math.max(0, (before?.runs?.count ?? 0) - (after?.runs?.count ?? 0));
  const models = Math.max(0, (before?.models?.count ?? 0) - (after?.models?.count ?? 0));
  return `Freed ${bytesText(freed)}: ${runs} kept run${runs === 1 ? "" : "s"} and ${models} converted model${models === 1 ? "" : "s"}. Settings are kept.`;
}

// Where a problem is met (the caches, the workers' answers, reportError), it
// is reported here; the page shows the notice (App.jsx). Nothing listens in
// node, so the caches stay testable.
const listeners = new Set();

/** Listen for memory problems: `fn({kind, where})`; returns the call that stops it. */
export function onMemoryProblem(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Report an error if it is a memory problem; returns its kind or null. */
export function reportMemoryProblem(error, where = null) {
  const kind = classifyMemoryError(error);
  if (kind) listeners.forEach((fn) => fn({ kind, where }));
  return kind;
}

/** Report a warning of `kind` ("storage", "heap"). */
export function reportMemoryWarning(kind, where = null) {
  if (WARNING_KINDS.includes(kind)) listeners.forEach((fn) => fn({ kind, where }));
}

/**
 * After a run or a model was kept: warn when the site's storage is over
 * STORAGE_WARN of its quota. Silent where the browser gives no estimate.
 */
export async function checkStorage(where = null, storage = globalThis.navigator?.storage) {
  try {
    const estimate = await storage?.estimate?.();
    if (storageNearlyFull(estimate)) reportMemoryWarning("storage", where);
  } catch {
    // No estimate: nothing to say.
  }
}

/** Warn when the page's heap is over HEAP_WARN of its limit (Chrome only). */
export function checkHeap(where = null, memory = globalThis.performance?.memory) {
  try {
    if (heapNearlyFull(memory)) reportMemoryWarning("heap", where);
  } catch {
    // No reading: nothing to say.
  }
}
