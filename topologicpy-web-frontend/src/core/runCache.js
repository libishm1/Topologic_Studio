/**
 * Finished Flamescapes runs kept in this browser (IndexedDB), so a run asked
 * for again, after a reload too, plays from its stored frames instead of
 * being computed. A run is found by a fingerprint of the walkable graph and
 * of the scene's geometry, the run's inputs, and RUN_CACHE_VERSION. The
 * store holds at most BUDGET bytes; the least recently used runs go first,
 * and a run larger than MAX_RUN is not kept.
 *
 * The frames live apart from a small record of each run (its size and when
 * it was last used), so making room never reads a run's frames.
 *
 * Any failure of the store (a private window, blocked storage, a full disk)
 * leaves the run to be computed as before: the store only saves time.
 */

/** Bump with any change to what the solver computes or how a frame is laid out. */
// 2 (2026-11-07): frames carry the ceiling detectors (flamescapesRun.js ceilingDetectors).
// 3 (2026-11-07): water by fire class (fireClasses.js waterResponse): liquid,
// metal and pack first items hold no water of their own and take no cap.
// 4 (2026-11-08): suspended ceilings in the fire grid, and the walls' own
// surface coefficient (flamescapes.js wallConvection).
// 5 (2026-11-08): the first item as a t-squared design fire by default
// (fireClasses.js designFireFor; flamescapesRun.js itemBurner).
// 6 (2026-11-08): the design-fire item's flame is in the frames' particles
// (flamescapesRun.js withBurnerFlame); kept runs had none until the contents caught.
// 7 (2026-11-08): that flame drawn as a flame (narrowing, flickering, at its
// cells' temperatures) instead of a column of points at McCaffrey's.
// 8 (2026-11-09): the CPU's radiation emits from the stage's starting
// temperatures, with the GPU's rays (each cell's own stream; flamescapes.js
// radiation, cellStream); flame particles
// beyond their share are refused by a hash on both paths, not in cell or
// thread order (particleStep, kernels.js spawnFlame).
// 9 (2026-11-11): spaces named as voids ("open to below") give the fire grid no
// floor, and a grid clipped round a fire under one reaches the atrium's top
// (flamescapesRun.js atriumTop): the NIBS Clinic's atria held their smoke at
// the ground floor.
// 10 (2026-11-11): the fire calibration's proposals A to C
// (wiki/research/fire-calibration-plan-2026-11-11.md): each frame's steps cut
// by the fastest gas and the confinement (flamescapes.js substepsFor), the
// confinement strength fitted to 6 (was 10), and a run that diverges stopped
// with its health in the result (healthVerdict).
export const RUN_CACHE_VERSION = 10;
export const BUDGET = 256 * 2 ** 20;
export const MAX_RUN = 96 * 2 ** 20;

const DB_NAME = "topologic-studio-runs";
const DB_VERSION = 1;
const META = "meta";
const FRAMES = "frames";

/** FNV-1a over the bytes of typed arrays, every `stride`-th 32-bit word. */
export function fingerprint(arrays, stride = 1) {
  let h = 0x811c9dc5;
  const mix = (word) => {
    for (let b = 0; b < 4; b += 1) {
      h ^= (word >>> (b * 8)) & 0xff;
      h = Math.imul(h, 0x01000193);
    }
  };
  for (const a of arrays) {
    if (!a) continue;
    const words = new Uint32Array(a.buffer, a.byteOffset, Math.floor(a.byteLength / 4));
    mix(words.length);
    for (let i = 0; i < words.length; i += stride) mix(words[i]);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

/** The bytes a run's frames hold (their typed arrays, nested one level). */
export function runBytes(frames) {
  let bytes = 0;
  const add = (v) => {
    if (ArrayBuffer.isView(v)) bytes += v.byteLength;
  };
  for (const f of frames || []) {
    for (const v of Object.values(f)) {
      add(v);
      if (v && typeof v === "object" && !ArrayBuffer.isView(v)) for (const w of Object.values(v)) add(w);
    }
  }
  return bytes;
}

/**
 * Which stored runs to drop so `incoming` bytes fit in `budget`: the least
 * recently used first. `metas` are [{key, bytes, usedAt}].
 */
export function evictionPlan(metas, incoming, budget = BUDGET) {
  let total = metas.reduce((a, m) => a + m.bytes, 0) + incoming;
  const drop = [];
  for (const m of [...metas].sort((a, b) => a.usedAt - b.usedAt)) {
    if (total <= budget) break;
    drop.push(m.key);
    total -= m.bytes;
  }
  return drop;
}

let opening = null;
function open() {
  if (opening) return opening;
  opening = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("no IndexedDB"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(META)) db.createObjectStore(META, { keyPath: "key" });
      if (!db.objectStoreNames.contains(FRAMES)) db.createObjectStore(FRAMES);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  opening.catch(() => {
    opening = null;
  });
  return opening;
}

const done = (tx) =>
  new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
const result = (request) =>
  new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

/** A stored run, `{info, frames, result, savedAt}`, or null; marks it used. */
export async function readRun(key) {
  try {
    const db = await open();
    const tx = db.transaction([META, FRAMES], "readwrite");
    const meta = await result(tx.objectStore(META).get(key));
    if (!meta) return null;
    const run = await result(tx.objectStore(FRAMES).get(key));
    if (!run) return null;
    tx.objectStore(META).put({ ...meta, usedAt: Date.now() });
    await done(tx);
    return { ...run, savedAt: meta.savedAt };
  } catch {
    return null;
  }
}

/** Keep a finished run; returns whether it was kept. */
export async function writeRun(key, { info, frames, result: runResult }) {
  try {
    const bytes = runBytes(frames);
    if (bytes > MAX_RUN) return false;
    const db = await open();
    const metas = await result(db.transaction(META).objectStore(META).getAll());
    const drop = evictionPlan(metas.filter((m) => m.key !== key), bytes);
    const tx = db.transaction([META, FRAMES], "readwrite");
    for (const k of drop) {
      tx.objectStore(META).delete(k);
      tx.objectStore(FRAMES).delete(k);
    }
    const now = Date.now();
    tx.objectStore(FRAMES).put({ info, frames, result: runResult }, key);
    tx.objectStore(META).put({ key, bytes, usedAt: now, savedAt: now });
    await done(tx);
    return true;
  } catch {
    return false;
  }
}

/** How many runs are kept and their bytes. */
export async function storedRuns() {
  try {
    const db = await open();
    const metas = await result(db.transaction(META).objectStore(META).getAll());
    return { count: metas.length, bytes: metas.reduce((a, m) => a + m.bytes, 0) };
  } catch {
    return { count: 0, bytes: 0 };
  }
}

/** Forget every kept run. */
export async function clearRuns() {
  try {
    const db = await open();
    const tx = db.transaction([META, FRAMES], "readwrite");
    tx.objectStore(META).clear();
    tx.objectStore(FRAMES).clear();
    await done(tx);
  } catch {
    // Nothing kept, or the store is unavailable: nothing to forget.
  }
}
