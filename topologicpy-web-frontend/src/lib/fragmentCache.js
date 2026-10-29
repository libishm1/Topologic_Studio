/**
 * IndexedDB cache of converted `.frag` buffers, keyed by IFC content hash.
 *
 * That Open's own guidance is that parsing IFC at runtime is too slow for
 * production and that models should be converted to fragments once and reused.
 * Classic re-parsed the IFC on every single load - and then parsed it a second
 * time in the same session just to read category IDs.
 *
 * Here the first load pays the conversion; every load after that reads a
 * ready-made fragments buffer straight out of the browser, which turns a
 * multi-second parse into a fetch from disk. The egress semantics read from
 * the same IFC (core/ifcSemantics.js) are kept with it, so a cached model
 * opens without starting web-ifc at all.
 */
import { converterId } from "../core/importSettings.js";

/**
 * What produced a cached buffer. The key was the IFC's hash alone, so a model
 * converted by an older fragments or web-ifc, or with other importer
 * settings, was served again after an upgrade
 * (wiki/research/ifc-handling-survey-2026-10-27.md, item 4). The settings
 * are numbered by IMPORT_SETTINGS in core/importSettings.js.
 *
 * The versions are the installed ones, put in at build time by
 * vite.config.js: package.json holds ranges ("~3.4.7" whatever patch is
 * installed), and importing it bundled the whole file (review m10).
 */
export const CONVERTER = converterId(import.meta.env.FRAGMENTS_VERSION, import.meta.env.WEBIFC_VERSION);

const DB_NAME = "topologic-studio";
const DB_VERSION = 1;
const STORE = "fragments";
const META = "meta";

/** Evict least-recently-used entries past this budget. */
const MAX_BYTES = 512 * 1024 * 1024;
const MAX_ENTRIES = 24;

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB unavailable"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
      if (!db.objectStoreNames.contains(META)) db.createObjectStore(META);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).catch((error) => {
    // Private browsing and hardened profiles can refuse IndexedDB outright.
    // The cache is an optimisation, so a failure here must never break loading.
    dbPromise = null;
    throw error;
  });
  return dbPromise;
}

function tx(db, stores, mode) {
  return db.transaction(stores, mode);
}

function promisify(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Content hash of a file. SHA-256 over the whole buffer via SubtleCrypto,
 * which is native and comfortably faster than parsing the IFC it identifies.
 * Falls back to a size+name key when SubtleCrypto is unavailable (http origins).
 */
export async function hashFile(file) {
  const buffer = await file.arrayBuffer();
  try {
    if (globalThis.crypto?.subtle) {
      const digest = await crypto.subtle.digest("SHA-256", buffer);
      const bytes = new Uint8Array(digest);
      let hex = "";
      for (let i = 0; i < bytes.length; i += 1) {
        hex += bytes[i].toString(16).padStart(2, "0");
      }
      return { key: `${hex}|${CONVERTER}`, buffer };
    }
  } catch {
    /* fall through */
  }
  const fallback = `${file.name}:${file.size}:${file.lastModified}|${CONVERTER}`;
  return { key: fallback, buffer };
}

/**
 * @returns {Promise<{buffer: ArrayBuffer, semantics: object|null}|null>}
 *          null when the model is not cached; semantics null when they were
 *          not stored with it.
 */
export async function readCached(key) {
  try {
    const db = await openDb();
    const record = await promisify(tx(db, [STORE], "readonly").objectStore(STORE).get(key));
    if (!record?.buffer) return null;
    // Touch the access time so eviction stays LRU rather than FIFO.
    void touch(db, key).catch(() => {});
    return { buffer: record.buffer, semantics: record.semantics ?? null };
  } catch {
    return null;
  }
}

/** Add semantics to a cached model that was stored without them. */
export async function writeCachedSemantics(key, semantics) {
  try {
    const db = await openDb();
    const transaction = tx(db, [STORE], "readwrite");
    const store = transaction.objectStore(STORE);
    const record = await promisify(store.get(key));
    if (record) store.put({ ...record, semantics }, key);
    await new Promise((resolve, reject) => {
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
    return Boolean(record);
  } catch {
    return false;
  }
}

export async function writeCached(key, buffer, meta = {}, semantics = null) {
  try {
    const db = await openDb();
    const size = buffer.byteLength ?? 0;
    const transaction = tx(db, [STORE, META], "readwrite");
    transaction.objectStore(STORE).put({ buffer, size, semantics }, key);
    transaction.objectStore(META).put(
      { key, size, accessed: Date.now(), created: Date.now(), ...meta },
      key,
    );
    await new Promise((resolve, reject) => {
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
    void evict(db).catch(() => {});
    askToPersist();
    return true;
  } catch {
    // A quota error just means no cache for this model; loading still works.
    return false;
  }
}

let persistAsked = false;

/**
 * Ask the browser not to clear the cache under storage pressure; it may say
 * no, which changes nothing. Once per session, after the first conversion is
 * stored (writes follow conversions only), and not when the cache is already
 * kept: Firefox asks the user each time it is called, and it was called on
 * every write.
 */
function askToPersist() {
  if (persistAsked) return;
  persistAsked = true;
  const storage = globalThis.navigator?.storage;
  if (!storage?.persist) return;
  Promise.resolve(storage.persisted?.())
    .then((kept) => (kept ? null : storage.persist()))
    .catch(() => {});
}

async function touch(db, key) {
  const transaction = tx(db, [META], "readwrite");
  const store = transaction.objectStore(META);
  const record = await promisify(store.get(key));
  if (record) store.put({ ...record, accessed: Date.now() }, key);
}

async function evict(db) {
  const entries = await promisify(tx(db, [META], "readonly").objectStore(META).getAll());
  if (!entries?.length) return;

  const total = entries.reduce((sum, entry) => sum + (entry.size || 0), 0);
  if (total <= MAX_BYTES && entries.length <= MAX_ENTRIES) return;

  const ordered = [...entries].sort((a, b) => (a.accessed || 0) - (b.accessed || 0));
  let running = total;
  let count = entries.length;
  const doomed = [];
  for (const entry of ordered) {
    if (running <= MAX_BYTES && count <= MAX_ENTRIES) break;
    doomed.push(entry.key);
    running -= entry.size || 0;
    count -= 1;
  }
  if (!doomed.length) return;

  const transaction = tx(db, [STORE, META], "readwrite");
  for (const key of doomed) {
    transaction.objectStore(STORE).delete(key);
    transaction.objectStore(META).delete(key);
  }
}

export async function listCached() {
  try {
    const db = await openDb();
    const entries = await promisify(tx(db, [META], "readonly").objectStore(META).getAll());
    return (entries || []).sort((a, b) => (b.accessed || 0) - (a.accessed || 0));
  } catch {
    return [];
  }
}

export async function clearCache() {
  try {
    const db = await openDb();
    const transaction = tx(db, [STORE, META], "readwrite");
    transaction.objectStore(STORE).clear();
    transaction.objectStore(META).clear();
    await new Promise((resolve) => {
      transaction.oncomplete = resolve;
      transaction.onerror = resolve;
    });
    return true;
  } catch {
    return false;
  }
}

export async function cacheSize() {
  const entries = await listCached();
  return {
    count: entries.length,
    bytes: entries.reduce((sum, entry) => sum + (entry.size || 0), 0),
  };
}
