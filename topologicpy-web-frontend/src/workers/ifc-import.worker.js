/**
 * IFC to fragments conversion, off the main thread.
 *
 * `IfcLoader.load` runs That Open's `IfcImporter` on the thread that calls it,
 * so converting a model froze the page: in tasks of up to 0.3 s on a desktop
 * and 1.5 s on a 4x-throttled phone for the 2.4 MB Duplex
 * (wiki/research/ifc-handling-survey-2026-10-27.md, items 27 and 3.7). Here the
 * same importer, with the same settings, runs in its own worker; the page only
 * loads the finished fragments buffer, which is the path a cached model takes.
 *
 * After the fragments, the same worker reads what the IFC says about egress
 * (exit doors, roof slabs; src/core/ifcSemantics.js). Its web-ifc is already
 * loaded and compiled, where a second worker cost the page another 419 KB
 * download and another start of the engine, and it began only after the model
 * was drawn.
 *
 * One worker per conversion: terminating it when the job ends hands web-ifc's
 * WASM heap (hundreds of MB on a large file) straight back to the browser.
 *
 * Protocol (plain objects with a `type`):
 *
 *   worker -> page  { type: "ready" }        this module loaded and evaluated
 *   page -> worker  { type: "convert", bytes, name, wasm, webIfc }
 *                                            bytes: the IFC, transferred
 *                   { type: "semantics", bytes, wasm }
 *                                            the semantics alone, for a model
 *                                            whose fragments were cached
 *   worker -> page  { type: "progress", value }       0..1, as IfcLoader reports
 *                   { type: "done", bytes }           fragments, transferred
 *                   { type: "failed", message }       a message for the user
 *                   { type: "unavailable", reason, bytes }
 *                       web-ifc could not start here; the IFC goes back so the
 *                       page can convert it in its own thread
 *                   { type: "semantics", result }     after "done", or alone
 *                   { type: "semanticsFailed", message }
 */
import { DataSet, IfcImporter } from "@thatopen/fragments";
import * as WEBIFC from "web-ifc";

import { trimImporter } from "../core/importSettings.js";
import { ifcSemantics } from "../core/ifcSemantics.js";

// The loader's tape is reserved up front, 64 MiB by default. Reading the
// relations needs far less: with 4 MiB the heap is 16 MiB after the Duplex
// and 33 MiB after the 12 MiB dental clinic, against 133 MiB, with the same
// result; a larger file takes more of it as it goes.
const TAPE_SIZE = 4 * 1024 * 1024;

// web-ifc picks its pthread build whenever the scope is cross-origin isolated.
// That build cannot start under Vite (see vite.config.js) and GitHub Pages
// cannot send the COOP/COEP headers it needs, so this worker is always single
// threaded. IfcImporter creates its own IfcAPI instances and calls `Init()`
// without arguments, so the flag is forced here rather than passed. A failure
// to start is remembered: it means this worker cannot convert anything, which
// is different from this file failing to convert.
let engineFailure = null;
const init = WEBIFC.IfcAPI.prototype.Init;
WEBIFC.IfcAPI.prototype.Init = async function initSingleThreaded(customLocateFileHandler) {
  try {
    return await init.call(this, customLocateFileHandler, true);
  } catch (error) {
    engineFailure = error;
    throw error;
  }
};

self.onmessage = (event) => {
  if (event.data?.type === "convert") void convert(event.data);
  else if (event.data?.type === "semantics") void semantics(new Uint8Array(event.data.bytes), event.data.wasm);
};

async function convert({ bytes: buffer, name = "the file", wasm = {}, webIfc = {} }) {
  const bytes = new Uint8Array(buffer);
  try {
    const importer = new IfcImporter();
    importer.wasm = { path: wasm.path ?? "", absolute: Boolean(wasm.absolute) };
    importer.webIfcSettings = { ...webIfc };
    // Only what the page reads from the fragments (core/importSettings.js).
    trimImporter(importer, WEBIFC, DataSet);

    // The importer reports per IFC class; the page shows whole percent, so
    // one message per percent is all it can use.
    let lastPercent = -1;
    const fragments = await importer.process({
      bytes,
      progressCallback: (value) => {
        const percent = Math.floor(value * 100);
        if (percent === lastPercent) return;
        lastPercent = percent;
        self.postMessage({ type: "progress", value });
      },
    });

    // A view onto part of a larger buffer would carry the rest along when it
    // is transferred and again when it is cached.
    const out =
      fragments.byteOffset === 0 && fragments.byteLength === fragments.buffer.byteLength
        ? fragments
        : fragments.slice();
    self.postMessage({ type: "done", bytes: out }, [out.buffer]);
  } catch (error) {
    if (engineFailure) {
      // web-ifc never opened the file, so its bytes are intact.
      self.postMessage(
        { type: "unavailable", reason: describeEngineFailure(engineFailure, wasm), bytes: buffer },
        [buffer],
      );
      return;
    }
    self.postMessage({ type: "failed", message: describeFailure(error, name, bytes) });
    return;
  }
  await semantics(bytes, wasm);
}

/** Read the egress semantics with a fresh web-ifc and post them. */
async function semantics(bytes, wasm = {}) {
  try {
    const api = new WEBIFC.IfcAPI();
    api.SetWasmPath(wasm.path ?? "", Boolean(wasm.absolute));
    await api.Init();
    api.SetLogLevel(WEBIFC.LogLevel.LOG_LEVEL_OFF);
    const t0 = performance.now();
    const model = api.OpenModel(bytes, { TAPE_SIZE });
    try {
      const result = ifcSemantics(api, model, WEBIFC);
      result.ms = performance.now() - t0;
      self.postMessage({ type: "semantics", result });
    } finally {
      api.CloseModel(model);
    }
  } catch (error) {
    self.postMessage({ type: "semanticsFailed", message: messageOf(error) });
  }
}

function messageOf(error) {
  return String(error?.message || error || "unknown error");
}

function describeEngineFailure(error, wasm) {
  return (
    `the IFC engine (web-ifc WebAssembly) could not start from ${wasm.path || "its default path"}: ` +
    `${messageOf(error)}`
  );
}

const SUPPORTED_SCHEMAS = ["IFC2X3", "IFC4", "IFC4X3"];

/** A message that says what went wrong and what to do about it. */
function describeFailure(error, name, bytes) {
  const raw = messageOf(error);
  const mb = (bytes.byteLength / (1024 * 1024)).toFixed(1);

  if (/out of memory|\bOOM\b|Cannot enlarge memory|allocation failed/i.test(raw)) {
    return (
      `The browser ran out of memory converting ${name} (${mb} MB). Close other tabs and try again, ` +
      "or open it on a desktop browser with more memory."
    );
  }

  // What the first bytes say the file is.
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, 4096)).trimStart();
  if (!head.startsWith("ISO-10303-21")) {
    if (bytes[0] === 0x50 && bytes[1] === 0x4b) {
      return `${name} looks like a zipped IFC (.ifczip). Unzip it and open the .ifc file inside.`;
    }
    if (head.startsWith("<")) {
      return `${name} looks like IFC-XML, which Studio cannot read. Export it from the authoring tool as an .ifc (STEP) file.`;
    }
    return (
      `${name} is not an IFC file: it does not start with the ISO-10303-21 header. ` +
      "Export the model from the authoring tool as IFC2x3 or IFC4 and open that file."
    );
  }
  const schema = /FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'/i.exec(head)?.[1]?.toUpperCase();
  if (schema && !SUPPORTED_SCHEMAS.some((s) => schema.startsWith(s))) {
    return `${name} uses the ${schema} schema, which the IFC engine does not read. Export it as IFC2x3 or IFC4.`;
  }
  return (
    `Could not convert ${name} (${mb} MB): ${raw}. The file may be damaged or use geometry the IFC ` +
    "engine cannot read; re-export it from the authoring tool, or check that it opens in another IFC viewer."
  );
}

self.postMessage({ type: "ready" });
