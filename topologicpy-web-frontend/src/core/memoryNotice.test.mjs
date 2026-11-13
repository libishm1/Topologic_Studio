/** Running out of memory or storage, and the notice that offers to clear the caches (memoryNotice.js). */
import test from "node:test";
import assert from "node:assert/strict";
import {
  bytesText,
  checkHeap,
  checkStorage,
  classifyMemoryError,
  freedText,
  heapNearlyFull,
  noticeText,
  onMemoryProblem,
  reportMemoryProblem,
  shouldShow,
  storageNearlyFull,
} from "./memoryNotice.js";

/** A DOMException as IndexedDB throws it, without needing the DOM. */
const domException = (message, name) => Object.assign(new Error(message), { name });

test("errors are told apart: a full store, a failed allocation, the GPU's memory, anything else", () => {
  assert.equal(classifyMemoryError(domException("The quota has been exceeded.", "QuotaExceededError")), "quota");
  assert.equal(classifyMemoryError({ name: "Error", code: 22, message: "" }), "quota");
  assert.equal(classifyMemoryError(new RangeError("Array buffer allocation failed")), "memory");
  assert.equal(classifyMemoryError(new RangeError("Invalid array length")), "memory");
  assert.equal(classifyMemoryError(new RangeError("Invalid typed array length: 4294967296")), "memory");
  // A worker's answer: its message as an Error, and its kind when it posted one.
  assert.equal(classifyMemoryError(new Error("Array buffer allocation failed")), "memory");
  assert.equal(classifyMemoryError(Object.assign(new Error("anything"), { kind: "gpu" })), "gpu");
  assert.equal(classifyMemoryError(new Error("The Flamescapes fire solver stopped, usually because the browser ran out of memory. Try the CPU or a shorter run.")), "memory");
  assert.equal(classifyMemoryError(domException("Out of memory.", "GPUOutOfMemoryError")), "gpu");
  // Not memory.
  assert.equal(classifyMemoryError(new RangeError("Maximum call stack size exceeded")), null);
  assert.equal(classifyMemoryError(new TypeError("x is undefined")), null);
  assert.equal(classifyMemoryError(new Error("cancelled")), null);
  assert.equal(classifyMemoryError(null), null);
});

test("storage over 80 % of its quota, and the heap over 85 % of its limit, warn; unknown never does", () => {
  assert.equal(storageNearlyFull({ usage: 81, quota: 100 }), true);
  assert.equal(storageNearlyFull({ usage: 80, quota: 100 }), false);
  assert.equal(storageNearlyFull({ usage: 81 }), false);
  assert.equal(storageNearlyFull(undefined), false);
  assert.equal(heapNearlyFull({ usedJSHeapSize: 3.5e9, jsHeapSizeLimit: 4e9 }), true);
  assert.equal(heapNearlyFull({ usedJSHeapSize: 3e9, jsHeapSizeLimit: 4e9 }), false);
  assert.equal(heapNearlyFull({}), false);
});

test("a warning shows once a visit; an error again each time it happens, never twice at once", () => {
  assert.equal(shouldShow("storage", { shown: new Set() }), true);
  assert.equal(shouldShow("storage", { shown: new Set(["storage"]) }), false);
  assert.equal(shouldShow("heap", { shown: new Set(["heap"]) }), false);
  assert.equal(shouldShow("quota", { shown: new Set(["quota"]) }), true);
  assert.equal(shouldShow("quota", { shown: new Set(["quota"]), showing: "quota" }), false);
  assert.equal(shouldShow("gpu", { showing: "quota" }), true);
  assert.equal(shouldShow("other"), false);
});

test("the notice says what happened and how much the caches hold; after clearing, what was freed", () => {
  const held = { runs: { count: 3, bytes: 120 * 2 ** 20 }, models: { count: 2, bytes: 40 * 2 ** 20 }, bytes: 160 * 2 ** 20, estimate: { usage: 170 * 2 ** 20, quota: 200 * 2 ** 20 } };
  assert.equal(noticeText("quota", { held }), "The browser's storage for this site is full, so a run or a model could not be kept. Cached runs and models hold 160.0 MB (3 runs, 2 models) of the 200.0 MB it allows this site.");
  assert.match(noticeText("memory", { held }), /^The browser ran out of memory\. .*reloading the page frees its memory\.$/);
  assert.match(noticeText("gpu", { held }), /standard grid, or Run on the CPU/);
  assert.match(noticeText("storage", { held }), /^This site's storage is 85 % full\./);
  assert.match(noticeText("heap", { heap: { usedJSHeapSize: 9, jsHeapSizeLimit: 10 } }), /^This page is using 90 % of the memory/);
  assert.match(noticeText("quota"), /Cached runs and models are kept in this browser\.$/);
  const after = { runs: { count: 0, bytes: 0 }, models: { count: 0, bytes: 0 }, bytes: 0 };
  assert.equal(freedText(held, after), "Freed 160.0 MB: 3 kept runs and 2 converted models. Settings are kept.");
  assert.equal(bytesText(0), "0 MB");
  assert.equal(bytesText(512), "1 KB");
  assert.equal(bytesText(3 * 2 ** 30), "3.0 GB");
});

test("a problem reported where it is met reaches the page's listener; any other error does not", async () => {
  const heard = [];
  const stop = onMemoryProblem((event) => heard.push(event));
  assert.equal(reportMemoryProblem(domException("full", "QuotaExceededError"), "keeping a run"), "quota");
  assert.equal(reportMemoryProblem(new Error("network"), "x"), null);
  await checkStorage("keeping a run", { estimate: async () => ({ usage: 95, quota: 100 }) });
  await checkStorage("keeping a run", { estimate: async () => ({ usage: 5, quota: 100 }) });
  await checkStorage("keeping a run", { estimate: async () => { throw new Error("unsupported"); } });
  await checkStorage("keeping a run", undefined);
  checkHeap("after a run", { usedJSHeapSize: 99, jsHeapSizeLimit: 100 });
  checkHeap("after a run", undefined);
  stop();
  reportMemoryProblem(domException("full", "QuotaExceededError"));
  assert.deepEqual(heard, [
    { kind: "quota", where: "keeping a run" },
    { kind: "storage", where: "keeping a run" },
    { kind: "heap", where: "after a run" },
  ]);
});
