/**
 * What a visit can use (core/session.js) and how a request's failure is told
 * apart (lib/api.js): wiki/architecture/state-machines.md, SM-4 and SM-6.
 *   node --test src/core/session.test.mjs
 */
import test, { mock } from "node:test";
import assert from "node:assert/strict";

import { effectiveEngine, isStaleSite } from "./session.js";
import { ApiError, api, isGraphGone } from "../lib/api.js";
import { humanCheckFailure } from "../lib/turnstile.js";

test("a failed human check says why, from Cloudflare's error code", () => {
  assert.match(humanCheckFailure("110200"), /does not accept this site's address \(code 110200\)/);
  assert.match(humanCheckFailure("600010"), /could not confirm you are human \(code 600010\)\. A VPN/);
  assert.match(humanCheckFailure("300030"), /could not confirm you are human/);
  assert.match(humanCheckFailure("110620"), /timed out/);
  assert.match(humanCheckFailure("400020"), /key is not valid/);
  assert.equal(humanCheckFailure(undefined), "The human check failed. Reload the page and try again.");
  assert.match(humanCheckFailure("999999"), /failed \(code 999999\)/);
});

test("a saved TopologicPy engine gives way to the browser's while the server lists no engines", () => {
  // Asleep, unreachable or absent: routing runs in the browser rather than refusing.
  assert.equal(effectiveEngine("topologicpy", null), "fast");
  assert.equal(effectiveEngine("topologicpy", []), "fast");
  assert.equal(effectiveEngine("fast", null), "fast");
  // Listed: the saved choice applies again.
  assert.equal(effectiveEngine("topologicpy", ["fast", "topologicpy"]), "topologicpy");
  assert.equal(effectiveEngine("topologicpy", ["fast"]), "fast");
});

test("a part of the site replaced by a newer deploy is told apart, in each browser's words", () => {
  assert.ok(isStaleSite(new TypeError("Failed to fetch dynamically imported module: https://x/assets/sprinklers-abc.js")));
  assert.ok(isStaleSite(new TypeError("error loading dynamically imported module: https://x/assets/a.js")));
  assert.ok(isStaleSite(new TypeError("Importing a module script failed.")));
  assert.ok(!isStaleSite(new Error("No route exists between those two points.")));
});

test("the server's lost graph is told apart from other refusals", () => {
  assert.ok(isGraphGone(new ApiError("x", { status: 400, detail: "No navigation graph available. Build one first (POST /api/ifc/graph)." })));
  assert.ok(isGraphGone(new ApiError("x", { status: 400, detail: "No graph available." })));
  assert.ok(!isGraphGone(new ApiError("x", { status: 400, detail: "Graph has 70000 nodes, above the 60000 limit" })));
  assert.ok(!isGraphGone(new Error("No navigation graph available.")));
});

/** A fetch that answers only when its signal aborts it. */
const hangingFetch = (_url, { signal }) =>
  new Promise((_resolve, reject) => {
    const abort = () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    if (signal.aborted) abort();
    else signal.addEventListener("abort", abort, { once: true });
  });

test("a request that times out or is cancelled is not a server gone away (only that marks it offline)", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = hangingFetch;
  try {
    // Cancelled by the caller.
    const controller = new AbortController();
    controller.abort();
    const cancelled = await api.health(controller.signal).catch((error) => error);
    assert.equal(cancelled.kind, "aborted");
    // Its own timer (8 s for the health probe).
    mock.timers.enable({ apis: ["setTimeout"] });
    const pending = api.health().catch((error) => error);
    mock.timers.tick(8000);
    const timedOut = await pending;
    assert.equal(timedOut.kind, "timeout");
  } finally {
    mock.timers.reset();
    globalThis.fetch = realFetch;
  }
  // Unreachable: the only kind that marks the backend offline.
  globalThis.fetch = () => Promise.reject(new TypeError("Failed to fetch"));
  try {
    assert.equal((await api.health().catch((error) => error)).kind, "network");
  } finally {
    globalThis.fetch = realFetch;
  }
});
