/**
 * Jobs for a module worker (lib/workerJob.js): the contract the page's
 * cancels rely on (wiki/architecture/state-machines.md, SM-1 and SM-2).
 *   node --test src/core/workerJob.test.mjs
 */
import test from "node:test";
import assert from "node:assert/strict";

import { isCancelled, workerClient } from "../lib/workerJob.js";

/** A worker that answers only when told to. */
class FakeWorker {
  constructor() {
    this.listeners = { message: [], error: [], messageerror: [] };
    this.posted = [];
    this.terminated = false;
  }
  addEventListener(type, fn) {
    this.listeners[type].push(fn);
  }
  postMessage(message) {
    this.posted.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  /** A message from the worker, as the browser would dispatch it. */
  send(data) {
    for (const fn of this.listeners.message) fn({ data });
  }
}

const clientWith = () => {
  const made = [];
  const client = workerClient(() => {
    const w = new FakeWorker();
    made.push(w);
    return w;
  });
  return { client, made };
};

const settled = (promise) => promise.then((value) => ({ value }), (error) => ({ error }));

test("cancel() ends the worker and every job in it, and the next job gets a fresh worker", async () => {
  const { client, made } = clientWith();
  const a = settled(client.run({ payload: "a" }));
  const b = settled(client.run({ payload: "b" }));
  assert.equal(made.length, 1);
  client.cancel();
  assert.ok(made[0].terminated);
  assert.ok(isCancelled((await a).error) && isCancelled((await b).error));
  assert.equal(client.busy, false);
  const c = client.run({ payload: "c" });
  assert.equal(made.length, 2);
  made[1].send({ id: made[1].posted[0].id, ok: true, result: 3 });
  assert.equal(await c, 3);
});

test("a message for a cancelled job reaches nobody: a frame already queued when the worker ended is dropped", async () => {
  const { client, made } = clientWith();
  const frames = [];
  const job = settled(client.run({ payload: "fire" }, [], { onProgress: (p) => frames.push(p) }));
  const { id } = made[0].posted[0];
  made[0].send({ id, progress: 1 });
  client.cancel();
  made[0].send({ id, progress: 2 });
  made[0].send({ id, ok: true, result: "late" });
  assert.deepEqual(frames, [1]);
  assert.ok(isCancelled((await job).error));
});

test("separate clients are separate workers: ending one lane leaves the others' jobs running (SM-2)", async () => {
  const plan = clientWith();
  const report = clientWith();
  const planned = settled(plan.client.run({ payload: "plan" }));
  const reported = settled(report.client.run({ payload: "report" }));
  // The report's Stop.
  report.client.cancel();
  assert.ok(isCancelled((await reported).error));
  // The plan is still in its worker and is answered.
  assert.equal(plan.made[0].terminated, false);
  plan.made[0].send({ id: plan.made[0].posted[0].id, ok: true, result: "routes" });
  assert.deepEqual(await planned, { value: "routes" });
});

test("a worker that fails rejects its jobs with the client's message, and is replaced", async () => {
  const made = [];
  const client = workerClient(
    () => {
      const w = new FakeWorker();
      made.push(w);
      return w;
    },
    { failure: "it stopped" },
  );
  const job = settled(client.run({ payload: 1 }));
  for (const fn of made[0].listeners.error) fn({ preventDefault() {} });
  assert.equal((await job).error.message, "it stopped");
  client.run({ payload: 2 });
  assert.equal(made.length, 2);
});
