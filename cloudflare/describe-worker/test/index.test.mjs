/**
 * The describer Worker with a stand-in for Workers AI:
 *   node --test cloudflare/describe-worker/test/
 */
import test from "node:test";
import assert from "node:assert/strict";
import worker, { INSTRUCTIONS, MAX_FACTS, textOf } from "../src/index.js";
import { DESCRIBE_INSTRUCTIONS } from "../../../topologicpy-web-frontend/src/core/describeFacts.js";

const SITE = "https://libishm1.github.io";
const env = (run) => ({ ALLOWED_ORIGINS: `${SITE},http://localhost:5173`, MODEL: "@cf/test/model", AI: { run } });
const post = (body, origin = SITE) =>
  new Request("https://describe.example/describe", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: origin },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });

test("the Worker's instructions are the page's, so both engines are told the same", () => {
  assert.equal(INSTRUCTIONS, DESCRIBE_INSTRUCTIONS);
});

test("a description from the facts: the model gets the Worker's instructions and the facts, the page gets the text", async () => {
  let seen = null;
  const response = await worker.fetch(post({ facts: "- Doors: 14." }), env(async (model, input) => {
    seen = { model, input };
    return { response: "A two-storey dwelling." };
  }));
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Access-Control-Allow-Origin"), SITE);
  assert.deepEqual(await response.json(), { text: "A two-storey dwelling.", model: "@cf/test/model" });
  assert.equal(seen.model, "@cf/test/model");
  assert.equal(seen.input.messages[0].content, INSTRUCTIONS);
  assert.equal(seen.input.messages[1].content, "Facts:\n- Doors: 14.");
});

test("other sites, empty or oversized facts and broken JSON are refused; facts are capped", async () => {
  const never = env(async () => assert.fail("the model must not run"));
  assert.equal((await worker.fetch(post({ facts: "x" }, "https://evil.example"), never)).status, 403);
  assert.equal((await worker.fetch(post({ facts: "  " }), never)).status, 400);
  assert.equal((await worker.fetch(post("{not json"), never)).status, 400);
  let length = 0;
  await worker.fetch(post({ facts: "y".repeat(MAX_FACTS + 500) }), env(async (m, input) => {
    length = input.messages[1].content.length;
    return { response: "ok" };
  }));
  assert.equal(length, "Facts:\n".length + MAX_FACTS);
  const preflight = await worker.fetch(new Request("https://describe.example/describe", { method: "OPTIONS", headers: { Origin: SITE } }), never);
  assert.equal(preflight.status, 204);
  const health = await worker.fetch(new Request("https://describe.example/health", { headers: { Origin: SITE } }), never);
  assert.deepEqual(await health.json(), { ok: true, model: "@cf/test/model" });
});

test("a model that fails, or answers in another shape, is handled", async () => {
  const failed = await worker.fetch(post({ facts: "- x" }), env(async () => {
    throw new Error("daily allowance used");
  }));
  assert.equal(failed.status, 503);
  assert.match((await failed.json()).error, /daily allowance used/);
  assert.equal(textOf({ choices: [{ message: { content: "chat" } }] }), "chat");
  assert.equal(textOf({ output: [{ content: [{ type: "output_text", text: "responses" }] }] }), "responses");
  assert.equal(textOf({}), "");
});
