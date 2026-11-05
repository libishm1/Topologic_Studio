/**
 * The describer Worker with a stand-in for Workers AI:
 *   node --test cloudflare/describe-worker/test/
 */
import test from "node:test";
import assert from "node:assert/strict";
import worker, { INSTRUCTIONS, MAX_FACTS, MAX_SUMMARY, ROOM_INSTRUCTIONS, checkSuggestions, jsonOf, textOf } from "../src/index.js";
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

// A drafted storey: a corridor traced in two pieces (c1, c2), an office with a
// door to c1, an unlabelled room beside it and a store off on its own.
const SUMMARY = {
  storey: "Ground",
  rooms: [
    { id: "c1", area: 12.4, labels: ["CORR."], neighbours: ["c2", "r1"], doors: ["r1"] },
    { id: "c2", area: 9.1, labels: [], neighbours: ["c1", "r2"], doors: ["r2"] },
    { id: "r1", area: 20, labels: ["OFF. 01"], neighbours: ["c1", "r2"], doors: ["c1"] },
    { id: "r2", area: 6, labels: [], neighbours: ["c2", "r1"], doors: ["c2"] },
    { id: "s1", area: 3, labels: ["ST"], neighbours: [], doors: [] },
  ],
};
const rooms = (body, origin = SITE) =>
  new Request("https://describe.example/rooms", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: origin },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });

test("room suggestions: the model gets the Worker's room instructions and the summary; the page gets checked suggestions", async () => {
  let seen = null;
  const answer = {
    merge: [{ rooms: ["c1", "c2"], why: "One corridor split\nwith no door." }],
    names: [{ room: "r2", name: "Store", class: "service", why: "Small, off the corridor." }],
  };
  const response = await worker.fetch(rooms({ summary: SUMMARY }), env(async (model, input) => {
    seen = input;
    return { response: "```json\n" + JSON.stringify(answer) + "\n```" };
  }));
  assert.equal(response.status, 200);
  assert.equal(seen.messages[0].content, ROOM_INSTRUCTIONS);
  assert.equal(seen.messages[1].content, `Rooms:\n${JSON.stringify(SUMMARY)}`);
  assert.deepEqual(await response.json(), {
    merge: [{ rooms: ["c1", "c2"], why: "One corridor split with no door." }],
    names: [{ room: "r2", name: "Store", class: "Service", why: "Small, off the corridor." }],
    dropped: 0,
    model: "@cf/test/model",
  });
});

test("suggestions that do not hold for the summary are dropped: unknown or repeated rooms, merges that do not touch", () => {
  const checked = checkSuggestions(
    {
      merge: [
        { rooms: ["c1", "s1"] }, // s1 touches nothing
        { rooms: ["c1", "x9"] }, // no such room
        { rooms: ["c2"] }, // one room is no merge
        { rooms: ["c1", "c2", "r2"] }, // connected through c2
        { rooms: ["r2", "r1"] }, // r2 is already merged
      ],
      names: [
        { room: "r2", name: "Store", class: "Bunker" }, // unknown class: name kept
        { room: "r2", name: "Again", class: "Unit" }, // one name per room
        { room: "x9", name: "Ghost", class: "Unit" },
        { room: "s1", name: "  ", class: "Unit" },
        { room: "s1", name: "Stair\u0007 one", class: "STAIR" },
      ],
    },
    SUMMARY,
  );
  assert.deepEqual(checked.merge, [{ rooms: ["c1", "c2", "r2"], why: "" }]);
  assert.deepEqual(checked.names, [
    { room: "r2", name: "Store", class: null, why: "" },
    { room: "s1", name: "Stair one", class: "Stair", why: "" },
  ]);
  assert.equal(checked.dropped, 7);
  assert.deepEqual(checkSuggestions(null, SUMMARY), { merge: [], names: [], dropped: 0 });
});

test("room requests: other sites, no rooms, oversized summaries and non-JSON answers are refused", async () => {
  const never = env(async () => assert.fail("the model must not run"));
  assert.equal((await worker.fetch(rooms({ summary: SUMMARY }, "https://evil.example"), never)).status, 403);
  assert.equal((await worker.fetch(rooms({ summary: { rooms: [] } }), never)).status, 400);
  assert.equal((await worker.fetch(rooms("{not json"), never)).status, 400);
  const big = { rooms: [{ id: "a", labels: ["x".repeat(MAX_SUMMARY)] }] };
  assert.equal((await worker.fetch(rooms({ summary: big }), never)).status, 413);
  const prose = await worker.fetch(rooms({ summary: SUMMARY }), env(async () => ({ response: "I think c1 and c2 are one corridor." })));
  assert.equal(prose.status, 502);
  assert.deepEqual(jsonOf('Sure: {"merge": []} done'), { merge: [] });
  assert.equal(jsonOf("{broken"), null);
});
