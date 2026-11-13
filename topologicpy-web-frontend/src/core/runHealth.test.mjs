import test from "node:test";
import assert from "node:assert/strict";
import { HEALTH } from "./flamescapes.js";
import { healthReason } from "./runHealth.js";

test("each reason the guard gives reads in words, with its limit", () => {
  assert.match(healthReason({ reason: "non-finite", value: null }), /not a number/);
  assert.match(healthReason({ reason: "temperature", value: 6412.3 }), /6,412 K, past the 3,000 K/);
  assert.match(healthReason({ reason: "speed", value: 10234 }), /10,234 m\/s, past the 50 m\/s/);
  assert.match(healthReason({ reason: "energy", value: 3e9 }), /one and a half times the heat the fire had put in/);
  assert.equal(HEALTH.maxT, 3000);
  assert.equal(HEALTH.maxSpeed, 50);
});

test("an unknown or missing reason still reads", () => {
  assert.equal(healthReason({ reason: "other" }), "the solver went unstable");
  assert.equal(healthReason(null), "the solver went unstable");
});
