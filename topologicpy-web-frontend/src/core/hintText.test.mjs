import test from "node:test";
import assert from "node:assert/strict";
import { HINT_WORDS, splitHint } from "./hintText.js";

test("a short hint shows whole", () => {
  const text = "Off by default. Each person's speed then drifts about their own.";
  assert.deepEqual(splitHint(text), [text, ""]);
  const exact = Array.from({ length: HINT_WORDS }, (_, i) => `w${i}`).join(" ");
  assert.deepEqual(splitHint(exact), [exact, ""]);
});

test("a long hint folds after its first sentence, and nothing is lost", () => {
  const first = "Each person keeps their own clock.";
  const rest =
    "A door passes at most 1.3 persons a second per metre of effective width, as Kirchner et al. (2002) and the SFPE Handbook give it, so queues form where they should.";
  const [shown, folded] = splitHint(`${first} ${rest}`);
  assert.equal(shown, first);
  assert.equal(folded, rest);
});

test("abbreviations and decimals do not end a sentence", () => {
  const text =
    "Drawn as Kirchner et al. (2002) and Burstedde et al. (2001) draw it at 1.3 persons a second per metre of clear width less 0.15 m a side. The rest follows here with more words.";
  const [shown, folded] = splitHint(text);
  assert.ok(shown.endsWith("a side."), shown);
  assert.equal(folded, "The rest follows here with more words.");
});

test("a first sentence over 30 words is cut at 22 with an ellipsis", () => {
  const words = Array.from({ length: 40 }, (_, i) => `w${i}`);
  const [shown, folded] = splitHint(`${words.join(" ")}.`);
  assert.equal(shown, `${words.slice(0, 22).join(" ")}…`);
  assert.equal(folded, `${words.slice(22).join(" ")}.`);
});
