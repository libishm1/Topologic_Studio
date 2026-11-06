/**
 * Field notes and hints (core/fieldNotes.js):
 *   node --test src/core/fieldNotes.test.mjs
 */
import test from "node:test";
import assert from "node:assert/strict";

import { FIELD_NOTES, FIELD_NOTE_COUNT, HINTS, fieldNote, nextHint } from "./fieldNotes.js";
import { GUIDED_RUNS } from "./guided.js";

const DETAIL = {
  "walk-out": { length: 23.46 },
  "second-way": { first: 20, second: 26 },
  "cut-off": { largest: 95.2 },
  "exit-separation": { passed: 3, checked: 4 },
  "no-safe-time": { places: 1234 },
  embers: { spots: 1 },
  ignition: { label: "Cooking equipment" },
};

test("twelve field notes, each with its own question and a text that reads its details", () => {
  assert.equal(FIELD_NOTE_COUNT, 12);
  assert.equal(new Set(FIELD_NOTES.map((n) => n.id)).size, FIELD_NOTE_COUNT);
  assert.equal(new Set(FIELD_NOTES.map((n) => n.teaser)).size, FIELD_NOTE_COUNT);
  for (const note of FIELD_NOTES) {
    assert.match(note.teaser, /\?$/, note.id);
    const text = note.text(DETAIL[note.id] || {});
    assert.ok(text.length > 30 && !/undefined|NaN/.test(text), `${note.id}: ${text}`);
  }
  assert.match(fieldNote("walk-out").text(DETAIL["walk-out"]), /23\.5 m/);
  assert.match(fieldNote("second-way").text(DETAIL["second-way"]), /26\.0 m against 20\.0 m, 30% longer/);
  assert.match(fieldNote("embers").text({ spots: 2 }), /2 spot fires/);
  assert.equal(fieldNote("nope"), null);
});

test("every guided run ends on a field note that exists", () => {
  for (const run of GUIDED_RUNS) assert.ok(fieldNote(run.note), run.id);
});

test("hints lead past the first screen, one step at a time, as features are used", () => {
  const base = { model: true, graph: false, split: false, route: false, appearance: "ghost", fireOrigin: false, busy: false };
  const used = new Set();
  const at = (step, s = {}) => nextHint({ ...base, ...s, step }, { used })?.id ?? null;

  assert.equal(at("model"), "more-steps");
  assert.equal(at("graph"), "more-steps");
  assert.equal(at("graph", { graph: true, appearance: "solid" }), "ghost");
  assert.equal(at("graph", { graph: true, split: true }), "parts");
  assert.equal(at("graph", { graph: true }), "route-next");
  assert.equal(at("graph", { graph: true, route: true }), null);

  const route = { graph: true, route: true };
  assert.equal(at("route", route), "alternatives");
  used.add("alternatives");
  assert.equal(at("route", route), "code-check");
  used.add("codeCheck");
  assert.equal(at("route", route), "fire-next");

  const sim = { graph: true, route: true };
  assert.equal(at("simulate", sim), "ignition");
  assert.equal(at("simulate", { ...sim, fireOrigin: true }), null);
  used.add("fire");
  assert.equal(at("simulate", { ...sim, fireOrigin: true }), "flamescapes");
  assert.equal(at("route", route), null);
  used.add("flamescapes");
  assert.equal(at("simulate", { ...sim, fireOrigin: true }), "doors-closed");
  used.add("flamescapesClosed");
  assert.equal(at("simulate", { ...sim, fireOrigin: true }), "plan");
  used.add("plan");
  assert.equal(at("simulate", { ...sim, fireOrigin: true }), "agent");
  used.add("agent");
  assert.equal(at("simulate", { ...sim, fireOrigin: true }), null);
});

test("no hint while something runs, when hints are off, or once dismissed", () => {
  const s = { step: "graph", model: true, graph: false };
  assert.equal(nextHint({ ...s, busy: true }), null);
  assert.equal(nextHint(s, { on: false }), null);
  assert.equal(nextHint(s, { dismissed: new Set(["more-steps"]) }), null);
});

test("hint actions go to a step of the workflow or set a known setting", () => {
  const steps = new Set(["model", "graph", "route", "simulate"]);
  for (const h of HINTS) {
    assert.ok(h.steps.every((s) => steps.has(s)), h.id);
    if (h.action?.tab) assert.ok(steps.has(h.action.tab), h.id);
    if (h.action?.settings) {
      for (const key of Object.keys(h.action.settings)) assert.ok(["modelAppearance", "flamescapesDoors"].includes(key), `${h.id}: ${key}`);
    }
  }
});
