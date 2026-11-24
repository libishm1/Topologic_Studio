// The people-leaving model's verification tests (tools/validation/crowd-validation.mjs:
// the RiMEA guideline's and NIST TN 1822's cases) run with the unit tests, so a change to
// crowd.js that breaks one shows at once. The two known failures are findings about the
// model, recorded in wiki/research/crowd-vv-report-2026-11-24.md; a third would be new.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const harness = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "tools", "validation", "crowd-validation.mjs");

test("the RiMEA and NIST verification cases: the same passes, and only the two known failures", () => {
  const out = execFileSync(process.execPath, [harness, "--quick", "--no-write"], { encoding: "utf8" });
  const line = out.split(/\r?\n/).find((l) => l.startsWith("SUMMARY "));
  assert.ok(line, "the harness ends with its summary");
  const summary = JSON.parse(line.slice(8));
  assert.deepEqual(summary.failed.sort(), ["RiMEA-12b", "RiMEA-16"]);
  assert.ok(summary.pass >= 20, `${summary.pass} pass`);
});
