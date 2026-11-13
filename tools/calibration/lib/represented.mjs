/**
 * Whether a run's grid represents its test's ventilation (the addendum of
 * 2026-11-12, 00:28, to wiki/research/fire-calibration-claim-protocol-2026-11-12.md;
 * plan section 12.6): the fire's open cells, flooded through open faces from
 * the burner's first cell, must reach the domain's open boundary (its sides
 * and top), unless the test was sealed by design (door shut, its leakage below
 * a cell). Geometry only: no measurement enters. On this grid, the app's
 * voxeliser seals Steckler's 0.24 to 0.25 m doors on every cell size and its
 * 0.36 m doors on 0.34 m cells.
 */
import { burnerBase, gridFor } from "./fds.mjs";

/** Tests sealed by design: the door shut, the measured leakage below a cell. */
export const SEALED_BY_DESIGN = Object.freeze({ nistnrc: ["01", "02", "07", "08", "13"], nbs: ["100O"] });

/** Does the fire's gas reach the open boundary on cell h, the grid built with the voxeliser's rule `snap` (plan 13.21)? */
export function reachesOpenBoundary(T, h, snap = "seal") {
  const G = gridFor({ ...T.geometry, fire: [] }, h, undefined, { snap });
  const { grid, solid, at } = G;
  const base = burnerBase(G, T.burner).cells;
  if (!base.length) return false;
  const seen = new Uint8Array(grid.cells);
  const stack = [base[0].c];
  seen[base[0].c] = 1;
  while (stack.length) {
    const c = stack.pop();
    const i = c % grid.nx, j = Math.floor(c / grid.nx) % grid.ny, k = Math.floor(c / (grid.nx * grid.ny));
    if (i === 0 || k === 0 || i === grid.nx - 1 || k === grid.nz - 1 || j === grid.ny - 1) return true;
    for (const [a, b, e] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
      const ii = i + a, jj = j + b, kk = k + e;
      if (ii < 0 || jj < 0 || kk < 0 || ii >= grid.nx || jj >= grid.ny || kk >= grid.nz) continue;
      const n = at(ii, jj, kk);
      if (!seen[n] && !solid[n]) {
        seen[n] = 1;
        stack.push(n);
      }
    }
  }
  return false;
}

/**
 * For each run, whether it represents its test: {id: {represented, why}}.
 * `tests` maps "dataset/test" to the reduced test (tools/calibration/data).
 */
export function representation(runs, tests) {
  const out = {};
  const memo = new Map();
  for (const r of runs) {
    const key = `${r.dataset}/${r.test}`;
    if ((SEALED_BY_DESIGN[r.dataset] || []).includes(String(r.test))) {
      out[r.id] = { represented: true, why: "sealed by design (door shut)" };
      continue;
    }
    const T = tests[key];
    // A test the data mark as shut (UL FSRI's closed-door tests, 2026-11-12) is sealed by design too.
    if (T?.sealedByDesign) {
      out[r.id] = { represented: true, why: "sealed by design (door shut)" };
      continue;
    }
    if (!T) {
      out[r.id] = { represented: true, why: "test not found; kept" };
      continue;
    }
    // Each run's grid as it was built (the suite records `snap` from 2026-11-13; a run without it was "seal").
    const mk = `${key}@${r.cell}@${r.snap ?? "seal"}`;
    if (!memo.has(mk)) memo.set(mk, reachesOpenBoundary(T, r.cell, r.snap ?? "seal"));
    out[r.id] = memo.get(mk) ? { represented: true, why: "the fire's gas reaches the open boundary" } : { represented: false, why: "not represented on this grid: the grid seals the test's opening" };
  }
  return out;
}
