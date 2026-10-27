// Run with: node --test src/core/float-error.property.test.mjs
//
// Property test for FireEgress/FloatError.lean (Lean project C:\Users\lmurugesan\lean\fireegress):
// floating-point error in the standard model fl(x op y) = (x op y)(1 + d), |d| <= u, with
// u = 2^-53 (doubles) and u = 2^-24 (Float32Array / Math.fround). Every "exact" value
// below is a BigInt dyadic rational built from the exact binary value of the doubles, so
// the checks are the theorems' real-number statements, not float approximations of them.
//
//   (1) flDet_err, flDet_err_exact, flDet_sign_certain(_shewchuk, _code),
//       segmentsCross_float_exact: each orientation determinant evaluated as lattice.js does,
//       f = (px - ox) * (qy - oy) - (py - oy) * (qx - ox), satisfies |f - d| <= errA(u)(|l| + |r|)
//       and |f - d| <= ((1+u)^4 - 1)(|X| + |Y|); so |f| > B certifies the sign, every sign
//       disagreement has |f| <= B, and segmentsCross is exact when all four clear B. The
//       bound the code would compute in doubles, fl((4 + 16u)u * fl(|l| + |r|)), is >= B.
//   (5) misclassified_near: whenever segmentsCross disagrees with exact arithmetic, some
//       determinant has the wrong sign, |d| <= B and d^2 <= kappa^2 |OP|^2 |OQ|^2.
//   (2) flSum_err / route_sum_rel_err: left-to-right double sums of non-negative doubles
//       (route lengths) are within gamma_{n-1} = (n-1)u / (1 - (n-1)u) of the exact sum.
//   (4) studio_float32_length_err: Math.fround coordinates within 100 m of the origin move a
//       link length by at most 2 * 2^-24 * 100 (checked exactly, squaring instead of sqrt).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

// segmentsCross is module-private; load the real function text from lattice.js.
const source = readFileSync(new URL("./lattice.js", import.meta.url), "utf8");
const match = source.match(/function segmentsCross\(([^)]*)\)\s*\{([\s\S]*?)\n\}/);
assert.ok(match, "segmentsCross not found in lattice.js");
const segmentsCross = new Function(...match[1].split(",").map((s) => s.trim()), match[2]);

/** mulberry32, as rngFrom in src/core/rl.js. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------ exact dyadic arithmetic

/** A finite double as { m, e } with value m * 2^e, exactly. */
function dy(x) {
  assert.ok(Number.isFinite(x), `not finite: ${x}`);
  if (x === 0) return { m: 0n, e: 0 };
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const hi = view.getUint32(0), lo = view.getUint32(4);
  const sign = hi >>> 31 ? -1n : 1n;
  const exp = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e;
  if (exp === 0) e = -1074;
  else {
    mant |= 1n << 52n;
    e = exp - 1075;
  }
  return { m: sign * mant, e };
}
const align = (a, b) => {
  const e = Math.min(a.e, b.e);
  return [a.m << BigInt(a.e - e), b.m << BigInt(b.e - e), e];
};
const add = (a, b) => { const [x, y, e] = align(a, b); return { m: x + y, e }; };
const sub = (a, b) => { const [x, y, e] = align(a, b); return { m: x - y, e }; };
const mul = (a, b) => ({ m: a.m * b.m, e: a.e + b.e });
const scale = (a, k) => ({ m: a.m * k, e: a.e }); // times a BigInt
const abs = (a) => ({ m: a.m < 0n ? -a.m : a.m, e: a.e });
const sgn = (a) => (a.m > 0n ? 1 : a.m < 0n ? -1 : 0);
const le = (a, b) => sgn(sub(a, b)) <= 0;
/** a <= (num / den) * b, for dyadics a, b and a positive rational num / den. */
const leRat = (a, { num, den }, b) => le(scale(a, den), scale(b, num));

// Constants of FloatError.lean as exact rationals, u = 1/N.
const N = 2n ** 53n;
const ERRA = { num: (N - 1n) ** 3n + N ** 4n - N * (N - 1n) ** 3n, den: N * (N - 1n) ** 3n }; // u + (1/(1-u))^3 - 1
const KAPPA = { num: ERRA.num * (N + 1n) ** 3n, den: ERRA.den * N ** 3n }; // errA (1+u)^3
const KAPPA2 = { num: KAPPA.num ** 2n, den: KAPPA.den ** 2n };
const EXACT4 = { num: (N + 1n) ** 4n - N ** 4n, den: N ** 4n }; // (1+u)^4 - 1
const SHEW = { num: 3n * N + 14n, den: N * N }; // (3 + 14u)u
const CODE = (4 + 16 * 2 ** -53) * 2 ** -53; // the double constant of code_constant_double

test("the constants are what FloatError.lean states", () => {
  // errA(2^-53) <= 4.4408920985006271e-16 (errA_double), the code constant is 2^-51 + 2^-102.
  assert.ok(ERRA.num * 10n ** 32n <= 44408920985006271n * ERRA.den); // errA_double
  assert.ok(ERRA.num * 10n ** 32n > 44408920985006268n * ERRA.den); // exactly 4u + 6u^2 + ... = 4.44089209850062690e-16
  assert.equal(CODE, 2 ** -51 + 2 ** -102); // code_constant_double
  assert.ok(KAPPA.num * 10n ** 19n <= 4441n * KAPPA.den); // kappa_double
  // gamma_10000_links: gamma_9999 = 9999 / (N - 9999) <= 1.1102e-12.
  assert.ok(9999n * 10n ** 16n <= 11102n * (N - 9999n));
});

// ------------------------------------------------------------ (1) and (5): orientation

/**
 * One determinant orient(O, P, Q), computed in doubles exactly as lattice.js computes
 * d1..d4 (O = c, P = d, Q = a for d1), checked against its exact value.
 */
function orientCase([ox, oy], [px, py], [qx, qy], stats) {
  const l = (px - ox) * (qy - oy), r = (py - oy) * (qx - ox), f = l - r;
  const [Ox, Oy, Px, Py, Qx, Qy] = [ox, oy, px, py, qx, qy].map(dy);
  const X = mul(sub(Px, Ox), sub(Qy, Oy)), Y = mul(sub(Py, Oy), sub(Qx, Ox));
  const d = sub(X, Y);
  const F = dy(f), T = add(abs(dy(l)), abs(dy(r)));
  const err = abs(sub(F, d));
  // flDet_err and flDet_err_exact.
  assert.ok(leRat(err, ERRA, T), `flDet_err: f=${f} at ${[ox, oy, px, py, qx, qy]}`);
  assert.ok(leRat(err, EXACT4, add(abs(X), abs(Y))), `flDet_err_exact: f=${f}`);
  // B = errA (|l| + |r|), exactly; inside means |f| <= B.
  const inside = leRat(abs(F), ERRA, T);
  const agree = sgn(F) === sgn(d);
  stats.dets += 1;
  if (inside) stats.inside += 1;
  if (inside && f === 0 && sgn(d) === 0) stats.insideExactZero += 1;
  if (!agree) stats.disagree += 1;
  if (!inside) assert.ok(agree, `flDet_sign_certain: |f| > B but signs differ, f=${f}`);
  if (!leRat(abs(F), SHEW, T)) assert.ok(agree, `flDet_sign_certain_shewchuk: f=${f}`);
  else stats.insideShewchuk += 1;
  // flDet_sign_certain_code: the bound the code computes in doubles is at least B.
  const bf = CODE * (Math.abs(l) + Math.abs(r));
  assert.ok(le(scale(T, ERRA.num), scale(dy(bf), ERRA.den)), `bound_eval_safe: computed bound ${bf} below B`);
  if (Math.abs(f) > bf) assert.ok(agree, `flDet_sign_certain_code: f=${f}`);
  // flDet_exact_of_zero.
  if (l === 0 && r === 0) assert.equal(sgn(d), 0);
  if (!agree) {
    // abs_le_of_not_sameSign + flDet_err: |d| <= B; orient_near_of_disagree: d^2 <= kappa^2 |OP|^2 |OQ|^2.
    assert.ok(leRat(abs(d), ERRA, T), `|d| <= B on a sign error: f=${f}`);
    const sq = (a) => mul(a, a);
    const sqdOP = add(sq(sub(Px, Ox)), sq(sub(Py, Oy))), sqdOQ = add(sq(sub(Qx, Ox)), sq(sub(Qy, Oy)));
    assert.ok(leRat(sq(d), KAPPA2, mul(sqdOP, sqdOQ)), `orient_near_of_disagree: f=${f}`);
  }
  return { f, sign: sgn(d), inside, agree };
}

const jsOpposite = (u, v) => (u > 0 && v < 0) || (u < 0 && v > 0);

/** Four determinants of segmentsCross(a, b, c, d) and the code's answer against exact. */
function crossCase(A, B, C, D, stats) {
  const o1 = orientCase(C, D, A, stats), o2 = orientCase(C, D, B, stats);
  const o3 = orientCase(A, B, C, stats), o4 = orientCase(A, B, D, stats);
  const code = segmentsCross(...A, ...B, ...C, ...D);
  // The determinants above are the code's: floatCross on them is segmentsCross.
  assert.equal(code, jsOpposite(o1.f, o2.f) && jsOpposite(o3.f, o4.f), "fidelity: determinants differ from lattice.js");
  const exact = o1.sign * o2.sign < 0 && o3.sign * o4.sign < 0; // Segments.crossTest
  stats.cases += 1;
  const all = [o1, o2, o3, o4];
  if (all.every((o) => !o.inside)) {
    stats.certainCases += 1;
    assert.equal(code, exact, "segmentsCross_float_exact: all four clear B but the answer differs");
  }
  if (code !== exact) {
    stats.wrongCases += 1;
    // misclassified_near: some determinant has the wrong sign (checked near the line in orientCase).
    assert.ok(all.some((o) => !o.agree), "misclassified_near: wrong answer with every sign right");
    assert.ok(all.some((o) => o.inside), "misclassified_near: wrong answer with every |f| > B");
  }
}

function newStats() {
  return { cases: 0, certainCases: 0, wrongCases: 0, dets: 0, inside: 0, insideExactZero: 0, insideShewchuk: 0, disagree: 0 };
}

function report(label, s) {
  console.log(
    `${label}: ${s.cases} segment pairs, ${s.certainCases} with all four |f| > B (all exact), ` +
      `${s.wrongCases} where segmentsCross differs from exact arithmetic; ${s.dets} determinants, ` +
      `${s.inside} inside |f| <= B (${s.insideExactZero} of them exactly 0 = 0, ` +
      `${s.inside - s.insideExactZero} genuinely uncertain; Shewchuk-style 3u bound: ${s.insideShewchuk}), ` +
      `${s.disagree} with the wrong sign, all inside B`,
  );
}

test("(1)+(5) orientation filter: random and near-degenerate segments in doubles", () => {
  const rand = mulberry32(20261027);
  const pick = (k) => Math.floor(rand() * k);
  const coord = () => (rand() - 0.5) * 200 + rand() * 1e-3; // within 100 m, odd low bits
  const p = () => [coord(), coord()];
  const lerp = (P, Q, t) => [P[0] + (Q[0] - P[0]) * t, P[1] + (Q[1] - P[1]) * t];
  const frac = () => [0.5, 0.25, 0.75, 1 / 3, 0.1, rand()][pick(6)];
  const stats = newStats();
  for (let i = 0; i < 1500; i += 1) {
    let A = p(), B = p(), C = p(), D = p();
    switch (pick(8)) {
      case 0: C = lerp(A, B, frac()); break; // outline vertex on the link (rounded)
      case 1: A = lerp(C, D, frac()); break; // link endpoint on the wall (rounded)
      case 2: C = lerp(A, B, -0.5 - rand()); D = lerp(A, B, 0.2 + rand()); break; // collinear overlap
      case 3: { const m = lerp(A, B, frac()); C = [2 * m[0] - D[0], 2 * m[1] - D[1]]; break; } // crossing at m
      case 4: { const m = lerp(A, B, frac()); C = m; D = lerp(m, D, 1e-9); break; } // tiny wall at the link
      case 5: B = lerp(A, B, 1e-7); break; // very short link
      default: break; // generic
    }
    crossCase(A, B, C, D, stats);
  }
  report("doubles", stats);
  assert.ok(stats.disagree > 0 && stats.wrongCases > 0, "the generator must reach rounding-decided cases");
  assert.ok(stats.certainCases > 500);
});

test("(1)+(5) Studio-like: float32 lattice links against double wall outlines", () => {
  // Nodes on a rotated 0.5 m lattice stored in a Float32Array (Math.fround), walls through
  // or along them computed in doubles, as buildLatticeGraph does before index.crosses.
  const rand = mulberry32(7);
  const pick = (k) => Math.floor(rand() * k);
  const stats = newStats();
  for (let i = 0; i < 1500; i += 1) {
    const s = 0.5, th = rand() < 0.3 ? 0 : rand() * Math.PI;
    const ox = (rand() - 0.5) * 140, oy = (rand() - 0.5) * 140;
    const node = (a, b) => [Math.fround(ox + s * (a * Math.cos(th) - b * Math.sin(th))),
      Math.fround(oy + s * (a * Math.sin(th) + b * Math.cos(th)))];
    const [di, dj] = [[1, 0], [0, 1], [1, 1], [1, -1]][pick(4)];
    const A = node(0, 0), B = node(di, dj);
    const lerp = (P, Q, t) => [P[0] + (Q[0] - P[0]) * t, P[1] + (Q[1] - P[1]) * t];
    const away = [A[0] + (rand() - 0.5) * 6, A[1] + (rand() - 0.5) * 6];
    let C, D;
    switch (pick(5)) {
      case 0: C = lerp(A, B, 0.5); D = away; break; // wall vertex at the link midpoint
      case 1: C = lerp(away, A, 2); D = away; break; // wall through node A
      case 2: C = lerp(A, B, -1.3); D = lerp(A, B, 2.7); break; // wall face along the grid line
      case 3: C = node(di, dj + 1); D = node(di + 1, dj); break; // wall between lattice nodes
      default: C = away; D = [A[0] + (rand() - 0.5) * 6, A[1] + (rand() - 0.5) * 6]; break;
    }
    crossCase(A, B, C, D, stats);
  }
  report("float32 lattice", stats);
  assert.ok(stats.certainCases > 500);
});

// ------------------------------------------------------------ (2) summation

test("(2) route lengths summed left to right are within gamma_{n-1} of the exact sum", () => {
  const rand = mulberry32(99);
  const N_ = Number(N);
  const check = (xs, label) => {
    let s = 0; // as findRoute / straightenRoute: let length = 0; length += Math.hypot(...)
    for (const x of xs) s += x;
    let S = { m: 0n, e: 0 };
    for (const x of xs) S = add(S, dy(x));
    const n1 = BigInt(xs.length - 1);
    // |s - S| <= gamma_{n-1} S, gamma_{n-1} = (n-1) / (N - (n-1)).
    const err = abs(sub(dy(s), S));
    assert.ok(le(scale(err, N - n1), scale(S, n1)), `route_sum_rel_err: ${label}, n = ${xs.length}`);
    const diff = sub(dy(s), S);
    const rel = (Number(diff.m) * 2 ** diff.e) / s;
    const gamma = (xs.length - 1) / N_ / (1 - (xs.length - 1) / N_);
    return gamma > 0 ? Math.abs(rel) / gamma : 0;
  };
  let worst = 0, worstLong = 0;
  for (const n of [1, 2, 10, 100, 1000, 10000]) {
    for (let k = 0; k < 4; k += 1) {
      // A lattice walk: float32 nodes 0.5 m apart, lengths by Math.hypot as edgeLengths does.
      const pts = [[Math.fround((rand() - 0.5) * 100), Math.fround((rand() - 0.5) * 100), Math.fround(0.75)]];
      const xs = [];
      for (let i = 0; i < n; i += 1) {
        const [dx, dz] = [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [1, -1]][Math.floor(rand() * 6)];
        const q = pts[pts.length - 1];
        const next = [Math.fround(q[0] + 0.5 * dx), Math.fround(q[1] + 0.5 * dz), q[2]];
        xs.push(Math.hypot(next[0] - q[0], next[1] - q[1], next[2] - q[2]));
        pts.push(next);
      }
      const ratio = check(xs, "lattice walk");
      worst = Math.max(worst, ratio);
      if (n >= 1000) worstLong = Math.max(worstLong, ratio);
    }
  }
  // An adversarial sum: 1 then 9999 terms just under half an ulp of 1, each rounded away.
  // The error reaches 99% of gamma_9999: the bound is essentially tight.
  const tiny = 0.99 * 2 ** -53;
  const ratio = check([1, ...new Array(9999).fill(tiny)], "adversarial");
  assert.ok(ratio > 0.98 && ratio <= 1, `adversarial ratio ${ratio}`);
  console.log(`summation: lattice walks reach at most ${worst.toFixed(3)} of gamma_{n-1} (short routes, ` +
    `few roundings), ${worstLong.toExponential(2)} for 1,000+ links ` +
    `(adversarial 10,000 terms: ${ratio.toFixed(4)}); gamma_9999 = ${(9999 / N_ / (1 - 9999 / N_)).toExponential(4)}`);
});

// ------------------------------------------------------------ (4) float32 coordinates

test("(4) Float32Array coordinates within 100 m move a length by at most 2 * 2^-24 * 100", () => {
  const rand = mulberry32(4);
  const T = dy(200 * 2 ** -24); // 2 u M, a dyadic, = 1.1920928955078125e-5
  const M2 = dy(100 * 100);
  const sq = (a) => mul(a, a);
  const norm2 = (P) => P.reduce((acc, x) => add(acc, sq(dy(x))), { m: 0n, e: 0 });
  const dist2 = (P, Q) => P.reduce((acc, x, k) => add(acc, sq(sub(dy(x), dy(Q[k])))), { m: 0n, e: 0 });
  /** |a - b| <= T for a = sqrt(a2), b = sqrt(b2), without square roots. */
  const sqrtGap = (a2, b2) => {
    // a <= b + T  <=>  a2 - b2 - T^2 <= 2 T b  <=>  (lhs <= 0) or lhs^2 <= 4 T^2 b2.
    const one = (x2, y2) => {
      const lhs = sub(sub(x2, y2), sq(T));
      return sgn(lhs) <= 0 || le(sq(lhs), scale(mul(sq(T), y2), 4n));
    };
    return one(a2, b2) && one(b2, a2);
  };
  let checked = 0, worst = 0;
  const point = () => {
    for (;;) {
      const P = [0, 0, 0].map(() => (rand() - 0.5) * 200);
      if (le(norm2(P), M2)) return P;
    }
  };
  for (let i = 0; i < 4000; i += 1) {
    const P = point();
    let Q;
    if (i % 2) Q = point();
    else {
      const step = [[0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0.3]][i % 4 === 0 ? 0 : (i >> 1) % 4];
      Q = P.map((x, k) => x + step[k]);
      if (!le(norm2(Q), M2)) continue;
    }
    const Pf = P.map(Math.fround), Qf = Q.map(Math.fround);
    // The model itself: |fround(x) - x| <= 2^-24 |x| (round_coords_norm's hypothesis).
    for (const [x, xf] of [...P.map((x, k) => [x, Pf[k]]), ...Q.map((x, k) => [x, Qf[k]])]) {
      assert.ok(le(scale(abs(sub(dy(xf), dy(x))), 2n ** 24n), abs(dy(x))), `fround model at ${x}`);
    }
    assert.ok(sqrtGap(dist2(Pf, Qf), dist2(P, Q)), `studio_float32_length_err at ${P} ${Q}`);
    checked += 1;
    const gap = Math.abs(Math.hypot(...Pf.map((x, k) => x - Qf[k])) - Math.hypot(...P.map((x, k) => x - Q[k])));
    worst = Math.max(worst, gap);
  }
  console.log(`float32: ${checked} pairs, worst length change ${worst.toExponential(3)} m ` +
    `(bound 2*2^-24*100 = ${(200 * 2 ** -24).toExponential(4)} m)`);
  assert.ok(worst <= 200 * 2 ** -24 + 1e-12);
});
