/**
 * The gas's thermal radiation by the rank correlated spectral line weighted
 * sum of gray gases (calibration plan 16.43, item 8, second attempt; points
 * 10 to 12): the tables' reader and the functions the radiation stage uses
 * (flamescapes.js radiationBands, marchBands). Off by default; with
 * gasRadiation 0 nothing here is called and no table is loaded.
 *
 * The tables are the absorption line blackbody distribution functions of
 * Pearson, Webb, Solovjov and Ma (2014, doi:10.1016/j.jqsrt.2014.01.019) at
 * 1 atm: F_c(C, T_g, T_b) for CO2 and F_w(C, T_g, T_b, Y) for H2O, the share
 * of black radiation at T_b that falls where the gas at T_g has an absorption
 * cross-section below C (m2/mol). T_g and T_b: 300 to 3000 K in steps of 100 K
 * (28 each). C: 71 values, 1e-4 x 10^(k/10) m2/mol. Y, the mole fraction of
 * H2O: 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.
 *
 * They are kept in this repository and credited (plan 16.44, decision 7 as
 * the user changed it): public/gas-radiation/gas-radiation-tables.f32, beside
 * a NOTICE.md that names their source, RadLib (Stephens, Jensen, Wheeler and
 * Lignell 2022, doi:10.1016/j.cpc.2021.108227; MIT licence), which ships
 * them, and the source files' SHA-256. One file of float32 values,
 * little-endian, holds them: CO2's 55 664 in the order [T_g][T_b][C], then
 * H2O's 500 976 in the order [Y][T_g][T_b][C], the last index fastest.
 * tools/gas-radiation-tables.mjs builds it from RadLib's two text files. It
 * is read only with gasRadiation on.
 *
 * Between its points a table is linear in T_g, T_b, Y and C. Outside them it
 * is held at its edge. The mixture is the product of the two species'
 * functions, each at C over its mole fraction (Solovjov and Webb 2000,
 * doi:10.1016/S0022-4073(99)00133-8); an absent species gives the factor 1.
 *
 * Eight gray gases and a clear one (Solovjov, Andre, Lemonnier and Webb 2017,
 * doi:10.1016/j.jqsrt.2017.01.034). Their levels are fixed once, at 600 K,
 * x_CO2 0.095 and x_H2O 0.19: nine cross-sections evenly spaced in ln C from
 * 3e-4 to 60 m2/mol give the levels Ft_k = F(C_k), the last set to 1; their
 * eight geometric means give F_j. In a cell at T with composition x, gray gas
 * j has the cross-section at which F(C; T, 600 K, x) = F_j, and its weight for
 * black radiation at T_b is F(Ct_j; T, T_b, x) - F(Ct_j-1; T, T_b, x), with
 * Ct_k the cross-section at which F(C; T, 600 K, x) = Ft_k.
 */

const N_T = 28, N_C = 71, N_Y = 9;
const T_MIN = 300, T_STEP = 100;
const Y_TABLE = Object.freeze([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]);
const N_CO2 = N_T * N_T * N_C;
const N_H2O = N_Y * N_T * N_T * N_C;

/** The gray gases (the clear gas is band 0 beside them). */
export const GAS_BANDS = 8;
/** The table file's layout: its counts of values and its size in bytes. */
export const GAS_TABLE = Object.freeze({ temperatures: N_T, crossSections: N_C, moleFractions: N_Y, co2: N_CO2, h2o: N_H2O, bytes: 4 * (N_CO2 + N_H2O) });
/** The reference state of the levels, and the range of the nine cross-sections (plan 16.43, point 12). */
export const GAS_REFERENCE = Object.freeze({ T: 600, xc: 0.095, xw: 0.19, Clow: 3e-4, Chigh: 60 });
/** The bisection's range in C (m2/mol) and its tolerance in ln C. */
const LN_LOW = Math.log(1e-7), LN_HIGH = Math.log(1e3), LN_TOLERANCE = 1e-6;
const PER_LN10 = 10 / Math.LN10;

/** The tables' file, relative to the app's base URL (and to public/ in the repository). */
export const GAS_TABLES_PATH = "gas-radiation/gas-radiation-tables.f32";
/** How often the tables' reader has run: the checks count it (plan 16.43, check (i)). */
export const gasTableLoads = { count: 0 };

/**
 * The tables from the file's bytes (an ArrayBuffer or a typed array over
 * them), with the gray gases' levels worked from them. Throws if the size is
 * not the file's.
 */
export function gasTablesFrom(bytes) {
  gasTableLoads.count += 1;
  const view = ArrayBuffer.isView(bytes) ? new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new DataView(bytes);
  if (view.byteLength !== GAS_TABLE.bytes) throw new Error(`The gas radiation tables need ${GAS_TABLE.bytes} bytes (float32, CO2 then H2O); this file has ${view.byteLength}.`);
  const co2 = new Float32Array(N_CO2), h2o = new Float32Array(N_H2O);
  for (let q = 0; q < N_CO2; q += 1) co2[q] = view.getFloat32(4 * q, true);
  for (let q = 0; q < N_H2O; q += 1) h2o[q] = view.getFloat32(4 * (N_CO2 + q), true);
  const C = new Float64Array(N_C);
  for (let k = 0; k < N_C; k += 1) C[k] = 1e-4 * 10 ** (k / 10);
  const M = { co2, h2o, C, rowC: new Float64Array(N_C), rowW: new Float64Array(N_C), levels: null, bandLevels: null, levelC: null };
  // The levels at the reference state (plan 16.43, point 12).
  const { T, xc, xw, Clow, Chigh } = GAS_REFERENCE;
  const levelC = new Float64Array(GAS_BANDS + 1);
  const levels = new Float64Array(GAS_BANDS + 1), bandLevels = new Float64Array(GAS_BANDS);
  for (let k = 0; k <= GAS_BANDS; k += 1) {
    levelC[k] = Math.exp(Math.log(Clow) + (k * (Math.log(Chigh) - Math.log(Clow))) / GAS_BANDS);
    levels[k] = k === GAS_BANDS ? 1 : albdf(M, levelC[k], T, T, xc, xw);
  }
  for (let j = 0; j < GAS_BANDS; j += 1) bandLevels[j] = albdf(M, Math.sqrt(levelC[j] * levelC[j + 1]), T, T, xc, xw);
  M.levels = levels;
  M.bandLevels = bandLevels;
  M.levelC = levelC;
  return M;
}

/**
 * The tables from a URL (the run layer's reader, used only with gasRadiation
 * on): the app's own file, GAS_TABLES_PATH under its base URL. Without a URL,
 * or if the fetch fails, it throws and says why.
 */
export async function fetchGasTables(url) {
  if (!url) throw new Error("gasRadiation needs the gas radiation tables' URL (the app's file is GAS_TABLES_PATH under its base URL).");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`gasRadiation: the gas radiation tables could not be fetched from ${url} (${response.status}).`);
  return gasTablesFrom(await response.arrayBuffer());
}

// Where a value lies in a table's axis: the lower index and the fraction to the next (module scratch).
let iT = 0, fT = 0;
function locateT(T) {
  const u = (Math.min(Math.max(T, T_MIN), T_MIN + T_STEP * (N_T - 1)) - T_MIN) / T_STEP;
  iT = Math.min(Math.floor(u), N_T - 2);
  fT = u - iT;
}
let iY = 0, fY = 0;
function locateY(x) {
  const y = Math.min(Math.max(x, 0), 1);
  let q = 0;
  while (q < N_Y - 2 && y >= Y_TABLE[q + 1]) q += 1;
  iY = q;
  fY = (y - Y_TABLE[q]) / (Y_TABLE[q + 1] - Y_TABLE[q]);
}
let iC = 0, fC = 0;
/** cx held at the table's ends; `guess` an index near it. */
function locateC(C, cx, guess) {
  if (cx <= C[0]) {
    iC = 0;
    fC = 0;
    return;
  }
  if (cx >= C[N_C - 1]) {
    iC = N_C - 2;
    fC = 1;
    return;
  }
  let k = guess < 0 ? 0 : guess > N_C - 2 ? N_C - 2 : guess;
  while (k > 0 && cx < C[k]) k -= 1;
  while (k < N_C - 2 && cx >= C[k + 1]) k += 1;
  iC = k;
  fC = (cx - C[k]) / (C[k + 1] - C[k]);
}
const guessOf = (cx) => Math.floor(10 * Math.log10(cx) + 40);

/** CO2's table at (cx, T_g, T_b): linear in each, held at the edges. */
export function albdfCO2(M, cx, Tg, Tb) {
  const F = M.co2;
  locateT(Tg);
  const i = iT, a = fT;
  locateT(Tb);
  const j = iT, b = fT;
  locateC(M.C, cx, cx > 0 ? guessOf(cx) : 0);
  const k = iC, t = fC;
  const o00 = (i * N_T + j) * N_C + k, o01 = o00 + N_C, o10 = o00 + N_T * N_C, o11 = o10 + N_C;
  const v00 = F[o00] * (1 - t) + F[o00 + 1] * t, v01 = F[o01] * (1 - t) + F[o01 + 1] * t;
  const v10 = F[o10] * (1 - t) + F[o10 + 1] * t, v11 = F[o11] * (1 - t) + F[o11 + 1] * t;
  return (v00 * (1 - b) + v01 * b) * (1 - a) + (v10 * (1 - b) + v11 * b) * a;
}

/** H2O's table at (cx, T_g, T_b, Y): linear in each, held at the edges. */
export function albdfH2O(M, cx, Tg, Tb, Y) {
  const F = M.h2o;
  locateT(Tg);
  const i = iT, a = fT;
  locateT(Tb);
  const j = iT, b = fT;
  locateY(Y);
  const y = iY, s = fY;
  locateC(M.C, cx, cx > 0 ? guessOf(cx) : 0);
  const k = iC, t = fC;
  let out = 0;
  for (let q = 0; q < 2; q += 1) {
    const o00 = (((y + q) * N_T + i) * N_T + j) * N_C + k, o01 = o00 + N_C, o10 = o00 + N_T * N_C, o11 = o10 + N_C;
    const v00 = F[o00] * (1 - t) + F[o00 + 1] * t, v01 = F[o01] * (1 - t) + F[o01 + 1] * t;
    const v10 = F[o10] * (1 - t) + F[o10 + 1] * t, v11 = F[o11] * (1 - t) + F[o11 + 1] * t;
    out += ((v00 * (1 - b) + v01 * b) * (1 - a) + (v10 * (1 - b) + v11 * b) * a) * (q ? s : 1 - s);
  }
  return out;
}

/**
 * The mixture's function F(C; T_g, T_b, x) (plan 16.43, point 11): C in
 * m2 per mole of mixture, xc and xw the mole fractions of CO2 and H2O.
 */
export function albdf(M, C, Tg, Tb, xc, xw) {
  let F = 1;
  if (xc > 0) F *= albdfCO2(M, C / xc, Tg, Tb);
  if (xw > 0) F *= albdfH2O(M, C / xw, Tg, Tb, xw);
  return F;
}

/** A row's value at cx (the row is a table's 71 values at fixed T_g, T_b and Y). */
function rowAt(M, row, cx, guess) {
  locateC(M.C, cx, guess);
  return row[iC] * (1 - fC) + row[iC + 1] * fC;
}

/** The two species' rows at (T_g, T_b) and, for H2O, Y = xw, into M.rowC and M.rowW. */
function fillRows(M, Tg, Tb, xc, xw) {
  locateT(Tg);
  const i = iT, a = fT;
  locateT(Tb);
  const j = iT, b = fT;
  const w00 = (1 - a) * (1 - b), w01 = (1 - a) * b, w10 = a * (1 - b), w11 = a * b;
  if (xc > 0) {
    const F = M.co2, row = M.rowC;
    const o00 = (i * N_T + j) * N_C, o01 = o00 + N_C, o10 = o00 + N_T * N_C, o11 = o10 + N_C;
    for (let k = 0; k < N_C; k += 1) row[k] = F[o00 + k] * w00 + F[o01 + k] * w01 + F[o10 + k] * w10 + F[o11 + k] * w11;
  }
  if (xw > 0) {
    locateY(xw);
    const y = iY, s = fY;
    const F = M.h2o, row = M.rowW;
    const p00 = ((y * N_T + i) * N_T + j) * N_C, p01 = p00 + N_C, p10 = p00 + N_T * N_C, p11 = p10 + N_C;
    const q00 = p00 + N_T * N_T * N_C, q01 = q00 + N_C, q10 = q00 + N_T * N_C, q11 = q10 + N_C;
    for (let k = 0; k < N_C; k += 1) {
      row[k] = (F[p00 + k] * w00 + F[p01 + k] * w01 + F[p10 + k] * w10 + F[p11 + k] * w11) * (1 - s) + (F[q00 + k] * w00 + F[q01 + k] * w01 + F[q10 + k] * w10 + F[q11 + k] * w11) * s;
    }
  }
}

/** F at ln C = `ln` from the rows fillRows left (lnxc and lnxw the logarithms of the mole fractions). */
function rowsAt(M, ln, xc, xw, lnxc, lnxw) {
  const C = Math.exp(ln);
  let F = 1;
  if (xc > 0) F *= rowAt(M, M.rowC, C / xc, Math.floor((ln - lnxc) * PER_LN10 + 40));
  if (xw > 0) F *= rowAt(M, M.rowW, C / xw, Math.floor((ln - lnxw) * PER_LN10 + 40));
  return F;
}

/** The cross-section at which the rows' F equals `level`: bisection in ln C between 1e-7 and 1e3 m2/mol, to 1e-6; 0 where F at 1e-7 is at or above the level. */
function invertRows(M, level, xc, xw, lnxc, lnxw) {
  if (rowsAt(M, LN_LOW, xc, xw, lnxc, lnxw) >= level) return 0;
  let lo = LN_LOW, hi = LN_HIGH;
  while (hi - lo > LN_TOLERANCE) {
    const mid = 0.5 * (lo + hi);
    if (rowsAt(M, mid, xc, xw, lnxc, lnxw) < level) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

/**
 * The cross-section C (m2/mol) at which F(C; T_g, T_b, x) = level (plan
 * 16.43, point 12): bisection in ln C between 1e-7 and 1e3, to 1e-6; 0 where
 * F at 1e-7 is already at or above the level.
 */
export function albdfInverse(M, level, Tg, Tb, xc, xw) {
  fillRows(M, Tg, Tb, xc, xw);
  return invertRows(M, level, xc, xw, xc > 0 ? Math.log(xc) : 0, xw > 0 ? Math.log(xw) : 0);
}

/**
 * A cell's gray gases (plan 16.43, point 12), by rank: into C[oC + j - 1] the
 * cross-section of gray gas j = 1 to 8 (m2/mol; times the molar density it is
 * kappa_j), and into Ct[oCt + k] the cross-section of level k = 0 to 7, for
 * gas at T with mole fractions xc of CO2 and xw of H2O.
 */
export function gasCellBands(M, T, xc, xw, C, oC, Ct, oCt) {
  fillRows(M, T, GAS_REFERENCE.T, xc, xw);
  const lnxc = xc > 0 ? Math.log(xc) : 0, lnxw = xw > 0 ? Math.log(xw) : 0;
  for (let j = 0; j < GAS_BANDS; j += 1) {
    C[oC + j] = invertRows(M, M.bandLevels[j], xc, xw, lnxc, lnxw);
    Ct[oCt + j] = invertRows(M, M.levels[j], xc, xw, lnxc, lnxw);
  }
}

/**
 * The nine weights a_0 to a_8 (plan 16.43, point 12) that gas at T with
 * composition x gives to black radiation at T_b, from the cell's level
 * cross-sections Ct[oCt + k] (gasCellBands): a_0 = F(Ct_0), a_j = F(Ct_j) -
 * F(Ct_j-1), and a_8 = 1 - F(Ct_7), the last level being 1. They sum to 1.
 */
export function gasWeights(M, Ct, oCt, T, Tb, xc, xw, out) {
  let prev = albdf(M, Ct[oCt], T, Tb, xc, xw);
  out[0] = prev;
  for (let j = 1; j < GAS_BANDS; j += 1) {
    const F = albdf(M, Ct[oCt + j], T, Tb, xc, xw);
    out[j] = F - prev;
    prev = F;
  }
  out[GAS_BANDS] = 1 - prev;
  return out;
}
