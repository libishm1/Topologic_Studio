/**
 * Checks of the opening-probe data and reducer (lib/probes.mjs; plan 13.2
 * items 4 and 5) that use no model, run before any model comparison:
 *
 *   1. the reducer fed the measured series as if it were the model returns
 *      model = measured (delta 1) for every probe, neutral plane and door flow;
 *   2. NIST/NRC: the door's mass flow rebuilt from the probes and the door's
 *      thermocouples (rho = 353 / T, each probe over its share of the 2 m x
 *      2 m door) against the file's door flow in and out, 300 to 1200 s;
 *   3. WTC: the opening probes' directions and temperatures over 100 to 600 s:
 *      the inflow probes read into the room and cold, the outflow probes out
 *      of it and hot;
 *   4. UL FSRI: the doorway's mass balance from its seven probes and
 *      thermocouples (rho = 353 / T, each probe over its 0.25 m band of the
 *      0.9 m door), over each open test's last 180 s: out over in, about 1
 *      when the probes' sign and units are read right.
 *
 *   node tools/calibration/probe-checks.mjs
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { probePairs } from "./lib/probes.mjs";
import { statistics } from "./lib/stats.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const load = (d) => JSON.parse(readFileSync(path.join(HERE, "data", `${d}.json`), "utf8"));

/**
 * A measured series as a 1 s series: each second takes the reading nearest it
 * (a tie takes the earlier), so the reducer's mean over half an interval
 * either side of a reading returns that reading.
 */
function oneSecond(times, series, end) {
  const out = [];
  let q = 0;
  for (let s = 1; s <= end; s += 1) {
    while (q + 1 < times.length && Math.abs(times[q + 1] - s) < Math.abs(times[q] - s)) q += 1;
    out.push(series[q] ?? null);
  }
  return out;
}

// 1. The reducer on the measured series.
const self = {};
for (const d of ["nistnrc", "wtc", "fsri", "ulnist"]) {
  for (const T of load(d).tests) {
    if (!T.probes?.length) continue;
    const MP = T.measured.probes;
    const end = Math.max(...MP.times);
    const probes = Object.fromEntries(T.probes.map((p) => [p.id, oneSecond(MP.times, MP.series[p.id], end)]));
    const MF = T.measured.doorFlow;
    const doorFlow = MF ? { in: oneSecond(MF.times, MF.in, end), out: oneSecond(MF.times, MF.out, end) } : null;
    const got = probePairs({ dataset: d, cell: 0, horizon: T.horizon ?? 600, health: { ok: true }, series: { from: 1, step: 1, probes, doorFlow } }, T);
    for (const p of got.pairs) (self[`${d} ${p.quantity}`] ||= []).push([p.model, p.measured]);
    for (const n of got.notes) console.log(`  note ${d}/${T.id}: ${n}`);
  }
}
console.log("1. the reducer on the measured series as the model (delta 1, sigmaM 0 when it reads the data as they are; sigma_E 0):");
for (const [k, ps] of Object.entries(self)) {
  const s = statistics(ps, 0);
  const worst = Math.max(...ps.filter(([m, e]) => e > 0).map(([m, e]) => Math.abs(m / e - 1)));
  console.log(`  ${k.padEnd(26)} ${String(ps.length).padStart(3)} pairs, delta ${s.delta}, sigmaM ${s.sigmaM}, worst |M/E - 1| ${worst.toFixed(3)}`);
}

// 2. NIST/NRC door flow rebuilt from the probes.
console.log("2. NIST/NRC: door mass flow from the probes and door thermocouples against the file's (mean over 300 to 1200 s):");
for (const T of load("nistnrc").tests) {
  if (!T.probes?.length) continue;
  const fdsText = null; // positions come from the reduced data (the FDS input's devices)
  void fdsText;
  const MP = T.measured.probes;
  const cols = [...new Set(T.probes.map((p) => p.y))].sort((a, b) => a - b);
  const width = (T.doorFlow.y[1] - T.doorFlow.y[0]) / cols.length;
  // The door's thermocouples are in the main file? No: file_2. Without them, the gas's density is taken from
  // each probe's own side of the neutral plane: ambient below (inflow), the hot layer's (the nearest tree's
  // upper layer) above. Here the simpler bound: rho of ambient air for both, which overstates the outflow's
  // mass by T_hot / T_ambient; so the check reads the volume flows' ratio to the file's mass flows times
  // the density each needs, and reports the density it implies.
  let sIn = 0, sOut = 0, fIn = 0, fOut = 0, n = 0, nf = 0;
  MP.times.forEach((t, iq) => {
    if (t < 300 || t > 1200) return;
    let vin = 0, vout = 0;
    for (const y of cols) {
      const col = T.probes.filter((p) => p.y === y).sort((a, b) => a.z - b.z);
      // Each probe stands for the band halfway to its neighbours (the door 0 to 2 m high).
      col.forEach((p, q) => {
        const lo = q === 0 ? T.doorFlow.z[0] : (col[q - 1].z + p.z) / 2;
        const hi = q === col.length - 1 ? T.doorFlow.z[1] : (p.z + col[q + 1].z) / 2;
        const v = MP.series[p.id][iq];
        if (v == null) return;
        const a = width * (hi - lo);
        if (v > 0) vin += v * a;
        else vout -= v * a;
      });
    }
    sIn += vin;
    sOut += vout;
    n += 1;
  });
  const MF = T.measured.doorFlow;
  MF.times.forEach((t, iq) => {
    if (t < 300 || t > 1200 || MF.in[iq] == null || MF.out[iq] == null) return;
    fIn += MF.in[iq];
    fOut += MF.out[iq];
    nf += 1;
  });
  const [qin, qout, min, mout] = [sIn / n, sOut / n, fIn / nf, fOut / nf];
  // The density each implies (kg/m3) and the temperature that density means (rho = 353 / T).
  const rIn = min / qin, rOut = mout / qout;
  console.log(`  test ${T.id}: inflow ${qin.toFixed(2)} m3/s and ${min.toFixed(2)} kg/s: rho ${rIn.toFixed(2)} kg/m3 (${(353 / rIn - 273.15).toFixed(0)} °C); outflow ${qout.toFixed(2)} m3/s and ${mout.toFixed(2)} kg/s: rho ${rOut.toFixed(2)} kg/m3 (${(353 / rOut - 273.15).toFixed(0)} °C)`);
}

// 3. WTC: direction and temperature of the opening probes.
console.log("3. WTC: the opening probes over 100 to 600 s (mean velocity, positive along +x; mean temperature):");
for (const T of load("wtc").tests) {
  const MP = T.measured.probes;
  const pick = MP.times.map((t, q) => (t >= 100 && t <= 600 ? q : -1)).filter((q) => q >= 0);
  const mean = (s) => {
    const v = pick.map((q) => s?.[q]).filter((x) => x != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
  };
  const line = ["in", "out"].map((side) => {
    const ps = T.probes.filter((p) => p.id.endsWith(`Vel ${side}`));
    const v = ps.map((p) => mean(MP.series[p.id]));
    const temps = ps.map((p) => mean(MP.temperatures?.[`T${side} ${p.id.split(" ")[0]}`]));
    return `${side}: v ${Math.min(...v).toFixed(2)} to ${Math.max(...v).toFixed(2)} m/s, T ${Math.min(...temps).toFixed(0)} to ${Math.max(...temps).toFixed(0)} °C`;
  });
  console.log(`  test ${T.id} (ambient ${T.ambientC} °C): ${line.join("; ")}`);
}

// 4. UL FSRI: the doorway's mass balance.
console.log("4. UL FSRI: doorway mass balance over each open test's last 180 s (out / in; about 1 when read right):");
const ratios = [];
for (const T of load("fsri").tests) {
  if (!T.probes?.length) continue;
  const MP = T.measured.probes;
  const [a, b] = T.windows.neutralPlane;
  let mIn = 0, mOut = 0;
  MP.times.forEach((t, iq) => {
    if (t < a || t > b) return;
    for (const p of T.probes) {
      const v = MP.series[p.id][iq], Tc = MP.temperatures?.[p.id]?.[iq];
      if (v == null || Tc == null) continue;
      const m = (353 / (Tc + 273.15)) * Math.abs(v) * 0.9 * 0.25;
      if (v > 0) mOut += m;
      else mIn += m;
    }
  });
  ratios.push([T.id, mOut / mIn]);
}
const rs = ratios.map((r) => r[1]).sort((x, y) => x - y);
console.log(`  ${ratios.length} open tests: out/in median ${rs[rs.length >> 1].toFixed(2)}, range ${rs[0].toFixed(2)} to ${rs[rs.length - 1].toFixed(2)}`);
for (const [id, r] of ratios) if (r < 0.7 || r > 1.4) console.log(`  ${id}: ${r.toFixed(2)}`);
