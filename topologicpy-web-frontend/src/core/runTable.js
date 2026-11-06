/**
 * A Flamescapes run as one table, for machine learning and other analysis:
 * one row per graph node per frame, with the node's place and kind, the
 * solver's readings at head height and the measures and doses derived from
 * them (core/tenability.js). CSV with a header row; the doses and the
 * tenability index are worked out again from the stored frames, in order, so
 * the table needs nothing the run did not keep.
 *
 * Columns: node, x, y, z (the IFC's frame, metres, at floor level), kind
 * (floor, stair, door), time_s, gas_temperature_C, visibility_m, then, when
 * the run recorded them, soot_mg_m3, extinction_per_m, obscuration_pct_per_m,
 * optical_density_per_m, co2_pct, o2_pct; then co_ppm, fed_asphyxiant,
 * fed_heat, tenability_index, limiting (the criterion nearest its limit).
 */
import { extinction, limitingCriterion, obscuration, opticalDensity, tenabilityTracker } from "./tenability.js";

const KINDS = ["floor", "stair", "door"];
const f = (x, digits) => (Number.isFinite(x) ? Number(x.toFixed(digits)).toString() : "");

/**
 * The table's text in chunks (one per frame, header first), so a large run
 * can be written to a Blob without one huge string.
 *
 * @param {object} run  frames: [{time, T, vis, co, soot?, co2?, o2?}],
 *                      frameSeconds, n; points: Float32Array of xyz in the
 *                      IFC's frame (3 n); kinds: per node
 * @param {object} opts the tenability settings (core/tenability.js)
 */
export function* runTableChunks({ frames, frameSeconds, n, points, kinds }, opts = {}) {
  const extras = Boolean(frames[0]?.soot && frames[0]?.co2 && frames[0]?.o2);
  const columns = ["node", "x", "y", "z", "kind", "time_s", "gas_temperature_C", "visibility_m"];
  if (extras) columns.push("soot_mg_m3", "extinction_per_m", "obscuration_pct_per_m", "optical_density_per_m", "co2_pct", "o2_pct");
  columns.push("co_ppm", "fed_asphyxiant", "fed_heat", "tenability_index", "limiting");
  yield `${columns.join(",")}\n`;
  const tracker = tenabilityTracker(n, opts);
  const place = Array.from({ length: n }, (_, v) => `${f(points[v * 3], 3)},${f(points[v * 3 + 1], 3)},${f(points[v * 3 + 2], 3)},${KINDS[kinds?.[v] ?? 0] ?? "floor"}`);
  for (let k = 0; k < frames.length; k += 1) {
    const frame = frames[k];
    const time = Number.isFinite(frame.time) ? frame.time : (k + 1) * frameSeconds;
    const ti = tracker.step(frame, frameSeconds / 60);
    const rows = new Array(n);
    for (let v = 0; v < n; v += 1) {
      let row = `${v},${place[v]},${f(time, 2)},${f(frame.T[v], 2)},${f(frame.vis[v], 2)}`;
      if (extras) {
        const soot = frame.soot[v];
        row += `,${f(soot, 3)},${f(extinction(soot), 4)},${f(obscuration(soot), 2)},${f(opticalDensity(soot), 4)},${f(frame.co2[v], 3)},${f(frame.o2[v], 3)}`;
      }
      const limiting = limitingCriterion({ T: frame.T[v], vis: frame.vis[v], fedGas: tracker.fedGas[v], fedHeat: tracker.fedHeat[v] }, opts);
      row += `,${f(frame.co[v], 1)},${f(tracker.fedGas[v], 5)},${f(tracker.fedHeat[v], 5)},${f(ti[v], 4)},${limiting}`;
      rows[v] = row;
    }
    yield `${rows.join("\n")}\n`;
  }
}

/** The table as a Blob, gzipped where the browser can (CompressionStream), with its file extension. */
export async function runTableBlob(run, opts = {}) {
  const parts = [...runTableChunks(run, opts)];
  const csv = new Blob(parts, { type: "text/csv" });
  if (typeof CompressionStream === "undefined") return { blob: csv, extension: "csv" };
  const gz = await new Response(csv.stream().pipeThrough(new CompressionStream("gzip"))).blob();
  return { blob: gz, extension: "csv.gz" };
}
