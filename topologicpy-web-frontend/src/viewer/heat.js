/**
 * Colours for the graph overlays that recolour it (heat, escape deadlines),
 * apart from ViewerManager so they run and are tested without WebGL.
 */
import { Color } from "three";

/**
 * Yellow -> orange -> red -> dark red across the temperature range: the stops
 * of .legend__ramp (styles/workflow.css), so the legend is literally what is
 * drawn. Lightness falls all the way, so heat reads in order for colour-blind
 * viewers too, which the rainbow this replaced did not. These are the light
 * scene's; SCENE_COLOURS has the dark scene's.
 */
export const HEAT_STOPS = [0xfacc15, 0xf97316, 0xdc2626, 0x7f1d1d].map((hex) => new Color(hex));

/**
 * The ramp and the deadline map's "no departure works" colour for each scene
 * background. On the dark scene (#0d1219) the light ramp's dark red was 1.9:1
 * against it and the old grey 2.4:1, and the two 1.3:1 apart for a protan
 * viewer: there the ramp ends in a red that keeps 3.9:1 (lightness still
 * falling all the way), and "none" is a light grey, 12.7:1 against the
 * background and 3.3:1 against the red. The light scene keeps its ramp, with
 * a mid grey (4.3:1). The legend draws the same (styles/workflow.css).
 */
export const SCENE_COLOURS = {
  light: { stops: HEAT_STOPS, none: new Color(0x71717a) },
  dark: { stops: [0xfacc15, 0xf97316, 0xef4444, 0xdc2626].map((hex) => new Color(hex)), none: new Color(0xd4d4d8) },
};

export function heatColor(target, temp, min, max, stops = HEAT_STOPS) {
  const t = (temp - min) / Math.max(1e-6, max - min);
  // Anything not a number reads as the cool end, rather than indexing past
  // the stops (HEAT_STOPS[NaN].r threw and took the page down).
  const x = (t >= 0 ? Math.min(1, t) : 0) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return target.copy(stops[i]).lerp(stops[i + 1], x - i);
}

/**
 * Colour both ends of every edge by its node's escape deadline (seconds): the
 * heat ramp reversed, so little time left reads as hot, and `none` where no
 * departure works (-Infinity) or the value is missing.
 *
 * @param {Float32Array} colors      the graph's vertex colours, 6 per edge
 * @param {ArrayLike<number>} edges  node pairs
 * @param {number} nodeCount         nodes in the graph drawn
 * @returns {boolean} false, writing nothing, when `deadlines` is not one value
 *          per node of this graph: a map made for a graph since rebuilt.
 */
export function paintDeadlines(colors, edges, nodeCount, deadlines, { max = 300, none, stops = HEAT_STOPS }) {
  if (!deadlines || deadlines.length !== nodeCount) return false;
  const scratch = new Color();
  const edgeCount = edges.length / 2;
  for (let e = 0; e < edgeCount; e += 1) {
    for (let side = 0; side < 2; side += 1) {
      const value = deadlines[edges[e * 2 + side]];
      const c = value > -Infinity ? heatColor(scratch, max - Math.max(0, value), 0, max, stops) : none;
      const offset = e * 6 + side * 3;
      colors[offset] = c.r;
      colors[offset + 1] = c.g;
      colors[offset + 2] = c.b;
    }
  }
  return true;
}
