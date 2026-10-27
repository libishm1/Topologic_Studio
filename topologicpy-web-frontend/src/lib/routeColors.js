/**
 * Colours shared by the viewer overlays and the panels that describe them.
 *
 * Chosen to stay clear of what is already on screen: the graph (blue), stairs
 * (violet), doors (amber), the exit (orange) and the fire (red-orange).
 */

/** Alternative routes, best first. The first is the main route's red. */
export const ROUTE_COLORS = [
  "#dc2626",
  "#0891b2",
  "#db2777",
  "#4f46e5",
  "#0d9488",
  "#9333ea",
  "#ca8a04",
  "#64748b",
];

/** TopologicPy's Navigation graph and its route. */
export const NAV_COLOR = "#84cc16";

export const routeColor = (index) => ROUTE_COLORS[index % ROUTE_COLORS.length];

export const hexToInt = (hex) => parseInt(hex.slice(1), 16);

/** Planning with the fire forecast: the earliest safe route, the least-dose
 *  route, and a route planned on the fire at departure that walks into it. */
export const FORECAST_COLORS = { earliest: "#0891b2", leastDose: "#4f46e5", snapshot: "#db2777" };
