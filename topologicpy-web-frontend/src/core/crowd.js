/**
 * People leaving the building, for the Route step's optional egress
 * animation: a floor-field model on the walkable graph, timed by the SFPE
 * Handbook's hydraulic values.
 *
 * Where everyone goes. By default (`choice` "quickest") each person weighs
 * every exit and takes the one that gets them out soonest: the walk there at
 * their own speed, or the people ahead of them for it (nearer it by the walk)
 * passing at the flow seen through it, whichever is longer. Each looks again
 * every `replan` seconds and turns to another exit only where the two ways
 * part and when it saves SWITCH_GAIN seconds and SWITCH_SHARE of their time,
 * so a queue's back leaves for a free door while its front stays, and no one
 * changes their mind over and over in a jam all the ways out share. The flow
 * through an exit is its door's capacity until someone heading for it is held
 * up, then what passes: a stair that feeds it slower than its door passes
 * makes it look as slow as it is. Pathfinder's "locally quickest" door choice
 * weighs a door the same way, by the walk and the person's place in its queue
 * at the door's observed flow, with a preference for the door already chosen
 * (Thunderhead Engineering, Pathfinder Technical Reference, Path Planning).
 * On the site's examples with no fire, the last one was out sooner where
 * queues formed at some exits while others stood free (the NIBS clinic 459
 * to 333 s, the NIBS office 433 to 350 s), and within 3 s on the others
 * (2026-11-10). With "nearest"
 * each person heads for the nearest exit along the shortest walk, the tree
 * the performance check walks (performanceCheck.js walkTree), as the
 * "nearest exit" policy that gave the shortest exit times in Carrera et al.
 * (2019, doi:10.1007/978-3-319-99608-0_16) and that people without a known
 * exit take (Park et al. 2019, doi:10.1109/iemcon.2019.8936208). The
 * quickest choice keeps a walk tree to each exit; on a graph where those
 * would take more than QUICKEST_CELLS, everyone takes the nearest.
 * A node holds one person at a time, as a cell of the static floor-field
 * automaton holds one (Burstedde et al. 2001, 0.4 m cells; the
 * graph's are about 0.5 m, a jam of about 4 persons/m^2). When the next node
 * on the tree is taken, a person steps to another free neighbour that is
 * also closer to an exit, so the crowd spreads over a corridor's width and
 * queues form behind doors rather than in single file.
 *
 * How fast. Unimpeded speeds spread about SFPE's maximum, 0.85 k = 1.19 m/s
 * with k = 1.40 m/s; with the others within DENSITY_RADIUS of the node ahead
 * at D persons/m^2 over the walkable area there (the walker not counted), a
 * step is at the person's own speed below FREE_DENSITY and at most
 * k (1 - a D), a = 0.266, above it; down a stair k is 1.08 m/s along the
 * slope (7 in risers, 11 in treads), 0.92 m/s at most for the mean walker,
 * up it STAIR_UP of that. A door passes at most the SFPE specific flow of
 * 1.3 persons/s per metre of effective width, the door's less 0.15 m a side
 * (performanceCheck.js doorCapacity; each door node's own width with
 * `doorWidths`): a door of 0.9 m lets about 0.8 persons a second through, and
 * in the queue before it (DOOR_QUEUE_M) the density slowing a step is taken
 * at most at that greatest flow's, so the door, not the crawl into the queue,
 * sets what passes. Each person keeps their own clock: a step begins when
 * they reached their node, the node ahead was left and its door opens,
 * whichever is latest, not at the next time step, so speeds do not depend on
 * a link's direction (tools/validation/crowd-validation.mjs, RiMEA Test 1).
 *
 * Corridors and stairs (`passageFlow`; off by default, as it changes the
 * times where they are narrow). Without it only doors carry a capacity: two
 * lattice rows of corridor pass about 1.19 persons/s, more than a door of that
 * width (RiMEA Test 12b), and a stair, one file of nodes whatever its width,
 * passes about 0.54. With it floor up to PASSAGE_MAX_M across passes at most
 * the SFPE specific flow on its effective width (0.2 m lost a side), and a
 * stair that carries its real width (`graph.widths`, from lattice.js) the
 * stair's (0.15 m a side; its lower specific flow follows from kStair), holding
 * as many abreast as its width takes: passageSections below, and the constants
 * before it. The floor's width is counted in nodes, so to about half a metre;
 * `passageWidths` gives real ones.
 *
 * With a fire (`closes`, each node's untenable time by the planners' rule,
 * core/forecast.js), no one steps on to a place the fire has made
 * untenable: every `replan` seconds each way out is worked out again with a
 * closed place costing FIRE_PENALTY metres, so people turn back and take
 * another exit when the fire cuts theirs, and anyone the fire has reached
 * walks out of it the shortest way. Someone whose only way out is through
 * the fire stays where they are (refuge in place) and is counted as cut off.
 * With the quickest choice a person weighs only the exits whose walk the
 * fire has not cut; once it has cut every one, or reached them, they take
 * the way out worked out with the fire, as with the nearest.
 *
 * On a stair a person steps on only when no one else stands within
 * `stairGap` of the next node on the flight (0.9 m, about a person's
 * length), so a stream down a stair keeps its gaps.
 *
 * When each sets off. By default within `start`, a short uniform stagger.
 * With `preMovement` (for the run, or a group's own) each person draws a
 * pre-movement time of their own, counted from the alarm (`alarm`, seconds
 * into the run): uniform between two times, log-normal, a normal cut off at
 * its ends, or a Weibull (drawPreMovement). The log-normal also takes its
 * 1st and 99th percentiles, the form of the performance check's design
 * values (rsetPresets.js RSET_PRESETS: perfAlarm, perfPre1, perfPre99), and
 * the Weibull the measured clusters there (MEASURED_PRE); no distribution is
 * built in here. The draws come from a stream of their own, so the same
 * people stand in the same places and walk at the same speeds with it or
 * without. Until they set off people stand where they are, fire or not:
 * no one starts early because they see smoke.
 *
 * In smoke (off by default). With a fire run's visibility at head height,
 * frame by frame (`smoke`, as the slime mould's `smokeFrames`), and a
 * relation (`smokeSpeed`), each step is at the person's own speed times a
 * factor read from the smoke where they stand when the step begins; the
 * extinction coefficient is K = 3 / visibility (tenability.js; none at the
 * 100 m cap). The one relation built in, "frantzichNilsson", is FDS+Evac's:
 * v = max(0.1 v0, v0 (alpha + beta K) / alpha), alpha 0.706 m/s, beta
 * -0.057 m^2/s (Korhonen and Hostikka 2009, FDS+Evac Technical Reference and
 * User's Guide, VTT Working Papers 119, Eq. 10, p. 22, after Frantzich and
 * Nilsson 2003, Lund report 3126; the same constants in FDS 6.7.5's
 * evac.f90). Their tunnel was filled with artificial smoke and acetic acid
 * as an irritant, at K of about 2 to 8 /m (as Ronchi, Gwynne and Purser
 * 2011 describe it), so thinner smoke is the fit's extrapolation, and FDS+Evac applies it as a share of anyone's speed, the
 * means only. Jin's curves for irritant and non-irritant smoke are not
 * built in: a curve of one's own may be given instead, as points of K or
 * visibility against the share of the speed kept. The loop's least step
 * speed, 0.15 m/s, still holds, so a walker slower than 1.5 m/s is not
 * slowed all the way to a tenth. The exit choice and the ways out do not
 * reckon with the smoke's slowing, and no one turns back from it.
 *
 * Randomness in how people move (off by default; the plan is
 * wiki/roadmap/crowd-motion-randomness-plan-2026-11-11.md). With `fluctuate`
 * each person's speed wobbles about their own as Corbetta et al. (2017)
 * measured walkers' do (WALK_FLUCTUATION). With `kS` each step is drawn as the
 * floor-field model draws it (Burstedde et al. 2001; Kirchner and
 * Schadschneider 2002): staying, or a free neighbour, with weight exp(kS x
 * the cells it brings nearer the exit), so people sometimes step aside or
 * hesitate, the more so the smaller kS. Both draw from a stream of their own,
 * so with both off a run is exactly as before; with either on, one run is one
 * sample, and the panel runs several.
 *
 * Out. Past an exit each person walks on to a gathering place a few metres
 * outside, clear of the building in plan (the graph's own places on every
 * storey, and the cells round them), at the exit's own level (the model shows
 * no way down from an upper exit, so none is drawn; `exits[].above` gives its
 * height over the lowest exit), and stands there, the later arrivals round the
 * earlier ones, so each exit's crowd shows how many chose it. With a
 * `shelterArea`, a gathering place holds what that area takes at the
 * spacing they stand at (GATHER_AREA a person); once full, each newcomer
 * takes the place of the earliest there, who leaves (`goneAt`) and fades
 * from the view.
 *
 * The result is each person's walk as straight segments (time, from, to),
 * which the view interpolates on the GPU from the playback time alone.
 */
import { Heap, csr, edgeLengths } from "./route.js";
import { KIND_DOOR, KIND_FLOOR, KIND_LIFT, KIND_STAIR } from "./lattice.js";
import { doorCapacity, lognormalFromPercentiles, walkTree } from "./performanceCheck.js";
import { MAX_VISIBILITY, SIGN_CONSTANT } from "./tenability.js";

/** The SFPE Handbook's hydraulic model (5th ed., 2016, ch. 59, Gwynne and Rosenbaum; doi:10.1007/978-1-4939-2565-0_59). */
export const SFPE = {
  // Speed k (1 - a D), m/s, D persons/m^2: corridors, ramps and doorways.
  k: 1.4,
  a: 0.266,
  // Along a stair's slope, 7 in risers and 11 in treads.
  kStair: 1.08,
  // The unimpeded speed, a share of k (the Handbook's table: 1.19 m/s for k = 1.40).
  free: 0.85,
};
/** The unimpeded walking speed, SFPE's 0.85 k: 1.19 m/s. */
export const FREE_SPEED = SFPE.free * SFPE.k;

/**
 * Below this density (persons/m^2) people walk at their own unimpeded speed;
 * above it, at k (1 - a D): SFPE's own threshold for the speed-density
 * relation. A walker never counts themselves (tools/validation: counting the
 * mover capped a lone walker at 1.235 m/s in the open, 0.90 in single file).
 */
export const FREE_DENSITY = 0.54;

/** The density round a node is the persons within this many metres of it over the walkable area there (its nodes' cells). */
export const DENSITY_RADIUS = 1.0;

/**
 * How the density that slows a step is read (`densityReading`). The default,
 * as before: the others within DENSITY_RADIUS of the node ahead over all the
 * cells there, and a least speed of 0.15 m/s. That reads low in a moving
 * crowd (the free node ahead and the walker's own cell count as floor but
 * hold no one, and people step into gaps), so dense flow is too fast.
 * The parameters, and the rules they are calibrated by, are in
 * wiki/research/crowd-calibration-2026-11-24.md:
 *   centre  "ahead" (the node stepped on to) or "own" (the walker's node)
 *   radius  m, the disc the others are counted in
 *   area    "all" (every cell in the disc) or "standing" (less the walker's
 *           cell and the target's: the floor the others could stand on)
 *   floor   m/s, the least speed of a step
 */
export const DENSITY_READING_DEFAULT = { centre: "ahead", radius: DENSITY_RADIUS, area: "all", floor: 0.15 };
/**
 * The reading chosen on RiMEA Test 16's single-file data alone, of 36 combinations
 * (tools/validation/crowd-calibrate.mjs, 2026-11-24): the others within 2 m of the
 * node ahead over the floor they could stand on. All 14 points then lie inside the
 * experiments' 10 to 90 % band (8 before). That is a fit to single-file data, not
 * a validation, and on the data held back it failed the rules set for it
 * (wiki/research/crowd-calibration-2026-11-24.md, section 7): in a 10 m corridor
 * at 2.3 persons/m^2 the speed is still 26 % above the SFPE relation (56 % before),
 * and the queue before a corridor or a stair crawls again, so a 1 m stair passes
 * 0.53 persons/s with passageFlow (0.71 by the default reading) and 0.15 without.
 * It is therefore not the default and the app does not offer it.
 */
export const DENSITY_READING_CALIBRATED = { centre: "ahead", radius: 2.0, area: "standing", floor: 0.15 };

/**
 * Within this walk (m) of a door or an exit, the density slowing a step is
 * taken at most at QUEUE_DENSITY, the density of SFPE's greatest specific
 * flow, so that the queue feeds the door and the door's capacity (its
 * specific flow times its effective width) governs what passes. Crawling into
 * a packed queue at the 0.15 m/s floor passed 0.29 persons/s through a door
 * of any width (tools/validation, RiMEA Test 12d).
 */
export const DOOR_QUEUE_M = 2.0;
export const QUEUE_DENSITY = 1.9;

/**
 * The flow a passage takes (`passageFlow`, off by default). A corridor and a
 * stair pass at most SFPE's greatest specific flow, (1 - a D) k D at
 * D = QUEUE_DENSITY, for each metre of effective width: the width less the
 * Handbook's boundary layer a side, 0.2 m at a corridor's walls and 0.15 m at
 * a stair's. With k = 1.40 m/s that is 1.32 persons/s/m on the level, the
 * doors' value (performanceCheck.js DOOR_FLOW); with the stair's k = 1.08 m/s
 * (7 in risers, 11 in treads) it is 1.01 persons/s/m, which is the Handbook's
 * tabulated maximum for that stair; up a stair STAIR_UP of it, as the speed.
 * A passage 1 m wide then passes 0.79 persons/s as a corridor and 0.71 as a
 * stair (a door 0.92), the values Pathfinder's SFPE mode reports.
 */
export const CORRIDOR_BOUNDARY = 0.2;
export const STAIR_BOUNDARY = 0.15;
export const CORRIDOR_FLOW = (1 - SFPE.a * QUEUE_DENSITY) * SFPE.k * QUEUE_DENSITY;
export const STAIR_FLOW = (1 - SFPE.a * QUEUE_DENSITY) * SFPE.kStair * QUEUE_DENSITY;
/** Persons a second a corridor (or a stair, going down) `width` m wide passes, less its boundary layers (at least 0.1 m left). */
export const passageCapacity = (width, stair = false) => (stair ? STAIR_FLOW : CORRIDOR_FLOW) * Math.max(0.1, width - 2 * (stair ? STAIR_BOUNDARY : CORRIDOR_BOUNDARY));
/**
 * Floor no wider than this (m) across is a passage with a capacity. Wider
 * floor is left alone: one person a node already passes less there than the
 * SFPE flow (about 1.15 persons/s for each metre against 1.32 for each metre
 * of effective width, which cross near 3 m).
 */
export const PASSAGE_MAX_M = 3.0;
/** A stair with a width holds one person abreast for each of these metres of it (Fruin's body ellipse is 0.61 m across the shoulders). */
export const STAIR_ABREAST_M = 0.6;

/**
 * Up a stair at this share of the speed down it: Fruin's ascending over
 * descending speeds on internal stairs, averaged over his three age groups
 * (0.55/0.76, 0.50/0.65, 0.42/0.55), as RiMEA 4.1.1 Table 3 gives them. SFPE's
 * kStair is kept for the way down.
 */
export const STAIR_UP = 0.75;

/** Metres a place the fire has made untenable adds to a way out: only taken by those already in it. */
export const FIRE_PENALTY = 1000;

/** How far from its exit a gathering place is put (m): the nearest of these that is clear of the building. */
const GATHER_REACH = [3.5, 5, 7, 10, 14, 20];
/** The radius about a gathering place's centre that must be clear of the building (m). */
const GATHER_CLEAR = 1.5;
/**
 * Populations for a group's `speed`: unimpeded walking speeds on the level, m/s.
 * - weidmann: the normal distribution RiMEA's analyses start from (Weidmann 1993:
 *   mean 1.34 m/s, standard deviation 0.26), cut off at 0.6 and 2.0 m/s here.
 * - imoMales30to50: IMO MSC.1/Circ.1533's males of 30 to 50 years on flat terrain,
 *   uniform between 0.97 and 1.62 m/s (its Test 7).
 */
export const POPULATIONS = {
  weidmann: { mean: 1.34, sd: 0.26, min: 0.6, max: 2.0 },
  imoMales30to50: [0.97, 1.62],
  // IMO MSC.1/Circ.1533, annex 3, appendix, Table 3.4: walking speed on flat terrain,
  // each group uniform between its least and greatest.
  imoFemalesUnder30: [0.93, 1.55],
  imoFemales30to50: [0.71, 1.19],
  imoFemalesOver50: [0.56, 0.94],
  imoFemalesOver50Impaired1: [0.43, 0.71],
  imoFemalesOver50Impaired2: [0.37, 0.61],
  imoMalesUnder30: [1.11, 1.85],
  imoMalesOver50: [0.84, 1.4],
  imoMalesOver50Impaired1: [0.64, 1.06],
  imoMalesOver50Impaired2: [0.55, 0.91],
  imoCrewFemales: [0.93, 1.55],
  imoCrewMales: [1.11, 1.85],
  // RiMEA 4.1.1, Table 2: persons with impaired mobility, in the plane.
  rimeaImpaired: [0.46, 0.76],
};

/**
 * IMO MSC.1/Circ.1533, annex 3, appendix, Table 3.1: the passengers' composition by
 * age and sex, in per cent (the keys are POPULATIONS'). RiMEA's own speeds by age are
 * given only as a curve (its Fig. 3, after Weidmann), so none is tabulated here.
 */
export const IMO_PASSENGERS = {
  imoFemalesUnder30: 7,
  imoFemales30to50: 7,
  imoFemalesOver50: 16,
  imoFemalesOver50Impaired1: 10,
  imoFemalesOver50Impaired2: 10,
  imoMalesUnder30: 7,
  imoMales30to50: 7,
  imoMalesOver50: 16,
  imoMalesOver50Impaired1: 10,
  imoMalesOver50Impaired2: 10,
};

/**
 * Groups for simulateCrowd that put `count` people on `nodes` in the shares of `mix`
 * (per cent by population, as IMO_PASSENGERS), each walking at its population's speeds.
 * The head counts are whole, by largest remainder; `extra` is added to every group
 * (an exit, start times).
 */
export function populationMix(nodes, count, mix = IMO_PASSENGERS, extra = {}) {
  const keys = Object.keys(mix);
  const total = keys.reduce((s, k) => s + mix[k], 0);
  const exact = keys.map((k) => (count * mix[k]) / total);
  const heads = exact.map(Math.floor);
  const rest = count - heads.reduce((s, x) => s + x, 0);
  keys.map((_, i) => i).sort((a, b) => exact[b] - heads[b] - (exact[a] - heads[a]) || a - b).slice(0, rest).forEach((i) => { heads[i] += 1; });
  return keys.map((k, i) => ({ ...extra, name: k, nodes, count: heads[i], speed: POPULATIONS[k] })).filter((g) => g.count > 0);
}

/** One speed from a population: [least, most], uniform, or {mean, sd, min, max}, a normal cut off at its ends. */
function drawSpeed(population, rand) {
  if (Array.isArray(population)) return population[0] + (population[1] - population[0]) * rand();
  const { mean, sd, min = mean - 3 * sd, max = mean + 3 * sd } = population;
  for (let tries = 0; tries < 200; tries += 1) {
    const x = mean + sd * Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
    if (x >= min && x <= max) return x;
  }
  return Math.min(max, Math.max(min, mean));
}

/**
 * One pre-movement time, seconds after the alarm, from a distribution:
 * - [least, most]: uniform;
 * - {median or mu, sigma, min?, max?}: log-normal, mu and sigma those of the
 *   logarithm of the seconds (mu = ln median); or {p1, p99, min?, max?}, the
 *   log-normal through its 1st and 99th percentiles (performanceCheck.js
 *   lognormalFromPercentiles);
 * - {mean, sd, min?, max?}: normal;
 * - {scale, shape, min?, max?}: Weibull (rsetPresets.js MEASURED_PRE's form).
 * `min` (0 when left out) and `max` (none) cut the distribution off: a draw
 * outside them is drawn again, so what is left keeps its shape.
 * `rand` is a uniform generator on [0, 1).
 */
export function drawPreMovement(distribution, rand) {
  const d = distribution;
  if (Array.isArray(d)) return d[0] + (d[1] - d[0]) * rand();
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  let draw, middle;
  if (d.scale !== undefined && d.shape !== undefined) {
    draw = () => d.scale * (-Math.log(1 - rand())) ** (1 / d.shape);
    middle = d.scale * Math.LN2 ** (1 / d.shape);
  } else if (d.sigma !== undefined || d.p1 !== undefined) {
    const { mu, sigma } = d.p1 !== undefined ? lognormalFromPercentiles(d.p1, d.p99) : { mu: d.mu ?? Math.log(d.median), sigma: d.sigma };
    draw = () => Math.exp(mu + sigma * gauss());
    middle = Math.exp(mu);
  } else if (d.mean !== undefined && d.sd !== undefined) {
    draw = () => d.mean + d.sd * gauss();
    middle = d.mean;
  } else throw new Error("simulateCrowd: a pre-movement distribution is [least, most], {median or mu, sigma}, {p1, p99}, {mean, sd} or {scale, shape}");
  const min = d.min ?? 0, max = d.max ?? Infinity;
  if (!Number.isFinite(middle) || !(min <= max)) throw new Error("simulateCrowd: a pre-movement distribution's numbers are not usable");
  for (let tries = 0; tries < 200; tries += 1) {
    const x = draw();
    if (x >= min && x <= max) return x;
  }
  return Math.min(max, Math.max(min, middle));
}

/**
 * Walking speed in smoke, for `smokeSpeed`: the share of their own speed a
 * person keeps at the extinction coefficient K (1/m) where they stand.
 * - frantzichNilsson: max(floor, (alpha + beta K) / alpha), as FDS+Evac
 *   applies Frantzich and Nilsson's fit (Korhonen and Hostikka 2009, VTT
 *   Working Papers 119, Eq. 10, p. 22: alpha 0.706 m/s, beta -0.057 m^2/s,
 *   least speed 0.1 of the person's own; the fit's standard deviations,
 *   0.069 and 0.015, are not used there or here). The experiments were in
 *   irritant artificial smoke at K of about 2 to 8 /m.
 * A relation of one's own: {alpha, beta, floor}, or a curve {by: "K" or
 * "visibility", points: [[x, share], ...]} with x rising, read by straight
 * lines between its points and held at its ends.
 */
export const SMOKE_SPEED = {
  frantzichNilsson: { alpha: 0.706, beta: -0.057, floor: 0.1 },
};

/** The share of their speed kept at extinction coefficient K (1/m) under a relation (SMOKE_SPEED's forms). */
export function smokeSpeedFactor(relation, K) {
  if (relation.points) {
    const x = relation.by === "visibility" ? (K > 0 ? SIGN_CONSTANT / K : Infinity) : K;
    const pts = relation.points;
    if (x <= pts[0][0]) return Math.max(0, pts[0][1]);
    for (let k = 1; k < pts.length; k += 1) {
      if (x <= pts[k][0]) return Math.max(0, pts[k - 1][1] + ((x - pts[k - 1][0]) / (pts[k][0] - pts[k - 1][0])) * (pts[k][1] - pts[k - 1][1]));
    }
    return Math.max(0, pts[pts.length - 1][1]);
  }
  return Math.min(1, Math.max(relation.floor ?? 0, 1 + (relation.beta / relation.alpha) * K));
}

/** m^2 a person standing at a gathering place: the sunflower's spacing, pi 0.45^2. */
export const GATHER_AREA = Math.PI * 0.45 ** 2;

/** How each person picks an exit: the one that gets them out soonest, or the nearest. */
export const EXIT_CHOICES = ["quickest", "nearest"];

/** Seconds, and the share of their time out, another exit must save before someone heading for one turns to it. */
export const SWITCH_GAIN = 5;
export const SWITCH_SHARE = 0.2;

/** At most this many exits x nodes for the quickest choice (a walk tree an exit, 12 bytes a node); above it, the nearest. */
export const QUICKEST_CELLS = 2e6;

/**
 * Walking-speed fluctuations as Corbetta, Lee, Benzi, Muntean and Toschi
 * measured them on 72,376 walkers (2017, doi:10.1103/PhysRevE.95.032316;
 * arXiv:1610.07429, Table 1): the speed u along the walk follows
 * du = -4 alpha u (u^2 - up^2) dt + sigma dW, alpha 0.0625 m^-2 s,
 * sigma 0.16 m s^-3/2, up 1.0 m/s; about up it relaxes at 8 alpha up^2 =
 * 0.5 per second (about 2 s) with a spread of 0.16 m/s. Here as a factor
 * f = u / up on each person's own unimpeded speed, worked in steps of at most
 * `step` seconds and kept above `floor`, the unstable point (about 0.2 m/s):
 * the one walker in about 450 to 1,000 who turns back is left out.
 */
export const WALK_FLUCTUATION = { alpha: 0.0625, sigma: 0.16, up: 1.0, step: 0.1, floor: 0.2 };

/** The drift of the speed factor f: -4 alpha up^2 f (f^2 - 1), zero at f = 1. */
export const fluctuationDrift = (f) => -4 * WALK_FLUCTUATION.alpha * WALK_FLUCTUATION.up ** 2 * f * (f * f - 1);

/**
 * The floor field's sensitivity kS for the step choice (Kirchner and
 * Schadschneider 2002, doi:10.1016/S0378-4371(02)00857-9, who ran 0.1, 0.4
 * and 10): null steps as before (the limit of a large kS); 10 their "strong",
 * near-ordered walking; 0.4 their intermediate, people far less sure of the
 * way; 2 between.
 */
export const STEP_RANDOMNESS = { exact: null, direct: 10, wandering: 2, disoriented: 0.4 };

/** Seconds the flow seen through an exit is averaged over. */
export const FLOW_SECONDS = 20;

/** The least share of its door's capacity an exit's flow is taken at, so one no one has reached yet is not thought shut. */
export const FLOW_FLOOR = 0.1;

/** A seeded uniform generator (mulberry32), so a crowd is the same each time. */
function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The passages of a graph that carry a flow capacity (`passageFlow`).
 *
 * Floor. Through each floor node run the lattice's two lines (its links about
 * `spacing` long, and across them); the shorter of the two runs of floor
 * nodes is the passage's cross-section there, when it is at most `maxWidth`
 * across and runs from wall to wall (a run that ends at a door, an exit or a
 * stair is a way through, not a cross-section). The nodes of one cross-section share a capacity: who steps on to it
 * from outside it waits for the one before. Its width is its nodes times the
 * spacing, a coarse measure (to half a metre on a 0.5 m lattice); a single
 * file of nodes, whose floor may be anything under two nodes wide, is taken
 * as two nodes wide. `passageWidths` gives a node's passage its real width.
 * Passages askew to the lattice are not found.
 *
 * Stairs. A stair drawn as one file of nodes (lattice.js) carries its real
 * width in `graph.widths` (m a node, 0: none), or `passageWidths`. Each such
 * node is its own cross-section at the stair's capacity, holds one person
 * abreast for each STAIR_ABREAST_M of width, and counts as its width times
 * its length of walk in the density; in place of the one person a `stairGap`
 * the stair fills to QUEUE_DENSITY, the density of SFPE's greatest flow, counted
 * over the stair ahead of the walker; the floor node at each end stands for
 * the stair's mouth and holds as many. A stair node without a width is as
 * before.
 *
 * The queue. Within DOOR_QUEUE_M of a passage's mouth (its nodes next to open
 * floor) and of a stair with a width, the density slowing a step is taken at
 * most at QUEUE_DENSITY, as before a door, so the passage's capacity and not
 * the crawl into the queue sets what passes.
 *
 * @returns {{section: Int32Array (each node's cross-section, -1: none), capacity: number[] (persons/s),
 *   width: number[] (m), slots: Uint16Array (persons a node holds), area: Float32Array (m^2 a node, 0: its cell),
 *   mouth: Uint8Array (a passage's nodes next to open floor, and a stair's with a width: the queue before
 *   them is fed as a door's is), corridors: number, stairs: number}}
 */
export function passageSections(graph, { exits = [], blocked = null, passageWidths = null, maxWidth = PASSAGE_MAX_M } = {}) {
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const [h1, h2] = [0, 1, 2].filter((c) => c !== up);
  const kinds = graph.kinds || new Uint8Array(n);
  const spacing = graph.spacing || 0.5;
  const P = graph.nodes;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const length = edgeLengths(graph);
  const isExit = new Uint8Array(n);
  for (const e of exits) if (e >= 0) isExit[e] = 1;
  const floor = (v) => kinds[v] === KIND_FLOOR && !isExit[v];
  const section = new Int32Array(n).fill(-1);
  const capacity = [], width = [];
  const slots = new Uint16Array(n).fill(1);
  const area = new Float32Array(n);
  const mouth = new Uint8Array(n);
  const dir = (a, b) => {
    const ex = P[b * 3 + h1] - P[a * 3 + h1], ey = P[b * 3 + h2] - P[a * 3 + h2], l = Math.hypot(ex, ey);
    return l > 1e-6 ? [ex / l, ey / l] : null;
  };
  // A lattice line's link: floor to floor, open, and about a spacing long (not a diagonal).
  const lineLink = (k) => !blocked?.[edgeIds[k]] && floor(neighbours[k]) && length[edgeIds[k]] <= 1.2 * spacing;
  const along = (x, dx, dy) => {
    for (let k = indptr[x]; k < indptr[x + 1]; k += 1) {
      if (!lineLink(k)) continue;
      const d = dir(x, neighbours[k]);
      if (d && d[0] * dx + d[1] * dy > 0.95) return neighbours[k];
    }
    return -1;
  };
  // Whether the line goes on from x into a door, an exit or a stair: then it is a way through, not wall to wall.
  const opensOn = (x, dx, dy) => {
    for (let k = indptr[x]; k < indptr[x + 1]; k += 1) {
      if (blocked?.[edgeIds[k]] || floor(neighbours[k])) continue;
      const d = dir(x, neighbours[k]);
      if (d && d[0] * dx + d[1] * dy > 0.95) return true;
    }
    return false;
  };
  const keys = new Map();
  for (let v = 0; v < n; v += 1) {
    if (!floor(v)) continue;
    const own = passageWidths && passageWidths[v] > 0 ? passageWidths[v] : 0;
    const limit = Math.max(maxWidth, 2 * own);
    const lines = [];
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      if (!lineLink(k)) continue;
      const d = dir(v, neighbours[k]);
      if (!d) continue;
      for (const [dx, dy] of [d, [-d[1], d[0]]]) if (!lines.some(([x, y]) => Math.abs(x * dx + y * dy) > 0.95)) lines.push([dx, dy]);
    }
    let best = null;
    for (const [dx, dy] of lines) {
      const run = [v];
      let within = true;
      for (const s of [1, -1]) {
        let end = v;
        for (let x = along(v, s * dx, s * dy); x >= 0 && within; x = along(x, s * dx, s * dy)) {
          run.push(x);
          end = x;
          if (run.length * spacing > limit + 1e-6) within = false;
        }
        if (within && opensOn(end, s * dx, s * dy)) within = false;
      }
      if (within && (!best || run.length < best.length)) best = run;
    }
    if (!best) continue;
    const key = best.sort((a, b) => a - b).join(",");
    let id = keys.get(key);
    if (id === undefined) {
      id = capacity.length;
      keys.set(key, id);
      width.push(Math.max(best.length, 2) * spacing);
      capacity.push(passageCapacity(width[id]));
    }
    section[v] = id;
    if (own > 0) {
      width[id] = own;
      capacity[id] = passageCapacity(own);
    }
  }
  let stairs = 0;
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] !== KIND_STAIR) continue;
    const w = passageWidths && passageWidths[v] > 0 ? passageWidths[v] : graph.widths && graph.widths[v] > 0 ? graph.widths[v] : 0;
    if (!(w > 0)) continue;
    stairs += 1;
    section[v] = capacity.length;
    width.push(w);
    capacity.push(passageCapacity(w, true));
    slots[v] = Math.max(1, Math.floor(w / STAIR_ABREAST_M + 1e-6));
    let sum = 0, links = 0;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      sum += length[edgeIds[k]];
      links += 1;
      const b = neighbours[k];
      if (floor(b)) {
        slots[b] = Math.max(slots[b], slots[v]);
        area[b] = Math.max(area[b], w * spacing);
      }
    }
    area[v] = w * (links ? sum / links : spacing);
    mouth[v] = 1;
  }
  // Where open floor meets a passage: its mouth, fed as a door is (DOOR_QUEUE_M).
  for (let v = 0; v < n; v += 1) {
    if (!floor(v) || section[v] < 0) continue;
    for (let k = indptr[v]; k < indptr[v + 1] && !mouth[v]; k += 1) if (!blocked?.[edgeIds[k]] && floor(neighbours[k]) && section[neighbours[k]] < 0) mouth[v] = 1;
  }
  return { section, capacity, width, slots, area, mouth, corridors: keys.size, stairs };
}

/**
 * The crowd's walk out.
 *
 * @param {object} graph  the walkable graph (nodes, edges, kinds, blocked, spacing, agentHeight, upAxis)
 * @param {object} opts
 *   exits         the exit nodes (the plan's: exits out of the building)
 *   occupantArea  m^2 of walkable floor per person (the performance check's)
 *   maxPeople     at most this many, spread alike over the floors
 *   speed         the mean unimpeded speed, m/s (FREE_SPEED)
 *   spread        each person's speed is speed x (1 +- spread), uniform
 *   start         [earliest, latest] start, s: a short stagger, not a pre-movement time
 *   preMovement   null (the default: `start`), or the distribution each person's pre-movement
 *                 time is drawn from (drawPreMovement), seconds after the alarm
 *   alarm         s into the run at which the alarm is given: added to each pre-movement time (0)
 *   doorWidth     m, every door's that `doorWidths` gives none
 *   doorWidths    m a node, each door node's own clear width (0: doorWidth); null: doorWidth for all
 *   passageFlow   corridors and stairs pass at most the SFPE flow for their width (passageSections;
 *                 default off: only doors carry a capacity, as before)
 *   densityReading how the density that slows a step is read: null (the default: DENSITY_READING_DEFAULT,
 *                 as before), "calibrated" (DENSITY_READING_CALIBRATED), or {centre, radius, area, floor}
 *   passageWidths m a node, the real width of the passage a floor or stair node stands in (0: the
 *                 graph's: the floor's nodes across, a stair's `graph.widths`); null: the graph's for all
 *   dt            s, the step
 *   horizon       s, stop then; anyone still inside is reported
 *   seed          for where people are and how fast they walk
 *   blocked       edges no one may cross (walls)
 *   stairGap      m kept clear round the next node on a stair (0: none)
 *   shelterArea   m^2 each gathering place holds (Infinity: everyone)
 *   closes        each node's untenable time, s (a fire's; null: none)
 *   smoke         null, or a fire run's visibility at head height: {K frames, frameSeconds,
 *                 vis: Float32Array of K x n metres}, on the run's clock (the last frame holds after it)
 *   smokeSpeed    null (the default: smoke slows no one), "frantzichNilsson" (SMOKE_SPEED), or a
 *                 relation of one's own ({alpha, beta, floor}, or a curve {by, points}); needs `smoke`
 *   replan        s between ways out worked out again with a fire, and exits weighed again
 *   choice        "quickest" (default) or "nearest" (EXIT_CHOICES)
 *   fluctuate     each person's speed wobbles in time (WALK_FLUCTUATION; default off)
 *   kS            the floor field's step choice (STEP_RANDOMNESS; null, the default: as before)
 *   groups        null (the default: people over all the floor, as above), or a list of
 *                 {nodes, count, exit?, speed?, start?}: `count` people on places drawn from
 *                 `nodes` (a start area), each heading for `exit` if one is given (an exit
 *                 node of `exits`; they keep it whatever the queue, until a fire cuts their walk to it: then
 *                 they choose as everyone else does, and `released` counts them),
 *                 walking at `speed` (a population: [least, most] m/s, uniform, or
 *                 {mean, sd, min, max}, a normal cut off at min and max; POPULATIONS) and
 *                 setting off within `start` ([earliest, latest] s), or after a pre-movement
 *                 time of the group's own (`preMovement`, with its own `alarm` if given; it
 *                 takes the place of the group's `start`). What a group leaves
 *                 out is as the options above give it. `occupantArea` and `maxPeople`
 *                 are then not used.
 * @returns {{
 *   people: number, points: Float32Array, segments: Float32Array, first: Int32Array, count: Int32Array,
 *   exitOf: Int32Array, outAt: Float32Array, exits: Array<{node: number, people: number, last: number}>,
 *   unreached: number, trapped: number (the people the unreached places would hold), inside: number, lastOut: number, longestWait: number, capped: boolean,
 *   goneAt: Float32Array, shelter: number, cutOff: number, choice: string, tooLarge: boolean,
 *   switches: Int32Array, groupOf: Int32Array|null (each person's group), short: number
 *   (people the groups asked for that their places could not hold), released: number,
 *   passages: null | {corridors: number, stairs: number} (with passageFlow: the floor cross-sections
 *   and the stair nodes that carry a capacity),
 *   startAt: Float32Array (when each was free to set off, s: the stagger, or the alarm and
 *   their pre-movement time), smokeSpeed: string|null (the relation applied: its name, or
 *   "custom"), slowed: number (people who took a step slowed by smoke)
 * }}
 *   choice: the one taken, "nearest" when the quickest was asked with a
 *   single exit or on a graph too large for it (tooLarge, QUICKEST_CELLS);
 *   switches: how many times each person turned to another exit after the
 *   start (0 with the nearest);
 *   points: xyz of the graph's nodes at foot level, then the gathering places;
 *   segments: 4 numbers each, start s, end s, point from, point to, each person's
 *   from first[i], count[i] of them, in time order (a gap between two is a wait).
 */
export function simulateCrowd(graph, opts = {}) {
  const {
    exits = [],
    occupantArea = 9,
    maxPeople = 4000,
    speed = FREE_SPEED,
    spread = 0.25,
    start = [0, 10],
    doorWidth = 0.9,
    doorWidths = null,
    passageFlow = false,
    passageWidths = null,
    densityReading = null,
    dt = 0.1,
    horizon = 900,
    seed = 1,
    blocked = null,
    stairGap = 0.9,
    shelterArea = Infinity,
    closes = null,
    replan = 2,
    choice = "quickest",
    fluctuate = false,
    kS = null,
    groups = null,
    preMovement = null,
    alarm = 0,
    smoke = null,
    smokeSpeed = null,
  } = opts;
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const kinds = graph.kinds || new Uint8Array(n);
  const cell = (graph.spacing || 0.5) ** 2;
  const { indptr, neighbours, edgeIds } = csr(graph);
  const length = edgeLengths(graph);
  // Metres to the nearest exit, and the next node on the way.
  let { time: dist, next } = walkTree(graph, exits, { speed: 1, blocked });
  // The same with the places the fire has closed by t costing FIRE_PENALTY each.
  const closedAt = (v, t) => closes !== null && closes[v] <= t;
  const openTree = (t) => {
    const time = new Float64Array(n).fill(Infinity);
    const via = new Int32Array(n).fill(-1);
    const heap = new Heap();
    for (const e of exits) {
      if (e < 0) continue;
      const start = closedAt(e, t) ? FIRE_PENALTY : 0;
      if (start >= time[e]) continue;
      time[e] = start;
      heap.push(start, e);
    }
    const done = new Uint8Array(n);
    while (heap.size) {
      const a = heap.pop();
      if (done[a]) continue;
      done[a] = 1;
      for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
        if (blocked?.[edgeIds[k]]) continue;
        const b = neighbours[k];
        const d = time[a] + length[edgeIds[k]] + (closedAt(b, t) ? FIRE_PENALTY : 0);
        if (d < time[b]) {
          time[b] = d;
          via[b] = a;
          heap.push(d, b);
        }
      }
    }
    return { time, next: via };
  };
  const isExit = new Uint8Array(n);
  for (const e of exits) if (e >= 0) isExit[e] = 1;

  // Where people are: walkable nodes that reach an exit (not doors, stairs
  // or lifts), as many as the floor's area holds at the occupant area.
  const rand = random(seed);
  const places = [];
  let unreached = 0;
  for (let v = 0; v < n; v += 1) {
    if (kinds[v] === KIND_DOOR || kinds[v] === KIND_STAIR || kinds[v] === KIND_LIFT || isExit[v]) continue;
    if (Number.isFinite(dist[v])) places.push(v);
    else unreached += 1;
  }
  // With groups: each group's people on places drawn from its own start area.
  let grouped = null;
  let short = 0;
  if (groups) {
    const usable = new Uint8Array(n);
    for (const v of places) usable[v] = 1;
    const taken = new Uint8Array(n);
    grouped = { at: [], group: [] };
    groups.forEach((g, k) => {
      const own = [...new Set(g.nodes)].filter((v) => v >= 0 && v < n && usable[v] && !taken[v]);
      const count = Math.min(g.count ?? own.length, own.length);
      short += Math.max(0, (g.count ?? own.length) - count);
      for (let i = 0; i < count; i += 1) {
        const j = i + Math.floor(rand() * (own.length - i));
        [own[i], own[j]] = [own[j], own[i]];
        taken[own[i]] = 1;
        grouped.at.push(own[i]);
        grouped.group.push(k);
      }
    });
  }
  const wanted = Math.ceil((places.length * cell) / Math.max(0.1, occupantArea));
  const people = grouped ? grouped.at.length : Math.min(wanted, maxPeople, places.length);
  // A partial Fisher-Yates shuffle picks them.
  if (!grouped)
    for (let i = 0; i < people; i += 1) {
      const j = i + Math.floor(rand() * (places.length - i));
      [places[i], places[j]] = [places[j], places[i]];
    }
  const at = Int32Array.from(grouped ? grouped.at : places.slice(0, people));
  const groupOf = grouped ? Int32Array.from(grouped.group) : null;
  // The exit node each person was given (-1: they choose).
  const assigned = new Int32Array(people).fill(-1);
  const vmax = new Float64Array(people);
  const ready = new Float64Array(people);
  // Pre-movement times are drawn from a stream of their own: the places and speeds are the same with them or without.
  const preRand = random((seed ^ 0x85ebca6b) >>> 0);
  for (let i = 0; i < people; i += 1) {
    const g = groupOf ? groups[groupOf[i]] : null;
    vmax[i] = g?.speed ? drawSpeed(g.speed, rand) : speed * (1 + spread * (2 * rand() - 1));
    const [early, late] = g?.start ?? start;
    ready[i] = early + (late - early) * rand();
    // A pre-movement time of their own after the alarm, the group's or the run's (a group's own `start` stands before the run's distribution).
    const pre = g?.preMovement ?? (g?.start ? null : preMovement);
    if (pre) ready[i] = (g?.alarm ?? alarm) + drawPreMovement(pre, preRand);
    if (g && g.exit !== undefined && g.exit !== null) {
      if (!isExit[g.exit]) throw new Error("simulateCrowd: a group's exit is not one of the exits");
      assigned[i] = g.exit;
    }
  }
  const startAt = Float32Array.from(ready);
  // How many stand on each node: one at most, but for a stair with a width (passageFlow).
  const load = new Uint16Array(n);
  for (let i = 0; i < people; i += 1) load[at[i]] += 1;

  // Foot-level points: the nodes stand agentHeight above their floor.
  const lift = graph.agentHeight || 0;
  const extra = [];
  const pointOf = (v) => [graph.nodes[v * 3] - (up === 0 ? lift : 0), graph.nodes[v * 3 + 1] - (up === 1 ? lift : 0), graph.nodes[v * 3 + 2] - (up === 2 ? lift : 0)];
  // Each exit's gathering place: a few metres out, away from the nodes inside, at the exit's own level.
  const gather = new Map();
  const ground = Math.min(...exits.filter((e) => e >= 0).map((e) => pointOf(e)[up]));
  // The building in plan: every walkable place on every storey, and the cells round it
  // (its walls). A gathering place and each person standing at one keep off it.
  const [h1, h2] = [0, 1, 2].filter((c) => c !== up);
  const step = graph.spacing || 0.5;
  const planKey = (a, b) => `${Math.round(a / step)},${Math.round(b / step)}`;
  const building = new Set();
  for (let v = 0; v < n; v += 1) {
    const a = Math.round(graph.nodes[v * 3 + h1] / step), b = Math.round(graph.nodes[v * 3 + h2] / step);
    for (let i = -1; i <= 1; i += 1) for (let j = -1; j <= 1; j += 1) building.add(`${a + i},${b + j}`);
  }
  const indoors = (q) => building.has(planKey(q[h1], q[h2]));
  // How much of the building a gathering place `reach` m from p along `dir` meets: on the
  // way to it (past the doorway's own cells) and where its first people stand.
  const meets = (p, dir, reach) => {
    let hits = 0;
    for (let r = 1.5; r < reach; r += step) if (building.has(planKey(p[h1] + dir[h1] * r, p[h2] + dir[h2] * r))) hits += 1;
    const c1 = p[h1] + dir[h1] * reach, c2 = p[h2] + dir[h2] * reach;
    for (let a = -GATHER_CLEAR; a <= GATHER_CLEAR + 1e-9; a += step)
      for (let b = -GATHER_CLEAR; b <= GATHER_CLEAR + 1e-9; b += step) if (a * a + b * b <= GATHER_CLEAR * GATHER_CLEAR && building.has(planKey(c1 + a, c2 + b))) hits += 1;
    return hits;
  };
  const outward = (e) => {
    if (gather.has(e)) return gather.get(e);
    const p = pointOf(e);
    const d = [0, 0, 0];
    for (let k = indptr[e]; k < indptr[e + 1]; k += 1) {
      const q = pointOf(neighbours[k]);
      for (let c = 0; c < 3; c += 1) if (c !== up) d[c] += p[c] - q[c];
    }
    let norm = Math.hypot(d[0], d[1], d[2]);
    if (norm < 1e-9) {
      d[(up + 1) % 3] = 1;
      norm = 1;
    }
    // Away from the neighbours is the first guess. Where that way leads back over the
    // building (a porch or terrace beyond the door, a recessed entrance, an inner corner),
    // the nearest clear place is taken: the other way, then round the compass, then further out.
    const first = d.map((x) => x / norm);
    const ways = [first, first.map((x) => -x || 0)];
    for (let k = 1; k < 16; k += 1) {
      if (k === 8) continue;
      const a = (k * Math.PI) / 8, w = [0, 0, 0];
      w[h1] = first[h1] * Math.cos(a) - first[h2] * Math.sin(a);
      w[h2] = first[h1] * Math.sin(a) + first[h2] * Math.cos(a);
      ways.push(w);
    }
    let dir = first, reach = GATHER_REACH[0], least = Infinity;
    search: for (const r of GATHER_REACH)
      for (const w of ways) {
        const hits = meets(p, w, r);
        if (hits < least) [dir, reach, least] = [w, r, hits];
        if (hits === 0) break search;
      }
    const side = [0, 0, 0];
    // The horizontal perpendicular: dir rotated a quarter turn about up.
    side[h1] = -dir[h2];
    side[h2] = dir[h1];
    const g = { centre: p.map((x, c) => (c === up ? x : x + dir[c] * reach)), above: p[up] - ground, dir, side, people: 0, last: 0, spots: [], here: [], next: 0 };
    gather.set(e, g);
    return g;
  };
  // How many a gathering place holds, and when each person left a full one.
  const shelter = Number.isFinite(shelterArea) ? Math.max(1, Math.floor(shelterArea / GATHER_AREA)) : Infinity;
  const goneAt = new Float32Array(people).fill(Infinity);
  const spot = (g, i, t) => {
    // A sunflower: the k-th place 0.45 sqrt(k) m out at the golden angle; in
    // a full one the newcomer takes the place of the earliest still there.
    const j = g.people;
    g.people += 1;
    g.here.push(i);
    if (j >= shelter) goneAt[g.here[j - shelter]] = t;
    const k = j % shelter;
    if (g.spots[k] === undefined) {
      // The sunflower's places that fall on the building are passed over, so a large
      // crowd spreads along the outside of the wall and not back through it.
      let place = null;
      for (let tries = 0; tries < 4000; tries += 1) {
        const m = g.next;
        g.next += 1;
        const r = 0.45 * Math.sqrt(m + 0.5), a = m * 2.399963;
        place = g.centre.map((x, c) => x + r * (Math.cos(a) * g.dir[c] + Math.sin(a) * g.side[c]));
        if (!indoors(place)) break;
      }
      extra.push(place);
      g.spots[k] = n + extra.length - 1;
    }
    return g.spots[k];
  };

  // Each door node's capacity: its own width where `doorWidths` gives one.
  const capacity = doorCapacity(doorWidth);
  const capOf = (u) => (doorWidths && doorWidths[u] > 0 ? doorCapacity(doorWidths[u]) : capacity);
  // The earliest time each door may be stepped on to next, one 1/capacity after the last.
  const doorFree = new Float64Array(n);
  // With passageFlow: the cross-sections of narrow floor and of stairs with a width, each
  // with its capacity and the earliest time it may be stepped on to next; kept on the graph.
  let passages = null;
  if (passageFlow) {
    const kept = graph._crowdPassages;
    const exitKey = exits.join(",");
    passages = kept && kept.n === n && kept.blocked === blocked && kept.passageWidths === passageWidths && kept.widths === graph.widths && kept.exitKey === exitKey
      ? kept.value
      : (graph._crowdPassages = { n, blocked, passageWidths, widths: graph.widths, exitKey, value: passageSections(graph, { exits, blocked, passageWidths }) }).value;
  }
  const section = passages ? passages.section : null;
  const sectionFree = passages ? new Float64Array(passages.capacity.length) : null;
  // How many a node holds, where any holds more than one.
  const abreast = passages && passages.stairs > 0 ? passages.slots : null;
  // When each node was last left (its holder's step away began), and each
  // person's own clock: when they reached the node they stand on. A step begins
  // at the later of the two (and of the door's next opening), not at the next
  // time step, so speeds keep to the person's whatever the link's direction.
  const leftAt = new Float64Array(n).fill(-Infinity);
  const clock = Float64Array.from(ready);
  // The time of each person's last look that found no way on (-Infinity once they step).
  const tried = new Float64Array(people).fill(-Infinity);
  const segs = Array.from({ length: people }, () => []);
  const exitOf = new Int32Array(people).fill(-1);
  const outAt = new Float32Array(people).fill(Infinity);
  let longestWait = 0;
  let left = people;

  // The walk from each node to the nearest door or exit, for the queue before it (DOOR_QUEUE_M).
  const toDoor = new Float64Array(n).fill(Infinity);
  {
    const heap = new Heap();
    for (let v = 0; v < n; v += 1) {
      // With passageFlow a passage's mouth and a stair with a width are fed as a door is, so their capacity governs.
      if (kinds[v] === KIND_DOOR || isExit[v] || (passages !== null && passages.mouth[v])) {
        toDoor[v] = 0;
        heap.push(0, v);
      }
    }
    const done = new Uint8Array(n);
    while (heap.size) {
      const a = heap.pop();
      if (done[a]) continue;
      done[a] = 1;
      for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
        if (blocked?.[edgeIds[k]]) continue;
        const b = neighbours[k], d = toDoor[a] + length[edgeIds[k]];
        if (d < toDoor[b] && d <= DOOR_QUEUE_M) {
          toDoor[b] = d;
          heap.push(d, b);
        }
      }
    }
  }
  // The nodes within DENSITY_RADIUS of each (in space, so a storey above is not
  // near), as lists; kept on the graph, as route.js keeps its csr, for the next run.
  const reading = { ...DENSITY_READING_DEFAULT, ...(densityReading === "calibrated" ? DENSITY_READING_CALIBRATED : densityReading || {}) };
  const readOwn = reading.centre === "own", readStanding = reading.area === "standing";
  const near = graph._crowdNear?.radius === reading.radius && graph._crowdNear.n === n ? graph._crowdNear : (graph._crowdNear = (() => {
    const R = reading.radius, P = graph.nodes;
    const cellOf = (v) => [0, 1, 2].map((c) => Math.floor(P[v * 3 + c] / R));
    const buckets = new Map();
    for (let v = 0; v < n; v += 1) {
      const key = cellOf(v).join(",");
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(v);
    }
    const start = new Int32Array(n + 1);
    const list = [];
    for (let v = 0; v < n; v += 1) {
      start[v] = list.length;
      const [a, b, c] = cellOf(v);
      for (let da = -1; da <= 1; da += 1) for (let db = -1; db <= 1; db += 1) for (let dc = -1; dc <= 1; dc += 1) {
        for (const w of buckets.get(`${a + da},${b + db},${c + dc}`) || []) {
          const d2 = (P[w * 3] - P[v * 3]) ** 2 + (P[w * 3 + 1] - P[v * 3 + 1]) ** 2 + (P[w * 3 + 2] - P[v * 3 + 2]) ** 2;
          if (d2 <= R * R + 1e-9) list.push(w);
        }
      }
    }
    start[n] = list.length;
    return { radius: R, n, start, list: Int32Array.from(list) };
  })());

  // The quickest choice: a walk tree to each exit, and the one each person
  // heads for (goal, an index into them; -1: the way out worked out with the
  // fire, as everyone's with the nearest choice).
  const exitList = [...new Set(exits.filter((e) => e >= 0))];
  // Whether people weigh the exits; an exit given to someone needs the walk trees too.
  const weighing = choice === "quickest" && exitList.length > 1 && exitList.length * n <= QUICKEST_CELLS;
  const trees = weighing || assigned.some((e) => e >= 0) ? exitList.map((e) => walkTree(graph, [e], { speed: 1, blocked })) : null;
  const goal = new Int32Array(people).fill(-1);
  // How many given an exit were let go of it when the fire cut their walk there.
  let released = 0;
  for (let i = 0; i < people; i += 1) if (assigned[i] >= 0) goal[i] = exitList.indexOf(assigned[i]);
  // How many times each turned to another exit after the start.
  const switches = new Int32Array(people);
  // Each person's metres to their exit, and the next node on the way.
  const distTo = (i) => (goal[i] >= 0 ? trees[goal[i]].time : dist);
  const nextTo = (i) => (goal[i] >= 0 ? trees[goal[i]].next : next);
  // Who heads for each exit, by metres from it: a Fenwick tree, a slot every
  // 0.1 m (a slot a metre wide left out a dense queue's band round each person).
  const slot = (d) => Math.floor(d * 10);
  let slots = 2;
  if (trees) for (const { time } of trees) for (let v = 0; v < n; v += 1) if (time[v] < Infinity) slots = Math.max(slots, slot(time[v]) + 2);
  const heading = trees ? trees.map(() => new Int32Array(slots + 1)) : [];
  const tally = (g, d, k) => {
    for (let x = Math.min(slots, slot(d) + 1); x <= slots; x += x & -x) heading[g][x] += k;
  };
  // How many heading for exit g are nearer it than d (by a slot or more).
  const ahead = (g, d) => {
    let s = 0;
    for (let x = Math.min(slots, slot(d)); x > 0; x -= x & -x) s += heading[g][x];
    return s;
  };
  // The flow seen out through each exit, persons/s: its door's capacity at
  // first, then an exponential average over FLOW_SECONDS of those who passed
  // while someone heading for it was held up, so an exit a stair feeds slower
  // than its door passes looks as slow as it is. With no one held up for it,
  // what passes says how many want it, not how many it takes: its flow goes
  // back towards the door's capacity.
  const exitCap = Float64Array.from(exitList, (e) => capOf(e));
  const flow = Float64Array.from(exitCap);
  const passed = new Int32Array(exitList.length);
  const held = new Uint8Array(exitList.length);
  const exitIndex = new Map(exitList.map((e, g) => [e, g]));
  let lastWeigh = 0;
  // Seconds till person i at v is out by exit g: the walk there, or those ahead through it at its flow, whichever is longer.
  const outBy = (i, g, v) => {
    const before = ahead(g, trees[g].time[v]);
    return Math.max(trees[g].time[v] / vmax[i], before > 0 ? before / Math.max(flow[g], FLOW_FLOOR * exitCap[g]) : 0);
  };
  // Whether the walk from v to exit g is clear of the fire at t.
  const clear = (g, v, t) => {
    if (closes === null) return true;
    for (let u = v; u >= 0; u = trees[g].next[u]) if (closedAt(u, t)) return false;
    return true;
  };
  // Everyone still inside weighs the exits again, nearer first, each counted
  // for the exit they take before the next weighs theirs. At the start each
  // takes the quickest clear one; later they keep theirs unless another saves
  // SWITCH_GAIN or the fire cuts it, and once the fire has cut every one they
  // keep to the way out worked out with it.
  const weigh = (t) => {
    if (t > lastWeigh) {
      const span = t - lastWeigh, a = 1 - Math.exp(-span / FLOW_SECONDS);
      for (let g = 0; g < flow.length; g += 1) flow[g] += ((held[g] ? passed[g] / span : exitCap[g]) - flow[g]) * a;
      passed.fill(0);
      held.fill(0);
      lastWeigh = t;
    }
    for (const h of heading) h.fill(0);
    for (const i of order) if (goal[i] >= 0) tally(goal[i], trees[goal[i]].time[at[i]], 1);
    for (const i of order) {
      const v = at[i], own = goal[i];
      // Someone given an exit keeps it until the fire cuts their walk to it; then they
      // are let go, and choose as everyone else does. Without the quickest choice no one weighs.
      if (assigned[i] >= 0) {
        if (closes === null || clear(own, v, t)) continue;
        assigned[i] = -1;
        released += 1;
        if (!weighing) {
          tally(own, trees[own].time[v], -1);
          goal[i] = -1;
          continue;
        }
      } else if (!weighing) continue;
      if (own < 0 && t > 0) continue;
      let keep = Infinity;
      if (own >= 0) {
        tally(own, trees[own].time[v], -1);
        if (clear(own, v, t)) keep = outBy(i, own, v);
      }
      let pick = keep < Infinity ? own : -1;
      let bar = keep < Infinity ? Math.min(keep - SWITCH_GAIN, keep * (1 - SWITCH_SHARE)) : Infinity;
      for (let g = 0; g < trees.length; g += 1) {
        if (g === own || !(trees[g].time[v] < Infinity)) continue;
        // While the two ways go on together there is nothing to choose yet.
        if (keep < Infinity && trees[g].next[v] === trees[own].next[v]) continue;
        const out = outBy(i, g, v);
        if (out < bar && clear(g, v, t)) {
          pick = g;
          bar = out;
        }
      }
      if (t > 0 && pick !== own && pick >= 0) switches[i] += 1;
      goal[i] = pick;
      if (pick >= 0) tally(pick, trees[pick].time[v], 1);
    }
  };

  // Nearer the exits first, so a queue moves up from its front in one step.
  let order = Array.from({ length: people }, (_, i) => i);
  const sortOrder = () => {
    order = order.filter((i) => exitOf[i] < 0);
    order.sort((p, q) => distTo(p)[at[p]] - distTo(q)[at[q]]);
  };
  sortOrder();
  if (trees) {
    weigh(0);
    sortOrder();
  }
  // The stair nodes within stairGap of each stair node, by the walk, found when first needed.
  const nearStair = new Map();
  const stairNear = (u) => {
    if (nearStair.has(u)) return nearStair.get(u);
    const found = [];
    const seen = new Map([[u, 0]]);
    const queue = [u];
    while (queue.length) {
      const a = queue.shift();
      for (let k = indptr[a]; k < indptr[a + 1]; k += 1) {
        const b = neighbours[k], d = seen.get(a) + length[edgeIds[k]];
        if (d > stairGap || (seen.has(b) && seen.get(b) <= d) || kinds[b] !== KIND_STAIR) continue;
        seen.set(b, d);
        found.push(b);
        queue.push(b);
      }
    }
    nearStair.set(u, found);
    return found;
  };
  // A stair that carries a width (passageFlow) keeps its gap ahead only, and by density.
  const wide = (u) => abreast !== null && section[u] >= 0;
  // Whether a person at v may step on to u: no one holds it, and on a stair no one else stands
  // close by. D is the person's metres to their exit a node.
  const free = (u, v, D) => {
    if (load[u] >= (abreast ? abreast[u] : 1)) return false;
    if (stairGap > 0 && kinds[u] === KIND_STAIR) {
      let close = load[u];
      if (wide(u)) {
        // On a stair with a width: those on u and on the stair nodes within the gap of it that
        // are ahead of the walker (nearer the exit than v), fewer than bring that stretch to
        // QUEUE_DENSITY, the density of SFPE's greatest flow, so the stair can carry its capacity
        // and fills no further. Those behind or beside the walker are not counted: they cannot
        // clear before the walker moves. Counted, as at first, they held the foremost back for
        // good once the stair had filled while its foot was held up, and no one left it again.
        let area = passages.area[u];
        for (const w of stairNear(u)) {
          if (w === v || !(D[w] < D[v])) continue;
          close += load[w];
          area += passages.area[w] > 0 ? passages.area[w] : cell;
        }
        return close < Math.max(1, Math.floor(QUEUE_DENSITY * area + 1e-9));
      }
      // The others within the gap, the one stepping from v left out.
      for (const w of stairNear(u)) close += w === v ? load[w] - 1 : load[w];
      if (close >= 1) return false;
    }
    return true;
  };
  // The density round a node: the others within DENSITY_RADIUS of it (not the
  // one stepping from v), over the walkable area there.
  // The walkable area within DENSITY_RADIUS of each node, where some nodes stand for more than their cell (a stair's width).
  const areaNear = abreast ? new Float64Array(n).fill(-1) : null;
  const cellOf = (w) => (areaNear && passages.area[w] > 0 ? passages.area[w] : cell);
  const densityAt = (u, v) => {
    // The disc's centre: the node ahead, or the walker's own (densityReading).
    const c = readOwn ? v : u;
    let held = 0;
    const { start, list } = near;
    for (let k = start[c]; k < start[c + 1]; k += 1) {
      const w = list[k];
      held += w === v ? load[w] - 1 : load[w];
    }
    let area = (start[c + 1] - start[c]) * cell;
    if (areaNear) {
      if (areaNear[c] < 0) {
        let sum = 0;
        for (let k = start[c]; k < start[c + 1]; k += 1) sum += cellOf(list[k]);
        areaNear[c] = sum;
      }
      area = areaNear[c];
    }
    if (!readStanding) return held / area;
    // The floor the others could stand on: not the walker's cell, nor the free one ahead.
    area -= cellOf(u) + cellOf(v);
    return area > 1e-9 ? held / area : 0;
  };

  // Randomness in how people move (wiki/roadmap/crowd-motion-randomness-plan-2026-11-11.md,
  // stage 1), drawn from a stream of its own so that with it off a run is
  // exactly as it was. Speed: each person's factor on their unimpeded speed
  // follows Corbetta et al.'s Langevin equation, worked to the time asked.
  const wobble = random((seed ^ 0x9e3779b9) >>> 0);
  const gauss = () => {
    const a = Math.max(1e-12, wobble()), b = wobble();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  };
  const factor = new Float64Array(people).fill(1);
  const factorAt = new Float64Array(people);
  const wobbleFactor = (i, t) => {
    if (!fluctuate) return 1;
    let f = factor[i];
    for (let h = t - factorAt[i]; h > 1e-9; h -= WALK_FLUCTUATION.step) {
      const s = Math.min(WALK_FLUCTUATION.step, h);
      f += fluctuationDrift(f) * s + (WALK_FLUCTUATION.sigma / WALK_FLUCTUATION.up) * Math.sqrt(s) * gauss();
      f = Math.max(WALK_FLUCTUATION.floor, f);
    }
    factor[i] = f;
    factorAt[i] = t;
    return f;
  };
  // In smoke (`smoke`, `smokeSpeed`): the share of their speed kept at the extinction
  // coefficient where the person stands, K = 3 / visibility, in the frame the step begins in.
  if (typeof smokeSpeed === "string" && !SMOKE_SPEED[smokeSpeed]) throw new Error(`simulateCrowd: no smoke speed relation named ${smokeSpeed}`);
  const smokeRelation = smoke && smokeSpeed ? (typeof smokeSpeed === "string" ? SMOKE_SPEED[smokeSpeed] : smokeSpeed) : null;
  if (smokeRelation && !(smoke.K > 0 && smoke.frameSeconds > 0 && smoke.vis?.length === smoke.K * n)) throw new Error("simulateCrowd: smoke is {K, frameSeconds, vis} with K x n visibilities");
  const slowedBy = new Uint8Array(people);
  const speedFactor = (i, t) => {
    const f = wobbleFactor(i, t);
    if (!smokeRelation) return f;
    const frame = Math.min(smoke.K - 1, Math.max(0, Math.floor(t / smoke.frameSeconds)));
    const seen = smoke.vis[frame * n + at[i]];
    const share = seen < MAX_VISIBILITY ? smokeSpeedFactor(smokeRelation, SIGN_CONSTANT / Math.max(1e-6, seen)) : 1;
    if (share < 1) slowedBy[i] = 1;
    return f * share;
  };
  // Steps: with kS set, staying or each free open neighbour, weighed as the
  // floor field weighs a cell (Kirchner and Schadschneider 2002, Eq. 1, with
  // no dynamic field): exp(kS x the cells it brings nearer the exit).
  const spacing = graph.spacing || 0.5;
  const pickStep = (i, v, D, inFire, t) => {
    const options = [v], logw = [0];
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const w = neighbours[k];
      if (kinds[w] === KIND_LIFT || (closedAt(w, t) && !inFire) || !(D[w] < Infinity)) continue;
      if (!isExit[w] && !free(w, v, D)) continue;
      options.push(w);
      logw.push((-kS * (D[w] - D[v])) / spacing);
    }
    let top = -Infinity;
    for (const l of logw) if (l > top) top = l;
    let total = 0;
    const weights = logw.map((l) => {
      const e = Math.exp(l - top);
      total += e;
      return e;
    });
    let r = wobble() * total;
    for (let j = 0; j < options.length; j += 1) {
      r -= weights[j];
      if (r <= 0) return options[j];
    }
    return options[options.length - 1];
  };

  let t = 0;
  const every = Math.max(1, Math.round(replan / dt));
  for (let step = 0; left > 0 && t < horizon; step += 1, t = step * dt) {
    // With a fire, the ways out over the places still open, now and then.
    if (closes !== null && step > 0 && step % every === 0) ({ time: dist, next } = openTree(t));
    if (trees && step > 0 && step % every === 0) weigh(t);
    if (step % 10 === 0) sortOrder();
    for (const i of order) {
      if (exitOf[i] >= 0 || ready[i] > t + 1e-9) continue;
      const v = at[i];
      const D = distTo(i);
      // The next node on the tree, or the best free neighbour as near the
      // exit (any open one nearer, from a place the fire has closed).
      let u = nextTo(i)[v];
      // From an open place no one steps into the fire; from inside it, anywhere out.
      const inFire = closedAt(v, t);
      if (kS !== null) {
        // The floor field's step, or staying a step where they are.
        u = pickStep(i, v, D, inFire, t);
        if (u === v) {
          ready[i] = t + dt;
          clock[i] = t + dt;
          continue;
        }
      } else if (u >= 0 && closedAt(u, t) && !inFire) u = -1;
      if (kS === null && (u < 0 || (!isExit[u] && !free(u, v, D)))) {
        const limit = u >= 0 ? D[u] + 1.0 : Infinity;
        let best = -1;
        for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
          if (blocked?.[edgeIds[k]]) continue;
          const w = neighbours[k];
          if (!free(w, v, D) || (closedAt(w, t) && !inFire) || !(D[w] < D[v]) || D[w] > limit || kinds[w] === KIND_LIFT) continue;
          if (best < 0 || D[w] < D[best]) best = w;
        }
        u = best;
      }
      // The passage's cross-section stepped on to from outside it (passageFlow), and when it, or the door, opens.
      const gate = section !== null && u >= 0 && section[u] >= 0 && section[u] !== section[v] ? section[u] : -1;
      const opens = u < 0 ? -Infinity : Math.max(kinds[u] === KIND_DOOR || isExit[u] ? doorFree[u] : -Infinity, gate >= 0 ? sectionFree[gate] : -Infinity);
      const doorShut = opens > t + 1e-9;
      if (u < 0 || doorShut) {
        if (goal[i] >= 0) held[goal[i]] = 1;
        tried[i] = t;
        // Cut off by the fire: wait for the next look at the ways out.
        const cut = u < 0 && goal[i] < 0 && dist[v] >= FIRE_PENALTY;
        ready[i] = doorShut ? Math.max(t + dt, Math.ceil((opens - 1e-9) / dt) * dt) : cut ? (Math.floor(step / every) + 1) * every * dt : t + dt;
        continue;
      }
      // The step begins on the person's own clock: when they reached v, the
      // node ahead was left and the door opens, whichever is latest, and not
      // before a look that found no way on; never later than now.
      const door = kinds[u] === KIND_DOOR || isExit[u];
      // On a stair, also not before the others near the next node had left (stairGap).
      // A node that holds several abreast is not waited for: its capacity spaces the steps on to it.
      const single = !abreast || abreast[u] === 1;
      let gapClear = -Infinity;
      if (single && stairGap > 0 && kinds[u] === KIND_STAIR && !wide(u)) for (const w of stairNear(u)) if (w !== v) gapClear = Math.max(gapClear, leftAt[w]);
      const t0 = Math.max(clock[i], single ? leftAt[u] : -Infinity, tried[i], gapClear, opens);
      longestWait = Math.max(longestWait, t0 - clock[i]);
      tried[i] = -Infinity;
      let len = 0;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) if (neighbours[k] === u) len = length[edgeIds[k]];
      // Stairs at SFPE's kStair down, STAIR_UP of it up; the floor at k.
      const stair = kinds[u] === KIND_STAIR || kinds[v] === KIND_STAIR;
      const rise = graph.nodes[u * 3 + up] - graph.nodes[v * 3 + up];
      const k0 = stair ? SFPE.kStair * (rise > 0.02 ? STAIR_UP : 1) : SFPE.k;
      const own = vmax[i] * (k0 / SFPE.k) * speedFactor(i, t0);
      // Slowed by the others round the node ahead above SFPE's threshold; in
      // the queue before a door at most as by its greatest flow's density.
      const around = densityAt(u, v);
      const crowd = toDoor[u] <= DOOR_QUEUE_M ? Math.min(around, QUEUE_DENSITY) : around;
      // The least speed of a step never lifts a walker above their own (someone slow, in smoke).
      const vel = Math.max(Math.min(reading.floor, own), crowd < FREE_DENSITY ? own : Math.min(own, k0 * (1 - SFPE.a * crowd)));
      const arrive = t0 + len / vel;
      segs[i].push(t0, arrive, v, u);
      if (door) doorFree[u] = Math.max(doorFree[u], t0) + 1 / capOf(u);
      // Up a stair the capacity is STAIR_UP of it, as the speed is.
      if (gate >= 0) sectionFree[gate] = Math.max(sectionFree[gate], t0) + 1 / (passages.capacity[gate] * (kinds[u] === KIND_STAIR && rise > 0.02 ? STAIR_UP : 1));
      leftAt[v] = t0;
      load[v] -= 1;
      if (isExit[u]) {
        // Out: on to the gathering place, and the exit is free at once.
        const g = outward(u);
        const p = spot(g, i, arrive);
        const q = extra[p - n], e = pointOf(u);
        const walk = Math.hypot(q[0] - e[0], q[1] - e[1], q[2] - e[2]);
        segs[i].push(arrive, arrive + walk / vmax[i], u, p);
        exitOf[i] = u;
        outAt[i] = arrive;
        if (trees) passed[exitIndex.get(u)] += 1;
        g.last = Math.max(g.last, arrive);
        left -= 1;
        continue;
      }
      load[u] += 1;
      at[i] = u;
      ready[i] = arrive;
      clock[i] = arrive;
    }
  }

  // Anyone who never moved (boxed in till the horizon) stands where they are.
  for (let i = 0; i < people; i += 1) if (!segs[i].length) segs[i].push(t, t, at[i], at[i]);
  // Packed for the view.
  const first = new Int32Array(people), count = new Int32Array(people);
  let total = 0;
  for (let i = 0; i < people; i += 1) {
    first[i] = total;
    count[i] = segs[i].length / 4;
    total += count[i];
  }
  const segments = new Float32Array(total * 4);
  for (let i = 0; i < people; i += 1) segments.set(segs[i], first[i] * 4);
  const points = new Float32Array((n + extra.length) * 3);
  for (let v = 0; v < n; v += 1) points.set(pointOf(v), v * 3);
  extra.forEach((p, j) => points.set(p, (n + j) * 3));
  let lastOut = 0;
  for (let i = 0; i < people; i += 1) if (Number.isFinite(outAt[i])) lastOut = Math.max(lastOut, outAt[i]);
  // Still inside with no open way out at the end: cut off by the fire.
  let cutOff = 0;
  if (closes !== null) for (let i = 0; i < people; i += 1) if (exitOf[i] < 0 && dist[at[i]] >= FIRE_PENALTY) cutOff += 1;
  return {
    people,
    points,
    segments,
    first,
    count,
    exitOf,
    outAt,
    exits: [...gather.entries()].map(([node, g]) => ({ node, people: g.people, last: g.last, above: g.above })).sort((a, b) => b.people - a.people),
    unreached,
    trapped: Math.ceil((unreached * cell) / Math.max(0.1, occupantArea)),
    inside: left,
    lastOut,
    longestWait,
    capped: wanted > people,
    doorCapacity: capacity,
    goneAt,
    shelter,
    cutOff,
    choice: weighing ? "quickest" : "nearest",
    tooLarge: choice === "quickest" && exitList.length > 1 && !weighing,
    groupOf,
    short,
    released,
    switches,
    passages: passages ? { corridors: passages.corridors, stairs: passages.stairs } : null,
    startAt,
    smokeSpeed: smokeRelation ? (typeof smokeSpeed === "string" ? smokeSpeed : "custom") : null,
    slowed: slowedBy.reduce((s, x) => s + x, 0),
  };
}

/**
 * One person running down a stair on a loop, to show the stair: of the
 * crowd's walks, the one that goes down the most stair nodes, from a few
 * steps before its first stair node to a few after its last, at that walk's
 * own speeds with its waits left out, as a one-person crowd from time 0.
 * Null when no walk takes a stair down at least `drop` metres.
 *
 * @returns {null | {people: 1, points: Float32Array, segments: Float32Array, first: Int32Array, count: Int32Array, outAt: Float32Array, duration: number}}
 */
export function stairRunner(graph, crowd, { margin = 4, drop = 1.5 } = {}) {
  if (!crowd?.people) return null;
  const n = graph.nodes.length / 3;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const kinds = graph.kinds || new Uint8Array(n);
  let best = null;
  for (let i = 0; i < crowd.people; i += 1) {
    // The walk's moves between graph nodes (not the last, to the gathering place).
    const moves = [];
    for (let j = 0; j < crowd.count[i]; j += 1) {
      const q = (crowd.first[i] + j) * 4;
      const a = crowd.segments[q + 2], b = crowd.segments[q + 3];
      if (a < n && b < n && a !== b) moves.push({ a, b, time: crowd.segments[q + 1] - crowd.segments[q] });
    }
    const onStair = moves.map((m) => kinds[m.a] === KIND_STAIR || kinds[m.b] === KIND_STAIR);
    const firstStair = onStair.indexOf(true), lastStair = onStair.lastIndexOf(true);
    if (firstStair < 0) continue;
    const from = Math.max(0, firstStair - margin), to = Math.min(moves.length - 1, lastStair + margin);
    const height = crowd.points[moves[from].a * 3 + up] - crowd.points[moves[to].b * 3 + up];
    const steps = onStair.slice(firstStair, lastStair + 1).filter(Boolean).length;
    if (height >= drop && (!best || steps > best.steps)) best = { steps, moves: moves.slice(from, to + 1) };
  }
  if (!best) return null;
  // Its own points, so the view uploads a few rather than the whole graph.
  const nodes = [best.moves[0].a, ...best.moves.map((m) => m.b)];
  const points = new Float32Array(nodes.length * 3);
  nodes.forEach((v, k) => points.set(crowd.points.subarray(v * 3, v * 3 + 3), k * 3));
  const segments = new Float32Array(best.moves.length * 4);
  let t = 0;
  best.moves.forEach((m, k) => {
    segments.set([t, t + m.time, k, k + 1], k * 4);
    t += m.time;
  });
  return { people: 1, points, segments, first: Int32Array.of(0), count: Int32Array.of(best.moves.length), outAt: Float32Array.of(Infinity), duration: t };
}

/**
 * One person on the forecast's way out (core/forecast.js planWithForecast:
 * its earliest safe route, a node list and the time each is reached, in
 * seconds from ignition), as a one-person crowd on that timeline: standing
 * at the start until the plan sets off, out at its arrival. Null without a
 * route.
 */
export function forecastRunner(graph, route) {
  const nodes = route?.nodes, times = route?.times;
  if (!nodes?.length || nodes.length < 2 || times?.length !== nodes.length) return null;
  const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
  const lift = graph.agentHeight || 0;
  const points = new Float32Array(nodes.length * 3);
  nodes.forEach((v, k) => {
    for (let c = 0; c < 3; c += 1) points[k * 3 + c] = graph.nodes[v * 3 + c] - (c === up ? lift : 0);
  });
  const segments = new Float32Array((nodes.length - 1) * 4);
  for (let k = 0; k + 1 < nodes.length; k += 1) segments.set([times[k], Math.max(times[k], times[k + 1]), k, k + 1], k * 4);
  const arrival = times[times.length - 1];
  return { people: 1, points, segments, first: Int32Array.of(0), count: Int32Array.of(nodes.length - 1), outAt: Float32Array.of(arrival), duration: arrival };
}
