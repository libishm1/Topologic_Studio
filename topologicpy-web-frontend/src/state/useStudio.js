/**
 * Application state and actions.
 *
 * One hook owns the workflow so the components stay presentational. State that
 * belongs to the 3D scene (camera, overlays, materials) deliberately lives in
 * ViewerManager instead - putting it here is what made the Classic viewer
 * rebuild itself whenever an unrelated checkbox changed.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, api, isGraphGone, openFireStream } from "../lib/api.js";
import { cacheSize, clearCache } from "../lib/fragmentCache.js";
import { mark } from "../lib/perf.js";
import { migrateSettings, withoutSessionOnly } from "../core/settingsMigration.js";
import { STALE_SITE_MESSAGE, effectiveEngine as engineFor, isStaleSite } from "../core/session.js";

const THEME_KEY = "ts.theme";
const SETTINGS_KEY = "ts.settings";

export const DEFAULT_SETTINGS = {
  upAxis: "y",
  agentHeight: 0.75,
  floorSpacing: 0.5,
  maxEdgeFloor: 2.25,
  maxEdgeStair: 0.4,
  // Keeps the floor mesh flat: a floor link may not span more height than
  // this, and a vertical column of samples collapses onto one surface.
  maxEdgeRise: 0.35,
  columnGap: 0.9,
  maxDegree: 12,
  useWalls: true,
  rectilinear: false,
  // Lifts and escalators join floors only when on: a fire evacuation does not
  // use them unless they are evacuation lifts (src/core/cores.js).
  useLifts: false,
  straightenRoutes: true,
  gridSnap: false,
  gridCellSize: 1.5,
  maxPoints: 40000,
  engine: "fast",
  altCount: 3,
  altMethod: "distinct",
  navSpacing: 0.5,
  showNavGraph: true,

  fireModel: "temperature",
  fireMaxSteps: 60,
  fireDelayMs: 120,
  reroute: true,
  // The rerouted route goes to "any" exit out of the building, or only the "picked" one.
  rerouteExits: "any",
  rerouteInterval: 5,
  hazardAlpha: 0.5,
  lethalityThreshold: null,

  // "shielded" (rlSafe.js: proved never to walk into the fire, certified
  // earliest escape) or "qlearning" (rl.js, after Jabi et al. 2019).
  rlMethod: "shielded",
  // The egress code check (core/codeCheck.js): whether the building is sprinklered.
  codeSprinklered: false,
  // The performance-based check (core/performanceCheck.js): the occupancy
  // preset (core/rsetPresets.js) and its times, seconds; detection from the
  // Flamescapes run's smoke when it has one, else the time set; and the
  // criterion (RSET_CRITERIA).
  perfOccupancy: "office",
  perfDetectionFromRun: true,
  perfDetection: 60,
  perfAlarm: 30,
  perfPre1: 60,
  perfPre99: 180,
  perfCriterion: "base",
  // The share who never respond (per cent), the area per occupant (m²) for
  // the queues at doors, the measured pre-movement shown beside the design
  // (MEASURED_PRE), whether doors queue, and whether planning also maps the
  // deadlines with the exit nearest the fire blocked (a second map: off).
  perfNoResponse: 0,
  perfOccupantArea: 9,
  perfMeasured: "business-1",
  perfQueues: true,
  perfBlockedExit: false,
  // The shielded learner escapes by "any" exit door or the "chosen" one.
  rlExits: "any",
  rlEpisodes: 400,
  rlMaxSteps: 200,
  rlUseFire: true,
  rlFireSpeed: 1,

  // Planning with the fire forecast (src/core/forecast.js). The fire models
  // count steps, so a step's length in seconds is the user's modelling choice.
  forecastSecondsPerStep: 5,
  walkSpeed: 1.2,
  untenableAbove: 80,
  doseLimit: 0.3,
  departAfter: 0,
  // "model": every exit door the IFC marks (or, failing that, doors with floor
  // on one side); "picked": the exit picked in the Route step.
  forecastExits: "model",
  showDeadlineMap: true,
  // People leaving (core/crowd.js): off until switched on; drawn as
  // "figures" or "dots", played at crowdRate times real time.
  crowd: false,
  crowdStyle: "figures",
  crowdRate: 10,
  // One figure running down a stair on a loop, when a walk takes one; or,
  // with crowdRunner "forecast" and a plan made, the forecast's evacuee on
  // its earliest safe route and timeline.
  crowdStairRunner: true,
  crowdRunner: "stair",
  // People keep the fire's playback time (Simulate) instead of their own.
  crowdWithFire: false,
  // Each person takes the exit that gets them out soonest, queues counted
  // ("quickest"), or the nearest ("nearest"; core/crowd.js EXIT_CHOICES).
  crowdChoice: "quickest",
  // People leave by every exit out of the building ("all"), or only by the
  // exit picked in Route ("picked"); not the forecast plan's Exits setting.
  crowdExits: "all",
  // Randomness in how people move (core/crowd.js; wiki/roadmap/crowd-motion-
  // randomness-plan-2026-11-11.md): each person's speed wobbles in time, and
  // each step is drawn by the floor field ("exact", or STEP_RANDOMNESS's
  // "direct", "wandering", "disoriented"); with either on, crowdRuns seeds.
  crowdFluctuate: false,
  crowdSteps: "exact",
  crowdRuns: 5,
  // The slime mould (core/slimeEgress.js) grows as if there were no fire
  // ("none"), held back by the fire and its smoke on the fire's clock
  // ("soft"), or walled off from every place the fire makes untenable
  // ("hard"); its skeletons (core/skeleton.js) drawn when asked.
  slimeFire: "none",
  slimeDesire: false,
  slimeMedial: false,
  // Readings from the slime's run (the desire-lines plan's gaps 1 and 2):
  // bottlenecks on its desire lines, dead ends (one way out longer than the
  // code's limit) and the branches it explored and abandoned.
  slimeBottlenecks: false,
  slimeDeadEnds: false,
  slimeAbandoned: false,
  // Gathering places hold crowdShelterArea m^2 of people when on; once full,
  // the earliest there fade as newcomers arrive.
  crowdShelterOn: false,
  crowdShelterArea: 25,
  // The two-zone room model (fireModel "zone"; src/core/zone.js), on a clock
  // in seconds with gas temperatures in degrees C: a t-squared fire of this
  // growth up to this peak (and the room's ventilation limit), untenable from
  // this gas temperature, or as soon as the smoke layer is at head height.
  zoneGrowth: "medium",
  zonePeakKw: 5000,
  zoneUntenableC: 60,
  zoneSmoke: true,
  zoneHorizon: 600,

  // The optional Flamescapes solver (src/core/flamescapes.js): off until
  // switched on, then two more fire models, "flamescapes" (fire and smoke)
  // and "flamescapes-embers" (with embers).
  flamescapes: false,
  flamescapesClass: "A", // fire class of the first item (src/core/fireClasses.js)
  // How the first item grows (fireClasses.js designFireFor): "class", its fire
  // class's t-squared growth (medium for A and C); a growth class; or
  // "solver", the solver burning it itself as before 2026-11-08 (about 500 kW
  // within 15 s, faster than any design fire).
  flamescapesGrowth: "class",
  flamescapesHorizon: 180, // s
  flamescapesFireLoad: 780, // MJ / m^2 of floor, the contents
  flamescapesUntenableC: 60, // gas at head height
  flamescapesSmoke: true, // visibility below the limit is untenable
  flamescapesVisibility: 10, // m
  flamescapesBackend: "auto", // "auto" (WebGPU where offered) or "cpu"
  // The fire's grid: "standard", the same budget on every device so a GPU
  // and a CPU give the same fire (lib/hardware.js STANDARD_CELLS; a phone's
  // CPU keeps a smaller one); or, on the GPU, "speed" (the finest level its
  // measured speed steps at three times real time) or "finest" (the finest
  // its memory allows). The results depend on the grid, so comparisons should
  // keep one. Named flamescapesGrid before 2026-11-09, defaulting to "speed".
  flamescapesGridPolicy: "standard",
  // Runs of each fire on the GPU, the first watched and the rest after it with
  // the room a thousandth of a degree warmer each, for the range on the card;
  // 1 for one run. Not on the CPU, where each takes as long as the first.
  flamescapesRepeats: 1,
  // Openings for the smoke: doors "open" or "closed"; windows "closed" or "open".
  flamescapesDoors: "open",
  // With random doors (core/doorStates.js): the share closed and the seed that picks them.
  flamescapesDoorShare: 0.5,
  flamescapesDoorSeed: 1,
  flamescapesWindows: "closed",
  // Ventilation through the model's air terminals: "off" (fans stopped and
  // dampers shut, as a fire alarm does) or "running" (the ducts mix the air
  // of every terminal), and the air each terminal moves (L/s, an assumption).
  flamescapesVentilation: "off",
  flamescapesTerminalFlow: 100,
  // Beams, and a second model's ducts and cable trays, as solids in the fire
  // grid (ViewerManager.overheadMeshes); off, the ceilings are as the walls
  // and slabs make them.
  flamescapesOverhead: false,
  // Sprinklers (core/sprinklers.js, core/suppression.js), off by default and
  // costing nothing off. The heads come from the IFC ("ifc"), are placed per
  // room by NFPA 13 ("placed"), or the IFC's when it has any ("auto"); the
  // rooms' hazard by name ("auto") or one for all; quick or standard
  // response. A run opens them on the "estimate" (Alpert's ceiling jet under
  // the fire's ceiling) or the "late" bound (the grid), and they only open
  // ("activation"), cap the fuel ("evans", Evans 1993) or cap it and spray
  // ("spray", Fire-X's evaporation on the grid).
  // Portable extinguishers (core/extinguishers.js), off by default and costing
  // nothing off: read from the IFC by name ("auto": the IFC's, else placed on
  // the walk by the 22.86 m rule), checked against SBC 801 906 and NFPA 10;
  // after a fire, one responder's attempt with the chosen unit, setting off
  // when they see the fire or after the alarm and the design pre-movement.
  extinguishers: false,
  extinguisherSource: "auto",
  extinguisherHazard: "light",
  extinguisherUnit: "abc-2.3",
  extinguisherStart: "seen",
  extinguisherTrained: false,
  sprinklers: false,
  sprinklerSource: "auto",
  sprinklerHazard: "auto",
  sprinklerResponse: "quick",
  // "wetting" by default: across 12 test fires on 4 models it cut most fires by
  // 20 to 79 % (tools/sprinkler-check.mjs); Evans's cap and the spray ranged
  // from 40 % less to 78 % more (2026-11-07, before the activation fix of
  // 2026-11-08: wiki/research/sprinkler-activation-fix-2026-11-08.md).
  sprinklerMode: "wetting",
  sprinklerOpenOn: "estimate",
  // Suppression systems by room use (core/suppressionSystems.js), off by
  // default and costing nothing off: clean agent (HFC-227ea) in IT, telecom,
  // electrical and battery rooms, foam in generator and fuel rooms, water
  // elsewhere, apartments water only, read from room names and the
  // performance check's occupancy. Agent only drops those rooms' sprinklers.
  suppressionSystems: false,
  systemsAgentOnly: false,
  // The graph's colour while a run plays (core/measures.js): "heat", "smoke",
  // "gases" or "tenability", and which smoke or gas measure.
  flamescapesShow: "heat",
  flamescapesSmokeMeasure: "visibility",
  flamescapesGasMeasure: "co",
  // Record soot, CO2 and O2 too (core/tenability.js); off, or past the
  // memory budget, a run keeps temperature, visibility and CO only.
  flamescapesMeasures: true,
  // Keep finished Flamescapes runs in the browser and replay them (core/runCache.js).
  flamescapesKeepRuns: true,

  showGraph: true,
  modelAppearance: "solid",
};

/**
 * The settings a forecast plan is made with. A plan keeps the values it used,
 * so its card describes that plan rather than the fields as they read now,
 * and says so when they differ (review M2).
 */
export const PLAN_SETTINGS = [
  "fireModel",
  "fireMaxSteps",
  "forecastSecondsPerStep",
  "untenableAbove",
  "walkSpeed",
  "departAfter",
  "doseLimit",
  "useWalls",
  "forecastExits",
  "zoneGrowth",
  "zonePeakKw",
  "zoneUntenableC",
  "zoneSmoke",
  "zoneHorizon",
  "flamescapesClass",
  "flamescapesGrowth",
  "flamescapesHorizon",
  "flamescapesFireLoad",
  "flamescapesUntenableC",
  "flamescapesSmoke",
  "flamescapesVisibility",
  "flamescapesDoors",
  "flamescapesDoorShare",
  "flamescapesDoorSeed",
  "flamescapesWindows",
  "flamescapesVentilation",
  "flamescapesTerminalFlow",
  "flamescapesGridPolicy",
];

export const planSettings = (settings) => Object.fromEntries(PLAN_SETTINGS.map((key) => [key, settings[key]]));

/** Whether the loaded model has IfcSpace rooms, which the two-zone fire model needs. */
export function hasRooms(modelInfo) {
  return (modelInfo?.stats?.spaces ?? 0) > 0;
}

/** Whether a fire model is one of the Flamescapes solver's two. */
export const isFlamescapes = (model) => model === "flamescapes" || model === "flamescapes-embers";

/** The fire model in use: a saved choice of the two-zone model falls back to
 *  heat diffusion on a model without rooms, and of a Flamescapes model when
 *  the solver is switched off. */
export function fireModelIn(settings, modelInfo) {
  if (settings.fireModel === "zone" && !hasRooms(modelInfo)) return "temperature";
  if (isFlamescapes(settings.fireModel) && !settings.flamescapes) return "temperature";
  return settings.fireModel;
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...withoutSessionOnly(migrateSettings(JSON.parse(raw))) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function initialTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* private mode */
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

let toastSeq = 0;

export function useStudio() {
  const [theme, setTheme] = useState(initialTheme);
  const [settings, setSettingsState] = useState(loadSettings);
  const [capabilities, setCapabilities] = useState(null);
  const [serverStatus, setServerStatus] = useState("checking");

  const [file, setFile] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loadState, setLoadState] = useState({ busy: false, stage: null, detail: null });

  const [graph, setGraph] = useState(null);
  const [graphBusy, setGraphBusy] = useState(false);

  const [pickMode, setPickMode] = useState(null);
  const [points, setPoints] = useState({ start: null, exit: null, fire: null });

  const [path, setPath] = useState(null);
  const [pathBusy, setPathBusy] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [alternatives, setAlternatives] = useState(null);
  const [navCheck, setNavCheck] = useState(null);
  const [navBusy, setNavBusy] = useState(false);

  const [fire, setFire] = useState({ running: false, step: 0, model: null });
  const [dynamicPath, setDynamicPath] = useState(null);
  const [rl, setRl] = useState({ busy: false, path: null, reachedExit: false, result: null });
  const [forecastPlan, setForecastPlan] = useState({ busy: false, result: null });
  // What the IFC says about egress beyond its geometry (core/ifcSemantics.js).
  const [semantics, setSemantics] = useState(null);

  const [toasts, setToasts] = useState([]);
  const [cacheInfo, setCacheInfo] = useState({ count: 0, bytes: 0 });

  const fireHandle = useRef(null);
  const abortRef = useRef(null);

  // ------------------------------------------------------------- utilities

  const toast = useCallback((message, { title, variant = "info", ttl = 6000 } = {}) => {
    const id = ++toastSeq;
    // The same message again replaces the old one rather than stacking a
    // copy: three tries in a row used to leave three identical toasts.
    setToasts((current) => [
      ...current.filter((t) => t.message !== message || t.title !== title).slice(-3),
      { id, message, title, variant },
    ]);
    if (ttl) {
      setTimeout(() => {
        setToasts((current) => current.filter((t) => t.id !== id));
      }, ttl);
    }
    return id;
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  const reportError = useCallback(
    (error, title) => {
      // A part of the site that a newer deploy replaced: say to reload (SM-6).
      const message = isStaleSite(error)
        ? STALE_SITE_MESSAGE
        : error instanceof ApiError
          ? error.message
          : error?.message || "Something went wrong.";

      // A server that has gone away must flip the indicator immediately.
      // Previously the status was only sampled at mount, so the status bar
      // could read "backend up" while every request was failing.
      if (error instanceof ApiError && error.kind === "network") {
        setServerStatus("offline");
      }

      // The graph store is in-memory, so restarting the backend orphans the
      // graph id the client is holding. Say what to do about it.
      const graphGone = isGraphGone(error);

      // Log the full object; the toast only carries what a user can act on.
      console.error(title || "Error", error);
      toast(
        graphGone
          ? "The server no longer has this graph - it restarts empty. Build the egress graph again."
          : message,
        { title: graphGone ? "Graph no longer on the server" : title, variant: "error", ttl: 9000 },
      );
    },
    [toast],
  );

  const setSettings = useCallback((patch) => {
    setSettingsState((current) => {
      const next = typeof patch === "function" ? patch(current) : { ...current, ...patch };
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(withoutSessionOnly(next)));
      } catch {
        /* not fatal */
      }
      return next;
    });
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((current) => {
      const next = current === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch {
        /* not fatal */
      }
      document.documentElement.dataset.theme = next;
      return next;
    });
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // ------------------------------------------------------------ server ping

  const refreshCapabilities = useCallback(async (timeout) => {
    try {
      const info = await api.capabilities(undefined, timeout);
      setCapabilities(info);
      setServerStatus("online");
      return info;
    } catch {
      setServerStatus("offline");
      setCapabilities(null);
      return null;
    }
  }, []);

  // The first contact gets a long timeout. A free-tier Render service asleep
  // after 15 idle minutes took 33 s to answer (measured 2026-10-27); with the
  // old 8 s timeout the UI declared it offline while it was still waking, and
  // the user saw a red badge for a server that was fine. Render holds the
  // request until the instance is up, so one patient request is enough.
  useEffect(() => {
    let alive = true;
    const waking = setTimeout(() => {
      if (alive) setServerStatus((s) => (s === "checking" ? "waking" : s));
    }, 2500);
    (async () => {
      await refreshCapabilities(90000);
      clearTimeout(waking);
      const info = await cacheSize();
      if (alive) setCacheInfo(info);
    })();
    return () => {
      alive = false;
      clearTimeout(waking);
    };
  }, [refreshCapabilities]);

  // While the server is unreachable, keep checking so the indicator recovers
  // on its own, backing off 4 s, 8 s, 16 s ... to a minute, and not at all
  // while the tab is hidden. Every 4 s for good was 19 failed requests in one
  // session on a site with no server (GitHub Pages alone), and needless radio
  // wake-ups on a phone.
  useEffect(() => {
    if (serverStatus !== "offline") return undefined;
    let delay = 4000;
    let timer = null;
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      if (document.visibilityState === "visible") {
        const info = await refreshCapabilities();
        if (info || !alive) return;
        delay = Math.min(delay * 2, 60000);
      }
      timer = setTimeout(tick, delay);
    };
    timer = setTimeout(tick, delay);
    const onVisible = () => {
      if (document.visibilityState === "visible" && alive) {
        clearTimeout(timer);
        delay = 4000;
        timer = setTimeout(tick, 0);
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [serverStatus, refreshCapabilities]);

  // The engine the server can actually serve (core/session.js). Derived rather
  // than corrected in an effect: writing the preference back would fight the
  // user's own choice every time the backend restarted without topologicpy.
  const effectiveEngine = useMemo(() => engineFor(settings.engine, capabilities?.engines), [capabilities, settings.engine]);

  // --------------------------------------------------------------- workflow

  const resetDownstream = useCallback(() => {
    setGraph(null);
    setPath(null);
    setComparison(null);
    setAlternatives(null);
    setNavCheck(null);
    setDynamicPath(null);
    setRl({ busy: false, path: null, reachedExit: false, result: null });
    setForecastPlan({ busy: false, result: null });
    setFire({ running: false, step: 0, model: null });
    fireHandle.current?.close();
    fireHandle.current = null;
  }, []);

  const beginLoad = useCallback(
    (nextFile) => {
      setFile(nextFile);
      setModelInfo(null);
      setSemantics(null);
      setPoints({ start: null, exit: null, fire: null });
      setPickMode(null);
      resetDownstream();
    },
    [resetDownstream],
  );

  const stopFire = useCallback(() => {
    fireHandle.current?.close();
    fireHandle.current = null;
    setFire((current) => ({ ...current, running: false, starting: false }));
  }, []);

  const setFireHandle = useCallback((handle) => {
    fireHandle.current?.close();
    fireHandle.current = handle;
  }, []);

  useEffect(() => () => fireHandle.current?.close(), []);

  const purgeCache = useCallback(async () => {
    await clearCache();
    const info = await cacheSize();
    setCacheInfo(info);
    toast("Local fragment cache cleared.", { variant: "success", ttl: 4000 });
  }, [toast]);

  const refreshCacheInfo = useCallback(async () => {
    setCacheInfo(await cacheSize());
  }, []);

  const value = useMemo(
    () => ({
      // state
      theme,
      settings,
      capabilities,
      serverStatus,
      effectiveEngine,
      file,
      modelInfo,
      loadState,
      graph,
      graphBusy,
      pickMode,
      points,
      path,
      pathBusy,
      comparison,
      alternatives,
      navCheck,
      navBusy,
      fire,
      dynamicPath,
      rl,
      forecastPlan,
      semantics,
      toasts,
      cacheInfo,

      // setters used by the orchestration layer in App
      setModelInfo,
      setLoadState,
      setGraph,
      setGraphBusy,
      setPickMode,
      setPoints,
      setPath,
      setPathBusy,
      setComparison,
      setAlternatives,
      setNavCheck,
      setNavBusy,
      setFire,
      setDynamicPath,
      setRl,
      setForecastPlan,
      setSemantics,
      setCacheInfo,
      setFireHandle,
      abortRef,

      // actions
      setSettings,
      toggleTheme,
      toast,
      dismissToast,
      reportError,
      refreshCapabilities,
      beginLoad,
      resetDownstream,
      stopFire,
      purgeCache,
      refreshCacheInfo,
    }),
    [
      theme,
      settings,
      capabilities,
      serverStatus,
      effectiveEngine,
      file,
      modelInfo,
      loadState,
      graph,
      graphBusy,
      pickMode,
      points,
      path,
      pathBusy,
      comparison,
      alternatives,
      navCheck,
      navBusy,
      fire,
      dynamicPath,
      rl,
      forecastPlan,
      semantics,
      toasts,
      cacheInfo,
      setSettings,
      toggleTheme,
      toast,
      dismissToast,
      reportError,
      refreshCapabilities,
      beginLoad,
      resetDownstream,
      stopFire,
      purgeCache,
      refreshCacheInfo,
      setFireHandle,
    ],
  );

  return value;
}

export { openFireStream, api, mark };
