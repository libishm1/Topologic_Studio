/**
 * Application state and actions.
 *
 * One hook owns the workflow so the components stay presentational. State that
 * belongs to the 3D scene (camera, overlays, materials) deliberately lives in
 * ViewerManager instead - putting it here is what made the Classic viewer
 * rebuild itself whenever an unrelated checkbox changed.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, api, openFireStream } from "../lib/api.js";
import { cacheSize, clearCache } from "../lib/fragmentCache.js";
import { mark } from "../lib/perf.js";

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
  rerouteInterval: 5,
  hazardAlpha: 0.5,
  lethalityThreshold: null,

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

  showGraph: true,
  modelAppearance: "solid",
};

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
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
      const message =
        error instanceof ApiError
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
      const graphGone =
        error instanceof ApiError &&
        error.status === 400 &&
        /no (navigation )?graph/i.test(String(error.detail ?? error.message));

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
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
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

  // The engine the server can actually serve. Derived rather than corrected in
  // an effect: writing the preference back would fight the user's own choice
  // every time the backend restarted without topologicpy.
  const effectiveEngine = useMemo(() => {
    const engines = capabilities?.engines;
    if (!engines?.length) return settings.engine;
    return engines.includes(settings.engine) ? settings.engine : engines[0];
  }, [capabilities, settings.engine]);

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
    setFire((current) => ({ ...current, running: false }));
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
