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
  gridSnap: false,
  gridCellSize: 1.5,
  maxPoints: 40000,
  engine: "fast",

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

  const [fire, setFire] = useState({ running: false, step: 0, model: null });
  const [dynamicPath, setDynamicPath] = useState(null);
  const [rl, setRl] = useState({ busy: false, path: null, reachedExit: false });

  const [toasts, setToasts] = useState([]);
  const [cacheInfo, setCacheInfo] = useState({ count: 0, bytes: 0 });

  const fireHandle = useRef(null);
  const abortRef = useRef(null);

  // ------------------------------------------------------------- utilities

  const toast = useCallback((message, { title, variant = "info", ttl = 6000 } = {}) => {
    const id = ++toastSeq;
    setToasts((current) => [...current.slice(-3), { id, message, title, variant }]);
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
      // Log the full object; the toast only carries what a user can act on.
      console.error(title || "Error", error);
      toast(message, { title, variant: "error", ttl: 9000 });
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

  const refreshCapabilities = useCallback(async () => {
    try {
      const info = await api.capabilities();
      setCapabilities(info);
      setServerStatus("online");
      return info;
    } catch {
      setServerStatus("offline");
      setCapabilities(null);
      return null;
    }
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      await refreshCapabilities();
      const info = await cacheSize();
      if (alive) setCacheInfo(info);
    })();
    return () => {
      alive = false;
    };
  }, [refreshCapabilities]);

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
    setDynamicPath(null);
    setRl({ busy: false, path: null, reachedExit: false });
    setFire({ running: false, step: 0, model: null });
    fireHandle.current?.close();
    fireHandle.current = null;
  }, []);

  const beginLoad = useCallback(
    (nextFile) => {
      setFile(nextFile);
      setModelInfo(null);
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
      fire,
      dynamicPath,
      rl,
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
      setFire,
      setDynamicPath,
      setRl,
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
      fire,
      dynamicPath,
      rl,
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
