/**
 * Application shell and workflow orchestration.
 *
 * The pipeline, end to end:
 *
 *   IFC file
 *     -> hash + fragment cache lookup        (lib/fragmentCache)
 *     -> convert or load fragments           (viewer/ViewerManager)
 *     -> category ids from the fragments     (viewer/categories)
 *     -> geometry extract + walkable sample  (workers/sampler.worker)
 *     -> point cloud upload                  (lib/api)
 *     -> navigation graph + routing          (backend)
 *
 * The expensive middle is off the main thread, and only the sampled points
 * cross the network - not the triangles.
 */
import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { api, legacyApi } from "./lib/api.js";
import { PhaseTimer, formatBytes, formatMs } from "./lib/perf.js";
import { isCancelled, workerClient } from "./lib/workerJob.js";
import { FORECAST_COLORS, NAV_COLOR, hexToInt, routeColor } from "./lib/routeColors.js";
import { graphIsSplit, graphStats } from "./core/graph.js";
import { findRoute, snap } from "./core/route.js";
import { earliestArrival } from "./core/forecast.js";
import { exitDoors } from "./core/ifcSemantics.js";
import { floorLevels } from "./core/lattice.js";
import { straightenRoute, withStraight } from "./core/straighten.js";
import { alternativeRoutes } from "./core/alternatives.js";
import { runFire } from "./core/fire.js";
import { RL_MAX_STEPS } from "./core/rl.js";
import { planSettings, useStudio } from "./state/useStudio.js";
import { GraphPanel, ModelPanel, RoutePanel, SimulatePanel } from "./ui/Panels.jsx";
import { SECTION_ABOVE_FLOOR } from "./viewer/sectionConstants.js";
import { Stepper } from "./ui/Stepper.jsx";
import { TopologyPanel } from "./ui/TopologyPanel.jsx";
import { Viewport } from "./ui/Viewport.jsx";

import { Badge, Button, FileButton, Icon, Segmented } from "./ui/primitives.jsx";
import { ErrorBoundary } from "./ui/ErrorBoundary.jsx";
import logo from "./assets/topologicStudio-white-logo400x400.png";

/** The points a route is drawn with: straightened when it has been. */
const drawn = (route) => route?.straight_points || route?.points || null;

/** Lowest and highest point of the egress geometry along the up axis. */
function heightRange(geometry, upAxis) {
  const up = upAxis === "z" ? 2 : upAxis === "x" ? 0 : 1;
  let low = Infinity, high = -Infinity;
  for (const group of [geometry.floors, geometry.walls, geometry.stairs]) {
    for (const mesh of group || []) {
      const p = mesh.positions;
      for (let i = up; p && i < p.length; i += 3) {
        if (p[i] < low) low = p[i];
        if (p[i] > high) high = p[i];
      }
    }
  }
  return [low, high];
}

// Three plus OrbitControls only matter in this mode, so the chunk is loaded
// on demand rather than shipped to everyone who opens the IFC viewer.
const TopologyViewer = lazy(() => import("./viewer/TopologyViewer.jsx"));
// The Methods page brings KaTeX (about 270 KB and its fonts); it loads only
// when someone opens the page. A failed load is kept by React.lazy, so a
// retry needs a new one.
const loadMethods = () => lazy(() => import("./ui/MethodsDialog.jsx").then((m) => ({ default: m.MethodsDialog })));

// The inspector is the workflow, one step per panel (src/ui/Panels.jsx).
// Methods (what was proved, the limits, the server) opens from the header.
const IFC_TABS = [
  { id: "model", label: "Model" },
  { id: "graph", label: "Graph" },
  { id: "route", label: "Route" },
  { id: "simulate", label: "Simulate" },
];

const TOPOLOGY_TABS = [{ id: "topology", label: "Topology" }];

const MODES = [
  { value: "ifc", label: "IFC", hint: "Fragment viewer, egress graph and fire simulation" },
  {
    value: "topology",
    label: "Topology JSON",
    hint: "TopologicPy JSON contract viewer (legacy workflow)",
  },
];

const NO_PLAN = { busy: false, result: null };

/**
 * The fields check what is typed, but settings saved before they did can be
 * out of range: a walking speed of 0 planned "too late" for the wrong
 * reason, and -1 failed with "Invalid array length" (review m1).
 */
function planSettingsProblem(s) {
  if (!(s.walkSpeed > 0)) return "Walking speed must be above 0 m/s.";
  if (!(s.forecastSecondsPerStep > 0)) return "Seconds per fire step must be above 0.";
  if (!(s.fireMaxSteps >= 1)) return "The fire needs at least one step (Steps, under Rerouting and timing).";
  if (!(s.doseLimit > 0)) return "The dose limit must be above 0.";
  if (!(s.departAfter >= 0)) return "Leave after cannot be negative.";
  return null;
}

/**
 * Which layer colours the graph: "deadlines", "fire", or null for the graph's
 * own colours. A small store rather than state, since the deadline effect
 * sets it and the legend reads it.
 */
function createColourOwner() {
  let owner = null;
  const listeners = new Set();
  return {
    get: () => owner,
    set: (next) => {
      if (next === owner) return;
      owner = next;
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

const DEFAULT_TOPOLOGY_OPTIONS = {
  showFaces: true,
  showVerts: false,
  wireframe: true,
  translucent: true,
  tiltMin: 0.3,
  maxZSpan: 1.0,
  minFloorArea: 9,
};

export default function App() {
  const studio = useStudio();
  const viewerRef = useRef(null);
  const samplerRef = useRef(null);
  const forecastRef = useRef(null);
  const semanticsRef = useRef(null);
  const rlRef = useRef(null);
  const geometryRef = useRef(null);
  // Counts file loads, so only the latest one updates the page.
  const loadSeqRef = useRef(0);
  // Fire accumulators live in refs: they are written from SSE callbacks that
  // outlive any single render, and must not be re-created when the callback
  // identity changes mid-simulation.
  const burningRef = useRef(new Set());
  const temperatureRef = useRef(new Map());
  const [tab, setTab] = useState("model");
  const [panelOpen, setPanelOpen] = useState(true);
  const [methodsOpen, setMethodsOpen] = useState(false);
  const [MethodsDialog, setMethodsDialog] = useState(loadMethods);
  // Storey heights for the section cut, and the chosen one (null = whole).
  const [levels, setLevels] = useState([]);
  const [section, setSection] = useState(null);
  const geometryPromiseRef = useRef(null);
  const closeMethods = useCallback(() => setMethodsOpen(false), []);
  // Its chunk failed to load (offline, or a deploy replaced it): say so, close
  // it, and load afresh at the next opening. Unhandled, this unmounted the app.
  const methodsFailed = useCallback(
    (error) => {
      console.warn("The Methods page could not load.", error);
      setMethodsOpen(false);
      setMethodsDialog(() => loadMethods());
      studio.toast("The Methods page could not load. Check the connection and open it again.", {
        title: "Methods",
        variant: "warning",
      });
    },
    [studio],
  );
  const [lastSample, setLastSample] = useState(null);

  // Legacy TopologicPy JSON-contract workflow.
  const [mode, setMode] = useState("ifc");
  const [topology, setTopology] = useState(null);
  const [topologySelection, setTopologySelection] = useState(null);
  const [topologyBusy, setTopologyBusy] = useState(false);
  const [topologySource, setTopologySource] = useState(null);
  const [topologyOptions, setTopologyOptionsState] = useState(DEFAULT_TOPOLOGY_OPTIONS);
  const [fitRequest, setFitRequest] = useState(0);

  const {
    settings,
    setSettings,
    theme,
    toggleTheme,
    file,
    modelInfo,
    graph,
    points,
    pickMode,
    fire,
    path,
    dynamicPath,
    alternatives,
    navCheck,
    rl,
    forecastPlan,
    toast,
    reportError,
  } = studio;

  // ------------------------------------------------------------ the workers

  // Each job is a promise that also settles when its worker fails to load or
  // dies (lib/workerJob.js); a failed worker is replaced at the next job.
  useEffect(
    () => () => {
      samplerRef.current?.cancel();
      forecastRef.current?.cancel();
      semanticsRef.current?.cancel();
      rlRef.current?.cancel();
    },
    [],
  );

  // Exit flags and roof slabs from the IFC itself, read by a second, small
  // web-ifc in a worker while the geometry converts (core/ifcSemantics.js).
  // One worker per file, ended when it answers or when another file is
  // opened: web-ifc's heap never shrinks, so a kept worker held it all session.
  const semanticsJob = useCallback((bytes) => {
    semanticsRef.current?.cancel();
    const job = workerClient(
      () => new Worker(new URL("./workers/semantics.worker.js", import.meta.url), { type: "module" }),
      { failure: "Could not read the model's exit flags." },
    );
    semanticsRef.current = job;
    // Against the page, as the viewer resolves it: a relative path would be
    // read against the worker's own URL, under assets/.
    const wasmPath = new URL(
      import.meta.env.VITE_WEBIFC_WASM_PATH || `${import.meta.env.BASE_URL}wasm/`,
      window.location.href,
    ).href;
    return job.run({ bytes, wasmPath }, [bytes]).finally(() => {
      job.cancel();
      if (semanticsRef.current === job) semanticsRef.current = null;
    });
  }, []);

  // Planning with the forecast has its own worker: its deadline map takes a
  // few seconds on a large graph and must not hold up a graph rebuild.
  const forecastJob = useCallback((payload) => {
    forecastRef.current ||= workerClient(
      () => new Worker(new URL("./workers/forecast.worker.js", import.meta.url), { type: "module" }),
      { failure: "Planning with the forecast stopped: its worker did not load or ran out of memory. Reload the page and try again." },
    );
    return forecastRef.current.run({ payload });
  }, []);

  const sample = useCallback((payload, transfers) => {
    samplerRef.current ||= workerClient(
      () => new Worker(new URL("./workers/sampler.worker.js", import.meta.url), { type: "module" }),
      { failure: "The graph builder stopped, usually because the browser ran out of memory. Reload the page and try again." },
    );
    return samplerRef.current.run({ payload }, transfers);
  }, []);

  // ------------------------------------------------------- forecast plan state

  // Counts plan requests and the changes that make a plan stale; a reply to
  // an older one is dropped (review M3).
  const planGenRef = useRef(0);

  // The layer colouring the graph, so that one layer does not wipe another's
  // colours and the legend shows the one drawn (review M6).
  const [colours] = useState(createColourOwner);
  const graphColours = useSyncExternalStore(colours.subscribe, colours.get);

  // The plan's deadline map, when it was made on the graph on screen.
  const deadlines = forecastPlan.result && forecastPlan.nodes === graph?.nodes ? forecastPlan.result.deadlines : null;
  const planHorizon = forecastPlan.result?.horizon ?? 300;

  // Whether the alternative-route overlays hold the plan's routes rather than
  // the alternatives: a ref for the handlers, state for the legend.
  const planRoutesRef = useRef(false);
  const [planRoutesShown, setPlanRoutesShown] = useState(false);
  const showPlanRoutes = useCallback((shown) => {
    planRoutesRef.current = shown;
    setPlanRoutesShown(shown);
  }, []);

  /**
   * A plan holds for the model, graph, start, exit and fire it was made with.
   * When any of them changes the plan is dropped in one step: its card, the
   * routes it drew, its deadline colours (they follow the state) and any plan
   * still running (review M1, M3). A map left on a rebuilt graph, indexed by
   * the new graph's nodes, read past its end and took the page down (C1).
   */
  const dropPlan = useCallback(() => {
    planGenRef.current += 1;
    if (forecastRef.current?.busy) forecastRef.current.cancel();
    studio.setForecastPlan(NO_PLAN);
    if (planRoutesRef.current) {
      viewerRef.current?.setRoutes(null);
      showPlanRoutes(false);
    }
  }, [studio, showPlanRoutes]);

  // --------------------------------------------------------------- IFC load

  // A file is about to be opened (pointer or focus on a file button or an
  // example, or a file dragged over the page): load the import worker now.
  const warmImport = useCallback(() => viewerRef.current?.warmImport?.(), []);

  const handleFile = useCallback(
    async (nextFile) => {
      if (!nextFile) return;
      if (!nextFile.name.toLowerCase().endsWith(".ifc")) {
        toast("Only .ifc files can be opened in the viewer.", {
          title: "Unsupported file",
          variant: "warning",
        });
        return;
      }
      const viewer = viewerRef.current;
      if (!viewer?.ready) {
        toast("The 3D viewer is still starting. Try again in a moment.", {
          variant: "warning",
        });
        return;
      }

      // Another file may be opened while this one loads. Only the latest load
      // may touch the page: an older one's progress, model info, semantics
      // and geometry are dropped when they arrive (the viewer disposes of its
      // model, ViewerManager.loadIfc).
      const seq = ++loadSeqRef.current;
      const latest = () => seq === loadSeqRef.current;

      rlRef.current?.cancel();
      dropPlan();
      // The new model starts with no graph to colour.
      colours.set(null);
      studio.beginLoad(nextFile);
      geometryRef.current = null;
      setLastSample(null);
      studio.setLoadState({ busy: true, stage: "reading", detail: nextFile.name });

      try {
        const result = await viewer.loadIfc(nextFile, {
          onProgress: ({ stage, detail }) => {
            if (!latest()) return;
            const percent =
              typeof detail === "number"
                ? Math.max(0, Math.min(100, detail * 100))
                : undefined;
            studio.setLoadState({
              busy: true,
              stage,
              detail: typeof detail === "string" ? detail : undefined,
              percent,
            });
          },
        });
        if (!latest()) return;

        studio.setModelInfo(result);
        viewer.setModelAppearance(settings.modelAppearance);
        nextFile
          .arrayBuffer()
          .then((bytes) => semanticsJob(bytes))
          .then(async (semantics) => {
            if (!latest()) return;
            studio.setSemantics(semantics);
            // The exit doors can change with them.
            dropPlan();
            // Declared storeys, moved into the viewer frame by the model's
            // coordination shift, where the slab geometry collapses to one
            // level (dental clinic, Office A, example.ifc: survey strategy 5).
            // Elsewhere the geometry stays: the Duplex declares a foundation
            // and a roof "storey" around its two floors.
            const shift = (await viewer.getCoordinationMatrix())?.[13] ?? 0;
            const geometry = await geometryPromiseRef.current;
            if (!latest()) return;
            const fromGeometry = geometry ? floorLevels(geometry.floors, settings.upAxis) : [];
            const declared = [...new Set((semantics.storeys || []).map((s) => +(s.elevation + shift).toFixed(2)))].sort(
              (a, b) => a - b,
            );
            const [low, high] = geometry ? heightRange(geometry, settings.upAxis) : [-Infinity, Infinity];
            if (fromGeometry.length <= 1 && declared.length >= 2 && declared.every((h) => h >= low - 1 && h <= high + 1)) {
              setLevels(declared);
            }
          })
          .catch((error) => {
            if (!isCancelled(error)) console.warn("Could not read the model's exit flags; exits fall back to geometry.", error);
          });
        setTab("graph");

        // Extract the egress geometry now rather than at the first build:
        // the outlines and the storey cut need it, and the build then skips it.
        setLevels([]);
        setSection(null);
        viewer.setSectionHeight(null);
        geometryPromiseRef.current = viewer
          .extractEgressGeometry()
          .then((geometry) => {
            if (!latest()) return null;
            geometryRef.current = geometry;
            viewer.setOutlines([
              ...geometry.walls,
              ...geometry.floors,
              ...geometry.stairs,
              ...geometry.doors,
              ...(geometry.obstacles || []),
            ]);
            setLevels(floorLevels(geometry.floors, settings.upAxis));
            return geometry;
          })
          .catch((error) => {
            if (!latest()) return null;
            console.warn("Geometry extraction failed; it will be retried at build.", error);
            geometryPromiseRef.current = null;
            return null;
          });

        void studio.refreshCacheInfo();

        toast(
          result.cached
            ? `Loaded from cache in ${formatMs(result.totalMs)}.`
            : `Converted and cached in ${formatMs(result.totalMs)}.`,
          { title: nextFile.name, variant: "success", ttl: 5000 },
        );
      } catch (error) {
        if (latest() && !isCancelled(error)) {
          reportError(error, "Could not load the IFC file");
        }
      } finally {
        if (latest()) studio.setLoadState({ busy: false, stage: null, detail: null });
      }
    },
    [reportError, semanticsJob, settings.modelAppearance, settings.upAxis, studio, toast, dropPlan, colours],
  );

  // --------------------------------------------------------------- examples

  // Shipped with the site (public/examples, written by
  // tools/make_example_ifcs.py), so anyone can try Studio without a model.
  const [examples, setExamples] = useState([]);
  useEffect(() => {
    let alive = true;
    fetch(`${import.meta.env.BASE_URL}examples/index.json`)
      .then((res) => (res.ok ? res.json() : []))
      .then((list) => alive && setExamples(Array.isArray(list) ? list : []))
      .catch(() => alive && setExamples([]));
    return () => {
      alive = false;
    };
  }, []);

  const openExample = useCallback(
    async (entry) => {
      // A file opened while the example downloads wins over it.
      const seq = ++loadSeqRef.current;
      try {
        studio.setLoadState({ busy: true, stage: "downloading", detail: entry.name });
        const res = await fetch(`${import.meta.env.BASE_URL}examples/${entry.file}`);
        if (!res.ok) throw new Error(`Could not download ${entry.name} (${res.status}).`);
        const blob = await res.blob();
        if (seq !== loadSeqRef.current) return;
        await handleFile(new File([blob], entry.file, { type: "application/octet-stream" }));
      } catch (error) {
        if (seq !== loadSeqRef.current) return;
        studio.setLoadState({ busy: false, stage: null, detail: null });
        reportError(error, "Could not open the example");
      }
    },
    [handleFile, reportError, studio],
  );

  // ------------------------------------------------------------ graph build

  const buildGraph = useCallback(async () => {
    const viewer = viewerRef.current;
    if (!viewer?.model) {
      toast("Load an IFC model first.", { variant: "warning" });
      return;
    }

    studio.setGraphBusy(true);
    studio.setLoadState({ busy: true, stage: "extracting", detail: null });
    // A policy still training is for the graph being replaced.
    rlRef.current?.cancel();
    const timer = new PhaseTimer("graph.build");

    try {
      // Geometry extraction is the expensive part, so it is cached against the
      // loaded model: changing a slider re-samples but does not re-extract.
      if (!geometryRef.current) {
        geometryRef.current = await timer.run(
          "extract",
          async () => (await geometryPromiseRef.current) || viewer.extractEgressGeometry(),
        );
      }
      const geometry = geometryRef.current;

      studio.setLoadState({ busy: true, stage: "sampling", detail: null });

      // The worker receives copies so the cached geometry survives transfer.
      const clone = (meshes) =>
        meshes.map((mesh) => ({
          // itemId must survive: the worker groups door meshes by it, and
          // without it a single door yields a waypoint per sub-mesh.
          itemId: mesh.itemId,
          positions: mesh.positions.slice(),
          indices: mesh.indices.slice(),
        }));
      const payload = {
        floors: clone(geometry.floors),
        stairs: clone(geometry.stairs),
        doors: clone(geometry.doors),
        walls: clone(geometry.walls),
        spaces: clone(geometry.spaces || []),
        obstacles: clone(geometry.obstacles || []),
        upAxis: settings.upAxis,
        floorSpacing: settings.floorSpacing,
        agentHeight: settings.agentHeight,
        maxEdgeRise: settings.maxEdgeRise,
        rectilinear: settings.rectilinear,
        maxPoints: settings.maxPoints,
        columnGap: settings.columnGap,
      };
      const transfers = [];
      for (const key of ["floors", "stairs", "doors", "walls", "spaces", "obstacles"]) {
        for (const mesh of payload[key]) {
          transfers.push(mesh.positions.buffer, mesh.indices.buffer);
        }
      }

      // The whole graph is built in the worker (src/core/lattice.js).
      const built = await timer.run("lattice", () => sample(payload, transfers));
      setLastSample({ nodes: built.stats.nodes, ms: built.stats.ms });
      const local = {
        ...built,
        upAxis: settings.upAxis,
        agentHeight: settings.agentHeight,
      };

      // Only the TopologicPy features need the graph on the server, so it is
      // uploaded in the background: the graph is usable at once, and on mobile
      // data the build no longer waits for the upload. Its id is filled in
      // when the server answers (below).
      const response = {
        graphId: null,
        stats: graphStats(local),
        upAxis: settings.upAxis,
        bounds: [],
        timings: {},
        nodes: local.points,
        edges: local.edges,
        kinds: local.kinds,
        blocked: local.blocked,
        // For drawing routes straight against the same walls (straighten.js).
        sections: built.sections,
        sectionHeight: built.sectionHeight,
        agentHeight: settings.agentHeight,
        spacing: built.spacing,
        lattice: { angleDeg: built.stats.angleDeg, spacing: built.spacing, ms: built.stats.ms },
        doors: built.doors || [],
        roofs: geometry.roofs || 0,
        // Why there is no graph at all, rather than "Graph ready, 0 nodes"
        // (survey item 3): shown beside the Build button (Panels.jsx).
        emptyCause: built.stats.nodes
          ? null
          : !geometry.floors?.length
            ? "The model has no floors or slabs to walk on: it may be a structural, MEP or site model."
            : "Its floors gave no walkable surface: they may be modelled only as spaces or coverings, or be too steep or too small.",
      };

      studio.setGraph(response);
      studio.setPath(null);
      studio.setDynamicPath(null);
      studio.setAlternatives(null);
      studio.setNavCheck(null);
      dropPlan();
      // A fire still running spreads over the old graph's node ids.
      studio.stopFire();
      viewer.setPath(null);
      viewer.setPath(null, { dynamic: true });
      viewer.setRoutes(null);

      viewer.setGraph(response.nodes, response.edges, response.kinds);
      colours.set(null);
      viewer.setGraphVisible(settings.showGraph);

      // A freshly built graph is completely hidden behind solid walls and a
      // roof, so the build reads as "nothing happened". Ghost the shell once,
      // the first time, and leave the choice with the user afterwards.
      if (settings.modelAppearance === "solid") {
        setSettings({ modelAppearance: "ghost" });
      }

      timer.finish({ nodes: response.stats.nodes, edges: response.stats.edges });

      if (studio.serverStatus === "online") {
        api
          .importGraph(local)
          .then((uploaded) => {
            // Only if this is still the graph on screen.
            studio.setGraph((current) =>
              current && current.nodes === response.nodes
                ? { ...current, graphId: uploaded.graph_id, bounds: uploaded.bounds ?? [] }
                : current,
            );
          })
          .catch((error) => console.warn("Graph upload failed; continuing in the browser only.", error));
      }

      // A graph in several parts, or none at all, is explained beside the
      // graph (Panels.jsx) and flagged on the stepper; a toast would vanish.
      // With no graph the Graph step stays on screen, since the Route step
      // has nothing to offer (review m7).
      if (!response.stats.nodes) return;
      if (!graphIsSplit(response.stats)) {
        toast(
          `${response.stats.nodes.toLocaleString()} nodes, ` +
            `${response.stats.edges.toLocaleString()} edges.`,
          { title: "Graph ready", variant: "success", ttl: 4000 },
        );
      }
      setTab("route");
    } catch (error) {
      reportError(error, "Could not build the navigation graph");
    } finally {
      studio.setGraphBusy(false);
      studio.setLoadState({ busy: false, stage: null, detail: null });
    }
  }, [reportError, sample, setSettings, settings, studio, toast, dropPlan, colours]);

  /**
   * Server-backed actions need this graph on the server. Without an id the
   * server would fall back to whatever graph it built last, which may be a
   * different model, so refuse plainly instead.
   */
  const needsServerGraph = useCallback(
    (what) => {
      if (graph?.graphId) return false;
      toast(
        `${what} still runs on the backend, which is not connected. The graph itself was built in your browser.`,
        { title: "Backend needed", variant: "warning", ttl: 8000 },
      );
      return true;
    },
    [graph, toast],
  );

  // -------------------------------------------------------------- picking

  /**
   * Nearest navigation node to an arbitrary 3D point.
   *
   * A ray cast into the model hits whatever surface is in front - typically an
   * exterior wall or the roof, not a floor. Routing snaps to the nearest node
   * anyway, so snapping here too means the marker shows the point the route
   * will actually use instead of a spot the walker could never stand on.
   */
  const snapToGraph = useCallback(
    (point) => {
      const nodes = graph?.nodes;
      if (!nodes?.length || !point) return point;
      let best = -1;
      let bestDist = Infinity;
      for (let i = 0; i < nodes.length; i += 3) {
        const dx = nodes[i] - point[0];
        const dy = nodes[i + 1] - point[1];
        const dz = nodes[i + 2] - point[2];
        const d = dx * dx + dy * dy + dz * dz;
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      if (best < 0) return point;
      return [nodes[best], nodes[best + 1], nodes[best + 2]];
    },
    [graph],
  );

  const handlePick = useCallback(
    (mode, rawPoint) => {
      const point = snapToGraph(rawPoint);
      studio.setPoints((current) => ({ ...current, [mode]: point }));
      viewerRef.current?.setMarker(mode, point);
      studio.setPickMode(null);
      // A plan is for one start, exit and fire: moving any of them drops it.
      dropPlan();
      if (mode !== "fire") {
        studio.setPath(null);
        studio.setAlternatives(null);
        viewerRef.current?.setPath(null);
        viewerRef.current?.setRoutes(null);
      }
    },
    [snapToGraph, studio, dropPlan],
  );

  const requestPick = useCallback(
    (mode) => {
      studio.setPickMode((current) => (current === mode ? null : mode));
      // Show the row being set: start and exit live in Route, the fire in
      // Simulate (a shortcut can arm either from any step).
      setTab(mode === "fire" ? "simulate" : "route");
    },
    [studio],
  );

  const clearPoints = useCallback(() => {
    studio.setPoints({ start: null, exit: null, fire: null });
    ["start", "exit", "fire"].forEach((name) => viewerRef.current?.setMarker(name, null));
    dropPlan();
    studio.setPath(null);
    studio.setDynamicPath(null);
    studio.setAlternatives(null);
    viewerRef.current?.setPath(null);
    viewerRef.current?.setPath(null, { dynamic: true });
    viewerRef.current?.setRoutes(null);
  }, [studio, dropPlan]);

  // ------------------------------------------------------------- find path

  // Routes are drawn straight, as Wire.Straighten draws them (straighten.js);
  // the lattice route stays in the result for its length and node ids.
  const straighten = useCallback(
    (route) =>
      graph && settings.straightenRoutes ? withStraight(graph, route, { useWalls: settings.useWalls }) : route,
    [graph, settings.straightenRoutes, settings.useWalls],
  );

  const findPath = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    // The built-in engine runs here, with no round trip; only the TopologicPy
    // engine needs the server.
    const onServer = studio.effectiveEngine === "topologicpy";
    if (onServer && needsServerGraph("The TopologicPy route engine")) return;
    studio.setPathBusy(true);
    studio.setComparison(null);
    studio.setAlternatives(null);
    // The route overlays held the alternatives or the plan's routes.
    viewerRef.current?.setRoutes(null);
    showPlanRoutes(false);
    try {
      const result = onServer
        ? await api.findPath({
            graph_id: graph.graphId,
            start_point: points.start,
            end_point: points.exit,
            engine: studio.effectiveEngine,
            use_walls: settings.useWalls,
          })
        : findRoute(graph, points.start, points.exit, { useWalls: settings.useWalls });
      const shown = straighten(result);
      studio.setPath(shown);
      viewerRef.current?.setPath(drawn(shown));
      if (!result.found) {
        toast(result.note || "No route exists between those two points.", {
          title: "No route",
          variant: "warning",
        });
      }
    } catch (error) {
      reportError(error, "Could not compute a route");
    } finally {
      studio.setPathBusy(false);
    }
  }, [graph, points, reportError, settings.useWalls, studio, toast, needsServerGraph, straighten, showPlanRoutes]);

  const comparePath = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    if (needsServerGraph("Engine comparison")) return;
    studio.setPathBusy(true);
    try {
      const result = await api.comparePath({
        graph_id: graph.graphId,
        start_point: points.start,
        end_point: points.exit,
        use_walls: settings.useWalls,
      });
      studio.setComparison(result);
      const shown = straighten(result.fast);
      studio.setPath(shown);
      viewerRef.current?.setPath(drawn(shown));
      toast(
        result.same_route
          ? "Both engines returned the same route."
          : `Routes differ; cost delta ${result.cost_delta.toFixed(3)}.`,
        {
          title: "Engine comparison",
          variant: result.same_route ? "success" : "warning",
        },
      );
    } catch (error) {
      reportError(error, "Engine comparison failed");
    } finally {
      studio.setPathBusy(false);
    }
  }, [graph, points, reportError, settings.useWalls, studio, toast, needsServerGraph, straighten]);

  const findAlternatives = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    // In the browser unless the TopologicPy engine is chosen (its Yen is
    // TGraph.KShortestPaths on the server).
    const onServer = studio.effectiveEngine === "topologicpy";
    if (onServer && needsServerGraph("Alternative routes on the TopologicPy engine")) return;
    studio.setPathBusy(true);
    studio.setComparison(null);
    try {
      const raw = onServer
        ? await api.alternatives({
            graph_id: graph.graphId,
            start_point: points.start,
            end_point: points.exit,
            engine: studio.effectiveEngine,
            use_walls: settings.useWalls,
            k: settings.altCount,
            method: settings.altMethod,
          })
        : alternativeRoutes(graph, points.start, points.exit, {
            k: settings.altCount,
            method: settings.altMethod,
            useWalls: settings.useWalls,
          });
      const result = { ...raw, routes: raw.routes.map(straighten) };
      studio.setAlternatives(result);
      // The best alternative is the route, so it takes the route's place
      // rather than being drawn twice.
      studio.setPath(result.routes[0] || null);
      viewerRef.current?.setPath(null);
      viewerRef.current?.setRoutes(
        result.routes.map((route, i) => ({ points: drawn(route), color: hexToInt(routeColor(i)) })),
      );
      showPlanRoutes(false);
      // "Only N of k" is shown in the panel next to the list it explains.
      if (!result.routes.length) {
        toast(result.note || "No route exists between those two points.", {
          title: "No route",
          variant: "warning",
        });
      }
    } catch (error) {
      reportError(error, "Could not compute alternative routes");
    } finally {
      studio.setPathBusy(false);
    }
  }, [graph, points, reportError, settings, studio, toast, needsServerGraph, straighten, showPlanRoutes]);

  // ------------------------------------------------- Navigation cross-check

  const runNavigationCheck = useCallback(async () => {
    const viewer = viewerRef.current;
    if (!graph || !file || !viewer) return;
    if (needsServerGraph("The TopologicPy cross-check")) return;
    studio.setNavBusy(true);
    try {
      const result = await api.navigationCheck(file, {
        graphId: graph.graphId,
        spacing: settings.navSpacing,
        startPoint: points.start,
        endPoint: points.exit,
        useWalls: settings.useWalls,
        coordination: await viewer.getCoordinationMatrix(),
      });
      studio.setNavCheck(result);
      const pct = (v) => `${Math.round(v * 100)}%`;
      toast(
        `Navigation covers ${pct(result.studio_covered)} of our nodes; ` +
          `we cover ${pct(result.navigation_covered)} of its.`,
        { title: "TopologicPy cross-check", variant: "success", ttl: 7000 },
      );
    } catch (error) {
      reportError(error, "Navigation cross-check failed");
    } finally {
      studio.setNavBusy(false);
    }
  }, [file, graph, points, reportError, settings, studio, toast, needsServerGraph]);

  // TopologicPy's graph and route are drawn from state, so hiding and showing
  // them is a toggle rather than another server round trip.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const show = Boolean(navCheck && settings.showNavGraph);
    viewer.setReferenceGraph(show ? navCheck.nodes : null, show ? navCheck.edges : null);
    viewer.setReferenceRoute(show && navCheck.route?.found ? navCheck.route.points : null);
  }, [navCheck, settings.showNavGraph]);

  // ------------------------------------------------------------------- fire

  const startFire = useCallback(() => {
    if (!graph || !points.fire) return;
    // A plan still running would land mid-fire and clear its route.
    if (forecastRef.current?.busy) dropPlan();
    studio.stopFire();
    studio.setDynamicPath(null);
    viewerRef.current?.setPath(null, { dynamic: true });

    burningRef.current = new Set();
    temperatureRef.current = new Map();
    studio.setFire({ running: true, step: 0, model: settings.fireModel });

    // Every model reroutes now: temperature by hazard cost, flood and radial by
    // treating burning nodes as impassable.
    const wantsReroute = settings.reroute && Boolean(points.start && points.exit);

    // Runs in the browser (src/core/fire.js), emitting the messages the
    // server's stream did, so no round trip per step and no backend needed.
    const handle = runFire(
      graph,
      {
        model: settings.fireModel,
        firePoint: points.fire,
        startPoint: points.start,
        exitPoint: points.exit,
        maxSteps: settings.fireMaxSteps,
        delayMs: settings.fireDelayMs,
        useWalls: settings.useWalls,
        reroute: wantsReroute,
        interval: settings.rerouteInterval,
        alpha: settings.hazardAlpha,
        lethality: settings.lethalityThreshold || null,
      },
      {
        onMessage: (message) => {
          const viewer = viewerRef.current;
          if (message.type === "step") {
            const burning = burningRef.current;
            (message.nodes || []).forEach((n) => burning.add(n));
            studio.setFire((current) => ({ ...current, step: message.step }));
            viewer?.setBurningNodes(burning);
            colours.set("fire");
          } else if (message.type === "temperature_step") {
            const temperatures = temperatureRef.current;
            temperatures.clear();
            for (const [key, value] of Object.entries(message.temperatures || {})) {
              temperatures.set(Number(key), value);
            }
            studio.setFire((current) => ({ ...current, step: message.step }));
            viewer?.setTemperatures(temperatures);
            colours.set("fire");
          } else if (message.type === "path_update") {
            studio.setDynamicPath(message);
            // Straight runs keep clear of burning or heated nodes.
            const shown =
              settings.straightenRoutes && message.node_ids?.length > 2
                ? straightenRoute(graph, message.node_ids, { useWalls: settings.useWalls, avoid: message.avoid }).points
                : message.path;
            viewer?.setPath(shown, { color: 0xc026d3, dynamic: true });
          }
        },
        onDone: () => {
          studio.setFire((current) => ({ ...current, running: false }));
          toast("Fire simulation finished.", { variant: "info", ttl: 3500 });
        },
        onError: (error) => {
          studio.setFire((current) => ({ ...current, running: false }));
          reportError(error, "Fire simulation");
        },
      },
    );
    studio.setFireHandle(handle);
  }, [graph, points, reportError, settings, studio, toast, dropPlan, colours]);

  // A fire that ends by itself keeps its final colours. Stopping it clears
  // them, back to what the graph showed before: the deadline map if it is
  // wanted, else the graph's own colours.
  const stopFire = useCallback(() => {
    studio.stopFire();
    const viewer = viewerRef.current;
    viewer?.setTemperatures(null);
    viewer?.setBurningNodes(null);
    if (settings.showDeadlineMap && deadlines) {
      viewer?.setDeadlines(deadlines, { max: planHorizon });
      colours.set("deadlines");
    } else {
      colours.set(null);
    }
  }, [studio, settings.showDeadlineMap, deadlines, planHorizon, colours]);

  // --------------------------------------------------------------------- RL

  // Training and the exact optimum it is scored against run in a worker
  // (src/workers/rl.worker.js): the optimum alone held the page for over a
  // second on a large graph with a long step limit.
  const rlJob = useCallback((graphData, options) => {
    rlRef.current ||= workerClient(
      () => new Worker(new URL("./workers/rl.worker.js", import.meta.url), { type: "module" }),
      { failure: "RL training stopped: its worker did not load or ran out of memory. Reload the page and try again." },
    );
    return rlRef.current.run({ graph: graphData, options });
  }, []);

  const trainRl = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    studio.setRl({ busy: true, path: null, reachedExit: false, result: null });
    try {
      // In the browser (src/core/rl.js), with the review's terminal-potential
      // and horizon corrections.
      const trained = await rlJob(
        { nodes: graph.nodes, edges: graph.edges, blocked: graph.blocked, graphId: graph.graphId ?? null },
        {
          startPoint: points.start,
          exitPoint: points.exit,
          // The fire starts at the picked origin, not at the evacuee.
          firePoint: points.fire,
          episodes: settings.rlEpisodes,
          maxSteps: Math.min(settings.rlMaxSteps, RL_MAX_STEPS),
          useFire: settings.rlUseFire,
          fireSpeed: settings.rlFireSpeed,
          useWalls: settings.useWalls,
        },
      );
      const result = straighten(trained);
      studio.setRl({
        busy: false,
        path: result.path,
        reachedExit: result.reached_exit,
        result,
      });
      // Drawn where the hazard-aware route was, which goes with its drawing.
      studio.setDynamicPath(null);
      viewerRef.current?.setPath(drawn(result), { color: 0x14b8a6, dynamic: true });
      if (!result.reached_exit || result.note) {
        // The server says *why*: no escape exists, the step limit is too low,
        // or the policy needs more episodes. Those call for different actions.
        toast(result.note || "The policy did not reach the exit.", {
          title: "RL training",
          variant: result.reached_exit ? "info" : "warning",
        });
      }
    } catch (error) {
      studio.setRl({ busy: false, path: null, reachedExit: false, result: null });
      // Cancelled when the model or the graph changed under it.
      if (!isCancelled(error)) reportError(error, "RL training failed");
    }
  }, [graph, points, reportError, settings, studio, toast, straighten, rlJob]);

  // -------------------------------------------------------------- exit doors

  const exits = useMemo(() => (graph?.doors ? exitDoors(graph.doors, studio.semantics) : null), [graph, studio.semantics]);

  const nearestExit = useCallback(() => {
    if (!graph || !points.start || !exits?.nodes.length) return;
    const open = new Float64Array(graph.nodes.length / 3).fill(Infinity);
    const found = earliestArrival(graph, snap(graph, points.start), 0, {
      exits: exits.nodes,
      closes: open,
      speed: 1,
      blocked: settings.useWalls ? graph.blocked : null,
    });
    if (found.exit < 0) {
      toast("None of the exit doors can be reached from the start.", { title: "No exit", variant: "warning" });
      return;
    }
    const i = found.exit * 3;
    handlePick("exit", [graph.nodes[i], graph.nodes[i + 1], graph.nodes[i + 2]]);
  }, [graph, points.start, exits, settings.useWalls, handlePick, toast]);

  // ------------------------------------------------- plan with the forecast

  const pointsOf = useCallback(
    (nodes) => (nodes || []).map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]),
    [graph],
  );

  const planForecast = useCallback(async () => {
    const exitNodes =
      settings.forecastExits === "model" && exits?.nodes.length
        ? exits.nodes
        : points.exit && graph
          ? [snap(graph, points.exit)]
          : [];
    if (!graph || !points.start || !points.fire || !exitNodes.length) return;
    const problem = planSettingsProblem(settings);
    if (problem) {
      toast(problem, { title: "Check the plan's settings", variant: "warning" });
      return;
    }
    // A plan still running is for older inputs: end its worker rather than
    // queue behind a deadline map that can take seconds.
    if (forecastRef.current?.busy) forecastRef.current.cancel();
    const gen = ++planGenRef.current;
    // What the card describes: the values this plan used (review M2).
    const inputs = planSettings(settings);
    studio.setForecastPlan((current) => ({ ...current, busy: true }));
    try {
      const result = await forecastJob({
        graph: { nodes: graph.nodes, edges: graph.edges, blocked: graph.blocked },
        start: snap(graph, points.start),
        exits: exitNodes,
        seed: snap(graph, points.fire),
        model: settings.fireModel,
        steps: settings.fireMaxSteps,
        secondsPerStep: settings.forecastSecondsPerStep,
        threshold: settings.untenableAbove,
        speed: settings.walkSpeed,
        depart: settings.departAfter,
        limit: settings.doseLimit,
        useWalls: settings.useWalls,
        // The map is the slow part; it is made only when it will be drawn.
        withMap: settings.showDeadlineMap,
      });
      // Inputs changed (or another plan began) while it ran: not this one's.
      if (gen !== planGenRef.current) return;
      // Tagged with the graph it was made on: it is drawn only on that graph.
      studio.setForecastPlan({ busy: false, result, nodes: graph.nodes, inputs });
      // The plan's routes take the place of the others on screen, and their
      // results go with their drawings, so the legend lists what is drawn.
      const viewer = viewerRef.current;
      studio.setPath(null);
      studio.setComparison(null);
      studio.setAlternatives(null);
      studio.setDynamicPath(null);
      viewer?.setPath(null);
      viewer?.setPath(null, { dynamic: true });
      const routes = [];
      if (result.earliest) routes.push({ points: pointsOf(result.earliest.nodes), color: hexToInt(FORECAST_COLORS.earliest) });
      if (result.leastDose && result.front.length > 1) {
        routes.push({ points: pointsOf(result.leastDose.nodes), color: hexToInt(FORECAST_COLORS.leastDose) });
      }
      if (result.snapshot && !result.snapshot.safe && !result.snapshot.beyondForecast) {
        routes.push({ points: pointsOf(result.snapshot.nodes), color: hexToInt(FORECAST_COLORS.snapshot) });
      }
      viewer?.setRoutes(routes);
      showPlanRoutes(routes.length > 0);
      if (!result.earliest) {
        toast("No route reaches the exit before the fire closes it, within the forecast.", {
          title: "No safe route",
          variant: "warning",
        });
      }
    } catch (error) {
      // Cancelled, or overtaken: a newer request or a change owns the state.
      if (isCancelled(error) || gen !== planGenRef.current) return;
      studio.setForecastPlan(NO_PLAN);
      reportError(error, "Planning with the forecast failed");
    }
  }, [graph, points, settings, studio, exits, forecastJob, pointsOf, toast, reportError, showPlanRoutes]);

  // The deadline map colours the graph when a plan arrives or the toggle
  // changes, and only then. A fire that has coloured the graph since keeps
  // its final colours when it ends: this ran again when the fire stopped
  // running and wiped them (review M6).
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (settings.showDeadlineMap && deadlines) {
      viewer.setDeadlines(deadlines, { max: planHorizon });
      colours.set("deadlines");
    } else if (colours.get() === "deadlines") {
      viewer.setDeadlines(null);
      colours.set(null);
    }
  }, [deadlines, planHorizon, settings.showDeadlineMap, colours]);

  // --------------------------------------------------------- reactive push

  useEffect(() => {
    viewerRef.current?.setGraphVisible(settings.showGraph);
  }, [settings.showGraph, graph]);

  useEffect(() => {
    viewerRef.current?.setModelAppearance(settings.modelAppearance);
  }, [settings.modelAppearance, modelInfo]);

  // ---------------------------------------------------------------- hotkeys

  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.tagName === "INPUT" ||
          target.tagName === "SELECT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      switch (event.key.toLowerCase()) {
        case "escape":
          studio.setPickMode(null);
          break;
        case "s":
          if (graph) requestPick("start");
          break;
        case "e":
          if (graph) requestPick("exit");
          break;
        case "f":
          if (graph) requestPick("fire");
          break;
        case "g":
          setSettings((s) => ({ ...s, showGraph: !s.showGraph }));
          break;
        case "m":
          // Cycle solid -> ghost -> hidden -> solid.
          setSettings((s) => ({
            ...s,
            modelAppearance:
              s.modelAppearance === "solid"
                ? "ghost"
                : s.modelAppearance === "ghost"
                  ? "hidden"
                  : "solid",
          }));
          break;
        case "b":
          if (modelInfo) void buildGraph();
          break;
        case "enter":
          if (graph && points.start && points.exit) void findPath();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [buildGraph, findPath, graph, modelInfo, points, requestPick, setSettings, studio]);

  // ------------------------------------------------- legacy topology mode

  const setTopologyOptions = useCallback((patch) => {
    setTopologyOptionsState((current) => ({ ...current, ...patch }));
  }, []);

  const loadTopologyJson = useCallback(
    async (jsonFile) => {
      setTopologyBusy(true);
      setTopology(null);
      setTopologySelection(null);
      setTopologySource(null);
      try {
        const text = await jsonFile.text();
        let parsed;
        try {
          parsed = JSON.parse(text);
        } catch {
          throw new Error("That file is not valid JSON.");
        }
        const payload = await legacyApi.uploadTopology(parsed);
        if (!payload?.vertices || !payload?.faces) {
          throw new Error("The backend returned an unexpected topology shape.");
        }
        payload.edges ||= [];
        payload.raw ||= [];
        setTopology(payload);
        setFitRequest((v) => v + 1);
        toast(`${jsonFile.name} loaded.`, { variant: "success", ttl: 4000 });
      } catch (error) {
        reportError(error, "Could not load the topology JSON");
      } finally {
        setTopologyBusy(false);
      }
    },
    [reportError, toast],
  );

  const processIfcOnServer = useCallback(
    async (ifcFile, includePath) => {
      setTopologyBusy(true);
      setTopologySelection(null);
      try {
        const payload = await legacyApi.uploadIfc(ifcFile, {
          includePath,
          tiltMin: topologyOptions.tiltMin,
          maxZSpan: topologyOptions.maxZSpan,
          minFloorArea: topologyOptions.minFloorArea,
        });
        if (!payload?.vertices || !payload?.faces) {
          throw new Error("The backend returned an unexpected topology shape.");
        }
        payload.edges ||= [];
        payload.raw ||= [];
        setTopology(payload);
        setTopologySource({ file: ifcFile, includePath });
        setFitRequest((v) => v + 1);
        toast(
          includePath
            ? "Floors extracted and shortest path computed."
            : `${ifcFile.name} processed on the server.`,
          { variant: "success", ttl: 4000 },
        );
      } catch (error) {
        reportError(error, "Server-side IFC processing failed");
      } finally {
        setTopologyBusy(false);
      }
    },
    [reportError, toast, topologyOptions],
  );

  const reprocessTopology = useCallback(() => {
    if (!topologySource?.file) return;
    void processIfcOnServer(topologySource.file, true);
  }, [processIfcOnServer, topologySource]);

  const topologySummary = useMemo(() => {
    if (!topology) return null;
    return {
      vertices: topology.vertices?.length || 0,
      edges: topology.edges?.length || 0,
      faces: topology.faces?.length || 0,
      raw: topology.raw?.length || 0,
    };
  }, [topology]);

  const topologyEntityById = useMemo(() => {
    const map = new Map();
    for (const entity of topology?.raw || []) {
      const id = entity.uid ?? entity.uuid;
      if (id) map.set(id, entity);
    }
    return map;
  }, [topology]);

  const selectedTopologyEntity = useMemo(
    () => (topologySelection ? topologyEntityById.get(topologySelection.uid) || null : null),
    [topologySelection, topologyEntityById],
  );

  // Translucency is applied to a derived copy so toggling it never mutates the
  // payload the inspector reads from.
  const displayTopology = useMemo(() => {
    if (!topology) return null;
    if (!topologyOptions.translucent) return topology;
    const faces = (topology.faces || []).map((face) => ({
      ...face,
      opacity: 0.25,
      dictionary: { ...(face.dictionary || {}), opacity: 0.25 },
    }));
    return { ...topology, faces };
  }, [topology, topologyOptions.translucent]);

  // Keep the active tab valid when the viewer mode changes.
  const tabs = mode === "ifc" ? IFC_TABS : TOPOLOGY_TABS;
  const activeTab = tabs.some((t) => t.id === tab) ? tab : tabs[0].id;

  // ------------------------------------------------------- doors, by storey

  // Exits are usually doors, so the Route step offers them as a list: also
  // the way to set an exit without a mouse.
  const doorList = useMemo(() => {
    if (!graph?.kinds) return [];
    const up = graph.upAxis === "z" ? 2 : graph.upAxis === "x" ? 0 : 1;
    const out = [];
    for (let i = 0; i < graph.kinds.length; i += 1) {
      if (graph.kinds[i] !== 2) continue;
      const point = [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]];
      out.push({ id: i, point });
    }
    out.sort((a, b) => a.point[up] - b.point[up] || a.point[0] - b.point[0]);
    return out.map((door, k) => ({
      ...door,
      label: `Door ${k + 1}: ${door.point.map((v) => v.toFixed(1)).join(", ")}`,
    }));
  }, [graph]);

  // ------------------------------------------------------------ step status

  const stepStatus = useMemo(
    () => ({
      model: modelInfo ? "done" : "todo",
      graph: graph ? (!graph.stats.nodes || graphIsSplit(graph.stats) ? "warning" : "done") : "todo",
      route: path?.found ? "done" : "todo",
      simulate: fire.step > 0 || rl.result ? "done" : "todo",
    }),
    [modelInfo, graph, path, fire.step, rl.result],
  );

  // ----------------------------------------------------------------- legend

  const legend = useMemo(() => {
    const items = [];
    // The deadline map recolours every link, so the graph's own colours are
    // not on screen then (review m4); heat and burning leave the rest as is.
    if (graph && graphColours !== "deadlines") {
      items.push({ label: "Walkable graph", color: "var(--viz-graph)" });
      if (graph.stats.stair_nodes) {
        items.push({ label: "Stairs", color: "var(--viz-stair)" });
      }
      if (graph.stats.door_nodes) {
        items.push({ label: "Doors", color: "var(--viz-door)" });
      }
    }
    if (graphColours === "fire" && fire.model !== "temperature") {
      items.push({ label: "Burning", color: "var(--viz-fire)" });
    }
    if (points.start) items.push({ label: "Start", color: "var(--viz-start)", dot: true });
    if (points.exit) items.push({ label: "Exit", color: "var(--viz-exit)", dot: true });
    if (points.fire) items.push({ label: "Fire origin (spike)", color: "var(--viz-fire)", dot: true });
    if (alternatives?.routes.length > 1) {
      alternatives.routes.forEach((route, i) =>
        items.push({ label: `Route ${route.rank} (${route.length.toFixed(1)} m)`, color: routeColor(i) }),
      );
    } else if (path?.found) {
      items.push({ label: "Egress route", color: "var(--viz-path)" });
    }
    if (dynamicPath) {
      items.push({ label: "Hazard-aware route", color: "var(--viz-path-dynamic)" });
    }
    if (navCheck && settings.showNavGraph) {
      items.push({ label: "TopologicPy Navigation", color: NAV_COLOR });
    }
    // The plan's routes, while they are the ones drawn.
    const plan = planRoutesShown ? forecastPlan.result : null;
    if (plan?.earliest) items.push({ label: "Earliest safe route (forecast)", color: FORECAST_COLORS.earliest });
    if (plan?.leastDose && plan.front.length > 1) items.push({ label: "Least-dose route", color: FORECAST_COLORS.leastDose });
    if (plan?.snapshot && !plan.snapshot.safe && !plan.snapshot.beyondForecast) {
      items.push({ label: "Planned on the fire now: unsafe", color: FORECAST_COLORS.snapshot });
    }
    return items;
  }, [
    graph,
    points,
    path,
    dynamicPath,
    alternatives,
    navCheck,
    settings.showNavGraph,
    forecastPlan.result,
    planRoutesShown,
    graphColours,
    fire.model,
  ]);

  // ------------------------------------------------------------------ render

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <img src={logo} alt="" className="brand__mark" />
          <span className="brand__name">Topologic Studio</span>
          <span className="brand__tag">Next</span>
        </div>

        <div className="header__divider" />

        <div className="header__mode">
          <Segmented
            value={mode}
            onChange={setMode}
            options={MODES}
            label="Viewer mode"
          />
        </div>

        <div className="header__divider" />

        <div className="header__group">
          {mode === "ifc" ? (
            <>
              <FileButton accept=".ifc" onFile={handleFile} variant="primary" onIntent={warmImport}>
                Open IFC
              </FileButton>
              {file && (
                <span className="badge" title={file.name}>
                  {file.name.length > 24 ? `${file.name.slice(0, 22)}...` : file.name}
                </span>
              )}
            </>
          ) : (
            <>
              <FileButton
                accept=".json"
                onFile={loadTopologyJson}
                variant="primary"
                disabled={topologyBusy}
              >
                Open JSON
              </FileButton>
              <FileButton
                accept=".ifc"
                onFile={(f) => processIfcOnServer(f, false)}
                disabled={topologyBusy}
              >
                Process IFC on server
              </FileButton>
              <FileButton
                accept=".ifc"
                onFile={(f) => processIfcOnServer(f, true)}
                disabled={topologyBusy}
              >
                Extract floors + path
              </FileButton>
            </>
          )}
        </div>

        <div className="header__spacer" />

        <div className="header__group header__group--wide">
          {studio.serverStatus === "waking" && (
            <Badge variant="warning" pulse>
              Waking backend…
            </Badge>
          )}
          {studio.serverStatus === "offline" && (
            <Badge variant="danger">Backend offline</Badge>
          )}
          {modelInfo?.cached && <Badge variant="success">cached</Badge>}
        </div>

        <div className="header__group">
          <Button variant="ghost" onClick={() => setMethodsOpen(true)} aria-haspopup="dialog" aria-label="Methods">
            <Icon name="book" size={18} />
            <span className="header__label">Methods</span>
          </Button>
          <Button
            icon
            variant="ghost"
            onClick={toggleTheme}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            aria-label="Toggle colour theme"
            aria-pressed={theme === "dark"}
          >
            <Icon name={theme === "dark" ? "sun" : "moon"} size={18} />
          </Button>
          <Button
            icon
            variant="ghost"
            onClick={() => setPanelOpen((v) => !v)}
            title={panelOpen ? "Hide the workflow panel" : "Show the workflow panel"}
            aria-label="Show the workflow panel"
            aria-pressed={panelOpen}
          >
            <Icon name="panel" size={18} />
          </Button>
        </div>
      </header>

      <div className="app-body" data-panel-collapsed={!panelOpen}>
        {mode === "ifc" ? (
          <Viewport
            viewerRef={viewerRef}
            onPick={handlePick}
            onFile={handleFile}
            theme={theme}
            pickMode={pickMode}
            hasModel={Boolean(modelInfo)}
            loadState={studio.loadState}
            graphStats={graph?.stats}
            fire={fire}
            levels={levels}
            section={section}
            onSection={(index) => {
              setSection(index);
              const viewer = viewerRef.current;
              viewer?.setSectionHeight(index === null ? null : levels[index] + SECTION_ABOVE_FLOOR);
              // A cut is read in solid: ghosted, the cut walls show no section.
              if (index !== null && settings.modelAppearance !== "solid") setSettings({ modelAppearance: "solid" });
            }}
            fireSteps={settings.fireMaxSteps}
            heatLegend={graphColours === "fire" && fire.model === "temperature"}
            deadlineLegend={graphColours === "deadlines" ? { max: planHorizon } : null}
            onStopFire={stopFire}
            legend={legend}
            examples={examples}
            onExample={openExample}
            onIntent={warmImport}
          />
        ) : (
          <div className="viewport">
            {topology ? (
              <Suspense
                fallback={
                  <div className="placeholder">
                    <div className="placeholder__card">
                      <div
                        className="btn__spinner"
                        style={{ color: "var(--accent)", width: 22, height: 22 }}
                      />
                      <p>Loading the topology viewer...</p>
                    </div>
                  </div>
                }
              >
                <TopologyViewer
                  data={displayTopology || topology}
                  selection={topologySelection}
                  onSelectionChange={setTopologySelection}
                  showFaces={topologyOptions.showFaces}
                  showVerts={topologyOptions.showVerts}
                  wireframe={topologyOptions.wireframe}
                  fitRequest={fitRequest}
                  theme={theme}
                />
              </Suspense>
            ) : (
              <div className="placeholder">
                <div className="placeholder__card">
                  <img src={logo} alt="" className="placeholder__logo" />
                  <h2>Topology JSON viewer</h2>
                  <p>
                    Inspect a TopologicPy JSON export, or have the server extract a
                    topology from an IFC file. Click any face, edge or vertex to walk
                    up its hierarchy and read its dictionary.
                  </p>
                </div>
              </div>
            )}
            {topologyBusy && (
              <div className="overlay overlay--top-left">
                <div className="card progress">
                  <div className="progress__row">
                    <span className="progress__label">Processing on the server</span>
                  </div>
                  <div className="progress__track">
                    <div className="progress__fill progress__fill--indeterminate" />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {panelOpen && (
          <aside className="panel" aria-label="Workflow">
            {mode === "ifc" ? (
              <Stepper steps={tabs} active={activeTab} status={stepStatus} onSelect={setTab} />
            ) : (
              <div className="panel__tabs" role="tablist" aria-label="Panels">
                {tabs.map((entry) => (
                  <button
                    key={entry.id}
                    id={`step-${entry.id}`}
                    type="button"
                    role="tab"
                    className="panel__tab"
                    aria-selected={activeTab === entry.id}
                    aria-controls="workflow-panel"
                    data-step={entry.id}
                    onClick={() => setTab(entry.id)}
                  >
                    {entry.label}
                  </button>
                ))}
              </div>
            )}

            <div className="panel__scroll" id="workflow-panel" role="tabpanel" aria-labelledby={`step-${activeTab}`}>
              {activeTab === "topology" && (
                <TopologyPanel
                  topology={topology}
                  summary={topologySummary}
                  selection={topologySelection}
                  selectedEntity={selectedTopologyEntity}
                  options={topologyOptions}
                  setOptions={setTopologyOptions}
                  busy={topologyBusy}
                  onReprocess={reprocessTopology}
                  canReprocess={Boolean(topologySource?.file)}
                  onFit={() => setFitRequest((v) => v + 1)}
                />
              )}
              {activeTab === "model" && (
                <ModelPanel
                  studio={studio}
                  examples={examples}
                  onExample={openExample}
                  onClearCache={studio.purgeCache}
                />
              )}
              {activeTab === "graph" && (
                <GraphPanel studio={studio} onBuildGraph={buildGraph} onNavigationCheck={runNavigationCheck} />
              )}
              {activeTab === "route" && (
                <RoutePanel
                  studio={studio}
                  onPick={requestPick}
                  onClearPoints={clearPoints}
                  onFindPath={findPath}
                  onCompare={comparePath}
                  onAlternatives={findAlternatives}
                  onPlace={handlePick}
                  doors={doorList}
                  exits={exits}
                  onNearestExit={nearestExit}
                  onGoTo={setTab}
                />
              )}
              {activeTab === "simulate" && (
                <SimulatePanel
                  studio={studio}
                  onPick={requestPick}
                  onPlace={handlePick}
                  onStartFire={startFire}
                  onStopFire={stopFire}
                  onTrainRl={trainRl}
                  onPlanForecast={planForecast}
                  deadlinesShown={graphColours === "deadlines"}
                  exits={exits}
                  onGoTo={setTab}
                />
              )}
            </div>

            <div className="statusbar">
              <span className="statusbar__item">
                <span
                  className="badge__dot"
                  style={{
                    background:
                      studio.serverStatus === "online"
                        ? "var(--success)"
                        : studio.serverStatus === "offline"
                          ? "var(--danger)"
                          : "var(--warning)",
                  }}
                />
                {studio.serverStatus === "online"
                  ? "backend up"
                  : studio.serverStatus === "offline"
                    ? "backend down"
                    : "backend waking"}
              </span>
              {lastSample && (
                <>
                  <span className="statusbar__sep" />
                  <span className="statusbar__item">
                    graph {lastSample.nodes.toLocaleString()} nodes in {Math.round(lastSample.ms)} ms
                  </span>
                </>
              )}
              {studio.cacheInfo.bytes > 0 && (
                <>
                  <span className="statusbar__sep" />
                  <span className="statusbar__item">
                    cache {formatBytes(studio.cacheInfo.bytes)}
                  </span>
                </>
              )}
            </div>
          </aside>
        )}
      </div>

      <div className="toasts" role="status" aria-live="polite">
        {studio.toasts.map((entry) => (
          <div key={entry.id} className={`toast toast--${entry.variant}`}>
            <div className="toast__body">
              {entry.title && <div className="toast__title">{entry.title}</div>}
              <div className="toast__message">{entry.message}</div>
            </div>
            <button
              type="button"
              className="toast__close"
              onClick={() => studio.dismissToast(entry.id)}
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {methodsOpen && (
        <ErrorBoundary onError={methodsFailed}>
          <Suspense
            fallback={
              <div className="methods-loading" role="status">
                <div className="card">
                  <div className="btn__spinner" aria-hidden="true" />
                  <span>Loading the Methods page...</span>
                </div>
              </div>
            }
          >
            <MethodsDialog open onClose={closeMethods} studio={studio} />
          </Suspense>
        </ErrorBoundary>
      )}
    </div>
  );
}
