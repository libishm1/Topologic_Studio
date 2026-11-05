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
import { CONVERTER } from "./lib/fragmentCache.js";
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
import { zoneNodeFrames } from "./core/zone.js";
import { RL_MAX_STEPS } from "./core/rl.js";
import { fireModelIn, isFlamescapes, planSettings, useStudio } from "./state/useStudio.js";
import { FIRE_CLASSES, fuelsFor } from "./core/fireClasses.js";
import { centroidOf } from "./core/centroid.js";
import { GUIDED_RUNS, farthestFromExits, nodeAlong, stagesFor } from "./core/guided.js";
import { describeFacts } from "./core/describeFacts.js";
import { codeCheck } from "./core/codeCheck.js";
import { CLOUD_URL, describeInCloud, describeOnDevice, onDeviceStatus } from "./lib/describe.js";
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

/** The two-zone model's gas temperatures are coloured from 20 to this many degrees C. */
const ZONE_COLOUR_PEAK = 600;
// The smoke cloud at full strength in the smoke view; lighter in the heat
// view, where the graph's colours have to show through it.
const SMOKE_IN_HEAT_VIEW = 0.35;
// Frames a second the view draws while the fire solver computes on the GPU
// (and whenever the camera moves): the playback changes it about 8 times a second.
const GPU_RUN_DRAW_RATE = 15;
const GPU_RATE_KEY = "topologic-studio.flamescapes.gpuRate";

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
  const rlRef = useRef(null);
  // The Flamescapes solver's worker, and its current run: inputs, frames as
  // they arrive, and the stacked result once done (reused by the plan).
  const flamescapesRef = useRef(null);
  const combustionRef = useRef(null);
  // {name, rate}: the cells a millisecond this device's GPU stepped in its
  // first measured run, kept in this browser so the same GPU always gets the
  // same grid level (the results depend on the grid).
  const gpuRateRef = useRef(null);
  if (gpuRateRef.current === null) {
    try {
      gpuRateRef.current = JSON.parse(localStorage.getItem(GPU_RATE_KEY) || "false") || false;
    } catch {
      gpuRateRef.current = false;
    }
  }
  // The run whose frames are on screen (playback reads its view from a ref,
  // so switching heat and smoke mid-run holds: review 4).
  const playingRunRef = useRef(null);
  const showRef = useRef("heat");
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
  // The two-zone model only where the model has IfcSpace rooms (useStudio.js).
  const fireModel = fireModelIn(settings, modelInfo);

  // ------------------------------------------------------------ the workers

  // Each job is a promise that also settles when its worker fails to load or
  // dies (lib/workerJob.js); a failed worker is replaced at the next job.
  useEffect(
    () => () => {
      samplerRef.current?.cancel();
      forecastRef.current?.cancel();
      rlRef.current?.cancel();
      flamescapesRef.current?.cancel();
    },
    [],
  );

  // Planning with the forecast has its own worker: its deadline map takes a
  // few seconds on a large graph and must not hold up a graph rebuild.
  const forecastJob = useCallback((payload, transfers = []) => {
    forecastRef.current ||= workerClient(
      () => new Worker(new URL("./workers/forecast.worker.js", import.meta.url), { type: "module" }),
      { failure: "Planning with the forecast stopped: its worker did not load or ran out of memory. Reload the page and try again." },
    );
    return forecastRef.current.run({ payload }, transfers);
  }, []);

  // The Flamescapes solver streams a frame at a time as progress.
  const flamescapesJob = useCallback((payload, transfers, onProgress) => {
    flamescapesRef.current ||= workerClient(
      () => new Worker(new URL("./workers/flamescapes.worker.js", import.meta.url), { type: "module" }),
      { failure: "The Flamescapes fire solver stopped, usually because the browser ran out of memory. Try the CPU or a shorter run." },
    );
    return flamescapesRef.current.run({ payload }, transfers, { onProgress });
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
  // Whether a plan is running in the forecast worker, which also runs the
  // two-zone frames a fire starts with: only a plan is ended for a new one.
  const planRunningRef = useRef(false);

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
    if (planRunningRef.current) {
      planRunningRef.current = false;
      forecastRef.current?.cancel();
      // A Flamescapes run started for this plan, and not on screen, is ended
      // too (review 8).
      const run = combustionRef.current;
      if (run && !run.done && run !== playingRunRef.current) {
        flamescapesRef.current?.cancel();
        combustionRef.current = null;
      }
    }
    studio.setForecastPlan(NO_PLAN);
    if (planRoutesRef.current) {
      viewerRef.current?.setRoutes(null);
      showPlanRoutes(false);
    }
  }, [studio, showPlanRoutes]);

  // --------------------------------------------------------------- IFC load

  // A file is about to be opened (pointer or focus on a file button or an
  // example, or a file dragged over the page): load the import worker now.
  // An example converted at build time needs only the fragments worker.
  const warmImport = useCallback(
    (entry) => viewerRef.current?.warmImport?.({ convert: !entry?.prepared }),
    [],
  );

  // ------------------------------------------------------ rooms (LOD1) models

  // A rooms model with no IFC behind it (core/lod1.js): a TopologicPy LOD1
  // export with its circulation and adjacency graphs, a GeoJSON plan, or the
  // rooms drafted from floor-plan drawings. It becomes the egress geometry an
  // IFC gives (lod1ToEgressGeometry), so the graph, routes, the code check and
  // both fire models run on it unchanged, and is drawn as prisms.
  const openLod1 = useCallback(
    async (model, { name, size = 0 }) => {
      const viewer = viewerRef.current;
      if (!viewer?.ready) {
        toast("The 3D viewer is still starting. Try again in a moment.", { variant: "warning" });
        return;
      }
      // An IFC still loading is dropped.
      ++loadSeqRef.current;
      rlRef.current?.cancel();
      dropPlan();
      colours.set(null);
      studio.beginLoad({ name, size, lod1: true });
      setLastSample(null);
      setSection(null);
      viewer.setSectionHeight(null);
      try {
        const t0 = performance.now();
        const { lod1ToEgressGeometry } = await import("./core/lod1.js");
        // The model's frame is the IFC's: the viewer's up axis is y.
        if (settings.upAxis !== "y") setSettings({ upAxis: "y" });
        const geometry = lod1ToEgressGeometry(model, { upAxis: "y" });
        viewer.loadLod1(geometry);
        viewer.setModelAppearance(settings.modelAppearance);
        geometryRef.current = geometry;
        geometryPromiseRef.current = Promise.resolve(geometry);
        studio.setSemantics(geometry.semantics);
        setLevels(floorLevels(geometry.floors, "y"));
        viewer.setOutlines([...geometry.walls, ...geometry.floors, ...geometry.stairs, ...geometry.doors]);
        studio.setModelInfo({
          lod1: model,
          totalMs: performance.now() - t0,
          stats: {
            floors: geometry.floors.length,
            stairs: model.links.filter((l) => l.kind !== "lift").length,
            doors: model.doors.length,
            walls: geometry.walls.length,
            spaces: model.rooms.length,
            storeys: model.levels.length,
          },
        });
        setTab("graph");
        const warnings = model.meta?.warnings?.length || 0;
        toast(`${model.rooms.length} rooms on ${model.levels.length} storeys${warnings ? `, ${warnings} warning${warnings > 1 ? "s" : ""} (step 1)` : ""}.`, {
          title: name,
          variant: warnings ? "warning" : "success",
          ttl: 5000,
        });
      } catch (error) {
        studio.beginLoad(null);
        reportError(error, "Could not open the rooms model");
      }
    },
    [colours, dropPlan, reportError, setSettings, settings.modelAppearance, settings.upAxis, studio, toast],
  );

  // JSON files chosen or dropped together: the TopologicPy CellComplex with
  // its graphs, or a GeoJSON plan.
  const openLod1Files = useCallback(
    async (files) => {
      try {
        const jsons = await Promise.all(
          files.map(async (f) => {
            try {
              return JSON.parse(await f.text());
            } catch {
              throw new Error(`${f.name} is not valid JSON.`);
            }
          }),
        );
        const { lod1FromJsonFiles, lod1JsonKind } = await import("./core/lod1.js");
        const model = lod1FromJsonFiles(jsons);
        const main = files[jsons.findIndex((j) => ["cellcomplex", "geojson"].includes(lod1JsonKind(j)))] || files[0];
        const graphs = files.length - 1;
        await openLod1(model, {
          name: graphs > 0 ? `${main.name} + ${graphs} graph${graphs > 1 ? "s" : ""}` : main.name,
          size: files.reduce((s, f) => s + (f.size || 0), 0),
        });
      } catch (error) {
        reportError(error, "Could not open the rooms model");
      }
    },
    [openLod1, reportError],
  );

  const exportLod1 = useCallback(async () => {
    const model = modelInfo?.lod1;
    if (!model) return;
    const { lod1ToGeoJSON } = await import("./core/lod1.js");
    const blob = new Blob([JSON.stringify(lod1ToGeoJSON(model))], { type: "application/geo+json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${(file?.name || "rooms").replace(/(\.(geo)?json)?( \+ .*)?$/i, "")}-plan.geojson`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }, [modelInfo, file]);

  // `prepared`: an example's fragments and semantics, converted at build time
  // (openExample); nextFile then stands in for its IFC. JSON and GeoJSON files
  // (one, or several chosen or dropped together) are rooms models.
  const handleFile = useCallback(
    async (chosen, { prepared = null } = {}) => {
      const files = Array.isArray(chosen) ? chosen : chosen ? [chosen] : [];
      const isJson = (f) => /\.(geo)?json$/i.test(f.name);
      const nextFile = files.find((f) => f.name.toLowerCase().endsWith(".ifc")) || null;
      if (!nextFile && files.length && files.every(isJson)) {
        await openLod1Files(files);
        return;
      }
      if (!nextFile) {
        if (files.length) {
          toast("Open an .ifc file, or a rooms model: a TopologicPy LOD1 export with its graphs, or a GeoJSON plan.", {
            title: "Unsupported file",
            variant: "warning",
          });
        }
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
          prepared,
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

        // Exit flags and roof slabs from the IFC itself (core/ifcSemantics.js),
        // read by the import worker after the fragments, or kept with them in
        // the cache. They may come after the model is drawn.
        const { semantics: reading, ...info } = result;
        studio.setModelInfo(info);
        viewer.setModelAppearance(settings.modelAppearance);
        Promise.resolve(reading)
          .then(async (semantics) => {
            if (!latest()) return;
            if (!semantics) throw new Error("No worker could read the IFC.");
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
          result.prepared
            ? `Opened in ${formatMs(result.totalMs)}.`
            : result.cached
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
    [reportError, settings.modelAppearance, settings.upAxis, studio, toast, dropPlan, colours, openLod1Files],
  );

  // --------------------------------------------------------------- examples

  // Shipped with the site (public/examples, written by
  // tools/make_example_ifcs.py), so anyone can try Studio without a model.
  // Converted at build time too (scripts/prepare-examples.mjs): an example
  // whose fragments match this build's converter opens from them, with no
  // IFC download, no web-ifc and no conversion.
  const [examples, setExamples] = useState([]);
  useEffect(() => {
    let alive = true;
    const base = `${import.meta.env.BASE_URL}examples/`;
    const json = (url, fallback) =>
      fetch(url)
        .then((res) => (res.ok ? res.json() : fallback))
        .catch(() => fallback);
    Promise.all([json(`${base}index.json`, []), json(`${base}prepared/index.json`, null)]).then(([list, prepared]) => {
      if (!alive) return;
      const ready = prepared?.converter === CONVERTER ? prepared.examples || {} : {};
      setExamples(Array.isArray(list) ? list.map((entry) => ({ ...entry, prepared: ready[entry.id] || null })) : []);
    });
    return () => {
      alive = false;
    };
  }, []);

  const openExample = useCallback(
    async (entry) => {
      // A file opened while the example downloads wins over it.
      const seq = ++loadSeqRef.current;
      const base = `${import.meta.env.BASE_URL}examples/`;
      const download = async (path, read) => {
        const res = await fetch(`${base}${path}`);
        if (!res.ok) throw new Error(`Could not download ${entry.name} (${res.status}).`);
        return read(res);
      };
      try {
        studio.setLoadState({ busy: true, stage: "downloading", detail: entry.name });
        if (entry.prepared) {
          const [fragments, semantics] = await Promise.all([
            download(entry.prepared.fragments, (res) => res.arrayBuffer()),
            download(entry.prepared.semantics, (res) => res.json()),
          ]);
          if (seq !== loadSeqRef.current) return;
          // The IFC itself is fetched only if the server cross-check needs it.
          const standIn = {
            name: entry.file,
            size: entry.bytes || 0,
            download: async () =>
              new File([await download(entry.file, (res) => res.blob())], entry.file, {
                type: "application/octet-stream",
              }),
          };
          await handleFile(standIn, { prepared: { fragments, semantics } });
          return;
        }
        const blob = await download(entry.file, (res) => res.blob());
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
    if (!viewer?.model && !viewer?.lod1) {
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
        // Lifts, escalators and stair cores (src/core/cores.js).
        coreMeshes: clone(geometry.coreMeshes || []),
        cores: geometry.cores || [],
        useLifts: settings.useLifts,
        upAxis: settings.upAxis,
        floorSpacing: settings.floorSpacing,
        agentHeight: settings.agentHeight,
        maxEdgeRise: settings.maxEdgeRise,
        rectilinear: settings.rectilinear,
        maxPoints: settings.maxPoints,
        columnGap: settings.columnGap,
      };
      const transfers = [];
      for (const key of ["floors", "stairs", "doors", "walls", "spaces", "obstacles", "coreMeshes"]) {
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
        cores: built.cores || [],
        useLifts: settings.useLifts,
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
    if (viewer.lod1) {
      toast("The TopologicPy cross-check reads an IFC; a rooms model has none to send.", { variant: "warning" });
      return;
    }
    if (needsServerGraph("The TopologicPy cross-check")) return;
    studio.setNavBusy(true);
    try {
      // An example opened from its prepared fragments fetches its IFC now.
      const ifc = file.download ? await file.download() : file;
      const result = await api.navigationCheck(ifc, {
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

  // What the two-zone model needs besides the graph (src/core/rooms.js): the
  // IfcSpace meshes (copied, then handed to the worker), the graph's doors with
  // their directions, the IFC's door sizes and exit flags, and the fire.
  const zoneInputs = useCallback(() => {
    const spaces = (geometryRef.current?.spaces || []).map((mesh) => ({
      itemId: mesh.itemId,
      positions: mesh.positions.slice(),
      indices: mesh.indices.slice(),
    }));
    if (!spaces.length) throw new Error("The two-zone model needs the model's IfcSpace rooms, and none were extracted.");
    return {
      zone: {
        spaces,
        doors: graph.doors || [],
        declaredDoors: (studio.semantics?.doors || []).map(({ id, widthM, heightM, exit }) => ({ id, widthM, heightM, exit })),
        upAxis: graph.upAxis,
        agentHeight: graph.agentHeight,
        growth: settings.zoneGrowth,
        peak: settings.zonePeakKw,
        smoke: settings.zoneSmoke,
        horizon: settings.zoneHorizon,
        untenableC: settings.zoneUntenableC,
      },
      transfers: spaces.flatMap((mesh) => [mesh.positions.buffer, mesh.indices.buffer]),
    };
  }, [graph, settings, studio.semantics]);

  // What the Flamescapes solver needs (src/core/flamescapesRun.js): the
  // IFC's walls, slabs, roofs, stairs and obstacles, the graph's nodes, the
  // fire and the first item's fire class (src/core/fireClasses.js).
  const combustionInputs = useCallback(() => {
    const geometry = geometryRef.current;
    if (!geometry) throw new Error("Build the walkable graph first.");
    const clone = (meshes) => (meshes || []).map((m) => ({ positions: m.positions.slice(), indices: m.indices.slice() }));
    // Openings for the smoke: open windows leave the obstacles (their walls
    // are cut round them), closed doors join the solids.
    const ids = viewerRef.current?.categoryIds || {};
    const windowIds = new Set(ids.windows || []);
    const windowsOpen = settings.flamescapesWindows === "open";
    const doorsClosed = settings.flamescapesDoors === "closed";
    const geo = {
      walls: clone(geometry.walls),
      floors: clone(geometry.floors),
      roofs: clone(geometry.roofMeshes),
      stairs: clone(geometry.stairs),
      obstacles: clone(windowsOpen ? (geometry.obstacles || []).filter((m) => !windowIds.has(m.itemId)) : geometry.obstacles),
      // A rooms model's openings (doorways with no leaf) stay open.
      ...(doorsClosed ? { doors: clone((geometry.doors || []).filter((m) => m.kind !== "opening")) } : {}),
    };
    const transfers = Object.values(geo).flatMap((list) => list.flatMap((m) => [m.positions.buffer, m.indices.buffer]));
    const { contentsFuel, ignitionFuel, item } = fuelsFor(settings.flamescapesClass);
    // At most 120 frames of at least 1 s: memory is frames x nodes x 12 bytes
    // (review 6), and a frame shorter than the solver's step means nothing.
    const frames = Math.max(1, Math.min(120, settings.fireMaxSteps, Math.floor(settings.flamescapesHorizon)));
    const payload = {
      geometry: geo,
      upAxis: graph.upAxis,
      fire: points.fire,
      nodes: graph.nodes.slice(),
      agentHeight: graph.agentHeight,
      headHeight: 1.8,
      horizon: settings.flamescapesHorizon,
      frameSeconds: settings.flamescapesHorizon / frames,
      embers: fireModel === "flamescapes-embers",
      openings: {
        doors: doorsClosed ? "closed" : "open",
        doorCount: new Set((geometry.doors || []).map((m) => m.itemId)).size,
        windows: windowsOpen ? "open" : "closed",
        windowCount: windowIds.size,
        terminals: (geometry.terminals || []).length,
        ventilation: settings.flamescapesVentilation,
      },
      // The model's air terminals, when the ventilation runs (flamescapesRun.js ductsFor).
      ventilation:
        settings.flamescapesVentilation === "running" && geometry.terminals?.length
          ? { terminals: geometry.terminals, flow: settings.flamescapesTerminalFlow / 1000 }
          : null,
      contentsFuel,
      ignitionFuel,
      item,
      fireLoad: settings.flamescapesFireLoad * 1e6,
      backend: settings.flamescapesBackend,
      // The GPU's measured speed (cells a ms) sizes its grid, unless the finest grid is asked for.
      sizeBySpeed: settings.flamescapesGrid !== "finest",
      gpuRate: gpuRateRef.current?.rate ?? null,
      gpuName: gpuRateRef.current?.name ?? null,
    };
    transfers.push(payload.nodes.buffer);
    const key = JSON.stringify([
      points.fire,
      fireModel,
      settings.flamescapesClass,
      settings.flamescapesHorizon,
      settings.fireMaxSteps,
      settings.flamescapesFireLoad,
      settings.flamescapesBackend,
      settings.flamescapesGrid,
      settings.flamescapesDoors,
      settings.flamescapesWindows,
      settings.flamescapesVentilation,
      settings.flamescapesTerminalFlow,
    ]);
    return { payload, transfers, key };
  }, [graph, points.fire, settings, fireModel]);

  // Start the Flamescapes solver, or return the run already made for the
  // same graph and inputs. Its frames collect in run.frames as they arrive;
  // run.promise settles with the stacked frames.
  const runCombustion = useCallback(() => {
    const { payload, transfers, key } = combustionInputs();
    const current = combustionRef.current;
    if (current && current.key === key && current.graph === graph && !current.failed) return current;
    flamescapesRef.current?.cancel();
    const run = { key, graph, frames: [], done: false, failed: false, result: null, info: null };
    run.promise = flamescapesJob(payload, transfers, (progress) => {
      if (progress.type === "scene") {
        // A second scene is a restart on the CPU after the GPU stopped: the
        // frames so far came from the abandoned run (a playback waiting on
        // frame k resumes when the new run reaches it).
        if (progress.info.restarted) run.frames.length = 0;
        run.info = progress.info;
        // The view shares the GPU with the solver: it draws less while the GPU computes.
        viewerRef.current?.setDrawRate(progress.info.backend?.kind === "gpu" ? GPU_RUN_DRAW_RATE : 0);
        studio.setFire((c) => ({ ...c, combustion: { ...progress.info, time: 0, hrr: 0, peak: 0, horizon: payload.horizon, embersOn: payload.embers } }));
        return;
      }
      run.frames.push(progress);
      studio.setFire((c) => ({
        ...c,
        combustion: {
          ...(c.combustion || run.info),
          time: progress.time,
          hrr: progress.hrr,
          peak: Math.max(c.combustion?.peak || 0, progress.hrr),
          embers: progress.embers,
          spots: progress.spots,
          embersOn: payload.embers,
          horizon: payload.horizon,
        },
      }));
    }).then(
      (result) => {
        run.done = true;
        run.result = result;
        // The first measurement on a GPU stands, so its grid does not drift between runs.
        const name = result?.info?.backend?.name;
        if (result?.info?.gpuRate > 0 && name && gpuRateRef.current?.name !== name) {
          gpuRateRef.current = { name, rate: result.info.gpuRate };
          try {
            localStorage.setItem(GPU_RATE_KEY, JSON.stringify(gpuRateRef.current));
          } catch {
            // Private windows and blocked storage: the next session measures again.
          }
        }
        // A run replaced by a newer one leaves the newer one's draw rate alone.
        if (!combustionRef.current || combustionRef.current === run) viewerRef.current?.setDrawRate(0);
        return result;
      },
      (error) => {
        run.done = true;
        run.failed = true;
        if (!combustionRef.current || combustionRef.current === run) viewerRef.current?.setDrawRate(0);
        throw error;
      },
    );
    combustionRef.current = run;
    return run;
  }, [combustionInputs, flamescapesJob, graph, studio]);

  // The graph, the route and the fire's results per node, as the JSON
  // TopologicPy reads with Graph.ByJSONFile (src/core/topologicExport.js),
  // in the IFC's world frame.
  const exportTopologic = useCallback(async () => {
    const viewer = viewerRef.current;
    if (!graph) return;
    try {
      const [{ topologicGraph, frameChange, viewerToIfc }, coordination] = await Promise.all([
        import("./core/topologicExport.js"),
        viewer?.getCoordinationMatrix?.() ?? null,
      ]);
      const run = combustionRef.current;
      const flamescapesRun = isFlamescapes(fire.model) && run && run.done && !run.failed && run.graph === graph ? run : null;
      const shape = { points: graph.nodes, edges: graph.edges, kinds: graph.kinds, upAxis: graph.upAxis, agentHeight: graph.agentHeight };
      const origin = points.fire ? viewerToIfc(Float32Array.from(points.fire), frameChange(coordination, graph.upAxis || "y", graph.agentHeight || 0)) : null;
      const json = topologicGraph({
        graph: shape,
        coordination,
        route: path?.node_ids?.length ? path.node_ids : null,
        frames: flamescapesRun ? flamescapesRun.frames : null,
        burning: !flamescapesRun && fire.step > 0 && burningRef.current?.size ? burningRef.current : null,
        tenability: { untenableC: settings.flamescapesUntenableC, smoke: settings.flamescapesSmoke, visibilityLimit: settings.flamescapesVisibility },
        properties: {
          source_file: file?.name || null,
          exported_at: new Date().toISOString(),
          route_length_m: path?.length ?? null,
          fire_model: fire.step > 0 ? fire.model : null,
          fire_origin_ifc: origin ? Array.from(origin, (v) => Math.round(v * 1000) / 1000) : null,
          ...(flamescapesRun
            ? {
                fire_class: settings.flamescapesClass,
                horizon_s: settings.flamescapesHorizon,
                embers: fire.model === "flamescapes-embers" ? 1 : 0,
                doors: flamescapesRun.info?.openings?.doors ?? null,
                windows: flamescapesRun.info?.openings?.windows ?? null,
                ventilation: flamescapesRun.info?.openings?.ventilation ?? null,
                grid: flamescapesRun.info?.grid ?? null,
                cell_m: flamescapesRun.info?.cell ?? null,
                backend: flamescapesRun.info?.backend?.name ?? null,
              }
            : {}),
        },
      });
      const blob = new Blob([JSON.stringify(json)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${(file?.name || "model").replace(/\.ifc(\.gz)?$/i, "")}-topologicpy.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      toast(`Exported ${Object.keys(json.vertices).length.toLocaleString()} nodes${flamescapesRun ? " with the fire's results" : ""} for TopologicPy.`, { variant: "info", ttl: 3500 });
    } catch (error) {
      reportError(error, "Exporting for TopologicPy");
    }
  }, [graph, fire.model, fire.step, path, points.fire, file, settings, reportError, toast]);

  // What the elements are (src/core/identify.js): the IFC class where the
  // exporter typed it, else the name, else the box; then, when asked, IFCNet's
  // shape classifier for the generic elements the rules could only size up
  // (src/core/identifyGeometry.js). Both load only when asked for.
  const [identifyState, setIdentify] = useState({ busy: false, modelKey: null });
  const identification = identifyState.modelKey === modelInfo ? identifyState : null;
  const identifyDataRef = useRef(null);
  const identifyElements = useCallback(async () => {
    const viewer = viewerRef.current;
    if (!viewer || !modelInfo) return;
    setIdentify({ busy: true, modelKey: modelInfo });
    try {
      const [{ identify }, data] = await Promise.all([import("./core/identify.js"), viewer.identifyData()]);
      const answers = data.records.map((record) => ({ record, answer: identify(record) }));
      identifyDataRef.current = { modelKey: modelInfo, answers, meshesOf: data.meshesOf };
      setIdentify({ busy: false, modelKey: modelInfo, answers: answers.map((a) => a.answer) });
    } catch (error) {
      setIdentify({ busy: false, modelKey: null });
      reportError(error, "Identifying the elements");
    }
  }, [modelInfo, reportError]);
  const identifyByShape = useCallback(async () => {
    const data = identifyDataRef.current;
    if (!data || data.modelKey !== modelInfo) return;
    const open = data.answers.filter((a) => a.answer.by === "box" || a.answer.by === "none");
    if (!open.length) return;
    setIdentify((c) => ({ ...c, shapeBusy: true, shapeDone: 0, shapeTotal: open.length }));
    const { IFCNET_MODEL, combine, createGeometryIdentifier, samplePoints } = await import("./core/identifyGeometry.js");
    const classifier = createGeometryIdentifier({ modelUrl: `${import.meta.env.BASE_URL}${IFCNET_MODEL}` });
    try {
      let byShape = 0;
      for (let i = 0; i < open.length; i += 8) {
        const chunk = open.slice(i, i + 8).map((a) => ({ a, cloud: samplePoints(data.meshesOf.get(a.record.itemId), { seed: a.record.itemId }) })).filter((c) => c.cloud);
        const shapes = chunk.length ? await classifier.classify(chunk.map((c) => c.cloud)) : [];
        chunk.forEach(({ a }, k) => {
          const combined = combine(a.answer, shapes[k]);
          if (combined.by === "shape") byShape += 1;
          a.answer = combined;
        });
        if (data !== identifyDataRef.current) return;
        setIdentify((c) => ({ ...c, shapeDone: Math.min(open.length, i + 8) }));
      }
      setIdentify((c) => ({ ...c, shapeBusy: false, answers: data.answers.map((a) => a.answer), byShape, shapeRun: true }));
    } catch (error) {
      setIdentify((c) => ({ ...c, shapeBusy: false }));
      reportError(error, "Identifying elements by shape");
    } finally {
      classifier.terminate();
    }
  }, [modelInfo, reportError]);

  // Likely ignition points (src/core/ignition.js): the model's appliances,
  // furniture and rooms ranked by the ignition-source review.
  const [ignitionState, setIgnition] = useState({ busy: false, candidates: null, occupancy: "residential", graphKey: null });
  // A token per graph: candidates found for another graph are not shown, and
  // the old graph's arrays are not kept alive by them (review 13).
  const graphKey = useMemo(() => ({ graph: Boolean(graph) }), [graph]);
  const ignition = ignitionState.graphKey === graphKey ? ignitionState : { ...ignitionState, candidates: null };
  const findIgnition = useCallback(
    async (occupancy = ignition.occupancy) => {
      const viewer = viewerRef.current;
      if (!viewer || !graph) return;
      setIgnition((c) => ({ ...c, busy: true, occupancy }));
      try {
        const { records, rooms } = await viewer.ignitionData();
        // Rooms are placed at their meshes' centroid (extracted with the graph).
        const spaceMeshes = new Map();
        for (const mesh of geometryRef.current?.spaces || []) {
          if (!spaceMeshes.has(mesh.itemId)) spaceMeshes.set(mesh.itemId, []);
          spaceMeshes.get(mesh.itemId).push(mesh);
        }
        const placedRooms = rooms.map((r) => ({ ...r, position: centroidOf(spaceMeshes.get(r.itemId)) })).filter((r) => r.position);
        // The catalogue (about 35 KB) loads only when the list is asked for.
        const { rankIgnitionPoints } = await import("./core/ignition.js");
        const ranked = rankIgnitionPoints(records, placedRooms, { occupancy, limit: 8 });
        const positions = await viewer.positionsOf(ranked.filter((c) => !c.byRoom).map((c) => c.itemId));
        const candidates = ranked
          .map((c) => ({ ...c, position: c.byRoom ? c.position : positions.get(c.itemId) }))
          .filter((c) => c.position);
        setIgnition({ busy: false, candidates, occupancy, graphKey });
      } catch (error) {
        setIgnition((c) => ({ ...c, busy: false, candidates: [] }));
        reportError(error, "Finding likely ignition points");
      }
    },
    [graph, graphKey, ignition.occupancy, reportError],
  );
  const useIgnition = useCallback(
    (candidate) => {
      const [x, y, z] = candidate.position;
      const up = graph.upAxis === "z" ? 2 : graph.upAxis === "x" ? 0 : 1;
      let best = -1, bestD = Infinity;
      const P = graph.nodes;
      for (let v = 0; v < P.length / 3; v += 1) {
        const d = [P[v * 3] - x, P[v * 3 + 1] - y, P[v * 3 + 2] - z];
        d[up] *= 4;
        const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (dd < bestD) {
          bestD = dd;
          best = v;
        }
      }
      if (best < 0) return;
      handlePick("fire", [P[best * 3], P[best * 3 + 1], P[best * 3 + 2]]);
      // The first item takes the scenario's fire class (cooking: K, appliances: C).
      const cls = candidate.scenario.fireClass;
      if (FIRE_CLASSES[cls]) setSettings({ flamescapesClass: cls });
      toast(`Fire origin set at ${candidate.byRoom ? "the room" : "the element"} for: ${candidate.scenario.label}.`, { variant: "info", ttl: 4000 });
    },
    [graph, handlePick, setSettings, toast],
  );

  const startFire = useCallback(async () => {
    if (!graph || !points.fire) return;
    // A plan still running would land mid-fire and clear its route.
    if (planRunningRef.current) dropPlan();
    studio.stopFire();
    studio.setDynamicPath(null);
    viewerRef.current?.setPath(null, { dynamic: true });

    // The two-zone model is run first, in the forecast worker; the simulation
    // then plays its node temperatures, one frame per step over its horizon.
    let zoneFrames = null;
    if (fireModel === "zone") {
      try {
        const frames = Math.max(1, settings.fireMaxSteps);
        const { zone, transfers } = zoneInputs();
        const table = await forecastJob(
          {
            kind: "zoneFrames",
            graph: { nodes: graph.nodes, edges: graph.edges, blocked: graph.blocked },
            seed: snap(graph, points.fire),
            zone,
            frames,
            frameSeconds: settings.zoneHorizon / frames,
          },
          transfers,
        );
        zoneFrames = zoneNodeFrames(table.nodeRoom, table.table, table.rooms, table.frames);
      } catch (error) {
        // Cancelled: the worker was ended for a newer request, or the page.
        if (!isCancelled(error)) reportError(error, "The two-zone fire model");
        return;
      }
    }

    // Flames, embers and smoke of an earlier run go (review 7).
    viewerRef.current?.setParticles(null);
    viewerRef.current?.setSmokeCloud(null);
    // The Flamescapes solver streams its frames from its worker; the
    // simulation plays each as it arrives, waiting while the next is computed.
    let combustion = null;
    if (isFlamescapes(fireModel)) {
      try {
        combustion = runCombustion();
      } catch (error) {
        reportError(error, "The Flamescapes fire model");
        return;
      }
      combustion.promise.catch((error) => {
        if (!isCancelled(error)) reportError(error, "The Flamescapes fire model");
      });
      playingRunRef.current = combustion;
      let k = 0;
      zoneFrames = {
        next() {
          if (k < combustion.frames.length) return { value: combustion.frames[k++].T, done: false };
          return combustion.done ? { value: undefined, done: true } : { value: undefined, done: false };
        },
        [Symbol.iterator]() {
          return this;
        },
      };
    }

    burningRef.current = new Set();
    temperatureRef.current = new Map();
    studio.setFire((current) => ({
      running: true,
      step: 0,
      model: fireModel,
      combustion: combustion ? current.combustion : null,
    }));

    // Every model reroutes now: temperature by hazard cost, flood and radial by
    // treating burning nodes as impassable.
    const wantsReroute = settings.reroute && Boolean(points.start && points.exit);

    // Runs in the browser (src/core/fire.js), emitting the messages the
    // server's stream did, so no round trip per step and no backend needed.
    const handle = runFire(
      graph,
      {
        model: fireModel,
        zoneFrames,
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
            // Gas temperatures run far past heat diffusion's 120-degree index.
            // The Flamescapes frames can show the smoke instead, and their
            // flames and embers.
            const frame = combustion?.frames[message.step];
            if (frame && showRef.current === "smoke") viewer?.setSmoke(frame.vis);
            else viewer?.setTemperatures(temperatures, fireModel === "zone" || combustion ? { peak: ZONE_COLOUR_PEAK } : undefined);
            if (combustion) {
              viewer?.setParticles(frame?.particles || null);
              viewer?.setSmokeCloud(frame?.smoke || null, { strength: showRef.current === "smoke" ? 1 : SMOKE_IN_HEAT_VIEW });
            }
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
          // A Flamescapes run that failed has said so already (review 9).
          if (combustion?.failed) return;
          toast("Fire simulation finished.", { variant: "info", ttl: 3500 });
        },
        onError: (error) => {
          studio.setFire((current) => ({ ...current, running: false }));
          reportError(error, "Fire simulation");
        },
      },
    );
    studio.setFireHandle(handle);
  }, [graph, points, reportError, settings, studio, toast, fireModel, zoneInputs, forecastJob, dropPlan, colours, runCombustion]);

  // A fire that ends by itself keeps its final colours. Stopping it clears
  // them, back to what the graph showed before: the deadline map if it is
  // wanted, else the graph's own colours.
  const stopFire = useCallback(() => {
    studio.stopFire();
    // A Flamescapes run still computing is ended: it holds a CPU core or the GPU.
    if (combustionRef.current && !combustionRef.current.done) {
      flamescapesRef.current?.cancel();
      combustionRef.current = null;
    }
    const viewer = viewerRef.current;
    viewer?.setTemperatures(null);
    viewer?.setBurningNodes(null);
    viewer?.setParticles(null);
    viewer?.setSmokeCloud(null);
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

  const exits = useMemo(() => (graph?.doors ? exitDoors(graph.doors, studio.semantics, graph) : null), [graph, studio.semantics]);

  const trainRl = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    const shielded = settings.rlMethod !== "qlearning";
    // The shielded learner may escape by any exit door (rlSafe.js exitPoints).
    const exitPoints =
      shielded && settings.rlExits === "any" && exits?.nodes?.length
        ? exits.nodes.map((v) => [graph.nodes[v * 3], graph.nodes[v * 3 + 1], graph.nodes[v * 3 + 2]])
        : null;
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
          ...(shielded ? { method: "shielded", ...(exitPoints ? { exitPoints } : {}) } : {}),
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
        // No escape at all is a finding about the fire, not a failure of the agent.
        const noEscape = result.escape_steps == null && result.fire_start_id != null;
        toast(result.note || "The policy did not reach the exit.", {
          title: "RL training",
          variant: result.reached_exit || noEscape ? "info" : "warning",
        });
      }
    } catch (error) {
      studio.setRl({ busy: false, path: null, reachedExit: false, result: null });
      // Cancelled when the model or the graph changed under it.
      if (!isCancelled(error)) reportError(error, "RL training failed");
    }
  }, [graph, points, exits, reportError, settings, studio, toast, straighten, rlJob]);

  // -------------------------------------------------------------- exit doors

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

  // ------------------------------------------------- egress code check

  // Walking distance, exits per storey, separation, door widths and
  // high-rise, against IBC 2021's limits for offices, to be confirmed
  // against SBC 201 (core/codeCheck.js). Kept with the graph it was made on.
  const [codeState, setCodeState] = useState({ result: null, graph: null, shown: false });
  const code = codeState.graph === graph ? codeState : { result: null, graph: null, shown: false };
  const runCodeCheck = useCallback(() => {
    if (!graph) return;
    try {
      const result = codeCheck(graph, {
        exits: exits?.nodes || [],
        levels,
        exitDoors: (studio.semantics?.doors || []).filter((d) => d.exit),
        sprinklered: settings.codeSprinklered,
        useWalls: settings.useWalls,
      });
      setCodeState({ result, graph, shown: false });
    } catch (error) {
      reportError(error, "Code check");
    }
  }, [graph, exits, levels, studio.semantics, settings.codeSprinklered, settings.useWalls, reportError]);
  const showTravel = useCallback(
    (on) => {
      const viewer = viewerRef.current;
      const result = code.result;
      if (!viewer || !result) return;
      if (on) {
        // Distance to spare against the limit: red at or over it, yellow well within it.
        const spare = Array.from(result.travel, (d) => (Number.isFinite(d) ? result.limit - d : -Infinity));
        viewer.setDeadlines(spare, { max: result.limit });
        colours.set("travel");
      } else {
        viewer.setDeadlines(null);
        colours.set(null);
      }
      setCodeState((c) => ({ ...c, shown: on }));
    },
    [code.result, colours],
  );

  // ------------------------------------------------- describe this model

  // A plain description written by a language model from the facts Studio
  // has computed (core/describeFacts.js): on this device with Chrome's
  // built-in model, or by the Cloudflare Worker when the site has one
  // (lib/describe.js). The facts are shown before anything is sent.
  const [describe, setDescribe] = useState({ status: "checking", engine: null, busy: false, text: "", error: null, progress: null, model: null });
  const describeAbortRef = useRef(null);
  useEffect(() => {
    let live = true;
    onDeviceStatus().then((status) => {
      if (!live) return;
      const onDevice = status !== "unsupported" && status !== "unavailable";
      setDescribe((d) => ({ ...d, status, engine: d.engine ?? (onDevice || !CLOUD_URL ? "device" : "cloud") }));
    });
    return () => {
      live = false;
    };
  }, []);
  const describeFactsText = useMemo(
    () =>
      describeFacts({
        file,
        stats: modelInfo?.stats,
        semantics: studio.semantics,
        identification: identification?.answers,
        graph: graph ? { stats: graph.stats, cores: graph.cores } : null,
        route: path,
        walkSpeed: settings.walkSpeed,
        fire,
        combustion: fire.combustion,
        plan: forecastPlan?.result ? forecastPlan : null,
        rl: rl.result,
      }).text,
    [file, modelInfo, studio.semantics, identification, graph, path, settings.walkSpeed, fire, forecastPlan, rl.result],
  );
  const runDescribe = useCallback(async () => {
    describeAbortRef.current?.abort();
    const abort = new AbortController();
    describeAbortRef.current = abort;
    const engine = describe.engine || "device";
    setDescribe((d) => ({ ...d, busy: true, text: "", error: null, progress: null, model: engine === "device" ? "Chrome's built-in model" : null }));
    try {
      if (engine === "device") {
        await describeOnDevice(describeFactsText, {
          signal: abort.signal,
          onText: (text) => setDescribe((d) => ({ ...d, text })),
          onProgress: (progress) => setDescribe((d) => ({ ...d, progress })),
        });
      } else {
        const { text, model } = await describeInCloud(describeFactsText, { signal: abort.signal });
        setDescribe((d) => ({ ...d, text, model: `${model} on Cloudflare Workers AI` }));
      }
      setDescribe((d) => ({ ...d, busy: false, progress: null }));
    } catch (error) {
      if (abort.signal.aborted) setDescribe((d) => ({ ...d, busy: false }));
      else setDescribe((d) => ({ ...d, busy: false, error: error?.message || String(error) }));
    }
  }, [describe.engine, describeFactsText]);
  const stopDescribe = useCallback(() => describeAbortRef.current?.abort(), []);
  const setDescribeEngine = useCallback((engine) => setDescribe((d) => ({ ...d, engine })), []);

  // ------------------------------------------------- guided runs

  // The welcome screen's presets (core/guided.js) step through the app as a
  // user would and stop at the result. One stage at a time: each acts once,
  // then advances when its result appears (a new model, graph, point, route
  // or finished fire); Stop, or a stage that waits too long, ends the run.
  const [guided, setGuided] = useState(null);
  const startGuided = useCallback((run) => setGuided({ run, stage: "open", acted: null, before: null, error: null }), []);
  const stopGuided = useCallback(() => setGuided(null), []);
  // useIgnition is a callback, not a hook; the alias keeps the hooks lint from reading it as one.
  const applyIgnition = useIgnition;
  useEffect(() => {
    if (!guided || guided.stage === "done" || guided.error) return undefined;
    // Deferred a tick, so the stage's state changes happen outside the effect;
    // a re-run before then cancels it, so no stage acts twice.
    const timer = setTimeout(() => guidedStep(guided), 0);
    return () => clearTimeout(timer);
  });
  const guidedStep = (guided) => {
    const { run, stage, acted, before } = guided;
    const ids = stagesFor(run).map((s) => s.id);
    const advance = () => setGuided((g) => (g && g.run === run && g.stage === stage ? { ...g, stage: ids[ids.indexOf(stage) + 1], acted: null, before: null } : g));
    const act = (then, was) => {
      setGuided((g) => (g && g.run === run && g.stage === stage ? { ...g, acted: stage, before: was } : g));
      then();
    };
    const fail = (error) => setGuided((g) => (g && g.run === run ? { ...g, error } : g));
    const at = (v) => [graph.nodes[v * 3], graph.nodes[v * 3 + 1], graph.nodes[v * 3 + 2]];
    if (stage === "open") {
      if (acted !== stage) {
        const entry = examples.find((e) => e.id === run.example);
        if (!entry) return fail("The example is not available.");
        setTab("model");
        act(() => openExample(entry), modelInfo);
      } else if (modelInfo && modelInfo !== before && !studio.loadState.busy) advance();
    } else if (stage === "graph") {
      if (acted !== stage) {
        setTab("graph");
        act(() => void buildGraph(), graph);
      } else if (graph && graph !== before && !studio.graphBusy) advance();
    } else if (stage === "start") {
      if (acted !== stage) {
        const v = farthestFromExits(graph, exits?.nodes);
        if (v < 0) return fail("No room can be reached from an exit door.");
        setTab("route");
        act(() => handlePick("start", at(v)), points.start);
      } else if (points.start && points.start !== before) advance();
    } else if (stage === "exit") {
      if (acted !== stage) act(() => nearestExit(), points.exit);
      else if (points.exit && points.exit !== before) advance();
    } else if (stage === "route") {
      if (acted !== stage) act(() => void findPath(), path);
      else if (path && path !== before) {
        if (path.found === false) return fail("No route was found.");
        advance();
      }
    } else if (stage === "origin") {
      if (acted !== stage) {
        setTab("simulate");
        if (run.fire.model === "flamescapes") {
          setSettings({ flamescapes: true, fireModel: "flamescapes", flamescapesDoors: run.fire.doors || "open", flamescapesHorizon: run.fire.horizon || 60 });
          act(() => void findIgnition("residential"), points.fire);
        } else {
          setSettings({ flamescapes: false, fireModel: run.fire.model, reroute: true });
          const v = nodeAlong(path?.node_ids, run.fire.along ?? 0.5);
          if (v < 0) return fail("The route has no nodes to place the fire on.");
          act(() => handlePick("fire", at(v)), points.fire);
        }
      } else if (run.fire.model === "flamescapes" && !guided.picked && ignition.candidates) {
        if (!ignition.candidates.length) return fail("No likely ignition point was found in the model.");
        setGuided((g) => (g && g.run === run ? { ...g, picked: true } : g));
        applyIgnition(ignition.candidates[0]);
      } else if (points.fire && points.fire !== before && (run.fire.model !== "flamescapes" || guided.picked)) advance();
    } else if (stage === "fire") {
      if (acted !== stage) act(() => void startFire(), null);
      else if (fire.running && !guided.sawRunning) setGuided((g) => (g && g.run === run ? { ...g, sawRunning: true } : g));
      else if (guided.sawRunning && !fire.running && fire.step > 0) advance();
    }
  };
  // A stage that waits too long ends the run (a fire on the GPU or a CPU core can take minutes).
  const guidedStage = guided && !guided.error && guided.stage !== "done" ? guided.stage : null;
  useEffect(() => {
    if (!guidedStage) return undefined;
    const limit = guidedStage === "fire" ? 15 * 60000 : 4 * 60000;
    const timer = setTimeout(() => setGuided((g) => (g && g.stage === guidedStage ? { ...g, error: "This step took too long." } : g)), limit);
    return () => clearTimeout(timer);
  }, [guidedStage]);

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
    if (planRunningRef.current) forecastRef.current?.cancel();
    const gen = ++planGenRef.current;
    // What the card describes: the values this plan used (review M2).
    const inputs = planSettings(settings);
    studio.setForecastPlan((current) => ({ ...current, busy: true }));
    try {
      // The two-zone model brings its rooms and fire, and is untenable from a
      // gas temperature; the graph models from a model-degree threshold.
      const zone = fireModel === "zone" ? zoneInputs() : null;
      planRunningRef.current = true;
      // The Flamescapes models plan on the solver's frames: the run already
      // made for these inputs, or a new one, waited for.
      let frames = null;
      if (isFlamescapes(fireModel)) {
        const run = runCombustion();
        let done;
        try {
          done = await run.promise;
        } catch (error) {
          // The run was stopped (Stop, or a newer run): this plan ends
          // with it rather than waiting for ever.
          if (isCancelled(error) && gen === planGenRef.current) {
            planRunningRef.current = false;
            studio.setForecastPlan(NO_PLAN);
            return;
          }
          throw error;
        }
        if (gen !== planGenRef.current) return;
        // Stacked here from the run's frames, sent as copies (the worker no
        // longer sends them twice: review 6).
        const { n, K } = done;
        const T = new Float32Array(K * n), vis = new Float32Array(K * n), co = new Float32Array(K * n);
        run.frames.forEach((f, k) => {
          T.set(f.T, k * n);
          vis.set(f.vis, k * n);
          co.set(f.co, k * n);
        });
        frames = {
          n,
          K,
          frameSeconds: done.frameSeconds,
          T,
          vis,
          co,
          smoke: settings.flamescapesSmoke,
          visibilityLimit: settings.flamescapesVisibility,
          // What the grid covered, for the plan card (review 3).
          clipped: Boolean(run.info?.clipped),
          inGrid: done.inGrid / Math.max(1, n),
        };
      }
      const result = await forecastJob(
        {
          graph: { nodes: graph.nodes, edges: graph.edges, blocked: graph.blocked },
          start: snap(graph, points.start),
          exits: exitNodes,
          seed: snap(graph, points.fire),
          model: fireModel,
          steps: settings.fireMaxSteps,
          secondsPerStep: settings.forecastSecondsPerStep,
          threshold: zone ? settings.zoneUntenableC : frames ? settings.flamescapesUntenableC : settings.untenableAbove,
          speed: settings.walkSpeed,
          depart: settings.departAfter,
          limit: settings.doseLimit,
          useWalls: settings.useWalls,
          // The map is the slow part; it is made only when it will be drawn.
          withMap: settings.showDeadlineMap,
          zone: zone?.zone,
          frames,
        },
        zone?.transfers || (frames ? [frames.T.buffer, frames.vis.buffer, frames.co.buffer] : []),
      );
      // Inputs changed (or another plan began) while it ran: not this one's.
      if (gen !== planGenRef.current) return;
      planRunningRef.current = false;
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
      planRunningRef.current = false;
      studio.setForecastPlan(NO_PLAN);
      reportError(error, "Planning with the forecast failed");
    }
  }, [graph, points, settings, studio, exits, forecastJob, pointsOf, toast, reportError, showPlanRoutes, fireModel, zoneInputs, runCombustion]);

  // A Flamescapes run belongs to the graph it was made on: a new graph ends
  // it and lets its frames go.
  useEffect(() => {
    if (combustionRef.current && combustionRef.current.graph !== graph) {
      if (!combustionRef.current.done) flamescapesRef.current?.cancel();
      combustionRef.current = null;
      playingRunRef.current = null;
      viewerRef.current?.setParticles(null);
      viewerRef.current?.setSmokeCloud(null);
      viewerRef.current?.setDrawRate(0);
    }
  }, [graph]);

  // Switching a Flamescapes run between heat and smoke redraws the frame on
  // screen, also once the run has ended.
  useEffect(() => {
    showRef.current = settings.flamescapesShow;
    const run = playingRunRef.current;
    if (!run || !isFlamescapes(fire.model) || graphColours !== "fire") return;
    const frame = run.frames[Math.min(fire.step, run.frames.length - 1)];
    if (!frame) return;
    const viewer = viewerRef.current;
    viewer?.setSmokeCloud(frame.smoke || null, { strength: settings.flamescapesShow === "smoke" ? 1 : SMOKE_IN_HEAT_VIEW });
    if (settings.flamescapesShow === "smoke") viewer?.setSmoke(frame.vis);
    else {
      const temperatures = new Map();
      for (let i = 0; i < frame.T.length; i += 1) if (frame.T[i] > 21) temperatures.set(i, frame.T[i]);
      viewer?.setTemperatures(temperatures, { peak: ZONE_COLOUR_PEAK });
    }
    // Only a change of view redraws; frames redraw themselves as they play.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.flamescapesShow]);

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
      if (graph.stats.lift_nodes) {
        items.push({ label: "Lifts", color: "var(--viz-lift)" });
      }
    }
    // Flood and distance-band fires colour burning nodes; the others, heat.
    if (graphColours === "fire" && (fire.model === "flood" || fire.model === "radial")) {
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
            // The legends follow the layer colouring the graph, which a fire
            // keeps after it ends (review M6).
            heatLegend={
              graphColours === "fire" &&
              (fire.model === "zone"
                ? {
                    title: "Gas temperature (°C)",
                    max: ZONE_COLOUR_PEAK,
                    note: "Two-zone model: the smoke layer's temperature where it is at or below head height; ambient elsewhere.",
                  }
                : isFlamescapes(fire.model)
                  ? settings.flamescapesShow === "smoke"
                    ? {
                        title: "Visibility at head height",
                        ramp: "smoke",
                        scale: ["30 m", "5 m", "1 m"],
                        note: "Flamescapes: seeing through the smoke 1.8 m above each node's floor; clear air keeps the graph's colour.",
                      }
                    : {
                        title: "Gas temperature at head height (°C)",
                        max: ZONE_COLOUR_PEAK,
                        note: "Flamescapes: the solver's gas temperature 1.8 m above each node's floor.",
                      }
                  : fire.model === "temperature")
            }
            deadlineLegend={
              graphColours === "deadlines"
                ? { max: planHorizon }
                : graphColours === "travel" && code.result
                  ? {
                      max: code.result.limit,
                      title: "Walk to an exit: distance to spare",
                      unit: "m",
                      note: `Red: at or over the ${code.result.limit.toFixed(1)} m travel distance (IBC 2021 Table 1017.2, to be confirmed against SBC 201).`,
                      noneNote: "Grey: reaches no exit on its storey.",
                    }
                  : null
            }
            onStopFire={stopFire}
            legend={legend}
            examples={examples}
            onExample={openExample}
            onIntent={warmImport}
            guidedRuns={GUIDED_RUNS}
            onGuided={startGuided}
            guided={guided ? { title: guided.run.title, stages: stagesFor(guided.run), stage: guided.stage, error: guided.error } : null}
            onStopGuided={stopGuided}
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
                  identification={identification}
                  onIdentify={identifyElements}
                  onIdentifyByShape={identifyByShape}
                  describe={{ ...describe, facts: describeFactsText, cloud: Boolean(CLOUD_URL) }}
                  onDescribe={runDescribe}
                  onStopDescribe={stopDescribe}
                  onDescribeEngine={setDescribeEngine}
                  onOpenRooms={handleFile}
                  onExportRooms={exportLod1}
                />
              )}
              {activeTab === "graph" && (
                <GraphPanel studio={studio} onBuildGraph={buildGraph} onNavigationCheck={runNavigationCheck} onExportTopologic={exportTopologic} />
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
                  code={{ ...code, shown: code.shown && graphColours === "travel" }}
                  onCodeCheck={runCodeCheck}
                  onShowTravel={showTravel}
                />
              )}
              {activeTab === "simulate" && (
                <SimulatePanel
                  studio={studio}
                  onPick={requestPick}
                  onPlace={handlePick}
                  ignition={ignition}
                  onFindIgnition={findIgnition}
                  onUseIgnition={useIgnition}
                  onStartFire={startFire}
                  onStopFire={stopFire}
                  onTrainRl={trainRl}
                  onPlanForecast={planForecast}
                  deadlinesShown={graphColours === "deadlines"}
                  exits={exits}
                  onGoTo={setTab}
                  onExportTopologic={exportTopologic}
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
