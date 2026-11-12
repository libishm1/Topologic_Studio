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

import { api, isGraphGone, legacyApi } from "./lib/api.js";
import { PhaseTimer, formatBytes, formatMs } from "./lib/perf.js";
import { CONVERTER } from "./lib/fragmentCache.js";
import { isPhone } from "./lib/hardware.js";
import { isCancelled, workerClient } from "./lib/workerJob.js";
import { FORECAST_COLORS, NAV_COLOR, hexToInt, routeColor } from "./lib/routeColors.js";
import { graphIsSplit, graphStats, reachablePct } from "./core/graph.js";
import { findRoute, snap } from "./core/route.js";
import { VOID_NAME, floorsFromSpaces, slabsMissing } from "./core/spaceFloors.js";
import { DETECTOR_VISIBILITY, buildForecast, earliestArrival, firstDetection, forecastFromFrames, runRange, runSummary, stackFrames } from "./core/forecast.js";
import { exitDoors } from "./core/ifcSemantics.js";
import { healthReason } from "./core/runHealth.js";
import { solverOptionsFor } from "./core/solverVersions.js";
import { KIND_DOOR, floorLevels } from "./core/lattice.js";
import { straightenRoute, withStraight } from "./core/straighten.js";
import { alternativeRoutes } from "./core/alternatives.js";
import { runFire } from "./core/fire.js";
import { RUN_CACHE_VERSION, clearRuns, fingerprint, readRun, storedRuns, writeRun } from "./core/runCache.js";
import { closedDoorIds, doorsLabel } from "./core/doorStates.js";
import { zoneNodeFrames } from "./core/zone.js";
import { RL_MAX_STEPS } from "./core/rl.js";
import { RUN_SETTINGS, fireModelIn, isFlamescapes, keepGuidedBefore, planSettings, useStudio } from "./state/useStudio.js";
import { runHorizon } from "./core/fireService.js";
import { validationVerdict } from "./core/calibrationStatus.js";
import { FIRE_CLASSES, designFireFor, fuelsFor } from "./core/fireClasses.js";
import { centroidOf } from "./core/centroid.js";
import { GUIDED_HOLD_MS, GUIDED_RUNS, farthestFromExits, fireStageNext, guidedHeadline, guidedOrigin, guidedResultPlace, nodeAlong, stagesFor } from "./core/guided.js";
import { withoutSessionOnly } from "./core/settingsMigration.js";
import { describeFacts } from "./core/describeFacts.js";
import { EGRESS_B, codeCheck, storeysOf } from "./core/codeCheck.js";
import * as reportPages from "./core/perfReport.js";
import * as rsetPresets from "./core/rsetPresets.js";
import { fittedHeatPeak, measureView } from "./core/measures.js";
import { runIndex } from "./core/tenability.js";
import { fieldNote, nextHint } from "./core/fieldNotes.js";
import { useFieldNotes } from "./state/useFieldNotes.js";
import { FieldNotesButton, FieldNotesDialog, HintCard } from "./ui/FieldNotes.jsx";
import { CLOUD_URL, describeInCloud, describeOnDevice, onDeviceStatus } from "./lib/describe.js";
import { humanToken } from "./lib/turnstile.js";
import { GraphPanel, ModelPanel, RoutePanel, SimulatePanel } from "./ui/Panels.jsx";
import { SECTION_ABOVE_FLOOR } from "./viewer/sectionConstants.js";
import { Stepper } from "./ui/Stepper.jsx";
import { TopologyPanel } from "./ui/TopologyPanel.jsx";
import { Viewport } from "./ui/Viewport.jsx";
import { revealSections } from "./ui/reveal.js";

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
// v2 (2026-11-12): the whole run's speed, cut steps and all (the worker);
// speeds kept before were the first frames' on the old solver, and are dropped.
const GPU_RATE_KEY = "topologic-studio.flamescapes.gpuRate.v2";
// The recording while its module loads (a click then does nothing).
const RECORDER_STARTING = { cancel() {} };

/**
 * The hottest gas at head height in a run's frames so far, °C, or null: each
 * frame is read once and the answer kept with the run (a restart's frames
 * start it again, runCombustion). For the heat ramp fitted to the run.
 */
function runHotGas(run) {
  if (!run?.frames) return null;
  if (!(run.hotRead <= run.frames.length)) {
    run.hotRead = 0;
    run.hotGas = -Infinity;
  }
  for (let k = run.hotRead; k < run.frames.length; k += 1) {
    const T = run.frames[k].T;
    for (let i = 0; i < T.length; i += 1) if (T[i] > run.hotGas) run.hotGas = T[i];
  }
  run.hotRead = run.frames.length;
  return Number.isFinite(run.hotGas) ? run.hotGas : null;
}

/** No slime mould grown (core/slimeEgress.js). */
const SLIME_NONE = { nodes: null, result: null, busy: false, error: null };

/**
 * Each graph node's door width, m (0 where no door): the IFC's declared
 * width where it gives one, else the width the graph measured.
 */
function slimeDoorWidths(graph, semantics) {
  const widths = new Float32Array(graph.nodes.length / 3);
  const declared = new Map((semantics?.doors || []).map((d) => [d.id, d.widthM]));
  for (const d of graph.doors || []) {
    if (!(d.node >= 0)) continue;
    const w = declared.get(d.itemId) > 0 ? declared.get(d.itemId) : d.width;
    if (w > 0) widths[d.node] = w;
  }
  return widths;
}
// How long a step is on screen, with nothing running, before its hint shows.
const HINT_SETTLE_MS = 2500;
// The code's occupancy group for the performance check's occupants setting
// (sprinkler requirement, extinguishers, suppression systems by room use).
const OCCUPANCY_GROUP = { office: "B", classroom: "university", apartment: "R-2", "apartment-origin": "R-2", hotel: "R-1", assembly: "A-3" };

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
// The performance-based check loads when first used; its presets and report
// are small and already on the page with the panel.
const loadPerf = () => import("./core/performanceCheck.js").then((check) => [check, rsetPresets, reportPages]);

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
const NO_LINKED = [];

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
  // The forecast worker's lanes, one worker each: plan, zone, report.
  const forecastRef = useRef({});
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
  // How fast the last Flamescapes run computed in this visit (simulated s,
  // wall ms, grid set-up ms, its backend and grid), for the estimate beside
  // the run's length (core/fireService.js computeSeconds).
  const [lastPace, setLastPace] = useState(null);
  // The run whose frames are on screen (playback reads its view from a ref,
  // so switching heat and smoke mid-run holds: review 4).
  const playingRunRef = useRef(null);
  // Counts fire starts and stops: a start still waiting for its two-zone
  // frames goes on only if nothing came after it.
  const fireStartRef = useRef(0);
  // Whether the last fire was stopped rather than ending by itself.
  const fireStoppedRef = useRef(false);
  // The load a graph build is running for (buildGraph), or null.
  const buildingRef = useRef(null);
  // Ends a two-zone fire still computing its frames: Stop, a new model, a
  // rebuilt graph or the viewer closed (SM-2, SM-7).
  const endFireStart = useCallback(() => {
    fireStartRef.current += 1;
    forecastRef.current.zone?.cancel();
  }, []);
  // What a Flamescapes run colours the graph by (core/measures.js), and the
  // limits its tenability index is worked out with (core/tenability.js).
  const showRef = useRef(measureView({}, false));
  const tenabilityRef = useRef({ limitC: 60, visibilityLimit: 10, smoke: true });
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
  // On a phone the panel takes half the screen; "Taller panel" gives it most
  // of the screen for a long step and hands it back (workflow.css).
  const [panelTall, setPanelTall] = useState(false);
  const [methodsOpen, setMethodsOpen] = useState(false);
  // Field notes and hints (core/fieldNotes.js): found from the app's own
  // results where they appear below, kept in this browser.
  const notes = useFieldNotes();
  const { markFound, markUsed } = notes;
  const [notesOpen, setNotesOpen] = useState(false);
  const [MethodsDialog, setMethodsDialog] = useState(loadMethods);
  // Storey heights for the section cut, and the chosen one (null = whole).
  const [levels, setLevels] = useState([]);
  // The model's coordination shift (m): a height in the viewer frame is the IFC's plus this.
  const [levelShift, setLevelShift] = useState(0);
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
      Object.values(forecastRef.current).forEach((lane) => lane.cancel());
      rlRef.current?.cancel();
      flamescapesRef.current?.cancel();
    },
    [],
  );

  // Planning with the forecast has its own worker: its deadline map takes a
  // few seconds on a large graph and must not hold up a graph rebuild. The
  // plan, the two-zone fire's frames and the report's fires each have a lane
  // of their own, since ending a worker ends every job in it: one shared, the
  // report's Stop left the plan busy for good and a second Plan click lost a
  // starting fire (wiki/architecture/state-machines.md, SM-2).
  const forecastJob = useCallback((payload, transfers = [], lane = "plan") => {
    forecastRef.current[lane] ||= workerClient(
      () => new Worker(new URL("./workers/forecast.worker.js", import.meta.url), { type: "module" }),
      { failure: "Planning with the forecast stopped: its worker did not load or ran out of memory. Reload the page and try again." },
    );
    return forecastRef.current[lane].run({ payload }, transfers);
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
   *
   * keepRun: the fire about to start plays the plan's Flamescapes run, so it
   * is kept (its start reuses it, or ends it if its inputs differ).
   */
  const dropPlan = useCallback((opts) => {
    planGenRef.current += 1;
    if (planRunningRef.current) {
      planRunningRef.current = false;
      forecastRef.current.plan?.cancel();
      // A Flamescapes run started for this plan, and not on screen, is ended
      // too (review 8), unless the fire about to start plays it: ended, it
      // was computed again from the start (SM-1).
      const run = combustionRef.current;
      if (run && !run.done && run !== playingRunRef.current && opts?.keepRun !== true) {
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

  // --------------------------------------------- Flamescapes frames on screen

  // The tenability index of frame k of a run (core/tenability.js): worked
  // out in order and kept with the run, again from the start when the limits
  // change, or when a restarted run replaced its frames.
  const tenabilityOf = useCallback((run, k) => runIndex(run, k, tenabilityRef.current), []);

  // Frame k of a run in what is on show (core/measures.js): heat on the
  // run's ramp, the smoke measures by visibility (each is a function of the
  // soot), the gases and the index as levels.
  const paintCombustion = useCallback((viewer, run, k) => {
    const frame = run?.frames[Math.min(k, run.frames.length - 1)];
    if (!viewer || !frame) return;
    const view = showRef.current;
    viewer.setSmokeCloud(frame.smoke || null, { strength: view.kind === "smoke" ? 1 : SMOKE_IN_HEAT_VIEW });
    // The heads open by this frame (core/suppression.js).
    if (run.sprinklerPoints) viewer.setSprinklers?.(run.sprinklerPoints, new Set(frame.sprinklers?.open || []), run.sprinklerDrops);
    // The rooms flooded or foamed by this frame (core/suppressionSystems.js).
    if (run.systemRooms) viewer.setSystemRooms?.(run.systemRooms, new Set(frame.systems?.discharged || []), run.systemUnits);
    if (view.kind === "smoke") viewer.setSmoke(frame.vis);
    else if (view.kind === "levels") viewer.setNodeLevels(view.levels(frame, view.key === "tenability" ? tenabilityOf(run, k) : null));
    else {
      const temperatures = new Map();
      for (let i = 0; i < frame.T.length; i += 1) if (frame.T[i] > 21) temperatures.set(i, frame.T[i]);
      // On the fixed ramp, or one fitted to the run (drawing only).
      viewer.setTemperatures(temperatures, { peak: view.fit ? fittedHeatPeak(runHotGas(run)) : ZONE_COLOUR_PEAK });
    }
  }, [tenabilityOf]);

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
      // An IFC still loading is dropped, its progress card with it, and a
      // graph build or a fire start for the old model ends (SM-7).
      ++loadSeqRef.current;
      studio.setLoadState({ busy: false, stage: null, detail: null });
      samplerRef.current?.cancel();
      endFireStart();
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
    [colours, dropPlan, reportError, setSettings, settings.modelAppearance, settings.upAxis, studio, toast, endFireStart],
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
        // An example's "Downloading" card goes too (SM-7).
        studio.setLoadState({ busy: false, stage: null, detail: null });
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

      // A graph build or a fire start for the old model ends with it (SM-7).
      samplerRef.current?.cancel();
      endFireStart();
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
            setLevelShift(shift);
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
        setLevelShift(0);
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

        // A converted or cached file says how long it took; an example opened
        // from its prepared fragments says nothing (the Model step gives its
        // time, and the toast covered the view: the interface audit of 2026-11-12).
        if (!result.prepared) {
          toast(result.cached ? `Loaded from cache in ${formatMs(result.totalMs)}.` : `Converted and cached in ${formatMs(result.totalMs)}.`, {
            title: nextFile.name,
            variant: "success",
            ttl: 5000,
          });
        }
      } catch (error) {
        if (latest() && !isCancelled(error)) {
          reportError(error, "Could not load the IFC file");
        }
      } finally {
        if (latest()) studio.setLoadState({ busy: false, stage: null, detail: null });
      }
    },
    [reportError, settings.modelAppearance, settings.upAxis, studio, toast, dropPlan, colours, openLod1Files, endFireStart],
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
    // One build at a time for a model: the "b" key started one per key
    // repeat. And the build is for the model it started on: one still running
    // when another model was opened set its graph on the new model
    // (wiki/architecture/state-machines.md, SM-7).
    const seq = loadSeqRef.current;
    if (buildingRef.current === seq) return;
    buildingRef.current = seq;
    const latest = () => seq === loadSeqRef.current;

    studio.setGraphBusy(true);
    studio.setLoadState({ busy: true, stage: "extracting", detail: null });
    // A policy still training is for the graph being replaced.
    rlRef.current?.cancel();
    const timer = new PhaseTimer("graph.build");

    try {
      // Geometry extraction is the expensive part, so it is cached against the
      // loaded model: changing a slider re-samples but does not re-extract.
      if (!geometryRef.current) {
        const extracted = await timer.run(
          "extract",
          async () => (await geometryPromiseRef.current) || viewer.extractEgressGeometry(),
        );
        if (!latest()) return;
        geometryRef.current = extracted;
      }
      const geometry = geometryRef.current;

      studio.setLoadState({ busy: true, stage: "sampling", detail: null });

      // Spaces outside the building that no door opens onto (roofs drawn as
      // rooms) and voids are not floor to walk on (core/outdoorSpaces.js).
      let outdoor = [];
      if (geometry.spaces?.length) {
        const { outdoorSpaces } = await import("./core/outdoorSpaces.js");
        if (!latest()) return;
        const names = new Map((studio.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
        outdoor = await timer.run("outdoor", async () =>
          outdoorSpaces(geometry.spaces, {
            names,
            overhead: [...(geometry.floors || []), ...(geometry.roofMeshes || [])],
            ceilings: geometry.ceilings || [],
            walls: geometry.walls,
            doors: geometry.doors,
            upAxis: settings.upAxis,
          }),
        );
        if (!latest()) return;
      }
      const leftOut = new Set(outdoor.map((s) => s.id));

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
        spaces: clone((geometry.spaces || []).filter((mesh) => !leftOut.has(mesh.itemId))),
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
      if (!latest()) return;
      setLastSample({ nodes: built.stats.nodes, ms: built.stats.ms });
      const local = {
        ...built,
        upAxis: settings.upAxis,
        agentHeight: settings.agentHeight,
      };

      // Only the TopologicPy features need the graph on the server, so it is
      // uploaded when one of them is first asked for (serverGraphId, below).
      const response = {
        // What the upload sends (lib/api.js importGraph).
        forServer: local,
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
        // The spaces left out of the floors (core/outdoorSpaces.js), for the Graph step.
        outdoorSpaces: outdoor.map(({ id, name, why }) => ({ id, name, why })),
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
      if (graphIsSplit(response.stats)) markFound("cut-off", { largest: reachablePct(response.stats) });
      studio.setPath(null);
      studio.setDynamicPath(null);
      studio.setAlternatives(null);
      studio.setComparison(null);
      studio.setNavCheck(null);
      dropPlan();
      // A fire still running spreads over the old graph's node ids, and one
      // still starting would start on them. A fire that ended left its step
      // and burning nodes, which the export then wrote onto the new graph's
      // vertices; the agent's answer was for the old graph too (SM-7).
      studio.stopFire();
      endFireStart();
      studio.setFire({ running: false, step: 0, model: null });
      burningRef.current = new Set();
      temperatureRef.current = new Map();
      studio.setRl({ busy: false, path: null, reachedExit: false, result: null });
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
      // Ended because another model was opened: nothing to report.
      if (latest() && !isCancelled(error)) reportError(error, "Could not build the navigation graph");
    } finally {
      // A newer build, or the newer model's load, owns the busy flags now.
      if (buildingRef.current === seq) {
        buildingRef.current = null;
        studio.setGraphBusy(false);
      }
      if (latest()) studio.setLoadState({ busy: false, stage: null, detail: null });
    }
  }, [reportError, sample, setSettings, settings, studio, toast, dropPlan, colours, markFound, endFireStart]);

  /**
   * Server-backed actions need the backend up. They always name the graph on
   * screen (serverGraphId): without an id the server would fall back to
   * whatever graph it built last, which may be another model.
   */
  const needsServerGraph = useCallback(
    (what) => {
      if (studio.serverStatus === "online") return false;
      toast(
        studio.serverStatus === "offline"
          ? `${what} still runs on the backend, which is not connected. The graph itself was built in your browser.`
          : `${what} runs on the backend, which is still waking up. Try again in a moment.`,
        { title: "Backend needed", variant: "warning", ttl: 8000 },
      );
      return true;
    },
    [studio.serverStatus, toast],
  );

  // The server's copy of the graph on screen, kept beside the graph rather
  // than in it. Written into the graph, the upload's answer made a new graph
  // object, and everything kept "with the graph it was made on" (a fire, the
  // code check, the report, the plans) took it for a new graph and ended or
  // dropped itself; and a graph built while the backend was waking was never
  // uploaded (wiki/architecture/state-machines.md, SM-4).
  const serverGraphRef = useRef({ nodes: null, id: null, pending: null });
  const serverGraphId = useCallback(
    (again = false) => {
      const held = serverGraphRef.current;
      if (held.nodes === graph.nodes && !again) {
        if (held.id) return Promise.resolve(held.id);
        if (held.pending) return held.pending;
      }
      const nodes = graph.nodes;
      const pending = api.importGraph(graph.forServer).then(
        (uploaded) => {
          if (serverGraphRef.current.pending === pending) serverGraphRef.current = { nodes, id: uploaded.graph_id, pending: null };
          return uploaded.graph_id;
        },
        (error) => {
          if (serverGraphRef.current.pending === pending) serverGraphRef.current = { nodes: null, id: null, pending: null };
          throw error;
        },
      );
      // The copy of a graph no longer on screen goes: the server keeps only a few.
      if (held.id && held.nodes !== nodes) api.deleteGraph(held.id).catch(() => {});
      serverGraphRef.current = { nodes, id: null, pending };
      return pending;
    },
    [graph],
  );
  // A request on the graph on screen: uploaded first if the server has no
  // copy, and once more if the server lost it (it sleeps and restarts empty,
  // and keeps a few graphs for an hour) (SM-4).
  const onServerGraph = useCallback(
    async (ask) => {
      try {
        return await ask(await serverGraphId());
      } catch (error) {
        if (!isGraphGone(error)) throw error;
        return ask(await serverGraphId(true));
      }
    },
    [serverGraphId],
  );
  // What a server answer must still match when it lands: the graph, the
  // points and the model on screen. An answer for older ones was drawn over
  // the newer (SM-5).
  const askedRef = useRef(null);
  askedRef.current = { nodes: graph?.nodes ?? null, start: points.start, exit: points.exit, file };
  const sameRoute = useCallback((asked) => asked.nodes === askedRef.current.nodes && asked.start === askedRef.current.start && asked.exit === askedRef.current.exit, []);
  const sameModel = useCallback((asked) => asked.nodes === askedRef.current.nodes && asked.file === askedRef.current.file, []);

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
      // So is the agent's training, and its answer: still training, it drew
      // a route for the old points when it ended (SM-3).
      if (mode !== "fire" || settings.rlUseFire) {
        rlRef.current?.cancel();
        if (rl.busy || rl.result) {
          studio.setRl({ busy: false, path: null, reachedExit: false, result: null });
          if (rl.path) viewerRef.current?.setPath(null, { dynamic: true });
        }
      }
      if (mode !== "fire") {
        studio.setPath(null);
        studio.setAlternatives(null);
        studio.setComparison(null);
        viewerRef.current?.setPath(null);
        viewerRef.current?.setRoutes(null);
      }
    },
    [snapToGraph, studio, dropPlan, settings.rlUseFire, rl],
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
    studio.setComparison(null);
    viewerRef.current?.setPath(null);
    viewerRef.current?.setPath(null, { dynamic: true });
    viewerRef.current?.setRoutes(null);
  }, [studio, dropPlan]);

  // The markers follow the points, whichever way they changed: opening
  // another model cleared the points and left the old start, exit and fire
  // drawn on it (wiki/architecture/state-machines.md, SM-7).
  useEffect(() => {
    for (const name of ["start", "exit", "fire"]) viewerRef.current?.setMarker(name, points[name]);
  }, [points]);

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
    const asked = askedRef.current;
    try {
      const result = onServer
        ? await onServerGraph((graphId) =>
            api.findPath({
              graph_id: graphId,
              start_point: points.start,
              end_point: points.exit,
              engine: studio.effectiveEngine,
              use_walls: settings.useWalls,
            }),
          )
        : findRoute(graph, points.start, points.exit, { useWalls: settings.useWalls });
      if (!sameRoute(asked)) return;
      const shown = straighten(result);
      studio.setPath(shown);
      viewerRef.current?.setPath(drawn(shown));
      if (result.found && shown.length > 0) markFound("walk-out", { length: shown.length });
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
  }, [graph, points, reportError, settings.useWalls, studio, toast, needsServerGraph, onServerGraph, sameRoute, straighten, showPlanRoutes, markFound]);

  const comparePath = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    if (needsServerGraph("Engine comparison")) return;
    studio.setPathBusy(true);
    const asked = askedRef.current;
    try {
      const result = await onServerGraph((graphId) =>
        api.comparePath({
          graph_id: graphId,
          start_point: points.start,
          end_point: points.exit,
          use_walls: settings.useWalls,
        }),
      );
      if (!sameRoute(asked)) return;
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
  }, [graph, points, reportError, settings.useWalls, studio, toast, needsServerGraph, onServerGraph, sameRoute, straighten]);

  const findAlternatives = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    // In the browser unless the TopologicPy engine is chosen (its Yen is
    // TGraph.KShortestPaths on the server).
    const onServer = studio.effectiveEngine === "topologicpy";
    if (onServer && needsServerGraph("Alternative routes on the TopologicPy engine")) return;
    studio.setPathBusy(true);
    studio.setComparison(null);
    const asked = askedRef.current;
    try {
      const raw = onServer
        ? await onServerGraph((graphId) =>
            api.alternatives({
              graph_id: graphId,
              start_point: points.start,
              end_point: points.exit,
              engine: studio.effectiveEngine,
              use_walls: settings.useWalls,
              k: settings.altCount,
              method: settings.altMethod,
            }),
          )
        : alternativeRoutes(graph, points.start, points.exit, {
            k: settings.altCount,
            method: settings.altMethod,
            useWalls: settings.useWalls,
          });
      if (!sameRoute(asked)) return;
      const result = { ...raw, routes: raw.routes.map(straighten) };
      studio.setAlternatives(result);
      markUsed("alternatives");
      if (result.routes.length >= 2) markFound("second-way", { first: result.routes[0].length, second: result.routes[1].length });
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
  }, [graph, points, reportError, settings, studio, toast, needsServerGraph, onServerGraph, sameRoute, straighten, showPlanRoutes, markFound, markUsed]);

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
    const asked = askedRef.current;
    try {
      // An example opened from its prepared fragments fetches its IFC now.
      const ifc = file.download ? await file.download() : file;
      const coordination = await viewer.getCoordinationMatrix();
      const result = await onServerGraph((graphId) =>
        api.navigationCheck(ifc, {
          graphId,
          spacing: settings.navSpacing,
          startPoint: points.start,
          endPoint: points.exit,
          useWalls: settings.useWalls,
          coordination,
        }),
      );
      // Another model or graph opened meanwhile: not drawn over it.
      if (!sameModel(asked)) return;
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
  }, [file, graph, points, reportError, settings, studio, toast, needsServerGraph, onServerGraph, sameModel]);

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

  // ---------------------------------------------------------- second models

  // MEP or structural models beside the first (core/federation.js): drawn
  // in its frame and checked against it; their sprinklers, ducts and beams
  // feed the sprinkler layout and, when asked, the fire grid. Nothing is
  // read until one is added; they go with the first model (kept per model).
  const [linked, setLinked] = useState({ owner: null, list: NO_LINKED, busy: null });
  const linkedList = linked.owner === modelInfo ? linked.list : NO_LINKED;
  const linkedBusy = linked.owner === modelInfo ? linked.busy : null;
  const mainName = file?.name || "the model";
  const addLinkedModel = useCallback(
    async (linkedFile) => {
      const viewer = viewerRef.current;
      if (!viewer || !modelInfo || !linkedFile) return;
      const owner = modelInfo;
      setLinked((current) => ({ owner, list: current.owner === owner ? current.list : NO_LINKED, busy: linkedFile.name }));
      try {
        const F = await import("./core/federation.js");
        const main = studio.semantics;
        const result = await viewer.loadLinkedIfc(linkedFile, { relativeFor: (semantics) => F.alignment(main, semantics) });
        const check = F.federationCheck({ name: mainName, semantics: main, box: viewer.mainBox() }, { name: result.name, semantics: result.semantics, box: result.box }, { how: result.how });
        const ids = result.categoryIds || {};
        const entry = {
          id: result.id,
          name: result.name,
          how: result.how,
          check,
          semantics: result.semantics,
          sprinklers: result.semantics?.sprinklers?.length ?? 0,
          beams: ids.beams?.length ?? 0,
          ducts: ids.ducts?.length ?? 0,
          // IFC2x3 writes ducts, pipes and trays alike as flow segments;
          // the ducts and trays among them are told by name when read.
          flowSegments: ids.flowSegments?.length ?? 0,
          cables: ids.cableCarriers?.length ?? 0,
        };
        setLinked((current) => (current.owner !== owner ? current : { owner, busy: null, list: [...current.list.filter((m) => m.id !== entry.id), entry] }));
        markUsed("linkedModel");
      } catch (error) {
        setLinked((current) => (current.owner !== owner ? current : { ...current, busy: null }));
        if (!isCancelled(error)) reportError(error, "Adding a second model");
      }
    },
    [modelInfo, mainName, studio.semantics, reportError, markUsed],
  );
  // Moved by hand so the plans' centres and lowest points meet, when the
  // files were exported from different bases (the check's offset).
  const alignLinkedModel = useCallback(
    async (id) => {
      const viewer = viewerRef.current;
      const entry = linkedList.find((m) => m.id === id);
      if (!viewer || !entry?.check.offset) return;
      const F = await import("./core/federation.js");
      const shift = entry.check.offset;
      const box = viewer.shiftLinked(id, shift);
      const check = F.federationCheck({ name: mainName, semantics: studio.semantics, box: viewer.mainBox() }, { name: entry.name, semantics: entry.semantics, box }, { how: entry.how });
      check.checks.push({
        id: "moved",
        status: "warning",
        text: `Moved by hand ${shift.map((v) => v.toFixed(2)).join(", ")} m so the plans meet: compare it with the drawings before trusting the sprinklers' places`,
      });
      const owner = modelInfo;
      setLinked((current) => (current.owner !== owner ? current : { ...current, list: current.list.map((m) => (m.id === id ? { ...m, check, how: "hand, after the files' frames missed" } : m)) }));
    },
    [linkedList, mainName, studio.semantics, modelInfo],
  );
  const removeLinkedModel = useCallback(
    (id) => {
      viewerRef.current?.removeLinked(id);
      const owner = modelInfo;
      setLinked((current) => (current.owner !== owner ? current : { ...current, list: current.list.filter((m) => m.id !== id) }));
    },
    [modelInfo],
  );

  // Beams, ducts and cable trays (ViewerManager.overheadMeshes), read only
  // while the sprinklers or the fire grid's beams and ducts are on.
  const overheadKey =
    graph && (settings.sprinklers || settings.flamescapesOverhead) ? `${mainName}|${linkedList.map((m) => m.id).join(",")}` : null;
  const [overhead, setOverhead] = useState({ key: null, meshes: null });
  useEffect(() => {
    if (!overheadKey) return undefined;
    let live = true;
    viewerRef.current
      ?.overheadMeshes()
      .then((meshes) => live && setOverhead({ key: overheadKey, meshes }))
      .catch(() => live && setOverhead({ key: overheadKey, meshes: [] }));
    return () => {
      live = false;
    };
  }, [overheadKey]);
  const overheadMeshes = overhead.key === overheadKey ? overhead.meshes : null;

  // ------------------------------------------------------------ sprinklers

  // The heads a Flamescapes run can open (core/sprinklers.js): read from the
  // IFC where it has them, else placed per room by NFPA 13 (on the IfcSpace
  // rooms, or a lattice over the walkable floor without rooms), with the
  // rule report and whether the codes require sprinklers. Kept with the
  // graph it was made on; nothing is made while the switch is off.
  const [sprinklerState, setSprinklerState] = useState({ graph: null, plan: null, busy: false });
  const sprinklerPlan = sprinklerState.graph === graph ? sprinklerState.plan : null;
  const layOutSprinklers = useCallback(async () => {
    const geometry = geometryRef.current;
    if (!graph || !geometry) return;
    setSprinklerState({ graph, plan: null, busy: true });
    try {
      const S = await import("./core/sprinklers.js");
      const upAxis = graph.upAxis || "y";
      const hazard = settings.sprinklerHazard === "auto" ? null : settings.sprinklerHazard;
      // Heads in the IFC, at their elements' centres in the viewer.
      let fromIfc = null;
      const listed = studio.semantics?.sprinklers || [];
      if (settings.sprinklerSource !== "placed" && listed.length) {
        const positions = (await viewerRef.current?.positionsOf?.(listed.map((r) => r.id)).catch(() => null)) || new Map();
        const heads = S.sprinklersFromIfc(listed, positions, { response: settings.sprinklerResponse, hazard: hazard || "light" });
        if (heads.length) fromIfc = { source: "ifc", heads, rooms: [], note: `${heads.length} of ${listed.length} sprinklers in the IFC have a position` };
      }
      // Else a second model's (an MEP model), in this model's frame.
      if (!fromIfc && settings.sprinklerSource !== "placed") {
        const heads = [];
        let count = 0;
        const from = [];
        for (const m of linkedList) {
          const own = m.semantics?.sprinklers || [];
          if (!own.length) continue;
          count += own.length;
          from.push(m.name);
          const positions = (await viewerRef.current?.positionsOf?.(own.map((r) => r.id), { linked: m.id }).catch(() => null)) || new Map();
          for (const h of S.sprinklersFromIfc(own, positions, { response: settings.sprinklerResponse, hazard: hazard || "light" })) heads.push({ ...h, id: `${m.name}:${h.id}` });
        }
        if (heads.length) fromIfc = { source: "ifc", heads, rooms: [], note: `${heads.length} of ${count} sprinklers in ${from.join(" and ")} have a position` };
      }
      // What hangs below the ceilings (this model's beams, a second model's
      // ducts and trays), one record per element, for the obstruction rules.
      const obstructions = [];
      if (overheadMeshes?.length) {
        const byItem = new Map();
        for (const m of overheadMeshes) {
          const key = `${m.source}:${m.itemId}`;
          if (!byItem.has(key)) byItem.set(key, []);
          byItem.get(key).push(m);
        }
        for (const [key, list] of byItem) {
          const o = S.obstructionFromMesh(list, { upAxis });
          if (o) obstructions.push({ ...o, itemId: key });
        }
      }
      if (fromIfc && obstructions.length) {
        const checked = S.checkObstructions(fromIfc.heads, obstructions, { upAxis });
        fromIfc.obstructions = { status: checked.status, notes: checked.notes, failing: checked.heads.filter((h) => h.status === "fail").length, wide: checked.wide.length };
      }
      let plan = fromIfc;
      if (!plan && settings.sprinklerSource !== "ifc") {
        const names = new Map((studio.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
        // The slabs and roofs over the rooms: a space with none over it (drawn
        // on the roof) gets no heads, and heads hang under the real ceiling,
        // a suspended one where the model has it.
        const overhead = [...(geometry.floors || []), ...(geometry.roofMeshes || [])];
        const rooms = geometry.spaces?.length ? S.roomsFromSpaces(geometry.spaces, { upAxis, names, overhead, ceilings: geometry.ceilings, walls: geometry.walls }) : null;
        const usable = rooms?.rooms ? rooms.rooms : rooms;
        if (usable?.length) {
          const laid = S.layoutRooms(usable, { hazardFor: hazard ? () => hazard : undefined, response: settings.sprinklerResponse, obstructions: obstructions.length ? obstructions : null });
          plan = { source: "rooms", ...laid, obstructionCount: obstructions.length };
        } else {
          const laid = S.layoutStoreys(graph, levels, hazard || "light", { response: settings.sprinklerResponse });
          plan = { source: "storeys", ...laid, rooms: (laid.storeys || []).map((st) => ({ ...st, id: st.index, name: `Storey ${st.index + 1}` })) };
        }
      }
      if (!plan) {
        setSprinklerState({ graph, plan: { source: "none", heads: [], rooms: [] }, busy: false });
        return;
      }
      // Whether the codes require sprinklers (SBC 801-CR-2018 903.2 with IBC
      // 2021 where stricter), for the occupancy the performance check uses and
      // the storeys' walkable areas at its area per occupant.
      const group = OCCUPANCY_GROUP[settings.perfOccupancy] ?? null;
      const storeyOf = storeysOf(graph, levels);
      const kinds = graph.kinds || new Uint8Array(storeyOf.length);
      const cell = (graph.spacing || 0.5) ** 2;
      const areas = new Array(Math.max(1, levels.length)).fill(0);
      for (let v = 0; v < storeyOf.length; v += 1) if (storeyOf[v] >= 0 && kinds[v] !== KIND_DOOR) areas[storeyOf[v]] += cell;
      const perStorey = areas.map((a) => Math.ceil(a / Math.max(0.1, settings.perfOccupantArea)));
      const need = S.requirementReport({
        occupancy: group ? [group] : [],
        storeyElevations: levels,
        occupantsPerStorey: perStorey,
        area: areas.reduce((a, b) => a + b, 0),
        occupants: perStorey.reduce((a, b) => a + b, 0),
      });
      plan = {
        ...plan,
        requirement: [{ status: need.status, text: need.text }, ...need.triggers.filter((t) => t.status !== "pass").map((t) => ({ status: t.status, text: `${t.text} (${t.clause})` }))],
      };
      setSprinklerState({ graph, plan, busy: false });
      markUsed("sprinklers");
    } catch (error) {
      setSprinklerState({ graph, plan: null, busy: false });
      reportError(error, "Laying out sprinklers");
    }
  }, [graph, levels, settings.sprinklerHazard, settings.sprinklerResponse, settings.sprinklerSource, settings.perfOccupancy, settings.perfOccupantArea, studio.semantics, linkedList, overheadMeshes, reportError, markUsed]);
  // On with the switch, and again when its choices change; off clears the heads.
  useEffect(() => {
    if (settings.sprinklers && graph && geometryRef.current) layOutSprinklers();
  }, [settings.sprinklers, layOutSprinklers, graph]);
  useEffect(() => {
    const heads = settings.sprinklers ? sprinklerPlan?.heads : null;
    viewerRef.current?.setSprinklers?.(heads?.length ? heads.map((h) => h.position) : null);
  }, [settings.sprinklers, sprinklerPlan]);

  // Suppression systems by room use (core/suppressionSystems.js), off by
  // default: each room's system from its name and the occupancy, the rooms a
  // run floods with clean agent or blankets with foam, and their own
  // extinguishers. The rooms are the sprinklers' (roomsFromSpaces).
  const [systemsState, setSystemsState] = useState({ graph: null, plan: null, busy: false });
  const systemsPlan = systemsState.graph === graph ? systemsState.plan : null;
  const layOutSystems = useCallback(async () => {
    const geometry = geometryRef.current;
    if (!graph || !geometry) return;
    setSystemsState({ graph, plan: null, busy: true });
    try {
      const [S, Y] = await Promise.all([import("./core/sprinklers.js"), import("./core/suppressionSystems.js")]);
      const upAxis = graph.upAxis || "y";
      const occupancy = OCCUPANCY_GROUP[settings.perfOccupancy] ?? null;
      const names = new Map((studio.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
      const overhead = [...(geometry.floors || []), ...(geometry.roofMeshes || [])];
      const found = geometry.spaces?.length ? S.roomsFromSpaces(geometry.spaces, { upAxis, names, overhead, ceilings: geometry.ceilings, walls: geometry.walls }) : null;
      const rooms = (found?.rooms ? found.rooms : found) || [];
      const plan = Y.planSystems(rooms, { occupancy: occupancy ? [occupancy] : null, sprinklered: settings.sprinklers, agentOnly: settings.systemsAgentOnly });
      const units = Y.specialExtinguishers(plan, rooms, { upAxis });
      // The rooms a run floods or foams, as the solver takes them.
      const special = plan.rooms.flatMap((p, i) =>
        (p.system === "clean-agent" || p.system === "foam") && rooms[i]?.footprint
          ? [{ id: p.id, name: p.name, system: p.system, footprint: rooms[i].footprint, floor: rooms[i].floor, ceiling: rooms[i].ceiling, rect: rooms[i].rect, area: rooms[i].area }]
          : [],
      );
      setSystemsState({ graph, plan: { ...plan, occupancy, special, units, noRooms: rooms.length === 0 }, busy: false });
      markUsed("systems");
    } catch (error) {
      setSystemsState({ graph, plan: null, busy: false });
      reportError(error, "Assigning suppression systems");
    }
  }, [graph, settings.perfOccupancy, settings.sprinklers, settings.systemsAgentOnly, studio.semantics, reportError, markUsed]);
  useEffect(() => {
    if (settings.suppressionSystems && graph && geometryRef.current) layOutSystems();
  }, [settings.suppressionSystems, layOutSystems, graph]);
  useEffect(() => {
    const plan = settings.suppressionSystems ? systemsPlan : null;
    viewerRef.current?.setSystemRooms?.(plan?.special ?? null, null, plan?.units ?? null);
  }, [settings.suppressionSystems, systemsPlan]);

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
    // The doors this run closes (core/doorStates.js); a rooms model's
    // openings (doorways with no leaf) stay open.
    const leaves = (geometry.doors || []).filter((m) => m.kind !== "opening");
    const shut = closedDoorIds(
      leaves.map((m) => m.itemId),
      settings.flamescapesDoors,
      {
        selfClosing: new Set((studio.semantics?.doors || []).filter((d) => d.selfClosing).map((d) => d.id)),
        seed: settings.flamescapesDoorSeed,
        share: settings.flamescapesDoorShare,
      },
    );
    const closedLeaves = leaves.filter((m) => shut.has(m.itemId));
    const doorCount = new Set((geometry.doors || []).map((m) => m.itemId)).size;
    const doorWords =
      settings.flamescapesDoors === "asbuilt"
        ? `as built (${shut.size} closed)`
        : settings.flamescapesDoors === "random"
          ? `random (${shut.size} closed, seed ${settings.flamescapesDoorSeed})`
          : settings.flamescapesDoors;
    // Spaces named as voids ("open to below"): no floor, and an atrium over a
    // fire under one, which the grid then reaches the top of (flamescapesRun.js).
    const spaceNames = new Map((studio.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
    const isVoid = (m) => VOID_NAME.test(spaceNames.get(m.itemId) || "");
    const geo = {
      walls: clone(geometry.walls),
      // A model whose floors are its spaces' bottoms (no slabs) gets them as
      // floor slabs, or the fire would have no floor to stand its contents on;
      // a void's would seal its atrium (the NIBS Clinic's smoke stopped there).
      floors: [
        ...clone(geometry.floors),
        ...(geometry.spaces?.length && slabsMissing(geometry.floors, geometry.spaces, { upAxis: graph.upAxis || "y" })
          ? floorsFromSpaces(geometry.spaces.filter((m) => !isVoid(m)), { upAxis: graph.upAxis || "y" })
          : []),
      ],
      voids: clone((geometry.spaces || []).filter(isVoid)),
      roofs: clone(geometry.roofMeshes),
      ceilings: clone(geometry.ceilings),
      stairs: clone(geometry.stairs),
      obstacles: [
        ...clone(windowsOpen ? (geometry.obstacles || []).filter((m) => !windowIds.has(m.itemId)) : geometry.obstacles),
        // Beams, ducts and cable trays under the ceilings, when asked.
        ...(settings.flamescapesOverhead && overheadMeshes?.length ? clone(overheadMeshes) : []),
      ],
      ...(closedLeaves.length ? { doors: clone(closedLeaves) } : {}),
    };
    const transfers = Object.values(geo).flatMap((list) => list.flatMap((m) => [m.positions.buffer, m.indices.buffer]));
    const { contentsFuel, ignitionFuel, item } = fuelsFor(settings.flamescapesClass);
    // The time set, or until the fire service arrives (core/fireService.js).
    const horizon = runHorizon(settings);
    // At most 120 frames of at least 1 s: memory is frames x nodes x 12 bytes
    // (review 6), and a frame shorter than the solver's step means nothing.
    const frames = Math.max(1, Math.min(120, settings.fireMaxSteps, Math.floor(horizon)));
    // Suppression systems by room use: the rooms the run floods or foams, and
    // the heads it leaves out there (a foam room's ordinary heads; an agent
    // room's too when agent only).
    const systemRooms = settings.suppressionSystems && systemsPlan?.special?.length ? systemsPlan.special : null;
    const noHeads = new Set((systemRooms || []).filter((r) => r.system === "foam" || settings.systemsAgentOnly).map((r) => r.id));
    const heads = settings.sprinklers && sprinklerPlan?.heads?.length ? sprinklerPlan.heads.filter((h) => h.room === null || !noHeads.has(h.room)) : null;
    const payload = {
      geometry: geo,
      upAxis: graph.upAxis,
      fire: points.fire,
      nodes: graph.nodes.slice(),
      agentHeight: graph.agentHeight,
      headHeight: 1.8,
      horizon,
      frameSeconds: horizon / frames,
      embers: fireModel === "flamescapes-embers",
      // Soot, CO2 and O2 too, unless switched off or past 64 MB of stored frames
      // (12 bytes a node a frame): the run then keeps the basic three.
      extras: settings.flamescapesMeasures && ((graph.nodes.length / 3) * frames * 12) <= 64 * 2 ** 20,
      extrasWanted: settings.flamescapesMeasures,
      openings: {
        doors: doorWords,
        doorCount,
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
      // The first item as a t-squared design fire (fireClasses.js designFireFor), or null for the solver's own burning.
      designFire: designFireFor(settings.flamescapesClass, settings.flamescapesGrowth),
      // The heads, when the sprinklers are on (core/suppression.js).
      sprinklers: heads?.length ? { heads, mode: settings.sprinklerMode, openOn: settings.sprinklerOpenOn, source: sprinklerPlan.source } : null,
      // The clean-agent and foam rooms (core/suppressionSystems.js systemsSetup).
      systems: systemRooms ? { rooms: systemRooms } : null,
      fireLoad: settings.flamescapesFireLoad * 1e6,
      backend: settings.flamescapesBackend,
      // The grid: "standard" (the same on every device), "speed" (sized to this GPU's measured speed, cells a ms),
      // "finest", or "calibrated" (0.27 m cells, kept under its own setting so a version without it reads "standard").
      grid: settings.flamescapesCalibratedGrid ? "calibrated" : settings.flamescapesGridPolicy,
      // The solver (core/solverVersions.js): the validated one, or the previous one for comparison.
      solver: settings.flamescapesSolver,
      solverOptions: solverOptionsFor(settings.flamescapesSolver),
      gpuRate: gpuRateRef.current?.rate ?? null,
      gpuName: gpuRateRef.current?.name ?? null,
    };
    transfers.push(payload.nodes.buffer);
    const key = JSON.stringify([
      points.fire,
      fireModel,
      settings.flamescapesClass,
      settings.flamescapesGrowth,
      // The length it comes out at: the time set, so kept runs keep their
      // keys, or detection plus the response; a kept run of the same length
      // plays again whichever way the length was set.
      horizon,
      settings.fireMaxSteps,
      settings.flamescapesFireLoad,
      settings.flamescapesBackend,
      settings.flamescapesGridPolicy,
      // Only when not the defaults, so a run with them keeps its kept-run key.
      ...(settings.flamescapesCalibratedGrid ? [["grid", "calibrated"]] : []),
      ...(settings.flamescapesSolver && settings.flamescapesSolver !== "validated" ? [["solver", settings.flamescapesSolver]] : []),
      settings.flamescapesDoors,
      // The doors closed, so a random or as-built set is part of the run.
      [...shut].sort((a, b) => a - b).join(","),
      settings.flamescapesWindows,
      settings.flamescapesVentilation,
      settings.flamescapesTerminalFlow,
      settings.flamescapesMeasures,
      // Only a run with them adds the beams and ducts, and the sprinklers, so
      // a run without keeps its kept-run key.
      ...(settings.flamescapesOverhead && overheadMeshes?.length ? [["overhead", overheadMeshes.length, overheadMeshes.reduce((n, m) => n + m.indices.length, 0)]] : []),
      ...(payload.sprinklers
        ? [["sprinklers", settings.sprinklerMode, settings.sprinklerOpenOn, payload.sprinklers.heads.map((h) => [h.id, ...h.position.map((x) => Math.round(x * 100)), h.rti, h.c, h.activationC, h.flowLpm]).join(";")]]
        : []),
      ...(payload.systems ? [["systems", payload.systems.rooms.map((r) => [r.id, r.system, Math.round(r.floor * 100), Math.round(r.ceiling * 100)].join(":")).join(";")]] : []),
    ]);
    return { payload, transfers, key };
  }, [graph, points.fire, settings, fireModel, studio.semantics, sprinklerPlan, systemsPlan, overheadMeshes]);

  // Start the Flamescapes solver, or return the run already made for the
  // same graph and inputs. Its frames collect in run.frames as they arrive;
  // run.promise settles with the stacked frames.
  const runCombustion = useCallback(() => {
    const { payload, transfers, key } = combustionInputs();
    const current = combustionRef.current;
    if (current && current.key === key && current.graph === graph && !current.failed) return current;
    flamescapesRef.current?.cancel();
    // The model this run is built from: its repeat runs start only on it.
    const geometryAtStart = geometryRef.current;
    const sprinklerPoints = payload.sprinklers?.heads.map((h) => h.position) ?? null;
    // How far each head's spray falls: to the highest floor at least 1 m below it.
    const up = { x: 0, y: 1, z: 2 }[graph.upAxis || "y"] ?? 1;
    const sprinklerDrops = sprinklerPoints?.map((p) => {
      const below = levels.filter((l) => l <= p[up] - 1);
      return below.length ? Math.min(12, p[up] - Math.max(...below)) : 2.4;
    }) ?? null;
    const run = {
      key, graph, frames: [], done: false, failed: false, result: null, info: null, sprinklerPoints, sprinklerDrops,
      systemRooms: payload.systems?.rooms ?? null,
      systemUnits: payload.systems ? (systemsPlan?.units ?? null) : null,
    };
    // The last run's open heads, spray and flooded rooms go now, not when
    // this run's first frame arrives (building its grid takes seconds).
    if (sprinklerPoints) viewerRef.current?.setSprinklers?.(sprinklerPoints, new Set(), sprinklerDrops);
    if (run.systemRooms) viewerRef.current?.setSystemRooms?.(run.systemRooms, null, run.systemUnits);
    // Where this run is kept in the browser (core/runCache.js): the graph,
    // the scene's geometry (every 16th word) and the run's inputs.
    const storeKey = settings.flamescapesKeepRuns
      ? [
          RUN_CACHE_VERSION,
          CONVERTER,
          fingerprint([graph.nodes, graph.edges, graph.blocked]),
          fingerprint(Object.values(payload.geometry).flatMap((list) => list.map((m) => m.positions)), 16),
          key,
        ].join("|")
      : null;
    // A kept run: its frames at once, and its card as it ended.
    const replay = (stored) => {
      run.info = stored.info;
      for (const frame of stored.frames) run.frames.push(frame);
      run.done = true;
      run.result = stored.result;
      const last = stored.frames.at(-1) || {};
      studio.setFire((c) => ({
        ...c,
        combustion: {
          ...stored.info,
          time: last.time ?? 0,
          hrr: last.hrr ?? 0,
          peak: stored.frames.reduce((m, f) => Math.max(m, f.hrr || 0), 0),
          embers: last.embers,
          spots: last.spots,
          detectors: last.detectors?.length ?? 0,
          detectedAt: firstDetection(stored.frames),
          sprinklerSummary: stored.result?.sprinklers ?? null,
          systemsSummary: stored.result?.systems?.rooms ?? null,
          itemSummary: stored.result?.designFire ?? null,
          health: stored.result?.health ?? null,
          horizon: payload.horizon,
          embersOn: payload.embers,
          extras: payload.extras,
          extrasDropped: payload.extrasWanted && !payload.extras,
          replayed: stored.savedAt,
        },
      }));
      return stored.result;
    };
    const compute = () => flamescapesJob(payload, transfers, (progress) => {
      // Only the current run writes the card (SM-1).
      if (combustionRef.current !== run) return;
      if (progress.type === "scene") {
        // A second scene is a restart on the CPU after the GPU stopped: the
        // frames so far came from the abandoned run (a playback waiting on
        // frame k resumes when the new run reaches it).
        if (progress.info.restarted) {
          run.frames.length = 0;
          run.hotRead = 0;
          run.hotGas = -Infinity;
        }
        run.info = progress.info;
        // The view shares the GPU with the solver: it draws less while the GPU computes.
        viewerRef.current?.setDrawRate(progress.info.backend?.kind === "gpu" ? GPU_RUN_DRAW_RATE : 0);
        studio.setFire((c) => ({
          ...c,
          combustion: { ...progress.info, time: 0, hrr: 0, peak: 0, horizon: payload.horizon, embersOn: payload.embers, extras: payload.extras, extrasDropped: payload.extrasWanted && !payload.extras },
        }));
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
          // Sprinklers: how many heads, and when each opened (core/suppression.js).
          sprinklers: payload.sprinklers
            ? {
                heads: payload.sprinklers.heads.length,
                source: payload.sprinklers.source,
                mode: payload.sprinklers.mode,
                openOn: payload.sprinklers.openOn,
                open: progress.sprinklers?.open.length ?? 0,
                first: c.combustion?.sprinklers?.first ?? progress.sprinklers?.opened[0]?.time ?? null,
                water: progress.sprinklers?.water ?? 0,
              }
            : null,
          // The rooms with a system, how many have been detected and have discharged.
          systems: payload.systems
            ? {
                rooms: payload.systems.rooms.length,
                detected: (c.combustion?.systems?.detected ?? 0) + (progress.systems?.detected.length ?? 0),
                discharged: progress.systems?.discharged.length ?? 0,
              }
            : null,
          // The ceiling detectors round the fire (flamescapesRun.js ceilingDetectors) and when the first responds.
          detectors: progress.detectors?.length ?? 0,
          detectedAt: c.combustion?.detectedAt ?? (progress.detectors?.some((s) => s <= DETECTOR_VISIBILITY) ? progress.time : null),
        },
      }));
    }).then(
      (result) => {
        run.done = true;
        run.result = result;
        // When each head opened by both readings, and the water (core/suppression.js).
        // And when each room's system was detected, discharged and acted (core/suppressionSystems.js).
        // And what became of a design-fire first item.
        // And a run the health guard stopped: why and when (flamescapes.js HEALTH).
        const stopped = result?.health && !result.health.ok ? result.health : null;
        if ((result?.sprinklers || result?.systems || result?.designFire || stopped) && (!combustionRef.current || combustionRef.current === run)) {
          studio.setFire((c) => ({
            ...c,
            combustion: c.combustion
              ? { ...c.combustion, sprinklerSummary: result.sprinklers ?? null, systemsSummary: result.systems?.rooms ?? null, itemSummary: result.designFire ?? null, health: stopped }
              : c.combustion,
          }));
        }
        if (stopped && (!combustionRef.current || combustionRef.current === run)) {
          toast(`The fire run was stopped at ${Math.round(stopped.time)} s: ${healthReason(stopped)}. The frames before it play.`, { variant: "warning", ttl: 8000 });
        }
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
        // Its pace, for the next run's estimate; not a run the health guard cut short.
        if (!stopped && result?.info?.ms > 0 && result.horizon > 0) {
          const info = result.info;
          setLastPace({ simulated: result.horizon, wallMs: info.ms, setupMs: info.setupMs ?? 0, kind: info.backend?.kind, name: info.backend?.name, backend: payload.backend, grid: payload.grid });
        }
        return result;
      },
      (error) => {
        run.done = true;
        run.failed = true;
        if (!combustionRef.current || combustionRef.current === run) viewerRef.current?.setDrawRate(0);
        throw error;
      },
    );
    // Repeat runs on the GPU (settings.flamescapesRepeats): the same fire
    // with the room a thousandth of a degree warmer each time, so the card
    // gives the range of a fire that is partly chance, not one run. They go
    // after this run, to its worker, on its grid; a newer run ends them.
    const repeat = (result) => {
      // A run the health guard stopped starts no repeats.
      if (result?.health?.ok === false) return;
      const count = settings.flamescapesRepeats - 1;
      if (!(count > 0) || run.info?.backend?.kind !== "gpu" || !run.info.budget || !result?.hrr) return;
      if (combustionRef.current !== run || geometryRef.current !== geometryAtStart) return;
      const tenability = { threshold: settings.flamescapesUntenableC, smoke: settings.flamescapesSmoke, visibilityLimit: settings.flamescapesVisibility };
      const frames = { ...stackFrames(run.frames, graph.nodes.length / 3), frameSeconds: result.frameSeconds, hrr: result.hrr, stats: result.stats };
      const members = [{ k: 0, ...runSummary(frames, tenability) }];
      let stopped = 0;
      const show = (status) => {
        if (combustionRef.current !== run) return;
        run.repeats = { of: count + 1, status, tenability, range: runRange(members), stopped };
        studio.setFire((c) => (c.combustion ? { ...c, combustion: { ...c.combustion, repeats: run.repeats } } : c));
      };
      show("running");
      // This run's own inputs again (the callback of its start), on its grid.
      const again = combustionInputs();
      // While they run the worker is busy though the run is done: Stop and a
      // new graph end them too (SM-1).
      run.repeating = true;
      flamescapesJob({ ...again.payload, maxCells: run.info.budget, repeats: { count, tenability } }, again.transfers, (progress) => {
        if (progress.type !== "member") return;
        // A member the health guard stopped is left out of the range and counted.
        if (progress.stopped) stopped += 1;
        else members.push(progress.summary);
        show("running");
      }).then(
        (answer) => {
          run.repeating = false;
          show(answer?.skipped ? "stopped" : "done");
        },
        // Ended by Stop or a newer run (its card is another), or the GPU failed: the range stays as far as it got.
        () => {
          run.repeating = false;
          show("stopped");
        },
      );
    };
    run.promise = (storeKey ? readRun(storeKey) : Promise.resolve(null)).then((stored) => {
      // Stopped or replaced while the kept runs were read (behind the last
      // run's write, that can take seconds): nothing of it was in the worker
      // to cancel, so it must not start now. Started, it ran to its end with
      // no Stop to end it, beside the newer run in the same worker (SM-1).
      if (combustionRef.current !== run) {
        run.done = true;
        run.failed = true;
        throw new Error("cancelled");
      }
      if (stored) {
        const result = replay(stored);
        repeat(result);
        return result;
      }
      return compute().then((result) => {
        // Kept once the browser is idle: copying the frames takes a moment.
        // A run the health guard stopped is not kept, so it never replays as a result.
        if (storeKey && result?.health?.ok !== false) {
          const keep = () => void writeRun(storeKey, { info: run.info, frames: run.frames, result });
          if (typeof requestIdleCallback === "function") requestIdleCallback(keep, { timeout: 5000 });
          else setTimeout(keep, 1000);
        }
        repeat(result);
        return result;
      });
    });
    combustionRef.current = run;
    return run;
  }, [combustionInputs, flamescapesJob, graph, studio, toast, settings.flamescapesKeepRuns, settings.flamescapesRepeats, settings.flamescapesUntenableC, settings.flamescapesSmoke, settings.flamescapesVisibility, levels, systemsPlan]);

  const forgetRuns = useCallback(async () => {
    const { count, bytes } = await storedRuns();
    await clearRuns();
    toast(count ? `Forgot ${count} kept run${count === 1 ? "" : "s"} (${(bytes / 2 ** 20).toFixed(1)} MB).` : "No runs were kept.", { variant: "info", ttl: 3500 });
  }, [toast]);

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
                horizon_s: runHorizon(settings),
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

  // The finished Flamescapes run as one table for analysis and machine
  // learning (core/runTable.js): a row per node per frame, gzipped where the
  // browser can.
  const exportTable = useCallback(async () => {
    const run = combustionRef.current;
    if (!graph || !run?.done || run.failed || run.graph !== graph) return;
    try {
      const [{ runTableBlob }, { frameChange, viewerToIfc }, coordination] = await Promise.all([
        import("./core/runTable.js"),
        import("./core/topologicExport.js"),
        viewerRef.current?.getCoordinationMatrix?.() ?? null,
      ]);
      const points = viewerToIfc(graph.nodes, frameChange(coordination, graph.upAxis || "y", graph.agentHeight || 0));
      const n = graph.nodes.length / 3;
      const { blob, extension } = await runTableBlob(
        { frames: run.frames, frameSeconds: run.result?.frameSeconds ?? runHorizon(settings) / Math.max(1, run.frames.length), n, points, kinds: graph.kinds },
        { limitC: settings.flamescapesUntenableC, visibilityLimit: settings.flamescapesVisibility, smoke: settings.flamescapesSmoke },
      );
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${(file?.name || "model").replace(/\.ifc(\.gz)?$/i, "")}-flamescapes.${extension}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      toast(`Exported ${(n * run.frames.length).toLocaleString()} rows: one per node per frame.`, { variant: "info", ttl: 3500 });
    } catch (error) {
      reportError(error, "Exporting the run as a table");
    }
  }, [graph, file, settings, reportError, toast]);

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
    const asked = askedRef.current;
    try {
      const [{ identify }, data] = await Promise.all([import("./core/identify.js"), viewer.identifyData()]);
      // Another model opened meanwhile: this answer is not its (SM-7).
      if (asked.file !== askedRef.current.file) return;
      const answers = data.records.map((record) => ({ record, answer: identify(record) }));
      identifyDataRef.current = { modelKey: modelInfo, answers, meshesOf: data.meshesOf };
      setIdentify({ busy: false, modelKey: modelInfo, answers: answers.map((a) => a.answer) });
    } catch (error) {
      if (asked.file !== askedRef.current.file) return;
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
    // Loaded inside the try: a module that fails to load (offline, or a tab
    // open across a deploy) left the button spinning for good (SM-3).
    let classifier = null;
    try {
      const { IFCNET_MODEL, combine, createGeometryIdentifier, samplePoints } = await import("./core/identifyGeometry.js");
      classifier = createGeometryIdentifier({ modelUrl: `${import.meta.env.BASE_URL}${IFCNET_MODEL}` });
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
      classifier?.terminate();
    }
  }, [modelInfo, reportError]);

  // Likely ignition points (src/core/ignition.js): the model's appliances,
  // furniture and rooms ranked by the ignition-source review.
  const [ignitionState, setIgnition] = useState({ busy: false, candidates: null, occupancy: "residential", context: "any", graphKey: null });
  // A token per graph: candidates found for another graph are not shown, and
  // the old graph's arrays are not kept alive by them (review 13).
  const graphKey = useMemo(() => ({ graph: Boolean(graph) }), [graph]);
  const ignition = ignitionState.graphKey === graphKey ? ignitionState : { ...ignitionState, candidates: null };
  const findIgnition = useCallback(
    async (occupancy = ignition.occupancy, context = ignition.context) => {
      const viewer = viewerRef.current;
      if (!viewer || !graph) return;
      setIgnition((c) => ({ ...c, busy: true, occupancy, context }));
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
        const ranked = rankIgnitionPoints(records, placedRooms, { occupancy, context, limit: 8 });
        const positions = await viewer.positionsOf(ranked.filter((c) => !c.byRoom).map((c) => c.itemId));
        const candidates = ranked
          .map((c) => ({ ...c, position: c.byRoom ? c.position : positions.get(c.itemId) }))
          .filter((c) => c.position);
        setIgnition({ busy: false, candidates, occupancy, context, graphKey });
      } catch (error) {
        setIgnition((c) => ({ ...c, busy: false, candidates: [] }));
        reportError(error, "Finding likely ignition points");
      }
    },
    [graph, graphKey, ignition.occupancy, ignition.context, reportError],
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
      markUsed("ignition");
      markFound("ignition", { label: candidate.scenario.label });
      toast(`Fire origin set at ${candidate.byRoom ? "the room" : "the element"} for: ${candidate.scenario.label}.`, { variant: "info", ttl: 4000 });
    },
    [graph, handlePick, setSettings, toast, markFound, markUsed],
  );

  // The model's exit doors; `final` are the ones out of the building, which
  // the nearest exit, rerouting, the forecast and the agent use.
  const exits = useMemo(() => (graph?.doors ? exitDoors(graph.doors, studio.semantics, graph) : null), [graph, studio.semantics]);

  // Rings at the exits the nearest exit and rerouting use.
  useEffect(() => {
    const shown = graph && exits?.final.length ? exits.final : null;
    viewerRef.current?.setExitDoors(shown?.map((v) => [graph.nodes[v * 3], graph.nodes[v * 3 + 1], graph.nodes[v * 3 + 2]]) ?? null);
  }, [graph, exits]);

  const startFire = useCallback(async () => {
    if (!graph || !points.fire) return;
    markUsed("fire");
    // A plan still running would land mid-fire and clear its route. Its
    // Flamescapes run is the one this fire plays, if the inputs match.
    if (planRunningRef.current) dropPlan({ keepRun: isFlamescapes(fireModel) });
    studio.stopFire();
    studio.setDynamicPath(null);
    viewerRef.current?.setPath(null, { dynamic: true });
    const start = ++fireStartRef.current;
    fireStoppedRef.current = false;

    // The two-zone model is run first, in the forecast worker; the simulation
    // then plays its node temperatures, one frame per step over its horizon.
    // Until they come the fire is starting: Start and Plan wait, Stop ends it.
    let zoneFrames = null;
    if (fireModel === "zone") {
      studio.setFire((current) => ({ ...current, starting: true }));
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
          "zone",
        );
        zoneFrames = zoneNodeFrames(table.nodeRoom, table.table, table.rooms, table.frames);
      } catch (error) {
        // Cancelled: Stop, or the page closing.
        if (start === fireStartRef.current) studio.setFire((current) => ({ ...current, starting: false }));
        if (!isCancelled(error)) reportError(error, "The two-zone fire model");
        return;
      }
      if (start !== fireStartRef.current) return;
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
      markUsed("flamescapes");
      if (settings.flamescapesDoors === "closed") markUsed("flamescapesClosed");
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
      // The doors this Flamescapes run was started with: the switch may change
      // while it plays, for the next run (SimulatePanel offers a restart).
      doors: combustion ? doorsLabel(settings.flamescapesDoors, { seed: settings.flamescapesDoorSeed, share: settings.flamescapesDoorShare }) : null,
      // What it was started with, so Run again in the view can say when the next run differs.
      started: combustion ? { settings, fire: points.fire } : null,
    }));

    // Every model reroutes now: temperature by hazard cost, flood and radial by
    // treating burning nodes as impassable.
    const wantsReroute = settings.reroute && Boolean(points.start && points.exit);
    // The route the fire first gives; a later one that differs is the fire's doing.
    let firstRoute = null;
    // The step it first left that route, and the step it was back on it (the fire's heat gone from it).
    let firstChange = null;
    let backOnFirst = null;
    // With "any exit", the route may leave by another exit out of the building.
    const otherExits = settings.rerouteExits === "any" && exits?.final.length ? exits.final : null;
    const pickedExit = points.exit ? snap(graph, points.exit) : -1;

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
        exits: otherExits,
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
            if (frame) paintCombustion(viewer, combustion, message.step);
            else viewer?.setTemperatures(temperatures, fireModel === "zone" || combustion ? { peak: ZONE_COLOUR_PEAK } : undefined);
            if (combustion) viewer?.setParticles(frame?.particles || null);
            colours.set("fire");
          } else if (message.type === "path_update") {
            const otherExit = message.found && message.exit_node !== null && message.exit_node !== undefined && message.exit_node !== pickedExit;
            const key = (message.node_ids || []).join(",");
            if (firstRoute === null) firstRoute = key;
            else if (message.found && key && key !== firstRoute) {
              markFound("reroute");
              if (firstChange === null) firstChange = message.step;
              backOnFirst = null;
            } else if (firstChange !== null && key === firstRoute && backOnFirst === null) backOnFirst = message.step;
            // The first change is kept: the card read only the latest update,
            // and said "still clear" after a route that changed and came back.
            studio.setDynamicPath({ ...message, otherExit, firstChange, clearBy: backOnFirst });
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
          if (combustion) {
            if (settings.flamescapesDoors === "closed") markFound("closed-doors");
            const spots = combustion.frames[combustion.frames.length - 1]?.spots || 0;
            if (spots > 0) markFound("embers", { spots });
          }
          // No toast: the playback bar goes and the cards say the run is complete.
        },
        onError: (error) => {
          studio.setFire((current) => ({ ...current, running: false }));
          reportError(error, "Fire simulation");
        },
      },
    );
    studio.setFireHandle(handle);
  }, [graph, points, exits, reportError, settings, studio, fireModel, zoneInputs, forecastJob, dropPlan, colours, runCombustion, markFound, markUsed, paintCombustion]);

  // A fire that ends by itself keeps its final colours. Stopping it clears
  // them, back to what the graph showed before: the deadline map if it is
  // wanted, else the graph's own colours.
  const stopFire = useCallback(() => {
    studio.stopFire();
    // A two-zone fire still starting is ended with its frames (SM-2).
    endFireStart();
    // A Flamescapes run still computing, or its repeat runs, is ended: they
    // hold a CPU core or the GPU. A finished run stays, to be played again.
    const run = combustionRef.current;
    if (run && (!run.done || run.repeating)) flamescapesRef.current?.cancel();
    if (run && !run.done) combustionRef.current = null;
    // Stopped, not ended by itself: a guided run ends here (SM-3).
    fireStoppedRef.current = true;
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
  }, [studio, settings.showDeadlineMap, deadlines, planHorizon, colours, endFireStart]);

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
    const shielded = settings.rlMethod !== "qlearning";
    // The shielded learner may escape by any exit out of the building (rlSafe.js exitPoints).
    const exitPoints =
      shielded && settings.rlExits === "any" && exits?.final?.length
        ? exits.final.map((v) => [graph.nodes[v * 3], graph.nodes[v * 3 + 1], graph.nodes[v * 3 + 2]])
        : null;
    studio.setRl({ busy: true, path: null, reachedExit: false, result: null });
    try {
      // In the browser (src/core/rl.js), with the review's terminal-potential
      // and horizon corrections.
      const trained = await rlJob(
        { nodes: graph.nodes, edges: graph.edges, blocked: graph.blocked },
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
      markUsed("agent");
      if (result.certified) markFound("certified");
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
  }, [graph, points, exits, reportError, settings, studio, toast, straighten, rlJob, markFound, markUsed]);

  // -------------------------------------------------------------- exit doors

  const nearestExit = useCallback(() => {
    if (!graph || !points.start || !exits?.final.length) return;
    const open = new Float64Array(graph.nodes.length / 3).fill(Infinity);
    const found = earliestArrival(graph, snap(graph, points.start), 0, {
      exits: exits.final,
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

  // A much nearer exit out of the building than the one the route went to:
  // the user moved the start beside an exit and the route still crossed the
  // building to the exit picked before (NIBS Clinic, 2026-11-11: 54.9 m
  // against 5.7 m). Shown under the route; the route stays the one asked for.
  const nearerExit = useMemo(() => {
    const path = studio.path;
    if (!graph || !points.start || !points.exit || !exits?.final.length || !path?.found) return null;
    const open = new Float64Array(graph.nodes.length / 3).fill(Infinity);
    const found = earliestArrival(graph, snap(graph, points.start), 0, {
      exits: exits.final,
      closes: open,
      speed: 1,
      blocked: settings.useWalls ? graph.blocked : null,
    });
    if (found.exit < 0 || found.exit === snap(graph, points.exit)) return null;
    const walk = found.arrive[found.exit];
    // Only when it saves half the walk and at least 10 m: a near tie is no news.
    return walk < 0.5 * path.length && path.length - walk >= 10 ? { walk, route: path.length } : null;
  }, [graph, points.start, points.exit, exits, studio.path, settings.useWalls]);

  // ------------------------------------------------- egress code check

  // Walking distance, exits per storey, separation, door widths and
  // high-rise, against the stricter of SBC 801-CR-2018 and IBC 2021 for offices
  // (core/codeCheck.js). Kept with the graph it was made on.
  const [codeState, setCodeState] = useState({ result: null, graph: null, shown: false });
  const code = codeState.graph === graph ? codeState : { result: null, graph: null, shown: false };
  const runCodeCheck = useCallback(() => {
    if (!graph) return;
    try {
      // Exits are the doors out of the building and, on the storeys above,
      // the stairs (IBC 202: a flat's front door is exit access, not an exit).
      const out = new Set(exits?.final || []);
      const outItems = new Set((graph.doors || []).filter((d) => out.has(d.node)).map((d) => d.itemId));
      const result = codeCheck(graph, {
        exits: exits?.final || [],
        levels,
        exitDoors: (studio.semantics?.doors || []).filter((d) => outItems.has(d.id)),
        sprinklered: settings.codeSprinklered,
        useWalls: settings.useWalls,
      });
      setCodeState({ result, graph, shown: false });
      markUsed("codeCheck");
      markUsed(settings.codeSprinklered ? "codeCheckSprinklered" : "codeCheckPlain");
      if (notes.used.has(settings.codeSprinklered ? "codeCheckPlain" : "codeCheckSprinklered")) markFound("sprinklers");
      const separations = result.storeys.flatMap((s) => s.checks.filter((c) => c.id === "separation" && c.status !== "n/a"));
      if (separations.length) markFound("exit-separation", { passed: separations.filter((c) => c.status === "pass").length, checked: separations.length });
    } catch (error) {
      reportError(error, "Code check");
    }
  }, [graph, exits, levels, studio.semantics, settings.codeSprinklered, settings.useWalls, reportError, markFound, markUsed, notes.used]);

  // The performance-based check (core/performanceCheck.js): the forecast's
  // escape-deadline map against detection, alarm and pre-movement, with the
  // queues at doors, the share who never respond and a measured spread
  // beside. Kept with the plan it was made on, so a new plan asks for it
  // again. Each check also joins the report (core/perfReport.js), which
  // gathers fires until the graph changes or it is cleared.
  const [perfState, setPerfState] = useState({ result: null, plan: null });
  const perf = perfState.plan && perfState.plan === forecastPlan.result ? perfState.result : null;
  const [perfReportState, setPerfReport] = useState({ graph: null, entries: [] });
  const perfReport = useMemo(() => (perfReportState.graph === graph ? perfReportState.entries : []), [perfReportState, graph]);
  // Fires added in a batch: { done, total } while it runs.
  const [perfBatch, setPerfBatch] = useState(null);
  const perfBatchStopRef = useRef(false);
  const planExitNodes = useCallback(
    () => (settings.forecastExits === "model" && exits?.final.length ? exits.final : points.exit && graph ? [snap(graph, points.exit)] : []),
    [settings.forecastExits, exits, points.exit, graph],
  );
  // The exits people leaving and the slime mould use: every exit out of the
  // building (the picked one only where the model marks none), or, for people
  // leaving when asked (crowdExits "picked"), the exit picked. Not the plan's
  // setting: with the forecast on "the exit you picked", stored in the
  // browser, the whole NIBS Office crowd went through one door (2026-11-11).
  const buildingExitNodes = useCallback(
    () => (exits?.final.length ? exits.final : points.exit && graph ? [snap(graph, points.exit)] : []),
    [exits, points.exit, graph],
  );
  const crowdExitNodes = useCallback(
    () => (settings.crowdExits === "picked" && points.exit && graph ? [snap(graph, points.exit)] : buildingExitNodes()),
    [settings.crowdExits, points.exit, graph, buildingExitNodes],
  );

  // ------------------------------------------------------- people leaving

  // The egress animation (core/crowd.js, viewer/crowd.js): off until switched
  // on, then everyone on the floors walks to the plan's nearest exit and
  // gathers outside it. Worked out once per graph and inputs; the GPU moves
  // them from this clock, so playing costs the page nothing per frame.
  const [crowdState, setCrowdState] = useState({ key: null, result: null, busy: false, error: null });
  const crowdClockRef = useRef({ base: 0, since: 0, rate: 10, playing: false, end: 0, withFire: false });
  const [crowdTime, setCrowdTime] = useState(0);
  // The fire's playback time (s since ignition), for people that play with
  // it (settings.crowdWithFire): a Flamescapes frame's own time, else steps
  // times seconds a step; between steps it runs on at the playback's pace.
  const fireClockRef = useRef({ t: 0, dt: 0, at: 0, delay: 1, running: false, active: false, endedAt: 0 });
  useEffect(() => {
    const c = fireClockRef.current;
    const frames = isFlamescapes(fire.model) ? combustionRef.current?.frames : null;
    const dt = frames ? frames[0]?.time ?? 0 : settings.forecastSecondsPerStep;
    const t = frames ? frames[fire.step]?.time ?? (frames.length ? frames.at(-1).time : 0) : fire.step * dt;
    const now = performance.now();
    Object.assign(c, { t, dt, at: now, delay: Math.max(1, settings.fireDelayMs), active: fire.running || fire.step > 0, endedAt: c.running && !fire.running ? now : c.endedAt, running: fire.running });
  }, [fire.step, fire.running, fire.model, settings.forecastSecondsPerStep, settings.fireDelayMs]);
  const crowdNow = useCallback(() => {
    const c = crowdClockRef.current;
    const f = fireClockRef.current;
    const now = performance.now();
    if (c.withFire) {
      // Before a fire they wait; while it plays they keep its time; once it
      // has played out they carry on from its last moment at their own rate.
      if (!f.active) return 0;
      const t = f.running ? f.t + Math.min(1, (now - f.at) / f.delay) * f.dt : f.t + ((now - f.endedAt) / 1000) * c.rate;
      return Math.min(t, c.end);
    }
    const t = c.playing ? c.base + ((now - c.since) / 1000) * c.rate : c.base;
    return Math.min(t, c.end);
  }, []);
  // People leave only when asked for the graph on screen: a new graph
  // switches them off (the demo turns them on again itself, after its
  // graph). A graph is told by its nodes: the copy the server upload makes
  // (graphId added) is the same graph.
  const graphNodes = graph?.nodes ?? null;
  const crowdGraphRef = useRef(graphNodes);
  useEffect(() => {
    if (crowdGraphRef.current === graphNodes) return;
    crowdGraphRef.current = graphNodes;
    if (settings.crowd) setSettings({ crowd: false });
  }, [graphNodes, settings.crowd, setSettings]);
  const crowdShelter = settings.crowdShelterOn ? settings.crowdShelterArea : 0;
  // Egress with the fire: where and when it makes each place untenable, by
  // the planners' rule (core/forecast.js): a finished Flamescapes run's
  // frames, or the graph fire model's own forecast. Null without the fire.
  const crowdRun = combustionRef.current;
  // Not a run the health guard stopped: past its stop, nodes would read as safe.
  const crowdFlames = settings.crowdWithFire && isFlamescapes(fire.model) && crowdRun?.done && !crowdRun.failed && crowdRun.graph?.nodes === graphNodes && crowdRun.result && crowdRun.result.health?.ok !== false ? crowdRun : null;
  const crowdGraphFire = settings.crowdWithFire && !isFlamescapes(fire.model) && fire.model !== "zone" && fire.step > 0 && points.fire && graph ? fire.model : null;
  const crowdFireKey = crowdFlames
    ? ["flames", crowdFlames.key, settings.flamescapesUntenableC, settings.flamescapesSmoke, settings.flamescapesVisibility]
    : crowdGraphFire
      ? ["graph", crowdGraphFire, points.fire, settings.fireMaxSteps, settings.forecastSecondsPerStep, settings.untenableAbove]
      : null;
  const crowdKey =
    settings.crowd && graph
      ? JSON.stringify([
          crowdExitNodes(),
          settings.perfOccupantArea,
          settings.useWalls,
          settings.departAfter,
          crowdShelter,
          crowdFireKey,
          settings.crowdChoice,
          settings.crowdFluctuate,
          settings.crowdSteps,
          settings.crowdRuns,
          // The IFC's door widths may arrive after the graph: the walk is worked out again with them.
          (studio.semantics?.doors || []).length,
        ])
      : null;
  const crowdAvoiding = Boolean(crowdFireKey);
  const crowdResult = crowdState.key === crowdKey && crowdState.nodes === graphNodes ? crowdState.result : null;
  // The runner: down a stair on a loop, or the forecast's way out on its
  // timeline (when a plan was made on this graph; until then, the stair).
  const crowdPlanRoute = forecastPlan.result && forecastPlan.nodes === graph?.nodes ? forecastPlan.result.earliest : null;
  const crowdRunnerKind = settings.crowdRunner === "forecast" && crowdPlanRoute ? "forecast" : "stair";
  const crowdRunner = useMemo(() => {
    if (!crowdResult || !settings.crowdStairRunner) return null;
    if (crowdRunnerKind === "forecast") return crowdState.forecastRunner?.(graph, crowdPlanRoute) ?? null;
    return crowdState.runner;
  }, [crowdResult, settings.crowdStairRunner, crowdRunnerKind, crowdState, graph, crowdPlanRoute]);
  // The inputs as they read now, for the effect below, which runs on the key
  // alone: depending on its own state too, its cleanup cancelled the walk it
  // had just scheduled when it marked itself busy.
  const crowdInputsRef = useRef(null);
  crowdInputsRef.current = {
    exits: crowdKey ? crowdExitNodes() : [],
    occupantArea: settings.perfOccupantArea,
    useWalls: settings.useWalls,
    departAfter: settings.departAfter,
    shelter: crowdShelter,
    choice: settings.crowdChoice,
    // Randomness in how people move (core/crowd.js; off by default) and how many seeds to run with it.
    motion: { fluctuate: settings.crowdFluctuate, steps: settings.crowdSteps },
    runs: settings.crowdRuns,
    // Each door's own width (the IFC's, else measured): its capacity sets the flow through it.
    doorWidths: () => (graph ? slimeDoorWidths(graph, studio.semantics) : null),
    // Each node's untenable time (s; Infinity never), worked out when the walk is.
    closes: () => {
      if (crowdFlames) {
        const frames = { ...stackFrames(crowdFlames.frames, graph.nodes.length / 3), frameSeconds: crowdFlames.result.frameSeconds };
        return forecastFromFrames(frames, { threshold: settings.flamescapesUntenableC, smoke: settings.flamescapesSmoke, visibilityLimit: settings.flamescapesVisibility }).untenable;
      }
      if (crowdGraphFire) {
        return buildForecast(graph, {
          model: crowdGraphFire,
          seed: snap(graph, points.fire),
          steps: settings.fireMaxSteps,
          dt: settings.forecastSecondsPerStep,
          threshold: settings.untenableAbove,
          blocked: settings.useWalls ? graph.blocked : null,
        }).untenable;
      }
      return null;
    },
  };
  // The key and graph of the walk last worked out, so switching off and on again does not work it out twice.
  const crowdDoneRef = useRef({ key: null, nodes: null, state: null });
  useEffect(() => {
    if (!crowdKey) return undefined;
    // Worked out already: shown again. A walk for another key may have been
    // started in between and dropped, which left the panel on that key with
    // nothing drawn (wiki/architecture/state-machines.md, SM-3).
    const done = crowdDoneRef.current;
    if (done.key === crowdKey && done.nodes === graph.nodes) {
      setCrowdState(done.state);
      return undefined;
    }
    const { exits: exitNodes, occupantArea, useWalls, departAfter, shelter, closes, choice, motion, runs, doorWidths: widthsOf } = crowdInputsRef.current;
    let live = true;
    const finish = (state) => {
      if (!live) return;
      const next = { key: crowdKey, nodes: graph.nodes, busy: false, result: null, error: null, ...state };
      crowdDoneRef.current = { key: crowdKey, nodes: graph.nodes, state: next };
      setCrowdState(next);
    };
    setCrowdState({ key: crowdKey, nodes: graph.nodes, result: null, busy: true, error: null });
    // A moment for the panel to say it is working before the walk is worked out.
    const timer = setTimeout(() => {
      if (!exitNodes.length) {
        finish({ error: "No exit to walk to: the model marks none out of the building, so pick one in Route." });
        return;
      }
      import("./core/crowd.js")
        .then(({ simulateCrowd, stairRunner, forecastRunner, STEP_RANDOMNESS }) => {
          if (!live) return;
          const kS = STEP_RANDOMNESS[motion.steps] ?? null;
          const varied = Boolean(motion.fluctuate) || kS !== null;
          const fire = closes();
          const doorWidths = widthsOf();
          const optsFor = (seed) => ({
            exits: exitNodes,
            occupantArea,
            doorWidths,
            blocked: useWalls ? graph.blocked : null,
            start: [departAfter, departAfter + 10],
            shelterArea: shelter > 0 ? shelter : Infinity,
            closes: fire,
            choice,
            fluctuate: Boolean(motion.fluctuate),
            kS,
            seed,
            // A phone draws fewer, so it keeps its pace.
            maxPeople: isPhone() ? 1500 : 4000,
          });
          const t0 = performance.now();
          const result = simulateCrowd(graph, optsFor(1));
          result.ms = performance.now() - t0;
          // With randomness one run is one sample: the view plays the first,
          // and the other seeds run one at a time after it, for the range.
          const count = varied ? Math.max(1, Math.min(isPhone() ? 3 : 10, runs || 1)) : 1;
          const sample = (r) => ({ lastOut: r.lastOut, inside: r.inside, cutOff: r.cutOff });
          finish({ result, runner: stairRunner(graph, result), forecastRunner, runs: count > 1 ? { of: count, samples: [sample(result)] } : null });
          let seed = 2;
          const more = () => {
            if (!live || seed > count) return;
            const r = simulateCrowd(graph, optsFor(seed));
            seed += 1;
            setCrowdState((s) => {
              if (s.key !== crowdKey || !s.runs) return s;
              const next = { ...s, runs: { ...s.runs, samples: [...s.runs.samples, sample(r)] } };
              crowdDoneRef.current = { key: crowdKey, nodes: graph.nodes, state: next };
              return next;
            });
            setTimeout(more, 0);
          };
          if (count > 1) setTimeout(more, 0);
        })
        .catch((error) => finish({ error: error?.message || String(error) }));
    }, 30);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [crowdKey, graph]);
  // The people on screen: set again when the walk or the look changes, and
  // played from the start; cleared when switched off.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return undefined;
    if (!crowdResult) {
      viewer.setCrowd?.(null);
      return undefined;
    }
    const clock = crowdClockRef.current;
    Object.assign(clock, { base: 0, since: performance.now(), playing: true, end: crowdResult.lastOut + 10 });
    viewer.setCrowd?.(crowdResult, { style: settings.crowdStyle, clock: crowdNow });
    return () => viewer.setCrowd?.(null);
  }, [crowdResult, settings.crowdStyle, crowdNow]);
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return undefined;
    // The forecast's runner keeps the people's clock; the stair's loops in real time.
    viewer.setStairRunner?.(crowdRunner, crowdRunnerKind === "forecast" ? { clock: crowdNow } : {});
    return () => viewer.setStairRunner?.(null);
  }, [crowdRunner, crowdRunnerKind, crowdNow]);
  useEffect(() => {
    const clock = crowdClockRef.current;
    // The rate changes from where the walk is now.
    Object.assign(clock, { base: crowdNow(), since: performance.now(), rate: settings.crowdRate });
  }, [settings.crowdRate, crowdNow]);
  useEffect(() => {
    const clock = crowdClockRef.current;
    // Unlinked again: the own clock goes on from where the fire had them.
    if (clock.withFire && !settings.crowdWithFire) Object.assign(clock, { base: crowdNow(), since: performance.now(), playing: true });
    clock.withFire = settings.crowdWithFire;
  }, [settings.crowdWithFire, crowdNow]);
  // The time shown under the controls, four times a second while playing.
  useEffect(() => {
    if (!crowdResult) return undefined;
    const timer = setInterval(() => {
      const t = crowdNow();
      setCrowdTime(t);
      const clock = crowdClockRef.current;
      if (clock.playing && t >= clock.end) Object.assign(clock, { base: clock.end, playing: false });
    }, 250);
    return () => clearInterval(timer);
  }, [crowdResult, crowdNow]);
  const playCrowd = useCallback(() => {
    const clock = crowdClockRef.current;
    if (clock.playing) Object.assign(clock, { base: crowdNow(), playing: false });
    else Object.assign(clock, { base: clock.base >= clock.end ? 0 : clock.base, since: performance.now(), playing: true });
    setCrowdTime(crowdNow());
  }, [crowdNow]);
  const restartCrowd = useCallback(() => {
    Object.assign(crowdClockRef.current, { base: 0, since: performance.now(), playing: true });
    setCrowdTime(0);
  }, []);

  // ------------------------------------------------------- slime mould

  // Jones's Physarum agents on the walkable graph (core/slimeEgress.js), in a
  // worker of their own, grown when asked for; a result is for the graph it
  // was grown on.
  const slimeRef = useRef(null);
  useEffect(() => () => slimeRef.current?.cancel(), []);
  const [slimeState, setSlimeState] = useState(SLIME_NONE);
  const slime = slimeState.nodes === graphNodes && graphNodes ? slimeState : SLIME_NONE;
  // The fire's untenable times, as the people's are worked out: a Flamescapes
  // run once it has finished, else the graph fire.
  const slimeFire = useCallback(() => {
    if (!graph) return null;
    const run = combustionRef.current;
    if (isFlamescapes(fire.model)) {
      if (!run?.done || run.failed || run.graph?.nodes !== graph.nodes || !run.result || run.result.health?.ok === false) return null;
      const frames = { ...stackFrames(run.frames, graph.nodes.length / 3), frameSeconds: run.result.frameSeconds };
      const closes = forecastFromFrames(frames, { threshold: settings.flamescapesUntenableC, smoke: settings.flamescapesSmoke, visibilityLimit: settings.flamescapesVisibility }).untenable;
      // The run's own smoke: its visibility at head height, frame by frame.
      return { closes, smokeFrames: { K: frames.K, frameSeconds: frames.frameSeconds, vis: frames.vis } };
    }
    if (fire.model === "zone" || !(fire.step > 0) || !points.fire) return null;
    const closes = buildForecast(graph, {
      model: fire.model,
      seed: snap(graph, points.fire),
      steps: settings.fireMaxSteps,
      dt: settings.forecastSecondsPerStep,
      threshold: settings.untenableAbove,
      blocked: settings.useWalls ? graph.blocked : null,
    }).untenable;
    return { closes, smokeFrames: null };
  }, [graph, fire.model, fire.step, points.fire, settings]);
  const growSlime = useCallback(() => {
    if (!graph) return;
    const exits = buildingExitNodes();
    if (!exits.length) {
      toast("No exit to grow towards: the model marks none out of the building, so pick one in Route.", { variant: "warning" });
      return;
    }
    const withFire = settings.slimeFire === "soft" || settings.slimeFire === "hard";
    const fireNow = withFire ? slimeFire() : null;
    const closes = fireNow?.closes ?? null;
    if (withFire && !closes) {
      toast("Start a fire in Simulate first; a Flamescapes run must have finished.", { variant: "warning" });
      return;
    }
    slimeRef.current ||= workerClient(
      () => new Worker(new URL("./workers/slime.worker.js", import.meta.url), { type: "module" }),
      { failure: "The slime mould stopped, usually because the browser ran out of memory. Reload the page and try again." },
    );
    const nodes = graph.nodes;
    setSlimeState({ nodes, result: null, busy: true, error: null });
    const payload = {
      graph: { nodes: graph.nodes.slice(), edges: graph.edges.slice(), kinds: graph.kinds ? graph.kinds.slice() : null, upAxis: graph.upAxis, spacing: graph.spacing },
      opts: {
        exits,
        start: points.start ? snap(graph, points.start) : -1,
        closes: closes ? Float64Array.from(closes) : null,
        smokeFrames: fireNow?.smokeFrames ? { ...fireNow.smokeFrames, vis: Float32Array.from(fireNow.smokeFrames.vis) } : null,
        visibilityLimit: settings.flamescapesVisibility,
        blocked: settings.useWalls && graph.blocked ? graph.blocked.slice() : null,
        fire: settings.slimeFire === "hard" ? "hard" : "soft",
        // For the bottlenecks and the dead ends.
        occupantArea: settings.perfOccupantArea,
        doorWidth: slimeDoorWidths(graph, studio.semantics),
        sprinklered: Boolean(settings.codeSprinklered),
        seed: 1,
      },
    };
    const transfers = [payload.graph.nodes.buffer, payload.graph.edges.buffer, payload.opts.doorWidth.buffer];
    for (const typed of [payload.graph.kinds, payload.opts.closes, payload.opts.blocked, payload.opts.smokeFrames?.vis]) if (typed) transfers.push(typed.buffer);
    const t0 = performance.now();
    slimeRef.current.run({ payload }, transfers).then(
      (result) => setSlimeState((s) => (s.nodes === nodes ? { ...s, busy: false, result: { ...result, ms: performance.now() - t0, withFire: Boolean(closes) } } : s)),
      (error) => {
        if (!isCancelled(error)) setSlimeState((s) => (s.nodes === nodes ? { ...s, busy: false, error: error?.message || String(error) } : s));
      },
    );
  }, [graph, buildingExitNodes, points.start, settings.slimeFire, settings.useWalls, settings.perfOccupantArea, settings.codeSprinklered, settings.flamescapesVisibility, studio.semantics, slimeFire, toast]);
  const clearSlime = useCallback(() => setSlimeState(SLIME_NONE), []);
  // The weights on the graph, slime yellow, and the route along the tubes.
  useEffect(() => {
    const viewer = viewerRef.current;
    const result = slime.result;
    if (!viewer || !result || !graph) return undefined;
    viewer.setSlimeWeights?.(result.weights);
    const route = result.route?.nodes;
    if (route) viewer.setPath(route.map((v) => [graph.nodes[v * 3], graph.nodes[v * 3 + 1], graph.nodes[v * 3 + 2]]), { name: "slimePath", color: 0xe5b800 });
    return () => {
      viewer.setSlimeWeights?.(null);
      viewer.setPath(null, { name: "slimePath" });
    };
  }, [slime.result, graph]);
  // Its skeletons (core/skeleton.js), when asked: the desire lines in a dark
  // ink with arrows to the exits, over the yellow; the floor's medial axis in cyan.
  useEffect(() => {
    const viewer = viewerRef.current;
    const result = slime.result;
    if (!viewer || !result || !graph) return undefined;
    const P = graph.nodes;
    const segmentsOf = (links) => {
      const out = new Float32Array(links.length * 3);
      for (let k = 0; k < links.length; k += 1) out.set(P.subarray(links[k] * 3, links[k] * 3 + 3), k * 3);
      return out;
    };
    if (settings.slimeDesire && result.desireLinks?.length) {
      // Thicker where more ways out bundle: 0.6 to 2 times, by the square root of the flow.
      const widths = Float32Array.from(result.desireFlow, (f) => 0.6 + 1.4 * Math.sqrt(f / Math.max(1, result.desireMax)));
      viewer.setSkeleton?.("slimeDesire", segmentsOf(result.desireLinks), { color: 0x1e1b4b, arrows: true, widths });
    }
    if (settings.slimeMedial && result.medialLinks?.length) viewer.setSkeleton?.("slimeMedial", segmentsOf(result.medialLinks), { color: 0x06b6d4 });
    if (settings.slimeDeadEnds && result.deadLinks?.length) viewer.setSkeleton?.("slimeDead", segmentsOf(result.deadLinks), { color: 0x7f1d1d });
    if (settings.slimeAbandoned && result.abandonedLinks?.length) viewer.setSkeleton?.("slimeAbandoned", segmentsOf(result.abandonedLinks), { color: 0x64748b });
    if (settings.slimeBottlenecks && result.bottlenecks?.length) {
      const worst = result.bottlenecks[0].seconds;
      viewer.setMarkers?.(
        "slimeBottlenecks",
        result.bottlenecks.map((b) => [P[b.node * 3], P[b.node * 3 + 1], P[b.node * 3 + 2]]),
        { color: 0xbe123c, sizes: result.bottlenecks.map((b) => 0.6 + 0.8 * Math.sqrt(b.seconds / worst)) },
      );
    }
    return () => {
      for (const layer of ["slimeDesire", "slimeMedial", "slimeDead", "slimeAbandoned"]) viewer.setSkeleton?.(layer, null);
      viewer.setMarkers?.("slimeBottlenecks", null);
    };
  }, [slime.result, graph, settings.slimeDesire, settings.slimeMedial, settings.slimeDeadEnds, settings.slimeAbandoned, settings.slimeBottlenecks]);
  // ------------------------------------------------------- egress map and accessibility

  // The egress map (core/egressMap.js) and, when asked, each storey's
  // measuring grid with the wheelchair audit (core/floorRaster.js,
  // core/accessAudit.js), in a worker (workers/egress.worker.js). Kept with
  // the graph they were made on; nothing runs until asked.
  const [egressState, setEgressState] = useState({ nodes: null, result: null, fine: null, busy: null, error: null });
  const egressRef = useRef(null);
  const egress = egressState.nodes === graphNodes && graphNodes ? egressState : { nodes: null, result: null, fine: null, busy: null, error: null };
  const runEgress = useCallback(
    (kind) => {
      if (!graph) return;
      const exits = buildingExitNodes();
      if (!exits.length) {
        toast("No exit to map towards: the model marks none out of the building, so pick one in Route.", { variant: "warning" });
        return;
      }
      egressRef.current ||= workerClient(() => new Worker(new URL("./workers/egress.worker.js", import.meta.url), { type: "module" }), {
        failure: "The egress map stopped, usually because the browser ran out of memory. Reload the page and try again.",
      });
      const geometry = geometryRef.current;
      if (kind === "fine" && !geometry) {
        toast("Build the walkable graph again first: the model's geometry is not loaded.", { variant: "warning" });
        return;
      }
      const nodes = graph.nodes;
      setEgressState((s) => ({ ...(s.nodes === nodes ? s : { result: null, fine: null }), nodes, busy: kind, error: null }));
      const names = (studio.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]);
      const doorNames = {};
      for (const d of graph.doors || []) {
        const sem = (studio.semantics?.doors || []).find((x) => x.id === d.itemId);
        if (d.node >= 0 && sem?.name) doorNames[d.node] = sem.name;
      }
      const limit = settings.codeSprinklered ? EGRESS_B.travel.sprinklered : EGRESS_B.travel.unsprinklered;
      const clone = (meshes) => (meshes || []).map((m) => ({ itemId: m.itemId, positions: m.positions.slice(), indices: m.indices.slice() }));
      const payload = {
        kind,
        graph: { nodes: graph.nodes.slice(), edges: graph.edges.slice(), kinds: graph.kinds ? graph.kinds.slice() : null, blocked: graph.blocked ? graph.blocked.slice() : null, upAxis: graph.upAxis, spacing: graph.spacing, agentHeight: graph.agentHeight },
        opts: {
          exits,
          levels,
          useWalls: settings.useWalls,
          limit,
          sprinklered: Boolean(settings.codeSprinklered),
          occupantArea: settings.perfOccupantArea,
          doorWidth: slimeDoorWidths(graph, studio.semantics),
          doorNames,
          spaceNames: names,
          profile: settings.accessProfile,
          fastMarching: true,
          // Phones measure at 10 cm: four times fewer cells.
          cell: window.matchMedia?.("(max-width: 720px)").matches ? 0.1 : 0.05,
        },
        // The rooms (their IfcSpaces, checked against the slabs, ceilings and walls) for both;
        // the fine grid also cuts the obstacles, stairs and doors.
        geometry: {
          floors: clone(geometry?.floors),
          spaces: clone(geometry?.spaces),
          walls: clone(geometry?.walls),
          ceilings: clone(geometry?.ceilings),
          roofMeshes: clone(geometry?.roofMeshes),
          ...(kind === "fine" ? { obstacles: clone(geometry?.obstacles), stairs: clone(geometry?.stairs), doors: clone(geometry?.doors) } : {}),
        },
      };
      const t0 = performance.now();
      // Each storey by its IFC name within 0.6 m, else its IFC height (the viewer's less the coordination shift).
      const storeyLabel = (h) => {
        const ifc = h - (levelShift || 0);
        const near = (studio.semantics?.storeys || []).filter((x) => Number.isFinite(x.elevation)).reduce((best, x) => (!best || Math.abs(x.elevation - ifc) < Math.abs(best.elevation - ifc) ? x : best), null);
        return near?.name && Math.abs(near.elevation - ifc) <= 0.6 ? near.name : `the floor at ${ifc.toFixed(1)} m`;
      };
      egressRef.current.run({ payload }).then(
        (result) =>
          setEgressState((s) =>
            s.nodes !== nodes
              ? s
              : kind === "map"
                ? { ...s, busy: null, result: { ...result, ms: performance.now() - t0 } }
                : {
                    ...s,
                    busy: null,
                    result: { ...result.map, ms: result.map.ms },
                    fine: { ...result, storeys: result.storeys.map((x) => ({ ...x, label: storeyLabel(x.level) })), profile: settings.accessProfile, ms: performance.now() - t0 },
                  },
          ),
        (error) => {
          if (!isCancelled(error)) setEgressState((s) => (s.nodes === nodes ? { ...s, busy: null, error: error?.message || String(error) } : s));
        },
      );
    },
    [graph, buildingExitNodes, levels, levelShift, settings.useWalls, settings.codeSprinklered, settings.perfOccupantArea, settings.accessProfile, studio.semantics, toast],
  );
  const clearEgress = useCallback(() => setEgressState({ nodes: null, result: null, fine: null, busy: null, error: null }), []);
  // What the map draws, each layer when asked.
  useEffect(() => {
    const viewer = viewerRef.current;
    const r = egress.result;
    if (!viewer || !r) return undefined;
    if (settings.egressRoutes && r.routeLinks?.length) viewer.setSkeleton?.("egressRoutes", r.routeLinks, { color: 0xdc2626, arrows: true });
    if (settings.egressContours && r.contours?.length) viewer.setSkeleton?.("egressContours", r.contours, { color: 0x334155 });
    if (settings.egressDeadEnds && r.deadLinks?.length) viewer.setSkeleton?.("egressDeadEnds", r.deadLinks, { color: 0x9d174d });
    if (settings.egressCritical && r.critical?.doors?.length) {
      viewer.setMarkers?.(
        "egressCritical",
        r.critical.doors.map((d) => d.position),
        { color: 0x7c3aed, sizes: r.critical.doors.map((d) => 0.5 + 0.25 * Math.min(4, d.rooms.length)) },
      );
    }
    return () => {
      for (const layer of ["egressRoutes", "egressContours", "egressDeadEnds"]) viewer.setSkeleton?.(layer, null);
      viewer.setMarkers?.("egressCritical", null);
    };
  }, [egress.result, settings.egressRoutes, settings.egressContours, settings.egressDeadEnds, settings.egressCritical]);
  // The fine grid's walk, and the wheelchair audit, laid on the floors.
  useEffect(() => {
    const viewer = viewerRef.current;
    const f = egress.fine;
    if (!viewer || !f?.images?.length) return undefined;
    if (settings.egressFine) viewer.setFloorImages?.("egressFine", f.images.map((im) => ({ ...im, rgba: im.distance })));
    if (settings.accessOverlay) {
      viewer.setFloorImages?.("accessAudit", f.images.map((im) => ({ ...im, rgba: im.access })), { lift: 0.06 });
      const issues = f.storeys.flatMap((s) => s.audit.issues).filter((i) => i.at && i.severity !== "low");
      viewer.setMarkers?.("accessIssues", issues.map((i) => i.at), { color: 0xb91c1c, sizes: issues.map((i) => (i.severity === "high" ? 1 : 0.7)) });
    }
    return () => {
      viewer.setFloorImages?.("egressFine", null);
      viewer.setFloorImages?.("accessAudit", null);
      viewer.setMarkers?.("accessIssues", null);
    };
  }, [egress.fine, settings.egressFine, settings.accessOverlay]);

  // ------------------------------------------------------- recording

  // The view recorded as a GIF (lib/recorder.js): the fire, the smoke and the
  // people as they play. Saved when stopped or at the device's limit.
  const recordRef = useRef(null);
  const [recording, setRecording] = useState({ state: "idle", seconds: 0, limit: 0 });
  const toggleRecording = useCallback(async () => {
    const current = recordRef.current;
    // Starting: its module is still loading. A second click then started a
    // second recorder that no Stop could reach (SM-3).
    if (current === RECORDER_STARTING) return;
    if (current) {
      setRecording((r) => ({ ...r, state: "encoding" }));
      current.stop();
      return;
    }
    const viewer = viewerRef.current;
    if (!viewer) return;
    recordRef.current = RECORDER_STARTING;
    try {
      const { recordView, recordingLimits } = await import("./lib/recorder.js");
      const limits = recordingLimits();
      const rec = recordView(viewer, {
        ...limits,
        onTick: ({ seconds, stopped }) => setRecording((r) => (r.state === "idle" ? r : { ...r, seconds, state: stopped ? "encoding" : r.state })),
      });
      recordRef.current = rec;
      setRecording({ state: "recording", seconds: 0, limit: limits.maxSeconds });
      rec.done
        .then(({ blob, frames }) => {
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
          a.download = `${(file?.name || "topologic-studio").replace(/\.ifc(\.gz)?$/i, "")}-${stamp}.gif`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          setTimeout(() => URL.revokeObjectURL(a.href), 10000);
          toast(`Saved a GIF of ${frames} frames (${formatBytes(blob.size)}).`, { variant: "info", ttl: 4000 });
        })
        .catch((error) => {
          if (!isCancelled(error)) reportError(error, "Recording the view");
        })
        .finally(() => {
          recordRef.current = null;
          setRecording({ state: "idle", seconds: 0, limit: 0 });
        });
    } catch (error) {
      recordRef.current = null;
      reportError(error, "Recording the view");
    }
  }, [file, toast, reportError]);
  useEffect(() => () => recordRef.current?.cancel(), []);

  // --------------------------------------------------------- extinguishers

  // Portable extinguishers (core/extinguishers.js): the IFC's, told by name
  // (here and in a second model), else placed on the walk by the travel
  // rule; checked against SBC 801-CR-2018 906 and NFPA 10. Kept with the
  // graph they were made on; nothing is read while the switch is off.
  const [extinguisherState, setExtinguisherState] = useState({ graph: null, plan: null, busy: false });
  const extinguisherPlan = extinguisherState.graph === graph ? extinguisherState.plan : null;
  const occupancyGroup = OCCUPANCY_GROUP[settings.perfOccupancy] ?? null;
  const layOutExtinguishers = useCallback(async () => {
    if (!graph) return;
    setExtinguisherState({ graph, plan: null, busy: true });
    try {
      const E = await import("./core/extinguishers.js");
      const viewer = viewerRef.current;
      const hazard = settings.extinguisherHazard;
      let sites = [];
      let note = null;
      if (settings.extinguisherSource !== "placed" && viewer?.extinguisherCandidates) {
        const candidates = await viewer.extinguisherCandidates().catch(() => []);
        const groups = new Map();
        for (const r of candidates) {
          if (!groups.has(r.linked)) groups.set(r.linked, []);
          groups.get(r.linked).push({ itemId: r.itemId, name: r.name, ifcClass: r.type, predefinedType: r.predefined });
        }
        let found = 0;
        for (const [linked, list] of groups) {
          const records = E.extinguisherRecords(list);
          found += records.length;
          const positions = (await viewer.positionsOf(records.map((r) => r.itemId), linked ? { linked } : {}).catch(() => null)) || new Map();
          for (const r of records) {
            const p = positions.get(r.itemId);
            if (p) sites.push({ node: snap(graph, p), rating: r.rating, unit: r.unit, name: r.name });
          }
        }
        if (sites.length) note = `${sites.length} of ${found} in the IFC have a place`;
      }
      let source = sites.length ? "ifc" : "none";
      let laid = null;
      if (!sites.length && settings.extinguisherSource !== "ifc") {
        laid = E.layoutExtinguishers(graph, { levels, hazard, exits: planExitNodes(), occupancy: occupancyGroup ?? undefined });
        sites = laid.sites.map((s) => ({ node: s.node, rating: s.rating }));
        source = "placed";
      }
      const check = sites.length || source === "none"
        ? E.placementCheck(graph, sites.map((s) => ({ node: s.node, rating: s.rating ?? undefined })), { levels, hazard, occupancy: occupancyGroup ?? undefined })
        : null;
      setExtinguisherState({ graph, plan: { source, sites, check, note, notes: laid?.notes || [], api: E }, busy: false });
      markUsed("extinguishers");
    } catch (error) {
      setExtinguisherState({ graph, plan: null, busy: false });
      reportError(error, "Placing extinguishers");
    }
  }, [graph, levels, settings.extinguisherHazard, settings.extinguisherSource, occupancyGroup, planExitNodes, reportError, markUsed]);
  useEffect(() => {
    if (settings.extinguishers && graph) layOutExtinguishers();
  }, [settings.extinguishers, layOutExtinguishers, graph]);
  useEffect(() => {
    const sites = settings.extinguishers && graph ? extinguisherPlan?.sites : null;
    viewerRef.current?.setExtinguishers?.(sites?.length ? sites.map((s) => [graph.nodes[s.node * 3], graph.nodes[s.node * 3 + 1] - (graph.agentHeight || 0), graph.nodes[s.node * 3 + 2]]) : null);
  }, [settings.extinguishers, extinguisherPlan, graph]);

  // One responder's attempt on the fire on show (core/extinguishers.js
  // attempt): from the fire's place, after they see it or after the alarm
  // and the design pre-movement, to the nearest suitable unit and back to
  // attack it; judged on the fire's heat release when they start, from the
  // Flamescapes run, else a t-squared design fire (the two-zone settings).
  const [attemptState, setAttemptState] = useState({ graph: null, result: null });
  const extinguisherAttempt = attemptState.graph === graph ? attemptState.result : null;
  const runExtinguisherAttempt = useCallback(() => {
    if (!settings.extinguishers || !graph || !points.fire || !extinguisherPlan?.sites?.length) return;
    const E = extinguisherPlan.api;
    if (!E) return;
    const fireNode = snap(graph, points.fire);
    const run = combustionRef.current;
    const frames = run && run.graph === graph && isFlamescapes(fireModel) ? run.frames : null;
    let hrrAt, source, horizon = Infinity;
    if (frames?.length) {
      source = "the Flamescapes run";
      horizon = frames[frames.length - 1].time;
      hrrAt = (t) => {
        let prev = { time: 0, hrr: 0 };
        for (const f of frames) {
          if (f.time >= t) return prev.hrr + ((f.hrr - prev.hrr) * (t - prev.time)) / Math.max(1e-9, f.time - prev.time);
          prev = f;
        }
        return prev.hrr;
      };
    } else {
      const design = E.designFire(settings.zoneGrowth, settings.zonePeakKw);
      source = design.text;
      hrrAt = design.hrrAt;
    }
    // After the alarm: detection as the performance check takes it, from the
    // plan's run when asked and found, else the time typed.
    const plan = forecastPlan.nodes === graph.nodes ? forecastPlan.result : null;
    const detected = reportPages.planDetection(plan, { fromRun: settings.perfDetectionFromRun, typed: settings.perfDetection });
    const median = Math.sqrt(Math.max(1, settings.perfPre1) * Math.max(1, settings.perfPre99));
    const alarm = settings.extinguisherStart === "alarm";
    const tGo = alarm ? detected.seconds + settings.perfAlarm + median : 30;
    const start = alarm
      ? {
          detection: detected.seconds,
          // No plan and no Flamescapes fire: no run would give one either.
          detectionFrom: detected.from === "no plan" && !isFlamescapes(fireModel) ? "no run" : detected.from,
          detectors: plan?.detectors ?? 0,
          alarm: settings.perfAlarm,
          median,
        }
      : null;
    try {
      const out = E.attempt({
        graph,
        fireNode,
        startNode: fireNode,
        sites: extinguisherPlan.sites.map((s) => ({ node: s.node, ...(s.unit ? { unit: s.unit } : {}) })),
        tGo,
        speed: settings.walkSpeed,
        unit: settings.extinguisherUnit,
        tTake: settings.extinguisherTrained ? E.T_TAKE.instructed ?? 7.2 : E.T_TAKE.value,
        hrrAt,
        fireClass: isFlamescapes(fireModel) ? settings.flamescapesClass : "A",
        exits: planExitNodes(),
        useWalls: settings.useWalls,
      });
      const late = out.times?.tS > horizon;
      setAttemptState({ graph, result: { ...out, source, tGo, start, late, horizon } });
      markUsed("extinguisherAttempt");
    } catch (error) {
      setAttemptState({ graph, result: { outcome: "cannot tell", reasons: [{ id: "error", status: "cannot tell", text: error.message }], source, tGo, start } });
    }
  }, [settings.extinguishers, settings.extinguisherStart, settings.extinguisherUnit, settings.extinguisherTrained, settings.zoneGrowth, settings.zonePeakKw, settings.perfPre1, settings.perfPre99, settings.perfDetection, settings.perfDetectionFromRun, settings.perfAlarm, settings.walkSpeed, settings.flamescapesClass, settings.useWalls, graph, points.fire, extinguisherPlan, fireModel, forecastPlan.nodes, forecastPlan.result, planExitNodes, markUsed]);

  // A report entry's name: the fire's storey and how it was placed, and the model.
  const fireLabel = useCallback(
    (node, how) => {
      const storey = node >= 0 ? storeysOf(graph, levels)[node] : -1;
      const model =
        { flamescapes: "Flamescapes", "flamescapes-embers": "Flamescapes with embers", zone: "two-zone rooms", temperature: "heat diffusion", flood: "spread along links", radial: "distance bands" }[
          fireModel
        ] || fireModel;
      return { label: `Fire on ${storey >= 0 ? `storey ${storey + 1}` : "the model"}${how ? `, ${how}` : ""}`, model: `Fire model: ${model}` };
    },
    [graph, levels, fireModel],
  );
  /** The check's options for one plan of the forecast, from the settings. */
  const perfCase = useCallback(
    (presets, plan, planExits) => {
      const criterion = presets.criterionOf(settings.perfCriterion);
      const sensitivity = presets.criterionOf("sensitivity");
      const detected = reportPages.planDetection(plan, { fromRun: settings.perfDetectionFromRun, typed: settings.perfDetection });
      const measured = presets.measuredOf(settings.perfMeasured);
      // Door widths where the IFC gives them; the stricter code minimum elsewhere.
      let widths = null;
      if (settings.perfQueues) {
        const byId = new Map((studio.semantics?.doors || []).map((d) => [d.id, d.widthM]));
        widths = new Float64Array(graph.nodes.length / 3);
        for (const d of graph.doors || []) {
          const w = byId.get(d.itemId);
          if (w > 0) widths[d.node] = w;
        }
      }
      return {
        criterion,
        sensitivity,
        fromCeiling: detected.from === "ceiling",
        detectionFrom: detected.from,
        levels,
        exits: planExits,
        horizon: plan.horizon,
        step: plan.mapStep ?? 0,
        speed: forecastPlan.result === plan ? (forecastPlan.inputs?.walkSpeed ?? settings.walkSpeed) : settings.walkSpeed,
        detection: detected.seconds,
        alarm: settings.perfAlarm,
        pre1: settings.perfPre1,
        pre99: settings.perfPre99,
        noResponse: settings.perfNoResponse / 100,
        measured: measured ? { scale: measured.scale, shape: measured.shape, label: measured.label } : null,
        queues: settings.perfQueues ? { occupantArea: settings.perfOccupantArea, cellArea: (graph.spacing || 0.5) ** 2, widths, defaultWidth: EGRESS_B.doorWidth.value } : null,
        useWalls: settings.useWalls,
      };
    },
    [graph, levels, settings, studio.semantics, forecastPlan.result, forecastPlan.inputs],
  );
  /** A plan's cases into the report, replacing an entry for the same plan. */
  const addToReport = useCallback(
    (entry) =>
      setPerfReport((current) => ({
        graph,
        entries: [...(current.graph === graph ? current.entries : []).filter((e) => e.plan !== entry.plan), entry],
      })),
    [graph],
  );
  const runPerformanceCheck = useCallback(async () => {
    const plan = forecastPlan.result;
    if (!graph || !deadlines || !plan) return;
    try {
      const [{ performanceCheck }, presets, report] = await loadPerf();
      const { detectionFrom, ...opts } = perfCase(presets, plan, planExitNodes());
      const cases = report.planCases(performanceCheck, graph, plan, opts);
      const result = { ...cases[0].result, detectionFrom, criterion: opts.criterion, cases, detectors: plan.detectors, blockedAsked: settings.perfBlockedExit };
      setPerfState({ result, plan });
      addToReport({ plan, ...fireLabel(points.fire ? snap(graph, points.fire) : -1, "as planned"), detectionFrom, cases });
      markUsed("performanceCheck");
    } catch (error) {
      reportError(error, "Performance check");
    }
  }, [graph, deadlines, forecastPlan.result, perfCase, planExitNodes, points.fire, fireLabel, addToReport, settings.perfBlockedExit, reportError, markUsed]);

  // More fires for the report, with a graph or two-zone fire (a Flamescapes
  // run takes minutes a fire: move the fire and plan again instead): on each
  // storey, one at the place nearest an exit (at least 3 m from it, so it
  // can block it) and one at the place farthest from every exit, at most
  // eight, each planned with its deadline map in the forecast worker.
  const addPerfFires = useCallback(async () => {
    const exitNodes = planExitNodes();
    if (!graph || isFlamescapes(fireModel) || !exitNodes.length || perfBatch) return;
    const problem = planSettingsProblem(settings);
    if (problem) {
      toast(problem, { title: "Check the plan's settings", variant: "warning" });
      return;
    }
    try {
      const [{ performanceCheck, walkToExit }, presets, report] = await loadPerf();
      const storeyOf = storeysOf(graph, levels);
      const kinds = graph.kinds || new Uint8Array(graph.nodes.length / 3);
      // Metres, at 1 m/s.
      const walk = walkToExit(graph, exitNodes, { speed: 1, blocked: settings.useWalls ? graph.blocked : null });
      const near = new Map(), far = new Map();
      for (let v = 0; v < walk.length; v += 1) {
        const s = storeyOf[v];
        if (s < 0 || kinds[v] === KIND_DOOR || !(walk[v] < Infinity)) continue;
        if (walk[v] >= 3 && (!near.has(s) || walk[v] < walk[near.get(s)])) near.set(s, v);
        if (!far.has(s) || walk[v] > walk[far.get(s)]) far.set(s, v);
      }
      const origins = [];
      for (const s of [...far.keys()].sort((a, b) => a - b)) {
        if (near.has(s)) origins.push({ node: near.get(s), how: "next to an exit" });
        if (far.get(s) !== near.get(s)) origins.push({ node: far.get(s), how: "farthest from the exits" });
      }
      const chosen = origins.slice(0, 8);
      if (origins.length > chosen.length) toast(`Eight fires of ${origins.length}: the lowest storeys first.`, { variant: "info", ttl: 4000 });
      perfBatchStopRef.current = false;
      setPerfBatch({ done: 0, total: chosen.length });
      for (const [i, origin] of chosen.entries()) {
        if (perfBatchStopRef.current) break;
        const zone = fireModel === "zone" ? zoneInputs() : null;
        const plan = await forecastJob(
          {
            graph: { nodes: graph.nodes, edges: graph.edges, blocked: graph.blocked },
            start: points.start ? snap(graph, points.start) : origin.node,
            exits: exitNodes,
            seed: origin.node,
            model: fireModel,
            steps: settings.fireMaxSteps,
            secondsPerStep: settings.forecastSecondsPerStep,
            threshold: zone ? settings.zoneUntenableC : settings.untenableAbove,
            speed: settings.walkSpeed,
            depart: settings.departAfter,
            limit: settings.doseLimit,
            useWalls: settings.useWalls,
            withMap: true,
            blockNearExit: settings.perfBlockedExit,
            zone: zone?.zone,
          },
          zone?.transfers || [],
          "report",
        );
        const { detectionFrom, ...opts } = perfCase(presets, plan, exitNodes);
        addToReport({ plan, ...fireLabel(origin.node, origin.how), detectionFrom, cases: report.planCases(performanceCheck, graph, plan, opts) });
        setPerfBatch({ done: i + 1, total: chosen.length });
      }
      markUsed("performanceFires");
    } catch (error) {
      if (!isCancelled(error)) reportError(error, "Adding fires to the report");
    } finally {
      setPerfBatch(null);
    }
  }, [graph, fireModel, planExitNodes, perfBatch, settings, levels, points.start, zoneInputs, forecastJob, perfCase, fireLabel, addToReport, toast, reportError, markUsed]);
  const stopPerfFires = useCallback(() => {
    perfBatchStopRef.current = true;
    forecastRef.current.report?.cancel();
  }, []);
  const clearPerfReport = useCallback(() => setPerfReport({ graph: null, entries: [] }), []);

  // The report as a standalone HTML page: every fire and case, the inputs
  // with their sources, and what the check does not show.
  const downloadPerfReport = useCallback(async () => {
    if (!perfReport.length) return;
    try {
      const [{ DOOR_FLOW }, presets, report] = await loadPerf();
      const preset = presets.presetOf(settings.perfOccupancy);
      const criterion = presets.criterionOf(settings.perfCriterion);
      const sensitivity = presets.criterionOf("sensitivity");
      const measured = presets.measuredOf(settings.perfMeasured);
      const detection = settings.perfDetectionFromRun
        ? "From a Flamescapes run: virtual smoke detectors under the ceiling, 7.0 m from the fire or as far as its room allows, at an optical density of 0.097 per metre (runs kept from before read it at head height); with other fire models the time set"
        : `${settings.perfDetection} s after ignition, as set`;
      const meta = {
        title: "Performance-based egress report",
        model: file?.name ? `Model: ${file.name}` : "Model",
        date: new Date().toLocaleString(),
        inputs: [
          ["Occupants", preset.label, preset.source],
          ["Occupant load", `${settings.perfOccupantArea} m² of walkable floor per occupant`, preset.load],
          ["Detection", detection, "NZ C/VM2 2nd edition (2025) Table 3.2.2.2"],
          ["Alarm", `${settings.perfAlarm} s after detection`, "C/VM2 3.3.1.1 and ABCB FSVM 1.1.3 (30 s); SFEG 2025 4.2 (120 s for a staged alarm)"],
          ["Pre-movement, design", `${settings.perfPre1} s and ${settings.perfPre99} s at the 1st and 99th percentiles, a log-normal through both`, preset.source],
          ["Pre-movement, measured (beside)", measured ? `${measured.label}: a Weibull of scale ${measured.scale} s and shape ${measured.shape}` : "none", measured?.source || ""],
          ["Never respond", `${settings.perfNoResponse} %`, presets.NO_RESPONSE_SOURCE],
          [
            "Queues at doors",
            settings.perfQueues
              ? `On: ${DOOR_FLOW.toFixed(2)} persons a second per metre of effective width (the clear width less 0.15 m a side); doors with no width in the IFC at ${EGRESS_B.doorWidth.value} m; everyone walks to the nearest exit with no fire`
              : "Off",
            "C/VM2 3.5.3, after the SFPE hydraulic model",
          ],
          ["Criterion", criterion.label, criterion.source],
          ["Sensitivity cases", sensitivity.label, sensitivity.source],
          ["Tenability", `ISO 13571 fractional effective dose up to ${settings.doseLimit}; walking at ${settings.walkSpeed} m/s`, "The forecast's settings (Simulate step)"],
        ],
        limits: [
          "The fire models are not validated for engineering use: the results rank places and times under the model, not a compliance demonstration.",
          "The deadline map gives each place the best route with foresight of the fire; the queues send everyone to the nearest exit with no fire; walking is not slowed by smoke and every occupant walks at one speed.",
          "Time spent queueing is added to RSET, as in a hand calculation (C/VM2 Eq. 3.1); the dose taken while queueing is not reckoned apart, and stairs' flow is not modelled, only doors'.",
          "The occupant load is spread over the walkable floor, which is less than the gross area the codes divide, so it is a lower bound.",
          "No Saudi numeric criterion was found: the criteria are Singapore SFEG 2025's, the Italian fire code's and PD 7974-6's, offered under SBC 801-CR-2018 104.9.",
          "People with disabilities, assisted evacuation and the M3 management level are not modelled; the room of fire origin is judged like any other place.",
        ],
      };
      const blob = new Blob([report.reportHtml(perfReport, meta)], { type: "text/html" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${(file?.name || "model").replace(/\.ifc(\.gz)?$/i, "")}-performance-report.html`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    } catch (error) {
      reportError(error, "Downloading the report");
    }
  }, [perfReport, settings, file, reportError]);
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
  // A description belongs to its model: another model opened ends a run
  // still going and clears the text. It stayed, or landed, under the new
  // model's facts (wiki/architecture/state-machines.md, SM-5).
  useEffect(() => {
    describeAbortRef.current?.abort();
    setDescribe((d) => (d.text || d.error || d.busy ? { ...d, busy: false, text: "", error: null, progress: null, model: null } : d));
  }, [file]);
  // Egress by storey for the facts (core/codeCheck.js on the graph): the
  // code check's own result when the user ran it, else its measures alone,
  // worked out only while the panel is open (2 to 15 ms on the examples).
  const [describeOpen, setDescribeOpen] = useState(false);
  const describeEgress = useMemo(() => {
    if (!describeOpen || !graph) return null;
    if (code.result) return code.result;
    try {
      return codeCheck(graph, { exits: exits?.final || [], levels, sprinklered: settings.codeSprinklered, useWalls: settings.useWalls });
    } catch {
      return null;
    }
  }, [describeOpen, graph, code.result, exits, levels, settings.codeSprinklered, settings.useWalls]);
  const describeFactsText = useMemo(
    () =>
      describeFacts({
        file,
        stats: modelInfo?.stats,
        semantics: studio.semantics,
        identification: identification?.answers,
        graph: graph ? { stats: graph.stats, cores: graph.cores, outdoorSpaces: graph.outdoorSpaces } : null,
        egress: describeEgress,
        codeChecked: Boolean(code.result),
        levelShift,
        route: path,
        nearerExit,
        walkSpeed: settings.walkSpeed,
        fire,
        combustion: fire.combustion,
        plan: forecastPlan?.result ? forecastPlan : null,
        rl: rl.result,
        crowd: crowdResult
          ? {
              result: crowdResult,
              withFire: crowdAvoiding,
              occupantArea: settings.perfOccupantArea,
              exits: settings.crowdExits,
              fluctuate: settings.crowdFluctuate,
              steps: settings.crowdSteps,
              runs: crowdState.runs,
            }
          : null,
        slime: slime.result,
        perf,
        sprinklers: settings.sprinklers ? sprinklerPlan : null,
        systems: settings.suppressionSystems ? systemsPlan : null,
        extinguishers: settings.extinguishers ? extinguisherPlan : null,
      }).text,
    [
      file,
      modelInfo,
      studio.semantics,
      identification,
      graph,
      describeEgress,
      code.result,
      levelShift,
      path,
      nearerExit,
      settings.walkSpeed,
      fire,
      forecastPlan,
      rl.result,
      crowdResult,
      crowdAvoiding,
      settings.perfOccupantArea,
      settings.crowdExits,
      settings.crowdFluctuate,
      settings.crowdSteps,
      crowdState.runs,
      slime.result,
      perf,
      settings.sprinklers,
      sprinklerPlan,
      settings.suppressionSystems,
      systemsPlan,
      settings.extinguishers,
      extinguisherPlan,
    ],
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
        // The human check's token (lib/turnstile.js); none if the site has no site key.
        const token = await humanToken({ signal: abort.signal });
        const { text, model } = await describeInCloud(describeFactsText, { signal: abort.signal, token });
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
  // and when its result appears (a new model, graph, point, route or finished
  // fire) it is held on screen for GUIDED_HOLD_MS, or on a phone until Next,
  // before the next stage; Stop, or a stage that waits too long, ends the run.
  const [guided, setGuided] = useState(null);
  // The settings the run changes, as they were before it (key: value), given
  // back when it is stopped, closed or left for the guided runs: a run used
  // to leave its fire, doors and people set for the user's next model. Kept
  // in the browser too, so a reload mid-run gives them back (useStudio.js).
  const guidedBeforeRef = useRef(null);
  const startGuided = useCallback((run) => {
    guidedBeforeRef.current = {};
    keepGuidedBefore(null);
    setGuided({ run, stage: "open", acted: null, before: null, error: null, ready: null, manual: isPhone() });
  }, []);
  // A setting the run changes: the user's value is kept first.
  const guidedSet = (patch) => {
    const before = guidedBeforeRef.current;
    if (before) {
      for (const key of Object.keys(patch)) if (!(key in before)) before[key] = settings[key];
      keepGuidedBefore(withoutSessionOnly(before));
    }
    setSettings(patch);
  };
  // Its field note was shown in the run's own card: not again in the panel.
  const { closeLatest } = notes;
  // The run ends and the user's settings come back.
  const endGuided = useCallback(() => {
    setGuided(null);
    closeLatest();
    const before = guidedBeforeRef.current;
    guidedBeforeRef.current = null;
    keepGuidedBefore(null);
    if (before && Object.keys(before).length) setSettings(before);
  }, [closeLatest, setSettings]);
  // Playing: not yet done or stopped. While it plays its card's Stop is the
  // one Stop (the playback's and the panel's are not shown), and it ends a
  // fire playing or starting too, as Stop fire does (M7). Once the run has
  // ended the panel's Start and Stop are back, and Close only closes the card.
  const guidedPlaying = Boolean(guided) && guided.stage !== "done" && !guided.error;
  const stopGuided = useCallback(() => {
    if (guidedPlaying && (fire.running || fire.starting)) stopFire();
    endGuided();
  }, [guidedPlaying, fire.running, fire.starting, stopFire, endGuided]);
  // A stage whose result is on screen goes on after the hold; on a phone, at Next.
  const nextGuided = useCallback(() => {
    setGuided((g) => {
      if (!g?.ready) return g;
      const ids = stagesFor(g.run).map((s) => s.id);
      return { ...g, stage: ids[ids.indexOf(g.stage) + 1], acted: null, before: null, ready: null };
    });
  }, []);
  useEffect(() => {
    if (!guided || guided.stage === "done" || guided.error) return undefined;
    if (guided.ready) {
      if (guided.manual) return undefined;
      const timer = setTimeout(nextGuided, Math.max(0, guided.ready + GUIDED_HOLD_MS - Date.now()));
      return () => clearTimeout(timer);
    }
    // Deferred a tick, so the stage's state changes happen outside the effect;
    // a re-run before then cancels it, so no stage acts twice.
    const timer = setTimeout(() => guidedStep(guided), 0);
    return () => clearTimeout(timer);
  });
  const guidedStep = (guided) => {
    const { run, stage, acted, before } = guided;
    // The stage's result is on screen: held (the effect above), then the next stage.
    const advance = () => setGuided((g) => (g && g.run === run && g.stage === stage && !g.ready ? { ...g, ready: Date.now() } : g));
    // Every setting a stage changes keeps the user's value first.
    const setSettings = guidedSet;
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
        const v = farthestFromExits(graph, exits?.final);
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
    } else if (stage === "map") {
      // The egress map's readings drawn: each room's worst walk and the dead ends.
      if (acted !== stage) {
        setTab("route");
        setSettings({ egressRoutes: true, egressDeadEnds: true });
        act(() => runEgress("map"), egress.result);
      } else if (egress.error) return fail(egress.error);
      else if (egress.result && egress.result !== before && !egress.busy) advance();
    } else if (stage === "measure") {
      // The wheelchair audit laid on the floors, every room's route kept drawn over it.
      if (acted !== stage) {
        setSettings({ accessOverlay: true });
        act(() => runEgress("fine"), egress.fine);
      } else if (egress.error) return fail(egress.error);
      else if (egress.fine && egress.fine !== before && !egress.busy) advance();
    } else if (stage === "origin") {
      const f = run.fire;
      if (acted !== stage) {
        setTab("simulate");
        if (f.model === "flamescapes") {
          // A phone runs a shorter fire where the run says so. The run's own
          // length, not until the fire service arrives (core/fireService.js);
          // what the graph shows, what burns and how fast, sprinklers and the
          // grid, where the run says so.
          const horizon = (isPhone() && f.phoneHorizon) || f.horizon || 60;
          setSettings({
            flamescapes: true,
            fireModel: "flamescapes",
            flamescapesDoors: f.doors || "open",
            flamescapesHorizon: horizon,
            flamescapesUntilService: false,
            ...(f.show ? { flamescapesShow: f.show } : {}),
            ...(f.fireClass ? { flamescapesClass: f.fireClass } : {}),
            ...(f.growth ? { flamescapesGrowth: f.growth } : {}),
            ...(f.sprinklers ? { sprinklers: true } : {}),
            ...(f.calibratedGrid ? { flamescapesCalibratedGrid: true, flamescapesGridPolicy: "standard" } : {}),
            ...(f.onRoute ? { reroute: true } : {}),
          });
        } else {
          setSettings({ flamescapes: false, fireModel: f.model, reroute: true });
        }
        if (f.model === "flamescapes" && !f.onRoute) act(() => void findIgnition("residential"), points.fire);
        else {
          // Part-way along the route.
          const v = nodeAlong(path?.node_ids, f.along ?? 0.5);
          if (v < 0) return fail("The route has no nodes to place the fire on.");
          act(() => handlePick("fire", at(v)), points.fire);
        }
      } else if (f.model === "flamescapes" && !f.onRoute && !guided.picked && ignition.candidates) {
        // The likeliest origin near the route, on a floor that reaches an exit,
        // clear of the start and the exits (core/guided.js guidedOrigin).
        const pick = guidedOrigin(graph, {
          candidates: ignition.candidates,
          route: path?.node_ids || [],
          exits: exits?.final || [],
          start: points.start ? snap(graph, points.start) : -1,
        });
        if (!pick) return fail("No place for the fire was found in the model.");
        const cls = pick.candidate?.scenario.fireClass;
        // The caption names what burns: a pan of oil (class K) or a liquid (B) burns as a pool.
        const label = pick.candidate
          ? `${pick.candidate.scenario.label}${cls === "K" ? ", a pan of oil burning as a pool" : cls === "B" ? ", a pool of flammable liquid" : ""}`
          : null;
        setGuided((g) => (g && g.run === run ? { ...g, picked: true, originLabel: label } : g));
        handlePick("fire", at(pick.node));
        if (FIRE_CLASSES[cls]) setSettings({ flamescapesClass: cls });
        // As picking a likely ignition point by hand says (useIgnition).
        if (pick.candidate) {
          markUsed("ignition");
          markFound("ignition", { label: pick.candidate.scenario.label });
        }
      } else if (
        points.fire &&
        points.fire !== before &&
        (f.model !== "flamescapes" || f.onRoute || guided.picked) &&
        // Under sprinklers, once the heads are laid out: a run started before has none.
        (!f.sprinklers || (sprinklerPlan && !sprinklerState.busy))
      )
        advance();
    } else if (stage === "people") {
      // Everyone on the floors: walking out (the way out), or leaving with the
      // fire, the runner on the forecast's timeline once the plan is made; dots on a phone.
      if (acted !== stage) {
        const style = isPhone() ? "dots" : "figures";
        act(
          () =>
            setSettings(
              run.people === "walk"
                ? { crowd: true, crowdWithFire: false, crowdStyle: style }
                : { crowd: true, crowdWithFire: true, crowdStairRunner: true, crowdRunner: "forecast", crowdStyle: style },
            ),
          null,
        );
      } else if (crowdResult) advance();
    } else if (stage === "fire") {
      if (acted !== stage) act(() => void startFire(), null);
      else if (fire.running && !guided.sawRunning) setGuided((g) => (g && g.run === run ? { ...g, sawRunning: true } : g));
      else {
        const next = fireStageNext({ sawRunning: guided.sawRunning, running: fire.running, step: fire.step, stopped: fireStoppedRef.current });
        if (next === "advance") advance();
        else if (next === "stopped") fail("The fire was stopped, so the guided run ends here.");
      }
    } else if (stage === "plan") {
      if (acted !== stage) act(() => void planForecast(), forecastPlan.result);
      else if (forecastPlan.result && forecastPlan.result !== before && !forecastPlan.busy) advance();
    } else if (stage === "tour") {
      // The fire plays again from its frames, the people and the runner with
      // it, while the camera turns once round the building.
      if (acted !== stage) {
        act(() => {
          viewerRef.current?.tour?.({ seconds: isPhone() ? 45 : 75 });
          void startFire();
        }, null);
      } else if (fire.running && !guided.sawReplay) setGuided((g) => (g && g.run === run ? { ...g, sawReplay: true } : g));
      else if (guided.sawReplay && !fire.running) advance();
    }
  };
  // A stage that waits too long ends the run (a fire on the GPU or a CPU core
  // can take minutes); not one whose result is on screen, held or waiting for Next.
  const guidedStage = guided && !guided.error && guided.stage !== "done" && !guided.ready ? guided.stage : null;
  useEffect(() => {
    if (!guidedStage) return undefined;
    const limit = guidedStage === "fire" ? 15 * 60000 : 4 * 60000;
    const timer = setTimeout(() => setGuided((g) => (g && g.stage === guidedStage ? { ...g, error: "This step took too long." } : g)), limit);
    return () => clearTimeout(timer);
  }, [guidedStage]);
  // A finished run ends on its field note: found, with what was found, or
  // still a question if this run did not show it.
  const guidedNote = useMemo(() => {
    if (guided?.stage !== "done" || !guided.run.note) return null;
    const note = fieldNote(guided.run.note);
    if (!note || !notes.found[note.id]) return note ? { teaser: note.teaser, text: null } : null;
    // This run's own walk, not the one the note was first found with.
    const own = note.id === "walk-out" && path?.found !== false && path?.length > 0 ? note.text({ length: path.length }) : null;
    return { teaser: note.teaser, text: own || notes.found[note.id].text };
  }, [guided, notes.found, path]);
  // At the end the result is brought into view: its step, its folded
  // sections opened (ui/reveal.js) and its card scrolled to the top of the
  // panel; "Done: the result is in the panel" was not true on screen.
  const guidedDone = guided?.stage === "done" ? guided.run : null;
  useEffect(() => {
    if (!guidedDone) return undefined;
    const place = guidedResultPlace(guidedDone);
    setTab(place.tab);
    revealSections(place.sections);
    // After the step and its opened sections have drawn; then the request
    // ends, so the sections open as usual when the step is shown again.
    const timer = setTimeout(() => {
      document.querySelector(place.scroll)?.scrollIntoView({ block: "start" });
      revealSections(null);
    }, 150);
    return () => {
      clearTimeout(timer);
      revealSections(null);
    };
  }, [guidedDone]);
  // And the card leads with its one or two numbers (core/guided.js).
  const guidedResult = useMemo(() => {
    if (!guidedDone) return null;
    const plan = forecastPlan.result;
    const depart = forecastPlan.inputs?.departAfter || 0;
    const safe = plan?.front?.find((p) => p.dose <= (forecastPlan.inputs?.doseLimit ?? Infinity));
    const storeys = (egress.fine?.storeys || []).filter((s) => s.audit?.rooms.onFoot > 0);
    return guidedHeadline(guidedDone, {
      route: path ? { found: path.found !== false, length: path.straight_length ?? path.length } : null,
      crowd: crowdResult ? { people: crowdResult.people, lastOut: crowdResult.lastOut } : null,
      reroute: dynamicPath ? { found: dynamicPath.found, step: dynamicPath.step, firstChange: dynamicPath.firstChange, clearBy: dynamicPath.clearBy } : null,
      combustion: fire.combustion
        ? {
            peak: fire.combustion.peak ?? 0,
            detectedAt: fire.combustion.detectedAt,
            firstSprinkler: fire.combustion.sprinklers?.first,
            // The run card's one line (core/calibrationStatus.js).
            verdict: fire.combustion.cell > 0 ? validationVerdict(fire.combustion.cell, { peakKW: fire.combustion.peak ?? null, horizon: fire.combustion.horizon ?? null }) : null,
          }
        : null,
      plan: plan ? { out: safe ? safe.arrival - depart : null, leave: plan.leaveBy?.good, leaveAtLeast: Boolean(plan.leaveBy?.horizon) } : null,
      egress: egress.result
        ? {
            farthest: egress.result.summary.farthest,
            deadEnds: egress.result.deadEnds?.corridors.length,
            byChair: storeys.reduce((n, s) => n + s.audit.rooms.byChair, 0),
            onFoot: storeys.reduce((n, s) => n + s.audit.rooms.onFoot, 0),
          }
        : null,
    });
  }, [guidedDone, forecastPlan.result, forecastPlan.inputs, egress.result, egress.fine, path, dynamicPath, fire.combustion, crowdResult]);
  // A stage's caption with what it is on: what burns, once picked (a pan of
  // oil burns as a pool), and the GPU or CPU a Flamescapes fire runs on.
  const guidedCaption = useMemo(() => {
    if (!guidedPlaying) return null;
    const base = stagesFor(guided.run).find((s) => s.id === guided.stage)?.caption;
    if (guided.stage === "origin" && guided.originLabel) return `Placing the fire: ${guided.originLabel}`;
    const backend = fire.combustion?.backend;
    if (guided.stage === "fire" && isFlamescapes(fire.model) && backend) return `${base}; running on ${backend.kind === "gpu" ? backend.name : "one CPU core"}`;
    return null;
  }, [guidedPlaying, guided, fire.combustion?.backend, fire.model]);

  // ------------------------------------------------- back to the guided runs

  // The welcome screen and its guided runs again, without a reload, from a
  // load, a guided run, a fire playing or one that has ended. A fire playing
  // or starting ends as Stop fire ends it (M7); a guided run ends as its Stop
  // ends it, with the user's settings back; a recording is saved as its Stop
  // saves it (M9); then the model goes as when another is opened (L1): its
  // load, graph build, agent, plan, report fires, egress map and slime mould
  // end, and what is kept with the graph drops itself (a Flamescapes run
  // still computing or repeating is ended by the graph's effect below: M4).
  // The viewer stays, without its model (ViewerManager.unloadModel).
  const backToWelcome = useCallback(() => {
    if (fire.running || fire.starting) stopFire();
    endGuided();
    if (recordRef.current && recordRef.current !== RECORDER_STARTING) void toggleRecording();
    ++loadSeqRef.current;
    samplerRef.current?.cancel();
    endFireStart();
    rlRef.current?.cancel();
    dropPlan();
    stopPerfFires();
    egressRef.current?.cancel();
    slimeRef.current?.cancel();
    geometryRef.current = null;
    geometryPromiseRef.current = null;
    studio.beginLoad(null);
    studio.setLoadState({ busy: false, stage: null, detail: null });
    // No newer load takes the busy flag from a build still ending (G1).
    studio.setGraphBusy(false);
    setLastSample(null);
    setLevels([]);
    setLevelShift(0);
    setSection(null);
    viewerRef.current?.unloadModel?.();
    colours.set(null);
    setTab("model");
  }, [endGuided, fire.running, fire.starting, stopFire, toggleRecording, endFireStart, dropPlan, stopPerfFires, studio, colours]);

  // ------------------------------------------------- run again, in the view

  // The Flamescapes fire started again with the settings as they are, as the
  // panel's Start fire starts it: offered in the view once a run has ended or
  // been stopped, not while one computes (a plan's or the repeat runs too), a
  // guided run plays or a pick waits for its click; and saying so when the
  // next run will differ from the last.
  const replay = useMemo(() => {
    const started = fire.started;
    const offered =
      Boolean(started && graph && points.fire) &&
      isFlamescapes(fire.model) &&
      isFlamescapes(fireModel) &&
      !fire.running &&
      !fire.starting &&
      !forecastPlan.busy &&
      fire.combustion?.repeats?.status !== "running" &&
      !guided &&
      !pickMode;
    if (!offered) return null;
    const changed =
      points.fire !== started.fire ||
      runHorizon(settings) !== runHorizon(started.settings) ||
      RUN_SETTINGS.some((key) => settings[key] !== started.settings[key]);
    return { changed };
  }, [fire.started, fire.model, fire.running, fire.starting, fire.combustion?.repeats?.status, graph, points.fire, fireModel, forecastPlan.busy, guided, pickMode, settings]);

  // ------------------------------------------------- hints

  // One hint for the step on screen (core/fieldNotes.js nextHint), none while
  // a model loads, a graph builds, a fire runs or a guided run plays.
  const hintState = useMemo(
    () => ({
      step: tab,
      model: Boolean(modelInfo),
      graph: Boolean(graph),
      split: Boolean(graph && graphIsSplit(graph.stats)),
      route: Boolean(path && path.found !== false),
      appearance: settings.modelAppearance,
      fireOrigin: Boolean(points.fire),
      busy: Boolean(studio.loadState.busy || studio.graphBusy || fire.running || fire.starting || guided),
    }),
    [tab, modelInfo, graph, path, settings.modelAppearance, points.fire, studio.loadState.busy, studio.graphBusy, fire.running, fire.starting, guided],
  );
  const hint = nextHint(hintState, { used: notes.used, dismissed: notes.dismissed, on: notes.hints });
  // A hint shows only once the step has been on screen for a moment, so it
  // does not flash past while the user moves on.
  const hintKey = hint ? `${tab}:${hint.id}` : "";
  const [settledHint, setSettledHint] = useState("");
  useEffect(() => {
    if (!hintKey) return undefined;
    const timer = setTimeout(() => setSettledHint(hintKey), HINT_SETTLE_MS);
    return () => clearTimeout(timer);
  }, [hintKey]);
  const shownHint = hint && settledHint === hintKey ? hint : null;
  const takeHint = useCallback(
    (h) => {
      if (h.action?.tab) setTab(h.action.tab);
      if (h.action?.settings) setSettings(h.action.settings);
      notes.dismiss(h.id);
    },
    [notes, setSettings],
  );
  const closeNotes = useCallback(() => setNotesOpen(false), []);

  // ------------------------------------------------- plan with the forecast

  const pointsOf = useCallback(
    (nodes) => (nodes || []).map((i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]),
    [graph],
  );

  const planForecast = useCallback(async () => {
    const exitNodes =
      settings.forecastExits === "model" && exits?.final.length
        ? exits.final
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
    if (planRunningRef.current) forecastRef.current.plan?.cancel();
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
        // A run the health guard stopped is cut short: nodes it never reached
        // would read as never untenable, so the plan does not use it.
        if (done?.health?.ok === false) {
          planRunningRef.current = false;
          studio.setForecastPlan(NO_PLAN);
          toast(`The fire run was stopped at ${Math.round(done.health.time)} s: ${healthReason(done.health)}. The plan does not use a stopped run.`, { variant: "warning", ttl: 8000 });
          return;
        }
        // Stacked here from the run's frames, sent as copies (the worker no
        // longer sends them twice: review 6).
        const { n, K } = done;
        const T = new Float32Array(K * n), vis = new Float32Array(K * n), co = new Float32Array(K * n);
        // CO2 and O2 when the run recorded them: the plan's doses then use them (core/tenability.js).
        const extras = Boolean(run.frames[0]?.co2 && run.frames[0]?.o2);
        const co2 = extras ? new Float32Array(K * n) : null, o2 = extras ? new Float32Array(K * n) : null;
        // The ceiling detectors' visibility, when the run read them (runs kept before did not).
        const D = run.frames.every((f) => f.detectors?.length) ? run.frames[0].detectors.length : 0;
        const detectors = D ? new Float32Array(K * D) : null;
        run.frames.forEach((f, k) => {
          T.set(f.T, k * n);
          vis.set(f.vis, k * n);
          co.set(f.co, k * n);
          if (extras) {
            co2.set(f.co2, k * n);
            o2.set(f.o2, k * n);
          }
          if (D) detectors.set(f.detectors, k * D);
        });
        frames = {
          n,
          K,
          frameSeconds: done.frameSeconds,
          T,
          vis,
          co,
          ...(extras ? { co2, o2 } : {}),
          ...(D ? { detectors, detectorCount: D } : {}),
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
          // A second map with the exit nearest the fire blocked, for the performance check.
          blockNearExit: settings.perfBlockedExit,
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
      markUsed("plan");
      if (result.deadlines) {
        let places = 0;
        for (const d of result.deadlines) if (d === -Infinity) places += 1;
        if (places > 0) markFound("no-safe-time", { places });
      }
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
      // Overtaken: a newer request or a change owns the state.
      if (gen !== planGenRef.current) return;
      // Still this plan's: it ends, whatever ended it. A cancel no newer
      // request asked for left it busy, its button spinning for good (SM-2).
      planRunningRef.current = false;
      studio.setForecastPlan(NO_PLAN);
      if (!isCancelled(error)) reportError(error, "Planning with the forecast failed");
    }
  }, [graph, points, settings, studio, exits, forecastJob, pointsOf, toast, reportError, showPlanRoutes, fireModel, zoneInputs, runCombustion, markFound, markUsed]);

  // A Flamescapes run belongs to the graph it was made on: a new graph ends
  // it and lets its frames go.
  useEffect(() => {
    if (combustionRef.current && combustionRef.current.graph !== graph) {
      if (!combustionRef.current.done || combustionRef.current.repeating) flamescapesRef.current?.cancel();
      combustionRef.current = null;
      playingRunRef.current = null;
      viewerRef.current?.setParticles(null);
      viewerRef.current?.setSmokeCloud(null);
      viewerRef.current?.setDrawRate(0);
    }
  }, [graph]);

  // Switching what a Flamescapes run shows (core/measures.js), or the limits
  // of its tenability index, redraws the frame on screen, also once the run
  // has ended.
  useEffect(() => {
    const run = playingRunRef.current;
    showRef.current = measureView(settings, Boolean(run?.frames?.[0]?.co2));
    tenabilityRef.current = { limitC: settings.flamescapesUntenableC, visibilityLimit: settings.flamescapesVisibility, smoke: settings.flamescapesSmoke };
    if (!run || !isFlamescapes(fire.model) || graphColours !== "fire") return;
    paintCombustion(viewerRef.current, run, fire.step);
    // Only a change of view or limits redraws; frames redraw themselves as they play.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.flamescapesShow, settings.flamescapesHeatScale, settings.flamescapesSmokeMeasure, settings.flamescapesGasMeasure, settings.flamescapesUntenableC, settings.flamescapesVisibility, settings.flamescapesSmoke]);

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

  // The topology viewer replaces the model's viewer, which is made new and
  // empty on the way back. The model, graph and points go with the old one:
  // kept, the panels offered a graph and a route with nothing drawn, and a
  // build said "Load an IFC model first" (SM-7).
  const switchMode = useCallback(
    (next) => {
      if (next === mode) return;
      if (mode === "ifc") {
        ++loadSeqRef.current;
        samplerRef.current?.cancel();
        endFireStart();
        rlRef.current?.cancel();
        dropPlan();
        geometryRef.current = null;
        studio.beginLoad(null);
        studio.setLoadState({ busy: false, stage: null, detail: null });
      }
      setMode(next);
    },
    [mode, studio, dropPlan, endFireStart],
  );

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
          if (modelInfo && !event.repeat) void buildGraph();
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
    if (graph && exits?.final.length) {
      items.push({ label: exits.finalSource ? "Exits out of the building (rings)" : "Exit doors (rings)", color: "var(--viz-exit)" });
    }
    if (points.fire) items.push({ label: "Fire origin (spike)", color: "var(--viz-fire)", dot: true });
    if (settings.sprinklers && sprinklerPlan?.heads?.length) items.push({ label: "Sprinkler heads (blue once open, with their spray)", color: "#0284c7", dot: true });
    if (settings.extinguishers && extinguisherPlan?.sites?.length) items.push({ label: "Extinguishers", color: "#dc2626", dot: true });
    if (settings.suppressionSystems && systemsPlan?.special?.some((r) => r.system === "clean-agent")) items.push({ label: "Clean agent rooms (filled once flooded)", color: "#0d9488" });
    if (settings.suppressionSystems && systemsPlan?.special?.some((r) => r.system === "foam")) items.push({ label: "Foam rooms (filled once foamed)", color: "#ca8a04" });
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
    exits,
    path,
    dynamicPath,
    alternatives,
    navCheck,
    settings.showNavGraph,
    forecastPlan.result,
    planRoutesShown,
    graphColours,
    fire.model,
    settings.sprinklers,
    sprinklerPlan?.heads?.length,
    settings.extinguishers,
    extinguisherPlan?.sites?.length,
    settings.suppressionSystems,
    systemsPlan?.special,
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
            onChange={switchMode}
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
          <FieldNotesButton found={notes.found} onOpen={() => setNotesOpen(true)} />
          <Button variant="ghost" onClick={() => setMethodsOpen(true)} aria-haspopup="dialog" aria-label="Methods">
            <Icon name="book" size={18} />
            <span className="header__label">Methods</span>
          </Button>
          {/* The guide (public/docs/), named in full: Vite's dev server answers docs/ with the app.
              header__group--wide hides it at 900 px and below, where the header has no room. */}
          <a
            className="btn btn--ghost header__group--wide"
            href={`${import.meta.env.BASE_URL}docs/index.html`}
            target="_blank"
            rel="noopener"
            title="The guide to Studio, in a new tab"
            style={{ textDecoration: "none" }}
          >
            Docs
          </a>
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

      <div className="app-body" data-panel-collapsed={!panelOpen} data-panel-tall={panelTall}>
        {mode === "ifc" ? (
          <Viewport
            viewerRef={viewerRef}
            onPick={handlePick}
            onFile={handleFile}
            theme={theme}
            pickMode={pickMode}
            hasModel={Boolean(modelInfo)}
            loadState={studio.loadState}
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
                  ? (measureView(settings, Boolean(fire.combustion?.extras)).legend ?? {
                      title: "Gas temperature at head height (°C)",
                      // The fixed ramp, or one fitted to the run, chosen in the legend: drawing only.
                      max: settings.flamescapesHeatScale === "run" ? fittedHeatPeak(runHotGas(playingRunRef.current)) : ZONE_COLOUR_PEAK,
                      note:
                        settings.flamescapesHeatScale === "run"
                          ? "Fitted to this run's hottest gas at head height so far: the colours only, not the fire; a colour means another temperature in another run."
                          : "Flamescapes: the solver's gas temperature 1.8 m above each node's floor.",
                      fit: {
                        on: settings.flamescapesHeatScale === "run",
                        onToggle: () => setSettings({ flamescapesHeatScale: settings.flamescapesHeatScale === "run" ? "fixed" : "run" }),
                      },
                    })
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
                      note: `Red: at or over the ${code.result.limit.toFixed(1)} m travel distance (Table 1017.2, the stricter of SBC 801-CR-2018 and IBC 2021).`,
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
            guided={
              guided
                ? {
                    title: guided.run.title,
                    stages: stagesFor(guided.run),
                    stage: guided.stage,
                    error: guided.error,
                    note: guidedNote,
                    result: guidedResult,
                    caption: guidedCaption,
                    playing: guidedPlaying,
                    // Its result on screen and waiting for Next (a phone).
                    waiting: Boolean(guided.ready && guided.manual),
                  }
                : null
            }
            onNextGuided={nextGuided}
            onOpenNotes={() => setNotesOpen(true)}
            onStopGuided={stopGuided}
            onHome={backToWelcome}
            replay={replay}
            onReplay={startFire}
            recording={recording}
            onRecord={toggleRecording}
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
            <button
              type="button"
              className="panel__grow"
              aria-pressed={panelTall}
              onClick={() => setPanelTall((v) => !v)}
            >
              {panelTall ? "Shorter panel" : "Taller panel"}
            </button>
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
                  onDescribeOpen={setDescribeOpen}
                  onOpenRooms={handleFile}
                  onExportRooms={exportLod1}
                  linked={{ list: linkedList, busy: linkedBusy }}
                  onAddLinked={addLinkedModel}
                  onAlignLinked={alignLinkedModel}
                  onRemoveLinked={removeLinkedModel}
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
                  nearerExit={nearerExit}
                  onGoTo={setTab}
                  code={{ ...code, shown: code.shown && graphColours === "travel" }}
                  onCodeCheck={runCodeCheck}
                  onShowTravel={showTravel}
                  perf={perf}
                  perfReady={Boolean(deadlines)}
                  onPerformanceCheck={runPerformanceCheck}
                  perfReport={perfReport}
                  perfBatch={perfBatch}
                  perfCanAddFires={Boolean(graph) && !isFlamescapes(fireModel) && planExitNodes().length > 0}
                  onAddPerfFires={addPerfFires}
                  onStopPerfFires={stopPerfFires}
                  onDownloadPerfReport={downloadPerfReport}
                  onClearPerfReport={clearPerfReport}
                  crowd={{
                    result: crowdResult,
                    runs: crowdResult ? crowdState.runs : null,
                    runner: crowdResult ? crowdState.runner : null,
                    gpu: crowdResult ? viewerRef.current?.gpuName?.() : null,
                    fireActive: fire.running || fire.step > 0,
                    avoiding: crowdAvoiding,
                    runnerKind: crowdRunnerKind,
                    planned: Boolean(crowdPlanRoute),
                    busy: crowdState.key === crowdKey && crowdState.busy,
                    error: crowdState.key === crowdKey ? crowdState.error : null,
                    time: crowdTime,
                    playing: crowdClockRef.current.playing,
                  }}
                  onPlayCrowd={playCrowd}
                  onRestartCrowd={restartCrowd}
                  slime={slime}
                  slimeFireReady={Boolean(graph) && (isFlamescapes(fire.model) ? Boolean(combustionRef.current?.done) : fire.model !== "zone" && fire.step > 0)}
                  onGrowSlime={growSlime}
                  onClearSlime={clearSlime}
                  egress={egress}
                  onMapEgress={() => runEgress("map")}
                  onMeasureFinely={() => runEgress("fine")}
                  onClearEgress={clearEgress}
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
                  onExportTable={exportTable}
                  onForgetRuns={forgetRuns}
                  sprinklerPlan={settings.sprinklers ? sprinklerPlan : null}
                  sprinklerBusy={sprinklerState.busy}
                  onLayOutSprinklers={layOutSprinklers}
                  overheadCount={overheadMeshes ? new Set(overheadMeshes.map((m) => `${m.source}:${m.itemId}`)).size : null}
                  extinguisherPlan={settings.extinguishers ? extinguisherPlan : null}
                  systemsPlan={settings.suppressionSystems ? systemsPlan : null}
                  systemsBusy={systemsState.busy}
                  onLayOutSystems={layOutSystems}
                  extinguisherBusy={extinguisherState.busy}
                  onLayOutExtinguishers={layOutExtinguishers}
                  extinguisherAttempt={settings.extinguishers ? extinguisherAttempt : null}
                  onExtinguisherAttempt={runExtinguisherAttempt}
                  pace={lastPace}
                  gpuSpeed={gpuRateRef.current || null}
                  guided={guidedPlaying}
                />
              )}
            </div>

            <HintCard
              key={notes.latest && !guided ? `note:${notes.latest.teaser}` : shownHint?.id || "none"}
              hint={shownHint}
              step={tab}
              latest={guided ? null : notes.latest}
              onAction={takeHint}
              onDismiss={notes.dismiss}
              onCloseLatest={notes.closeLatest}
              onOpenNotes={() => setNotesOpen(true)}
            />

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

      <FieldNotesDialog
        open={notesOpen}
        onClose={closeNotes}
        found={notes.found}
        hints={notes.hints}
        onHints={notes.setHints}
        onResetHints={notes.resetHints}
      />
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
