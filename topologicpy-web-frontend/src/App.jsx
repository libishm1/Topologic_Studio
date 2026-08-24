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
import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, legacyApi, openFireStream } from "./lib/api.js";
import { PhaseTimer, formatBytes, formatMs } from "./lib/perf.js";
import { useStudio } from "./state/useStudio.js";
import { AboutPanel, ModelPanel, RoutePanel, SimulatePanel } from "./ui/Panels.jsx";
import { TopologyPanel } from "./ui/TopologyPanel.jsx";
import { Viewport } from "./ui/Viewport.jsx";

import { Badge, Button, FileButton, Segmented } from "./ui/primitives.jsx";
import logo from "./assets/topologicStudio-white-logo400x400.png";

// Three plus OrbitControls only matter in this mode, so the chunk is loaded
// on demand rather than shipped to everyone who opens the IFC viewer.
const TopologyViewer = lazy(() => import("./viewer/TopologyViewer.jsx"));

const IFC_TABS = [
  { id: "model", label: "Model" },
  { id: "route", label: "Route" },
  { id: "simulate", label: "Simulate" },
  { id: "about", label: "About" },
];

const TOPOLOGY_TABS = [
  { id: "topology", label: "Topology" },
  { id: "about", label: "About" },
];

const MODES = [
  { value: "ifc", label: "IFC", hint: "Fragment viewer, egress graph and fire simulation" },
  {
    value: "topology",
    label: "Topology JSON",
    hint: "TopologicPy JSON contract viewer (legacy workflow)",
  },
];

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
  const workerRef = useRef(null);
  const geometryRef = useRef(null);
  // Fire accumulators live in refs: they are written from SSE callbacks that
  // outlive any single render, and must not be re-created when the callback
  // identity changes mid-simulation.
  const burningRef = useRef(new Set());
  const temperatureRef = useRef(new Map());
  const [tab, setTab] = useState("model");
  const [panelOpen, setPanelOpen] = useState(true);
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
    toast,
    reportError,
  } = studio;

  // ------------------------------------------------------------- the worker

  const getWorker = useCallback(() => {
    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL("./workers/sampler.worker.js", import.meta.url),
        { type: "module" },
      );
    }
    return workerRef.current;
  }, []);

  useEffect(() => () => workerRef.current?.terminate(), []);

  const sample = useCallback(
    (payload, transfers) =>
      new Promise((resolve, reject) => {
        const worker = getWorker();
        const id = Math.random().toString(36).slice(2);
        const onMessage = (event) => {
          if (event.data?.id !== id) return;
          worker.removeEventListener("message", onMessage);
          if (event.data.ok) resolve(event.data.result);
          else reject(new Error(event.data.error || "Sampling failed."));
        };
        worker.addEventListener("message", onMessage);
        worker.postMessage({ id, payload }, transfers);
      }),
    [getWorker],
  );

  // --------------------------------------------------------------- IFC load

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

      studio.beginLoad(nextFile);
      geometryRef.current = null;
      setLastSample(null);
      studio.setLoadState({ busy: true, stage: "reading", detail: nextFile.name });

      try {
        const result = await viewer.loadIfc(nextFile, {
          onProgress: ({ stage, detail }) => {
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

        studio.setModelInfo(result);
        viewer.setModelAppearance(settings.modelAppearance);
        setTab("model");

        void studio.refreshCacheInfo();

        toast(
          result.cached
            ? `Loaded from cache in ${formatMs(result.totalMs)}.`
            : `Converted and cached in ${formatMs(result.totalMs)}.`,
          { title: nextFile.name, variant: "success", ttl: 5000 },
        );
      } catch (error) {
        if (error?.message !== "cancelled") {
          reportError(error, "Could not load the IFC file");
        }
      } finally {
        studio.setLoadState({ busy: false, stage: null, detail: null });
      }
    },
    [reportError, settings.modelAppearance, studio, toast],
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
    const timer = new PhaseTimer("graph.build");

    try {
      // Geometry extraction is the expensive part, so it is cached against the
      // loaded model: changing a slider re-samples but does not re-extract.
      if (!geometryRef.current) {
        geometryRef.current = await timer.run("extract", () =>
          viewer.extractEgressGeometry(),
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
        upAxis: settings.upAxis,
        floorSpacing: settings.floorSpacing,
        maxPoints: settings.maxPoints,
        columnGap: settings.columnGap,
      };
      const transfers = [];
      for (const key of ["floors", "stairs", "doors", "walls"]) {
        for (const mesh of payload[key]) {
          transfers.push(mesh.positions.buffer, mesh.indices.buffer);
        }
      }

      const sampled = await timer.run("sample", () => sample(payload, transfers));
      setLastSample(sampled.stats);

      studio.setLoadState({ busy: true, stage: "building", detail: null });

      const toTriples = (flat) => {
        const out = [];
        for (let i = 0; i < flat.length; i += 3) {
          out.push([flat[i], flat[i + 1], flat[i + 2]]);
        }
        return out;
      };

      const response = await timer.run("upload", () =>
        api.buildGraph({
          floor_points: toTriples(sampled.floorPoints),
          stair_points: toTriples(sampled.stairPoints),
          door_points: sampled.doorPoints,
          walls: sampled.walls,
          options: {
            up_axis: settings.upAxis,
            agent_height: settings.agentHeight,
            max_edge_floor: settings.maxEdgeFloor,
            max_edge_stair: settings.maxEdgeStair,
            max_edge_rise: settings.maxEdgeRise,
            column_gap: settings.columnGap,
            max_degree: settings.maxDegree,
            use_walls: settings.useWalls,
            rectilinear: settings.rectilinear,
            grid_snap: settings.gridSnap,
            grid_cell_size: settings.gridSnap ? settings.gridCellSize : null,
            max_points: settings.maxPoints,
          },
        }),
      );

      studio.setGraph(response);
      studio.setPath(null);
      studio.setDynamicPath(null);

      viewer.setGraph(response.nodes, response.edges, response.kinds);
      viewer.setGraphVisible(settings.showGraph);

      // A freshly built graph is completely hidden behind solid walls and a
      // roof, so the build reads as "nothing happened". Ghost the shell once,
      // the first time, and leave the choice with the user afterwards.
      if (settings.modelAppearance === "solid") {
        setSettings({ modelAppearance: "ghost" });
      }

      timer.finish({ nodes: response.stats.nodes, edges: response.stats.edges });

      const reachable = response.stats.nodes
        ? Math.round((response.stats.largest_component / response.stats.nodes) * 100)
        : 0;
      if (response.stats.components > 1 && reachable < 98) {
        toast(
          `${reachable}% of the walkable area is one connected network. ` +
            `The rest is isolated and has no route out.`,
          { title: "Some floor area is unreachable", variant: "warning", ttl: 9000 },
        );
      } else {
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
  }, [reportError, sample, setSettings, settings, studio, toast]);

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
      if (mode !== "fire") {
        studio.setPath(null);
        viewerRef.current?.setPath(null);
      }
    },
    [snapToGraph, studio],
  );

  const requestPick = useCallback(
    (mode) => {
      studio.setPickMode((current) => (current === mode ? null : mode));
    },
    [studio],
  );

  const clearPoints = useCallback(() => {
    studio.setPoints({ start: null, exit: null, fire: null });
    ["start", "exit", "fire"].forEach((name) => viewerRef.current?.setMarker(name, null));
    studio.setPath(null);
    studio.setDynamicPath(null);
    viewerRef.current?.setPath(null);
    viewerRef.current?.setPath(null, { dynamic: true });
  }, [studio]);

  // ------------------------------------------------------------- find path

  const findPath = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    studio.setPathBusy(true);
    studio.setComparison(null);
    try {
      const result = await api.findPath({
        graph_id: graph.graphId,
        start_point: points.start,
        end_point: points.exit,
        engine: studio.effectiveEngine,
        use_walls: settings.useWalls,
      });
      studio.setPath(result);
      viewerRef.current?.setPath(result.points);
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
  }, [graph, points, reportError, settings.useWalls, studio, toast]);

  const comparePath = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    studio.setPathBusy(true);
    try {
      const result = await api.comparePath({
        graph_id: graph.graphId,
        start_point: points.start,
        end_point: points.exit,
        use_walls: settings.useWalls,
      });
      studio.setComparison(result);
      studio.setPath(result.fast);
      viewerRef.current?.setPath(result.fast.points);
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
  }, [graph, points, reportError, settings.useWalls, studio, toast]);

  // ------------------------------------------------------------------- fire

  const startFire = useCallback(() => {
    if (!graph || !points.fire) return;
    studio.stopFire();
    studio.setDynamicPath(null);
    viewerRef.current?.setPath(null, { dynamic: true });

    burningRef.current = new Set();
    temperatureRef.current = new Map();
    studio.setFire({ running: true, step: 0, model: settings.fireModel });

    const wantsReroute =
      settings.reroute &&
      settings.fireModel === "temperature" &&
      Boolean(points.start && points.exit);

    const handle = openFireStream(
      {
        graph_id: graph.graphId,
        model: settings.fireModel,
        max_steps: settings.fireMaxSteps,
        delay_ms: settings.fireDelayMs,
        use_walls: settings.useWalls,
        start_x: points.fire[0],
        start_y: points.fire[1],
        start_z: points.fire[2],
        ...(points.exit
          ? { end_x: points.exit[0], end_y: points.exit[1], end_z: points.exit[2] }
          : {}),
        ...(wantsReroute
          ? {
              stream_path: true,
              path_start_x: points.start[0],
              path_start_y: points.start[1],
              path_start_z: points.start[2],
              path_recompute_interval: settings.rerouteInterval,
              path_alpha: settings.hazardAlpha,
              path_engine: studio.effectiveEngine,
              ...(settings.lethalityThreshold
                ? { path_lethality_threshold: settings.lethalityThreshold }
                : {}),
            }
          : {}),
      },
      {
        onMessage: (message) => {
          const viewer = viewerRef.current;
          if (message.type === "step") {
            const burning = burningRef.current;
            (message.nodes || []).forEach((n) => burning.add(n));
            studio.setFire((current) => ({ ...current, step: message.step }));
            viewer?.setBurningNodes(burning);
          } else if (message.type === "temperature_step") {
            const temperatures = temperatureRef.current;
            temperatures.clear();
            for (const [key, value] of Object.entries(message.temperatures || {})) {
              temperatures.set(Number(key), value);
            }
            studio.setFire((current) => ({ ...current, step: message.step }));
            viewer?.setTemperatures(temperatures);
          } else if (message.type === "path_update") {
            studio.setDynamicPath(message);
            viewer?.setPath(message.path, { color: 0xc026d3, dynamic: true });
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
  }, [graph, points, reportError, settings, studio, toast]);

  const stopFire = useCallback(() => {
    studio.stopFire();
    viewerRef.current?.setTemperatures(null);
    viewerRef.current?.setBurningNodes(null);
  }, [studio]);

  // --------------------------------------------------------------------- RL

  const trainRl = useCallback(async () => {
    if (!graph || !points.start || !points.exit) return;
    studio.setRl({ busy: true, path: null, reachedExit: false });
    try {
      const result = await api.trainRl({
        graph_id: graph.graphId,
        start_point: points.start,
        exit_point: points.exit,
        episodes: settings.rlEpisodes,
        max_steps: settings.rlMaxSteps,
        use_fire: settings.rlUseFire,
      });
      studio.setRl({ busy: false, path: result.path, reachedExit: result.reached_exit });
      viewerRef.current?.setPath(result.points, { color: 0x14b8a6, dynamic: true });
      if (!result.reached_exit) {
        toast("The policy did not reach the exit. Try more episodes.", {
          title: "RL training",
          variant: "warning",
        });
      }
    } catch (error) {
      studio.setRl({ busy: false, path: null, reachedExit: false });
      reportError(error, "RL training failed");
    }
  }, [graph, points, reportError, settings, studio, toast]);

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

  // ----------------------------------------------------------------- legend

  const legend = useMemo(() => {
    const items = [];
    if (graph) {
      items.push({ label: "Walkable graph", color: "var(--viz-graph)" });
      if (graph.stats.stair_nodes) {
        items.push({ label: "Stairs", color: "var(--viz-stair)" });
      }
      if (graph.stats.door_nodes) {
        items.push({ label: "Doors", color: "var(--viz-door)" });
      }
    }
    if (points.start) items.push({ label: "Start", color: "var(--viz-start)", dot: true });
    if (points.exit) items.push({ label: "Exit", color: "var(--viz-exit)", dot: true });
    if (points.fire) items.push({ label: "Fire origin (spike)", color: "var(--viz-fire)", dot: true });
    if (path?.found) items.push({ label: "Egress route", color: "var(--viz-path)" });
    if (dynamicPath) {
      items.push({ label: "Hazard-aware route", color: "var(--viz-path-dynamic)" });
    }
    return items;
  }, [graph, points, path, dynamicPath]);

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

        <Segmented
          value={mode}
          onChange={setMode}
          options={MODES}
          label="Viewer mode"
        />

        <div className="header__divider" />

        <div className="header__group">
          {mode === "ifc" ? (
            <>
              <FileButton accept=".ifc" onFile={handleFile} variant="primary">
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
          {studio.serverStatus !== "online" && (
            <Badge variant="danger">Backend offline</Badge>
          )}
          {modelInfo?.cached && <Badge variant="success">cached</Badge>}
          {fire.running && (
            <Badge variant="danger" pulse>
              fire step {fire.step}
            </Badge>
          )}
        </div>

        <div className="header__group">
          <Button
            icon
            variant="ghost"
            onClick={toggleTheme}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            aria-label="Toggle colour theme"
          >
            {theme === "dark" ? "☀" : "☾"}
          </Button>
          <Button
            icon
            variant="ghost"
            onClick={() => setPanelOpen((v) => !v)}
            title={panelOpen ? "Hide inspector" : "Show inspector"}
            aria-label="Toggle inspector panel"
          >
            {panelOpen ? "▸" : "◂"}
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
            legend={legend}
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
          <aside className="panel">
            <div className="panel__tabs" role="tablist">
              {tabs.map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  role="tab"
                  className="panel__tab"
                  aria-selected={activeTab === entry.id}
                  onClick={() => setTab(entry.id)}
                >
                  {entry.label}
                </button>
              ))}
            </div>

            <div className="panel__scroll">
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
                  onBuildGraph={buildGraph}
                  onClearCache={studio.purgeCache}
                />
              )}
              {activeTab === "route" && (
                <RoutePanel
                  studio={studio}
                  onPick={requestPick}
                  onClearPoints={clearPoints}
                  onFindPath={findPath}
                  onCompare={comparePath}
                />
              )}
              {activeTab === "simulate" && (
                <SimulatePanel
                  studio={studio}
                  onStartFire={startFire}
                  onStopFire={stopFire}
                  onTrainRl={trainRl}
                />
              )}
              {activeTab === "about" && <AboutPanel studio={studio} />}
            </div>

            <div className="statusbar">
              <span className="statusbar__item">
                <span
                  className="badge__dot"
                  style={{
                    background:
                      studio.serverStatus === "online"
                        ? "var(--success)"
                        : "var(--danger)",
                  }}
                />
                {studio.serverStatus === "online" ? "backend up" : "backend down"}
              </span>
              {lastSample && (
                <>
                  <span className="statusbar__sep" />
                  <span className="statusbar__item">
                    sampled {(lastSample.floorPoints + lastSample.stairPoints).toLocaleString()} pts
                    in {Math.round(lastSample.ms)} ms
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
    </div>
  );
}
