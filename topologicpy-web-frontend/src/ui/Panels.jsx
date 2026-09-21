/**
 * Inspector panels.
 *
 * Grouped by the actual workflow - Model, Graph, Simulate, About - instead of
 * the single scrolling wall of controls the Classic sidebar had. Controls that
 * cannot do anything yet are disabled with a reason rather than hidden, so the
 * sequence of steps stays visible.
 */
import React from "react";

import { formatBytes, formatMs } from "../lib/perf.js";
import {
  Badge,
  Button,
  Empty,
  KeyValue,
  NumberField,
  Section,
  Segmented,
  SelectField,
  Slider,
  Stat,
  Toggle,
} from "./primitives.jsx";

const fmt = (n) => (n === null || n === undefined ? "-" : n.toLocaleString());
const coord = (p) => (p ? p.map((v) => v.toFixed(2)).join(", ") : "not set");
const reachablePct = (stats) =>
  stats?.nodes ? Math.round((stats.largest_component / stats.nodes) * 100) : 0;

// -------------------------------------------------------------------- model

export function ModelPanel({ studio, onBuildGraph, onClearCache }) {
  const { modelInfo, file, graph, graphBusy, settings, setSettings, cacheInfo } = studio;

  return (
    <>
      <Section title="Model" badge={file ? file.name : undefined}>
        {!modelInfo ? (
          <Empty>No model loaded. Open an IFC file to begin.</Empty>
        ) : (
          <>
            <div className="stats">
              <Stat label="Floors" value={fmt(modelInfo.stats.floors)} />
              <Stat label="Stairs" value={fmt(modelInfo.stats.stairs)} />
              <Stat label="Doors" value={fmt(modelInfo.stats.doors)} />
              <Stat label="Walls" value={fmt(modelInfo.stats.walls)} />
            </div>
            <KeyValue
              entries={[
                ["Storeys", fmt(modelInfo.stats.storeys)],
                ["Spaces", fmt(modelInfo.stats.spaces)],
                ["File size", formatBytes(file?.size || 0)],
                ["Load time", formatMs(modelInfo.totalMs)],
                ["Source", modelInfo.cached ? "local cache" : "converted now"],
              ]}
            />
            {modelInfo.cached ? (
              <p className="field__hint">
                Loaded from the local fragment cache, skipping IFC parsing entirely.
              </p>
            ) : (
              <p className="field__hint">
                Converted and cached. Re-opening this file will skip the parse.
              </p>
            )}
          </>
        )}

        <div className="field">
          <div className="field__label">
            <span>IFC geometry</span>
          </div>
          <Segmented
            value={settings.modelAppearance}
            onChange={(modelAppearance) => setSettings({ modelAppearance })}
            label="IFC geometry display"
            options={[
              { value: "solid", label: "Solid" },
              { value: "ghost", label: "Ghost" },
              { value: "hidden", label: "Hidden" },
            ]}
          />
          <p className="field__hint">
            The navigation graph sits inside the building, so solid geometry hides it.
            Ghost keeps the building as context while letting the graph show through.
          </p>
        </div>
      </Section>

      <Section title="Navigation graph" badge={graph ? `${fmt(graph.stats.nodes)} nodes` : null}>
        <SelectField
          label="Up axis"
          value={settings.upAxis}
          onChange={(upAxis) => setSettings({ upAxis })}
          options={[
            { value: "y", label: "Y up (typical for viewers)" },
            { value: "z", label: "Z up (typical for IFC)" },
            { value: "x", label: "X up (unusual)" },
          ]}
          hint="Which axis points up in the loaded geometry. Get this wrong and floors read as walls."
        />

        <Slider
          label="Sampling spacing"
          value={settings.floorSpacing}
          min={0.2}
          max={2}
          step={0.05}
          onChange={(floorSpacing) => setSettings({ floorSpacing })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="Distance between walkable sample points. Smaller is more faithful and slower."
        />

        <Slider
          label="Floor connectivity"
          value={settings.maxEdgeFloor}
          min={0.5}
          max={8}
          step={0.25}
          onChange={(maxEdgeFloor) => setSettings({ maxEdgeFloor })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="Longest edge allowed between two floor points."
        />

        <Slider
          label="Stair connectivity"
          value={settings.maxEdgeStair}
          min={0.15}
          max={2}
          step={0.05}
          onChange={(maxEdgeStair) => setSettings({ maxEdgeStair })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="Longest edge between stair treads. Around twice the tread rise works well."
        />

        <Slider
          label="Max vertical step"
          value={settings.maxEdgeRise}
          min={0.05}
          max={1.5}
          step={0.05}
          onChange={(maxEdgeRise) => setSettings({ maxEdgeRise })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="How much height a floor-to-floor link may span. Low keeps each storey a flat mesh; raise it only for ramps or split levels."
        />

        <Slider
          label="Neighbours per node"
          value={settings.maxDegree}
          min={4}
          max={32}
          step={1}
          onChange={(maxDegree) => setSettings({ maxDegree })}
          format={(v) => `${v}`}
          hint="Caps graph density. Dense sampling with a high cap makes very large graphs for no routing benefit."
        />

        <Slider
          label="Agent height"
          value={settings.agentHeight}
          min={0}
          max={2}
          step={0.05}
          onChange={(agentHeight) => setSettings({ agentHeight })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="Lifts the graph off the slab so it renders above the floor."
        />

        <Toggle
          label="Walls block routes"
          checked={settings.useWalls}
          onChange={(useWalls) => setSettings({ useWalls })}
          hint="Edges crossing a wall are excluded from pathfinding. Doors stay open."
        />

        <Toggle
          label="Rectilinear edges only"
          checked={settings.rectilinear}
          onChange={(rectilinear) => setSettings({ rectilinear })}
          hint="Drops diagonal links so routes follow corridors."
        />

        <Toggle
          label="Snap to grid"
          checked={settings.gridSnap}
          onChange={(gridSnap) => setSettings({ gridSnap })}
          hint="Rebuilds the graph on a regular lattice. Cleaner, but coarser."
        />

        {settings.gridSnap && (
          <Slider
            label="Grid cell size"
            value={settings.gridCellSize}
            min={0.3}
            max={3}
            step={0.1}
            onChange={(gridCellSize) => setSettings({ gridCellSize })}
            format={(v) => `${v.toFixed(1)} m`}
          />
        )}

        <Button
          variant="primary"
          block
          busy={graphBusy}
          disabled={!modelInfo}
          onClick={onBuildGraph}
        >
          {graphBusy ? "Building graph" : graph ? "Rebuild graph" : "Build egress graph"}
        </Button>

        {graph && (
          <>
            <div className="stats">
              <Stat label="Nodes" value={fmt(graph.stats.nodes)} />
              <Stat label="Edges" value={fmt(graph.stats.edges)} />
              <Stat label="Doors" value={fmt(graph.stats.door_nodes)} />
              <Stat
                label="Reachable"
                value={`${reachablePct(graph.stats)}%`}
                warn={reachablePct(graph.stats) < 90}
              />
            </div>
            {graph.stats.components > 1 && (
              <p className="field__hint">
                {fmt(graph.stats.largest_component)} of {fmt(graph.stats.nodes)} nodes form one
                connected network; the rest sit in {graph.stats.components - 1} isolated
                {graph.stats.components === 2 ? " pocket" : " pockets"}. Isolated floor area has
                no route out, which is worth knowing — but it can also mean a level change the
                model does not bridge with stairs or a ramp. Raise{" "}
                <strong>Max vertical step</strong> to link surfaces at slightly different
                heights.
              </p>
            )}
            <Toggle
              label="Show graph overlay"
              checked={settings.showGraph}
              onChange={(showGraph) => setSettings({ showGraph })}
            />
          </>
        )}
      </Section>

      <Section title="Local cache" defaultOpen={false} badge={formatBytes(cacheInfo.bytes)}>
        <p className="field__hint">
          Converted models are stored in this browser so repeat loads skip IFC parsing.
          {cacheInfo.count > 0 && ` Currently holding ${cacheInfo.count} model(s).`}
        </p>
        <Button block onClick={onClearCache} disabled={!cacheInfo.count}>
          Clear cached models
        </Button>
      </Section>
    </>
  );
}

// --------------------------------------------------------------- pathfinding

export function RoutePanel({ studio, onPick, onClearPoints, onFindPath, onCompare }) {
  const {
    graph,
    points,
    pickMode,
    path,
    pathBusy,
    comparison,
    settings,
    setSettings,
    capabilities,
  } = studio;

  const engines = capabilities?.engines || ["fast"];
  const ready = Boolean(graph);

  return (
    <>
      <Section title="Start and exit">
        {!ready && <Empty>Build a navigation graph first.</Empty>}

        <PickRow
          name="Start"
          color="var(--viz-start)"
          point={points.start}
          active={pickMode === "start"}
          disabled={!ready}
          onPick={() => onPick("start")}
          shortcut="S"
        />
        <PickRow
          name="Exit"
          color="var(--viz-exit)"
          point={points.exit}
          active={pickMode === "exit"}
          disabled={!ready}
          onPick={() => onPick("exit")}
          shortcut="E"
        />
        <PickRow
          name="Fire origin"
          color="var(--viz-fire)"
          point={points.fire}
          active={pickMode === "fire"}
          disabled={!ready}
          onPick={() => onPick("fire")}
          shortcut="F"
        />

        <Button block size="sm" onClick={onClearPoints} disabled={!points.start && !points.exit && !points.fire}>
          Clear all points
        </Button>
      </Section>

      <Section title="Route">
        <SelectField
          label="Path engine"
          value={settings.engine}
          onChange={(engine) => setSettings({ engine })}
          options={[
            { value: "fast", label: "Fast (built-in A*)" },
            {
              value: "topologicpy",
              label: engines.includes("topologicpy")
                ? "TopologicPy Graph.ShortestPath"
                : "TopologicPy (unavailable)",
              disabled: !engines.includes("topologicpy"),
            },
          ]}
          hint={
            engines.includes("topologicpy")
              ? "Both return the same optimum route. Measured on a 2,600-node model: built-in 5.8 ms, TopologicPy 9.3 ms per query plus a one-off graph build. Either is fine interactively."
              : "The server could not load topologicpy, so only the built-in engine is offered."
          }
        />

        {settings.engine === "topologicpy" && (
          <p className="field__hint">
            Uses TopologicPy&apos;s <code>TGraph.ShortestPath</code>. Around 1.6x the
            cost of the built-in engine, so live rerouting during a fire keeps up.
          </p>
        )}

        <Button
          variant="primary"
          block
          busy={pathBusy}
          disabled={!ready || !points.start || !points.exit}
          onClick={onFindPath}
        >
          Find egress route
        </Button>

        <Button
          block
          size="sm"
          disabled={!ready || !points.start || !points.exit || engines.length < 2}
          onClick={onCompare}
        >
          Compare both engines
        </Button>

        {path && (
          <>
            <div className="stats">
              <Stat label="Found" value={path.found ? "yes" : "no"} warn={!path.found} />
              <Stat label="Length" value={`${path.length.toFixed(1)} m`} />
              <Stat label="Waypoints" value={fmt(path.points.length)} />
              <Stat label="Engine" value={path.engine} />
            </div>
            {path.fallback_from && (
              <p className="field__hint" style={{ color: "var(--warning)" }}>
                Fell back from {path.fallback_from} to {path.engine}.
              </p>
            )}
            {path.note && <p className="field__hint">{path.note}</p>}
          </>
        )}

        {comparison && (
          <KeyValue
            entries={[
              ["Same route", comparison.same_route ? "yes" : "no"],
              ["Cost delta", comparison.cost_delta.toFixed(4)],
              ["Fast", `${comparison.fast_ms.toFixed(1)} ms`],
              ["TopologicPy", `${comparison.topologicpy_ms.toFixed(1)} ms`],
            ]}
          />
        )}
      </Section>
    </>
  );
}

function PickRow({ name, color, point, active, disabled, onPick, shortcut }) {
  return (
    <div className="picker" data-active={active}>
      <span className="picker__dot" style={{ background: color }} />
      <div className="picker__body">
        <div className="picker__name">{name}</div>
        <div className="picker__coord">{active ? "click in the model..." : coord(point)}</div>
      </div>
      <Button size="sm" onClick={onPick} disabled={disabled} title={`Shortcut: ${shortcut}`}>
        {active ? "Cancel" : "Set"}
      </Button>
    </div>
  );
}

// ----------------------------------------------------------------- simulate

export function SimulatePanel({ studio, onStartFire, onStopFire, onTrainRl }) {
  const { graph, points, settings, setSettings, fire, dynamicPath, rl } = studio;
  const ready = Boolean(graph);

  return (
    <>
      <Section title="Fire spread">
        {!ready && <Empty>Build a navigation graph first.</Empty>}

        <SelectField
          label="Model"
          value={settings.fireModel}
          onChange={(fireModel) => setSettings({ fireModel })}
          options={[
            { value: "temperature", label: "Temperature diffusion" },
            { value: "flood", label: "Spread along edges" },
            { value: "radial", label: "Radial distance bands" },
          ]}
          hint={
            settings.fireModel === "temperature"
              ? "Each node relaxes toward its neighbours. Required for hazard-weighted rerouting."
              : settings.fireModel === "flood"
                ? "Breadth-first along graph edges, so walls and doors shape the front."
                : "Ignition banded by straight-line distance. Cheapest, ignores geometry."
          }
        />

        <div className="grid-2">
          <NumberField
            label="Steps"
            value={settings.fireMaxSteps}
            min={1}
            max={2000}
            onChange={(fireMaxSteps) => setSettings({ fireMaxSteps: fireMaxSteps ?? 60 })}
          />
          <NumberField
            label="Step delay (ms)"
            value={settings.fireDelayMs}
            min={0}
            max={2000}
            step={10}
            onChange={(fireDelayMs) => setSettings({ fireDelayMs: fireDelayMs ?? 0 })}
          />
        </div>

        <div className="row">
          <Button
            variant="primary"
            disabled={!ready || fire.running || !points.fire}
            onClick={onStartFire}
          >
            Start fire
          </Button>
          <Button variant="danger" disabled={!fire.running} onClick={onStopFire}>
            Stop
          </Button>
        </div>
        {ready && !points.fire && (
          <p className="field__hint">Set a fire origin to start the simulation.</p>
        )}
      </Section>

      <Section title="Live rerouting" defaultOpen={settings.fireModel === "temperature"}>
        <Toggle
          label="Recompute the route as the fire grows"
          checked={settings.reroute}
          onChange={(reroute) => setSettings({ reroute })}
          disabled={settings.fireModel !== "temperature"}
          hint={
            settings.fireModel !== "temperature"
              ? "Needs the temperature model, which is what supplies the hazard field."
              : "The route is re-solved against current temperatures while the fire spreads."
          }
        />

        <Slider
          label="Hazard weight"
          value={settings.hazardAlpha}
          min={0}
          max={5}
          step={0.1}
          onChange={(hazardAlpha) => setSettings({ hazardAlpha })}
          format={(v) => v.toFixed(1)}
          disabled={!settings.reroute}
          hint="Edge cost is length x (1 + weight x hazard). Zero ignores the fire entirely."
        />

        <Slider
          label="Recompute every"
          value={settings.rerouteInterval}
          min={1}
          max={30}
          step={1}
          onChange={(rerouteInterval) => setSettings({ rerouteInterval })}
          format={(v) => `${v} steps`}
          disabled={!settings.reroute}
        />

        <NumberField
          label="Lethality threshold (deg C)"
          value={settings.lethalityThreshold}
          min={30}
          max={200}
          step={5}
          placeholder="none"
          onChange={(lethalityThreshold) => setSettings({ lethalityThreshold })}
          hint="Edges hotter than this are impassable. Left blank, temperature only adds cost."
        />

        {dynamicPath && (
          <div className="stats">
            <Stat label="Cost" value={dynamicPath.cost?.toFixed(1) ?? "-"} />
            <Stat
              label="Status"
              value={dynamicPath.changed ? "rerouted" : "stable"}
              warn={dynamicPath.changed}
            />
          </div>
        )}
      </Section>

      <Section title="Reinforcement learning" defaultOpen={false}>
        <p className="field__hint">
          Tabular Q-learning over the same graph, with ignition times as a penalty. Exploratory
          rather than a production route solver.
        </p>
        <div className="grid-2">
          <NumberField
            label="Episodes"
            value={settings.rlEpisodes}
            min={10}
            step={50}
            onChange={(rlEpisodes) => setSettings({ rlEpisodes: rlEpisodes ?? 200 })}
          />
          <NumberField
            label="Max steps"
            value={settings.rlMaxSteps}
            min={10}
            step={25}
            onChange={(rlMaxSteps) => setSettings({ rlMaxSteps: rlMaxSteps ?? 200 })}
          />
        </div>
        <Toggle
          label="Penalise reaching burning nodes"
          checked={settings.rlUseFire}
          onChange={(rlUseFire) => setSettings({ rlUseFire })}
        />
        <Button
          block
          busy={rl.busy}
          disabled={!ready || !points.start || !points.exit}
          onClick={onTrainRl}
        >
          Train policy
        </Button>
        {rl.path && (
          <div className="stats">
            <Stat label="Steps" value={fmt(rl.path.length)} />
            <Stat
              label="Reached exit"
              value={rl.reachedExit ? "yes" : "no"}
              warn={!rl.reachedExit}
            />
          </div>
        )}
      </Section>
    </>
  );
}

// -------------------------------------------------------------------- about

export function AboutPanel({ studio }) {
  const { capabilities, serverStatus } = studio;
  const topo = capabilities?.topologicpy || {};

  return (
    <>
      <Section title="Backend">
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Badge variant={serverStatus === "online" ? "success" : "danger"}>
            {serverStatus === "online" ? "Connected" : "Unreachable"}
          </Badge>
          {capabilities?.version && <span className="mono">v{capabilities.version}</span>}
        </div>

        <KeyValue
          entries={[
            ["API", import.meta.env.VITE_API_BASE || "http://localhost:8000"],
            ["TopologicPy", topo.topologicpy || "not installed"],
            ["topologic-core", topo.topologic_core || "not installed"],
            ["Path engines", (capabilities?.engines || []).join(", ") || "-"],
            ["IfcOpenShell", capabilities?.server_side_ifc ? "available" : "not installed"],
          ]}
        />

        {topo.error && (
          <p className="field__hint" style={{ color: "var(--warning)" }}>
            {topo.error}
          </p>
        )}
        {serverStatus !== "online" && (
          <p className="field__hint">
            Start the API with <code>uvicorn app.main:app --port 8000</code> from the backend
            folder, then reload.
          </p>
        )}
      </Section>

      <Section title="Keyboard" defaultOpen={false}>
        <KeyValue
          entries={[
            ["S", "Set start point"],
            ["E", "Set exit point"],
            ["F", "Set fire origin"],
            ["G", "Toggle graph overlay"],
            ["M", "Toggle IFC geometry"],
            ["B", "Build / rebuild graph"],
            ["Enter", "Find route"],
            ["Esc", "Cancel picking"],
          ]}
        />
      </Section>
    </>
  );
}
