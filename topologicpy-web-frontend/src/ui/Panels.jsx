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
import { routeColor } from "../lib/routeColors.js";
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

export function ModelPanel({ studio, onBuildGraph, onClearCache, onNavigationCheck }) {
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
          label="Grid spacing"
          value={settings.floorSpacing}
          min={0.2}
          max={2}
          step={0.05}
          onChange={(floorSpacing) => setSettings({ floorSpacing })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="One grid for the whole building, turned to its walls. Smaller follows narrow passages better and makes a bigger graph."
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
          label="No diagonal links"
          checked={settings.rectilinear}
          onChange={(rectilinear) => setSettings({ rectilinear })}
          hint="Links only along the building's grid. Routes then follow corridors exactly, but a route across an open room can be up to 41% longer than walking straight."
        />

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

      <NavigationCheckSection studio={studio} onRun={onNavigationCheck} />

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

/**
 * TopologicPy builds its own walkable graph from the same IFC; this compares
 * the two. Lives under Model because it checks the graph, not a route.
 */
function NavigationCheckSection({ studio, onRun }) {
  const { graph, file, navCheck, navBusy, settings, setSettings, capabilities, points } = studio;
  const available = Boolean(capabilities?.navigation_check);
  const pct = (v) => (v === null || v === undefined ? "-" : `${Math.round(v * 100)}%`);
  const route = navCheck?.route;

  return (
    <Section title="Cross-check with TopologicPy" defaultOpen={false} badge={navCheck ? pct(navCheck.studio_covered) : null}>
      <p className="field__hint">
        TopologicPy&apos;s <code>Navigation</code> class builds its own walkable graph from the IFC:
        walkable faces, obstacles cut out at walking height, stairs as connectors. It is an
        independent estimate of the same space, so where the two disagree, one of them is wrong
        about the building. The IFC is uploaded and parsed on the server: seconds here, up to a
        minute on a small host.
      </p>
      <NumberField
        label="Grid spacing (m)"
        value={settings.navSpacing}
        min={0.25}
        max={5}
        step={0.25}
        onChange={(navSpacing) => setSettings({ navSpacing: navSpacing ?? 0.5 })}
        hint="Match the graph's sampling spacing for a fair comparison. Also the coverage radius."
      />
      <Button
        block
        busy={navBusy}
        disabled={!available || !graph || !file}
        onClick={onRun}
        title={available ? undefined : "The server's topologicpy has no Navigation class"}
      >
        {navBusy ? "Building TopologicPy's graph" : "Run Navigation cross-check"}
      </Button>
      {!available && (
        <p className="field__hint">The server cannot run this: it needs topologicpy 0.9.86+ and IfcOpenShell.</p>
      )}
      {available && graph && !(points.start && points.exit) && !navCheck && (
        <p className="field__hint">Set a start and exit first to compare a route on each graph as well.</p>
      )}

      {navCheck && (
        <>
          <div className="stats">
            <Stat label="Its nodes we cover" value={pct(navCheck.navigation_covered)} warn={navCheck.navigation_covered < 0.9} />
            <Stat label="Our nodes it covers" value={pct(navCheck.studio_covered)} warn={navCheck.studio_covered < 0.9} />
            <Stat label="Its nodes" value={fmt(navCheck.navigation_nodes)} />
            <Stat label="Its parts" value={fmt(navCheck.navigation_components)} warn={navCheck.navigation_components > 1} />
          </div>
          <KeyValue
            entries={[
              ...navCheck.storeys.map((s) => [
                s.name,
                `ours ${fmt(s.studio_nodes)} · its ${fmt(s.navigation_nodes)} · ${pct(s.navigation_covered)} / ${pct(s.studio_covered)}`,
              ]),
              ...Object.entries(navCheck.studio_covered_by_kind).map(([kind, v]) => [
                `Our ${kind} nodes covered`,
                pct(v),
              ]),
              ...(route
                ? [
                    [
                      "Route length",
                      route.found
                        ? `ours ${route.studio_length?.toFixed(1)} m · its ${route.navigation_length.toFixed(1)} m (${route.ratio?.toFixed(2)}x)`
                        : "no route on its graph",
                    ],
                  ]
                : []),
              ["Alignment", `${navCheck.alignment}, median gap ${navCheck.residual_median.toFixed(2)} m`],
              [
                "Time",
                navCheck.cached
                  ? `${formatMs(navCheck.timings.align_ms)} (graph cached)`
                  : `${formatMs(navCheck.timings.parse_ms + navCheck.timings.graph_ms)} to build`,
              ],
            ]}
          />
          {route?.note && <p className="field__hint">{route.note}</p>}
          {navCheck.notes.map((note) => (
            <p key={note} className="field__hint" style={{ color: "var(--warning)" }}>
              {note}
            </p>
          ))}
          <p className="field__hint">
            Per storey: ours / its node counts, then the share of its nodes we cover and of ours it
            covers, within {navCheck.radius} m.
          </p>
          <Toggle
            label="Show TopologicPy's graph"
            checked={settings.showNavGraph}
            onChange={(showNavGraph) => setSettings({ showNavGraph })}
          />
        </>
      )}
    </Section>
  );
}

// --------------------------------------------------------------- pathfinding

export function RoutePanel({ studio, onPick, onClearPoints, onFindPath, onCompare, onAlternatives }) {
  const {
    graph,
    points,
    pickMode,
    path,
    pathBusy,
    comparison,
    alternatives,
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
                ? "TopologicPy TGraph.ShortestPath"
                : "TopologicPy (unavailable)",
              disabled: !engines.includes("topologicpy"),
            },
          ]}
          hint={
            engines.includes("topologicpy")
              ? "Both return the same optimum route. Either is fine interactively."
              : "The server could not load topologicpy, so only the built-in engine is offered."
          }
        />

        {settings.engine === "topologicpy" && (
          <p className="field__hint">
            Uses TopologicPy&apos;s <code>TGraph.ShortestPath</code>. About 30 ms on a
            3,000-node floor against 3 ms for the built-in engine, so live rerouting during a
            fire still keeps up.
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

      <Section
        title="Alternative routes"
        badge={alternatives ? `${alternatives.routes.length} found` : null}
      >
        <p className="field__hint">
          Is there a second way out? <strong>Different space</strong> keeps a route only if at most
          half of it runs within 1.5 m of a better one, up to 1.5x the shortest length; fewer
          routes than asked means the building offers no more. <strong>Yen</strong> is
          TopologicPy&apos;s <code>KShortestPaths</code>: the next cheapest paths, which on a dense
          graph are usually the best route with a step moved.
        </p>
        <div className="grid-2">
          <NumberField
            label="Routes"
            value={settings.altCount}
            min={2}
            max={8}
            step={1}
            onChange={(altCount) => setSettings({ altCount: altCount ?? 3 })}
          />
          <SelectField
            label="Method"
            value={settings.altMethod}
            onChange={(altMethod) => setSettings({ altMethod })}
            options={[
              { value: "distinct", label: "Different space" },
              { value: "yen", label: "Yen k-shortest" },
            ]}
          />
        </div>
        <Button
          block
          busy={pathBusy}
          disabled={!ready || !points.start || !points.exit}
          onClick={onAlternatives}
        >
          Find alternative routes
        </Button>
        {alternatives?.routes.length > 0 && (
          <div className="kv">
            {alternatives.routes.map((route, i) => (
              <React.Fragment key={route.rank}>
                <div className="kv__k" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span className="picker__dot" style={{ background: routeColor(i) }} />
                  Route {route.rank}
                </div>
                <div className="kv__v">
                  {route.length.toFixed(1)} m
                  {route.rank === 1
                    ? " · shortest"
                    : ` · +${Math.round((route.stretch - 1) * 100)}% · ${Math.round(route.overlap * 100)}% shared`}
                </div>
              </React.Fragment>
            ))}
          </div>
        )}
        {alternatives && (
          <p className="field__hint">
            {alternatives.note ? `${alternatives.note} ` : ""}
            {alternatives.method === "yen" ? "Yen" : "Different space"}, {alternatives.engine} engine,{" "}
            {alternatives.ms.toFixed(0)} ms.
          </p>
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
          hint={
            settings.fireModel === "temperature"
              ? "The route is re-solved against current temperatures while the fire spreads."
              : "The route is re-solved with burning nodes treated as impassable."
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
          disabled={!settings.reroute || settings.fireModel !== "temperature"}
          hint="Temperature model only. Edge cost is length x (1 + weight x hazard); zero ignores the fire."
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
          Tabular Q-learning over the same graph, with the flood fire's ignition times as a
          penalty. Exploratory rather than a production route solver.
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
          hint={
            settings.rlUseFire && !points.fire
              ? "Set a fire origin (F); without one the policy trains without fire."
              : undefined
          }
        />
        <NumberField
          label="Fire speed (hops per move)"
          value={settings.rlFireSpeed}
          min={0.05}
          max={10}
          step={0.05}
          disabled={!settings.rlUseFire}
          onChange={(rlFireSpeed) => setSettings({ rlFireSpeed: rlFireSpeed ?? 1 })}
          hint="1 = the front moves exactly as fast as the evacuee (Classic). On a real plan that often leaves no escape; the result says when."
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
            <Stat label="Steps" value={fmt(rl.path.length - 1)} />
            <Stat
              label="Reached exit"
              value={rl.reachedExit ? "yes" : rl.result?.burned ? "burned" : "no"}
              warn={!rl.reachedExit}
            />
            {rl.result?.optimal_length != null && (
              <Stat
                label="Length vs shortest"
                value={`${rl.result.length.toFixed(1)} / ${rl.result.optimal_length.toFixed(1)} m`}
              />
            )}
            <Stat
              label="Late-training success"
              value={`${Math.round((rl.result?.success_rate ?? 0) * 100)}%`}
            />
            {settings.rlUseFire && rl.result?.fire_start_id != null && (
              <Stat
                label="Escape possible"
                value={
                  rl.result.escape_steps == null ? "no" : `yes, ${rl.result.escape_steps} moves`
                }
                warn={rl.result.escape_steps == null}
              />
            )}
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
          <Badge
            variant={
              serverStatus === "online"
                ? "success"
                : serverStatus === "waking" || serverStatus === "checking"
                  ? "warning"
                  : "danger"
            }
          >
            {serverStatus === "online"
              ? "Connected"
              : serverStatus === "waking"
                ? "Waking up"
                : serverStatus === "checking"
                  ? "Checking"
                  : "Unreachable"}
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
        {serverStatus === "waking" && (
          <p className="field__hint">
            A hosted backend on a free plan sleeps when idle and takes up to a minute to wake.
            Requests made now will wait for it.
          </p>
        )}
        {serverStatus === "offline" && (
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
