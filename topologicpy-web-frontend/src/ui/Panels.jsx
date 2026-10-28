/**
 * Inspector panels, one per workflow step: Model, Graph, Route, Simulate.
 *
 * Each step opens with its one primary action, puts the result right under
 * it, and folds the settings most people leave alone. Explanations of a
 * result (a graph in several parts, fewer alternatives than asked) sit beside
 * it as notices rather than in toasts. Controls that cannot do anything yet
 * are disabled with a reason, so the order of the steps stays visible.
 */
import React, { useId, useState } from "react";

import { graphIsSplit, reachablePct } from "../core/graph.js";
import { RL_MAX_STEPS } from "../core/rl.js";
import { formatBytes, formatMs } from "../lib/perf.js";
import { FORECAST_COLORS, routeColor } from "../lib/routeColors.js";
import { PLAN_SETTINGS } from "../state/useStudio.js";
import {
  Button,
  Choices,
  Disclosure,
  Empty,
  Icon,
  KeyValue,
  Notice,
  NumberField,
  Section,
  Segmented,
  SelectField,
  Slider,
  Stat,
  StepHeading,
  Toggle,
} from "./primitives.jsx";

const fmt = (n) => (n === null || n === undefined ? "-" : n.toLocaleString());
const coord = (p) => (p ? p.map((v) => v.toFixed(2)).join(", ") : "not set");

/**
 * Separate parts are not necessarily a fault: the Duplex is two dwellings,
 * each with its own front door, and splits 51/49.
 */
function PartsNotice({ stats }) {
  if (!stats || stats.components < 2) return null;
  if (!graphIsSplit(stats)) {
    return (
      <Notice>
        {fmt(stats.nodes - stats.largest_component)} nodes sit in{" "}
        {stats.components - 1 === 1 ? "a small pocket" : `${stats.components - 1} small pockets`} with no
        route out, typically floor poking past an outside wall.
      </Notice>
    );
  }
  return (
    <Notice tone="warning">
      The graph is in {stats.components} parts; the largest holds {reachablePct(stats)}%. A route can
      only join two points in the same part. Separate dwellings, or a room whose door is not in the
      model, show up this way; a level change with no stair in the model can be joined by raising{" "}
      <strong>Max vertical step</strong> in the graph settings.
    </Notice>
  );
}

// -------------------------------------------------------------------- model

/**
 * What the file itself says about egress (core/ifcSemantics.js): read beside
 * the geometry, so it appears a moment after the model. Missing exit flags
 * are the most common reason an egress result needs care, so it says so.
 */
function EgressCheck({ semantics }) {
  if (!semantics) return null;
  const doors = semantics.doors || [];
  const exits = doors.filter((d) => d.exit);
  const sources = [...new Set(exits.map((d) => d.exit))].map((s) =>
    s.startsWith("Pset_DoorCommon") ? s.split(".")[1] : "external space boundary",
  );
  return (
    <>
      <KeyValue
        entries={[
          ["Schema", semantics.schema || "-"],
          ["Exit doors", doors.length ? `${exits.length} of ${doors.length}${sources.length ? ` (${sources.join(", ")})` : ""}` : "no doors"],
          ["Roof slabs", semantics.roofSlabs?.length ? `${semantics.roofSlabs.length}, not walked on` : "none typed"],
        ]}
      />
      {doors.length > 0 && !exits.length && (
        <Notice tone="warning">
          <strong>No exit doors in the file.</strong> No door is flagged as an exit (Pset_DoorCommon FireExit or IsExternal) or has an external space boundary, so
          exits fall back to doors with floor on one side. Check them, or pick the exit yourself.
        </Notice>
      )}
      {!doors.length && (
        <Notice tone="warning">
          <strong>No doors in the file.</strong> Rooms connect only through gaps the walls leave; a route may not find a way out.
        </Notice>
      )}
    </>
  );
}

export function ModelPanel({ studio, examples, onExample, onClearCache }) {
  const { modelInfo, file, settings, setSettings, cacheInfo } = studio;

  return (
    <>
      <StepHeading
        step={1}
        title="Model"
        lead="An IFC file, opened in your browser. It is only sent to the server for the TopologicPy cross-check."
      />
      <div className="step-body">
        {!modelInfo ? (
          <Empty>No model open. Use Open IFC above, drop a file on the viewport, or pick an example below.</Empty>
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
                ["File", file?.name || "-"],
                ["Storeys", fmt(modelInfo.stats.storeys)],
                ["Spaces", fmt(modelInfo.stats.spaces)],
                ["File size", formatBytes(file?.size || 0)],
                ["Load time", formatMs(modelInfo.totalMs)],
                ["Source", modelInfo.cached ? "local cache" : "converted now"],
              ]}
            />
            <p className="field__hint">
              {modelInfo.cached
                ? "The geometry came from the local fragment cache, without converting the IFC again. The exit flags and storeys are still read from the file."
                : "Converted and cached: opening this file again skips the conversion. The exit flags and storeys are read from the file each time."}
            </p>
            <EgressCheck semantics={studio.semantics} />
          </>
        )}

        <div className="field">
          <div className="field__label">
            <span>Show the building as</span>
          </div>
          <Segmented
            value={settings.modelAppearance}
            onChange={(modelAppearance) => setSettings({ modelAppearance })}
            label="Show the building as"
            options={[
              { value: "solid", label: "Solid" },
              { value: "ghost", label: "Ghost" },
              { value: "hidden", label: "Hidden" },
            ]}
          />
          <p className="field__hint">Ghost keeps the building as context and lets the graph inside show through.</p>
        </div>
      </div>

      {examples?.length > 0 && (
        <Section title="Examples" defaultOpen={false}>
          <ul className="example-list">
            {examples.map((entry) => (
              <li key={entry.id}>
                <button
                  type="button"
                  className="example"
                  onClick={() => onExample(entry)}
                  aria-label={entry.bytes ? `${entry.name}, ${formatBytes(entry.bytes)}` : entry.name}
                  aria-describedby={`model-example-${entry.id}`}
                >
                  <span className="example__head">
                    <span className="example__name">{entry.name}</span>
                    {entry.bytes ? <span className="example__size">{formatBytes(entry.bytes)}</span> : null}
                  </span>
                  <span className="example__desc" id={`model-example-${entry.id}`}>
                    {entry.description}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </Section>
      )}

      <Section title="Local cache" defaultOpen={false} badge={formatBytes(cacheInfo.bytes)}>
        <p className="field__hint">
          Converted models are stored in this browser so repeat loads skip the IFC conversion.
          {cacheInfo.count > 0 && ` Currently holding ${cacheInfo.count} model(s).`}
        </p>
        <Button block onClick={onClearCache} disabled={!cacheInfo.count}>
          Clear cached models
        </Button>
      </Section>
    </>
  );
}

// -------------------------------------------------------------------- graph

export function GraphPanel({ studio, onBuildGraph, onNavigationCheck }) {
  const { modelInfo, graph, graphBusy, settings, setSettings } = studio;

  return (
    <>
      <StepHeading
        step={2}
        title="Walkable graph"
        lead="A grid turned to the building's walls, with stairs and doors. Start, exit and fire can only be picked on it."
      />
      <div className="step-body">
        {!modelInfo && <Empty>Open a model first (step 1).</Empty>}
        <Button variant="primary" size="lg" block busy={graphBusy} disabled={!modelInfo} onClick={onBuildGraph}>
          {graphBusy ? "Building graph" : graph ? "Rebuild graph" : "Build egress graph"}
        </Button>

        {graph && (
          <>
            <div className="stats">
              <Stat label="Nodes" value={fmt(graph.stats.nodes)} />
              <Stat label="Links" value={fmt(graph.stats.edges)} />
              <Stat label="Doors" value={fmt(graph.stats.door_nodes)} />
              <Stat label="Reachable" value={`${reachablePct(graph.stats)}%`} warn={reachablePct(graph.stats) < 90} />
            </div>
            <PartsNotice stats={graph.stats} />
            <Toggle
              label="Show the graph"
              checked={settings.showGraph}
              onChange={(showGraph) => setSettings({ showGraph })}
            />
          </>
        )}
      </div>

      <Section title="Graph settings" defaultOpen={false}>
        <p className="field__hint">Changes apply when the graph is rebuilt.</p>
        <Slider
          label="Grid spacing"
          value={settings.floorSpacing}
          min={0.2}
          max={2}
          step={0.05}
          onChange={(floorSpacing) => setSettings({ floorSpacing })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="Smaller follows narrow passages better and makes a bigger graph."
        />
        <Slider
          label="Max vertical step"
          value={settings.maxEdgeRise}
          min={0.05}
          max={1.5}
          step={0.05}
          onChange={(maxEdgeRise) => setSettings({ maxEdgeRise })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="Height one floor link may span. Raise it only for ramps or split levels."
        />
        <Slider
          label="Agent height"
          value={settings.agentHeight}
          min={0}
          max={2}
          step={0.05}
          onChange={(agentHeight) => setSettings({ agentHeight })}
          format={(v) => `${v.toFixed(2)} m`}
          hint="Lifts the graph off the slab so it draws above the floor."
        />
        <Toggle
          label="Walls block routes"
          checked={settings.useWalls}
          onChange={(useWalls) => setSettings({ useWalls })}
          hint="Links that cross a wall 0.75 m above the floor are left out of routing. Links that only touch a wall stay in. Door links always stay open."
        />
        <Toggle
          label="No diagonal links"
          checked={settings.rectilinear}
          onChange={(rectilinear) => setSettings({ rectilinear })}
          hint="Routes then use only grid-aligned links. On an open floor the grid route can be up to 41% (the square root of 2) longer than with diagonals; drawing routes straight recovers much of that."
        />
        <SelectField
          label="Up axis"
          value={settings.upAxis}
          onChange={(upAxis) => setSettings({ upAxis })}
          options={[
            { value: "y", label: "Y up (typical for viewers)" },
            { value: "z", label: "Z up (typical for IFC)" },
            { value: "x", label: "X up (unusual)" },
          ]}
          hint="Get this wrong and floors read as walls."
        />
      </Section>

      <NavigationCheckSection studio={studio} onRun={onNavigationCheck} />
    </>
  );
}

/**
 * TopologicPy builds its own walkable graph from the same IFC; this compares
 * the two. Lives under Graph because it checks the graph, not a route.
 */
function NavigationCheckSection({ studio, onRun }) {
  const { graph, file, navCheck, navBusy, settings, setSettings, capabilities, points } = studio;
  const available = Boolean(capabilities?.navigation_check);
  const pct = (v) => (v === null || v === undefined ? "-" : `${Math.round(v * 100)}%`);
  const route = navCheck?.route;

  return (
    <Section title="Cross-check with TopologicPy" defaultOpen={false} badge={navCheck ? pct(navCheck.studio_covered) : null}>
      <p className="field__hint">
        TopologicPy&apos;s <code>Navigation</code> class builds its own walkable graph from the IFC, an
        independent estimate of the same space: where the two disagree, one of them is wrong about the
        building. The IFC is uploaded and parsed on the server, which takes seconds here and up to a
        minute on a small host.
      </p>
      <NumberField
        label="Grid spacing (m)"
        value={settings.navSpacing}
        min={0.25}
        max={5}
        step={0.25}
        onChange={(navSpacing) => setSettings({ navSpacing: navSpacing ?? 0.5 })}
        hint="Match the graph's spacing for a fair comparison. Also the coverage radius."
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
              ...Object.entries(navCheck.studio_covered_by_kind).map(([kind, v]) => [`Our ${kind} nodes covered`, pct(v)]),
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
            <Notice key={note} tone="warning">
              {note}
            </Notice>
          ))}
          <p className="field__hint">
            Per storey: ours / its node counts, then the share of its nodes we cover and of ours it covers,
            within {navCheck.radius} m.
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

/** The graph in one line, at the top of the steps that use it. */
function GraphSummary({ graph, onGoTo }) {
  if (!graph) return null;
  return (
    <div className="summary">
      <div className="summary__row">
        <span>
          <strong>Graph</strong> {fmt(graph.stats.nodes)} nodes · {fmt(graph.stats.edges)} links
        </span>
        <button type="button" className="linklike" onClick={() => onGoTo("graph")}>
          Graph step
        </button>
      </div>
      {graphIsSplit(graph.stats) && <PartsNotice stats={graph.stats} />}
    </div>
  );
}

// --------------------------------------------------------------- pathfinding

/**
 * Changes of direction of more than 25 degrees between runs of at least
 * 0.75 m. Counting vertices instead reported every stair node, which the
 * straightened route keeps, as a turn; so did the half-step zigzag of a
 * flight, which merging short runs removes.
 */
function countTurns(raw) {
  const points = raw.length ? [raw[0]] : [];
  for (const p of raw.slice(1, -1)) {
    if (Math.hypot(...p.map((v, k) => v - points[points.length - 1][k])) >= 0.75) points.push(p);
  }
  if (raw.length > 1) points.push(raw[raw.length - 1]);
  let turns = 0;
  for (let i = 1; i + 1 < points.length; i += 1) {
    const a = points[i].map((v, k) => v - points[i - 1][k]);
    const b = points[i + 1].map((v, k) => v - points[i][k]);
    const la = Math.hypot(...a), lb = Math.hypot(...b);
    if (la < 1e-6 || lb < 1e-6) continue;
    const cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb);
    if (cos < Math.cos((25 * Math.PI) / 180)) turns += 1;
  }
  return turns;
}

function RouteResult({ path }) {
  const turns = countTurns(path.straight_points || path.points);
  return (
    <section className="result" aria-live="polite" aria-label="Route result">
      <h3 className="result__title" data-tone={path.found ? "ok" : "warn"}>
        <Icon name={path.found ? "check" : "alert"} size={16} />
        {path.found ? "Route found" : "No route"}
      </h3>
      {path.found && (
        <dl className="result__grid">
          <div className="result__item">
            <dt className="result__label">Length</dt>
            <dd className="result__value result__value--lead">
              {(path.straight_length ?? path.length).toFixed(1)} m
            </dd>
          </div>
          {path.straight_length != null && (
            <div className="result__item">
              <dt className="result__label">On the grid</dt>
              <dd className="result__value result__value--muted">{path.length.toFixed(1)} m</dd>
            </div>
          )}
          <div className="result__item">
            <dt className="result__label">Turns</dt>
            <dd className="result__value">{fmt(turns)}</dd>
          </div>
          <div className="result__item">
            <dt className="result__label">Computed</dt>
            <dd className="result__value">
              {path.engine === "browser" ? "In your browser" : `On the server (${path.engine})`}
            </dd>
          </div>
        </dl>
      )}
      {path.fallback_from && (
        <p className="result__note">
          Fell back from {path.fallback_from} to {path.engine}.
        </p>
      )}
      {path.note && <p className="result__note">{path.note}</p>}
    </section>
  );
}

export function RoutePanel({ studio, onPick, onPlace, doors, exits, onNearestExit, onClearPoints, onFindPath, onCompare, onAlternatives, onGoTo }) {
  const { graph, points, pickMode, path, pathBusy, comparison, alternatives, settings, setSettings, capabilities } =
    studio;

  const engines = capabilities?.engines || ["fast"];
  const ready = Boolean(graph);
  const canRoute = ready && points.start && points.exit;

  return (
    <>
      <StepHeading step={3} title="Route" lead="The shortest route on the walkable graph to the exit you pick, drawn straight, and any real alternatives." />
      <GraphSummary graph={graph} onGoTo={onGoTo} />
      <div className="step-body" style={graph ? { paddingTop: "var(--space-4)" } : undefined}>
        {!ready && <Empty>Build the walkable graph first (step 2).</Empty>}

        <fieldset className="choices" disabled={!ready}>
          <legend className="choices__legend">Start and exit</legend>
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
          <p className="field__hint">Points snap to the graph: only its nodes can be picked.</p>
          {exits?.nodes.length > 0 && (
            <>
              <Button size="sm" disabled={!points.start} onClick={onNearestExit}>
                Use the nearest exit door
              </Button>
              <p className="field__hint">
                {exits.nodes.length} exit door{exits.nodes.length === 1 ? "" : "s"}, from {exitSourceText(exits.source)}.
                {!points.start && " Set a start first."}
              </p>
            </>
          )}
        </fieldset>
        {ready && <PlaceWithoutMouse kinds={["start", "exit"]} doors={doors} onPlace={onPlace} />}

        <Button variant="primary" size="lg" block busy={pathBusy} disabled={!canRoute} onClick={onFindPath}>
          Find egress route
        </Button>

        {path && <RouteResult path={path} />}

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

        <Disclosure summary="Route options">
          <SelectField
            label="Engine"
            value={settings.engine}
            onChange={(engine) => setSettings({ engine })}
            options={[
              { value: "fast", label: "Built-in A* (in your browser)" },
              {
                value: "topologicpy",
                label: engines.includes("topologicpy") ? "TopologicPy TGraph (server)" : "TopologicPy (unavailable)",
                disabled: !engines.includes("topologicpy"),
              },
            ]}
            hint={
              engines.includes("topologicpy")
                ? "Both return a route of the same optimum cost (proved for the built-in A*, checked in tests for TGraph); where routes tie they may draw different ones. TopologicPy's TGraph.ShortestPath runs on the server, about 30 ms against 3 ms."
                : "The server could not load topologicpy, so only the built-in engine is offered."
            }
          />
          <Toggle
            label="Draw routes straight"
            checked={settings.straightenRoutes}
            onChange={(straightenRoutes) => setSettings({ straightenRoutes })}
            hint="Straight runs, as TopologicPy's Wire.Straighten draws them, never longer than the grid route. A run whose ends both stand 0.25 m clear keeps about 0.24 m from walls at the 0.5 m grid (less on a coarser grid). Doors and stairs stay on the route. Applies to the next route."
          />
          <Button block disabled={!canRoute || engines.length < 2} onClick={onCompare}>
            Compare both engines
          </Button>
          <Button block disabled={!points.start && !points.exit && !points.fire} onClick={onClearPoints}>
            Clear all points
          </Button>
        </Disclosure>
      </div>

      <Section title="Alternative routes" badge={alternatives ? `${alternatives.routes.length} found` : null}>
        <p className="field__hint">
          Is there a second way out? <strong>Different space</strong> keeps a route only if at most half of it
          runs within 1.5 m of a route found before it, up to 1.5 times the shortest; it can stop before
          finding every such route. <strong>Yen</strong>, the method of TopologicPy&apos;s{" "}
          <code>KShortestPaths</code>, lists the next cheapest loopless routes exactly (proved), which are
          usually the best route with a step moved.
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
        <Button block busy={pathBusy} disabled={!canRoute} onClick={onAlternatives}>
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
                  {(route.straight_length ?? route.length).toFixed(1)} m
                  {route.rank === 1
                    ? " · shortest on the grid"
                    : ` · +${Math.round((route.stretch - 1) * 100)}% · ${Math.round(route.overlap * 100)}% shared`}
                </div>
              </React.Fragment>
            ))}
          </div>
        )}
        {alternatives && (
          <Notice>
            {alternatives.note ? `${alternatives.note} ` : ""}
            {alternatives.method === "yen" ? "Yen" : "Different space"}, {alternatives.engine} engine,{" "}
            {alternatives.ms.toFixed(0)} ms.
          </Notice>
        )}
      </Section>
    </>
  );
}

const KIND_LABEL = { start: "Start", exit: "Exit", fire: "Fire origin" };

/**
 * Picking on the 3D model needs a mouse. This places the same points from
 * the keyboard: an exit at one of the graph's doors, or any point by its
 * coordinates, snapped to the nearest graph node as a click would be.
 */
function PlaceWithoutMouse({ kinds, doors, onPlace }) {
  const [kind, setKind] = useState(kinds[0]);
  const [xyz, setXyz] = useState(["", "", ""]);
  const id = useId();
  const valid = xyz.every((v) => v !== "" && Number.isFinite(Number(v)));
  return (
    <Disclosure summary="Place without the mouse">
      {doors?.length > 0 && kinds.includes("exit") && (
        <div className="field">
          <label className="field__label" htmlFor={`${id}-door`}>
            <span>Exit at a door</span>
          </label>
          <select
            id={`${id}-door`}
            value=""
            onChange={(event) => {
              const door = doors[Number(event.target.value)];
              if (door) onPlace("exit", door.point);
            }}
          >
            <option value="" disabled>
              Choose a door...
            </option>
            {doors.map((door, i) => (
              <option key={door.id} value={i}>
                {door.label}
              </option>
            ))}
          </select>
        </div>
      )}
      <fieldset className="choices">
        <legend className="choices__legend">By coordinates (model units, as shown under each point)</legend>
        {kinds.length > 1 && (
          <SelectField
            label="Point"
            value={kind}
            onChange={setKind}
            options={kinds.map((k) => ({ value: k, label: KIND_LABEL[k] }))}
          />
        )}
        <div className="grid-3">
          {["x", "y", "z"].map((axis, i) => (
            <NumberField
              key={axis}
              label={axis}
              value={xyz[i] === "" ? null : Number(xyz[i])}
              step={0.1}
              onChange={(v) => setXyz((cur) => cur.map((c, j) => (j === i ? (v === null ? "" : String(v)) : c)))}
            />
          ))}
        </div>
        <Button block disabled={!valid} onClick={() => onPlace(kind, xyz.map(Number))}>
          Place {KIND_LABEL[kind].toLowerCase()}
        </Button>
      </fieldset>
    </Disclosure>
  );
}

function PickRow({ name, color, point, active, disabled, onPick, shortcut }) {
  return (
    <div className="picker" data-active={active}>
      <span className="picker__dot" style={{ background: color }} />
      <div className="picker__body">
        <div className="picker__name">{name}</div>
        <div className="picker__coord">{active ? "click on the graph..." : coord(point)}</div>
      </div>
      <Button size="sm" onClick={onPick} disabled={disabled} title={`Shortcut: ${shortcut}`} aria-keyshortcuts={shortcut}>
        {active ? "Cancel" : point ? "Move" : "Set"} <span className="kbd">{shortcut}</span>
      </Button>
    </div>
  );
}

// ----------------------------------------------------------------- simulate

const FIRE_MODELS = [
  {
    value: "temperature",
    label: "Heat diffusion",
    description: "Each node relaxes toward its neighbours; walls slow it. Needed for hazard-weighted rerouting.",
  },
  { value: "flood", label: "Spread along links", description: "One link per step, so walls and doors shape the front." },
  { value: "radial", label: "Distance bands", description: "Straight-line distance from the origin. Ignores walls." },
];

function RerouteResult({ dynamicPath, step }) {
  const unsafe = dynamicPath.found === false;
  return (
    <section className="result" aria-live="polite" aria-label="Route during the fire">
      <h3 className="result__title" data-tone={unsafe ? "warn" : "ok"}>
        <Icon name={unsafe ? "alert" : "check"} size={16} />
        {unsafe
          ? "No safe route"
          : dynamicPath.changed
            ? `The route changed at step ${dynamicPath.step}`
            : `Route still clear at step ${step}`}
      </h3>
      <dl className="result__grid">
        <div className="result__item">
          <dt className="result__label">Cost</dt>
          <dd className="result__value result__value--lead">{dynamicPath.cost?.toFixed(1) ?? "-"}</dd>
        </div>
        <div className="result__item">
          <dt className="result__label">Last checked</dt>
          <dd className="result__value">step {dynamicPath.step}</dd>
        </div>
      </dl>
      {dynamicPath.note && <p className="result__note">{dynamicPath.note}</p>}
    </section>
  );
}

/**
 * Seconds, as "42.0 s" or "3 min 5 s". `round` is Math.floor for a deadline,
 * so it never reads later than it is, and Math.ceil for the first time that
 * fails, so the two never print the same.
 */
function secs(s, round = Math.round) {
  if (s === null || s === undefined || Number.isNaN(s)) return "-";
  if (s === Infinity) return "never";
  if (s < 90) return `${(round(s * 10) / 10).toFixed(1)} s`;
  const whole = round(s);
  return `${Math.floor(whole / 60)} min ${whole % 60} s`;
}

/** A fractional effective dose: small ones in scientific notation. */
function fed(d) {
  if (d === null || d === undefined) return "-";
  return d > 0 && d < 0.001 ? d.toExponential(1) : d.toFixed(3);
}

function Swatch({ color }) {
  return <span className="legend__swatch legend__swatch--inline" style={{ background: color }} aria-hidden="true" />;
}

/**
 * The plan's heading. "Safe route" only when some route is within the dose
 * limit, timed by the first such route; when every route exceeds it the
 * heading says so (review M11).
 */
function planHeading(result, inputs) {
  const depart = inputs.departAfter || 0;
  const within = result.front.find((point) => point.dose <= inputs.doseLimit);
  if (within) return { ok: true, text: `Safe route: out in ${secs(within.arrival - depart)}` };
  if (result.earliest) return { ok: false, text: `Out in ${secs(result.earliest.arrival - depart)}, above the dose limit` };
  return { ok: false, text: "No safe route within the forecast" };
}

/** The leaving time: rounded down, and "at least" when the forecast's end, not the fire, sets it. */
function leaveWithin(leaveBy) {
  if (!leaveBy) return "too late";
  return `${leaveBy.horizon ? "at least " : ""}${secs(leaveBy.good, Math.floor)}`;
}

function ForecastResult({ plan, settings, deadlinesShown }) {
  const { result, inputs } = plan;
  const { earliest, leastDose, leaveBy, snapshot, front, truncated, coarsened, mapStep } = result;
  // What the plan was made with, not what the fields say now (review M2).
  const depart = inputs.departAfter || 0;
  const limit = inputs.doseLimit;
  const tradeoff = front.length > 1;
  const unsafePlan = snapshot && !snapshot.safe && !snapshot.beyondForecast;
  const heading = planHeading(result, inputs);
  const changed = PLAN_SETTINGS.some((key) => settings[key] !== inputs[key]);
  // Announced by the section's status line, not here: a live region that
  // mounts with its content is not read, and this one was read again at
  // every keystroke in the settings (review m5).
  return (
    <section className="result" aria-label="Plan with the forecast" aria-busy={plan.busy || undefined}>
      <h3 className="result__title" data-tone={heading.ok ? "ok" : "warn"}>
        <Icon name={heading.ok ? "check" : "alert"} size={16} />
        {heading.text}
      </h3>
      {changed && (
        <Notice tone="warning">The settings have changed since this plan was made. Plan again to use them.</Notice>
      )}
      <dl className="result__grid">
        <div className="result__item">
          <dt className="result__label">Leave within</dt>
          <dd className="result__value result__value--lead">{leaveWithin(leaveBy)}</dd>
        </div>
        <div className="result__item">
          <dt className="result__label">Closest call</dt>
          <dd className="result__value">
            {!earliest ? "-" : earliest.slack === Infinity ? "fire never near" : `${secs(earliest.slack)} ahead`}
          </dd>
        </div>
        <div className="result__item">
          <dt className="result__label">Least dose (FED)</dt>
          <dd className="result__value">{fed(leastDose?.dose)}</dd>
        </div>
        <div className="result__item">
          <dt className="result__label">Forecast</dt>
          <dd className="result__value">{secs(result.horizon)}</dd>
        </div>
      </dl>
      <div className="result__note forecast__notes">
        {earliest && (
          <span>
            <Swatch color={FORECAST_COLORS.earliest} />
            {earliest.slack === Infinity
              ? "The earliest safe route never passes a point the fire reaches within the forecast."
              : `The earliest safe route reaches every point before the fire makes it untenable; at its closest it is ${secs(earliest.slack)} ahead.`}
          </span>
        )}
        {leaveBy && !leaveBy.horizon && (
          <span>
            Leaving by {secs(leaveBy.good, Math.floor)} after ignition still gets out with a dose of at most {limit};
            leaving at {secs(leaveBy.bad, Math.ceil)} does not.
          </span>
        )}
        {leaveBy?.horizon && (
          <span>
            Leaving by {secs(leaveBy.good, Math.floor)} after ignition still gets out with a dose of at most {limit}. The
            forecast ends at {secs(result.horizon)}, so no later departure was checked: the real deadline may be later.
          </span>
        )}
        {leaveBy && depart > leaveBy.good && (
          <span>The departure planned, {secs(depart)} after ignition, is later than that.</span>
        )}
        {earliest && !leaveBy && <span>A safe route exists, but every one accrues more than the dose limit.</span>}
        {tradeoff && leastDose && earliest && (
          <span>
            <Swatch color={FORECAST_COLORS.leastDose} />
            The least-dose route takes {secs(leastDose.arrival - earliest.arrival)} longer and accrues{" "}
            {Math.round((1 - leastDose.dose / front[0].dose) * 100)}% less dose; {front.length} routes trade time against
            dose.
          </span>
        )}
        {unsafePlan && (
          <span>
            <Swatch color={FORECAST_COLORS.snapshot} />
            A route planned on the fire as it is at departure reaches a point {secs(snapshot.lateBy)} after it becomes
            untenable. The forecast route avoids it.
          </span>
        )}
        {mapStep && deadlinesShown && (
          <span>
            The graph is coloured by escape deadline: for each point, the latest departure time on a{" "}
            {mapStep.toFixed(2)} s grid that still gets out within the dose limit. It is never later than the exact
            deadline, but it can be several seconds earlier, because every walking time is rounded up to the grid.
          </span>
        )}
        {!result.deadlines && settings.showDeadlineMap && <span>Plan again to colour the graph by escape deadline.</span>}
        {truncated && (
          <span>
            The dose search hit its label budget, so the least dose shown may not be the least. The route shown is safe
            and has the dose shown.
          </span>
        )}
        {leaveBy?.truncated && (
          <span>A search for the leaving time hit its label budget, so that time may be too early.</span>
        )}
        {coarsened && (
          <span>The forecast was stored every {secs(coarsened)} to fit in memory, which only makes it more cautious.</span>
        )}
        <span className="field__hint">
          Times are counted from ignition, on the clock set for this plan ({inputs.forecastSecondsPerStep} s per fire
          step). The heat is the model&apos;s index, so the dose ranks routes under the model; it is not a measured
          dose.
        </span>
      </div>
    </section>
  );
}

/** One line for screen readers, kept mounted so that a change is read. */
function planStatus(plan) {
  if (plan.busy) return "Planning with the forecast.";
  if (!plan.result || !plan.inputs) return "";
  const heading = planHeading(plan.result, plan.inputs);
  return `${heading.text}. Leave within ${leaveWithin(plan.result.leaveBy)}.`;
}

/** Where the exit doors came from, in words. */
function exitSourceText(source) {
  if (!source) return "nowhere";
  if (source.startsWith("Pset_DoorCommon")) return `the model's ${source.split(".")[1]} flags`;
  if (source.startsWith("IfcRelSpaceBoundary")) return "the model's external space boundaries";
  return source;
}

function ForecastSection({ studio, exits, onPlan, deadlinesShown }) {
  const { graph, points, settings, setSettings, forecastPlan, fire } = studio;
  const modelExits = settings.forecastExits === "model" && exits?.nodes.length > 0;
  const ready = Boolean(graph && points.start && points.fire && (modelExits || points.exit));
  return (
    <Section title="Escape deadlines and forecast routes" badge={forecastPlan.result?.earliest ? "planned" : null}>
      <p className="field__hint">
        Runs the fire to its last step first, then routes with the state each point will be in when you would reach it,
        and works out how long you can wait before leaving.
      </p>
      {exits?.nodes.length > 0 && (
        <Choices
          legend="Exits"
          name="forecast-exits"
          value={settings.forecastExits}
          onChange={(forecastExits) => setSettings({ forecastExits })}
          options={[
            {
              value: "model",
              label: `All ${exits.nodes.length} exit door${exits.nodes.length === 1 ? "" : "s"}`,
              description: `From ${exitSourceText(exits.source)}.`,
            },
            { value: "picked", label: "The exit you picked", description: "Set in the Route step." },
          ]}
        />
      )}
      <Disclosure summary="Clock, speed and limits">
        <div className="grid-2">
          <NumberField
            label="Seconds per fire step"
            value={settings.forecastSecondsPerStep}
            min={0.5}
            max={60}
            step={0.5}
            onChange={(v) => setSettings({ forecastSecondsPerStep: v ?? 5 })}
            hint="The fire models count steps; this sets the clock."
          />
          <NumberField
            label="Walking speed (m/s)"
            value={settings.walkSpeed}
            min={0.2}
            max={3}
            step={0.1}
            onChange={(v) => setSettings({ walkSpeed: v ?? 1.2 })}
          />
          <NumberField
            label="Untenable from (model °C)"
            value={settings.untenableAbove}
            min={25}
            max={120}
            step={5}
            onChange={(v) => setSettings({ untenableAbove: v ?? 80 })}
          />
          <NumberField
            label="Leave after (s)"
            value={settings.departAfter}
            min={0}
            step={5}
            onChange={(v) => setSettings({ departAfter: v ?? 0 })}
            hint="Counted from ignition."
          />
        </div>
        <NumberField
          label="Dose limit (FED)"
          value={settings.doseLimit}
          min={0.01}
          max={1}
          step={0.05}
          onChange={(v) => setSettings({ doseLimit: v ?? 0.3 })}
          hint="ISO 13571 uses an FED of 0.3 to protect most occupants and 1 for the average adult."
        />
        <Toggle
          label="Colour the graph by escape deadline"
          checked={settings.showDeadlineMap}
          onChange={(showDeadlineMap) => setSettings({ showDeadlineMap })}
        />
      </Disclosure>
      <Button block busy={forecastPlan.busy} disabled={!ready || fire.running} onClick={onPlan}>
        Plan with the forecast
      </Button>
      {!ready && graph && (
        <p className="field__hint">
          Needs a start{modelExits ? "" : " and an exit"} (Route step) and a fire origin.
        </p>
      )}
      <p className="visually-hidden" role="status">
        {planStatus(forecastPlan)}
      </p>
      {forecastPlan.result && forecastPlan.inputs && (
        <ForecastResult plan={forecastPlan} settings={settings} deadlinesShown={deadlinesShown} />
      )}
    </Section>
  );
}

/**
 * How far the policy's return falls short of the best within the step limit,
 * as a share of the best's size. Flagged above 5%: a gap of a few thousandths
 * is a tie broken another way, not a worse policy.
 */
const RETURN_SHORTFALL_WARN = 0.05;
function returnShortfall(r) {
  return Math.max(0, r.optimal_return - r.policy_return) / Math.max(Math.abs(r.optimal_return), 1e-9);
}
function returnVsBest(r) {
  const gap = returnShortfall(r);
  const pct = gap * 100;
  const short = pct < 0.5 ? "" : `, ${pct < 10 ? pct.toFixed(1) : Math.round(pct)}% short`;
  return `${r.policy_return.toFixed(2)} / ${r.optimal_return.toFixed(2)}${short}`;
}

export function SimulatePanel({
  studio,
  exits,
  onPick,
  onPlace,
  onStartFire,
  onStopFire,
  onTrainRl,
  onPlanForecast,
  deadlinesShown,
  onGoTo,
}) {
  const { graph, points, pickMode, settings, setSettings, fire, dynamicPath, rl } = studio;
  const ready = Boolean(graph);

  return (
    <>
      <StepHeading step={4} title="Simulate a fire" lead="Watch it spread and the route change, or train an agent against it." />
      <GraphSummary graph={graph} onGoTo={onGoTo} />
      <div className="step-body" style={graph ? { paddingTop: "var(--space-4)" } : undefined}>
        {!ready && <Empty>Build the walkable graph first (step 2).</Empty>}

        <PickRow
          name="Fire origin"
          color="var(--viz-fire)"
          point={points.fire}
          active={pickMode === "fire"}
          disabled={!ready}
          onPick={() => onPick("fire")}
          shortcut="F"
        />
        {ready && <PlaceWithoutMouse kinds={["fire"]} onPlace={onPlace} />}

        <Choices
          legend="How the fire spreads"
          name="fire-model"
          value={settings.fireModel}
          onChange={(fireModel) => setSettings({ fireModel })}
          options={FIRE_MODELS}
          disabled={fire.running}
        />

        <div className="row">
          <Button
            variant="primary"
            size="lg"
            style={{ flex: 1 }}
            disabled={!ready || fire.running || !points.fire}
            onClick={onStartFire}
          >
            Start fire
          </Button>
          <Button variant="danger" size="lg" disabled={!fire.running} onClick={onStopFire}>
            Stop
          </Button>
        </div>
        {ready && !points.fire && <p className="field__hint">Set a fire origin to start the simulation.</p>}
        {ready && points.fire && settings.reroute && !(points.start && points.exit) && (
          <p className="field__hint">Set a start and exit in the Route step to see the route react.</p>
        )}

        {dynamicPath && <RerouteResult dynamicPath={dynamicPath} step={fire.step} />}

        <Disclosure summary="Rerouting and timing">
          <Toggle
            label="Recompute the route as the fire grows"
            checked={settings.reroute}
            onChange={(reroute) => setSettings({ reroute })}
            hint={
              settings.fireModel === "temperature"
                ? "Re-solved against the current heat while the fire spreads."
                : "Re-solved with burning nodes treated as impassable."
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
            hint="Heat diffusion only. Link cost is length x (1 + weight x heat); zero ignores the fire."
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
            label="Impassable above (model °C)"
            value={settings.lethalityThreshold}
            min={30}
            max={200}
            step={5}
            placeholder="none"
            onChange={(lethalityThreshold) => setSettings({ lethalityThreshold })}
            hint="Links hotter than this are closed. Left blank, heat only adds cost. The scale is an index, not a measured temperature (see Methods)."
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
        </Disclosure>
      </div>

      <ForecastSection studio={studio} exits={exits} onPlan={onPlanForecast} deadlinesShown={deadlinesShown} />

      <Section title="Train an agent against this fire" defaultOpen={false}>
        <p className="field__hint">
          Tabular Q-learning over the same graph, with the spreading fire as a penalty. Studio first checks
          whether any escape exists, so a failed agent and an impossible escape are told apart.
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
            max={RL_MAX_STEPS}
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
          label="Fire speed (links per move)"
          value={settings.rlFireSpeed}
          min={0.05}
          max={10}
          step={0.05}
          disabled={!settings.rlUseFire}
          onChange={(rlFireSpeed) => setSettings({ rlFireSpeed: rlFireSpeed ?? 1 })}
          hint="1 = the front moves exactly as fast as the evacuee. On a real plan that often leaves no escape; the result says when."
        />
        <Button block busy={rl.busy} disabled={!ready || !points.start || !points.exit} onClick={onTrainRl}>
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
            <Stat label="Late-training success" value={`${Math.round((rl.result?.success_rate ?? 0) * 100)}%`} />
            {rl.result?.optimal_return != null && rl.result.policy_return != null && (
              <Stat
                label="Return vs best within the step limit"
                value={returnVsBest(rl.result)}
                warn={returnShortfall(rl.result) > RETURN_SHORTFALL_WARN}
              />
            )}
            {settings.rlUseFire && rl.result?.fire_start_id != null && (
              <Stat
                label="Escape possible"
                value={rl.result.escape_steps == null ? "no" : `yes, ${rl.result.escape_steps} moves`}
                warn={rl.result.escape_steps == null}
              />
            )}
          </div>
        )}
      </Section>
    </>
  );
}
