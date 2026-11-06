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
import { PLAN_SETTINGS, fireModelIn, hasRooms, isFlamescapes } from "../state/useStudio.js";
import { FIRE_CLASSES, FIRE_CLASS_KEYS } from "../core/fireClasses.js";
import {
  Button,
  Choices,
  Disclosure,
  Empty,
  FileButton,
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

/**
 * Vertical cores found in the model (core/cores.js): stairs joined along an
 * assumed walking line because their treads are not modelled, and lifts and
 * escalators, which join floors only when the setting allows them. Warns when
 * a model has several storeys and nothing joins them.
 */
function CoresNotice({ cores, stats, storeys, useLifts }) {
  const list = cores || [];
  const assumed = list.filter((c) => c.kind === "stair" && c.linked > 0);
  const lifts = list.filter((c) => c.kind !== "stair");
  const name = (c) => c.names?.[0] || c.sources?.[0] || c.kind;
  const joined = (stats?.stair_nodes ?? 0) > 0 || (stats?.lift_nodes ?? 0) > 0;
  const rows = [];
  if (assumed.length) {
    rows.push([
      "Stairs without treads",
      `${assumed.length}, walked along an assumed line (${assumed.slice(0, 3).map(name).join(", ")}${assumed.length > 3 ? ", ..." : ""})`,
    ]);
  }
  if (lifts.length) {
    const used = lifts.filter((c) => c.linked > 0);
    rows.push([
      "Lifts and escalators",
      `${lifts.length} (${lifts.slice(0, 3).map(name).join(", ")}${lifts.length > 3 ? ", ..." : ""}): ${useLifts ? `${used.length} joining floors` : "not used"}`,
    ]);
  }
  return (
    <>
      {rows.length > 0 && <KeyValue entries={rows} />}
      {assumed.length > 0 && (
        <p className="field__hint">
          A stair whose treads are not in the model is walked along an assumed dog-leg line inside its footprint, with 175 mm risers and 280 mm goings, so a
          route through it is about as long as the stair. Its width and landings are not known.
        </p>
      )}
      {lifts.length > 0 && !useLifts && (
        <p className="field__hint">Lifts are not used in a fire unless they are evacuation lifts. Turn them on in the graph settings to route through them.</p>
      )}
      {storeys > 1 && !joined && (
        <Notice tone="warning">
          <strong>Nothing joins the {storeys} storeys.</strong>{" "}
          {lifts.length && !useLifts
            ? "The model has no stair, ramp or stair core the graph could use, and its lifts are off,"
            : "The model has no stair, ramp, lift or stair core the graph could use,"}{" "}
          so a route cannot change floor. Each floor can only be left at its own level.
        </Notice>
      )}
    </>
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
    s.startsWith("Pset_DoorCommon") ? s.split(".")[1] : s === "LOD1 exit" ? "marked in the rooms model" : "external space boundary",
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

const LABEL_NAMES = {
  wall: "Walls",
  slab: "Slabs and floors",
  roof: "Roofs",
  door: "Doors",
  window: "Windows",
  stair: "Stairs and ramps",
  railing: "Railings",
  column: "Columns",
  beam: "Beams",
  covering: "Coverings and ceilings",
  furniture: "Furniture",
  appliance: "Appliances",
  sanitary: "Sanitary fittings",
  heater: "Heaters",
  lighting: "Lights",
  mep: "Building services",
  space: "Rooms",
  other: "Other",
};

/**
 * What the elements are (core/identify.js): typed elements by their class,
 * generic ones by name, then box, then, when asked, IFCNet's shape classifier
 * (core/identifyGeometry.js) for those only the box could place.
 */
function ElementIdentification({ identification, onIdentify, onIdentifyByShape }) {
  const answers = identification?.answers;
  const counts = new Map();
  for (const a of answers || []) counts.set(a.label, (counts.get(a.label) || 0) + 1);
  const generic = (answers || []).filter((a) => a.by !== "class");
  const by = (...kinds) => generic.filter((a) => kinds.includes(a.by)).length;
  const open = by("box", "none");
  return (
    <Disclosure summary="What the elements are">
      <p className="field__hint">
        Each element by its IFC class where the exporter typed it. Generic ones (IfcBuildingElementProxy and the like), as furniture and appliances often
        arrive, by their name, else by their size.
      </p>
      <Button block busy={identification?.busy} onClick={onIdentify}>
        Identify elements
      </Button>
      {answers && (
        <>
          <KeyValue entries={[...counts].sort((a, b) => b[1] - a[1]).map(([label, n]) => [LABEL_NAMES[label] || label, fmt(n)])} />
          <p className="field__hint">
            {generic.length
              ? `${fmt(generic.length)} generic: ${fmt(by("name", "predefined"))} by name or type, ${fmt(by("box"))} by size, ${fmt(by("shape"))} by shape, ${fmt(by("none"))} not identified.`
              : "Every element is typed by its class."}
          </p>
          {(open > 0 || identification.shapeRun) && (
            <>
              <Button block busy={identification.shapeBusy} disabled={identification.shapeRun || !open} onClick={onIdentifyByShape}>
                {identification.shapeBusy
                  ? `By shape: ${fmt(identification.shapeDone)} of ${fmt(identification.shapeTotal)}`
                  : identification.shapeRun
                    ? "Checked by shape"
                    : `Identify ${fmt(open)} by shape (IFCNet)`}
              </Button>
              <p className="field__hint">
                IFCNet&apos;s point-cloud network (Emunds et al. 2021) looks at the shape of each element the name could not place, and its answer replaces
                the size&apos;s when it is at least 90% sure. The first use downloads 4.5 MB (the model from this site, onnxruntime-web from jsDelivr) and
                takes about 0.2 s an element.
                {identification.shapeRun && ` It identified ${fmt(identification.byShape)}; the rest keep the rules' answer.`}
              </p>
            </>
          )}
        </>
      )}
    </Disclosure>
  );
}

/**
 * Describe this model: a plain description written from the facts Studio
 * computed (core/describeFacts.js), on this device or by the site's
 * Cloudflare describer (lib/describe.js). The facts are shown first.
 */
function DescribeModel({ describe, onDescribe, onStop, onEngine }) {
  if (!describe) return null;
  const onDevice = describe.status !== "unsupported" && describe.status !== "unavailable" && describe.status !== "checking";
  const engine = describe.engine || "device";
  const ready = engine === "device" ? onDevice : describe.cloud;
  const deviceNote = {
    checking: "Checking for this browser's built-in model...",
    unsupported: "This browser has no built-in model (Chrome 148 or later on a desktop computer has one).",
    unavailable: "This browser's built-in model cannot run here (it needs more than 4 GB of GPU memory, or 16 GB of RAM).",
    downloadable: "Chrome downloads its model the first time (once, a few GB); nothing leaves this computer.",
    downloading: "Chrome is downloading its model; nothing leaves this computer.",
    available: "Written on this computer by Chrome's built-in model; nothing is sent.",
  }[describe.status];
  return (
    <Disclosure summary="Describe this model">
      <p className="field__hint">
        A short description of the building, its ways out and its fire, written by a language model from the facts Studio has worked out from the IFC and
        its own simulations, and from nothing else. It grows as you go: describe again after building the graph, finding a route or running a fire.
      </p>
      <div className="field">
        <div className="field__label">
          <span>Written by</span>
        </div>
        <Segmented
          label="Written by"
          value={engine}
          onChange={onEngine}
          disabled={describe.busy}
          options={[
            { value: "device", label: "This device" },
            { value: "cloud", label: "Cloudflare" },
          ]}
        />
        <p className="field__hint">
          {engine === "device"
            ? deviceNote
            : describe.cloud
              ? "The facts below are sent to the site's describer on Cloudflare Workers AI, which says it neither keeps nor trains on them; the IFC itself is not sent."
              : "This site has no Cloudflare describer set up yet (cloudflare/describe-worker in the repository)."}
        </p>
      </div>
      <Disclosure summary="The facts it is given">
        <pre className="describe__facts">{describe.facts}</pre>
      </Disclosure>
      <div className="row">
        <Button block variant="primary" busy={describe.busy} disabled={!ready} onClick={onDescribe}>
          {engine === "cloud" ? "Send the facts and describe" : "Describe"}
        </Button>
        {describe.busy && (
          <Button onClick={onStop}>
            Stop
          </Button>
        )}
      </div>
      {describe.progress != null && describe.progress < 1 && (
        <p className="field__hint">Downloading the model: {Math.round(describe.progress * 100)}%</p>
      )}
      {describe.error && <Notice tone="warning">{describe.error}</Notice>}
      {describe.text && (
        <div className="describe__text" aria-live="polite">
          {describe.text.split(/\n{2,}/).map((para, i) => (
            <p key={i}>{para}</p>
          ))}
          <p className="field__hint">
            Written by {describe.model || "a language model"} from the facts above. Check it against the panels: a language model can still word things
            wrongly.
          </p>
        </div>
      )}
    </Disclosure>
  );
}

/** What a rooms (LOD1) model holds, what its reading warned of, and its export as a GeoJSON plan. */
function RoomsModel({ model, file, modelInfo, semantics, onExport }) {
  const walked = model.rooms.filter((r) => r.walkable !== false).length;
  const kinds = (kind) => model.links.filter((l) => l.kind === kind).length;
  const warnings = model.meta?.warnings || [];
  const source = { topologicpy: "TopologicPy LOD1 export", geojson: "GeoJSON plan", "ifc-spaces": "IFC spaces", drawing: "floor-plan drawings" };
  return (
    <>
      <div className="stats">
        <Stat label="Rooms" value={fmt(model.rooms.length)} />
        <Stat label="Storeys" value={fmt(model.levels.length)} />
        <Stat label="Doors" value={fmt(model.doors.length)} />
        <Stat label="Stairs" value={fmt(kinds("stair") + kinds("ramp"))} />
      </div>
      <KeyValue
        entries={[
          ["File", file?.name || "-"],
          ["Source", source[model.meta?.source] || model.meta?.source || "rooms model"],
          ["Walked in", `${fmt(walked)} of ${fmt(model.rooms.length)} rooms`],
          ["Lifts", fmt(kinds("lift"))],
          ["Opened in", formatMs(modelInfo.totalMs)],
        ]}
      />
      <p className="field__hint">
        Drawn as rooms: walls 0.2 m thick on the rooms&apos; outlines, slabs under each floor, stairs as treads along their walking lines. Shafts, voids
        and tanks are not walked in.
      </p>
      {warnings.length > 0 && (
        <Notice tone="warning">
          <strong>
            {warnings.length} warning{warnings.length > 1 ? "s" : ""} while reading the model.
          </strong>{" "}
          {warnings.slice(0, 4).join(" ")}
          {warnings.length > 4 ? ` And ${warnings.length - 4} more.` : ""}
        </Notice>
      )}
      <EgressCheck semantics={semantics} />
      <div className="field">
        <Button block onClick={onExport}>
          Export as a GeoJSON plan
        </Button>
        <p className="field__hint">Rooms, doors, stairs and storeys in plan metres, the format Studio reads back and other tools can edit.</p>
      </div>
    </>
  );
}

export function ModelPanel({
  studio,
  examples,
  onExample,
  onClearCache,
  identification,
  onIdentify,
  onIdentifyByShape,
  describe,
  onDescribe,
  onStopDescribe,
  onDescribeEngine,
  onOpenRooms,
  onExportRooms,
}) {
  const { modelInfo, file, settings, setSettings, cacheInfo } = studio;
  const rooms = modelInfo?.lod1;

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
        ) : rooms ? (
          <RoomsModel model={rooms} file={file} modelInfo={modelInfo} semantics={studio.semantics} onExport={onExportRooms} />
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
                ["Source", modelInfo.prepared ? "converted with the site" : modelInfo.cached ? "local cache" : "converted now"],
              ]}
            />
            <p className="field__hint">
              {modelInfo.prepared
                ? "This example was converted when the site was built, so it opened without reading the IFC. Its exit flags and storeys were read then too."
                : modelInfo.cached
                  ? "The geometry, exit flags and storeys came from the local cache, without reading the IFC again."
                  : "Converted and cached: opening this file again skips the conversion, and keeps the exit flags and storeys read from it."}
            </p>
            <EgressCheck semantics={studio.semantics} />
            <ElementIdentification identification={identification} onIdentify={onIdentify} onIdentifyByShape={onIdentifyByShape} />
            <DescribeModel describe={describe} onDescribe={onDescribe} onStop={onStopDescribe} onEngine={onDescribeEngine} />
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

      <Section title="Rooms models (LOD1)" defaultOpen={false}>
        <p className="field__hint">
          A building as rooms, storeys, doors and stairs, with no IFC: a TopologicPy LOD1 export (the CellComplex JSON, with its circulation and
          adjacency graphs chosen together with it) or a GeoJSON plan. The graph, routes, code check and both fire models run on it as on an IFC.
        </p>
        <FileButton accept=".json,.geojson" multiple onFile={onOpenRooms}>
          Open a rooms model
        </FileButton>
      </Section>

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

/** Download the graph (and, after a run, the fire's results per node) for TopologicPy. */
function ExportTopologic({ onExport, disabled, withRun }) {
  return (
    <div className="field">
      <Button block disabled={disabled} onClick={onExport}>
        Export for TopologicPy (JSON)
      </Button>
      <p className="field__hint">
        The walkable graph in the IFC&apos;s own coordinates, with the route
        {withRun ? " and, per node, the fire's gas temperature, visibility and CO over time and when it became untenable" : ""}, as the JSON
        TopologicPy reads with <code>Graph.ByJSONFile</code>.
      </p>
    </div>
  );
}

export function GraphPanel({ studio, onBuildGraph, onNavigationCheck, onExportTopologic }) {
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

        {graph && !graph.stats.nodes && (
          <Notice tone="warning">
            <strong>No walkable graph.</strong> {graph.emptyCause}
          </Notice>
        )}
        {graph && graph.stats.nodes > 0 && (
          <>
            <div className="stats">
              <Stat label="Nodes" value={fmt(graph.stats.nodes)} />
              <Stat label="Links" value={fmt(graph.stats.edges)} />
              <Stat label="Doors" value={fmt(graph.stats.door_nodes)} />
              <Stat label="Reachable" value={`${reachablePct(graph.stats)}%`} warn={reachablePct(graph.stats) < 90} />
            </div>
            <PartsNotice stats={graph.stats} />
            <CoresNotice cores={graph.cores} stats={graph.stats} storeys={modelInfo?.stats?.storeys ?? 0} useLifts={graph.useLifts} />
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
        <Toggle
          label="Use lifts and escalators"
          checked={settings.useLifts}
          onChange={(useLifts) => setSettings({ useLifts })}
          hint="Off by default: in a fire, lifts are not used unless they are evacuation lifts. When on, a lift joins every floor beside it, and a ride counts as walking its rise, with no waiting for the car."
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
      {graph && (
        <div className="step-body">
          <ExportTopologic onExport={onExportTopologic} disabled={graphBusy} />
        </div>
      )}
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

/**
 * The egress code check (core/codeCheck.js): per storey, the walk to an exit,
 * the exits against the occupant load and their separation; the exit doors'
 * widths; high-rise. IBC 2021's values for offices, to be confirmed against
 * SBC 201, whose text is not open.
 */
function CodeCheck({ code, settings, setSettings, onRun, onShow, ready }) {
  const result = code?.result;
  const tone = { pass: "ok", fail: "warn" };
  const mark = { pass: "Pass", fail: "Fail", "cannot tell": "Cannot tell", "n/a": "Not applicable" };
  return (
    <Section title="Code check" defaultOpen={false}>
      <p className="field__hint">
        Offices (occupancy group B), against the International Building Code 2021, which the Saudi Building Code SBC 201-2024 appears to follow. The
        SBC&apos;s own text is not open, so every limit is to be confirmed against SBC 201. Distances are walked on the graph to the nearest exit door, or,
        above and below it, to the nearest stair; the occupant load uses the walkable area, a lower bound. Exit doors come from the IFC&apos;s exit flags,
        which some exporters set on every door along an escape route: then exits are counted too generously.
      </p>
      <Toggle
        label="The building has automatic sprinklers"
        checked={settings.codeSprinklered}
        onChange={(codeSprinklered) => setSettings({ codeSprinklered })}
        hint="Sprinklers allow longer walks (91.44 m against 60.96 m) and exits closer together (1/3 of the diagonal against 1/2)."
      />
      <Button block disabled={!ready} onClick={onRun}>
        Check the egress
      </Button>
      {result && (
        <>
          <Notice tone={result.building.highRise ? "warning" : "neutral"}>
            <strong>{result.building.highRise ? "High-rise." : "Not high-rise."}</strong> {result.building.text}.
            {result.building.provisions.length > 0 && ` Check by hand: ${result.building.provisions.join("; ")}.`}
          </Notice>
          <ul className="code-list">
            {result.storeys.map((s) => (
              <li key={s.index} className="code-list__storey">
                <strong>
                  {s.name}
                  {s.discharge ? ", exit doors" : ""}
                </strong>{" "}
                <span className="field__hint">
                  {Math.round(s.area)} m² walkable, at least {s.occupants} occupants
                </span>
                <ul>
                  {s.checks.map((c) => (
                    <li key={c.id} data-tone={tone[c.status] || "neutral"}>
                      <span className="pill">{mark[c.status]}</span> {c.text}. <span className="field__hint">{c.clause}.</span>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
            <li className="code-list__storey">
              <strong>Exit doors</strong>
              <ul>
                <li data-tone={tone[result.doors.status] || "neutral"}>
                  <span className="pill">{mark[result.doors.status]}</span> {result.doors.text}. <span className="field__hint">{result.doors.clause}.</span>
                </li>
              </ul>
            </li>
          </ul>
          <Toggle
            label="Show the distance to spare on the graph"
            checked={Boolean(code.shown)}
            onChange={onShow}
            hint="Red where a place is at or over the walking-distance limit, yellow where it is well within; grey where no exit can be reached on its storey."
          />
          <p className="field__hint">
            Not checked yet: the common path of travel, dead ends, stair widths and capacity, and whether each stair is enclosed. {result.limits.source}.
          </p>
        </>
      )}
    </Section>
  );
}

export function RoutePanel({
  studio,
  onPick,
  onPlace,
  doors,
  exits,
  onNearestExit,
  onClearPoints,
  onFindPath,
  onCompare,
  onAlternatives,
  onGoTo,
  code,
  onCodeCheck,
  onShowTravel,
}) {
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
      <CodeCheck code={code} settings={settings} setSettings={setSettings} onRun={onCodeCheck} onShow={onShowTravel} ready={Boolean(graph)} />
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
              optional
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

// Offered only on a model with IfcSpace rooms (src/core/zone.js, rooms.js).
const ZONE_MODEL = {
  value: "zone",
  label: "Two-zone rooms (degrees C)",
  description:
    "A smoke layer and its gas temperature in each room, on a clock in seconds, spreading through open doors. Needs the model's rooms (IfcSpace).",
};

// Offered when the optional Flamescapes solver is switched on
// (src/core/flamescapes.js; Wrede, Wagner et al. 2026).
const FLAMESCAPES_MODELS = [
  {
    value: "flamescapes",
    label: "Flamescapes: fire and smoke",
    description:
      "The Flamescapes solver on a grid built from the IFC: gas flow, burning of the room contents, radiation and smoke, read at head height.",
  },
  {
    value: "flamescapes-embers",
    label: "Flamescapes: fire, smoke and embers",
    description: "As above, with embers carried by the hot gas that can start spot fires away from the flame.",
  },
];

/** What the Flamescapes solver leaves out, shown with its controls (Methods has more). */
const FLAMESCAPES_LIMITS =
  "A browser version of a graphics fire solver: grid cells of 0.2 to 0.45 m, not centimetres, so flames and " +
  "thin walls are coarse; room contents are a statistical fire load, not the model's furniture; walls and slabs " +
  "do not burn; doors open, windows closed. It shows how heat and smoke may move through this building; it is " +
  "not a calibrated fire engineering model.";

const ZONE_GROWTH = [
  { value: "slow", label: "Slow (1 MW in 600 s)" },
  { value: "medium", label: "Medium (1 MW in 300 s)" },
  { value: "fast", label: "Fast (1 MW in 150 s)" },
  { value: "ultra-fast", label: "Ultra-fast (1 MW in 75 s)" },
];

/** What the two-zone model leaves out (specification section 8), shown with every result. */
const ZONE_LIMITS =
  "Pre-flashover correlations (MQH) for rooms of ordinary size; one fire; doors open; no ceiling jet, radiant heat, " +
  "toxic gases or visibility beyond the smoke layer's height; smoke spreads one way, outward from the fire room; " +
  "lower layers at ambient. It ranks routes and deadlines under these assumptions, and is not a substitute for CFAST, " +
  "FDS or an engineer's assessment.";

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

/** The two-zone model's account of its fire, and its limits (specification section 8). */
function ZoneNotes({ zone, roomName }) {
  return (
    <>
      <span>
        The fire starts in {roomName || "a room"} ({zone.fireRoomArea.toFixed(0)} m², {zone.fireRoomHeight.toFixed(1)} m
        high) and grows to {Math.round(zone.Qmax).toLocaleString()} kW
        {zone.leakage
          ? ": the room has no door in the graph, so it burns on a nominal 0.1 m² leak"
          : zone.ventLimited
            ? `, as much as its doors (${zone.ventArea.toFixed(1)} m²) let it burn`
            : ""}
        .{" "}
        {zone.flashover !== null
          ? `It flashes over at ${secs(zone.flashover)}.`
          : "It does not flash over within the forecast."}{" "}
        Smoke reaches {zone.roomsReached} of {zone.rooms} rooms, and {zone.roomsUntenable} become untenable.
      </span>
      {zone.nodesInRooms < 0.999 && (
        <span>
          {Math.round(zone.nodesInRooms * 100)}% of the graph lies in a room; the rest (outside, or where the model has no
          IfcSpace) sees no fire.
        </span>
      )}
      <span className="field__hint">
        Times are seconds after ignition, and the dose is of the smoke layer&apos;s gas temperature (convective heat
        only). Limits: {ZONE_LIMITS}
      </span>
    </>
  );
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

function ForecastResult({ plan, settings, deadlinesShown, roomName }) {
  const { result, inputs } = plan;
  const { earliest, leastDose, leaveBy, snapshot, front, truncated, coarsened, mapStep, zone, coverage } = result;
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
      {coverage?.clipped && (
        <Notice tone="warning">
          The Flamescapes grid covered only the area round the fire ({Math.round(coverage.inGrid * 100)}% of the graph&apos;s
          nodes). Outside it this plan reads clear air, so routes and deadlines there are optimistic.
        </Notice>
      )}
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
        {zone && (
          <>
            <div className="result__item">
              <dt className="result__label">Flashover</dt>
              <dd className="result__value">{zone.flashover === null ? "not reached" : secs(zone.flashover)}</dd>
            </div>
            <div className="result__item">
              <dt className="result__label">Smoke reaches</dt>
              <dd className="result__value">
                {zone.roomsReached} of {zone.rooms} rooms
              </dd>
            </div>
          </>
        )}
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
        {zone ? (
          <ZoneNotes zone={zone} roomName={roomName} />
        ) : (
          <span className="field__hint">
            Times are counted from ignition, on the clock set for this plan ({inputs.forecastSecondsPerStep} s per fire
            step). The heat is the model&apos;s index, so the dose ranks routes under the model; it is not a measured
            dose.
          </span>
        )}
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
  const { graph, points, settings, setSettings, forecastPlan, fire, modelInfo, semantics } = studio;
  const modelExits = settings.forecastExits === "model" && exits?.nodes.length > 0;
  const ready = Boolean(graph && points.start && points.fire && (modelExits || points.exit));
  const zone = fireModelIn(settings, modelInfo) === "zone";
  // The fire room's IfcSpace name, from the model's own data (ifcSemantics.js).
  const fireRoomId = forecastPlan.result?.zone?.fireRoomId;
  const space = fireRoomId !== null && fireRoomId !== undefined ? semantics?.spaces?.find((s) => s.id === fireRoomId) : null;
  const roomName = space ? [space.longName, space.name].filter(Boolean).join(" ") || null : null;
  return (
    <Section title="Escape deadlines and forecast routes" badge={forecastPlan.result?.earliest ? "planned" : null}>
      <p className="field__hint">
        Runs the fire to its last step first, then routes with the state each point will be in when you would reach it,
        and works out how long you can wait before leaving.
        {zone && " The fire is the two-zone room model chosen above: gas temperatures in degrees C, on a clock in seconds."}
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
      {zone && (
        <Disclosure summary="The room fire" defaultOpen>
          <div className="grid-2">
            <SelectField
              label="Growth"
              value={settings.zoneGrowth}
              options={ZONE_GROWTH}
              onChange={(zoneGrowth) => setSettings({ zoneGrowth })}
            />
            <NumberField
              label="Peak heat release (kW)"
              value={settings.zonePeakKw}
              min={50}
              max={50000}
              step={250}
              onChange={(v) => setSettings({ zonePeakKw: v ?? 5000 })}
            />
          </div>
          <p className="field__hint">
            A t-squared fire in the room of the fire origin, capped at this peak or at what the room&apos;s doors let it
            burn, whichever is lower.
          </p>
          <Toggle
            label="Smoke at head height is untenable"
            checked={settings.zoneSmoke}
            onChange={(zoneSmoke) => setSettings({ zoneSmoke })}
            hint="A point closes once its room's smoke layer is down to 1.8 m, however hot; off, only from the gas temperature below."
          />
        </Disclosure>
      )}
      <Disclosure summary="Clock, speed and limits">
        <div className="grid-2">
          {zone ? (
            <NumberField
              label="Forecast length (s)"
              value={settings.zoneHorizon}
              min={60}
              max={1800}
              step={30}
              onChange={(v) => setSettings({ zoneHorizon: v ?? 600 })}
            />
          ) : (
            <NumberField
              label="Seconds per fire step"
              value={settings.forecastSecondsPerStep}
              min={0.5}
              max={60}
              step={0.5}
              onChange={(v) => setSettings({ forecastSecondsPerStep: v ?? 5 })}
              hint="The fire models count steps; this sets the clock."
            />
          )}
          <NumberField
            label="Walking speed (m/s)"
            value={settings.walkSpeed}
            min={0.2}
            max={3}
            step={0.1}
            onChange={(v) => setSettings({ walkSpeed: v ?? 1.2 })}
          />
          {zone ? (
            <NumberField
              label="Untenable from (gas °C)"
              value={settings.zoneUntenableC}
              min={30}
              max={200}
              step={5}
              onChange={(v) => setSettings({ zoneUntenableC: v ?? 60 })}
            />
          ) : (
            <NumberField
              label="Untenable from (model °C)"
              value={settings.untenableAbove}
              min={25}
              max={120}
              step={5}
              onChange={(v) => setSettings({ untenableAbove: v ?? 80 })}
            />
          )}
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
        <ForecastResult plan={forecastPlan} settings={settings} deadlinesShown={deadlinesShown} roomName={roomName} />
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

/** Progress and state of a Flamescapes run: backend, grid, time, heat release, embers. */
function CombustionStatus({ combustion, running }) {
  if (!combustion) return null;
  const done = combustion.time >= (combustion.horizon ?? Infinity) - 1e-6;
  return (
    <section className="result" aria-live="polite" aria-label="Flamescapes run">
      <h3 className="result__title" data-tone="ok">
        <Icon name={done ? "check" : "info"} size={16} />
        {done ? "Flamescapes run complete" : running ? "Flamescapes is computing" : "Flamescapes run"}
      </h3>
      <dl className="result__grid">
        <div className="result__item">
          <dt className="result__label">Simulated</dt>
          <dd className="result__value result__value--lead">
            {Math.round(combustion.time ?? 0)} / {Math.round(combustion.horizon ?? 0)} s
          </dd>
        </div>
        <div className="result__item">
          <dt className="result__label">Heat release</dt>
          <dd className="result__value">
            {Math.round(combustion.hrr ?? 0)} kW (peak {Math.round(combustion.peak ?? 0)} kW)
          </dd>
        </div>
        {combustion.openings && (
          <div className="result__item">
            <dt className="result__label">Openings</dt>
            <dd className="result__value">
              {combustion.openings.doorCount} doors {combustion.openings.doors}, {combustion.openings.windowCount} windows {combustion.openings.windows}
              {combustion.openings.ventilation === "running"
                ? `; ventilation running through ${combustion.openings.ductCells} of ${combustion.openings.terminals} air terminals`
                : combustion.openings.terminals
                  ? `; ventilation off (${combustion.openings.terminals} air terminals)`
                  : "; no air terminals in the model"}
            </dd>
          </div>
        )}
        {/* Off says so, so "0 thrown" never reads as embers that failed to fly. */}
        <div className="result__item">
          <dt className="result__label">Embers</dt>
          <dd className="result__value">
            {combustion.embersOn ? `on: ${combustion.embers ?? 0} thrown, ${combustion.spots ?? 0} spot fires` : "off"}
          </dd>
        </div>
        {combustion.grid && (
          <div className="result__item">
            <dt className="result__label">Grid</dt>
            <dd className="result__value">
              {combustion.grid.join(" x ")} of {combustion.cell?.toFixed(2)} m
            </dd>
          </div>
        )}
      </dl>
      {combustion.backend && (
        <p className="result__note">
          Running on {combustion.backend.kind === "gpu" ? "the GPU (WebGPU)" : "one CPU core"}: {combustion.backend.name}.
          {combustion.resized
            ? " The GPU stepped the first grid slower than three times real time, so the fire started again on a coarser grid it keeps up with."
            : combustion.restarted
              ? " The GPU stopped partway, so the fire started again on the CPU."
              : ""}
          {combustion.clipped ? " The model is larger than the grid allows, so the grid covers the area round the fire; nodes outside it read as clear air." : ""}
        </p>
      )}
    </section>
  );
}

/** Likely ignition points: the review's scenarios matched in this model, ranked. */
function IgnitionPoints({ ignition, onFind, onUse, disabled }) {
  const { busy, candidates, occupancy } = ignition || {};
  const pct = (x) => (x == null ? "unknown" : `${(x * 100).toFixed(x < 0.01 ? 1 : 0)}%`);
  return (
    <Disclosure summary="Likely ignition points">
      <p className="field__hint">
        Where a fire most often starts, matched in this model by class, type, name or room, and ranked by each
        cause&apos;s share of fire deaths (residential: US homes 2019-2023). Sources in the Methods page and{" "}
        <code>wiki/research/ignition-sources-2026-11-05.md</code>.
      </p>
      <SelectField
        label="Occupancy"
        value={occupancy || "residential"}
        onChange={(value) => onFind(value)}
        options={[
          { value: "residential", label: "Residential" },
          { value: "non-residential", label: "Non-residential" },
        ]}
      />
      <Button block busy={busy} disabled={disabled} onClick={() => onFind(occupancy)}>
        Find in this model
      </Button>
      {candidates && !candidates.length && (
        <p className="field__hint">
          No appliance, furniture, heater or named room was found to suggest an origin. Pick one on the graph.
        </p>
      )}
      {candidates?.length > 0 && (
        <ul className="ignition-list">
          {candidates.map((c) => (
            <li key={`${c.byRoom ? "r" : "e"}${c.itemId}`} className="ignition-list__item">
              <div>
                <strong>{c.scenario.label}</strong>{" "}
                <span className="pill">Class {c.scenario.fireClass}</span>
                <div className="field__hint">
                  {c.byRoom ? "Room" : "Element"}: {c.name || "unnamed"}. {pct(c.weight)} of fire deaths
                  {c.estimated ? " (estimated)" : ""}; match {Math.round(c.confidence * 100)}%.
                </div>
              </div>
              <Button size="sm" onClick={() => onUse(c)}>
                Start here
              </Button>
            </li>
          ))}
        </ul>
      )}
    </Disclosure>
  );
}

export function SimulatePanel({
  studio,
  exits,
  onPick,
  onPlace,
  ignition,
  onFindIgnition,
  onUseIgnition,
  onStartFire,
  onStopFire,
  onTrainRl,
  onPlanForecast,
  deadlinesShown,
  onGoTo,
  onExportTopologic,
}) {
  const { graph, points, pickMode, settings, setSettings, fire, dynamicPath, rl, modelInfo } = studio;
  const ready = Boolean(graph);
  const rooms = hasRooms(modelInfo);
  const fireModel = fireModelIn(settings, modelInfo);
  // Heat diffusion, the two-zone model and Flamescapes give temperatures to reroute on.
  const flamescapes = isFlamescapes(fireModel);
  const shielded = settings.rlMethod !== "qlearning";
  const heat = fireModel === "temperature" || fireModel === "zone" || flamescapes;
  const fireClass = FIRE_CLASSES[settings.flamescapesClass] || FIRE_CLASSES.A;
  const models = [...FIRE_MODELS, ...(rooms ? [ZONE_MODEL] : []), ...(settings.flamescapes ? FLAMESCAPES_MODELS : [])];

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
        {ready && <IgnitionPoints ignition={ignition} onFind={onFindIgnition} onUse={onUseIgnition} disabled={fire.running} />}

        <Choices
          legend="How the fire spreads"
          name="fire-model"
          value={fireModel}
          onChange={(fireModel) => setSettings({ fireModel })}
          options={models}
          disabled={fire.running}
        />
        {fire.running && <p className="field__hint">Stop the fire to change how it spreads.</p>}
        <Toggle
          label="Flamescapes combustion solver (optional)"
          checked={settings.flamescapes}
          disabled={fire.running}
          onChange={(on) => setSettings(on ? { flamescapes: true, fireModel: "flamescapes" } : { flamescapes: false, fireModel: isFlamescapes(settings.fireModel) ? "temperature" : settings.fireModel })}
          hint="Adds two fire models from the Flamescapes paper (Wrede, Wagner et al. 2026): fire and smoke, with or without embers. Heavier than the others: it runs on your GPU where the browser offers WebGPU, otherwise on one CPU core, for a minute or more."
        />
        {flamescapes && (
          <>
            <SelectField
              label="What burns first"
              value={settings.flamescapesClass}
              onChange={(flamescapesClass) => setSettings({ flamescapesClass })}
              options={FIRE_CLASS_KEYS.map((key) => ({ value: key, label: FIRE_CLASSES[key].label }))}
              hint={`${fireClass.examples}. ISO 3941 class: ${fireClass.iso}.`}
            />
            <Notice tone={fireClass.agents.unsuitable.length ? "warning" : "neutral"}>
              <strong>Extinguishing a class {fireClass.short} fire.</strong> Suitable: {fireClass.agents.suitable.join(", ")}.
              {fireClass.agents.unsuitable.map((entry) => (
                <span key={entry.agent}>
                  {" "}
                  Never {entry.agent.toLowerCase()}: {entry.reason}
                </span>
              ))}
            </Notice>
            {/* Open or closed doors change the fire more than any other setting: here, not folded away. */}
            <div className="field">
              <div className="field__label">
                <span>Doors</span>
              </div>
              {/* Never locked: a run that is playing keeps its doors, and the
                  switch sets the next run's, with a restart offered at once. */}
              <Segmented
                label="Doors"
                value={settings.flamescapesDoors}
                onChange={(flamescapesDoors) => setSettings({ flamescapesDoors })}
                options={[
                  { value: "open", label: "Open" },
                  { value: "closed", label: "Closed" },
                ]}
              />
              {fire.running && fire.doors && fire.doors !== settings.flamescapesDoors && (
                <Notice tone="neutral">
                  This fire is running with the doors {fire.doors}.{" "}
                  <Button size="sm" onClick={onStartFire}>
                    Restart the fire with the doors {settings.flamescapesDoors}
                  </Button>
                </Notice>
              )}
              <p className="field__hint">
                {settings.flamescapesDoors === "closed"
                  ? "Each door's leaf fills its opening, so the smoke stays in the rooms it starts in until it finds another way. Routes still pass through every door."
                  : "Smoke and heat pass through every doorway. Close them to see what shut doors hold back; routes use the doors either way."}
              </p>
            </div>
            <Segmented
              label="Show on the graph"
              value={settings.flamescapesShow}
              onChange={(flamescapesShow) => setSettings({ flamescapesShow })}
              options={[
                { value: "heat", label: "Heat" },
                { value: "smoke", label: "Smoke" },
              ]}
            />
          </>
        )}
        {modelInfo && !rooms && (
          <p className="field__hint">
            Two-zone rooms (degrees C) needs rooms: the model must have IfcSpace elements, and this one has none.
          </p>
        )}
        {fireModel === "zone" && (
          <p className="field__hint">
            The rooms&apos; fire is computed first and played one step per {secs(settings.zoneHorizon / Math.max(1, settings.fireMaxSteps))};
            set its growth and peak under Escape deadlines below.
          </p>
        )}

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

        {flamescapes && <CombustionStatus combustion={fire.combustion} running={fire.running} />}
        {dynamicPath && <RerouteResult dynamicPath={dynamicPath} step={fire.step} />}
        {ready && fire.step > 0 && !fire.running && <ExportTopologic onExport={onExportTopologic} withRun={flamescapes} />}
        {flamescapes && (
          <Disclosure summary="Flamescapes settings">
            <div className="grid-2">
              <NumberField
                label="Simulated time (s)"
                value={settings.flamescapesHorizon}
                min={30}
                max={900}
                step={30}
                onChange={(flamescapesHorizon) => setSettings({ flamescapesHorizon: flamescapesHorizon ?? 180 })}
              />
              <NumberField
                label="Fire load (MJ/m²)"
                value={settings.flamescapesFireLoad}
                min={50}
                max={3000}
                step={10}
                onChange={(flamescapesFireLoad) => setSettings({ flamescapesFireLoad: flamescapesFireLoad ?? 780 })}
                hint="Room contents per floor area; 780 is the Eurocode mean for dwellings."
              />
              <NumberField
                label="Untenable from (gas °C)"
                value={settings.flamescapesUntenableC}
                min={30}
                max={200}
                step={5}
                onChange={(flamescapesUntenableC) => setSettings({ flamescapesUntenableC: flamescapesUntenableC ?? 60 })}
              />
              <NumberField
                label="Untenable below visibility (m)"
                value={settings.flamescapesVisibility}
                min={1}
                max={30}
                step={1}
                onChange={(flamescapesVisibility) => setSettings({ flamescapesVisibility: flamescapesVisibility ?? 10 })}
              />
            </div>
            <Toggle
              label="Smoke makes a place untenable"
              checked={settings.flamescapesSmoke}
              onChange={(flamescapesSmoke) => setSettings({ flamescapesSmoke })}
              hint="Visibility below the limit closes a node for the plan, as gas above the temperature does."
            />
            <SelectField
              label="Run on"
              value={settings.flamescapesBackend}
              onChange={(flamescapesBackend) => setSettings({ flamescapesBackend })}
              options={[
                { value: "auto", label: "The GPU where the browser offers WebGPU, else the CPU" },
                { value: "cpu", label: "The CPU (one core, coarser grid)" },
              ]}
            />
            <SelectField
              label="Windows"
              value={settings.flamescapesWindows}
              onChange={(flamescapesWindows) => setSettings({ flamescapesWindows })}
              options={[
                { value: "closed", label: "Closed" },
                { value: "open", label: "Open: smoke vents through the window openings" },
              ]}
            />
            <SelectField
              label="Ventilation"
              value={settings.flamescapesVentilation}
              onChange={(flamescapesVentilation) => setSettings({ flamescapesVentilation })}
              options={[
                { value: "off", label: "Off: fans stopped and dampers shut, as a fire alarm should do" },
                { value: "running", label: "Running: the ducts carry the air between the model's air terminals" },
              ]}
            />
            {settings.flamescapesVentilation === "running" && (
              <NumberField
                label="Air through each terminal (L/s)"
                value={settings.flamescapesTerminalFlow}
                min={10}
                max={1000}
                step={10}
                onChange={(flamescapesTerminalFlow) => setSettings({ flamescapesTerminalFlow: flamescapesTerminalFlow ?? 100 })}
              />
            )}
            <p className="field__hint">
              For the smoke only: a route still goes through every door. Closed doors and windows do not leak or burn. Running ventilation joins
              every air terminal in the model (IfcAirTerminal) through one well-mixed duct volume, the case that spreads smoke furthest: which
              system a terminal belongs to is not read, and 100 L/s a terminal is an assumption, not a measured value.
            </p>
            <SelectField
              label="Grid on the GPU"
              value={settings.flamescapesGrid}
              onChange={(flamescapesGrid) => setSettings({ flamescapesGrid })}
              options={[
                { value: "speed", label: "Sized to keep three times real time on this GPU" },
                { value: "finest", label: "The finest this GPU's memory allows (slower)" },
              ]}
            />
            <p className="field__hint">
              The fire depends on the grid: on the Duplex a sofa fire peaked at 0.09 MW on a 0.20 m grid, 2.5 MW on 0.23 m and 6.1 MW on
              0.27 m. Keep one grid when comparing runs; the run card gives it. This browser remembers the GPU&apos;s speed, so the first
              setting picks the same grid each time on the same GPU.
            </p>
            <p className="field__hint">{FLAMESCAPES_LIMITS}</p>
          </Disclosure>
        )}

        <Disclosure summary="Rerouting and timing">
          <Toggle
            label="Recompute the route as the fire grows"
            checked={settings.reroute}
            onChange={(reroute) => setSettings({ reroute })}
            hint={
              heat
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
            disabled={!settings.reroute || !heat}
            hint="Heat diffusion and two-zone rooms only. Link cost is length x (1 + weight x degrees above ambient / 100); zero ignores the fire."
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
            optional
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
        <div className="field">
          <div className="field__label">
            <span>Method</span>
          </div>
          <Segmented
            label="Learning method"
            value={settings.rlMethod}
            onChange={(rlMethod) => setSettings({ rlMethod })}
            options={[
              { value: "shielded", label: "Shielded (proved)" },
              { value: "qlearning", label: "Q-learning" },
            ]}
          />
          <p className="field__hint">
            {shielded
              ? "Learns the escape on (node, time) behind a shield that never lets it step where the fire will seal it in, and stops when it can prove its route is the earliest escape. Both properties are proved in Lean."
              : "Tabular Q-learning over the same graph, with the spreading fire as a penalty (after Jabi et al. 2019). It is drawn by a walk that never revisits a node, and on the study's corpus it missed 15% of the escapes that existed."}{" "}
            Studio first checks whether any escape exists, so a failed agent and an impossible escape are told apart.
          </p>
        </div>
        {shielded && (
          <div className="field">
            <div className="field__label">
              <span>Escape by</span>
            </div>
            <Segmented
              label="Escape by"
              value={settings.rlExits}
              onChange={(rlExits) => setSettings({ rlExits })}
              options={[
                { value: "any", label: "Any exit door" },
                { value: "chosen", label: "The chosen exit" },
              ]}
            />
          </div>
        )}
        <div className="grid-2">
          {!shielded && (
            <NumberField
              label="Episodes"
              value={settings.rlEpisodes}
              min={10}
              step={50}
              onChange={(rlEpisodes) => setSettings({ rlEpisodes: rlEpisodes ?? 200 })}
            />
          )}
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
          hint="1 = the fire advances one link per move, as the evacuee does. With the fire on or beside the route at that speed, the study's runs had no escape in 70 of 72; the result says when."
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
            {rl.result?.method === "shielded" ? (
              <>
                {rl.result.escape_steps != null && rl.result.moves != null && (
                  <Stat label="Moves vs earliest escape" value={`${rl.result.moves} / ${rl.result.escape_steps}`} warn={rl.result.moves > rl.result.escape_steps} />
                )}
                <Stat label="Certified earliest" value={rl.result.certified ? "yes" : "no"} warn={!rl.result.certified && rl.reachedExit} />
                <Stat label="Trials" value={fmt(rl.result.episodes)} />
              </>
            ) : (
              <Stat label="Late-training success" value={`${Math.round((rl.result?.success_rate ?? 0) * 100)}%`} />
            )}
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
                value={rl.result.escape_steps == null ? "no, within the step limit" : `yes, ${rl.result.escape_steps} moves`}
                warn={rl.result.escape_steps == null}
              />
            )}
          </div>
        )}
        {rl.path && rl.result?.method === "shielded" && (
          <p className="field__hint">
            Certified earliest: the learner proved, from its own bounds, that no route leaves sooner. Moves vs earliest
            escape: its route against the earliest escape found by search; they match whenever it is certified.
          </p>
        )}
        {rl.path && rl.result?.method !== "shielded" && (
          <p className="field__hint">
            Late-training success: share of the last fifth of training episodes, with exploration, that reached the
            exit. Return vs best within the step limit: the learned policy&apos;s discounted return, and the most any
            policy can collect within the step limit (backward induction).
          </p>
        )}
      </Section>
    </>
  );
}
