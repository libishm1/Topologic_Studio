/**
 * Inspector panels, one per workflow step: Model, Graph, Route, Simulate.
 *
 * Each step opens with its one primary action, puts the result right under
 * it, and folds the settings most people leave alone. Explanations of a
 * result (a graph in several parts, fewer alternatives than asked) sit beside
 * it as notices rather than in toasts. Controls that cannot do anything yet
 * are disabled with a reason, so the order of the steps stays visible.
 */
import React, { useEffect, useId, useRef, useState } from "react";

import { graphIsSplit, reachablePct } from "../core/graph.js";
import { RL_MAX_STEPS } from "../core/rl.js";
import { formatBytes, formatMs } from "../lib/perf.js";
import { STANDARD_CELLS } from "../lib/hardware.js";
import { CROWD_COLOURS } from "../viewer/crowd.js";
import { FORECAST_COLORS, routeColor } from "../lib/routeColors.js";
import { PLAN_SETTINGS, fireModelIn, hasRooms, isFlamescapes } from "../state/useStudio.js";
import { FIRE_CLASSES, FIRE_CLASS_KEYS, designFireFor, waterResponse } from "../core/fireClasses.js";
import { GAS_MEASURES, SHOW_GROUPS, SMOKE_MEASURES } from "../core/measures.js";
import { IGNITION_CONTEXTS } from "../core/ignitionContexts.js";
import { DOOR_MODES, doorsLabel } from "../core/doorStates.js";
import { MEASURED_PRE, NO_RESPONSE_SOURCE, RSET_CRITERIA, RSET_PRESETS, criterionOf, measuredOf, presetOf } from "../core/rsetPresets.js";
import { worstStatus } from "../core/perfReport.js";
import { TURNSTILE_SITE_KEY, mountHumanCheck, unmountHumanCheck } from "../lib/turnstile.js";
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
function PartsNotice({ stats, brief = false }) {
  if (!stats || stats.components < 2) return null;
  if (brief) {
    return (
      <Notice tone="warning">
        In {stats.components} parts; the largest holds {reachablePct(stats)}%. A route joins points in the same part only (why: Graph step).
      </Notice>
    );
  }
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
/** The "I'm human" check (Cloudflare Turnstile, lib/turnstile.js): one slim strip while it is shown. */
function HumanCheck({ theme }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    let alive = true;
    // Its script may load after the panel has gone: then it is removed at once.
    mountHumanCheck(el, { theme: theme === "dark" ? "dark" : "light" })
      .then(() => alive || unmountHumanCheck(el))
      .catch(() => {});
    return () => {
      alive = false;
      unmountHumanCheck(el);
    };
  }, [theme]);
  return <div ref={ref} className="human-check" />;
}

function DescribeModel({ describe, onDescribe, onStop, onEngine, onOpen, theme }) {
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
    <Disclosure summary="Describe this model" onToggle={onOpen}>
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
              ? `The facts below are sent to the site's describer on Cloudflare Workers AI, which says it neither keeps nor trains on them; the IFC itself is not sent.${
                  TURNSTILE_SITE_KEY ? " Cloudflare's check that you are human comes first; it rarely needs a click." : ""
                }`
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
      {engine === "cloud" && TURNSTILE_SITE_KEY && <HumanCheck theme={theme} />}
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
  onDescribeOpen,
  onOpenRooms,
  onExportRooms,
  linked,
  onAddLinked,
  onAlignLinked,
  onRemoveLinked,
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
            <DescribeModel describe={describe} onDescribe={onDescribe} onStop={onStopDescribe} onEngine={onDescribeEngine} onOpen={onDescribeOpen} theme={studio.theme} />
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

      {modelInfo && !rooms && linked && (
        <LinkedModels linked={linked} onAdd={onAddLinked} onAlign={onAlignLinked} onRemove={onRemoveLinked} />
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
function ExportTopologic({ onExport, disabled, withRun, onExportTable }) {
  return (
    <div className="field">
      <Button block disabled={disabled} onClick={onExport}>
        Export for TopologicPy (JSON)
      </Button>
      <p className="field__hint">
        The walkable graph in the IFC&apos;s own coordinates, with the route
        {withRun
          ? " and, per node, the fire's readings over time (gas temperature, visibility, CO, and soot, CO₂ and O₂ when recorded), the tenability index, its doses and when it reached 1"
          : ""}
        , as the JSON TopologicPy reads with <code>Graph.ByJSONFile</code>.
      </p>
      {withRun && onExportTable && (
        <>
          <Button block onClick={onExportTable}>
            Export the run as a table (CSV)
          </Button>
          <p className="field__hint">
            One row per node per frame: place, kind, time, every reading and measure, the doses and the tenability index. Gzipped; ready for
            pandas or a machine-learning pipeline.
          </p>
        </>
      )}
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

const FEDERATION_WORDS = { aligned: ["ok", "Aligned"], check: ["warn", "Check"], offset: ["bad", "Offset"], "no geometry": ["bad", "No geometry"] };
const CHECK_TONES = { ok: "ok", warning: "warn", note: "plan", "cannot tell": "plan" };

/**
 * Second models beside the first (core/federation.js): an MEP model's
 * sprinklers and ducts, a structural model's beams. Each with the checks of
 * the MEP report's section 6.4, what it brings, and a way to move it onto
 * the first when their plans miss.
 */
function LinkedModels({ linked, onAdd, onAlign, onRemove }) {
  const { list, busy } = linked;
  return (
    <Section title="Second model (MEP, structure)" defaultOpen={false} badge={list.length ? `${list.length} added` : null}>
      <p className="field__hint">
        An MEP or structural IFC of the same building. It is drawn in this model&apos;s frame and checked against it; its sprinklers, ducts and beams are
        used by the sprinklers in step 4. The graph and the fire&apos;s walls stay this model&apos;s.
      </p>
      <FileButton accept=".ifc" onFile={onAdd} disabled={Boolean(busy)}>
        {busy ? `Adding ${busy}...` : "Add a model"}
      </FileButton>
      {list.map((m) => {
        const [tone, word] = FEDERATION_WORDS[m.check.status] || ["plan", m.check.status];
        const brings = [
          m.sprinklers && `${m.sprinklers} sprinklers`,
          m.beams && `${m.beams} beams`,
          m.ducts && `${m.ducts} ducts`,
          m.cables && `${m.cables} cable trays`,
          m.flowSegments && `${m.flowSegments.toLocaleString()} flow segments and fittings (ducts and trays among them by name)`,
        ].filter(Boolean);
        return (
          <div key={m.id} className="linked-model">
            <div className="summary__row">
              <span className="linked-model__name" title={m.name}>
                <strong>{m.name}</strong>
              </span>
              <span className={`pill pill--${tone}`}>{word}</span>
            </div>
            <p className="field__hint">
              Placed by {m.how}. {brings.length ? `Brings ${brings.join(", ")}.` : "Nothing the sprinklers use."}
            </p>
            <ul className="code-list">
              {m.check.checks.map((c) => (
                <li key={c.id} data-tone={CHECK_TONES[c.status] || "plan"}>
                  {c.text}
                </li>
              ))}
            </ul>
            <div className="button-row">
              {m.check.offset && (
                <Button size="sm" onClick={() => onAlign(m.id)}>
                  Move it onto this model
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={() => onRemove(m.id)}>
                Remove
              </Button>
            </div>
          </div>
        );
      })}
    </Section>
  );
}

const EXTINGUISHER_UNITS = [
  { value: "abc-2.3", label: "ABC dry chemical, 2.3 kg (2-A:10-B:C), 14 s" },
  { value: "abc-4.5", label: "ABC dry chemical, 4.5 kg (4-A:80-B:C), 20 s" },
  { value: "water-9.5", label: "Water, 9.5 L (2-A), 60 s" },
  { value: "foam-9.5", label: "Foam, 9.5 L (3-A:20-B), 50 s" },
  { value: "co2-2", label: "CO2, 2 kg (34B), 8.5 s" },
];
const OUTCOME_WORDS = { success: ["ok", "Put out"], "cannot tell": ["plan", "Cannot tell"], failure: ["bad", "Not put out"] };
const kwText = (x) => (x >= 1000 ? `${(x / 1000).toFixed(2)} MW` : `${Math.round(x)} kW`);

/** The extinguishers placed or read, and the placement check (core/extinguishers.js). */
function ExtinguisherLayout({ plan, busy, onRedo }) {
  if (busy) return <p className="field__hint">Placing the extinguishers…</p>;
  if (!plan) return null;
  const check = plan.check;
  const where = plan.source === "ifc" ? "read from the IFC" : plan.source === "placed" ? "placed on the walk by the travel rule" : "none found in the IFC";
  const failing = (check?.storeys || []).filter((s) => s.status === "fail");
  return (
    <div className="sprinkler-layout">
      <Notice tone={check?.status === "fail" ? "warning" : "neutral"}>
        <strong>{plan.sites.length} extinguishers</strong>, {where}
        {plan.note ? ` (${plan.note})` : ""}.
        {check && ` The check: ${check.status}. Every place within ${check.limit} m of walking to one (${check.passWithin?.toFixed?.(2) ?? check.passWithin} m on this graph, as its nodes sample the floor).`}
      </Notice>
      {failing.length > 0 && (
        <ul className="code-list">
          {failing.slice(0, 6).map((s, i) => (
            <li key={i} className="code-list__storey" data-tone="warn">
              <span className="pill">Fail</span> <strong>{s.name || `Storey ${i + 1}`}</strong>{" "}
              <span className="field__hint">
                {(s.checks || [])
                  .filter((c) => c.status !== "pass")
                  .map((c) => c.text)
                  .join("; ")}
                .
              </span>
            </li>
          ))}
        </ul>
      )}
      <Button size="sm" onClick={onRedo}>
        Place again
      </Button>
    </div>
  );
}

/** One person's attempt with an extinguisher on the fire on show (core/extinguishers.js attempt). */
function ExtinguisherAttempt({ attempt }) {
  if (!attempt) return null;
  const [tone, word] = OUTCOME_WORDS[attempt.outcome] || ["plan", attempt.outcome];
  const t = attempt.times || {};
  const s = (x) => (Number.isFinite(x) ? `${Math.round(x)} s` : "-");
  return (
    <div className="linked-model">
      <div className="summary__row">
        <strong>The attempt</strong>
        <span className={`pill pill--${tone}`}>{word}</span>
      </div>
      {attempt.attempted !== false && Number.isFinite(t.tS) && (
        <p className="field__hint">
          Sets off at {s(attempt.tGo)}, reaches the extinguisher at {s(attempt.tGo + (t.toExtinguisher || 0))}, readies it, and starts at {s(t.tS)}
          {Number.isFinite(attempt.hrr) ? ` against ${kwText(attempt.hrr)}` : ""}
          {attempt.capacity?.units === "kW"
            ? `; the unit puts out up to ${kwText(attempt.capacity.low)} for sure and nothing above ${kwText(attempt.capacity.high)}`
            : attempt.capacity?.units === "m²"
              ? `; the unit puts out up to ${attempt.capacity.low.toFixed(2)} m² of burning liquid for sure, nothing above ${attempt.capacity.high.toFixed(2)} m²`
              : ""}
          .
          {Number.isFinite(attempt.timeLost) ? ` Time lost against leaving at once: ${s(attempt.timeLost)}.` : ""}
        </p>
      )}
      {attempt.late && <p className="field__hint">The run ended at {s(attempt.horizon)}, before the attack: its last heat release is used.</p>}
      <ul className="code-list">
        {(attempt.reasons || []).map((r, i) => (
          <li key={i} data-tone={r.status === "pass" ? "ok" : r.status === "fail" ? "warn" : "plan"}>
            {r.text.replace(/ at t_s\b/g, " when the attack starts").replace(/\bt_s\b/g, "the attack's start")}
          </li>
        ))}
      </ul>
      <p className="field__hint">The fire: {attempt.source}. Its size a rating can put out is uncertain (0.2 to 0.45 MW per A unit, and the operator), so the answer is put out, not put out, or cannot tell between.</p>
    </div>
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
      {graphIsSplit(graph.stats) && <PartsNotice stats={graph.stats} brief />}
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
 * widths; high-rise. For offices, the stricter of SBC 801-CR-2018 (Chapter 10)
 * and IBC 2021 for each limit; SBC 2024 to replace the 2018 values.
 */
function CodeCheck({ code, settings, setSettings, onRun, onShow, ready }) {
  const result = code?.result;
  const tone = { pass: "ok", fail: "warn" };
  const mark = { pass: "Pass", fail: "Fail", "cannot tell": "Cannot tell", "n/a": "Not applicable" };
  return (
    <Section title="Code check" defaultOpen={false}>
      <p className="field__hint">
        Offices (occupancy group B). Each limit is the stricter of the Saudi Fire Code SBC 801-CR-2018 (Chapter 10, means of egress) and the
        International Building Code 2021, so a pass meets both; SBC 2024 replaces the 2018 values when its text is available, and the high-rise
        provisions are the IBC&apos;s until SBC 201 is read. Distances are walked on the graph to the nearest exit out of the building, or, above and
        below it, to the nearest stair; the occupant load uses the walkable area, a lower bound.
      </p>
      <Toggle
        label="The building has automatic sprinklers"
        checked={settings.codeSprinklered}
        onChange={(codeSprinklered) => setSettings({ codeSprinklered })}
        hint="Sprinklers allow longer walks (90 m against 60 m) and exits closer together (1/3 of the diagonal against 1/2)."
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

/**
 * The performance-based ("simulated fire") check (core/performanceCheck.js):
 * the forecast's escape deadlines against detection, alarm and pre-movement,
 * per storey. Evidence for an alternative method, beside the code check.
 */
const SPRINKLER_MODES = {
  activation: "they only open",
  evans: "they cap the fuel (Evans 1993)",
  spray: "they cap the fuel and spray (Fire-X's evaporation)",
  wetting: "they spray and wet the fuel (calibrated to Evans at 0.07 mm/s)",
};
const SPRINKLER_SOURCES = { ifc: "from the IFC", rooms: "placed per room (NFPA 13)", storeys: "on a lattice over the floor (no rooms)" };

/** A run's sprinklers in a line: heads, when the first opened by both readings, how many opened, and the water. */
function sprinklerLine(live, summary) {
  const s = (x) => `${Math.round(x)} s`;
  const kw = (x) => (x >= 1000 ? `${(x / 1000).toFixed(1)} MW` : `${Math.round(x)} kW`);
  const heads = `${live.heads} heads ${SPRINKLER_SOURCES[live.source] || ""}; ${SPRINKLER_MODES[live.mode] || live.mode}`;
  if (!summary) return `${heads}; ${live.open ? `${live.open} open, the first at ${s(live.first)}` : "none open yet"}`;
  const first = (key, hrr) => {
    const opened = summary.heads.filter((h) => h[key] !== null);
    if (!opened.length) return null;
    const head = opened.reduce((a, b) => (b[key] < a[key] ? b : a));
    return `${s(head[key])} (${kw(head[hrr])})`;
  };
  const estimate = first("estimate", "hrrEstimate"), late = first("late", "hrrLate");
  const offGrid = summary.heads.filter((h) => !h.onGrid).length;
  const water = summary.stats?.emitted > 0
    ? `; ${summary.stats.emitted.toFixed(0)} kg of water${summary.stats.evaporated !== null && summary.stats.evaporated !== undefined ? `, ${summary.stats.evaporated.toFixed(1)} kg evaporated` : ""}`
    : "";
  const opened = `${summary.open.length} opened (on the ${summary.reading === "late" ? "late bound" : "estimate"})`;
  return `${heads}; the first opens at ${estimate || "none"} by the estimate, ${late || "none"} by the late bound; ${opened}${summary.covered ? `, ${summary.covered} fuel cells under them` : ""}${water}${offGrid ? `; ${offGrid} outside the fire grid` : ""}`;
}

/**
 * A run's suppression systems by room use in a line: per room whose
 * detectors saw smoke, when the system was released and acted; a release
 * due after the run's end says so.
 */
function systemsLine(live, summary, horizon) {
  const s = (x) => (x === null || x === undefined ? "never" : `${Math.round(x)} s`);
  if (!summary) return live ? `${live.rooms} room${live.rooms === 1 ? "" : "s"} with a clean agent or foam; ${live.detected} detected, ${live.discharged} released so far` : "";
  const seen = summary.filter((r) => r.detectedAt !== null || r.manual);
  const parts = seen.map((r) => {
    const name = r.name || `Room ${r.id}`;
    const after = r.dischargeAt !== null && r.dischargeAt > (horizon ?? Infinity);
    const smoke = r.manual ? "released by hand" : `smoke seen at ${s(r.detectedAt)}`;
    if (after) return `${name}: ${r.system === "foam" ? "foam" : "clean agent"}, ${smoke}, release due at ${s(r.dischargeAt)}, after the run's end`;
    if (r.system === "clean-agent") return `${name}: clean agent, ${smoke}, released at ${s(r.dischargeAt)}, flames stop from ${s(r.reachesA)} (${r.extinguishing.A} %), ${r.design} % at ${s(r.floodedAt)}`;
    return `${name}: foam, ${smoke}, on from ${s(r.dischargeAt)}, a pool controlled by ${s(r.controlAt)} and out by ${s(r.outAt)}`;
  });
  const quiet = summary.length - seen.length;
  return [`${summary.length} room${summary.length === 1 ? "" : "s"} with a clean agent or foam`, ...parts, quiet ? `${quiet} saw no smoke` : null].filter(Boolean).join("; ");
}

/** The share of occupants started by t seconds after the alarm, design and measured, with their 99th percentiles. */
function PreMovementChart({ curves, measuredLabel }) {
  const W = 300, H = 128;
  const pad = { l: 30, r: 10, t: 8, b: 26 };
  const x = (t) => pad.l + (t / curves.tMax) * (W - pad.l - pad.r);
  const y = (p) => H - pad.b - p * (H - pad.t - pad.b);
  const path = (pts) => pts.map(([t, p], i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(p).toFixed(1)}`).join("");
  // Ticks every round number of seconds, four or five of them.
  const unit = [30, 60, 120, 300, 600, 1200, 1800, 3600].find((u) => curves.tMax / u <= 5) || 7200;
  const ticks = Array.from({ length: Math.floor(curves.tMax / unit) + 1 }, (_, i) => i * unit);
  const label = `Share of occupants who have started, by seconds after the alarm: design log-normal, 99th percentile ${Math.round(curves.p99)} s${
    curves.measured ? `; measured Weibull, 99th percentile ${Math.round(curves.measuredP99)} s` : ""
  }.`;
  return (
    <figure className="perf-chart" style={{ margin: "var(--space-2) 0" }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={{ width: "100%", height: "auto", display: "block" }}>
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line x1={pad.l} x2={W - pad.r} y1={y(p)} y2={y(p)} stroke="currentColor" strokeOpacity={p ? 0.15 : 0.5} />
            <text x={pad.l - 4} y={y(p) + 3} textAnchor="end" fontSize="9" fill="currentColor" fillOpacity="0.7">
              {p * 100} %
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - pad.b + 11} textAnchor="middle" fontSize="9" fill="currentColor" fillOpacity="0.7">
            {t}
          </text>
        ))}
        <text x={(pad.l + W - pad.r) / 2} y={H - 3} textAnchor="middle" fontSize="9" fill="currentColor" fillOpacity="0.7">
          seconds after the alarm
        </text>
        <line x1={x(curves.p99)} x2={x(curves.p99)} y1={pad.t} y2={y(0)} stroke="var(--viz-path)" strokeOpacity="0.6" strokeDasharray="2 2" />
        <path d={path(curves.design)} fill="none" stroke="var(--viz-path)" strokeWidth="2" />
        {curves.measured && (
          <>
            <line x1={x(curves.measuredP99)} x2={x(curves.measuredP99)} y1={pad.t} y2={y(0)} stroke="var(--viz-fire)" strokeOpacity="0.6" strokeDasharray="2 2" />
            <path d={path(curves.measured)} fill="none" stroke="var(--viz-fire)" strokeWidth="2" strokeDasharray="6 3" />
          </>
        )}
      </svg>
      <figcaption className="field__hint">
        <span style={{ color: "var(--viz-path)" }}>Solid</span>: the design log-normal (99th percentile {Math.round(curves.p99)} s).
        {curves.measured && (
          <>
            {" "}
            <span style={{ color: "var(--viz-fire)" }}>Dashed</span>: {measuredLabel}, measured (99th percentile {Math.round(curves.measuredP99)} s).
          </>
        )}{" "}
        Dotted lines mark the 99th percentiles; a curve that levels off below 100 % is the share who never respond.
      </figcaption>
    </figure>
  );
}

function PerformanceCheck({ perf, settings, setSettings, onRun, ready, onGoTo, report, batch, canAddFires, onAddFires, onStopFires, onDownloadReport, onClearReport }) {
  const tone = { pass: "ok", fail: "warn" };
  const mark = { pass: "Pass", fail: "Fail", "cannot tell": "Cannot tell" };
  const preset = presetOf(settings.perfOccupancy);
  const criterion = criterionOf(settings.perfCriterion);
  const measured = measuredOf(settings.perfMeasured);
  const choose = (id) => {
    const p = presetOf(id);
    setSettings({ perfOccupancy: id, ...(p.values || {}) });
  };
  // Typing a preset's value makes it custom: the values no longer are the preset's.
  const set = (key) => (v) => setSettings({ [key]: Math.max(0, v ?? 0), perfOccupancy: "custom" });
  const badPre = !(settings.perfPre1 > 0 && settings.perfPre99 > settings.perfPre1);
  const badArea = settings.perfQueues && !(settings.perfOccupantArea > 0);
  const detectedBy = {
    ceiling: `from the run's ${perf?.detectors || ""} ceiling detectors`.replace("  ", " "),
    run: "from the run's smoke at head height (a run kept from before the ceiling detectors)",
    "not in the run": "as set: the run's smoke never reached a detector's threshold",
    "no run": "as set: only a Flamescapes run gives smoke to detect",
    set: "as set",
  };
  const seconds = (x) => (x === -Infinity ? "no safe time" : Number.isFinite(x) ? `${Math.round(x)} s` : "");
  const status = report.length ? worstStatus(report.flatMap((e) => e.cases.map((c) => c.result?.status ?? "cannot tell"))) : null;
  return (
    <Section title="Performance check (with the fire)" defaultOpen={false}>
      <p className="field__hint">
        Performance-based: each place passes when someone there can wait for detection, the alarm, the 99th percentile of pre-movement
        time and any queue at the doors, and still get out with a dose under the limit, with the margin the criterion asks. It reads the
        escape-deadline map of the forecast (Simulate step, Plan with the forecast); every check joins a report of fires. Evidence for an
        alternative method (SBC 801-CR-2018 104.9) beside the code check, not a replacement for it.
      </p>
      <SelectField label="Occupants" value={settings.perfOccupancy} onChange={choose} options={RSET_PRESETS.map((p) => ({ value: p.id, label: p.label }))} hint={preset.source} />
      <Toggle
        label="Detection from the fire run's smoke"
        checked={settings.perfDetectionFromRun}
        onChange={(perfDetectionFromRun) => setSettings({ perfDetectionFromRun })}
        hint="With a Flamescapes fire: when the smoke reaches a smoke detector's optical density, 0.097 per metre, at virtual detectors under the ceiling 7.0 m from the fire, or as far as its room allows (NZ C/VM2 Table 3.2.2.2). Otherwise the time below."
      />
      <div className="grid-2">
        <NumberField label="Detection (s)" value={settings.perfDetection} min={0} max={1800} step={5} onChange={(v) => setSettings({ perfDetection: Math.max(0, v ?? 0) })} />
        <NumberField label="Alarm after it (s)" value={settings.perfAlarm} min={0} max={1800} step={5} onChange={set("perfAlarm")} />
      </div>
      <div className="grid-2">
        <NumberField label="Pre-movement, 1st percentile (s)" value={settings.perfPre1} min={1} max={3600} step={5} onChange={set("perfPre1")} />
        <NumberField label="Pre-movement, 99th percentile (s)" value={settings.perfPre99} min={2} max={7200} step={5} onChange={set("perfPre99")} />
      </div>
      <NumberField label="Never respond (%)" value={settings.perfNoResponse} min={0} max={50} step={1} onChange={set("perfNoResponse")} hint={NO_RESPONSE_SOURCE} />
      <SelectField
        label="Measured pre-movement, shown beside"
        value={measured ? measured.id : "none"}
        onChange={(perfMeasured) => setSettings({ perfMeasured })}
        options={[...MEASURED_PRE.map((m) => ({ value: m.id, label: m.label })), { value: "none", label: "None" }]}
        hint={measured ? `${measured.source} Judged the same way and reported beside the design values, not in the verdict.` : "Only the design values."}
      />
      <Toggle
        label="Queues at doors"
        checked={settings.perfQueues}
        onChange={(perfQueues) => setSettings({ perfQueues })}
        hint="Occupants on every place walk to the nearest exit and queue at each door on the way: 1.32 persons a second per metre of the door's width less 0.15 m a side (NZ C/VM2 3.5.3). Widths from the IFC; 0.813 m where it gives none."
      />
      {settings.perfQueues && (
        <NumberField label="Floor area per occupant (m²)" value={settings.perfOccupantArea} min={0.3} max={100} step={0.5} onChange={set("perfOccupantArea")} hint={preset.load} />
      )}
      <Toggle
        label="Also with the exit nearest the fire blocked"
        checked={settings.perfBlockedExit}
        onChange={(perfBlockedExit) => setSettings({ perfBlockedExit })}
        hint="Planning makes a second deadline map without that exit, about doubling the map's time; judged as a sensitivity case (SFEG 2025 test 5). Plan again after switching it on."
      />
      <SelectField
        label="Criterion"
        value={criterion.id}
        onChange={(perfCriterion) => setSettings({ perfCriterion })}
        options={RSET_CRITERIA.map((c) => ({ value: c.id, label: c.label }))}
        hint={`${criterion.source} Sensitivity cases are judged at 1.2.`}
      />
      {badPre && <Notice tone="warning">The 99th percentile must be longer than the 1st, and both above zero.</Notice>}
      {badArea && <Notice tone="warning">Set the floor area per occupant above zero.</Notice>}
      <Button block disabled={!ready || badPre || badArea} onClick={onRun}>
        Check against the fire
      </Button>
      {!ready && (
        <p className="field__hint">
          Plan with the forecast first, with the escape-deadline map on.{" "}
          <Button size="sm" variant="ghost" onClick={() => onGoTo?.("simulate")}>
            Go to Simulate
          </Button>
        </p>
      )}
      {perf && (
        <>
          <Notice tone={perf.status === "pass" ? "neutral" : "warning"}>
            <strong>{mark[perf.status]}.</strong> Occupants must be able to set off up to {Math.round(perf.start)} s after ignition
            {perf.factor > 1 ? " beside an exit, and later farther away" : ""}: detection at {Math.round(perf.detection)} s,{" "}
            {detectedBy[perf.detectionFrom] || "as set"}; the alarm at {Math.round(perf.cue)} s; then pre-movement
            {perf.queues ? " and the queues at doors" : ""}. {perf.criterion?.label}.
          </Notice>
          <ul className="code-list">
            {perf.storeys.map((s) => (
              <li key={s.index} className="code-list__storey" data-tone={tone[s.status] || "neutral"}>
                <span className="pill">{mark[s.status]}</span> <strong>{s.name}</strong> <span className="field__hint">{s.text}.</span>
                {s.worstWait !== null && (
                  <span className="field__hint">
                    {" "}
                    Queues: {Math.round(s.occupants)} occupants; the last to start waits up to {Math.round(s.worstWait)} s at the doors.
                  </span>
                )}
                {s.measured && (
                  <span className="field__hint">
                    {" "}
                    With the measured spread: {mark[s.measured.status].toLowerCase()}
                    {s.measured.fail ? `, ${s.measured.fail} of ${s.judged} places short` : ""}; {(100 * s.measured.share).toFixed(1)} % start in time.
                  </span>
                )}
              </li>
            ))}
          </ul>
          {perf.queues?.busiest && perf.queues.busiest.wait > 0 && (
            <p className="field__hint">
              The busiest door passes {Math.round(perf.queues.busiest.persons)} people at {perf.queues.busiest.capacity.toFixed(2)} a second and holds them up to{" "}
              {Math.round(perf.queues.busiest.wait)} s; its queue clears {Math.round(perf.queues.busiest.clears)} s after ignition.
            </p>
          )}
          <PreMovementChart curves={perf.curves} measuredLabel={perf.measured?.label || ""} />
          {perf.cases?.length > 1 && (
            <ul className="code-list">
              {perf.cases.slice(1).map((c) => (
                <li key={c.id} className="code-list__storey" data-tone={tone[c.result?.status] || "neutral"}>
                  <span className="pill">{mark[c.result?.status ?? "cannot tell"]}</span> <strong>{c.label}</strong>{" "}
                  <span className="field__hint">
                    {c.result ? `${c.criterion}; detection at ${Math.round(c.detection)} s; the worst storey: ${c.result.storeys.find((s) => s.status === c.result.status)?.text ?? ""}.` : `${c.note}.`}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {perf.blockedAsked && !perf.cases?.some((c) => c.id === "exit-blocked") && (
            <p className="field__hint">The case with an exit blocked needs a plan made with that switch on and at least two exits: plan again.</p>
          )}
          <p className="field__hint">
            ASET from the forecast (ISO 13571 dose against the limit set under Clock, speed and limits), RSET as detection, alarm,
            pre-movement (a log-normal through the two percentiles, PD 7974-6), the walk with no fire and any queue. The margin is ASET less
            RSET at the 99th percentile. One walking speed; time in a queue is added to RSET as in a hand calculation, without its own dose;
            the room of fire origin is judged like any other place.
          </p>
        </>
      )}
      {(report.length > 0 || canAddFires) && (
        <div className="perf-report">
          <h4 className="choices__legend">Report: {report.length} fire{report.length === 1 ? "" : "s"}{status ? `, ${mark[status].toLowerCase()} overall` : ""}</h4>
          {report.length > 0 && (
            <ul className="code-list">
              {report.map((e, i) => (
                <li key={i} className="code-list__storey" data-tone={tone[worstStatus(e.cases.map((c) => c.result?.status ?? "cannot tell"))] || "neutral"}>
                  <strong>{e.label}</strong>{" "}
                  <span className="field__hint">
                    {e.cases
                      .map((c) => {
                        const worst = c.result?.storeys.reduce((m, s) => Math.min(m, s.worstMargin), Infinity);
                        return `${c.id === "base" ? "base case" : c.id === "exit-blocked" ? "exit blocked" : "detector fails"}: ${mark[c.result?.status ?? "cannot tell"].toLowerCase()}${
                          c.result ? ` (worst margin ${seconds(worst)})` : ""
                        }`;
                      })
                      .join("; ")}
                    .
                  </span>
                </li>
              ))}
            </ul>
          )}
          {batch ? (
            <div className="grid-2">
              <p className="field__hint">
                Planning fire {Math.min(batch.done + 1, batch.total)} of {batch.total}…
              </p>
              <Button size="sm" variant="ghost" onClick={onStopFires}>
                Stop
              </Button>
            </div>
          ) : (
            <Button size="sm" block disabled={!canAddFires || badPre || badArea} onClick={onAddFires}>
              Add fires on every storey
            </Button>
          )}
          <p className="field__hint">
            {canAddFires
              ? "Two fires a storey, one next to an exit and one farthest from the exits (at most eight), each planned with its own deadline map and checked with the settings above."
              : "With a Flamescapes fire a run takes minutes: move the fire, plan again and check; each fire joins the report."}
          </p>
          {report.length > 0 && (
            <div className="grid-2">
              <Button size="sm" onClick={onDownloadReport}>
                Download the report
              </Button>
              <Button size="sm" variant="ghost" onClick={onClearReport}>
                Clear the report
              </Button>
            </div>
          )}
        </div>
      )}
    </Section>
  );
}

/**
 * People leaving (core/crowd.js): the switch, the look and the playback
 * speed, the clock and what the walk came to.
 */
function PeopleLeaving({ crowd, settings, setSettings, onPlay, onRestart, ready }) {
  const result = crowd?.result;
  // How many turned to another exit on the way (the quickest choice).
  const turned = result?.switches ? result.switches.reduce((sum, s) => sum + (s > 0 ? 1 : 0), 0) : 0;
  const swatch = (key, label) => (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4, marginRight: 10 }}>
      <span className="picker__dot" style={{ background: `#${CROWD_COLOURS[key].toString(16).padStart(6, "0")}` }} />
      {label}
    </span>
  );
  return (
    <Section title="People leaving" badge={result ? `${result.people.toLocaleString()} people` : null} defaultOpen={false}>
      <Toggle
        label="Show people leaving"
        checked={settings.crowd}
        disabled={!ready}
        onChange={(on) => setSettings({ crowd: on })}
        hint="Everyone on the floors walks out by the exit that gets them out soonest and gathers outside it, played faster than real time."
      />
      {settings.crowd && (
        <>
          <div className="grid-2">
            <SelectField
              label="Show them as"
              value={settings.crowdStyle}
              onChange={(crowdStyle) => setSettings({ crowdStyle })}
              options={[
                { value: "figures", label: "Figures" },
                { value: "dots", label: "Dots" },
              ]}
            />
            <SelectField
              label="Playback"
              value={String(settings.crowdRate)}
              onChange={(value) => setSettings({ crowdRate: Number(value) })}
              options={[1, 5, 10, 30].map((r) => ({ value: String(r), label: r === 1 ? "Real time" : `${r} times real time` }))}
            />
          </div>
          <SelectField
            label="Egress"
            value={settings.crowdWithFire ? "fire" : "none"}
            onChange={(value) => setSettings({ crowdWithFire: value === "fire" })}
            options={[
              { value: "none", label: "Without the fire" },
              { value: "fire", label: "With the fire" },
            ]}
            hint={
              !settings.crowdWithFire
                ? "Everyone walks out as if there were no fire."
                : !crowd?.fireActive
                  ? "They wait for the fire: start it in Simulate and they set off with it."
                  : crowd?.avoiding
                    ? "They keep the fire's time and keep out of places it has made untenable (as the plan reads them): they turn back for another exit, and anyone with no way out but through the fire stays where they are."
                    : "They keep the fire's time; they keep out of it once its run has finished computing, so play it again to see them turn back."
            }
          />
          <SelectField
            label="Each person takes"
            value={settings.crowdChoice}
            onChange={(crowdChoice) => setSettings({ crowdChoice })}
            options={[
              { value: "quickest", label: "The quickest exit, queues counted" },
              { value: "nearest", label: "The nearest exit" },
            ]}
            hint={
              settings.crowdChoice === "quickest"
                ? "Each weighs every exit out of the building, the walk there against the queue ahead of them at that door, and turns to a free door where the ways part when it is clearly quicker."
                : "Each walks to the nearest exit and waits in its queue, even while another door stands free."
            }
          />
          <Toggle
            label="A runner in an orange shirt"
            checked={settings.crowdStairRunner}
            onChange={(crowdStairRunner) => setSettings({ crowdStairRunner })}
          />
          {settings.crowdStairRunner && (
            <SelectField
              label="The runner"
              value={settings.crowdRunner}
              onChange={(crowdRunner) => setSettings({ crowdRunner })}
              options={[
                { value: "stair", label: "Down a stair, on a loop" },
                { value: "forecast", label: "The forecast's way out, on its timeline" },
              ]}
              hint={
                settings.crowdRunner === "forecast"
                  ? crowd?.planned
                    ? "The forecast's evacuee on its earliest safe route, reaching each place when the plan says: with the fire on, at the fire's time."
                    : "Plan with the forecast in Simulate first; until then the runner takes a stair."
                  : crowd?.result && !crowd.runner
                    ? "No walk here goes down a stair."
                    : "One of the walks down a stair, looped in real time at that walk's own speeds."
              }
            />
          )}
          <Toggle
            label="Full gathering places clear"
            checked={settings.crowdShelterOn}
            onChange={(crowdShelterOn) => setSettings({ crowdShelterOn })}
            hint="A gathering place holds only what its area takes, about 0.64 m² a person standing; once it is full, the earliest there fade away as newcomers arrive."
          />
          {settings.crowdShelterOn && (
            <SelectField
              label="Each gathering place"
              value={String(settings.crowdShelterArea)}
              onChange={(value) => setSettings({ crowdShelterArea: Number(value) })}
              options={[10, 25, 50, 100].map((a) => ({ value: String(a), label: `${a} m², about ${Math.floor(a / 0.636)} people` }))}
            />
          )}
          {crowd?.busy && <p className="field__hint">Working out everyone&apos;s walk…</p>}
          {crowd?.error && <Notice tone="warning">{crowd.error}</Notice>}
          {result && (
            <>
              {!settings.crowdWithFire && (
                <div className="grid-2">
                  <Button size="sm" onClick={onPlay}>
                    {crowd.playing ? "Pause" : "Play"}
                  </Button>
                  <Button size="sm" onClick={onRestart}>
                    From the start
                  </Button>
                </div>
              )}
              <p className="field__hint" aria-live="off">
                {Math.round(crowd.time)} s of {Math.ceil(result.lastOut)} s
                {settings.crowdWithFire ? ", on the fire's clock" : settings.crowdRate > 1 ? `, at ${settings.crowdRate} times real time` : ""}.
              </p>
              <p className="field__hint">
                {swatch("waiting", "not yet off")}
                {swatch("walking", "walking")}
                {swatch("held", "held up")}
                {swatch("out", "out")}
                {result.inside > 0 && swatch("inside", "still inside")}
              </p>
              <Notice>
                {result.people.toLocaleString()} people, one per {settings.perfOccupantArea} m² of walkable floor
                {result.capped ? " (capped)" : ""}: the last is out at {Math.round(result.lastOut)} s
                {result.inside > 0 ? `, ${result.inside} still inside when the walk stopped` : ""}, by{" "}
                {result.exits.map((e) => `${e.people}`).join(", ")} through {result.exits.length} exit{result.exits.length === 1 ? "" : "s"}; the longest hold-up{" "}
                {Math.round(result.longestWait)} s.
                {turned > 0 ? ` ${turned.toLocaleString()} turned to another exit on the way.` : ""}
                {result.tooLarge ? " The graph is too large to weigh every exit for everyone, so each took the nearest." : ""}
                {result.unreached > 0 ? ` ${result.unreached.toLocaleString()} walkable places reach no exit and hold no one.` : ""}
                {Number.isFinite(result.shelter) ? ` Each gathering place holds ${result.shelter}; ${Array.from(result.goneAt).filter(Number.isFinite).length} leave a full one.` : ""}
                {result.cutOff > 0 ? ` ${result.cutOff} cut off by the fire stay where they are.` : ""}
              </Notice>
              {crowd.gpu && <p className="field__hint">Moved and drawn on the GPU: {crowd.gpu}.</p>}
              <p className="field__hint">
                {result.choice === "quickest"
                  ? "Each person takes the exit that gets them out soonest: the walk there, or the people ahead of them through it at the flow seen at its door, whichever is longer, looked at again every 2 s, as Pathfinder's “locally quickest” door choice weighs a door. "
                  : "Each person heads for the nearest exit by the shortest walk (the “nearest exit” choice that gave the shortest times in Carrera et al. 2019). "}
                Each holds one place of the graph at a time, as a cell of a floor-field model does
                (Burstedde et al. 2001), stepping round whoever is ahead. Speeds spread about 1.19 m/s and slow as the crowd
                thickens, 0.92 m/s at most on stairs, and a 0.9 m door passes about 0.8 people a second (SFPE Handbook). Everyone
                sets off within 10 s of the plan&apos;s departure time: the performance check, not this animation, models how
                long people take to start. An illustration of the walk out, not an evacuation model to quote.
              </p>
            </>
          )}
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
  perf,
  perfReady,
  onPerformanceCheck,
  perfReport = [],
  perfBatch = null,
  perfCanAddFires = false,
  onAddPerfFires,
  onStopPerfFires,
  onDownloadPerfReport,
  onClearPerfReport,
  crowd = null,
  onPlayCrowd,
  onRestartCrowd,
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
                {exits.finalSource &&
                  ` The nearest is one of the ${exits.final.length} that lead out of the building (${exits.finalSource}).`}
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
          <Button block busy={pathBusy} disabled={!canRoute || engines.length < 2} onClick={onCompare}>
            Compare both engines
          </Button>
          <Button block disabled={!points.start && !points.exit && !points.fire} onClick={onClearPoints}>
            Clear all points
          </Button>
        </Disclosure>
      </div>

      <Section title="Alternative routes" badge={alternatives ? `${alternatives.routes.length} found` : null}>
        <p className="field__hint">
          Is there a second way out? <strong>Different space</strong>: routes that share at most half their length
          with an earlier one, up to 1.5 times the shortest. <strong>Yen</strong>: the next cheapest routes exactly, often the
          best route with a step moved.
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
      <PeopleLeaving crowd={crowd} settings={settings} setSettings={setSettings} onPlay={onPlayCrowd} onRestart={onRestartCrowd} ready={Boolean(graph)} />
      <CodeCheck code={code} settings={settings} setSettings={setSettings} onRun={onCodeCheck} onShow={onShowTravel} ready={Boolean(graph)} />
      <PerformanceCheck
        perf={perf}
        settings={settings}
        setSettings={setSettings}
        onRun={onPerformanceCheck}
        ready={perfReady}
        onGoTo={onGoTo}
        report={perfReport}
        batch={perfBatch}
        canAddFires={perfCanAddFires}
        onAddFires={onAddPerfFires}
        onStopFires={onStopPerfFires}
        onDownloadReport={onDownloadPerfReport}
        onClearReport={onClearPerfReport}
      />
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
  "do not burn; doors and windows as set above. The flow is incompressible, so heated gas takes no extra volume " +
  "and smoke may reach head height later than it would; the grid's sides are open to outside air, so a large " +
  "model clipped round the fire is never fully closed. It shows how heat and smoke may move through this " +
  "building; it is not a calibrated fire engineering model.";

const ZONE_GROWTH = [
  { value: "slow", label: "Slow (1 MW in 600 s)" },
  { value: "medium", label: "Medium (1 MW in 300 s)" },
  { value: "fast", label: "Fast (1 MW in 150 s)" },
  { value: "ultra-fast", label: "Ultra-fast (1 MW in 75 s)" },
];

/** How the Flamescapes first item grows (fireClasses.js designFireFor). */
const itemGrowthOptions = (cls) => [
  { value: "class", label: `As its fire class (${cls.growth?.class ?? "its own burning"})` },
  ...ZONE_GROWTH,
  { value: "solver", label: "As the solver burns it (before 2026-11-08)" },
];

/** A design-fire first item on the run card: its curve, then what water or agent did to it. */
/** What grid a run was built on, for its card's note (flamescapes.worker.js describe). */
function gridNote(c) {
  if (c.policy === "speed") return " A finer grid sized to this GPU's speed: its fire differs from the standard grid's, which every device builds by default.";
  if (c.policy === "finest") return " The finest grid this GPU's memory allows: its fire differs from the standard grid's, which every device builds by default.";
  if (c.policy === "standard" && c.budget < STANDARD_CELLS)
    return ` This device's CPU builds a lighter grid (at most ${c.budget.toLocaleString()} cells) to keep ahead of real time: its fire can differ from the standard grid's.`;
  return c.policy === "standard" ? " The standard grid, the same on every device." : "";
}

/** The range over a fire's repeat runs (core/forecast.js runRange), for its card. */
function repeatsLine({ of, status, tenability: t, range }) {
  const span = (s, f) => (!s ? "none" : s.min === s.max ? f(s.median) : `${f(s.median)} (${f(s.min)} to ${f(s.max)})`);
  const runs =
    status === "running" ? `${range.runs} of ${of} runs so far` : status === "stopped" ? `${range.runs} of ${of} runs (the rest stopped)` : `${range.runs} runs`;
  const first = range.firstUntenable ? span(range.firstUntenable, (s) => `${Math.round(s)} s`) : "never";
  const never = range.firstUntenable && range.noneUntenable ? `, none in ${range.noneUntenable} of them` : "";
  return (
    `${runs}, the median and its range: ${span(range.untenable, Math.round)} nodes untenable, the first at ${first}${never}; ` +
    `peak ${span(range.peakKw, (kw) => `${(kw / 1000).toFixed(2)} MW`)}; ${span(range.releasedMJ, (mj) => `${mj.toFixed(1)} MJ`)} released. ` +
    `Untenable at ${t.threshold} °C${t.smoke ? `, below ${t.visibilityLimit} m visibility` : ""} or by dose, as the plan reads it.`
  );
}

function itemLine(design, summary, horizon) {
  const base = `${/^[aeiou]/i.test(design.growth) ? "an" : "a"} ${design.growth} t-squared fire (${design.alpha} kW/s²), ${design.mass} kg`;
  if (!summary) return base;
  const stopped = summary.stoppedAt != null && summary.stoppedAt <= (horizon ?? Infinity);
  const what = stopped
    ? `the clean agent put it out at ${Math.round(summary.stoppedAt)} s`
    : summary.suppressed
      ? `${summary.suppressed === "foam" ? "foam-water" : "sprinkler water"} on it from ${Math.round(summary.suppressedAt)} s, decaying as Evans measured`
      : "no water reached it";
  return `${base}; ${what}; it released ${(summary.released / 1e6).toFixed(1)} MJ`;
}

/** What the first item does, for the growth choice's hint. */
function itemGrowthHint(clsKey, growth) {
  const cls = FIRE_CLASSES[clsKey] || FIRE_CLASSES.A;
  const fire = designFireFor(clsKey, growth);
  if (growth === "solver") return "The solver lights and burns the item itself: about 500 kW within 15 s in the examples, faster than any design fire, so sprinklers open when the fire is already large.";
  if (!fire) return `A class ${cls.short} item keeps its own burning (a pool, a metal or a battery pack is not a t-squared item).`;
  const energy = cls.item.mass * cls.fuel.heatOfCombustion;
  const end = Math.cbrt((3 * energy) / (fire.alpha * 1000));
  return (
    `${/^[aeiou]/i.test(fire.growth) ? "An" : "A"} ${fire.growth} t-squared fire, ${fire.alpha} kW/s²: the ${cls.item.mass} kg item (${Math.round(energy / 1e6)} MJ) would reach ${Math.round(fire.alpha * end * end)} kW ` +
    `when it burns out at ${Math.round(end)} s. The room's contents catch from it. Sprinkler water over it decays it as Evans measured; clean agent at the class A concentration puts it out.`
  );
}

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
        {!unsafe && (
          <div className="result__item">
            <dt className="result__label">Exit</dt>
            <dd className="result__value">{dynamicPath.otherExit ? "another exit" : "the one picked"}</dd>
          </div>
        )}
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
  const modelExits = settings.forecastExits === "model" && exits?.final.length > 0;
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
      {exits?.final.length > 0 && (
        <Choices
          legend="Exits"
          name="forecast-exits"
          value={settings.forecastExits}
          onChange={(forecastExits) => setSettings({ forecastExits })}
          options={[
            {
              value: "model",
              label: `All ${exits.final.length} exit${exits.final.length === 1 ? "" : "s"} out of the building`,
              description: `From ${exits.finalSource || exitSourceText(exits.source)}.`,
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
      <Button block busy={forecastPlan.busy} disabled={!ready || fire.running || fire.starting} onClick={onPlan}>
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
      {combustion.replayed && (
        <p className="result__note">
          Played from this browser&apos;s kept runs (computed {new Date(combustion.replayed).toLocaleString()}), not computed again.
        </p>
      )}
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
        <div className="result__item">
          <dt className="result__label">Measures</dt>
          <dd className="result__value">
            {combustion.extras ? "heat, smoke, CO, CO₂, O₂" : combustion.extrasDropped ? "heat, smoke, CO (the rest would not fit in memory)" : "heat, smoke, CO"}
          </dd>
        </div>
        {combustion.designFire && (
          <div className="result__item">
            <dt className="result__label">First item</dt>
            <dd className="result__value">{itemLine(combustion.designFire, combustion.itemSummary, combustion.horizon)}</dd>
          </div>
        )}
        {combustion.sprinklers && (
          <div className="result__item">
            <dt className="result__label">Sprinklers</dt>
            <dd className="result__value">
              {sprinklerLine(combustion.sprinklers, combustion.sprinklerSummary)}
              <span className="field__hint">
                {" "}
                Against measured tests the estimate opens the first head at 0.8 to 1.3 of the measured time, about a quarter late in a small
                closed room. With the first item as a design fire, heads in rooms of ordinary height open at tens to a few hundred kW. The
                late bound is not a time to quote.
              </span>
            </dd>
          </div>
        )}
        {(combustion.systems || combustion.systemsSummary) && (
          <div className="result__item">
            <dt className="result__label">Systems</dt>
            <dd className="result__value">{systemsLine(combustion.systems, combustion.systemsSummary, combustion.horizon)}</dd>
          </div>
        )}
        {combustion.detectors > 0 && (
          <div className="result__item">
            <dt className="result__label">Smoke detectors</dt>
            <dd className="result__value">
              {combustion.detectors} under the ceiling, 7 m from the fire or as far as its room allows:{" "}
              {combustion.detectedAt !== null && combustion.detectedAt !== undefined ? `the first responds at ${Math.round(combustion.detectedAt)} s` : "none has responded"}
            </dd>
          </div>
        )}
        {combustion.grid && (
          <div className="result__item">
            <dt className="result__label">Grid</dt>
            <dd className="result__value">
              {combustion.grid.join(" x ")} of {combustion.cell?.toFixed(2)} m
            </dd>
          </div>
        )}
        {combustion.repeats && (
          <div className="result__item">
            <dt className="result__label">Repeat runs</dt>
            <dd className="result__value">{repeatsLine(combustion.repeats)}</dd>
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
          {gridNote(combustion)}
          {combustion.clipped
            ? " The model is larger than the grid allows, so the grid covers the area round the fire; its sides are open to outside air, so smoke that reaches them leaves the run even where they cut through the building, and nodes outside it read as clear air."
            : ""}
        </p>
      )}
    </section>
  );
}

/** Likely ignition points: the review's scenarios matched in this model, ranked. */
function IgnitionPoints({ ignition, onFind, onUse, disabled }) {
  const { busy, candidates, occupancy, context } = ignition || {};
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
        onChange={(value) => onFind(value, context)}
        options={[
          { value: "residential", label: "Residential" },
          { value: "non-residential", label: "Non-residential" },
        ]}
      />
      <SelectField
        label="Cause"
        value={context || "any"}
        onChange={(value) => onFind(occupancy, value)}
        options={IGNITION_CONTEXTS.map((c) => ({ value: c.id, label: c.label }))}
        hint="Narrows the list to one cause; a gas fire starts as class B, whatever the appliance."
      />
      <Button block busy={busy} disabled={disabled} onClick={() => onFind(occupancy, context)}>
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

/** The sprinkler layout in brief: how many heads and from where, and the rooms whose rules fail or cannot be told. */
function SprinklerLayout({ plan, busy, onRedo }) {
  const mark = { pass: "Pass", fail: "Fail", "cannot tell": "Cannot tell", required: "Required", "not required": "Not required" };
  const tone = { pass: "ok", fail: "warn", required: "warn" };
  if (busy) return <p className="field__hint">Laying out the sprinklers…</p>;
  if (!plan) return null;
  // Rooms laid out, judged by their placement rules; obstructions (beams,
  // ducts) only when a structural or MEP model gave them (step 1), else the
  // check cannot be told for every room alike.
  const all = plan.rooms || [];
  const rooms = all.filter((r) => r.status !== "n/a");
  const skipped = all.length - rooms.length;
  const placement = (r) => r.placement ?? r.status;
  const failing = rooms.filter((r) => placement(r) === "fail");
  const open = rooms.filter((r) => placement(r) === "cannot tell");
  const where =
    plan.source === "ifc"
      ? "read from the IFC"
      : plan.source === "rooms"
        ? `placed by NFPA 13 in ${rooms.length} rooms`
        : plan.source === "storeys"
          ? "placed on a lattice over the walkable floor (the model has no rooms)"
          : "none";
  return (
    <div className="sprinkler-layout">
      <Notice tone={failing.length ? "warning" : "neutral"}>
        <strong>{plan.heads.length} sprinkler heads</strong>, {where}
        {plan.note ? ` (${plan.note})` : ""}.
        {rooms.length > 0 && ` ${rooms.length - failing.length - open.length} meet every placement rule, ${failing.length} fail one, ${open.length} cannot be told.`}
        {skipped > 0 && ` ${skipped} spaces not laid out (shafts, spaces drawn over smaller rooms, or outside the building: roofs, terraces, balconies).`}
        {rooms.length > 0 && !plan.obstructionCount && " Obstructions (beams, ducts) are not checked without a structural or MEP model (step 1)."}
        {plan.obstructionCount > 0 && ` Checked against ${plan.obstructionCount} beams and ducts (NFPA 13's beam, three times and wide rules).`}
        {plan.obstructions &&
          ` Against ${plan.obstructions.notes?.[0] || "the beams and ducts"}: ${plan.obstructions.failing} heads fail the beam or three times rule${plan.obstructions.wide ? `, ${plan.obstructions.wide} obstructions wider than 1.2 m` : ""}.`}
      </Notice>
      {(plan.requirement || []).length > 0 && (
        <ul className="code-list">
          {plan.requirement.map((r, i) => (
            <li key={i} className="code-list__storey" data-tone={tone[r.status] || "neutral"}>
              <span className="pill">{mark[r.status] || r.status}</span> <span className="field__hint">{r.text}</span>
            </li>
          ))}
        </ul>
      )}
      {(plan.notes || []).filter((n) => /^obstructions:/.test(n)).map((n) => (
        <p key={n} className="field__hint">
          {n[0].toUpperCase() + n.slice(1)}.
        </p>
      ))}
      {failing.length + open.length > 0 && (
        <ul className="code-list">
          {[...failing, ...open].slice(0, 8).map((r) => (
            <li key={r.id} className="code-list__storey" data-tone={tone[r.status] || "neutral"}>
              <span className="pill">{mark[r.status] || r.status}</span> <strong>{r.name || `Room ${r.id}`}</strong>{" "}
              <span className="field__hint">
                {(r.checks || [])
                  .filter((c) => c.status !== "pass" && (c.id !== "obstructions" || plan.obstructionCount > 0))
                  .map((c) => c.text)
                  .join("; ")}
                .
              </span>
            </li>
          ))}
          {failing.length + open.length > 8 && <li className="field__hint">and {failing.length + open.length - 8} more rooms.</li>}
        </ul>
      )}
      <Button size="sm" onClick={onRedo}>
        Lay out again
      </Button>
    </div>
  );
}

const SYSTEM_WORDS = { water: "Water", "clean-agent": "Clean agent", foam: "Foam", none: "None" };
const UNIT_WORDS = { "co2-2": "carbon dioxide", "foam-9.5": "class B foam", "wet-chemical-6": "class K wet chemical" };

/** Suppression systems by room use in brief: the counts, the clean-agent and foam rooms and why, and their own extinguishers. */
function SystemsPlan({ plan, busy, onRedo }) {
  if (busy) return <p className="field__hint">Assigning the systems…</p>;
  if (!plan) return null;
  if (plan.noRooms) {
    return <Notice tone="warning">The model has no rooms (IfcSpace). Systems are assigned by room name, so none can be given here.</Notice>;
  }
  const special = plan.rooms.filter((r) => r.system === "clean-agent" || r.system === "foam");
  const units = {};
  for (const u of plan.units || []) units[u.unit] = (units[u.unit] || 0) + 1;
  return (
    <div className="sprinkler-layout">
      <Notice tone="neutral">
        {plan.lines.map((line) => `${line[0].toUpperCase()}${line.slice(1)}.`).join(" ")}
      </Notice>
      {special.length > 0 && (
        <ul className="code-list">
          {special.slice(0, 12).map((r, i) => (
            <li key={`${r.id}-${i}`} className="code-list__storey" data-tone="neutral">
              <span className="pill">{SYSTEM_WORDS[r.system]}</span> <strong>{r.name || `Room ${r.id}`}</strong>{" "}
              <span className="field__hint">
                {r.reason}
                {r.agentKg ? `; ${r.agentKg.toFixed(1)} kg of agent` : ""} ({r.clause.split(":")[0]}).
              </span>
            </li>
          ))}
          {special.length > 12 && <li className="field__hint">and {special.length - 12} more rooms.</li>}
        </ul>
      )}
      {Object.keys(units).length > 0 && (
        <p className="field__hint">
          Their own extinguishers, at the door (dark, amber and grey in the view):{" "}
          {Object.entries(units)
            .map(([u, n]) => `${UNIT_WORDS[u] || u} at ${n} room${n === 1 ? "" : "s"}`)
            .join(", ")}
          ; ABC powder on the routes is in Extinguishers.
        </p>
      )}
      <Button size="sm" onClick={onRedo}>
        Assign again
      </Button>
    </div>
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
  onExportTable,
  onForgetRuns,
  sprinklerPlan = null,
  sprinklerBusy = false,
  onLayOutSprinklers,
  overheadCount = null,
  extinguisherPlan = null,
  extinguisherBusy = false,
  onLayOutExtinguishers,
  extinguisherAttempt = null,
  onExtinguisherAttempt,
  systemsPlan = null,
  systemsBusy = false,
  onLayOutSystems,
}) {
  const { graph, points, pickMode, settings, setSettings, fire, dynamicPath, rl, modelInfo, semantics } = studio;
  // Running, or starting: the two-zone model computes its frames first.
  const fireBusy = fire.running || Boolean(fire.starting);
  const doorsNow = doorsLabel(settings.flamescapesDoors, { seed: settings.flamescapesDoorSeed, share: settings.flamescapesDoorShare });
  const selfClosingCount = (semantics?.doors || []).filter((d) => d.selfClosing).length;
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
        {ready && <IgnitionPoints ignition={ignition} onFind={onFindIgnition} onUse={onUseIgnition} disabled={fireBusy} />}

        <Choices
          legend="How the fire spreads"
          name="fire-model"
          value={fireModel}
          onChange={(fireModel) => setSettings({ fireModel })}
          options={models}
          disabled={fireBusy}
        />
        {fireBusy && <p className="field__hint">Stop the fire to change how it spreads.</p>}
        <Toggle
          label="Flamescapes combustion solver (optional)"
          checked={settings.flamescapes}
          disabled={fireBusy}
          onChange={(on) => setSettings(on ? { flamescapes: true, fireModel: "flamescapes" } : { flamescapes: false, fireModel: isFlamescapes(settings.fireModel) ? "temperature" : settings.fireModel })}
          hint="Fire and smoke from the Flamescapes paper (Wrede, Wagner et al. 2026), with or without embers. Heavier: on your GPU where the browser offers WebGPU, otherwise a minute or more on the CPU."
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
            <SelectField
              label="How it grows"
              value={settings.flamescapesGrowth}
              onChange={(flamescapesGrowth) => setSettings({ flamescapesGrowth })}
              options={itemGrowthOptions(fireClass)}
              hint={itemGrowthHint(settings.flamescapesClass, settings.flamescapesGrowth)}
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
              <Segmented label="Doors" value={settings.flamescapesDoors} onChange={(flamescapesDoors) => setSettings({ flamescapesDoors })} options={DOOR_MODES} />
              {settings.flamescapesDoors === "random" && (
                <div className="grid-2">
                  <NumberField
                    label="Closed (%)"
                    value={Math.round(settings.flamescapesDoorShare * 100)}
                    min={0}
                    max={100}
                    step={5}
                    onChange={(v) => setSettings({ flamescapesDoorShare: Math.min(1, Math.max(0, (v ?? 50) / 100)) })}
                  />
                  <NumberField label="Seed" value={settings.flamescapesDoorSeed} min={1} max={9999} step={1} onChange={(v) => setSettings({ flamescapesDoorSeed: Math.max(1, Math.round(v ?? 1)) })} />
                </div>
              )}
              {fire.running && fire.doors && fire.doors !== doorsNow && (
                <Notice tone="neutral">
                  This fire is running with the doors {fire.doors}.{" "}
                  <Button size="sm" onClick={onStartFire}>
                    Restart the fire with the doors {doorsNow}
                  </Button>
                </Notice>
              )}
              <p className="field__hint">
                {settings.flamescapesDoors === "closed"
                  ? "Each door's leaf fills its opening, so the smoke stays in the rooms it starts in until it finds another way. Routes still pass through every door."
                  : settings.flamescapesDoors === "asbuilt"
                    ? `The doors the model says close themselves are closed: SelfClosing, or a fire rating with -C (EN 13501-2); ${selfClosingCount} of ${semantics?.doors?.length ?? 0} doors here. The rest stand open.`
                    : settings.flamescapesDoors === "random"
                      ? "Each door is closed with the share above, drawn with the seed: the same seed closes the same doors, so runs can be repeated and compared."
                      : "Smoke and heat pass through every doorway, the outside doors too, so smoke also leaves the building through them. Close them to see what shut doors hold back; routes use the doors either way."}
              </p>
            </div>
            <Segmented
              label="Show on the graph"
              value={settings.flamescapesShow}
              onChange={(flamescapesShow) => setSettings({ flamescapesShow })}
              options={SHOW_GROUPS}
            />
            {settings.flamescapesShow === "smoke" && (
              <SelectField
                label="Smoke measure"
                value={settings.flamescapesSmokeMeasure}
                onChange={(flamescapesSmokeMeasure) => setSettings({ flamescapesSmokeMeasure })}
                options={Object.entries(SMOKE_MEASURES).map(([value, m]) => ({ value, label: m.label }))}
                hint="All five come from the soot at head height, so they shade the graph alike; the legend reads in the one chosen."
              />
            )}
            {settings.flamescapesShow === "gases" && (
              <SelectField
                label="Gas"
                value={settings.flamescapesGasMeasure}
                onChange={(flamescapesGasMeasure) => setSettings({ flamescapesGasMeasure })}
                options={Object.entries(GAS_MEASURES).map(([value, m]) => ({ value, label: m.label, disabled: m.extras && !settings.flamescapesMeasures }))}
                hint={settings.flamescapesMeasures ? undefined : "CO₂ and O₂ need the extra measures (Flamescapes settings)."}
              />
            )}
            {settings.flamescapesShow === "tenability" && (
              <p className="field__hint">
                The largest share of a limit each place has reached: gas temperature, visibility, and the asphyxiant (CO, with CO₂&apos;s
                faster breathing and low O₂) and heat doses against FED 0.3, after ISO 13571. Red at 1: untenable. The Methods page gives the
                formulas and sources.
              </p>
            )}
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
            disabled={!ready || fireBusy || !points.fire}
            onClick={onStartFire}
          >
            Start fire
          </Button>
          <Button variant="danger" size="lg" disabled={!fireBusy} onClick={onStopFire}>
            Stop
          </Button>
        </div>
        {ready && !points.fire && <p className="field__hint">Set a fire origin to start the simulation.</p>}
        {ready && points.fire && settings.reroute && !(points.start && points.exit) && (
          <p className="field__hint">Set a start and exit in the Route step to see the route react.</p>
        )}

        {flamescapes && <CombustionStatus combustion={fire.combustion} running={fire.running} />}
        {dynamicPath && <RerouteResult dynamicPath={dynamicPath} step={fire.step} />}
        {ready && fire.step > 0 && !fire.running && <ExportTopologic onExport={onExportTopologic} withRun={flamescapes} onExportTable={onExportTable} />}
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
                { value: "cpu", label: "The CPU (one core)" },
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
            <Toggle
              label="Beams and ducts in the fire grid"
              checked={settings.flamescapesOverhead}
              onChange={(flamescapesOverhead) => setSettings({ flamescapesOverhead })}
              hint={
                overheadCount === null
                  ? "Beams, and a second model's ducts and cable trays (step 1), as solids under the ceilings: they hold back the ceiling jet a sprinkler reads."
                  : `${overheadCount} beams, ducts and cable trays found; they hold back the ceiling jet a sprinkler reads.`
              }
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
              For the smoke only: a route still goes through every door. A closed door or window is airtight and never breaks or burns through, so
              smoke leaves the building only through open doorways and open windows. Real ones leak at their gaps and glass can fail in a hot layer,
              so a run with everything closed keeps the most smoke in its rooms. Running ventilation joins
              every air terminal in the model (IfcAirTerminal) through one well-mixed duct volume, the case that spreads smoke furthest: which
              system a terminal belongs to is not read, and 100 L/s a terminal is an assumption, not a measured value.
            </p>
            <SelectField
              label="Grid"
              value={settings.flamescapesGridPolicy}
              onChange={(flamescapesGridPolicy) => setSettings({ flamescapesGridPolicy })}
              options={[
                { value: "standard", label: "Standard: the same on every device, so the GPU and the CPU give one fire" },
                { value: "speed", label: "Finer on the GPU: sized to keep three times real time" },
                { value: "finest", label: "Finest on the GPU: all its memory allows (slower)" },
              ]}
            />
            <p className="field__hint">
              The fire depends on the grid: on the Duplex a sofa fire peaked at 0.09 MW on a 0.20 m grid, 2.5 MW on 0.23 m and 6.1 MW on
              0.27 m. The standard grid (at most {STANDARD_CELLS.toLocaleString()} cells) is the same on every device, so the CPU gives the
              GPU&apos;s fire at about a ninth of its speed; a phone&apos;s or small machine&apos;s CPU builds a lighter one to keep up, and
              its card says so. The finer grids are for the GPU (the CPU builds the standard one) and give a different fire; which grid is
              nearer a real fire has not been measured. Keep one grid when comparing runs; the run card gives it.
            </p>
            <SelectField
              label="Repeat runs on the GPU"
              value={String(settings.flamescapesRepeats)}
              onChange={(value) => setSettings({ flamescapesRepeats: Number(value) })}
              options={[
                { value: "1", label: "One run" },
                { value: "4", label: "4: the run you watch, then 3 more" },
                { value: "8", label: "8: the run you watch, then 7 more" },
              ]}
            />
            <p className="field__hint">
              A fire that catches the room&apos;s contents is partly chance: the same run with the room a thousandth of a degree warmer can
              hold or die back. After the run you watch, the GPU makes the rest on the same grid, each a thousandth of a degree warmer,
              while the view plays on, and the card gives the median and range of the untenable nodes, the first untenable time, the peak
              and the energy. On the standard grid each takes a few seconds; none are made on the CPU, where each would take as long as
              the first.
            </p>
            <Toggle
              label="Record soot, CO₂ and O₂ too"
              checked={settings.flamescapesMeasures}
              onChange={(flamescapesMeasures) => setSettings({ flamescapesMeasures })}
              hint="For the smoke and gas measures, the tenability index's CO₂ and O₂ terms, and the exports. The solver works them out anyway; recording them doubles what each stored frame holds, so a run past 64 MB of frames keeps temperature, visibility and CO only, and its card says so."
            />
            <Toggle
              label="Keep finished runs in this browser"
              checked={settings.flamescapesKeepRuns}
              onChange={(flamescapesKeepRuns) => setSettings({ flamescapesKeepRuns })}
              hint="The same fire on the same model, asked for again (after a reload too), plays from its kept frames instead of being computed. At most 256 MB; the runs used least recently go first."
            />
            <Button size="sm" onClick={onForgetRuns}>
              Forget kept runs
            </Button>
            <p className="field__hint">{FLAMESCAPES_LIMITS}</p>
          </Disclosure>
        )}

        {flamescapes && (
          <Disclosure summary="Sprinklers">
            <Toggle
              label="Sprinklers"
              checked={settings.sprinklers}
              onChange={(sprinklers) => setSettings({ sprinklers })}
              hint="Off by default. Off, nothing is laid out and a run is exactly as before, at the same speed. On, the heads are read from the IFC or placed per room by NFPA 13, and a Flamescapes run opens them as the fire heats their links."
            />
            {settings.sprinklers && waterResponse(FIRE_CLASSES[settings.flamescapesClass]?.fuel).note && (
              <Notice tone="warning">
                <strong>{FIRE_CLASSES[settings.flamescapesClass].label}:</strong> {waterResponse(FIRE_CLASSES[settings.flamescapesClass].fuel).note} The
                room&apos;s other contents are wetted and capped as usual.
              </Notice>
            )}
            {settings.sprinklers && (
              <>
                <SelectField
                  label="Heads"
                  value={settings.sprinklerSource}
                  onChange={(sprinklerSource) => setSettings({ sprinklerSource })}
                  options={[
                    { value: "auto", label: "The IFC's (this model's, else a second model's); else placed per room" },
                    { value: "ifc", label: "Only the model's (IfcFireSuppressionTerminal)" },
                    { value: "placed", label: "Placed per room by NFPA 13" },
                  ]}
                />
                <SelectField
                  label="Hazard"
                  value={settings.sprinklerHazard}
                  onChange={(sprinklerHazard) => setSettings({ sprinklerHazard })}
                  options={[
                    { value: "auto", label: "By room name: kitchens, plant and parking ordinary 1, the rest light" },
                    { value: "light", label: "Light hazard (offices, homes, classrooms): 20.9 m² a head" },
                    { value: "oh1", label: "Ordinary hazard 1: 12.1 m² a head" },
                    { value: "oh2", label: "Ordinary hazard 2: 12.1 m² a head, more water" },
                  ]}
                />
                <SelectField
                  label="Response"
                  value={settings.sprinklerResponse}
                  onChange={(sprinklerResponse) => setSettings({ sprinklerResponse })}
                  options={[
                    { value: "quick", label: "Quick response (RTI 50), required in light hazard rooms" },
                    { value: "standard", label: "Standard response (RTI 100)" },
                  ]}
                />
                <SelectField
                  label="An open head"
                  value={settings.sprinklerMode}
                  onChange={(sprinklerMode) => setSettings({ sprinklerMode })}
                  options={[
                    { value: "wetting", label: "Sprays and wets the fuel it reaches (the default: it cut most test fires)" },
                    { value: "spray", label: "Caps the fuel under it (Evans 1993) and sprays: steam and cooling in the gas" },
                    { value: "evans", label: "Caps the fuel under it (Evans 1993) only" },
                    { value: "activation", label: "Only opens: the fire is left as it is" },
                  ]}
                />
                <SelectField
                  label="Open the heads on"
                  value={settings.sprinklerOpenOn}
                  onChange={(sprinklerOpenOn) => setSettings({ sprinklerOpenOn })}
                  options={[
                    { value: "estimate", label: "The estimate: Alpert's ceiling jet under the fire's own ceiling" },
                    { value: "late", label: "The late bound: the grid's top cell, which reads the jet low" },
                  ]}
                />
                <SprinklerLayout plan={sprinklerPlan} busy={sprinklerBusy} onRedo={onLayOutSprinklers} />
                <p className="field__hint">
                  A head opens when its link reaches its rating: the RTI equation with conduction to the mount, stepped exactly on the gas at
                  the top cell of its column and, under the fire&apos;s ceiling, on Alpert&apos;s jet for the run&apos;s heat release. The run
                  card gives both times. What an open head then does is the choice above. Wetting: the water left at the end of each column
                  soaks into the first fuel it reaches and cools it. The cap: the fuel under the head burns at its rate when the head opened,
                  decaying as Evans measured for the spray&apos;s density (conservative: the least reduction to expect). With the spray, part of
                  the water evaporates in the hot gas on the way down (1 mm drops), the steam diluting the oxygen. Liquids, metals and cooking
                  oil are not wetted (the notice above). The heads read no RTI or C from the IFC (none carries them): quick 50, standard 100,
                  C 0.4. Portable extinguishers have their own section below; clean agent and foam rooms are in Systems by room use.
                </p>
              </>
            )}
          </Disclosure>
        )}

        {flamescapes && (
          <Disclosure summary="Systems by room use">
            <Toggle
              label="Systems by room use"
              checked={settings.suppressionSystems}
              onChange={(suppressionSystems) => setSettings({ suppressionSystems })}
              hint="Off by default. Off, nothing is assigned and a run is exactly as before. On, each room gets its system from its name and the building's occupancy: clean agent (HFC-227ea, FM-200) in server, telecom, electrical and battery rooms; foam-water sprinklers in generator, fuel and flammable-liquid rooms; water everywhere else. Apartments and hotels (Group R) get water in every room. A Flamescapes run floods or foams those rooms once their own detectors see smoke."
            />
            {settings.suppressionSystems && (
              <>
                <SelectField
                  label="Occupancy"
                  value={settings.perfOccupancy}
                  onChange={(id) => setSettings({ perfOccupancy: id, ...(presetOf(id).values || {}) })}
                  options={RSET_PRESETS.map((p) => ({ value: p.id, label: p.label }))}
                  hint="The performance check's occupants, shared with it: apartments and hotels (Group R) get water only, the rest by room name."
                />
                <Toggle
                  label="Agent only in the agent rooms"
                  checked={settings.systemsAgentOnly}
                  onChange={(systemsAgentOnly) => setSettings({ systemsAgentOnly })}
                  hint="Off: those rooms keep their sprinklers and the agent is a supplement (NFPA 75 9.1.1, 9.4.1). On: their sprinklers are left out, which needs the fire official's approval (SBC 801-CR-2018 903.3.1.1.1)."
                />
                {!settings.sprinklers && <Notice tone="warning">Sprinklers are off, so the water rooms have no system. Switch them on in Sprinklers above.</Notice>}
                <SystemsPlan plan={systemsPlan} busy={systemsBusy} onRedo={onLayOutSystems} />
                <p className="field__hint">
                  Clean agent: once a detector in each of two zones sees smoke and 30 s have passed (specification practice; the code sets no
                  number), the agent rises to 7 % over 10 s and holds for 10 minutes. At 5.2 % or more (6.62 % over a liquid&apos;s vapour)
                  flames and smouldering stop; hot fuel goes on giving off vapour, which can burn again outside the room or once the agent is
                  gone. The room is taken as closed and well mixed: the agent stays in it, does not cool or take oxygen, and the release does
                  not close the model&apos;s doors. Foam: once one detector sees smoke and 30 s have passed (assumed), the blanket cuts a
                  pool&apos;s evaporation to a tenth in about 10 s and to nothing in about 30 s (Back and Farley 2020), and its water wets the
                  ordinary fuel it lands on, as sprinklers do.
                </p>
              </>
            )}
          </Disclosure>
        )}

        <Disclosure summary="Extinguishers">
          <Toggle
            label="Extinguishers"
            checked={settings.extinguishers}
            onChange={(extinguishers) => setSettings({ extinguishers })}
            hint="Off by default. On, the model's portable extinguishers are read from the IFC by name, or placed on the walk so that nowhere is more than 22.86 m from one, and checked against SBC 801-CR-2018 906 and NFPA 10. After a fire, one person near it can try to put it out."
          />
          {settings.extinguishers && (
            <>
              <SelectField
                label="Extinguishers"
                value={settings.extinguisherSource}
                onChange={(extinguisherSource) => setSettings({ extinguisherSource })}
                options={[
                  { value: "auto", label: "The IFC's (this model's, else a second model's); else placed on the walk" },
                  { value: "ifc", label: "Only the IFC's" },
                  { value: "placed", label: "Placed on the walk (22.86 m rule)" },
                ]}
              />
              <SelectField
                label="Hazard (NFPA 10)"
                value={settings.extinguisherHazard}
                onChange={(extinguisherHazard) => setSettings({ extinguisherHazard })}
                options={[
                  { value: "light", label: "Light: offices, classrooms, hotels (279 m² per A unit)" },
                  { value: "ordinary", label: "Ordinary: shops, car parks, workshops (139 m²)" },
                  { value: "extra", label: "Extra: kitchens, storage of flammables (92.9 m², 4-A)" },
                ]}
              />
              <ExtinguisherLayout plan={extinguisherPlan} busy={extinguisherBusy} onRedo={onLayOutExtinguishers} />
              <SelectField
                label="The unit used"
                value={settings.extinguisherUnit}
                onChange={(extinguisherUnit) => setSettings({ extinguisherUnit })}
                options={EXTINGUISHER_UNITS}
              />
              <SelectField
                label="The person sets off"
                value={settings.extinguisherStart}
                onChange={(extinguisherStart) => setSettings({ extinguisherStart })}
                options={[
                  { value: "seen", label: "When they see the fire (30 s after ignition)" },
                  { value: "alarm", label: "After detection, the alarm and the median pre-movement (Route step, performance check)" },
                ]}
              />
              <Toggle
                label="Trained to use it (7.2 s to ready it, not 13.4 s)"
                checked={settings.extinguisherTrained}
                onChange={(extinguisherTrained) => setSettings({ extinguisherTrained })}
              />
              <Button size="sm" onClick={onExtinguisherAttempt} disabled={!extinguisherPlan?.sites?.length || !studio.points.fire}>
                Try to put the fire out
              </Button>
              <ExtinguisherAttempt attempt={extinguisherAttempt} />
            </>
          )}
        </Disclosure>

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
          {exits?.final.length > 1 && (
            <Choices
              legend="Reroute towards"
              name="reroute-exits"
              value={settings.rerouteExits}
              onChange={(rerouteExits) => setSettings({ rerouteExits })}
              options={[
                {
                  value: "any",
                  label: "Any exit out of the building",
                  description: `The cheapest of the ${exits.final.length} and the one you picked, so the route can move to another stair and door as the fire grows.`,
                },
                { value: "picked", label: "The exit you picked", description: "Only the path to it changes." },
              ]}
            />
          )}
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
