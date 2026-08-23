/**
 * Inspector for the legacy TopologicPy JSON-contract workflow.
 *
 * This is the `/upload-topology` and `/upload-ifc` path that predates the IFC
 * egress pipeline. It is kept because the backend still serves it and it is
 * how TopologicPy exports are inspected, but it lives behind its own viewer
 * mode rather than sharing a shell with the IFC controls the way Classic did.
 */
import React from "react";

import { Button, Empty, KeyValue, Section, Slider, Stat, Toggle } from "./primitives.jsx";

const fmt = (n) => (n === null || n === undefined ? "-" : n.toLocaleString());

function formatValue(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" || typeof value === "string") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function TopologyPanel({
  topology,
  summary,
  selection,
  selectedEntity,
  options,
  setOptions,
  busy,
  onReprocess,
  canReprocess,
  onFit,
}) {
  return (
    <>
      <Section title="Topology">
        {!topology ? (
          <Empty>
            Load a TopologicPy JSON export, or an IFC file processed server-side,
            to inspect cells, shells, faces and their graphs.
          </Empty>
        ) : (
          <div className="stats">
            <Stat label="Vertices" value={fmt(summary?.vertices)} />
            <Stat label="Edges" value={fmt(summary?.edges)} />
            <Stat label="Faces" value={fmt(summary?.faces)} />
            <Stat label="Entities" value={fmt(summary?.raw)} />
          </div>
        )}

        <Toggle
          label="Show faces"
          checked={options.showFaces}
          onChange={(showFaces) => setOptions({ showFaces })}
        />
        <Toggle
          label="Show vertices"
          checked={options.showVerts}
          onChange={(showVerts) => setOptions({ showVerts })}
        />
        <Toggle
          label="Wireframe"
          checked={options.wireframe}
          onChange={(wireframe) => setOptions({ wireframe })}
        />
        <Toggle
          label="Translucent faces"
          checked={options.translucent}
          onChange={(translucent) => setOptions({ translucent })}
        />
        <Button block size="sm" onClick={onFit} disabled={!topology}>
          Fit view
        </Button>
      </Section>

      <Section title="Server-side floor extraction" defaultOpen={false}>
        <p className="field__hint">
          Applies when an IFC file is processed by the backend rather than loaded
          into the 3D viewer. Changing these re-runs the extraction.
        </p>
        <Slider
          label="Minimum tilt"
          value={options.tiltMin}
          min={0}
          max={1}
          step={0.05}
          onChange={(tiltMin) => setOptions({ tiltMin })}
          format={(v) => v.toFixed(2)}
          hint="How close to horizontal a face must be to count as a floor."
        />
        <Slider
          label="Maximum height span"
          value={options.maxZSpan}
          min={0}
          max={3}
          step={0.1}
          onChange={(maxZSpan) => setOptions({ maxZSpan })}
          format={(v) => `${v.toFixed(2)} m`}
        />
        <Slider
          label="Minimum floor area"
          value={options.minFloorArea}
          min={1}
          max={50}
          step={1}
          onChange={(minFloorArea) => setOptions({ minFloorArea })}
          format={(v) => `${v.toFixed(0)} m2`}
        />
        <Button block busy={busy} disabled={!canReprocess} onClick={onReprocess}>
          Re-extract floors
        </Button>
        {!canReprocess && (
          <p className="field__hint">
            Load an IFC file through &ldquo;Process on server&rdquo; to enable this.
          </p>
        )}
      </Section>

      <Section title="Selection">
        {!selection ? (
          <Empty>
            Click a face, edge or vertex to inspect it. Clicking the same spot again
            walks up the hierarchy: CellComplex, Cell, Shell, Face, Edge, Vertex.
          </Empty>
        ) : (
          <>
            <KeyValue
              entries={[
                ["Type", selection.level],
                ["UID", selection.uid],
              ]}
            />
            {(() => {
              const dictionary = selectedEntity?.dictionary || {};
              const entries = Object.entries(dictionary).sort(([a], [b]) =>
                a.localeCompare(b),
              );
              if (!entries.length) {
                return <p className="field__hint">No dictionary entries on this entity.</p>;
              }
              return <KeyValue entries={entries.map(([k, v]) => [k, formatValue(v)])} />;
            })()}
          </>
        )}
      </Section>
    </>
  );
}
