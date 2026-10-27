/**
 * The 3D viewport.
 *
 * Mounts the engine exactly once and exposes it through a ref. Everything that
 * changes afterwards is pushed into ViewerManager imperatively, so React never
 * re-creates the WebGL context.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";

import { formatBytes } from "../lib/perf.js";
import { Badge, Button, FileButton, Icon } from "./primitives.jsx";

const STAGE_LABELS = {
  reading: "Reading file",
  converting: "Converting IFC",
  caching: "Caching fragments",
  cached: "Loading from cache",
  rendering: "Drawing model",
  indexing: "Reading categories",
  extracting: "Extracting surfaces",
  sampling: "Sampling walkable area",
  building: "Building graph",
};

export function Viewport({
  viewerRef,
  onPick,
  onReady,
  onFile,
  theme,
  pickMode,
  hasModel,
  loadState,
  graphStats,
  fire,
  levels,
  section,
  onSection,
  fireSteps,
  heatLegend,
  onStopFire,
  legend,
  examples,
  onExample,
}) {
  const containerRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [initError, setInitError] = useState(null);
  const [booting, setBooting] = useState(true);
  // Open on a wide screen; on a phone the legend would cover most of the
  // model, so it starts folded to one line.
  const [legendOpen, setLegendOpen] = useState(
    () => typeof window === "undefined" || window.matchMedia("(min-width: 761px)").matches,
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    let cancelled = false;
    let manager = null;

    // Three plus the That Open engine is ~6 MB. Importing it lazily lets the
    // shell paint immediately and streams the engine in behind the loading
    // state, instead of blocking first paint on the whole bundle.
    (async () => {
      try {
        const { ViewerManager } = await import("../viewer/ViewerManager.js");
        if (cancelled) return;
        manager = new ViewerManager();
        viewerRef.current = manager;
        // Dev-only handle so the browser test harness (and a human with the
        // console open) can inspect the live scene.
        // ?debug exposes it in a production build too, so tools/browser-test.mjs
        // can aim picks at graph nodes on the deployed site.
        if (import.meta.env.DEV || new URLSearchParams(window.location.search).has("debug")) {
          window.__viewer = manager;
        }
        await manager.init(container, { theme, onPick });
        if (cancelled) {
          manager.dispose();
          return;
        }
        setBooting(false);
        onReady?.(manager);
      } catch (error) {
        if (cancelled) return;
        console.error("Viewer init failed", error);
        setBooting(false);
        setInitError(error?.message || "The 3D viewer failed to start.");
      }
    })();

    return () => {
      cancelled = true;
      manager?.dispose();
      viewerRef.current = null;
    };
    // Mount once. Theme and pick handler are pushed in below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    viewerRef.current?.applyTheme(theme);
  }, [theme, viewerRef]);

  useEffect(() => {
    viewerRef.current?.setPickMode(pickMode);
  }, [pickMode, viewerRef]);

  // Keep the pick callback fresh without re-mounting the engine.
  useEffect(() => {
    if (viewerRef.current) viewerRef.current._onPick = onPick;
  }, [onPick, viewerRef]);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setDragOver(false);
      const dropped = Array.from(event.dataTransfer?.files || []);
      const ifc = dropped.find((f) => f.name.toLowerCase().endsWith(".ifc"));
      if (ifc) onFile(ifc);
      else if (dropped.length) {
        onFile(dropped[0]);
      }
    },
    [onFile],
  );

  const setView = useCallback(
    (which) => viewerRef.current?.setStandardView(which),
    [viewerRef],
  );
  const fitView = useCallback(() => viewerRef.current?.fitToModel(), [viewerRef]);

  return (
    <main
      className="viewport"
      aria-label="Model view"
      data-picking={pickMode ? "true" : "false"}
      onDragOver={(event) => {
        event.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <div ref={containerRef} className="viewport__canvas" />

      {booting && !initError && (
        <div className="placeholder">
          <div className="placeholder__card">
            <div className="btn__spinner" style={{ color: "var(--accent)", width: 22, height: 22 }} />
            <p>Starting the 3D engine...</p>
          </div>
        </div>
      )}

      {!booting && !hasModel && !loadState.busy && (
        <div className="placeholder">
          <section className="welcome" aria-labelledby="welcome-title">
            <div>
              <h1 id="welcome-title" className="welcome__title">
                Open a building model to find its way out
              </h1>
              <p className="welcome__lead">
                Studio turns an IFC model into a walkable graph, then finds egress routes, alternatives, and
                how a spreading fire changes them. The model stays in your browser unless you run the
                TopologicPy cross-check.
              </p>
            </div>

            <div className="dropzone" data-over={dragOver}>
              <Icon name="upload" size={32} />
              <p style={{ fontWeight: 650, color: "var(--text-primary)" }}>Drop an .ifc file here</p>
              <p style={{ fontSize: 13 }}>
                IFC 2x3 or IFC4. Converted once and cached in this browser, so the same file opens faster
                the second time.
              </p>
              <FileButton accept=".ifc" onFile={onFile}>
                Choose a file
              </FileButton>
            </div>

            {examples?.length > 0 && (
              <div>
                <h2 className="welcome__subtitle">Or start from an example</h2>
                <ul className="example-grid">
                  {examples.map((entry) => (
                    <li key={entry.id}>
                      <button
                        type="button"
                        className="example"
                        onClick={() => onExample(entry)}
                        aria-label={entry.bytes ? `${entry.name}, ${formatBytes(entry.bytes)}` : entry.name}
                        aria-describedby={`welcome-example-${entry.id}`}
                      >
                        <span className="example__head">
                          <span className="example__name">{entry.name}</span>
                          {entry.bytes ? <span className="example__size">{formatBytes(entry.bytes)}</span> : null}
                        </span>
                        <span className="example__desc" id={`welcome-example-${entry.id}`}>
                          {entry.description}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
                <p className="examples__credit" style={{ marginTop: 12 }}>
                  {examples
                    .filter((entry) => entry.source)
                    .map((entry, i) => (
                      <React.Fragment key={entry.id}>
                        {i > 0 && " "}
                        {entry.name}:{" "}
                        <a href={entry.source} target="_blank" rel="noreferrer">
                          {entry.credit}
                        </a>
                      </React.Fragment>
                    ))}{" "}
                  The other examples are generated for Studio and are CC0.
                </p>
              </div>
            )}
          </section>
        </div>
      )}

      {initError && (
        <div className="placeholder">
          <div className="placeholder__card">
            <h2>The 3D viewer could not start</h2>
            <p>{initError}</p>
            <p style={{ fontSize: 12 }}>
              This usually means WebGL is unavailable or the web-ifc WASM files are
              missing from <code>public/wasm/</code>. Run <code>npm run sync-assets</code>.
            </p>
          </div>
        </div>
      )}

      {loadState.busy && (
        <div className="overlay overlay--top-left">
          <div className="card progress">
            <div className="progress__row">
              <span className="progress__label">
                {STAGE_LABELS[loadState.stage] || "Working"}
              </span>
              {loadState.percent !== undefined && loadState.percent !== null && (
                <span className="progress__detail">{Math.round(loadState.percent)}%</span>
              )}
            </div>
            <div className="progress__track">
              <div
                className={`progress__fill${
                  loadState.percent === undefined || loadState.percent === null
                    ? " progress__fill--indeterminate"
                    : ""
                }`}
                style={{ width: `${loadState.percent ?? 35}%` }}
              />
            </div>
            {loadState.detail && (
              <div className="progress__detail">{String(loadState.detail).slice(0, 60)}</div>
            )}
          </div>
        </div>
      )}

      {pickMode && (
        <div className="overlay overlay--top-right">
          <div className="card" style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Badge variant="accent" pulse>
              Picking
            </Badge>
            <span style={{ fontSize: 12 }}>
              Click in the model to place the{" "}
              <strong>
                {pickMode === "start" ? "start" : pickMode === "exit" ? "exit" : "fire origin"}
              </strong>
            </span>
            <span className="kbd">Esc</span>
          </div>
        </div>
      )}

      {fire?.running && (
        <div className="overlay overlay--bottom-center">
          <div className="card playback" role="group" aria-label="Fire simulation">
            <div className="playback__info">
              <div className="playback__row">
                <strong>
                  Fire step <span className="mono">{fire.step}</span>
                  {fireSteps ? ` of ${fireSteps}` : ""}
                </strong>
              </div>
              {fireSteps ? (
                <div
                  className="playback__track"
                  role="progressbar"
                  aria-label="Fire simulation progress"
                  aria-valuemin={0}
                  aria-valuemax={fireSteps}
                  aria-valuenow={Math.min(fire.step, fireSteps)}
                >
                  <div className="playback__fill" style={{ width: `${Math.min(100, (fire.step / fireSteps) * 100)}%` }} />
                </div>
              ) : null}
            </div>
            <Button variant="danger" onClick={onStopFire}>
              Stop fire
            </Button>
          </div>
        </div>
      )}

      {hasModel && (
        <>
          <div className="overlay overlay--bottom-left">
            {(legend?.length > 0 || heatLegend) && (
              <details
                className="card legend"
                open={legendOpen}
                onToggle={(event) => setLegendOpen(event.currentTarget.open)}
              >
                <summary className="legend__summary">Legend</summary>
                {heatLegend && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 6 }}>
                    <span className="legend__title">Model temperature</span>
                    <span className="legend__ramp" aria-hidden="true" />
                    <span className="legend__scale">
                      <span>20</span>
                      <span>70</span>
                      <span>120</span>
                    </span>
                    <span className="legend__note">An index scaled to read in °C, not a measured temperature.</span>
                  </div>
                )}
                {legend.map((item) => (
                  <div className="legend__row" key={item.label}>
                    <span
                      className={`legend__swatch${item.dot ? " legend__swatch--dot" : ""}`}
                      style={{ background: item.color }}
                    />
                    <span>{item.label}</span>
                  </div>
                ))}
              </details>
            )}
          </div>

          <div className="overlay overlay--bottom-right">
            {graphStats && (
              <div className="card" style={{ fontSize: 11 }}>
                <span className="mono">{graphStats.nodes.toLocaleString()}</span> nodes
                {"  ·  "}
                <span className="mono">{graphStats.edges.toLocaleString()}</span> edges
              </div>
            )}
            <div className="card">
              {levels?.length > 1 && (
                <div className="section-cut">
                  <label htmlFor="section-cut">Show</label>
                  <select
                    id="section-cut"
                    value={section === null ? "" : String(section)}
                    onChange={(event) => onSection(event.target.value === "" ? null : Number(event.target.value))}
                  >
                    <option value="">Whole building</option>
                    {levels.map((height, i) => (
                      <option key={height} value={i}>
                        Cut at level {i + 1} ({height - levels[0] >= 0 ? "+" : ""}
                        {(height - levels[0]).toFixed(1)} m)
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <div className="viewcube">
                <Button size="sm" onClick={() => setView("top")} title="Top view" data-secondary="true">
                  Top
                </Button>
                <Button size="sm" onClick={() => setView("front")} title="Front view" data-secondary="true">
                  Front
                </Button>
                <Button size="sm" onClick={() => setView("right")} title="Right view" data-secondary="true">
                  Right
                </Button>
                <Button size="sm" onClick={() => setView("iso")} title="Isometric view">
                  Iso
                </Button>
                <Button size="sm" onClick={() => setView("left")} title="Left view" data-secondary="true">
                  Left
                </Button>
                <Button
                  size="sm"
                  onClick={fitView}
                  title="Fit model in view"
                >
                  Fit
                </Button>
              </div>
            </div>
          </div>
        </>
      )}
    </main>
  );
}
