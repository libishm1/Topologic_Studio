/**
 * The 3D viewport.
 *
 * Mounts the engine exactly once and exposes it through a ref. Everything that
 * changes afterwards is pushed into ViewerManager imperatively, so React never
 * re-creates the WebGL context.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";

import { Badge, Button } from "./primitives.jsx";
import logo from "../assets/topologicStudio-white-logo400x400.png";

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
  legend,
}) {
  const containerRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [initError, setInitError] = useState(null);
  const [booting, setBooting] = useState(true);

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
        if (import.meta.env.DEV) window.__viewer = manager;
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
    <div
      className="viewport"
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
          <div className="placeholder__card">
            <img src={logo} alt="" className="placeholder__logo" />
            <h2>Topologic Studio</h2>
            <p>
              Load an IFC model to explore it in 3D, build an egress navigation
              graph, and simulate fire spread with live rerouting.
            </p>
            <div className="dropzone" data-over={dragOver}>
              <p style={{ marginBottom: 8, fontWeight: 600, color: "var(--text-primary)" }}>
                Drop an .ifc file here
              </p>
              <p style={{ fontSize: 12 }}>
                or use <strong>Open IFC</strong> above. Models are converted once and
                cached locally, so opening the same file again is near-instant.
              </p>
            </div>
            <p style={{ fontSize: 11, color: "var(--text-tertiary)" }}>
              <span className="kbd">S</span> set start &nbsp;
              <span className="kbd">E</span> set exit &nbsp;
              <span className="kbd">F</span> set fire &nbsp;
              <span className="kbd">G</span> toggle graph
            </p>
          </div>
        </div>
      )}

      {initError && (
        <div className="placeholder">
          <div className="placeholder__card">
            <h2>The 3D viewer could not start</h2>
            <p>{initError}</p>
            <p style={{ fontSize: 12 }}>
              This usually means WebGL is unavailable or the web-ifc WASM files are
              missing from <code>/wasm/</code>. Run <code>npm run sync-wasm</code>.
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
        <div className="overlay overlay--top-right" style={{ top: pickMode ? 62 : undefined }}>
          <div className="card" style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Badge variant="danger" pulse>
              Fire
            </Badge>
            <span style={{ fontSize: 12 }}>
              step <strong className="mono">{fire.step}</strong>
            </span>
          </div>
        </div>
      )}

      {hasModel && (
        <>
          <div className="overlay overlay--bottom-left">
            {legend?.length > 0 && (
              <div className="card legend">
                {legend.map((item) => (
                  <div className="legend__row" key={item.label}>
                    <span
                      className={`legend__swatch${item.dot ? " legend__swatch--dot" : ""}`}
                      style={{ background: item.color }}
                    />
                    <span>{item.label}</span>
                  </div>
                ))}
              </div>
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
              <div className="viewcube">
                <Button size="sm" onClick={() => setView("top")} title="Top view">
                  Top
                </Button>
                <Button size="sm" onClick={() => setView("front")} title="Front view">
                  Front
                </Button>
                <Button size="sm" onClick={() => setView("right")} title="Right view">
                  Right
                </Button>
                <Button size="sm" onClick={() => setView("iso")} title="Isometric view">
                  Iso
                </Button>
                <Button size="sm" onClick={() => setView("left")} title="Left view">
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
    </div>
  );
}
