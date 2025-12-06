// src/App.jsx
import React, { useState, useMemo } from "react";
import axios from "axios";
import TopologyViewer from "./TopologyViewer.jsx";
import "./App.css"; // ← make sure this line is present
import logoImg from "./assets/topologicStudio-white-logo400x400.png"; // adjust path/name if needed
const API_BASE = "http://localhost:8000"; // FastAPI backend

export default function App() {
  const [topology, setTopology] = useState(null);
  const [selection, setSelection] = useState(null);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState("");

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setSelection(null);
    setTopology(null);
    setFileName(file.name);

    try {
      const text = await file.text();

      let originalJson;
      try {
        originalJson = JSON.parse(text);
      } catch (parseErr) {
        console.error("JSON parse error:", parseErr);
        setError("JSON parse error – is this a valid JSON file?");
        return;
      }

      let res;
      try {
        res = await axios.post(`${API_BASE}/upload-topology`, originalJson);
      } catch (apiErr) {
        console.error("API error:", apiErr);
        if (apiErr.response) {
          setError(
            `API error ${apiErr.response.status}: ` +
              JSON.stringify(apiErr.response.data)
          );
        } else {
          setError(
            `Network error: ${apiErr.code || apiErr.message || "unknown"}`
          );
        }
        return;
      }

      const payload = res.data;

      if (!payload.vertices || !payload.faces) {
        setError("Unexpected response format from backend.");
        return;
      }

      if (!payload.edges) payload.edges = [];
      if (!payload.raw) payload.raw = [];

      setTopology(payload);
    } catch (err) {
      console.error("Unexpected error:", err);
      setError("Unexpected error while loading topology.");
    }
  }

  const rawById = useMemo(() => {
    if (!topology || !topology.raw) return new Map();
    const map = new Map();
    topology.raw.forEach((e) => {
      const id = e.uid ?? e.uuid;
      if (!id) return;
      map.set(id, e);
    });
    return map;
  }, [topology]);

  const selectedEntity = useMemo(() => {
    if (!selection || !rawById.size) return null;
    return rawById.get(selection.uid) || null;
  }, [selection, rawById]);

  const summary = useMemo(() => {
    if (!topology) return null;
    return {
      numVertices: topology.vertices?.length || 0,
      numEdges: topology.edges?.length || 0,
      numFaces: topology.faces?.length || 0,
    };
  }, [topology]);

  const formatDictValue = (value) => {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return value;
    if (typeof value === "string") return value;
    // Arrays / objects
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  };

  return (
    <div className="app-root">
      {/* Top bar */}
      <header className="app-header">
        <div className="app-header-left">
          <div className="app-logo-circle">
            <img src={logoImg} alt="TopologicStudio logo" className="app-logo-image" />
          </div>
          <div className="app-title-block">
            <h1 className="app-title">TopologicStudio</h1>
            <span className="app-subtitle">
              Interactive topology + graph viewer
            </span>
          </div>
        </div>
        <div className="app-header-middle">
          {summary ? (
            <span className="app-summary">
              <span>{summary.numVertices} vertices</span>
              <span className="app-summary-dot" />
              <span>{summary.numEdges} edges</span>
              <span className="app-summary-dot" />
              <span>{summary.numFaces} faces</span>
            </span>
          ) : (
            <span className="app-summary app-summary-faded">
              No topology loaded
            </span>
          )}
        </div>
        <div className="app-header-right">
          <label className="file-upload-button">
            <input
              type="file"
              accept=".json"
              onChange={handleFileChange}
              className="file-upload-input"
            />
            <span>Load JSON</span>
          </label>
          {fileName && (
            <span className="file-chip" title={fileName}>
              {fileName}
            </span>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="app-main">
        <div className="viewer-panel">
          {topology ? (
            <TopologyViewer
              data={topology}
              selection={selection}
              onSelectionChange={setSelection}
            />
          ) : (
            <div className="viewer-placeholder">
              <div className="viewer-placeholder-card">
                <h2>Welcome to TopologicStudio</h2>
                <p>
                  Load a TopologicPy JSON export to explore cells, shells,
                  faces, and their graphs in an interactive 3D view.
                </p>
                <p className="viewer-placeholder-hint">
                  Use the button in the top right to load a file.
                </p>
              </div>
            </div>
          )}
          {error && <div className="error-banner">{error}</div>}
        </div>

        {/* Sidebar / Inspector */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <span className="sidebar-title">Properties</span>
            {selection && (
              <span className="sidebar-pill">{selection.level}</span>
            )}
          </div>

          {!selection && (
            <div className="sidebar-empty">
              <p>Click on a face, edge, or vertex to inspect its metadata.</p>
              <p className="sidebar-empty-hint">
                Repeated clicks on the same location will cycle through the
                hierarchy (CellComplex → Cell → Shell → Face → Edge → Vertex).
              </p>
            </div>
          )}

          {selection && (
            <>
              <div className="sidebar-section">
                <div className="sidebar-section-header">Selection</div>
                <div className="sidebar-section-body sidebar-selection-body">
                  <div>
                    <span className="sidebar-label">Type</span>
                    <span className="sidebar-value">{selection.level}</span>
                  </div>
                  <div>
                    <span className="sidebar-label">UID</span>
                    <span className="sidebar-value sidebar-value-mono">
                      {selection.uid}
                    </span>
                  </div>
                </div>
              </div>

              <div className="sidebar-section">
              <div className="sidebar-section-header">Dictionary</div>
              <div className="sidebar-section-body">
                {selectedEntity ? (
                  (() => {
                    const dict = selectedEntity.dictionary || {};
                    const entries = Object.entries(dict).sort(([a], [b]) =>
                      a.localeCompare(b)
                    );

                    if (entries.length === 0) {
                      return (
                        <p className="sidebar-empty-hint">
                          No dictionary entries for this entity.
                        </p>
                      );
                    }

                    return (
                      <div className="sidebar-dict-container">
                        {entries.map(([key, value]) => (
                          <div className="sidebar-dict-row" key={key}>
                            <div className="sidebar-dict-key">{key}</div>
                            <div className="sidebar-dict-value">
                              {formatDictValue(value)}
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  })()
                ) : (
                  <p className="sidebar-empty-hint">
                    No dictionary found for this entity.
                  </p>
                )}
              </div>
            </div>

            </>
          )}
        </aside>
      </div>
    </div>
  );
}
