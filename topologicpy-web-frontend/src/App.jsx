// src/App.jsx
import React, { useState, useMemo, useEffect, useRef } from "react";
import axios from "axios";
import TopologyViewer from "./TopologyViewer.jsx";
import "./App.css";
import logoImg from "./assets/topologicStudio-white-logo400x400.png";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"; // FastAPI backend

export default function App() {
  const spinnerStyle = { __html: `@keyframes spin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }` };

  const [topology, setTopology] = useState(null);
  const [selection, setSelection] = useState(null);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState("");
  const [translucent, setTranslucent] = useState(true);
  const [floorTilt, setFloorTilt] = useState(0.3);
  const [floorMaxZ, setFloorMaxZ] = useState(1.0);
  const [floorMinArea, setFloorMinArea] = useState(9);
  const [lastIfcFile, setLastIfcFile] = useState(null);
  const [lastIncludePath, setLastIncludePath] = useState(false);
  const [showFaces, setShowFaces] = useState(false);
  const [showVerts, setShowVerts] = useState(false);
  const [wireframe, setWireframe] = useState(true);
  const [loading, setLoading] = useState(false);
  const [fitRequest, setFitRequest] = useState(0);

  const fireTimerRef = useRef(null);
  const fireSseRef = useRef(null);
  const [graphMode, setGraphMode] = useState("cell");
  const [pickMode, setPickMode] = useState(null);
  const [startPoint, setStartPoint] = useState(null);
  const [exitPoint, setExitPoint] = useState(null);
  const [startId, setStartId] = useState(null);
  const [exitId, setExitId] = useState(null);
  const [fireRunning, setFireRunning] = useState(false);
  const [fireTimeline, setFireTimeline] = useState([]);
  const [fireStep, setFireStep] = useState(0);
  const [fireNodes, setFireNodes] = useState([]);
  const [fireUsePrecompute, setFireUsePrecompute] = useState(true);
  const [fireDelayMs, setFireDelayMs] = useState(200);
  const [fireMaxSteps, setFireMaxSteps] = useState(60);
  const [fireCellBboxes, setFireCellBboxes] = useState([]);
  const [cellDisplayNodes, setCellDisplayNodes] = useState([]);
  const [rlEpisodes, setRlEpisodes] = useState(200);
  const [rlMaxSteps, setRlMaxSteps] = useState(200);
  const [rlUseFire, setRlUseFire] = useState(true);
  const [rlPath, setRlPath] = useState([]);
  const [rlLoading, setRlLoading] = useState(false);
  const emptyExtras = useMemo(() => [], []);


  const stopFire = () => {
    if (fireTimerRef.current) {
      clearInterval(fireTimerRef.current);
      fireTimerRef.current = null;
    }
    if (fireSseRef.current) {
      fireSseRef.current.close();
      fireSseRef.current = null;
    }
    setFireRunning(false);
  };

  const resetSimulationState = () => {
    stopFire();
    setFireTimeline([]);
    setFireNodes([]);
    setFireStep(0);
    setFireCellBboxes([]);
    setRlPath([]);
    setStartPoint(null);
    setExitPoint(null);
    setStartId(null);
    setExitId(null);
    setPickMode(null);
  };

  async function uploadIfc(file, includePath) {
    setLoading(true);
    setError(null);
    resetSimulationState();
    setSelection(null);
    setTopology(null);
    setFileName(includePath ? `${file.name} (path)` : file.name);

    const form = new FormData();
    form.append("file", file);

    const query = includePath
      ? `include_path=true&tilt_min=${floorTilt}&max_z_span=${floorMaxZ}&min_floor_area=${floorMinArea}`
      : `include_path=false`;

    try {
      const res = await axios.post(`${API_BASE}/upload-ifc?${query}`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const payload = res.data;
      if (!payload?.vertices || !payload?.faces) {
        setError("Unexpected response format from IFC upload.");
        return;
      }
      if (!payload.edges) payload.edges = [];
      if (!payload.raw) payload.raw = [];
      setTopology(payload);
    } catch (apiErr) {
      setError(
        apiErr.response?.data?.detail || apiErr.message || "IFC upload failed"
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    resetSimulationState();
    setSelection(null);
    setTopology(null);
    setShowFaces(false);
    setFileName(file.name);

    try {
      const text = await file.text();
      let originalJson;
      try {
        originalJson = JSON.parse(text);
      } catch (parseErr) {
        console.error("JSON parse error:", parseErr);
        setError("JSON parse error: is this a valid JSON file?");
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
    } finally {
      setLoading(false);
    }
  }

  const handleSelectionChange = (nextSelection) => {
    setSelection(nextSelection);
    if (!pickMode || !nextSelection?.point) return;
    if (pickMode === "start") {
      setStartPoint(nextSelection.point);
      setStartId(nextSelection.level === "Vertex" ? nextSelection.uid : null);
    } else if (pickMode === "exit") {
      setExitPoint(nextSelection.point);
      setExitId(nextSelection.level === "Vertex" ? nextSelection.uid : null);
    }
    setPickMode(null);
  };

  async function handleIfcUpload(event, includePath = false) {
    const file = event.target.files?.[0];
    if (!file) return;
    event.target.value = null;
    setLastIfcFile(file);
    setLastIncludePath(includePath);
    setShowFaces(!includePath);
    await uploadIfc(file, includePath);
  }

  const handlePickStart = () => {
    setPickMode("start");
    setError(null);
  };

  const handlePickExit = () => {
    setPickMode("exit");
    setError(null);
  };

  const clearStartExit = () => {
    setStartPoint(null);
    setExitPoint(null);
    setStartId(null);
    setExitId(null);
    setPickMode(null);
  };

  const stopFireSimulation = () => {
    stopFire();
    setFireNodes([]);
    setFireTimeline([]);
    setFireStep(0);
  };

  const startFireSimulation = async () => {
    stopFire();
    setError(null);
    setFireNodes([]);
    setFireTimeline([]);
    setFireStep(0);
    setFireRunning(true);

    if (fireUsePrecompute) {
      try {
        const res = await axios.post(`${API_BASE}/fire-sim`, {
          mode: graphMode,
          start_id: startId,
          end_id: exitId,
          start_point: startPoint,
          end_point: exitPoint,
          max_steps: fireMaxSteps,
          precompute: true,
          radial: true,
          delay_ms: fireDelayMs,
        });
        const payload = res.data || {};
        const timeline = payload.timeline || [];
        setFireCellBboxes(payload.cell_bboxes || []);
        setFireTimeline(timeline);
        if (timeline.length === 0) {
          setFireRunning(false);
          return;
        }
        let idx = 0;
        setFireStep(0);
        setFireNodes(timeline[0] || []);
        if (timeline.length > 1) {
          fireTimerRef.current = setInterval(() => {
            idx += 1;
            if (idx >= timeline.length) {
              stopFire();
              return;
            }
            setFireStep(idx);
            setFireNodes(timeline[idx] || []);
          }, Math.max(50, fireDelayMs));
        } else {
          setFireRunning(false);
        }
      } catch (apiErr) {
        setFireRunning(false);
        setError(
          apiErr.response?.data?.detail || apiErr.message || "Fire simulation failed."
        );
      }
      return;
    }

    const params = new URLSearchParams();
    params.set("mode", graphMode);
    params.set("max_steps", String(fireMaxSteps));
    params.set("precompute", "false");
    params.set("radial", "true");
    params.set("delay_ms", String(fireDelayMs));
    if (startId) params.set("start_id", startId);
    if (exitId) params.set("end_id", exitId);
    if (startPoint && startPoint.length >= 3) {
      params.set("start_x", startPoint[0]);
      params.set("start_y", startPoint[1]);
      params.set("start_z", startPoint[2]);
    }
    if (exitPoint && exitPoint.length >= 3) {
      params.set("end_x", exitPoint[0]);
      params.set("end_y", exitPoint[1]);
      params.set("end_z", exitPoint[2]);
    }
    const url = `${API_BASE}/fire-sim/stream?${params.toString()}`;
    const es = new EventSource(url);
    fireSseRef.current = es;
    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "meta") {
          setFireCellBboxes(msg.cell_bboxes || []);
        } else if (msg.type === "step") {
          setFireStep(msg.step ?? 0);
          setFireNodes(msg.nodes || []);
        } else if (msg.type === "done") {
          stopFire();
        }
      } catch {
        // ignore parse errors
      }
    };
    es.onerror = () => {
      stopFire();
      setError("Fire stream error.");
    };
  };

  const trainRlPath = async () => {
    if (!startPoint && !startId) {
      setError("Pick a start point first.");
      return;
    }
    if (!exitPoint && !exitId) {
      setError("Pick an exit point first.");
      return;
    }
    setError(null);
    setRlLoading(true);
    setRlPath([]);
    try {
      const res = await axios.post(`${API_BASE}/rl/train`, {
        mode: graphMode,
        start_id: startId,
        exit_id: exitId,
        start_point: startPoint,
        exit_point: exitPoint,
        episodes: rlEpisodes,
        max_steps: rlMaxSteps,
        use_fire: rlUseFire,
      });
      const payload = res.data || {};
      setRlPath(payload.path || []);
      if (payload.cell_bboxes) {
        setFireCellBboxes(payload.cell_bboxes || []);
      }
    } catch (apiErr) {
      setError(
        apiErr.response?.data?.detail || apiErr.message || "RL training failed."
      );
    } finally {
      setRlLoading(false);
    }
  };

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

  const vertexList = useMemo(() => {
    if (!topology || !Array.isArray(topology.vertices)) return [];
    return topology.vertices
      .map((v) => {
        const id = v.uid ?? v.uuid;
        const coord = v.coordinates || v.Coordinates;
        if (!id || !Array.isArray(coord) || coord.length < 3) return null;
        return { id, coord: coord.slice(0, 3) };
      })
      .filter(Boolean);
  }, [topology]);

  const edgeList = useMemo(() => {
    if (!topology) return [];
    if (Array.isArray(topology.edges) && topology.edges.length > 0) {
      return topology.edges;
    }
    if (Array.isArray(topology.raw)) {
      return topology.raw.filter(
        (e) => e.type === "Edge" && Array.isArray(e.vertices)
      );
    }
    return [];
  }, [topology]);

  const edgeByKey = useMemo(() => {
    const map = new Map();
    edgeList.forEach((e) => {
      const verts = e.vertices || [];
      if (verts.length < 2) return;
      const a = verts[0];
      const b = verts[1];
      const key = a < b ? `${a}|${b}` : `${b}|${a}`;
      const id = e.uid ?? e.uuid;
      if (id && !map.has(key)) {
        map.set(key, id);
      }
    });
    return map;
  }, [edgeList]);


  const cellDisplayVertices = useMemo(() => {
    if (!cellDisplayNodes.length) return [];
    return cellDisplayNodes.map((cell) => {
      const center =
        cell.center || [
          0.5 * (cell.minx + cell.maxx),
          0.5 * (cell.miny + cell.maxy),
          cell.z ?? cell.z_min ?? 0,
        ];
      return {
        id: cell.id,
        coord: center,
        color: "#22d3ee",
      };
    });
  }, [cellDisplayNodes]);

  const fireVertexIds = useMemo(() => {
    if (!fireNodes.length) return [];
    return fireNodes;
  }, [fireNodes]);

  const fireEdgeIds = useMemo(() => {
    if (graphMode !== "wire" || !edgeList.length || !fireVertexIds.length) {
      return [];
    }
    const fireSet = new Set(fireVertexIds);
    return edgeList
      .filter((e) => {
        const verts = e.vertices || [];
        return verts.length >= 2 && fireSet.has(verts[0]) && fireSet.has(verts[1]);
      })
      .map((e) => e.uid ?? e.uuid)
      .filter(Boolean);
  }, [edgeList, fireVertexIds, graphMode]);

  const pathVertexIds = useMemo(() => {
    if (!rlPath.length) return [];
    return rlPath;
  }, [rlPath]);

  const pathEdgeIds = useMemo(() => {
    if (pathVertexIds.length < 2) return [];
    const ids = [];
    for (let i = 0; i < pathVertexIds.length - 1; i += 1) {
      const a = pathVertexIds[i];
      const b = pathVertexIds[i + 1];
      const key = a < b ? `${a}|${b}` : `${b}|${a}`;
      const edgeId = edgeByKey.get(key);
      if (edgeId) ids.push(edgeId);
    }
    return ids;
  }, [pathVertexIds, edgeByKey]);

  const displayTopology = useMemo(() => {
    if (!topology) return null;
    if (!translucent) return topology;
    const faces = (topology.faces || []).map((f) => {
      const opacity = 0.25;
      return { ...f, opacity, dictionary: { ...(f.dictionary || {}), opacity } };
    });
    return { ...topology, faces };
  }, [topology, translucent]);




  useEffect(() => {
    return () => {
      stopFire();
    };
  }, []);

  useEffect(() => {
    stopFireSimulation();
    setRlPath([]);
    setFireCellBboxes([]);
    if (graphMode !== "cell") {
      setCellDisplayNodes([]);
      return;
    }
    if (!topology) return;

    let active = true;
    axios
      .get(`${API_BASE}/graph-meta?mode=cell`)
      .then((res) => {
        if (!active) return;
        const nodes = res.data?.cell_bboxes || [];
        setCellDisplayNodes(nodes);
        if (nodes.length) {
          setFireCellBboxes(nodes);
        } else {
          setError("Cell model not available. Load IFC with wires first.");
        }
      })
      .catch((err) => {
        if (!active) return;
        setError(err.response?.data?.detail || err.message || "Cell model load failed.");
      });

    return () => {
      active = false;
    };
  }, [graphMode, topology]);

  useEffect(() => {
    let timer;
    if (lastIncludePath && lastIfcFile) {
      timer = setTimeout(() => {
        uploadIfc(lastIfcFile, true);
      }, 400);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [floorTilt, floorMaxZ, floorMinArea, lastIncludePath, lastIfcFile]);


  const formatPoint = (point) => {
    if (!point || point.length < 3) return "not set";
    return point.map((v) => Number(v).toFixed(2)).join(", ");
  };

  const formatDictValue = (value) => {
    if (value === null || value === undefined) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return value;
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  };

  return (
    <div className="app-root">
      <style dangerouslySetInnerHTML={spinnerStyle} />
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
          <label className="file-upload-button">
            <input
              type="file"
              accept=".ifc"
              onChange={(e) => handleIfcUpload(e, false)}
              className="file-upload-input"
            />
            <span>Load IFC</span>
          </label>
          <label className="file-upload-button">
            <input
              type="file"
              accept=".ifc"
              onChange={(e) => handleIfcUpload(e, true)}
              className="file-upload-input"
            />
            <span>Load IFC, generate wires, and calculate shortest path</span>
          </label>
          <button
            type="button"
            className="file-upload-button"
            onClick={() => setTranslucent((v) => !v)}
          >
            {translucent ? "Show opaque" : "Make translucent"}
          </button>
          <button
            type="button"
            className="file-upload-button"
            onClick={() => setShowFaces((v) => !v)}
          >
            {showFaces ? "Hide meshes" : "Show meshes"}
          </button>
          <button
            type="button"
            className="file-upload-button"
            onClick={() => setShowVerts((v) => !v)}
          >
            {showVerts ? "Hide vertices" : "Show vertices"}
          </button>
          <button
            type="button"
            className="file-upload-button"
            onClick={() => setWireframe((v) => !v)}
          >
            {wireframe ? "Disable wireframe" : "Enable wireframe"}
          </button>
          <button
            type="button"
            className="file-upload-button"
            onClick={() => setFitRequest((v) => v + 1)}
          >
            Fit view
          </button>
          {fileName && (
            <span className="file-chip" title={fileName}>
              {fileName}
            </span>
          )}
          <div className="slider-panel">
            <label>
              Tilt min
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={floorTilt}
                onChange={(e) => setFloorTilt(Number(e.target.value))}
              />
              <span className="slider-value">{floorTilt.toFixed(2)}</span>
            </label>
            <label>
              Max z-span (m)
              <input
                type="range"
                min="0"
                max="3"
                step="0.1"
                value={floorMaxZ}
                onChange={(e) => setFloorMaxZ(Number(e.target.value))}
              />
              <span className="slider-value">{floorMaxZ.toFixed(2)}</span>
            </label>
            <label>
              Min area (m?)
              <input
                type="range"
                min="1"
                max="50"
                step="1"
                value={floorMinArea}
                onChange={(e) => setFloorMinArea(Number(e.target.value))}
              />
              <span className="slider-value">{floorMinArea.toFixed(0)}</span>
            </label>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="app-main">
        <div className="viewer-panel" style={{ position: "relative" }}>
          {topology ? (
            <TopologyViewer
              data={displayTopology || topology}
              selection={selection}
              onSelectionChange={handleSelectionChange}
              showFaces={showFaces}
              showVerts={showVerts}
              wireframe={wireframe}
              fitRequest={fitRequest}
              fireVertices={fireVertexIds}
              fireEdges={fireEdgeIds}
              pathEdges={pathEdgeIds}
              pathVertices={pathVertexIds}
              extraVertices={graphMode === "cell" ? cellDisplayVertices : emptyExtras}
              extraVerticesVisible={graphMode === "cell"}
            />
          ) : (
            <div className="viewer-placeholder">
              <div className="viewer-placeholder-card">
                <h2>Welcome to TopologicStudio</h2>
                <p>
                  IFC viewer and fire egress calculator for TopologicPy models.
                </p>
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
          {loading && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                background: "rgba(10,12,24,0.55)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                zIndex: 15,
                backdropFilter: "blur(2px)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: "12px",
                  color: "#fff",
                  fontWeight: 600,
                }}
              >
                <img
                  src={logoImg}
                  alt="Loading"
                  style={{ width: "64px", height: "64px", opacity: 0.9 }}
                />
                <div
                  style={{
                    width: "36px",
                    height: "36px",
                    borderRadius: "50%",
                    border: "4px solid rgba(255,255,255,0.35)",
                    borderTopColor: "#7ad7ff",
                    animation: "spin 0.9s linear infinite",
                  }}
                />
                <span>Loading?</span>
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

          <div className="sidebar-section">
            <div className="sidebar-section-header">Fire + egress</div>
            <div className="sidebar-section-body">
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                <div>
                  <span className="sidebar-label">Graph mode</span>
                  <select
                    value={graphMode}
                    onChange={(e) => setGraphMode(e.target.value)}
                    style={{
                      width: "100%",
                      padding: "6px 8px",
                      borderRadius: "8px",
                      border: "1px solid rgba(148, 163, 184, 0.4)",
                      background: "rgba(15, 23, 42, 0.85)",
                      color: "#e5e7eb",
                    }}
                  >
                    <option value="wire">Wire grid (detailed)</option>
                    <option value="cell">Cell model (simple)</option>
                  </select>
                </div>

                <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                  <button
                    type="button"
                    className="file-upload-button"
                    onClick={handlePickStart}
                    style={{ flex: "1 1 120px", justifyContent: "center" }}
                  >
                    {pickMode === "start" ? "Click in viewer..." : "Set start"}
                  </button>
                  <button
                    type="button"
                    className="file-upload-button"
                    onClick={handlePickExit}
                    style={{ flex: "1 1 120px", justifyContent: "center" }}
                  >
                    {pickMode === "exit" ? "Click in viewer..." : "Set exit"}
                  </button>
                  <button
                    type="button"
                    className="file-upload-button"
                    onClick={clearStartExit}
                    style={{ flex: "1 1 120px", justifyContent: "center" }}
                  >
                    Clear points
                  </button>
                </div>

                <div>
                  <span className="sidebar-label">Start</span>
                  <span className="sidebar-value sidebar-value-mono">
                    {startId || formatPoint(startPoint)}
                  </span>
                </div>
                <div>
                  <span className="sidebar-label">Exit</span>
                  <span className="sidebar-value sidebar-value-mono">
                    {exitId || formatPoint(exitPoint)}
                  </span>
                </div>

                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <input
                      type="checkbox"
                      checked={fireUsePrecompute}
                      onChange={(e) => setFireUsePrecompute(e.target.checked)}
                    />
                    <span className="sidebar-value">Precompute timeline</span>
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <input
                      type="checkbox"
                      checked={rlUseFire}
                      onChange={(e) => setRlUseFire(e.target.checked)}
                    />
                    <span className="sidebar-value">Use fire in RL</span>
                  </label>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                  <label>
                    <span className="sidebar-label">Step delay (ms)</span>
                    <input
                      type="number"
                      min="50"
                      step="50"
                      value={fireDelayMs}
                      onChange={(e) => setFireDelayMs(Number(e.target.value))}
                      style={{ width: "100%", padding: "6px", borderRadius: "6px" }}
                    />
                  </label>
                  <label>
                    <span className="sidebar-label">Fire steps</span>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={fireMaxSteps}
                      onChange={(e) => setFireMaxSteps(Number(e.target.value))}
                      style={{ width: "100%", padding: "6px", borderRadius: "6px" }}
                    />
                  </label>
                  <label>
                    <span className="sidebar-label">RL episodes</span>
                    <input
                      type="number"
                      min="10"
                      step="10"
                      value={rlEpisodes}
                      onChange={(e) => setRlEpisodes(Number(e.target.value))}
                      style={{ width: "100%", padding: "6px", borderRadius: "6px" }}
                    />
                  </label>
                  <label>
                    <span className="sidebar-label">RL max steps</span>
                    <input
                      type="number"
                      min="10"
                      step="10"
                      value={rlMaxSteps}
                      onChange={(e) => setRlMaxSteps(Number(e.target.value))}
                      style={{ width: "100%", padding: "6px", borderRadius: "6px" }}
                    />
                  </label>
                </div>

                <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                  <button
                    type="button"
                    className="file-upload-button"
                    onClick={startFireSimulation}
                    disabled={fireRunning}
                    style={{ flex: "1 1 120px", justifyContent: "center" }}
                  >
                    {fireRunning ? "Fire running" : "Start fire"}
                  </button>
                  <button
                    type="button"
                    className="file-upload-button"
                    onClick={stopFireSimulation}
                    disabled={!fireRunning}
                    style={{ flex: "1 1 120px", justifyContent: "center" }}
                  >
                    Stop fire
                  </button>
                  <button
                    type="button"
                    className="file-upload-button"
                    onClick={trainRlPath}
                    disabled={rlLoading}
                    style={{ flex: "1 1 120px", justifyContent: "center" }}
                  >
                    {rlLoading ? "Training..." : "Train RL path"}
                  </button>
                </div>

                <div className="sidebar-value" style={{ opacity: 0.8 }}>
                  Fire step: {fireStep} {fireRunning ? (fireUsePrecompute ? "(precomputed)" : "(streaming)") : ""}
                </div>
              </div>
            </div>
          </div>

          {!selection && (
            <div className="sidebar-empty">
              <p>Click on a face, edge, or vertex to inspect its metadata.</p>
              <p className="sidebar-empty-hint">
                Repeated clicks on the same location will cycle through the
                hierarchy (CellComplex -> Cell -> Shell -> Face -> Edge -> Vertex).
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


