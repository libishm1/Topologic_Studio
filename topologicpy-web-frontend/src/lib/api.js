/**
 * Backend client.
 *
 * Plain `fetch` rather than axios: one less dependency, and the streaming and
 * abort behaviour the fire simulation needs is native. Every call returns
 * parsed JSON or throws an `ApiError` carrying the server's `detail`, so the
 * UI can show what actually went wrong instead of "Request failed".
 */

export const API_BASE = (
  import.meta.env.VITE_API_BASE || "http://localhost:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, url = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.url = url;
  }
}

async function request(path, { method = "GET", body, signal, timeout = 120000 } = {}) {
  const url = `${API_BASE}${path}`;
  const controller = new AbortController();
  const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new ApiError("Request cancelled or timed out.", { url });
    }
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE}. Is it running?`,
      { url },
    );
  } finally {
    if (timer) clearTimeout(timer);
  }

  if (!response.ok) {
    let detail = null;
    try {
      const payload = await response.json();
      detail = payload?.detail ?? payload;
    } catch {
      detail = await response.text().catch(() => null);
    }
    const message =
      typeof detail === "string" && detail
        ? detail
        : `Request failed with status ${response.status}.`;
    throw new ApiError(message, { status: response.status, detail, url });
  }

  if (response.status === 204) return null;
  return response.json();
}

/** Decode a base64 payload into a typed array without an intermediate string copy. */
function decodeBase64(b64, TypedArray) {
  if (!b64) return new TypedArray(0);
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new TypedArray(bytes.buffer);
}

/**
 * Turn a graph response into typed arrays.
 *
 * The server sends nodes and edges as base64 float32/uint32 rather than JSON
 * numbers: about a sixth of the bytes, and no per-number parse on the main
 * thread. On a 20k-node graph that is the difference between a visible stall
 * and an instant overlay.
 */
export function decodeGraph(payload) {
  return {
    graphId: payload.graph_id,
    stats: payload.stats,
    upAxis: payload.up_axis,
    bounds: payload.bounds,
    timings: payload.timings || {},
    nodes: decodeBase64(payload.nodes_b64, Float32Array),
    edges: decodeBase64(payload.edges_b64, Uint32Array),
    kinds: decodeBase64(payload.kinds_b64, Uint8Array),
  };
}

export const api = {
  health: (signal) => request("/health", { signal, timeout: 8000 }),
  capabilities: (signal) => request("/api/capabilities", { signal, timeout: 8000 }),

  buildGraph: (payload, signal) =>
    request("/api/ifc/graph", { method: "POST", body: payload, signal }).then(decodeGraph),

  getGraph: (graphId, signal) =>
    request(`/api/ifc/graph/${graphId}`, { signal }).then(decodeGraph),

  deleteGraph: (graphId, signal) =>
    request(`/api/ifc/graph/${graphId}`, { method: "DELETE", signal }),

  findPath: (payload, signal) =>
    request("/api/ifc/path", { method: "POST", body: payload, signal }),

  comparePath: (payload, signal) =>
    request("/api/ifc/path/compare", { method: "POST", body: payload, signal }),

  fireTimeline: (payload, signal) =>
    request("/api/fire/timeline", { method: "POST", body: payload, signal }),

  trainRl: (payload, signal) =>
    request("/api/rl/train", { method: "POST", body: payload, signal, timeout: 300000 }),
};

/**
 * Open the fire simulation SSE stream.
 *
 * Returns a handle with `close()`. `EventSource` reconnects automatically on
 * error, which for a one-shot simulation means it silently restarts from step
 * zero; the guard below closes it instead and reports the failure once.
 */
export function openFireStream(params, handlers = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      query.set(key, String(value));
    }
  });

  const source = new EventSource(`${API_BASE}/api/fire/stream?${query.toString()}`);
  let finished = false;

  const close = () => {
    if (!finished) {
      finished = true;
      source.close();
    }
  };

  source.onmessage = (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message.type === "done") {
      close();
      handlers.onDone?.();
      return;
    }
    handlers.onMessage?.(message);
  };

  source.onerror = () => {
    if (finished) return;
    close();
    handlers.onError?.(new ApiError("The fire simulation stream was interrupted."));
  };

  return { close, get closed() { return finished; } };
}

/**
 * Legacy TopologicPy JSON-contract endpoints.
 *
 * Multipart and raw-JSON shapes that predate the IFC egress pipeline. Kept
 * because the backend still serves them and they are how TopologicPy exports
 * are inspected.
 */
export const legacyApi = {
  uploadTopology: (json, signal) =>
    request("/upload-topology", { method: "POST", body: json, signal, timeout: 300000 }),

  async uploadIfc(file, { includePath = false, tiltMin, maxZSpan, minFloorArea } = {}, signal) {
    const query = new URLSearchParams({ include_path: String(includePath) });
    if (includePath) {
      query.set("tilt_min", String(tiltMin));
      query.set("max_z_span", String(maxZSpan));
      query.set("min_floor_area", String(minFloorArea));
    }
    const form = new FormData();
    form.append("file", file);

    const response = await fetch(`${API_BASE}/upload-ifc?${query}`, {
      method: "POST",
      body: form,
      signal,
    });
    if (!response.ok) {
      let detail = null;
      try {
        detail = (await response.json())?.detail;
      } catch {
        detail = await response.text().catch(() => null);
      }
      throw new ApiError(
        typeof detail === "string" && detail
          ? detail
          : `Server-side IFC processing failed (${response.status}).`,
        { status: response.status, detail },
      );
    }
    return response.json();
  },
};
