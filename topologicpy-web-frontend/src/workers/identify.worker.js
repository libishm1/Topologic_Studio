/**
 * IFCNet's geometry classifier off the main thread (src/core/identifyGeometry.js).
 * onnxruntime-web is imported from a pinned CDN URL on the first "init", so
 * neither it nor its WASM is in the app bundle or loaded before it is asked
 * for. One session per worker; inputs arrive as one batch of point clouds
 * [count, points, 3] and are run one by one (the model's shape is static).
 *
 * Messages: {id, type: "init", payload: {modelUrl, points, backend, ortBase}}
 *           {id, type: "classify", payload: {batch: Float32Array, count}}
 * Answers:  {id, ok: true, result} or {id, ok: false, error}
 */
let ort = null;
let session = null;
let points = 0;
let info = null;

async function init({ modelUrl, points: n, backend = "wasm", ortBase }) {
  const t0 = performance.now();
  const wantGpu = backend === "webgpu" && typeof navigator !== "undefined" && "gpu" in navigator;
  const script = ortBase + (wantGpu ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs");
  try {
    ort = await import(/* @vite-ignore */ script);
  } catch (error) {
    throw new Error(`Could not load onnxruntime-web (${error?.message || error}). Are you offline? The name rules' answers stand.`);
  }
  ort.env.wasm.wasmPaths = ortBase;
  // Threads need cross-origin isolation, which static hosts (GitHub Pages) cannot set.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`The geometry model did not download (${response.status} ${response.statusText}).`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const providers = wantGpu ? ["webgpu", "wasm"] : ["wasm"];
  try {
    session = await ort.InferenceSession.create(bytes, { executionProviders: providers, graphOptimizationLevel: "all" });
  } catch (error) {
    if (!wantGpu) throw error;
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }
  points = n;
  info = { backend: wantGpu ? "webgpu" : "wasm", threads: ort.env.wasm.numThreads, bytes: bytes.length, loadMs: performance.now() - t0 };
  return info;
}

async function classify({ batch, count }) {
  if (!session) throw new Error("The geometry classifier is not loaded.");
  const per = points * 3;
  let classes = 0;
  let logits = null;
  for (let i = 0; i < count; i += 1) {
    const input = new ort.Tensor("float32", batch.subarray(i * per, (i + 1) * per), [1, points, 3]);
    const out = await session.run({ points: input });
    const data = out.logits.data;
    if (!logits) {
      classes = data.length;
      logits = new Float32Array(count * classes);
    }
    for (let c = 0; c < classes; c += 1) {
      const value = data[c];
      // A NaN would read as a confident answer downstream; refuse it here.
      if (!Number.isFinite(value)) throw new Error("The geometry classifier returned a non-finite value.");
      logits[i * classes + c] = value;
    }
  }
  return { logits: logits || new Float32Array(0), classes };
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data || {};
  try {
    if (type === "init") self.postMessage({ id, ok: true, result: await init(payload) });
    else if (type === "classify") {
      const result = await classify(payload);
      self.postMessage({ id, ok: true, result }, [result.logits.buffer]);
    } else throw new Error(`Unknown message type: ${type}`);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
