/**
 * Recording the 3D view as a GIF: the fire, the smoke and the people as they
 * play. Each frame is read from the view's canvas just after it is drawn
 * (ViewerManager.onRendered: a WebGL canvas keeps its picture only until the
 * page hands it to the screen), scaled down on a small canvas and sent to
 * workers/gif.worker.js, which adds it to the GIF at once. A phone records a
 * smaller, shorter GIF than a desktop, so it keeps its pace and memory.
 */
import { isPhone } from "./hardware.js";

/** Width (px), frames a second and longest recording (s) for this device. */
export function recordingLimits(nav = globalThis.navigator) {
  return isPhone(nav) ? { width: 360, fps: 8, maxSeconds: 15 } : { width: 640, fps: 10, maxSeconds: 30 };
}

/**
 * Starts recording `viewer`'s view. `onTick({ seconds, frames, encoded })`
 * reports progress; recording stops by itself after maxSeconds.
 * @returns {{ stop: () => Promise<{ blob: Blob, frames: number, seconds: number }>, cancel: () => void }}
 */
export function recordView(viewer, { width, fps, maxSeconds, onTick = null } = recordingLimits()) {
  const source = viewer.canvas();
  if (!source) throw new Error("The view is not ready to record.");
  const worker = new Worker(new URL("../workers/gif.worker.js", import.meta.url), { type: "module" });
  const scale = Math.min(1, width / Math.max(1, source.width));
  const w = Math.max(2, Math.round(source.width * scale)), h = Math.max(2, Math.round(source.height * scale));
  const frame = document.createElement("canvas");
  frame.width = w;
  frame.height = h;
  const ctx = frame.getContext("2d", { willReadFrequently: true });
  const t0 = performance.now();
  let last = -Infinity, frames = 0, encoded = 0, stopped = false;
  let finish = null;
  const done = new Promise((resolve, reject) => {
    finish = { resolve, reject };
  });
  worker.onmessage = (event) => {
    if (event.data.type === "frame") {
      encoded = event.data.frames;
      onTick?.({ seconds: (performance.now() - t0) / 1000, frames, encoded, stopped });
    } else if (event.data.type === "done") {
      worker.terminate();
      finish.resolve({ blob: new Blob([event.data.bytes], { type: "image/gif" }), frames: event.data.frames, seconds: frames / fps });
    }
  };
  worker.onerror = (event) => {
    worker.terminate();
    finish.reject(new Error(event?.message || "The GIF could not be made."));
  };
  worker.postMessage({ type: "start", delay: Math.round(1000 / fps) });
  const capture = () => {
    const now = performance.now();
    if (stopped || now - last < 1000 / fps - 2) return;
    last = now;
    ctx.drawImage(source, 0, 0, w, h);
    const { data } = ctx.getImageData(0, 0, w, h);
    worker.postMessage({ type: "frame", data: data.buffer, width: w, height: h }, [data.buffer]);
    frames += 1;
    if ((now - t0) / 1000 >= maxSeconds) stop();
  };
  let unsubscribe = () => {};
  const stop = () => {
    if (!stopped) {
      stopped = true;
      unsubscribe();
      worker.postMessage({ type: "finish" });
    }
    return done;
  };
  const cancel = () => {
    stopped = true;
    unsubscribe();
    worker.terminate();
    finish.reject(new Error("cancelled"));
  };
  unsubscribe = viewer.onRendered(capture);
  return { stop, cancel, done };
}
