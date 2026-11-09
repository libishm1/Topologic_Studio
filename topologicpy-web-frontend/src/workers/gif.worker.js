/**
 * A GIF written as its frames arrive (lib/recorder.js), off the page's
 * thread: each frame is quantised to its own 256 colours and added at once
 * (gifenc, MIT), so only the GIF's bytes are kept, not the frames.
 *
 * Messages: { type: "start", delay } (ms a frame), { type: "frame", data,
 * width, height } (RGBA bytes, transferred), { type: "finish" }; answers
 * { type: "frame", frames } after each and { type: "done", bytes } at the end.
 */
import { GIFEncoder, applyPalette, quantize } from "gifenc";

let gif = null;
let delay = 100;
let frames = 0;

self.onmessage = (event) => {
  const { type } = event.data || {};
  if (type === "start") {
    gif = GIFEncoder();
    delay = event.data.delay || 100;
    frames = 0;
    return;
  }
  if (!gif) return;
  if (type === "frame") {
    const { data, width, height } = event.data;
    const rgba = new Uint8ClampedArray(data);
    const palette = quantize(rgba, 256, { format: "rgb565" });
    const index = applyPalette(rgba, palette, "rgb565");
    // The first frame says to loop for ever.
    gif.writeFrame(index, width, height, { palette, delay, ...(frames === 0 ? { repeat: 0 } : {}) });
    frames += 1;
    self.postMessage({ type: "frame", frames });
    return;
  }
  if (type === "finish") {
    gif.finish();
    const bytes = gif.bytes().slice();
    gif = null;
    self.postMessage({ type: "done", bytes, frames }, [bytes.buffer]);
  }
};
