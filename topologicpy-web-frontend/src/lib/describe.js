/**
 * "Describe this model": the language model that writes the description from
 * the facts in core/describeFacts.js.
 *
 *   on this device  Chrome's built-in Prompt API (Gemini Nano, Chrome 148 or
 *                   later on a desktop computer): nothing leaves the computer;
 *   Cloudflare      the Worker in cloudflare/describe-worker, which runs a
 *                   Workers AI model; only when the site is built with its
 *                   address (VITE_DESCRIBE_URL), and the page shows the facts
 *                   before they are sent.
 */
import { DESCRIBE_INSTRUCTIONS } from "../core/describeFacts.js";

export const CLOUD_URL = (import.meta.env?.VITE_DESCRIBE_URL || "").replace(/\/+$/, "");

const LANGUAGES = {
  expectedInputs: [{ type: "text", languages: ["en"] }],
  expectedOutputs: [{ type: "text", languages: ["en"] }],
};

/** "available", "downloadable", "downloading", "unavailable", or "unsupported" (no Prompt API). */
export async function onDeviceStatus() {
  const LM = globalThis.LanguageModel;
  if (!LM?.availability) return "unsupported";
  try {
    return await LM.availability(LANGUAGES);
  } catch {
    return "unavailable";
  }
}

/**
 * The description, written on this device; `onText` gets the text so far as
 * it streams, `onProgress` the model download's share (0 to 1) the first time.
 */
export async function describeOnDevice(facts, { onText, onProgress, signal } = {}) {
  const LM = globalThis.LanguageModel;
  if (!LM?.create) throw new Error("This browser has no built-in language model; Chrome 148 or later on a desktop computer has one.");
  const session = await LM.create({
    ...LANGUAGES,
    initialPrompts: [{ role: "system", content: DESCRIBE_INSTRUCTIONS }],
    signal,
    monitor(m) {
      m.addEventListener("downloadprogress", (e) => onProgress?.(e.loaded));
    },
  });
  try {
    let text = "";
    for await (const chunk of session.promptStreaming(`Facts:\n${facts}`, { signal })) {
      // Chrome streams new text; an older build streamed the whole text so far.
      text = chunk.startsWith(text) && text ? chunk : text + chunk;
      onText?.(text);
    }
    return text;
  } finally {
    session.destroy();
  }
}

/**
 * The description, written by the Cloudflare Worker (CLOUD_URL) from the facts
 * sent to it; `token` is the human check's (lib/turnstile.js), when the site has one.
 */
export async function describeInCloud(facts, { url = CLOUD_URL, signal, token = null } = {}) {
  if (!url) throw new Error("This site has no Cloudflare describer set up (VITE_DESCRIBE_URL).");
  const response = await fetch(`${url}/describe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(token ? { facts, turnstile: token } : { facts }),
    signal,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `The describer answered ${response.status}.`);
  return { text: body.text, model: body.model };
}

/**
 * Suggestions for a storey drafted from a floor-plan drawing (draftSummary in
 * core/drawingToLod1.js): rooms to merge, and names and classes for rooms,
 * already checked against the summary by the Worker. The page offers each
 * one; none is applied unless the user accepts it.
 */
export async function suggestRoomsInCloud(summary, { url = CLOUD_URL, signal, token = null } = {}) {
  if (!url) throw new Error("This site has no Cloudflare describer set up (VITE_DESCRIBE_URL).");
  const response = await fetch(`${url}/rooms`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(token ? { summary, turnstile: token } : { summary }),
    signal,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `The describer answered ${response.status}.`);
  return { merge: body.merge || [], names: body.names || [], dropped: body.dropped || 0, model: body.model };
}
