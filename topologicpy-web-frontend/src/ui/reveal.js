import { useSyncExternalStore } from "react";

/**
 * Panel sections to open from outside the panel: {ids, seq}, matched against
 * each Section's id (primitives.jsx). A finished guided run opens the ones
 * that hold its result (App.jsx), and ends the request (null) once they have
 * drawn, so a section shown again later opens as it usually does. Each
 * request opens them once, as a click would. A small store rather than a
 * context, so nothing wraps the panel.
 */
let current = null;
let seq = 0;
const listeners = new Set();

export function revealSections(ids) {
  current = ids ? { ids, seq: (seq += 1) } : null;
  listeners.forEach((listener) => listener());
}

const subscribe = (listener) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** The latest request, or null. */
export const useReveal = () => useSyncExternalStore(subscribe, () => current);
