/**
 * The field notes and hints a user has met, kept in this browser
 * (core/fieldNotes.js holds what they say). Survives reloads; a private
 * window or blocked storage starts afresh each time.
 *
 *   found      id -> {text, at}: notes found, with the text said then
 *   used       features used at least once: their hints no longer apply
 *   dismissed  hints closed by the user
 *   hints      whether hints are shown (on by default)
 *   latest     the note found most recently in this session, shown until closed
 */
import { useCallback, useMemo, useRef, useState } from "react";
import { fieldNote } from "../core/fieldNotes.js";

const KEY = "topologic-studio.fieldNotes";

function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || "null");
    if (raw && raw.v === 1) return raw;
  } catch {
    // Unreadable or blocked: start afresh.
  }
  return { v: 1, hints: true, found: {}, used: [], dismissed: [] };
}

function save(state) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // Private windows and blocked storage: kept for this session only.
  }
}

export function useFieldNotes() {
  const [state, setState] = useState(load);
  const [latest, setLatest] = useState(null);
  // The state as last set, read at once: an updater passed to setState may
  // run later, or twice, so it cannot say whether a note was new.
  const stateRef = useRef(state);

  const update = useCallback((change) => {
    const current = stateRef.current;
    const next = change(current);
    if (next === current) return false;
    stateRef.current = next;
    save(next);
    setState(next);
    return true;
  }, []);

  // A note is found once; finding it again does nothing.
  const markFound = useCallback(
    (id, detail = {}) => {
      const note = fieldNote(id);
      if (!note) return;
      let text;
      try {
        text = note.text(detail);
      } catch {
        return;
      }
      const isNew = update((current) =>
        current.found[id] ? current : { ...current, found: { ...current.found, [id]: { text, at: new Date().toISOString() } } },
      );
      if (isNew) setLatest({ id, teaser: note.teaser, text });
    },
    [update],
  );

  const markUsed = useCallback(
    (feature) => update((current) => (current.used.includes(feature) ? current : { ...current, used: [...current.used, feature] })),
    [update],
  );

  const dismiss = useCallback(
    (hintId) => update((current) => (current.dismissed.includes(hintId) ? current : { ...current, dismissed: [...current.dismissed, hintId] })),
    [update],
  );

  const setHints = useCallback((on) => update((current) => ({ ...current, hints: Boolean(on) })), [update]);
  // Hints come back; found notes and used features stay.
  const resetHints = useCallback(() => update((current) => ({ ...current, hints: true, dismissed: [] })), [update]);
  const closeLatest = useCallback(() => setLatest(null), []);

  const used = useMemo(() => new Set(state.used), [state.used]);
  const dismissed = useMemo(() => new Set(state.dismissed), [state.dismissed]);

  return {
    found: state.found,
    used,
    dismissed,
    hints: state.hints,
    latest,
    markFound,
    markUsed,
    dismiss,
    setHints,
    resetHints,
    closeLatest,
  };
}
