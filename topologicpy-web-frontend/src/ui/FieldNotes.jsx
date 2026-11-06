/**
 * Field notes and hints on screen (core/fieldNotes.js, state/useFieldNotes.js):
 * the card at the foot of the workflow panel, the header button with the
 * count of notes found, and the list of notes.
 */
import React, { useEffect, useRef } from "react";

import { FIELD_NOTES, FIELD_NOTE_COUNT } from "../core/fieldNotes.js";
import { Button, Icon, Toggle } from "./primitives.jsx";

/**
 * One card at the foot of the panel: a field note just found, else the hint
 * for this step, else nothing. Announced politely, never over the 3D view.
 */
export function HintCard({ hint, latest, onAction, onDismiss, onCloseLatest, onOpenNotes }) {
  if (latest) {
    return (
      <div className="hint hint--found" role="status" aria-live="polite">
        <div className="hint__head">
          <Icon name="book" size={14} />
          <strong>Field note found</strong>
        </div>
        <p className="hint__teaser">{latest.teaser}</p>
        <p className="hint__text">{latest.text}</p>
        <div className="hint__actions">
          <Button size="sm" onClick={onOpenNotes}>
            Open field notes
          </Button>
          <Button size="sm" variant="ghost" onClick={onCloseLatest}>
            Close
          </Button>
        </div>
      </div>
    );
  }
  if (!hint) return null;
  return (
    <div className="hint" role="status" aria-live="polite">
      <div className="hint__head">
        <Icon name="info" size={14} />
        <strong>Hint</strong>
      </div>
      <p className="hint__text">{hint.text}</p>
      <div className="hint__actions">
        {hint.action && (
          <Button size="sm" onClick={() => onAction(hint)}>
            {hint.action.label}
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={() => onDismiss(hint.id)}>
          Got it
        </Button>
      </div>
    </div>
  );
}

/** The header button: how many notes are found, out of how many. */
export function FieldNotesButton({ found, onOpen }) {
  const count = Object.keys(found || {}).length;
  return (
    <Button variant="ghost" onClick={onOpen} aria-haspopup="dialog" aria-label={`Field notes, ${count} of ${FIELD_NOTE_COUNT} found`}>
      <Icon name="book" size={18} />
      <span className="header__label">Field notes</span>
      <span className="field-notes__count mono">
        {count}/{FIELD_NOTE_COUNT}
      </span>
    </Button>
  );
}

/** The notes, found ones with what was found and the rest as their questions; and the hints switch. */
export function FieldNotesDialog({ open, onClose, found, hints, onHints, onResetHints }) {
  const sheetRef = useRef(null);
  const headingRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const before = document.activeElement;
    headingRef.current?.focus();
    // Captured before the app's single-key shortcuts, as in the Methods dialog.
    const onKey = (event) => {
      event.stopPropagation();
      if (event.key === "Escape") onClose();
      else if (event.key === "Tab") {
        const focusable = sheetRef.current?.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])');
        if (!focusable?.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      before?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;
  const count = Object.keys(found || {}).length;
  return (
    <div
      className="field-notes"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="field-notes__sheet card" ref={sheetRef} role="dialog" aria-modal="true" aria-labelledby="field-notes-title">
        <div className="field-notes__bar">
          <h2 id="field-notes-title" ref={headingRef} tabIndex={-1}>
            Field notes <span className="mono">{count}/{FIELD_NOTE_COUNT}</span>
          </h2>
          <Button onClick={onClose}>
            <Icon name="close" size={16} />
            Close
          </Button>
        </div>
        <p className="field__hint">
          Things to find out about a building as you work through the steps. Each is a question until you find the answer in the app; then it is
          written down here, with what you found.
        </p>
        <ol className="field-notes__list">
          {FIELD_NOTES.map((note) => {
            const entry = found?.[note.id];
            return (
              <li key={note.id} data-found={Boolean(entry)}>
                <span className="field-notes__mark" aria-hidden="true">
                  {entry ? <Icon name="check" size={14} /> : "?"}
                </span>
                <div>
                  <strong>{note.teaser}</strong>
                  {entry ? <p>{entry.text}</p> : <p className="field__hint">Not found yet.</p>}
                </div>
              </li>
            );
          })}
        </ol>
        <div className="field-notes__foot">
          <Toggle label="Show hints as I work" checked={hints} onChange={onHints} />
          <Button size="sm" variant="ghost" onClick={onResetHints}>
            Show closed hints again
          </Button>
        </div>
      </div>
    </div>
  );
}
