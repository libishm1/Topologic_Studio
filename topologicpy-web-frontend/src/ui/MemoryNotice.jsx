/**
 * The notice on running out of memory or storage (core/memoryNotice.js): one
 * sentence on what happened and how much the caches hold, the button that
 * clears the cached runs and models (lib/caches.js clearCaches), and Dismiss;
 * after clearing, how much was freed. A toast among the toasts, not a modal.
 */
import React from "react";

import { Button } from "./primitives.jsx";

const TITLES = {
  quota: "Storage full",
  memory: "Out of memory",
  gpu: "GPU out of memory",
  storage: "Storage nearly full",
  heap: "Memory nearly full",
};

export function MemoryNotice({ notice, onClear, onDismiss }) {
  return (
    <div className="toast toast--warning memory-notice" data-kind={notice.kind}>
      <div className="toast__body">
        <div className="toast__title">{TITLES[notice.kind] || "Memory"}</div>
        <div className="toast__message">{notice.freed || notice.text}</div>
        <div className="memory-notice__actions">
          {!notice.freed && (
            <Button size="sm" variant="primary" busy={notice.busy} onClick={onClear}>
              Clear cached runs and models
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={onDismiss}>
            Dismiss
          </Button>
        </div>
      </div>
    </div>
  );
}
