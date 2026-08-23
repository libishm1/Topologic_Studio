/**
 * Timing markers.
 *
 * The roadmap asks to measure before rewriting, so these stay in the shipped
 * build behind a flag rather than being dev-only. `VITE_PERF_LOG=1` prints to
 * the console; the collected phases are also surfaced in the app's own status
 * bar so a non-developer can report a number.
 */

const ENABLED =
  (import.meta.env.VITE_PERF_LOG ?? (import.meta.env.DEV ? "1" : "")) === "1";

const listeners = new Set();

export function onPerf(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function emit(entry) {
  if (ENABLED) {
    const detail = entry.detail ? ` ${JSON.stringify(entry.detail)}` : "";
    console.log(`PERF ${entry.label} ${entry.ms.toFixed(1)}ms${detail}`);
  }
  listeners.forEach((listener) => {
    try {
      listener(entry);
    } catch {
      /* a broken listener must not break the measured code */
    }
  });
}

/** Start a timer. The returned function stops it and records the entry. */
export function mark(label) {
  const t0 = performance.now();
  return (detail) => {
    const ms = performance.now() - t0;
    emit({ label, ms, detail, at: Date.now() });
    return ms;
  };
}

/** Time an async operation and return its result. */
export async function timed(label, fn, detail) {
  const done = mark(label);
  try {
    return await fn();
  } finally {
    done(typeof detail === "function" ? detail() : detail);
  }
}

/** Collects named phases for one multi-step operation. */
export class PhaseTimer {
  constructor(name) {
    this.name = name;
    this.t0 = performance.now();
    this.phases = {};
  }

  async run(phase, fn) {
    const start = performance.now();
    try {
      return await fn();
    } finally {
      this.phases[phase] = (this.phases[phase] || 0) + (performance.now() - start);
    }
  }

  get total() {
    return performance.now() - this.t0;
  }

  finish(detail) {
    const entry = {
      label: this.name,
      ms: this.total,
      detail: { ...this.phases, ...detail },
      at: Date.now(),
    };
    emit(entry);
    return entry;
  }
}

export function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatMs(ms) {
  if (ms === null || ms === undefined) return "-";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}
