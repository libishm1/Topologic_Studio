/**
 * Folding a long field hint (ui/primitives.jsx Hint). The interface audit
 * (2026-11-11) found hints carrying method notes and sources sitting in the
 * panel for good, up to 1,300 words on one step; a hint now shows its first
 * sentence and the rest behind "More".
 */

/** Hints up to this many words show whole; longer ones show their first sentence and "More". */
export const HINT_WORDS = 25;

/**
 * A hint's first sentence and the rest, or [text, ""] when it is short. A
 * sentence ends at ". ", "? " or "! " before a capital, so "et al. (2017)" and
 * "1.3 persons" do not end one; a first sentence over 30 words is cut at 22.
 */
export function splitHint(text) {
  const words = text.trim().split(/\s+/);
  if (words.length <= HINT_WORDS) return [text, ""];
  const m = text.match(/^(.+?[.!?])\s+(?=[A-Z])/);
  if (m && m[1].split(/\s+/).length <= 30) return [m[1], text.slice(m[0].length)];
  return [`${words.slice(0, 22).join(" ")}…`, words.slice(22).join(" ")];
}
