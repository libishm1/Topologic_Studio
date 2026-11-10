/**
 * Cloudflare Turnstile, the "I'm human" check in front of the site's Cloudflare
 * Worker (cloudflare/describe-worker), so its free AI allowance is spent on
 * people. Only when the site is built with a site key (VITE_TURNSTILE_SITE_KEY),
 * and only where a cloud feature is offered: the panel mounts the check (one
 * slim strip, in the app's theme) while Cloudflare is chosen, and its script
 * loads then. It checks at once, usually with no click, so a token is ready
 * when the request is sent; tokens are single-use, so the check runs again
 * after each request.
 *
 * The Worker checks the token with its secret (TURNSTILE_SECRET); without one
 * it does not ask for a token.
 */

export const TURNSTILE_SITE_KEY = import.meta.env?.VITE_TURNSTILE_SITE_KEY || "";

const SCRIPT = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
let loading = null;
// The mounted check: {id, container, token, waiters}.
let widget = null;

/** The Turnstile API, loading its script the first time. */
function loadTurnstile() {
  if (globalThis.turnstile) return Promise.resolve(globalThis.turnstile);
  loading ||= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = SCRIPT;
    script.async = true;
    script.onload = () => (globalThis.turnstile ? resolve(globalThis.turnstile) : reject(new Error("The human check did not start.")));
    script.onerror = () => {
      loading = null;
      reject(new Error("The human check (Cloudflare Turnstile) could not load: check the connection, or a blocker."));
    };
    document.head.appendChild(script);
  });
  return loading;
}

const failAll = (state, message) => state.waiters.splice(0).forEach((w) => w.reject(new Error(message)));

/** Show the check in `container` (theme "light", "dark" or "auto"). */
export async function mountHumanCheck(container, { theme = "auto" } = {}) {
  if (!TURNSTILE_SITE_KEY || !container) return;
  const turnstile = await loadTurnstile();
  if (widget?.container === container) return;
  unmountHumanCheck();
  const state = { container, token: null, waiters: [] };
  state.id = turnstile.render(container, {
    sitekey: TURNSTILE_SITE_KEY,
    // The Worker accepts this check's tokens for /describe only (its ACTIONS).
    action: "describe",
    appearance: "always",
    size: "flexible",
    theme,
    callback: (token) => {
      state.token = token;
      state.waiters.splice(0).forEach((w) => w.resolve(token));
    },
    "error-callback": () => failAll(state, "The human check failed. Try again."),
    "expired-callback": () => {
      state.token = null;
      turnstile.reset(state.id);
    },
  });
  widget = state;
}

/** Remove the check (the panel closed, or Cloudflare is no longer chosen). */
export function unmountHumanCheck(container = null) {
  if (!widget || (container && widget.container !== container)) return;
  try {
    globalThis.turnstile?.remove(widget.id);
  } catch {
    // Gone with its container already.
  }
  failAll(widget, "The human check was closed.");
  widget = null;
}

/**
 * A token for one request: the check's current one, or the next it gives
 * (up to two minutes, if Cloudflare asks for a click). The check then runs
 * again for the next request. Null when the site has no site key. `signal`
 * ends the wait: Stop did nothing while it lasted
 * (wiki/architecture/state-machines.md, SM-5).
 */
export async function humanToken({ signal } = {}) {
  if (!TURNSTILE_SITE_KEY) return null;
  const state = widget;
  if (!state) throw new Error("The human check is not on the page: choose Cloudflare again.");
  const token =
    state.token ??
    (await new Promise((resolve, reject) => {
      const waiter = { resolve, reject };
      const leave = (error) => {
        const i = state.waiters.indexOf(waiter);
        if (i < 0) return;
        state.waiters.splice(i, 1);
        reject(error);
      };
      state.waiters.push(waiter);
      setTimeout(() => leave(new Error("The human check was not completed.")), 120000);
      if (signal?.aborted) leave(new Error("cancelled"));
      signal?.addEventListener("abort", () => leave(new Error("cancelled")), { once: true });
    }));
  state.token = null;
  try {
    globalThis.turnstile?.reset(state.id);
  } catch {
    // Removed meanwhile: the next request mounts a new one.
  }
  return token;
}
