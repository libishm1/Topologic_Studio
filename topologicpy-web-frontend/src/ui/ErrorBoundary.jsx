/**
 * Catches an error thrown while rendering below it and shows `fallback`
 * instead: with no boundary, React unmounts the whole tree and the page goes
 * blank, which is what a stale deadline map did after a rebuild (review C1).
 *
 * `fallback` is an element, or a function of { error, reset } where reset
 * renders the children again. `onError` hears every error caught.
 */
import React from "react";

import { Button } from "./primitives.jsx";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
    this.reset = () => this.setState({ error: null });
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    this.props.onError?.(error, info);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const { fallback = null } = this.props;
    return typeof fallback === "function" ? fallback({ error, reset: this.reset }) : fallback;
  }
}

/** The page shown when the app itself fails: what happened, and the ways back. */
export function AppCrash({ error, reset }) {
  return (
    <div className="crash">
      <div className="placeholder" role="alert">
        <div className="placeholder__card">
          <h2>Studio stopped</h2>
          <p>
            Something went wrong and the page could not go on. Your settings are kept; an open model has to be
            opened again.
          </p>
          <p className="mono">{error?.message || String(error)}</p>
          <div className="row">
            <Button variant="primary" onClick={reset}>
              Try again
            </Button>
            <Button onClick={() => window.location.reload()}>Reload the page</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
