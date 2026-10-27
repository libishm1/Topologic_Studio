/**
 * Methods and verification: what each result is, which properties are
 * machine-checked (formal/fireegress, Lean 4 + Mathlib), and where the models
 * stop being physics. The limits come from the literature review
 * (wiki/research/fire-egress-methods-2026-10-27.md, findings W2-W4) and are
 * stated plainly so a result is not read for more than it is.
 *
 * A modal page: focus moves into it on open and back on close, Escape and
 * the close button both close it, and Tab stays inside while it is open.
 */
import React, { useEffect, useMemo, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

import { Badge, Button, Icon, KeyValue } from "./primitives.jsx";
import { PROOFS, leanUrl } from "./proofs.js";

/**
 * A display equation. The TeX is the app's own (proofs.js), never user
 * input, so injecting KaTeX's markup is safe. KaTeX also writes MathML, which
 * screen readers read; a wide equation scrolls inside its own box.
 */
function Equation({ tex }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: true, throwOnError: false, output: "htmlAndMathml" }),
    [tex],
  );
  return <div className="equation" tabIndex={0} dangerouslySetInnerHTML={{ __html: html }} />;
}

const SHORTCUTS = [
  ["S", "Set the start"],
  ["E", "Set the exit"],
  ["F", "Set the fire origin"],
  ["G", "Show or hide the graph"],
  ["M", "Show or hide the building"],
  ["B", "Build or rebuild the graph"],
  ["Enter", "Find the route"],
  ["Esc", "Cancel picking; close this page"],
];

export function MethodsDialog({ open, onClose, studio }) {
  const sheetRef = useRef(null);
  const headingRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const before = document.activeElement;
    headingRef.current?.focus();
    // Captured before the app's single-key shortcuts, which must not act on
    // the model behind the page.
    const onKey = (event) => {
      event.stopPropagation();
      if (event.key === "Escape") {
        onClose();
      } else if (event.key === "Tab") {
        const focusable = sheetRef.current?.querySelectorAll(
          'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
        );
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
  const { capabilities, serverStatus } = studio;
  const topo = capabilities?.topologicpy || {};

  return (
    <div
      className="methods"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="methods__sheet" ref={sheetRef} role="dialog" aria-modal="true" aria-labelledby="methods-title">
        <div className="methods__bar">
          <strong>Topologic Studio</strong>
          <Button onClick={onClose}>
            <Icon name="close" size={16} />
            Close
          </Button>
        </div>
        <div className="methods__body">
          <nav className="methods__toc" aria-label="On this page">
            <ol>
              <li><a href="#methods-computes">What Studio computes</a></li>
              <li><a href="#methods-proved">What has been proved</a></li>
              <li><a href="#methods-limits">Limits to keep in mind</a></li>
              <li><a href="#methods-sources">Sources and credits</a></li>
              <li><a href="#methods-server">Server and shortcuts</a></li>
            </ol>
          </nav>

          <article className="methods__article">
            <div>
              <h2 id="methods-title" ref={headingRef} tabIndex={-1}>
                Methods and verification
              </h2>
              <p style={{ marginTop: 8, fontSize: 17 }}>
                How each result is computed, which properties are machine-checked, and where the models stop
                being physics.
              </p>
            </div>

            <section id="methods-computes" aria-labelledby="h-computes">
              <h3 id="h-computes">What Studio computes</h3>
              <dl className="methods__cards">
                <div className="methods__card">
                  <dt>Walkable graph</dt>
                  <dd>
                    One grid per building, turned to its walls, 0.5 m apart by default. Walls are cut 0.75 m
                    above each floor. A link is blocked where it crosses a wall outline, or where a point a
                    quarter, half or three quarters along it lies inside a wall; door links are never blocked.
                    Stairs and doors join the floors.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>Shortest route</dt>
                  <dd>
                    A* with the straight-line distance as its guide, then drawn straight as TopologicPy&apos;s
                    Wire.Straighten does, never longer than the grid route and checked against the walls at
                    points 1/8 of a grid step apart.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>Fire spread</dt>
                  <dd>
                    Heat diffusion on the graph, a front along links, or distance bands. Routes are reweighted
                    by heat and recomputed as it grows.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>Learned escape</dt>
                  <dd>
                    Tabular Q-learning with distance-based reward shaping, compared against the earliest escape
                    found by search.
                  </dd>
                </div>
              </dl>
            </section>

            <section id="methods-proved" aria-labelledby="h-proved">
              <h3 id="h-proved">What has been proved</h3>
              <p style={{ marginBottom: 12 }}>
                These statements are proved in Lean 4 with Mathlib, more than 450 machine-checked results in all,
                each resting only on Lean&apos;s standard axioms. The proofs are about idealised models in exact
                arithmetic: they do not cover floating-point rounding unless a statement says so, and each file
                lists what it leaves out. Sources: <code>formal/fireegress</code> in the repository; each
                statement links to its Lean file.
              </p>
              <ol className="proofs">
                {PROOFS.map((proof) => (
                  <li className="proof" key={proof.property}>
                    <h4 className="proof__title">{proof.property}</h4>
                    <p className="proof__meaning">{proof.meaning}</p>
                    <div className="proof__statement" role="group" aria-label={`Formal statement: ${proof.property}`}>
                      {proof.tex.map((tex) => (
                        <Equation key={tex} tex={tex} />
                      ))}
                    </div>
                    <p className="proof__meta">
                      <span className="pill pill--ok">{proof.status}</span>
                      <span>
                        Lean:{" "}
                        {proof.files.map((file, i) => (
                          <React.Fragment key={file}>
                            {i > 0 && ", "}
                            <a href={leanUrl(file)} target="_blank" rel="noreferrer">
                              {file}.lean
                            </a>
                          </React.Fragment>
                        ))}
                      </span>
                    </p>
                  </li>
                ))}
              </ol>
            </section>

            <section id="methods-limits" aria-labelledby="h-limits">
              <h3 id="h-limits">Limits to keep in mind</h3>
              <ul>
                <li>
                  <strong>The temperature is an index.</strong> It is scaled to read 20 to 120 °C, but it is not
                  a temperature a sensor would measure, so do not read it against tenability limits.
                </li>
                <li>
                  <strong>Time is counted in steps, not seconds.</strong> A step is one link of spread; walking
                  speed and fire growth are not yet in physical units.
                </li>
                <li>
                  <strong>The evacuee waits while the fire grows.</strong> Each reroute starts from the original
                  start. A route that has to pass through fire is drawn but reported as not safe.
                </li>
                <li>
                  <strong>A route stays within one part of the graph.</strong> Separate dwellings, or a room whose
                  door is missing from the model, are separate parts; the Duplex is two homes, so it has two.
                </li>
              </ul>
            </section>

            <section id="methods-sources" aria-labelledby="h-sources">
              <h3 id="h-sources">Sources and credits</h3>
              <ul style={{ fontSize: 14 }}>
                <li>
                  Hart, Nilsson and Raphael (1968). A formal basis for the heuristic determination of minimum cost
                  paths. IEEE Trans. Systems Science and Cybernetics 4, 100-107.{" "}
                  <a href="https://doi.org/10.1109/TSSC.1968.300136" target="_blank" rel="noreferrer">
                    doi:10.1109/TSSC.1968.300136
                  </a>
                </li>
                <li>
                  Ng, Harada and Russell (1999). Policy invariance under reward transformations. ICML 1999,
                  278-287.
                </li>
                <li>
                  Jabi et al. (2019). The synergy of non-manifold topology and reinforcement learning for fire
                  egress. eCAADe/SIGraDi 2019, 85-96.{" "}
                  <a href="https://doi.org/10.5151/proceedings-ecaadesigradi2019_671" target="_blank" rel="noreferrer">
                    doi:10.5151/proceedings-ecaadesigradi2019_671
                  </a>
                </li>
                <li>
                  Purser and McAllister (2016). Assessment of hazards to occupants from smoke, toxic gases, and
                  heat. SFPE Handbook of Fire Protection Engineering.{" "}
                  <a href="https://doi.org/10.1007/978-1-4939-2565-0_63" target="_blank" rel="noreferrer">
                    doi:10.1007/978-1-4939-2565-0_63
                  </a>
                </li>
                <li>
                  TopologicPy (Jabi) for the TGraph route engine and the Navigation cross-check; That Open Engine
                  and web-ifc for the viewer.
                </li>
                <li>
                  Example models: Duplex Apartment, NIBS buildingSMART alliance and U.S. Army ERDC (2011);
                  buildingSMART International sample scene; two CC0 models made for Studio.
                </li>
              </ul>
            </section>

            <section id="methods-server" aria-labelledby="h-server">
              <h3 id="h-server">Server and shortcuts</h3>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
                <Badge
                  variant={
                    serverStatus === "online"
                      ? "success"
                      : serverStatus === "waking" || serverStatus === "checking"
                        ? "warning"
                        : "danger"
                  }
                >
                  {serverStatus === "online"
                    ? "Connected"
                    : serverStatus === "waking"
                      ? "Waking up"
                      : serverStatus === "checking"
                        ? "Checking"
                        : "Unreachable"}
                </Badge>
                {capabilities?.version && <span className="mono">v{capabilities.version}</span>}
              </div>
              <p style={{ marginBottom: 12 }}>
                Routing, alternatives, fire and RL run in your browser. The server adds the TopologicPy engine,
                the Navigation cross-check and the Topology JSON tools.
                {serverStatus === "waking" && " A hosted server on a free plan sleeps when idle and takes up to a minute to wake."}
              </p>
              <KeyValue
                entries={[
                  ["API", import.meta.env.VITE_API_BASE || "http://localhost:8000"],
                  ["TopologicPy", topo.topologicpy || "not installed"],
                  ["topologic-core", topo.topologic_core || "not installed"],
                  ["Path engines", (capabilities?.engines || []).join(", ") || "-"],
                  ["IfcOpenShell", capabilities?.server_side_ifc ? "available" : "not installed"],
                ]}
              />
              <h4 style={{ margin: "20px 0 8px", fontSize: 15 }}>Keyboard</h4>
              <KeyValue entries={SHORTCUTS} />
            </section>
          </article>
        </div>
      </div>
    </div>
  );
}
