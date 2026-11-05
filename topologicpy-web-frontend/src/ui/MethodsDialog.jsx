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
                    A diagonal is blocked when both of its axis detours are. Each stair is its walking line, a
                    node per tread on the flight's centre line, joined to the floor once at each end; doors
                    join the rooms. A stair is read with the parts aggregated into it. A stair whose treads are
                    not in the model (an IfcStair with no steps, or a room named as a stair) is walked along an
                    assumed dog-leg line inside its footprint, with 175 mm risers and 280 mm goings. Lifts and
                    escalators (IfcTransportElement, or rooms and objects named as lifts) join the floors beside
                    them only when turned on in the graph settings.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>Shortest route</dt>
                  <dd>
                    A* with the straight-line distance as its guide, then drawn straight as TopologicPy&apos;s
                    Wire.Straighten does, never longer than the grid route and checked against the walls at
                    points at most a quarter of a grid step apart.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>Fire spread</dt>
                  <dd>
                    A heat index that relaxes toward its neighbours, a spread one link per step, or distance
                    bands. In the live simulation routes are reweighted by heat and recomputed as it grows; the
                    forecast runs the model to its end first and routes on the state each point will be in when
                    it is reached.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>Flamescapes fire and smoke (optional)</dt>
                  <dd>
                    A browser version of the Flamescapes solver (Wrede, Wagner et al. 2026), on a grid of 0.2 to
                    0.45 m cells built from the IFC&apos;s walls, slabs, roofs and stairs: buoyant gas flow with
                    pressure projection, solid fuel that dries, pyrolyses and oxidises with the paper&apos;s Arrhenius
                    rates, gas combustion with the heat of Fire-X&apos;s 417 kJ per mol of oxygen, radiation by rays
                    absorbed by soot, flame-front particles and, in the second option, embers carried by the gas
                    that start spot fires. The room contents are the paper&apos;s statistical fire load; the first
                    item burns with the fuel of its fire class (NFPA 10: A, B, C, D, K; ISO 3941:2026: L, lithium-ion
                    batteries); a liquid (B, K) evaporates at most at its pool&apos;s burning rate. Read at head height
                    (1.8 m) at each graph node: gas temperature, visibility through the smoke (S = 3 / (8700 m²/kg
                    times soot density)) and CO. The smoke itself, the paper&apos;s residual species carried by the gas, is
                    drawn as a cloud: its soot density averaged over blocks of cells, each as opaque as light crossing
                    the block finds it. With the ventilation running, the model&apos;s air terminals (IfcAirTerminal) swap
                    gas each step with one well-mixed duct volume, carrying smoke to every room with a terminal (not
                    the paper&apos;s: Flamescapes models no ducts). The plan adds the CO dose of ISO 13571 to the heat dose. It runs on the GPU
                    through WebGPU where the browser offers it, otherwise on one CPU core.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>What the elements are</dt>
                  <dd>
                    Each element by its IFC class where the exporter typed it; generic elements (IfcBuildingElementProxy
                    and the like) by their name, in English, German, Norwegian, Dutch, French, Spanish, Italian and a few
                    Arabic words, else by the size and height of their box. On request, IFCNet&apos;s DGCNN point-cloud
                    classifier (Emunds et al. 2021), converted to ONNX and run in a worker, looks at the shape of those the
                    name could not place, and its answer replaces the box&apos;s when it is at least 90% sure. With their
                    class hidden, the rules named 86% of 6,066 typed elements from 45 models correctly.
                  </dd>
                </div>
                <div className="methods__card">
                  <dt>Learned escape</dt>
                  <dd>
                    Tabular Q-learning with distance-based reward shaping, compared against the earliest escape
                    found by search and against the best return within the step limit.
                  </dd>
                </div>
              </dl>
            </section>

            <section id="methods-proved" aria-labelledby="h-proved">
              <h3 id="h-proved">What has been proved</h3>
              <p style={{ marginBottom: 12 }}>
                These statements are proved in Lean 4 with Mathlib, more than 1,260 machine-checked theorems and lemmas in 27 modules,
                each resting only on Lean&apos;s standard axioms or none. The proofs are about idealised models in exact
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
                  a temperature a sensor would measure. The forecast applies ISO 13571&apos;s convective dose law
                  to this index, so its doses rank routes under the model and the ISO limits do not carry over.
                </li>
                <li>
                  <strong>Fire time is counted in steps.</strong> The fire models count steps; the forecast
                  converts them to seconds with a factor you set and walks at a set speed. Fire growth is not
                  calibrated.
                </li>
                <li>
                  <strong>In the live simulation the evacuee waits while the fire grows.</strong> Each reroute
                  starts from the original start. A route that has to pass through fire is drawn but reported as not safe.
                </li>
                <li>
                  <strong>A route stays within one part of the graph.</strong> Separate dwellings, or a room whose
                  door is missing from the model, are separate parts; the Duplex is two homes, so it has two.
                </li>
                <li>
                  <strong>Assumed stairs and lifts are approximate.</strong> An assumed stair has no known width
                  or landing, so its line can be longer or shorter than the real stair. On five models with their
                  treads removed, routes to another floor through assumed stairs averaged 1.0 to 1.4 times the
                  length of routes through the real ones. A lift ride counts as walking its rise, with no
                  wait for the car, and a lift is taken to stop at every floor beside it.
                </li>
                <li>
                  <strong>Flamescapes in the browser is coarse.</strong> The paper resolves centimetres on a desktop
                  GPU; here a cell is 0.2 to 0.45 m and a step 0.1 s. So the fuel&apos;s surface layer, a sub-grid
                  rule for where an ember lands, the first item alight from the start, a raised solid-gas heat
                  transfer coefficient (the paper&apos;s own 200) and exact integration of each reaction over a step
                  stand in for what the paper resolves. Walls and slabs do not burn; doors are open and windows
                  closed unless the run closes the doors (each leaf fills its opening, without leaks) or opens the
                  windows (their wall openings vent the smoke); running ventilation joins every air terminal through
                  one duct volume at an assumed 100 L/s a terminal, without reading which system a terminal belongs to
                  or the ducts themselves; and contents are a statistical fire load. It shows how heat and smoke may move through
                  the building; it is not calibrated, and its fire class values marked provisional are still being
                  checked. Indoors at these cells embers seldom land hot enough to light anything: in a 120 s fire in
                  the Duplex they lit no spot fire. A liquid pool lights, but much of its vapour stays unburnt on this
                  grid: a 1 m² heptane pool reached 0.3 MW after 30 s in a test room, against the 3.2 MW its burning rate allows.
                  The heat release depends strongly on the grid, and not in one direction: a sofa fire in the Duplex peaked at
                  0.09 MW on a 0.20 m grid, 2.5 MW on 0.23 m, 6.1 MW on 0.27 m and 0.75 MW on the CPU&apos;s 0.34 m. On the GPU the
                  grid is sized to the GPU&apos;s speed unless the finest is asked for; compare runs on one grid.
                </li>
                <li>
                  <strong>Identifying generic elements is a best guess.</strong> On 159 proxies labelled by hand from their
                  names, the rules were right for 55 to 60%. IFCNet on its own was right for 41% of typed elements (61% within
                  the 20 classes it knows) and added almost nothing to the rules in the benchmark. Its weights are published
                  by its authors without a stated licence and are shipped with credit.
                </li>
              </ul>
            </section>

            <section id="methods-sources" aria-labelledby="h-sources">
              <h3 id="h-sources">Sources and credits</h3>
              <ul style={{ fontSize: 14 }}>
                <li>
                  Wrede, Wagner, Pałubicki, Michels and Pirk (2026). Flamescapes: simulating the wildlandfire urban
                  interface. ACM Transactions on Graphics 45(6), 246 (CC BY 4.0); DOI 10.1145/3842518 printed
                  on the paper, not yet registered.{" "}
                  <a href="https://helgewrede.github.io/flamescapes/" target="_blank" rel="noreferrer">
                    Project page
                  </a>
                </li>
                <li>
                  Wrede, Wagner, Mahfuz, Pałubicki, Michels and Pirk (2025). Fire-X: extinguishing fire with
                  stoichiometric heat release. ACM Transactions on Graphics 44(6), 268.{" "}
                  <a href="https://doi.org/10.1145/3763338" target="_blank" rel="noreferrer">
                    doi:10.1145/3763338
                  </a>
                </li>
                <li>
                  Huggett (1980). Estimation of rate of heat release by means of oxygen consumption measurements.
                  Fire and Materials 4, 61-65.{" "}
                  <a href="https://doi.org/10.1002/fam.810040202" target="_blank" rel="noreferrer">
                    doi:10.1002/fam.810040202
                  </a>
                </li>
                <li>
                  Mulholland and Croarkin (2000). Specific extinction coefficient of flame generated smoke. Fire
                  and Materials 24, 227-230.{" "}
                  <a href="https://doi.org/10.1002/1099-1018(200009/10)24:5<227::AID-FAM742>3.0.CO;2-9" target="_blank" rel="noreferrer">
                    doi:10.1002/1099-1018(200009/10)24:5&lt;227::AID-FAM742&gt;3.0.CO;2-9
                  </a>
                </li>
                <li>
                  Babrauskas (1983). Estimating large pool fire burning rates. Fire Technology 19(4), 251-261 (the
                  liquid pool&apos;s burning rate, through the SFPE Handbook table in NUREG-1805 Suppl. 1).{" "}
                  <a href="https://doi.org/10.1007/BF02380810" target="_blank" rel="noreferrer">
                    doi:10.1007/BF02380810
                  </a>
                </li>
                <li>
                  Lyon and Quintiere (2007). Criteria for piloted ignition of combustible solids. Combustion and
                  Flame 151(4), 551-559 (the floor below which a heated solid is not counted as burning).{" "}
                  <a href="https://doi.org/10.1016/j.combustflame.2007.07.020" target="_blank" rel="noreferrer">
                    doi:10.1016/j.combustflame.2007.07.020
                  </a>
                </li>
                <li>
                  Emunds, Pauen, Richter, Frisch and van Treeck (2021). IFCNet: a benchmark dataset for IFC entity
                  classification. EG-ICE 2021. Code MIT (RWTH-E3D/ifcnet-models); the DGCNN weights from
                  ifcnet.e3d.rwth-aachen.de, converted and pruned, with no stated licence (see models/NOTICE-ifcnet.md).{" "}
                  <a href="https://doi.org/10.48550/arXiv.2106.09712" target="_blank" rel="noreferrer">
                    doi:10.48550/arXiv.2106.09712
                  </a>
                </li>
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
                  and web-ifc for the viewer; onnxruntime-web (MIT) for the shape classifier.
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
