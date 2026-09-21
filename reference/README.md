# Reference material

## `topologicpy-explorer-local.zip`

TopologicPy Explorer, Local Preview 0.5 — a small Python server plus Plotly web
UI that runs auto-discovered TopologicPy examples locally. Not part of this
app; kept because it is the reference for current TopologicPy usage.

What it showed us, and what changed here as a result:

- It calls **`TGraph`**, not the legacy `Graph`. That prompted the benchmark
  that found `TGraph.ShortestPath` is ~500x faster per query than the legacy
  class, which reversed this project's "TopologicPy is too slow to route with"
  conclusion. See `wiki/roadmap/handoff-2026-08-24.md` and the engine module
  docstring.
- Its architecture is a different shape from Topologic Studio: examples are
  auto-discovered modules that each own metadata, parameter validation and a
  Plotly figure, and the browser never submits Python. Worth borrowing if an
  examples gallery is ever wanted; it is not a replacement for the egress
  pipeline.
