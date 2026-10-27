/**
 * What the Methods page says is proved, one entry per property: the
 * plain-language meaning, the formal statement as LaTeX (rendered with KaTeX),
 * and the Lean module(s) in formal/fireegress that prove it. Each statement
 * transcribes the theorem it cites; the wording follows the adversarial
 * reviews of those statements against the code.
 *
 * Display maths cannot wrap, so each line is kept short enough to fit a phone
 * (about 30 symbols); a long statement is split over several lines or an
 * aligned block rather than left to scroll sideways.
 */

const REPO = "https://github.com/libishm1/Topologic_Studio/blob/main/formal/fireegress/FireEgress";
export const leanUrl = (file) => `${REPO}/${file}.lean`;

export const PROOFS = [
  {
    property: "Grid routes are shortest on the walkable graph",
    meaning:
      "The straight-line guide is consistent: across any link it falls by no more than the link's cost, also with fire weighting, because heat only raises a cost. A* as Studio runs it, closing each node once, is Dijkstra on the reduced costs, so the node v* it settles next already has its least cost, and it returns a least-cost route on the walkable graph between the snapped start and exit.",
    tex: [
      String.raw`h(x) = \lVert x - x_t \rVert, \quad w(u,v) \ge \lVert x_u - x_v \rVert`,
      String.raw`h(u) \le w(u,v) + h(v)`,
      String.raw`w'(u,v) = w(u,v) - h(u) + h(v) \ge 0`,
      String.raw`g(v^{\ast}) = \min_{P \,:\, s \leadsto v^{\ast}} c(P)`,
    ],
    files: ["AStar", "Heuristic"],
    status: "Proved",
  },
  {
    property: "Straight runs are never longer than the grid route",
    meaning:
      "The drawn route keeps a subset of the grid route's points p_0 to p_n with the same ends, so by the triangle inequality it cannot be longer.",
    tex: [
      String.raw`0 = i_0 < i_1 < \dots < i_m = n`,
      String.raw`\begin{aligned} &\textstyle\sum_{k=0}^{m-1} \lVert p_{i_{k+1}} - p_{i_k} \rVert \\ &\quad \le \textstyle\sum_{j=0}^{n-1} \lVert p_{j+1} - p_j \rVert \end{aligned}`,
    ],
    files: ["Straighten"],
    status: "Proved",
  },
  {
    property: "How far straight runs keep from walls",
    meaning:
      "Walls are checked at points at most a quarter of a grid step s apart. A run whose ends both stand c = 0.25 m clear keeps about 0.24 m from walls between those points at the 0.5 m grid (less on a coarser grid); near doorways and wall faces it keeps only as much as its ends do.",
    tex: [
      String.raw`\delta \le s/4`,
      String.raw`\operatorname{dist}(x, W) \ge \sqrt{c^{2} - (\delta/2)^{2}}`,
      String.raw`\sqrt{0.25^{2} - 0.0625^{2}} \approx 0.242\ \mathrm{m}`,
    ],
    files: ["Straighten"],
    status: "Proved, with that bound",
  },
  {
    property: "Wall crossings are found exactly",
    meaning:
      "The crossing test flags a link [a, b] exactly when it crosses a wall outline [c, d] at a point inside both. A link that only touches a wall, at a corner or along a face, is not flagged by it; three sample points along each link catch most of those.",
    tex: [
      String.raw`\begin{aligned} o(a,b,c) = {} &(b_x - a_x)(c_y - a_y) \\ - {} &(b_y - a_y)(c_x - a_x) \end{aligned}`,
      String.raw`o(c,d,a)\,o(c,d,b) < 0`,
      String.raw`\wedge\ \ o(a,b,c)\,o(a,b,d) < 0`,
      String.raw`\iff [a,b] \text{ crosses } [c,d] \text{ inside both}`,
    ],
    files: ["Segments"],
    status: "Proved",
  },
  {
    property: "The grid's detour is bounded",
    meaning:
      "On an open floor no grid route is shorter than the octile distance, and that is at most 8.2% longer than walking straight; walls only make the grid route longer. A diagonal link is kept only if one of its two axis detours is open.",
    tex: [
      String.raw`\begin{aligned} \operatorname{oct}(a,b) = {} &\max(|a|,|b|) \\ + {} &(\sqrt{2} - 1)\min(|a|,|b|) \end{aligned}`,
      String.raw`\sqrt{a^{2} + b^{2}} \le \operatorname{oct}(a,b)`,
      String.raw`\operatorname{oct}(a,b) \le \sqrt{4 - 2\sqrt{2}}\,\sqrt{a^{2} + b^{2}}`,
      String.raw`\sqrt{4 - 2\sqrt{2}} \approx 1.0824`,
    ],
    files: ["Lattice"],
    status: "Proved",
  },
  {
    property: "Lengths survive the change of frame",
    meaning:
      "Turning IFC coordinates into the viewer's frame is a rotation that keeps handedness, and the model's placement is a rigid motion, so route lengths stay in metres. For a model rotated in plan, That Open rounds the placement to about 1e-3, which can change lengths by about that fraction.",
    tex: [
      String.raw`R(x, y, z) = (x,\, z,\, -y), \quad \det R = +1`,
      String.raw`C^{\top} C = I \;\Rightarrow`,
      String.raw`\lVert (Cx + p) - (Cy + p) \rVert = \lVert x - y \rVert`,
    ],
    files: ["Frames"],
    status: "Proved",
  },
  {
    property: "Heat never overshoots or leaves its range",
    meaning:
      "A node never gets hotter than the mean of its neighbours, and over any number of steps every node stays within its starting range. The previous rule could push a dead end from 20 to 140 in one step; the coefficient is now capped.",
    tex: [
      String.raw`\beta_i = \min\!\left(1, \kappa / d_i\right), \quad b = \bar T^{\,n}_{N(i)}`,
      String.raw`T_i^{n+1} = T_i^{n} + \beta_i \bigl(b - T_i^{n}\bigr)`,
      String.raw`\Rightarrow\ T_i^{n+1} \in \bigl[\min(T_i^n, b),\, \max(T_i^n, b)\bigr]`,
      String.raw`\Delta t\,(G_i + H_i) \le C_i \;\Rightarrow`,
      String.raw`\min\bigl(0, \min_j \theta_j^{0}\bigr) \le \theta_i^{n} \le \max\bigl(0, \max_j \theta_j^{0}\bigr)`,
    ],
    files: ["Temperature", "GraphNumerics"],
    status: "Proved",
  },
  {
    property: "Escape search is exact as the fire grows",
    meaning:
      "For the search as specified, the step at which a node is first reached is the least number of moves k of any fire-free walk to it, provided each node's ignition time is fixed. The code's step limit is not covered.",
    tex: [
      String.raw`\ell(v) = \min\{\, k : s = v_0 \to \dots \to v_k = v \,\}`,
      String.raw`\text{over fire-free walks}`,
    ],
    files: ["GraphNumerics"],
    status: "Proved, specification",
  },
  {
    property: "Earliest arrival while the fire changes the routes",
    meaning:
      "When each link's arrival time never overtakes (leaving later never arrives earlier) and travel never takes negative time, label setting gives the exact earliest arrival everywhere, and waiting never helps. Both conditions are needed: a counterexample without the first is proved.",
    tex: [
      String.raw`s \le t \Rightarrow A_e(s) \le A_e(t), \qquad t \le A_e(t)`,
      String.raw`\Longrightarrow\ \mathrm{label}(v) = \mathrm{EA}(v)`,
      String.raw`\mathrm{EA}(v) = \min_{s = v_0 \to \dots \to v_k = v} \bigl(A_{e_k} \circ \dots \circ A_{e_1}\bigr)(t_0)`,
    ],
    files: ["TimeDependent"],
    status: "Proved (for the planned router)",
  },
  {
    property: "Least heat dose on the way out",
    meaning:
      "Backward induction over time gives the least total dose of any route that reaches an exit by the horizon H, and a route below the incapacitation threshold exists exactly when that value is below it. If doses only grow with time, some optimal route never waits.",
    tex: [
      String.raw`\begin{aligned} V_t(v) = \min\Bigl( &w_v(t) + V_{t+1}(v), \\ &\min_{e = (v,u)} \bigl[ d_e(t) + V_{t + \tau_e}(u) \bigr] \Bigr) \end{aligned}`,
      String.raw`V_H(v) = \begin{cases} 0 & v \text{ an exit} \\ \infty & \text{otherwise} \end{cases}`,
      String.raw`d_e(t) \le d_e(t+1), \; w_v(t) \ge 0 \;\Rightarrow`,
      String.raw`\text{some optimal route never waits}`,
    ],
    files: ["MinDose"],
    status: "Proved (for a planned feature)",
  },
  {
    property: "Yen's alternative routes are the next cheapest",
    meaning:
      "Every list Yen returns, at any length it reaches, is a list of k shortest loopless routes: none left out is cheaper than the last one listed. The different-space mode is bounded by its stretch and overlap rules but may stop before finding every such route.",
    tex: [
      String.raw`P_1, \dots, P_k \ \text{loopless}`,
      String.raw`c(P_1) \le c(P_2) \le \dots \le c(P_k)`,
      String.raw`c(P_k) \le c(Q) \ \text{ for every loopless}`,
      String.raw`Q \notin \{P_1, \dots, P_k\}`,
    ],
    files: ["Yen"],
    status: "Proved",
  },
  {
    property: "Shaping keeps the best escape best",
    meaning:
      "The agent's extra reward cannot reorder two escapes from the same start, provided every terminal state, an exit, a burning node or (with fire) the step limit, carries zero potential. A counterexample shows the order flips otherwise; Studio sets all three to zero.",
    tex: [
      String.raw`\tilde R(\tau) = R(\tau) + \gamma^{T}\,\Phi(s_T) - \Phi(s_0)`,
      String.raw`\Phi(s_T) = 0 \;\Rightarrow`,
      String.raw`\tilde R(\tau_1) \le \tilde R(\tau_2) \iff R(\tau_1) \le R(\tau_2)`,
    ],
    files: ["Shaping"],
    status: "Proved",
  },
  {
    property: "Every fire and tenability equation is dimensionally consistent",
    meaning:
      "All 90 equations taken from the literature close in SI units. Unit scales a dimension check cannot see (minutes against seconds, degrees C against kelvin) are pinned numerically where they matter.",
    tex: [
      String.raw`\Delta T = 6.85 \left( \frac{\dot Q^{2}}{A_o \sqrt{H_o}\; h_k A_T} \right)^{1/3}`,
      String.raw`[6.85] = \mathrm{K}^{2/3}\,\mathrm{kW}^{-1/3}\,\mathrm{m}^{5/6}`,
      String.raw`X_{\mathrm{FED}} = \sum_i \frac{\Delta t_i}{t_I(T_i)}`,
      String.raw`t_{I,\mathrm{conv}} = 5 \times 10^{7}\, T^{-3.4}\ \mathrm{min}`,
      String.raw`(T \text{ in } {}^{\circ}\mathrm{C})`,
    ],
    files: ["FireDynamics", "Tenability"],
    status: "Proved",
  },
];
