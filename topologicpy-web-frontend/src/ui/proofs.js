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
      "Walls are checked at points at most a quarter of a grid step s apart. A run whose ends both stand c = 0.25 m clear keeps at least 0.24 m from walls between those points at the 0.5 m grid (less on a coarser grid); near doorways and wall faces it keeps only as much as its ends do.",
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
      "On an open floor no grid route is shorter than the octile distance, and that is at most 8.24% longer than walking straight; walls only make a route between lattice nodes longer. A diagonal link is kept only if one of its two axis detours is open.",
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
      "Turning IFC coordinates into the viewer's frame is a rotation that keeps handedness, and the model's placement is a rigid motion, so route lengths stay in metres. For a model rotated in plan, That Open rounds the placement's direction to 1e-3, which changes lengths by 0.015% for a model turned 45 degrees.",
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
      "One step never moves a node past the mean of its neighbours, so it stays between its own value and that mean. The previous rule could push a dead end from 20 to 140 in one step; the coefficient is now capped. For the physical graph heat equation, which Studio does not run yet, a time step within the stability limit keeps every node within its starting range for all steps.",
    tex: [
      String.raw`\beta_i = \min\!\left(1, \kappa / d_i\right), \quad b = \bar T^{\,n}_{N(i)}`,
      String.raw`T_i^{n+1} = T_i^{n} + \beta_i \max\bigl(0,\, b - T_i^{n}\bigr)`,
      String.raw`\Rightarrow\ T_i^{n+1} \in \bigl[\min(T_i^n, b),\, \max(T_i^n, b)\bigr]`,
      String.raw`\text{Physical heat equation:}`,
      String.raw`\Delta t\,(G_i + H_i) \le C_i \;\Rightarrow`,
      String.raw`\min\bigl(0, \min_j \theta_j^{0}\bigr) \le \theta_i^{n} \le \max\bigl(0, \max_j \theta_j^{0}\bigr)`,
    ],
    files: ["Temperature", "GraphNumerics"],
    status: "Proved",
  },
  {
    property: "Escape search is exact as the fire grows",
    meaning:
      "For the search as specified, the step at which a node is first reached is the least number of moves k of any walk to it in which each move is made before the next node ignites, provided each node's ignition time is fixed. If none is found within the step limit, no such walk of that many moves exists.",
    tex: [
      String.raw`\ell(v) = \min\{\, k : s = v_0 \to \dots \to v_k = v \,\}`,
      String.raw`\text{over walks whose every move}`,
      String.raw`\text{is made before the next node ignites}`,
    ],
    files: ["GraphNumerics"],
    status: "Proved, specification",
  },
  {
    property: "Earliest arrival while the fire changes the routes",
    meaning:
      "When each link's arrival time never overtakes (leaving later never arrives earlier) and travel never takes negative time, label setting gives the exact earliest arrival everywhere, and waiting never helps. FIFO cannot be dropped: a counterexample without it is proved. Studio's forecast router walks each link at constant speed and closes a point once the forecast makes it untenable, which meets both, so its route reaches an exit as early as any safe route can. A route planned once on the fire as it is at departure can walk into it; a proved example shows this.",
    tex: [
      String.raw`s \le t \Rightarrow A_e(s) \le A_e(t), \qquad t \le A_e(t)`,
      String.raw`\Longrightarrow\ \mathrm{label}(v) = \mathrm{EA}(v)`,
      String.raw`\mathrm{EA}(v) = \min_{s = v_0 \to \dots \to v_k = v} \bigl(A_{e_k} \circ \dots \circ A_{e_1}\bigr)(t_0)`,
    ],
    files: ["TimeDependent", "Forecast"],
    status: "Proved, and met by the forecast router",
  },
  {
    property: "Least heat dose on the way out",
    meaning:
      "Backward induction over time gives the least total dose of any route that reaches an exit by the horizon H, and a route below the incapacitation threshold exists exactly when that value is below it. If doses only grow with time, some optimal route never waits.",
    tex: [
      String.raw`\begin{aligned} V_t(v) = \min\Bigl( &w_v(t) + V_{t+1}(v), \\ &\min_{e = (v,u)} \bigl[ d_e(t) + V_{t + \tau_e}(u) \bigr] \Bigr) \end{aligned}`,
      String.raw`\text{for } v \text{ not an exit}`,
      String.raw`V_H(v) = \begin{cases} 0 & v \text{ an exit} \\ \infty & \text{otherwise} \end{cases}`,
      String.raw`d_e(t) \le d_e(t+1), \; w_v(t) \ge 0 \;\Rightarrow`,
      String.raw`\text{some optimal route never waits}`,
    ],
    files: ["MinDose", "Forecast"],
    status: "Proved",
  },
  {
    property: "The time-dose trade-off is exact",
    meaning:
      "The forecast keeps each point's heat as its running maximum, so heat never falls, a link's dose never falls with the time it is entered, and waiting never helps. A route that is no later and no more dosed then stays so along every continuation, so, when the search finishes within its label budget, its routes to the exits are exactly the trade-off between arriving early and accruing little dose: the first is the earliest safe route and the last the least-dose one; once routes tied in arrival are reduced to the least-dose one, none is beaten on both. The dose is the convective-heat FED of the model's heat index, so it ranks routes under the model; it is not a measured dose.",
    tex: [
      String.raw`r(T) = \frac{T^{3.4}}{5 \times 10^{7} \cdot 60}\ \mathrm{s}^{-1}`,
      String.raw`d_e(t) = \int_{t}^{t + \tau_e} \max\bigl(r_u(s), r_w(s)\bigr)\, ds`,
      String.raw`t_1 \le t_2,\ D_1 \le D_2 \;\Rightarrow`,
      String.raw`D_1 + d_e(t_1) \le D_2 + d_e(t_2)`,
    ],
    files: ["Forecast", "Tenability"],
    status: "Proved for the search; the tie filter follows from it",
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
    property: "How long you can wait before leaving",
    meaning:
      "When heat never falls, leaving earlier is never worse, so the departures that still get out form an interval, and its end is the escape deadline L: leaving before it is safe, leaving at or after it is not. Without a dose limit L is computed exactly by settling the latest deadline first. With a dose limit, for the chosen start the deadline is bracketed to within 0.1 s by bisection on the exact search; for every point it is computed on a time grid that rounds walking times up, so the map is never later than the exact deadline. The proofs do not bound how much earlier the map can be; in tests on 23 models it was up to 7.8 s earlier.",
    tex: [
      String.raw`L(x) = u(x) \quad \text{for an exit } x`,
      String.raw`L(v) = \max_{(v,w)} \min\bigl(u(v),\ L(w) - \tau_{vw}\bigr)`,
      String.raw`\text{leaving at } t \text{ is safe} \iff t < L(v)`,
      String.raw`T^{\ast}(v) = \sup\{\, t : D^{\ast}(v, t) \le \theta \,\}`,
      String.raw`\mathrm{map}(v) \le T^{\ast}(v) \le L(v)`,
    ],
    files: ["Deadline", "MinDose"],
    status: "Proved in exact arithmetic; a rounding fault the proofs found and one the tests found are fixed",
  },
  {
    property: "Q-learning converges to the best table",
    meaning:
      "For Studio's deterministic problem the Bellman operator is a contraction, and tabular Q-learning with any step in (0, 1] bounded away from 0 converges to the optimal table on the state-action pairs reachable from the start, provided each of them is updated infinitely often; with fire and a step limit the optimum is exact backward induction. This says nothing about a finite run, so the panel shows the policy's return beside the best return within the step limit. Shaping can change the optimal policy unless the step limit also has zero potential; Studio sets it so.",
    tex: [
      String.raw`\lVert TQ - TQ' \rVert_\infty \le \gamma\, \lVert Q - Q' \rVert_\infty`,
      String.raw`\alpha_n \in [\alpha_{\min}, 1],\ \alpha_{\min} > 0,`,
      String.raw`\text{every reachable pair}`,
      String.raw`\text{updated infinitely often}`,
      String.raw`\Rightarrow\ Q_n \to Q^{\ast}`,
      String.raw`Q^{\ast}_{\Phi} = Q^{\ast} - \Phi`,
      String.raw`\text{if } \Phi = 0 \text{ at every end}`,
    ],
    files: ["QLearning"],
    status: "Proved (deterministic, asymptotic)",
  },
  {
    property: "How a graph fire front at speed F relates to the real one",
    meaning:
      "A front spreading at speed F from one point arrives at distance over F, the eikonal solution away from the source. Along the graph, provided no link passes through a wall, it is never earlier than the front that must go round walls, and on an open floor at most 8.24% later; with walls only 'never early' is proved, and refining the grid does not remove the 8.24%. Studio's fire models do not compute this front: of them only the distance-band fire is covered, and it is never later than the walled front; the spread-along-links fire is a hop count.",
    tex: [
      String.raw`T(x) = \lVert x - x_0 \rVert / F`,
      String.raw`\lVert \nabla T(x) \rVert = 1/F \quad (x \ne x_0)`,
      String.raw`\lVert x - x_0 \rVert \le d_{\Omega}(x_0, x) \le F\, T_h(x)`,
      String.raw`F\, T_h(x) \le 1.0824\, \lVert x - x_0 \rVert`,
      String.raw`\text{on an unbounded open lattice}`,
    ],
    files: ["Eikonal"],
    status: "Proved",
  },
  {
    property: "Rounding has a bounded effect on wall tests and routes",
    meaning:
      "In double precision the wall-crossing test can disagree with exact arithmetic only when a link endpoint lies within about 1e-13 m of a wall's line, or a wall corner that close to the link's line (for distances under 300 m). With a guide made inconsistent by rounding by at most a small amount, A* is within that amount times the number of links of the optimum, and summing a 10,000-link route adds a relative error of at most 1.1e-12. Storing node coordinates in single precision moves a link's length by at most 1.2e-5 m for nodes within 100 m of the origin.",
    tex: [
      String.raw`|\tilde o - o| \le \bigl((1+u)^{4} - 1\bigr)\bigl(|X| + |Y|\bigr)`,
      String.raw`|\tilde o| > (4 + 16u)\,u\,\bigl(|\tilde l| + |\tilde r|\bigr)`,
      String.raw`\Rightarrow\ \operatorname{sgn} \tilde o = \operatorname{sgn} o`,
      String.raw`u = 2^{-53}, \qquad 2\,u_{32}\,M \approx 1.2 \times 10^{-5}\ \mathrm{m}`,
    ],
    files: ["FloatError"],
    status: "Proved, in the standard rounding model",
  },
  {
    property: "The fire and tenability equations close in SI units",
    meaning:
      "Of the 90 equations taken from the literature, 71 were confirmed against a source, 11 corrected and 8 could not be verified. The 47 with physical units are checked in Lean and close in SI units; two burn-injury laws can be checked only in normalised form. Unit scales a dimension check cannot see (minutes against seconds, degrees C against kelvin) are pinned numerically where they matter.",
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
  {
    property: "A forecast close to another gives deadlines close to its deadlines",
    meaning:
      "If a cheaper fire model's closing times are within e of a reference model's at every node, every escape deadline moves by at most e. Shifting the cheaper times e earlier makes them conservative: never later than the reference deadline and within 2e of it. On the same open links the spread-along-links fire is at every step at least as hot as the heat-index fire, so it closes no node later and its deadlines and routes are conservative. Exact arithmetic; a surrogate that also changes dose rates is shown only to keep the deadline monotone.",
    tex: [
      String.raw`\lvert L_{c}(v) - L_{c'}(v) \rvert \le \lVert c - c' \rVert_{\infty}`,
      String.raw`\lVert \hat c - c \rVert_{\infty} \le \varepsilon \;\Rightarrow`,
      String.raw`L_{c}(v) - 2\varepsilon \le L_{\hat c - \varepsilon}(v) \le L_{c}(v)`,
    ],
    files: ["Surrogate", "Deadline"],
    status: "Proved",
  },
  {
    property: "The two-zone room model only gets worse with time",
    meaning:
      "Heat release never falls, the hot layer's interface never rises, a room's gas temperature never falls, and a node that becomes untenable stays untenable. These are the conditions the forecast routers' proofs ask of a fire forecast. Proved in exact arithmetic; that each floating-point operation keeps them is tested, and the accuracy of the Euler time step is tested numerically, not proved.",
    tex: [
      String.raw`t \le t' \;\Rightarrow\; \dot Q(t) \le \dot Q(t')`,
      String.raw`z_i(t') \le z_i(t), \qquad T_i(t) \le T_i(t')`,
      String.raw`T_i(t) \ge T_{\mathrm{crit}} \;\Rightarrow\; T_i(t') \ge T_{\mathrm{crit}}`,
    ],
    files: ["Zone", "FireDynamics"],
    status: "Proved",
  },
  {
    property: "An assumed stair is at least as long as the stair it stands for",
    meaning:
      "Where a stair's treads are not modelled, its assumed line rises from one floor to the next in flights whose plan length is the rise over the riser times the going, and its height never falls along it. So it is at least about 1.89 times the rise long with 175 mm risers and 280 mm goings, as a real stair of those steps would be. Separately, with the per-part exit rule, every part of the graph at or above the size threshold that holds a fallback door has an exit that every node in it can reach.",
    tex: [
      String.raw`P = \frac{h_b - h_a}{r}\, g`,
      String.raw`\ell \ge \sqrt{P^{2} + (h_b - h_a)^{2}}`,
      String.raw`\quad = (h_b - h_a)\,\frac{\sqrt{g^{2} + r^{2}}}{r}`,
      String.raw`\ell \ge 1.886\,(h_b - h_a) \quad (r = 0.175,\ g = 0.28)`,
    ],
    files: ["Cores"],
    status: "Proved",
  },
  {
    property: "The Flamescapes port's reactions and heat exchange are exact over a step",
    meaning:
      "On a coarse grid the browser cannot take the paper's millisecond steps, so each reaction is integrated exactly over a step: the fuel left is the exact solution of first-order decay, never negative and never more than before, and the paper's explicit update is its first-order expansion. Each solid-gas exchange is the exact relaxation of two bodies: energy is conserved and neither temperature passes the other.",
    tex: [
      String.raw`m' = m\,e^{-k\Delta t}, \quad 0 \le m' \le m`,
      String.raw`\bigl| m(1 - k\Delta t) - m' \bigr|`,
      String.raw`\quad \le \tfrac{1}{2} m (k\Delta t)^{2}`,
      String.raw`q = (T_a - T_b)\,\mu \bigl(1 - e^{-G\Delta t/\mu}\bigr)`,
      String.raw`\mu = \frac{C_a C_b}{C_a + C_b}`,
      String.raw`C_a T_a' + C_b T_b' = C_a T_a + C_b T_b`,
    ],
    files: ["FlamescapesPort"],
    status: "Proved",
  },
  {
    property: "Fire-X's heat release is a fixed heat per kilogram of oxygen",
    meaning:
      "Fire-X derives the heat of combustion of a fuel with c carbon, h hydrogen and o oxygen atoms per molecule from the oxygen it consumes: 417 kJ for each mole of O2, whatever the fuel. That is 13.03 MJ per kilogram of oxygen, within 0.53% of Huggett's oxygen-consumption constant, so Studio's solver releases heat in proportion to the oxygen each fuel burns (metals, class D, excepted).",
    tex: [
      String.raw`\Delta_c H = -417\,\bigl(c + \tfrac{h}{4} - \tfrac{o}{2}\bigr)`,
      String.raw`\frac{417\ \mathrm{kJ/mol}}{0.032\ \mathrm{kg/mol}}`,
      String.raw`\quad = 13.03\ \mathrm{MJ/kg}`,
      String.raw`\Bigl|1 - \frac{13.03}{13.1}\Bigr| < 0.0053`,
    ],
    files: ["FireX", "FlamescapesPort"],
    status: "Proved",
  },
  {
    property: "Running ventilation moves smoke and heat between rooms without making or losing any",
    meaning:
      "With the ventilation running, each air terminal draws a share f of its cell's gas into the ducts, taken as one well-mixed volume, and gets the same mass of the mixture back. Over the system, the mass of each gas (soot and CO among them) and the heat are unchanged, and no terminal's cell leaves the range of values the terminals held before, so a mass fraction stays between 0 and 1.",
    tex: [
      String.raw`x_t' = (1 - f)\,x_t + f\,\bar{x}`,
      String.raw`\bar{x} = \frac{\sum_t M_t x_t}{\sum_t M_t}`,
      String.raw`\sum_t M_t x_t' = \sum_t M_t x_t`,
      String.raw`\min_u x_u \le x_t' \le \max_u x_u`,
    ],
    files: ["DuctMixing"],
    status: "Proved",
  },
  {
    property: "A queue at a door never passes more people than the door can, and has passed everyone by the time the check reads",
    meaning:
      "In the performance check, each door passes at most its capacity c in a time bin of length Δ (Lindley's recursion). What arrived up to bin b has left by the end of bin b + ⌈Q/(cΔ)⌉, that bound never falls for a later arrival, and the time spent queueing enters the check exactly as ASET at least f times RSET and at least the margin over it, with RSET counting the wait.",
    tex: [
      String.raw`Q_{b+1} = \max(0,\ Q_b + a_b - c\,\Delta)`,
      String.raw`\textstyle\sum_{j \le b} a_j`,
      String.raw`\quad \le \textstyle\sum_{j < \text{exit}(b)} \text{out}_j`,
      String.raw`\text{exit}(b) = b + 1`,
      String.raw`\quad + \lceil Q_{b+1}/(c\,\Delta) \rceil`,
      String.raw`L \ge \text{req}(\tau + W) + W`,
      String.raw`\quad \iff f\,(\text{base} + \tau + W) \le L + \tau`,
    ],
    files: ["DoorQueue", "PreMovement"],
    status: "Proved",
  },
  {
    property: "A sprinkler opens no later in hotter gas, and an open head's cap on the fuel never rises",
    meaning:
      "The link's exact step never leaves the range of its own, the gas's and the mount's temperatures, for any step length, and gas at least as hot never opens a head later. Under an open head each fuel cell is held to its rate when the head opened times exp(-(t - t_act)/tau), tau from Evans's spray density; the cap never rises, and the heat it lets through after opening is at most the rate times (tau + dt).",
    tex: [
      String.raw`T_l' = T_\infty + (T_l - T_\infty)\,e^{-(a+b)\,\Delta t}`,
      String.raw`a = \sqrt{u}/\mathrm{RTI},\ b = C/\mathrm{RTI}`,
      String.raw`\tau = 3.0\,w''^{-1.85}`,
      String.raw`\textstyle\sum_n r\,e^{-n\,\Delta t/\tau}\,\Delta t \le r\,(\tau + \Delta t)`,
    ],
    files: ["Sprinklers", "Suppression"],
    status: "Proved",
  },
  {
    property: "The sprinklers' spray keeps its water and the gas's energy, and only dilutes the oxygen",
    meaning:
      "Down each column the water emitted equals the water evaporated plus the water reaching the floor, and the flux never grows downward. Evaporation conserves the energy, the vapour carrying its own heat, never cools a cell below boiling, keeps every mass fraction between 0 and 1, and never raises the oxygen's mole fraction. At a fixed temperature, steam scales the gas burn down by the dilution factor.",
    tex: [
      String.raw`m_0 = \textstyle\sum_j e_j + m_n`,
      String.raw`C_g T = (C_g + c_v\,dv)\,T'`,
      String.raw`\quad + dv\,(e - c_v T_B)`,
      String.raw`T_B \le T' \le T`,
      String.raw`X_{O_2}' \le X_{O_2}`,
    ],
    files: ["Suppression"],
    status: "Proved",
  },
  {
    property: "The shielded learner never walks into the fire, and stops only at the earliest escape",
    meaning:
      "D(w) is the last time step at which node w can still be left for an exit before the fire closes the way. The shield allows a move into w at step t only if t + 1 is at most D(w). Every walk inside the shield reaches an exit whenever any escape exists, and the shield is never empty while one does. When a learning trial changes no value, the route it walked is the earliest escape.",
    tex: [
      String.raw`\text{move } v \to w \text{ at } t \text{ allowed}`,
      String.raw`\quad \iff t + 1 \le D(w)`,
      String.raw`\exists\ \text{escape from } (s, 0)`,
      String.raw`\quad \Rightarrow \text{every shielded walk escapes}`,
      String.raw`\text{a trial without change} \Rightarrow`,
      String.raw`\quad \text{moves} = \text{earliest escape}`,
    ],
    files: ["SafeEscape"],
    status: "Proved",
  },
  {
    property: "The slime mould's trail spreads without making or losing any, and two agents never share a place",
    meaning:
      "One step of Jones's diffusion makes each cell (1 - d) times the mean of its 3 by 3 window. On the wrap-around grid this leaves exactly (1 - d) times the total trail and, for d at most 1, never makes it negative or lifts a cell above (1 - d) times the old largest; where walls absorb, the total only falls faster. An agent keeps its heading when the sensor ahead is strictly the strongest, else turns towards the stronger side; on a tie the code turns left, where the 2010 paper turns at random. Moving, in any order, never puts two agents on one place or one on a wall.",
    tex: [
      String.raw`D_d(T)(x) = (1-d)\,\overline{T}_{3\times3}(x)`,
      String.raw`\textstyle\sum_x D_d(T)(x)`,
      String.raw`\quad = (1-d)\textstyle\sum_x T(x)`,
      String.raw`0 \le T \le M \;\Rightarrow`,
      String.raw`\quad 0 \le D_d(T) \le (1-d)\,M`,
      String.raw`f > f_L,\ f > f_R \Rightarrow \text{keep}`,
      String.raw`\text{valid}(p) \Rightarrow \text{valid}(\text{move}(p))`,
    ],
    files: ["Physarum"],
    status: "Proved",
  },
  {
    property: "The fine grid's walk on 16 directions is at most 2.75 % longer than the straight walk in the open",
    meaning:
      "The egress map's measuring grid walks in 16 directions, the king's and the knight's moves. A walk to a point between two adjacent directions goes a along one and b along the other, and the straight walk squared is a² + b² + 2ab cos θ. Then (a + b)²(1 + cos θ) is at most twice that, the difference being (1 − cos θ)(a − b)². The widest gap between adjacent directions is the knight's, cos θ = 2/√5, so the walk is under 1.0276 times the straight one; the graph's 8 directions allow 8.24 %.",
    tex: [
      String.raw`(a+b)^2 (1+c) \le 2\,(a^2+b^2+2abc)`,
      String.raw`\quad c \le 1`,
      String.raw`c = 2/\sqrt5 > 0.8944`,
      String.raw`(a+b)^2 \le 1.0558\,(a^2+b^2+2abc)`,
      String.raw`\sqrt{1.0558} < 1.0276`,
    ],
    files: ["Sixteen"],
    status: "Proved",
  },
];
