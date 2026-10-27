# Fire spread and egress in Topologic Studio: methods for a research-grade, Lean-checkable model

Date: 2026-10-27
Status: literature and methods study. No code was changed.
Scope: `topologicpy-web-backend/app/graphs/{fire,pathfinding,rl,alternatives}.py` and `app/routers/simulation.py` on branch `next/studio-2`, read against [../roadmap/review-2026-10-27.md](../roadmap/review-2026-10-27.md), section 5.
Bibliography: [fire-egress-methods.bib](fire-egress-methods.bib). Citation keys in square brackets refer to it. Every DOI there was checked against Crossref and OpenAlex on 2026-10-27; section 11 lists what could not be checked.

Conventions used below. "Derivation" marks a result worked out in this study, with its steps shown; it is not a quotation. "Flag" marks a constant or equation that is reproduced from a secondary source or from recall and should be checked against the primary text before it is implemented. Equations without either mark carry the citation of the source they come from.

---

## 0. Summary

Studio's fire and egress stack is a set of graph algorithms on a navigation lattice (about 0.5 m spacing, degree about 12). The algorithms are mostly sound as graph algorithms. The modelling around them is not yet physical: there is no time unit, no length scale in the fire, and the "temperature" field is an arrival-order indicator rather than a temperature. To reach the standard of the KAUST Visual Computing Center work the user named (Michels's Computational Sciences Group on combustion and atmospheric simulation; Wonka's group on building layouts), the models need governing equations with units and sources, discretisations with stated stability conditions, and comparison against a reference solver or measurements.

The main findings are these.

1. Five things in the current code are mathematically wrong rather than simplified (section 2.3). The Classic temperature step violates the discrete maximum principle at degree-1 nodes. Its field has a uniform steady state at the source value, so it cannot be read as a temperature. The hop clock makes fire and walking speeds depend on lattice orientation and on A* tie-breaking. Live rerouting re-solves from the original start with a hazard snapshot taken at the wrong time. The RL potential shaping gives burn terminals a non-zero potential, so the policy-invariance result it cites does not cover the implementation.
2. Several things that look questionable are correct: the earliest-arrival BFS is exact under the hop model (proof in section 2.4), the Euclidean A* heuristic is consistent and not only admissible, and the time-indexed RL state is what a finite-horizon problem requires.
3. A tractable, physically grounded replacement fits the graph in three layers: an arrival-time (eikonal) front on the lattice, a two-zone model (CFAST-style) on a room graph, and fractional effective dose (FED) tenability evaluated at the evacuee's own arrival time. Each layer has equations and parameters in the fire-safety literature (section 3, 4).
4. Routing in a growing fire is a time-dependent shortest-path problem. With a monotone fire the arrival functions satisfy the FIFO (non-overtaking) condition, so time-dependent Dijkstra and A* stay exact for earliest arrival. Exposure-minimising routes need a time-expanded dynamic programme, which is also exact and cheap at Studio's graph sizes (section 5).
5. For a known, deterministic fire the RL problem has an exact polynomial-time solution (backward induction on the time-expanded graph). RL results should be reported as a gap to that optimum (section 7).
6. About twenty theorems are candidates for Lean 4 + Mathlib. Roughly half are hours to days of work for someone fluent in Mathlib; the algorithm-correctness theorems (Dijkstra, A*, stochastic MDP shaping) are weeks each (section 8).

---

## 1. What "KAUST VCC level" implies

### 1.1 Attributions, checked

The user's reading of "KAUST VCC" is the Visual Computing Center at King Abdullah University of Science and Technology. The attributions were checked against Crossref author lists and OpenAlex affiliation records.

| Group | Paper | What the abstract reports | Check |
| --- | --- | --- | --- |
| Michels (Computational Sciences Group) | Pirk et al. 2017, interactive wood combustion [pirk2017wood] | Particles carrying biological and physical attributes drive the exothermic reaction; coupled rod physics; plausibility evaluated against combustion of real wood samples. | Confirmed. TOG 36(6), SIGGRAPH Asia 2017. OpenAlex lists Hädrich and Michels at KAUST, Pirk at Stanford. |
| Michels | Hädrich et al. 2021, "Fire in Paradise" [hadrich2021fire] | A mathematical formulation for plant combustion with heat transfer, char insulation and mass loss, and for fire propagation through an ecosystem, at interactive rates. | Confirmed. TOG 40(4). Five authors; Hädrich and Michels at KAUST. |
| Michels | Hädrich et al. 2020, "Stormscapes" [hadrich2020stormscapes] | First-principle buoyancy and pressure formulations; quantitative assessment with cloud-fraction profiles. | Added; shows the group's validation practice. |
| Michels | Kokosza et al. 2024, "Scintilla" [kokosza2024scintilla] | Convection, combustion and heat transfer between vegetation, soil and atmosphere; firebrands; comparison to real wildfire data. | Added. |
| Michels | Liu et al. 2025, "FLAMEFORGE" [liu2025flameforge] | Volumetric combustion of wooden structures on a voxel grid, a signed distance field for char insulation, multiresolution data structures; quantitative comparison with sub-surface measurements of a real combustion experiment. | Added; the closest of the group's papers to building fire. |
| Michels | Wrede et al. 2025, "Fire-X" [wrede2025firex] | Multi-species thermodynamics, stoichiometry-dependent heat release, sprinkler and spray suppression, fire detectors, indoor scenarios. | Added. |
| Michels (numerics) | Michels et al. 2014 [michels2014exponential]; Michels, Luan and Tokman 2017 [michels2017stiffly]; Shao et al. 2022 [shao2022multigrid] | Exponential and stiffly accurate integrators that remove step-size limits from stiff terms; algebraic multigrid for the pressure Poisson equation. | Added. The 2017 co-authors are Luan and Tokman (UC Merced); Michels is listed at Stanford on that paper. |
| Wonka | Bao et al. 2013 [bao2013layouts]; Peng et al. 2014 [peng2014templates]; Peng et al. 2016 [peng2016network]; Wu et al. 2018 [wu2018miqp]; Para et al. 2021 [para2021constraint] | Analytic characterisation of the space of valid building layouts; discrete-then-continuous layout optimisation; integer programming of networks from functional specifications, including floorplans that "determine the flow of people"; MIQP interior layout; constraint-graph generative layouts. | Confirmed. OpenAlex lists Wonka at Arizona State University on the 2013 and 2014 papers and at ASU and KAUST on the 2016 paper. |
| Not Wonka | Wu et al. 2019, data-driven interior plan generation [wu2019interior] | | Often grouped with Wonka's layout work, but its authors are Wu, Fu, Tang, Wang, Qi and Liu; no Wonka or KAUST author. |

Older graphics fire and fluid work that the Michels papers build on includes stable fluids [stam1999stable] and physically based fire [nguyen2002fire]. Continuum crowds [treuille2006continuum] is the graphics paper closest to egress: a dynamic potential field (an eikonal-type solve) drives a continuum crowd.

### 1.2 What level of mathematics this implies for Studio

Judged from these abstracts, the group's standard has four parts. Full texts were not read, so this is a reading of what the papers claim, not of how they prove it.

1. Governing equations from first principles, with closures for the parts that are not resolved (char insulation, mass loss, heat release by stoichiometry).
2. Discretisations chosen for stability: exponential and stiffly accurate integrators that tolerate large time steps, and multigrid for the elliptic pressure solve. Stability is engineered rather than hoped for.
3. Quantitative comparison with measurements (wood samples, sub-surface temperatures, wildfire records, cloud fraction).
4. In the layout work, optimisation formulations whose hard constraints are guaranteed by construction (integer programming).

Formal convergence proofs do not appear among the claims in these abstracts. For Studio the same standard reads as follows:

| Ingredient | Meaning for a building graph |
| --- | --- |
| Continuum formulation | State the PDE or ODE each graph model approximates, with units: the eikonal equation for a front, conservation of mass and energy for smoke layers, a dose integral for tenability. |
| Discretisation | Say what the graph operator approximates and how badly: metrication error of graph distance (section 3.1), conservative form of graph diffusion (section 3.5). |
| Stability | Prove the time-step condition (explicit Euler on the graph Laplacian) or use an unconditionally monotone integrator (implicit Euler, exponential integrator [hochbruck2010exponential]). |
| Convergence | Where theory exists, use it: monotone, stable and consistent schemes converge to the viscosity solution of the eikonal equation [barles1991convergence; crandall1983viscosity]. Where it does not (graph Dijkstra on a fixed stencil), bound the error and state the bound. |
| Validation | Compare the zone model with CFAST on the same rooms [peacock2015cfast], and egress times with the SFPE hydraulic hand calculation [gwynne2016hydraulic]. |
| Formal check | Lean 4 for the discrete statements (maximum principle, FIFO label-setting, shaping), and for dimensional homogeneity of every equation. |

What transfers to a navigation graph: arrival-time fronts, conservation-law ODEs on a room graph (the graph analogue of a finite-volume method), positivity and maximum principles of graph operators, implicit or exponential time stepping, dimensional bookkeeping, and network optimisation in the style of Peng et al. [peng2016network]. What does not transfer: resolved fluid mechanics and combustion chemistry. The 0.5 m lattice has no momentum equation and its edges are not control-volume faces, so it cannot be turned into a CFD mesh. Where resolved fields are needed, they should come from a solver such as FDS [mcgrattan2013fds] and be sampled onto the graph.

---

## 2. Diagnosis of the current models

### 2.1 Notation

$G=(V,E)$ is the navigation graph, $n=|V|$, $x_i\in\mathbb{R}^3$ the node positions, $\ell_{ij}=\lVert x_i-x_j\rVert$ the edge lengths, $N(i)$ the neighbours of $i$, $d_i=|N(i)|$ its degree, and $A_i\subseteq N(i)$ the neighbours across edges not blocked by walls, with $c_i=|A_i|$.

### 2.2 What each model computes

| Model | Code | Rule |
| --- | --- | --- |
| Radial fire | `fire.radial_timeline` | Node $i$ ignites at step $\lfloor \lVert x_i-x_s\rVert / \tilde\ell \rfloor$, with $\tilde\ell$ the median edge length. Walls ignored. |
| Flood fire | `fire.flood_timeline` | BFS hop count from the seed over unblocked edges. |
| Temperature fire | `fire.temperature_steps` | $m_i=\frac1{c_i}\sum_{j\in A_i}T_j$; $T_i^{+}=T_i+\frac{\kappa}{d_i}\max(m_i-T_i,0)$ with $\kappa=1.2$; source reset to $T_f=120$; clip to $[20,120]$. |
| Hazard routing | `pathfinding.hazard_weights`, `astar` | $w_e=\ell_e\,(1+\alpha\,h_e)$ with $h_e=\max(0,(\bar T_e-20)/100)$ from the current step; A* with $h(i)=\lVert x_i-x_{goal}\rVert$. Edges with $\bar T_e$ above a threshold are blocked. |
| Live rerouting | `simulation.fire_stream.reroute` | Every $N$ steps, A* from `walker_start` to the exit with the current step's weights and blocks. |
| RL | `rl.q_learning_path` | Tabular Q-learning; state $(v,t)$ under fire; reward $-c\,\ell_e$, $+10$ at the exit, $-10$ on entering a node with $t\ge t^{ign}_v$; shaping $F=\gamma\Phi(s')-\Phi(s)$ with $\Phi(v)=-c\,d(v,\text{exit})$. |
| Escape check | `rl.earliest_escape` | BFS in time layers; a move at step $t$ into $v$ is allowed if $t<t^{ign}_v$; visited nodes are not re-entered. |

### 2.3 Errors (mathematically wrong, not only simplified)

W1. The temperature step violates the discrete maximum principle at degree-1 nodes (derivation). For $d_i\ge 2$ the update is the convex combination
$$T_i^{+}=(1-\beta_i)\,T_i+\beta_i\,m_i,\qquad \beta_i=\kappa/d_i\in(0,1],$$
so $T_i\le T_i^{+}\le m_i\le\max_{j\in A_i}T_j$ whenever $m_i>T_i$. For $d_i=1$ with one hotter neighbour $j$, $\beta_i=1.2$ and
$$T_i^{+}-T_j=(1-1.2)(T_i-T_j)=0.2\,(T_j-T_i)>0 .$$
The new value of the dead end exceeds the previous maximum of its closed neighbourhood, which is a violation of the discrete maximum principle $T_i^{n+1}\le\max_{j\in\{i\}\cup N(i)}T_j^{n}$. Whether it also exceeds the neighbour's new value depends on how fast the neighbour heats: with $e^n=T_j^n-T_i^n$, $T_i^{n+1}-T_j^{n+1}=0.2\,e^n-(T_j^{n+1}-T_j^{n})$. If the neighbour is static (a local plateau), the dead end ends up hotter than its only neighbour and, because the rule never cools, stays so until the neighbour passes it; next to the held source the overshoot (for example $20+1.2\cdot100=140$) is removed by the clip at 120. Check on the unmodified code (this study): on a dead end attached to a degree-12 hub, the maximum principle failed at every step from step 2 (for example 21.000 against a previous neighbourhood maximum of 20.833); on that graph and on a three-node path the neighbour heated fast enough that no same-step inversion appeared. The clip to $[20,120]$ bounds the global range only. The docstring's "each node relaxes toward the mean of its neighbours" is false for $d_i=1$. The coefficient is a property of the formula; the review records it as inherited from Classic [jabi2019synergy].

W2. The temperature field is not a temperature (derivation). The update has no time step, no length scale, no loss to ambient and no conservation: heat gained by a cool node is not lost by the hot one, so $\sum_i T_i$ is non-decreasing even without a source. Its only steady state on the connected component of the source is $T\equiv T_f$. Proof sketch: each $T_i$ is non-decreasing and bounded by $T_f$, so the iteration converges, and the limit $T^\ast$ is a fixed point of the (continuous) update. Let $\mu=\min T^\ast$ over the component and suppose $\mu<T_f$. The set $U=\{i:T^\ast_i=\mu\}$ excludes the source, so connectivity gives $i\in U$ with an active neighbour $j\notin U$. Then $m_i>\mu=T^\ast_i$, so the gain at $i$ is positive, contradicting the fixed point. Hence $\mu=T_f$. The field therefore encodes arrival order and rate only; reading it in °C, or comparing it to a lethality threshold in °C, has no physical meaning.

W3. The hop clock is orientation- and mesh-dependent (derivation). Fire and evacuee advance one edge per tick, while edges run from 0.5 to 2.25 m. On an ideal square sampling of spacing $h$, the 12 nearest neighbours of a node are the stencil $S_{12}=\{(\pm1,0),(0,\pm1),(\pm1,\pm1),(\pm2,0),(0,\pm2)\}h$. The hop distance to lattice point $(a,b)$ is
$$\operatorname{hop}(a,b)=\left\lceil \tfrac{|a|+|b|}{2}\right\rceil .$$
Lower bound: every step in $S_{12}$ changes $|x|+|y|$ by at most 2. Upper bound, for $a\ge b\ge0$: take $b$ diagonal steps and cover the remaining $a-b$ with $\lceil (a-b)/2\rceil$ steps of length 2 or 1, giving $b+\lceil (a-b)/2\rceil=\lceil (a+b)/2\rceil$. The flood front after $k$ ticks is therefore the $L^1$ ball of radius $2kh$: it advances $2h$ per tick along a lattice axis and $\sqrt2\,h$ per tick along a diagonal, a ratio of $\sqrt2$ (1.0 against 0.71 m per tick at $h=0.5$ m). The evacuee is no better defined: along an axial corridor one $2h$ edge and two $h$ edges have equal metric cost, so A* tie-breaking decides whether a tick is 0.5 m or 1.0 m of walking. A fire race in this model is decided by sampling orientation and tie-breaking, and the outcome has no limit as the lattice is refined. Check on the unmodified code (this study): an ideal 0.5 m square sampling wired by `build_adjacency` (interior degree 12) and burned by `flood_timeline` from its centre ignites a point 10.00 m away along an axis at step 10 (1.000 m per step) and a point 9.90 m away along the diagonal at step 14 (0.707 m per step); the 5 m points give the same two speeds. Real samplings are not perfectly square, so the anisotropy on a model graph will differ in detail but not in kind.

W4. Live rerouting plans from the wrong place at the wrong time. `reroute()` always solves from `walker_start`, so the path shown at step $k$ starts where the evacuee was at step 0. It also weights or blocks every edge by its state at step $k$, not at the time the evacuee would reach it. With a growing fire, an edge clear at step $k$ can be burning on arrival, so a returned route is not guaranteed feasible. When no route survives the threshold, the fallback returns a route through the fire with `found = true` and a note; the RL path instead reports "no route escapes", which is the correct semantics.

W5. The RL shaping does not satisfy the hypothesis of the invariance result it cites (derivation). With $F(s,s')=\gamma\Phi(s')-\Phi(s)$ applied on every transition and no bootstrap after a terminal transition, the shaped discounted return of an episode $s_0,\dots,s_T$ telescopes:
$$G'=\sum_{t=0}^{T-1}\gamma^t\big(r_t+\gamma\Phi(s_{t+1})-\Phi(s_t)\big)=G+\gamma^{T}\Phi(s_T)-\Phi(s_0).$$
At the exit $\Phi=0$, so escaping episodes are shifted by the constant $-\Phi(s_0)$. At a burn terminal $b$, $\Phi(b)=-c\,d(b,\text{exit})<0$, so burning episodes are penalised by a further, trajectory-dependent $\gamma^T\Phi(b)$. Policy invariance in episodic tasks needs the terminal potential to be zero (or terminals modelled as absorbing states that keep receiving shaping) [ng1999policy; grzes2017episodic]. What does hold here is weaker: since shaping only lowers burning returns and shifts all escaping returns by the same constant, an escaping policy that is optimal without shaping is still optimal with it (deterministic environment). When no escape exists, shaping can change which burning behaviour is optimal. A related defect: in the timed mode the last step bootstraps from untrained prior values at $t=H$, whereas a finite-horizon problem with time in the state should treat $t=H$ as terminal with value 0 [pardo2018timelimits; puterman1994mdp]. The fix for both is to use $\Phi=0$ on terminal transitions and stop at $t=H$.

A precision note, not an error: A* with a closed set and no reopening, as implemented, needs a consistent heuristic, not only an admissible one. The Euclidean heuristic is consistent because $w_{ij}\ge\ell_{ij}\ge h(i)-h(j)$ by the triangle inequality (section 5.4). Edge lengths are stored as float32 while the heuristic is evaluated in float64, so consistency can fail by rounding at about $10^{-7}$ relative. The "true optimum" claim holds up to that tolerance.

### 2.4 Correct as implemented

C1. `earliest_escape` is exact under the hop model (derivation). Let $e^\ast(v)$ be the earliest step at which a safe walk (one that enters each node $u$ at a step $t<t^{ign}_u$) reaches $v$. Dominance lemma: if a walk from $v$ starting at step $t'$ is safe, the same walk starting at any $t\le t'$ is safe, because each node is entered earlier and the fire is monotone. Induction on $t$: the BFS layer that first contains $v$ is $e^\ast(v)$. If an optimal walk reaches $w$ at $e^\ast(w)$ through $v$, then $e^\ast(v)\le e^\ast(w)-1$; by induction BFS has $v$ at $e^\ast(v)$, and by dominance the step into $w$ at $e^\ast(v)+1\le e^\ast(w)$ is safe. Conversely every BFS label is realised by a safe walk. Pruning visited nodes loses nothing, and no waiting is needed.

C2. The greedy rollout's "skip visited nodes" rule is justified by the same dominance lemma plus positive step costs: removing a cycle from a safe walk keeps it safe and lowers its cost.

C3. The time-indexed state $(v,t)$ is required: optimal policies of finite-horizon problems are in general non-stationary [puterman1994mdp].

C4. The radial model is the exact solution of the eikonal equation $F|\nabla T|=1$ in free space with uniform $F=\tilde\ell$ per tick. Its error is that buildings are not free space: it passes through walls.

---

## 3. Fire and smoke on a building graph (Question 1)

### 3.1 Arrival-time, eikonal and level-set formulations

Continuum. With front speed $F(x)>0$ normal to the front, the first arrival time $T(x)$ solves the static Hamilton-Jacobi (eikonal) problem
$$F(x)\,\lvert\nabla T(x)\rvert=1,\qquad T=0 \text{ on the ignition set } \Gamma,$$
in the viscosity sense [crandall1983viscosity]. The level-set method evolves the front as the zero set of a function satisfying a Hamilton-Jacobi equation [osher1988fronts]; the fast marching method solves the eikonal case in one ordered pass, borrowing from Dijkstra's ordering [sethian1996fast]. Anisotropic spread (wind or slope, the elliptical Huygens model of wildland fire [richards1990elliptical], with spread rates from Rothermel [rothermel1972]) needs ordered upwind methods; Sethian and Vladimirsky prove convergence of one such scheme to the viscosity solution [sethian2003ordered]. The general convergence argument is that monotone, stable and consistent schemes converge when the limit equation has a comparison principle [barles1991convergence].

Graph. On a weighted graph the discrete analogue is the Bellman equation
$$T_i=\min_{j\in A_i}\Big(T_j+\frac{\ell_{ij}}{F_{ij}}\Big),\qquad T_s=0,$$
which Dijkstra's algorithm solves exactly in $O(|E|\log|V|)$ [dijkstra1959note]. Tsitsiklis gives Dijkstra-like and Dial-like one-pass algorithms for a discretised Hamilton-Jacobi equation of this type [tsitsiklis1995efficient], and Finney uses minimum travel time on a two-dimensional network as a fire-growth model and reports perimeters essentially identical to perimeter-expansion methods [finney2002fire]. Exact Euclidean geodesics on polyhedral surfaces need different algorithms again [mitchell1987discrete].

Metrication error of graph distance (derivation). Graph Dijkstra solves the graph problem exactly but approximates the Euclidean metric with an error that does not shrink under refinement. Let two consecutive stencil directions be separated by angle $\varphi$, and let the target $x$ make angle $\psi\in[0,\varphi]$ with the first. Writing $x=a\,u_1+b\,u_2$ with $a,b\ge0$, the sine rule gives $a=|x|\sin(\varphi-\psi)/\sin\varphi$ and $b=|x|\sin\psi/\sin\varphi$, so
$$\frac{a+b}{|x|}=\frac{\sin(\varphi-\psi)+\sin\psi}{\sin\varphi}=\frac{\cos(\varphi/2-\psi)}{\cos(\varphi/2)}\le\frac1{\cos(\varphi/2)} .$$
For the 8-direction stencil ($\varphi=45^\circ$) the worst overestimate is $1/\cos 22.5^\circ\approx1.082$; adding the $(\pm2,\pm1)$ and $(\pm1,\pm2)$ directions (largest gap $26.57^\circ$) reduces it to $1/\cos 13.28^\circ\approx1.027$. Studio's metric routes and a metric Dijkstra fire inherit at most the 8.2% figure in open floor; the hop-count flood (W3) has a 41% anisotropy. The 8% route-length difference against TopologicPy's `Navigation` graph in the review is of the same size as this bound, which is consistent with, but does not establish, a stencil effect.

Room scale. Inside buildings the front is not a material property: fire spreads by item-to-item ignition, flashover, and failure of barriers. Network models represent this at room scale, with rooms as nodes and spread delays (time to flashover, barrier fire resistance) on edges [elms1984modeling; ling1985network; platt1994probabilistic; cheng2011dynamic]. The arrival time of fire in room $j$ is then the same Bellman equation with edge weights in seconds,
$$T_j=\min_{i}\big(T_i+t^{fo}_i+t^{bar}_{ij}\big),$$
solved by Dijkstra. This is the graph form of Finney's minimum travel time at building scale.

Recommendation. Replace the hop-count flood with Dijkstra on $\ell_e/F$ over unblocked edges (cheap, exact on the graph, error bounded by the metrication factor), and treat flame spread between rooms with the room-graph form above. Keep the radial model only as a labelled free-space reference.

### 3.2 Design fire: t-squared growth

The heat release rate in the growth phase is modelled as
$$\dot Q(t)=\alpha\,(t-t_0)^2,\qquad \alpha=\frac{\dot Q_{ref}}{t_g^2},\quad \dot Q_{ref}=1055\ \text{kW},$$
following Heskestad and Delichatsios on the initial convective flow in fire [heskestad1979initial] and the growth classes tabulated in NFPA 72 and in Karlsson and Quintiere [nfpa72; karlsson2022enclosure].

| Class | $t_g$ (s) | $\alpha$ (kW s$^{-2}$) |
| --- | --- | --- |
| Slow | 600 | 0.00293 |
| Medium | 300 | 0.0117 |
| Fast | 150 | 0.0469 |
| Ultrafast | 75 | 0.188 |

The $\alpha$ column is computed from $\alpha=1055/t_g^2$. Growth is capped by fuel, ventilation, or flashover.

Flashover estimate. Babrauskas gives the heat release rate needed for flashover as [babrauskas1980flashover]
$$\dot Q_{fo}\approx750\,A_o\sqrt{H_o}\quad(\text{kW};\ A_o\text{ in m}^2,\ H_o\text{ in m}),$$
with $A_o$ and $H_o$ the area and height of the opening (flag: constant as reproduced in [karlsson2022enclosure; drysdale2011introduction]). Worked example (derivation): one door of $0.9\times2.0$ m gives $\dot Q_{fo}\approx750\cdot1.8\cdot\sqrt2\approx1.9$ MW, reached $\sqrt{\dot Q_{fo}/\alpha}\approx200$ s after $t_0$ for a fast fire and about 400 s for a medium one. The time scale of the problem is minutes, which is what a metric clock must resolve.

### 3.3 Plume, ceiling jet and room temperature correlations

Plume (Heskestad, flag: constants as reproduced in [heskestad2016plumes]; original papers [heskestad1983luminous; heskestad1983virtual; heskestad1984engineering]). Mean flame height and virtual origin, with $\dot Q$ in kW and fire diameter $D$ in m:
$$L=0.235\,\dot Q^{2/5}-1.02\,D,\qquad z_0=0.083\,\dot Q^{2/5}-1.02\,D .$$
Plume mass flow above the flame, with $\dot Q_c$ the convective part of $\dot Q$:
$$\dot m_p=0.071\,\dot Q_c^{1/3}(z-z_0)^{5/3}\Big[1+0.026\,\dot Q_c^{2/3}(z-z_0)^{-5/3}\Big]\ \ \text{kg s}^{-1},\qquad z>L .$$
Zukoski, Kubota and Cetegen give the point-source form $\dot m_p\propto \dot Q^{1/3}z^{5/3}$ with an empirically fitted coefficient [zukoski1981entrainment]. Which entrainment law a zone model uses is a modelling choice that should be stated; the CFAST technical reference documents its own [peacock2015cfast].

Ceiling jet (Alpert, flag: constants as reproduced in [alpert2016ceiling]; original [alpert1972]). Under an unconfined ceiling of height $H$ above the fuel, at radius $r$, before a hot layer forms:
$$\Delta T=\begin{cases}16.9\,\dot Q^{2/3}H^{-5/3}, & r/H\le0.18,\\ 5.38\,(\dot Q/r)^{2/3}H^{-1}, & r/H>0.18,\end{cases}\qquad U=\begin{cases}0.96\,(\dot Q/H)^{1/3}, & r/H\le0.15,\\ 0.195\,\dot Q^{1/3}H^{1/2}r^{-5/6}, & r/H>0.15.\end{cases}$$
Worked example (derivation): $\dot Q=1$ MW, $H=2.7$ m, $r=5$ m gives $\Delta T\approx68$ K and $U\approx0.84$ m s$^{-1}$. The early smoke front moves at walking speed, which gives the review's "fire speed ratio of 1" a physical reading for smoke near a large fire, but not for flame spread.

Room temperature before flashover (McCaffrey, Quintiere and Harkleroad, MQH; flag: as reproduced in [karlsson2022enclosure]; original [mccaffrey1981estimating]):
$$\Delta T_g=6.85\left(\frac{\dot Q^2}{A_o\sqrt{H_o}\,h_k A_T}\right)^{1/3},\qquad h_k=\begin{cases}\sqrt{k\rho c/t}, & t<t_p,\\ k/\delta, & t\ge t_p,\end{cases}\qquad t_p=\frac{\rho c}{k}\Big(\frac{\delta}{2}\Big)^2,$$
with $A_T$ the enclosing surface area and $k,\rho,c,\delta$ the wall conductivity, density, specific heat and thickness. The constant 6.85 carries units of $\text{K}^{2/3}\,\text{kW}^{-1/3}\,\text{m}^{5/6}$ (derivation, section 8.4): it is a dimensional correlation, valid only in the stated units, with ambient air properties absorbed into the constant.

### 3.4 Two-zone model on a room graph

Each room $k$ is split into a hot upper layer (U) and a cool lower layer (L) at uniform pressure $p$. CFAST solves this system for multi-room buildings [peacock2015cfast; jones1985multicompartment]; Forney and Moss analyse its numerical characteristics [forney1994zone]; Walton et al. summarise zone models [walton2016zone].

Derivation of the layer equations. Assume an ideal gas with constant $c_p,c_v$, $\gamma=c_p/c_v$, $R=c_p-c_v$, so a layer's internal energy is $c_v mT=pV_{layer}/(\gamma-1)$. Let $\dot h_U,\dot h_L$ (W) be the net rates of enthalpy and heat added to each layer (plume, vents, wall losses) and $\dot m_U,\dot m_L$ the net mass rates.

Step 1, whole room (fixed volume $V=V_U+V_L$): $\frac{d}{dt}\frac{pV}{\gamma-1}=\dot h_U+\dot h_L$, so
$$\frac{dp}{dt}=\frac{\gamma-1}{V}\big(\dot h_U+\dot h_L\big).$$
Step 2, upper layer as an open system with a moving boundary: $\frac{d}{dt}\frac{pV_U}{\gamma-1}=\dot h_U-p\frac{dV_U}{dt}$. Expanding, $V_U\dot p+\gamma p\dot V_U=(\gamma-1)\dot h_U$, so
$$\frac{dV_U}{dt}=\frac{(\gamma-1)\dot h_U-V_U\,\dot p}{\gamma p}.$$
Step 3, $T_U=pV_U/(Rm_U)$ with $\dot m_U$ given. Logarithmic differentiation gives $\dot T_U/T_U=\dot p/p+\dot V_U/V_U-\dot m_U/m_U$. Substituting step 2 and using $(\gamma-1)/\gamma=R/c_p$ and $T_UR/(pV_U)=1/m_U$:
$$\frac{dT_U}{dt}=\frac{1}{c_p\,\rho_U V_U}\Big(\dot h_U-c_p\,\dot m_U T_U+V_U\,\dot p\Big),$$
and the same with U replaced by L. This is the predictive equation set presented in the CFAST technical reference [peacock2015cfast; forney1994zone]. The layer interface height in a prismatic room of floor area $A_f$ is $z_i=(V-V_U)/A_f$.

On the room graph, rooms are nodes with state $(p,V_U,T_U,T_L)$ and doors are edges carrying vent mass flows computed from the hydrostatic pressure difference across the opening with an orifice law, $\dot m=C_d A\sqrt{2\rho\Delta p}$ integrated over the opening height [tanaka2016vent]. The plume term uses section 3.3, the fire term section 3.2. The pressure equation relaxes much faster than the layer equations, which makes the system stiff in practice, and the CFAST technical reference describes solving it with an implicit differential-algebraic solver (flag: check against the current CFAST version) [peacock2015cfast]. This is the point of contact with the stiff and exponential integrators of the Michels group [michels2014exponential; michels2017stiffly].

Mapping to the lattice. A lattice node $i$ in room $k$ at floor height $z^f_i$ sees the upper-layer state when $z_i(t)<z^f_i+z_{head}$ and the lower-layer state otherwise, with $z_{head}$ a design parameter. This gives every node a gas temperature history $T_i(t)$ in kelvin with a physical time axis, which is what the tenability model (section 4) needs.

Fire dynamics background for all of section 3: [drysdale2011introduction; quintiere2006fundamentals; karlsson2022enclosure; sfpe2016handbook]. A sixth edition of the SFPE Handbook exists [sfpe2026handbook]; the constants above are from the fifth edition and should be re-checked against the sixth.

### 3.5 Heat diffusion on graphs via the graph Laplacian

This section derives the stability and maximum-principle conditions and the corrected Classic step. All results in it are derivations, using standard matrix facts [berman1994nonnegative; chung1997spectral; anderson1985eigenvalues] and the finite-difference stability theory in LeVeque [leveque2007finite] and Patankar [patankar1980].

Model. Symmetric conductances $g_{ij}=g_{ji}\ge0$ (W K$^{-1}$), positive capacities $C_i$ (J K$^{-1}$), loss coefficients $H_i\ge0$ to ambient $T_a$, sources $q_i$ (W):
$$C_i\,\dot T_i=\sum_j g_{ij}(T_j-T_i)-H_i(T_i-T_a)+q_i .$$
With $G_i=\sum_j g_{ij}$, $L_g=\operatorname{diag}(G)-[g_{ij}]$ (the weighted Laplacian, $L_g\mathbf1=0$) and $\theta=T-T_a\mathbf1$: $C\dot\theta=-(L_g+H)\theta+q$.

Explicit Euler. $\theta^{n+1}=M\theta^n+\Delta t\,C^{-1}q^n$ with $M=I-\Delta t\,C^{-1}(L_g+H)$:
$$M_{ii}=1-\frac{\Delta t\,(G_i+H_i)}{C_i},\qquad M_{ij}=\frac{\Delta t\,g_{ij}}{C_i}\ (j\ne i),\qquad \sum_jM_{ij}=1-\frac{\Delta t\,H_i}{C_i}.$$

Lemma A (monotonicity). $M\ge0$ entrywise if and only if
$$\Delta t\le\min_i\frac{C_i}{G_i+H_i}.$$
Then each row of $M$ is non-negative and sums to at most 1 (substochastic).

Theorem B (discrete maximum principle). Under Lemma A and with $q=0$, $\theta^{n+1}_i=\sum_jM_{ij}\theta^n_j+\big(1-\sum_jM_{ij}\big)\cdot0$ is a convex combination of the values $\theta^n_j$ and 0, so by induction
$$\min\big(0,\min_j\theta^0_j\big)\le\theta^n_i\le\max\big(0,\max_j\theta^0_j\big)\quad\text{for all }n.$$
A node held at $\theta_f$ (Dirichlet source) extends the upper bound to $\max(\theta_f,\max_j\theta^0_j)$.

Theorem C ($\ell^2$ stability). Take uniform $C_i=C$. $L_g+H$ is symmetric positive semidefinite; by Gershgorin its eigenvalues lie in $[0,\lambda_{max}]$ with $\lambda_{max}\le\max_i(2G_i+H_i)$ (for unweighted graphs Anderson and Morley give the sharper $\max_{ij\in E}(d_i+d_j)$ [anderson1985eigenvalues]). The eigenvalues of $M$ are $1-\Delta t\lambda_k/C$, so the scheme is non-expansive in $\ell^2$ if and only if $\Delta t\le2C/\lambda_{max}$. Lemma A's condition implies this; the converse fails. For $\Delta t$ between $C/(G+H)$ and $2C/\lambda_{max}$ the scheme is stable but not monotone: it oscillates and overshoots. On a uniform $d$-dimensional lattice with $g=kh^{d-2}$ and $C=\rho c_ph^d$, Lemma A becomes the familiar Fourier-number condition $\alpha\Delta t/h^2\le1/(2d)$ [leveque2007finite].

Why Classic overshoots. Classic's gain on a node heating up is $T_i^{+}=(1-\beta_i)T_i+\beta_i m_i$, with per-node coefficient $\beta_i=\kappa/d_i$. Its row in the effective iteration matrix has diagonal $1-\kappa/d_i$, which is negative exactly when $d_i<\kappa=1.2$, that is $d_i=1$. That node sits in the "stable but not monotone" regime of Theorem C: the local error $e=T_j-T_i$ maps to $-0.2e$, contracting but changing sign, which is the overshoot of W1.

Corrected update, minimal change (derivation). Replace $\kappa/d_i$ by
$$\beta_i=\min\!\Big(1,\frac{\kappa}{d_i}\Big),\qquad T_i^{+}=T_i+\beta_i\max(m_i-T_i,0).$$
Then $T_i\le T_i^{+}\le\max(T_i,m_i)\le\max_{j\in\{i\}\cup A_i}T_j$, so if all temperatures start in $[T_a,T_f]$ they stay there and the clip becomes provably redundant. The update is bit-identical to Classic for every $d_i\ge2$ and differs only at dead ends, which keeps comparability with Classic.

Corrected update, physical form. If a graph diffusion is kept, it should be the conservative form above, with $g_{ij}$ and $C_i$ derived from geometry, a loss term $H_i$ so that the steady state decays away from the source instead of filling the building (W2), and either $\Delta t$ chosen by Lemma A or an unconditionally monotone integrator:

- Implicit Euler, $(I+\Delta t\,C^{-1}(L_g+H))\theta^{n+1}=\theta^n$. The matrix is a non-singular M-matrix, so its inverse is entrywise non-negative with row sums at most 1, and Theorem B holds for every $\Delta t$ [berman1994nonnegative].
- Exponential integrator, $\theta(t+\Delta t)=e^{-\Delta t\,C^{-1}(L_g+H)}\theta(t)$, exact for the linear system [hochbruck2010exponential]. Because $-(L_g+H)$ has non-negative off-diagonal entries, $e^{-\Delta tC^{-1}(L_g+H)}$ is entrywise non-negative and substochastic for every $\Delta t$.

Conduction through air is not how heat and smoke move through a building at this scale; buoyant plumes, ceiling jets and layer filling are (section 3.3, 3.4). A graph diffusion is at best a phenomenological smoother and should be labelled and calibrated as such.

### 3.6 Recommended fire model for Studio

| Layer | Graph | Model | Output |
| --- | --- | --- | --- |
| Flame spread | Room graph | Dijkstra with $t^{fo}+t^{bar}$ delays (3.1); t-squared growth and flashover per room (3.2) | Fire arrival time per room |
| Smoke and heat | Room graph | Two-zone ODEs, plume entrainment, vent flows (3.3, 3.4) | $z_i(t)$, $T_U(t)$, $T_L(t)$ per room |
| Near field | Lattice, fire room only | Alpert ceiling jet (3.3) until a layer forms | Early $\Delta T(r)$ |
| Node hazard | Lattice | Layer state at head height mapped to nodes (3.4) | $T_i(t)$ in kelvin, for FED (4) |
| Fallback front | Lattice | Dijkstra on $\ell_e/F$ (3.1) | Ignition time per node, when no room graph exists |

---

## 4. Tenability and hazard (Question 2)

### 4.1 Fractional effective dose

ISO 13571 [iso13571] and Purser and McAllister [purser2016hazards; purser2026hazards] estimate the time to compromised tenability by accumulating a fractional effective dose,
$$X_{FED}(t)=\int_0^t\frac{d\tau}{t_I\big(c(\tau)\big)}\approx\sum_k\frac{\Delta t_k}{t_{I,k}},$$
where $t_I(c)$ is the exposure time that would incapacitate at the constant condition $c$. Incapacitation is predicted when $X_{FED}$ reaches 1. ISO 13571 discusses lower design thresholds to protect more susceptible people (flag: the commonly quoted value is 0.3; confirm against the standard). $X_{FED}$ is dimensionless by construction (time over time).

Convective heat (flag: as reproduced from ISO 13571 in secondary sources):
$$t_{I,conv}=5\times10^{7}\,T^{-3.4}\ \text{min},\qquad T\ \text{in °C}.$$
Evaluated (derivation): about 45 min at 60 °C, 17 min at 80 °C, 7.9 min at 100 °C and 4.3 min at 120 °C. Taken literally, Classic's "fire" constant of 120 °C would be a 4-minute tolerance. Radiant heat has its own power law above a threshold flux below which no dose accumulates; the constants should be taken from the standard [iso13571; purser2016hazards]. Combined heat dose: $X_{FED,heat}=\sum_k(1/t_{I,conv}+1/t_{I,rad})\Delta t_k$.

Asphyxiant gases (flag): for CO, ISO 13571 uses an incapacitating exposure dose of the order of 35,000 µL L$^{-1}$ min, so $X_{FED,CO}\approx\sum_k\varphi_{CO,k}\Delta t_k/35000$; HCN and CO$_2$ hyperventilation enter as further terms with constants in the standard. Studio has no species model, so heat and the smoke-layer position are the operative criteria until the zone model carries species yields.

Note on units. $t_{I,conv}=5\times10^7T^{-3.4}$ is homogeneous only after $T$ is made dimensionless, $T/(1\,°\text{C})$, with the constant carrying minutes. Because °C is an interval scale, the relation is valid only in the unit system in which it was fitted. Section 8 formalises this.

### 4.2 ASET and RSET

The engineering criterion compares the available safe egress time with the required one:
$$\text{RSET}=t_{det}+t_{alarm}+t_{pre}+t_{trav},\qquad \text{ASET}_i=\inf\{t: X_{FED,i}(t)\ge X_{crit}\ \text{or another criterion at }i\},$$
with the safety margin $\text{ASET}-\text{RSET}>0$ along each occupant's route [purser2001quantification; purser2016developments; bsi2019pd79746; isotr16738]. Babrauskas, Fleming and Russell argue that the concept is flawed when used deterministically: it ignores the variation of occupants' capabilities and of fire scenarios, and it assumes "robotic" movement to the best exit [babrauskas2010rset]. The defensible use is distributional: pre-movement and speeds as distributions, several fire scenarios, and the margin reported as a distribution rather than a single number.

### 4.3 From exposure along a route to a route cost

For a route $P$ left at $t_0$, the evacuee reaches the tail of edge $e$ at $a_e$ (computed along $P$ from walking times, section 6) and is exposed on $e$ for $\tau_e$:
$$X_{FED}(P,t_0)=\sum_{e\in P}\int_{a_e}^{a_e+\tau_e}\frac{dt}{t_I\big(T_e(t)\big)} .$$
The cost of an edge depends on when it is entered, so a static snapshot (the current `hazard_weights`) evaluates it at the wrong time. Three exact formulations fit the graph.

1. Hard tenability windows. If node $v$ is untenable from $b_v$ onwards (fire arrival, $T>T_{crit}$, or layer below head height), edge $e=(u,v)$ can be entered at $t$ only if $t+\tau_e<b_v$. Earliest arrival is then a FIFO time-dependent shortest path, exact by Dijkstra (section 5).
2. Minimum dose. Discretise time with step $\Delta t$ and form the time-expanded graph with nodes $(v,k)$. Edge costs $c(e,k)$ are fixed numbers on a layered acyclic graph, so the minimum-FED route is an exact dynamic programme in $O(K|E|)$ [chabini1998discrete].
3. Earliest arrival under a cumulative dose limit. On the same layered graph compute $F^\ast(v,k)$, the least dose with which $v$ can be reached at layer $k$. The earliest feasible arrival is $\min\{k:F^\ast(\text{exit},k)\le X_{crit}\}$. Time is explicit in the layer, so the cumulative constraint needs no Pareto labels.

At Studio's sizes (about 6,300 nodes and 38,000 edges, 600 one-second layers) formulation 2 or 3 is about $4.6\times10^7$ edge relaxations, about one second in vectorised numpy (estimate, not measured). A weighted sum, time plus $\lambda X_{FED}$, is also additive and exact on the same graph.

---

## 5. Routing in a fire that grows (Question 3)

### 5.1 Time-dependent shortest paths

Let $\tau_e(t)$ be the traversal time of edge $e$ entered at time $t$, and $A_e(t)=t+\tau_e(t)$ its arrival function. Cooke and Halsey formulated the problem with time-dependent transit times [cooke1966shortest]. Dreyfus observed that Dijkstra's method extends to it by labelling nodes with earliest arrival times [dreyfus1969appraisal]. Kaufman and Smith identified the condition under which this is exact: a consistency (FIFO, non-overtaking) condition on the forecasts, under which the time-dependent fastest path costs the same as a static one [kaufman1993fastest]. Orda and Rom treat arbitrary time-dependent delays under different waiting constraints; they show that if the departure time from the source is unrestricted, a simple path exists that is as fast as the most unrestricted one, while under restricted transit the minimum delay can be finite and yet be achieved only by an infinite path [orda1990shortest]. Chabini gives optimal-time algorithms for discrete-time dynamic shortest paths [chabini1998discrete], and Foschini, Hershberger and Suri analyse the complexity of the earliest-arrival function over all departure times [foschini2014complexity].

FIFO condition:
$$t\le t'\ \Rightarrow\ A_e(t)\le A_e(t'),\qquad\text{equivalently}\quad \tau_e(t')\ge\tau_e(t)-(t'-t).$$
Under FIFO, (i) waiting at a node never gives an earlier arrival, and (ii) label-setting with labels $a(v)$ and relaxation $a(v)\leftarrow\min(a(v),A_e(a(u)))$ returns exact earliest arrival times.

### 5.2 A monotone fire gives FIFO (derivation)

With walking time $\tau_e$ and the edge closed from $b_e$ onwards,
$$A_e(t)=\begin{cases}t+\tau_e,& t+\tau_e<b_e,\\ +\infty,&\text{otherwise},\end{cases}$$
which is non-decreasing in $t$, so FIFO holds. If smoke slows walking, $\tau_e(t)$ is non-decreasing in $t$ for a monotone hazard, and $A_e(t)=t+\tau_e(t)$ is again non-decreasing. Every fire model in Studio is monotone (a burning node stays burning), so time-dependent Dijkstra is exact for earliest arrival. This replaces both the snapshot rerouting (W4) and, under a metric clock, the hop BFS (C1).

### 5.3 Indexing hazard by the evacuee's arrival time

The hazard on $e=(u,v)$ must be read at the label time $a(u)$, the moment this evacuee would enter $e$, not at the current simulation step. Since Studio's fire is precomputed and deterministic, this forecast exists: ignition time per node for the radial and flood models, $T_i(t)$ for the temperature or zone model. Rerouting then becomes receding-horizon control: at replanning time $t_r$, solve from the evacuee's current position $p(t_r)$ (a node, or a point on an edge, which is handled by a virtual start node) with departure time $t_r$.

Replanning consistency (derivation). In a FIFO network with an unchanged forecast, if $P$ is an earliest-arrival path from $s$ at $t_0$ and the evacuee reaches $v\in P$ at $a_P(v)$, the suffix of $P$ from $v$ is an earliest-arrival path from $v$ at $a_P(v)$. If a faster suffix existed, joining it to $P$'s prefix would beat $P$. Replanning with unchanged information therefore cannot improve the arrival time. With a deterministic precomputed fire, "live rerouting" is a display of the plan; it carries new information only when the forecast changes (new detection data, a stochastic fire). The research result worth reporting is the comparison between the snapshot route and the time-dependent route.

### 5.4 When Dijkstra and A* stay correct

| Setting | Dijkstra | A* with heuristic $h$ |
| --- | --- | --- |
| Static non-negative weights | Exact [dijkstra1959note] | Exact if $h$ is consistent, $h(u)\le w_{uv}+h(v)$, which implies admissibility [hart1968formal] |
| Studio hazard weights $w_e=\ell_e(1+\alpha h_e)\ge\ell_e$ | Exact | $h(u)=\lVert x_u-x_{goal}\rVert$ is consistent: $h(u)-h(v)\le\lVert x_u-x_v\rVert=\ell_{uv}\le w_{uv}$ (derivation) |
| Time-dependent, FIFO, earliest arrival | Exact [kaufman1993fastest; dreyfus1969appraisal] | Exact with $h(u)=\lVert x_u-x_{goal}\rVert/v_{max}$ (derivation below) |
| Time-dependent cost not equal to time (dose) | Not exact in general | Not exact in general; use the time-expanded DP (4.3) |
| Non-FIFO without waiting | Not exact [orda1990shortest] | Not exact |

Time-dependent A* (derivation). Every traversal satisfies $\tau_e(t)\ge\ell_e/v_{max}\ge h(u)-h(v)$, so the reduced label $a(u)+h(u)$ never decreases along an edge, and $t\mapsto A_e(t)+h(v)$ remains non-decreasing (FIFO is preserved). Label-setting on reduced labels is therefore exact.

---

## 6. Egress movement (Question 4)

### 6.1 Speeds and the speed-density relation

Hydraulic model (SFPE; flag: constants as reproduced in [gwynne2016hydraulic], building on Nelson and Mowrer [nelson2002emergency] and on Fruin's data [fruin1971pedestrian]):
$$S=k\,(1-a D),\qquad a=0.266\ \text{m}^2\,\text{person}^{-1},\qquad 0.54\le D\le3.8\ \text{persons m}^{-2},$$
with $S\approx0.85k$ below $D=0.54$, $k=1.40$ m s$^{-1}$ for corridors, ramps and doorways, and for stairs $k=1.00,1.08,1.16,1.23$ m s$^{-1}$ for riser/tread of 191/254, 178/279, 165/305 and 165/330 mm (speed along the line of travel). Specific flow is $F_s=SD$ and flow is $F=F_sW_e$ with $W_e$ the effective width (clear width less boundary layers tabulated in the chapter). Maximising $F_s=kD(1-aD)$ gives $D^\ast=1/(2a)\approx1.88$ m$^{-2}$ and $F_{s,max}=k/(4a)\approx0.94\,k$ (derivation): about 1.32 persons s$^{-1}$ m$^{-1}$ for level routes and 0.94 to 1.16 on the four stair geometries. A newer edition of the chapter exists [gwynne2026hydraulic].

Weidmann's relation (flag: as reproduced in [seyfried2005fundamental]; original [weidmann1993transporttechnik]):
$$v(\rho)=v_0\Big[1-\exp\Big(-\gamma\Big(\frac1\rho-\frac1{\rho_{max}}\Big)\Big)\Big],\qquad v_0=1.34\ \text{m s}^{-1},\ \gamma=1.913\ \text{m}^{-2},\ \rho_{max}=5.4\ \text{m}^{-2}.$$
At $\rho=1$ m$^{-2}$ it gives 1.06 m s$^{-1}$ against 1.03 m s$^{-1}$ from the hydraulic model (derivation), a useful cross-check. Seyfried et al. report a linear relation between velocity and inverse density in single-file experiments, and an unexpected agreement between single-file and planar fundamental diagrams in a density range above 1 m$^{-2}$, which suggests lateral interference matters little there [seyfried2005fundamental].

Stairs. Drill data from buildings up to 31 storeys [peacock2012movement] and a stair-specific analysis [peacock2017stairs] give measured speeds that can replace handbook defaults. On Studio's graph, stair edges are 3D segments, so $\tau_e=\ell_e/S_{stair}$ with $\ell_e$ along the pitch line applies directly.

Pre-movement. Pre-movement time is usually the largest and most variable part of RSET; it should be a distribution by occupancy and alarm type [purser2001quantification; bsi2019pd79746; isotr16738].

For a single evacuee (Studio's current case), $D$ is small and $S=0.85k$: 1.19 m s$^{-1}$ on level floors and 0.92 m s$^{-1}$ on a 178/279 mm stair. The metric clock is then $\tau_e=\ell_e/S_e$ in seconds, which removes W3 for the evacuee.

### 6.2 Many evacuees: dynamic network flows

A dynamic network has capacities $u_e$ (persons per time step) and transit times $\tau_e$ on its arcs. The time-expanded network copies each node for each time step, with arcs $(u,t)\to(v,t+\tau_e)$ of capacity $u_e$ and holdover arcs $(v,t)\to(v,t+1)$ for waiting [ford1958dynamic; skutella2009flows].

| Result | Statement (from the abstracts where OpenAlex has one; otherwise as commonly summarised, flag) |
| --- | --- |
| Ford and Fulkerson 1958 [ford1958dynamic] | The maximum flow from one node to another within $T$ periods can be built from one static computation and repeated chain flows; holdovers are not needed. |
| Gale 1959 [gale1959transient] | Earliest-arrival flows (maximal by every deadline at once) exist for one source and one sink. |
| Jarvis and Ratliff 1982 [jarvis1982equivalent] | Several dynamic-flow objectives coincide. |
| Burkard, Dlaska and Klinz 1993 [burkard1993quickest] | Quickest flow: the least time to send a given amount. |
| Hoppe and Tardos 2000 [hoppe2000quickest] | First polynomial algorithm for the quickest transshipment problem (several sources and sinks), with integral flows; the single-sink case is the usual building-evacuation model. |
| Chalmet, Francis and Saunders 1982 [chalmet1982network] | Network models for building evacuation. |
| Lu, George and Shekhar 2005 [lu2005ccrp] | Capacity-constrained routing heuristics for large evacuation networks. |

For Studio the flow model belongs on the room graph, not the lattice: rooms and corridor segments as nodes with storage (floor area times a maximum density), doors and stair flights as arcs with capacity $F_{s,max}W_e\Delta t$ and transit time $\ell/S$. The classical theory assumes transit times that do not depend on the flow; the hydraulic model's speed-density coupling breaks that, so a flow model is an approximation whose capacities and speeds must be chosen for the expected density.

### 6.3 Agent-based tools, for context

Agent-based egress models resolve individual occupants: social-force dynamics [helbing1995social; helbing2000panic], FDS+Evac coupled to the FDS fire solver [korhonen2010fdsevac], buildingEXODUS [owen1996exodus], and commercial tools such as Pathfinder, reviewed with others by Kuligowski and co-authors [kuligowski2010review; kuligowski2016computer]. Continuum crowd models couple a density field to an eikonal potential [hughes2002continuum; treuille2006continuum]. Studio does not need to become one of these. It can use their published speed and flow data, and an agent tool can serve as a reference for verifying a flow or single-evacuee model.

---

## 7. Reinforcement learning (Question 5)

### 7.1 What Ng, Harada and Russell prove

Ng, Harada and Russell (1999) consider an MDP $M=(S,A,T,\gamma,R)$ and a shaped MDP $M'=(S,A,T,\gamma,R+F)$ [ng1999policy]. As we recall the result (flag: the full text was not consulted in this study; confirm the wording):

- Sufficiency. If $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ for some $\Phi:S\to\mathbb{R}$, every optimal policy of $M'$ is optimal in $M$ and vice versa, and $Q^\ast_{M'}(s,a)=Q^\ast_M(s,a)-\Phi(s)$.
- Necessity. If $F$ is not of this form, there exist transition and reward functions for which no optimal policy of $M'$ is optimal in $M$.
- Conditions. $\gamma<1$; or $\gamma=1$ with an absorbing state $s_0$, proper policies, and $\Phi(s_0)=0$.

The secondary record agrees on the main claim: Wiewiora describes the method as one that "guarantees the learner will learn optimal behavior" and proves that potential-based shaping and initialising Q-values with the potential produce the same updates [wiewiora2003potential]; Devlin and Kudenko describe it as "proven to not alter the optimal policy of an agent learning alone" and extend the guarantee to potentials that change over time [devlin2012dynamic]. Grześ analyses the episodic case [grzes2017episodic]. The telescoping identity in W5 shows directly why episodic implementations need $\Phi=0$ at terminal states: without it the shaped return carries $\gamma^T\Phi(s_T)$, which depends on where and when the episode ends.

The theorem is about which policies are optimal. It says nothing about learning speed or about what a finite run of Q-learning returns.

### 7.2 Time-indexed finite-horizon MDPs

With the fire, the problem is a finite-horizon MDP on states $(v,t)$, $t=0,\dots,H$. Its optimal policies are in general non-stationary and are computed exactly by backward induction [puterman1994mdp]:
$$V_H(v)=0,\qquad V_t(v)=\max_{u\in A_v}\big[r(v,u,t)+\gamma\,V_{t+1}(u)\,\mathbb 1\{(u,t+1)\text{ not terminal}\}\big].$$
With deterministic transitions and a known fire this is a longest-path (shortest-cost) computation on the time-expanded acyclic graph, $O(H|E|)$: on the 3,111-node floor with $H=200$, about $8\times10^6$ updates. Q-learning is a sampling approximation of this computation. Its use is justified when the model is unknown or the fire is stochastic; with Studio's precomputed fire, the exact optimum is available and RL should be reported against it.

Potentials on the augmented state, $\Phi(v,t)$, are allowed; a static $\Phi(v)$ is a special case. The horizon must be terminal, $V_H=0$ and $\Phi(\cdot,H)=0$; treating the time limit as part of the state is the correct choice when the state carries $t$ [pardo2018timelimits].

### 7.3 Convergence

Watkins and Dayan prove convergence of tabular Q-learning with probability 1 for finite MDPs, given that every state-action pair is sampled infinitely often and learning rates satisfy $\sum_n\alpha_n=\infty$, $\sum_n\alpha_n^2<\infty$ (flag: conditions as commonly stated) [watkins1992qlearning]; Tsitsiklis and Jaakkola, Jordan and Singh give stochastic-approximation proofs [tsitsiklis1994asynchronous; jaakkola1994convergence]. Studio uses a constant $\alpha=0.5$ and 200 episodes. In a deterministic environment the update target has no noise, and we expect a constant step to suffice given infinite visits, but that is outside the cited conditions and needs its own argument. Either way the guarantee is asymptotic, so finite-run results are empirical.

### 7.4 Keeping the RL claims honest

1. Set $\Phi=0$ on terminal transitions (exit and burn) and make $t=H$ terminal. Only then does "shaping does not change which policy is optimal" follow from [ng1999policy].
2. Compute the exact optimum by backward induction on the time-expanded graph and report the RL result as a gap: return, arrival time, and whether the escape set matches.
3. Report success rates as empirical, with seeds and episode counts; do not call them convergence.
4. Under a metric clock, replace the hop BFS by time-dependent Dijkstra for escape feasibility (section 5.2); the hop BFS stays exact only for the hop model.
5. State the MDP explicitly in any paper: states, actions, horizon, rewards and their units, terminal conditions, discount.

---

## 8. Lean formalisation plan (Question 6)

### 8.1 Toolchain and packages

Existing setup (read, not modified). `C:\Users\lmurugesan\lean\earth3dp` pins `leanprover/lean4:v4.34.1` and Mathlib tag `v4.34.1` (manifest revision `d13f23b`), with `relaxedAutoImplicit = false` and the `weak.linter.mathlibStandardSet` linter. Its manifest shows the packages Mathlib brings in transitively: Plausible, LeanSearchClient, import-graph, ProofWidgets4, Aesop, Qq (quote4), Batteries and Cli. Installed toolchains: v4.32.2, v4.34.1, v4.35.0-rc4, v4.35.0. The README is the default template and carries no project information. `Earth3dp/Dimensions.lean` and the skill template `PaperTemplate.lean` both use a decidable `Dim` structure with integer exponents, closed by `decide`, and earth3dp checks axioms with `#print axioms`.

State on 2026-10-27 (GitHub API): the latest stable Lean 4 release [moura2021lean4] is v4.35.0 (2026-10-12), Mathlib has a matching `v4.35.0` tag, and v4.36.0-rc1 is a pre-release.

Recommendation:

- New project outside OneDrive, for example `C:\Users\lmurugesan\lean\fireegress`, to keep `.lake` (several GB) out of sync and away from long-path problems.
- `lean-toolchain`: `leanprover/lean4:v4.35.0`. `lakefile.toml`: copy earth3dp's `[leanOptions]` and require Mathlib at `rev = "v4.35.0"`. If disk or bandwidth matters more than currency, `v4.34.1` reuses the Mathlib cache earth3dp already downloaded.
- Lean projects pin one Mathlib revision; adding a second library (for example PhysLib) requires that it builds against the same revision.

```toml
name = "fireegress"
version = "0.1.0"
defaultTargets = ["FireEgress"]

[leanOptions]
pp.unicode.fun = true
relaxedAutoImplicit = false
weak.linter.mathlibStandardSet = true
maxSynthPendingDepth = 3

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4"
rev = "v4.35.0"

[[lean_lib]]
name = "FireEgress"
```

| Package | Needed? | Reason |
| --- | --- | --- |
| Mathlib [mathlib2020; mathlib4repo] | Yes | `SimpleGraph` with `lapMatrix`, `Matrix.rowStochastic`, Gershgorin (`eigenvalue_mem_ball`), the spectral theorem for Hermitian matrices, `Finset` sums and telescoping, `Real.rpow`, metric spaces, `ContractingWith`. All names were checked in the local Mathlib checkout. |
| Aesop [limperg2023aesop] | Comes with Mathlib | Proof search for the many small `Finset`, order and graph side goals. |
| Batteries, Qq, ProofWidgets4 | Come with Mathlib | Infrastructure; no direct use planned. |
| Plausible [plausiblerepo] | Comes with Mathlib; use it | Property-based testing of a conjecture before proving it, for example to find the degree-1 counterexample automatically. |
| LeanSearchClient [leansearchclientrepo] | Comes with Mathlib; optional | `#leansearch` natural-language lemma search during development; needs network access. |
| import-graph | Comes with Mathlib | Dependency inspection; not needed. |
| PhysLib [physlibrepo; toobysmith2025heplean] | Optional, not as a dependency at first | Its `Physlib/Units/` modules define dimensions with rational exponents through a reducible wrapper (`Dimension.Exponent`), because rational operations are irreducible in Lean and `decide` does not see through them. Studio's equations need exponents such as 1/3, 2/5, 5/3, 5/6 and 3.4. Copying the idea is lighter than depending on the library. |
| CSLib | Not needed | Its file list (checked) has no shortest-path or Dijkstra development. |
| Measure theory | Only for the continuous FED | `intervalIntegral.integral_mono_on` for monotonicity of the dose integral; the discrete sum needs only `Finset`. |

Dimensions with fractional exponents. The template's `Dim` uses `Int` exponents and `decide`. Two options keep `decide` working: store each exponent multiplied by 30 (the least common multiple of the denominators 2, 3, 5 and 6 that occur), or adopt a reducible rational wrapper modelled on PhysLib's. The first is a ten-line change to the template. Units-of-measure typing has a long precedent in programming languages [kennedy1997units].

### 8.2 Candidate theorems

Difficulty is for someone fluent in Mathlib, without human-in-the-loop latency; for this project's pace, multiply by 1.5 to 2. "Falsifies" marks a theorem whose statement shows the current code is wrong.

| ID | Statement in plain mathematics | Difficulty | Mathlib pieces |
| --- | --- | --- | --- |
| L1 | If $\Delta t\le C_i/(G_i+H_i)$ for all $i$, the explicit-Euler matrix $M=I-\Delta tC^{-1}(L_g+H)$ is entrywise non-negative with row sums at most 1. | Hours | `Matrix`, `Finset.sum`, `SimpleGraph.lapMatrix`, `SimpleGraph.lapMatrix_mulVec_apply` |
| L2 | Discrete maximum principle: for $M$ as in L1, $\min(0,\min\theta^0)\le(M^n\theta^0)_i\le\max(0,\max\theta^0)$ for all $n$; with $H=0$ the same with $M\in$ `rowStochastic`. | 1 to 2 days | `Matrix.rowStochastic`, `Matrix.nonneg_of_mem_rowStochastic`, `Matrix.sum_row_of_mem_rowStochastic`, `Finset.sup'`/`inf'`, induction on `n` |
| L3 | Classic overshoot (falsifies): $d_i=1$, $\kappa=6/5$, $T_j^n>T_i^n$ imply $T_i^{n+1}>\max(T_i^n,T_j^n)$, the previous neighbourhood maximum. | Hours | `linarith`, `norm_num` |
| L4 | Corrected step: with $\beta_i=\min(1,\kappa/d_i)$, $T_i\le T_i^{+}\le\max_{j\in\{i\}\cup A_i}T_j$; hence the clip is redundant. | Hours to 1 day | `min_le_left`, `Finset.sum`, `Finset.le_sup'` |
| L5 | $\ell^2$ stability: for symmetric positive semidefinite $L$ with eigenvalues in $[0,\lambda_{max}]$ and $\Delta t\le2/\lambda_{max}$, $\lVert(I-\Delta tL)x\rVert_2\le\lVert x\rVert_2$; Gershgorin gives $\lambda_{max}\le\max_i 2G_i$. | 3 to 7 days | `Matrix.IsHermitian.spectral_theorem`, `Matrix.IsHermitian.eigenvalues`, `eigenvalue_mem_ball`, `SimpleGraph.posSemidef_lapMatrix`, `EuclideanSpace` |
| L6 | Classic's one-sided update has $T\equiv T_f$ as the only steady state on the source's connected component. | 2 to 5 days | `Monotone`, `tendsto_atTop_ciSup`, `SimpleGraph.Connected`, `SimpleGraph.Reachable` |
| L7 | Monotone fire implies the time-layered BFS gives exact earliest arrival (spec level: the layer of first appearance equals the minimum over safe walks; pruning visited nodes is sound). | 3 to 7 days; refinement to the Python loop structure 1 to 2 weeks more | `SimpleGraph.Walk`, `SimpleGraph.Walk.length`, `SimpleGraph.Walk.support`, `Finset`, `Nat.find`, `ℕ∞` |
| L8 | Euclidean heuristic: if $w_{uv}\ge\lVert x_u-x_v\rVert$ then $h(u)\le w_{uv}+h(v)$ (consistency) and $h(u)\le d^\ast(u,goal)$ (admissibility, by induction on walks). | Hours (consistency); 1 day (admissibility) | `dist_triangle`, `EuclideanSpace ℝ (Fin 3)`, `SimpleGraph.Walk` induction, `List.sum` |
| L9 | A* with a closed set returns an optimal path under a consistent heuristic. | 2 to 4 weeks (no shortest-path library to build on) | `Finset`, well-founded recursion, the L8 lemmas |
| L10a | FIFO composition: if every $A_e$ is monotone, the arrival function of any path is monotone, and subpaths of earliest-arrival paths are earliest-arrival from their start times. | 1 to 2 days | `Monotone.comp`, `List.foldl`, `WithTop ℝ` or `EReal` |
| L10b | Under FIFO the earliest-arrival times are the least fixed point of the time-dependent Bellman equation; waiting never helps. | Days | `OrderHom`, `Finset.min'`, `WithTop ℝ` |
| L10c | Time-dependent Dijkstra with FIFO arrival functions is correct (loop invariant). | 2 to 4 weeks | As L10a, L10b, plus the algorithm |
| L11 | Replanning consistency: under an unchanged FIFO forecast, the suffix of an earliest-arrival path is earliest-arrival from its intermediate node and time. | Hours to 1 day | L10a |
| L12 | Shaping telescopes: $\sum_{t<T}\gamma^t(\gamma\Phi(s_{t+1})-\Phi(s_t))=\gamma^T\Phi(s_T)-\Phi(s_0)$. | Hours | `Finset.sum_range_sub` (the additive form of `Finset.prod_range_div`), `pow_succ` |
| L13 | Deterministic finite horizon: with $\Phi=0$ at terminal states and at $t=H$, every policy's shaped return equals its return minus $\Phi(s_0)$, so the sets of optimal policies coincide. | 1 to 2 days | L12, `Finset.exists_max_image`, `IsMaxOn` |
| L14 | Counterexample (falsifies the current claim): a three-state deterministic MDP with $\Phi(\text{terminal})\ne0$ whose shaped optimum differs from the unshaped optimum. | Hours | `norm_num`, `decide` |
| L15 | Ng et al. sufficiency for a finite stochastic MDP with $\gamma<1$: $Q^\ast_{M'}=Q^\ast_M-\Phi$, so the greedy policies coincide. | 1 to 3 weeks | `ContractingWith`, `ContractingWith.efixedPoint`, `Finset.sum` over `Fintype S`, sup norm on `S × A → ℝ` |
| L16 | Dimensional homogeneity: zone-model ODEs (pressure, layer volume, layer temperature), FED increments (dimensionless), MQH (units of 6.85), Alpert, hydraulic model, Weidmann exponent, t-squared ($\alpha$ in W s$^{-2}$). | Hours each, 1 to 2 days for the set | `Dim` with exponents scaled by 30, `decide` |
| L17 | Zone-model derivation: the three equations of section 3.4 follow from the ideal-gas law and energy conservation. | 1 to 3 days with derivatives as real variables and the chain rule as hypotheses; about 1 week with `HasDerivAt` | `field_simp`, `ring`, `HasDerivAt.mul`, `HasDerivAt.div` |
| L18 | Hop metric of $S_{12}$ on $\mathbb Z^2$ equals $\lceil(\lvert a\rvert+\lvert b\rvert)/2\rceil$ (falsifies isotropy of the flood model). | 2 to 4 days | `Int`, `Nat.ceil`, `SimpleGraph.dist` on an explicit graph |
| L19 | Metrication bound $1/\cos(\varphi/2)$ for two-direction lattice paths. | 2 to 4 days | `Real.cos`, `Real.sin`, trigonometric identities |
| L20 | Numerics: $\alpha$ values from $t_g$; $\dot Q_{fo}$ and time to flashover in the worked example; FED tolerance times. | Hours for the rational ones; days for `Real.rpow` with non-integer exponents (needs log/exp bounds) | `norm_num`, `Real.sqrt`, `Real.rpow`, `Real.log` bounds |

Suggested core set (about 4 to 8 weeks at this project's pace): L1 to L4, L7, L8, L10a, L10b, L11, L12 to L14, L16. Stretch set (another 6 to 12 weeks): L5, L6, L9, L10c, L15, L17 with calculus, L18, L19.

### 8.3 Statement sketches

These are signatures to fix the shape of the development; they have not been compiled.

```lean
-- L3: Classic's step overshoots at a degree-1 node.
theorem classic_overshoot (Ti Tj : ℝ) (h : Ti < Tj) :
    Tj < Ti + (6/5 : ℝ) / 1 * max (Tj - Ti) 0 := by
  sorry

-- L12: potential-based shaping telescopes along a trajectory.
theorem shaping_telescopes (γ : ℝ) (Φ : ℕ → ℝ) (T : ℕ) :
    ∑ t ∈ Finset.range T, γ ^ t * (γ * Φ (t + 1) - Φ t) = γ ^ T * Φ T - Φ 0 := by
  sorry

-- L16: the pressure equation of the zone model is homogeneous (W / m³ = Pa / s).
theorem zone_pressure_dims : W / m3 = Pa / s := by decide
```

### 8.4 Dimensional checks worked by hand (derivations)

| Equation | Check |
| --- | --- |
| $\dot p=(\gamma-1)(\dot h_U+\dot h_L)/V$ | W m$^{-3}$ = J s$^{-1}$ m$^{-3}$ = Pa s$^{-1}$. |
| $\dot V_U=[(\gamma-1)\dot h_U-V_U\dot p]/(\gamma p)$ | W Pa$^{-1}$ = m$^3$ s$^{-1}$; $V_U\dot p$ is also W. |
| $\dot T_U=(\dot h_U-c_p\dot m_UT_U+V_U\dot p)/(c_p\rho_UV_U)$ | Numerator W; denominator J K$^{-1}$; result K s$^{-1}$. |
| $X_{FED}=\sum\Delta t/t_I$ | s/s, dimensionless. |
| $t_{I,conv}=5\times10^7T^{-3.4}$ | Homogeneous only with $T/(1\,°\text{C})$; constant in minutes. |
| MQH, $\Delta T=6.85(\dot Q^2/(A_o\sqrt{H_o}h_kA_T))^{1/3}$ | Bracket: kW K m$^{-5/2}$; cube root: kW$^{1/3}$K$^{1/3}$m$^{-5/6}$; so 6.85 carries K$^{2/3}$ kW$^{-1/3}$ m$^{5/6}$. |
| Hydraulic, $S=k(1-aD)$ | $aD$: m$^2$ person$^{-1}$ times person m$^{-2}$, dimensionless. |
| Weidmann exponent $\gamma(1/\rho-1/\rho_{max})$ | m$^{-2}$ times m$^2$, dimensionless. |
| t-squared, $\dot Q=\alpha t^2$ | $\alpha$ in kW s$^{-2}$. |

---

## 9. Model map

| Studio model | Gap | Proposed model | Parameters and sources | Lean-provable claims | Implementation effort (single developer, with review) |
| --- | --- | --- | --- | --- | --- |
| Radial fire | Ignores walls; speed is "median edge per tick"; no time unit (C4, W3). | Keep only as a free-space reference; use Dijkstra on $\ell_e/F$ over unblocked edges. | $F$ in m s$^{-1}$, chosen per scenario; [sethian1996fast; finney2002fire; tsitsiklis1995efficient] | L19 | 1 to 2 days |
| Flood fire | Hop count: $\sqrt2$ anisotropy, mesh-dependent (W3). | Dijkstra on $\ell_e/F$ (lattice) or room-graph spread with $t^{fo}+t^{bar}$. | $F$; $t^{fo}$ from t-squared growth and $\dot Q_{fo}$; barrier times [babrauskas1980flashover; heskestad1979initial; platt1994probabilistic; cheng2011dynamic] | L18, L19 | 2 to 4 days (lattice); 2 to 3 weeks (room graph, after the room graph exists) |
| Temperature fire (Classic) | Degree-1 overshoot (W1); not a temperature (W2). | Immediately: $\beta_i=\min(1,\kappa/d_i)$. Research model: two-zone model on the room graph with plume, vents, t-squared fire; Alpert near field; layer state mapped to nodes. | $\alpha$, $\dot Q_{fo}$, wall $k,\rho,c,\delta$, $C_d$, $z_{head}$ [peacock2015cfast; forney1994zone; heskestad2016plumes; alpert2016ceiling; tanaka2016vent; mccaffrey1981estimating] | L1 to L6, L16, L17 | Fix: 0.5 to 1 day. Zone model with CFAST comparison: 6 to 12 weeks |
| Hazard routing cost | Snapshot hazard at the current step; unitless 0..1 hazard (W4). | Time-dependent: hard tenability windows with FIFO Dijkstra; dose-minimising or dose-constrained routes by time-expanded DP. | FED model and thresholds [iso13571; purser2016hazards]; walking speeds (section 6) | L10a, L10b, L11, L20 | 2 to 3 weeks |
| A* engine | Correct; justification should be consistency, not admissibility; float32 tolerance. | Keep; add time-dependent A* with $h=\lVert\cdot\rVert/v_{max}$. | $v_{max}$ from [gwynne2016hydraulic] | L8, L9 | 1 to 3 days |
| Lethality threshold | °C threshold on a non-temperature (W2). | ASET per node from FED or layer criteria; block edges after $b_v$. | [purser2016hazards; babrauskas2010rset] | L16 (FED dimensionless) | 1 week, after the zone model |
| Live rerouting | Solves from the original start, at the wrong time (W4); returns routes through fire as found. | Receding horizon from the current position and time; report "no tenable route" explicitly. | none new | L11 | 1 to 2 weeks |
| Time model (hop ticks) | No unit; orientation- and tie-dependent speeds (W3). | Seconds throughout: $\tau_e=\ell_e/S_e$ for the evacuee, fire in seconds from its model; pre-movement as a distribution. | $k$, $a$, stair $k$ [gwynne2016hydraulic]; [weidmann1993transporttechnik; peacock2012movement; purser2001quantification] | L16 | 2 to 4 weeks (touches fire, routing, RL and UI) |
| RL (Q-learning) | Non-zero terminal potential; horizon not terminal (W5); no exact baseline. | $\Phi=0$ at terminals and $t=H$; exact backward-induction baseline; report gap. | $\gamma$, $H$ [ng1999policy; grzes2017episodic; puterman1994mdp; pardo2018timelimits] | L12 to L15 | Fixes: 1 to 2 days. Exact baseline: 3 to 5 days |
| Escape check (BFS) | Correct for hop time (C1); not for metric time. | Time-dependent Dijkstra under FIFO once the clock is metric. | as above [kaufman1993fastest; dreyfus1969appraisal] | L7, L10a to L10c | 2 to 3 days, with the metric clock |
| Many evacuees (absent) | Single evacuee only. | Quickest transshipment or time-expanded max flow on the room graph. | $F_{s,max}$, $W_e$, storage [gwynne2016hydraulic; hoppe2000quickest; ford1958dynamic; chalmet1982network] | Not recommended at first (flows over time are a large formalisation) | 6 to 10 weeks |

---

## 10. Recommended order of work

1. Correctness fixes with small proofs (weeks 1 to 2). Clamp $\beta_i$ (W1); set $\Phi=0$ on terminal transitions and make $t=H$ terminal (W5); stop returning routes through fire as `found` (W4). Prove L3, L4, L12, L13 and L14 alongside, so each fix carries a checked statement.
2. Exact baselines (weeks 2 to 4). Backward induction on the time-expanded graph for the RL problem; report RL's gap to it. This gives every later change a ground truth.
3. Metric clock (weeks 3 to 7). Walking times from the hydraulic model; fire in seconds (Dijkstra on $\ell_e/F$ as the interim fire); time-dependent Dijkstra for escape and rerouting from the current position. Prove L7, L8, L10a, L10b, L11. This settles review decisions 1 to 3.
4. Tenability (weeks 6 to 10). FED heat along routes, ASET maps, dose-aware routing by time-expanded DP. Prove L16 for FED and the hydraulic model.
5. Room graph and two-zone model (months 3 to 5). Build the room graph (review section 6, item 4), implement the zone ODEs with an implicit integrator, compare against CFAST on the same rooms [peacock2015cfast]. Prove L16 and L17 for the zone equations. This settles review decision 4 and is the step that brings the model to the validation standard of section 1.2.
6. Many evacuees (optional, months 5 to 7). Dynamic network flows on the room graph; compare with an agent tool on one building.

The Lean core set runs in parallel with steps 1 to 4 (about 4 to 8 weeks); the stretch set is worth doing only if a paper will claim algorithm correctness formally.

---

## 11. Reference verification

Procedure. Each DOI was resolved through the Crossref REST API and through OpenAlex, and the title, year and first author were compared. Bibliographic fields came from Crossref's BibTeX transform and were cleaned by hand. Abstracts were read from OpenAlex where available; no full texts were downloaded.

Counts for the 118 entries in [fire-egress-methods.bib](fire-egress-methods.bib):

| Status | Count | Entries |
| --- | --- | --- |
| DOI verified in Crossref and OpenAlex | 106 | All entries with a DOI except the two below |
| DOI verified in DataCite and OpenAlex (not a Crossref DOI) | 2 | `weidmann1993transporttechnik`, `pardo2018timelimits` |
| No DOI exists; verified in two indexes | 1 | `ng1999policy` (OpenAlex W1777239053, Semantic Scholar CorpusId 5730166) |
| Software; existence checked on GitHub | 4 | `mathlib4repo`, `plausiblerepo`, `leansearchclientrepo`, `physlibrepo` |
| Not verified | 5 | `iso13571`, `isotr16738` (ISO catalogue returned HTTP 403), `nfpa72` (edition not fixed), `fruin1971pedestrian`, `nelson2002emergency` (not found in OpenAlex or Crossref) |

Corrections found while verifying:

- Crossref returns a wrong booktitle (AAMAS 2004) for both AAMAS papers [grzes2017episodic; devlin2012dynamic]; corrected from the IFAAMAS listings, which also confirm the page numbers.
- Wu et al. 2019 is not a Wonka paper [wu2019interior].
- Michels, Luan and Tokman 2017 is the correct author list [michels2017stiffly].
- The CFAST technical reference's Crossref record lists Peacock, Reneke and Forney [peacock2015cfast].
- Weidmann 1993 has two DataCite records (1992 and 1993); the 1993 one is cited.
- The Classic eCAADe paper has two DOIs (Blucher and eCAADe); the Blucher one is cited [jabi2019synergy].

Constants flagged in the text (plume, ceiling jet, MQH, Babrauskas, FED heat and CO, hydraulic model, Weidmann) are reproduced from secondary sources or recall and should be checked against the primary text before implementation. Their structure and units were checked here; their numerical values were not checked against the originals.

Other sources consulted for context and not otherwise cited: the FDS technical reference [mcgrattan2013fds] for what resolved CFD provides, and the SFPE Handbook evacuation and hazard chapters of the sixth edition [purser2026hazards; gwynne2026hydraulic].
