# Flamescapes: Simulating the Wildlandfire Urban Interface, Supplementary Material (transcription)

Wrede, H., Wagner, A. R., Pałubicki, W., Michels, D. L., Pirk, S. (2026). Flamescapes: Simulating the Wildlandfire Urban Interface, Supplementary Material. ACM Transactions on Graphics 45(6), Article 246. https://doi.org/10.1145/3842518

The DOI is as printed on the paper; Crossref did not resolve it on 2026-11-05 (the project page says "DOI (coming soon)").

Licence: Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/), as stated on the paper's first page. © 2026 the authors.
Changes: transcribed from the authors' PDF (project page https://helgewrede.github.io/flamescapes/, retrieved 2026-11-05) to Markdown on 2026-11-05; equations retyped in LaTeX; figures replaced by their captions; layout changed. Transcription errors are ours; cite the original.

Transcription conventions: page breaks of the original are marked with HTML comments such as `<!-- page 246:2 -->`, placed at the paragraph boundary nearest to the break; figure captions are placed at a paragraph boundary close to where the figure sits in the original. Small-capital state names of the original (e.g. DOOR(CENTER)) are written in ordinary capitalization (Door(Center)). The algorithms are reproduced as numbered code blocks with their mathematical content repeated in LaTeX below them. Unreadable glyphs would be marked `[?]` (none were needed). Possible typos or inconsistencies are listed, unchanged in the text, under "Transcriber's notes" at the end.

The main paper is transcribed in `flamescapes-2026.md`; the equation catalogue is `flamescapes-2026-equations.json` (supplement entries carry ids starting with `S`).

---

<!-- page 246:1 -->

# Flamescapes: Simulating the Wildlandfire Urban Interface Supplementary Material

HELGE WREDE\*, Kiel University, Germany
ANTON R. WAGNER\*, Kiel University, Germany
WOJTEK PAŁUBICKI, Adam Mickiewicz University, Poland
DOMINIK L. MICHELS, KAUST, KSA
SÖREN PIRK, Kiel University, Germany

\*Both authors contributed equally to this research.

Authors' Contact Information: Helge Wrede, helge.wrede@email.uni-kiel.de, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany; Anton R. Wagner, awa@informatik.uni-kiel.de, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany; Wojtek Pałubicki, wojciech.palubicki@amu.edu.pl, Adam Mickiewicz University, 61-614, Poznań, Poland; Dominik L. Michels, dominik.michels@kaust.edu.sa, KAUST, Thuwal 23955, KSA; Sören Pirk, soeren.pirk@gmail.com, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany.

This work is licensed under a Creative Commons Attribution 4.0 International License.
© 2026 Copyright held by the owner/author(s).
ACM 1557-7368/2026/12-ART246
https://doi.org/10.1145/3842518

## 1 Modeling Urban Scenes

Here we provide a description of the urban scene modeling components of our WUI framework.

### 1.1 Procedural Modeling of Combustion-Ready Buildings

To prepare a scene for simulation with our WUI framework we aim to represent buildings with *atomic modules*. To procedurally generate combustion-ready buildings we use WFC and SSA to model 3D buildings from OSM polygons.

*WFC and Semantic State Space.* To model buildings we define the vertical wall parts of the houses. Specifically, for each line segment of the footprint polygon, we define a two-dimensional uniform grid where the bottom of the grid is spanning the extent of the polygon line. Each grid cell is initialized with a set of possible semantic states representing architectural and structural elements, we call *atomic modules* (Fig. 4, a, b, main text). These include structural components such as beams (rafters, joists, studs, etc.) with different positional (left, bottom, right, top, center) and orientational (horizontal, vertical) roles as well as facade elements such as windows and doors to encode their relative position within an opening (left, right, bottom, or top), see Tab. 1 and 2 for a full list of states. Depending on the wall length and height, grids have approx. 150–250 cells per wall. We solve the semantic assignment on each wall grid using WFC [Alaka and Bidarra 2023; Karth and Smith 2017] over a 4-neighborhood (left, right, up, down). Following construction, materials are assigned according to Tab. 3.

### 1.2 WFC Rule Formalism

Each semantic state $t$ exposes a discrete *interface label* on every side, $\sigma(t, d)$ for $d \in \{left, right, up, down\}$. Two adjacent cells with states $(t_i, t_j)$ are compatible in direction $d$ iff $\sigma(t_i, d) = \sigma(t_j, \bar{d})$, where $\bar{d}$ is the opposite direction. We construct $\sigma$ deterministically from the state parameters (class, orientation, and relative position labels), so that adjacency encodes architectural principles such as: (1) continuity of structural members across cells, (2) consistent tiling of openings (windows, fireplaces) via start, center, end boundary labels, (3) framing of openings by structural states, and (4) support and anchoring constraints (e.g., door bottoms are restricted to ground level).

> Figure 1: To model buildings from OSM polygons we use WFC. For each edge we define a 2D grid with assignments of atomic modules for each grid cell. After resolving WFC we can obtain different plausible configurations of architectural elements for each polygon (a-d).

Backtracking is used to ensure the constraints are satisfied and that they produce a globally consistent assignment of semantic states across the wall grid. The state space is discrete and low-dimensional to keep WFC tractable while still capturing meaningful architectural variation. Moreover, we define multiple sets of semantic states and assign them randomly to each building polygon to create buildings with different architectural properties. A few examples of architectural variation of atomic modules are shown in Fig. 1.

### 1.3 Architectural constraints

The architectural constraints are represented as rules

$$
r = (S, T, E),
\tag{1}
$$

where $S$ is the rule scope, $T$ is the target condition, and $E$ is the rule effect. Effects either keep compatible states, remove incompatible states, or modify state weights. When a cell is collapsed to a concrete state $s$, the solver generates both grid-local rules, which constrain neighboring cells within the same wall grid, and building-global rules, which propagate constraints across larger building regions, such as suppressing repeated door placement. To construct our buildings we used the following constraints:

<!-- page 246:2 -->

> Figure 2: We use the OSM polygons to generate roofs. A polygon (a) extended before we compute roof compartments with the SSA algorithm (b). We then elevate the compartments to obtain 3D roofs.

**Table 1.** Atomic module classes and associated semantic state counts used for wall discretization.

| Class | Parameters | Allowed Values | # States |
|---|---|---|--:|
| Studs | $(n, o, p)$ | $n \in \{1, 2, 3\}$, $o \in \{\mathrm{H}, \mathrm{V}\}$, $p \in \{\text{start, center, end}\}$ | 156 |
| Windows | $(p_x, p_y)$ | $p_x, p_y \in \{\text{start, center, end}\}$ | 9 |
| Doors | $p_y$ | $\{\text{bottom, center, top}\}$ | 3 |
| Fireplace bases | $(p_x, p_y)$ | $p_x, p_y \in \{\text{start, center, end}\}$ | 9 |
| Chimneys | $p$ | $\{\text{start, end}\}$ | 2 |
| **Total** | | | **179** |

*Door constraints.* Doors are restricted to the ground-level portion of a wall. If a door is placed at row $j \le 4$, all cells in the corresponding column are forced into a coherent vertical door stack: rows 1–3 become Door(Center), row 4 becomes Door(Top), and row 0 becomes Door(Bottom). If a cell in the same ground-level band is resolved as a non-door state, all door states are removed from rows 0–4 of that column. A building-global suppression rule additionally reduces repeated door placement on the same wall or nearby building regions.

*Window constraints.* Window states encode their position within larger openings. Horizontal and vertical boundary states remove incompatible window states in adjacent cells and keep only states that can continue the opening. A Window(Center, Center) state forces a local $3 \times 3$ window neighborhood, producing coherent window blocks. Additional rules enforce horizontal and vertical line continuity, suppress conflicting window bands, and remove fireplaces, doors, and chimneys from immediate horizontal neighbors.

*Stud continuity constraints.* Structural states enforce continuity of framing members. A vertical stud at the right, left, or center position forces compatible vertical studs or supporting horizontal members in the cells above and below. Center studs also remove window states from neighboring cells when the corresponding opening would intersect the structural member. These rules maintain connected load-bearing elements while still allowing facade variation.

*Double-wall constraints.* Double states encode common structural combinations such as horizontal members with vertical supports. Bottom horizontal double states promote compatible top-edge windows, doors, or fireplace states in the cell below.

**Table 2.** Implementation-level WFC wall cell states $w \in W$.

| State | Description | Parameters |
|---|---|---|
| Straight$(o)$ | Single wall segment or stud | $o \in O$ |
| Double$(o_1, o_2)$ | Two independent wall segments | $o_1, o_2 \in O$ |
| Triple$(o_1, o_2, o_3)$ | Three wall segments; used for corners | $o_1, o_2, o_3 \in O$ |
| Junction$(o_1, o_2)$ | Two segments, one of which is half-length | $o_1, o_2 \in O$ |
| Door$(v)$ | Door opening at vertical location $v$ | $v \in V$ |
| Window$(h, v)$ | Window opening at horizontal and vertical location | $h \in H,\ v \in V$ |
| FirePlace$(h, v)$ | Fireplace opening | $h \in H,\ v \in V$ |
| Chimney$(h)$ | Chimney column | $h \in H$ |

**Table 3.** Role-based material class assignment at geometry spawn.

| Class | Assigned geometry |
|---|---|
| Dense biomass | framing, siding, trim studs, plates, doors, tree trunks and branches |
| Sparse biomass | tree canopy |
| Metal | gutter segments, spouts, fixtures |
| Stone | curbs, floor slabs, fireplace base and chimney, fireplace / chimney, ground plane, roads |
| Glass | window panes |

Scene-config meshes: class authored per entity in the scene file.

*Fireplace constraints.* If a wall is initialized with a fireplace feature $F = \mathrm{True}$, additional constraints are applied. Chimney columns are forced across all rows above the cell. For fireplaces, the lower rows of the selected column span are forced to fireplace states with appropriate horizontal and vertical position labels. Corner cells are assigned compatible triple states, left and right edge cells receive boundary vertical members, and top or bottom cells receive horizontal boundary members.

### 1.4 Roof Geometry

The roof, in contrast to the rest of the house, uses the SSA [Aichholzer and Aurenhammer 1996; Cheng and Vigneron 2007] instead of WFC. The original building footprint polygon, is first extended outward to create roof overhangs. We then compute the skeleton of the polygon which defines the roof geometry. Given the separate roof polygons we generate different roof elements such as rafters and shingles that are then assigned different material properties relevant for the combustion simulation. Together, these steps produce complex 3D building models that plausibly describe architectural components of real houses. Fig. 2 shows an extruded input polygon (a), the extended roof polygon computed with the SSA (b) and the final 3D roof geometry (c).

### 1.5 Modeling of Urban Scenes with Vegetation

We use the OSM building footprints and street center lines to compute a 2D SDF of the urban scene. The SDF is used for generating ground geometry and to assign areas with distinct materials, such as asphalt for roads, concrete for curbs and sidewalks. Additionally, we compute the shortest distances between building footprints and street centerlines to model driveways. Areas not occupied by roads or buildings are classified as vegetated ground, such as lawns or vegetated ground. This SDF-based formulation enables smooth

<!-- page 246:3 -->

transitions between surface types and robust handling of complex urban layouts, including irregular parcel shapes and intersections.

Vegetation density is controlled by adjusting both the minimum plant distance and the padding thresholds of the underlying distance field. For the placement of plant models, we employ Poisson disk-based sampling to distribute plant models within vegetated ground regions. Poisson disk sampling enforces the minimum distance between samples, producing a spatially uniform distribution that avoids overlap. By adjusting the padding thresholds and the minimum plant distance, we can generate vegetation layouts ranging from sparse plantings to high-density fuel distributions. To reduce the geometric complexity of plant models during rendering, we employ stochastic pruning [Cook et al. 2007] – a randomly selected subset of leaves is removed for each tree, while the remaining leaves are scaled to preserve overall projected coverage and silhouette.

## 2 Algorithms

The following algorithms show the procedures used for the ray-based thermal radiation as well as the ember and flame front particle to grid sampling. The thermal radiation step is shown in the main text in Algorithm 1 in line 15. The ember and flame front sampling are referenced in lines 16-18.

**ALGORITHM 1: Thermal Radiation.**

**Input:** Eulerian grid with gas and solid properties.
**Output:** Updated temperatures and incoming thermal radiation.

```
1  forall Grid Cells do
2  |   Determine available radiation directions;
3  |   Calculate outgoing thermal radiation;
4  |   Update temperatures T_s, T_g;
5  |   Cast rays in all available directions with a fixed energy budget;
6  |   forall Rays do
7  |   |   Deposit energy in solid surface cells and volumes on hit;
8  |   Update temperatures T_s, T_g;
```

Mathematical content in LaTeX: lines 4 and 8 update $T_s, T_g$.

**ALGORITHM 2: Particle to grid.**

**Input:** Eulerian grid and Lagrangian particles.
**Output:** Updated grid.

```
 1  Sort particles into grid cells;
 2  forall Grid Cells do
    |   // - Embers -
 3  |   Initialize thermal energy from embers: q̇ ← 0;
 4  |   forall Ember Particles in this Cell do
 5  |   |   Determine excess thermal energy: q̇_e ← m_e · (T_e − T_s) · C_e;
 6  |   |   Deposit excess thermal energy: q̇ ← q̇ + q̇_e;
 7  |   Update solid temperature: T_s ← T_s + q̇ · ρ_s C_{p,s};
    |   // - Flame Front -
 8  |   Initialize maximum flame front temperature: T_f ← 0;
 9  |   forall Flame Front Particles in this Cell do
10  |   |   Store maximum temperature: T_f ← max(T_f, T_p);
11  |   During gas combustion, use: max(T_f, T_s);
```

Mathematical content in LaTeX:

- line 3: $\dot{q} \leftarrow 0$
- line 5: $\dot{q}_e \leftarrow m_e \cdot (T_e - T_s) \cdot C_e$
- line 6: $\dot{q} \leftarrow \dot{q} + \dot{q}_e$
- line 7: $T_s \leftarrow T_s + \dot{q} \cdot \rho_s C_{p,s}$
- line 8: $T_f \leftarrow 0$
- line 10: $T_f \leftarrow max(T_f, T_p)$
- line 11: $max(T_f, T_s)$

### 2.1 Parameter Values for Real Experiment Comparisons

The simulation parameters used for the validation experiments (Sec. 6.2 in the main text) are summarized in the tables Tab. 4, 5, and 6.

**Table 4.** Simulation parameters for the radiative heat flux validation.

| Parameter | Range / Value | Units |
|---|--:|:-:|
| ***Combustion & Pyrolysis*** | | |
| Gas fuel type | IsoOctane | — |
| Combustion CO₂ fraction | 0.850 | — |
| Combustion residual fraction | 0.150 | — |
| Pyrolysis fuel fraction | 0.900 | — |
| Pyrolysis residual fraction | 0.100 | — |
| Gas residual reaction fraction | 0.600 | — |
| ***Heat Transfer*** | | |
| Convection coefficient | 3.0 | — |
| Radiation ray energy | 0.100 | J |
| Residual absorption coefficient | 500 | — |
| ***Fluid Dynamics*** | | |
| Vorticity confinement strength | 20.0 | — |
| Wind velocity $(x, y, z)$ | (0.0, 0.0, 0.0) | m s⁻¹ |
| Porous solid velocity multiplier | 0.100 | — |
| Ambient temperature | 300 | K |
| Ambient gas density | 1.1726 | kg m⁻³ |
| Gravitational acceleration | −9.81 | m s⁻² |
| Porous solid density threshold | 10.0 | kg m⁻³ |
| Solid density threshold | 100.0 | kg m⁻³ |
| ***Particle System*** | | |
| Particle capacity | 262,144 | — |
| Ember spawn rate | 1.0 | — |
| Ember lifetime | [1.0, 10.0] | s |
| Ember mass | [10⁻⁶, 10⁻⁴] | kg |
| Ember emissivity | 0.95 | — |
| Ember mass loss rate | — | kg s⁻¹ |
| Ember grid velocity factor | 0.500 | — |
| Flame-front lifetime | 0.5 | s |
| Flame-front speed | 8.0 | m s⁻¹ |
| ***Procedural Scene Generation*** | | |
| OSM request size | – | m |
| Vegetation density | – | — |
| Leaf thickness | – | m |
| Minimum house distance | – | m |
| Minimum road distance | – | m |
| ***Computational Domain*** | | |
| Fixed timestep | 0.00833 | s |
| Maximum simulation time | ∞ | s |
| Grid resolution $(x, y, z)$ | 164 × 164 × 328 | voxels |
| Domain size $(x, y, z)$ | 20.0 × 20.0 × 40.0 | m |

(The original uses a long dash "—" in the Units column and a short dash "–" in the value column for entries without a value; both are kept.)

<!-- page 246:4 -->

**Table 5.** Simulation parameters for the vegetation combustion behavior experiment.

| Parameter | Value | Units |
|---|--:|:-:|
| ***Combustion & Pyrolysis*** | | |
| Gas fuel type | Methane | — |
| Combustion CO₂ fraction | 0.900 | — |
| Combustion residual fraction | 0.100 | — |
| Pyrolysis fuel fraction | 0.900 | — |
| Pyrolysis residual fraction | 0.100 | — |
| Gas residual reaction fraction | 0.500 | — |
| ***Heat Transfer*** | | |
| Convection coefficient | 200.0 | — |
| Radiation ray energy | 0.100 | J |
| Residual absorption coefficient | 3000 | — |
| ***Fluid Dynamics*** | | |
| Vorticity confinement strength | 10.0 | — |
| Wind velocity $(x, y, z)$ | (0.0, 0.0, 0.0) | m s⁻¹ |
| Porous solid velocity multiplier | 0.500 | — |
| Ambient temperature | 300 | K |
| Ambient gas density | 1.1726 | kg m⁻³ |
| Gravitational acceleration | −9.81 | m s⁻² |
| Porous solid density threshold | 1.0 | kg m⁻³ |
| Solid density threshold | 600.0 | kg m⁻³ |
| ***Particle System*** | | |
| Particle capacity | 524,288 | — |
| Ember spawn rate | 0.500 | — |
| Ember lifetime | 10.0 | s |
| Ember mass | [10⁻⁶, 0.005] | kg |
| Ember emissivity | 0.95 | — |
| Ember mass loss rate | 0.006 | kg s⁻¹ |
| Ember grid velocity factor | 0.500 | — |
| Flame-front lifetime | 1.0 | s |
| Flame-front speed | 8.0 | m s⁻¹ |
| ***Procedural Scene Generation*** | | |
| OSM request size | – | m |
| Vegetation density | – | — |
| Leaf thickness | 10⁻⁶ | m |
| Minimum house distance | – | m |
| Minimum road distance | – | m |
| ***Computational Domain*** | | |
| Fixed timestep | 0.004 | s |
| Maximum simulation time | 60.0 | s |
| Grid resolution $(x, y, z)$ | 240 × 240 × 200 | voxels |
| Domain size $(x, y, z)$ | 3.60 × 3.60 × 3.00 | m |

**Table 6.** Simulation parameters for the wind-driven structure exposure experiment.

| Parameter | Range / Value | Units |
|---|--:|:-:|
| ***Combustion & Pyrolysis*** | | |
| Gas fuel type | Methane | — |
| Combustion CO₂ fraction | 0.999 | — |
| Combustion residual fraction | 0.0005 | — |
| Pyrolysis fuel fraction | 1.000 | — |
| Pyrolysis residual fraction | 0.0005 | — |
| Gas residual reaction fraction | 0.001 | — |
| ***Heat Transfer*** | | |
| Convection coefficient | 200.0 | — |
| Radiation ray energy | 0.500 | J |
| Residual absorption coefficient | 500 | — |
| ***Fluid Dynamics*** | | |
| Vorticity confinement strength | 11.0 | — |
| Wind velocity $(x, y, z)$ | (12.0, 0.0, 0.0) | m s⁻¹ |
| Porous solid velocity multiplier | 0.140 | — |
| Ambient temperature | 300 | K |
| Ambient gas density | 1.1726 | kg m⁻³ |
| Gravitational acceleration | −9.81 | m s⁻² |
| Porous solid density threshold | 1.0 | kg m⁻³ |
| Solid density threshold | 1000.0 | kg m⁻³ |
| ***Particle System*** | | |
| Particle capacity | 1,048,576 | — |
| Ember spawn rate | 0.0 | — |
| Ember lifetime | [0.0, 120.0] | s |
| Ember mass | [0.001, 0.002] | kg |
| Ember emissivity | 0.950 | — |
| Ember mass loss rate | $1e-6$ | kg s⁻¹ |
| Ember grid velocity factor | 0.100 | — |
| Flame-front lifetime | 5.0 | s |
| Flame-front speed | 5.0 | m s⁻¹ |
| ***Procedural Scene Generation*** | | |
| OSM request size | – | m |
| Vegetation density | – | — |
| Leaf thickness | – | m |
| Minimum house distance | – | m |
| Minimum road distance | – | m |
| ***Computational Domain*** | | |
| Fixed timestep | 0.005 | s |
| Maximum simulation time | 365.0 | s |
| Grid resolution $(x, y, z)$ | 320 × 160 × 80 | voxels |
| Domain size $(x, y, z)$ | 32.0 × 16.0 × 8.0 | m |

## References

- O. Aichholzer and F. Aurenhammer. 1996. Straight Skeletons for General Polygonal Figures in the Plane. In *Proceedings of the Second Annual International Conference on Computing and Combinatorics (COCOON '96)*. Springer-Verlag, 117–126.
- S. Alaka and R. Bidarra. 2023. Hierarchical Semantic Wave Function Collapse. In *Proceedings of the 18th International Conference on the Foundations of Digital Games* (Lisbon, Portugal) *(FDG '23)*. Association for Computing Machinery, New York, NY, USA, Article 68, 10 pages. doi:10.1145/3582437.3587209
- S.-W. Cheng and A. Vigneron. 2007. Motorcycle Graphs and Straight Skeletons. *Algorithmica* 47, 2 (Feb. 2007), 159–182.
- R. L. Cook, J. Halstead, M. Planck, and D. Ryu. 2007. Stochastic simplification of aggregate detail. In *ACM SIGGRAPH 2007 Papers* (San Diego, California) *(SIGGRAPH '07)*. Association for Computing Machinery, New York, NY, USA, 79–es. doi:10.1145/1275808.1276476
- I. Karth and A. M. Smith. 2017. WaveFunctionCollapse is constraint solving in the wild. In *Proceedings of the 12th International Conference on the Foundations of Digital Games* (Hyannis, Massachusetts) *(FDG '17)*. Association for Computing Machinery, New York, NY, USA, Article 68, 10 pages. doi:10.1145/3102071.3110566

---

## Unreadable glyphs

None. Every equation, algorithm line and table cell was legible in renders of the PDF at 280 to 420 dpi, so no `[?]` markers were needed.

## Transcriber's notes

These notes are ours, not the authors'. The transcription above is unchanged; the points below flag possible typos or inconsistencies. Page numbers are supplement pages (246:n of the supplement PDF); "main" refers to the main paper.

1. **Stud state count (p. 2, Table 1).** The Studs row allows $n \in \{1,2,3\}$, $o \in \{\mathrm{H},\mathrm{V}\}$, $p \in \{\text{start, center, end}\}$, which gives $3 \times 2 \times 3 = 18$ combinations, yet the row lists 156 states. The column total of 179 ($156 + 9 + 3 + 9 + 2$) is arithmetically consistent with the printed rows.
2. **Undefined sets (p. 2, Table 2).** The parameter sets $O$, $H$, $V$ and the state set $W$ are not defined in the text. Table 1 gives Chimneys a parameter $p \in \{\text{start, end}\}$, while Table 2 writes Chimney$(h)$ with $h \in H$.
3. **Ember heat deposit units (p. 3, Algorithm 2, line 5).** $\dot{q}_e \leftarrow m_e (T_e - T_s) C_e$ is an energy (J) if $C_e$ is a specific heat capacity, although it carries a rate dot. $T_e$ and $C_e$ are not defined (presumably ember temperature and ember specific heat). The symbol $\dot{q}_e$ also resembles the main paper's evaporation term $\dot{q}_{e,s}$.
4. **Solid temperature update (p. 3, Algorithm 2, line 7).** $T_s \leftarrow T_s + \dot{q} \cdot \rho_s C_{p,s}$ multiplies by $\rho_s C_{p,s}$; a temperature increment from deposited energy requires division by $\rho_s C_{p,s}$ (and by the cell volume).
5. **Temperature used for gas ignition (p. 3, Algorithm 2, line 11).** The line uses $max(T_f, T_s)$ with the solid temperature $T_s$, while main Sec. 4.7 says the particle temperature supersedes the local cell temperature during gaseous combustion, whose rate (main Eq. 10) is evaluated at $T_g$.
6. **Ground materials (p. 2, Sec. 1.5 versus Table 3).** Sec. 1.5 assigns asphalt to roads and concrete to curbs and sidewalks, while Table 3 assigns curbs and roads to the Stone class, and main Table 4 has no asphalt or concrete entry. The same paragraph repeats itself: "vegetated ground, such as lawns or vegetated ground".
7. **Class names versus material names (p. 2, Table 3; main p. 18, Table 4).** Table 3 uses classes Dense biomass, Sparse biomass, Metal, Stone and Glass; main Table 4 lists Steel, Wood, Leaves, Stone, Glass and Moisture. The mapping (Dense biomass to Wood, Sparse biomass to Leaves, Metal to Steel) is implied but not stated.
8. **Values outside the main Table 5 ranges (main p. 19).** Table 4 (p. 3): grid resolution $z = 328$ exceeds the stated maximum of 300 voxels, and the maximum simulation time is ∞ against a stated range of [0.0, 300.0] s. Table 5 (p. 4): domain size 3.60 × 3.60 × 3.00 m is below the stated minimum of 16 × 16 × 8 m. Table 6 (p. 4): maximum simulation time 365.0 s exceeds 300.0 s, and the ember lifetime lower bound 0.0 s is below the stated minimum of 1.0 s.
9. **Embers in the structure-exposure run (p. 4, Table 6).** Ember spawn rate is 0.0, so this validation ran without embers, while ember lifetime, mass, emissivity and mass-loss values are still listed. The mass-loss rate is printed in math italics as "$1e-6$".
10. **Column header (p. 4, Table 5).** Table 5 heads its value column "Value", whereas Tables 4 and 6 use "Range / Value".
11. **Symbol reuse (p. 1).** Sec. 1.2 uses $\sigma$ for the interface label, which is the Stefan–Boltzmann constant in the main paper, and Sec. 1.3 uses $T$ for the target condition, which is temperature in the main paper. Not an error, but relevant when reading the equation catalogue.
12. **Figure 2 wording (p. 2).** The caption says a polygon (a) is "extended before" the SSA step, while Sec. 1.4 describes panel (a) as "an extruded input polygon" and (b) as "the extended roof polygon computed with the SSA".
