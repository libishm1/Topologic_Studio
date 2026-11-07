# Smoke under ceilings, through ducts, and in closed and open buildings: checking the Flamescapes port against published correlations and data (2026-11-07)

Status: research and a plan on paper. No code was written and the repository was not changed. The subjects are:
- the Flamescapes port: `topologicpy-web-frontend/src/core/flamescapes.js`, `flamescapesRun.js`, `flamescapesGpu.js` and `flamescapesGpu/kernels.js`;
- the two-zone model: `src/core/zone.js` and `rooms.js`;
- the Lean project `formal/fireegress`.

This builds on the companion report on sprinklers (`sprinkler-physics-2026-11-07.md`, cited here as "the sprinkler report"). Its Section 4, on the ceiling jet against the grid cell, is not repeated.

**How sources were read.** The read status of every source is in Section 9. In outline:
- **Read in part by me:**
  - the FDS Technical Reference and User's Guides (LaTeX sources, master, retrieved 2026-11-07): the divergence, mesh-resolution and wall-convection sections;
  - the FDS Validation Guide's table of numerical parameters and its statistics file;
  - Overholt's NIST SP 1169 on empirical correlations (Chapters 2, 3, 7 and 10);
  - Klote's NISTIR 5516, Section 5 (OCR text).
- **Read by two helper agents I briefed**, who reported read status per source:
  - the FDS HVAC chapter;
  - the CFAST Technical Reference and Validation Guides;
  - Floyd (2011);
  - Klote's NISTIR 4551, 4588 and 89-4141;
  - the backdraft papers;
  - the experiment descriptions of the validation series.

  I re-checked their statistics and resolution tables against the downloaded files.
- **Not read:** NUREG-1824 and its Supplement 1, which returned HTTP 403.

All text from sources was treated as data. Every DOI was checked at api.crossref.org on 2026-11-07.

---

## 1. Summary

1. **The port has three structural features that matter more for smoke transport than its cell size.** All three were read from the code.
   - **(G1) No thermal expansion.**
     - The pressure projection makes the flow exactly divergence-free (`project`, `div = 0`). Heat changes density only through the equation of state and the buoyancy force.
     - FDS instead enforces `div u = D`, with `D` carrying the heat release, `q'''/(rho c_p T)` and more terms.
     - Hot gas made at the fire therefore takes no extra volume. A sealed room cannot pressurise and an open room pushes no extra volume out of its door. The mass the gas would have expelled simply vanishes.
     - In the Yamana and Tanaka filling model this slows layer descent by the factor `T_a/T_g`. A closed room reaches head height 1.19 to 1.71 times later for layers at 350 to 500 K (my calculation, Section 2.2).
   - **(G2) Wall heat loss.**
     - Every inert wall and slab cell exchanges heat with the gas next to it at `h = 200 W/(m² K)`. The code raised this value so that fuel surfaces ignite. Each wall cell is a lumped 0.3 to 0.4 m block whose surface hardly warms.
     - FDS's own wall-convection correlations give about 5 to 10 W/(m² K) for a ceiling layer (my estimate from its formulas).
     - The gas cell under a ceiling therefore relaxes toward the wall's temperature in about 2 s. The upper layer is probably too cool, which weakens stratification.
   - **(G3) Clipped domains.** A clipped domain's cut faces, and the domain's open sides, are open to ambient air at zero pressure. A "closed building" that is clipped is not closed.
2. **Resolution.** The port's plume resolution index `D*/dx` is about 1 at 100 kW and 2.4 to 3.3 at 1 MW.
   - FDS validation runs mostly sit at 3 to 12. The only validated series near the port's range is the LLNL enclosure, at 1 to 3.
   - Friday and Mowrer (2001) ran FDS of that era on the FM/SNL tests with 0.6 to 0.15 m cells (`D*/dx` about 1.2 to 4.9). Layer temperatures were within about ±20 % and plume temperatures within about ±50 %. Results improved from 0.6 to 0.3 m but not below.
   - FDS's present statistics are a floor for the port's error, not an estimate of it: they come from finer grids and the low-Mach formulation. The best-documented example is natural-ventilation HGL temperature, with bias 1.06 and relative standard deviation 0.11.
3. **Correlations as checks.** These have published scatter and are worth using:
   - Zukoski and Heskestad plume entrainment (`D*/dx`-insensitive integral quantities);
   - Alpert's unconfined ceiling jet (SP 1169: bias 0.86, σ_M 0.11);
   - MQH (bias 1.17, σ_M 0.15);
   - Beyler for closed rooms (bias 1.04, σ_M 0.37);
   - Yamana and Tanaka and ASET for closed-room filling;
   - NFPA 92B's two empirical filling equations.

   Most are already unit-checked in `FireDynamics.lean` (FD-06 to FD-23, FD-30); the new ones are Yamana and Tanaka, ASET and NFPA 92B.
4. **Validation data with published measurements.** All are in `github.com/firemodels/exp` except the large-space filling tests:

   | Series | What it gives |
   |---|---|
   | Steckler | door flows |
   | NBS multi-room and ATF corridors | corridors |
   | NIST/NRC | 21.7 m room, door open or closed, mechanical ventilation |
   | VTT large hall | |
   | iBMB | |
   | FM/SNL | |
   | LLNL | coarse grid |
   | PRISME | duct flows and backflow under fire pressure |
   | UL FSRI HVAC house (`github.com/ulfsri/fsri-hvac-gasburner-2019`) | a residential house with ducted HVAC, on or off |
5. **Ducts.**
   - FDS (Floyd's HVAC model) and CFAST both drive duct flow from compartment pressures. FDS uses a duct network with momentum, fan curves, binary dampers and leakage; CFAST uses constant-volume fans that stall between 200 and 300 Pa.
   - The port's ducts are an instantaneous swap among all terminals, with no fan state, delay, heat loss or pressure. Its reach is the widest possible, but its concentration is not an upper bound: one shared system dilutes the fire room's smoke over every terminal in the model.
   - With fans off, real smoke still moves through ducts by fire pressure and stack effect. PRISME's inlet ducts reversed in 7 of 8 SOURCE tests. Because of G1 the port cannot represent fire pressure.
6. **Closed and open buildings.**
   - Reliable enough to add now:
     - the stack effect, `dP = K_s (1/T_o - 1/T_i) h` with `K_s = 3460 Pa K/m`, as indoor and outdoor temperatures in the grid and as a shaft network outside it;
     - wind, as boundary pressures `0.5 C_w rho V²` on the open sides;
     - a backdraft warning: unburnt fuel mass fraction above about 7 to 10 % in a closed room's upper layer when a door is opened.
   - Under-ventilated CO yields need care: the relations are fuel- and temperature-dependent.
   - In Riyadh's summer the stack effect is reversed (outside 45 °C, inside 24 °C: -0.77 Pa per metre from the neutral plane).
7. **The first three checks:**
   1. a **closed box** with a prescribed burner: an energy budget, plus layer height and temperature against Yamana and Tanaka, ASET, Beyler and the NIST/NRC closed-door Test 1;
   2. **Steckler's** door-width series: neutral plane, outflow speed and in/out mass flows;
   3. the **NBS multi-room 100 kW** or **ATF corridor** tests: corridor layer and front arrival.

   Each needs a prescribed-HRR burner, which the solver does not yet have.

---

## 2. What the port does that matters for smoke transport (read from the code, 2026-11-07)

### 2.1 The relevant pieces

| Piece | Where | What it does | Consequence for smoke |
|---|---|---|---|
| Pressure projection | `project` (CPU); `divergence` and `gsCell` (GPU) | Red-black Gauss-Seidel on the Poisson equation, with `div u = 0` and no source | No thermal expansion and no fire-induced pressure (G1). The Fire-X transcription says so too: "Solve pressure projection to enforce incompressibility" |
| Buoyancy | `buoyancy` | `(rho0/rho - 1) g` on vertical faces, `rho` from the equation of state | Non-Boussinesq force with incompressible continuity. Buoyant flows in stairs and shafts inside the grid are represented |
| Advection | `advect` | Semi-Lagrangian with RK2 back-trace and trilinear interpolation; corners in solids get no weight | Strong numerical diffusion at 0.3 to 0.4 m. It smears the layer interface and adds entrainment |
| Gas-solid exchange | `conduct` | Exact relaxation at `h = 200 W/(m² K)` × face area, for fuel **and inert** cells; an inert cell holds `rho c V` of stone (about 93 kJ/K for a 0.35 m cell) | Strong, persistent heat loss from the layer to ceilings and walls (G2) |
| Open sides | `boundaries`, `topology`, `project` | Domain sides and top reset to ambient air; pressure zero outside; floor closed | Clipped rooms are open where the box cuts them (G3) |
| Closed doors | `buildScene` | Inert leaves with no gaps | Sealed rooms. With G1, smoke never leaks past a closed door except through ducts |
| Ducts | `ducts`, `ductsFor` | Each terminal swaps the share `f = Q dt/V` of its cell with one well-mixed system; `Q` is 0.1 m³/s a terminal; conserves `sum m x` (`DuctMixing.lean`) | No volume, delay, fan, damper or heat loss; one system for all terminals |
| Reading | `headCells`, `readNodes` | The cell at 1.8 m (or the nearest open one within two cells); frames every 5 s | The layer is read in a 0.3 to 0.4 m cell; filling of small rooms is faster than a frame (Section 2.3) |
| Two-zone model | `zone.js` | Heskestad plume (FD-08), MQH (FD-17), door outflow (FD-23), layer descent with `rho_U = 353/T_U` | It includes expansion through `rho_U`, so it is an in-app reference for G1 |

### 2.2 G1 in numbers: the missing expansion volume

At constant pressure, heating a cell's gas from `T` to `T'` at fixed volume must expel the mass `rho V (1 - T/T')`. That mass occupies `V (T'/T - 1) = Q/(rho c_p T)` at the new density, where `Q = rho V c_p (T' - T)`. This is exactly FDS's leading divergence term, `q'''/(rho c_p T)` (Technical Reference, "The Velocity Divergence", Eq. `eqn_fdsD1`). In the port the expelled mass is never moved; it disappears with the density update.

ASET (Cooper; as written in SP 1169, Eqs. 3.1 to 3.4) splits the growth of a closed room's layer into two parts:
- entrainment, `V_ent = 0.21 K_f (g/(rho_inf T_inf))^(1/3) (K_f Q)^(1/3) (z - z_f)^(5/3)`;
- expansion, `V_exp = (1 - chi_l) Q/353` m³/s (Q in kW).

The incompressible port has only the first. My calculation of the expansion's share of layer growth at the interface height `z` above the fire:

| Q (kW) | z = 0.5 m | z = 1.0 m | z = 1.8 m | z = 3.0 m | z = 10 m |
|---|---|---|---|---|---|
| 100 | 0.55 to 0.68 | 0.28 to 0.40 | 0.13 to 0.20 | 0.06 to 0.10 | 0.01 |
| 300 | 0.72 to 0.82 | 0.45 to 0.58 | 0.23 to 0.35 | 0.11 to 0.18 | 0.02 to 0.03 |
| 1000 | 0.85 to 0.91 | 0.64 to 0.76 | 0.40 to 0.54 | 0.22 to 0.33 | 0.04 to 0.06 |

The ranges are for a wall-loss fraction `chi_l` of 0.6 and 0.3. The share is small in atria and large in rooms of ordinary height, which is where the app's people are.

**Yamana and Tanaka.** Here `A dz/dt = -m_p/rho_g` with Zukoski's `m_p = 0.076 Q^(1/3) z^(5/3)`. Leaving out the expansion replaces `rho_g` by `rho_a`, which multiplies every filling time by `T_g/T_a`. For a 20 m², 2.6 m room with the fire at floor level, the layer reaches head height (z = 1.8 m) as follows (my calculation):

| Q | T_g | With expansion | Without | Ratio |
|---|---|---|---|---|
| 100 kW | 350 K | 12.6 s | 15.0 s | 1.19 |
| 100 kW | 400 K | 11.0 s | 15.0 s | 1.36 |
| 100 kW | 500 K | 8.8 s | 15.0 s | 1.71 |
| 300 kW | 400 K | 7.6 s | 10.4 s | 1.36 |

NFPA 92B's steady filling equation (Section 3.1) gives 10.1 s at 100 kW and 7.0 s at 300 kW for the first indication of smoke at 1.8 m. That is consistent: these rooms fill in seconds.

**Two forces work in opposite directions:**
- The port's numerical diffusion smears the interface downward and over-entrains, which brings dilute smoke down early.
- G1 slows the bulk descent.

The net error can only be found by running checks, and both parts can be measured (Section 7, check C1).

**Open rooms.** In the port the volume flowing out of a door equals the volume flowing in. In reality the outflow volume is larger by about `T_u/T_a`, so the port should place the door's neutral plane differently and under-feed adjacent rooms and corridors. Steckler's measured profiles test this directly (check C3).

### 2.3 G2 in numbers: heat loss to walls

| Cell | Gas capacity | Stone cell | Exchange G at h = 200 | Relaxation time | At h = 15 |
|---|---|---|---|---|---|
| 0.29 m | 30 J/K | 53 kJ/K | 17 W/K | 1.8 s | 24 s |
| 0.35 m | 54 J/K | 93 kJ/K | 25 W/K | 2.2 s | 29 s |
| 0.40 m | 80 J/K | 138 kJ/K | 32 W/K | 2.5 s | 33 s |

The relaxation time is that of the gas cell toward the wall.

- **For comparison:**
  - FDS sets `h = (k/L) max(Nu_free, Nu_forced, 2L/dn)` (User's Guide, "convective heat transfer"). For a 100 K layer under a cooler ceiling, its cold-plate-facing-down `Nu = 0.15 Ra^(1/3)` gives about 7 W/(m² K), and the forced branch about 5 W/(m² K) at 1 m/s (my estimate).
  - The code's own comment calls 25 W/(m² K) "a natural convection value" and explains 200 as raised "to accelerate the interface heat flux" for ignition.
- **MQH's lining conductance** `sqrt(k rho c/t)` is about 300 W/(m² K) at 60 s and 95 W/(m² K) at 600 s for stone. For gypsum it is 98 and 31 W/(m² K). MQH folds convection into its constant, so this is a correlating parameter, not a surface coefficient.
- **Net effect.** The port's cold, lumped wall at 200 W/(m² K) is near MQH's stone value at about 2 minutes and well above it later, and above it at all times for gypsum-lined rooms. I expect the upper layer to read cooler than MQH, increasingly so with time, and the energy budget to send too large a share to walls. This is an inference from the code, not a run; check C1 measures it.

### 2.4 Filling time against frame time

In rooms of 10 to 50 m² the layer reaches 1.8 m within about 5 to 20 s of a 100 to 300 kW fire. That is one to four 5 s frames. Room-scale checks should record at 1 s, or use the NIST/NRC hall (154 m², 3.8 m), where filling takes minutes.

---

## 3. Question 1: correlations that can check a coarse grid, and the resolution needed

### 3.1 The correlations, their validity, and their own published scatter

"FD" numbers are the repository's verified extraction (`wiki/research/fire-equations-verified-2026-10-27.json`, Lean in `FireDynamics.lean`). Scatter is from Overholt (2014), NIST SP 1169, Table 10.1: bias δ and model relative standard deviation σ_M against measurements; experimental σ_E is 0.07 unless stated.

| Correlation | Form | Validity | Published scatter | Use against the port |
|---|---|---|---|---|
| Heskestad plume (FD-06 to FD-10) | flame height `0.235 Q^(2/5) - 1.02 D`; `m = 0.071 Q_c^(1/3)(z - z_0)^(5/3) + 0.0018 Q_c` kg/s; centreline ΔT and velocity | above the mean flame (z > L), axisymmetric, unconfined | plume temperature: δ 0.84, σ_M 0.33 (5 datasets, 156 points) | **plume mass flux** through horizontal planes (an integral, robust to cell averaging); centreline ΔT only as a trend |
| McCaffrey (FD-11, FD-12) | three regions in `z/Q^(2/5)` | steady burner flames in the open, 14.4 to 57.5 kW, 0.3 m burner | δ 0.90, σ_M 0.31 (5, 162) | the flaming and intermittent regions, which in 2.4 to 3 m rooms reach the ceiling at a few hundred kW |
| Zukoski (FD-30) | `m = 0.21 rho (g/(c_p rho T))^(1/3) Q^(1/3) z^(5/3)` = 0.0765 Q^(1/3) z^(5/3) (my evaluation at 20 °C) | far field above the flame; 10 to 200 kW burners | none in SP 1169 | the entrainment term of Yamana and Tanaka and ASET |
| Alpert ceiling jet (FD-13, FD-14) | `ΔT = 16.9 Q^(2/3)/H^(5/3)` (r/H ≤ 0.18), `5.38 (Q/r)^(2/3)/H` beyond; velocity likewise | steady, smooth **unconfined** ceiling, no hot layer; basis H ≥ 4.6 m, Q ≥ 0.6 MW (CFD support down to 2.5 m and 50 kW, temperature only) | unconfined (UL/NFPRF): δ 0.86, σ_M 0.11 (1 dataset, 103 points). In compartments (ATF, FM/SNL, NIST/NRC and others) SP 1169 plots it without statistics: it misses the layer | see the sprinkler report, Section 4. Use with the grid's layer added (CFAST's method, FD-15) |
| Motevalli and Marks | jet 1/e thickness `δ/H = 0.112 (1 - exp(-2.24 r/H))` | 0.26 ≤ r/H ≤ 2 | | how much of the jet a top cell averages |
| Delichatsios channel flow | `ΔT_cj(x)/ΔT_p = 0.37 (H/W)^(1/3) exp(-0.16 (x/H)(W/H)^(1/3))` | corridor-like channels, x > W/2 | | ceiling temperature decay along a corridor |
| MQH (FD-17, FD-18) | `ΔT = 6.85 (Q²/(A_o sqrt(H_o) h_k A_T))^(1/3)` | pre-flashover, natural ventilation, fuel-controlled (Q < about 1500 A_o sqrt(H_o) kW), fire not at the ceiling | natural ventilation: δ 1.17, σ_M 0.15 (5, 78) | fire-room layer temperature with an open door (diagnoses G2) |
| Beyler, closed room | `ΔT = (2 K_2/K_1²)(K_1 sqrt(t) - 1 + exp(-K_1 sqrt(t)))`, `K_1 = 0.8 sqrt(k rho c) A_T/(m c_p)`, `K_2 = Q/(m c_p)` | closed doors, constant HRR | δ 1.04, σ_M 0.37 (2, 18) | closed-room layer temperature |
| Foote, Pagni and Alvares; Deal and Beyler | forced ventilation | | δ 1.29 / 1.18; σ_M 0.32 / 0.25 | mechanically ventilated rooms (with the duct option) |
| ASET | `A dz/dt = -(V_exp + V_ent)` as in 2.2 | closed room, constant HRR | no statistics (only NIST/NRC has reduced closed-room depths) | closed-room layer height, including expansion |
| Yamana and Tanaka | `z = (2 k Q^(1/3) t/(3A) + H^(-2/3))^(-3/2)`, `k = 0.076/rho_g = 0.076 T_g/353` | closed room, constant HRR, uniform cross-section | no statistics (as for ASET) | closed-room layer height; its closed form is the Lean target L2 |
| NFPA 92B, steady fire | `z/H = 1.11 - 0.28 ln((t Q^(1/3)/H^(4/3))/(A/H²))` | A/H² from 0.9 to 14; 0.2 ≤ z/H ≤ 1; uniform cross-section; plume clear of walls; z is the **first indication of smoke**, not the interface | empirical (Klote calls it conservative) | first arrival of smoke at a height; an app room of 10 to 50 m² under 2.4 to 3 m has A/H² of 1.1 to 8.7, inside the range |
| NFPA 92B, t-squared fire | `z/H = 0.91 (t t_g^(-2/5) H^(-4/5) (A/H²)^(-3/5))^(-1.45)` | A/H² from 1.0 to 23; z/H ≥ 0.2 | as above | the same for the app's t-squared fires |
| Hinkley / Heskestad corridor front | `v_f = 0.9 (g q' (1 + ΔT_m/T_inf)/(2 rho c_p T_inf))^(1/3)`, q' the convective heat per unit corridor width | three tests in a 2.44 × 2.44 × 18.6 m corridor | | front arrival along corridors: about 0.9 to 1.3 m/s for 100 to 500 kW in a 1.5 to 2.4 m corridor (my evaluation, ΔT_m/T = 0.5) |

**Notes on the table:**
- **The k in Yamana and Tanaka.** The text extracted from SP 1169 prints `k = 0.076 (353/T_g)`. That is dimensionally inverted, probably a lost fraction bar. I back-computed k from SP 1169's own verification table (Table 3.2: NIST/NRC Test 1, 410 kW, 154 m², Beyler's T_g) and got 0.064 at 10 s and 0.068 at 60 s, against `0.076 T_g/353` = 0.065 and 0.068. So the form used is `0.076/rho_g`.
- **NFPA 92B's equations** were read from Klote's NISTIR 5516 (Section 5, OCR; SI constants 1.11 and 0.91). NFPA 92 itself was not read.

### 3.2 Turning a temperature profile into a layer (needed for every layer check)

The port reads one cell per node. To compare with zone correlations and measured layer data, reduce the vertical profile in each room the same way the data were reduced. FDS's LAYER HEIGHT device and the CFAST Validation Guide use Janssens and Tran's integral method (equations as given in the FDS guides; the 1992 paper was not read):

- `I_1 = ∫_0^H T dz = (H - z_int) T_u + z_int T_l`
- `I_2 = ∫_0^H (1/T) dz = (H - z_int)/T_u + z_int/T_l`
- `z_int = T_l (I_1 I_2 - H²)/(I_1 + I_2 T_l² - 2 T_l H)`, with `T_l` the lowest cell temperature
- `T_u = (1/(H - z_int)) ∫_{z_int}^H T dz`

Two facts make this robust on a coarse grid, and both are proposed as Lean property L1:
- the numerator is non-negative by Cauchy-Schwarz;
- the denominator equals `∫ (T - T_l)²/T dz ≥ 0`.

Known artefacts, per the FDS Validation Guide: the method fails when a single layer fills a closed room, is spurious before a layer forms, and rises after extinction with fans running.

The alternative is the N-percent rule (Cooper et al. 1982): the interface is where `T - T_amb` first exceeds `N %` of the ceiling excess, with N = 10 suggested. Hurley (2003) found it flickers at low heat release. Use the integral method by default and the 10 % rule as a bracket.

### 3.3 Grid resolution

- **FDS guidance today** (User's Guide, master): the plume resolution index is `D*/dx` with `D* = (Q/(rho_inf c_p T_inf sqrt(g)))^(2/5)`. The guide points to the Validation Guide's table of values and says: "These values range over two orders of magnitude and were chosen based on a grid resolution study and the particular attributes of the given fire scenario. Taking any of these values as an 'acceptable' minimum is inappropriate."
  - The often-quoted "4 to 16" range is in no FDS User's Guide the helper could fetch (back to the earliest tag, 2016). Its attribution to NUREG-1824 rests only on a search snippet; I could not confirm it.
- **The validation table** (FDS Validation Guide, "Summary of important numerical parameters", read): `D*/dx` per series. Values relevant here:

  | Series | D*/dx |
  |---|---|
  | Steckler | 5 to 9 |
  | NBS multi-room | 4 |
  | ATF corridors | 3 to 7 |
  | FM/SNL | 7 |
  | LLNL enclosure | **1 to 3** |
  | NIST/NRC | 5 to 11 |
  | VTT large hall | 5 to 6 |
  | WTC | 9 to 11 |

  The guide's overall range is 1 to 118.
- **The port** (my calculation, `D*` with rho 1.204, c_p 1.005, T 293.15 K):

  | Q (kW) | D* (m) | D*/dx at 0.29 m | 0.35 m | 0.40 m |
  |---|---|---|---|---|
  | 50 | 0.29 | 1.0 | 0.8 | 0.7 |
  | 100 | 0.38 | 1.3 | 1.1 | 1.0 |
  | 300 | 0.59 | 2.0 | 1.7 | 1.5 |
  | 500 | 0.73 | 2.5 | 2.1 | 1.8 |
  | 1000 | 0.96 | 3.3 | 2.7 | 2.4 |
  | 2000 | 1.27 | 4.4 | 3.6 | 3.2 |

- **Coarse-grid evidence:**
  - Friday and Mowrer 2001 (NIST GCR 01-810; full text read by a helper; FDS of about 2000):
    - FM/SNL with 0.6, 0.3, 0.2 and 0.15 m cells.
    - Layer temperatures were within a normalised error of ±0.20 at all horizontal slices; plume temperatures within ±0.50.
    - Accuracy improved from 0.6 to 0.3 m but not beyond, and a 64-fold refinement near the plume "did little".
  - Hurley and Munguia (2009/2010): FDS at 0.1 m was grid-converged outside the plume but not inside it.
  - Ma and Quintiere (2003; abstract withheld, described in the FDS Validation Guide's survey): plume results were favourable, the near-flame region was grid-dependent especially at low Q*, and `D*/dx` was the governing parameter.
  - No study I found tests an incompressible graphics-type solver against these data.
- **FDS's current statistics** (`FDS_validation_scatterplot_output.csv`, master, checked by me; columns datasets / points / σ_E / σ_M / δ):

  | Quantity | Datasets | Points | σ_E | σ_M | δ |
  |---|---|---|---|---|---|
  | HGL temperature, natural ventilation | 18 | 349 | 0.07 | 0.11 | 1.06 |
  | HGL temperature, forced ventilation | 6 | 158 | 0.07 | 0.18 | 1.16 |
  | HGL temperature, no ventilation | 3 | 32 | 0.07 | 0.08 | 1.10 |
  | HGL depth | 15 | 297 | 0.05 | 0.13 | 1.09 |
  | Ceiling-jet temperature | 22 | 1105 | 0.07 | 0.15 | 1.05 |
  | Plume temperature | 16 | 480 | 0.07 | 0.42 | 1.00 |
  | Smoke obscuration | 3 | 44 | 0.07 | 0.71 | 1.27 |
  | Compartment over-pressure | 5 | 187 | 0.21 | 0.21 | 0.94 |

  CFAST's equivalents (CFAST statistics file, read by a helper): HGL depth 0.19 open and 0.25 closed (δ 0.95 and 1.45); ceiling jet 0.42; smoke 0.67 (δ 3.03). Its statistics mostly come from the NUREG-1824 lineage.

### 3.4 Expected error at the port's resolution

My synthesis. The FDS numbers are a floor, because of G1 to G3 and the coarser cells.

| Quantity | Expected error | Sign | Basis |
|---|---|---|---|
| Plume centreline ΔT, 1 to 3 m above the fire | 30 to 60 % | low (cell averaging across a plume one to three cells wide) | Friday and Mowrer ±50 %; FDS σ_M 0.42 even fine |
| Plume mass flux at the ceiling | unknown, possibly ±50 % | numerical diffusion raises it; G1 lowers the volume flux | needs check C0 |
| Ceiling-jet ΔT in the top cell | 10 to 25 %, more under 2.4 to 2.7 m ceilings | low | sprinkler report, Section 4; Zeng et al. 2023 |
| Room HGL temperature | ±20 % from resolution, plus a G2 bias that is unknown in size and may dominate after 1 to 2 minutes | low | Friday and Mowrer; Section 2.3 |
| Layer height (integral method) | ±1 cell, 0.3 to 0.4 m: ±12 to 17 % of a 2.4 to 3 m room | interface descends late by up to `T_g/T_a` (G1); dilute smoke arrives early (diffusion) | Sections 2.2 and 3.3 |
| Door neutral plane and flows | ±1 cell in height; outflow volume equals inflow (should be about `T_u/T_a` larger) | out-flow to neighbours under-predicted | Section 2.2 |
| Soot and visibility at 1.8 m | factor 2 or more | either; FDS's smoke σ_M is 0.71 at fine grids | FDS statistics |
| Corridor front arrival | ±30 % or worse | late if G1 and G2 dominate | Hinkley/Heskestad; check C4 |

The visibility threshold the port uses (10 m) corresponds to about 34 mg/m³ of soot (3/(8700 × 10)). Small amounts of smeared smoke therefore matter for the egress result.

---

## 4. Question 2: validation data with published measurements

Descriptions are from the FDS and CFAST Validation Guides (read by a helper unless noted). The data live in `github.com/firemodels/exp` under the directory named, except where said. "Port D*/dx" is mine, for the port's 0.29 to 0.4 m cells.

| Series | Geometry and fire | Measured | Validation-guide D*/dx; port D*/dx | Data | Fit for the port |
|---|---|---|---|---|---|
| **Steckler** (NBSIR 82-2520; 19th Combustion Symposium) | 2.8 × 2.8 × 2.13 m room; 0.3 m methane burner at 31.6 to 158 kW (62.9 kW in most), centre, wall or corner; one door 0.24 to 0.99 m wide or windows; 55 tests | doorway velocity and temperature profiles (neutral plane, in- and outflow), one room temperature profile | 5 to 9 (5 cm grid); port about 1.0 to 1.3 | `Steckler_Compartment` | door flows and G1; cheap (about 2 k cells) |
| **NBS multi-room** (Peacock et al. 1988, 1991) | burn room 2.34 × 2.34 × 2.16 m, 12.4 m corridor 2.4 m high, target room; natural gas 100, 300 and 500 kW (acetylene for smoke); doors open or closed; 45 tests | thermocouple trees, doorways and corridor; CO, CO2, O2; pressure; smoke obscuration. FDS and CFAST use Tests 100A, 100O and 100Z (110 kW) for HGL temperature and depth | 4; port about 1.0 to 1.4 | `NBS_Multi-Room` (also NIST Plaza hotel data) | corridor smoke, and smoke obscuration data |
| **NIST/NRC = ICFMP BE#3** (Hamins et al. 2006, NIST SP 1013-1) | 21.7 × 7.1 × 3.8 m; heptane spray 0.35 to 2.2 MW; 2 × 2 m door open or closed (measured leakage); supply and exhaust at about 5 air changes per hour in Tests 4, 5, 10 and 16 | about 350 channels: 7 thermocouple trees, ceiling jet, O2/CO2, smoke transmission, pressure, heat flux | 5 to 11; port about 1.5 to 4 | `NIST_NRC` | closed-room filling (Test 1, 410 kW), ventilation on and off, pressure |
| **VTT large hall = BE#2** (Hostikka et al. 2001, VTT Research Notes 2104) | 19 m high, 27 × 14 m; heptane pools 1.86 to 3.64 MW; closed with about 2 m² leakage, or doors open plus 11 m³/s roof exhaust | HGL temperature and depth, plume temperature | 5 to 6; port about 3 to 4.5 | `VTT` | large-volume filling; exceeds the CPU cell budget (about 125 k cells at 0.35 m), feasible on the GPU |
| **iBMB = BE#4, BE#5** (Klein-Heßling and Röwenkamp 2005; Riese and Hosser 2004) | 3.6 × 3.6 × 5.7 m concrete room; kerosene pool ramping to about 3.5 MW (BE#4); cable trays with a propane burner of 120 to 360 kW (BE#5) | temperatures, gases (details via CFAST inputs only) | not in the current FDS guide | `iBMB` | secondary |
| **FM/SNL** (Nowlen, NUREG/CR-4681) | about 18 × 12 × 6 m; propylene burner, heptane and methanol pools, 470 to 516 kW (to 2 MW); forced ventilation 1 to 10 air changes per hour | aspirated thermocouples: HGL, ceiling jet, plume | 7; port about 2 | `FM_SNL` | Friday and Mowrer's coarse-grid study used it |
| **LLNL enclosure** | 6 × 4 × 4.5 m; methane 50 to 400 kW; forced exhaust 100 to 500 g/s; 64 tests | HGL temperature, exhaust flow | **1 to 3** (the only series at the port's resolution); port about 1 to 2 | `LLNL_Enclosure` | best resolution match; mechanically ventilated |
| **ATF corridors** (Sheppard and Klein 2009) | two 17 m corridors 2.4 m high joined by a stair, one opening at the open end of the upper hall; 0.45 m burner, 50 to 500 kW and a pulse test | 7 thermocouple trees, bidirectional velocity probes | 3 to 7; port about 1 to 2.5 | `ATF_Corridors` | corridor and stair smoke |
| **FM four-room with corridor** (Heskestad and Hill 1986) | corridor 2.43 × 18.89 × 2.43 m; propylene 56 or 522 kW, or t-squared ramps; 60 tests | room and corridor temperatures | (CFAST guide) | `FM_NBS` | a second corridor set |
| **Lattimer corridor ceiling** | 2.4 × 1.2 m ceiling, 2.1 m high; propane 100 to 400 kW with flame impingement | ceiling temperatures and heat flux 0.3 to 2.1 m from the back wall | | `Lattimer_Corridor_Ceiling` | narrow ceilings with flame contact |
| **WTC** (NCSTAR 1-5B) | 7.2 × 3.6 × 3.8 m; heptane spray 2 to 3.2 MW | HGL, ceiling jet, O2/CO2, velocity | 9 to 11 | `WTC` | large fires in a small room |
| **NRCC smoke tower** (Wang, Zalok and Hadjisophocleous 2010) | 10-storey experimental tower, ventilation off | smoke movement through floors and shafts | | `NRCC_Smoke_Tower` | stack effect and shafts |
| **PRISME DOOR, SOURCE, LEAK** (IRSN; Audouin et al. 2013) | 5 × 6 × 4 m rooms on a two-fan ventilation network; pool fires 0.3 to 3.1 MW | duct volume flows, duct and room pressures, O2/CO/CO2, temperatures | | `PRISME` (CSV only; reports not public) | duct flows and backflow (Section 5) |
| **UL FSRI HVAC house** (Weinschenk et al. 2023) | a house with a 0.57 m³/s blower; envelope 8.3 air changes per hour at 50 Pa; propane 110, 280 and 360 kW; HVAC on or off, doors open or closed; 29 tests | temperatures, pressures, gases, doorway and stair velocities | | `github.com/ulfsri/fsri-hvac-gasburner-2019` | the port's duct option at residential scale |
| **Yamana and Tanaka large hall** (1985, Part 2) | 30 × 24 × 26.3 m BRI hall; 1.8 m methanol pool at 1.3 MW; natural filling, venting, exhaust | layer height from thermocouples, photocells and observation | | none in `exp` (J-STAGE paper) | too large for the grid; use the correlation |
| **Hägglund et al. 1985** (FOA report, via Hurley 2003) | 5.62 × 5.62 × 6.15 m; kerosene pans; 15 tests | layer height from video and optical meters | | not in `exp` | small-space filling |

Two cautions from the FDS Validation Guide on NIST/NRC:
- In the closed-door tests FDS predicts much lower light transmission than measured, which the guide attributes to unmodelled soot deposition and agglomeration.
- The layer reduction fails once smoke fills the room.

The same will hold for the port.

---

## 5. Question 3: smoke through HVAC ducts

### 5.1 How FDS and CFAST model ducts

**FDS (Floyd; Technical Reference "HVAC" chapter, read by a helper; Floyd 2011, Fire Safety Science 10).**
- **Network.** A network of zero-volume nodes and ducts, after MELCOR:
  - node mass and energy balances, `sum rho_j u_j A_j = 0` and `sum rho_j u_j A_j h_j = 0`;
  - duct momentum, `rho_j L_j du_j/dt = (p_i - p_k) + (rho g dz)_j + dp_fan - 0.5 K_j rho_j |u_j| u_j`, with `K = K_minor + fL/D`.
- **Coupling.** The network is coupled to each pressure zone's background pressure, `dp_m/dt = (∫D dV - ∫u·dS)/∫P dV`, with `P = 1/p_m - 1/(rho c_p T)`.
- **Fans:**
  - constant volume flow;
  - a quadratic curve, `V = V_max sign(dp_max - dp) sqrt(|dp - dp_max|/dp_max)`, which reverses above the stall pressure `dp_max`;
  - user curves with a density correction;
  - spin-up time, and `LOSS = 1` when off.
- **Dampers** are binary, with a loss-ramp option for leaky dampers.
- **Filters** remove species by efficiency.
- **No transport delay by default.** An optional 1-D upwind duct advection (`N_CELLS`) adds one.
- **Leakage between zones** is a duct, `V = C_d A_L sign(dp) sqrt(2|dp|/rho)`, with the reference-pressure scaling `A_L = A_L,ref (|dp|/dp_ref)^(n - 0.5)`. In the guide's words it "cannot account for effects like exterior wind or the stack effect" because it uses background pressure only.
- **Floyd 2011:**
  - Duct pressure drops in the ASHRAE example were within 0 to 1.8 %.
  - Against the NRL 23-compartment ship facility, duct velocities were well predicted.
  - Smoke arrived too early in one test **because there was no transport delay**.

**CFAST (Technical Reference and User's Guide, read by a helper).**
- CFAST "does not include duct work or fan curves".
- Fans give a constant volume flow, falling from full at 200 Pa to zero at 300 Pa (defaults, tanh transition), with no reverse flow.
- Filters act on soot and trace species. Leakage is a wall vent to outside.
- Its Validation Guide over-predicts the pressure in NIST/NRC Test 16 (2.3 MW, ventilation on) because the fan flow stays fixed until the cutoff.
- The older Klote and Cooper (1989) fan-resistance model, FANRES, had backflow regions on the fan curve and no validation.

**Fans running or off.**
- **Running:** forced flow dominates and spreads smoke system-wide within the duct transit time.
- **Off:** smoke still enters ducts by fire pressure, buoyancy and stack effect. Klote's NISTIR 4551 is explicit that shutdown does not stop movement through ducts and shafts; smoke dampers do.
- **Codes** (secondary sources only; the standards were not read):
  - NFPA 90A requires a smoke detector in the supply duct of systems over 2000 cfm (0.94 m³/s), and in the return of systems over 15 000 cfm serving more than one storey; detection stops the fans.
  - UL 555 fire dampers close on heat (1.5 or 3 h ratings). Dynamic dampers close against 10 m/s and about 1 kPa.
  - UL 555S smoke dampers leak at most about 41, 102 or 406 L/(s m²) at 1 kPa (Classes I to III).

### 5.2 Measured data on duct smoke spread

- **PRISME SOURCE.** A helper analysed the CSVs in `firemodels/exp/PRISME`; the French column names were interpreted by the helper, so treat this as tentative.
  - The inlet duct reversed in 7 of 8 tests, reaching -0.05 to -0.21 m³/s against +0.03 to +0.27 m³/s before the fire.
  - The reversal came with the ignition over-pressure of +270 to +3050 Pa.
  - Under-pressure later reached -310 to -2760 Pa.
  - Prétrel, Le Saux and Audouin (2012) correlate the over-pressure peak (abstract only).
  - Fire-induced pressure drives smoke backward into **supply** ducts of confined rooms. That direction cannot arise in the port (G1).
- **UL FSRI HVAC house** (Weinschenk et al. 2023, data paper read in part; Ghanekar et al. 2022, fragment).
  - HVAC on against off, doors open against closed, at 110 to 360 kW, with gases and temperatures across a house.
  - Ghanekar et al. report that HVAC status mainly changed the rate of change of H2O.
  - Chaudhari, Weinschenk and Floyd (2023, abstract) ran FDS on these tests. Through ducts it under-predicted the temperature rise by about 9 % and over-predicted O2 by 21 %. **The closed-room result improved after duct heat loss to ambient was added.**
- **NIST/NRC** Tests 4, 5, 10 and 16 (supply and exhaust, door open or closed): pressure and HGL with ventilation on.
- **NRL ship facility** (Floyd et al. 2005; no DOI found): duct velocities.
- **NIST Plaza Hotel** (Klote 1990, via the CFAST guide): pressurised stairs and exhaust at up to 3 MW.

### 5.3 What the port's duct option is, and how to check it

**What it is:**
- an instantaneous, volume-less swap among **all** IFC air terminals as one system;
- 0.1 m³/s a terminal (an assumption);
- no supply and return distinction, no fan state, no shutdown on detection, no delay, no duct heat loss or soot deposition, and no pressure;
- it conserves species and heat exactly (`DuctMixing.lean`, `ducts_conserve`, `ducts_after_mem`).

**What it is not.** "The furthest ducts could spread smoke" is right for **reach**, but not for **concentration**. A real system with one return in the fire room serving a few supply grilles concentrates smoke in those rooms. The port dilutes the same smoke over every terminal in the building, so the README's claim should say "reach, not concentration".

**Checks:**
- **D1 (verification, exact).** Two sealed rooms of volumes `V_1` and `V_2` joined only by one terminal each, with no fire and soot initially in room 1 only. For well-mixed rooms the difference decays as `exp(-Q (1/V_1 + 1/V_2) t)` and the total is conserved.
  - The port's rooms are not perfectly mixed, so use a forced mixing step or tiny rooms, or compare the terminal cells.
  - Accept: total soot conserved to float32 round-off; decay constant within 10 % for rooms of a few cells.
- **D2 (sensitivity).** A run on one model with the system split by IFC system (once the import reads `IfcDistributionSystem`), against one shared system. Report the peak soot concentration at the most-affected remote node. This quantifies the dilution caveat above.
- **D3 (validation, qualitative).** The UL FSRI HVAC house, HVAC on and off, doors closed: the sign and order of magnitude of the O2 depression and CO2 rise in a remote bedroom.
  - Accept: same sign, and within a factor of 2 of the measured change.
  - Rebuilding the house as an IFC is the cost.
- **D4 (known-limit test).** PRISME SOURCE, the backflow in the supply duct. The port should fail this until G1 and zone pressure exist. Keep it as a regression target.

---

## 6. Question 4: closed and open buildings

### 6.1 What the literature supports

**Ventilation-controlled burning.**
- Air inflow through a single opening in a fully involved room is about `0.5 A sqrt(H)` kg/s (FDS Validation Guide: `m_O2 = 0.5 × 0.23 A_0 sqrt(H_0)`), hence a maximum of about `1500 A sqrt(H)` kW at 13.1 MJ per kg of O2.
- Kawagoe's (1958, BRI Report 27, not read) wood burning rate of about `5.5 A sqrt(H)` kg/min was reached only through secondary sources (Gross and Robertson quote about 6).
- The repository already uses `1500 A_o sqrt(H_o)` (zone.js F1) and Babrauskas's flashover estimate (FD-03, ±40 % scatter). The global equivalence ratio is `phi = Q/(13 100 m_O2)`.
- **CO under-ventilated:**
  - Gottuk's compartment data (dissertation, read in part) show the CO yield rising to about 0.20 to 0.22 g/g, levelling off for phi above 1.5 at upper-layer temperatures above 900 K.
  - Pitts (NIST Monograph 179, read in part) warns that the GER correlations are fuel- and temperature-dependent and valid only under limited conditions.
  - A Tewarson-type yield curve was recalled from memory by a helper and not verified, so it is not given here.

**Backdraft.**
- Fleischmann, Pagni and Williamson (1994, read by a helper): a 1.2 × 1.2 × 2.4 m compartment with methane at 70 kW. Backdrafts needed an unburnt hydrocarbon mass fraction **above about 10 %**, and their over-pressure grew with it.
- Weng and Fan (2003 abstract; 2005 read): critical fractions of **7.0 to 9.8 %** depending on the opening. They are lower for high or ceiling openings (slot ceiling 7.0 %, window ceiling 7.7 %).
- The inflow is a gravity current. Fleischmann and McGrattan (1999) matched its transit time numerically.
- The deflagration that follows is beyond an incompressible solver. The **condition** is a cheap and defensible flag.

**Stack effect.**
- Klote (NISTIR 4551, read by a helper): `dP_so = K_s (1/T_o - 1/T_s) h`, with `K_s = 3460 Pa K/m`. I checked it as `g p_atm/R_air` = 3463.
  - Pressure inside a warm shaft exceeds outside above the neutral plane (normal stack effect). The sign reverses when the outside is warmer.
- Neutral plane of a shaft with a top and a bottom opening (Klote, NISTIR 4588; rebuilt by the helper and checked against the report's worked numbers; I re-derived it from orifice mass balance):
  - `h_b = H/(1 + (T_s/T_o)(A_b/A_t)²)` above the bottom opening;
  - `H/(1 + (T_s/T_o)^(1/3))` for a continuous opening.
- Tamura and Wilson's building measurements and the Tamura and Klote fire-tower tests support the theory (via NISTIR 4551, not read). Klote saw reverse stack effect in tight buildings with exterior stairs.
- My numbers:

  | Case | Pressure difference |
  |---|---|
  | -18 °C outside, 22 °C inside, 30 m from the neutral plane | 55 Pa |
  | Riyadh summer, 45 °C out and 24 °C in | -0.77 Pa/m (-23 Pa at 30 m, reversed) |
  | A stair filled with 100 °C smoke | +2.5 Pa/m |
  | Equal openings (22 / -18 °C) | neutral plane at 46 % of the shaft height |
  | Bottom opening twice the top | 18 % |
  | Top opening twice the bottom | 78 % |

- NIST's CONTAM is the standard multizone network for such flows and for smoke control design (TN 1887r1, read in part).

**Wind.**
- `P_w = 0.5 C_w rho V²`. C_w is about +0.7 to +0.8 on windward walls and -0.25 to -0.8 on leeward walls (Klote, NISTIR 4551, after MacDonald).
- Power-law profile `V = V_0 (z/z_0)^n`, with n about 0.16 open and 0.40 urban.
- 10 m/s gives 48 Pa windward at C_w = 0.8, comparable with stack and fire pressures.
- Measured: the NIST seven-storey wind-driven tests (Kerber and Madrzykowski 2009, TN 1629, read in part): 9 to 11 m/s imposed wind, corridor and stair temperatures above 400 °C, and velocities about 10 m/s. The laboratory series (TN 1618, via TN 1629) reached 600 °C in the corridor within 30 s of window failure.
- The Flamescapes paper itself carries a scene wind velocity (its Table 5, -15 to 15 m/s) that the port dropped.

### 6.2 What to add, and in what form

| Effect | Reliable enough? | Form | Cost when off | Cost when on |
|---|---|---|---|---|
| Thermal expansion (G1) | yes: FDS's leading divergence term | `div u = D` with `D = q_gas/(rho c_p T V)` per cell, `q_gas` the heat the step put into the gas (chemistry, radiation absorbed, wall exchange); in sealed regions subtract the region mean (a uniform pressure rise) or balance it with leakage (next row) | zero (branch skipped) | one array and one subtraction in the divergence; the heat per cell is already computed in `chemistry` and `conduct` |
| Zone pressure and door/wall leakage | yes: FDS and CFAST practice | per sealed region a background pressure `p_m` and per closed door an orifice leak `V = C_d A_L sign(dp) sqrt(2|dp|/rho)` (`A_L` about 0.01 to 0.03 m² per door is a typical value to confirm against Klote's tables) | zero | a connected-components pass once, then a scalar ODE per region per step |
| Indoor/outdoor temperatures (stack effect in the grid) | yes | initial indoor air at `T_i`, boundary resets at `T_o`, `rho0 = rho(T_o)` | zero when equal | none |
| Building-scale stack effect | yes, as a network | storeys × shafts (stairs, lifts, risers), Klote's formulas with leakage defaults; fed by the grid's or the zone model's fire floor; gives shaft temperatures and pressures and which floors receive smoke | zero | microseconds per step |
| Wind | yes, as boundary pressure | per open side a pressure `0.5 C_w rho V²` (windward +0.8, leeward -0.3, sides -0.5, from the cited ranges) | zero | none |
| Duct improvements | yes | fan state (running, off, shut down on a detector reading), supply/return roles per terminal, a FIFO transport delay per system, duct heat loss as exact relaxation to ambient | zero (as today) | per terminal per step |
| Backdraft flag | yes, as a warning | room upper-layer mean `Y_FUEL ≥ 0.07` (warn) or `≥ 0.10` (high) at the moment a door is opened or a window fails | zero | a per-room mean when doors change |
| CO yield against phi | not yet | wait for a verified published relation | | |

---

## 7. Question 5: the plan

### 7.1 Prerequisites, cheap and test-only

- **P1. Prescribed-HRR burner.** A source that adds a set heat release and the burner fuel's products in chosen cells, with no solid fuel. Every validation series uses gas or spray burners. The sprinkler report needs the same thing.
- **P2. Diagnostics.** Run once per frame, so negligible:
  - per IfcSpace, the integral layer height and `T_u`, `T_l` (Section 3.2);
  - the plume mass flux through horizontal planes;
  - the energy budget per step: heat released; convected out of open sides; to walls by `conduct`; to walls by radiation; the **expansion volume and enthalpy** removed by the density update (G1, Section 2.2); storage;
  - `D*/dx` on the run card.
- **P3.** A box-geometry builder for tests (rooms, doors, corridors as meshes), and 1 s frames for checks.

### 7.2 Checks to run against the existing solver

The acceptance bands are my proposals, not published criteria. "Pass" means within roughly the 2σ_M of the corresponding FDS or correlation statistic, widened for the port's resolution. "Late" errors that make smoke arrive later than measured fail at a tighter band, because they are the unsafe direction for egress.

| ID | Case | Quantity compared | Reference | Pass | Amber | Cost |
|---|---|---|---|---|---|---|
| C0 | Open-air plume, 100, 300 and 1000 kW burner, no ceiling | plume mass flux at 1, 2 and 3 m; centreline ΔT trend | Heskestad FD-08, Zukoski FD-30; McCaffrey FD-11/12 | mass flux within ±40 % | ±40 to 80 % | minutes |
| C1 | **Closed box**, 10 × 8 × 3 m, 300 kW burner, then NIST/NRC Test 1 (410 kW, door closed) | energy budget closure; integral layer height and `T_u` against time | budget to round-off; Yamana and Tanaka (`k = 0.076/rho_g`), ASET, Beyler; NIST/NRC measured layer | budget within 1 %; layer at 1.8 m not later than YT by more than 20 %; `T_u` rise within ±30 % of Beyler | layer 20 to 50 % late; `T_u` ±30 to 50 % | an hour (NIST/NRC at 0.35 m is about 14 k cells) |
| C2 | Open fire room, one door, 100 to 500 kW | `T_u` rise against time | MQH with stone and gypsum `h_k`; zone.js on the same room with the burner's HRR | within ±30 % of MQH (stone) at 60 to 300 s | ±30 to 50 % | minutes |
| C3 | **Steckler**, 62.9 kW centre burner, door widths 0.24 to 0.99 m | neutral-plane height; peak outflow velocity and temperature; mass in and out | measured profiles | neutral plane within one cell; velocity within ±30 %; mass out/in ratio within ±20 % of measured | velocity ±30 to 50 % | minutes |
| C4 | **NBS multi-room 100A/100O/100Z** (110 kW), then **ATF corridors** 50 to 500 kW | corridor layer depth and `T_u`; arrival time of a 10 % rise at each tree; ceiling ΔT decay along the corridor | measured; Delichatsios decay; Heskestad front speed | arrival not later than measured by more than 20 % or one frame; `T_u` ±30 % | arrival 20 to 50 % late | an hour |
| C5 | Unconfined and confined ceiling jet | top-cell ΔT against r/H | Alpert (FD-13), CFAST hybrid (FD-15), Vettori flat ceiling | as in the sprinkler report, Section 4 | | (shared with the sprinkler work) |
| C6 | LLNL enclosure (forced exhaust), at the port's own resolution | `T_u`; exhaust temperature | measured | ±25 % | ±25 to 50 % | an hour |
| C7 | Ducts D1 to D4 | Section 5.3 | exact; UL FSRI; PRISME | D1 to round-off and 10 %; D3 sign and factor 2 | | D1 minutes; D3 days |
| C8 | Clip-boundary effect | layer height in a room cut by the clip box, against the same room run unclipped | the port itself | difference under one cell | | minutes |
| C9 | CPU/GPU parity for C1 to C4 | all of the above | the CPU run | within float32 tolerance as the existing parity tests | | minutes |

**Run first:**
1. **C1.** Closed box, then NIST/NRC Test 1. The energy budget says at once how big G1 and G2 are, and the layer comparison says whether they matter for egress.
2. **C3.** Steckler. Door flows are how smoke leaves the fire room, and the in/out mass ratio exposes G1 in an open room.
3. **C4.** NBS multi-room and ATF corridors. Corridor arrival is the quantity that most directly drives the routes.

### 7.3 Candidate improvements, by value per cost

Each sits behind an option that is off by default and skips its code path, so the default run costs nothing extra.

1. **Prescribed burner, diagnostics, `D*/dx` on the card (P1 to P3).** High value, low cost, no change to results.
2. **A separate gas-to-inert-wall coefficient** (e.g. 10 to 25 W/(m² K)), keeping 200 for fuel surfaces.
   - Zero runtime cost: a second uniform. But the default changes, so only after C1 and C2 show G2.
   - A better step is a two-node or exact semi-infinite surface temperature for walls (MQH-like `h_k(t)`), at a few flops per wall face.
3. **Thermal expansion (G1),** first for regions open to the domain boundary. Moderate value, low cost: one array. GPU: one more field through the existing fixed-point `atom` pattern, applied in the `divergence` kernel.
4. **Zone pressure with door leakage** for sealed regions. High value for "closed building" scenarios and for duct backflow, moderate cost: connected components once, a scalar ODE per region.
5. **Duct realism.** Low cost:
   - fan states including shutdown when the return terminal's soot or temperature passes a detector threshold;
   - supply/return roles;
   - a transport delay (FIFO of system-mean states, length `L/v`, with L and v defaults when IFC lacks them);
   - duct heat loss.
6. **Stack effect and wind** inside the grid (indoor/outdoor temperatures, side pressures). Zero cost when off; a small change to `boundaries` and `project`.
7. **Building-scale shaft network** (CONTAM-like, Klote's formulas). Moderate cost, mostly import work: shafts and leakage areas from IFC. Negligible runtime. It extends `rooms.js`/`zone.js` to the storeys the clipped grid does not hold.
8. **Backdraft warning.** Low cost and low risk.
9. **Closing clip faces** (treat them as walls, or as zero-gradient outflow) for closed-building runs. Low cost, and only after C8 shows the effect.

### 7.4 Lean-provable properties

These are new; `Zone.lean` already proves `layer_antitone` (monotone descent in the two-zone model) and `DuctMixing.lean` the duct sums. Place them in `FireEgress/SmokeChecks.lean` (L1 to L3) and `FireEgress/Pressure.lean` (L4 to L6). All use real arithmetic; float32 is not modelled.

1. **L1 `layerHeight_nonneg_den`, `layerHeight_num_nonneg`, `layerHeight_step_exact`: the Janssens and Tran reduction.**
   - Statements, for cell temperatures `T_i > 0` with lowest `T_l`, sums over cells of height `dz`:
     - `sum (T_i - T_l)²/T_i dz = I_1 + I_2 T_l² - 2 T_l H ≥ 0`;
     - `I_1 I_2 ≥ H²` (Cauchy-Schwarz);
     - for a two-layer step profile (`T_u` above a cell boundary `z_i`, `T_l` below, `T_u ≠ T_l`), `z_int = z_i` and the upper mean equals `T_u` exactly.
   - Kind: algebraic, with Mathlib's `inner_mul_le_norm_mul_norm` or `Finset.sum_mul_sq_le_sq_mul_sq`.
   - Use: the diagnostic in P2 is exact where it should be.
2. **L2 `yamanaTanaka_solves`, `_antitone`, `_anti_k`: closed-room filling.**
   - Statement: `z(t) = (2kQ^(1/3) t/(3A) + H^(-2/3))^(-3/2)` satisfies `A z' = -k Q^(1/3) z^(5/3)` with `z(0) = H`, is strictly decreasing, and stays in `(0, H]`. It is antitone in `k`.
   - Corollary: since `k_a = 0.076/rho_a ≤ 0.076/rho_g = k_g` for `T_g ≥ T_a`, the expansion-free layer is never lower, `z_a(t) ≥ z_g(t)`. That is the direction of G1's bias, stated exactly in the zone model.
   - Kind: Mathlib analysis (`HasDerivAt` for `rpow`), as `survives_unique` does; monotonicity is algebraic.
3. **L3 `expansion_volume` and `vanished_mass`: what the incompressible step loses.**
   - Statement: at fixed pressure and composition, heating a cell's gas by `Q = rho V c_p (T' - T)` makes the mass `rho V (1 - T/T')` disappear in the density update. That mass occupies `V (T' - T)/T = Q/(rho c_p T)` at `T'`, which is FDS's leading divergence term times V.
   - Summed over cells with constant `c_p`, the missing volume flux is `sum q_c/(rho_c c_p T_c)`.
   - Kind: algebraic (`field_simp`).
   - Use: the P2 budget's expansion term is right by construction.
4. **L4 `zonePressure_exists_unique` and `leak_sign`: sealed regions with leakage.**
   - Statement: for a leak law `F(p) = sum_d C_d A_d sign(p) sqrt(2|p|/rho)` (strictly increasing, continuous, `F(0) = 0`, unbounded), for every expansion rate `E` there is exactly one `p` with `F(p) = E`. It has the sign of `E`, and it is monotone in `E` and antitone in each leak area.
   - Kind: intermediate value theorem (Mathlib) for existence, strict monotonicity for uniqueness.
   - Use: the quasi-steady zone pressure is well defined, and opening a gap never raises it.
5. **L5 `stack_sign`, `Ks_eq`, `neutralPlane_mem`, `neutralPlane_anti`: stack effect.**
   - Statements:
     - `K_s = g p_atm/R_air`;
     - `dP(h) = K_s (1/T_o - 1/T_i) h` is positive exactly when `h > 0` and `T_i > T_o` (or `h < 0` and `T_i < T_o`);
     - mass balance of two orifices, `A_b² rho_o h_b = A_t² rho_s h_t` with `h_b + h_t = H`, has the unique solution `h_b = H/(1 + (T_s/T_o)(A_b/A_t)²)` in `(0, H)`;
     - it is strictly decreasing in `A_b/A_t` and in `T_s/T_o`;
     - swapping `T_s` and `T_o` gives the reverse stack effect.
   - Kind: algebraic.
6. **L6 `ductDelay_conserve`, `ductLoss_mem`, `fanCurve_reverse_iff`: ducts with delay, loss and a fan curve.**
   - Statements:
     - with a FIFO of `K` stored system states, the sum over rooms plus store of `m x` is unchanged by a step (extending `ducts_conserve`);
     - with exact relaxation to ambient at rate `λ ≥ 0`, delivered temperatures stay in `[min(T_amb, min_t T_t), max(T_amb, max_t T_t)]` and total heat does not increase;
     - for FDS's quadratic fan, `V < 0` exactly when `dp > dp_max`, and `V` is antitone in `dp`.
   - Kind: algebraic (`Finset.sum_range_succ`, `Real.sqrt` monotonicity).
   - Use: the D4 backflow regression is a theorem about the fan, not the grid.

Prove L1 and L3 before P2; L2 before C1; L4 before improvement 4; L5 before 6 and 7; L6 before 5.

---

## 8. Claims I am least sure of

- **The size of G1 and G2 in the port.**
  - The direction and the closed-form ratios are solid.
  - How much numerical diffusion offsets G1, and whether G2 cools the layer by a few per cent or by half, is inference from the code. Only C1 and C2 can tell.
  - My FDS wall-convection estimate (5 to 10 W/(m² K)) was computed from the guide's formulas with assumed properties, not taken from an FDS run.
- **The acceptance bands** are my proposals, scaled from FDS and SP 1169 statistics and Friday and Mowrer's coarse-grid study. That study used FDS of about 2000, which already had a low-Mach divergence; the port does not.
- **The "4 to 16" `D*/dx` range** and anything from NUREG-1824 and its Supplement 1: not read (HTTP 403).
- **The Yamana and Tanaka constant.** My `k = 0.076/rho_g` rests on back-computing SP 1169's verification table. The printed form seen in the extracted text is inverted. Part 1 of the 1985 paper was not read.
- **NFPA 92B's two equations** come from Klote's 1994 OCR text. The current NFPA 92 may differ in constants or ranges.
- **PRISME backflow numbers** depend on a helper's reading of French column names in 10 s data.
- **Ducts in practice.** The NFPA 90A thresholds and UL 555S leakage classes come from secondary sources. The door-gap leakage areas (0.01 to 0.03 m² per door) are typical values I have not checked against Klote's tables.
- **Backdraft thresholds** come from methane in small boxes (1.2 m scale). Their transfer to room-scale contents fires is unproven.
- **Data reached only as summaries.** Hu et al. 2005 (corridor decay) was seen only in search summaries, and the Hinkley exponents came from a garbled OCR of Steckler (1989). For the corridor checks, use the ATF and NBS data, not these formulas, as the reference.
- **The domain size of the VTT case** on the GPU (about 125 k cells) is an estimate.

---

## 9. Sources and DOI checks

Every DOI below was checked at `api.crossref.org/works/<doi>` on 2026-11-07. Title, authors and year are as Crossref returned them. "Helper" means read by an agent I briefed, which reported the read status.

| DOI / identifier | Crossref title; authors; year | Read |
|---|---|---|
| 10.1145/3842518 | Flamescapes (Wrede et al. 2026): **not found at Crossref (404)**, as before | repository transcription |
| 10.1145/3763338 | Fire-X: Extinguishing Fire with Stoichiometric Heat Release; Wrede, Wagner, Mahfuz, Palubicki, Michels, Pirk; 2025 | repository transcription |
| 10.6028/NIST.SP.1169 | Verification and validation of commonly used empirical correlations for fire scenarios; Overholt; 2014 | **read in part by me** (Chapters 2, 3, 7, 10) |
| 10.6028/NIST.IR.5516 | Method of predicting smoke movement in atria with application to smoke management; Crossref lists NIST as author (the report is Klote's); 1994 | **read in part by me** (Section 5, OCR) |
| 10.6028/NIST.SP.1018; 10.6028/NIST.SP.1019 | Fire dynamics simulator (version 4); McGrattan (2006); McGrattan, Forney (2004). Crossref returns version 4 records, not the current editions | current Technical Reference and User's Guide **read in part by me** from LaTeX (master, 2026-11-07); HVAC chapter by a helper |
| FDS Validation Guide (no current DOI) | | numerical-parameters table and statistics CSV **read by me**; experiment descriptions by a helper |
| CFAST Technical Reference and Validation Guides (no current DOI) | | helpers (LaTeX, master) |
| 10.1016/0379-7112(84)90005-5 | Engineering relations for fire plumes; Heskestad; 1984 | not read; equations from FD-06 to FD-10 |
| 10.6028/NBS.IR.79-1910 | Purely buoyant diffusion flames; McCaffrey; 1979 | not read by me (FD-11, FD-12) |
| 10.1016/0379-7112(81)90037-0 | Entrainment in fire plumes; Zukoski, Kubota, Cetegen; 1981 | not read (FD-30; via SP 1169) |
| 10.1007/BF02590543 | Calculation of response time of ceiling-mounted fire detectors; Alpert; 1972 | not read (via SP 1169, FD-13) |
| 10.3801/IAFSS.FSS.3-301 | Characterizing The Unconfined Ceiling Jet Under Steady-state Conditions: A Reassessment; Motevalli, Marks; 1991 | not read (via CFAST guide) |
| 10.1007/s10694-023-01461-0 | Revisiting Alpert's Correlations: Numerical Exploration of Early-Stage Building Fire and Detection; Zeng, Wong, Węgrzyński, Huang; 2023 | via the sprinkler report (pp. 1 to 10) |
| 10.3210/fst.5.31 | Smoke control in large scale spaces. Part 1: Analytic theories for simple smoke control problems; Tanaka, Yamana; 1985 | not read (formula via SP 1169) |
| 10.3210/fst.5.41 | Smoke Control in Large Scale Spaces [Part 2]; Yamana, Tanaka; 1985 | pp. 41 to 45 (helper) |
| 10.1016/s0082-0784(82)80267-1 | Flow induced by fire in a compartment; Steckler, Quintiere, Rinkinen; 1982 | via FDS Validation Guide |
| 10.6028/nbs.ir.82-2520 | Flow induced by fire in a compartment; Steckler, Quintiere, Rinkinen; 1982 | via FDS Validation Guide |
| 10.6028/jres.096.022 | Data for room fire model comparisons; Peacock, Davis, Babrauskas; 1991 | part (helper) |
| 10.6028/nbs.ir.88-3752 | An experimental data set for the accuracy assessment of room fire models; Peacock, Davis, Lee; 1988 | via validation guides |
| 10.6028/NIST.SP.1013-1 | Report of experimental results for the international fire model benchmarking and validation exercise #3; Hamins, Maranghides, Johnsson, Donnelly, Yang, Mulholland, Anleitner; 2006 | via validation guides (whether it is the same report as NUREG/CR-6905 is not confirmed) |
| 10.1177/073490419201000604 | Data Reduction of Room Tests for Zone Model Validation; Janssens, Tran; 1992 | not read (equations via FDS guides) |
| 10.1016/s0379-7112(97)00064-7 | Determination of interface height from measured parameter profile in enclosure fire experiment; He, Fernando, Luo; 1998 | not read |
| 10.1115/1.3245194 | An Experimental Study of Upper Hot Layer Stratification in Full-Scale Multiroom Fire Scenarios; Cooper, Harkleroad, Quintiere, Rinkinen; 1982 | abstract (helper) |
| 10.1177/1042391503013001003 | ASET-B: Comparison of Model Predictions with Full-Scale Test Data; Hurley; 2003 | full (helper) |
| 10.6028/NIST.IR.6470 | Comparison of FPETool: ...; Vettori, Madrzykowski; 2000 | not read (cited by Hurley for the N-percent rule) |
| 10.1016/j.buildenv.2008.12.010 | Experimental data and numerical modelling of 1.3 and 2.3MW fires in a 20m cubic atrium; Gutiérrez-Montes, Sanmiguel-Rojas, Viedma, Rein; 2009 | via FDS Validation Guide survey |
| 10.1007/s10694-012-0261-1 | Heat Fluxes and Flame Lengths from Fires Under Ceilings; Lattimer, Mealy, Beitel; 2012 | via FDS Validation Guide |
| 10.1007/s10694-007-0007-7 | Using FDS to Simulate Smoke Layer Interface Height in a Simple Atrium; Kerber, Milke; 2007 | metadata only |
| 10.1016/j.firesaf.2004.09.005 | Development of a hybrid field and zone model for fire smoke propagation simulation in buildings; Hua, Wang, Kumar; 2005 | metadata only |
| 10.1177/1042391509360272 | Analysis of Prediction Capability of FDS for Response of Thermal Detectors; Hurley, Munguia; 2010 | abstract (helper) |
| 10.6028/NIST.GCR.09-921 | Analysis of FDS thermal detector response prediction capability; Hurley, Munguia; 2009 | full (helper) |
| NIST GCR 01-810 (DOI 10.6028/NIST.GCR.01-810 **not found at Crossref, 404**) | Friday, Mowrer 2001, FDS comparison with FM/SNL tests | full (helper, from tsapps.nist.gov) |
| 10.1016/s0379-7112(02)00082-6 | Numerical simulation of axi-symmetric fire plumes: accuracy and limitations; Ma, Quintiere; 2003 | via FDS Validation Guide survey (abstract withheld) |
| 10.1088/0026-1394/48/3/011 | Quantifying the predictive uncertainty of complex numerical models; McGrattan, Toman; 2011 | not read (the statistics method behind δ and σ_M) |
| 10.6028/NIST.IR.89-4050 | Fire induced flows in corridors: a review of efforts to model key features; Steckler; 1989 | full scan, partly garbled (helper) |
| 10.1016/j.tust.2004.08.007 | Full-scale burning tests on studying smoke temperature and velocity along a corridor; Hu, Huo, Li, Wang, Chow; 2005 | search summaries only |
| 10.1177/10423910260620473 | Development and Validation of Corridor Flow Submodel for CFAST; Bailey, Forney, Tatem, Jones; 2002 | abstract |
| 10.1016/0010-2180(81)90002-x | The flow of fire gases under a beamed ceiling; Delichatsios; 1981 | not read (correlation via CFAST Technical Reference) |
| 10.3801/IAFSS.FSS.10-459 | Coupling a Network HVAC Model to a Computational Fluid Dynamics Model Using Large Eddy Simulation; Floyd; 2011 | full (helper) |
| 10.1016/j.buildenv.2020.107321 | Operation point fan model: a new evaluated approach for atypical pressure conditions in buildings; Ralph, Mofidi; 2020 | not read (cited by FDS guide) |
| 10.1080/19401493.2019.1608304 | Coupled hybrid modelling within the Fire Dynamics Simulator: transient transport and mass storage; Ralph, Carvel, Floyd; 2019 | not read |
| 10.1016/j.firesaf.2018.08.008 | Coupled hybrid modelling in fire safety engineering; a literature review; Ralph, Carvel; 2018 | not read |
| 10.1007/s10694-023-01390-y | Numerical Simulations of Gas Burner Experiments in a Residential Structure with HVAC System; Chaudhari, Weinschenk, Floyd; 2023 | abstract (helper) |
| 10.1016/j.dib.2022.108825 | Experimental data from gas burner fires in residential structure with HVAC system; Weinschenk, Ghanekar, Stakes, Quiat, Kessler, Lee; 2023 | part (helper) |
| 10.1016/j.firesaf.2022.103534 | Effects of HVAC on combustion-gas transport in residential structures; Ghanekar, Weinschenk, Horn, Stakes, Kesler, Lee; 2022 | fragment (secondary) |
| 10.1016/j.firesaf.2012.04.005 | Pressure variations induced by a pool fire in a well-confined and force-ventilated compartment; Prétrel, Le Saux, Audouin; 2012 | secondary |
| 10.1016/j.firesaf.2013.07.008 | OECD PRISME project: Fires in confined and ventilated nuclear-type multi-compartments, overview and main experimental results; Audouin, Rigollet, Prétrel, Le Saux, Röwekamp; 2013 | not read (descriptions via validation guides; CSV data analysed by a helper) |
| 10.6028/NIST.IR.89-4141 | Model of a simple fan-resistance ventilation system and its application to fire modeling; Klote, Cooper; 1989 | full, garbled OCR (helper) |
| 10.6028/NIST.IR.4551 | Design manual for smoke control systems; Klote; 1991 | full (helper) |
| 10.6028/NIST.IR.4588 | A general routine analysis of stack effect; Klote; 1991 | full, garbled OCR (helper); neutral-plane formula re-derived by me |
| 10.3801/IAFSS.FSS.2-761 | Experimental Fire Tower Studies On Mechanical Pressurization On Control Smoke Movement Caused By Fire Pressures; Tamura, Klote; 1989 | not read |
| 10.1520/STP23527S | Experimental Fire Tower Studies on Controlling Smoke Movement Caused by Stack and Wind Action; Tamura, Klote; 1990 | not read |
| 10.1007/s10694-009-0132-6 | An Experimental Study of Smoke Movement in Multi-Storey Buildings; Wang, Zalok, Hadjisophocleous; 2010 | secondary |
| 10.6028/NIST.TN.1887r1 | CONTAM User Guide and Program Documentation Version 3.4; Dols, Polidoro; 2020 | part (helper) |
| 10.6028/NIST.TN.1618 | Fire fighting tactics under wind driven conditions: ...; Madrzykowski, Kerber; 2009 | via TN 1629 |
| 10.6028/NIST.TN.1629 | Fire fighting tactics under wind driven conditions: ...; Kerber, Madrzykowski; 2009 | part (helper) |
| 10.3801/IAFSS.FSS.4-337 | Quantitative Backdraft Experiments; Fleischmann, Pagni, Williamson; 1994 | full (helper) |
| 10.1007/BF01052526 | Exploratory backdraft experiments; Fleischmann, Pagni, Williamson; 1993 | not read |
| 10.1016/S0379-7112(98)00046-0 | Numerical and experimental gravity currents related to backdrafts; Fleischmann, McGrattan; 1999 | NIST record (secondary) |
| 10.1016/S0950-4230(02)00088-8 | Critical condition of backdraft in compartment fires: a reduced-scale experimental study; Weng, Fan; 2003 | abstract (helper) |
| 10.3801/IAFSS.FSS.8-1181 | Experimental Study Of Backdraft In A Compartment With Different Opening Geometries And Its Mitigation With Water Mist; Weng, Fan; 2005 | full (helper) |
| Gojkovic 2000, Lund report 3121 (no DOI) | Initial backdraft experiments | full (helper); no usable threshold |
| 10.6028/NIST.MONO.179 | The global equivalence ratio concept and the prediction of carbon monoxide formation in enclosure fires; Pitts; 1994 | part (helper) |
| 10.1016/0360-1285(95)00004-2 | The global equivalence ratio concept and the formation mechanisms of carbon monoxide in enclosure fires; Pitts; 1995 | not read |
| 10.1177/104239159200400402 | Carbon Monoxide Production in Compartment Fires; Gottuk, Roby, Peatross, Beyler; 1992 | not read (Gottuk dissertation read in part by a helper) |
| Kawagoe 1958, BRI Report 27 (no DOI) | Fire behaviour in rooms | not read (via Gross and Robertson, and the FDS Validation Guide) |
| NUREG-1824 (2007) and Supplement 1 (no DOI) | Verification and validation of selected fire models for nuclear power plant applications | **not reached** (nrc.gov returned 403; not bypassed) |
| Hostikka, Kokkala, Vaari 2001, VTT Research Notes 2104; Nowlen, NUREG/CR-4681; Sheppard and Klein 2009 (ATF); Klein-Heßling and Röwenkamp 2005; Riese and Hosser 2004; Heskestad and Hill 1986, NBS-GCR 86-502 (DOI 404 per a helper) | | via the FDS and CFAST Validation Guides |
| NFPA 92, NFPA 90A, UL 555, UL 555S | | not read; NFPA 92B equations via Klote 1994; the others via secondary sources |
