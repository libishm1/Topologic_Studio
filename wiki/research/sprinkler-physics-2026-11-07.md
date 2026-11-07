# Sprinklers and water suppression for the Flamescapes port: physics, a design, and what to prove first (2026-11-07)

Status: research and a design on paper. No code was written and the repository was not changed. The design is for `topologicpy-web-frontend/src/core/flamescapes*.js` (CPU), `flamescapesGpu.js` and `flamescapesGpu/kernels.js` (WebGPU), and the Lean project `formal/fireegress`.

How the sources were read is stated beside each one in Section 12. In short: the FDS Technical Reference, User's and Validation Guides were read from their LaTeX sources on GitHub (master, retrieved 2026-11-07), together with the FDS Fortran routine that updates the sprinkler link; the CFAST Technical Reference Guide likewise; Evans (1993) was read in full from the NIST scan; Zeng et al. (2023) was read in part (pages 1 to 10 of the accepted manuscript); Fire-X and Flamescapes were read from the repository's transcriptions. Heskestad and Bill (1988), Heskestad and Smith (1976), Alpert (1972), Evans and Stroup (1986), Madrzykowski and Vettori (1992), the UL/NFPRF, Vettori and Bittern reports, and Bullen (1977) were not read; what is said about them comes from the guides above or from abstracts, and is marked so. Text and code from these sources were treated as data.

---

## 1. Summary and recommendation

1. **Activation** is a per-sprinkler linear ODE (Heskestad and Bill 1988, as used by FDS, CFAST and DETACT):
   `dT_l/dt = (sqrt|u| (T_g - T_l) - C (T_l - T_m) - C2 beta |u| (T_l - T_w)) / RTI`.
   With the gas frozen over a step it has an exact solution, a relaxation toward a weighted mean of the gas, mount and water temperatures. That exact step fits the port's rule (every exchange is integrated exactly), needs no time-step bound and keeps the link between its sinks.
2. **The grid problem.** The ceiling jet under a 2.2 to 2.7 m ceiling is about 0.2 to 0.3 m thick (Motevalli and Marks, via CFAST) and the link sits 2 to 11 cm below the ceiling. A 0.29 to 0.4 m cell therefore averages the whole jet and some air beneath it. FDS reads the single cell containing the link, with no sub-grid model, and gives no cell-size rule for sprinklers beyond the general `D*/dx` measure; at the heat release rates where sprinklers open (0.3 to 1 MW) the port's `D*/dx` is only about 1.7 to 2.7. A mesh study in Zeng et al. (2023) shows 0.4 m cells reading roughly 10 to 25 % less excess temperature than 0.05 to 0.2 m cells, even under an 8.6 m ceiling. Under a t-squared fire, a 25 % low excess temperature and a 30 % low speed delay activation by about 20 to 25 % and raise the heat release rate at activation by 45 to 55 % (my calculation, Section 4.3). The recommended reading is CFAST's approach with the grid supplying the layer: the upper-layer temperature from the grid plus Alpert's ceiling-jet excess from the solver's own heat release and plume position, with the plain top-cell reading kept as a late bound.
3. **Suppression models**, from simple to detailed:
   - Evans (1993): `Q(t) = Q(t_act) exp(-(t - t_act)/tau)`, with `tau = 3.0 w''^-1.85` s and `w''` in mm/s. It is a conservative upper envelope for unshielded furnishings, intended for spray densities of 0.07 mm/s and above (`tau = 410 s` there).
   - FDS: Lagrangian droplets with Ranz-Marshall heat and mass transfer, plus an empirical burning-rate reduction on wetted fuel, `exp(-integral of a m_w'' dt)` (`E_COEFFICIENT`).
   - Fire-X: energy-limited droplet evaporation on the grid (Eqs. 26 to 31), with vapour mixed in at boiling point. Suppression comes only through the Arrhenius rate, from cooling and from oxygen dilution, with heat tied to the oxygen consumed.

   **Implement Evans first, as a per-cell cap on the solid fuel's reaction.** It is the only one of the three calibrated on full-scale sprinklered furnishings. It needs nothing the grid cannot resolve, its effect is bounded and monotone (provable in a few lines), it is what CFAST does (so results can be cross-checked), and it costs one float per cell on both backends.

   The Fire-X-style evaporation is the second step. It adds what Evans lacks and egress needs: gas cooling, steam that dilutes the oxygen, and the loss of buoyancy that brings smoke down. Wetting the fuel through the existing `moisture` field is attractive but unsafe as a first step: on this grid it would probably stop a covered cell within one or two steps, far faster than Evans's envelope (Section 5.5).
4. **Per sprinkler** the solver needs:
   - position (link point), orientation and distance below the ceiling;
   - activation temperature, RTI and C;
   - K-factor and pressure, or the flow;
   - coverage area and footprint, and design density;
   - spray half-angle, droplet median diameter and initial speed;
   - the room it serves.

   IFC's `Pset_FireSuppressionTerminalTypeSprinkler` gives activation temperature, a QUICK/STANDARD response class, coverage area, flow, residual pressure and a discharge coefficient, but **no RTI or C**, so defaults by class are needed. On the grid a head maps to:
   - a sensor cell (the top open cell of its column);
   - a static set of spray-zone cells (the columns under its footprint that it can see, layer by layer to the floor or the fuel);
   - the fuel cells it wets.
5. **Spray representation.** The spray crosses a 2.4 m room in about 0.3 to 0.6 s, which is 3 to 6 solver steps. So an Eulerian "spray zone" source term (steady footprint, layer-by-layer budget) captures what matters at this resolution, for well under a millisecond per step on either backend. Lagrangian parcels are a later option for visuals and footprint checks. On the GPU they should get their own buffer or a reserved quota: flame-front particles may fill 75 % of the shared buffer. Fire-X itself uses SPH particles (106k active in its sprinklered kitchen, 15 ms per step on an RTX 4090). SPH is unnecessary for a dilute spray, where the liquid volume fraction is about 1e-5.
6. **Validation data** exist for activation and for the suppressed heat release rate:
   - activation: Vettori 1998 and 2003 (residential sprinklers, t-squared methane burner), Bittern 2004 (chair fires, two sprinkler types), and UL/NFPRF 1998 (heptane spray burner, flowing sprinklers, number of activations);
   - heat release after activation: Madrzykowski and Vettori 1992 and Walton 1988 (the data behind Evans), and Madrzykowski 2008 (sprinklered Christmas-tree room: peak 1.8 MW against more than 6 MW unsprinklered).

   Fire-X's own sprinkler case is code-to-code against FDS with no measurements: 33 kW, 300 L/min switched on at 15 s.
7. **Lean**: 15 statements are listed in Section 9:
   - seven for activation (maximum principle, comparison, the closed form, monotone activation time, step composition, and the wet-link term);
   - eight for suppression (water budget, evaporation energy, species after adding vapour, oxygen dilution, evaporative cooling, an extinction criterion, Evans's envelope, and wet fuel held at 100 °C).

   Most are algebraic. The ODE uniqueness, the activation-time closed form and the cooling inequality need Mathlib's real analysis, in the pattern `survives_unique` already uses.

**Recommended first implementation step:**
- list the sprinklers (IFC or placed);
- step every link with the exact exponential update after each solver step, reading the gas two ways (the top-cell grid reading and the layer-plus-Alpert reading);
- report "head N opens at t = ... s" with both times, and change nothing in the fire.

Prove items 1 to 6 of Section 9 before writing it.

---

## 2. What the port already gives a suppression model

From the code (read 2026-11-07):

- **Fields per cell.**
  - Gas: temperature `T` (K), `rho` from the equation of state, and mass fractions `Y` of FUEL, O2, CO2, H2O, SOOT and CO, with nitrogen the rest (`SPECIES`, `NS = 6`).
  - Solid or porous cells: temperature `Ts`, fuel mass `fuelMass`, exposed area `area`, and the surface layer's water `moisture` (kg).
  - Velocities on a staggered grid (`u`, `v`, `w`).
  - Constants: liquid water's heat capacity 4186 J/(kg K) (in `solidCapacity`), vapour's 1864 J/(kg K) (`CP[H2O]`), and latent heat 2.259 MJ/kg (`latentHeat`, Flamescapes' Table 3).
- **Order of a step** (`STAGES`): pilot, particles, radiation, particleToGrid, advect, density, buoyancy, vorticity, diffuse, ducts, chemistry, cullFlameFront, conduct, density2, boundaries, project.
- **Exact integration.** Each reaction is integrated exactly for its rate frozen at the step's start: `survives(k, dt) = exp(-k dt)`. Each two-body exchange is an exact relaxation (`exchange`). Both are proved in `FlamescapesPort.lean`.
- **Moisture already dries before burning.** `chemistry()` runs, per fuel cell:
  - `dw = min(moisture, C (Ts - 373.15)/L)` and `Ts -= dw L / C`;
  - the vapour is added to the gas with `addGas` (`changeGas`, which keeps nitrogen and renormalises);
  - only then oxidation and pyrolysis, evaluated at the dried `Ts`.

  Water put into `moisture` therefore holds the surface at 100 °C until it is gone.
- **Stoichiometric cap.** Gas combustion takes `dF = min(yF m (1 - exp(-omega dt)), yO m / st.o2)`, and solid oxidation is capped at `o2 / st.o2`. For a fuel whose oxygen is Huggett's, `st.o2 = heat / HEAT_PER_O2`, so the heat released per step is at most `m_O2 x HEAT_PER_O2` (13.03 MJ per kg of O2; `addProducts_heat_per_O2`). This is the same bound FDS's critical-flame-temperature model starts from (`Q = m Y_O2 (Delta H / r_O2)`, about 13 100 kJ/kg).
- **Rate law.** `omega = A exp(-E/RT) cF^(a-1) cO^b` with a = -0.3 and b = 1.3, so `(a - 1) + b = 0`. This matters in Section 9, item 11.
- **Pyrolysis rises when oxygen falls**: `kp = arrhenius(...) (0.25 + 0.75 starved)`, with `starved = 1 - min(1, Y_O2/0.232)`. Steam that dilutes the oxygen at a hot fuel surface would raise its pyrolysis rate, by up to four times. That means more unburnt fuel vapour, soot and CO, an interaction to test (item 15).
- **GPU particles** are 12 floats: `p = (x, y, z, kind)`, `v = (vx, vy, vz, T)`, `q = (mass or direction, age, life, spare)`.
  - Kind 0 is a flame front, kind 1 an ember.
  - Spawning uses an atomic counter. Flame-front particles are capped at `flameShare = 0.75` of the buffer "so embers always find room". The buffer is compacted once per step.
  - Per-cell deposits use fixed-point `atomicAdd` into the `atom` buffer. Adding fields there avoids new storage bindings; the widest kernel uses five of the eight guaranteed.
  - The per-cell solid loss already exists on both backends (`solidLoss`, GPU `CF.loss`).
- **Cost baseline** (`wiki/research/tenability-index-2026-11-06.md`): the CPU takes 33 to 44 ms per step on the five examples; the GPU 3.0 ms per step at 27 300 cells and 16 ms at 186 760 cells.

---

## 3. Sprinkler activation (question 1)

### 3.1 The ODE

Heskestad and Bill (1988) added a conduction loss to the mount to the RTI model of Heskestad and Smith (1976). FDS adds a term for water droplets cooling the link. The **FDS Technical Reference Guide** (Device chapter) prints:

`dT_l/dt = sqrt|u|/RTI (T_g - T_l) - C/RTI (T_l - T_m) - C2/RTI (T_g - T_l) beta |u|`

- `T_l` is the link temperature, `T_g` the gas temperature at the link and `T_m` the mount temperature (taken as ambient).
- `|u|` is the gas speed and `beta` the volume fraction of liquid water.
- `C2 = 2.23e5`, "empirically determined based on work of DiMarzo and co-workers" (Ruffino and di Marzo 2003).

The **FDS code** (`Source/dump.f90`, quantity `SPRINKLER LINK TEMPERATURE`, master on 2026-11-07) differs in the third term:

```
RHS = (VELSR*(TMP_G - TMP_L) - C_FACTOR*(TMP_L - TMP_INITIAL) - C_DIMARZO*VEL*WATER_VOL_FRAC*(TMP_L - WATER_TEMP)) / RTI
TMP_L = MAX(MIN(TMP_G, TMP_INITIAL), TMP_L + DT*RHS)
```

- The code's wet term relaxes the link toward the **water** temperature, `-(C2/RTI) beta |u| (T_l - T_w)`. The guide's printed form, `-(C2/RTI) beta |u| (T_g - T_l)`, has the opposite structure.
- The guide's form is unstable at realistic water loadings (Section 9, item 7); the code's is not.
- The guide's unit for C2, (m s)^-1/2, does not balance; the term needs (s/m)^1/2.
- FDS steps the link by explicit Euler and clips it from below at `min(T_g, T_ambient)`.
- `VEL` is the cell-centred speed from the six face velocities of **the cell containing the link**, and `TMP_G` is that cell's temperature: no interpolation and no sub-grid model.

The heat detector is the first term alone (FDS and CFAST). DETACT-QS (Evans and Stroup 1986; not read, described in CFAST and FDS) is that first term driven by Alpert's quasi-steady ceiling-jet correlations for a growing fire. CFAST steps the same first-term ODE by a semi-implicit (trapezoidal) update. For rooms without a fire, or below the jet, CFAST uses the layer temperature and a fixed speed of 0.1 m/s.

### 3.2 Parameter values

| Quantity | Value | Source and confidence |
|---|---|---|
| RTI, quick (fast) response | 50 (m s)^1/2 or less | NFPA 13 definition, from secondary web sources only; NFPA 13 not read |
| RTI, standard response | 80 (m s)^1/2 or more | as above. ISO 6182-1 classes not read |
| FDS defaults | RTI 100, C 0, 74 °C | FDS User's Guide source |
| UL/NFPRF ELO-231 upright (standard response) | RTI 148, C 0.7, 74 °C; deflector 8 cm and link 11 cm below the ceiling; 3 m spacing | FDS Validation Guide, quoting UL |
| Vettori's residential (quick response) | RTI 56 with 11 % relative standard uncertainty; 68 ± 2.4 °C | FDS Validation Guide |
| Bittern: residential A and B (3 mm bulb) | RTI 36, C 0.4, 68 °C | FDS Validation Guide; C chosen by sensitivity analysis |
| Bittern: standard SS68 and SS93 (5 mm bulb) | RTI 95, C 0.4, 68 or 93 °C | as above |
| C2 (wet link) | 2.23e5 (s/m)^1/2 | FDS guide and code (`C_DIMARZO`); unit corrected as noted |
| Common bulb ratings | orange 57, red 68, yellow 79, green 93, blue 141, mauve 182 °C | common industry values, not checked against ISO 6182-1. IFC lists these six colours |
| Uncertainty of a measured activation time from RTI, temperature and HRR | 6 % relative | FDS Validation Guide, Table "Summary of uncertainty estimates" |

### 3.3 Closed form for constant gas temperature and speed

Write `a = sqrt|u|/RTI`, `b = C/RTI`, `c = C2 beta |u|/RTI` (the code's form) and `k = a + b + c`. With `T_g`, `T_m` and `T_w` constant, the ODE is linear with constant coefficients:

`T_l(t) = T_inf + (T_l(0) - T_inf) exp(-k t)`, where `T_inf = (a T_g + b T_m + c T_w)/k`.

With dry gas (`c = 0`) and `T_m = T_l(0) = T_0`:

- The link opens only if `T_inf > T_a`, that is, `T_g - T_0 > (T_a - T_0)(1 + C/sqrt|u|)`. This is Heskestad and Bill's point: conduction raises the gas excess temperature needed by the factor `1 + C/sqrt u`. With C = 0.4 and u = 1 m/s the factor is 1.4.
- When it opens, at
  `t_a = RTI/(sqrt|u| + C) x ln( dT_inf / (dT_inf - dT_a) )`, with `dT_inf = (T_g - T_0) sqrt|u|/(sqrt|u| + C)` and `dT_a = T_a - T_0`.
- With C = 0 this is the textbook form `t_a = (RTI/sqrt|u|) ln( (T_g - T_0)/(T_g - T_a) )`.

Worked values (T_0 = 20 °C, T_a = 68 °C, T_g = 100 °C, u = 1 m/s):

| RTI | C | t_a |
|---|---|---|
| 50 | 0 | 45.8 s |
| 50 | 0.4 | 65.4 s |
| 100 | 0 | 91.6 s |
| 36 | 0.4 | 47.1 s |

With the gas 25 % cooler in excess (80 °C), RTI 50 and C 0 give 80.5 s instead of 45.8 s. Near the activation threshold a modest under-reading of the gas changes the time a great deal, and can prevent activation altogether.

`t_a` is linear in RTI, because `T_inf` does not depend on RTI. It falls as `T_g` rises and as `u` rises (for C ≥ 0).

### 3.4 The update for this solver

- **Step:** use the exact step, `T_l <- T_inf + (T_l - T_inf) exp(-k dt)`, with `T_g`, `|u|` and `beta` read once per solver step. It is the port's `exchange` with one infinite body. It never leaves the interval spanned by `T_l`, `T_g`, `T_m` and `T_w`, for any `dt` (item 1).
- **Euler's bound:** explicit Euler stays inside only for `k dt ≤ 1`, that is `dt ≤ RTI/(sqrt|u| + C)`, about 14 s for RTI 36, C 0.4 and u = 5 m/s. It is never binding at the port's 0.05 to 0.2 s. The exact form is still preferred, for consistency with the rest of the port and so that the GPU and CPU can share a proof.
- **Speed floor:** apply 0.1 m/s, as CFAST does. Otherwise a link in a stagnant hot layer never heats.
- **Where it runs:** CPU, after `project` (the velocity is final). GPU, a one-thread-per-sprinkler kernel at the end of each step. It reads `T` and the six faces of each sensor cell, keeps `T_l`, an opened flag and `t_act` in a small buffer, and copies them back once per frame with the node readings. The spray kernels read the flags on the GPU, so opening a head needs no round trip.
- **The wet term (`c`)** matters only for the delay of later heads under earlier sprays, which FDS's validation calls "somewhat random". Leave it out of the first version. If added, use the code's form, never the guide's.

---

## 4. Reading the gas at a sprinkler on a 0.29 to 0.4 m grid (question 2)

### 4.1 Geometry of the ceiling jet against the cell

- **Thickness.**
  - Alpert's experiments put the ceiling jet's thickness at about 5 to 12 % of the ceiling height (Zeng et al. 2023, quoting Alpert).
  - Motevalli and Marks give the 1/e thickness as `delta/H = 0.112 (1 - exp(-2.24 r/H))` for `0.26 ≤ r/H ≤ 2` (CFAST guide).
  - For a fuel surface 2.2 m below the ceiling and a head 2 m from the plume, `delta` is about 0.21 m; for H = 2.7 m and r = 2.5 m, about 0.26 m.
- **Where the link sits.** Links sit 2 cm (Bittern's bulbs) to 11 cm (UL/NFPRF) below the ceiling.
- **Where the port reads.** The port's top open cell has its centre 0.15 to 0.2 m below the ceiling and averages 0 to 0.3 or 0.4 m. That is the whole jet plus cooler air beneath it.
- **Illustration** (my assumption, not a measured profile). Take a Gaussian-like excess `exp(-(z/delta)^2)` from the ceiling with `delta` = 0.27 m. A 0.35 m cell then averages 64 % of the peak excess, while a link at 0.11 m sees 85 % of it. The cell reads about 25 % low in excess temperature.
- **Velocity** is worse: the jet's speed peaks within centimetres of the ceiling, and the solver's top-cell horizontal faces carry a smeared value.

### 4.2 What FDS does and recommends

- **How FDS reads a sprinkler.** Section 3.1: the cell containing the link, the cell-centred speed and no sub-grid model. By default a device reports the value of the cell containing the point. The User's Guide offers an eight-cell interpolation, but warns against it for "a sprinkler device located under a stair-stepped ceiling", because the average can take values from the other side of the ceiling.
- **No cell-size rule for sprinklers.** I found none in the User's Guide source. The general guidance is the plume resolution index `D*/dx`, with `D* = (Q/(rho c_p T sqrt g))^(2/5)`. The guide refers to the table of values used in validation and warns against taking any as an acceptable minimum. A range of 4 to 16 is often quoted in practice, but I did not find that range in the current guide.
- **Where the port stands.** For the port at dx = 0.35 m, `D*/dx` is about 1.7 at 300 kW and 2.7 at 1 MW, the range where residential and light-hazard sprinklers open.
- **Studies.**
  - Zeng et al. (2023), FDS 6.7.7, 0.62 MW, 8.6 m ceiling: 0.05, 0.1 and 0.2 m cells gave nearly the same ceiling-jet excess. At 0.4 m it was lower, by roughly 10 to 25 % read from their Fig. 4b. For their other cases they chose 0.1 m. Their own model differed from the measurements by 15 % in the ceiling-jet region, against 12 % for Alpert's fit. They also note that sprinklers in confined rooms respond much sooner than Alpert's (unconfined) correlations predict, because hot gas accumulates.
  - Hopkin, Spearpoint and Bittern (2018; abstract only) re-modelled Bittern's 22 experiments with sensitivity to grid size, C-factor, radiative fraction and leakage. FDS predicted the activation times within a Euclidean relative difference of 0.18.

### 4.3 Expected error (my calculation)

This is a DETACT-style run with Alpert's unconfined correlations, a fuel surface 2.2 m below the ceiling, the head at r = 2.0 m, T_0 = 20 °C and T_a = 68 °C, using the exact link step. "Coarse" multiplies the jet's excess temperature by 0.75 and its speed by 0.7, a guess for a 0.35 m cell from Sections 4.1 and 4.2. Each entry is the activation time and, in brackets, the heat release rate when the head opens.

| Fire | RTI, C | Alpert | Coarse (0.75 dT, 0.7 u) | Coarser (0.6 dT, 0.5 u) |
|---|---|---|---|---|
| medium t² (0.0117 t² kW) | 36, 0.4 | 180 s (379 kW) | 223 s (585 kW) | 267 s (836 kW) |
| medium t² | 100, 0 | 200 s (468 kW) | 241 s (678 kW) | 280 s (921 kW) |
| fast t² (0.0469 t² kW) | 50, 0 | 100 s (468 kW) | 120 s (680 kW) | 140 s (922 kW) |
| fast t² | 148, 0.7 | 148 s (1023 kW) | 179 s (1506 kW) | 211 s (2086 kW) |

A coarse cell delays activation by about 20 to 25 % and lets the fire grow 45 to 55 % larger before the head opens. Under Evans's model the suppressed fire then decays from that larger value. The sign is known (late) as long as the grid under-reads the jet. The size is not known for this solver until it is measured against Vettori or Bittern (Section 8).

### 4.4 Options and recommendation

- **(A) Grid reading.** Take the top open cell of the sprinkler's column: march up from the link point until the first SOLID cell, then step back one. Use its `T`, and the cell-centred speed from its six faces with a floor of 0.1 m/s.
  - Cheap and local.
  - Biased late, for the reasons above.
  - Use this for the late bound.
- **(B) Alpert sub-grid estimate.** Use the solver's own convective heat release `Q`, the plume axis, the ceiling height above the burning fuel `H` and the radial distance `r`:
  - `dT = 16.9 Q^(2/3)/H^(5/3)` for `r/H ≤ 0.18`, and `5.38 (Q/r)^(2/3)/H` beyond;
  - `U = 0.947 (Q/H)^(1/3)` for `r/H ≤ 0.15`, and `0.197 Q^(1/3) H^(1/2)/r^(5/6)` beyond;
  - SI units with Q in kW, as quoted by Zeng et al. Many texts round the speeds to 0.96 and 0.195. The branches meet within 0.2 % for temperature and 1.1 % for speed.

  The plume axis is the heat-release-weighted centroid of the burning cells. The CPU's `chemistry()` would accumulate the per-cell heat it already computes; the GPU would add a reduction.
  - Alone, Alpert is unconfined, so it is also late in a room. It needs `Q` and a plume.
  - Several fires need a rule; CFAST takes the fire giving the hottest jet.
  - A plume against a wall or in a corner is usually treated by the image method (twice or four times `Q`). That is common practice, not checked here.
- **(C) CFAST's hybrid.** CFAST adds Alpert's excess onto the upper-layer temperature instead of ambient. Here the grid gives the layer. Then:
  - `T_g = T_u,grid + dT_Alpert(Q, H, r)`, with `T_u,grid` the mean top-cell temperature of the same space, excluding columns within `r < 0.2 H` of the plume so that the jet is not counted twice;
  - `u = max(U_Alpert, |u_cell|, 0.1 m/s)`.

**Recommendation.** Run the link ODE twice per head, on (A) and on (C).
- Report (C) as the estimate and (A) as the late bound.
- Use (C) to open the head in the simulation, so the suppressed fire starts from a fire size near the real one.
- Keep a switch to use (A) for a conservative study.

The sprinkler card should show both times and the heat release rate at each, so a reader sees how wide the bracket is.

---

## 5. Suppression models (question 3)

### 5.1 Evans (1993), NISTIR 5254 (read in full)

- **Data.**
  - Madrzykowski and Vettori (1992) burned unshielded office furnishings and wood cribs under pendent sprinklers at 0.07 mm/s.
  - Measured decay constants ranged from 9.1 s (a vinyl sofa) to 713 s (workstation I).
  - All were bounded by `Q(t - t_act)/Q(t_act) = exp(-(t - t_act)/435)` (Evans's Eq. 1).
- **Generalisation.** Evans extended this with Walton's (1988) crib tests at other densities:
  - `tau = 2.0e-5 (w''/H_c)^-1.85` for crib height `H_c` in mm, within ±50 %;
  - for the 610 mm crib, `tau = 3.0 (w'')^-1.85` with `w''` in mm/s.

  Recommended form (his Eq. 4): `Q(t - t_act) = Q(t_act) exp(-(t - t_act)/(3.0 w''^-1.85))`.

  | w'' (mm/s) | tau |
  |---|---|
  | 0.026 | 2567 s |
  | 0.05 | 766 s |
  | 0.07 | 411 s |
  | 0.10 | 212 s |
  | 0.133 | 125 s |
  | 0.2 | 59 s |
- **Scope, in Evans's words:**
  - "appropriate for spray densities greater than or equal to 0.07 mm/s";
  - "a conservative bound, meaning minimum expected reduction in heat release rate", for fuel "not shielded from the water spray";
  - below about 0.026 mm/s water had no measurable effect on the cribs;
  - the power law over-predicts the long extinction times at low densities, which is conservative;
  - "the model does not consider the interaction of the spray with the fire driven gas flow".
- **CFAST's current version** uses `tau = tau_0 (u_w,0/u_w)^1.8`. Past versions fixed `tau_0 u_w,0^1.8 = 3` (mm/s and s), giving 360 s at 0.07 mm/s. CFAST notes that the model cannot represent a fire that overwhelms the sprinkler, and that a second head does not speed suppression.
- **Spray density versus head type.**
  - Common design densities are 4.1 mm/min (0.068 mm/s) for NFPA 13 light hazard and 2.25 mm/min (0.0375 mm/s) for EN 12845 light hazard. These are from memory, for the placement agent to confirm.
  - Residential densities near 0.034 mm/s lie below Evans's validated range; there `tau` is about 1500 s, effectively "no help within the egress window".

### 5.2 FDS

- **Droplets** are Lagrangian parcels (default `PARTICLES_PER_SECOND = 5000` per head), sized from a Rosin-Rammler-lognormal distribution (`GAMMA_D` 2.4 by default).
- **Drag** uses a sphere drag curve.
- **Heat and mass transfer** use Ranz-Marshall-type correlations (`Nu_0 = 2 + 0.6 Re^1/2 Pr^1/3`, `Sh_0 = 2 + 0.6 Re^1/2 Sc^1/3`) with Spalding blowing corrections. The equilibrium vapour fraction comes from Clausius-Clapeyron. The coupled droplet, gas and solid equations are solved implicitly over a gas step.
- **Super drops.** The User's Guide warns that a single super drop can cool a cell too fast, and suggests more particles.
- **On surfaces**, droplets form a film (minimum 1e-5 m) and drip at about 0.5 m/s.
- **Suppression of a burning surface:**
  - When the fuel has a pyrolysis model, water acts through surface cooling.
  - For a prescribed heat-release surface: `m''_f(t) = m''_f,0(t) exp(-integral of a m''_w(t) dt)`, with `m''_w` the water per unit area (kg/m²) and `a` (`E_COEFFICIENT`, m²/(kg s)) empirical (Hamins and McGrattan 1999, 2003). FDS motivates the local exponential from Yu, Lee and Kung's (1994) global `Q = Q_0 exp(-k (t - t_0))` for rack storage.
- **Gas-phase extinction** uses a critical flame temperature, set from a limiting oxygen volume fraction.
  - `EXTINCTION 1`, meant for coarse grids: a cell does not burn if `X_O2 < X_OI (T_OI - T)/(T_OI - T_inf)` below a "free-burn" temperature (600 °C default), with `X_OI = 0.135` at 20 °C (Morehart et al. 1991 found 12.4 to 14.3 %).
  - `EXTINCTION 2` compares the potential heat release with the enthalpy needed to reach the critical flame temperature. Water vapour counts as a diluent product.

### 5.3 Fire-X (repository transcription)

- **Gas chemistry.** Heat is tied to stoichiometry (Eqs. 4 to 7): about 417 kJ per mole of O2 for any hydrocarbon.
- **Liquid.**
  - Eq. 9 removes liquid in proportion to its sensible energy above ambient.
  - Eq. 10 couples droplet and gas temperatures.
  - Eqs. 26 to 28 evaporate `n_e = E_l/E_d` whole droplets: the cell's liquid sensible energy above ambient over the energy to heat one droplet to boiling and vaporise it.
  - Eqs. 29 to 31 mix the vapour in at `T_B` and rescale the species.
  - Eq. 32 blends the gas velocity toward the liquid's (the "displacement" term; Fig. 13 shows that both dispersion and displacement are needed).
- **No extinction rule.** Suppression emerges only through the Arrhenius rate: lower `T` and lower `c_O`.
- **Gaps.** Fire-X has no solid fuel model (solids are static obstacles), no detector model and no sprinkler model beyond Table 3's emitter ranges. The transcription's notes record units issues in Eqs. 9, 10 and 19 to 21, and that Eq. 31 leaves `sum Y < 1` unless vapour is added elsewhere.

### 5.4 How each maps onto the port's fields

| Model | Reads | Writes | New state | Notes |
|---|---|---|---|---|
| Activation (all) | `T`, `u, v, w` at the sensor cell; Alpert needs the per-cell heat release this step and the plume centroid | nothing | per head: `T_l`, opened, `t_act` | no effect on the fire |
| Evans | `solidLoss` (per-cell mass loss, already on both backends) at activation; time | caps `dOx`, `dPy` and the liquid's `dv` in covered fuel cells, hence `fuelMass`, `Ts`, `Y` through `addProducts`/`addGas` | per cell: mass-loss rate at activation (kg/s), `tau` (from the local `w''`), `t_act` (or head index) | fuel vapour already in the gas at activation still burns: a finite extra energy `m_F,gas x heat` |
| FDS E-coefficient | water per unit area on a fuel cell (`moisture/area`) | multiplies the fuel cell's reaction by `exp(-integral of a m''_w dt)` | per cell: the running integral | `a` is fuel-specific and unknown for "contents" |
| FDS droplets | `T`, `Y`, velocities, `Ts` | `T`, `Y_H2O`, velocities (drag), `moisture` or `Ts` | droplet buffer | Section 7 |
| Fire-X / Eulerian spray zone | `T`, `rho`, `Y`, `Ts`, `moisture` | `T` (cooling), `Y_H2O` via `changeGas` (O2 diluted), `v` faces (drag), `moisture` (wetting) | per head: static zone (cells, weights) | soot unchanged; buoyancy falls through `T` and the equation of state; flame-front particles in sprayed cells should be culled so a stale flame temperature does not relight a cooled cell |

### 5.5 Which first, and why

1. **Evans, as a per-cell cap on the solid fuel's reaction in covered cells**, after activation:
   - **Mechanism.** For each fuel cell under an open head, record its mass-loss rate `r_c` at activation. Use a frame average of `solidLoss/dt`, since one step's value is noisy. Thereafter scale `dOx`, `dPy` and the liquid's `dv` so that their sum is at most `r_c exp(-(t - t_act)/tau_c) dt`.
   - **Exactness.** If every burning cell is covered and opened together, the total heat release follows Evans exactly, because a sum of identical exponentials is one exponential.
   - **New ignitions.** A covered cell that was not burning at activation has `r_c = 0` and never ignites, which matches Evans's premise that a sprinklered fire does not grow.
   - **Outside coverage**, the fire is unaffected, which is right.
   - **Why first.**
     - Its numbers come from full-scale tests and it is conservative by construction.
     - It needs no sub-grid spray physics.
     - It is identical on CPU and GPU.
     - It is what CFAST does, so CFAST runs can check it.
     - Its properties are easy to prove (item 14).
   - **Limits.**
     - It is a heat-release envelope only: it neither cools the gas nor makes steam.
     - It is calibrated on single unshielded items.
     - Its per-cell use is my assumption.
     - Below 0.07 mm/s it is outside its data.
2. **The Fire-X-style energy-limited evaporation in the gas**, in an Eulerian spray zone. It is the second step because it adds the egress-relevant effects: cooling and loss of buoyancy (smoke comes down and visibility at head height can fall), steam, and oxygen dilution. Its energy bookkeeping follows `chemistry_liquid_energy` (item 9). The per-cell evaporated fraction is the uncertain parameter (Section 7).
3. **Wetting the fuel through `moisture`.** This is mechanistic and already wired, but not as a first step. On this grid every covered contents cell is a fully exposed surface layer, and a design density of 0.068 kg/(m² s) can absorb about 176 kW/m². The gas-to-solid convection the port uses (h = 200 W/(m² K)) brings in about 140 kW/m² from 800 °C gas, so a wetted cell would be held at 100 °C almost at once, where the oxidation constant is about 2e-5 per second. That is likely far faster than any of Evans's furnishings except the sofa: shielding, vertical surfaces and deep-seated burning are not on this grid. It needs a "wetted fraction" calibrated so that the heat release never falls faster than Evans's envelope in the cases Evans covers; until then, use the envelope.
4. **FDS's Lagrangian model** last, if ever. It is the most detailed, but its benefits (spray-plume interaction, droplet size effects) depend on resolving what a 0.35 m cell cannot.

---

## 6. Data per sprinkler and mapping to cells (question 4)

### 6.1 Record per sprinkler

| Field | Unit | From IFC (`Pset_FireSuppressionTerminalTypeSprinkler`) | Default if absent |
|---|---|---|---|
| id, storey, space (room) | | element GUID, containment | from the placement step |
| link point | m, viewer frame | placement | deflector point minus about 0.03 m |
| orientation / type | | `SprinklerType` (CEILING, CONCEALED, CUTOFF, PENDANT, RECESSEDPENDANT, SIDEWALL, UPRIGHT) | PENDANT |
| distance of link below ceiling | m | not in the Pset; from geometry | 0.05 to 0.10 m |
| activation temperature | K | `ActivationTemperature`; else `BulbLiquidColor` (ORANGE, RED, YELLOW, GREEN, BLUE, MAUVE) | 68 °C (bulb), 74 °C (link) |
| activation element | | `Activation` (BULB, FUSIBLESOLDER) | BULB |
| response class | | `Response` (QUICK, STANDARD) | by occupancy |
| RTI | (m s)^1/2 | **not in the Pset** | QUICK 50, STANDARD 100 (FDS default); allow 36 to 150 |
| C-factor | (m/s)^1/2 | **not in the Pset** | 0.4 (Bittern's estimate), with a sensitivity switch to 0 and 0.7 |
| K-factor | L/(min bar^1/2) | `DischargeCoefficient` ("coefficient of flow at the sprinkler"; unit not stated, so check) | 80 (K5.6 US is about 80.7) |
| pressure | bar | `ResidualFlowingPressure` | from the hydraulic design; 0.5 bar as a placeholder |
| flow | L/min | `DischargeFlowRate`, else `K sqrt(p)` | |
| coverage area and footprint | m² and polygon | `CoverageArea` | from the placement rules (another agent) |
| design density | mm/s | derived: flow / area | |
| spray half-angle, pattern | degrees | not in IFC | FDS example: 30° to 80° band for an upright; a uniform floor footprint matching the coverage is simpler |
| droplet median diameter | m | not in IFC | 0.5 to 1 mm (Fire-X Table 3 range 0.5 to 5 mm; FDS example 1 mm) |
| initial droplet speed | m/s | not in IFC | 10 m/s (FDS example) |
| provenance | | IFC / code default / user | shown on the sprinkler card |

There is no RTI or C in IFC, so every run states the defaults it used, and the card offers the bracket of activation times for RTI 36 to 150.

### 6.2 Mapping to cells

- **Sensor cell.** The top open cell in the link's column: march up from the link point's cell to the first SOLID cell and step back one. If the link point is inside a SOLID cell (a thick ceiling slab), march down to the first open cell. This is the `headCells` pattern reversed. FDS's warning about interpolating under a stair-stepped ceiling applies, so do not average across the ceiling.
- **Spray zone** (static, built once when the head opens, or once per run):
  - for each grid column whose centre lies inside the head's floor footprint, the open cells from the sensor layer down to the first fuel (POROUS) or SOLID cell;
  - keep only cells the head can see (a solid-mask march from the link point, as `march` does for rays), so walls and slabs block the spray;
  - a weight per column, normalised to 1 over the visible footprint. Uniform matches the "design density" idea; a cone-shaped radial weight is optional.
- **Wetted fuel cells.** The POROUS fuel cell that ends each column, if any.
- **Evans coverage.** The fuel cells under the footprint, with the local density `w''_c = sum over open heads of flow x weight / column area`, so overlapping heads add.
- **Particle emitter**, if Lagrangian parcels are used later: the link point, an orientation, the spray band and the speed. Parcels are spawned in the sensor cell or the next one down (FDS's `OFFSET`, 0.05 m by default).

---

## 7. The spray in the solver: Lagrangian droplets or an Eulerian source (question 5)

### 7.1 Time and length scales

- **Transit time.** Droplets leave at about 10 m/s and settle at a few metres per second. They cross 2.4 m in about 0.3 to 0.6 s, which is 3 to 6 steps at `dt = 0.1 s` and far below the 5 s frames. The spray in a room is therefore effectively steady from the step after a head opens.
- **Evaporation per cell** (my estimate from the FDS correlations without blowing correction, droplet already at 100 °C, gas 300 K hotter):
  - A 1 mm droplet at about 4 m/s (Re about 150, Nu about 8.5) gains about 0.26 W and needs 1.36 J to evaporate. Crossing a 0.35 m cell, it evaporates about 1.7 %.
  - A 0.5 mm droplet at about 2 m/s gains about 0.09 W against 0.17 J, so about 9 % per cell.
  - So most water from ordinary sprinklers reaches the floor and the fuel. Gas cooling acts mainly in the hot layer near the head, and droplet size is the main uncertain parameter.
- **Cooling capacity is ample.** A K80 head at 0.5 bar delivers about 0.94 kg/s, which could absorb about 2.4 MW if it all evaporated. Only a small fraction need evaporate to cool a few-hundred-kW fire's layer. Fire-X's 300 L/min (5 kg/s, about 13 MW of capacity) on a 33 kW flame is far more water than one head delivers, so its validation tests the coupling more than the quantities.
- **Smoke-layer drag** (my estimate). Droplets falling at terminal speed pass their weight to the gas: a downward force per volume equal to the water's bulk density times g.
  - At 0.068 kg/(m² s) and 4 to 5 m/s that is about 0.014 to 0.017 kg/m³, or 0.14 to 0.17 N/m³.
  - A 100 °C layer's buoyancy is about 2.5 N/m³.
  - So the spray brings smoke down mainly by cooling it; drag decides only once the layer is within about 10 K of ambient (0.4 N/m³).
  - This is the drag-against-buoyancy balance studied by Bullen (1977) and Morgan (1979); both are unread here.

### 7.2 Eulerian spray zone (recommended for step 2)

Per step, for each open head, process each visible column from the top down:

- the column's water this step is `m_0 = flow x weight x dt`;
- in each cell `j`, evaporate `dv_j = min(eta_j m_j, energy limit, saturation limit)`:
  - `eta_j`, the fraction per cell, comes from droplet diameter and gas temperature (a closed form like the estimate above), or is a single calibrated parameter;
  - the energy limit is that of item 9;
- apply the energy update (item 9) and `changeGas` with `+dv` of H2O;
- add the drag force `(m_j/(v_t dt h²)) g` to the cell's lower `v` face;
- pass `m_{j+1} = m_j - dv_j` down;
- at the bottom, add the remainder to the fuel cell's `moisture`, scaled by a wetted fraction (step 3), or drop it at the floor.

Cost:

- **CPU.** A head covering 12 m² at 0.35 m is about 100 columns of about 7 cells, so 700 cell updates; ten heads give 7000. That is well under 1 ms per step against 33 to 44 ms (estimate).
- **GPU.** One thread per zone cell, ordered by layer, or one thread per column looping down. Results accumulate through fixed-point `atomicAdd` into new `atom` fields (water and heat per cell) followed by a per-cell apply kernel, the pattern the embers already use. No new storage bindings are needed.
- **Agreement.** CPU and GPU agree up to float32 and summation order, as the existing parity checks expect.

### 7.3 Lagrangian parcels (later, optional)

- **On the GPU**, a droplet is a third kind (`p.w = 2`):
  - `q.x` holds the parcel mass and `q.w` the diameter;
  - motion is an exact exponential relaxation toward the gas velocity plus gravity, as the embers' `relax` does;
  - `particleToGrid` adds the parcel's mass to an `atom` field;
  - a per-cell kernel evaporates energy-limited on the grid (Fire-X Sec. 5.4), and the parcels are scaled back by their cell's evaporated fraction. Doing the evaporation on the grid, not per parcel, removes FDS's super-drop instability.
- **Buffer.** Use a separate buffer, or a reserved quota in the shared one. In the shared buffer flame-front particles may take 75 %, and a spray starved of slots would silently deliver less water. The water budget (item 8) would then fail, which is a good runtime check.
- **Counts.** At FDS's default 5000 parcels per second per head and about 0.5 s in flight, each head has about 2500 parcels aloft; ten heads give 25k.
- **Cost.**
  - GPU: trivial next to the grid's 3 ms per step.
  - CPU: each parcel samples three face velocities and the gas temperature, about 1 µs in JavaScript by analogy with the ember code (estimate). That is about 25 ms per step for 25k parcels, roughly doubling a desktop step and worse on a phone.
- **Fire-X** carries liquid as SPH particles with pressure, near-pressure and viscosity forces (Eqs. 12 to 17), particle-to-grid and back (Eqs. 18 to 25), and evaporation on the grid. Its Table 2: 6.7k active particles (Fig. 8 nozzles, 6 to 8 ms per step for particles) and 106k (Fig. 18, the kitchen with a detector-triggered sprinkler: 15 ms for particles, 7 ms for the grid), on an RTX 4090. SPH matters for jets and pooling, not for a dilute sprinkler spray, so the port can drop it.

---

## 8. Validation data (question 6)

| Case | What was measured | Use for the port | Source status |
|---|---|---|---|
| **Closed form** (Section 3.3) | exact `t_a` for constant gas | unit test of the link step; Lean items 3 and 4 | derived |
| **Vettori flat ceiling**, NISTIR 6253 (1998), doi:10.6028/NIST.IR.6253 | 45 tests: residential pendent heads (RTI 56, 68 °C), smooth or beamed ceiling, slow, medium and fast t-squared methane burner at three positions; activation times of up to four heads (no water); temperatures (including at the heads) and some ceiling-jet speeds | activation from a prescribed heat release rate (the port needs a prescribed-HRR gas burner mode); compare readings A and C with measured times | via the FDS Validation Guide; report not read |
| **Vettori sloped ceilings**, NISTIR 7079 (2003), doi:10.6028/NIST.IR.7079 | 72 tests: flat, 13° and 24° slopes, smooth or obstructed, fast or slow fire, three positions; thermocouples 15 cm below the ceiling | as above, for sloped IFC roofs | via the FDS Validation Guide |
| **Bittern 2004** (University of Canterbury thesis); Wade et al. 2007, doi:10.1007/s10694-007-0009-5; Hopkin et al. 2018, doi:10.1177/0734904118772306 | 22 single-chair fires in a gypsum-lined room; two heads (RTI 36 or 95, C 0.4, 68 or 93 °C), not flowing; activation times; chair mass loss (HRR); gas temperature profiles; door open or closed | the best match to the app's room scale with a real burning item | via the FDS Validation Guide; Hopkin abstract read |
| **UL/NFPRF**, NISTIR 6196-1 (1998), doi:10.6028/NIST.IR.6196-1 | 22 tests at 4.4 MW and 12 at 10 MW, heptane spray burner (t-squared ramp, then held), 7.6 m ceiling, 3 m spacing, flowing ELO-231 heads (RTI 148, C 0.7, 74 °C) at 0.34 L/(s m²); first activation (54 to 106 s in Series I) and total count; ceiling and layer temperatures | first activation and the spray's cooling of the jet (activation count); large domain (about 100k cells at 0.4 m) | via the FDS Validation Guide |
| **Madrzykowski and Vettori 1992**, NISTIR 4833, doi:10.6028/NIST.IR.4833 (also J. Fire Prot. Eng., doi:10.1177/104239159200400403) | heat release rates of office furnishings and cribs, free and sprinklered at 0.07 mm/s; decay constants 9.1 to 713 s; heat release at activation 280 to 1365 kW | the Evans envelope and the step-3 calibration | via Evans 1993 (Table 1) |
| **Walton 1988**, NBSIR 88-3696, doi:10.6028/NBS.IR.88-3696 | 305 and 610 mm crib heat release rates at spray densities 0.026 to 0.133 mm/s (Evans excluded 0.013 and 0.14 mm/s runs); decay constants 44 to 543 s | the density dependence of `tau` | via Evans 1993 (Table 2) |
| **Madrzykowski 2008**, NISTIR 7506, doi:10.6028/NIST.IR.7506 | dry Fraser fir trees 3.2 to 4.3 MW free; furnished room with one residential sprinkler: peak about 1.8 MW, no flashover; without the sprinkler, more than 6 MW | a whole-room "does the sprinkler stop flashover" test | abstract read |
| **Milke, Evans and Hayes 1988**, NBSIR 88-3745, doi:10.6028/NBS.IR.88-3745 | water spray on fully developed crib fires in a compartment | the gas-phase cooling step | title only (Crossref) |
| **Hamins and McGrattan 1999, 2003**, NISTIR 6439 and doi:10.3801/IAFSS.FSS.7-457 | reduced-scale rack-storage commodity suppression; the E-coefficient | only if FDS's E-coefficient form is adopted | not read |
| **Bullen 1977**, doi:10.1007/BF02338883; **Morgan 1979**, doi:10.1002/fam.810030106 | stability of a smoke layer under a sprinkler spray; heat transfer from the layer to the spray | the smoke-descent effect the egress forecast must not hide | titles only |
| **FDS Validation Guide water cases**: VTT water spray (nozzle LN-2; droplet speed, flux and median diameter at 40 and 62 cm); USCG/HAI water mist (7 by 5 by 3 m machinery space, 250 kW to 1 MW heptane spray, extinction times) | | spray footprint and extinction checks | via the guide |
| **Fire-X validation** (Sec. 6.2.1) | enclosure of NBSIR 79-1910 (McCaffrey 1979, doi:10.6028/NBS.IR.79-1910), 1.5 by 1.5 by 3.9 m as described by Fire-X, 0.72 m pedestal; flames of 14.4 to 57.5 kW; 21 thermocouples 0.05 to 3.0 m above the pedestal. Sprinkler case: 33 kW, a 300 L/min sprinkler started at 15 s, compared with FDS only (mean temperature and per-sensor mean and standard deviation, Fig. 14g to i) | a cheap smoke test of the coupling; no measurement behind the sprinkler case | transcription |

A coarse solver should be compared on quantities it can resolve:
- activation time, in the bracket of readings A and C;
- the number of activations;
- the room-average upper-layer temperature before and after activation;
- the total heat release rate after activation, against Evans's envelope.

Point thermocouples in the ceiling jet are not a fair comparison.

---

## 9. Lean 4: what to prove before implementing (question 7)

Placement: a new `FireEgress/Sprinkler.lean` for items 1 to 7 and `FireEgress/Suppression.lean` for items 8 to 15. Both import `FlamescapesPort` (`survives`, `arrhenius`, `changeGas`, `exp_neg_le_quadratic`) and `FireX` (`X29`, `X30`). Real arithmetic, as elsewhere; the GPU's float32 is not modelled. Each counterexample is kept beside its theorem, as the repository does (`exchange_explicit_overshoots`, `survives_euler_negative`).

Notation: `a = sqrt u / RTI ≥ 0`, `b = C / RTI ≥ 0`, `c = C2 beta u / RTI ≥ 0`, `k = a + b + c`, `T_inf = (a T_g + b T_m + c T_w)/k`.
`linkStep Tl Tg Tm Tw a b c dt = if 0 < k then T_inf + (Tl - T_inf) exp(-k dt) else Tl`.

1. **`linkStep_mem`: discrete maximum principle for the exact link step.**
   - In words: the exact update never leaves the interval spanned by the old link temperature and the three sink temperatures, for any time step.
   - Hypotheses: `0 ≤ a, b, c`; `0 ≤ dt`.
   - Statement: `min Tl (min Tg (min Tm Tw)) ≤ linkStep ... ∧ linkStep ... ≤ max Tl (max Tg (max Tm Tw))`.
   - Corollary `linkSteps_mem` (induction on n): if `lo ≤ Tl₀ ≤ hi` and `lo ≤ Tg n, Tm, Tw ≤ hi` for all n, then `lo ≤ Tl n ≤ hi`. In particular, with `Tm = Tw = T_amb ≤ Tl₀` and gas in `[T_amb, T_max]`, the link stays in `[T_amb, T_max]`.
   - Companion `linkEuler_mem` and counterexample `linkEuler_overshoots`: the explicit Euler update stays in the hull if `k dt ≤ 1`; with `k dt = 2`, `Tl = 0`, `Tg = 1`, `b = c = 0`, it returns 2.
   - Kind: algebraic (`Real.exp_pos`, `Real.exp_le_one_iff`, convex combination).
2. **`linkStep_mono` and `activationIndex_antitone`: hotter gas never opens a head later.**
   - In words: the step is monotone in the old link temperature and in each sink. Two runs with the same speeds, one with gas at least as hot at every step and starting at least as hot, keep the order of link temperatures at every step. The first step at which `Tl ≥ Ta` is therefore no later for the hotter run.
   - Hypotheses: same `a, b, c, dt ≥ 0` per step; `Tg₁ n ≤ Tg₂ n` for all n; `Tl₁ 0 ≤ Tl₂ 0`; and run 1 opens at step n₁. (The speed is held equal on purpose: in a cooling gas a faster flow also cools the link faster.)
   - Statement: `∀ n, Tl₁ n ≤ Tl₂ n`, and `Nat.find h₂ ≤ Nat.find h₁` for the activation predicates.
   - Kind: algebraic plus induction (coefficients `exp(-k dt) ≥ 0` and `(a/k)(1 - exp(-k dt)) ≥ 0`).
3. **`link_ode_unique`: closed form of the RTI ODE for constant conditions.**
   - In words: with constant gas temperature and speed, the link temperature is `T_inf + (T₀ - T_inf) exp(-k t)`, and this is the only solution.
   - Hypotheses: `0 < k`; `f : ℝ → ℝ` with `∀ t, HasDerivAt f (a (Tg - f t) + b (Tm - f t) + c (Tw - f t)) t`; `f 0 = T₀`.
   - Statement: `∀ t, f t = T_inf + (T₀ - T_inf) * exp (-k t)`, together with the derivative of the right-hand side.
   - Kind: Mathlib real analysis (`HasDerivAt`, `is_const_of_deriv_eq_zero`), the pattern of `survives_unique`.
4. **`activation_time` and `activates_iff`: when and whether a head opens.**
   - In words: if `T₀ < Ta < T_inf`, the link reaches `Ta` exactly at `t_a = log((T_inf - T₀)/(T_inf - Ta))/k` and is below it before. If `T_inf ≤ Ta` it never reaches `Ta`. With `c = 0` and `Tm = T₀`, `T_inf > Ta` holds exactly when `(Tg - T₀) sqrt u > (Ta - T₀)(sqrt u + C)` (Heskestad and Bill's conduction criterion).
   - Hypotheses: as item 3; `0 < u`; `0 ≤ C`; `0 < RTI`.
   - Statements: `f t_a = Ta`; `∀ t ∈ [0, t_a), f t < Ta`; `T_inf ≤ Ta → ∀ t ≥ 0, f t < Ta`; and the iff.
   - Kind: Mathlib (`Real.exp_log`, `Real.log_lt_log`, strict monotonicity of `exp`).
5. **`activationTime_mono`: dependence on gas temperature, speed and RTI.**
   - In words: with `c = 0`, `Tm = T₀` and the head opening, `t_a` is strictly decreasing in `Tg`, non-increasing in `u`, and exactly proportional to `RTI` (`t_a = RTI · G(Tg, u, C)`).
   - Hypotheses: `0 < u`, `0 ≤ C`, `0 < RTI`, `T₀ < Ta`, and the activation condition.
   - Kind: algebraic plus `Real.log` monotonicity. For `u`: `T_inf - T₀ = (Tg - T₀) sqrt u /(sqrt u + C)` increases with `u`, and so does `k`.
6. **`linkStep_add` and `linkStep_euler_error`: the step is the ODE's solution and composes.**
   - In words: two exact steps of `dt₁` and `dt₂` under the same gas equal one of `dt₁ + dt₂`, and equal item 3's solution at that time. Explicit Euler differs from the exact step by at most `(k dt)² |Tl - T_inf| / 2`.
   - Hypotheses: `0 ≤ dt₁, dt₂`; `0 < k`.
   - Kind: algebraic (`Real.exp_add`) plus the existing `exp_neg_le_quadratic` and `Real.add_one_le_exp`.
7. **`wetLink_code_mem` and `wetLink_guide_unstable`: FDS's wet-link term, code against guide.**
   - In words: with the code's form (a sink at the water temperature), item 1 applies with `c ≥ 0` and the link stays in the hull of `{Tl, Tg, Tm, Tw}`. With the guide's printed form, `dT/dt = (a - c)(Tg - T) + b (Tm - T)`, the relaxation rate is `a - c + b`. For `c > a + b` it is negative and the solution grows without bound.
   - Witness: realistic values `C2 = 2.23e5`, `beta = 1.36e-5` (0.068 kg/(m² s) falling at 5 m/s), `u = 1`, `RTI = 50`, `C = 0.4` give `c/a ≈ 3.0` and `a - c + b < 0`.
   - Hypotheses: as item 1; the numerals for the counterexample.
   - Kind: algebraic (`norm_num`), plus item 3 for the unbounded growth if stated in continuous time.
8. **`spray_budget`: the Eulerian spray conserves water.**
   - In words: down each column, what enters the top equals the water evaporated in its cells plus what leaves the bottom. Every amount is non-negative and the flux never increases downward. Summed over columns with weights summing to one, and over heads, the water emitted in a step equals the water evaporated plus the water deposited (on fuel or floor) plus any leaving the domain.
   - Hypotheses: `m (j+1) = m j - e j` with `0 ≤ e j ≤ m j`; weights `w ≥ 0` with `∑ w = 1`.
   - Kind: algebraic (`Finset.sum_range_succ`, telescoping).
9. **`evaporate_energy` and `evaporate_bounds`: the evaporation step conserves energy (the form of `chemistry_liquid_energy`).**
   - Definitions: water arrives at `Tw` and is heated to `T_B` and vaporised with `e = c_w (T_B - Tw) + L`. The vapour joins the gas at its new temperature `T'`:
     `T' = (C_g T - dv (e - c_v T_B)) / (C_g + c_v dv)`, with `dv = min m_w (C_g (T - T_ref) / (e + c_v (T_ref - T_B)))`.
   - Statement (i), the identity: `C_g T = (C_g + c_v dv) T' + dv (e - c_v T_B)`. Energy relative to liquid water at `Tw` is conserved, and the vapour carries its sensible heat. The identity reduces to Fire-X's `X30_enthalpy` when `T' = T_B`.
   - Statement (ii): `0 ≤ dv ≤ m_w` for `T ≥ T_ref`.
   - Statement (iii): `T_ref ≤ T' ≤ T` for `e + c_v (T_ref - T_B) > 0`.
   - Hypotheses: `0 < C_g`, `0 ≤ c_v`, `0 ≤ m_w`, `T ≥ T_ref`, and that positivity condition.
   - Kind: algebraic (`field_simp`, `min` lemmas).
   - Remarks:
     - The existing moisture and liquid-item steps add vapour at the gas temperature without crediting its sensible heat. That is negligible for grams per step but not for the kilograms a sprinkler delivers, so the spray step should use this form.
     - If a saturation cap from Clausius-Clapeyron is added, proving that a unique saturated `dv` exists needs the intermediate value theorem (Mathlib).
10. **`vapour_changeGas` and `vapour_dilutes_O2`: species stay valid after adding water vapour.**
    - In words: adding `dv ≥ 0` of H2O through `changeGas` leaves every other species' mass unchanged and raises H2O's by `dv`. Every mass fraction stays non-negative and the tracked fractions sum to at most one (nitrogen non-negative). The cell's mass rises by `dv`, and the O2 mole fraction `n_O2/(n + dv/M_w)` does not increase (strictly decreases if `dv > 0` and `n_O2 > 0`).
    - Hypotheses: those of `changeGas_species` with `δ = Pi.single H2O dv`; `0 < V`; `0 < M_w`.
    - Kind: algebraic, reusing `changeGas_species`, `changeGas_mass` and `changeGas_nitrogen`.
11. **`dilution_scales_burn`: at fixed temperature, water vapour can only scale the gas burn down.**
    - In words: at fixed `T` and pressure, adding vapour multiplies the molar concentrations of fuel and oxygen by the same `λ = n/(n + n_w) ∈ (0, 1]`. Because the port's exponents satisfy `(a - 1) + b = 0`, the rate constant `omega` is unchanged. The fuel burnt in the step, `min(λ m_F (1 - exp(-omega dt)), λ m_O / s)`, is exactly `λ` times the undiluted value, so the heat released and the stoichiometric bound `m_O2 HEAT_PER_O2` are scaled by `λ ≤ 1`.
    - Hypotheses: `0 < λ ≤ 1`; `0 < cF, cO`; `(a - 1) + b = 0`, discharged for `a = -0.3`, `b = 1.3` by `norm_num`.
    - Kind: algebraic with `Real.mul_rpow` and `Real.rpow_add`; `min` scales by positive factors.
12. **`cooling_first_order` and `cooling_raises_cap`: cooling raises the cell's gas mass but not its first-order burn.**
    - In words: the equation of state raises a cooled cell's gas mass by `T/T'` at fixed mass fractions, and the stoichiometric cap with it (counterexample: 1000 K to 600 K raises the cap by 5/3). But for `0 < T' < T ≤ E/R`, `(T/T') exp(-(E/R)(1/T' - 1/T)) < 1`. The first-order burn rate `m_F omega` therefore falls despite the extra mass, for the port's `E = 1.26e5 J/mol` up to 15 155 K.
    - Second counterexample: in the fast-chemistry limit, where both before and after the whole fuel reacts within the step, the exact scheme burns `T/T'` times more fuel after cooling. The theorem is a first-order (small `omega dt`) statement only.
    - Hypotheses: `0 < T' < T`, `T ≤ E/R`, `0 < E`, `0 < R`.
    - Kind: Mathlib (`Real.log_lt_sub_one_of_pos` or `Real.add_one_lt_exp`): `log(T/T') < T/T' - 1 ≤ (E/(R T))(T - T')/T'`.
13. **`extinct_monotone`: a coarse-grid extinction rule, if adopted, is never undone by water.**
    - In words: for FDS's `EXTINCTION 1` limit, `X_lim(T) = X_OI (T_OI - T)/(T_OI - T_inf)` for `T < T_fb` and 0 above, with `T_inf ≤ T_fb < T_OI`, `X_lim` is antitone in `T`. A cell extinct at `(X_O2, T)` therefore stays extinct at any `(X', T')` with `X' ≤ X_O2` and `T' ≤ T`: cooling and dilution never re-light it.
    - Hypotheses: `0 < X_OI`, `T_inf < T_OI`, `T_fb < T_OI`.
    - Kind: algebraic.
14. **`evans_envelope`: the Evans cap is non-increasing, bounded and has finite energy.**
    - In words, (i): `Q(t) = Q_act exp(-(t - t_act)/tau)` is non-increasing for `t ≥ t_act` and at most `Q_act`.
    - (ii): `tau(w) = 3.0 w^(-1.85)` is strictly decreasing in `w > 0`, so the envelope is non-increasing in spray density.
    - (iii): if each covered cell's heat release in step n after activation is at most `q_c · survives(1/tau, n dt)` with `∑ q_c = Q_act`, then the total is at most `Q_act · survives(1/tau, n dt)`; with per-cell `tau_c`, at most `Q_act · survives(1/tau_max, n dt)`.
    - (iv): the energy released after activation is at most `∑_{n<N} Q_act e^(-n dt/tau) dt ≤ Q_act dt/(1 - e^(-dt/tau)) ≤ Q_act (tau + dt)`. For example, 500 kW at `tau = 411 s` gives at most about 206 MJ. Fuel vapour already in the gas at activation adds at most its own heat content.
    - Hypotheses: `0 ≤ Q_act`, `0 < tau`, `0 < dt`, `0 < w`.
    - Kind: algebraic (`survives_add`, the geometric sum, and `1 - e^(-x) ≥ x/(1 + x)` from `Real.add_one_le_exp`). The continuous integral `Q_act tau` optionally via Mathlib's `integral_exp`.
15. **`wet_skin_pinned` and `pyrolysis_dilution_counterexample`: wet fuel cannot burn hotter than at 100 °C, but steam raises pyrolysis.**
    - In words, (i): in the existing drying step (`dw = min(m_w, C (Ts - T_B)/L)`, `Ts' = Ts - dw L/C`), if water remains after it (`m_w - dw > 0`), then `Ts' ≤ T_B`.
    - (ii): with `arrhenius_strictMonoOn`, the oxidation constant that step is at most `arrhenius(A, E, T_B)`, so the solid's heat release is bounded by `skin (1 - survives(k(T_B), dt)) heat`, about 2e-5 per second times the layer's heat content for the port's kinetics.
    - (iii), the counterexample: the pyrolysis multiplier `0.25 + 0.75 (1 - min(1, Y_O2/0.232))` is antitone in `Y_O2`. Steam that dilutes the oxygen at a hot, dry fuel surface raises its pyrolysis rate, up to four times. This documents an interaction to test, not a defect.
    - Hypotheses: `0 < C`, `0 < L`, `0 ≤ m_w`.
    - Kind: algebraic plus the existing `arrhenius_strictMonoOn`.

Before step 1 (activation): items 1 to 6. Item 7 only if the wet term is implemented. Before step 2 (Evans): item 14. Before step 3 (spray): items 8 to 13 and 15.

---

## 10. Order of work

1. **Sprinklers listed and activation only** (CPU and GPU):
   - IFC `IfcFireSuppressionTerminal` (SPRINKLER) and placed heads, with the Section 6.1 record;
   - sensor cells;
   - the exact link step on readings A and C;
   - opening times on the sprinkler card;
   - no effect on the fire.

   Tests: the closed form (3.3) to round-off; CPU and GPU parity; Vettori and Bittern brackets.
2. **Evans cap** on covered fuel cells (CPU, then GPU through `CF.loss` and one new cell field).

   Tests: item 14 numerically; a burner case whose total heat release after activation stays at or under the envelope; comparison with CFAST on a single room.
3. **Eulerian spray zone**: gas evaporation, steam, drag, and wetting with a wetted fraction calibrated so that Evans's furnishings cases are not out-suppressed. Flame-front particles culled in sprayed cells.

   Tests: item 8's budget at runtime; the energy identity per step; Fire-X's 33 kW case as a smoke test; Madrzykowski 2008 (no flashover with one head).
4. **Manual suppression** (the earlier note's step) and, optionally, Lagrangian parcels on the GPU for visuals.

---

## 11. Claims I am least sure of

- **The size of the coarse-grid error** (Section 4.3). The 0.75 and 0.7 factors are my guesses, informed by Zeng et al.'s 0.4 m result under an 8.6 m ceiling and by an assumed jet profile. Under a 2.7 m ceiling the error may be larger. Only Vettori or Bittern comparisons in the port can settle it.
- **RTI class limits and bulb ratings** (50 and 80 (m s)^1/2; 57 to 182 °C) come from secondary sources. NFPA 13 and ISO 6182-1 were not read.
- **Heskestad and Bill (1988) and Alpert (1972) were not read.** Their equations are as given by the FDS and CFAST guides and by Zeng et al. (2023), which agree with each other.
- **The FDS guide against code discrepancy** in the wet-link term, and the corrected unit of C2, rest on the master branch read on 2026-11-07. An older FDS release may differ.
- **Applying Evans per cell** and to the app's statistical "contents" is my design assumption. Evans's data are single unshielded items at 0.07 mm/s and above.
- **The per-cell evaporation fractions** (1.7 % and 9 %), the drag-against-buoyancy magnitudes and the CPU parcel cost are order-of-magnitude estimates of mine.
- **That wetting through `moisture` would stop a covered cell within a step or two** is an inference from the code, not a run.
- **Fire-X details.**
  - The transcription calls McCaffrey's NBSIR 79-1910 set-up a 1.5 by 1.5 by 3.9 m "compartment". I did not check McCaffrey's report for that.
  - Fire-X's emitter "Mass 0.1 to 1.0 kg" in Table 3 is ambiguous: per particle or per emission.
- **The Flamescapes DOI** (10.1145/3842518) still does not resolve at Crossref (404 on 2026-11-07), as the repository noted on 2026-11-05.

---

## 12. Sources and DOI checks

Each DOI was checked at `api.crossref.org/works/<doi>` on 2026-11-07. Title, authors and year are as Crossref returned them, with the read status.

| DOI / identifier | Crossref title; authors; year | Read |
|---|---|---|
| 10.1145/3763338 | Fire-X: Extinguishing Fire with Stoichiometric Heat Release; Wrede, Wagner, Mahfuz, Palubicki, Michels, Pirk; 2025 (ACM TOG 44(6)) | repository transcription, in full |
| 10.1145/3842518 | Flamescapes (Wrede, Wagner, Pałubicki, Michels, Pirk 2026): **not found at Crossref (HTTP 404)** | repository transcription, searched |
| 10.1016/0379-7112(88)90049-5 | Quantification of thermal responsiveness of automatic sprinklers including conduction effects; Heskestad, Bill; 1988 (Fire Safety Journal 14(1-2) 113-125) | not read; via FDS Technical Reference Guide |
| FMRC Serial 22485 (RC 76-T-50), no DOI | Heskestad and Smith 1976, Investigation of a new sprinkler sensitivity approval test: the plunge test | not read; citation found in a secondary reference list |
| 10.1007/BF01040244 | Methods to calculate the response time of heat and smoke detectors installed below large unobstructed ceilings; Evans, Stroup; 1986 (Fire Technology 22(1) 54-65) | not read |
| 10.1007/BF02590543 | Calculation of response time of ceiling-mounted fire detectors; Alpert; 1972 (Fire Technology 8(3) 181-195) | not read; correlations via Zeng et al. and the FDS Validation Guide |
| 10.6028/NIST.IR.5254 | Sprinkler fire suppression algorithm for HAZARD; Crossref lists "National Institute of Standards and Technology" as author (the report names David D. Evans); 1993 | **read in full** (NIST scan) |
| 10.6028/NIST.IR.4833 | A sprinkler fire suppression algorithm for the GSA engineering fire assessment system; Madrzykowski, Vettori; 1992 | not read; via Evans 1993 |
| 10.1177/104239159200400403 | A Sprinkler Fire Suppression Algorithm; Madrzykowski, Vettori; 1992 (J. Fire Prot. Eng.) | not read |
| 10.6028/NBS.IR.88-3696 | Suppression of wood crib fires with sprinkler sprays; Walton; 1988 | not read; via Evans 1993 |
| 10.6028/NBS.IR.88-3745 | Water spray suppression of fully-developed wood crib fires in a compartment; Milke, Evans, Hayes; 1988 | title only |
| 10.6028/NIST.IR.6196-1 | Sprinkler, smoke & heat vent, draft curtain interaction; McGrattan, Hamins, Stroup; 1998 | not read; via the FDS Validation Guide |
| 10.6028/NIST.IR.6253 | Effect of an obstructed ceiling on activation time of a residential sprinkler; Vettori; 1998 | not read; via the FDS Validation Guide |
| 10.6028/NIST.IR.7079 | Effect of beamed, sloped, and sloped beamed ceilings on the activation time of a residential sprinkler; Vettori; 2003 | not read; via the FDS Validation Guide |
| 10.1007/s10694-007-0009-5 | Assessing the Sprinkler Activation Predictive Capability of the BRANZFIRE Fire Model; Wade, Spearpoint, Bittern, Tsai; 2007 | not read |
| 10.1177/0734904118772306 | Using experimental sprinkler actuation times to assess the performance of Fire Dynamics Simulator; Hopkin, Spearpoint, Bittern; 2018 (J. Fire Sciences 36(4)) | abstract only |
| 10.6028/NIST.IR.7506 | Impact of a residential sprinkler on the heat release rate of a Christmas tree fire; Madrzykowski; 2008 | abstract only |
| 10.6028/NIST.IR.6439 | Reduced-scale experiments to characterize the suppression of rack-storage commodity fires; Hamins, McGrattan; 1999 | not read |
| 10.3801/IAFSS.FSS.7-457 | Reduced-scale Experiments On The Water Suppression Of A Rack-storage Commodity Fire For Calibration Of A Cfd Fire Model; Hamins, McGrattan; Crossref year 2003 (the FDS bibliography says 2002) | not read |
| 10.3801/iafss.fss.4-901 | Suppression Of Rack-storage Fires By Water; Yu, Lee, Kung (Crossref also lists a duplicate Yu and "Brown, W."); 1994 | not read |
| 10.3801/iafss.fss.3-575 | Characteristics Of Large Diffusion Flames Burning In A Vitiated Atmosphere; Morehart, Zukoski, Kubota; 1991 | not read; via the FDS Technical Reference Guide |
| 10.1115/1.1561453 | Temperature and Volumetric Fraction Measurements in a Hot Gas Laden With Water Droplets; Ruffino, di Marzo; 2003 (J. Heat Transfer 125(2)) | not read; source of C2 per FDS |
| 10.1007/s10694-023-01461-0 | Revisiting Alpert's Correlations: Numerical Exploration of Early-Stage Building Fire and Detection; Zeng, Wong, Węgrzyński, Huang; 2023 (Fire Technology 59(5)) | accepted manuscript, pp. 1 to 10 |
| 10.1007/s10694-023-01441-4 | Automatic Fire Sprinkler Activation Time with Quadratic Fire Growth; Toldrà; 2023 (Fire Technology 59(5)) | title only (an analytical t-squared counterpart to Section 3.3; worth reading before item 4) |
| 10.1007/BF02338883 | The effect of a sprinkler on the stability of a smoke layer beneath a ceiling; Bullen; 1977 (Fire Technology 13(1)) | title only |
| 10.1002/fam.810030106; 10.1002/fam.810030107 | Heat transfer from a buoyant smoke layer beneath a ceiling to a sprinkler spray, 1 (tentative theory) and 2 (experiment); Morgan; Morgan, Baines; 1979 (Fire and Materials) | titles only (Crossref search) |
| 10.1016/S0082-0784(79)80106-X; 10.1016/0379-7112(89)90017-9 | The initial convective flow in fire, and its update; Heskestad, Delichatsios; 1979, 1989 | not read (t-squared ceiling-jet correlations behind DETACT-T2) |
| 10.6028/NBS.IR.79-1910 | Purely buoyant diffusion flames; McCaffrey; 1979 | not read |
| 10.6028/NBS.IR.81-2329 | Characterizing the thermal response of fusible-link sprinklers; Evans, Madrzykowski; 1981 | not read |
| 10.6028/NIST.SP.1018, 10.6028/NIST.SP.1019 | FDS Technical Reference Guide and User's Guide. **Crossref returns the version 4 records (2006, 2004)**, so the DOI does not identify the current edition | current editions read from LaTeX sources (github.com/firemodels/fds, `Manuals/`, master, 2026-11-07); Validation Guide likewise; `Source/dump.f90` for the link update |
| CFAST Technical Reference Guide (NIST) | | LaTeX source (github.com/firemodels/cfast, `Manuals/CFAST_Tech_Ref`, master, 2026-11-07): ceiling jet, sprinkler, suppression sections |
| buildingSMART IFC 4.3 `Pset_FireSuppressionTerminalTypeSprinkler` | | `reference_schemas/psd/Pset_FireSuppressionTerminalTypeSprinkler.xml` in github.com/buildingSMART/IFC4.3.x-development (file says IFC4x3_RC4). The IFC4 ADD2 TC1 page returned HTTP 403 |
| NFPA 13 RTI definitions | | secondary web sources only (manufacturer and distributor pages) |
