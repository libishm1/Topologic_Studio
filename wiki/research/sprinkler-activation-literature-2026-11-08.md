# Sprinkler activation on a coarse grid: surface heat transfer, the hot layer, FDS's benchmark and plume resolution (2026-11-08)

Status: literature study only. No code was changed and nothing was committed. This note answers the questions raised by the validation in `sprinkler-validation-data-2026-11-07.md` (R1 to R9) and is meant to be read beside `sprinkler-activation-fix-2026-11-08.md`. Section 6 and Section 3.3 also answer two questions added during the study: how coarse-grid and zone-type tools obtain the upper-layer temperature a head sees, and whether Studio's additive "layer plus Alpert excess" is an upper or a lower bound on Evans's substitution.

How the sources were read. The FDS 6.11.1 guides (Technical Reference, User's, Validation) and the CFAST 7.7.7 guides were downloaded from the GitHub releases of firemodels/fds and firemodels/cfast and read as text. NIST and NBS reports were read from nvlpubs.nist.gov (NBSIR 83-2670, 83-2807, 85-3134, 86-3307, 82-2578; NISTIR 4705, 6178, 6976). The 2nd edition SFPE Handbook chapter "Ceiling Jet Flows" by D. D. Evans was read from the copy NIST hosts (tsapps.nist.gov, pub_id 916864) for its equations and worked example only. The FDS predictions on the NIST data sets were taken from github.com/firemodels/out (commit 1a2fc67, outputs from FDS-6.11.1-77-gc444053 and FDS-6.11.1-154-g59da822), the measurements from github.com/firemodels/exp (704e0d9), and the pairing rules from fds `Utilities/Python/FDS_validation_dataplot_inputs.csv` (master, 7ea9716). Journal papers behind paywalls were not read; where they are used, only their Crossref or OpenAlex abstract was used and this is said. The downloads and the extraction scripts were kept out of the repository. Everything read was treated as data. No paragraph of any source is reproduced; equations, numbers and short attributed phrases only.

Conventions. "Source:" marks what a document says. "My calculation" marks numbers I computed from a source's equations (scripts `htc_numbers.py`, `evans_vs_additive.py`, `dstar_lag.py`, `fds_act_stats.py`, `fds_act_subsets.py`). "Inference" marks my own reading. Air properties for my calculations: ρ = 1.204 kg/m³, cp = 1.005 kJ/(kg K), T∞ = 293.15 K, ν = 1.51e-5 m²/s, Pr = 0.71.

---

## 1. Answers in brief

1. Inert surfaces. The measured and correlated convective coefficients from a ceiling jet to a ceiling lie between about 5 and 50 W/(m² K) for Studio's room fires, highest at the plume's impingement point and falling with radius (Cooper 1982 correlation as restated by Cooper and Woodhouse 1985 and Cooper 1991; my calculation in Section 2.2). FDS's default model gives 4 to 14 W/(m² K) on ceilings and walls (about 1 on the floor) and CFAST's natural-convection rule 4 to 8 W/(m² K) for the same conditions. A separate inert coefficient of about 10 W/(m² K), range 5 to 25, or better the FDS form h = max(C|ΔT|^(1/3), (k/L) 0.0296 Re_L^0.8 Pr^(1/3)) with L = 1 m, is defensible; 200 W/(m² K) is 4 to 40 times the literature (inference, Section 2.4).
2. Evans's substitution replaces the fire by a weaker source further below the ceiling in an environment at the layer temperature (Section 3.1). Against it, Studio's additive form is neither an upper nor a lower bound: it is higher by up to 11 K for weak plumes under thin hot layers and lower by up to 19 K for strong plumes under deep layers, over 100 kW to 1 MW, 0.3 to 1.0 m layers and 35 to 100 °C (my calculation, Section 3.3). Under CFAST's own layer treatment the layer form is always the hotter one (by 0.4 to 30 K). For early activation, with thin and cool layers, the differences are under about 5 K, so the substitution form is not why heads open late; the grid's layer temperature is.
3. FDS on the four NIST sets: the FDS Validation Guide (NIST SP 1018-3, sixth edition, revision FDS-6.11.1-0-gff928db, 10 July 2026), Section 7.2.1 and Table 16.1, gives for sprinkler activation time a bias factor of 1.04 and a model relative standard deviation of 0.16 over 6 data sets and 272 points. Per set, from the FDS outputs (my extraction): Vettori 1998 median 1.02 (0.78 to 1.36), Vettori 2003 0.96 (0.62 to 1.64), Bittern 1.07 (0.96 to 1.79), UL/NFPRF Series I 1.06 (0.85 to 3.28). On the subsets Studio ran: 1.13, 0.73, 0.90 (beams), 1.06 and 0.98 to 1.09 for UL first heads with one outlier (Section 4). Grids 0.067 to 0.2 m.
4. D*/dx on Studio's grids is 0.6 to 4.8 for 50 kW to 1 MW at 0.2 to 0.45 m cells (my calculation, Section 5.1), against 5.7 to 9.8 for FDS's sprinkler validation runs and 3 to 24 in its plume resolution studies. FDS gives no acceptable minimum. No published sub-grid ceiling-jet model for coarse CFD was found; zone models use the jet correlations as their "virtual detector".
5. Large rooms: transport lag (14 to 46 s at Studio's scales, Heskestad and Delichatsios), the flattening of the jet's radial decay once a layer forms (Davis et al. 1998, hangar data), dilution that reduces the estimate to Alpert's jet over ambient, and the distance from the head to the actual plume axis all bear on it (Section 7).

---

## 2. Convective heat transfer to inert compartment surfaces before flashover

### 2.1 What the sources give

Ceiling, plume-driven (Cooper). Source: Cooper (1982, J. Heat Transfer 104, 446 to 451; abstract only read) reviewed free-jet and fire-plume data and gave "easily applicable formulae" for heat transfer from a plume-driven ceiling jet to an unconfined ceiling. The formulae are restated in NBSIR 85-3134 (Cooper and Woodhouse 1985, Eq. 6) and in NISTIR 4705 (Cooper 1991, Eqs. 13 to 18), both read:

- flux q″ = h_l (T_ad − T_s), where T_ad is the temperature an adiabatic ceiling would reach (the ceiling jet's recovery temperature) and T_s the ceiling surface;
- h_l / h̄ = 8.82 Re_H^(−1/2) Pr^(−2/3) [1 − (5.0 − 0.284 Re_H^0.2)(r/H)] for 0 ≤ r/H < 0.2, and 0.283 Re_H^(−0.3) Pr^(−2/3) (r/H)^(−1.2) (r/H − 0.0771)/(r/H + 0.279) for r/H ≥ 0.2;
- h̄ = ρ cp g^(1/2) H^(1/2) Q*^(1/3), Re_H = g^(1/2) H^(3/2) Q*^(1/3) / ν, Q* = (1 − χr) Q / (ρ cp T g^(1/2) H^(5/2)), so it is the convective heat release that enters;
- (T_ad − T_u)/(T_u Q*^(2/3)) = 10.22 − 14.9 r/H for r/H < 0.2 and 8.39 f(r/H) beyond, with f = [1 − 1.10 x^0.8 + 0.808 x^1.6] / [1 − 1.10 x^0.8 + 2.20 x^1.6 + 0.690 x^2.4], x = r/H (NISTIR 4705 Eqs. 15 and 16; my check: the two branches meet at 7.24 at r/H = 0.2).

Range of validity. Source: NBSIR 85-3134 (p. 4): the h correlation gave reasonable fluxes for Re_H of 0.73e4 to 27e4, and full-scale fires of 200 to 2000 kW convective at 2 to 3 m have Re_H of 20e4 to 50e4. Cooper and Woodhouse (1986, J. Heat Transfer 108, 822 to 826; abstract) extended the T_ad distribution from 0 ≤ r/H < 0.7 to 0 ≤ r/H ≤ 2.2. The underlying experiment is Veldman's (1975 Caltech thesis; abstract via DataCite and OpenAlex): 1.0 to 1.6 kW burners under a 0.91 m steel plate at 0.5 to 0.91 m, with h defined on the adiabatic recovery temperature.

Other ceiling data (abstracts only). You and Faeth (1979, Fire and Materials 3, 140 to 147): 50 to 7890 W fires, ceilings 58 to 940 mm up; flux nearly independent of position in the stagnation region (r < 0.2 H); plume stagnation fluxes 25 to 40 % of those of impinging jets; confinement raised fluxes. Kokkala (1991, Fire Safety Science 3, 261 to 270): 2.9 to 10.5 kW flames; stagnation flux above You and Faeth's in the flaming regions, close to it in the plume region; ceiling-jet flux in agreement with earlier work. Motevalli and Marks (1991, Fire Safety Science 3, 301 to 312): 0.5 to 2.0 kW; thermal and momentum thicknesses of the jet differ. CFAST 7.7.7 (Technical Reference Eq. 3.33) quotes their 1/e thickness as δ/H = 0.112 [1 − exp(−2.24 r/H)] for 0.26 ≤ r/H ≤ 2.0. Evans's SFPE chapter gives the jet's thickness as 5 to 12 % of H, with its temperature and velocity maxima within 1 % of H of the ceiling.

Walls. Source: Cooper (1986, NBSIR 86-3307; abstract): where the jet turns down a wall, wall heat transfer is 1 to 2.5 times the ceiling's just upstream, for 200 to 2000 kW fires 2 to 3 m below the ceiling. Below the jet, natural convection applies (CFAST and FDS rules below).

FDS's default model. Source: FDS Technical Reference Guide 6.11.1, Section 7.1.2, and User's Guide Section 8.2.2. In LES, q″ = h (Tg − Tw), with Tg the gas temperature at the centre of the first gas cell, and h = (k/L) max(Nu_free, Nu_forced, 2L/δn), L = 1 m for planar surfaces. Nu_free = 0.15 Ra^(1/3) for a cold surface facing down (a ceiling over hot gas) when Ra > 1e7 (0.54 Ra^(1/4) below), 0.52 Ra^(1/5) for a cold surface facing up, (0.825 + 0.324 Ra^(1/6))² for vertical plates, and h = 1.31 ΔT^(1/3) on a vertical surface backed by void. Nu_forced = 0.0296 Re^0.8 Pr^(1/3) for a flat plate. The 2L/δn term is the DNS limit. The User's Guide adds two warnings that bear on coarse cells: at an impingement point the tangential velocity is poorly resolved, so the default gives a "fictitiously low" h unless an impinging-jet profile is set by the user (its example: peak 50 W/(m² K), Gaussian width 0.5 m, peak about 1.3 times the area mean); and the log-law wall model "is not well-suited for buoyant flows" and needs a well-resolved near-wall flow.

CFAST's model. Source: CFAST Technical Reference Guide 7.7.7, Section 5.2. Walls and floor: h = C |Tg − Tw|^(1/3), C = 1.31 for walls and 1.52 for floor and ceiling. Ceiling, early: the flux follows Alpert's SFPE correlation q″ = 1.323 f (Q̇c/H²)(r/H)^(−1.36) with f = 0.03, integrated over the ceiling to q″_avg = 0.27 Q̇c / ((LW)^0.68 H^0.64), and h = max(q″_avg/(Tu − Ts), C |Tu − Tw|^(1/3)) so that the ceiling moves to the natural-convection rule once a layer forms.

Whole-room loss in the early stage. Source: ASET (Cooper and Stroup 1982, NBSIR 82-2578, read): the fraction λc of the total heat release lost to the room's surfaces is roughly constant early in single-room fires and lies in 0.6 to 0.9, with χr = 0.35; about 0.6 for wide rooms (span large against height) with smooth ceilings and the fire away from walls, toward 0.9 for narrow rooms, fires within about a room height of a wall, or irregular ceilings.

Values used in codes for later stages. EN 1991-1-2 uses 25 W/(m² K) on the exposed side under the standard temperature-time curve (checked only in secondary sources; the standard itself was not read; unverified). Veloo and Quintiere (2013, J. Fire Sciences 31, 410 to 423; abstract): in fully developed fires (gas to 1000 °C, total flux to 200 kW/m²), convection was about 25 % of the total and h was correlated with the temperature rise; their values were not read. Both concern post-flashover conditions and are not a basis for the pre-flashover inert coefficient (inference). Tanaka and Yamada (1999, "Reduced scale experiments for convective heat transfer in the early stage of fires") exists as an OpenAlex record without DOI; it was not read and no number from it is used.

A counterpoint. Husted and Holmstedt (2008, J. Fire Prot. Eng. 18, 29 to 54; abstract) found that ceiling heat transfer had little influence on sprinkler activation in their CFD runs of a large entertainment centre. In a large room with heads near the plume, the jet reaches the head before much heat is lost; in Studio's coarse grid the loss acts on every cell the jet crosses at once (inference, Section 2.3).

### 2.2 The numbers for Studio's cases

Cooper's coefficient (my calculation, convective heat release, ambient air properties), in W/(m² K):

| Case | Re_H | r = 0 | 0.3 m | 0.6 m | 1.0 m | 2.2 m | 3.0 m | 5.0 m |
|---|---|---|---|---|---|---|---|---|
| Vettori burner 300 kW, H = 2.13 m, χr 0.2 | 2.1e5 | 43 | 33 | 27 | 21 | 11 | 8 | 5 |
| Vettori burner 600 kW | 2.6e5 | 48 | 38 | 31 | 24 | 13 | 10 | 6 |
| Bittern chair 100 kW, H = 1.75 m, χr 0.46 | 1.1e5 | 38 | 25 | 19 | 14 | 7 | 5 | 3 |
| 1 MW, H = 2.5 m, χr 0.3 | 3.3e5 | 46 | 39 | 33 | 27 | 15 | 12 | 7 |
| UL/NFPRF 4 MW, H = 7.1 m, χr 0.35 | 1.0e6 | 29 | 28 | 28 | 27 | 24 | 20 | 14 |

Averaged over a disk of the ceiling's area centred on the plume (my calculation): 10.6 and 12.5 W/(m² K) for the 1998 room (9.2 x 5.6 m) at 300 and 600 kW, 8.7 for Bittern's room (8 x 4 m) at 100 kW, 5.6 for a 20 x 20 m room at 1 MW and H = 2.4 m. The UL case lies above the correlation's validated Re_H (Section 2.1) and is an extrapolation.

FDS's default (my calculation, L = 1 m, film properties): on a ceiling 0.3 to 1.5 m/s jets give 4.7 to 7.8 W/(m² K) (natural branch governs), 2.4 m/s gives 8.7 to 9.5, 4 m/s gives 13 to 14, for gas 20 to 120 K above the surface; on walls in still gas 3.8 to 6.1, on the floor about 1. CFAST's C|ΔT|^(1/3) gives 3.6 to 6.5 on walls and 4.1 to 7.5 on ceilings and floors for ΔT = 20 to 120 K. The median gas speeds of Studio's top cells at the measured activations were 0.64 to 2.36 m/s across the data sets (validation R4), so the FDS form would give about 5 to 9 W/(m² K) there.

CFAST's early-ceiling flux expressed as a coefficient on the layer (my construction, dividing Eq. 5.25 by an assumed Tu − Ts): 55 W/(m² K) for the 1998 room at 300 kW and 50 K, 68 at 600 kW and 80 K, 32 for Bittern at 100 kW and 30 K. These are large because the flux is driven by the jet, which is hotter than the layer mean; they show what an effective coefficient against a too-cool reference temperature has to be (inference).

Room energy budget (my calculation from validation R6.1). In the 1998-size box at 45 s on the fast tabulated ramp, the gas held 1.26 MJ (h = 200) and 4.30 MJ (h = 25) of the burner's 4.80 MJ convective input. With χr = 0.2 the total release was 6.0 MJ, so the effective λc was 0.79 at h = 200 and 0.28 at h = 25. ASET's range for a wide room with a detached fire is near 0.6. On this crude test, the coarse grid with any literature-sized h keeps too much of the heat in the gas, yet the coordinator's run at 10 W/(m² K) still reads the ceiling cells too cool (median first-head ratio about 1.25 in the 1998 room, reported to me and not checked by me). Together these say the heat is in the gas but not at the ceiling (inference; Section 6).

### 2.3 Coarse cells: what temperature the coefficient multiplies

Every correlation above multiplies a coefficient by a near-wall reference temperature: T_ad (Cooper), the first cell's centre (FDS), the layer mean (CFAST). On a 0.2 to 0.45 m grid with H ≈ 2.4 m the top cell holds the whole jet, whose 1/e thickness is 0.18 to 0.27 m at r/H = 0.5 to 2 (Motevalli and Marks via CFAST Eq. 3.33). Taking a Gaussian profile with that thickness, the top cell's mean excess is 0.71 to 0.84 of the peak at 0.2 m cells, 0.52 to 0.70 at 0.3 m and 0.36 to 0.51 at 0.45 m; under a 7.6 m ceiling it is 0.83 to 0.98 (my calculation). A cell-mean coefficient that reproduces Cooper's flux therefore needs to be about 1.2 to 2.8 times Cooper's h in a 2.4 m room (inference).

The other side of the coarse cell is speed. The exchange in `conduct` relaxes a gas cell toward its wall with a time constant τ = ρ cp δx / h per face (my reading of `flamescapes.js`, `exchange` with P.convection A). This is 1.2, 1.8 and 2.7 s at h = 200 for 0.2, 0.3 and 0.45 m cells; 9.7, 14.5 and 21.8 s at h = 25; 24, 36 and 54 s at h = 10 (my calculation). The ceiling jet crosses 2 m in about 1 to 2.5 s (speeds 0.8 to 2.7 m/s measured at the Vettori heads, validation Section 3), so at 200 W/(m² K) and 0.3 m cells the top cells give up 43 to 75 % of their excess to a cold ceiling within the transit (1 − exp(−t/τ), my calculation); the real jet, about 0.2 m thick with h of 10 to 40 W/(m² K), has a time constant of about 6 to 25 s (my calculation). This is why 200 removes the jet before it reaches a head and a smaller cell does not help (validation R7: τ falls with δx).

None of the sources gives a wall function designed for a cell that contains the whole jet. FDS uses its correlation on the first cell regardless of δn, warns that it under-predicts at impingement and offers a user-set impingement profile; CFAST avoids the issue by integrating Alpert's flux correlation over the ceiling; Cooper (1991) evaluates h_l and T_ad from the plume (or its substitute source in a layer), not from any gas cell.

### 2.4 Recommendation for a separate inert-surface coefficient (inference)

- Keep 200 W/(m² K) for fuel surfaces only, as the calibration for flame spread it was.
- For walls, floors and ceilings use the FDS default form with L = 1 m on the cell's own speed and temperature difference: h = max(Nu_free, 0.0296 Re_L^0.8 Pr^(1/3)) k/L, Nu_free by orientation as in Section 2.1. It gives about 4 to 14 W/(m² K) in Studio's rooms and needs only the cell's speed, which the solver has. A constant fallback of 10 W/(m² K) is consistent with it and with Cooper's area-mean (5.6 to 12.5).
- Range to test: 5 to 25 W/(m² K) for the constant. 25 is the upper end: it equals Cooper's value about 1 m from the axis of a 300 to 1000 kW fire, the Eurocode post-flashover value, and the cell-mean correction at r/H ≈ 1 for 0.3 m cells.
- Optional, for heat transfer to the ceiling over the plume: Cooper's h_l(r) in the impingement zone (about 30 to 50 W/(m² K) within r/H < 0.2 for 100 kW to 1 MW), as FDS's impinging-jet option does by hand.
- Check the room budget against ASET's λc of 0.6 to 0.9 (0.6 for broad rooms with detached fires). A run that loses much less than 0.6 of the total release in the early stage is losing too little, whatever h is used, and the ceiling reading should then be checked for stratification (Section 6) rather than tuned through h.

---

## 3. Activation under a hot layer: Evans and successors

### 3.1 Evans's substitution

Source: Evans (1985, Fire Safety Journal 9, 147 to 155; read through its record and abstract only) estimates actuation in compartments of about 24 to 24,300 m² by computing a substitute source for the fire that stands in an environment at the upper-layer temperature, for RTI 25, 100 and 400 (m s)^(1/2), with the unconfined ceiling as the limiting case (firedoc.nist.gov record). NBSIR 83-2807 (Evans 1984, abstract read) states that the warm layer's effect on plume and jet is handled by substituting an equivalent point source whose strength and location are given relative to the real fire. The equations, as given in Evans's SFPE Handbook chapter (2nd ed., Eqs. 23 to 26) and checked against its worked example:

- Q*_I1 = Q / (ρ∞ cp T∞ g^(1/2) Z_I1^(5/2)), Z_I1 the distance from the fire to the layer interface;
- Q*_I2 = [(1 + C_T Q*_I1^(2/3)) / (ξ C_T) − 1/C_T]^(3/2), ξ = T_u/T_l, C_T = 9.115;
- Z_I2 = [ξ Q*_I1 C_T / (Q*_I2^(1/3) ((ξ − 1)(β² + 1) + ξ C_T Q*_I2^(2/3)))]^(2/5) Z_I1, β² = 0.913;
- Q_2 = Q*_I2 ρ∞ cp T∞ g^(1/2) Z_I2^(5/2); H_2 = H_1 − Z_I1 + Z_I2;
- then the unconfined-ceiling correlation is used with Q_2, H_2 and the upper layer as the ambient: T_jet = T_u + ΔT(Q_2, H_2, r; T_u, ρ_u), with ρ_u T_u = ρ∞ T∞.

Check. The chapter's example (2500 kW, fire 3 m below a 2 m deep 50 °C layer, ceiling 5 m) gives Q*_I1 = 0.1452, Q*_I2 = 0.1179, Z_I2 = 3.161 m, Q_2 = 2313 kW, H_2 = 5.161 m and a jet excess of 106 K at r = 4 m. My implementation gives Q_2 = 2312 kW, H_2 = 5.161 m and 105 K with the same (Heskestad and Delichatsios) correlation. Cooper's alternative (partial penetration of the plume, the chapter's Eqs. 27 to 31) gives 2308 kW and 5.176 m for the same case (source: the chapter).

In equation form for comparison with Studio: Studio's estimate is T = T_layer + ΔT_Alpert(Q, H, r) with ΔT over ambient (`suppression.js`, `activationFrame`). Evans's is T = T_u + (T_u/T∞) ΔT_Alpert(Q_2, H_2, r), since the excess scales with the ambient temperature at fixed Q* when ρT is constant. The ratio of the two excesses is (T_u/T∞)(Q_2/Q)^(2/3)(H_1/H_2) for r/H > 0.18 (my derivation).

The plume stops reaching the ceiling in Evans's model when Q*_I2 falls to zero, at T_u/T_l ≥ 1 + 9.115 Q*_I1^(2/3) (my rearrangement): for 50 kW 1.5 m below the interface at T_u ≥ 192 °C, for 100 kW at 293 °C (my calculation). The additive form keeps adding Alpert's excess there.

### 3.2 Successors

- CFAST 7.7.7 (Technical Reference Sections 3.4, 3.5 and 6.1). The plume's centreline excess in the upper layer uses Heskestad's correlation with T_u and ρ_u and a virtual origin moved to z′0 = z_I − (T_u/T_l)^(3/5)(z_I − z0) (Eqs. 3.27 and 3.28). The jet is T_cj = T_u + ΔT0(H) for r/H ≤ 0.2 and T_u + 0.182 ΔT0(H)(0.225 + 0.27 r/H)^(−4/3) beyond (Eq. 3.30, with T∞ replaced by T_u), and the jet speed follows Eq. 3.31. Heads below the jet or in rooms without a fire see the layer temperature at 0.1 m/s. The link is integrated semi-implicitly (Eq. 6.2).
- Cooper (1990, Fire Safety Journal 16, 137 to 163; and the LAVENT reports NBSIR 88-3734, NISTIR 89-4122) and Cooper (1991, NISTIR 4705): the substitute-source idea is applied to the jet and to ceiling heat transfer in a two-layer zone, with the jet's temperature profile below the ceiling (NISTIR 4705 abstract and Section "Calculating the Convective Heat Transfer to the Ceiling", read).
- Davis, Notarianni and Tapper (1998, NISTIR 6178, read), the basis of Davis's JET model (NISTIR 6324, 1999; scanned report, not read). Findings: with no hot layer Alpert's and Heskestad's correlations agreed with the 15 m and 22 m hangar data; once a layer formed they "substantially" under-predicted plume and jet temperatures. FPEtool applies Evans's method once the layer is deeper than 0.12 H and takes the layer temperature and depth from a single-room zone model, but uses Alpert's radial decay, which falls too fast when a layer is present. Entrainment of layer gas, and for large fires a radiative fraction that varies with fire diameter, must be modelled to predict the centreline temperature. In a 2.8 MW JP-5 test at the Keflavik hangar the power-law fits printed on their Fig. 4 flatten from about r^(−0.66) at 80 s, through r^(−0.53), to r^(−0.16) to r^(−0.27) at the later times (150 to 300 s), against r^(−0.67) for Alpert; the authors attribute it to the layer cutting the jet's heat loss and its entrainment of cool air.
- Gandhi (1995, Fire Technology 31, 137 to 157) gives temperature and velocity correlations in room fires for actuation time; abstract not available, not read.

### 3.3 Is the additive form a bound? (answer to the added question b)

My calculation (`evans_vs_additive.py`): Alpert's correlation in both, fire 2.1 m below the ceiling, head at r = 2.2 m, Q = 100, 250, 500 and 1000 kW, layer depths 0.3, 0.6 and 1.0 m, layer at 35, 60 and 100 °C (36 cases).

| | Evans minus additive | Ratio of excesses, Evans/additive |
|---|---|---|
| Range over the 36 cases | −10.9 to +19.0 K | 0.67 to 1.13 |
| 100 kW, 0.3 m layer at 100 °C (weak plume, thin hot layer) | −10.9 K | 0.67 |
| 250 kW, 0.6 m at 60 °C | −1.3 K | 0.96 |
| 500 kW, 0.6 m at 60 °C | +1.3 K | 1.02 |
| 1 MW, 1.0 m at 100 °C (strong plume, deep layer) | +19.0 K | 1.13 |

So the additive form is an upper bound on Evans for weak plumes under thin, hot layers (where Evans's Q_2/Q falls to 0.44) and a lower bound for strong plumes under deep layers (where the T_u/T∞ factor wins). For 35 °C layers, the early-activation regime, the two agree within 4 K. With the Heskestad and Delichatsios correlation in the SFPE example the additive excess is 103 K against Evans's 105 K. Under CFAST's treatment (Section 3.2), which keeps the full heat release and raises the excess by T_u/T∞, the layer form exceeds the corresponding additive form in all 36 cases, by 0.4 to 30 K. Against the hangar data the Alpert- and Evans-type predictions were low once a layer formed (Davis et al. 1998). The weight of evidence is that at Studio's scales the additive form is close to Evans, may err either way by about 10 % of the excess, and is more likely low than high in a well-developed layer (inference). It cannot explain the late openings; the layer temperature it is fed can.

---

## 4. What FDS achieves on the same NIST data

Source: FDS Validation Guide, NIST SP 1018-3, sixth edition, revision FDS-6.11.1-0-gff928db, dated 10 July 2026 (release asset of FDS-6.11.1). Section 7.2.1 (Fig. 7.56) and Table 16.1: sprinkler activation time, 6 data sets (Bittern; UL/NFPRF Series I, Series II and Group A commodity; Vettori flat; Vettori sloped), 272 points, experimental relative standard deviation 0.06, model relative standard deviation 0.16, bias factor 1.04. Section 4.3.2 explains the 0.06: propagated HRR, RTI and activation temperature uncertainty, matching the 6 % scatter of Vettori's replicates. The guide's statistics are bin-weighted (regressogram over 10 bins, Chapter 4). For UL/NFPRF only the first activation is compared because the spray makes later ones "somewhat random" (Section 7.2.1); the number of activations has its own statistics (7.2.2: 3 sets, 38 points, σ_M 0.31, bias 1.19). For context, Table 16.1 gives ceiling-jet temperature σ_M 0.13 and bias 1.08 (22 sets), plume temperature 0.42 and 1.00 (16 sets), and hot-layer temperature 0.09 to 0.16 and 1.07 to 1.16.

Per data set (my extraction, `fds_act_stats.py`: activation is the last time the device is still unactuated, as the config's comparison window defines it; statistics unweighted; 273 pairs against the guide's 272, the difference not traced; pooled unweighted bias 1.02 and σ_M 0.18 against the guide's 1.04 and 0.16):

| Data set (FDS grid) | Pairs | Median M/E | Range | Geometric mean | SD of ln(M/E) | Early |
|---|---|---|---|---|---|---|
| Vettori 1998, NISTIR 6253 (0.096 x 0.104 x 0.081 m) | 68 | 1.02 | 0.78 to 1.36 | 1.02 | 0.13 | 46 % |
| of which smooth | 45 | 1.06 | 0.80 to 1.36 | 1.05 | 0.13 | 40 % |
| Vettori 2003, NISTIR 7079 (0.1 x 0.1 x 0.098 m) | 126 | 0.96 | 0.62 to 1.64 | 0.96 | 0.18 | 57 % |
| Bittern 2004 (0.067 m) | 41 | 1.07 | 0.96 to 1.79 | 1.12 | 0.14 | 17 % |
| UL/NFPRF Series I, NISTIR 6196-1 (0.2 m), first head | 21 | 1.06 | 0.85 to 3.28 | 1.09 | 0.27 | 43 % |
| UL/NFPRF Series II | 12 | 0.82 | 0.67 to 1.06 | 0.82 | 0.13 | 92 % |
| UL/NFPRF commodity | 5 | 1.40 | 1.12 to 1.61 | 1.34 | 0.15 | 0 % |

On the subsets Studio ran (validation R4), with Studio's medians at its 200 and 25 W/(m² K) for comparison:

| Subset | FDS median (range), heads | Studio, h = 200 | Studio, h = 25 |
|---|---|---|---|
| 1998 smooth, detached (tests 1 to 9, heads 1 and 2) | 1.13 (0.96 to 1.36), 18 | 1.68 | 1.31 |
| 2003 flat smooth, detached (1 to 4) | 0.73 (0.64 to 0.92), 12 | 1.09 | 0.86 |
| 2003 flat, two beams, detached (13 to 16) | 0.90 (0.86 to 1.09), 9 | none opened | 1.28 |
| Bittern centre chair (3, 4, 8, 10, 12 to 15) | 1.06 (0.97 to 1.65), 16 | 1.18 | 1.07 |
| UL I-17, I-18, I-20, I-22, first head | 1.09, 1.08, 3.28, 0.98 | 1.09 (first heads) | 1.08 |

Input differences to keep in mind (source: the FDS inputs read from firemodels/fds): Vettori runs use RTI 56 and the default C = 0, where Studio used 0.25 (C = 0.25 delays the replayed link by about 4 %, validation R2); both use the tabulated HRR ramp; Bittern uses RTI 36 or 95 with C 0.4 and the link 2 cm below the ceiling; UL uses RTI 148, C 0.7, 74 °C.

Two readings (inference). FDS is also early on the 2003 smooth set (0.73), so Studio's early result there at 25 W/(m² K) (0.86) points to the inputs (the tabulated ramp or the time origin, as the validation note's energy check suspected) rather than to the solver. And FDS opens the beamed heads at 0.90, where Studio opened none at 200 W/(m² K). Suggested target metrics for Studio's estimate, per subset: median within about ±0.15 of FDS's median (that is, about 1.0 to 1.3 for the 1998 set, 0.6 to 0.9 for the 2003 smooth set, 0.9 to 1.2 for Bittern and UL first heads), every measured head opening within twice the measured time, and an overall SD of ln(M/E) of 0.2 or less, against FDS's 0.13 to 0.18 on the room sets.

---

## 5. Plume and ceiling-jet resolution on coarse grids

### 5.1 The plume resolution index

Source: FDS User's Guide Eq. 6.2 and Validation Guide Eq. 3.40: D* = (Q / (ρ∞ cp T∞ √g))^(2/5), with Q the total heat release; D*/δx is the number of cells across the fire's characteristic diameter. The User's Guide states that the validation runs span two orders of magnitude of D*/δx and that taking any of them as an acceptable minimum is inappropriate. The Validation Guide (Section 2) reports that Ma and Quintiere and NIST found the near-flame plume grid-dependent, especially at low Q*, and D*/δx critical. Its resolution studies use D*/δx of 3, 6, 12 and 24 (McCaffrey plume, Section 6.1.4) and 5, 10 and 20 (flame height, Section 6.2.1). Its Table 3.56 lists D*/δx of 12 for Vettori flat, 10 for Vettori sloped and 8 to 12 for UL/NFPRF (D* from the peak heat release); Bittern is not listed.

My calculation:

| Q (kW) | D* (m) | D*/δx at 0.2 m | 0.3 m | 0.45 m | 0.507 m |
|---|---|---|---|---|---|
| 50 | 0.29 | 1.4 | 1.0 | 0.6 | 0.6 |
| 100 | 0.38 | 1.9 | 1.3 | 0.8 | 0.8 |
| 250 | 0.55 | 2.8 | 1.8 | 1.2 | 1.1 |
| 500 | 0.73 | 3.6 | 2.4 | 1.6 | 1.4 |
| 1000 | 0.96 | 4.8 | 3.2 | 2.1 | 1.9 |
| 2000 | 1.27 | 6.3 | 4.2 | 2.8 | 2.5 |
| 4400 | 1.73 | 8.7 | 5.8 | 3.9 | 3.4 |

At the heat release that matters for activation, the FDS validation runs sit at D*/δx of 5.9 to 7.8 (Vettori, 300 to 600 kW on 0.1 m cells), 5.7 to 6.7 (Bittern, 100 to 150 kW on 0.067 m) and 8.7 (UL, 4.4 MW on 0.2 m) (my calculation). Studio's validation runs were at about 1.3 (100 kW) to 2.4 (500 kW) in the rooms on 0.3 m cells and 3.4 in UL, three to five times coarser. A ceiling-jet LES for sprinkler activation with FireFOAM (Chatterjee et al. 2014, Fire Safety Science 11, 458 to 471; read) needed 3 cm cells near the plume and ceiling for mesh independence at 1500 and 4500 kW convective and 1.5 cm above 5000 kW, a D*/δx of about 38 at 1500 kW (my calculation).

What D*/δx is needed for activation is not stated in any source read. FDS's sprinkler results (Section 4) come from D*/δx of about 6 to 10, which is the only demonstrated range (inference). Studio's open-plume diagnostic (validation R6.2), 4.6 K against Heskestad's 85 to 105 K at 2.25 m for 100 kW on 0.3 m cells, is at D*/δx = 1.3.

### 5.2 Sub-grid jets and "virtual detectors"

- FDS has no sub-grid ceiling-jet model for links: a sprinkler reads the gas in its cell (Technical Reference and User's Guide; in the Vettori inputs the link at 2.375 m lies in the top 0.081 m cell). Husted and Holmstedt (2008; abstract) found FDS 4.07 more sensitive than CFX to where the sprinkler sits in the vertical grid, and a zone model with a ceiling-jet formula 10 to 20 % slower than the CFD when the head is close to the ceiling.
- Zone models are the established "virtual detector": CFAST evaluates the link on the plume and jet correlations with the layer temperature (Section 3.2); FPEtool applies Evans's method on the zone's layer (Davis et al. 1998); JET adds layer-dependent radial decay and ceiling heat loss (NISTIR 6178).
- Węgrzyński et al. (2020, Energies 13, 1450; abstract) integrate the RTI equation in every cell of an ANSYS Fluent run to map activation time in 3D; it is a resolved-field method, not a sub-grid one.
- Zeng et al. (2023, Fire Technology 59, 2925 to 2948), the paper `suppression.js` cites for Alpert's SI form, compares FDS with Alpert's correlations for early-stage detection; its abstract could not be read and no number from it is used.
- I found no published sub-grid ceiling-jet model for coarse CFD or a zone-CFD hybrid built for activation (searches of NIST, IAFSS and general literature; a negative finding from a limited search). Studio's "estimate" (grid layer plus a correlation jet) is itself such a hybrid, closest in spirit to FPEtool's use of Evans on a zone layer (inference).

---

## 6. The upper-layer temperature when the grid smears the stratification (answer to the added question a)

What the sources do.
- Reduce a column profile to two zones. Janssens and Tran (1992, J. Fire Sciences 10, 528 to 555; abstract) derived interface height and uniform upper and lower temperatures from measured profiles while conserving mass. FDS implements it as the LAYER HEIGHT and UPPER TEMPERATURE devices (User's Guide Section 22.10.7): with I1 = ∫T dz and I2 = ∫(1/T) dz over the column, z_int = T_l (I1 I2 − H²)/(I1 + I2 T_l² − 2 T_l H), T_l the lowest cell, and (H − z_int) T_u = ∫ from z_int to H of T dz. This is how FDS's hot-layer validation (Table 16.1) is scored.
- Compute the layer with a zone energy balance and put the jet on top. CFAST solves the upper layer's mass and energy and adds the jet correlation over T_u (Section 3.2). FPEtool takes T_u and the interface from a single-room zone model and applies Evans (Davis et al. 1998). ASET does the same with a constant loss fraction λc (Section 2.1). Cooper's CEILHT (NISTIR 4705) computes the jet and the ceiling heat transfer from the zone's T_u and the substitute source.
- Read the device's own cell (FDS). On FDS's 0.07 to 0.2 m cells the top cell sits inside the jet, which is what the validation in Section 4 rests on.

No source read describes reading below the top cell to recover a smeared layer. In a smeared profile the cell below is no hotter (inference).

What this suggests for Studio (inference). The coordinator's result (Section 2.2) says the ceiling cells are too cool even when the walls take little heat, and the budget says the gas keeps more heat than ASET's λc allows. The energy is then present but spread too deep, which a column-integral layer would expose (z_int low, T_u low) but not cure. Two remedies follow from the zone practice:
1. A zone-type layer for the estimate. Take the gas enthalpy rise of the fire's room from the grid (it conserves energy whatever the profile) and assign it to an upper layer whose depth comes from a plume-filling model (Heskestad entrainment, as CFAST and ASET do), or from the Janssens and Tran interface where that is sharp. For scale, 4.3 MJ (validation R6.1 at h = 25) spread over a 0.9 m layer of the 1998 room (46 m³) is about 77 K (my calculation; an upper bound, because the lower part of the room holds some of it).
2. Then apply Evans (Section 3.1) or keep the additive form (Section 3.3) on that layer.
This keeps the grid for transport between rooms and for the spray, and takes from the zone model only what the coarse grid cannot resolve. It is the FPEtool and CFAST arrangement with the grid's energy in place of a zone's.

---

## 7. Why heads open late in large rooms, and what corrects it

- Transport lag. Source: Evans's SFPE chapter (Heskestad and Delichatsios 1979): for t-squared fires the jet excess is zero until t*_2 = 0.954 (1 + r/H), with t*_2 = t / (A^(−1/5) α^(−1/5) H^(4/5)), A = g/(cp T∞ ρ∞), and afterwards it is the steady value scaled by ((t − t_f)/t)^(4/3). My calculation: the lag is 14, 18 and 24 s at H = 2.4 m, r = 2.2 m for fast, medium and slow fires, 23 to 40 s at H = 2.7 m, r = 5 m, and 27 to 46 s at H = 3.5 m, r = 6 m. Studio's estimate is quasi-steady and has no lag, so the lag would make it early, not late. It does mean a far head in a large room really does open tens of seconds after a near one.
- Layer dilution. In a large room the grid's layer adds little and the estimate falls back to Alpert's jet over ambient. On Studio's own replay that alone opened the 1998 heads at 2.07 times the measured time and UL's at 1.26 (validation R6.4), so the fallback is late. Davis et al. (1998) found Alpert's radial decay too steep once a layer forms in a 15 m to 22 m hangar (Section 3.2), so far heads see more than Alpert predicts. Correction (inference): a zone-type layer (Section 6) and, where the layer is developed, a flatter radial decay as in JET.
- Fire position. Cooper (1991, NISTIR 4705) treats the fire's position in the room as a parameter in its own right for ceiling heat transfer and detection, and ASET's λc rises toward 0.9 for fires near walls. Studio measured r between cell centres (validation R6.5) and from the first item; in a spreading fire the plume axis moves. Correction (inference): r from each head to the heat-release-weighted centroid of the burning cells, at the head's true position. This matches D2 of the activation-fix note.
- Plume tilt and wander. Studio's open-plume run (validation R6.2) saw the hottest cell move 1 to 4 m in 10 s at D*/δx = 1.3. I found no source that quantifies plume wander on coarse grids for activation; FDS's plume temperature σ_M of 0.42 (Table 16.1) at much finer grids shows how hard the plume is even there.
- Geometry. The activation-fix note (D1) found suspended ceilings missing from the grid in the NIBS models, which lifts the heads and the jet; no literature is needed for that, but it dominates the clinic case.

---

## 8. Not verified, and gaps

- EN 1991-1-2's 25 W/(m² K): from secondary sources only.
- Cooper 1982, Evans 1985, Gandhi 1995, Ma and Quintiere 2003, Veloo and Quintiere 2013, Zeng et al. 2023, Hopkin et al. 2018, Wade et al. 2007, Husted and Holmstedt 2008: paywalled; only bibliographic records and, where available, abstracts were read. Cooper's correlation was read in NBSIR 85-3134 and NISTIR 4705, Evans's in his NIST-hosted SFPE chapter.
- Tanaka and Yamada (1999): record found, paper not read.
- Davis's JET report (NISTIR 6324) is a scan without a text layer; its contents are known here only through NISTIR 6178.
- The per-set FDS statistics are my extraction; they reproduce the guide's pooled numbers to within 0.02 in bias and σ_M, not exactly, because the guide weights by bins and has one point fewer.
- The coordinator's 10 W/(m² K) result is reported, not checked.
- ASET's λc range is for single rooms with flaming fires and χr = 0.35; applying it to the methane burner (χr 0.2) is approximate.

---

## 9. References and DOI checks

Every DOI was queried on 2026-11-08 at api.crossref.org/works and at api.openalex.org/works/doi (no email or identifier sent). "Both" means both returned the matching title, authors and year. Veldman's thesis carries a DataCite DOI, so DataCite stood in for Crossref.

| Reference | DOI | Crossref | OpenAlex |
|---|---|---|---|
| Alpert 1972, Fire Technology 8, 181 to 195 | 10.1007/BF02590543 | yes | yes |
| Alpert 1975, Combust. Sci. Technol. 11, 197 to 213 | 10.1080/00102207508946699 | yes | yes |
| Cooper 1982, J. Heat Transfer 104, 446 to 451 | 10.1115/1.3245113 | yes | yes |
| Cooper and Woodhouse 1986, J. Heat Transfer 108, 822 to 826 | 10.1115/1.3247018 | yes | yes |
| Cooper and Woodhouse 1985, NBSIR 85-3134 | 10.6028/NBS.IR.85-3134 | yes | yes |
| Cooper 1986, NBSIR 86-3307 | 10.6028/NBS.IR.86-3307 | yes | yes |
| Cooper 1991, NISTIR 4705 | 10.6028/NIST.IR.4705 | yes | yes |
| Cooper 1985, 20th Symp. (Int.) Combust., 1567 to 1573 | 10.1016/S0082-0784(85)80651-2 | yes | yes |
| Cooper 1990, Fire Safety J. 16, 137 to 163 | 10.1016/0379-7112(90)90030-I | yes | yes |
| Davis and Cooper 1989, NISTIR 89-4122 | 10.6028/NIST.IR.89-4122 | yes | yes |
| Cooper 1982, Fire and Materials 6, 135 to 144 (ASET model) | 10.1002/fam.810060307 | yes | yes |
| Cooper and Stroup 1982, NBSIR 82-2578 (ASET program) | 10.6028/NBS.IR.82-2578 | yes | yes |
| Veldman 1975, Caltech thesis | 10.7907/NJR4-M471 | DataCite yes (Crossref 404, not a Crossref DOI) | yes |
| You and Faeth 1979, Fire and Materials 3, 140 to 147 | 10.1002/fam.810030305 | yes | yes |
| Kokkala 1991, Fire Safety Science 3, 261 to 270 | 10.3801/IAFSS.FSS.3-261 | yes | yes |
| Motevalli and Marks 1991, Fire Safety Science 3, 301 to 312 | 10.3801/IAFSS.FSS.3-301 | yes | yes |
| Evans 1983, NBSIR 83-2670 | 10.6028/NBS.IR.83-2670 | yes | yes |
| Evans 1984, NBSIR 83-2807 | 10.6028/NBS.IR.83-2807 | yes | yes |
| Evans 1984, Fire Technology 20, 39 to 63 | 10.1007/BF02384163 | yes | yes |
| Evans 1985, Fire Safety J. 9, 147 to 155 | 10.1016/0379-7112(85)90002-5 | yes | yes |
| Evans and Stroup 1986, Fire Technology 22, 54 to 65 | 10.1007/BF01040244 | yes | yes |
| Heskestad and Delichatsios 1979, 17th Symp. (Int.) Combust., 1113 to 1123 | 10.1016/S0082-0784(79)80106-X | yes | yes |
| Heskestad 1984, Fire Safety J. 7, 25 to 32 | 10.1016/0379-7112(84)90005-5 | yes | yes |
| Beyler 1984, Fire Technology 20, 5 to 16 | 10.1007/BF02389990 | yes | yes |
| Gandhi 1995, Fire Technology 31, 137 to 157 | 10.1007/BF01040710 | yes | yes |
| Davis, Notarianni and McGrattan 1996, NISTIR 5927 | 10.6028/NIST.IR.5927 | yes | yes |
| Gott et al. 1997, NIST TN 1423 | 10.6028/NIST.TN.1423 | yes | yes |
| Davis, Notarianni and Tapper 1998, NISTIR 6178 | 10.6028/NIST.IR.6178 | yes | yes |
| Davis 1999, NISTIR 6324 (JET) | 10.6028/NIST.IR.6324 | yes | yes |
| Davis et al. 2003, NISTIR 6976 | 10.6028/NIST.IR.6976 | yes | yes |
| McCaffrey, Quintiere and Harkleroad 1981, Fire Technology 17, 98 to 119 | 10.1007/BF02479583 | yes | yes |
| Janssens and Tran 1992, J. Fire Sciences 10, 528 to 555 | 10.1177/073490419201000604 | yes | yes |
| McGrattan, Baum and Rehm 1998, Fire Safety J. 30, 161 to 178 | 10.1016/S0379-7112(97)00041-6 | yes | yes |
| Ma and Quintiere 2003, Fire Safety J. 38, 467 to 492 | 10.1016/S0379-7112(02)00082-6 | yes | yes |
| Veloo and Quintiere 2013, J. Fire Sciences 31, 410 to 423 | 10.1177/0734904113479001 | yes | yes |
| Husted and Holmstedt 2008, J. Fire Prot. Eng. 18, 29 to 54 | 10.1177/1042391507080813 | yes | yes |
| Wade et al. 2007, Fire Technology 43, 175 to 193 | 10.1007/s10694-007-0009-5 | yes | yes |
| Hopkin, Spearpoint and Bittern 2018, J. Fire Sciences 36, 342 to 361 | 10.1177/0734904118772306 | yes | yes |
| Chatterjee et al. 2014, Fire Safety Science 11, 458 to 471 | 10.3801/IAFSS.FSS.11-458 | yes | yes |
| Węgrzyński et al. 2020, Energies 13, 1450 | 10.3390/en13061450 | yes | yes |
| Zeng et al. 2023, Fire Technology 59, 2925 to 2948 | 10.1007/s10694-023-01461-0 | yes | yes |
| Maragkos and Beji 2021, Appl. Sci. 11, 5240 (review of convective heat transfer in fire LES; abstract only, centred on fuel surfaces) | 10.3390/app11115240 | yes | yes |
| Lattimer, SFPE Handbook 5th ed., Heat Transfer from Fires to Surfaces (not read) | 10.1007/978-1-4939-2565-0_25 | yes | yes |
| Alpert, SFPE Handbook 5th ed., Ceiling Jet Flows (not read; cited through CFAST) | 10.1007/978-1-4939-2565-0_14 | yes | yes |
| Vettori 1998, NISTIR 6253 | 10.6028/NIST.IR.6253 | yes | yes |
| Vettori 2003, NISTIR 7079 | 10.6028/NIST.IR.7079 | yes | yes |
| McGrattan, Hamins and Stroup 1998, NISTIR 6196-1 | 10.6028/NIST.IR.6196-1 | yes | yes |
| CFAST 7 Technical Reference Guide, NIST TN 1889v1 (2015 record; 7.7.7 edition read from the release) | 10.6028/NIST.TN.1889v1 | yes | yes |
| FDS guides, NIST SP 1018 series (DOI resolves to the version 4 record of 2006; the 6.11.1 editions were read from the release) | 10.6028/NIST.SP.1018 | yes, version 4 record | yes, version 4 record |

Without DOI: Evans, "Ceiling Jet Flows", SFPE Handbook 2nd ed. (1995), Section 2 Chapter 4, NIST-hosted copy (tsapps.nist.gov pub_id 916864); EN 1991-1-2 (not read); Tanaka and Yamada 1999 (not read); Bittern 2004 thesis (ir.canterbury.ac.nz handle 10092/14748, not read). Unverified DOIs: none used. Three DOIs I first guessed were wrong and are not used: 10.1002/fam.810060306 (a different paper), 10.1016/S0379-7112(02)00088-7 (a different paper), 10.1007/BF02390046 (a different paper).
