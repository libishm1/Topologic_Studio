# A quasi-steady two-zone fire model on the IFC room graph (specification, 2026-10-28)

This specifies gap LG8 of the gap analysis (`gap-analysis-2026-10-27/README.md`): a lightweight fire model generated from an IFC model and coupled to Studio's routers, giving a gas temperature in degrees C on a physical clock, which the present graph models do not (their heat is an index). It is written so that the implementation (`src/core/rooms.js`, `src/core/zone.js`) and the Lean module (`formal/fireegress/FireEgress/Zone.lean`) can be built from it independently and then checked against each other. Section 3.4 of `fire-egress-methods-2026-10-27.md` derives the full CFAST-style system; this is a deliberately simpler quasi-steady form of it, built from correlations whose units are already checked in `FireDynamics.lean` (FD-02 t-squared growth, FD-03 Babrauskas flashover, FD-07 virtual origin, FD-08 plume mass, FD-17 MQH).

## 1. The room graph (from IFC)

- Rooms are IfcSpace elements. For room k: floor area A_f,k (m², upward-facing area of the space's lowest surfaces), floor level z_f,k and height H_k (m, from the space mesh), internal surface area A_T,k = 2 A_f,k + P_k H_k (m²; P_k the footprint perimeter, or 4 sqrt(A_f,k) when unknown) minus the room's door areas.
- Lattice nodes are assigned to rooms: a node belongs to room k when it lies in k's footprint at k's floor level (within 0.5 m vertically). Nodes in no footprint take the room of the nearest assigned node along the graph (multi-source breadth-first search); nodes in a part with no room get no fire exposure.
- Doors join rooms: a graph door (lattice.js doorReport, with its two sides) joins the rooms of the nodes on its two sides; a door with one side outside every room, or flagged as an exit (ifcSemantics.js), vents to the outside. Door width W_d and height H_d (m): IfcDoor OverallWidth and OverallHeight when given (in the file's length unit, scaled), else the width measured from the door mesh and H_d = 2.0 m. Doors are assumed open.

## 2. The fire (room k_f, the room of the fire origin)

Heat release rate, t seconds after ignition (kW):
  (F1) Q(t) = min(α t², Q_max),
with growth coefficient α (kW s⁻²; slow 0.00293, medium 0.0117, fast 0.0469, ultra-fast 0.188, FD-02) and Q_max = min(Q_user, 1500 A_o √H_o) (ventilation limit, kW, with A_o = Σ W_d H_d over k_f's doors and H_o their area-weighted height). Convective part Q_c = χ_c Q with χ_c = 0.7. Fire diameter D = sqrt(4 Q_max / (π q″)), q″ = 500 kW m⁻². Virtual origin (FD-07): z_0(t) = 0.083 Q(t)^(2/5) − 1.02 D (m, above the floor).

Upper-layer temperature rise in the fire room (MQH, FD-17; K):
  (F2) ΔT_U,f(t) = 6.85 (Q(t)² / (A_o √H_o h_k(t) A_T))^(1/3),
with the lining's effective conductance h_k(t) = sqrt(k ρ c / t) / 1000 for t < t_p and k / (1000 δ) after (kW m⁻² K⁻¹), t_p = (ρ c / k)(δ/2)². Default lining gypsum board: k = 0.48 W m⁻¹ K⁻¹, ρ = 1440 kg m⁻³, c = 840 J kg⁻¹ K⁻¹, δ = 0.016 m (t_p ≈ 161 s). A room with no door gets a nominal leakage vent A_o = 0.1 m², H_o = 2.0 m.
  (F3) Flashover: from the first time ΔT_U,f ≥ 500 K (Babrauskas's criterion; FD-03 gives the matching heat release), ΔT_U,f = 900 K for all later times.

## 3. Smoke layers

Each room has a layer interface height z_k(t) above its floor, starting at z_k(0) = H_k, and an upper-layer temperature T_U,k = T_a + ΔT_U,k (K; T_a = 293.15 K). Gas density ρ_U,k = 353 / T_U,k (kg m⁻³); ambient ρ_a = 353 / T_a.

Plume mass flow into the fire room's layer (FD-08, kg s⁻¹), at height s = z_f(t) − z_0(t) above the virtual origin, s ≥ 0.1 m:
  (S1) ṁ_p(t) = 0.071 Q_c^(1/3) s^(5/3) + 0.001846 Q_c.

Outflow through a door d of room i, of upper-layer gas above the interface (kg s⁻¹), with C_d = 0.7, g = 9.81 m s⁻²:
  (S2) ṁ_d(t) = (2/3) C_d W_d ρ_a sqrt(2 g (1 − T_a / T_U,i)) max(0, H_d − z_i)^(3/2),
which is zero while the layer is above the door's top.

Layer descent, explicit Euler with step Δt = 0.5 s, never rising:
  (S3) z_k ← clamp(z_k − Δt max(0, ṁ_in,k − ṁ_out,k) / (ρ_U,k A_f,k), 0, H_k),
where for the fire room ṁ_in = ṁ_p, and ṁ_out,k is the sum of (S2) over k's doors.

## 4. Spread to other rooms

Rooms are processed in breadth-first order from k_f over the room graph; each room's whole history is computed before its successors'. A room j is fed by the doors that join it to rooms i processed before it:
  (R1) ṁ_in,j(t) = Σ over such doors d of ṁ_d(t) computed from room i;
  (R2) ΔT_in,j(t) = max over those rooms i of ΔT_U,i(t);
  (R3) ΔT_U,j(t) = ΔT_in,j(t) · ṁ_in,j c_p / (ṁ_in,j c_p + h_c A_T,j), with c_p = 1.0 kJ kg⁻¹ K⁻¹ and wall-loss coefficient h_c = 0.015 kW m⁻² K⁻¹ (a steady energy balance with losses to the walls);
and its layer follows (S3). Flow back towards the fire room is not modelled (one-way propagation along the breadth-first tree, all doors open).

## 5. What a lattice node sees

A node n in room k is in the upper layer when the interface is at or below head height, z_k(t) ≤ z_head (default 1.8 m above the room's floor); then its gas temperature is T_n(t) = T_U,k(t), otherwise T_n(t) = T_a (the lower layer is taken at ambient).
  (N1) T_n(t) in degrees C = T_n(t) − 273.15.
A node is untenable from the first time either the layer is at or below head height (the smoke criterion, default on) or T_n(t) ≥ θ (θ default 60 °C). The routers' convective-heat FED rate is r(T_n) per second (ISO 13571, as in forecast.js); with this model it is a dose of real gas temperature, convective heat only.

## 6. Coupling to the routers

The model is sampled every Δt_f seconds (default 1 s) up to the horizon H (default 600 s) into the same forecast object `buildForecast` returns (per-frame FED rates of the running maximum of T_n, untenable and closing times), so `earliestArrival`, `doseFront`, `escapeDeadlines`, `doseDeadline` and `doseDeadlineMap` run on it unchanged.

## 7. Properties the implementation and the Lean module must both establish

P1. Q(t) is non-decreasing; h_k(t) is non-increasing; hence ΔT_U,f(t) is non-decreasing, including across t_p and flashover.
P2. Each Euler step (S3) never raises z_k, keeps 0 ≤ z_k ≤ H_k, and with non-decreasing inflow, temperature and outflow-coefficient inputs the computed z_k(t) is non-increasing.
P3. ṁ_d is non-decreasing in T_U,i and non-increasing in z_i; (R3) is non-decreasing in ṁ_in and in ΔT_in; hence, by induction along the breadth-first order, every room's T_U,k(t) is non-decreasing and z_k(t) non-increasing. Note that (S3) uses the same room's outflow, so P2 needs ṁ_in − ṁ_out handled by the clamp, not by monotonicity of the difference.
P4. Hence every node's T_n(t) is non-decreasing, its untenability is an up-set in time, and the forecast hypotheses of Forecast.lean and Deadline.lean (heat that never falls; closing times that never reopen) hold, so the routers' exactness and the deadline map's conservativeness carry over to this model.
P5. Units of (S2) and (S3) close (Dim.lean); (F1), (F2) and (S1) are FD-02, FD-17 and FD-08.
P6 (numerics, implementation only). For a closed room (no doors) and constant Q with the linear term of (S1) omitted, (S3) converges to the closed-form filling solution s(t) = (s_0^(−2/3) + (2/3) a t)^(−3/2), a = 0.071 Q_c^(1/3) / (ρ_U A_f), as Δt falls; report the error at Δt = 0.5 s.

## 8. Validity and limits (to state wherever results are shown)

Pre-flashover correlations (MQH) for rooms of ordinary size; one fire; open doors; no ceiling jet, radiant heat, toxic gases or visibility model beyond the layer height; one-way smoke spread along the breadth-first tree; lower layers at ambient; IfcSpace needed (models without spaces fall back to the graph models). The model ranks routes and deadlines under these assumptions; it is not a substitute for CFAST, FDS or an engineer's assessment.
