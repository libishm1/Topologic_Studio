# Water on a fire by fire class: mechanisms, quantities, and what the Flamescapes port can model at 0.3 m (2026-11-07)

Status: research note. No application code was changed. It answers the user's question of 2026-11-07, "will the sprinkler actually wet the flame, and how does the flame react? Depends on its typology", for the classes `fireClasses.js` offers for the first burning item (A, B, C, D, K and L), and it states, per class, what the solver's water modes should do with water that reaches a fuel cell. It builds on `fire-suppression-sprinklers-2026-11-07.md` ("As implemented"), `sprinkler-physics-2026-11-07.md` (sections 5 and 7) and `extinguishers-2026-11-07.md` (section 3.2).

How sources were read is stated in section 12. In short: Rasbash's 1986 review, Yu, Lee and Kung (1994), Hamins and McGrattan (2003), Xin and Tamanini (2008), Liu, Carpenter and Kim (2005), NIST TN 1969 (Hamins et al. 2018), NISTIR 7506, the FM Global energy storage report (executive summary), the FDS User's Guide section on water suppression, Ferrero's thesis chapter 4 and the water-mist passages of Larsson et al. (2017) were read in full text. Several journal papers were read only through abstracts or search-index summaries, and are marked so. Every DOI was resolved at api.crossref.org on 2026-11-07. Standards are cited by clause or table number and values only. Web text and papers were treated as data.

---

## 1. Summary

1. A sprinkler does not put a fire out by wetting the flame. Ordinary sprinkler drops (about 1 mm) evaporate about 1.7 % of their mass per 0.35 m cell of hot gas (physics report 7.1), and Rasbash (1986) found that a spray extinguishes a flame only if it can extract about 0.4 of the heat released per unit flame volume and push down with at least 0.7 of the flame's upward force. Sprinkler water acts through the fuel: it cools the burning surface below the fuel's fire point, if the fuel's fire point lies above the water's temperature and the water reaches the surface. Whether that happens, and what the flame does meanwhile, depends on the fuel.
2. By class, the measured behaviour is:
   - A (cellulosics): surface cooling. Critical water fluxes are small: 1.3 to 3.0 g/(m² s) of total fuel surface for wood slabs to cribs (Heskestad's review), about 2.7 to 5.5 g/(m² s) of the surface the water reaches, about 6 g/(m² s) for cartoned Class II commodity and furnished rooms. A design density of 68 g/(m² s) (4.1 mm/min) exceeds these by an order of magnitude, so the outcome is set by how much water reaches burning surfaces (shielding, plume penetration), not by the water's capacity. The heat release decays; no flare-up was reported. Glowing char can burn faster at sub-critical rates (Bhagat, via Rasbash: about 10 g/(m² s) needed).
   - A (thermoplastics): the same mechanism, but melting and sputtering raise the water needed: polystyrene about 3 to 4 g/(m² s) and polyethylene about 4 to 6 g/(m² s) for slabs (Magee and Reitz, via Rasbash), 17 to 20 g/(m² s) for the cartoned plastic commodity (Yu et al.), and exposed plastics respond all or nothing (Xin and Tamanini).
   - B, fire point below the water's temperature (heptane, petrol): water cannot cool the liquid below its fire point. The drops sink through the pool (heptane's specific gravity is 0.68 and it is practically insoluble). The flame is put out only by fine, intense sprays (petrol: drops of at most 0.6 mm and at least 130 g/(m² s)); sprinklers control the surroundings, not the pool. Rasbash judged the extinction of low flash point liquids by plain water unreliable.
   - B, fire point above the water's temperature (kerosene, diesel, transformer oil): cooling extinguishes, at 100 to 330 g/(m² s), with a flame upsurge in the first 1 to 2 s for diesel and transformer oil and splashing by coarse drops. These rates lie at or above the extra hazard densities (203 and 272 g/(m² s)).
   - B, miscible (ethanol): dilution of the surface layer extinguished an alcohol fire with coarser sprays; fine sprays put the flame out directly (Rasbash). No critical dilution was read.
   - C: after de-energising, the burning plastics behave as class A. While energised, the electrical fault keeps heating and water can carry current; no source read gives a quantity a fire solver can use.
   - D: water is an oxidant for a burning metal, not a coolant. Per kg of water reacting, magnesium releases 17.5 MJ at the metal and 0.112 kg of hydrogen, which can burn in the gas for a further 13.4 MJ; aluminium 15.1 MJ and 0.112 kg; sodium 10.2 MJ and 0.056 kg; lithium 12.4 MJ and 0.056 kg (section 7, from CODATA values). No measurement of the fraction of water that reacts at a burning metal surface was found.
   - K: water reaching oil above about 210 °C boils explosively and ejects burning oil; cooking oils autoignite at 330 to 445 °C (canola, corn, peanut), so a burning pan is always in that regime. Under a residential sprinkler delivering 69 g/(m² s) on the cooktop, NIST extinguished one of two cooking oil fires, and both lost fuel containment. Single 2 to 4.5 mm drops raised the flame height of burning peanut oil 6 to 15 times. No study read measured the heat release jump or the oil ejected per kg of water.
   - Lithium-ion: water cannot reach the cells inside a pack or module, so ceiling sprinklers did not suppress the burning unit; they delayed or prevented spread to neighbours (FM Global, at 12 mm/min) and an NFPA 13D system prevented flashover in e-scooter room fires (FSRI). Stopping cell-to-cell propagation needs water on the cells early: cooling the cell in runaway below 126 °C (18650 cells, water mist). Water mist raised the HF production rate temporarily but not the total (Larsson et al.).
3. For the solver. The wetting mode now in the code puts a share (0.52, calibrated to Evans on the test room) of the water reaching a fuel cell into its `moisture`, which then holds the surface near 100 °C. That is the right mechanism for class A and for high fire point liquids, and the wrong one for low fire point liquids (it is harmless there by a coincidence of boiling points), cooking oil, metals and batteries, where it would extinguish a fire that water does not extinguish or makes worse. The recommended behaviour per class is in section 10: wet A and C (after de-energising), wet B only above its fire point and with the measured penetration share, never wet K, D or L cells (state the hazard instead), and drop Evans's cap from liquid pools.

---

## 2. What the solver does now with water

From `src/core/flamescapes.js`, `src/core/suppression.js` and `src/core/flamescapesGpu/kernels.js`, read 2026-11-07 (the wetting code was being written concurrently; line numbers will move):

- Every fuel cell, of any class, dries first: `dw = min(moisture, C (Ts - 373.15)/L)`, and the vapour joins the gas (`chemistry`, CPU and GPU alike).
- A liquid first item (classes B and K) then evaporates only above its boiling point, `dv = min(skin, C (Ts - T_boil)/L_f, poolCap)`. Heptane boils at 371.6 K, below water's 373.15 K; the class K oil stand-in boils at 663 K.
- A solid or metal cell oxidises and pyrolyses with Arrhenius rates in `Ts`; a metal's oxide stays as ash.
- Evans's cap (`evansFactor`) multiplies both the solid reactions and the liquid's evaporation `dv` of every covered fuel cell, whatever its class.
- In "wetting" mode, the spray's water left at the end of a column, times `WETTED_FRACTION` (0.52), joins the top fuel cell's `moisture`, mixed in at 20 °C. The share was calibrated on the 5 x 4 m class A test room so that the covered fuel burnt in the 120 s after the head opens matches Evans's cap at 0.068 mm/s (`benchmark/results/2026-11-07-wetting-calibration.txt`). The wetting has no class test.
- Two existing details bear on water: every fuel cell starts with `moisture = 0.1` kg per kg of its surface layer, liquid and metal first items included; and a solid or metal cell's surface layer is replenished from the bulk "with its water" (0.1 kg per kg of fresh fuel). For magnesium both put water where there is none and make it a heat sink held at 100 °C, the opposite sign to water's real effect on a burning metal (section 7).

What the wetting does to each class, by the code's own rules (inferences from the code, not runs):

| First item | What `moisture` does in the solver | What water does in reality |
|---|---|---|
| A, C (plastics stand-in) | holds the surface near 100 °C while the water lasts; oxidation and pyrolysis slow | the same, if the water reaches the burning surface |
| B, heptane | almost nothing: the pool evaporates above 371.6 K, below the 373.15 K at which the water boils, so the pool still burns at its pool cap | the water sinks; the surface stays near 98 °C and burns on |
| B, a high fire point liquid (not offered) | holds the pool below its boiling point: extinguished as soon as the water's heat sink exceeds the heating | extinguished by cooling at 100 to 330 g/(m² s), after an upsurge |
| K, cooking oil | holds the oil at 100 °C, far below its 663 K "boiling": extinguished | the water sinks, boils explosively and throws burning oil out |
| D, magnesium | holds the metal at 100 °C, where its oxidation stops: extinguished | the water oxidises the metal and releases hydrogen |
| L (plastics stand-in) | as A: the pack is extinguished | the cells keep reacting inside the casing |

So, for the user's question: in the solver as it stands, every class is wetted and slowed alike; in reality only classes A, C (de-energised) and high fire point B respond to water that way.

---

## 3. Water's heat sinks and the ways water acts on a fire

- Heat absorbed. Heating water from 20 °C to 100 °C and evaporating it takes 4186 x 80 + 2.259e6 = 2.594 MJ/kg, the solver's own constants. Rasbash's Table 1 gives 2.612 MJ/kg from 15 °C, 2.900 MJ/kg to 250 °C as steam, and 5.317 MJ/kg to cool a flame's reaction zone to 1300 °C including condensation; Yu et al. used 2.595 MJ/kg.
- Cooling the fuel to its fire point. Rasbash's surface balance: the spray extinguishes when it removes heat faster than the flame returns it, at a rate S that must hold down to the fire point. His heat transfer factor phi (the fraction of the heat of combustion returned to the surface by convection at the fire point) is about 0.2 to 0.4 for organic fuels, with 0.3 suggested. Liu et al. (2005) wrote the same balance for an oil pool, giving the critical water flux reaching the surface as
  `m''_w,cr = (f_c dH_c - L_v) m''_f / (L_w + c_w (100 - T_w))`, with f_c 0.1 to 0.4 (their reference [8]).
- Cooling the flame. For 30 cm liquid fires the spray had to absorb about 0.4 of the heat released per unit flame volume, and its downward force had to exceed about 0.7 of the flame's upward force (Rasbash). Sprinkler drops of about 1 mm do not meet the first condition in a solver cell (physics report 7.1).
- Steam and oxygen dilution. Diesel pool flames of 30 and 60 cm went out when the oxidant, steam with nitrogen, held 14.5 to 15 % oxygen (Rasbash). FDS's coarse-grid criterion uses a limiting oxygen volume fraction of 0.135 at 20 °C (physics report 5.2). Rasbash judged steam secondary to fuel cooling in room fires at the scale of his Table 4.
- Dilution of a water-miscible liquid lowers its vapour pressure (Rasbash: alcohol, section 5.4).
- Emulsification of viscous oils is named in water spray practice as an extinguishing mechanism, but no source read here quantified it for sprinkler sprays.
- Mechanisms that make the fire worse: splashing of liquid fuel by coarse drops and sputtering by fine drops boiling at a hot surface (Rasbash, kerosene); frothing and eruption when water enters liquid above 100 °C (K, section 8); boilover of a water layer beneath a burning pool (section 5.5); oxidation of a burning metal by water (D, section 7); spreading of a floating pool by water collecting beneath it (section 5.5).

---

## 4. Class A: ordinary solids

### 4.1 Mechanism

Water that reaches a burning cellulosic surface is evaporated there almost completely: Magee and Reitz found that the extra water needed to extinguish PMMA and polyoxymethylene under external radiation balanced the extra radiation through its heat of vaporisation (via Rasbash). The surface cools below its fire point and flaming stops; the heat release decays. In compartments, Rasbash judged that fuel cooling, not steam, extinguished furnished room fires at water rates above about 6 g/(m² s) of fuel area. What limits suppression is the water's access to the burning surfaces: plume penetration (the actual delivered density), shielding by the fuel's own geometry, and run-off.

### 4.2 Critical water fluxes

| Fuel | Critical flux, g/(m² s) | Area basis | Source |
|---|---|---|---|
| Wood, slabs to cribs | 1.3 to 3.0 | entire fuel surface | Heskestad's review, via Rasbash (1986) and Yu et al. (1994) |
| Wood cribs and pallets | 2.7 to 5.5 | surface exposed to the water (0.55 of the total) | Yu et al. (1994), converting Heskestad |
| Rotating crib, moving jet | about 1.6 | fuel surface | Bryan, via Rasbash |
| 0.9 m crib, 25 mm sticks | about 4 (187 on the top) | fuel surface | O'Dogherty et al., via Rasbash |
| Furnished and crib room fires | least rate used about 6 (5.6, 6.25) | fuel area | Rasbash Table 4 |
| FM Class II commodity (cartons with metal liners) | about 6 | exposed surface, 240 s | Yu et al. (1994) |
| FM plastic commodity (polystyrene cups in cartons) | 17 to 20 | exposed surface, 240 s | Yu et al. (1994) |
| PMMA, slab | about 1.3 to 2.7 | burning surface | Magee and Reitz, via Rasbash Table 2 (see section 11 on the table) |
| Polyoxymethylene, slab | about 1.9 to 2.7 | burning surface | the same |
| Polystyrene, slab | about 3 to 4.1 | burning surface | the same |
| Polyethylene, slab | about 4.3 to 6.0 | burning surface | the same |
| Glowing charcoal (60 to 120 µm drops in a 10.4 m/s air stream) | about 10; lower rates increased burning by removing ash | surface | Bhagat, via Rasbash |

Yu et al. derived the net heat flux the flame returns to the burning surface from their plastic commodity data as 47.8 kW/m². The critical flux is that flux over 2.595 MJ/kg, 18 g/(m² s), the measured 17 to 20. For wood, 2.7 to 5.5 g/(m² s) corresponds to 7 to 14 kW/m² (my conversion at 2.594 MJ/kg).

Design densities, for comparison, in g/(m² s): light hazard 4.1 mm/min, 68; ordinary hazard 1 and 2, 6.1 and 8.1 mm/min, 102 and 135 (codes report); extra hazard 1 and 2, 12.2 and 16.3 mm/min, 203 and 272 (NFPA 13-2022 Table 19.2.3.1.1, over 230 m², from a secondary copy of the table). A room's contents have more fuel surface than floor; the solver takes 2 m² per m² of floor (`contentsArea`). Even then, light hazard water spread over all of it is 34 g/(m² s), several times every critical flux above except the plastic commodity's.

### 4.3 Heat release under water: decay constants beyond Evans

| Model | Form | Values | Basis |
|---|---|---|---|
| Evans (1993) | Q = Q_act exp(-(t - t_act)/tau), tau = 3.0 w''^-1.85 s, w'' in mm/s | 433 s at 0.068 mm/s, 122 s at 0.135, 33 s at 0.272 | upper envelope of unshielded furnishings and cribs; measured furnishing decays 9.1 s (vinyl sofa) to 713 s (workstation) at 0.07 mm/s (physics report 5.1) |
| Yu, Lee, Kung (1994) | Q = Q_0 exp(-k (t - t_0)), fitted over 240 s | Class II: k = 0.536 m''_w - 0.0040 s^-1 for 0.006 < m''_w < 0.024 kg/(m² s); plastic: k = 0.716 m''_w - 0.0131 s^-1 for 0.012 < m''_w < 0.041 kg/(m² s); m''_w per unit exposed surface | 98 FM delivered-density tests, rack storage; k's water term follows from a surface energy balance with alpha = 4.29 /m (Class II) and 7.27 /m (plastic) |
| Hamins and McGrattan (2003), FDS's E_COEFFICIENT | local q'' = q''_0 [exp(-k1 (t - t_0)) + k2 (t - t_0)], k1 = a1 M''_w, k2 = a2 M''_w + b2 | a1 = 1.0 m²/(kg s) (fitted as -1.02 +/- 0.12 with their sign), a2 = -0.020 m²/(kg s), b2 = 0.0015 s^-1; M''_w = 0.24 s x the water flux at the box tops | 19 tests on the Group A plastic commodity, fluxes 0.03 to 0.66 kg/(m² s) from nozzles near the fuel; the k2 term is the slow regrowth seen at low or late application |
| FDS (current User's Guide) | m''_f = m''_f,0 exp(-integral of chi m''_w dt), m''_w the water per unit area on the surface | chi must be measured; the guide's example uses 1 | for surfaces with a prescribed heat release |
| Xin and Tamanini (2008) | actual delivered density over sprinkler discharge flux: ADD/SDF = 1 - 0.171 (1 - exp(-Q_conv/1116 kW)) SDF^0.327, SDF in mm/min | 0.94 to 0.70 for SDF 4.1 to 16.3 mm/min and Q_conv 0.3 to 1.4 MW (my evaluation) | 3.05 m ceiling clearance; the ADD data spanned SDF 10.5 to 31.8 mm/min, so 4.1 to 8.1 mm/min is an extrapolation |
| Madrzykowski (2008) | measured | a residential sprinkler held a Christmas tree room fire to about 1.8 MW against more than 6 MW without | NISTIR 7506 |
| Noaki et al. (2018) | measured | peak heat release of wood cribs reduced in proportion to the water flow, about 12 kJ per g of water | abstract (extinguishers report 3.2) |

The three exponential forms disagree by an order of magnitude at the same nominal flux because their fluxes are defined on different areas and at different stages: Evans's w'' is the ceiling density, Yu's m''_w is spread over the exposed surface of the array, and Hamins and McGrattan's water was delivered from close above the fuel. At 0.068 kg/(m² s) on the commodity's top, Hamins and McGrattan's k1 gives tau = 1/(0.24 x 0.068) = 61 s, against Evans's 433 s for the ceiling density.

### 4.4 Thermoplastic furnishings

- Polystyrene and polyethylene melt (240 and 138 °C) below their fire points (350 and 330 °C): a horizontal surface carries a layer of very hot liquid and a vertical one burns as a running liquid fire. Spray on these sputtered fuel into the flame; the critical fuel flow at extinction was higher than for PMMA or polyoxymethylene, and a coarse spray worked better than a fine one (Magee and Reitz, via Rasbash).
- Exposed plastics (plastic pallets, exposed polystyrene trays) respond all or nothing: the damage fraction stays below about 10 % when controlled, and the fire goes out of control just beyond; cartoned commodities degrade gradually, with damage fractions of 20 to 40 % still controlled (Xin and Tamanini 2008).
- Evans's own furnishing data do not single thermoplastics out as slow: the vinyl sofa decayed fastest (9.1 s).
- The app's contents are one statistical fuel (`FIRE_CLASSES.A.fuel`), so it cannot distinguish a polyurethane sofa from a bookcase; the factor of about 3 between the plastic commodity's critical flux and Class II's is the size of what is lost.

### 4.5 What a 0.3 m solver can do

- Wetting is the right mechanism. The water arriving at a fuel cell's top is a column budget the spray stage already carries. The calibrated share 0.52 lumps three things the data separate: plume penetration (ADD/SDF about 0.8 to 0.94 at light hazard by Xin and Tamanini, extrapolated), the share of the cell's exposed area the water reaches (Yu et al. used top plus vertical surfaces; the solver's cell has 2 m² of exposed fuel per m² of floor), and the solver's own surface heating. The last matters: the solver couples gas and solid at h = 200 W/(m² K), so a fuel cell in 800 °C gas receives about 140 kW/m², against the 7 to 14 kW/m² (wood) and 48 kW/m² (plastic commodity) that the measured critical fluxes imply. The solver's own critical flux is therefore set by its exchange rule, and is probably several times the measured one (an inference, not a run).
- Checks. (1) A critical-flux test: one burning contents cell under one column, water flux stepped from 1 to 60 g/(m² s) of exposed area; the flux at which the cell stops should be compared with 2.7 to 6 g/(m² s) (cellulosics) and 17 to 20 g/(m² s) (plastic commodity). (2) The per-cell decay at 0.07 mm/s should lie within the measured furnishing range, 9.1 to 713 s; the calibration run's fitted decays (63 to 85 s) do. (3) The density dependence: the calibration run's burnt fuel fell from 9.0 to 4.5 kg between 0.026 and 0.133 mm/s, much less steeply than Evans's w''^-1.85; Yu et al.'s linear k(m''_w), with an intercept below which water does not suppress, is the measured shape to compare with.
- Evans's cap remains the conservative choice (it bounds the heat release from above for unshielded items) and is what CFAST does.
- Warnings only: thermoplastic-rich contents may need about three times the water (no per-item fuel in the app); smouldering char can survive and re-ignite after the water stops, and sub-critical water can raise a glowing char's burning rate (Bhagat); no quantity usable on this grid.

---

## 5. Class B: flammable liquids

### 5.1 The two boundaries: fire point against water temperature, density and miscibility against water

| Liquid | Flash or fire point | Specific gravity, solubility | Source |
|---|---|---|---|
| n-heptane (the app's class B) | flash -4 °C (25 °F); boils 98 °C | 0.68; 0.0003 % | NIOSH Pocket Guide |
| Petrol (gasoline) | below ambient | floats | Rasbash (qualitative) |
| Ethanol | flash 13 °C (55 °F); boils 78 °C | 0.79; miscible | NIOSH Pocket Guide |
| Kerosene | closed cup flash 40 to 45 °C; open cup fire point 58 to 68 °C | floats (40 to 50 % of a spray passed below the burning layer) | Rasbash |
| Diesel oil | fire point 104 to 115 °C | floats (burned on a water bed in boilover tests) | Rasbash; Ferrero et al. |
| Transformer oil | fire point 175 to 180 °C | | Rasbash |

Water cannot cool a liquid below its own temperature, 20 °C in the solver (`WATER_TEMPERATURE`; stored water in a hot climate is warmer, which raises the boundary). A liquid whose fire point lies below the water's temperature cannot be extinguished by cooling; one whose fire point lies above it can, given enough water at the surface.

### 5.2 Rasbash's measurements (1957, 1960; summarised 1986)

- Kerosene (30 cm and 11 cm pools, 6 cm layer). Unextinguished fires settled at a lower surface temperature and burning rate: at the most effective drop size (0.35 to 0.7 mm, falling with spray pressure) 40 to 50 % of the spray penetrated below the burning fuel, and the burning rate fell to 2 to 3 g/(m² s) from 10 and 14 g/(m² s) without spray. Coarser drops (more than 1.7 times the best size) splashed fuel into the flames; finer drops (less than 0.7 times) penetrated only 10 to 20 % and sputtered fuel by boiling at the hot surface; both raised the burning rate. Extinction by cooling to the fire point needed 100 g/(m² s) at 0.4 mm drops, rising to 300 g/(m² s) at 1.2 mm; the flames shrank before going out and the fuel did not re-ignite at once. Extinction of the flame itself, with the surface still above its fire point, was faster but the fuel re-ignited at once from a pilot.
- Petrol was extinguished only by putting the flame out: not with drops above 0.6 mm nor below 130 g/(m² s) at 5.9 bar; extinction time scaled with the fourth power of drop size and the inverse square of flow. With less than 10 s of preburn, sprays that put out longer-burning fires failed: the flame burned as a horizontal sheet in the spray's entrained air.
- Diesel oil and transformer oil under 267 g/(m² s) of 0.28 to 0.49 mm spray: a large upsurge of flame in the first 1 to 2 s, then extinction a few seconds later, faster than kerosene; the coarser sprays worked better. No magnitude of the upsurge was given.
- Alcohol: the two coarser sprays extinguished it by diluting the surface layers; the finest put the flame out in a few seconds.
- Practice: water spray systems for high flash point liquids (power stations) used about 330 g/(m² s) (0.4 imperial gal/(ft² min)); a tube rig of burning transformer oil needed 250 to 850 g/(m² s) as the tubes heated from 100 to 500 °C, film boiling limiting the water's contact with hot metal. Extinction of low flash point liquids by plain water was judged unreliable, and sprays were used to cool neighbouring risks.

Heskestad (2003, abstract only) found the water flow at extinction for methane and heptane pool fires under downward nozzles roughly proportional to an effective nozzle diameter and to the 0.4 power of nozzle height and free-burn heat release, with liquid pools needing somewhat more water than gas burners; the coefficient was not read.

### 5.3 Design densities

NFPA 13-2022 Table 19.2.3.1.1 (values only, from a secondary copy): extra hazard group 1, 12.2 mm/min over 230 m² (203 g/(m² s)); group 2, for occupancies with appreciable flammable liquids, 16.3 mm/min over 230 m² (272 g/(m² s)). The liquid-specific storage tables of NFPA 30 and FM Global Data Sheet 7-29 were not read. Against Rasbash's numbers, light hazard water (68 g/(m² s)) is below every extinguishing rate for a liquid pool, and extra hazard 2 is at the low end of the high fire point range: sprinklers control a liquid fire's surroundings; they extinguish a high fire point pool only at the upper densities and with drops that do not splash.

### 5.4 Miscible liquids

Water entering ethanol mixes with it and lowers its vapour pressure; the flash point of an aqueous solution rises with water content. Rasbash's coarse sprays put an alcohol fire out this way. No measured flash or fire point table for ethanol and water was read (Liaw and Chiu 2003 model them; not read), so the critical dilution is not given here.

### 5.5 Water beneath a burning pool

- Spreading. Sprinkler water that collects in a bund, tray or pan floats the burning liquid up and can carry it over the edge. The time to overflow is the free volume over the water inflow, a geometric quantity the app does not have (no pan or bund geometry).
- Thin-layer boilover. When a thin burning diesel layer heats the water bed beneath it to boiling, the burning rate rises. Ferrero's tests (diesel on water, 1.5 to 6 m pools; thesis Table 4.10, read) gave a mean burning rate during boilover 89 % above the steady rate and a peak 140 % above at 1.5 m; 2 to 87 % (mean) and 23 to 81 % (peak) at 3 m; and at 4 to 6 m, mean rates below the steady one. Petrol showed no thin-layer boilover in the same series (Ferrero et al. 2006, abstract). The increase falls with pool diameter and depends on the fuel layer's thickness at onset.

### 5.6 What a 0.3 m solver can do

- Low fire point, immiscible (the app's heptane): do not wet the pool cell; the water passes under it and does not change the burning. The present wetting is nearly harmless for heptane because heptane boils below water (section 2), but it adds a heat capacity the pool does not have. Do not apply Evans's cap to the pool's evaporation: it is calibrated on furnishings and cribs, and Rasbash's data show no suppression of such pools by sprinkler-sized drops. Keep the spray's gas cooling and steam. Warnings: the pool keeps burning under sprinklers; water collecting under it can float it over a rim and spread it.
- High fire point, immiscible (not offered; diesel, transformer oil): wet the pool cell with the share that penetrates the burning layer, 0.4 to 0.5 for the most effective drops and 0.1 to 0.2 for finer ones (Rasbash, kerosene). Check the outcome against the critical flux, 100 to 330 g/(m² s) measured, or (f_c dH_c - L_v) m''_f / 2.594 MJ/kg with f_c 0.1 to 0.4 (Liu et al.; Rasbash's phi 0.3). Warnings: an upsurge of 1 to 2 s on first contact (no magnitude), splashing by coarse drops, and boilover if water lies beneath a thin layer (burning rate up to 2.4 times steady at 1.5 m; a multiplier on `poolCap` would need the layer thickness, which the grid does not carry).
- Miscible (not offered; ethanol): water joining the liquid should lower its evaporation by its share in the mixture (Raoult's law with an activity coefficient). No critical dilution was read, so this is a modelling route, not a parameter.

---

## 6. Class C: energised electrical equipment

- Once the equipment burns, its fuel is plastics (`fuelClass: "A"`), and water acts on it as on class A, if the water reaches the burning parts through the casing.
- While the circuit is live, the fault that started the fire keeps heating, so a fire knocked down by water can re-ignite; water can also cause new faults and arcing.
- The person-safety hazard of water on live equipment (leakage current along the spray, shock on touching wet equipment) is real but was not quantified in any source read for sprinkler sprays; a patent and a 2024 Russian study measured leakage currents for fine sprays and nozzles (not read).
- Solver: wet as class A. If the app later models the energised state, keep the ignition source (the pilot) on while energised. Warnings only: re-ignition while energised; shock risk to occupants and responders; no quantity.

---

## 7. Class D: combustible metals

### 7.1 Reactions with water (CODATA key values, 298 K)

Enthalpies of formation (kJ/mol): H2O(l) -285.830, H2O(g) -241.826, MgO(cr) -601.60, Al2O3 (corundum) -1675.7, Li+(aq) -278.47, Na+(aq) -240.34, OH-(aq) -230.015.

| Reaction | Heat at the metal | Per kg water reacting | Metal per kg water | H2 per kg metal | H2 per kg water |
|---|---|---|---|---|---|
| Mg + H2O(l) -> MgO + H2 | 315.8 kJ/mol, 13.0 MJ/kg Mg (14.8 with steam) | 17.5 MJ | 1.35 kg | 0.083 kg | 0.112 kg |
| 2 Al + 3 H2O(l) -> Al2O3 + 3 H2 | 409.1 kJ/mol Al, 15.2 MJ/kg Al (17.6 with steam) | 15.1 MJ | 1.00 kg | 0.112 kg | 0.112 kg |
| Na + H2O(l) -> Na+(aq) + OH-(aq) + 1/2 H2 | 184.5 kJ/mol, 8.0 MJ/kg Na | 10.2 MJ | 1.28 kg | 0.044 kg | 0.056 kg |
| Li + H2O(l) -> Li+(aq) + OH-(aq) + 1/2 H2 | 222.7 kJ/mol, 32.1 MJ/kg Li | 12.4 MJ | 0.39 kg | 0.145 kg | 0.056 kg |

The hydrogen burns in air at 241.8 kJ/mol (120 MJ/kg H2; 7.94 kg of O2 per kg; 15.1 MJ per kg of O2, against Huggett's 13.1 for organic fuels). For magnesium, the water reaction followed by the hydrogen's combustion releases 557.6 kJ/mol, which is the 601.6 kJ/mol of burning magnesium in air less the 44.0 kJ/mol needed to evaporate the water: water on burning magnesium removes almost nothing from the fire's total heat if the hydrogen burns; it moves about 40 % of it from the metal into the gas. Each kg of water fully reacted with magnesium or aluminium gives 1.34 m³ of hydrogen at 20 °C (my calculation).

### 7.2 What is known of the behaviour

- Magnesium particles burn in water vapour (Prachukho et al. 1971, title only; Nishii and Koizumi 2024 measured the condensed products' temperature in O2 and steam, title only). Steam therefore does not smother burning magnesium.
- Alkali metals react with water at room temperature; the explosive onset of the reaction was attributed to a Coulomb explosion of the metal (Mason et al. 2015, not read).
- At a hot metal surface water may film-boil instead of wetting it (Rasbash's oil-coated tubes above 500 °C), so the fraction of water that reaches and reacts with a burning metal is unknown. No controlled study of sprinkler water on burning magnesium, aluminium, sodium or lithium was found; the guidance found (data sheets, `fireClasses.js` from NFPA 10 and CAMEO) only advises against water.

### 7.3 What a 0.3 m solver can do

- Never put water into a metal cell's `moisture`. Set the metal cells' initial moisture and the replenishment water to zero (the 0.1 kg/kg default is wood's).
- Default: water reaching a burning metal cell neither cools it nor feeds it; show the hazard.
- Optional bound, "all water reacts": at Ts above the metal's ignition temperature (473 °C for magnesium, `fireClasses.js`), each kg of water reaching the cell consumes 1.35 kg of magnesium, releases 17.5 MJ in the cell (oxide to ash, as now) and puts 0.112 kg of hydrogen into the gas. It uses only stoichiometry, so it is an upper bound on heat at the metal and on hydrogen. It needs a hydrogen fuel kind with its own stoichiometry (o2 7.94, h2o 8.94, heat 120 MJ/kg); the solver keeps one vapour kind per gas cell (`fuelVapourKind`), so hydrogen would overwrite pyrolysis vapour's kind in a shared cell.
- Warnings: water on burning metal releases hydrogen and can intensify the fire or explode (molten aluminium with water is a known steam explosion hazard; sodium and lithium react at room temperature); lithium also burns in nitrogen and CO2. No usable rate exists for how much of the water reacts.

---

## 8. Class K: cooking oils and fats

### 8.1 Mechanism

Cooking oil floats on water (0.914 to 0.920 against 1.0) and burns far above water's boiling point: corn oil flashes at 254 °C and autoignites at 392 °C, peanut at 282 and 445 °C (NIST TN 1969 Table 3-2), canola at 232 to 290 and 330 to 360 °C (Liu et al. 2005); canola in Liu's pans autoignited at 351 to 361 °C. Water that reaches it sinks, superheats and boils explosively, throwing burning oil out as droplets. Single water drops on oil above about 210 °C produced vapour explosions (Xu et al. 2018, abstract, oil at 205 to 260 °C).

### 8.2 Quantities found

- Residential sprinkler over a cooktop (Hamins et al. 2018, NIST TN 1969). One pendent sprinkler, K about 45 (3.1 US), 68 °C bulb, centred in a 3.6 x 3.4 x 2.4 m kitchen; 69 g/(m² s) on average on the cooktop (spatial variation +/- 80 %, local maximum 180 g/(m² s)), measured without fire. Two tests, 4.5 L of oil: water at 176 s after ignition extinguished the fire; at 307 s it did not. Both lost fuel containment, which the report attributed to the violent vaporisation of water drops on the oil surface. Tenability limits were exceeded after activation in one test; the report found room sprinklers slowed spread to nearby items but did not reliably suppress the pan.
- Sprinkler sprays on industrial oil cookers (earlier tests summarised by Liu et al. 2005): extinguished in 10 to 145 s, but with pronounced flare-up and much oil spilled over the cooker, forming large fires on the ground, as coarse drops sank and boiled in the hot oil. Fine water mist (Dv99 below 1000 µm) put out 13 MW cooker fires in 4 to 18 s without substantial flare-up; discharge was continued after extinction to cool the oil below its autoignition temperature and prevent re-ignition.
- Single drops on burning peanut oil (Fan, Wang and Guo 2020, abstract): 2.0 to 4.5 mm drops falling 45 to 95 cm raised the maximum flame height 6 to 15 times and the flame volume by factors of 6.7e4 to 4.8e5; drop size set the scale through the steam mixed with fuel vapour. On a stratified oil and water pool (diesel), splashed oil droplets drove the flame growth (Wang et al. 2023, abstract).
- Critical flux by the surface balance for the app's oil stand-in (m''_f 0.039 kg/(m² s), dH_c 37.4 MJ/kg, L_v 0.25 MJ/kg; all `fireClasses.js`, the last two provisional): 52, 165 and 221 g/(m² s) for f_c 0.1, 0.3 and 0.4 (my evaluation of Liu's equation). The NIST sprinkler's 69 g/(m² s) lies at the low end, consistent with one success in two, but the balance ignores the eruption.
- Not found: no study read measured the heat release jump on water application to cooking oil, or the mass of oil ejected per mass of water. Flame height and volume ratios (single drops) and containment loss (yes or no) are the only measured quantities.

### 8.3 What a 0.3 m solver can do

- Never wet the oil cell: `moisture` would hold the oil at 100 °C and extinguish it at once, the opposite of the eruption. Set its initial moisture to zero. Do not apply Evans's cap to it.
- Wet the surrounding class A contents as usual: the kitchen's cabinets and furnishings are protected (TN 1969).
- Warnings, stated with the measured values above: eruption and flame growth on contact (6 to 15 times flame height for single drops), loss of containment and burning oil outside the pan (both NIST sprinkler tests), extinguishment unreliable (one of two), re-ignition while the oil stays above autoignition and while the hob is on (TN 1969 notes the cooktop was not de-energised).

---

## 9. Lithium-ion batteries (class L)

### 9.1 Mechanism

A cell in thermal runaway generates heat internally; ceiling water reaches only the casing of a pack or module, so it cannot stop the reaction in progress. It can cool neighbouring cells, packs and furnishings below their onset temperatures and so stop spread, if it reaches them in time. Water does not react with the cells' chemistry in a way that adds heat at these scales (no source found); it does change the gas: water mist raised the HF production rate during application but not the total HF released (Larsson et al. 2017, pouch cells).

### 9.2 Water to stop propagation against cooling only

- Ceiling sprinklers, energy storage racks (Ditch and Zeng 2019, FM Global): 12 mm/min (200 g/(m² s)), K 81, 74 °C, RTI 27.6 (m s)^1/2. The racks' design shielded the fire from the water; the sprinklers delayed or prevented spread to adjacent racks without suppressing the rack of origin. With iron phosphate (LFP, 83.6 kWh) one sprinkler held the fire to the rack of origin; with NMC (125 kWh) it spread to the adjacent rack and the sprinklers that opened covered more than 230 m². The report recommends water supplies for 230 m² over at least 90 min (LFP) and a fire watch until damaged equipment is removed, as re-ignition occurred from minutes to days later.
- Residential sprinklers, e-scooters (Madrzykowski, Fleischmann and Weinschenk 2025; FSRI summary read): an NFPA 13D system controlled e-scooter fires started by thermal runaway, prevented flashover in the living room and kept the fire from spreading to other fuels; the summary gives no temperatures, heat release or gas data.
- Water on the cells (18650 cells, water mist; abstracts via search index): stopping propagation required cooling the cell in runaway below 126 °C, and the mist raised the critical runaway temperature by 36 °C (Xu et al. 2022); mist was effective when applied at a cell temperature at least 20 °C below runaway onset, conservatively 186.5 °C (Liu et al. 2019); below 100 °C the mist cools only by its sensible heat and its efficiency falls (Liu et al. 2021). On large-format NCM cells (750 Wh), modules (7.5 kWh) and a pack (74 kWh), water injected in direct contact with the battery immediately after runaway onset prevented propagation (Choi et al. 2025; the abstract gives no rates).
- So "cooling only" (ceiling water on the outside) limits spread beyond the burning unit; "stopping propagation" inside a pack needs water on the cells, early, which ceiling sprinklers do not deliver.

### 9.3 What a 0.3 m solver can do

- The app burns the class L item as appliance plastics (`provisional: ["fuel", "agents"]`). Wetting it, or capping it by Evans, would extinguish a pack that water does not extinguish. Do neither while the item has fuel; let it burn out at its own rate.
- Wet the neighbouring contents as usual: this reproduces the measured effect, spread stopped where water reaches.
- Warnings: re-ignition from minutes to days; water does not stop runaway inside a pack; HF production rate rises temporarily under water (total unchanged in Larsson's tests); the app does not model runaway, so propagation inside the pack is not computed.

---

## 10. Recommended solver behaviour per class

One gate per fuel kind decides what the spray's water does at the top fuel cell of a column (a flag on `fuels[k]`, read in the spray stage on both backends; nothing changes when sprinklers are off). The spray's gas cooling, steam and drag stay as they are for every class.

| First item | Water at the fuel cell | Evans's cap | Parameter values | Source | Warning shown |
|---|---|---|---|---|---|
| A (contents always) | into `moisture` | conservative alternative | wetted share 0.52 (calibrated; lumps plume penetration 0.8 to 0.94, area reached, and the solver's surface heating); check the critical flux against 2.7 to 6 g/(m² s) (cellulosics), 17 to 20 (cartoned plastic); decay at 0.07 mm/s within 9.1 to 713 s | Heskestad via Rasbash; Yu et al.; Xin and Tamanini; Evans; the calibration run | plastic-rich contents need about 3 times the water; char can smoulder and re-ignite |
| B, fire point below water temperature (heptane) | not absorbed; passes under the pool | not applied | none | Rasbash; NIOSH (heptane -4 °C, SG 0.68) | pool burns on; may float and spread; flame may intensify in the spray's air current |
| B, fire point above water temperature (if added) | into `moisture` with share 0.4 to 0.5 | not applied | critical flux 100 to 330 g/(m² s), or (f_c dH_c - L_v) m''_f / 2.594 MJ/kg, f_c 0.1 to 0.4 | Rasbash; Liu et al. 2005 | 1 to 2 s upsurge; splashing; boilover up to 2.4 times the burning rate if water lies beneath |
| B, miscible (if added) | mixes into the liquid; evaporation scaled by the fuel's share | not applied | none read for the critical dilution | Rasbash | extinction by dilution needs much water |
| C | as A, after de-energising | as A | as A | as A | shock; re-ignition while energised |
| D | not absorbed; initial and replenishment moisture zero | not applied | optional bound: per kg water, 1.35 kg Mg, 17.5 MJ in the cell, 0.112 kg H2 to the gas | CODATA | hydrogen; explosion; water never to be used |
| K | not absorbed; initial moisture zero | not applied | none (no measured jump or ejected mass) | NIST TN 1969; Fan et al.; Liu et al. 2005; Xu et al. 2018 | eruption, flame growth 6 to 15 times (single drops), burning oil spilled, 1 in 2 extinguished, re-ignition |
| L | not absorbed while the item has fuel | not applied | none | Ditch and Zeng; Madrzykowski et al. 2025; Xu et al. 2022 | runaway continues inside; re-ignition minutes to days; HF rate rises under water |

Further options, each behind its own switch and each measured before it is trusted:

- A gas-phase extinction check (FDS's coarse-grid criterion, limiting oxygen 0.135 at 20 °C, rising with temperature to a free-burn 600 °C; or Rasbash's 14.5 to 15 % for diesel flames in steam and nitrogen). The solver now extinguishes only through the Arrhenius rate and the oxygen cap, as Fire-X does.
- The Hamins and McGrattan form as an alternative to wetting for cartoned commodities (a1 = 1.0 m²/(kg s), M''_w = 0.24 s x flux, with the k2 regrowth term).
- A boilover multiplier for high fire point pools on water, once a layer thickness exists.

Statements for Lean, if the gates are built: water routed to a gated cell leaves its `moisture` and `Ts` unchanged (so the off and gated paths agree); the spray's water budget still closes with "passed under" as a fourth sink; the metal-water bound releases exactly 315.8 kJ per mol of water at the metal and, with the hydrogen burnt, 557.6 kJ, the air value less water's 44.0 kJ/mol (a Hess's-law identity on the table's values).

---

## 11. Claims I am least sure of

1. Magee and Reitz's critical fluxes (section 4.2) come from Rasbash's Table 2 as extracted from the PDF; the table's columns were scrambled, and the assignment of values to polymers and orientations is my reading, guided by Rasbash's text (polystyrene and polyethylene needed more water).
2. The solver's own critical flux is several times the measured one (section 4.5) is an inference from h = 200 W/(m² K), not a run; the critical-flux test would settle it.
3. Xin and Tamanini's ADD correlation was fitted for 3.05 m clearance and 10.5 to 31.8 mm/min; its use at 4.1 to 8.1 mm/min and under 2.4 to 2.7 m ceilings is an extrapolation.
4. f_c of 0.1 to 0.4 in Liu et al.'s balance is their citation of another source; Rasbash's phi of 0.3 is a suggested mean. The cooking oil critical fluxes (52 to 221 g/(m² s)) also rest on the app's provisional oil properties.
5. NFPA 13-2022 extra hazard densities come from a secondary copy of the table (an errata sheet seen through a search index), not the standard.
6. Abstract-only claims: Fan et al. 2020 (flame expansion ratios), Xu et al. 2018 (210 °C threshold), Wang et al. 2023, Heskestad 2003, Xu et al. 2022 (126 °C, +36 °C), Liu et al. 2019 and 2021, Choi et al. 2025, Ferrero et al. 2006 and 2007 (the thesis table was read).
7. Ferrero's thesis Table 4.10 was read from an OCR of a Spanish scan; the mean and peak columns may be offset in one row (the 1.04 mm/m² row).
8. The sodium and lithium heats use aqueous products at 298 K, right for cold water on cold metal; at a burning metal's temperature the products (molten hydroxide, oxide) differ and the heats change, by an amount not computed here.
9. That no study measured the heat release jump or ejected oil mass for water on cooking oil rests on the searches made on 2026-11-07; a study outside the indexes searched may exist.
10. Hamins and McGrattan's Table 2, test 15, lists 0.13 kg/m² for a 0.053 kg/(m² s) flux, inconsistent with the stated 0.24 s factor (0.013 expected); probably a typographical error in the paper.

---

## 12. Sources

DOIs were resolved at `api.crossref.org/works/<doi>` on 2026-11-07; title, authors and year below are as Crossref returned them. Read status follows each entry.

Standards, codes and reference data (values and clause numbers only)

- NFPA 13, Standard for the Installation of Sprinkler Systems, 2022 edition, Table 19.2.3.1.1 (extra hazard densities). Seen through a search-index summary of an NFPA errata sheet; the standard was not read.
- NFPA 10, Standard for Portable Fire Extinguishers (the class list and agents, via `fireClasses.js`).
- CODATA Key Values for Thermodynamics (Cox, Wagman, Medvedev, 1989), web table at codata.info/resources/databases/key1.html. Read (H2O, MgO, Al2O3, Li+, Na+, OH-).
- NIOSH Pocket Guide to Chemical Hazards, n-heptane (npgd0312) and ethyl alcohol (npgd0262). Read.

Reports and theses without DOIs

- Ditch, B., Zeng, D. (2019). Development of Sprinkler Protection Guidance for Lithium Ion Based Energy Storage Systems. FM Global, Project RW000029. Executive summary read (copy at lak.bremen.de).
- Ferrero, F. (2006). Incendios de hidrocarburos: estudio de la formación y evolución del boilover de capa fina. Doctoral thesis, Universitat Politècnica de Catalunya. Chapter 4 read (Table 4.10).
- FDS User's Guide, section "Suppression by Water" (E_COEFFICIENT), LaTeX source, github.com/firemodels/fds, master, 2026-11-07. Read. The FDS Validation Guide's suppression chapter was checked: it no longer covers the commodity tests.
- UL FSRI research update, "Journal article examines impact of residential fire sprinklers on e-scooter fires" (fsri.org). Read.

Articles and reports with DOIs

- Choi, S., Lee, K., Kim, J., et al. (2025). Early Detection and Suppression of Thermal Runaway in Large-Format Lithium-Ion Batteries: Insights from Experimental Analysis. Energies 18(1), 155. doi:10.3390/en18010155 (abstract).
- Evans, D. D. (1993). Sprinkler fire suppression algorithm for HAZARD. NISTIR 5254. doi:10.6028/NIST.IR.5254 (read in full for the physics report).
- Fan, X., Wang, C., Guo, F. (2020). Experimental study of flame expansion induced by water droplet impact on the burning cooking oil. Fuel 270, 117497. doi:10.1016/j.fuel.2020.117497 (abstract via search index).
- Ferrero, F., Muñoz, M., Kozanoglu, B., Casal, J., Arnaldos, J. (2006). Experimental study of thin-layer boilover in large-scale pool fires. Journal of Hazardous Materials 137(3), 1293-1302. doi:10.1016/j.jhazmat.2006.04.050 (abstract via search index).
- Ferrero, F., Muñoz, M., Arnaldos, J. (2007). Thin-layer boilover in diesel-oil fires: determining the increase of thermal hazards and safety distances. Journal of Hazardous Materials 140(1-2), 361-368. doi:10.1016/j.jhazmat.2006.09.062 (abstract via search index).
- Fleischmann, C., Weinschenk, C., Madrzykowski, D., Schraiber, A., Gaudet, B. (2025). Quantifying the Fire Hazard from Li-Ion Battery Fires Caused by Thermal Runaway in E-scooters. Fire Technology 61(5), 2865-2887. doi:10.1007/s10694-025-01707-z (via `fireClasses.js`).
- Hamins, A., McGrattan, K. (1999). Reduced-scale experiments to characterize the suppression of rack-storage commodity fires. NISTIR 6439. doi:10.6028/NIST.IR.6439 (the NIST file is a scan without text; not read).
- Hamins, A., McGrattan, K. (2003). Reduced-scale Experiments On The Water Suppression Of A Rack-storage Commodity Fire For Calibration Of A CFD Fire Model. Fire Safety Science 7, 457-468. doi:10.3801/IAFSS.FSS.7-457 (read in full).
- Hamins, A., Kim, S. C., Madrzykowski, D., Kent, J. (2018). Investigation of residential cooking fire suppression technologies. NIST Technical Note 1969. doi:10.6028/NIST.TN.1969 (read: sections 2.2, 3.1, 5, 6.2.4, 7).
- Hamins, A., Kim, S., Madrzykowski, D. (2018). Characterization of stovetop cooking oil fires. Journal of Fire Sciences 36(3), 224-239. doi:10.1177/0734904118761884 (via `fireClasses.js`).
- Heskestad, G. (2003). Extinction of gas and liquid pool fires with water sprays. Fire Safety Journal 38(4), 301-317. doi:10.1016/S0379-7112(02)00085-1 (abstract via search index).
- Hu, Y., Chen, J., Wang, Z., et al. (2021). The Character of Residential Cooktop Fires. NIST Technical Note 2133. doi:10.6028/NIST.TN.2133 (via `fireClasses.js`).
- Kung, H.-C., Hill, J. P. (1975). Extinction of wood crib and pallet fires. Combustion and Flame 24, 305-317. doi:10.1016/0010-2180(75)90164-9 (not read; via Yu et al.).
- Larsson, F., Andersson, P., Blomqvist, P., Mellander, B.-E. (2017). Toxic fluoride gas emissions from lithium-ion battery fires. Scientific Reports 7, 10018. doi:10.1038/s41598-017-09784-z (water mist passages read).
- Liu, T., Liu, Y., Wang, X., Kong, X., Li, G. (2019). Cooling control of thermally-induced thermal runaway in 18,650 lithium ion battery with water mist. Energy Conversion and Management 199, 111969. doi:10.1016/j.enconman.2019.111969 (abstract via search index).
- Liu, T., Tao, C., Wang, X. (2020). Cooling control effect of water mist on thermal runaway propagation in lithium ion battery modules. Applied Energy 267, 115087. doi:10.1016/j.apenergy.2020.115087 (title only).
- Liu, T., Hu, J., Tao, C., Zhu, X., Wang, X. (2021). Effect of parallel connection on 18650-type lithium ion battery thermal runaway propagation and active cooling prevention with water mist. Applied Thermal Engineering 184, 116291. doi:10.1016/j.applthermaleng.2020.116291 (abstract via search index).
- Liu, Z., Carpenter, D., Kim, A. (2005). Application Of Water Mist To Extinguish Large Oil Pool Fires For Industrial Oil Cooker Protection. Fire Safety Science 8, 741-752. doi:10.3801/IAFSS.FSS.8-741 (read).
- Madrzykowski, D., Vettori, R. (1992). A sprinkler fire suppression algorithm for the GSA engineering fire assessment system. NISTIR 4833. doi:10.6028/NIST.IR.4833 (not read; via Evans).
- Madrzykowski, D. (2008). Impact of a residential sprinkler on the heat release rate of a Christmas tree fire. NISTIR 7506. doi:10.6028/NIST.IR.7506 (abstract and results read).
- Madrzykowski, D., Fleischmann, C., Weinschenk, C. (2025). Examining the Impact of Residential Fire Sprinklers on e-Scooter Fires Initiated by Thermal Runaway of Li-Ion Batteries. Fire Technology 61(7), 5753-5772. doi:10.1007/s10694-025-01808-9 (not read; FSRI summary read).
- Magee, R. S., Reitz, R. D. (1975). Extinguishment of radiation augmented plastic fires by water sprays. Symposium (International) on Combustion 15(1), 337-347. doi:10.1016/S0082-0784(75)80309-2 (not read; via Rasbash 1986).
- Mason, P. E., Uhlig, F., Vaněk, V., Buttersack, T., Bauerecker, S., Jungwirth, P. (2015). Coulomb explosion during the early stages of the reaction of alkali metals with water. Nature Chemistry 7(3), 250-254. doi:10.1038/nchem.2161 (not read).
- Nishii, K., Koizumi, H. (2024). Temperature measurement on condensed combustion products of magnesium with oxygen and water vapor. Combustion and Flame 262, 113362. doi:10.1016/j.combustflame.2024.113362 (title only).
- Noaki, M., Delichatsios, M. A., Yamaguchi, J., Ohmiya, Y. (2018). Heat release rate of wooden cribs with water application for fire suppression. Fire Safety Journal 95, 170-179. doi:10.1016/j.firesaf.2017.10.002 (abstract, via the extinguishers report).
- Prachukho, V. P., Ozerov, E. S., Yurinov, A. A. (1971). Burning of magnesium particles in water vapor. Combustion, Explosion and Shock Waves 7(2), 195-198. doi:10.1007/BF00748971 (title only).
- Rasbash, D. J., Rogowski, Z. W. (1957). Extinction of fires in liquids by cooling with water sprays. Combustion and Flame 1(4), 453-466. doi:10.1016/0010-2180(57)90006-8 (not read; via Rasbash 1986).
- Rasbash, D. J., Rogowski, Z. W., Stark, G. W. V. (1960). Mechanisms of extinction of liquid fires with water sprays. Combustion and Flame 4, 223-234. doi:10.1016/S0010-2180(60)80026-0 (not read; via Rasbash 1986).
- Rasbash, D. J. (1986). The Extinction Of Fire With Plain Water: A Review. Fire Safety Science 1, 1145-1163. doi:10.3801/IAFSS.FSS.1-1145 (read, pp. 1145-1159).
- Tamanini, F. (1976). The Application of Water Sprays to the Extinguishment of Crib Fires. Combustion Science and Technology 14(1-3), 17-23. doi:10.1080/00102207608946742 (not read; via Rasbash and Yu et al.).
- Walton, W. D. (1988). Suppression of wood crib fires with sprinkler sprays. NBSIR 88-3696. doi:10.6028/NBS.IR.88-3696 (not read; via Evans).
- Wang, C., Zhang, X., Huang, Y., Xie, J., Xu, M., Zhang, J. (2023). Flame expansion behavior induced by a water droplet impacting on burning and stratified oil-water pool. International Journal of Thermal Sciences 191, 108342. doi:10.1016/j.ijthermalsci.2023.108342 (abstract via search index).
- Wrede, H., Wagner, A., Mahfuz, S., Palubicki, W., Michels, D., Pirk, S. (2025). Fire-X: Extinguishing Fire with Stoichiometric Heat Release. ACM Transactions on Graphics 44(6). doi:10.1145/3763338 (repository transcription).
- Xin, Y., Tamanini, F. (2008). Assessment of Commodity Classification for Sprinkler Protection Using Representative Fuels. Fire Safety Science 9, 527-538. doi:10.3801/IAFSS.FSS.9-527 (read, pp. 527-535).
- Xu, J., Duan, Q., Zhang, L., Liu, Y., Zhao, C., Wang, Q. (2022). Experimental study of the cooling effect of water mist on 18650 lithium-ion battery at different initial temperatures. Process Safety and Environmental Protection 157, 156-166. doi:10.1016/j.psep.2021.10.034 (abstract via search index).
- Xu, M., Zhang, J., Wu, C., Li, C., Chen, X., Lu, S. (2018; Crossref lists 2017 online). Collision dynamics of a single water droplet impinging on a high-temperature pool of oil. Acta Mechanica 229(4), 1567-1577. doi:10.1007/s00707-017-2071-5 (abstract via the INIS record).
- Yu, H.-Z., Lee, J. L., Kung, H.-C. (1994). Suppression Of Rack-storage Fires By Water. Fire Safety Science 4, 901-912. doi:10.3801/IAFSS.FSS.4-901 (read in OCR; Crossref also lists a duplicate author record).
