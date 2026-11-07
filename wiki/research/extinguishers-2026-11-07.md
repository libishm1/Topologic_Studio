# Portable fire extinguishers: placement on the walking graph and manual suppression in a run (research, 2026-11-07)

Status: research only, no code. Companion to `fire-suppression-sprinklers-2026-11-07.md` (whose "Left" list ends with extinguishers), `sprinkler-codes-2026-11-07.md` (how codes are cited), `sprinkler-physics-2026-11-07.md` (Evans), `ignition-sources-2026-11-05.md` (fire classes) and `rset-components-2026-11-07.md` (pre-movement). The existing code this note builds on is `src/core/fireClasses.js` (suitable and unsuitable agents per class) and `src/core/suppression.js` (the Evans cap on the solver's fuel cells).

How sources were handled. SBC 801-CR-2018 Section 906 and Tables 906.1, 906.3(1) and 906.3(2) were read in the user's licensed copy, which stays outside the repository; only clause numbers and values appear here. NFPA 10 was read in NFPA's public Second Draft Report for the A2025 cycle (the 2022 text with the committee's revisions), so its clause and table numbers are those of the next edition and may differ by one revision from the 2022 edition. UL 711 was read in its 7th edition (2004, revised to 2013) through a public incorporation-by-reference copy; the current edition was not read. EN 3-7, ISO 7165, BS 5306-8 and the IFC/IBC 2021 Section 906 text were not read; their values come from secondary sources and are marked so. Standards are cited by clause and value only. Every DOI in section 8 was resolved at api.crossref.org on 2026-11-07 (the NIST database DOI at DataCite). Web text was treated as data. Two research agents gathered sections 2 to 4; their key values were checked here against the extracted texts (UL 711 7.1.3, 7.2.6, 7.2.8, 5.4 and Table 2; Rasbash 1986 pp. 1149 and 1157; the NIST crib record).

## 0. Summary

- Where required. SBC 801-CR-2018 906.1 requires portable extinguishers in new and existing Group A, B, E, F, H, I, M, R-1, R-2, R-4 and S occupancies, within 9 m of commercial cooking equipment, where flammable liquids are stored, used or dispensed, on each floor under construction, where Table 906.1 says so, and in special-hazard areas such as laboratories, computer rooms and generator rooms where the official requires. A university building (Group B, A-3 lecture halls, laboratories) is covered throughout.
- Placement, class A (SBC 801-CR-2018 Table 906.3(1); NFPA 10 Table 6.2.1.1): smallest single extinguisher 2-A (light and ordinary hazard) or 4-A (extra); at most 279 m² (3000 ft²), 139 m² (1500 ft²) or 92.9 m² (1000 ft²) of floor per unit of A; at most 1045 m² (11,250 ft²) per extinguisher; travel distance at most 22.86 m (75 ft; SBC prints 23 m, NFPA 22.9 m). NFPA 10 E.3.3 defines that distance as the actual walking distance, which is what the walking graph measures.
- Placement, class B spill fires (SBC Table 906.3(2); NFPA 10 Table 6.3.1.1): 5-B or 10-B (light), 10-B or 20-B (ordinary), 40-B or 80-B (extra), at most 9 m or 15 m of travel respectively (NFPA 30 ft and 50 ft, 9.14 and 15.25 m). Class K: 9 m (SBC 904.12.5; NFPA 6.6.2, 9.1 m). Class D: 22.9 m (NFPA 6.5.2).
- What a rating is. UL 711 class A ratings scale with the wood in a crib of 38 mm sticks (1-A: 72 sticks 500 mm long in 12 layers, about 0.052 m³ of wood), attacked at 55 % of its mass from at least 1.8 m, with no re-ignition for 15 min. A 9 L water extinguisher is 2-A without a fire test. The UL class B number is 40 % of the heptane pan area (in ft²) an expert consistently extinguishes. No measured heat release rate of any rated test fire was found; from the crib's wood and a measured NIST crib (36.5 kg, 396 kW peak), one UL A unit is about 0.3 MW at attack (0.2 to 0.45 MW, derived).
- Discharge. Hand units empty in 8 s to 60 s (NFPA 10 Annex text); 9.5 L water about 1 min at 9.1 to 12.2 m; 2.3 kg ABC powder about 14 s at 3.7 to 5.5 m; 4.5 kg ABC about 20 s; 2 kg CO2 about 8.5 s at about 3 m. UL 711 7.1.3 requires at least 13 s for 2-A and above.
- People. Extinguishers were used in 5.0 % of US residential fires and fully put the fire out in about half of those uses (CPSC 2004 to 2005); in Sweden hand extinguishers put out the fire in 54 % and limited it in 27 % of the attempts made with them before the fire service arrived (attended residential fires, 2022 to 2025); in Tokyo 81 % of all first-aid attempts (any method) succeeded (2022). About 15 % of people's first actions in Wood's 1972 UK study were fighting the fire. Untrained people took 13.4 s on average from "go" to discharge with the extinguisher at their side; 88 % aimed at the flames rather than the burning material.
- IFC. No IFC version has an extinguisher item: IfcFireSuppressionTerminalTypeEnum lists breeching inlet, fire hydrant, hose reel, sprinkler, sprinkler deflector and (since IFC4.3 RC3) fire monitor. Of 198 local IFC files, three distinct models carry extinguishers, all from Revit: 6 IfcBuildingElementProxy (a CO2 extinguisher family), 10 IfcFlowTerminal typed by IfcSanitaryTerminalType (cabinets, in four copies of the NIBS clinic) and 2 IfcBuildingElementProxy (cabinets). None uses IfcFireSuppressionTerminal, IfcFurnishingElement or the token "FEC".
- Recommendation (section 6). A placement check by multi-source Dijkstra on the walking graph (exact against the walking-distance rule, conservative against a radius check), a greedy cover layout, and a manual-suppression card: one responder walks to the nearest suitable extinguisher and back, and the fire's heat release within reach when discharge starts is compared with the extinguisher's capacity in three outcomes (success, cannot tell, failure). On success the reached cells take the existing Evans cap with tau = t_d / ln 100; on failure the fire is unchanged and the responder loses the walk, the readying and the discharge time. The simplest honest version is the card alone, with the fire unchanged.

## 1. Where extinguishers are required and how they are placed

### 1.1 SBC 801-CR-2018 Section 906 (read; values only)

| Clause | Value |
|---|---|
| 906.1 item 1 | New and existing Groups A, B, E, F, H, I, M, R-1, R-2, R-4 and S. Exception: in R-2, only the locations of items 2 to 6 where each dwelling unit has an extinguisher of at least 1-A:10-B:C |
| 906.1 item 2 | Within 9 m of commercial cooking equipment |
| 906.1 items 3 to 6 | Where flammable or combustible liquids are stored, used or dispensed; each floor of structures under construction (3315.1, Group R-3 excepted); the sections of Table 906.1 (40 special uses, from asphalt kettles to LP-gas); special-hazard areas such as laboratories, computer rooms and generator rooms where the official requires |
| 906.2 | Selection, installation and maintenance per this section and NFPA 10. Exceptions: travel distance not applied to A-5 spectator seating; electronic monitoring may replace 30-day inspections; I-3 at staff locations |
| 906.3.1, Table 906.3(1) | Class A, light / ordinary / extra hazard: smallest single extinguisher 2-A / 2-A / 4-A; floor area per unit of A 280 / 139 / 93 m²; area per extinguisher 1045 m² in all three; travel distance 23 m. Footnotes: two 9.5 L water units count as one 4-A; two 1-A water units count as one 2-A in light hazard |
| 906.3.2, Table 906.3(2) | Class B, liquids at most 6 mm deep: light 5-B within 9 m or 10-B within 15 m; ordinary 10-B within 9 m or 20-B within 15 m; extra 40-B within 9 m or 80-B within 15 m. Deeper liquids per NFPA 10 |
| 906.3.3 | Class C sized and placed on the anticipated class A or B hazard |
| 904.12.5 | Commercial cooking: extinguishers within 9 m of travel; class K for vegetable or animal oils and fats. 904.12.5.1: solid-fuel appliances with fireboxes up to 0.14 m³, one 9 L or two 6 L class K wet-chemical units. 904.12.5.2: one 6 L class K unit for up to four fryers of at most 36 kg of cooking medium each, and one more for each further group of four |
| 906.5, 906.6 | Easy to see and to reach, on the routes people normally walk unless the official decides otherwise; kept clear and in view, or their places signed |
| 906.8 | Cabinets not locked (exceptions for malicious use, I-3 and mental health areas of I-2) |
| 906.9.1 to 906.9.3 | Gross weight up to 18 kg: top at most 1500 mm above the floor; over 18 kg: top at most 1100 mm; bottom at least 100 mm above the floor |
| 906.10 | Wheeled units in a designated, conspicuous location |

### 1.2 IFC/IBC 2021 (not read)

The IFC/IBC 2021 text of Section 906 was not obtained. Two points matter for the user's rule (SBC 801-CR-2018, with IBC 2021 where stricter):

- The model code once exempted Groups A, B and E with quick-response sprinklers throughout from most extinguishers. A Fire Equipment Manufacturers' Association newsletter (2020) reports that the 2012 IFC dropped that exception; Virginia and Minnesota keep versions of it as state amendments. SBC 801-CR-2018 has no such exception, so no relaxation applies either way.
- State adoptions of IFC 2021 906.1 item 2 add domestic cooking equipment in Group I-1, I-2 Condition 1 and R-2 college dormitories (New York's wording; Minnesota's reads "R-2 congregate living facilities") to the 30 ft (9144 mm) cooking rule. If the base IFC 2021 text says the same, it is stricter than SBC 2018 and should be applied to dormitories. To confirm when the IFC 2021 text is read.

### 1.3 NFPA 10 (Second Draft Report, A2025 cycle; values only)

| Clause | Value |
|---|---|
| 5.4.1.1 to 5.4.1.3 | Hazard by room or area: light (class B flammables under 3.8 L in any room), ordinary (3.8 to 18.9 L), extra (over 18.9 L, or high quantities of class A materials) |
| Table 6.2.1.1 | Class A: 2-A / 2-A / 4-A; 3000 ft² (279 m²) / 1500 ft² (139 m²) / 1000 ft² (92.9 m²) per unit of A; 11,250 ft² (1045 m²) per extinguisher; 75 ft (22.9 m) travel |
| 6.2.1.2.1 to 6.2.1.2.3 | Number per floor = floor area / area per extinguisher; then add units until no point is beyond the travel distance |
| 6.2.1.3.1, 6.2.1.3.2 | Two 1-A water units may serve as one 2-A; two 9.46 L (2.5 gal) water units as one 4-A |
| 6.2.1.4 | Up to half the class A complement may be replaced by 38 mm (1.5 in) hose stations, no more than every other extinguisher |
| 6.2.1.5, E.3.2 | A floor smaller than the table area still needs one extinguisher of the minimum size |
| E.3.3 | Travel distance is the actual walking distance from any point to the nearest extinguisher |
| E.3.4 | The 1045 m² limit is the square (32 m x 32 m) inscribed in a 22.9 m circle |
| Table E.3.5 | Area per extinguisher, light / ordinary / extra (m²): 2-A 557 / 279 / none; 3-A 836 / 418 / none; 4-A 1045 / 557 / 372; 6-A 1045 / 836 / 557; 10-A 1045 / 1045 / 929; 20-A and up 1045 |
| Table 6.3.1.1 | Class B spill fires: as SBC Table 906.3(2), with 30 ft (9.14 m) and 50 ft (15.25 m); the table's note says the ratings do not imply fires of that size will occur |
| 6.3.2.1 to 6.3.2.6 | Liquids of appreciable depth: extinguishers not the sole protection above 0.93 m² (10 ft²) of surface; at least 2 B units per 0.09 m² (1 ft²) of the largest hazard (foam: 1 B unit); travel at most 15.25 m |
| 6.3.3.1 | Obstacle, three-dimensional and pressure fires: at most 9.1 m (30 ft) |
| 6.4.1 to 6.4.3 | Class C where energised equipment may be met; sized on the class A or B hazard |
| 6.5.2 | Class D: at most 22.9 m (75 ft) |
| 6.6.2 | Class K: at most 9.1 m (30 ft) |
| 6.7 | Solid-fuel cooking appliances up to 0.14 m³: a 2-A water unit or a 6 L class K wet-chemical unit |
| F.6.1 (annex, dwellings) | One 2-A:10-B:C per floor level of a living unit, at most 12 m (40 ft) of travel |
| F.6.2.2 (annex) | Near room exits that give an escape route; top at most 1.5 m, bottom at least 0.1 m |

NFPA 10 A.5.4.1.1 to A.5.4.1.3 give examples: light hazard for offices, classrooms, churches, assembly halls and hotel guest rooms; ordinary for dining areas, mercantile, light manufacturing, research operations, auto showrooms, workshops and support areas of light-hazard buildings; extra for woodworking, vehicle repair, parking garages, aircraft and boat servicing, cooking areas, product display showrooms and painting, dipping or coating. Two of these differ from the NFPA 13 classes the sprinkler layout uses (sprinkler note, row 43): parking garages are ordinary group 1 for sprinklers but extra hazard for extinguishers, and kitchens ordinary group 1 for sprinklers but extra for extinguishers. The extinguisher check therefore needs its own hazard per room, not the sprinkler hazard. A university laboratory is "research operations", ordinary hazard, with class B units where flammable liquids are used (A.5.5 gives the hospital-laboratory example).

### 1.4 BS 5306-8 and EN 3 for comparison (not read; secondary)

- BS 5306-8:2012 was withdrawn on 30 November 2023 and replaced by BS 5306-8:2023 (NSI technical bulletin 0065).
- Class A rule as reported by suppliers and fire services: the total class A rating per storey is at least 0.065 times the floor area in m² (one 13A per 200 m²), with at least 26A per storey, at least two class A units except on very small storeys (reports differ: 50 m² or 400 m²), and at most 30 m to the nearest extinguisher, which overrides the area rule.
- The 2023 revision bases class A provision on a 9 L water extinguisher achieving 13A. UL 711 Table 1 rates 9.0 L of water 2-A. On that bridge 13A is about 2-A, and an office (light hazard) needs about 2.8 times more class A rating per m² under BS 5306-8 (13A per 200 m²) than under NFPA 10 (2-A per 557 m²), against a longer travel distance (30 m against 22.9 m).
- EN 3-7 class A test fires are long wood cribs 500 mm wide, about 560 mm high (14 layers of 39 mm sticks), n x 100 mm long for rating nA (13A: 1.3 m), after O'Dogherty, Young and Lange (Fire Research Note 731, 1968). Class B trays hold B litres (one third water, two thirds heptane) over pi B / 100 m² (8B 0.25 m², 13B 0.41, 21B 0.66, 34B 1.07 m²: IS 15683:2006 Table 9, an EN-type Indian standard, read; larger ratings by the same rule, derived). The EN 3-7 tables of minimum ratings per charge and minimum discharge times were not obtained.

## 2. What a rating represents physically

### 2.1 UL 711 class A (7th edition, read)

- Crib (7.2.1, Table 2): 38 x 38 mm kiln-dried spruce or fir sticks at 9 to 15 % moisture, square cribs, on a stand 400 mm high (7.2.2), lit by heptane in a square pan beneath (7.2.4, Table 3).

| Rating | Sticks | Length (mm) | Layers | Wood (m³, derived) | Pan (mm), heptane (L) |
|---|---|---|---|---|---|
| 1-A | 72 | 500 | 12 of 6 | 0.052 | 400, 1.1 |
| 2-A | 112 | 635 | 16 of 7 | 0.103 | 535, 2.0 |
| 3-A | 144 | 735 | 18 of 8 | 0.153 | 635, 2.8 |
| 4-A | 180 | 800 | 20 of 9 | 0.208 | 700, 3.4 |
| 6-A | 230 | 925 | 23 of 10 | 0.307 | 825, 4.8 |
| 10-A | 324 | 1100 | 27 of 12 | 0.515 | 1000, 7.0 |

  The derived wood volume is in proportion to the rating number (ratios 1, 1.98, 2.94, 4.0, 5.9, 9.9). Ratings of 20-A and above use 38 x 89 mm sticks on edge.
- Attack (7.2.6): when the crib has lost 45 % of its mass (55 +/- 1 % left), from the front at 1.8 m (6 ft) or more, then closer and from the sides, top and bottom but never the back; discharge continuous to empty.
- Pass (7.1.1, 7.1.2, 7.2.8): no re-ignition or continued smouldering within 15 min after discharge, in 2 of 3 crib fires. Ratings up to 6-A also need the wood panel test (Table 4: 2.45 m square panel and 4.55 kg of excelsior for 1-A up to 5.20 m and 27.2 kg for 6-A).
- Discharge (7.1.3): at least 13 s for 2-A and above. NFPA 10 A.4.4.1(2) records that this came from UL's novice-operator tests of the early 1980s, which showed that longer discharge raised the chance of extinguishment; before 1984 most 2-A dry chemical units emptied in 8 to 10 s.
- Water equivalence (Table 1): water units up to 125 L are rated by capacity alone: 4.5 L 1-A, 9.0 L 2-A, 14 L 3-A, 18 L 4-A, 36 L 6-A, 64 L 10-A. About 4.5 L of water per A unit up to 4-A, then 6 to 6.4 L.
- NFPA 10 G.2 states the meaning: a 4-A unit should put out about twice the class A fire of a 2-A unit (a 9.46 L water unit).

### 2.2 UL 711 class B (read)

- 5.4: the B number is 40 % of the pan area (in ft²) an expert operator consistently extinguishes, an approximate relative indication. So the expert's pan is 0.232 m² (2.5 ft²) per B unit, and the B number in ft² (0.0929 m² per unit) is the conventional area for an ordinary operator. The "non-expert" reading is common in guidance; it is not UL's wording.
- Table 5 and 8.2: square steel pans, at least 51 mm of heptane over water, 60 s pre-burn; 1-B 0.23 m², 5-B 1.16 m², 10-B 2.32 m², 20-B 4.65 m²; minimum discharge 8 s up to 20-B.
- Ratings in use (NFPA 10 G.2): 1-A, 2-A, 3-A, 4-A, 6-A, 10-A, 20-A, 30-A, 40-A; 1-B to 640-B in 15 steps. Ratings up to 20-A and 20-B are from indoor tests.

### 2.3 Heat release of the test fires

No measured heat release rate of a rated UL or EN test fire was found. Two anchors:

- NIST Fire Calorimetry Database, Test13_woodCrib_r1: one crib 0.56 x 0.56 x 0.46 m (22 x 22 x 18 in), 39.4 kg, 36.5 kg burnt, peak 396 +/- 19 kW at 23 min, 597 +/- 31 MJ (16.3 MJ per kg burnt). Stick size not stated.
- The UL 1-A crib is about the same size (0.5 x 0.5 x 0.46 m) with about 0.052 m³ of wood, 21 to 26 kg at 400 to 500 kg/m³ (density assumed). Scaled by mass, the NIST crib gives about 0.24 MW at peak; Babrauskas's crib correlations (SFPE Handbook, not re-read here) give 0.39 MW (porosity-controlled) to 0.47 MW (surface-controlled). The UL attack at 45 % mass loss is near the crib's peak.

The model therefore takes q_A = 0.3 MW per UL A unit at attack, with a range of 0.2 to 0.45 MW. This is a derived value, the least certain physical number in the note.

Class B pans (heptane, the solver's own n-heptane values in `fireClasses.js`: 0.101 kg/(m² s), k beta 1.1 /m, 44.1 MJ/kg), free burning: UL 1-B pan 0.23 m² about 0.45 MW; 10-B expert pan 2.32 m² about 8.8 MW; the 10-B ordinary-operator area 0.93 m² about 2.9 MW; EN 21B tray 0.66 m² about 1.9 MW. The app's class B item (a 1 m² pool, about 3.2 MW) is close to a 10-B unit's ordinary-operator area.

### 2.4 Ratings across standards

- 13A (EN) is about 2-A (UL) by the 9 L water bridge (UL 711 Table 1; BS 5306-8:2023 as reported), so one EN A unit is about 2/13 of a UL A unit, about 0.046 MW on the estimate above.
- For powder the bridge fails: a 6 kg EN ABC unit is typically rated 34A to 55A and 183B to 233B (GLORIA 2026 catalogue), while a 4.5 kg (10 lb) UL ABC unit is typically 4-A:80-B:C (Amerex B456). The EN B tray of 233B is 7.3 m² (pi B / 100), the UL expert pan of 80-B is 18.6 m². Class B capacities across the two systems disagree by a factor of 2 to 3, and the A capacities of powder units by about 2.
- FRN 731 (O'Dogherty, Young and Lange 1968, read in OCR) found that the water needed rose in proportion to crib length, about 5 L per metre of its long crib, for 5.7 L and 9.1 L extinguishers. That supports a capacity linear in the rating for water units.

### 2.5 What real fire a rating can put out

No published study was found that reports the heat release rate or fire area of a real item (upholstered furniture, a mattress, cooking oil) at which a rated extinguisher fails. Candidate sources, found and DOI-checked but not read: Kokkala (1986) on portable chemical extinguishers and water in compartment fires; Liu, Kim and Carpenter (2007) on portable water-mist units against cooking oil, heptane, diesel and cribs; Rappsilber, Below and Krüger (2019) on crib tests by agent and jet type at close range. The capacity in section 6 is therefore the test fire's, scaled, not a measured capacity against furnishings.

## 3. Discharge and application rates

### 3.1 Agent, duration and throw

NFPA 10 Table H.2 (annex; the extracted table's columns are partly scrambled for the dry chemical rows, so those are given as ranges) and manufacturer sheets:

| Unit | Rating | Discharge | Range | Source |
|---|---|---|---|---|
| Water 6 L (1.6 gal), stored pressure | 1-A | 40 s | 9.1 to 12.2 m | NFPA 10 Table H.2 |
| Water 9.5 L (2.5 gal) | 2-A | 1 min (55 s, Amerex 240) | 9.1 to 12.2 m (13.7 to 16.8 m, Amerex) | Table H.2; Amerex |
| Foam 9.5 L | 3-A:20 to 40-B | 50 s | 6.1 to 7.6 m | Table H.2 |
| Water mist 6.8 to 9.5 L | 2-A:C | 50 to 80 s | 1.5 to 3.7 m | Table H.2 |
| CO2 1.1 to 2.3 kg (2.5 to 5 lb) | 1 to 5-B:C | 8 to 30 s | 0.9 to 2.4 m | Table H.2 |
| CO2 2 kg / 5 kg | 34B / 89B | 8.5 s / 13.5 s | about 3 m / 4 to 5 m | GLORIA 2026 |
| ABC powder 1.1 kg (2.5 lb) | 1-A:10-B:C | 10 s | 2.7 to 4.6 m | Amerex B417 (retailer copy) |
| ABC powder 2.3 kg (5 lb) | 2-A:10-B:C or 3-A:40-B:C | 14 s | 3.7 to 5.5 m | Amerex B500, B402 |
| ABC powder 4.5 kg (10 lb) | 4-A:80-B:C | 20 s | 4.6 to 6.4 m | Amerex B456 |
| ABC powder 9.1 kg (20 lb) | 10-A:120-B:C | 30 s | 4.6 to 6.4 m | Amerex A411 |
| ABC powder 6 kg (EN) | 34A to 55A, 183B to 233B | about 16 to 23 s | about 6 m | GLORIA 2026 (columns interleaved; medium confidence) |
| ABC powder, hand units 0.5 to 14 kg | 1-A to 20-A | 8 to 25 s | 1.5 to 6.1 m | Table H.2 (ranges) |

NFPA 10 Annex F (homes) notes that hand units of the sizes discussed empty in 8 to 60 s.

### 3.2 Water needed to put out wood

From Rasbash's 1986 review (read):

- Heskestad's review: critical water flux for wood, slabs to cribs, 1.3 to 3.0 g/(m² s) of exposed fuel surface; the higher values for densely packed cribs.
- O'Dogherty's overhead tests on 0.9 m cribs of 25 mm sticks (about 48 m² of fuel per m² of top): 187 g/(m² s) on the top, about 4 g/(m² s) of fuel surface, put the fire out in about 10 min once the flames were beaten down.
- O'Dogherty's extinguisher test (FRN 731): about 6 g/(m² s) put out a fairly open crib in about 1 min, which Rasbash reads as a critical rate of about 2.5 g/(m² s).
- Room fires (Rasbash Table 4): the least water rate that achieved extinction was about 6 g/(m² s) of fuel area.

Särdqvist's thesis (Lund 2000, read by the agent) gives a minimum extinguishing flow of about 2 g/(m² s) of fuel surface (printed as 0.002 per m² s; the unit is probably kg), an optimum of about ten times that, and a wood mass loss rate at extinction of about 0.0055 kg/(m² s).

Noaki et al. (2018), overhead water on wooden cribs: the reduction of peak heat release rate was in proportion to the total water flow, about 12 kJ per g of water (abstract via the publisher's index; full text not read). For a 9.5 L / 60 s unit (about 158 g/s) that bounds the reduction at about 1.9 MW, larger than the 2-A capacity of section 2.3 (about 0.6 MW at q_A = 0.3 MW); the rating, not the water, is the binding limit.

### 3.3 Powder

Särdqvist (2000): the ratio of powder mass flow to fuel vapour mass flow at extinction (REMP) is about 1.0 to 1.5 for pool fires and about ten times higher for gases; much of the powder applied misses the flame, and powder protects poorly against re-ignition. For heptane at 0.07 to 0.10 kg/(m² s), that is roughly 0.07 to 0.15 kg/(m² s) of powder reaching the flame (derived). A 2.3 kg / 14 s unit delivers 0.16 kg/s, enough by REMP for about 1 to 2 m² of heptane if all of it reached the flame. Ewing et al. (1989, 1992) measured dry chemical extinction concentrations and weights for small pan fires (DOIs checked, not read).

### 3.4 The heat release after application

No time history of a fire's heat release under a portable extinguisher was found. For sprinklers, Evans (NISTIR 5254, 1993, read) fits Q(t) = Q(t_act) exp(-(t - t_act)/tau), tau = 2.0e-5 (w''/H_c)^-1.85 within +/- 50 %, on Walton's cribs of 38 mm sticks; the app uses its H_c = 610 mm form. The rating tests define success instead: no flaming at the end of discharge (FRN 731) and no re-ignition within 15 min (UL 711 7.2.8), 10 min (IS 15683 8.2.2) or 3 min (EN 3-7, secondary). Section 6.3 uses that definition to set the decay constant.

## 4. Human factors and statistics

### 4.1 How often occupants try, and how often it works

| Population | Attempt | Result | Source |
|---|---|---|---|
| US residential fires, 2004 to 2005, about 7.43 million a year (7.18 million not attended by a fire department) | Someone in the household put the fire out in 77.7 %; an extinguisher was used in 5.0 % of all fires (4.5 % unattended, 17.7 % attended; 4.7 % in the 1984 survey); water 18.7 %, power off 18.0 %, smothering 15.8 % | Extinguisher put the fire out in 2.5 % of all fires, reduced it in 1.1 %, little or no effect in 1.0 %; in attended fires mostly little or no effect | Greene and Andres 2009, CPSC, Tables 8-4 and 8-5 (read; the attended column of 8-5 was scrambled in extraction) |
| Same survey: distance to an extinguisher | Extinguisher in the same room in 32.1 % of fires, used in about 10 % of those; in another room in 28.4 %, used in about 6 %; none in 35.5 % | | same |
| Tokyo residential fires, 2022 (N = 1,606) | Attempt by any method in 63.6 %; none in 24.7 %; fire already out in 11.7 % | 81.0 % of attempts succeeded; 48 to 54 % of all residential fires were put out by occupants each year 2007 to 2022 | Tokyo Fire Department 2023 report (read) |
| Tokyo, extinguisher only | | 74.9 % (2014), 78.6 % (2016), 80.4 % (2018), 73.7 % (2024 report) | TFD leaflets and annual report (secondary) |
| Sweden, residential fires attended by the fire service, 2022 to 2025 (about 5,500 to 5,900 a year) | 10,295 attempts before arrival (several methods per fire possible); hand extinguisher 3,313 (32 %) | Hand extinguisher: put out 53.9 %, limited 26.9 %, little effect 15.6 %, worse 0.2 %; all methods: put out 62.2 %; sink water 90.2 %, pot lid 82.0 %, fire blanket or wet towel 66.9 % | MSB/MCF statistics, tables B30 and B35 (read through the PxWeb API) |
| Norway, dwelling fires (2007) | Put out early with hand extinguishers 15 %, hose reels 5 % | | DSB via trade news (secondary, unverified) |
| Trade figures | Fires put out solely by extinguishers 94 % (1979), 91 % (1985), 95 % (1996) | | NAFED distributor reports, quoted by WPI 2012 (self-selected; not usable as rates) |

The UK publishes no current table (FIRE0101 to FIRE1404) on extinguishing method or on fires put out before the fire service arrived. One trade fact sheet misreads the CPSC figure as fires put out by extinguishers (5 %); CPSC's 5 % is use, about half of it successful.

Every data set except CPSC counts only fires reported to the fire service, which lowers the success rate: fires put out by occupants often go unreported.

### 4.2 Injuries and the fire at which attempts fail

- US residential fires 2017 to 2019 (NFIRS, about 11,650 civilian injuries a year, activity known for 52 %): 32.5 % were injured fighting the fire, 28.5 % escaping; fighting the fire was the leading activity for ages 20 to 59 (USFA Topical Fire Report 21(4), 2021, read). NFPA reports 54 to 55 % of non-fatal home cooking-fire injuries while fighting the fire (secondary).
- Tokyo 2022: 129 of 506 people injured in residential fires (25.5 %) were hurt during first-aid firefighting, the largest group; causes recorded include taking too long (50), getting too close (39) and touching burning material (23).
- Merseyside 2006 to 2016: 19.5 % of dwelling-fire injuries were sustained fighting the fire, men about twice as often as women (Taylor et al. 2022); 82 % of such injuries 2011 to 2022 happened in homes with a smoke alarm (Taylor et al. 2024; abstracts read).
- Fire size at failure: no quantitative data found. Wood (1972) found people less likely to fight a fire that seemed more serious, Bryan (1977) more likely when spread was limited (both through FCRC-TR 97-11). Tokyo's recorded reasons for failure are late discovery, panic, a smoke-filled room and choosing to leave. CPSC's extinguishers mostly failed in the fires a fire department later attended. NFPA 10 C.2.2 relates class A success to the quantity of combustibles involved in an incipient fire. Fire service guidance ("only a small fire") is guidance, not data.

### 4.3 Time to fetch and use

- WPI and Eastern Kentucky University (Poole et al. 2012; a student project hosted by the manufacturers' association; read): 276 untrained adults, a propane trainer fire, a 6 L water training unit with a pin placed beside a person standing 3 m from the fire, no carrying. From "go" to discharge (breaking the seal and pulling the pin included): mean 13.4 s on the first try (group means 11.6 to 15.2 s), 7.2 s after brief instruction. 98 % discharged on the first try; many struggled with the pin; 65 % put out the simulated fire on the first try and 90 % on the second, within about 15 s of agent.
- Thielsch et al. (2020, read): 64 laypeople met a burning 40 x 40 cm or 50 x 50 cm polypropylene-filled pillow in a mock break room, with a 625 mL foam spray already in the room. 95 % fought the fire; mean 17.9 s (median 17.5 s, SD 10.2 s, range 2 to 48 s) from entering the room to starting, 12.1 s on the second trial; mean 13.6 s to put it out. 88 % aimed at the flames rather than the burning material, 92 % stood upright, 87 % closed the door behind them.
- Bruck and Thomas (2010, as quoted in WPI 2012): 18 of 23 adults over 60 put out a small fire with an extinguisher, mean 38 s (SD 16.3 s).
- No data were found on walking speed while carrying an extinguisher, or on the time to find one and lift it from a bracket.

### 4.4 Who fights the fire

- Wood (1972, Fire Research Note 953, N = 2,193 first actions, through FCRC-TR 97-11): some firefighting action in 14.9 % of first actions and 16.6 % of first and second actions; by building, dwellings 10 %, flats 12 %, multiple occupancy 6 %, other occupancies 21 % (almost a quarter in factories); 24 % when the first cue was flames, 9 % when it was an alarm; women less likely than men. Thielsch et al. quote Wood differently (13 % try to extinguish); the original was not read.
- Bryan (1977, Project People, NBS-GCR-77-94, about 580 people mostly in dwellings, through FCRC-TR 97-11): firefighting 10.4 % of first actions; men went for an extinguisher more often.
- Kuligowski (2009) gives a conceptual model of the behaviour process in fires, without firefighting figures. Proulx (1997, 2006), Brennan (1999) and Kobes et al. (2010) were identified (DOIs checked) but not read for firefighting data.

## 5. Extinguishers in IFC

### 5.1 The schema

IfcFireSuppressionTerminalTypeEnum, read in the IFC 4.4 draft documentation, lists BREECHINGINLET, FIREHYDRANT, HOSEREEL, SPRINKLER, SPRINKLERDEFLECTOR, FIREMONITOR, USERDEFINED and NOTDEFINED; its change log dates FIREMONITOR to IFC4x3 RC3, so IFC2x3 and IFC4 have the other items only (their pages refused access and were not checked). No item covers a portable extinguisher, and the type is defined as a flow terminal delivering a fluid to suppress a fire, every listed item being a pipe-connected device. A buildingSMART forum thread (2020 to 2026) has no settled answer: one proposal (April 2026) is IfcFireSuppressionTerminal with USERDEFINED and ObjectType FIRE_EXTINGUISHER, optionally grouped in a user-defined IfcDistributionSystem; others suggest IfcFurniture, IfcFurnishingElement or IfcAsset. Exporters do what their category mapping gives, as the corpus shows.

### 5.2 The local corpus

Scanned: all 198 `.ifc` files of the local test corpus (12 at the top, 75 in `mep`, 98 in `more`, 11 in `more2`, 2 in `more3`), as text, with a read-only Python script run under `-I` from the scratchpad; nothing was copied. Each STEP entity's strings (IFC GlobalIds excluded) were matched case-insensitively against extinguish, fire_ext / fire ext, FEC as a whole token, and the German, French, Norwegian, Swedish, Dutch and Finnish words; type objects were resolved through IfcRelDefinesByType and property values through IfcPropertySet and IfcRelDefinesByProperties.

| File (under ifc-samples) | Schema | Count | Entity | Type object | Names and properties |
|---|---|---|---|---|---|
| `ISSUE_159_kleine_Wohnung_R22.ifc` | IFC4 | 6 | IfcBuildingElementProxy | IfcBuildingElementProxyType, NOTDEFINED | Revit family "Extinguisher - C02: Red" (CO2 written with a zero, a numeric prefix on the name); German Revit set "Sonstige" (Familie, Typ, Typ-ID); materials "Paint - Red Fire Extinguisher", "Paint - Black Fire Extinguisher"; IfcShapeAspect "Fire Extinguisher" |
| `mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Arch.ifc`, `...Arch-Optimized.ifc`, `mep\bsi-community-clinic\Clinic_Architectural.ifc`, `dental_clinic.ifc` (the last two byte-identical) | IFC2X3 | 10 in each (one model) | IfcFlowTerminal | IfcSanitaryTerminalType "305 x 215 x 813mm", NOTDEFINED | Name "M_Fire Extinguisher Cabinet:305 x 215 x 813mm:<id>"; the sanitary type suggests the family sat in a plumbing category (not checked) |
| `mep\auckland-openifc\20160124OTC-Conference Center.ifc` | IFC2X3 | 2 | IfcBuildingElementProxy | IfcBuildingElementProxyType, NOTDEFINED | "Fire Extinguisher Cabinet:9 1/2 x 4 1/2 x 24""; Pset_BuildingElementProxyCommon.Reference |
| `mep\duraark-sgd-bodo\SGD_BODO\SGD_BODO_Eng-HVAC-Plumbing.ifc` | IFC2X2_FINAL | 13 | IfcFlowTerminal | IfcFireSuppressionTerminalType, HOSEREEL | "FH": hose reels, not extinguishers; NFPA 10 6.2.1.4 lets 38 mm hose stations replace up to half the class A units |

Not found anywhere: an extinguisher as IfcFireSuppressionTerminal or its type (USERDEFINED or otherwise), as IfcFurnishingElement, with an underscore name (Fire_Extinguisher), or as the token FEC. Every FEC match was a false positive: fragments of hyphenated hexadecimal IDs in ArchiCAD properties, and a glazing property named "fec" in three Norwegian ArchiCAD models (`SGD_BODO_Arch-3.ifc`, `SGD_HiTOS_Arch.ifc`, `more2\osb-testfiles-hitos\HITOS_070308.ifc`). A reader that matches FEC must require it as a whole token near "cabinet" or "extinguisher". The other fire-suppression terminals in the corpus are the sprinklers already reported in the MEP note.

### 5.3 Rules for the reader

1. Candidates: any IfcProduct (IfcBuildingElementProxy, IfcFlowTerminal, IfcFireSuppressionTerminal, IfcFurnishingElement, IfcFurniture, IfcDistributionControlElement) whose Name, ObjectType, type name or a property value names an extinguisher or extinguisher cabinet in the languages above, or a USERDEFINED IfcFireSuppressionTerminal(Type) whose ElementType or ObjectType does. Hose reels (HOSEREEL or the name) are read as hose stations, never as extinguishers; sprinklers and hydrants are excluded as in `ifcSemantics.js`.
2. Agent and rating from the name where present (CO2 or C02, ABC, powder, Pulver, water, foam, wet chemical, a rating pattern such as 2-A:10-B:C or 34A 233B C); otherwise "unknown", shown as such, with the default of section 6.3 used and stated.
3. Position: the product's placement; snapped to the nearest walkable node on the same storey within 1.5 m (a cabinet sits in a wall), else reported as unreachable.
4. Mounting height (SBC 906.9.1 to 906.9.3) from the geometry's top and bottom above the storey's floor; "cannot tell" without geometry or gross weight.
5. The NIBS clinic is the best first test: its 10 cabinets, its 418 sprinklers in the MEP model of the same frame, two storeys and IfcSpace rooms.

## 6. Recommended model

All of it behind an "Extinguishers" switch next to the Sprinklers switch, off by default; off, nothing is read, laid out or computed, and a run keeps its kept-run key (the user's performance rule).

### 6.1 Placement check on the walking graph

For each storey, with E the extinguisher nodes (from IFC or the layout) and the walking graph's edge lengths in metres:

1. d(v) = shortest walking distance from node v to the nearest node of E, by one multi-source Dijkstra (all of E at distance 0). This is NFPA 10 E.3.3's actual walking distance, measured along the walk, not a straight line.
2. Travel rule per hazard area: class A, every node within D_A = 22.86 m (the stricter of SBC's 23 m and IBC's 75 ft); class B areas within 9 m or 15 m of a unit of the required rating (SBC Table 906.3(2), stricter than NFPA's 9.14 and 15.25 m); class K within 9 m of the cooking item; class D within 22.86 m. Dwelling units: the NFPA 10 F.6.1 advice (2-A:10-B:C per floor level, 12 m) reported as advice, not a code rule.
3. Node sampling: the walking graph samples the floor at spacing s (0.5 m), so a point of the floor can be up to delta = s / sqrt 2 (0.35 m) from its node along a straight walk. A node pass at D_A - delta guarantees the rule at every point; the check reports nodes within delta of the limit as "near the limit".
4. Count rule: n_E >= ceil(A_storey / min(n_A a_h, 1045 m²)), with a_h = 279, 139 or 92.9 m² per A unit by hazard (NFPA 10 6.2.1.2.1, Table E.3.5), and every unit at least the minimum single rating (2-A, 2-A, 4-A); EN ratings converted at 2/13 UL A unit per EN A unit (section 2.4), stated.
5. Where-required (SBC 906.1) for the occupancy chosen for the performance check, the mounting heights (906.9), the agent suitability per room class from `fireClasses.js`, and placement on the routes people normally walk (906.5) checked as a node on or within 1 m of an egress route; obstruction and visibility (906.6) "cannot tell".
6. Report as the code check does: pass, fail or cannot tell for each rule, with the clause (SBC 801-CR-2018 and NFPA 10), the worst node and its distance.

Cost: one Dijkstra per storey, O(E log V); Vienna's graph runs it in the time of one route query.

### 6.2 Greedy layout

Where the IFC has no extinguishers, or the check fails:

- Candidates C: walkable nodes within 0.6 m of a wall and either on an egress route or within 2 m of a door (NFPA 10 F.6.2.2 places them near exits; SBC 906.5 on the routes people normally walk), thinned to one per 1 m.
- Each candidate c covers the nodes within D_A - delta of it (a Dijkstra ball cut at that radius).
- Greedy set cover: repeatedly take the candidate covering the most uncovered nodes (ties: nearest an exit), until every node is covered, then add units until the count rule holds. Greedy is within a factor H(m) <= 1 + ln m of the fewest units for a cover of m nodes (Johnson 1974; Chvátal 1979). The result is checked again by 6.1, so the layout never has to be trusted, only the check.
- Separate passes for class B, K and D hazard areas with their own radii and ratings; class C from the A or B hazard.

### 6.3 Manual suppression in a run

One responder, one extinguisher, one attempt. The steps, each with its parameter:

1. Who goes. The occupant with the shortest walking time to the fire among those in the room of origin (the cue of flames: Wood's 24 % fought when the first cue was flames, 9 % on an alarm); none if the room is empty, or a named node (a fire warden) chosen by the user. The extinguisher: the nearest by walking time whose agent `fireClasses.js` lists as suitable for the first item's class; none for class L (the list says leave) or D without a class D unit.
2. When they start. t_go = t_cue + t_r, with t_cue the run's cue time for that occupant and t_r either the run's sampled pre-movement (design log-normal or Lovreglio Weibull, the conservative case for the fire) or 0 (the immediate-response bound). Measured context: Thielsch et al., mean 17.9 s from meeting the fire to fighting it, range 2 to 48 s.
3. The fetch. t_s = t_go + T(start, ext) + t_take + T(ext, attack), with T the graph's walking times at the run's speeds (carrying speed: no data; taken as unchanged, stated), t_take = 13.4 s untrained or 7.2 s instructed (WPI 2012), and the attack node the walkable node nearest the extinguisher in walking time that is at least 1.8 m from every burning cell (UL 711 7.2.6) with a clear line (the `clear` test of `sprinklerSetup`) to them. Every node on the way must be tenable when passed (the run's tenability index); otherwise the attempt is abandoned at the first untenable node and the responder evacuates from there.
4. The fire when discharge starts. From the solver at t_s: the burning cells B (reaction rate above a threshold), their heat release Q(t_s), and, for liquid fuels, their burning area A_B. Reach: the cells within the unit's range r (the lower end of its Table H.2 range: water 9.1 m, foam 6.1 m, ABC 1.5 m for units up to 2.3 kg and 4.6 m above, CO2 0.9 m) of the attack node.
5. Capacity. Class A: Q_cap = eta n_A q_A, q_A = 0.3 MW (0.2 to 0.45) per UL A unit, eta = 0.4 (0.2 to 1.0), the operator factor, by analogy with UL 711 5.4's 40 % for class B (an assumption: no class A derating for ordinary operators is published). Class B: A_cap = 0.0929 n_B m² for UL ratings, 0.4 pi B / 100 m² for EN ratings. Class C: the A or B rule with a non-conductive agent only. Class K: wet chemical only, and only while the burning cells are within the cooking item's own cells (the K test is pass or fail on one appliance; the app's pan fire peaks at 20 to 110 kW). Water units are also bounded by Noaki's 12 kJ/g times the water flow (not binding at the rated sizes).
6. Outcome, three-valued so the parameter uncertainty shows: success if every burning cell is within reach, the agent is suitable, and Q(t_s) <= Q_cap at the low ends (q_A 0.2 MW, eta 0.2; 0.04 n_A MW); failure if a burning cell is out of reach, the agent is unsuitable, or Q(t_s) > Q_cap at the high ends (0.45 n_A MW); cannot tell between. For a 2-A unit: success up to 80 kW, failure above 0.9 MW. For a 10-B unit: success up to 0.93 m² of burning liquid at UL's own figure.
7. On success (and, at the user's choice, on cannot tell): every reached burning cell takes the cap Q_c(t) <= r_c(t_s) exp(-(t - t_s)/tau_x), with r_c(t_s) its rate in the step after t_s and tau_x = t_d / ln(1/epsilon), epsilon = 0.01 (0.001 to 0.1): the rating tests' own criterion that flames are out when the unit is empty, followed by no re-ignition. This is `evansFactor` with a per-cell tau and t_act, so the solver needs no new kernel. Cells not burning at t_s are left alone (powder and CO2 leave no protection; water's wetting is not calibrated), unlike the sprinkler cap, which stops covered cells from igniting. Values: 9.5 L water, t_d 60 s, tau_x 13.0 s; 6 L water, 40 s, 8.7 s; 2.3 kg ABC, 14 s, 3.0 s; 4.5 kg ABC, 20 s, 4.3 s; 6 kg ABC (EN), 16 to 23 s, 3.5 to 5.0 s; 2 kg CO2, 8.5 s, 1.8 s. The heat released by a reached cell after t_s is then at most r_c(t_s) tau_x (continuous) or r_c (tau_x + dt) per step (Suppression.lean `evansCap_sum`).
8. On failure: the fire is unchanged (the 27 % of Swedish attempts that limited the fire are not credited, the conservative choice for egress); the responder discharges for t_d at the attack node, accumulating dose there, then evacuates from it. The time lost is (t_s + t_d + T(attack, exit)) - (t_go + T(start, exit)), which is never negative (6.5, item 8). An unsuitable agent (water on class B or K, water or foam on live class C) is reported as the wrong outcome with its reason from `fireClasses.js`; a flare-up is not modelled (no source gives its size).
9. Not modelled: the agent's gas-phase effect (9.5 L of water could absorb at most 24.6 MJ if all of it evaporated, about 0.41 MW over 60 s, most of which lands on the fuel instead), powder's loss of visibility (NFPA 10 H.2 item 5: up to several minutes in a small room), CO2's oxygen dilution, re-ignition after the cap, several responders, and a second extinguisher.

A worked example, with a t² fire standing in for the solver: a class A fire growing at the medium rate (alpha 0.0117 kW/s², the default of `fireClasses.js`), a responder 5 m from it, the extinguisher 15 m away along the walk and 15 m back, walking at 1.2 m/s: t_s = t_go + 12.5 + 13.4 + 12.5 s. With t_go = 30 s (an immediate response to flames), t_s = 68 s and Q = 54 kW: success with a 2-A unit even at the low ends. With t_go = 330 s (the ABCB design median of 300 s after a 30 s cue), t_s = 368 s and Q = 1.6 MW: failure even at the high ends. The outcome turns on the responder's delay far more than on the extinguisher, which is what the statistics of section 4.1 suggest: attempts succeed when made early.

Parameters, sources and uncertainty:

| Parameter | Value | Range | Source | Kind |
|---|---|---|---|---|
| D_A (class A travel) | 22.86 m | SBC 23 m | SBC 801-CR-2018 Table 906.3(1); NFPA 10 Table 6.2.1.1 | code |
| Class B travel | 9 m / 15 m | NFPA 9.14 / 15.25 m | SBC Table 906.3(2); NFPA 10 Table 6.3.1.1 | code |
| Class K, D travel | 9 m; 22.86 m | | SBC 904.12.5; NFPA 10 6.6.2, 6.5.2 | code |
| Area per A unit | 279 / 139 / 92.9 m² | SBC 280 / 139 / 93 | as above | code |
| Area per extinguisher | 1045 m² | | as above; E.3.4 | code |
| Node sampling margin delta | 0.35 m | s / sqrt 2 | geometry of a 0.5 m grid | derived |
| t_take | 13.4 s | 7.2 s instructed; over 15 s for some | WPI 2012 | measured, one study, trainer |
| t_r | the run's pre-movement, or 0 | 2 to 48 s measured for a fire met in a room | rsetPresets.js; Thielsch et al. 2020 | design / measured |
| Carrying speed | the run's walking speed | none found | none | assumption |
| Standoff | 1.8 m | | UL 711 7.2.6 | test procedure |
| Reach r | lower end of Table H.2 range | upper end | NFPA 10 Table H.2 | annex |
| t_d | by unit, 8.5 to 60 s | +/- 20 % between makers | Table H.2; Amerex; GLORIA | manufacturer |
| q_A | 0.3 MW per UL A unit | 0.2 to 0.45 MW | UL 711 Table 2 wood; NIST FCD Test13; crib correlations | derived |
| eta (class A operator) | 0.4 | 0.2 to 1.0 | analogy with UL 711 5.4 | assumption |
| Class B area per unit | 0.0929 m² (UL); 0.4 pi / 100 m² (EN) | factor 2 to 3 between systems | UL 711 5.4, Table 5; IS 15683 Table 9 | test definition |
| EN to UL class A | 2/13 UL unit per EN unit | factor 2 for powder | UL 711 Table 1; BS 5306-8:2023 basis (secondary) | derived |
| epsilon | 0.01 | 0.001 to 0.1 | rating tests' pass criteria | constructed |
| tau_x | t_d / ln(1/epsilon) | t_d/6.9 to t_d/2.3 | from epsilon | constructed |
| Water cap check | 12 kJ per g of water | not checked for jets | Noaki et al. 2018 | measured, overhead water |

### 6.4 The simplest honest version, and the order of work

1. The placement check of 6.1 and the IFC reader of 5.3, with no simulation. It is exact against the code's own measure and needs no physical parameter.
2. The suppression card: steps 1 to 6 of 6.3, reporting t_s, Q(t_s), the cells in reach, the three-valued outcome with the parameter ranges, and the time lost, with the fire unchanged. This is the simplest version that is still honest: it states what the codes and test fires support and shows the uncertainty instead of resolving it.
3. The greedy layout of 6.2.
4. The cap of 6.3 step 7 on success, behind a sub-switch, and the responder's changed evacuation in the egress run.

Measured, as for the sprinklers: the switch off must match today's timings and kept-run keys; on, the card is one Dijkstra and a read of the frame at t_s, and the cap one multiply per reached cell.

### 6.5 Statements to prove in Lean 4

Several reuse what is proved (Suppression.lean `evansCap_antitone`, `evansCap_sum`, `evans_cells`; Sprinklers.lean `decay_le`, `decay_antitone`; PreMovement.lean). New, all elementary:

1. `travel_mono_sites`: adding an extinguisher never increases any node's distance (a minimum over a superset), so a passing layout stays passing when units are added.
2. `graph_ge_euclid`: if each edge is at least as long as the straight line between its ends, every graph distance is at least the straight-line distance; so a pass on the graph implies a pass of the radius check, not the converse.
3. `sample_margin`: if every floor point lies within delta of a node by a straight walk, then a node pass at D - delta gives a pass at D for every point.
4. `count_rule_antitone`: the required number of units never rises with a higher rating or a lighter hazard, and is at least 1 on any non-empty floor.
5. `greedy_covers`, `greedy_card_le`: if the candidates' balls cover the storey, greedy ends with a cover of at most as many units as nodes; optionally Chvátal's H(m) bound.
6. `arrival_mono`: t_s is non-decreasing in t_r, t_take and each walking time.
7. `success_antitone`: if Q is non-decreasing on [t, t'] (the growth phase; the hypothesis is needed), success at t' implies success at t, so a later arrival never turns failure into success; `success_mono_capacity`: success is monotone in n_A, q_A and eta; `success_mono_reach`: in r. For a t² fire, `success_iff_tsq`: success iff t_s <= sqrt(Q_cap / alpha).
8. `time_lost_nonneg`: with T a shortest-path metric (triangle inequality), t_take >= 0 and t_d >= 0, the responder's evacuation start with an attempt is never earlier than without one.
9. `decay_at_discharge_end`: with tau_x = t_d / ln(1/epsilon) and 0 < epsilon < 1, the cap at t_s + t_d is exactly epsilon r_c(t_s); `tau_mono`: tau_x rises with t_d and with epsilon.
10. `reached_energy_le`: the heat released by the reached cells after t_s is at most the sum of r_c(t_s) (tau_x + dt), from `evansCap_sum` and `evans_cells`.
11. A counterexample to record, as for the sprinklers: the cap scales every reached cell's reactions by a factor of at most 1 in every step, but the fire's total heat release in a later frame can still exceed the free fire's, because pyrolysis gas made before t_s can burn elsewhere; the provable statements are about the reached cells' fuel and energy, not about global peaks.

## 7. The claims I am least sure of

1. q_A = 0.3 MW per UL A unit (0.2 to 0.45). No rated test fire's heat release was measured in any source found; it rests on the UL crib's wood volume, an assumed wood density and one NIST crib of unknown stick size.
2. The operator factor eta = 0.4 for class A, carried over from UL's class B definition. No study gives the share of a rated class A fire an untrained person puts out.
3. The decay constant tau_x = t_d / ln 100. The exponential form is Evans's for sprinklers; the constant is set by the rating tests' pass criterion, not measured for any extinguisher.
4. EN to UL conversions (13A as about 2-A; class B areas differing by a factor of 2 to 3 for powder).
5. The EN 3-7, ISO 7165, BS 5306-8 and IFC 2021 values, all secondary; the EN minimum ratings per charge and discharge times were not obtained.
6. t_take = 13.4 s comes from one non-peer-reviewed study with the extinguisher beside the person and a propane trainer; the fetch from a bracket and the carrying speed are unmeasured.
7. The Tokyo extinguisher-only success rates, the Norwegian figures and the NFPA cooking-injury shares are secondary; Wood's and Bryan's figures come from a 1997 review, and Thielsch et al. quote Wood differently.
8. The NFPA 10 Table H.2 dry chemical rows: the extracted table's columns are interleaved, so ratings and durations for ABC units are given as ranges and from manufacturer sheets.
9. NFPA 10 clause numbers are from the Second Draft Report for the next edition; the 2022 edition may number some annex items differently.

## 8. Sources

Standards and codes (values and clause numbers only)

- SBC 801-CR-2018, Saudi Fire Code, Code Requirements, Section 906, Tables 906.1, 906.3(1), 906.3(2), and 904.12.5. Licensed copy for internal use, read locally; not in the repository.
- NFPA 10, Standard for Portable Fire Extinguishers, Second Draft Report, A2025 revision cycle (2022 edition with revisions). https://docinfofiles.nfpa.org/files/AboutTheCodes/10/10_SDR_A2025.pdf (read: 5.4, 6.2 to 6.7, A.4.4.1, A.5.4.1.1 to A.5.4.1.3, A.5.5, C.2.2, E.3, F.6, G.1, G.2, H.1, H.2, Table H.2).
- ANSI/UL 711, CAN/ULC-S508, Rating and Fire Testing of Fire Extinguishers, 7th edition, 2004, revised 18 December 2013. https://law.resource.org/pub/us/cfr/ibr/inc/ul/ul.s711.7.2013.pdf (read: 5.3, 5.4, 7.1 to 7.3, 8.1 to 8.2, Tables 1 to 5).
- IS 15683:2006, Portable fire extinguishers: performance and construction, Bureau of Indian Standards. https://law.resource.org/pub/in/bis/S03/is.15683.2006.pdf (read: 8.2 to 8.4, Tables 7 to 9).
- O'Dogherty, M. J., Young, R. A., Lange, A. (1968). The performance of water-type extinguishers on experimental Class A fires. Fire Research Note 731. https://publications.iafss.org/publications/frn/731/-1/view/frn_731.pdf (read in OCR).
- IFC 2021 906.1 as adopted: Massachusetts 780 CMR 906.1, https://regulations.justia.com/states/massachusetts/780-cmr/chapter-9/section-906/subsection-906-1 ; Minnesota 7511.0906, https://www.revisor.mn.gov/rules/7511.0906/version/2020-04-16T11:42:58-05:00 ; New York DOS, https://dos.ny.gov/required-fire-extinguisher-locations-existing-buildings (search summaries).
- Fire Equipment Manufacturers' Association, newsletter issue 4 (2020), on the 2012 IFC removal of the quick-response sprinkler exception. https://femalifesafety.org/wp-content/uploads/2020/07/NewsletterIssueNo4Oct.pdf (search summary).
- NSI Technical Bulletin 0065, Publication of BS 5306-8:2023. https://documents.nsi.org.uk/wp-content/uploads/2023/12/NSI-Technical-Bulletin-No-0065-Publication-of-BS-5306-8_2023.pdf (search summary; the PDF refused access). Safelincs, extinguisher advice and buyers guide. https://safelincs.co.uk/fire-extinguisher-buyers-guide (search summary).
- MeyerFire, "Area per unit of A vs floor area per extinguisher". https://www.meyerfire.com/daily/area-per-unit-of-a-vs-floor-area-per-extinguisher
- buildingSMART, IFC 4.4 draft documentation, IfcFireSuppressionTerminalTypeEnum (via the tunnel documentation mirror). https://bsi-infraroom.github.io/IFC-Documentation-Tunnel/4_4_0_0/general/HTML/schema/ifcplumbingfireprotectiondomain/lexical/ifcfiresuppressionterminaltypeenum.htm
- buildingSMART forum, "How to categorise electric strikes, IDF panels, egress paths and more?" (2020 to 2026). https://forums.buildingsmart.org/t/how-to-categorise-electric-strikes-idf-panels-egress-paths-and-more/2018

Manufacturers

- Amerex Model 240 water. https://amerex-fire.com/products/fire-extinguishers/portable-fire-extinguishers/240 ; Amerex ABC dry chemical specifications (retailer copy). https://thefirestore.com/Amerex-ABC-Dry-Chemical-Fire-Extinguishers
- GLORIA, Übersicht tragbare Feuerlöscher 2026. https://www.gloria.de/wp-content/uploads/2026/03/uebersicht-tragbare-feuerloescher-2026.pdf

Data and reports without DOIs

- Greene, M. A., Andres, C. (2009). 2004-2005 National Sample Survey of Unreported Residential Fires. US Consumer Product Safety Commission. https://www.cpsc.gov/s3fs-public/UnreportedResidentialFires.pdf
- Tokyo Fire Department (2023). 住宅火災・放火火災の実態（令和4年中）. https://www.tfd.metro.tokyo.lg.jp/content/000044673.pdf
- MSB / MCF statistics database, tables B30 and B35 (residential fires; first-aid attempts before fire service arrival), 2022 to 2025. https://statistik.mcf.se/PxWeb/api/v1/sv/PxData/START/B/B3/B35
- US Fire Administration (2021). Civilian fire injuries in residential buildings (2017-2019). Topical Fire Report Series 21(4). https://www.usfa.fema.gov/downloads/pdf/statistics/v21i4.pdf
- Poole, B., Hicks, W., Notarianni, K., Harris, C., Hanks, C., Gorbett, G. (2012). Ordinary people and effective operation of fire extinguishers. Worcester Polytechnic Institute and Eastern Kentucky University. https://www.femalifesafety.org/wp-content/uploads/2021/02/Effective-Use-of-Portable-Fire-Extinguishers-by-Ordinary-People-%E2%80%93-WPI-EKU-Study.pdf
- Fire Code Reform Centre (1997). Selected literature reviews on human behaviour in fire. FCRC-TR 97-11 (reports Wood 1972, FRN 953, and Bryan 1977, NBS-GCR-77-94). https://www.abcb.gov.au/sites/default/files/resources/2020//FCRC-TR-97-11-Selected-literature-reviews-human-behaviour-fire.pdf
- Särdqvist, S. (2000). Demand for extinguishing media in manual fire fighting. Lund University, report TVBB-1021. https://lup.lub.lu.se/search/files/4546393/1049443.pdf

Journal articles and reports with DOIs (resolved at api.crossref.org, 2026-11-07, unless stated)

- Bruck, D., Thomas, I. (2010). Interactions between human behaviour and technology: implications for fire safety science. Fire Technology 46(4), 769-787. doi:10.1007/s10694-010-0161-1 (quoted through WPI 2012).
- Bundy, M. et al. (2020). NIST Fire Calorimetry Database (FCD), Test13_woodCrib_r1. doi:10.18434/mds2-2314 (DataCite). https://www.nist.gov/node/1674296
- Chvátal, V. (1979). A greedy heuristic for the set-covering problem. Mathematics of Operations Research 4(3), 233-235. doi:10.1287/moor.4.3.233
- Evans, D. D. (1993). Sprinkler fire suppression algorithm for HAZARD. NISTIR 5254. doi:10.6028/NIST.IR.5254
- Ewing, C. T., Faith, F. R., Romans, J. B., Hughes, J. T., Carhart, H. W. (1992). Flame extinguishment properties of dry chemicals: extinction weights for small diffusion pan fires and additional evidence for flame extinguishment by thermal mechanisms. Journal of Fire Protection Engineering 4(2), 35-51. doi:10.1177/104239159200400201 (not read).
- Johnson, D. S. (1974). Approximation algorithms for combinatorial problems. Journal of Computer and System Sciences 9(3), 256-278. doi:10.1016/S0022-0000(74)80044-9
- Kobes, M., Helsloot, I., de Vries, B., Post, J. G. (2010). Building safety and human behaviour in fire: a literature review. Fire Safety Journal 45(1), 1-11. doi:10.1016/j.firesaf.2009.08.005 (not read for this note).
- Kokkala, M. A. (1986). Extinguishment of compartment fires using portable chemical extinguishers and water. Fire Safety Journal 11(3), 201-209. doi:10.1016/0379-7112(86)90063-9 (not read).
- Kuligowski, E. D. (2009). The process of human behavior in fires. NIST Technical Note 1632. doi:10.6028/NIST.TN.1632
- Kung, H.-C., Hill, J. P. (1975). Extinction of wood crib and pallet fires. Combustion and Flame 24, 305-317. doi:10.1016/0010-2180(75)90164-9 (not read).
- Liu, Z., Kim, A. K., Carpenter, D. (2007). A study of portable water mist fire extinguishers used for extinguishment of multiple fire types. Fire Safety Journal 42(1), 25-42. doi:10.1016/j.firesaf.2006.06.008 (not read).
- Noaki, M., Delichatsios, M. A., Yamaguchi, J., Ohmiya, Y. (2018). Heat release rate of wooden cribs with water application for fire suppression. Fire Safety Journal 95, 170-179. doi:10.1016/j.firesaf.2017.10.002 (abstract only).
- Proulx, G. (1997). The time delay to start evacuation: review of five case studies. Fire Safety Science 5, 783-794. doi:10.3801/IAFSS.FSS.5-783 (not read for this note).
- Rappsilber, T., Below, P., Krüger, S. (2019). Wood crib fire tests to evaluate the influence of extinguishing media and jet type on extinguishing performance at close range. Fire Safety Journal 106, 136-145. doi:10.1016/j.firesaf.2019.04.014 (not read).
- Rasbash, D. J. (1986). The extinction of fire with plain water: a review. Fire Safety Science 1, 1145-1163. doi:10.3801/IAFSS.FSS.1-1145
- Särdqvist, S., Holmstedt, G. (2000). Correlation between firefighting operation and fire area: analysis of statistics. Fire Technology 36(2), 109-130. doi:10.1023/A:1015450308130 (not read).
- Särdqvist, S., Holmstedt, G. (2001). Water for manual fire suppression. Journal of Fire Protection Engineering 11(4), 209-231. doi:10.1177/104239101400934333 (not read).
- Tamanini, F. (1976). The application of water sprays to the extinguishment of crib fires. Combustion Science and Technology 14(1-3), 17-23. doi:10.1080/00102207608946742 (not read).
- Taylor, M., Appleton, D., Fielding, J., Oakford, G. (2022). An exploration of householder injuries sustained fighting dwelling fires. Fire Safety Journal 127, 103519. doi:10.1016/j.firesaf.2021.103519 (abstract).
- Taylor, M., Fielding, J., Reilly, D., Kwasnica, V. (2024). A Bayesian analysis of domestic fire response and fire injury. Fire Safety Journal 150, 104266. doi:10.1016/j.firesaf.2024.104266 (abstract).
- Thielsch, M. T., Kirsch, J., Thölking, H., Tangelder, L., Lamers, C. (2020). Fight or flight? Behaviour and experiences of laypersons in the face of an incipient fire. Ergonomics 64(2), 149-170. doi:10.1080/00140139.2020.1825824
- Walton, W. D. (1988). Suppression of wood crib fires with sprinkler sprays. NBSIR 88-3696. doi:10.6028/NBS.IR.88-3696 (as used by Evans).
