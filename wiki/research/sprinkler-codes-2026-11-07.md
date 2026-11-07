# Automatic sprinklers in the Saudi Building Code and NFPA 13 placement rules; outdoor assembly points (research, 2026-11-07)

Status: research only, no code. Companion to `wiki/research/fire-suppression-note-2026-11-05.md`, `sbc-firefighting-future-notes-2026-11-05.md` and `tall-office-sbc-egress-2026-11-05.md`.

**The main limitation.** No clause of SBC 201 or SBC 801 was read. The 2024 English "Code Requirements" are sold, and the free Arabic text opens only in a protected web viewer. The prior note (`tall-office-sbc-egress-2026-11-05.md`, section 2) records this, and I did not try to get round it. Only the official tables of contents are open PDFs. So every "where required" value below is an **IBC/IFC 2021 value, to be confirmed against SBC 201/801 (2024)**. The only exceptions are the few Saudi figures that secondary sources report, and the table marks those.

**How NFPA values were handled.** I did not open NFPA 13, 13R or 13D, and I created no account. Values come from secondary sources: manufacturer data sheets (Tyco), an AHJ plan-review checklist (City of Newport Beach), NFPA staff and trade-press articles, practitioner references (MeyerFire, QRFS), US government documents (VA, USFA, NIOSH, OSHA, UFC) and study guides. The table marks a value "yes" only where at least two independent sources agree. One source of the study-guide kind (OpenExamPrep NICET pages, "OEP" below) contradicted the better sources more than once. It is used only as a supporting second source, never alone. Section numbers are given for the NFPA 13 edition the source cites, because NFPA renumbered the standard in 2019 (chapter 8 became chapters 9 and 10).

Units: the US value is quoted first and the SI value follows in brackets. Where the source prints an SI figure it is used; otherwise the conversion is mine.

---

## 0. Summary table of values

Key to sources: see section 7. "IBC" means the 2021 International Building Code and "IFC" the 2021 International Fire Code. Their sprinkler sections (903) carry the same numbers in both codes.

| # | Value | Unit | Applies to | Clause | Sources | Confirmed |
|---|---|---|---|---|---|---|
| **Q1. Where sprinklers are required** | | | | | | |
| 1 | SBC 2018 is based on the 2015 International Codes | n/a | SBC 201/801-2018 | n/a | S1, S2 | yes |
| 2 | SBC 2024 aligns with IBC/IFC 2021 (inferred; no official statement) | n/a | SBC 201/801-2024 | n/a | S4 (SBC 801-2024 contents list Sections 916 Gas Detection and 917 Mass Notification, first added in IFC 2021), S5, S7 | partly |
| 3 | SBC 2024 mandatory from 30 June 2025 (one vendor says 1 July 2025) | date | all projects after the transition | n/a | S51, S52, S5 | yes |
| 4 | Sprinklers throughout every building that contains a Group R fire area | n/a | R-2 apartments | IBC/IFC 903.2.8 | S8, S53 | yes (IBC) |
| 5 | NFPA 13R permitted only up to 4 stories and 60 ft (18 288 mm) above grade plane, with the highest story's floor no more than 30 ft (9144 mm) above the lowest fire department (FD) vehicle access | ft (mm) | Group R | IBC 903.3.1.2 (2021) | S9, S53 | yes (IBC) |
| 6 | No sprinkler trigger for Group B based on use or area, except ambulatory care facilities (4 or more care recipients incapable of self-preservation, or 1 or more not on the level of exit discharge) | n/a | Group B offices | IBC 903.2.2 | S8 | partly |
| 7 | Sprinklers throughout when any story with an occupant load of 30 or more has its floor 55 ft (16 764 mm) or more above the lowest FD vehicle access (open parking structures and F-2 excepted) | ft (mm) | all groups, including B | IBC 903.2.11.3 | S8, S11 | yes (IBC) |
| 8 | High-rise: an occupied floor more than 75 ft (22 860 mm) above the lowest FD vehicle access; sprinklers throughout | ft (mm) | all groups | IBC 202, 403.3 | S10, S8, S45 | yes (IBC) |
| 9 | Saudi high-rise threshold "towers above 23 m", used in the SBC enforcement schedule | m | Saudi buildings | SBC implementing regulation (2018), not SBC 201 itself | S6 | partly |
| 10 | Stories and basements without qualifying openings: sprinklers throughout where the floor area exceeds 1,500 ft² (139 m²) | ft² (m²) | all groups | IBC 903.2.11.1 | S8, S56 | partly |
| 11 | Basement with any part more than 75 ft (22 860 mm) from the required openings: sprinklers throughout | ft (mm) | all groups | IBC 903.2.11.1.3 | S50, S56 | yes (IBC) |
| 12 | Education above grade 12 is Group B; lecture halls may be A-3 | n/a | university | IBC 304.1, 303.4 | S12a, S12b | yes (IBC) |
| 13 | A-3 fire area: over 12,000 ft² (1115 m²), occupant load 300 or more, or on a floor other than the level of exit discharge | ft² (m²), persons | lecture halls, assembly | IBC 903.2.1.3 | S8, S50 | yes (IBC) |
| 14 | Group E (K-12): fire area over 12,000 ft² (1115 m²), occupant load 300 or more, or on a floor other than the level of exit discharge | ft² (m²), persons | schools, not universities | IBC 903.2.3 | S8 | partly |
| 15 | Higher-education laboratory suites: sprinklers throughout | n/a | university labs | IBC 428 (S8 cites 428.3.9; SBC 201-2024 contents list Section 428) | S8, S7 | partly |
| 16 | S-2 parking: enclosed garage fire area over 12,000 ft² (1115 m²); open garage over 48,000 ft² (4459 m²); any garage beneath other groups (R-3 excepted) | ft² (m²) | car parks | IBC 903.2.10 | S8; S5 reports 4,460 m² for open parking in SBC 2024 | yes (IBC); SBC figure partly |
| 17 | Class III standpipes where the highest floor is more than 30 ft (9144 mm) above the lowest FD vehicle access; Class I permitted if the building is fully sprinklered | ft (mm) | standpipes | IFC/IBC 905.3.1; SBC 801: "more than 9 m" | S2, S13 | yes |
| 18 | At least two fire service access elevators where an occupied floor is more than 120 ft (36 576 mm) above FD access | ft (mm) | high-rise | IBC 403.6.1; SBC: "more than 36 m" | S7, S2 | yes |
| **Q2. Installation standards** | | | | | | |
| 19 | 903.3.1.1 requires NFPA 13; 903.3.1.2 NFPA 13R; 903.3.1.3 NFPA 13D; standpipes NFPA 14 (905); fire pumps NFPA 20 (913) | n/a | all | IFC/IBC 903.3.1, 905, 913 | S2 (NFPA 14 in SBC 801), S9, S53 | yes (IFC structure); SBC editions not read |
| 20 | IFC 2015 references the 2013 editions of NFPA 13, 13R, 14 and 20 | edition | SBC 2018 base | IFC 2015 ch. 80 | S49, S49b | partly |
| 21 | IFC 2021 references the 2019 editions of NFPA 13, 13R, 14 and 20 | edition | SBC 2024 base | IFC 2021 ch. 80 | S48 (three sources) | partly |
| **Q3. Placement, NFPA 13 standard spray upright and pendent** | | | | | | |
| 22 | Light hazard, hydraulically calculated, noncombustible or unobstructed combustible: max 225 ft² (20.9 m²) per sprinkler, max spacing 15 ft (4.6 m) | ft² (m²), ft (m) | LH | NFPA 13-2019/2022 Table 10.2.4.2.1(a); 2016 Table 8.6.2.2.1(a) | S15, S16, S14 (spacing), OEP | yes |
| 23 | Light hazard, pipe schedule: 200 ft² (18.6 m²), 15 ft | ft² | LH | same table | S15 | partly |
| 24 | Light hazard, combustible obstructed, members 3 ft (0.91 m) or more on centre: 168 ft² (15.6 m²), 15 ft | ft² | LH | same table | S15; OEP gives 168 but assigns it differently | partly |
| 25 | Light hazard, combustible, members less than 3 ft on centre: 130 ft² (12.1 m²), 15 ft; concealed pitched roof (4/12 or steeper) of wood joist or truss: 120 ft² (11.1 m²), 15 ft along the slope and 10 ft (3.05 m) across it | ft² | LH | same table | S15 | partly |
| 26 | Ordinary hazard (groups 1 and 2): 130 ft² (12.1 m²), 15 ft (4.6 m) | ft², ft | OH | Table 10.2.4.2.1(b); 2016 8.6.2.2.1(b) | S15, S16, S14, OEP | yes |
| 27 | Extra hazard: pipe schedule 90 ft² (8.4 m²) at 12 ft (3.7 m); hydraulic at 0.25 gpm/ft² (10.2 mm/min) or more, 100 ft² (9.3 m²) at 12 ft; hydraulic below 0.25, 130 ft² (12.1 m²) at 15 ft | ft², ft | EH, high-piled storage | Table 10.2.4.2.1(c), (d) | S15, S14 (spacing) | partly |
| 28 | Protection area A = S × L, where S (along the branch line) and L (between branch lines) are each the larger of the distance to the next sprinkler and twice the distance to the wall | n/a | all standard spray | NFPA 13-2022 9.5.2.1 | S23, OEP, patents US5366022 and US6976543 | yes |
| 29 | Max distance from a wall: half the allowable spacing, i.e. 7.5 ft (2.3 m) at 15 ft spacing | ft (m) | LH, OH | 2019 10.2.5.2 | S14, S16, OEP | yes |
| 30 | Small room rule: in a light hazard room of 800 ft² (74 m²) or less with unobstructed construction, a sprinkler may be up to 9 ft (2.7 m) from one wall | ft² (m²), ft (m) | LH | 2019 10.2.5.2.3 | S14, S16, OEP | yes |
| 31 | Min distance from a wall: 4 in (102 mm) | in (mm) | all standard spray | 2019 10.2.5.3 | S14, OEP | yes |
| 32 | Min distance between sprinklers: 6 ft (1.8 m), unless baffles are fitted | ft (m) | all standard spray | 2019 10.2.5.4 | S14, S16, OEP | yes |
| 33 | Deflector 1 to 12 in (25 to 305 mm) below an unobstructed ceiling (concealed, recessed and flush types per their listing) | in (mm) | unobstructed | 2019 10.2.6.1.1.1; 2016 8.6.4.1.1.1 | S18, S16, OEP | yes |
| 34 | Obstructed construction: deflector 1 to 6 in (25 to 152 mm) below the structural members and no more than 22 in (559 mm) below the deck; or at or above the member bottom within 22 in, subject to the beam rule; or one sprinkler in each bay, 1 to 12 in below the deck | in (mm) | obstructed | 2019 10.2.6.1.2 | S18, S16, OEP | yes |
| 35 | Three times rule: the sprinkler sits at least 3 × the largest dimension of an obstruction (column, pipe, fixture) away from it, but never more than 24 in (610 mm) is required | in (mm) | all standard spray | 2022 10.2.7.3.1.3 | S20, OEP, S22 | yes |
| 36 | In light and ordinary hazard, only structural members are subject to the three times rule; non-structural items remain under the other obstruction rules | n/a | LH, OH | 2022 10.2.7.3.1.4 | S20 | partly |
| 37 | Beam rule: for each horizontal distance A from the sprinkler to the side of the obstruction, a maximum height B of the deflector above the obstruction's bottom (table in section 3.4) | in (mm) | all standard spray | 2019 Table 10.2.7.1.2; 2016 Table 8.6.5.1.2 | S19; S21 checks two rows; OEP agrees up to 3.5 ft | partly |
| 38 | Obstructions wider than 4 ft (1.2 m), such as ducts and decks, need sprinklers beneath them | ft (m) | all | 2019 9.5.5.3 | S19, S21, S22 | yes |
| 39 | Keep 18 in (457 mm) clear between the deflector and the top of storage or contents | in (mm) | all standard spray | 2016 8.5.6.1; 2019 9.5.6 | S16, S21, S22 | yes |
| 40 | Residential pendent (one product, Tyco TY2234): listed coverage from 12 × 12 to 20 × 20 ft (3.7 × 3.7 to 6.1 × 6.1 m); min spacing 8 ft (2.4 m); deflector 1¼ to 4 in (32 to 102 mm) below a smooth ceiling | ft, in | residential sprinklers | TFP400 listing | S26; S15 says to follow the listing | partly (one product) |
| 41 | NFPA 13 with residential sprinklers: the four hydraulically most demanding sprinklers, each giving the listed flow and at least 0.1 gpm/ft² (4.1 mm/min) | n/a | R-2 under NFPA 13 | NFPA 13 residential criteria | S26, S29 | partly |
| 42 | NFPA 13D: up to 2 design sprinklers, 10 min supply (7 min for some small dwellings). NFPA 13R: up to 4 design sprinklers, 30 min supply. Minimum density 0.05 gpm/ft² (2.0 mm/min) or the listed flow | sprinklers, min | 13D, 13R | NFPA 13D/13R | S37 (three secondary sources) | partly |
| 43 | Hazard classes: offices, dwelling units and classrooms LH; restaurant service areas and kitchens, parking garages, and mechanical and electrical rooms OH1; storage rooms OH1 if 8 ft (2.4 m) or lower, OH2 up to 12 ft (3.7 m); mercantile OH2 | class | room types | NFPA 13-2022 4.3.2 to 4.3.3.2 and annex A.4.3 | S16, S24, S25, OEP | yes (storage rows partly) |
| **Q4. Hydraulics and sprinklers** | | | | | | |
| 44 | Design density: LH 0.10 gpm/ft² (4.1 mm/min); OH1 0.15 (6.1); OH2 0.20 (8.1); EH1 0.30 (12.2); EH2 0.40 (16.3) | gpm/ft² (mm/min) | occupancy hazard method | NFPA 13-2022 Table 19.2.3.1.1 (S57); 2007 to 2016 Figure 11.2.3.1.1 | OEP, S28 (LH and OH1), S57 | yes |
| 45 | Design area: 1,500 ft² (139 m²) for LH and OH; 2,500 ft² (232 m²) for EH (the smallest area on the NFPA curve) | ft² (m²) | same | same | OEP, S57; S28 (US DoD) uses 3,000 ft² (280 m²) | partly |
| 46 | Flow Q = K√P; K5.6 gpm/psi^½ = K80.6 L/min/bar^½ (the "K80" head); K4.9 = K70.6; conversion factor about 14.4 | n/a | all sprinklers | NFPA 13 hydraulics | S27, S26, S29 | yes |
| 47 | Minimum operating pressure at any sprinkler: 7 psi (0.48 bar; NFPA's metric figure is 0.5 bar) | psi (bar) | all | NFPA 13 | S29, S26, S57 | yes |
| 48 | Hose stream allowance and duration: LH total 100 gpm (380 L/min), 30 min; OH total 250 gpm (950 L/min), 60 to 90 min; EH total 500 gpm (1900 L/min), 90 to 120 min; inside hose 0, 50 or 100 gpm | gpm (L/min), min | occupancy hazard method | 2016 Table 11.2.3.1.2 | S31, S30, S57 | LH and OH yes; EH partly |
| 49 | The lower duration is allowed only where waterflow and supervisory devices are electrically supervised and monitored at a constantly attended location | n/a | same | 2016 11.2.3.1.3 | S30 | partly |
| 50 | Temperature classes: ordinary 135 to 170°F (57 to 77°C), for ceilings up to 100°F (38°C); intermediate 175 to 225°F (79 to 107°C), up to 150°F (66°C); high 250 to 300°F (121 to 149°C), up to 225°F (107°C) | °F (°C) | all | NFPA 13 temperature-rating table (2019 Table 7.2.4.1, numbering not checked) | S31, S32, S16 (intermediate), S27 | yes |
| 51 | Bulb colours: ordinary orange (135°F/57°C) or red (155°F/68°C); intermediate yellow (175°F/79°C) or green (200°F/93°C); high blue | colour | glass bulbs | same table | S31, S32 | yes |
| 52 | RTI: fast response 50 (m·s)^½ or less (about 90 (ft·s)^½); standard response 80 (m·s)^½ or more (about 145 (ft·s)^½); "special response" (50 to 80) exists in ISO only | (m·s)^½ | thermal elements | NFPA 13 definitions; ISO 6182-1 | S31, S33, S34 | yes |
| 53 | Quick-response sprinklers required throughout light hazard occupancies | n/a | LH | NFPA 13-2010 8.3.3.1 (as cited) | S16; S45 (VA policy, stricter) | partly |
| **Q5. Assembly points** | | | | | | |
| 54 | Safe dispersal area (only where no public way can be reached): at least 5 ft² (0.46 m²) per person, at least 50 ft (15 240 mm) from the building, on the same lot, permanently identified, with a safe path to it | ft² (m²), ft (m) | exit discharge | IBC 1028.5 exception | S39 (blog and two forum quotations) | yes (IBC) |
| 55 | The fire safety plan's site plan must show the occupants' assembly point, the hydrants and the normal FD vehicle access routes | n/a | plans | IFC 404.2.2 (SBC 801 Section 404 has the same title) | S40 | partly |
| 56 | Occupants are accounted for at the assembly point | n/a | drills, evacuation | IFC 405 ("Accountability"; 405.8 in the edition cited) | S40 | partly |
| 57 | Fire-ground collapse zone: at least 1.5 × the building height | m | fire service operations | NIOSH guidance, not a code | S42 (two NIOSH documents) | yes |
| 58 | Fire apparatus access road within 45 m of every part of the facility, at least 6 m wide | m | site | SBC 801 503 (IFC 503) | S2, S47 | yes |
| 59 | UK national guidance, US OSHA and Australian AS 3745 set no distance for an assembly point | n/a | n/a | n/a | S38, S41, S58 | yes (no number) |

---

## 1. When the Saudi Building Code requires automatic sprinklers

### 1.1 Edition and access

- **2018 edition.** SBC 201-2018 and SBC 801-2018 are based on the 2015 International Codes (ICC country page, S1). Pillai (2025, S2) says SBC 801 was built on IFC 2015 and NFPA standards with Saudi changes, and that the 2018 SBC 801 runs to 832 pages. The SBC 801-2018 Code and Commentaries table of contents (S3) keeps the IFC Chapter 9 sections 901 to 915 under their IFC titles.
- **2024 edition, mandatory from 30 June 2025.** Sources: S51, S52, and the Fire Middle East summary on ground.news. One vendor gives 1 July 2025 (S5).
  - No official statement names its base code.
  - The SBC 801-2024 contents (S4, read from the open PDF) add Section 916 Gas Detection Systems and Section 917 Mass Notification Systems after 915. Both first appeared in IFC 2021.
  - The SBC 201-2024 contents follow the IBC 2021 egress numbering, with 1029 Egress Courts and Section 428 Higher Education Laboratories (S7).
  - A Saudi fire-protection vendor says the 2024 edition aligns with IFC 2021 (S5).
  - Two Saudi values the vendor reports match IBC 2021 exactly:
    - the business occupant load factor moves from 9 to 14 m² per person, which is IBC 2015's 100 ft² and IBC 2018/2021's 150 ft²;
    - open parking areas over 4,460 m² need sprinklers, which is IBC 2021's 48,000 ft² (4459 m²) in 903.2.10.
- **Conclusion.** IBC/IFC 2021 is the best available stand-in for SBC 2024. Every value in sections 1.2 to 1.6 is an IBC/IFC 2021 value, to be confirmed by reading SBC 801/201 Section 903.
- **Where the clauses sit.** SBC 801-2024 Section 903 runs from page 150 to page 157 of the Code Requirements (S4). In the IBC family the 903 text appears in both the building code and the fire code ("[F]" sections), so the same clauses should be in SBC 201 Chapter 9 and SBC 801 Chapter 9.

### 1.2 Group R-2 apartments

- **Trigger.** IBC/IFC 903.2.8 requires sprinklers throughout every building that contains a Group R fire area (S8). NFSA (S53) stresses that the trigger is the whole building: separating the R fire area from other uses does not remove the requirement for the rest of the building.
- **Which standard.** NFPA 13 throughout (903.3.1.1). NFPA 13R is allowed instead (903.3.1.2, as rewritten in 2021) only when all three hold (S9, S53):
  - the building has 4 stories or fewer above grade plane;
  - it is no more than 60 ft (18 288 mm) above grade plane;
  - the floor of the highest story is no more than 30 ft (9144 mm) above the lowest level of FD vehicle access.

  NFPA 13D applies only to Group R-3 and R-4 condition 1 (903.3.1.3), not to apartments.
- **Saudi point to check.** Whether SBC 801-2024 keeps 903.2.8 unchanged. Saudi residential construction is largely low-rise concrete villas and apartment blocks, and the 2018-era cost study (Allahaim et al., cited in S2) suggests the code's demands were contested. No source found states a Saudi exception.

### 1.3 Group B offices

IBC 903.2 has no general trigger for Group B. A B building needs sprinklers when any of the following applies (S8, S11):

1. **Ambulatory care.** A Group B ambulatory care facility with 4 or more care recipients incapable of self-preservation, or 1 or more on a level other than the level of exit discharge (903.2.2).
2. **Tall building.** Any story with an occupant load of 30 or more has its finished floor 55 ft (16 764 mm) or more above the lowest level of FD vehicle access (903.2.11.3). Open parking structures and Group F-2 are excepted. The system is then required throughout the building.
3. **High-rise.** The building is a high-rise (403.3; definition in 1.5).
4. **Stories without openings.** A story or basement over 1,500 ft² (139 m²) lacks qualifying exterior openings (903.2.11.1), or any part of a basement is more than 75 ft (22 860 mm) from those openings (903.2.11.1.3; S50, S56).
5. **Mixed use.** The building contains another use that triggers sprinklers: assembly (1.4), a Group R fire area (1.2), a parking garage (1.6) or higher-education laboratories (IBC 428).

For a typical office, rule 2 is the one that bites. At about 4 m floor to floor, the sixth floor (fifth above ground) is already about 16 to 20 m up. Note also that the egress limits in `codeCheck.js` (travel distance 300 ft and dead ends 50 ft with sprinklers; the 1/3 diagonal) assume sprinklers designed to 903.3.1.1, that is NFPA 13.

### 1.4 A university building

- **IBC classification.** Educational use above the 12th grade is Group B (IBC 304.1; S12a, S12b). Lecture halls and other rooms used for assembly may be A-3 (303.4). Assembly rooms with fewer than 50 occupants that serve another use count as part of that use (303.1.2; S12a forum summary).
  - Group E sprinkler rules (903.2.3) therefore do not apply to a university.
  - Group B rules do (1.3), plus A-3 rules for large lecture halls: sprinklers where the A-3 fire area exceeds 12,000 ft² (1115 m²), its occupant load is 300 or more, or it is on a floor other than the level of exit discharge (903.2.1.3; S8, S50).
  - IBC 903.2.1.7 (S8) adds a rule for several assembly areas: where A occupancies share exit access and their combined occupant load reaches 300, sprinklers are required.
- **Higher-education labs.** IBC Section 428 (Higher Education Laboratories, an option for labs that exceed control-area limits) requires sprinklers throughout the laboratory suites (S8 cites 428.3.9). SBC 201-2024's contents include Section 428 under the same title (S7).
- **Whether SBC 201 differs.** Not confirmed. The SBC 201-2024 contents keep the IBC chapter 3 structure, but the text of 304.1 was not read.
  - The superseded 2007 SBC 801 was organised differently (an NFPA 101-style "educational" group, S46).
  - Under the 2007 code a Saudi university may have been treated as educational. This matters for older campus buildings.

### 1.5 High-rise buildings: threshold and how it is measured

- **IBC/IFC definition.** A high-rise building has an occupied floor more than 75 ft (22 860 mm) above the lowest level of FD vehicle access (IBC 202; S10). IBC 403.3 then requires NFPA 13 sprinklers throughout (S8).
- **How it is measured** (S10, and the Indiana DHS interpretation in S10):
  - up to the **floor** of the highest occupied story, not the roof or the parapet;
  - from the **lowest level where a fire department vehicle can stand**, not from grade plane or average grade.

  On a sloping site this access level can lie below the grade plane, which makes the building taller in code terms.
- **Occupied roofs.** The 2024 IBC adds occupied roofs to the definition (S10). The SBC 2024, if it is based on 2021, may not include this.
- **The 55 ft rule.** 903.2.11.3 (S8, S11) is measured the same way, but to the finished floor of a story with 30 or more occupants. It requires sprinklers in many buildings below the high-rise line.
- **Saudi practice.** The SBC enforcement schedule announced in November 2018 (Saudi Press Agency, reported by S6) put "high-rise towers above 23 m" in phase 1 and "buildings below 23 m" in phase 2. 23 m is the rounded SI form of 75 ft. Whether SBC 201-2024's definition prints 23 m, 22.86 m or 22 860 mm was not read.
- **Related heights in Pillai's account of SBC 801-2018** (S2), which match the IBC:
  - two fire service access elevators above 36 m (IBC 403.6.1: 120 ft, 36 576 mm);
  - Class III standpipes above 9 m, Class I if sprinklered (IFC 905.3.1: 30 ft, 9144 mm);
  - fire apparatus access within 45 m of every part of the facility and roads at least 6 m wide (also S47).

### 1.6 Other triggers likely to matter for the app's models

- **Car parks** (S8, S5):
  - enclosed S-2 garages with a fire area over 12,000 ft² (1115 m²);
  - open garages over 48,000 ft² (4459 m²; reported for SBC 2024 as 4,460 m², with podium car parks in high-rises included);
  - any garage beneath another use (R-3 excepted);
  - mechanical-access enclosed garages, which need an engineered system (903.2.10).
- **Saudi Civil Defense trade-off.** SBC 801-2024 Appendix D drops aerial apparatus roads for Type IA, IB and IIA buildings, but only if they are fully sprinklered, have a Class I standpipe in an enclosed stair to the roof, and Civil Defense approves (S5). So sprinklers buy relief from access-road requirements.
- **Atriums and malls.** Both require sprinklers throughout (IBC 404.3, 402.5; S8).

### 1.7 Saudi Civil Defense requirements

- **Enforcement.** SBC 801 is enforced by the General Directorate of Civil Defense (998.gov.sa; S1). SBC 2024 lets Civil Defense-certified third-party reviewers act as AHJ reviewers (S51), and some SBC 2024 exceptions need Civil Defense approval (S5).
- **Separate rules.** I found no current, separately published Civil Defense sprinkler thresholds that override SBC 801. Before SBC 801 was made mandatory, Civil Defense issued its own safety requirements by occupancy, but none were found online in a citable form.
- **Historical note: the superseded SBC 801-2007.** A 2014 lecture on it (King Saud University / King Fahd Security College, S46) lists triggers that differ from the IFC. Among them:
  - assembly halls over 465 m²;
  - basements over 139 m² (or 230 m² in some uses);
  - exhibition halls over 1115 m²;
  - retail with a storey over 1115 m², a building over 2230 m², or more than 3 storeys;
  - underground car parks whose ceiling is more than 0.6 m below street level.

  These do not apply to new work, but buildings designed between 2007 and 2018 may follow them.

---

## 2. Installation standards and editions

- **Which standard.** In the IFC/IBC structure that SBC 801/201 follow (S2, S3, S4):
  - 903.3.1.1 requires installation to NFPA 13;
  - 903.3.1.2 to NFPA 13R (residential, within the limits in 1.2);
  - 903.3.1.3 to NFPA 13D;
  - standpipes (905) to NFPA 14, which Pillai confirms for SBC 801 (S2);
  - fire pumps (913) to NFPA 20;
  - fire alarm (907) to NFPA 72 (S2).
- **Which edition.** Not read for either SBC edition. By inheritance:
  - **SBC 2018** (on IFC 2015): IFC 2015 chapter 80 lists the 2013 editions of NFPA 13, 13R, 13D, 14 and 20. Sources: the South Carolina State Fire Marshal list (S49), which states it reflects the ICC references, and Dallas's 2015 adoption (S49b).
  - **SBC 2024** (on IFC 2021): IFC 2021 lists the 2019 editions (S48). Ohio's rule amends the IFC 2021 references for NFPA 13 and 20 "from 2019 to 2022", a fire district lists NFPA 14 and 20 at 2019, and a standards seller lists NFPA 13R 2019 in IFC 2021.
  - The Saudi committee may have updated or localised these; the AESG article says SBC 2024 newly references SASO standards (S51). This is the first thing to read in SBC 801-2024 chapter 80 (contents p. 682, S4).
- **Newer NFPA editions.** NFPA 13 has since been issued in 2022 and 2025. Table numbers cited in this note follow the source's edition: 2016 (chapter 8) or 2019/2022 (chapters 9, 10 and 19).

---

## 3. NFPA 13 placement rules (standard spray upright and pendent)

### 3.1 Protection area and spacing

- **Protection area (S23, OEP, patents).** The area a sprinkler protects is A = S × L:
  - **S** is measured along the branch line: the larger of the distance to the farthest adjacent sprinkler upstream or downstream, or twice the distance to the wall where the sprinkler is the last on the line;
  - **L** is measured between branch lines in the same way: the larger of the distance to the farthest adjacent branch line, or twice the distance to the wall.
- **Small rooms.** Under the small room rule, the area per sprinkler is the room area divided by the number of sprinklers (S23).
- **Limits.** A must not exceed the table limit, and neither S nor L may exceed the maximum spacing.

| Hazard and construction | Max area per sprinkler | Max spacing | Sources |
|---|---|---|---|
| LH, noncombustible (obstructed or not), or combustible unobstructed or with no exposed members, hydraulically calculated | 225 ft² (20.9 m²) | 15 ft (4.6 m) | S15, S16, OEP |
| LH, as above, pipe schedule | 200 ft² (18.6 m²) | 15 ft | S15 |
| LH, combustible obstructed, exposed members 3 ft (0.91 m) or more on centre | 168 ft² (15.6 m²) | 15 ft | S15 |
| LH, combustible, exposed members under 3 ft on centre, general | 130 ft² (12.1 m²) | 15 ft | S15 |
| LH, concealed space under a pitched roof (4/12 or steeper) of wood joists or trusses under 3 ft on centre | 120 ft² (11.1 m²) | 15 ft along the slope, 10 ft (3.05 m) across | S15 |
| OH groups 1 and 2, any construction | 130 ft² (12.1 m²) | 15 ft | S15, S16, OEP |
| EH, pipe schedule | 90 ft² (8.4 m²) | 12 ft (3.7 m) | S15, S14 |
| EH or high-piled storage, hydraulic, density 0.25 gpm/ft² or more | 100 ft² (9.3 m²) | 12 ft (12 ft 6 in, 3.8 m, in 25 ft structural bays) | S15, S14 |
| EH or high-piled storage, hydraulic, density under 0.25 gpm/ft² | 130 ft² (12.1 m²) | 15 ft | S15, S14 |

Table: NFPA 13-2019/2022 Table 10.2.4.2.1(a) to (d); 2002 to 2016 Table 8.6.2.2.1(a) to (d). "Unobstructed" construction is defined by NFPA 13; the classification (S16 item 5) has to be supplied by the user or read from the IFC.

### 3.2 Distances to walls and between sprinklers

- **Maximum to a wall.** Half the allowable spacing, measured perpendicular to the wall: 7.5 ft (2.3 m) for LH and OH, 6 ft (1.8 m) at 12 ft spacing (S14 10.2.5.2, S16, OEP).
  - At corners formed by walls meeting at less than 90°, 0.75 × spacing applies (S14 10.2.5.2.2). S14 prints 11.5 ft, which is inconsistent with 0.75 × 15 ft; not confirmed.
- **Small rooms.** In a LH room of 800 ft² (74 m²) or less with unobstructed construction, a sprinkler may be up to 9 ft (2.7 m) from any single wall, as long as the 225 ft² and 15 ft limits hold (S14 10.2.5.2.3, S16, OEP).
- **Minimum to a wall.** 4 in (102 mm) (S14 10.2.5.3, OEP).
- **Minimum between sprinklers.** 6 ft (1.8 m) on centre (S14 10.2.5.4, S16, OEP). Closer spacing needs baffles midway between them.
  - S14 gives baffles no longer than 8 in (203 mm) and no higher than 6 in (152 mm), with the top 2 to 3 in (51 to 76 mm) above upright deflectors.
  - OEP gives minimum baffle dimensions of 6 × 3 in instead. Not confirmed either way.

### 3.3 Deflector distance below the ceiling

Sources: S18, S16, OEP.

- **Unobstructed construction.**
  - Deflector 1 to 12 in (25 to 305 mm) below the ceiling (2019 10.2.6.1.1.1).
  - Concealed, recessed and flush sprinklers follow their listing.
  - In light and ordinary hazard with noncombustible ceilings, a vertical change in ceiling elevation of more than 36 in (914 mm) is treated as a wall. A change of 36 in or less is spaced as if the ceiling were flat (10.2.6.1.1.3).
- **Obstructed construction** (2019 10.2.6.1.2): one of these options.
  1. Deflector 1 to 6 in (25 to 152 mm) below the structural members and no more than 22 in (559 mm) below the deck.
  2. Deflector at or above the bottom of the members, no more than 22 in below the deck, and complying with the beam rule.
  3. One sprinkler in each bay, deflector 1 to 12 in below the deck.

  Further options exist for composite wood joists (1 to 6 in below, 22 in maximum) and concrete tees (at or above 1 in below the stem).

### 3.4 Obstructions

**Beam rule (obstructions at or above deflector level, within 18 in of it).** The sprinkler may sit above the bottom of a beam, duct or soffit by at most B, depending on its horizontal distance A to the side of the obstruction. No interpolation. Source: NFPA 13-2019 Table 10.2.7.1.2, as reproduced by S19. Corroboration:

- S21 (CSE, 2016 edition) checks two rows: 3½ in at 18 to 24 in, and 12 in at 3½ to 4 ft.
- OEP agrees through 3½ ft but gives different and smaller values beyond. I take S19 and S21 as correct and OEP as erroneous; still mark the rows beyond 4 ft as unconfirmed.

| A, sprinkler to side of obstruction | B, max deflector height above obstruction bottom | Confirmed |
|---|---|---|
| less than 1 ft (0.3 m) | 0 in (0 mm) | yes (S19, OEP) |
| 1 ft to less than 1 ft 6 in | 2½ in (65 mm) | yes (S19, OEP) |
| 1 ft 6 in to less than 2 ft | 3½ in (90 mm) | yes (S19, S21, OEP) |
| 2 ft to less than 2 ft 6 in | 5½ in (140 mm) | yes (S19, OEP) |
| 2 ft 6 in to less than 3 ft | 7½ in (190 mm) | yes (S19, OEP) |
| 3 ft to less than 3 ft 6 in | 9½ in (240 mm) | yes (S19, OEP) |
| 3 ft 6 in to less than 4 ft | 12 in (300 mm) | yes (S19, S21) |
| 4 ft to less than 4 ft 6 in | 14 in (350 mm) | partly (S19) |
| 4 ft 6 in to less than 5 ft | 16½ in (415 mm) | partly (S19) |
| 5 ft to less than 5 ft 6 in | 18 in (450 mm) | partly (S19) |
| 5 ft 6 in to less than 6 ft | 20 in (500 mm) | partly (S19) |
| 6 ft to less than 6 ft 6 in | 24 in (600 mm) | partly (S19) |
| 6 ft 6 in to less than 7 ft | 30 in (750 mm) | partly (S19) |
| 7 ft to less than 7 ft 6 in | 35 in (875 mm) | partly (S19) |

**Three times rule (obstructions entirely below the deflector, within 18 in of it).** Examples are columns, pipes, light fixtures and truss webs. S20 gives the 2022 numbering (10.2.7.3.1.3); OEP and S22 corroborate.

- The sprinkler must be at least 3 times the obstruction's largest dimension (width or depth) away from it.
- No more than 24 in (610 mm) of separation is ever required, except for large columns.
- Truss members 1½ in (38 mm) wide or less may be ignored (OEP).
- In light and ordinary hazard only structural members are subject to this rule (2022 10.2.7.3.1.4, S20; single source). Smoke detectors and light fixtures then remain subject only to the other obstruction rules.
- In light and ordinary hazard, sprinklers may instead go on both sides of an obstruction up to 4 ft (1.2 m) wide, each within half the allowable spacing of its centreline (S19 10.2.1.2; S21).

**Wide obstructions.** Fixed obstructions wider than 4 ft (1.2 m), such as ducts, decks, open grating, and canopies or roofs over 4 ft, need sprinklers beneath them (S19 9.5.5.3, S21, S22; S16 for canopies).

- The lower sprinklers sit within 3 in (76 mm) of the outside edge and 1 to 12 in below the obstruction.
- Exempt (S19, S21): noncombustible fixed obstructions whose bottom is 24 in (610 mm) or less above the floor, and movable items such as conference tables.

**Clearance to contents.** Keep 18 in (457 mm) between the deflector and the top of storage or contents (S16 8.5.6.1, S21, S22).

### 3.5 Residential sprinklers (NFPA 13, 13R, 13D)

- **Spacing comes from the listing, not from NFPA 13's tables** (S15). One current product (Tyco TY2234, data sheet TFP400, September 2019, S26) gives:
  - square coverage areas of 12 × 12, 14 × 14, 16 × 16, 18 × 18 and 20 × 20 ft (3.7 × 3.7 to 6.1 × 6.1 m), each with a minimum flow (wet pipe, K4.9): 13 gpm (49.2 L/min) at 7.0 psi (0.48 bar) up to 16 × 16 ft, 17 gpm at 12.0 psi for 18 × 18 ft, 20 gpm at 16.7 psi for 20 × 20 ft;
  - a minimum of 8 ft (2.4 m) between sprinklers, and a maximum spacing no greater than the side of the coverage area used;
  - deflector 1¼ to 4 in (32 to 102 mm) below a smooth ceiling, or 1¼ to 1¾ in (32 to 44 mm) below the bottom of a beam;
  - temperature ratings of 155°F (68°C) or 175°F (79°C).

  Wall distances follow the listing; the 4 in (102 mm) minimum is the general rule.
- **Hydraulic basis** (S26, S37):
  - **NFPA 13D** (one- and two-family dwellings): up to 2 design sprinklers, a 10-minute water supply (7 minutes allowed for some small dwellings), and the greater of 0.05 gpm/ft² (2.0 mm/min) or the listed flow.
  - **NFPA 13R** (residential up to 4 stories): up to 4 design sprinklers and a 30-minute supply.
  - **NFPA 13 using residential sprinklers:** the four hydraulically most demanding sprinklers, each giving the greater of the listed flow and 0.1 gpm/ft² (4.1 mm/min) over its area.
- **Activation model.** Residential and quick-response sprinklers are fast-response elements (RTI at most 50 (m·s)^½). This is what the activation model in the suppression note needs.

### 3.6 Hazard classification of the rooms the app will see

NFPA 13 sets hazard classes by the quantity and combustibility of contents. The annex lists only examples, and the designer decides (S25, MeyerFire summary of NFPA 13-2022 4.3.2 to 4.3.3.2 and A.4.3). NFPA's classes do not align with building-code occupancy groups (S25).

| Space | Class | Sources | Confirmed |
|---|---|---|---|
| Offices, including data processing | Light hazard | S16, S24, OEP, S25 (classroom example), Koorsen | yes |
| Apartments (dwelling units), hotel rooms, dormitories | Light hazard | S24, OEP; S26 markets residential sprinklers for homes, apartments, dormitories and hotels | yes |
| Classrooms, educational | Light hazard | S25, OEP | yes |
| Restaurant seating, meeting rooms, corridors, toilets | Light hazard | S16, S24 | yes |
| Commercial kitchens, restaurant service areas | Ordinary hazard group 1 | S16, S24, OEP | yes |
| Parking garages, car showrooms | Ordinary hazard group 1 | S24, OEP | yes |
| Mechanical, electrical, elevator machine (plant) rooms | Ordinary hazard group 1 | S25, S24, OEP | yes |
| Storage rooms, storage up to 8 ft (2.4 m) and moderate combustibility | Ordinary hazard group 1 | S24, OEP | partly |
| Storage rooms, storage up to 12 ft (3.7 m) or more combustible | Ordinary hazard group 2 | S24, OEP | partly |
| Mercantile (shops) | Ordinary hazard group 2 | S16, S25 | yes |
| Library large stack rooms | Ordinary hazard group 2 (general library areas LH) | OEP; one search snippet put it in OH1 | partly |
| High-piled or rack storage above 12 ft | Outside occupancy classes; NFPA 13 storage chapters by commodity | S24, OEP | yes |

Where a few OH rooms sit in a LH building, each is designed to its own class. The light hazard 30-minute supply is kept only for single OH rooms of 400 ft² (37 m²) or less (MeyerFire forum citing NFPA 13-2016 11.1.6.1; one source, not confirmed). The US DoD and VA set stricter rules of their own (S28, S45), for example quick-response sprinklers everywhere at the VA.

---

## 4. Hydraulic design basics

### 4.1 Density and area (occupancy hazard method)

| Hazard | Density | Design area (minimum on the curve) | Sources |
|---|---|---|---|
| Light | 0.10 gpm/ft² (4.1 mm/min, i.e. L/min per m²) | 1,500 ft² (139 m²) | OEP, S28 (density), S57 |
| Ordinary group 1 | 0.15 gpm/ft² (6.1 mm/min) | 1,500 ft² (139 m²) | OEP, S28 (density), S57 |
| Ordinary group 2 | 0.20 gpm/ft² (8.1 mm/min) | 1,500 ft² (139 m²) | OEP, S57 |
| Extra group 1 | 0.30 gpm/ft² (12.2 mm/min) | 2,500 ft² (232 m²) | OEP, S57 |
| Extra group 2 | 0.40 gpm/ft² (16.3 mm/min) | 2,500 ft² (232 m²) | OEP, S57 |

- **Where the values sit.** Through the 2016 edition these were the low-area ends of the density/area curves (Figure 11.2.3.1.1 in 2007 to 2016, S57). MeyerFire cites Table 19.2.3.1.1 for the 2022 edition.
- **Not confirmed.**
  - Whether larger areas at lower densities along the old curves are still allowed in the 2022 and 2025 editions.
  - The design-area reduction allowed for quick-response sprinklers (not researched).
- **Design area shape.** The design area is the hydraulically most remote area. NFPA 13 also sets its shape (a rectangle about 1.2 × √A along the branch lines); not verified here.
- **US DoD difference.** UFC 3-600-01 (S28) is the US Department of Defense's criteria, not NFPA 13. It uses the same densities over a larger 3,000 ft² (280 m²), with 250 and 500 gpm hose allowances and 60 minutes. A source that quotes 3,000 ft² is therefore probably quoting the UFC.

### 4.2 K-factor, pressure, flow

- **Discharge.** Q = K√P, with Q in gpm and P in psi, or Q in L/min and P in bar (S29, OEP).
  - K in L/min/bar^½ ≈ 14.4 × K in gpm/psi^½. My conversion from unit definitions is 14.41.
  - Tyco prints K5.6 = 80.6 L/min/bar^½ and K4.9 = 70.6 (S27, S26); the common metric name is "K80".
- **Minimum pressure.** No sprinkler may operate below 7 psi (0.48 bar; NFPA's metric figure is 0.5 bar) (S29; S26's minimum is 7.0 psi, 0.48 bar). A K5.6 head at 7 psi gives 14.8 gpm (56 L/min).
- **Minimum flow per head.** Each head's flow must also meet density × its protection area (S29). For example, LH at 0.10 gpm/ft² × 144 ft² is 14.4 gpm.
- **Calculation basis (not researched).** Friction loss (Hazen-Williams, C = 120 for wet steel pipe) and velocity pressure follow NFPA 13 chapter 27 (2019 numbering), which this note did not research.

### 4.3 Hose stream allowance and water supply duration

| Hazard | Inside hose | Total inside plus outside hose | Duration | Sources | Confirmed |
|---|---|---|---|---|---|
| Light | 0, 50 or 100 gpm (0, 190 or 380 L/min) | 100 gpm (380 L/min) | 30 min | S31 (reproduced table row), OEP | yes |
| Ordinary | 0, 50 or 100 gpm | 250 gpm (950 L/min) | 60 to 90 min | S30, S57 | yes |
| Extra | 0, 50 or 100 gpm | 500 gpm (1900 L/min) | 90 to 120 min | S57 (a low-quality blog) and general knowledge | partly |

- **Table.** NFPA 13-2016 Table 11.2.3.1.2 (S30); the 2022 number was not verified.
- **Lower durations** are allowed only where waterflow and supervisory devices are electrically supervised to a constantly attended location (2016 11.2.3.1.3, S30).
- **Total supply.** The required water is the sprinkler demand plus the hose allowance, times the duration.

### 4.4 Temperature ratings and response

**Temperature classes** (NFPA 13 temperature-rating table; S31 USFA, S32, S16 for intermediate, S27 for the 286°F/141°C rating).

| Class | Rating | Max ceiling temperature | Bulb colour |
|---|---|---|---|
| Ordinary | 135 to 170°F (57 to 77°C); common nominals 135°F (57°C), 155°F (68°C) | 100°F (38°C) | orange (135°F), red (155°F) |
| Intermediate | 175 to 225°F (79 to 107°C); 175°F (79°C), 200°F (93°C) | 150°F (66°C) | yellow (175°F), green (200°F) |
| High | 250 to 300°F (121 to 149°C); 286°F (141°C) | 225°F (107°C) | blue |

- **Frame colours.** Frame colour codes (ordinary uncoloured or black, intermediate white, high blue) are a separate system from bulb colours (S32).
- **Where intermediate is required** (S16, NFPA 13-2010 8.3.2.5):
  - under skylights exposed to direct sun;
  - in unventilated concealed spaces and attics;
  - under uninsulated roofs;
  - in unventilated show windows.

  In Saudi roof spaces and plant rooms this rule will often apply.
- **Response time index** (RTI, the thermal responsiveness of the element; Heskestad and Bill 1988, S35):

  | Response class | RTI in (m·s)^½ | RTI in (ft·s)^½ | Sources |
  |---|---|---|---|
  | Fast response | 50 or less | about 90 or less | S31, S33, S34 |
  | Standard response | 80 or more | about 145 or more (my conversion, factor 1.811) | S31, S33, S34 |
  | Special response (ISO 6182-1 only) | 50 to 80 | n/a | S31, S34 |

  - Quick-response, residential, ESFR and quick-response extended-coverage sprinklers are all fast-response types (S31, S33).
  - Listing can differ by agency: FM treats concealed heads as standard response even when UL lists them as quick response (S33, S45).
  - Glass-bulb sprinklers of the same rating can differ in response; Arvidson (2018, S36) measured this in a residential room fire.
- **Where quick response is required** (S16, 2010 8.3.3.1): in light hazard occupancies. IBC 903.3.2 also requires quick-response or residential sprinklers in dwelling and sleeping units; a secondary source cites this, but the wording was not read. Standard and quick-response heads should not be mixed in one compartment (S16, S45).

---

## 5. Outdoor assembly points (places of safety after evacuation)

### 5.1 What the codes and guidance set

| Topic | Rule or guidance | Number set? | Source |
|---|---|---|---|
| Discharge to a public way, or a "safe dispersal area" | Exits must discharge to a public way. Where none can be reached, a safe dispersal area on the same lot is allowed: at least 5 ft² (0.46 m²) per person, at least 50 ft (15 240 mm) from the building, permanently maintained and identified, with a safe and unobstructed path from the building. Some forum posts mention I-2 variants (30 ft² per bed or stretcher patient, 6 ft² per ambulatory patient), not verified. | **Yes: the only code numbers found** | IBC 1028.5 exception (S39); SBC 201 Section 1028 has the same title, value not read |
| Assembly points in the fire safety plan | The plan's site plan must show the occupants' assembly point, the fire hydrants and the normal routes of FD vehicle access. Large sites may have several assembly points. | No | IFC 404.2.2 (S40); SBC 801-2024 Section 404 "Fire safety, evacuation and lockdown plans" (S4) |
| Accounting for occupants | On arrival at the assembly point, staff determine whether everyone has left or is accounted for | No | IFC 405 accountability clause (405.8 in the edition S40 cites; 405.9 in another); OSHA 29 CFR 1910.38(c)(4) (S41) |
| Fire service access | Fire apparatus access roads must stay unobstructed. SBC 801 requires a road within 45 m of every part of the facility, at least 6 m wide (IFC: 20 ft, 6096 mm). An assembly point must not sit on these roads or around hydrants and FD connections. | Road width and reach only | SBC 801/IFC 503 (S2, S47); UC Irvine (S43) |
| UK | Training must cover reaching "an assembly point at a place of total safety". A place of total safety is a place away from the premises with no immediate danger from fire. Roll call at the designated assembly points. | **No** | HM Government 2006, Fire safety risk assessment: offices and shops, pp. 35, 107 to 112 and glossary (S38) |
| US OSHA | An emergency action plan must include procedures to account for all employees after evacuation; roll call at the assembly area is one method | **No** | 29 CFR 1910.38(c)(4), OSHA eTool (S41) |
| Australia | AS 3745 requires assembly areas on evacuation diagrams and at a suitable distance; checklists give no figure | **No** (standard not read) | S58 |

### 5.2 Fire-service and institutional guidance that gives numbers (not code)

- **Collapse zone.** NIOSH fire-fighter fatality guidance sets a perimeter of at least 1.5 times the building height for defensive operations, plus an allowance for debris scatter (S42). It is written for firefighters, not occupants. Even so, it is the only authoritative distance found that scales with the building, and an assembly point inside it would conflict with fire-fighting.
- **Behind the responders.** UC Irvine Emergency Management (2022, S43) dropped assembly areas close to buildings, fire hydrants or fire lanes. Its rule: any space between the building and the responders' staging area is unsuitable, and evacuees should stand behind the responders, their vehicles and their equipment. It also keeps several areas per zone, so another can be used if the first is unsafe.
- **Secondary area for schools.** The District of Columbia school emergency plan template (2009, S44) asks for a second area 500 ft (152 m) from the building. It also lists hazards to avoid at assembly areas:
  - gas, sewer and power lines;
  - chain-link fences;
  - water towers and transformers;
  - multistorey buildings that could collapse;
  - balconies that could fall.
- **Practitioner rules of thumb.** Figures of 50 ft (15 m) and 18 m appear in commercial safety blogs without a cited standard. Do not use them as code values.
- **Upwind.** No building code found requires an assembly point to be upwind. "Approach and stay upwind" is standard hazmat practice (US DOT Emergency Response Guidebook, not opened here), and some safety blogs extend it to assembly points. The app could treat wind as a user input for choosing between candidate assembly points, labelled as good practice, not code.

### 5.3 Capacity per person

- **Only code figure.** The IBC safe dispersal area figure, 5 ft² (0.46 m²) per person (S39), is the only code value found for an outdoor area that must hold evacuated occupants.
  - The same value is the IBC occupant load factor for "standing space" (Table 1004.5, net), so it is a crowd density of about 2.2 persons per m².
  - The forum quotations attribute 3 ft² (0.28 m²) per person to stadium seating dispersal; not verified.
- **No assembly-point figure.** No source found sets a per-person area for an assembly point in particular. The app would have to label its choice: IBC 5 ft² per person as a minimum, with a more generous value for comfort.

### 5.4 What this means for the app's assembly-point check

Where there is no code number, each item must be a parameter with its basis stated:

| Item | Status | Basis |
|---|---|---|
| Distance from the facade | Parameter, defaults: 50 ft (15.24 m) minimum (code only for safe dispersal areas); 1.5 × building height (NIOSH, fire service) | S39, S42 |
| Clear of exits' discharge paths, fire apparatus roads (6 m), hydrants and FD connections | Good practice and plan content | IFC 404.2.2, 503; S43 |
| Not between the building and the fire service staging area | Guidance | S43 |
| Capacity | 0.46 m² per evacuee (code minimum, safe dispersal) | S39 |
| Wind | Good practice only | none in codes |
| A second assembly point | Guidance | S40, S43, S44 |

---

## 6. Open questions and what could not be confirmed

1. **Every SBC clause value.** No SBC 201 or SBC 801 text was read, for either edition.
   - The Arabic code provisions are free but open only in a protected viewer; a person with a browser can read them.
   - The English Code Requirements cost 500 SAR (SBC 801) and 550 SAR (SBC 201).
   - Priorities in SBC 801-2024 (Code Requirements):
     - Section 903, pp. 150 to 157: the triggers 903.2.8, 903.2.11.3 (55 ft in SI) and 903.2.10, and the NFPA 13R limits 903.3.1.2;
     - Section 905;
     - Chapter 80: which NFPA editions it cites and whether SASO or other standards are added;
     - the high-rise definition in SBC 201 chapter 2.
2. **Base of SBC 2024.** The IFC 2021 base is inferred from the table of contents and from one vendor's statement, not from an official committee document.
3. **The 23 m Saudi high-rise threshold.** It comes from a 2018 press report of the enforcement schedule, not from the SBC 201 definition.
4. **Saudi Civil Defense.** No current Civil Defense document setting sprinkler thresholds separately from SBC 801 was found.
5. **University classification under SBC 201.** Not read: whether 304.1 keeps "educational above the 12th grade" in Group B.
6. **NFPA 13 values with one source only:**
   - LH pipe-schedule 200 ft²;
   - the combustible construction areas (168, 130 and 120 ft²);
   - the extra hazard area and spacing table;
   - beam rule rows from 4 ft upward;
   - the light and ordinary hazard limit of the three times rule to structural members;
   - the narrow-corner 0.75 × S rule;
   - baffle dimensions (two sources disagree);
   - the 400 ft² small OH room allowance.
7. **Hydraulics:**
   - the extra hazard hose allowance and duration (500 gpm, 90 to 120 min);
   - whether the 2022 and 2025 editions keep the density/area curves or only the single points;
   - the quick-response area reduction;
   - the 2022 table numbers for hose allowance and temperature ratings.
8. **NFPA 13D/13R design rules** (number of design sprinklers, durations) come from secondary sources; NFPA 13D and 13R were not opened.
9. **IFC 404.2.2 and 405 wording** for assembly points was confirmed only through state and local review sheets, and their clause numbers differ by edition.
10. **Assembly points.** No national standard or fire service guidance found sets a distance for an assembly point. The only code numbers apply to safe dispersal areas (IBC 1028.5). SCDF's (Singapore) evacuation planning guideline was not reachable (404), and AS 3745 was not read.

---

## 7. Sources

All accessed 2026-11-07 unless stated. Web text was treated as data. No account was created and no agreement accepted. Copies of standards hosted on Scribd were not used.

**Saudi Building Code and its base**
- S1. ICC, Global Building Codes: Saudi Arabia. https://www.iccsafe.org/?p=240998
- S2. Pillai, S. R. (2025). From prescription to performance: a comparative analysis of SBC 801 and ADB fire codes. *American Academic Scientific Research Journal for Engineering, Technology, and Sciences* 102(1), 284 to 293. https://asrjetsjournal.org/American_Scientific_Journal/article/view/11808 . No DOI found at Crossref (bibliographic query returned no match).
- S3. SBCNC, SBC 801-CC 2018, table of contents. https://sbc.gov.sa/ar/BC/Documents/tableofcontent2018/SBC%20801/SBC801_CC_T2018.pdf
- S4. SBCNC, SBC 801-CR 2024, table of contents (read as text). https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20801/SBC801_CR_241224-FA.pdf
- S5. Vortex Fire, "SBC 201 & 801 2024 edition: the key changes you need to know" (vendor). https://vortexfire.com/insights/sbc-201-801-2024-edition-the-key-changes-you-need-to-know
- S6. Aleqtsad, report of the Saudi Building Code launch and enforcement schedule, 18 November 2018 (citing the Saudi Press Agency). https://www.aleqtsad.org/?p=98336
- S7. Repository note `TopologicStudio-Next/wiki/research/tall-office-sbc-egress-2026-11-05.md` (SBC access, 2024 table of contents numbering, IBC 2021 egress values).
- S46. Khalaf, T., King Saud University / King Fahd Security College lecture on SBC 801 (2007 edition) water-based systems, 2014. https://faculty.ksu.edu.sa/sites/default/files/lec_7.pdf (historical only).
- S47. ICC, 2018 SBC 801 Study Companion, sample pages (fire apparatus road 45 m). https://shop.iccsafe.org/media/wysiwyg/material/4407S18SBC-Sample.pdf
- S51. International Fire & Safety Journal, "Building compliance with confidence: AESG breaks down Saudi code changes". https://internationalfireandsafetyjournal.com/sbc-2024-compliance-aesg/
- S52. SIMA, "Update of the Saudi Building Code System 2024". https://sima-ksa.com/market-insight/update-of-the-saudi-building-code-system-2024 ; Fire Middle East summary via https://ground.news/article/fire-middle-east-magazine_9ba44b

**IBC/IFC 2021**
- S8. MeyerFire, IBC Sprinkler Cheatsheet (2018, 2021, 2024), August 2025. https://www.meyerfire.com/uploads/1/6/0/7/16072416/meyerfire_-_sprinkler_-_where_required_in_ibc__2025_.pdf
- S9. ICC, 2021 Significant Changes, IBC 903.3.1.2 (via search summary). https://www.iccsafe.org/wp-content/uploads/2021_SigChanges_IBC_903.3.1.2.pdf
- S10. ICC Building Safety Journal, "Talking in code: high-rise building definition". https://www.iccsafe.org/building-safety-journal/bsj-technical/talking-in-code-high-rise-building-definition/ ; NFSA, "Does your structure measure up to a high-rise building?" https://nfsa.org/2020/02/21/does-your-structure-measure-up-to-a-high-rise/ ; Indiana DHS, Section 905.3.1, how to measure height above the lowest level of FD vehicle access. https://secure.in.gov/dhs/files/Section-905.3.1-How-to-measure-height-above-lowest-level-of-fire-department-vehicle-access.pdf
- S11. Data Driven AEC, "IBC sprinkler requirements" (55 ft rule). https://datadrivenaec.com/insights/ibc-sprinkler-requirements
- S12a. The Building Code Forum, "College classrooms" (quotes IBC 303.1.2, 304.1). https://www.thebuildingcodeforum.com/forum/threads/college-classrooms.39212/post-314046
- S12b. MeyerFire, "What is a Business Occupancy?" https://www.meyerfire.com/university/what-is-a-business-occupancy ; Seattle Building Code 2021, chapter 3 (IBC 2021 base). https://seattle.gov/documents/departments/sdci/codes/seattlebuildingcode/2021sbcchapter3.pdf (listed, not opened)
- S13. NFSA, "When are standpipes required for fire protection?" https://nfsa.org/?p=12881 ; ICC interpretation BU 00-53-03. https://www3.iccsafe.org/cs/committeeArea/pdf_file/BU_00_53_03.pdf
- S48. Ohio Admin. Code 1301:7-7-80 (amends IFC 2021 references for NFPA 13 and 20 from 2019 to 2022). https://codes.ohio.gov/assets/laws/administrative-code/rules/1301/7/1301$7-7-80_eff_11_20_25.pdf ; Central Valley Fire District, adopted codes. https://cvfdmt.specialdistrict.org/adopted-codes-standards ; standards.ie listing of IFC 2021. https://shop.standards.ie/en-ie/standards/icc-ifc-international-fire-code-2021-1307227_saig_icc_icc_3185968
- S49. South Carolina State Fire Marshal, currently adopted standards (2016). https://statefire.llr.sc.gov/osfm/forms/engineering/Currently%20Adopted%20Standards_rev.2016.06.24.referenced%20in%20both%20SFMR%20and%20IFC.pdf ; S49b. Dallas, 2015 ICC codes. https://dallascityhall.com/departments/sustainabledevelopment/buildinginspection/Pages/2015-icc-codes.aspx
- S50. Washington WAC 51-54A-0903 (IFC 2021 as amended; 2024 archive). https://lawfilesext.leg.wa.gov/Law/WACArchive/2024/htm/WAC%20%2051%20-%2054A%20CHAPTER/WAC%20%2051%20-%2054A-0903.htm
- S53. NFSA, "What does the IBC say about NFPA 13R?" https://nfsa.org/2022/12/15/nfpa-13r-and-the-international-building-code/ ; NFSA, "When Group R is present, the building is the trigger for sprinklers" (2025). https://nfsa.org/2025/12/17/group-r-is-present-the-building-is-the-trigger-for-sprinklers/
- S56. The Building Code Forum and the Bellevue amendment on 903.2.11.1 (1,500 ft², 75 ft). https://bellevue.municipal.codes/BCC/23.11.903.2.11

**NFPA 13 placement and hazard (secondary)**
- S14. QRFS blog #214, "Maximum and minimum sprinkler distance rules, part 1" (NFPA 13-2019). https://blog.qrfs.com/214-maximum-and-minimum-sprinkler-distance-rules-part-1-standard-spray-fire-sprinklers/
- S15. MeyerFire, Fire Sprinkler Spacing flowchart, NFPA 13 2002 to 2019, April 2022. https://www.meyerfire.com/uploads/1/6/0/7/16072416/meyerfire_-_sprinkler_-_spacing__2022_.pdf
- S16. City of Newport Beach, Building Division, Tenant Improvement Fire Sprinkler Plan Review Comments (NFPA 13-2010), 3 March 2011. https://www.newportbeachca.gov/home/showpublisheddocument/6556/635682493202100000
- S18. MeyerFire, Fire Sprinkler Deflector Height flowchart, NFPA 13 2002 to 2019, July 2020. https://meyerfire.com/uploads/1/6/0/7/16072416/meyerfire_-_sprinkler_-_deflector_height__2020_.pdf
- S19. QRFS blog #225, "Distance rules part 2: obstruction distance rules for standard spray sprinklers" (NFPA 13-2019). https://blog.qrfs.com/225-distance-rules-part-2-sprinkler-head-obstruction-distance-rules-for-standard-spray-fire-sprinklers
- S20. QRFS blog #427, "Fire sprinkler obstructions: the rules for nonstructural objects" (NFPA 13-2022). https://blog.qrfs.com/427-fire-sprinkler-obstructions-the-rules-for-nonstructural-objects/
- S21. Consulting-Specifying Engineer, "Sprinkler installations and obstructions", 18 April 2019 (NFPA 13-2016). https://www.csemag.com/sprinkler-installations-and-obstructions/
- S22. O'Connor, B. (NFPA), "Understanding and resolving sprinkler obstructions", HFM Magazine, 24 March 2025. https://www.hfmmagazine.com/understanding-and-resolving-sprinkler-obstructions
- S23. MeyerFire course summary fx108.72h (coverage area, NFPA 13-2022 9.5.2.1; via search summary). https://meyerfire.com/uploads/1/6/0/7/16072416/fx108.72h_-_summary.pdf ; US patents 5,366,022 and 6,976,543 (definitions of S and L).
- S24. Building Code Resource Library, Hazard classification (SlideShare deck, c. 2019). https://www.slideshare.net/slideshow/hazard-classification-195543772/195543772
- S25. MeyerFire course summary fx108.61a, occupancy hazard classification (NFPA 13-2022 4.3, A.4.3). https://meyerfire.com/uploads/1/6/0/7/16072416/fx108.61a_-_summary.docx.pdf
- OEP. OpenExamPrep NICET water-based study guide pages (protection area, positioning, obstructions, hazard classes, density/area). https://open-exam-prep.com/study-guides/nicet-water-based/ (weak; contradicted better sources on the beam rule beyond 3.5 ft and on the LH obstructed area; used only as corroboration).
- Koorsen blog, commodity and occupancy classifications (2018). https://blog.koorsen.com/understanding-commodity-and-occupancy-classifications-for-fire-sprinklers

**Products, hydraulics, response**
- S26. Tyco (Johnson Controls), TFP400, Series LFII Residential 4.9K Pendent Sprinkler TY2234, September 2019. https://www.qrfs.com/content/attachments/Datasheet_Tyco_T2234_2019_September.pdf
- S27. Tyco (Johnson Controls), TFP171, Series TY-FRB quick response standard coverage sprinklers, December 2022. https://24hr.supply/content/TFP171_12_2022.pdf
- S28. US DoD, UFC 3-600-01, Table 4-1 (via search summary of the tpub mirror). https://ufgscriteria.tpub.com/3_600_01/3_600_010096.htm ; WBDG copy: https://www.wbdg.org/FFC/DOD/UFC/ufc_3_600_01_2016_c6.pdf (not opened)
- S29. Meyer, J., "The two drivers for minimum sprinkler flow", MeyerFire, 25 March 2020. https://meyerfire.com/blog/the-two-drivers-for-minimum-sprinkler-flow
- S30. MeyerFire forum, "Why is there a range for fire sprinkler supply?" (NFPA 13-2016 Table 11.2.3.1.2, 11.2.3.1.3). https://meyerfire.com/daily/why-is-there-a-range-for-fire-sprinkler-supply
- S31. US Fire Administration, Coffee Break Training bulletins (archived): hose stream table (cb-2006-50), temperature ratings (cb_fp_2010_8), RTI (cb-2007-1), content via search snippets. https://webarchive.library.unt.edu/web/20121014055108/http://www.usfa.fema.gov/downloads/pdf/coffee-break/cb-2007-1.pdf
- S32. Building Code Resource Library, Sprinkler types and requirements (temperature table). https://www.buildingcoderesourcelibrary.com/Sprinkler-Types-and-Requirements.pdf ; Victaulic data sheet. https://cdn11.bigcommerce.com/s-h7xitcpvze/content/products/specifications/VICS271AJQ410.pdf
- S33. QRFS blog #1, standard response vs quick response. https://blog.qrfs.com/1-fire-sprinklers-standard-response-vs-quick-response/ ; QRFS #460. https://blog.qrfs.com/460-how-can-the-same-fire-sprinkler-be-quick-and-standard-response/
- S34. Kiwa, RTI testing of existing sprinklers (ISO 6182-1 classes). https://www.kiwa.com/nl/nl/markten/brandveiligheid-en-beveiliging/nieuws/testen-van-gevoeligheid-van-bestaande-sprinklerkoppen-rti-essentieel/
- S35. Heskestad, G., Bill, R. G. Jr. (1988). Quantification of thermal responsiveness of automatic sprinklers including conduction effects. *Fire Safety Journal* 14(1 to 2), 113 to 125. doi:10.1016/0379-7112(88)90049-5 (resolved at Crossref).
- S36. Arvidson, M. (2018). The response time of different sprinkler glass bulbs in a residential room fire scenario. *Fire Technology* 54(5), 1265 to 1282. doi:10.1007/s10694-018-0729-8 (resolved at Crossref).
- S37. NFPA 13D/13R summaries: kordfire, "NFPA 13D vs 13R". https://kordfire.com/nfpa-13d-vs-13r-residential-sprinkler-guide/ ; AmTrust, sprinkler systems comparison. https://AmTrustFinancial.com/getmedia/edfc63ea-c1fa-4cd1-bef8-4ea718a55246/AFS_SZ_Sprinkler_Systems_Comparison_MKT5348.pdf ; PHCP Pros, "NFPA 13D and 13R". https://www.phcppros.com/articles/8258-nfpa-13d-and-13r
- S45. US Department of Veterans Affairs, Fire Protection Design Manual, 9th ed., 1 November 2023 (sections 6.1, 6.2). https://www.wbdg.org/FFC/VA/VADEMAN/dmfpfire.pdf
- S57. Search summaries of NICET-style guides, an NFPA 13 vs IS comparison, and an NFPA 13 2007 figure (density/area points, hose allowance), with the MeyerFire note citing NFPA 13-2022 Table 19.2.3.1.1. Weak; corroboration only.

**Assembly points**
- S38. HM Government (DCLG) (2006). Fire safety risk assessment: offices and shops. https://www.gov.uk/government/publications/fire-safety-risk-assessment-offices-and-shops ; PDF read as text: https://assets.publishing.service.gov.uk/media/5a815c78ed915d74e33fdc7e/9449_Offices_and_Shops_v2.pdf
- S39. I Dig Hardware (Allegion), safe dispersal areas. https://idighardware.com/?p=22744 ; The Building Code Forum, "Exiting through an internal courtyard" (quotes IBC 1028.5 exception). https://www.thebuildingcodeforum.com/forum/threads/exiting-through-a-internal-courtyard.37100/post-293764
- S40. Indiana DHS, Fire Safety Evacuation Plans Review Sheet (2025). https://secure.in.gov/dhs/files/Fire-Safety-Evacuation-Plans-Review-Sheet-2025-10-07-1.pdf ; Goodyear AZ and other local IFC 404 guidance (via search summary).
- S41. OSHA, Evacuation Plans and Procedures eTool, minimum requirements; 29 CFR 1910.38. https://www.osha.gov/etools/evacuation-plans-procedures/eap/minimum-requirements ; https://www.govinfo.gov/content/pkg/CFR-2013-title29-vol5/pdf/CFR-2013-title29-vol5-sec1910-38.pdf
- S42. NIOSH fire fighter fatality investigation reports and the NIOSH 2014-120 workplace solutions on collapse zones (via search summary). https://stacks.cdc.gov/view/cdc/259970 ; https://www.cdc.gov/niosh/docs/wp-solutions/2014-120/pdfs/2014-120.pdf
- S43. UC Irvine Emergency Management, Highlighted changes to evacuation and assembly areas, revised October 2022 (read as text). https://em.uci.edu/_pdf/highlighted-changes-to-evacuation-and-assembly-areas.pdf
- S44. District of Columbia, School Emergency Response Plan and Management Guide, Form F, Assembly Area: Outdoors, October 2009 (read as text). https://esa.dc.gov/sites/default/files/dc/sites/esa/publication/attachments/form-f-assembly-area-outdoors-standard-evacuation.pdf
- S58. Victorian Department of Education, evacuation diagram checklist (AS 3745). https://www.education.vic.gov.au/Documents/Evacuation-Diagram-Checklist.docx (via search summary)
