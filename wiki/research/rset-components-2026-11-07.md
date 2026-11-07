# RSET components for the simulated-fire compliance check (2026-11-07)

Scope: values the app needs to compare L(v), the latest safe departure time at each point of a storey, with t_det + t_alarm + t_pre, and to report the margin and P(v) = F_pre(L(v) - t_det - t_alarm). Research only; no code and no change to the repository.

**As implemented (2026-11-07).** `src/core/rsetPresets.js` carries the rows of section 1 as presets (office, classroom, apartments outside and inside the dwelling of origin, hotel, assembly), with t_alarm 30 s, or 120 s for the hotel's staged alarm (SFEG 2025 4.2). `src/core/performanceCheck.js` applies the criteria (a) to (c) of section 1 as three choices (`RSET_CRITERIA`): the base case (ratio 2 and at least 30 s), a sensitivity case (1.2 and 30 s) and the 99th percentile alone, with tau(v) the walk with no fire; `formal/fireegress/FireEgress/Performance.lean` proves the ratio's equivalence with a margin of (f - 1) RSET. The second round of the same day added the five items of section 7.2 that were left:

- **Ceiling detectors (section 4.1).** `src/core/flamescapesRun.js` `ceilingDetectors` places up to 16 virtual smoke detectors under the ceiling round the fire, at C/VM2's 7.0 m or as far as the fire's ceiling space allows (a wall or a lintel ends it), read every frame on the CPU and the GPU; `src/core/forecast.js` `ceilingDetection` takes t_det as the end of the first frame in which one reaches OD 0.097 /m. Runs kept from before fall back to head height and say so.
- **A measured F_pre beside the design (sections 2.3, 3.6).** `MEASURED_PRE` holds the Lovreglio et al. (2019) Weibull clusters of section 3.6, each preset with a default cluster; the check judges every place under it too and draws both distributions, but the verdict stays on the design values.
- **No response (section 3.5).** Each preset carries the ABCB share (0 % for offices and assembly, 1 % on an apartment fire floor and in hotels); it is never counted as starting in time, so P(v) is at most 1 - q, and the percentiles are the responders'.
- **Queues (section 6, gap G4).** Occupants at the area per occupant (the code check's 9 m² for offices, IBC 2021 Table 1004.5 for the others) walk to the nearest exit with no fire and pass each door at 1.32 persons a second per metre of effective width (C/VM2 3.5.3), first come first served, by Lindley's recursion on time bins; door widths come from the IFC, and doors with no width take the stricter code minimum (0.813 m). The 99th-percentile starter's wait W enters RSET as in C/VM2 Eq. 3.1: a place needs L >= required(tau + W) + W.
- **Several scenarios (section 7.2, point 2).** `src/core/perfReport.js` gathers every check into one report, adds the plan's sensitivity cases (the exit nearest the fire blocked, from a second deadline map; the first detector failing, timed by the ceiling detector the smoke reaches last), can add two fires a storey for graph and two-zone fires, and saves the report as one HTML page with its inputs and sources.

`DoorQueue.lean` and `PreMovement.lean` prove the queue's capacity, conservation, clearance and first-come-first-served order, the queue-adjusted requirement, the Weibull's quantile, the no-response bound and the report's verdict. Still open: the M3 refusal, the C/VM2 cap of 50 persons a minute through a self-closing leaf, stair flow, speed classes and the ABCB disability increments (section 7.2, point 3), and a faster or smouldering fire as an automatic case.

Read status used throughout: **P** = read in the primary source in this session; **S** = read in a secondary source that reproduces or summarises the primary; **N** = not read (metadata or abstract only). "Derivation" marks numbers computed here from read values.

## 1. Proposed defaults per occupancy

All pre-movement times are counted from the general alarm (or the local alarm for the dwelling of origin). Lognormal parameters are in seconds: ln t ~ Normal(mu, sigma), fitted through the 1st and 99th percentiles (section 2.2). C/VM2 = NZ Verification Method C/VM2, 2nd edition, 2025, Table 3.4.1.1; Singapore's SFEG 2025 Table F-1 repeats the same numbers.

| Occupancy and scenario assumed | t_det | t_alarm | Default F_pre (1st, 99th percentile; lognormal) | Regulatory single values (C/VM2; SFEG 2025) | Cross-checks | Read status |
|---|---|---|---|---|---|---|
| **Office.** PD 7974-6 category A (awake, familiar), management M2, simple multi-storey B2, automatic detection with an immediate general alarm A1 | From the run: first time a ceiling detector reaches optical density 0.097 /m (C/VM2 Table 3.2.2.2). That equals soot 25.7 mg/m3 at 8.7 m2/g (derivation) | 30 s (C/VM2 3.3.1.1; ABCB FSVM data sheets 1.1.3) | 60 s and 180 s. mu = 4.644 (median 104 s), sigma = 0.236 | 30 s in the enclosure of origin; 60 s remote | ABCB Table 14 (XM2, XA1-2, XB1-2): 60 s for 10 %, 120 s for 80 %, 180 s for 10 %. ISO/TR 16738 example (A, M2, complex B3, staged A2): 1.5 and 3.5 min. Lovreglio et al. (2019) business cluster 1, Weibull (scale 139.3 s, shape 1.195): median 102 s, 99th percentile 500 s | PD 7974-6:2019 values S (CFPA-E 2023, Table 7). PD 7974-6:2004 S (CFPA-E 2009 image of Table C.1, 1 + 2 min). C/VM2, ABCB, Lovreglio P. ISO/TR 16738 example S (via the Italian fire code, which is P) |
| **Apartment building, fire floor, outside the dwelling of origin.** Asleep, familiar; minimal management XM2; building-wide alarm XA1; simple XB1 | From the run, at the common-area (corridor) detector. ABCB 1.4.4 notes that a building-wide alarm may be fed only by detectors in public spaces | 30 s | ABCB Table 16, fire floor: 150 s for 9 %, 300 s for 80 %, 600 s for 10 %, no response 1 %. As a lognormal: median 300 s (mu = 5.704), sigma = 0.541, with F capped at 0.99 (derivation: 10th and 90th percentiles at 150 s and 600 s) | 60 s in the dwelling of origin; 300 s remote (standard alarm) | PD 7974-6:2019 Ci (dwellings, individual occupancy) M2 B1 A1: 5 and 10 min, printed in italics (greater uncertainty), "for other units in a block assume one hour". ABCB other floors: 600, 1800 and 2400 s, with 4 % no response. Lovreglio residential cluster 1 (daytime drills, working alarm): median 64 s, 99th percentile 750 s; cluster 2 (fires, poor alarm): median 498 s, 99th percentile 3454 s | ABCB, C/VM2, Lovreglio P; PD S |
| **Apartment, dwelling of fire origin** (sub-case) | From the run, at the in-dwelling smoke alarm | 30 s (C/VM2 applies the same notification time) | PD 7974-6:2019 Ci M2 B1 A1: 300 s and 600 s. mu = 6.050 (median 424 s), sigma = 0.149 | 60 s | PD 7974-6:2004 Table C.1: 5 + 5 min, the same | PD S; C/VM2 P |
| **Hotel.** Ciii (asleep, unfamiliar), M2, B2, staged alarm A2 | From the run (corridor or room detector) | 30 s for A1. For A2, add the staff investigation delay (section 3.3) | ISO/TR 16738 example (Ciii, A2, B2, ordinary management): 20 and 40 min. mu = 7.437 (median 1697 s), sigma = 0.149 | 60 s in the room of origin; remote, 600 s with a standard alarm and 300 s with a voice alarm | PD 7974-6:2004 Table C.1 Ciii M2 B2 A1-A2: 20 + 20 min (the same). ABCB Table 18 (XM2): 600 s for 9 %, 900 s for 80 %, 2400 s for 10 %, no response 1 %. Lovreglio hotel cluster 1 (Dutch drills on one floor, alarm by phone): median 74 s, 99th percentile 212 s. Two US high-rise hotel fires without an alarm: means of 21 and 74 min | ISO S (Italian code Table M.3-1, P); PD 2004 S; C/VM2, ABCB, Lovreglio P |
| **University classroom or lecture hall.** Treated as category A, M2, B2, A1 (PD has no educational row) | From the run | 30 s | As the office: 60 s and 180 s; mu = 4.644, sigma = 0.236 | No classroom row. The office row gives 30 s and 60 s | BSI DD 240:1997 Table 21 (via CFPA-E 2009) groups schools, colleges and universities with offices: under 1 min with live directives (W1), 3 min with recorded messages (W2), over 4 min with a bell only (W3). Lovreglio lecture halls (cluster 5), Weibull (28.9 s, 1.127): median 21 s, 99th percentile 112 s. Libraries (cluster 3): median 69 s, 99th percentile 197 s | PD and DD 240 S; C/VM2, Lovreglio P |
| **Assembly** (cinema, theatre, hall), category B, M2, B1, A1 or A2, voice alarm assumed | From the run. In the space of origin, movement may start when the fire reaches 500 kW if that comes first (C/VM2 Table 3.4.1.1 note 5) | 30 s, plus the time to speak a voice message twice (Purser 2003) | PD 7974-6:2019 B M2 B1 A1-A2: 60 s and 240 s; mu = 4.787 (median 120 s), sigma = 0.298. Add 0.5 min for B2 or 1.0 min for B3 complexity | Focused activity (cinemas, theatres, stadiums): 0 s in the space of origin. Retail and restaurants: 60 s (origin) and 120 s (remote) with a standard alarm; 30 s and 60 s with a voice alarm | ABCB Table 15 (XM2, XB1): 60 s, 150 s and 240 s. Lovreglio assembly case studies (cinemas and theatres in Sweden and the UK, drills): case means from 0.30 to 0.93 min; UK theatre (cluster 4): median 53 s, 99th percentile 110 s | PD S; C/VM2, ABCB, Lovreglio P |

Acceptance checks for every row (section 4):

- **(a)** Margin at the 99th percentile, m99(v) = L(v) - t_det - t_alarm - t_pre,99 >= 0. This is the PD 7974-6 sparse-population case.
- **(b)** A ratio criterion, ASET_eq / RSET >= 2 for the base case and >= 1.2 for sensitivity cases (Singapore SFEG 2025 Table 10.4). Here ASET_eq(v) = L(v) + tau(v) and RSET(v) = t_det + t_alarm + t_pre,99 + tau(v), where tau(v) is the walking time (derivation). This is the same as the Italian default, a margin of at least 100 % of RSET.
- **(c)** An absolute margin of at least 30 s in all cases (Italian Code M.3.2.2).
- **Report P(v) in every case.** These are proposed app conventions; no Saudi numeric criterion was found (section 4.6).

## 2. How the check uses the values

### 2.1 Decomposition

Every source read uses the same decomposition:

- ISO/TR 16738:2009 Eq. (2) (P, preview): t_RSET = t_det + t_warn + (t_pre + t_trav).
- C/VM2 Equation 3.1 (P): RSET = (td + tn + tpre) + (ttrav or tflow).
- Italian Code M.3-4 (P): RSET = tdet + ta + tpre + ttra.
- ABCB D1.1 (P) and SFEG 2025 chapter 4 (P): the same terms.
- ISO/TR 16738 Eq. (1) (P) defines the margin as t_marg = t_ASET - t_RSET.

L(v) already subtracts the walking time along the best route. So for one occupant starting at v:

- margin at percentile q: m_q(v) = L(v) - t_det - t_alarm - t_pre,q;
- probability of getting out: P(v) = F_pre(L(v) - t_det - t_alarm), with P = 0 when the argument is 0 or less.

With FIFO links (as the methods note shows for Studio's monotone fires), the departure slack m_q(v) equals the smallest time margin at any node of the route. It is therefore the per-occupant ASET - RSET (derivation).

### 2.2 Fitting a lognormal from two percentiles (derivation)

With z = 2.3263 (the standard normal 99th percentile):

- mu = (ln t1 + ln t99) / 2
- sigma = (ln t99 - ln t1) / (2 z)
- P = Phi((ln x - mu) / sigma), where x = L(v) - t_det - t_alarm

| Scenario (source) | t1, t99 (s) | mu | median (s) | sigma |
|---|---|---|---|---|
| A M1 B1-B2 A1-A2 (PD 2019) | 30, 90 | 3.951 | 52 | 0.236 |
| A M2 B1-B2 A1-A2 (PD 2019) | 60, 180 | 4.644 | 104 | 0.236 |
| A M2 B3 A2 (ISO/TR 16738 via Italy) | 90, 210 | 4.923 | 137 | 0.182 |
| B M1 B1 (PD 2019) | 30, 150 | 4.206 | 67 | 0.346 |
| B M2 B1 (PD 2019) | 60, 240 | 4.787 | 120 | 0.298 |
| Ci M2 B1 A1 (PD 2019) | 300, 600 | 6.050 | 424 | 0.149 |
| Cii M2 B2 (PD 2019) | 900, 2400 | 7.293 | 1470 | 0.211 |
| Ciii M2 B2 A2 (ISO/TR 16738 via Italy) | 1200, 2400 | 7.437 | 1697 | 0.149 |

Values written ">15" or ">30" (management level M3) have no upper bound. PD and the Italian summary of the levels both say M3 is not suitable for an engineered design without other measures, so the app should refuse to compute P for M3 rather than invent a bound.

The ABCB three-point profiles are not percentiles. The fits in the table of section 1 put the "prompt" and "slow" times at the 10th and 90th percentiles, which is a modelling choice (derivation). The "no response" share caps F below 1.

### 2.3 Worked example and the distribution choice (derivation)

Office, with t_det = 60 s and t_alarm = 30 s, chosen for illustration:

| L(v) | x | P, PD lognormal (60 s, 180 s) | P, Lovreglio business cluster 1 Weibull |
|---|---|---|---|
| 180 s | 90 s | 0.271 | 0.448 |
| 250 s | 160 s | 0.966 | 0.693 |
| 300 s | 210 s | 0.999 | 0.805 |

The PD-derived lognormal is narrow: sigma between 0.15 and 0.35 for every row. Measured populations are far wider. Lovreglio's Weibull shapes run from about 0.7 to 1.8, and their 1st-to-99th percentile ratio is about 160 for business cluster 1, against 3 for the PD row. The PD values are design values agreed for engineering use, not a population fit.

The app should show P under both, or under a design and a data-based distribution, rather than one curve.

## 3. Pre-movement time (question 1)

### 3.1 Categories and modifiers

**Definitions (Purser 2003, IAFSS Fire Safety Science 7, Table 1 and text; P).**

- **A:** awake and familiar (office, industrial).
- **B1:** awake and unfamiliar, no focal point (shop, restaurant, circulation space).
- **B2:** awake and unfamiliar, with a focal point (cinema, theatre).
- **Ci:** asleep, familiar, individual occupancy (dwelling without 24-hour on-site management).
- **Cii:** managed occupancy (serviced flats, halls of residence).
- **Ciii:** asleep and unfamiliar (hotel, hostel).
- **D:** medical care.
- **E:** transport (station, airport).

**Alarm levels:**

- **A1:** automatic detection throughout, with an immediate general alarm.
- **A2:** automatic detection giving a pre-alarm to management. The general alarm is sounded manually, or automatically after a fixed delay if the pre-alarm is not cancelled.
- **A3:** local detection and alarm only, or none, with a manual general alarm.

Purser adds that with a voice alarm under A1 or A2, the time to speak the message twice is added to the alarm time.

**Complexity:**

- **B1:** a simple single-storey building with few enclosures.
- **B2:** a simple multi-enclosure, usually multi-storey, building.
- **B3:** a large complex building.

**Management:**

- **M1:** trained staff, wardens, drills, independent certification and audit.
- **M2:** as M1 but with fewer staff and no independent audit.
- **M3:** basic management only, not suitable for an engineered design without other measures.

**Second readings.** CFPA-E Guideline 19:2023 (S) gives the same lists for PD 7974-6:2019 (its Table 6 and Annex 4.3). Its example for B2 is a simple multi-storey office under 60 m storey height.

**Ambiguity of the B labels.** In the tables, B1 to B3 after M is building complexity, not occupancy category B1 or B2. The ABCB data sheets add an X prefix (XM, XA, XB) for this reason (P).

### 3.2 PD 7974-6 values (minutes)

Neither PD edition was read directly.

- **2004 edition.** Read as an image of Table C.1, "Suggested pre-times for different design behavioural scenario categories", reproduced in CFPA-E Guideline 19:2009 (S).
- **2019 edition.** Read as an image of CFPA-E Guideline 19:2023 Table 7, cited to PD 7974-6:2019 (S). The image ends after category Cii.
- **2004 footnote a.** It defines the total pre-movement time as the first-occupant value plus the distribution value. The "2004 sum" column below is that sum (derivation).
- **Italics.** These mark values with greater uncertainty (2004 footnote a).

| Scenario | 2004: first occupants | 2004: distribution | 2004 sum | 2019: 1st percentile | 2019: 99th percentile |
|---|---|---|---|---|---|
| A M1 B1-B2 A1-A2 | 0.5 | 1.0 | 1.5 | 0.5 | 1.5 |
| A M2 B1-B2 A1-A2 | 1 | 2 | 3 | 1 | 3 |
| A M3 B1-B2 A1-A3 | >15 | >15 | >30 | >15 | >30 |
| A: add for B3 | +0.5 | | | +0.5 | |
| B M1 B1 A1-A2 | 0.5 | 2 | 2.5 | 0.5 | 2.5 |
| B M2 B1 A1-A2 | 1.0 | 3 | 4.0 | 1.0 | 4.0 |
| B M3 B1 A1-A3 | >15 | >15 | >30 | >15 | >30 |
| B: add for B2 / B3 | +0.5 / +1.0 | | | +0.5 / +1.0 | |
| Ci M2 B1 A1 | 5 | 5 | 10 | 5 | 10 |
| Ci M3 B1 A3 | 10 | >20 | >30 | 10 | **>40** |
| Ci, other units in a block | 1 hour | | | 1 hour | |
| Cii M1 B2 A1-A2 | 10 | 20 | 30 | 10 | 30 |
| Cii M2 B2 A1-A2 | 15 | 25 | 40 | 15 | 40 |
| Cii M3 B2 A1-A3 | >20 | >20 | >40 | >20 | >40 |
| Ciii M1 B2 A1-A2 | 15 | 15 | 30 | not seen | not seen |
| Ciii M2 B2 A1-A2 | 20 | 20 | 40 | not seen | not seen |
| Ciii M3 B2 A1-A3 | >20 | >20 | >40 | not seen | not seen |
| Ciii: add for B3 | +1.0 | | | | |
| D awake (clinic) M1 / M2 / M3 B1 | 0.5 / 1.0 / >15 | 2 / 3 / >15 | 2.5 / 4 / >30 | not seen | not seen |
| D asleep (ward, nursing home) M1 / M2 / M3 B2 | 5 / 10 / >10 | 10 / 20 / >20 | 15 / 30 / >30 | not seen | not seen |
| E M1 / M2 / M3 B3 | 1.5 / 2.0 / >15 | 4 / 5 / >15 | 5.5 / 7 / >30 | not seen | not seen |

Notes on the table:

- The D asleep rows depend on enough staff being present to help disabled occupants (2004 footnote b).
- A voice alarm or PA is normally needed for M1 in categories B, Ciii and D, for M1 and M2 in E, and for M1 in A where unfamiliar visitors are likely.

**Finding.** Every 2019 value read equals the 2004 sum, except Ci M3, which is >40 against >30. The 2019 table therefore prints the 99th-percentile time itself, while 2004 printed the spread to be added.

The ISO/TR 16738 definition supports this reading. It splits pre-travel into the first occupants and the distribution "between the first and last occupants to move" (cl. 3.9, NOTE 2, P). The Italian code's ISO-derived examples are also cumulative (section 3.4). If the app ever loads a 2004-style table, it must add the two columns.

**Distribution shape.** CFPA-E 2023 (S) says the PD 7974-6:2019 pre-movement times follow a log-normal distribution, characterised by the 1st and 99th percentiles. Neither PD edition, Purser 2003, nor the ISO/TR 16738 preview was found stating the shape in the text read. Lovreglio et al. (2019, P) fitted gamma, lognormal, loglogistic and Weibull forms with nearly equal R² in every cluster, so the data do not single out the lognormal.

**Population use in PD 7974-6:2019** (via CFPA-E 2023, S). Two cases are evaluated, and the longer is used for design:

- a sparse case with one third of the design population: RSET = t_det + t_a + t_pre,99 + t_walk;
- the full design population: RSET = t_det + t_a + t_pre,1 + t_walk + t_flow.

In both cases the largest exit is discounted. The per-point P(v) check is the single-occupant form of the sparse case. Crowds need the flow term (repo gap G4).

### 3.3 ISO/TR 16738:2009

**What the preview gives (P).** The iTeh and SIS previews give the scope, definitions, Eq. (1) and (2), and the contents list:

- Annex A: detection and warning times;
- Annex D: design behavioural scenarios for default RSET variables;
- Annex E: pre-travel activity time distribution data and derivations;
- Annex G: travel speeds and flow rates.

It defines the margin of safety (cl. 3.6) and says the margin must reflect occupancy risk, fire scenarios and the uncertainty of the ASET and RSET predictions (cl. 5.6). It also says the warning time can range from about zero to several or many minutes for staged systems (cl. 5.7).

**What was not read.** The annex values. The forum and calculator sites that attribute a Table E.2 to ISO/TR 16738 or to PD 7974-6:2019 (A1 B1 M2: 1 and 3 min) were not used.

**Values available from other primary documents:**

- **Italian Codice di prevenzione incendi, Table M.3-1** (coordinated text in force from 1 January 2023, P). It reproduces three ISO/TR 16738 examples:
  - a medium-complexity hotel (Ciii; automatic detection with a general alarm after staff verification; multi-storey simple layout; ordinary management): 20 and 40 min;
  - a large production plant (A; the same alarm; multi-storey complex layout; ordinary management): 1 min 30 s and 3 min 30 s;
  - a nursing home (D asleep; the same alarm; multi-storey simple layout; ordinary management; enough staff): 5 and 10 min.
- **ABCB FSVM data sheets, section 1.4** (2020, CC BY 4.0, P). Its default pre-movement tables were "derived from the base scenarios from Appendix D of ISO/TR 16738:2009" and turned into prompt, medium, slow and no-response profiles.

### 3.4 Regulatory single values

**C/VM2, 2nd edition** (effective 28 July 2025; the 1st edition may be used for applications before 1 August 2026), Table 3.4.1.1 (P). All values in seconds, for the enclosure of origin and remote from it:

| Occupants | Alarm | Enclosure of origin | Remote |
|---|---|---|---|
| Awake, alert, familiar (offices, warehouses not open to the public) | any | 30 | 60 |
| Awake, alert, unfamiliar (retail, exhibitions, restaurants) | standard | 60 | 120 |
| Awake, alert, unfamiliar | voice | 30 | 60 |
| Sleeping, familiar (apartments) | standard | 60 | 300 |
| Sleeping, unfamiliar (hotels, motels) | standard | 60 | 600 |
| Sleeping, unfamiliar | voice | 60 | 300 |
| Awake under trained staff (dental office, clinic) | any | 60 | 120 |
| Early childhood centres | any | 60 for staff to respond, then 60 per child per staff member | 120 |
| Hospitals and rest homes | any | 60 for staff to respond, then 120 per patient per 2 staff | 1800 |
| Focused activity (cinemas, theatres, stadiums) | any | 0 | as other rows |

Notes and comment boxes:

- **Focused activity.** In the space of origin, movement starts at detection plus notification, or when the fire reaches 500 kW, whichever is first (note 5).
- **Implicit safety factor.** The comment says the design fire ignores the incipient phase, which gives an inherent safety factor on these times.
- **Cues.** Occupants with direct or indirect cues may take the enclosure-of-origin values. Indirect cues are not to be assumed in apartments, hotels or separate office tenancies.

**Confirmations of the C/VM2 values:**

- **Amendment 7 Table 3.3** (archived MBIE copy, P) gives the same values. The Errata 1 and Amendment 3 and 4 markers show the 30 s office value dates from 2012.
- **Singapore SFEG 2025 Table F-1** (P) reproduces them. Clause 4.3 lets the fire engineer use them with justification, and Annex F notes that they ignore the alarm, management and complexity modifiers.

### 3.5 ABCB FSVM design occupant profiles

ABCB FSVM Data Sheets, Handbook annex, 2020 (P). Times are in seconds, with the share of occupants in brackets.

| Table | Scenario | Prompt | Medium | Slow | No response |
|---|---|---|---|---|---|
| 14, A | XM1 XA1-2 XB1-2 | 30 (10 %) | 60 (80 %) | 90 (10 %) | 0 |
| 14, A | XM2 XA1-2 XB1-2 | 60 | 120 | 180 | 0 |
| 14, A | XM1 XB3 | 60 | 90 | 120 | 0 |
| 14, A | XM2 XB3 | 90 | 160 | 210 | 0 |
| 15, B | XM1 XB1 | 30 | 90 | 150 | 0 |
| 15, B | XM2 XB1 | 60 | 150 | 240 | 0 |
| 15, B | XM1 XB2 | 60 | 120 | 180 | 0 |
| 15, B | XM2 XB2 | 90 | 180 | 270 | 0 |
| 15, B | XM1 XB3 | 90 | 150 | 210 | 0 |
| 15, B | XM2 XB3 | 120 | 210 | 300 | 0 |
| 16, C1 apartments, fire floor | XM2 XA1 XB1 | 150 (9 %) | 300 (80 %) | 600 (10 %) | 1 % |
| 16, C1 apartments, fire floor | XM3 XA3 XB1 | 300 (8 %) | 600 | 1200 | 2 % |
| 16, C1 apartments, other floors | XM2 XA1 XB1 | 600 (6 %) | 1800 | 2400 | 4 % |
| 16, C1 apartments, other floors | XM3 XA3 XB1 | 1200 (2 %) | 2400 | 3600 | 8 % |
| 18, C2 hotels | XM1 XA1-2 XB2-3 | 300 (9 %) | 600 | 1800 | 1 % |
| 18, C2 hotels | XM2 XA1-2 XB2-3 | 600 (9 %) | 900 | 2400 | 1 % |
| 18, C2 hotels | XM3 XA1-3 XB2-3 | 900 (8 %) | 1200 | 3000 | 2 % |

Further ABCB guidance:

- **Robustness when management fails** (categories A and B): 10 % no response, and 15 to 30 min for the rest.
- **Disabilities** (Table 17, increases to pre-movement in apartments): 120 to 300 s for mobility, 120 s for vision, 60 to 120 s for hearing, 120 s for cognitive impairment.
- **Caption errors.** The captions of Tables 16 and 18 say "Category B1"; the section headings show they are C1 and C2.

### 3.6 Measured data: Lovreglio, Kuligowski, Gwynne and Boyce (2019)

The database holds 112 case studies (9 fires and 103 drills) from 16 countries, and extends Gwynne and Boyce's SFPE Handbook chapter. The author accepted manuscript (Ulster University repository) was read (P).

**Parameters.** The tables give a and b in seconds and in minutes. The Weibull and gamma columns are internally consistent. The loglogistic "a" is the log of the scale: its seconds and minutes values differ by ln 60 = 4.094 in every row checked (derivation).

**Lognormal columns are not usable.** They are inconsistent (for example 381.651 in seconds against 40.919 in minutes, and negative values such as -186.262). Crossref lists an erratum, doi:10.1016/j.firesaf.2019.102829 (2019), whose content was not read. The lognormal columns of the manuscript should not be used.

**Weibull quantiles (derivation).**

| Cluster | Scale (s) | Shape | Median | 99th percentile |
|---|---|---|---|---|
| Business 1 (13 case studies, 2597 points) | 139.3 | 1.195 | 102 s | 500 s |
| Residential 1 (drills, working alarm) | 102.5 | 0.767 | 64 s | 750 s |
| Residential 2 (fires and poorly performing alarms) | 724.6 | 0.978 | 498 s | 3454 s |
| Hotel 1 | 90.5 | 1.79 | 74 s | 212 s |
| Library (educational cluster 3) | 84.4 | 1.798 | 69 s | 197 s |
| Lecture hall (educational cluster 5) | 28.9 | 1.127 | 21 s | 112 s |
| Theatre (assembly cluster 4) | 61.0 | 2.579 | 53 s | 110 s |

The residential set excludes the sleep data in the SFPE chapter, so it is mostly awake occupants.

### 3.7 SFPE sources (not read)

The brief names a Purser chapter on pre-movement in the SFPE Handbook, 5th edition. Crossref shows that Purser's 5th-edition chapters are on combustion toxicity (Ch. 62) and on hazards to occupants (Ch. 63, with McAllister). The pre-evacuation data compilation is Gwynne and Boyce's "Engineering Data" (Ch. 64). The behaviour chapter is Kuligowski's "Human Behavior in Fire" (Ch. 58). Scenario selection is Nilsson and Fahy (Ch. 57), and the hydraulic model is Gwynne and Rosenbaum (Ch. 59).

None of these, nor the SFPE Guide to Human Behavior in Fire (2nd edition, 2019), was read; all are paywalled. Lovreglio et al. state that 68 % of the Gwynne and Boyce case studies are in their database (P), so section 3.6 is the best open proxy for Ch. 64.

### 3.8 Values confirmed by two or more sources

| Value | Source 1 | Source 2 | Agreement |
|---|---|---|---|
| Office A M2: about 1 min first, about 3 min 99th | PD 2019 via CFPA-E 2023 (S) | PD 2004 Table C.1 via CFPA-E 2009 (S), 1 + 2; ABCB Table 14 (P), 60, 120 and 180 s | yes |
| A M2 with B3: 1.5 and 3.5 min | Italian Code Table M.3-1 citing ISO/TR 16738 (P / S) | PD 2004 arithmetic, 1 + 0.5 and then + 2 (S) | yes |
| Hotel Ciii M2 B2: 20 and 40 min | Italian Code citing ISO/TR 16738 | PD 2004, 20 + 20 (S) | yes |
| C/VM2 table | C/VM2 2nd edition (P) | C/VM2 Amendment 7 (P); SFEG 2025 Table F-1 (P) | yes |
| Notification 30 s for an immediate alarm | C/VM2 3.3.1.1 (P) | ABCB 1.1.3 (P) | yes |
| Nursing home D asleep, ordinary management: Italian code 5 and 10 min | Italian Code (P) | PD 2004: M1 is 5 + 10 = 15, M2 is 10 + 20 = 30 (S) | **no** (section 7) |

## 4. Detection and alarm times (question 2)

### 4.1 Detector models and parameters

| Source | Smoke detector | Heat detector | Notes |
|---|---|---|---|
| C/VM2 2nd edition Table 3.2.2.2 (P) | Optical density at alarm 0.097 /m; radial distance 7.0 m; at least 25 mm below the ceiling. Beam detectors: optical density from the beam length and the obscuration setting, OD = -(1/L) log10((100 - OBS)/100) | RTI 30 (m s)^1/2; T_act 57 °C; radial distance 4.2 m | The values must be used whatever device is actually specified (3.2.2.2). Detection must come from a ceiling-jet correlation or from CFD (3.2.2.1). Sprinklers: standard response RTI 135, C 0.85, 68 °C, 3.25 m; quick response RTI 50, C 0.65, 68 °C, 3.25 m |
| FDS User's Guide (firemodels/fds master, commit d1a1280, 2026-11-05; P) | "CHAMBER OBSCURATION". Default Heskestad model with characteristic length 1.8 m. Default activation 3.24 %/m (1 %/ft). Cleary four-parameter model optional | "LINK TEMPERATURE", the sprinkler link model without water cooling. Default activation 74 °C. The PROP default RTI is 100 (m s)^1/2, the sprinkler default | Set heat-detector RTI and T_act explicitly |
| CFAST User's Guide (firemodels/cfast master, commit 611a810, 2026-11-04; P) | Default activation 24 %/m, from the upper-layer average, not the ceiling jet. Smoke alarms are often modelled as heat alarms 5 to 10 °C above ambient: about 5 °C for ionisation alarms (Davis and Notarianni), and 10 °C with RTI 5 in NUREG-1805 | Default 57 °C; default RTI 50 (m s)^1/2 | Davis and Notarianni and NUREG-1805 were seen only through the CFAST guide (S) |
| Singapore SFEG 2025 4.1 (P) | First ring of detectors. CFD, Alpert's ceiling-jet correlation or a zone model, using the RTI, T_act, C and optical density of the devices specified, at code-compliant spacing | as for smoke | Without automatic detection: detected when the smoke layer reaches 10 % of the room height |

**Conversion to the app's soot field** (derivation; mass extinction 8.7 m²/g as in the tenability note; K = 2.303 x OD; S = 3/K):

| Threshold | K (1/m) | Soot (mg/m³) | Obscuration (%/m) | Visibility S = 3/K |
|---|---|---|---|---|
| C/VM2, OD 0.097 /m | 0.223 | 25.7 | 20.0 | 13.4 m |
| FDS default, 3.24 %/m | 0.0329 | 3.8 | 3.24 | 91 m |
| CFAST default, 24 %/m | 0.274 | 31.5 | 24 | 11 m |

FDS adds a Heskestad lag inside the detector chamber (L/u, with L = 1.8 m); C/VM2 has no lag, but its threshold is about six times higher.

**Recommendation for t_det in the app:**

- **Detector cells.** Read the top gas cell under the ceiling at real detector positions if the IFC model has them. Otherwise use virtual detectors at the C/VM2 radial distance from the fire (7.0 m for smoke, 4.2 m for heat), or closer if the room is smaller (Table 3.2.2.2 note 1).
- **Smoke detectors.** Take t_det as the first time OD >= 0.097 /m in any detector cell.
- **Heat detectors.** Integrate the RTI link equation already specified in `wiki/research/sprinkler-physics-2026-11-07.md` with RTI 30 and T_act 57 °C.
- **Per scenario.** t_det is one number for each fire scenario, and every point of the storey uses it.

### 4.2 t_alarm by alarm type

| Alarm arrangement | t_alarm | Source and status |
|---|---|---|
| Automatic detection, immediate general alarm (A1) | 30 s | C/VM2 3.3.1.1 (P); ABCB 1.1.3, a "nominal allowance" for comparative studies (P) |
| A1, alternative | 0 s | Italian Code M.3.4.2: zero when detection directly triggers the general alarm (P) |
| A1 with a voice alarm | Add the time to speak the message twice | Purser 2003 (P). No typical message length found |
| Staged: pre-alarm to staff, investigation (A2) | 120 s | SFEG 2025 4.2: the time for management to confirm with the monitoring company (P) |
| A2, US practice | 15 s to acknowledge plus up to 180 s to investigate, so up to 195 s | NFPA 72 positive alarm sequence, from search summaries of a City of Bellevue interpretation (citing NFPA 72-2010) and other secondary pages (S); NFPA 72 not read |
| A2, general | Near zero to several or many minutes | ISO/TR 16738 cl. 5.7 (P). Gwynne, Purser and Boswell (2011) treat staff pre-warning delay as a separate RSET term (N) |
| Phased evacuation (ABCB XA2(b): alert tone, then evacuation tone cascading from the fire floor) | Add the cascade delay for floors away from the fire | ABCB 1.1.3 and 1.3.3.2.2 (P) |
| Manual only or local alarm (A3), no detection | Detection by occupants. C/VM2 3.2.3 (large spaces only: average ceiling height at least 5 m, awake and familiar, fewer than 50 people): td = 10 + 2.4L (L <= 1.4w) or 10 + w + 1.7L (1.4w < L <= 4w) for storage up to 5 m, and 25 + 1.7L or 25 + w + L above 5 m, with no further pre-travel. PD M3 rows give pre-movement over 15 min | C/VM2 (P); SFEG Annex E repeats the formula (P); PD (S) |

## 5. Acceptance criteria and safety margins (question 3)

| Document | Criterion | Tenability limits | Margin | Status |
|---|---|---|---|---|
| NZ Building Code C4.3 and C4.4; C/VM2 2nd edition 2.1 | ASET > RSET | FED_CO 0.3; FED_thermal 0.3; visibility 10 m, or 5 m in rooms under 100 m². Measured 2.0 m above the floor (C/VM2 2.1.6.2). FED_CO includes CO, CO2 and O2; FED_thermal includes radiant and convective heat (2.1.6.3). C4.3(b) and (c) are waived where a sprinklered firecell cannot expose more than 1000 occupants (C4.4) | No numeric margin; the implicit factor from ignoring the incipient phase. Robustness (4.10): each key system failed in turn, with only FED_CO assessed; in sprinklered sleeping buildings, stairs keep at least 5.0 m visibility for the RSET period | P (C/VM2, and NZBC Clauses C1 to C6) |
| Italy, Codice M.3 (coordinated text, 2023) | ASET > RSET; t_marg = ASET - RSET | Visibility 10 m (5 m in rooms under 100 m²) at 1.80 m; FED and FEC 0.1, so about 1.1 % of occupants are incapacitated at the threshold; 60 °C; 2.5 kW/m². Simplified method: smoke layer at least 2 m above the floor, layer temperature at most 200 °C | t_marg >= 100 % of RSET unless the input data are specifically evaluated; then >= 10 % of RSET, which should rest on literature or consolidated standards; **always >= 30 s** | P |
| Singapore SFEG 2025 | ASET/RSET with safety factor | Visibility over 10 m; temperature under 60 °C, or under 200 °C in a smoke layer above 2.5 m; FED for toxic gases at most 0.3 where allowed | **Base case ASET/RSET >= 2; sensitivity cases >= 1.2** (Table 10.4). Mandatory sensitivity tests: buoyancy, wind, fan failure, faster growth, a blocked exit, failure of the first detector, smouldering fire. Simulations run to at least 1200 s or until the criteria are shown. Notification time 120 s | P |
| ABCB FSVM (Australia) | Comparison with a reference building that meets the deemed-to-satisfy provisions | Agreed in the performance-based design brief | An "appropriate safety margin" for an absolute approach; no number | P |
| BSI PD 7974-6:2019 and BS 7974 | Margin "appropriate" to scenario risk and uncertainty. PD 7974-6 covers parameter uncertainty; BS 7974 covers the overall margin | PD 7974-6:2019 adds full FED expressions | No number found | P for the BSI product page only; standard N |
| ISO/TR 16738:2009 | t_marg = t_ASET - t_RSET | n/a | The margin must reflect occupancy risk, fire scenarios and prediction uncertainty (cl. 3.6, 5.6) | P (preview) |
| SFPE Engineering Guide to Performance-Based Fire Protection (2nd edition, 2007); Hurley and Rosenbaum, SFPE Handbook 5th edition Ch. 37 | not read | not read | not read | N |
| Saudi Arabia | SBC 801-2018 Section 104.9 allows alternative methods approved by the fire code official (as stated in the brief). SBC 201 follows the IBC, where the matching clause is 104.11 | none found | none found | N (see 5.1) |

**Distributional criteria.** Two exist in the sources:

- the PD 7974-6 percentile cases (the 99th percentile with walking time, or the 1st percentile with flow);
- the "no response" shares in the ABCB profiles, to be carried in the model rather than designed away.

No source read sets a required probability of escape, such as P >= 0.99. Any such threshold would be an app convention.

**Equivalence of margin and ratio** (derivation). ASET - RSET >= 1.0 x RSET is the same as ASET/RSET >= 2. The Italian default and the Singapore base case therefore coincide; the Italian 10 % option corresponds to a ratio of 1.1. With Studio's quantities, ASET_eq(v) = L(v) + tau(v) and RSET_q(v) = t_det + t_alarm + t_pre,q + tau(v).

### 5.1 Gulf and Saudi guidance

- **SBC texts.** No openly readable Saudi performance-based design guide or numeric RSET criterion was found. The repository's note `tall-office-sbc-egress-2026-11-05.md` records that the SBC 201 and 801 texts are sold or shown only in a protected viewer. `fire-suppression-sprinklers-2026-11-07.md` mentions a licensed SBC 801-CR-2018 copy kept locally, which was not found in this session, so Section 104.9 was not read.
- **Qatar.** A search summary says the Qatar Civil Defence framework points performance-based submissions to the International Fire Engineering Guidelines and requires a fire engineer (S, from a Scribd copy; not read).
- **UAE.** The UAE Fire and Life Safety Code was not read.
- **Defensible route.** Under Section 104.9, propose a recognised method (PD 7974-6 and ISO/TR 16738, C/VM2 or SFEG) in a fire engineering brief agreed with Saudi Civil Defense. Show the strictest common criteria: a ratio of 2 in the base case, at least 30 s absolute, and a 99th-percentile margin of at least 0.

## 6. Walking speeds and door flow (question 4)

| Quantity | Value | Source |
|---|---|---|
| Unimpeded horizontal speed | 1.2 m/s cap; S = k - akD with k = 1.4 and a = 0.266, used when D > 0.54 persons/m² | C/VM2 3.5.2.2 (P); ABCB 1.5.2 (P); SFEG 4.4.1 (P) |
| Stairs (speed along the line of travel; riser/tread in mm, k) | 191/254 (k 1.00): 0.85 m/s; 178/279 (1.08): 0.95; 165/305 (1.16): 1.00; 165/330 (1.23): 1.05 | C/VM2 Table 3.5.2.4 (P); the same k values in SFEG Table 4.1 "from PD 7974-6" (P) and in the repo methods note 6.1 (SFPE hydraulic model) |
| Occupant classes | Mean speeds: adult 1.25, male 1.35, female 1.15, child 0.90, elderly 0.80 m/s | FDS+Evac source, as read in `open-source-evacuation-study-2026-10-28.md` [F10] |
| People with disabilities | Boyce, Shields and Silcock (1999) measured horizontal and ramp speeds by aid type; values not read | N |
| Door flow | Fc = (1 - aD) k D We with D = 1.9 persons/m² at doors. At most 50 persons per minute per self-closing leaf, which corresponds to a 0.95 m door with 0.65 m effective width; no cap without a closer. Boundary layer 0.15 m per side for doors, 0.20 m for corridors | C/VM2 3.5.3 (P) |
| Maximum specific flow | k/(4a), about 1.32 persons/s per metre of effective width on the level (about 1.3, as used in gap G4) | Repo methods note 6.1 (derivation from the SFPE hydraulic constants) |
| Smoke | ABCB notes walking speeds of 0.3 m/s or less at the visibility limit in irritant smoke (data sheet section on smoke) | P, brief look only |

## 7. Criticism and what a defensible report should show (question 5)

### 7.1 Babrauskas, Fleming and Russell (2010)

Fire and Materials 34(7), 341 to 355. Only the abstract was read (OpenAlex); the full text was not. The authors argue that RSET/ASET is intrinsically flawed:

- it ignores the wide variation in occupants' capabilities and condition;
- it assumes that, after a brief assessment, occupants go "robotically" to the best exit;
- this does not match fatal and serious-injury fires, especially residential ones;
- it ignores the variation in fire scenarios, which can change both ASET and RSET drastically for the same building;
- used for design, it permits preventable deaths.

**Related work, not read beyond the metadata:**

- Chow (2012), a letter commenting on the paper.
- Poon (2014), Procedia Engineering. The abstract was read: it proposes a space-utility-over-time assessment for transient 3D simulation results.
- Schröder, Arnold and Seyfried (2020), "A map representation of the ASET-RSET concept", Fire Safety Journal 115, 103154. The L(v) map is such a representation, and the paper should be read before publishing.
- Gwynne, Purser and Boswell (2011), on staff pre-warning delay.

### 7.2 What a defensible report from the app should show

These points answer the critique and follow the sources read.

1. **Distributions, not points.** Report P(v) under at least two F_pre:
   - a design distribution (PD or ISO/TR 16738 percentiles, or ABCB profiles);
   - a data-based one (Lovreglio Weibull for the occupancy cluster).

   Report the "no response" share as a separate mass.
2. **Several fire scenarios.** Cover several fire locations, including one near an exit. Also cover:
   - a faster growth rate (SFEG test 4);
   - failure of the first detector (SFEG test 6);
   - a blocked exit (SFEG test 5; PD discounts the largest exit);
   - each key system failed in turn, judged on FED_CO only (C/VM2 4.10);
   - a smouldering fire for sleeping uses.

   If scenario weights are given, report P(v) per scenario and a weighted P.
3. **Occupant classes.** Repeat L(v) for the speed classes (adult, elderly, impaired, with stairs blocked for wheelchair users). Use the ABCB disability increments for t_pre.
4. **Show the per-scenario inputs.** t_det per scenario (with the detector model and threshold), t_alarm (alarm type and any staged delay), and the scenario category with its M, B and A levels.
5. **Several margin readings.** Report margins at the 99th percentile, as a ratio (2 for the base case, 1.2 for sensitivity cases), and as an absolute (at least 30 s), each labelled with the document it comes from.
6. **State the tenability criteria.** FED 0.3 (NZ, and the app's default) or 0.1 (Italy); visibility 10 m or 5 m; head height 1.8 m (app and Italy) against 2.0 m (NZ). Also state what the solver cannot give: radiant heat at nodes, irritant gases and HCN, per `tenability-index-2026-11-06.md`.
7. **Model uncertainty.** The Flamescapes solver is not validated for engineering use. Results rank places and times under the model; they are not a compliance demonstration.
8. **Behaviour the router does not model.** Route choice with perfect foresight (gap G7), no queues (G4) and no smoke slowing (G2). Say so beside every P(v).

## 8. What I am least sure of

1. **PD 7974-6:2019 values beyond category Cii.** These were not seen. The CFPA-E 2023 image stops at Cii, so the Ciii, D and E rows for 2019 are inferred by adding the 2004 columns. That rule held for every 2019 cell read except Ci M3 (>40 against >30).
2. **The nursing-home example.** The Italian code's ISO/TR 16738 example gives 5 and 10 min. This matches the raw 2004 M1 columns, not their sum (15). The ISO D values may differ from PD, or the example may not be cumulative.
3. **That PD 7974-6:2019 prescribes a log-normal fitted through its 1st and 99th percentiles.** This rests on one secondary source (CFPA-E 2023). The two-percentile fit is my derivation, and the resulting sigma (0.15 to 0.35) is much narrower than any measured population. P(v) computed from it will look more certain than the data justify.
4. **Lovreglio et al. (2019) lognormal parameters.** They are inconsistent in the accepted manuscript. The erratum was not read; only the Weibull and gamma columns are used.
5. **Saudi and Gulf criteria.** Not found. The proposed acceptance rule is an app convention built from Italian, Singapore, NZ and UK sources, not a Saudi requirement.
6. **NFPA 72 positive alarm sequence (15 s plus 180 s).** Taken from secondary summaries; NFPA 72 was not read.
7. **ABCB "prompt" and "slow" times as 10th and 90th percentiles.** This is my modelling choice; the ABCB tables give three-point profiles, not percentiles.

## 9. Sources and checks

### 9.1 Documents read

| Source | What was used | Status |
|---|---|---|
| MBIE, C/VM2 Verification Method: Framework for fire safety design, 2nd edition, effective 28 July 2025 (Wayback copy of the building.govt.nz PDF, 13 February 2026; the live site blocks scripts) | 2.1.1, 2.1.6, 2.3.2.1, 3.1 to 3.5, 4.10, Tables 3.2.2.2, 3.4.1.1, 3.5.2.4 and 3.5.3.2 | P |
| MBIE, C/VM2 Amendment 7 (2023; Wayback copy) | Table 3.3; notification 30 s | P |
| NZ Building Code Clauses C1 to C6 (MBIE PDF, Wayback copy) | C4.3, C4.4 | P |
| Purser, D. A. (2003). ASET and RSET: addressing some issues in relation to occupant behaviour and tenability. Fire Safety Science 7, 91 to 102. doi:10.3801/IAFSS.FSS.7-91 (IAFSS open PDF) | Categories, A, B and M levels, voice-alarm rule | P |
| CFPA-E Guideline No 19:2023 F, Fire safety engineering concerning evacuation from buildings | Table 6 and Table 7 image (PD 7974-6:2019); two-case RSET | S |
| CFPA-E Guideline No 19:2009 | Image of PD 7974-6:2004 Table C.1; DD 240 Table 21 adaptation | S |
| ISO/TR 16738:2009 previews (iTeh, SIS) | Clauses 3, 5, 6, 7; contents | P (front matter only) |
| Ministero dell'Interno, Codice di prevenzione incendi, testo coordinato (in force 1 January 2023), Chapter M.3 (University of Parma copy) | M.3.2.2 margins, Table M.3-1, M.3.5 thresholds | P |
| ABCB, Fire Safety Verification Method Data Sheets, Handbook annex (2020, CC BY 4.0) | 1.1.2, 1.1.3, 1.3.3, 1.4 (Tables 14 to 18), 1.5.2 | P |
| SCDF, Singapore Fire Safety Engineering Guidelines 2025 | Ch. 2 sensitivity tests, Ch. 3 and 4, Table 10.4, Annex F Table F-1 | P |
| Lovreglio, Kuligowski, Gwynne, Boyce (2019), author accepted manuscript (Ulster University) | Tables 1 to 13 (parameters) | P (manuscript, not the version of record) |
| FDS User's Guide, LaTeX source, firemodels/fds (commit d1a1280, 2026-11-05) | Heat and smoke detector sections; PROP defaults | P |
| CFAST User's Guide, Input_File_Chapter.tex, firemodels/cfast (commit 611a810, 2026-11-04) | Detector defaults and the smoke-alarm surrogate | P |
| BSI Knowledge product page for PD 7974-6:2019 | Publication date 31 March 2019, current; first revision since 2004; statement on margins | P (page only) |
| OpenAlex abstract of Babrauskas et al. (2010) | Abstract | P (abstract only) |

### 9.2 DOIs checked at Crossref (api.crossref.org/works)

| DOI | Title | Authors | Year |
|---|---|---|---|
| 10.1002/fam.1025 | RSET/ASET, a flawed concept for fire safety assessment | V. Babrauskas, J. M. Fleming, B. D. Russell | 2010 |
| 10.1016/S0925-7535(00)00066-7 | Quantification of behaviour for engineering design standards and escape time calculations | D. A. Purser, M. Bensilum | 2001 |
| 10.1007/978-3-319-20708-7_2 | Developments in Tenability and Escape Time Assessment for Evacuation Modelling Simulations | D. A. Purser | 2016 |
| 10.3403/30359669 | Application of fire safety engineering principles to the design of buildings (BSI record; no authors or year in Crossref) | BSI | n/a |
| 10.3801/IAFSS.FSS.7-91 | ASET And RSET: Addressing Some Issues In Relation To Occupant Behaviour And Tenability | D. Purser | 2003 |
| 10.1016/j.firesaf.2018.12.009 | A pre-evacuation database for use in egress simulations | R. Lovreglio, E. Kuligowski, S. Gwynne, K. Boyce | 2019 |
| 10.1016/j.firesaf.2019.102829 | Corrigendum to "A pre-evacuation database for use in egress simulations" | the same authors | 2019 |
| 10.1007/978-1-4939-2565-0_57 | Selecting Scenarios for Deterministic Fire Safety Engineering Analysis: Life Safety for Occupants | D. Nilsson, R. Fahy | 2016 |
| 10.1007/978-1-4939-2565-0_58 | Human Behavior in Fire | E. D. Kuligowski | 2016 |
| 10.1007/978-1-4939-2565-0_59 | Employing the Hydraulic Model in Assessing Emergency Movement | S. M. V. Gwynne, E. R. Rosenbaum | 2016 |
| 10.1007/978-1-4939-2565-0_63 | Assessment of Hazards to Occupants from Smoke, Toxic Gases, and Heat | D. A. Purser, J. L. McAllister | 2016 |
| 10.1007/978-1-4939-2565-0_64 | Engineering Data | S. M. V. Gwynne, K. E. Boyce | 2016 |
| 10.1007/978-1-4939-2565-0_37 | Performance-Based Design | M. J. Hurley, E. R. Rosenbaum | 2016 |
| 10.1007/978-3-319-94697-9 | SFPE Guide to Human Behavior in Fire | Society of Fire Protection Engineers | 2019 |
| 10.1023/A:1015339216366 | Toward the Characterization of Building Occupancies for Fire Safety Engineering: Capabilities of Disabled People Moving Horizontally and on an Incline | K. E. Boyce, T. J. Shields, G. W. H. Silcock | 1999 |
| 10.1002/fam.2134 | Letter to the Editor: Comment on "RSET/ASET, a flawed concept for fire safety assessment" | W. K. Chow | 2012 |
| 10.1016/j.firesaf.2020.103154 | A map representation of the ASET-RSET concept | B. Schröder, L. Arnold, A. Seyfried | 2020 |
| 10.1007/978-1-4419-9725-8_22 | Pre-Warning Staff Delay: A Forgotten Component in ASET/RSET Calculations | S. M. V. Gwynne, D. Purser, D. L. Boswell | 2011 |
| 10.1016/j.proeng.2014.04.025 | A Dynamic Approach to ASET/RSET Assessment in Performance based Design | S. L. Poon | 2014 |

### 9.3 Identifiers without a DOI

- ISO/TR 16738:2009 (ISO catalogue 42887);
- PD 7974-6:2004;
- BSI DD 240:1997;
- NFPA 72;
- SFPE Engineering Guide to Performance-Based Fire Protection (2007);
- SBC 801-2018 and SBC 201;
- C/VM2, the Italian code, the ABCB data sheets and SFEG 2025 (government documents, cited by URL in section 9.1).

### 9.4 Not used

Copies of PD 7974-6 and ISO/TR 16738 on file-sharing sites (pdfcoffee, kupdf, Scribd and similar) were found in searches and deliberately not opened. Forum and calculator pages were not used for values.
