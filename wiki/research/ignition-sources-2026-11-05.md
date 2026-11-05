# Where fires start: ignition sources, design fires and fire classes for choosing a fire origin in Topologic Studio Next

Date: 2026-11-05
Status: literature study and integration plan. No application code was changed.
Companion catalogue: [ignition-sources-2026-11-05.json](ignition-sources-2026-11-05.json) (21 ignition scenarios, 26 measured fuel items, 6 fire classes, occupancy design fires, ranking rule, 76 sources). Keys in square brackets below are the catalogue's source keys.
Related: [zone-model-spec-2026-10-28.md](zone-model-spec-2026-10-28.md) (the two-zone model's inputs), [fire-equations-verified-2026-10-27.json](fire-equations-verified-2026-10-27.json) (FD-01 to FD-05 and FD-17), [../../formal/fireegress/README.md](../../formal/fireegress/README.md) (what is already proved about the zone model).

Conventions. Evidence types are labelled as peer-reviewed, standard, official statistics (government), industry statistics (NFPA, which builds on the US federal NFIRS data plus its own survey), measurement (a laboratory report or measurement database) or news. "Derived" marks a number computed in this study from published values, with the computation shown. "Judgement" marks a modelling choice that no source makes. Every DOI cited was resolved against Crossref on 2026-11-05 and its title compared (one DOI, for the NIST Fire Calorimetry Database, is registered with DataCite and was resolved there). Figures read by the delegated search agents rather than by the author are marked "(agent)" here and in the catalogue's `read` fields; the author re-read the NFPA and England headline figures, all calorimetry tables and all abstracts quoted here.

---

## 0. Summary

Today the user picks the fire origin by clicking any node. The statistics say that the origin is not arbitrary: in US homes from 2019 to 2023, cooking caused 49 % of reported fires but 16 % of deaths, while smoking materials caused 5 % of fires and 23 % of deaths [nfpa2025homes]; in England in the year ending March 2026, cooking appliances started 42 % of accidental dwelling fires and 11 % of the deaths, smoking materials 7.6 % and 27 % [mhclg2026detailed]. An origin chooser should therefore rank candidate items by frequency times consequence, which reduces to their share of deaths, and should give each a design fire from calorimetry of the item that actually burns.

Table 0 is the result. "Share" is of US home fires and deaths, 2019 to 2023, unless marked UK (England, accidental dwelling fires, year ending March 2026). The design fire is the pair Studio's two-zone model takes (growth class, peak heat release rate); the model caps the peak at the room's ventilation limit.

| Scenario | Fires | Deaths | First fuel | Room of origin | Design fire | IFC entity (IFC4; IFC2x3) | NFPA class | Sources |
|---|---|---|---|---|---|---|---|---|
| Cooktop or oven, unattended cooking | 49 % (UK 42 %) | 16 % (UK 11 %) | Cooking oil or fat, food | Kitchen | ultra-fast, 7,585 kW (room) | IfcElectricAppliance ELECTRICCOOKER; IfcFlowTerminal typed ELECTRICCOOKER; IfcBurner; proxies by name | K | nfpa2025homes, mhclg2026fire0602, hamins2018, nistfcd |
| Smoking materials on a sofa or armchair | 0.3 % (derived) | 6.0 % (derived); UK all smoking 27 % | Upholstered furniture | Living room | fast, 3,120 kW | IfcFurniture SOFA, CHAIR; IfcFurnishingElement by name | A | nfpa2025smoking, babrauskas1982nbsir |
| Smoking materials on a bed | 0.2 % (derived) | 4.1 % (derived) | Mattress, bedding | Bedroom | slow, 1,700 kW | IfcFurniture BED; IfcFurnishingElement by name | A | nfpa2025smoking, lawson1984nbsir |
| Electrical distribution, wiring, boards, outlets | 9 % (UK 14 %) | 16 % (UK 2 %) | Wire insulation, structure | Bedroom, concealed spaces | medium, 250 kW/m2 x floor area | IfcElectricDistributionBoard; IfcElectricDistributionPoint; IfcOutlet | C (fuel A) | nfpa2025homes, en1991-1-2 |
| Intentional, accelerant at an entrance or stair | 7 % | 5 % | Gasoline, then contents | Entrance, corridor, stair | ultra-fast, 2,180 kW (1 m2 pool, derived) | none (anchor at exit doors) | B | nfpa2025homes, babrauskas1983pool |
| Heating stove, fireplace | 9 % | 6 % | Furniture near the appliance | Living room | fast, 2,000 kW | IfcChimney; IfcSpaceHeater (EnergySource WOOD); proxies | A | nfpa2025homestables |
| Space heater too close to combustibles | 1 % (UK 3 %) | 6 % (UK 4 %) | Armchair, bedding, clothing | Living room, bedroom | fast, 2,000 kW | IfcElectricAppliance FREESTANDINGELECTRICHEATER; IfcSpaceHeater CONVECTOR | C (fuel A) | nfpa2025homestables, babrauskas1982nbsir |
| Tumble dryer | 3.7 % (derived, 2010 to 2014; UK 2.4 %) | 1 % (with washers; UK 1.2 %) | Lint, clothing | Laundry | medium (assumed), 333 kW (proxy) | IfcElectricAppliance TUMBLEDRYER | C (fuel A) | nfpa2017dryers, hietaniemi2001vtt |
| Candle | 2 % (UK 3.5 %) | 3 % (UK 6 %) | Bedding, curtains, furniture | Bedroom | slow, 1,700 kW | none (anchor at beds, sofas) | A | nfpa2025homes, lawson1984nbsir |
| E-bike or e-scooter battery | UK 3.7 % (batteries and generators) | UK 0.6 % (5.3 % of casualties) | Cells, then vehicle and furniture | Hallway, living area | ultra-fast, 3,120 kW (composite) | proxies by name; IfcElectricFlowStorageDevice | C (fuel B), ISO L | mhclg2026fire0602, fleischmann2025 |
| Extension lead or cord | 1 % | 3 % | Adjacent furniture or bedding | Living room, bedroom | fast, 2,000 kW | IfcOutlet POWEROUTLET | C (fuel A) | nfpa2025homes |
| TV and electronics | 0.7 % | 1.8 % | Plastic enclosure | Living room, bedroom | fast, 241 kW | IfcAudioVisualAppliance DISPLAY; IfcFlowTerminal typed TV | C (fuel A) | nfpa2025homestables, hietaniemi2001vtt |
| Refrigerator or freezer | 0.4 % (UK 0.6 %) | 1.0 % (UK 0.6 %) | Polyurethane insulation, liners | Kitchen | slow, 1,970 kW | IfcElectricAppliance REFRIGERATOR, FRIDGE_FREEZER, FREEZER | C (fuel A) | nfpa2025homestables, hietaniemi2001vtt |
| Dishwasher | 0.2 % (UK 0.5 %) | 0.2 % (UK 0) | Tub, seals, wiring | Kitchen | medium (assumed), 411 kW | IfcElectricAppliance DISHWASHER | C (fuel A) | nfpa2025homestables, hietaniemi2001vtt |
| Washing machine | 0.3 % (derived; UK 1.7 %) | n/a (UK 0) | Wire insulation, housing | Laundry, bathroom | medium (assumed), 333 kW | IfcElectricAppliance WASHINGMACHINE | C (fuel A) | nfpa2017dryers, hietaniemi2001vtt |
| Phone, power bank or charger on a bed | 0.3 % (UK 0.6 %) | 0 | Bedding | Bedroom | slow, 1,700 kW | none (anchor at beds) | C (fuel A), ISO L | nfpa2025homestables, lawson1984nbsir |
| Office: waste bin or paper at a workstation | non-res. UK: paper first ignited 8 % | n/a | Paper, then workstation | Office, classroom | medium, 3,500 kW | IfcFurniture DESK; IfcSystemFurnitureElement | A | mhclg2026fire0603, madrzykowski2004sp1021 |
| Office or institutional electrical equipment | US medical, non-confined: appliances 21 % | n/a | Plastics, then workstation | Office, copy room, ward | medium, 3,500 kW | IfcElectricAppliance PHOTOCOPIER, VENDINGMACHINE; IfcCommunicationsAppliance | C (fuel A) | usfa2018medical |
| Dry Christmas tree (seasonal) | not gathered | not gathered | Dry needles | Living room, lobby | ultra-fast, 7,362 kW | none (by name) | A | nistfcd |
| Boiler, burner or generator | not separated | not separated | Gas or oil | Plant room | medium, 250 kW/m2 (default) | IfcBoiler, IfcBurner; IfcGasTerminalType | B | en1991-1-2 |
| Battery room or UPS | non-res. UK: batteries and generators 5.9 % | 15 % of casualties | Electrolyte | Battery or electrical room | medium, 250 kW/m2 (default) | IfcElectricFlowStorageDevice BATTERY, UPS | C (fuel B), ISO L | mhclg2026fire0602, larsson2017 |

Five findings matter most for the implementation.

1. Ranking by deaths differs from ranking by fires. Per 1,000 US home fires (2019 to 2023), smoking-material fires caused 39.5 deaths, electrical distribution 13.6, candles 12.0, heating 6.6 and cooking 2.7 (derived from [nfpa2025homes]); England gives 23.6 for smoking against 1.8 for cooking (derived from FIRE0602).
2. Most first items that kill grow on the fast or ultra-fast class: foam sofas and armchairs about 0.025 to 0.055 kW s^-2, thin plywood wardrobes 0.38, a dry Christmas tree 7.6 (derived from [babrauskas1982nbsir], [lawson1984nbsir], [nistfcd]). Studio's default (medium, 5,000 kW) is neither conservative nor typical for them.
3. IFC models rarely say what an appliance is. In 45 test models no element is an IfcElectricAppliance, two IFC2x3 IfcElectricApplianceType records are lift cars, and real appliances are proxies or furniture named "Miele Induction Cooktop" or "M_Refrigerator". Name matching is the primary route.
4. The NFPA 10 class of the item at the origin (A, B, C, D, K) fixes fuel properties, smoke and CO yields and the safe extinguishers; lithium-ion packs, now ISO 3941:2026 Class L, have growth data but no agent guidance in the sources read.
5. `snapToGraph` (App.jsx) snaps to the nearest node in 3D, which can put an appliance's fire on another storey or in the next room. Placement must use rooms.js's room assignment so that `fireRoomOf` finds the item's own room.

---

## 1. Method and evidence base

Sources searched: peer-reviewed epidemiology of fire causes (Fire Safety Journal, Fire Technology, Journal of Fire Sciences, Journal of Safety Research, burns journals); NFPA Research reports; US Fire Administration NFIRS reports; England's tables FIRE0602 and FIRE0603; lithium-ion statistics from the London Fire Brigade, the UK product-safety regulator and the New York Fire Department; the NBS furniture-calorimeter reports, the VTT appliance study, NIST's cooktop, workstation and Fire Calorimetry Database data; NFPA 10, ISO 3941 and EN 2 as standards. Two search agents covered statistics and fire classes, a third the IFC enumerations; the author read the calorimetry, the headline statistics and every abstract quoted.

Verification. 64 DOIs were resolved against Crossref on 2026-11-05; 63 matched their titles and one (a mistyped Pitts 1995 DOI ending "-3") returned "not found" and is not cited. The catalogue cites 38 DOIs (37 Crossref, one DataCite); five are BSI standards records, which carry an approval date but no designation.

Not accessed: buildingSMART's HTML documentation, iso.org and codes.iccsafe.org (HTTP 403), and the paywalled SFPE Handbook chapters, including the yield tables of [khan2016sfpe]. Numbers that would have come from them are taken from secondary reproductions and marked. No full text was stored in the repository.

---

## 2. Where fires start

### 2.1 Dwellings

Table 1. US home structure fires, 2019 to 2023 annual averages: 328,590 fires, 2,600 civilian deaths and 10,770 injuries per year [nfpa2025homes]. Deaths per 1,000 fires are derived.

| Cause | Fires | Deaths | Injuries | Deaths per 1,000 fires |
|---|---|---|---|---|
| Cooking | 49 % (159,400) | 16 % (430) | 36 % (3,850) | 2.7 |
| Heating equipment | 20 % (65,000) | 17 % (430) | 14 % (1,530) | 6.6 |
| Electrical distribution and lighting | 9 % (31,650) | 16 % (430) | 12 % (1,300) | 13.6 |
| Intentional | 7 % (24,600) | 5 % (120) | 4 % (460) | 4.9 |
| Smoking materials | 5 % (15,200) | 23 % (600) | 10 % (1,050) | 39.5 |
| Candles | 2 % (5,830) | 3 % (70) | 5 % (570) | 12.0 |
| Exposure to another fire | 2 % (6,510) | 1 % (30) | 1 % (50) | 4.6 |
| All causes | 100 % | 100 % | 100 % | 7.9 |

NFPA warns that causes come from several NFIRS data elements, so double counting is possible. Wiring and related equipment accounted for 3 % of home fires and 5 % of deaths, cords or plugs for 1 % and 3 % [nfpa2025homes]. The equipment table [nfpa2025homestables] (agent) gives ranges or cooktops 25 % of fires and 14 % of deaths, heating stoves 9 % and 6 %, space heaters 1 % and 6 %; the appliance rows are in Table 0 and the catalogue.

England is consistent with different emphases (Table 2). Of 23,795 accidental dwelling fires in the year ending March 2026 (161 deaths, 4,314 non-fatal casualties), cooking appliances started 41.7 % and caused 11.2 % of deaths; smoking materials started 7.6 % and caused 26.7 %, "the largest proportion of fire-related fatalities" [mhclg2026detailed]. Electrical distribution is more frequent than in the US (14.3 %) but caused 3 deaths.

Table 2. England, accidental dwelling fires by source of ignition, year ending March 2026 (FIRE0602, aggregated from the published raw data by an agent; totals match the release) [mhclg2026fire0602].

| Source of ignition | Fires | Deaths | Note |
|---|---|---|---|
| Cooking appliances | 9,928 (41.7 %) | 18 (11.2 %) | cooker including oven 6,324; microwave 873; grill or toaster 822; deep-fat fryer 173 |
| Electrical distribution | 3,410 (14.3 %) | 3 (1.9 %) | wiring, cabling and plugs 2,497; batteries and generators 885 (230 casualties, 5.3 %) |
| Other electrical appliances | 2,739 (11.5 %) | 13 (8.1 %) | tumble dryer 579 (2 deaths); washing machine 409; battery charger 142; fridge or freezer 141 (1 death); dishwasher 119; TV 27 |
| Smokers' materials | 1,819 (7.6 %) | 43 (26.7 %) | |
| Candles | 844 (3.5 %) | 10 (6.2 %) | |
| Space heating appliances | 747 (3.1 %) | 7 (4.3 %) | |
| Cigarette lighters; matches | 156; 111 | 5; 4 | |
| Other or unspecified | 3,756 (15.8 %) | 56 (34.8 %) | |

Peer-reviewed studies of fatal fires agree on the pattern and add the first item. In 259 fatal unintentional dwelling fires investigated by the London Fire Brigade from 1996 to 2000, smokers' materials were the ignition source in 47 %, cooking appliances in 14 %, candles and heating appliances in 8 % each; bedclothes were the first item ignited in 23 %, upholstered furniture in 20 % and clothing in 16 % [holborn2003] (agent, full text). A Swedish study found that, compared with non-fatal fires, fatal residential fires "more often originated in the bedroom, were more often caused by smoking, and were more likely to occur at night" [jonsson2017] (abstract). Troitzsch's review reports upholstered furniture as the first item in the most fire deaths, followed by electrical distribution [troitzsch2016] (abstract). A 2026 analysis of US kitchen fires from 2012 to 2021 finds unattended equipment the leading contributing factor [salem2026] (abstract).

### 2.2 First item ignited and room of origin

In US homes (2019 to 2023) cooking materials were first ignited in 29 % of fires but 5 % of deaths, upholstered furniture in 1 % and 14 %, mattresses or bedding in 2 % and 12 %; one in 12 upholstered-furniture fires caused a death [nfpa2025homes] (agent). In England, textiles, upholstery and furnishings were "the item first ignited in 27 % of primary dwelling fires" [mhclg2026detailed]. By room of origin (Fig. 10 of [nfpa2025homes], read by the author), the kitchen accounts for 41 % of fires and 16 % of deaths, the living room for 5 % and 25 %, the bedroom for 6 % and 24 %; apartment fires start in the kitchen in 68 % of cases, one- and two-family home fires in 30 %. In England, 38 % of fatal dwelling fires from 2010/11 to 2018/19 started in the living room [homeoffice2021fatalities] (agent).

The cause reports give the conditional shares the catalogue uses (agent): oil, fat or grease in 51 % of fires that began with cooking materials [nfpa2023cooking]; in smoking fires, upholstered furniture first ignited in 6 % of fires and 26 % of deaths, bedding in 4 % and 18 % [nfpa2025smoking]; bedrooms as origin of 37 % of candle fires and 40 % of candle deaths [nfpa2020candles]; living rooms of 46 % of heating deaths [nfpa2022heating]; wire insulation (32 %) and structural members (34 %) first ignited in electrical fires [nfpa2022electrical]; lint (27 %) and clothing (26 %) in dryer fires [nfpa2017dryers]. A scenario's share is the cause share times the conditional share (smoking on upholstery: 0.23 x 0.26 = 0.060 of home fire deaths); where the two reports cover different years the catalogue says so.

### 2.3 Lithium-ion batteries

Battery fires are rising but are still a small share of dwelling fires. In England, batteries and generators started 885 accidental dwelling fires (3.7 %) in 2025/26, with one death but 5.3 % of casualties [mhclg2026fire0602]. The London Fire Brigade attended 206 e-bike and e-scooter fires in 2025, its highest count, "an e-bike or e-scooter fire every other day, on average" [lfb2026] (agent). The UK product-safety regulator was notified of 211 such fires in 2024, with 8 deaths; 66 % were indoors or residential, 39 % occurred on charge and 45 % of e-bike fires involved conversion kits [opss2025] (agent). New York reported 268 lithium-ion fires and 18 deaths in 2023 [fdny2025] (agent). Peer-reviewed work is mostly clinical or abstract-level: in 148 Massachusetts incidents, charging was involved in 40.5 % and online purchase raised the odds of a charging-related fire (odds ratio 1.87) [bucala2026]; a scoping review of e-bike and e-scooter burns found 185 cases in five reports [warnerlevy2024] (agent).

### 2.4 Non-residential buildings

In US non-residential buildings (2017 to 2019, 108,500 fires a year), cooking caused 30.3 % of fires with a known cause, intentional 10.2 %, electrical 7.5 %; among non-confined fires the order is other careless acts 20 %, electrical 14 %, intentional 9 %, cooking 3 % [usfa2021nonres] (agent). Cooking dominates stores and offices (32.7 %), educational (50.3 %) and institutional properties (69.0 %). In US schools 43 % of fires were intentional, and lavatories or locker rooms were the origin of 32 % in middle and high schools [nfpa2020schools] (agent). In medical facilities appliances (21 %) and electrical causes (18 %) led among non-confined fires [usfa2018medical] (agent). In England's non-dwelling buildings (2025/26) electrical distribution started 16.8 % of fires, batteries and generators 5.9 % of fires and 15.2 % of casualties, and paper or cardboard was first ignited in 8.0 % [mhclg2026fire0602, mhclg2026fire0603]. Offices and schools therefore point to kitchens, electrical equipment and deliberate fires in circulation and sanitary spaces rather than to furniture.

---

## 3. Design fires for the first items

### 3.1 The t-squared classes and Studio's input

Studio's two-zone model takes a t-squared fire, Q(t) = min(alpha t^2, Q_max), with alpha set by a growth class and Q_max the smaller of the user's peak and the ventilation limit 1500 A_o sqrt(H_o) (spec F1). The classes reach 1,055 kW in 600, 300, 150 and 75 s, so alpha is 0.00293, 0.0117, 0.0469 and 0.1876 kW s^-2 (FD-01 and FD-02 in the verified-equations file; [heskestad1979]); zone.js rounds the last to 0.188. To map a test to a class, this study takes the effective coefficient Q_peak / t_peak^2 and chooses the class nearest on a log scale, with boundaries at the geometric means 0.00586, 0.0234 and 0.0938 kW s^-2. Because t_peak in a furniture test runs from the test ignition (for example a 50 kW burner applied for 200 s), the derived coefficient is a lower bound on the growth once the item flames freely.

### 3.2 Upholstered furniture

The NBS furniture-calorimeter series [babrauskas1982nbsir] (Table 2A) pairs construction with peak heat release rate under one ignition (a burner simulating a 50 kW wastebasket for 200 s). A 51.5 kg sofa with California foam and polyolefin fabric peaked at 3,120 kW at 250 s (714 MJ); a 40.4 kg loveseat at 2,890 kW at 230 s; a 28.3 kg armchair at 1,970 and 2,130 kW at 280 and 260 s; a 14.0 kg armchair on a polypropylene frame at 1,950 kW at 220 s (effective heat of combustion 35.1 MJ/kg). Cotton fabric on the same foam cut the armchair to 700 kW at 650 s. Foam-and-polyolefin seating is therefore fast (derived 0.025 to 0.055 kW s^-2), cotton-covered seating slow (0.0004 to 0.003). Two of five easy chairs and two of three sofas tested by Lawson et al. could flash over "an average size waiting room" alone [lawson1984nbsir]; office chairs reached only about 250 and 500 kW [madrzykowski2004sp1021]. The catalogue uses fast with 3,120 kW for a sofa and fast with 2,000 kW for an armchair. The European CBUF report (Sundström 1995) was not accessed.

### 3.3 Mattresses and bedding

In two bedding tests by Lawson et al. (Table 6 of [lawson1984nbsir]), cigarettes ignited neither mattress but a lighter on the sheet did: a mattress without box spring peaked at 1,700 kW, "fully involved" 740 s after ignition (204 MJ; derived 0.0031 kW s^-2, slow), and a mattress on a box spring at 530 kW, then 660 kW as the box spring burned. These were hospital-type furnishings. New US mattresses are bounded by 16 CFR 1633: "the peak rate of heat release shall not exceed 200 kilowatts" within 30 minutes, and total heat release must not exceed 15 MJ in the first 10 minutes [cfr1633]. The bedroom default is slow with 1,700 kW, a measured but dated value; 200 kW bounds a recent US mattress from below.

### 3.4 Wardrobes

Wardrobes are not ignition sources but they are the bedroom's largest fuel package and show how construction dominates growth (Table 7 of [lawson1984nbsir]). A wardrobe of 3.2 mm unfinished plywood with clothing and newspaper peaked above 6,000 kW (table value 6,400 kW) at 130 s, "the fastest burning item with the greatest peak heat release rate of this research project" (derived 0.38 kW s^-2, ultra-fast). Two coats of intumescent paint halved the peak to 2,900 kW at 170 s (derived 0.10 kW s^-2); 12.7 mm plywood wardrobes averaged 3,300 kW (standard deviation 208 kW); a 19.1 mm particleboard wardrobe 1,900 kW; a metal wardrobe with 3.18 kg of rags 770 kW.

### 3.5 Kitchens and cooking oil

Cooking fires are pool fires with a heated base. Corn oil on an electric coil autoignites above about 392 C; a 26 cm pan with 7 mm of oil gave flames of almost 1.4 m and 110 kW, larger than gasoline in the same pans, and 150 g in a 9.7 cm pan boiled over at almost 50 kW [hu2021tn2133] (abstract, Tables 2 and 3). On a ceramic hob, 28 cm pans of sunflower, canola or olive oil peaked at 20 to 35 kW within 20 to 40 s, after 309 +/- 81 s of heating, and only 44 +/- 13 % of heated pans ignited [hostikka2023] (agent). The pan is small; spread to cabinets is what matters. In a 3.6 m x 3.4 m x 2.4 m kitchen with fiberboard cabinets, "even small cooktop fires spread and grew ultra-fast" [hamins2018] (abstract). In NIST's Kitchen Room Fire test (500 mL of canola oil in a 6 in cast-iron pan, OSB-lined wall), flashover came 9.35 min after the pan ignited and the heat release peaked at 7,585 +/- 451 kW, 6,813 MJ in all [nistfcd], a derived 0.0075 kW s^-2 (medium) from pan ignition to peak. The catalogue pairs the peer-reviewed ultra-fast growth with this measured room peak and lists the medium growth as the alternative; the ventilation limit in zone.js will usually bind first.

### 3.6 Electrical appliances

VTT burned TV sets, washing machines, dishwashers and refrigerator-freezers at large scale [hietaniemi2001vtt] (abstract, Tables 5.1 to 5.3, Fig. 5.10). Peaks were up to 2,000 kW for refrigerator-freezers (mean 1,970 kW free-burning; tests stopped with water), 350 to 750 kW for dishwashers (mean 411 kW, 548 kW in a cupboard), 300 to 450 kW for washing machines (mean 333 kW) and 250 to 300 kW for TV sets (mean 241 kW). Each burned first at low heat (1.5 to 3 min for TVs, 5 to 10 min for dishwashers and most refrigerators, 10 to 20 min for washing machines) and then grew rapidly; a TV peaked 1 to 1.5 min after rapid growth began (derived 0.03 to 0.07 kW s^-2, fast), the refrigerators took about 10 min to reach about 2 MW (read from Fig. 5.10; derived 0.004 to 0.006, slow). Effective heats of combustion were 28.1 to 32.3 MJ/kg, smoke yields 0.021 (washing machines) to 0.116 kg/kg (TV sets), CO yields 0.038 to 0.071 kg/kg; one refrigerator released 27 g of HF from its refrigerant. The authors judged that refrigerator-freezer fires "probably lead to flashover". No dryer calorimetry was read: the catalogue uses the washing-machine values as a labelled proxy and assumes the medium class where no growth time was reported.

### 3.7 Christmas trees, waste baskets and office workstations

A 7 ft Douglas fir cut four weeks before and not watered, standing in a room corner with an upholstered chair and a shelf, peaked at 7,362 +/- 445 kW 0.52 min (31 s) after an electric match, releasing 938 MJ [nistfcd]; the derived coefficient, about 7.6 kW s^-2, is forty times the ultra-fast class. A watered tree in the same series stayed below the calorimeter's detection limit. Waste baskets are small: the NBS series modelled a polyethylene wastebasket filled with milk cartons as "a constant Q = 50 kW applied for 200 s" [babrauskas1982nbsir]. An office workstation is not: the Cook County single workstation (3.15 m2, 83 kg/m2 fuel load) peaked at 3.5 MW, and earlier 3.8 m2 workstations from 2.8 to 6.9 MW depending on how many sides were enclosed [madrzykowski2004sp1021]. The time to peak was not read, so the catalogue pairs the workstation peak with EN 1991-1-2's office growth class.

### 3.8 Lithium-ion packs

An overcapacity e-bike pack heated to thermal runaway in a cabinet started runaway at 300 C and peaked at 68 kW at 80 % state of charge and 107 kW at 60 %, with jet flames [liu2025ebike] (abstract). The NIST database also holds lithium-polymer pack tests, but its listing does not say whether the ignition burner's heat is included, so they are not used. The pack is small; the room is not. In UL FSRI's experiments, an overcharged e-scooter in a closed bedroom ignited its vented gases about 20 s after visible smoke and the room reached flashover within 30 s of visible smoke [fsri2025] (research institute summary of [fleischmann2025], whose abstract was read). The catalogue therefore pairs the ultra-fast class with a composite peak (pack plus an adjacent sofa, 3,120 kW), labelled as a judgement. Toxic gases matter as much as heat: seven commercial cell types released 20 to 200 mg of HF per Wh of capacity and 15 to 22 mg/Wh of POF3 [larsson2017] (abstract); the two-zone model does not track either.

### 3.9 Occupancy design fires and design-fire standards

Without item data, a standard's occupancy design fire is the defensible default. EN 1991-1-2 Annex E (Table E.5, read as reproduced on a JRC workshop slide [jrc2012vassart]) gives medium growth with 250 kW/m2 of floor area for dwellings, hospital and hotel rooms, offices and schools; fast growth with 250 kW/m2 for shopping centres and 500 kW/m2 for libraries and theatres; slow growth for transport spaces. Its 80 % fractile fire loads are 948 MJ/m2 for dwellings, 511 for offices, 347 for schools and 280 for hospital rooms [en1991-1-2]. PD 7974-1:2019 [pd7974-1] and ISO/TS 16733-1 [iso16733-1] set out how to select design fire scenarios; neither was read and no number rests on them. Holborn, Nolan and Golt fitted log-normal distributions to growth rates from London investigations [holborn2004]; only its record was read, but it is the natural source for a probabilistic ranking.

### 3.10 Mapping to Studio

Each catalogue scenario carries `designFire: {growth, alpha_kW_per_s2, peak_kW | peak_kW_per_m2, basis, source}`, an optional `firstItemFire` (the item alone) and `alternatives`. `basis` says whether the peak is the item, the item plus the furniture it ignites (a composite and a judgement), a measured room or an occupancy default. A per-area peak is multiplied by the fire room's floor area from rooms.js. The `fuelItems` array holds every measured item above with mass, peak, time to peak, total heat, effective heat of combustion, yields and derived coefficient, so that a later combustion model can grow the fire item by item.

---

## 4. How a fire grows from the first item

This section covers only what bears on choosing and growing an origin; the flashover and zone equations are already in the verified-equations file and in Zone.lean (formal/fireegress/README.md, sixth batch), and are not repeated.

### 4.1 Will the second item ignite?

Babrauskas measured the irradiance around burning furniture and the ignition times of furnishing specimens at fixed fluxes [babrauskas1981nbsir] (pp. 21 to 28 read; journal version [babrauskas1981fsj]). He proposed three ignitability levels: especially easily ignitable, 10 kW/m2; normal, 20 kW/m2; difficult, 40 kW/m2. Most fabric and polyurethane-foam pieces ignited within 45 s at 20 kW/m2. The fastest-burning items produced about 80 kW/m2 at 0.05 m, but 40 kW/m2 was not recorded beyond 0.44 m nor 20 kW/m2 beyond 0.88 m, so common furnishings "would stand little hazard of fire involvement if placed at least 1 m away"; his Fig. 13 gives about 1.38, 0.88 and 0.40 m for the three levels at the largest first item (47 g/s). The limits hold only before flashover. The point-source relation q'' = chi_r Q / (4 pi R^2), valid beyond five fire diameters [hu2021tn2133, Eq. 5], sharpens the estimate where the radiative fraction is known (0.44 for corn-oil and 0.37 for gasoline pan fires; not read for furniture).

For Studio this gives a cheap growth rule: list the IFC furniture within 1 m of the origin item (bounding-box gap), treat items of normal ignitability as ignited when the first item nears its peak, and add their own t-squared curves. Zone.lean already proves that a larger Q never lowers the layer temperature, so added items can only bring untenability earlier.

### 4.2 Flashover and the size of the design fire

The flashover correlations of Babrauskas (750 A_o sqrt(H_o) kW), McCaffrey, Quintiere and Harkleroad (740 (h_k A_T A_o sqrt(H_o))^(1/2) kW) and Thomas (7.8 A_T + 378 A_o sqrt(H_o) kW) are FD-03, FD-04 and FD-05, and zone.js uses MQH with a 500 K latch (FD-17) [babrauskas1980, mccaffrey1981, thomas1981, peacock1999]. Babrauskas's 1981 report adds the scale that matters for choosing a peak: fires of 0.4 to 1.0 MW can flash over rooms whose ventilation factor A sqrt(h) is at most 0.5 to 1.3 m^(5/2), whereas a common doorway gives about 2.2 m^(5/2) [babrauskas1981nbsir]. With 750 A sqrt(h), such a doorway needs about 1.65 MW (derived), so a single sofa (3.1 MW) or wardrobe (3.3 to 6.4 MW) can flash over a room with one door, while a TV set (0.24 MW) or dishwasher (0.41 MW) needs a second item. The design peak in the catalogue is the item's peak, and zone.js decides flashover from the room.

---

## 5. Fire classes

### 5.1 Classification systems

The Saudi Fire Protection Code SBC 801 (2024 edition [sbc801]) follows the International Fire Code, whose section 906.2 requires portable extinguishers to be selected, installed and maintained in accordance with NFPA 10 [ifc906]; the IFC wording was seen only in state adoptions and SBC 801's own reference was not read. NFPA 10 (2022 edition) is the primary system here [nfpa10]: A, ordinary combustibles such as wood, cloth, paper, rubber and many plastics; B, flammable and combustible liquids, greases, oils, solvents, alcohols and flammable gases; C, energised electrical equipment; D, combustible metals such as magnesium, titanium, zirconium, sodium, lithium and potassium; K, cooking appliances with vegetable or animal oils and fats. The text was read in NFPA's second draft for the next edition (2022 text with revisions); a 2026 edition is reported published in September 2025.

ISO 3941:2007 and EN 2:1992 with A1:2004 classify by fuel: A solids, B liquids, C gases, D metals and, since A1, F cooking media; neither has an electrical class [iso3941-2007, en2]. ISO 3941:2026 adds Class L for lithium-ion cells and batteries without metallic lithium [iso3941-2026], according to trade and consultancy reports; neither ISO text was readable.

Table 3. Mapping between the systems.

| NFPA 10 | ISO 3941:2007 / EN 2 | Notes for Studio |
|---|---|---|
| A | A | Furniture, bedding, paper, wood, plastics |
| B | B (liquids), C (gases) | Accelerants, solvents, heating oil; natural gas and LPG at burners |
| C | none | Classify by the fuel that burns once the equipment is involved (catalogue `fuelClass`, usually A) |
| D | D | Out of scope for the two-zone model |
| K | F | Oil and fat in cooking appliances |
| none | L (ISO 3941:2026 only) | Lithium-ion; Studio records a charging pack as C with fuel class B |

### 5.2 Fuel properties per class

Table 4 gives one representative value per field with its source, and the range where the literature shows one. The catalogue's `fireClasses` object holds the same values with notes.

| Class | Representative fuels | Effective heat of combustion | CO yield | Soot yield | Growth | Ignition | Embers |
|---|---|---|---|---|---|---|---|
| A | Upholstered furniture, bedding, wood | 18.0 MJ/kg (13.9 to 20.9 for wood-framed seating; 35.1 for a polypropylene frame) [babrauskas1982nbsir, lawson1984nbsir] | 0.01 kg/kg well ventilated, about 0.2 under-ventilated [gannbryner, mckenna2016] | 0.015 kg/kg wood (0.008 to 0.194; flexible polyurethane foam 0.188), secondary [klotemilke2002] | medium (occupancy); furniture fast | 20 kW/m2 (10 to 40) [babrauskas1981nbsir]; plywood and foam 16 kW/m2, 390 C, secondary [sfpe2002ignition] | yes, charring solids [manzello2020] |
| B | Gasoline, solvents, heating oil, gas | 44.1 MJ/kg gasoline (20.0 methanol to 46.0 transformer oil) [hu2021tn2133, nureg1805s1] | 0.022 kg/kg (derived) [hu2021tn2133] | 0.072 kg/kg (derived) [hu2021tn2133] | pool: m''_inf 0.055 kg m^-2 s^-1, k beta 2.1 m^-1 (gasoline), secondary [babrauskas1983pool]; spills ultra-fast (judgement) | not read | no (inference) |
| C | Appliances, boards, wiring | 30.3 MJ/kg (28.1 to 32.3) [hietaniemi2001vtt] | 0.056 kg/kg (0.038 to 0.071) | 0.056 kg/kg (0.021 to 0.116) | medium (slow for refrigerators, fast for TVs) | internal fault, not a flux criterion | as the fuel |
| D | Magnesium, sodium, lithium metal | 24.75 MJ/kg Mg (derived from the enthalpy of formation of MgO, -601.60 kJ/mol) [nistwebbookMgO] | none from the metal | metal oxide fume, not soot | not characterised | Mg about 473 C, Na about 121 C, Li about 179 C, secondary [cameo, zhao2022] | not established |
| K | Cooking oil and fat | 37.4 MJ/kg corn oil [hu2021tn2133] | 0.021 kg/kg (derived) | 0.069 kg/kg (derived) | ultra-fast in a kitchen [hamins2018] | autoignition about 390 C (385 to 429) [hu2021tn2133, hostikka2023] | no (inference) |
| L (ISO) | Lithium-ion packs | not a single fuel | HF 20 to 200 mg/Wh, POF3 15 to 22 mg/Wh [larsson2017] | not read | ultra-fast in a room [fsri2025] | runaway at about 300 C [liu2025ebike] | not established |

Notes on the table. The class B and K yields are derived from TN 2133's relations, CO yield = 0.034 g/m2 x SEA and soot yield = 0.11 g/m2 x SEA (its Eq. 18; Tewarson's constant is 0.0994 g/m2), with measured specific extinction areas of 0.65 m2/g for gasoline (+/- 45 %) and 0.63 m2/g for corn oil (+/- 22 %). Tewarson's well-ventilated ratio of CO to smoke yield, 0.34 +/- 0.05 for ordinary polymers, liquids and gases [tewarson2008], gives a CO yield for any class A or B fuel whose soot yield is known; it does not hold for halogenated polymers. The per-material yields of Tewarson's SFPE chapter [khan2016sfpe] should replace the secondary class A values once the chapter is available. Ember production is supported in the literature only for charring solids, and only for outdoor spread (abstract-level); the "no" for liquids and cooking oil is an inference from the absence of any report, and boil-over throws burning oil, which is a spill rather than a firebrand.

### 5.3 Extinguishing agents

Table 5. Agents per class, with the reason an agent is dangerous. Sources: [nfpa10] (annex text, agent), [liu2005fss] for cooking oil, [cameo] for metals.

| Class | Suitable | Unsuitable and why |
|---|---|---|
| A | Water; foam rated for class A; multipurpose (ABC) dry chemical; wet chemical; halogenated agents | Carbon dioxide: listed for B and C only, with a short reach (about 1 to 2.4 m) and no cooling of deep-seated embers |
| B | Foam (AFFF, FFFP), which floats on the liquid and prevents re-ignition; CO2; dry chemical; halogenated agents; for pressurised jets, dry chemical once the supply can be shut off | Plain water: water extinguishers are for class A only, and a stream splashes and spreads burning liquid |
| C | CO2 and halocarbon clean agents (non-conductive); dry chemical (its residue can conduct on wet equipment); de-energise first, then treat as A or B | Water or foam on energised equipment: conducts current to the operator; NFPA 10 bars conductive agents from class C extinguishers |
| D | Listed class D powders; dry sand, dolomite, soda ash, sodium chloride or graphite applied gently | Water and foam: the burning metal releases hydrogen, and the reaction can be explosive; CO2 and halogenated agents: burning magnesium keeps reacting |
| K | Wet chemical, which saponifies the oil into a foam blanket and cools it; fine water mist from fixed systems | Water jet or coarse spray: drops sink, boil and expel the oil; CO2: knocks the flame down but leaves the oil above its autoignition temperature, so it re-ignites; wet chemical injected below the surface: a reaction NFPA 10 compares to an explosion |
| L | No agent comparison in a source read; packs re-ignite without sustained cooling | Not established |

### 5.4 How the app should use the classes

The class of the item at the origin should parameterise the new combustion model and the interface together. In the model, the class (for class C its `fuelClass`) selects the effective heat of combustion, which converts heat release to mass loss; the CO and soot yields, which turn the layer into a smoke density and a CO concentration for an ISO 13571 asphyxiant dose (the present model doses heat only); the default growth class; and the ember flag, which enables ignition of distant items for class A only. An item's own measured yields (TV sets, appliances) override the class values. Class D lies outside the model and should be reported so; class L should warn that HF is not modelled. In the interface, each likely ignition point shows its class as a letter in a neutral shape (no vendor pictograms), its name ("Class K: cooking oil and fat") and one line each for suitable and unsuitable extinguishers, for example "Use: wet chemical extinguisher. Do not use: water, which spreads burning oil."

---

## 6. Integration with IFC parsing and the interface

### 6.1 What the app reads today

`ifcSemantics(api, modelID, W)` (src/core/ifcSemantics.js, run by the import worker's `semantics`) makes one web-ifc pass over the relations: it maps property sets and types to elements (`ownSets`, `typeOf`), reads doors' exit flags and sizes, roof slabs, storeys and IfcSpace names. rooms.js builds the room graph from IfcSpace meshes in the viewer frame (`roomsFromSpaces`, `footprintIndex.roomAt`, `assignNodes`, `roomDoors`). zone.js finds the fire room with `fireRoomOf(nodeRoom, seed)`, which throws when the origin is in no room. `SimulatePanel` (src/ui/Panels.jsx) has the "Fire origin" `PickRow` and `PlaceWithoutMouse`; the forecast section's "The room fire" disclosure holds `settings.zoneGrowth` and `settings.zonePeakKw` (defaults medium and 5,000 kW, src/state/useStudio.js), which App.jsx passes to `buildZoneForecast`. Points go through `handlePick(mode, point)` and `snapToGraph`. Element data come from the fragments model through src/viewer/categories.js (`CATEGORY_PATTERNS`, `collectCategoryIds`, `extractMeshes`, `elementRecords`), the route cores.js uses to find lifts by name.

### 6.2 IFC entities and enumerations

Values below are from buildingSMART's EXPRESS files for IFC2x3 TC1 and IFC4 ADD2 TC1 and an IFC4X3_ADD2 copy cross-checked against buildingSMART's 4.3 documentation source [ifc2x3exp, ifc4exp, ifc43exp]; the HTML pages [ifc2x3html, ifc4html] returned HTTP 403 to both tools on 2026-11-05.

Table 6. Ignition-relevant entities.

| Purpose | IFC4 and IFC4.3 occurrence (PredefinedType values) | IFC2x3 equivalent |
|---|---|---|
| Domestic and office appliances | IfcElectricAppliance: DISHWASHER, ELECTRICCOOKER, FREESTANDINGELECTRICHEATER, FREESTANDINGFAN, FREESTANDINGWATERHEATER, FREESTANDINGWATERCOOLER, FREEZER, FRIDGE_FREEZER, HANDDRYER, KITCHENMACHINE, MICROWAVE, PHOTOCOPIER, REFRIGERATOR, TUMBLEDRYER, VENDINGMACHINE, WASHINGMACHINE, USERDEFINED, NOTDEFINED | No occurrence entity: IfcFlowTerminal typed by IfcElectricApplianceType (PredefinedType COMPUTER, DIRECTWATERHEATER, DISHWASHER, ELECTRICCOOKER, ELECTRICHEATER, FACSIMILE, FREESTANDINGFAN, FREEZER, FRIDGE_FREEZER, HANDDRYER, INDIRECTWATERHEATER, MICROWAVE, PHOTOCOPIER, PRINTER, REFRIGERATOR, RADIANTHEATER, SCANNER, TELEPHONE, TUMBLEDRYER, TV, VENDINGMACHINE, WASHINGMACHINE, WATERHEATER, WATERCOOLER) |
| Gas cookers and burners | IfcBurner (USERDEFINED, NOTDEFINED; Pset_BurnerTypeCommon.EnergySource) | IfcFlowTerminal typed by IfcGasTerminalType: GASAPPLIANCE, GASBOOSTER, GASBURNER; Pset_GasTerminalTypeGasAppliance.GasApplianceType GASFIRE or GASCOOKER |
| Space heaters | IfcSpaceHeater: CONVECTOR, RADIATOR; Pset_SpaceHeaterTypeCommon.EnergySource (ELECTRICITY, GAS, OIL, PROPANE, WOOD, WOOD_PELLET, COAL and others) | IfcEnergyConversionDevice typed by IfcSpaceHeaterType (SECTIONALRADIATOR, PANELRADIATOR, TUBULARRADIATOR, CONVECTOR, BASEBOARDHEATER, FINNEDTUBEUNIT, UNITHEATER; Pset HeatingSource); IfcElectricHeaterType (ELECTRICPOINTHEATER, ELECTRICCABLEHEATER, ELECTRICMATHEATER) |
| Boilers, generators, transformers | IfcBoiler (WATER, STEAM); IfcElectricGenerator (CHP, ENGINEGENERATOR, STANDALONE); IfcTransformer | IfcEnergyConversionDevice typed by the corresponding *Type |
| Distribution boards | IfcElectricDistributionBoard: CONSUMERUNIT, DISTRIBUTIONBOARD, MOTORCONTROLCENTRE, SWITCHBOARD | IfcElectricDistributionPoint (an IfcFlowController with attribute DistributionPointFunction: ALARMPANEL, CONSUMERUNIT, CONTROLPANEL, DISTRIBUTIONBOARD, GASDETECTORPANEL, INDICATORPANEL, MIMICPANEL, MOTORCONTROLCENTRE, SWITCHBOARD) |
| Outlets, lighting | IfcOutlet: POWEROUTLET and others; IfcLightFixture: POINTSOURCE, DIRECTIONSOURCE, SECURITYLIGHTING (IfcLampTypeEnum includes HALOGEN, TUNGSTENFILAMENT) | IfcFlowTerminal typed by IfcOutletType, IfcLightFixtureType |
| Electronics | IfcAudioVisualAppliance: DISPLAY, PLAYER, PROJECTOR and others; IfcCommunicationsAppliance: COMPUTER, PRINTER and others | IfcElectricApplianceType TV, COMPUTER, PRINTER |
| Batteries, UPS | IfcElectricFlowStorageDevice: BATTERY, UPS (IFC4.3 adds RECHARGER). "IfcElectricStorageDevice" does not exist in any version | IfcFlowStorageDevice typed by IfcElectricFlowStorageDeviceType |
| Furniture | IfcFurniture: BED, CHAIR, DESK, FILECABINET, SHELF, SOFA, TABLE (IFC4.3 adds TECHNICALCABINET); IfcSystemFurnitureElement: PANEL, WORKSURFACE | IfcFurnishingElement; IfcFurnitureType has only AssemblyPlace, no PredefinedType, so names carry the meaning |
| Chimneys | IfcChimney (USERDEFINED, NOTDEFINED) | none |
| Anything else | IfcBuildingElementProxy (PredefinedType USERDEFINED requires ObjectType) | IfcBuildingElementProxy (no PredefinedType; Name required by rule WR1) |
| Spaces | IfcSpace.PredefinedType: SPACE, PARKING, GFA, INTERNAL, EXTERNAL (no room function); Name, LongName, ObjectType; Pset_SpaceOccupancyRequirements.OccupancyType; Pset_SpaceFireSafetyRequirements.FireRiskFactor, FlammableStorage | IfcSpace (no PredefinedType; InteriorOrExteriorSpace); Pset_SpaceCommon.Category |

In IFC4 the type object's PredefinedType applies when the occurrence's is unset, and IfcFlowTerminal and IfcFurnishingElement are deprecated for instantiation but not abstract, so IFC4 files can still contain them. web-ifc 0.0.77 defines constants for every entity in Table 6 (IFCELECTRICFLOWSTORAGEDEVICE, not IFCELECTRICSTORAGEDEVICE), and `GetLineIDsWithType` returns an empty vector, without throwing, for an entity the file's schema lacks; with `includeInherited` it adds IFC4 subtypes of IfcFlowTerminal and IfcFurnishingElement, but IFC2x3 has no subtypes to add.

### 6.3 What real models contain

The local test corpus (45 distinct building models of at least 50 kB, local testing only) was scanned with the catalogue's patterns. Seventeen models contain a recognised element: 10 an appliance, heater, fireplace or distribution board, 13 upholstered furniture, beds or desks. No model contains an IfcElectricAppliance occurrence; IfcElectricApplianceType occurs in two IFC2x3 Revit exports with PredefinedType NOTDEFINED and names "2000 lbs" and "3500 lbs", which are lift cars. Appliances arrive as IfcBuildingElementProxy (a Miele cooktop, oven, microwave, dryer, washer, fridge-freezer and dishwasher in one IFC4 model; "M_Cook Top-4 Unit" and "M_Refrigerator" in an IFC2x3 house; "N_Oven-Built-in" in a Norwegian school), as IfcFlowTerminal ("M_Washing Machine") or as furniture types ("Refrigerator 24", "Cabinet Base Cooktop+Oven 24"). All 238 IfcFurniture occurrences have PredefinedType unset or NOTDEFINED. The only specifically typed entities are 44 hydronic IfcSpaceHeater RADIATOR and 2 IfcBoiler WATER in one model; distribution boards appear only as proxies named "Lighting and Appliance Panelboard".

### 6.4 Name and space heuristics

Each scenario has `nameRegex`, `nameExclude`, `nameHintsAr`, `spaceRegex` and `spaceHintsAr` (JavaScript, flag `i`). English tokens are preceded by `(?:^|[^a-z])` rather than `\b`: Revit names start with "M_" or "N_" and "_" is a word character, so `\b` fails in "M_Bed" (the first draft missed every Revit bed). German and Norwegian tokens have no leading boundary so that compounds match (Doppelbett, Elektroherd, oppvaskmaskin); short tokens carry a trailing guard (`hob(?![a-z])`). Arabic tokens (فرن oven, ثلاجة refrigerator, سرير bed, مطبخ kitchen) are matched as substrings because JavaScript's `\b` is ASCII-only. Exclusions remove false friends: range hoods, air outlets, hand and hair dryers, water heaters, ceiling panels, EV charging stations ("E-Ladestation"). On the corpus the final patterns returned no false positive among cooking, upholstery, bedding and TV names (a "TV Stand" matches, which locates the TV).

The text matched is the occurrence's Name, ObjectType and LongName (as `elementRecords` joins them), then its type's Name and ElementType (IfcRelDefinesByType, already parsed in `ifcSemantics`), then Revit's "Family" and "Type" properties. Space names raise the confidence of an item in a matching room and, when a model has no recognisable items, nominate rooms as origins (section 6.8).

### 6.5 Ranking candidate origins

The quantity to rank is the expected share of deaths. For a scenario s, the share of fires times the deaths per fire is proportional to the scenario's share of deaths, so the catalogue's `shareDeaths` is the scenario weight w_s. For candidate item i matched to s with confidence m_i, among N_s candidates of the same scenario in the model,

score_i = w_s x m_i / N_s,

with m_i = 1.0 for a typed PredefinedType match, 0.9 for a typed entity confirmed by name, 0.7 for a name on the element, 0.6 for a name on its type only and 0.4 for a space name alone, each multiplied by the catalogue's `weight` for weak entity matches (hydronic radiators 0.1, outlets 0.2). Dividing by N_s keeps a model with twelve sofas from filling the list with sofas; the list shows the best candidate of each scenario and "k more" beneath it. Occupancy selects the weights: residential models use the US shares with the England shares as a check, and non-residential models (offices, schools, healthcare, retail) use `shareNonResidential` where the catalogue has it, flagged where it does not. Occupancy is inferred from space names and Pset_SpaceOccupancyRequirements.OccupancyType and can be overridden in the panel. Injuries break ties. The ranking orders national frequency and consequence; it is not a risk assessment of the specific building, and the interface should say so.

### 6.6 Placing the origin on the walkable graph

The origin must be a lattice node in the item's own room, on its own storey.

1. Take the item's mesh from the fragments model with `extractMeshes`, in the same viewer frame as the IfcSpace meshes that rooms.js uses. Do not compute positions from IfcLocalPlacement: the viewer frame adds the coordination matrix and the axis swap.
2. Compute the item's plan centre (c0, c1) and base height zb (the lowest vertex on the up axis).
3. Room: r = `footprintIndex.roomAt(c0, c1, zb, FLOOR_TOLERANCE)`; if -1, retry within SEARCH_BAND (1 m), then use the IfcSpace that contains the element through IfcRelContainedInSpatialStructure if the exporter used one.
4. Node: among nodes v with `nodeRoom[v] === r` (prefer `direct[v] === r`), take the one with the least plan distance to (c0, c1). Furniture is not an obstacle in the lattice, so a node usually lies within half a grid step of an item's footprint; appliances against walls lie within a step of the wall.
5. Fallback: if the room has no node, take the nearest node on the same storey (node height minus the agent height within 0.5 m of zb) within 2 m in plan; otherwise drop the candidate and report why ("no walkable floor next to the oven").
6. Return `{node, point, room: r, planDistance}` and place it with `onPlace("fire", point)`. Because the point is a node, `snapToGraph` returns it unchanged and `fireRoomOf(nodeRoom, node)` returns r.

This replaces, for these candidates, the 3D nearest-node snap, which for a wall oven or a range hood can land on the storey above or in the next room.

### 6.7 Choosing the design fire

`designFireFor(scenario, room, occupancy)` returns `{growth, peakKw, basis, source}`: the scenario's `designFire`, with `peak_kW_per_m2` multiplied by `rooms[r].floorArea` when given, and the occupancy's EN 1991-1-2 growth class when the scenario has none. It sets `settings.zoneGrowth` and `settings.zonePeakKw` and, when the model has rooms, `settings.fireModel = "zone"`. After the run, the forecast's `zone.ventLimited` and `zone.Qmax` already say whether the room's doors capped the fire; the panel should show "limited by ventilation to Q kW" next to the chosen design fire. For the graph-only fire models (heat diffusion, spread along links, distance bands), which have no heat release rate, the list can still set the origin, and the panel should say that growth and peak apply only to the two-zone model.

### 6.8 What the interface should offer

A "Likely ignition points" section in the Simulate step, under the "Fire origin" row, lists up to ten candidates. Each row shows the class letter in a neutral shape, the label and room ("Cooktop, Kitchen 0.12"), one statistic with population and source ("Cooking: 49 % of US home fires, 16 % of deaths, 2019 to 2023, NFPA"), the design fire ("ultra-fast, 7.6 MW peak") and a "Use" button that places the origin and sets growth and peak in one action. A disclosure gives the evidence type, sources, first fuel and the extinguisher line of section 5.4. A footer states the occupancy used and that the ranking reflects national statistics, not an assessment of this building.

In 3D, candidates appear as small markers above their items, coloured by rank and labelled with the class letter; hovering highlights the item by its local id (the express id, as for doors), and clicking a marker equals "Use".

When a model has no recognisable items the section says so and falls back to rooms: "This model names no cookers, heaters, distribution boards or upholstered furniture that Studio recognises. Home fires most often start in kitchens (41 % of US home fires); deaths most often follow fires that start in living rooms (25 %) and bedrooms (24 %)." It then lists kitchens, living rooms and bedrooms by space name, each placing the origin at the room's most central node with that room's scenario. Without IfcSpace names, the user picks the origin by hand, as today.

### 6.9 Functions and data to add

- `wiki/research/ignition-sources-2026-11-05.json`, loaded as static data (or copied to `src/data/ignition-catalogue.json` with a schema version check).
- src/viewer/categories.js: a `CATEGORY_PATTERNS.ignition` entry covering IFCELECTRICAPPLIANCE, IFCFLOWTERMINAL, IFCFURNITURE, IFCFURNISHINGELEMENT, IFCSYSTEMFURNITUREELEMENT, IFCSPACEHEATER, IFCENERGYCONVERSIONDEVICE, IFCBURNER, IFCBOILER, IFCELECTRICDISTRIBUTIONBOARD, IFCELECTRICDISTRIBUTIONPOINT, IFCFLOWCONTROLLER, IFCOUTLET, IFCLIGHTFIXTURE, IFCAUDIOVISUALAPPLIANCE, IFCCOMMUNICATIONSAPPLIANCE, IFCELECTRICFLOWSTORAGEDEVICE, IFCFLOWSTORAGEDEVICE, IFCCHIMNEY, IFCELECTRICGENERATOR, IFCTRANSFORMER; proxies are already collected.
- src/core/ifcSemantics.js: return `elementTypes: Map(expressId -> {typeEntity, typeName, typePredefined, elementType})` for those entities, and for IFC2x3 the type's PredefinedType (fragments' `getItemsData` does not return relations), plus the property values named in Table 6 (EnergySource, HeatingSource, GasApplianceType, OccupancyType, FireRiskFactor). Bump IMPORT_SETTINGS, since cached imports lack these fields.
- src/core/ignition.js (new, pure, testable like cores.js): `matchIgnition(record, types, catalogue) -> {scenarioId, confidence, matchedBy} | null`; `ignitionItems(records, types, catalogue)`; `inferOccupancy(spaces, psets)`; `rankIgnitions(items, catalogue, occupancy)`; `placeIgnition(item, mesh, graph, roomGraph, {upAxis, agentHeight}) -> {node, point, room, planDistance} | {reason}`; `designFireFor(scenario, room, occupancy)`; `roomFallbacks(spaces, roomGraph, catalogue)`; and `neighboursWithin(itemId, meshes, 1.0)` for the second-item rule of section 4.1.
- src/ui/Panels.jsx: a `LikelyIgnitionPoints` component in `SimulatePanel`, taking `ignitions` and `onUseIgnition(candidate)`; App.jsx's handler calls `handlePick("fire", point)` and `setSettings({zoneGrowth, zonePeakKw, fireModel})`.
- Viewer: `setIgnitionMarkers(list)` and `highlightItem(localId)`.
- Tests: `ignition.test.mjs` with the corpus names quoted in section 6.3 as fixtures (positive and negative), placement on a two-storey fixture to prove the origin stays on the item's storey and room, and a check that every catalogue `source` key resolves.

---

## 7. Limitations and open items

The statistics are national (US, England) and mostly residential; no Saudi or Gulf data were found, and they should be sought before the ranking is presented as applicable there. Several NFPA cause reports were read only by an agent. The furniture and appliance calorimetry dates from 1982 to 2001, and modern products differ; the CBUF data and Babrauskas's SFPE chapter on heat release rates were not read. Growth for dishwashers, washing machines and dryers is assumed, the dryer peak is a proxy and composite peaks are judgements, all marked in the catalogue. Class A, B and D properties partly come from secondary reproductions and the class B and K yields are derived from extinction measurements; Tewarson's tables should replace them. The ISO 3941 and EN 2 texts, the ICC text of section 906.2 and SBC 801's NFPA 10 reference were not read. The two-zone model has no CO, HF or visibility; the class yields are for the planned combustion model.

---

## 8. References

Peer-reviewed

- [babrauskas1980] Babrauskas V. Estimating room flashover potential. Fire Technology 16(2):94-103, 1980. doi:10.1007/BF02351801. Crossref verified.
- [babrauskas1981fsj] Babrauskas V. Will the second item ignite? Fire Safety Journal 4(4):281-292, 1981/82. doi:10.1016/0379-7112(81)90031-X. Crossref verified; not read (the NBS report was read).
- [babrauskas1983pool] Babrauskas V. Estimating large pool fire burning rates. Fire Technology 19(4):251-261, 1983. doi:10.1007/BF02380810. Crossref verified; values via a handbook table.
- [bucala2026] Bucala M, Nguyen, Knight, Barowy. Device-specific fire risk analysis of lithium-ion battery incidents. Fire Technology 62(2), 2026. doi:10.1007/s10694-026-01876-5. Crossref verified; abstract.
- [fleischmann2025] Fleischmann C, Weinschenk C, Madrzykowski D, Schraiber A, Gaudet B. Quantifying the fire hazard from Li-ion battery fires caused by thermal runaway in e-scooters. Fire Technology 61(5):2865-2887, 2025. doi:10.1007/s10694-025-01707-z. Crossref verified; abstract summary.
- [hamins2018] Hamins A, Kim SC, Madrzykowski D. Characterization of stovetop cooking oil fires. Journal of Fire Sciences 36(3):224-239, 2018. doi:10.1177/0734904118761884. Crossref verified; abstract.
- [heskestad1979] Heskestad G, Delichatsios MA. The initial convective flow in fire. Symposium (International) on Combustion 17(1):1113-1123, 1979. doi:10.1016/S0082-0784(79)80106-X. Crossref verified.
- [holborn2003] Holborn PG, Nolan PF, Golt J. An analysis of fatal unintentional dwelling fires investigated by London Fire Brigade between 1996 and 2000. Fire Safety Journal 38(1):1-42, 2003. doi:10.1016/S0379-7112(02)00049-8. Crossref verified.
- [holborn2004] Holborn PG, Nolan PF, Golt J. An analysis of fire sizes, fire growth rates and times between events using data from fire investigations. Fire Safety Journal 39(6):481-524, 2004. doi:10.1016/j.firesaf.2004.05.002. Crossref verified; record only.
- [hostikka2023] Hostikka S, et al. Exploring the thermal characteristics, ignitions and heat release of oils and solid items at electric cooktops. Fire Safety Journal 140:103889, 2023. doi:10.1016/j.firesaf.2023.103889. Crossref verified.
- [hu2021jfs] Hu Y, et al. The character of residential cooktop fires. Journal of Fire Sciences 39(2):142-163, 2021. doi:10.1177/0734904121990392. Crossref verified; abstract.
- [jonsson2017] Jonsson A, Bonander C, Nilson F, Huss F. The state of the residential fire fatality problem in Sweden. Journal of Safety Research 62:89-100, 2017. doi:10.1016/j.jsr.2017.06.008. Crossref verified; abstract.
- [larsson2017] Larsson F, Andersson P, Blomqvist P, Mellander BE. Toxic fluoride gas emissions from lithium-ion battery fires. Scientific Reports 7, 2017. doi:10.1038/s41598-017-09784-z. Crossref verified; abstract.
- [liu2005fss] Liu Z, Carpenter DW, Kim AK. Application of water mist to extinguish large oil pool fires for industrial oil cooker protection. Fire Safety Science 8:741-752, 2005. doi:10.3801/IAFSS.FSS.8-741. Crossref verified.
- [liu2025ebike] Liu J, et al. Experimental investigation of the thermal runaway and fire characteristics of overcapacity lithium-ion batteries for electric bicycles in charging cabinets. Frontiers in Energy Research 13:1681969, 2025. doi:10.3389/fenrg.2025.1681969. Crossref verified; abstract.
- [manzello2020] Manzello SL, et al. Role of firebrand combustion in large outdoor fire spread. Progress in Energy and Combustion Science 76:100801, 2020. doi:10.1016/j.pecs.2019.100801. Crossref verified; abstract level.
- [mccaffrey1981] McCaffrey BJ, Quintiere JG, Harkleroad MF. Estimating room temperatures and the likelihood of flashover using fire test data correlations. Fire Technology 17(2):98-119, 1981. doi:10.1007/BF02479583. Crossref verified.
- [mckenna2016] McKenna ST, Hull TR. The fire toxicity of polyurethane foams. Fire Science Reviews 5:3, 2016. doi:10.1186/s40038-016-0012-3. Crossref verified.
- [peacock1999] Peacock RD, Reneke PA, Bukowski RW, Babrauskas V. Defining flashover for fire hazard calculations. Fire Safety Journal 32(4):331-345, 1999. doi:10.1016/S0379-7112(98)00048-4. Crossref verified.
- [salem2026] Salem J, et al. The influence of cooking practices and social circumstances on kitchen fire incidents of residential buildings in the USA. Fire Technology 62(4), 2026. doi:10.1007/s10694-026-01902-6. Crossref verified; abstract.
- [tewarson2008] Tewarson A. Smoke emissions in fires. Fire Safety Science 9:1153-1164, 2008. doi:10.3801/IAFSS.FSS.9-1153. Crossref verified.
- [thomas1981] Thomas PH. Testing products and materials for their contribution to flashover in rooms. Fire and Materials 5(3):103-111, 1981. doi:10.1002/fam.810050305. Crossref verified.
- [troitzsch2016] Troitzsch JH. Fires, statistics, ignition sources, and passive fire protection measures. Journal of Fire Sciences 34(3):171-198, 2016. doi:10.1177/0734904116636642. Crossref verified; abstract.
- [warnerlevy2024] Warner-Levy J, Herieka E, Sheikh Z. Riding toward danger. Journal of Burn Care and Research 45(5):1154-1159, 2024. doi:10.1093/jbcr/irae115. Crossref verified.
- [zhao2022] Zhao X, et al. The relationship between ignition and oxidation of molten magnesium alloys during the cooling process. Frontiers in Chemistry 10:980860, 2022. doi:10.3389/fchem.2022.980860. Crossref verified.

Standards, codes and regulations

- [cfr1633] 16 CFR 1633.3, Standard for the flammability (open flame) of mattress sets, test criteria. https://www.law.cornell.edu/cfr/text/16/1633.3
- [en1991-1-2] EN 1991-1-2:2002, Eurocode 1, Part 1-2, Annex E. BSI record doi:10.3403/02700262 (approved 28 February 2013). Read through [jrc2012vassart].
- [en2] EN 2:1992 + A1:2004, Classification of fires. BSI record doi:10.3403/00287011 (approved 15 December 1992). Not read.
- [ifc906] International Fire Code, section 906.2. Seen in state adoptions, e.g. https://regulations.justia.com/states/massachusetts/780-cmr/chapter-9/section-906/subsection-906-2
- [iso16733-1] ISO/TS 16733-1:2015, Selection of design fire scenarios and design fires, Part 1. BSI record doi:10.3403/30300271 (approved 31 January 2016). Not read.
- [iso3941-2007] ISO 3941:2007, Classification of fires. https://www.iso.org/standard/40042.html. Not read (HTTP 403).
- [iso3941-2026] ISO 3941:2026, Classification of fires, adding Class L. BSI record doi:10.3403/30482407 (approved 27 January 2026; designation not in the record). Not read.
- [jrc2012vassart] Vassart O. EN 1991-1-2: basic design methods, worked examples. JRC workshop, Brussels, 27-28 November 2012. https://eurocodes.jrc.ec.europa.eu/sites/default/files/2022-06/03a-VASSART-EC-FireDesign-WS.pdf
- [nfpa10] NFPA 10, Standard for Portable Fire Extinguishers, 2022 edition; read in https://docinfofiles.nfpa.org/files/AboutTheCodes/10/10_SDR_A2025.pdf
- [pd7974-1] PD 7974-1:2019, Application of fire safety engineering principles to the design of buildings, Part 1. BSI record doi:10.3403/30357143 (approved 27 March 2019). Not read.
- [sbc801] SBC 801, The Saudi Fire Protection Code, 2024 edition. https://sbc.gov.sa/
- [sfpe2002ignition] SFPE Engineering Guide: Piloted Ignition of Solid Materials Under Radiant Exposure, 2002. Read through [nureg1805s1].
- [ifc2x3exp], [ifc4exp], [ifc43exp], [ifc2x3html], [ifc4html]: buildingSMART schemas; URLs in the catalogue.

Statistics

- [nfpa2025homes] Hall S, McGree B. Home Structure Fires. NFPA, August 2025. https://content.nfpa.org/-/media/Project/Storefront/Catalog/Files/Research/NFPA-Research/Building-and-life-safety/oshomes.pdf; supporting tables [nfpa2025homestables].
- [nfpa2023cooking], [nfpa2022heating], [nfpa2022electrical], [nfpa2025smoking], [nfpa2020candles], [nfpa2017dryers], [nfpa2020schools]: NFPA Research cause and occupancy reports; URLs in the catalogue.
- [usfa2021nonres] US Fire Administration. Nonresidential Building Fires (2017-2019). Topical Fire Report Series 21(9), 2021. https://www.usfa.fema.gov/downloads/pdf/statistics/v21i9.pdf; [usfa2018medical] medical facility snapshot.
- [mhclg2026detailed], [mhclg2026fire0602], [mhclg2026fire0603] MHCLG, Fire statistics, England, year ending March 2026, published 19 August 2026; [homeoffice2021fatalities] in-depth review of fatalities 2010/11 to 2018/19.
- [lfb2026], [opss2025], [fdny2025]: lithium-ion fire statistics; URLs in the catalogue.

Reports, handbooks and data

- [babrauskas1981nbsir] Babrauskas V. Will the Second Item Ignite? NBSIR 81-2271, 1981. doi:10.6028/NBS.IR.81-2271. Crossref verified.
- [babrauskas1982nbsir] Babrauskas V, Lawson JR, Walton WD, Twilley WH. Upholstered Furniture Heat Release Rates Measured with a Furniture Calorimeter. NBSIR 82-2604, 1982. doi:10.6028/NBS.IR.82-2604. Crossref verified.
- [bryant2019tn2077] Bryant RA, Bundy MF. The NIST 20 MW Calorimetry Measurement System for Large-Fire Research. NIST TN 2077, 2019. doi:10.6028/NIST.TN.2077. Crossref verified.
- [cameo] NOAA CAMEO Chemicals. https://cameochemicals.noaa.gov/
- [gannbryner] Gann RG, Bryner NP. Combustion products and their effects on life safety. NFPA Fire Protection Handbook. https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=900093
- [hietaniemi2001vtt] Hietaniemi J, Mangs J, Hakkarainen T. Burning of Electrical Household Appliances: An Experimental Study. VTT Research Notes 2084, 2001. https://sarjaweb.vtt.fi/pdf/tiedotteet/2001/T2084.pdf
- [hu2021tn2133] Hu Y, et al. The Character of Residential Cooktop Fires. NIST TN 2133, 2021. doi:10.6028/NIST.TN.2133. Crossref verified.
- [khan2016sfpe] Khan MM, Tewarson A, Chaos M. Combustion characteristics of materials and generation of fire products. SFPE Handbook, 5th ed., pp. 1143-1232, 2016. doi:10.1007/978-1-4939-2565-0_36. Crossref verified; not read.
- [klotemilke2002] Klote JH, Milke JA. Principles of Smoke Management. ASHRAE, 2002. Read through [nureg1805s1].
- [lawson1984nbsir] Lawson JR, Walton WD, Twilley WH. Fire Performance of Furnishings as Measured in the NBS Furniture Calorimeter, Part I. NBSIR 83-2787, 1984. doi:10.6028/NBS.IR.83-2787. Crossref verified.
- [madrzykowski2004sp1021] Madrzykowski D, Walton WD. Cook County Administration Building Fire: Heat Release Rate Experiments and FDS Simulations. NIST SP 1021, 2004. doi:10.6028/NIST.SP.1021. Crossref verified.
- [nistfcd] Bundy M, Bryant R. NIST Fire Calorimetry Database (FCD). doi:10.18434/mds2-2314. DataCite verified.
- [nistwebbookMgO] NIST Chemistry WebBook, magnesium oxide. https://webbook.nist.gov/cgi/cbook.cgi?ID=C1309484&Mask=2
- [nureg1805s1] US NRC. NUREG-1805 Supplement 1, Fire Dynamics Tools, 2013. https://www.nrc.gov/reading-rm/doc-collections/nuregs/staff/sr1805/s1/index.html
- [fsri2025] UL FSRI research update, 1 May 2025. https://fsri.org/research-update/journal-article-quantifies-fire-hazards-lithium-ion-battery-fires-caused-thermal
- [studioFD02] wiki/research/fire-equations-verified-2026-10-27.json, FD-01 to FD-05, FD-17.
