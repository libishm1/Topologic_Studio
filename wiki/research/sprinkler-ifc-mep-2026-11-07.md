# Open IFC models with fire suppression (sprinklers) for Topologic Studio

Prepared 2026-11-07 for the fire egress work in Topologic Studio. The report covers four things: open IFC models that carry sprinklers, verification counts on every file that could be downloaded, how the IFC schema represents fire suppression, and how architectural and MEP models of one building are federated.

Ground rules followed. No account was created and no agreement was accepted. Downloaded files were treated as data: nothing in them was run. Third-party IFCs went only to `<local corpus>\mep\<source>\`, outside OneDrive, each source in its own new folder. None of them may be copied into the OneDrive repository.

Method. The entity counts in Section 2 are plain `grep -c -i` counts over the STEP text. They count lines, and every file here writes one entity per line, so a line count equals an entity count. The relational facts in Sections 1 and 3 come from a small read-only STEP text parser written for this check (local scripts, not kept in the repository). It does six things:

- follows IfcRelDefinesByType to count typed sprinkler instances;
- follows IfcRelAssignsToGroup to list system members;
- follows IfcRelDefinesByProperties to read property values;
- follows IfcRelContainedInSpatialStructure to count sprinklers per storey or space;
- resolves IfcLocalPlacement chains to world coordinates for the site, building, storeys, walls and sprinklers;
- reads length units, TrueNorth and any IfcMapConversion.

## Summary

- **No openly licensed tall office or apartment building with a sprinkler model was found.** The best verified material is three Revit-exported buildings of 2 to 7 storeys, each with an architectural companion:
  - the NIBS Medical Clinic (418 sprinklers);
  - the Autodesk West Riverside Hospital (1,354 sprinklers);
  - the NIBS Office Building (292 sprinklers).
- **No file uses the standard sprinkler semantics.** None of the roughly 80 files checked uses:
  - PredefinedType `SPRINKLER`;
  - `IfcDistributionSystemEnum.FIREPROTECTION`;
  - `Pset_FireSuppressionTerminalTypeSprinkler`.

  Every sprinkler found comes from a Revit exporter:
  - the type is `IfcFireSuppressionTerminalType` with `.NOTDEFINED.`;
  - the occurrence is `IfcFlowTerminal` (IFC2x3) or `IfcFireSuppressionTerminal` with no PredefinedType (IFC4);
  - fire systems are plain `IfcSystem` objects named "Fire Protection Wet/Dry/Pre-Action", or only a Revit `System Type` property;
  - sprinkler properties, when present at all, sit in `PSet_Revit_Type_Fire Protection` (K-Factor, Temperature Rating, Coverage, Response, Orifice Size) in Revit internal units.

  A Studio parser therefore needs fallbacks beyond the schema-correct path (Section 5.9).
- **The NIBS pairs are the cleanest to federate.** In both the Office and Clinic pairs, the MEP and architectural files share the IfcSite and IfcBuilding placement (origin 0,0,0, no rotation), the storey elevations and the units (metre). The MEP files name the architectural Revit link as the sprinkler host. Without any transform, the sprinkler XY extents fall inside the wall extents of the architectural model.
- **The hospital pair aligns only partly.** The metric architecture (millimetre) and the imperial sprinkler model (foot) share the site rotation (-5 degrees) and the ground-floor elevation (165,811.2 mm = 544 ft). They differ in two ways:
  - storey elevations drift by up to 0.27 m;
  - TrueNorth is encoded differently.

  The hospital architecture also has no IfcSpace.
- **Correction to the local note on `advanced_model.ifc`.** `more\web-ifc-tests\advanced_model.ifc` does have sprinkler instances: 6 `IfcFlowTerminal` typed by its one `IfcFireSuppressionTerminalType` (IFC2x3 pattern, so a search for `IFCFIRESUPPRESSIONTERMINAL(` returns 0). It is the same building as the opensourceBIM "HHS Office" MEP files: identical sprinkler element IDs, storey elevations and site latitude and longitude. Those IDs, elevations and site location point to Autodesk's Revit advanced sample project as the common origin (inference, not confirmed).
- **TalTech DS3 (CC BY 4.0) has no fire protection.** Of its 174 `IfcPipeSegment`:
  - 166 belong to `IfcDistributionSystem` "District Heating/Radiators-Flooring" (PredefinedType HEATING);
  - 1 belongs to "Cooling Ventilation";
  - 7 are unassigned.

## 1. Ranked candidates

The ranking criteria, in order of weight, are: has sprinklers, open licence, tall office or apartment building, architectural companion. Counts were verified on the downloaded files. "Frame" means the coordinate frame relative to the architectural companion.

| Rank | Candidate and source | Licence (exact wording or "none stated") | Building, storeys | Schema, sprinkler file size | Sprinklers (verified) | Fire systems and pipes | Architectural companion and frame | Local path |
|---|---|---|---|---|---|---|---|---|
| 1 | **NIBS Medical Clinic, `NBU_MedicalClinic_Eng-MEP` (Revit MEP 2011 export, 2011-09-08)**, from the DURAARK mirror at TIB: https://tib.eu/data/duraark/BuildingData/01_IFC/NBU_MedicalClinic_ifc.zip | The DURAARK path itself states no licence. buildingSMART's copy of the same NIBS Clinic dataset says: "These files were originally provided under Creative Commons Liscence. These files are provided here under this same liscence. Creative Commons (CC BY 4.0)" (attribution "BSI (2020) "Medical-Dental Test Files," buildingSMART International"). The re3data record for the DURAARK repository lists data licence "CC0". | Real two-storey medical and dental clinic (redacted), US South-West. Storeys: TOF Footing -1.0 m, Level 1 0, Level 2 4.57 m, Roof 9.25 m | IFC2X3. Eng-MEP is 207.3 MB, and the Solibri-optimised copy is 125.8 MB with identical counts | **418** IfcFlowTerminal typed by 1 IfcFireSuppressionTerminalType "15 mm Pendent" (`.NOTDEFINED.`). Every one is contained in an IfcSpace. | No IfcSystem and no ports. Revit `System Type` = "Fire Protection Wet" on 418 terminals, 1,015 IfcFlowSegment, 1,121 IfcFlowFitting and 1 IfcFlowController. | Yes. `NBU_MedicalClinic_Arch.ifc` (269 IfcSpace). The local `dental_clinic.ifc` is byte-identical to buildingSMART's `Clinic_Architectural.ifc`, which is a Solibri-optimised copy of the same 2011-09-06 export. Same site and building origin, storey elevations and units. The sprinklers' Revit `Host` reads "Linked Revit Model : Clinic_A.rvt". Sprinkler XY sits inside the wall extents. **Shared frame verified.** | `<local corpus>\mep\duraark-nbu-medicalclinic\` |
| 2 | **West Riverside Hospital, "Autodesk_Hospital_Sprinkle" (Revit 2015 export, 2016)**, Open IFC Model Repository, University of Auckland: https://openifcmodel.cs.auckland.ac.nz/ (download API `/api/download/<file>`) | The current site states no licence. The archived model page (2020 snapshot) says: "This model is provided by the courtesy of Autodesk Inc. for the purpose of research and other purposes as long as it carries the appropriate attribution to Autodesk Inc." and "Publishable under cc-by-sa-3.0 licence". | Hospital, "a 7 storey hospital building often featured in Autodesk's materials". The architecture has 8 storey entities (Level 1 to 7, plus 7A). The sprinkler model has Levels 1 to 5. | IFC4 edition 34.0 MB and IFC2X3 edition 35.7 MB | **1,354**. IFC4: IfcFireSuppressionTerminal with PredefinedType unset, typed by 2 types "1/2" Pendent" (`.NOTDEFINED.`). IFC2x3: 1,354 IfcFlowTerminal typed the same way. Per storey: 252, 229, 314, 274, 285. | 17 IfcSystem: 12 "Fire Protection Wet", 3 "Fire Protection Pre-Action", 2 "Sanitary". IFC4: 6,228 IfcPipeSegment, 5,900 IfcPipeFitting and 8 IfcValve; the IFC2x3 edition has 6,228 IfcFlowSegment. Full port graph: 24,758 IfcDistributionPort and 12,379 IfcRelConnectsPorts. No property sets beyond Pset_ManufacturerTypeInformation. | Yes, partly aligned. `Autodesk_Hospital_Metric_Architecture` (IFC4, 80.9 MB, mm) has 1,440 walls, 440 doors and 60 IfcStair, but **no IfcSpace**. Same site rotation and ground elevation; units ft against mm; storey drift up to 0.27 m; TrueNorth encoded differently (Section 6.3). The IFC4 sprinkler file also has a dangling reference (#19). | `<local corpus>\mep\auckland-westriverside-hospital\` |
| 3 | **NIBS Office Building, `NBU_OfficeBuilding_Eng-HVAC` (Revit MEP 2011 export, 2011-09-13)**, DURAARK mirror at TIB: https://tib.eu/data/duraark/BuildingData/01_IFC/NBU_OfficeBuilding_ifc.zip | None stated at TIB or on the WBDG page. The buildingSMART CC BY 4.0 statement covers the sister Duplex and Clinic datasets, not the Office. re3data lists "CC0" for the DURAARK repository. | Office: "The two story office building model was developed based on the published sample floor plans for a specific type of mid-size office building built in the United States." Storeys: Level 1 0, Level 2 4.267 m, Roof 7.925 m | IFC2X3. 65.8 MB, optimised copy 41.9 MB | **292** IfcFlowTerminal typed by 1 IfcFireSuppressionTerminalType "15 mm Pendent" (`.NOTDEFINED.`): 151 at z 2.5 m and 141 at z 6.77 m, all contained in IfcSpace. | No IfcSystem and no ports. Revit `System Type` = "Fire Protection Wet" on 292 terminals, 695 IfcFlowSegment and 813 IfcFlowFitting. Type properties include K-Factor 81, Temperature Rating 330.15 (K) and Orifice Size 0.0427 (ft). | Yes. `NBU_OfficeBuilding_Arch-1.ifc` is byte-identical to the local `Office_A_20110811.ifc`. Same origin, storeys and units. `Host` reads "Linked Revit Model : Office_A.rvt". Sprinklers x 1.4 to 48.8 m and y -34.0 to -0.7 m, against walls x -0.5 to 50.0 m and y -36.2 to 0.4 m. **Shared frame verified.** | `<local corpus>\mep\duraark-nbu-office\` |
| 4 | OTC Conference Center (Revit 2015 export), Auckland repository | Same Autodesk courtesy wording as rank 2, archived page; "Publishable under cc-by-sa-3.0 licence" | "7 storey Conference Center building with 2 basement stories", with the MEP model integrated. 12 storey entities, B2 at -29.5 ft up to 8 ROOF at 88 ft. | IFC2X3, 70.3 MB (an IFC4 edition of 71.7 MB exists and was not downloaded) | **47** IfcFlowTerminal typed by 3 types: 1/2" Dry Pendent With Guard, 1/2" Pendent, 1/2" Upright. Only in the lower levels. | 3 IfcSystem: Fire Protection Wet 1 and 2, and Dry 1. Together they hold 93 IfcFlowSegment, 96 IfcFlowFitting and 1 controller; only 20 sprinklers are system members. Ports present. | Integrated in the same file (94 IfcSpace). Units ft. Site placed at (261.2, 97.6) ft. | `<local corpus>\mep\auckland-openifc\` |
| 5 | "HHS Office" MEP (`MEP.ifc`, `MEP2.ifc`), opensourceBIM/IFC-files: https://github.com/opensourceBIM/IFC-files/tree/master/HHS%20Office | Repository README: "Post your IFC data under the Creative Commons Attribution-NoDerivatives license". Provenance doubt: probably derived from Autodesk's Revit advanced sample project. | Three-storey office (Level 1 to 3 plus roof) | IFC2X3, 39.6 MB | **6** IfcFlowTerminal "15 mm Pendent on Drop with Guard" | No fire system. 1,284 IfcFlowSegment (other services). | Yes. The local `more2\osb-ifcfiles-hhs-office\architect.ifc` shares site, storeys and units (mm). | `<local corpus>\mep\osb-ifcfiles-hhs-office\` |
| 6 | SGD Bodø, `SGD_BODO_Eng-HVAC-Plumbing` (Norway), DURAARK mirror | None stated (re3data lists CC0 for the repository) | Building type not stated. Storeys in this file: ground, basement, 1st and 2nd floor (named "Ventilation ...") | IFC2X2_FINAL, 72.8 MB | **0 sprinklers.** 13 IfcFlowTerminal typed by 2 IfcFireSuppressionTerminalType `.HOSEREEL.`, which carry `Pset_FireSuppressionTerminalTypeHoseReel`. Useful as a parser test for non-sprinkler fire terminals. | 1,980 IfcFlowSegment | Arch-1 to Arch-3 in the same zip (IFC2x2 and IFC2x3); not checked for alignment | `<local corpus>\mep\duraark-sgd-bodo\` |

How to choose:

- **For a first Studio integration test**, use rank 3, the NIBS Office. The architectural file is already in the corpus with 99 IfcSpace, the frame is identical, and the sprinkler properties are present.
- **For scale and the IFC4 entity path**, use rank 2, the hospital (1,354 IfcFireSuppressionTerminal and a real port graph). Its architecture lacks spaces, and the federation needs a unit conversion and a storey check.
- **Rank 1 combines the most sprinklers in a NIBS pair with the clearest CC BY 4.0 lineage.** The caveat is that the sprinkler-bearing file itself was obtained from the TIB mirror, whose download path carries no licence text.

## 2. Verification counts

The counts below are `grep -c -i` on each file. Column headings use these short names:

| Column | Pattern counted |
|---|---|
| FST( | `IFCFIRESUPPRESSIONTERMINAL(` |
| FSTType( | `IFCFIRESUPPRESSIONTERMINALTYPE(` |
| PipeSeg( | `IFCPIPESEGMENT(` |
| FlowSeg( | `IFCFLOWSEGMENT(` (the IFC2x3 pipe and duct carrier) |
| DistSys( | `IFCDISTRIBUTIONSYSTEM(` |
| System( | `IFCSYSTEM(` |
| .SPRINKLER. | `.SPRINKLER.` |
| FIREPROTECTION | `FIREPROTECTION` |
| Pset_..Sprinkler | `PSET_FIRESUPPRESSIONTERMINALTYPESPRINKLER` |
| FlowTerm( | `IFCFLOWTERMINAL(` |

The last two rows are the existing local files named in the brief.

| File (under <local corpus>\) | Schema | MB | FST( | FSTType( | PipeSeg( | FlowSeg( | DistSys( | System( | .SPRINKLER. | FIREPROTECTION | Pset_..Sprinkler | FlowTerm( |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mep\auckland-openifc\20160124OTC-Conference Center.ifc | IFC2X3 | 70.3 | 0 | 3 | 0 | 658 | 0 | 101 | 0 | 0 | 0 | 382 |
| mep\auckland-openifc\20191002Mechanical Piping.ifc | IFC2X3 | 43.6 | 0 | 0 | 0 | 3888 | 0 | 461 | 0 | 0 | 0 | 214 |
| mep\auckland-openifc\20200318SD-M-1637-L55-IFC 200318.IFC | IFC2X3 | 60.4 | 0 | 0 | 0 | 287 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\auckland-openifc\20220221KT-ZCB (combined).ifc | IFC2X3 | 117.2 | 0 | 0 | 0 | 1559 | 0 | 136 | 0 | 0 | 0 | 52 |
| mep\auckland-trapelo\20160125RME_2010_Trapelo_const-E20-F.ifc | IFC2X3 | 67.8 | 0 | 0 | 0 | 2095 | 0 | 793 | 0 | 0 | 0 | 2133 |
| mep\auckland-trapelo\20160125RME_2010_Trapelo_const.ifc | IFC4 | 65.3 | 0 | 0 | 506 | 0 | 0 | 793 | 0 | 0 | 0 | 0 |
| mep\auckland-trapelo\20160125Trapelo_Design_Intent.ifc | IFC4 | 27.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\auckland-westriverside-hospital\20160125WestRiverSide Hospital - IFC4-Autodesk_Hospital_Sprinkle.ifc | IFC4 | 34.0 | 1354 | 2 | 6228 | 0 | 0 | 17 | 0 | 0 | 0 | 0 |
| mep\auckland-westriverside-hospital\20160125WestRiverSide Hospital-Ifc2x3-Autodesk_Hospital_Sprinkle_2015.ifc | IFC2X3 | 35.7 | 0 | 2 | 0 | 6228 | 0 | 17 | 0 | 0 | 0 | 1354 |
| mep\auckland-westriverside-hospital\20190104WestRiverSide Hospital - IFC4-Autodesk_Hospital_Metric_Architecture.ifc | IFC4 | 80.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\2.3.0.1_Building-Architecture.ifc | IFC2X3 | 0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\2.3.0.1_Building-Hvac.ifc | IFC2X3 | 0.1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 2 |
| mep\bsi-certification-simple-scene\2.3.0.1_Infra-Plumbing.ifc | IFC2X3 | 0.2 | 0 | 0 | 0 | 24 | 0 | 1 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\4.0.2.1_Building-Architecture.ifc | IFC4 | 0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\4.0.2.1_Building-Hvac.ifc | IFC4 | 0.1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\4.0.2.1_Infra-Plumbing.ifc | IFC4 | 0.4 | 0 | 0 | 24 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\4.3.2.0_Building-Architecture.ifc | IFC4X3_ADD2 | 0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\4.3.2.0_Building-Hvac.ifc | IFC4X3_ADD2 | 0.1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-certification-simple-scene\4.3.2.0_Infra-Plumbing.ifc | IFC4X3_ADD2 | 0.4 | 0 | 0 | 24 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-community-clinic\Clinic_Architectural.ifc | IFC2X3 | 13.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 102 |
| mep\bsi-community-clinic\Clinic_Electrical.ifc | IFC2X3 | 6.8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2081 |
| mep\bsi-community-clinic\Clinic_HVAC.ifc | IFC2X3 | 26.9 | 0 | 0 | 0 | 1548 | 0 | 0 | 0 | 0 | 0 | 440 |
| mep\bsi-community-clinic\Clinic_Plumbing.ifc | IFC2X3 | 55.8 | 0 | 0 | 0 | 2893 | 0 | 0 | 0 | 1 | 0 | 119 |
| mep\bsi-community-duplex\Duplex_A_20110907.ifc | IFC2X3 | 2.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-community-duplex\Duplex_Electrical_20121207.ifc | IFC2X3 | 1.6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 82 |
| mep\bsi-community-duplex\Duplex_MEP_20110907.ifc | IFC2X3 | 17.9 | 0 | 0 | 0 | 427 | 0 | 0 | 0 | 0 | 0 | 105 |
| mep\bsi-community-duplex\Duplex_M_20111024_ROOMS_AND_SPACES.ifc | IFC2X3 | 8.8 | 0 | 0 | 0 | 246 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\bsi-community-duplex\Duplex_Plumbing_20121113.ifc | IFC2X3 | 31.6 | 0 | 0 | 0 | 231 | 0 | 0 | 0 | 1 | 0 | 16 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt-COBie_Arch-Design.ifc | IFC2X3 | 0.5 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 105 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt-COBie_Arch-Handover.ifc | IFC2X3 | 0.6 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 105 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt-COBie_Arch-ProductInstall.ifc | IFC2X3 | 0.6 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 105 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt-COBie_Arch-ProductSelect.ifc | IFC2X3 | 0.6 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 105 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt-COBie_Arch-Programming.ifc | IFC2X3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt_Arch-Optimized.ifc | IFC2X3 | 1.7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt_Arch.ifc | IFC2X3 | 2.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt_Eng-HVAC.ifc | IFC2X3 | 32.1 | 0 | 0 | 0 | 231 | 0 | 0 | 0 | 1 | 0 | 16 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt_Eng-MEP-1.ifc | IFC2X3 | 1.6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 82 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt_Eng-MEP-2.ifc | IFC2X3 | 8.9 | 0 | 0 | 0 | 246 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt_Eng-MEP-Optimized.ifc | IFC2X3 | 11.1 | 0 | 0 | 0 | 427 | 0 | 0 | 0 | 0 | 0 | 105 |
| mep\duraark-nbu-duplex\NBU_Duplex\NBU_Duplex-Apt_Eng-MEP.ifc | IFC2X3 | 18.2 | 0 | 0 | 0 | 427 | 0 | 0 | 0 | 0 | 0 | 105 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Arch-Optimized.ifc | IFC2X3 | 13.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 102 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Arch.ifc | IFC2X3 | 18.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 102 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Eng-CON-Optimized.ifc | IFC2X3 | 19.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Eng-CON.ifc | IFC2X3 | 19.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Eng-ELE.ifc | IFC2X3 | 12.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2081 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Eng-HVAC.ifc | IFC2X3 | 27.4 | 0 | 0 | 0 | 1548 | 0 | 0 | 0 | 0 | 0 | 440 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Eng-MEP-Optimized.ifc | IFC2X3 | 125.8 | 0 | 1 | 0 | 5952 | 0 | 0 | 0 | 0 | 0 | 3053 |
| mep\duraark-nbu-medicalclinic\NBU_MedicalClinic\NBU_MedicalClinic_Eng-MEP.ifc | IFC2X3 | 207.3 | 0 | 1 | 0 | 5952 | 0 | 0 | 0 | 0 | 0 | 3053 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Arch-1.ifc | IFC2X3 | 4.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Arch-2.ifc | IFC2X3 | 0.6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Arch-Optimized.ifc | IFC2X3 | 4.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Eng-CON-Optimized.ifc | IFC2X3 | 11.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Eng-CON.ifc | IFC2X3 | 11.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Eng-ELE.ifc | IFC2X3 | 6.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 870 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Eng-HVAC-Optimized.ifc | IFC2X3 | 41.9 | 0 | 1 | 0 | 1959 | 0 | 0 | 0 | 0 | 0 | 1456 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Eng-HVAC.ifc | IFC2X3 | 65.8 | 0 | 1 | 0 | 1959 | 0 | 0 | 0 | 0 | 0 | 1456 |
| mep\duraark-nbu-office\NBU_OfficeBuilding\NBU_OfficeBuilding_Eng-MEP.ifc | IFC2X3 | 4.1 | 0 | 1 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 1456 |
| mep\duraark-sgd-bodo\SGD_BODO\SGD_BODO_Arch-1.ifc | IFC2X2_FINAL | 23.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-sgd-bodo\SGD_BODO\SGD_BODO_Arch-2.ifc | IFC2X2_FINAL | 23.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-sgd-bodo\SGD_BODO\SGD_BODO_Arch-3.ifc | IFC2X3 | 64.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-sgd-bodo\SGD_BODO\SGD_BODO_Eng-HVAC-Plumbing.ifc | IFC2X2_FINAL | 72.8 | 0 | 2 | 0 | 1980 | 0 | 0 | 0 | 0 | 0 | 57 |
| mep\duraark-sgd-bodo\SGD_BODO\SGD_BODO_Eng-HVAC-Ventilation.ifc | IFC2X2_FINAL | 47.2 | 0 | 0 | 0 | 1869 | 0 | 22 | 0 | 0 | 0 | 322 |
| mep\duraark-sgd-hitos\SGD_HiTOS\SGD_HiTOS_Arch.ifc | IFC2X2_FINAL | 52.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68 |
| mep\duraark-sgd-hitos\SGD_HiTOS\SGD_HiTOS_Eng-ELE.ifc | IFC2X3 | 7.5 | 0 | 0 | 0 | 348 | 0 | 0 | 0 | 0 | 0 | 318 |
| mep\duraark-sgd-hitos\SGD_HiTOS\SGD_HiTOS_Eng-HVAC.ifc | IFC2X3 | 18.5 | 0 | 0 | 0 | 2577 | 0 | 3 | 0 | 0 | 0 | 340 |
| mep\duraark-sgd-munkerud\SGD_Munkerud\SGD_Munkerud_Arch-1.ifc | IFC2X2_FINAL | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-sgd-munkerud\SGD_Munkerud\SGD_Munkerud_Arch-2.ifc | IFC2X2_FINAL | 4.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-sgd-munkerud\SGD_Munkerud\SGD_Munkerud_Arch-3.ifc | IFC2X2_FINAL | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mep\duraark-sgd-munkerud\SGD_Munkerud\SGD_Munkerud_Eng-ELE.ifc | IFC2X2_FINAL | 15.1 | 0 | 0 | 0 | 278 | 0 | 0 | 0 | 0 | 0 | 326 |
| mep\duraark-sgd-munkerud\SGD_Munkerud\SGD_Munkerud_Eng-HVAC-Sanitary.ifc | IFC2X2_FINAL | 10.7 | 0 | 0 | 0 | 454 | 0 | 0 | 0 | 0 | 0 | 42 |
| mep\duraark-sgd-munkerud\SGD_Munkerud\SGD_Munkerud_Eng-HVAC-Ventilation.ifc | IFC2X2_FINAL | 5.5 | 0 | 0 | 0 | 205 | 0 | 4 | 0 | 0 | 0 | 36 |
| mep\duraark-sgd-munkerud\SGD_Munkerud\SGD_Munkerud_Merged.ifc | IFC2X2_FINAL | 35.2 | 0 | 0 | 0 | 937 | 0 | 4 | 0 | 0 | 0 | 404 |
| mep\osb-ifcfiles-hhs-office\MEP.ifc | IFC2X3 | 39.6 | 0 | 1 | 0 | 1284 | 0 | 0 | 0 | 0 | 0 | 725 |
| mep\osb-ifcfiles-hhs-office\MEP2.ifc | IFC2X3 | 37.6 | 0 | 1 | 0 | 1284 | 0 | 0 | 0 | 0 | 0 | 725 |
| mep\osb-testfiles-ifc4mep\ifc4Mep export 17-12-2013_IFC4.ifc | IFC4 | 29.6 | 0 | 0 | 818 | 23 | 15 | 0 | 0 | 0 | 0 | 0 |
| more\web-ifc-tests\advanced_model.ifc | IFC2X3 | 35.3 | 0 | 1 | 0 | 1365 | 0 | 240 | 0 | 0 | 0 | 736 |
| more2\zenodo-taltech-ds3\DS3_TalTech_V4.ifc | IFC4 | 38.7 | 0 | 0 | 174 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |

Reading notes:

- **The `.SPRINKLER.`, `FIREPROTECTION` and Pset_..Sprinkler columns are 0 in every file.** The FIREPROTECTION hits of 1 in some NIBS files are a Revit "FireProtectionClass" text property, not the enumeration.
- **In IFC2x3, sprinklers must be counted relationally,** as `IfcFlowTerminal` typed by `IfcFireSuppressionTerminalType`. The relational counts are in Section 1.
- **The NIBS Office `Eng-MEP.ifc` (4.1 MB) holds the same 292 terminals without geometry or placements.** It is a "BIMServer COBie Deserializer" output. Use `Eng-HVAC` instead.

## 3. Per-candidate notes

### 3.1 NIBS / buildingSMART alliance Common BIM files (Duplex, Office, Clinic)

**Original host.** The WBDG page "Common Building Information Model Files and Tools" (https://www.wbdg.org/bim/cobie/common-bim-files) is now a script-rendered site. I read it through the 2023 Internet Archive copy (page "Updated: 04-27-2021", by USACE ERDC-CERL). For each project it lists:

- COBie spreadsheets;
- an "IFC 2x3 Coordination MVD" file;
- the native RVT;
- per-discipline exchanges: Sparkie (electrical), HVACie (HVAC) and WSie (water systems).

The page lists no fire protection or sprinkler exchange and states no licence. Its download links (`portal.nibs.org/files/wl/?id=...` and `projects.buildingsmartalliance.org/files/?artifact_id=...`) no longer respond.

**buildingSMART copy.** https://github.com/buildingsmart-community/Community-Sample-Test-Files holds the Duplex and Clinic, under CC BY 4.0 as quoted in Section 1, but not the Office.

- Duplex: A, Electrical (Sparkie), MEP, rooms and spaces, Plumbing (WSie).
- Clinic: Architectural, Electrical, HVAC (HVACie), Plumbing (WSie), Structural.

All were downloaded except the Clinic Structural. **None contains a sprinkler.** The Clinic WSie plumbing file carries only a building-level Revit "Text" property set with `SprinklerProtection` = 'YES' (IfcLabel).

**DURAARK mirror.** The FP7 DURAARK project's TIB mirror (`https://tib.eu/data/duraark/BuildingData/01_IFC/NBU_*_ifc.zip`; directory listing forbidden, files open) keeps the earlier 2011 "Eng-MEP" and "Eng-HVAC" coordination exports. **These are the files that contain the sprinklers:**

- **Clinic Eng-MEP:** 418 sprinklers.
- **Office Eng-HVAC:** 292 sprinklers.
- **Duplex:** every variant, 0. A two-unit residential building would not be sprinklered in the US.

The bimdata list (https://github.com/bimdata/BIMData-Research-and-Development/blob/master/pages/IFC_FILES.md) points to the same mirror. It states no licence and mislinks NBU_OfficeBuilding to the Clinic zip.

**Sprinkler data in the 2011 Revit MEP exports.**

- **Occurrence property sets:** `PSet_Revit_Mechanical` holds `System Type` "Fire Protection Wet" and `System Name` "Fire Protection Wet 1". `PSet_Revit_Fire Protection` holds Flow 0.0706 and Pressure Drop 14721.84, both in Revit internal units. `PSet_Revit_Constraints` holds `Host` and `Elevation`.
- **Type property sets:**
  - `PSet_Revit_Type_Fire Protection`:
    - Temperature Rating 330.15, which is kelvin, i.e. 57 °C;
    - K-Factor 81, a metric K presumably in L/min/bar^0.5;
    - Orifice Size 0.04265, which is feet, i.e. 13 mm;
    - Response 0 and Coverage 0 (integers).
  - `PSet_Revit_Type_Dimensions` holds the escutcheon, protrusion and nominal diameter, also in feet as IfcReal.
- **What a reader must handle.** The values are untyped IfcReal in Revit internal units, while the file's declared length unit is metre. A reader must convert them by name, not by the IFC unit assignment.

### 3.2 West Riverside Hospital and other Autodesk-provided models on the Auckland repository

**Repository.** The repository (https://openifcmodel.cs.auckland.ac.nz/) has 172 models. Its JSON listing (`/api/get-ifc-models`) carries no licence fields. The pre-2023 model pages, read through the Internet Archive, carry the Autodesk courtesy statement and "Publishable under cc-by-sa-3.0 licence" for every Autodesk-provided set. The archived pages are:

- West Riverside Hospital: https://web.archive.org/web/2020/http://openifcmodel.cs.auckland.ac.nz/Model/Details/305 (IFC2x3) and `/308` (IFC4);
- OTC Conference Center: `/301`, `/304`;
- Trapelo St Office: `/302`, `/303`;
- Hospital Parking Garage: `/306`, `/307`.

All of them were uploaded in January 2016 by Wawan Solihin.

**Autodesk rule.** These are Autodesk models, and the brief allows downloading Autodesk material only where the terms read allow it. These were downloaded only after reading this statement, which explicitly allows research use with attribution. Attribution to Autodesk Inc. is required in any publication, and CC BY-SA 3.0 requires share-alike for derivatives.

**Hospital disciplines.** The set has 7 discipline files, each in IFC2x3 and IFC4:

- Architecture (~80 MB)
- Mechanical (~73 to 79 MB)
- Plumbing (25 MB)
- Electrical (4.6 MB)
- Fire Alarm (0.9 MB)
- Structural (6.3 MB)
- Sprinkler (34 to 36 MB)

Only the Sprinkler model (both editions) and the IFC4 Architecture were downloaded. The Mechanical model may carry the IfcSpace objects missing from the architecture; it was not checked.

**Hospital sprinkler network.** The IFC4 export is the only verified file whose sprinklers are native `IfcFireSuppressionTerminal`. It is also the only one with a usable flow network:

- every terminal, segment and fitting has ports, connected by `IfcRelConnectsPorts`;
- ports are attached through `IfcRelConnectsPortToElement`, which IFC4 deprecates, not `IfcRelNests`;
- systems are grouped with `IfcRelAssignsToGroup`;
- each system is linked to the building by `IfcRelServicesBuildings`.

The four large wet systems hold 424, 453, 218 and 200 sprinklers.

**OTC Conference Center.** Downloaded before its terms were read. They turned out to be the same courtesy and CC BY-SA 3.0 wording. Its 47 sprinklers cover only part of the basement.

**Trapelo St Office.** "A 3 storey office building where Revit HQ in Waltham is", with Architecture, MEP and Structure. The MEP model (both editions) and the IFC4 architecture were downloaded. **0 sprinklers.** The IFC4 MEP uses 793 IfcSystem and 506 IfcPipeSegment, but no fire protection.

**Other Auckland files** (licence not stated; downloaded for checking):

| File | Origin | Sprinklers |
|---|---|---|
| KT-ZCB (combined) | Revit 2020, 21 storey entities | 0 |
| Mechanical Piping | Revit 2019, contract-style number W912DS18C0023 in the header | 0 |
| SD-M-1637-L55 | Autodesk Fabrication | 0 |

### 3.3 Other sources checked

**buildingSMART Sample-Test-Files.** github.com/buildingSMART/Sample-Test-Files now redirects to `buildingSMART/Certification-datasets` (LICENSE: "This work is licensed under the Creative Commons Attribution 4.0 International License."). It holds 35 IFC files, including "Simple-Scene" Building-Hvac and Infra-Plumbing in IFC2x3, IFC4 and IFC4.3 (all downloaded): 0 sprinklers. The former user-contributed content moved to Community-Sample-Test-Files (Section 3.1). Its "ISO Spec - ReferenceView_V1.2" files, like the Community repository's "ISO Spec archive", hold no fire terminal by name: air terminal, basin and similar examples only.

**KIT IFC examples.** https://www.ifcwiki.org/index.php?title=KIT_IFC_Examples lists FZK-Haus, the Office Building (AC20 Institute Var 2), Smiley West, a bridge and a road, "for unrestricted use". None is described as containing building services. The local copies show no fire terminals.

**IfcOpenShell test files.** https://github.com/IfcOpenShell/files has no licence file. It holds 271 IFC files, which are bug reproductions; four names mention MEP (duct fitting, pipe segment, flow controller, flow segment) and none mentions a sprinkler. Not downloaded.

**opensourceBIM.**

- `IFC-files` (CC BY-ND 4.0): the HHS Office MEP (rank 5).
- `TestFiles` (GPL-3.0-or-later): `ifc4Mep export 17-12-2013_IFC4.ifc` (DDS-CAD 9, IFC4) has 15 IfcDistributionSystem and 818 IfcPipeSegment. It is a good example of typed systems (SEWAGE, DOMESTICCOLDWATER, HEATING, VENTILATION, AIRCONDITIONING), but has no fire protection.

**Zenodo.** I searched the API for "IFC sprinkler", "IfcFireSuppressionTerminal" (0 hits), "IFC MEP model", "fire protection IFC", "IFC building services dataset", "sprinkler BIM" and similar queries. **No record with a sprinkler model was found.**

- **GNI BIM Dataset** (record 19722012, CC BY 4.0, 224 TUM student models): its published class histogram shows no IfcFireSuppressionTerminal and only 5 IfcPipeSegment in total. Not downloaded (zips of 503 MB and 221 MB).
- **RestageIFC** (21470658, CC BY 4.0, 190 MB): structure and assembly only.
- **BIM4EEB** (6685810): architectural.

**TalTech and DURAARK Norwegian sets.** TalTech DS3 is covered in Section 4. The DURAARK Norwegian sets (`SGD_*`) gave these results:

- HiTOS (Tromsø): HVAC and ELE, 0 sprinklers;
- Munkerud: 0;
- Bodø: 13 hose reels (rank 6).

The GNI dataset's link list (`other_online_BIM_model_resources.csv`) was also scanned. It gives an FTP source for the full HiTOS set (ftp://ftp.buildingsmart.no/pub/ifcfiles/HITOS/), which was not tried (FTP). It also lists SimAUD datasets ("Copyright © 2010-2019 SimAUD"; architecture, electrical, mechanical), which were not checked.

**Not downloaded, by rule or because access needs an account, form or app:**

- **Autodesk Revit sample projects** (Snowdon Towers, rme_advanced_sample_project). The Autodesk help pages read:
  - https://help.autodesk.com/cloudhelp/2024/ENU/Revit-GetStarted/files/GUID-61EF2F22-3A1F-4317-B925-1E85F138BE88.htm
  - https://help.autodesk.com/cloudhelp/2024/ENU/Revit-GetStarted/files/GUID-5DA26FF7-85B3-4E9D-A9F2-FADDD02D88D6.htm

  These pages contain no licence or terms-of-use text; their only usage wording is "Use these files as an example of a complete coordinated project with Revit models from all disciplines." The files are native RVT, not IFC.
  - **Snowdon Towers:** Architectural, Site, Facades, Structural, Plumbing, HVAC and Electrical; no fire protection model is listed. "All models use the same coordinate system". It is a mixed-use building in Brownsville, PA.
  - **rme_advanced_sample_project:** whether it contains sprinklers is not stated. The local `advanced_model.ifc` and the HHS Office MEP files are probably exports of this project, and carry 6 sprinklers.
- **BIMcollab example project** (nine discipline IFCs, delivered with BIMcollab Zoom or a trial space): no licence found.
- **Molio-Huset:** download needs a form.
- **Kaggle collections** (for example "5000 IFC/RVT projects"): account required.

## 4. Local corpus note (TalTech and advanced_model)

**TalTech DS3** (`<local corpus>\more2\zenodo-taltech-ds3\DS3_TalTech_V4.ifc`):

- **Source:** Zenodo record 15782433, "SmartlivingEPC public building BIM model and measurements" (DOI 10.5281/zenodo.15782433; Veliskaki, Koltsios, Giannopoulos, Klonis, Bazakas, Papadopoulou, Kuivjõgi, Võsa, Ferrantelli, Kurnitski, Ioannidis, Tzovaras).
- **Licence:** `cc-by-4.0`, i.e. Creative Commons Attribution 4.0 International.
- **Model:** IFC4, Revit 2023, 5 storeys (+Kelder, 1. to 3. korrus, Katus).
- **Pipes:** all 174 `IfcPipeSegment` are of family "Pipe Types". 166 are members of `IfcDistributionSystem` "District Heating/Radiators-Flooring" (PredefinedType HEATING), 1 of "Cooling Ventilation" (HEATING), and 7 belong to no system. The heating system also holds 166 IfcPipeFitting, 44 IfcSpaceHeater and 1 IfcBoiler.
- **Fire content:** none.

**advanced_model.ifc** (`<local corpus>\more\web-ifc-tests\advanced_model.ifc`; IFC2x3, Revit 2021):

- Its one IfcFireSuppressionTerminalType ("M_Sprinkler - Pendent - Hosted:15 mm Pendent on Drop with Guard", `.NOTDEFINED.`) types **6 IfcFlowTerminal instances** at z 6,250 mm.
- Its storey elevations (94.172, 3,800, 7,300 and 10,900 mm) and site latitude and longitude (48°7'58.8"N, 11°34'58.8"E) match the HHS Office `MEP.ifc`, and the same Revit element IDs (545673, 545696) appear in both.
- Its building origin is offset by about (12,282, 12,229) mm against the HHS export. That offset is a live example of two exports of one building with different coordinate bases (Section 6.3).

## 5. How IFC represents fire suppression

Schema facts below were read from the official buildingSMART pages, which sit behind a Cloudflare check. Internet Archive snapshots were used where the live host refused; URLs are in Section 7.

### 5.1 Entities

- **Inheritance (IFC4 and IFC4.3):** `IfcFireSuppressionTerminal` is SUBTYPE OF `IfcFlowTerminal` < IfcDistributionFlowElement < IfcDistributionElement < IfcElement < IfcProduct. It is "New entity in IFC4". Its definition: "delivering a fluid (gas or liquid) that will suppress a fire ... all forms of sprinkler, spreader and other form of terminal that is connected to a pipework system".
- **Occurrence attribute 9:** `PredefinedType : OPTIONAL IfcFireSuppressionTerminalTypeEnum`. It is not used when a typing object with a PredefinedType exists.
- **Occurrence WHERE rules:**
  - `CorrectPredefinedType`: USERDEFINED requires ObjectType.
  - `CorrectTypeAssigned`: the type, if any, must be an IfcFireSuppressionTerminalType.
- **Type:** `IfcFireSuppressionTerminalType` is SUBTYPE OF `IfcFlowTerminalType`. Its `PredefinedType` is mandatory, and USERDEFINED requires ElementType.
- **IFC4 deprecation:** `IfcFlowTerminal` itself was "deprecated for instantiation" in IFC4. Readers should still accept it, because IFC2x3-style exports keep using it.

### 5.2 IfcFireSuppressionTerminalTypeEnum

| Value | IFC2x3 TC1 | IFC4 ADD2 TC1 | IFC4.3 ADD2 |
|---|---|---|---|
| BREECHINGINLET | yes | yes | yes |
| FIREHYDRANT | yes | yes | yes |
| FIREMONITOR | no | no | yes (new) |
| HOSEREEL | yes | yes | yes |
| SPRINKLER | yes | yes | yes ("Device for sprinkling water from a pipe under pressure over an area.") |
| SPRINKLERDEFLECTOR | yes | yes | deprecated in 4.3 |
| USERDEFINED, NOTDEFINED | yes | yes | yes |

### 5.3 IFC2x3 pattern and what a parser must check

IFC2x3 has no `IfcFireSuppressionTerminal` occurrence. A sprinkler is an `IfcFlowTerminal` typed through `IfcRelDefinesByType` by an `IfcFireSuppressionTerminalType` (new in IFC2x2). A parser should:

1. Take every IfcFlowTerminal, and in IFC4 or later also every IfcFireSuppressionTerminal.
2. Follow IsDefinedBy to the single IfcRelDefinesByType; IfcObject WR1 allows at most one.
3. Accept the instance when RelatingType is an IfcFireSuppressionTerminalType with PredefinedType SPRINKLER.
4. Because every real file here uses NOTDEFINED, also accept NOTDEFINED or USERDEFINED types whose Name, ElementType or the occurrence Name matches `sprinkler|pendent|upright|sidewall|concealed`, and exclude HOSEREEL, FIREHYDRANT and BREECHINGINLET.
5. Read properties from the type's HasPropertySets and the occurrence's IfcRelDefinesByProperties.
6. In IFC2x3, read systems as plain `IfcSystem` (IfcGroup) through IfcRelAssignsToGroup. There is no system enumeration, so fire systems can only be recognised by name.

### 5.4 Network and ports

- **Port attachment:** IFC4 replaces `HasPorts` (IfcRelConnectsPortToElement) with `NestedBy` (IfcRelNests). Revit 2015 IFC4 exports still use IfcRelConnectsPortToElement, as the hospital file shows.
- **IfcDistributionPort (IFC4):** FlowDirection, PredefinedType and SystemType (`IfcDistributionSystemEnum`). Two ports connect only with the same SystemType and opposite directions.
- **Port nesting table for the terminal (IFC4):**

| PredefinedType | Port | Flow | SystemType | Meaning |
|---|---|---|---|---|
| SPRINKLER | Line | SINK | FIREPROTECTION | supply line |
| FIREHYDRANT | Line | SINK | FIREPROTECTION | supply line |
| FIREHYDRANT | Hose | SOURCE | FIREPROTECTION | hose connection |

- **Network elements:** the rest of the network is `IfcPipeSegment`, `IfcPipeFitting` and `IfcValve` (or IfcFlowSegment, IfcFlowFitting and IfcFlowController in IFC2x3), joined port to port by `IfcRelConnectsPorts`.
- **Sensors and flow switches:** attached through `IfcRelFlowControlElements`. `IfcSensorTypeEnum` includes FIRESENSOR, HEATSENSOR, SMOKESENSOR and FLOWSENSOR; FLOWSENSOR is the closest fit for a sprinkler flow switch (interpretation).
- **Valves:** `IfcValveTypeEnum` has no alarm or deluge valve value.

### 5.5 Pset_FireSuppressionTerminalTypeSprinkler (IFC4.3 ADD2)

Applies to IfcFireSuppressionTerminal and IfcFireSuppressionTerminalType with PredefinedType SPRINKLER (template type PSET_TYPEDRIVENOVERRIDE).

| Property | Property entity | Data type | Values / definition |
|---|---|---|---|
| SprinklerType | IfcPropertyEnumeratedValue | PEnum_SprinklerType | CEILING, CONCEALED, CUTOFF, PENDANT, RECESSEDPENDANT, SIDEWALL, UPRIGHT, OTHER, NOTKNOWN, UNSET |
| Activation | IfcPropertyEnumeratedValue | PEnum_SprinklerActivation | BULB, FUSIBLESOLDER, OTHER, NOTKNOWN, UNSET |
| Response | IfcPropertyEnumeratedValue | PEnum_SprinklerResponse | QUICK, STANDARD, NOTKNOWN, UNSET |
| ActivationTemperature | IfcPropertySingleValue | IfcThermodynamicTemperatureMeasure | temperature at which it is designed to activate |
| CoverageArea | IfcPropertySingleValue | IfcAreaMeasure | area the sprinkler is designed to protect |
| HasDeflector | IfcPropertySingleValue | IfcBoolean | deflector (baffle) fitted to diffuse discharge |
| BulbLiquidColour | IfcPropertyEnumeratedValue | PEnum_SprinklerBulbLiquidColour | BLUE, GREEN, MAUVE, ORANGE, RED, YELLOW, OTHER, NOTKNOWN, UNSET |
| DischargeFlowRate | IfcPropertySingleValue | IfcVolumetricFlowRateMeasure | volumetric rate of discharge |
| ResidualFlowingPressure | IfcPropertySingleValue | IfcPressureMeasure | residual pressure at which the discharge rate is determined |
| DischargeCoefficient | IfcPropertySingleValue | IfcReal | coefficient of flow at the sprinkler (the K-factor slot) |
| MaximumWorkingPressure | IfcPropertySingleValue | IfcPressureMeasure | maximum pressure it withstands |
| ConnectionSize | IfcPropertySingleValue | IfcPositiveLengthMeasure | inlet connection size |

Version differences:

- **IFC2x3:** the same 12 properties, with these differences:
  - `BulbLiquidColor` (US spelling);
  - two extra IfcPropertyReferenceValue properties to IfcMaterial: `FrameMaterial` and `DeflectorMaterial`;
  - the property set applies only to the type;
  - enum items are written in mixed case (Ceiling, Cut-off, Pendant, Quick, Standard, Bulb, FusibleSolder).
- **IFC4:** the same 12 properties, with `BulbLiquidColor`, upper-case items and no material references.
- **IFC4.3:** renamed the property to `BulbLiquidColour`.
- **All versions:** the flow coefficient is `DischargeCoefficient`; there is no "DischargeFlowCoefficient".
- **Pset_FireSuppressionTerminalTypeCommon** (IFC4 and 4.3 only): Reference (deprecated in 4.3) and Status.

### 5.6 Systems

- **`IfcDistributionSystem`** is new in IFC4. It is SUBTYPE OF IfcSystem, with `LongName` and `PredefinedType : OPTIONAL IfcDistributionSystemEnum`; members come through IfcRelAssignsToGroup, and hierarchy through IfcRelAggregates. `IfcDistributionCircuit` partitions a system.
- **`IfcDistributionSystemEnum` (IFC4):** 42 values plus USERDEFINED and NOTDEFINED, among them FIREPROTECTION ("Fire protection sprinkler system."), DOMESTICCOLDWATER, HEATING, VENTILATION, SECURITY and SIGNAL. IFC4.3 adds 7 rail and telecom values.
- **No other fire value:** FIREPROTECTION is the only fire value, so a fire alarm system would need SECURITY, SIGNAL or USERDEFINED (interpretation).
- **IFC2x3:** has only IfcSystem, and links it to buildings through IfcRelServicesBuildings.
- **`IfcPipeSegmentTypeEnum`:** has no fire value; a pipe's fire role comes from its system or its port SystemType.

### 5.7 Pset_SpaceFireSafetyRequirements

| Property | Type | 2x3 | IFC4 | 4.3 |
|---|---|---|---|---|
| MainFireUse, AncillaryFireUse, FireHazardFactor | IfcLabel | yes | removed | no |
| FireRiskFactor | IfcLabel | yes | yes | yes |
| FlammableStorage | IfcBoolean | yes | yes | yes |
| FireExit | IfcBoolean ("designed to serve as an exit in fire") | yes | yes | yes |
| SprinklerProtection | IfcBoolean ("object is sprinkler protected") | yes | yes | yes |
| SprinklerProtectionAutomatic | IfcBoolean (only given if SprinklerProtection is TRUE) | yes | yes | yes |
| AirPressurization | IfcBoolean | yes | yes | yes |

Applicability by version:

- **2x3:** IfcSpace and IfcZone.
- **IFC4:** adds IfcSpatialZone.
- **IFC4.3:** IfcSpatialElement, IfcSpatialZone, IfcZone and their types.

**None of the architectural companions here uses this property set.** The NIBS files carry `SprinklerProtection` only as a Revit project "Text" property ('YES' or 'FALSE' as IfcLabel).

### 5.8 Related property sets (IFC4.3)

- **Hose reel:** `Pset_FireSuppressionTerminalTypeHoseReel` (HoseReelType, HoseReelMountingType, HoseNozzleType, InletConnectionSize, HoseDiameter, HoseLength, ClassOfService, ClassificationAuthority). The Bodø file uses its IFC2x2 form.
- **Hydrant and inlet:** `...FireHydrant` and `...BreechingInlet`.
- **Pipes:** `Pset_PipeSegmentTypeCommon` (NominalDiameter, InnerDiameter, OuterDiameter, WorkingPressure, PressureRange, TemperatureRange).
- **Systems:** `Pset_DistributionSystemCommon` holds only Reference, which is deprecated. No fire-specific system property set exists.

### 5.9 What real exports contain, and the reader Studio needs

| Need | Schema-correct source | What the verified files actually have |
|---|---|---|
| Is it a sprinkler? | IfcFireSuppressionTerminal(/Type) with SPRINKLER | Type `.NOTDEFINED.`. Names such as "M_Sprinkler - Pendent - Hosted:15 mm Pendent", "Sprinkler - Pendent - Hosted:1/2" Pendent", "1/2" Upright", "1/2" Dry Pendent With Guard". |
| Position | ObjectPlacement and Body geometry | Present in all files except the NIBS COBie-derived `Eng-MEP.ifc`. NIBS terminals are contained in IfcSpace, not the storey, so the storey must be found through the space's decomposition. |
| Activation temperature | ActivationTemperature (IfcThermodynamicTemperatureMeasure) | NIBS: `Temperature Rating` = 330.15 (IfcReal, kelvin). Hospital and OTC: absent. |
| K-factor | DischargeCoefficient (IfcReal) | NIBS: `K-Factor` = 81 (IfcReal). Hospital: absent. |
| Response | Response (QUICK, STANDARD) | NIBS: `Response` = 0 (IfcInteger code). |
| Coverage | CoverageArea (IfcAreaMeasure) | NIBS: `Coverage` = 0 (IfcInteger code). |
| System membership | IfcDistributionSystem FIREPROTECTION | IfcSystem named "Fire Protection Wet/Dry/Pre-Action" (hospital, OTC), or only Revit `System Type` = "Fire Protection Wet" (NIBS). |
| Network topology | ports and IfcRelConnectsPorts | Hospital and OTC only. |

## 6. Federation of architectural and MEP models

### 6.1 Mechanisms in the schema

- **IFC2x3:**
  - the project-wide frame is `IfcGeometricRepresentationContext.WorldCoordinateSystem`, plus an optional `TrueNorth` (a 2D direction of geographic north in that frame);
  - the site is placed by `IfcSite.ObjectPlacement`, which carries any shift or rotation from a shared point;
  - `IfcSite.RefLatitude`, `RefLongitude` and `RefElevation` locate the site only coarsely.

  There is no map conversion entity. The buildingSMART Australasia georeferencing guide fills the gap with the property sets `ePset_MapConversion` and `ePset_ProjectedCRS`.
- **IFC4:**
  - `IfcMapConversion` (SUBTYPE OF IfcCoordinateOperation) links the context (`SourceCRS`) to an `IfcProjectedCRS` (`TargetCRS`, for example an EPSG code). Its attributes are `Eastings`, `Northings`, `OrthogonalHeight`, `XAxisAbscissa`, `XAxisOrdinate` and an optional `Scale`;
  - it hangs off the context through `HasCoordinateOperation`;
  - `IfcProjectedCRS` adds `GeodeticDatum`, `VerticalDatum`, `MapProjection`, `MapZone` and `MapUnit`.
- **IFC4.3 ADD2:**
  - adds `IfcMapConversionScaled` (separate `FactorX`, `FactorY` and `FactorZ`) and `IfcRigidOperation`;
  - keeps IfcMapConversion as the common case.
- **Levels of georeferencing:** the LoGeoRef scheme (Clemen and Görne 2019) grades files from 10 (address only) through 20 (RefLatitude/Longitude), 30 (shifted site placement) and 40 (TrueNorth and WCS) to 50 (IfcMapConversion).
  - **The Revit-exported building files here sit at LoGeoRef 20 to 40.** None of the sprinkler-bearing files, their architectural companions or the other downloads carries an IfcMapConversion or ePset_MapConversion.
  - **The only files with IfcMapConversion are the buildingSMART Certification "Simple-Scene" IFC4 and IFC4.3 files** (`<local corpus>\mep\bsi-certification-simple-scene\`), all with `IfcProjectedCRS('EPSG:32760', ... 'WGS 84' ...)`. Building-Architecture and Building-Hvac share one IfcMapConversion: Eastings 729013.349, Northings 9063992.685, OrthogonalHeight 1.3, XAxisAbscissa 0.5 and XAxisOrdinate 0.866 (a 60 degree rotation), Scale 0.001 (mm to m). Infra-Plumbing uses a different one (729011.226, 9063960.608, 0, no rotation, 0.001).
  - **Studio test case.** These are a small CC BY 4.0 pair for testing federation through IfcMapConversion: a building model and an infrastructure model in different local frames that only line up after both map conversions are applied. They contain no sprinklers.

### 6.2 How discipline models are aligned in practice

- **Common reference point.** Each discipline model is authored against the same reference point. In Revit that is "shared coordinates", taken from a survey point or project base point agreed in the BIM execution plan, or simply the same internal origin when the consultants link the architect's model "origin to origin".
- **Consistent export settings.** Each discipline exports IFC with the same coordinate base option. In Revit's IFC exporter the choices are shared coordinates, internal coordinates, project base point or survey point.
- **Result.** If both models use the same base, the IfcSite placement, storey elevations and TrueNorth match, and a viewer can load both without any transform. That is the NIBS pattern here: the MEP models were authored with the architectural Revit model linked (the sprinklers' `Host` property names "Linked Revit Model : Office_A.rvt" and "Clinic_A.rvt"), and both disciplines were exported the same way.
- **Coordination checks.** In coordination tools (Solibri, Navisworks, BIMcollab Zoom, That Open), the check is a visual overlay plus clash detection. A mismatch shows as a uniform offset, a rotation about the site origin, or a vertical step at one storey.

### 6.3 What goes wrong, with examples from these files

1. **Different coordinate bases at export.** The same Revit building exported twice (HHS `MEP.ifc` in 2010, `advanced_model.ifc` in 2021) differs by about 12.3 m in both X and Y. Nothing in either file records the shift.
2. **Unit mismatch.** The hospital architecture is in millimetres and the sprinkler model in feet. IFC carries the units, but any reader that assumes metres, or applies a single scale to a federation, misplaces one model by a factor of 304.8.
3. **TrueNorth encoded two ways.** The two hospital models encode north differently:
   - the architecture rotates both its site placement (-5 degrees) and its context TrueNorth;
   - the sprinkler model rotates only the site placement and leaves TrueNorth as +Y.

   A viewer that applies TrueNorth as an extra rotation would turn one model 5 degrees against the other. Topologic Studio should use placements for geometry and treat TrueNorth as metadata.
4. **Storey drift between editions.** The hospital's metric architecture and imperial MEP were evidently modelled separately. Their Level 1 elevations agree (544 ft = 165,811.2 mm), but higher levels drift. Assigning sprinklers to architectural storeys by elevation band needs a tolerance, or must use the MEP model's own storeys.

   | Level | MEP minus architecture |
   |---|---|
   | 2 | +96 mm |
   | 3 | -27 mm |
   | 4 | -150 mm |
   | 5 | -274 mm |

5. **Storey names differ while elevations agree.** NIBS Clinic: MEP "Level 1" against architectural "First Floor". Matching must be by elevation, not by name.
6. **Containment in spaces, and duplicated spaces.** Revit MEP exports put terminals in IfcSpace, and the MEP file carries its own, larger set of spaces (NIBS Office 198 against 99 in the architecture; Clinic 526 against 269). A federated reader should map MEP spaces to architectural spaces by GlobalId or geometry, not trust either set alone.
7. **Default georeference.** The NIBS and Autodesk files carry Revit's default site location (42°21'30"N, 71°03'35"W, Boston), or another location that differs between the paired files. RefLatitude/RefLongitude cannot be used to align these files.
8. **Large coordinates in web viewers.** Large world coordinates lose precision in 32-bit GPU buffers. web-ifc's `COORDINATE_TO_ORIGIN` moves each model by its own offset, which breaks federation unless every model gets the same offset (That Open's coordination matrix taken from the first model). This matches the project memory note on viewer frame alignment.
9. **Invalid files.** The hospital IFC4 sprinkler export references a non-existent entity (#19, in an IfcAxis2Placement3D). Strict readers may reject it; tolerant readers should default the missing axis.

### 6.4 Suggested federation check for Studio

Before overlaying an MEP model on an architectural one, compare the following:

1. length units, normalised to metres;
2. the IfcSite and IfcBuilding world placement, origin and rotation;
3. context TrueNorth;
4. any IfcMapConversion. When both files have one, align through it, as in the buildingSMART Simple-Scene files; when only one has it, fall back to placements and warn;
5. storey elevations, matched by elevation within a tolerance (e.g. 0.3 m);
6. the overlap of the sprinkler XY bounding box with the wall bounding box of the architecture.

Accept the pair without a transform when items 1 to 3 agree and item 6 overlaps. Otherwise warn and show the offset. This is the procedure used for the frame checks in Section 1.

## 7. Sources

**Datasets and licence pages**

- WBDG Common BIM Files (archived 2023): https://web.archive.org/web/2023/https://www.wbdg.org/bim/cobie/common-bim-files
- buildingSMART Community Sample Test Files (Duplex and Clinic READMEs with CC BY 4.0): https://github.com/buildingsmart-community/Community-Sample-Test-Files
- buildingSMART Certification datasets (formerly Sample-Test-Files), with LICENSE CC BY 4.0: https://github.com/buildingSMART/Certification-datasets
- DURAARK mirror at TIB: https://tib.eu/data/duraark/BuildingData/01_IFC/ (files NBU_MedicalClinic_ifc.zip, NBU_OfficeBuilding_ifc.zip, NBU_Duplex_ifc.zip, SGD_BODO_ifc.zip, SGD_HiTOS_ifc.zip, SGD_Munkerud_ifc.zip)
- re3data record for DURAARK datasets: https://www.re3data.org/repository/r3d100012506 (API: https://www.re3data.org/api/v1/repository/r3d100012506)
- BIMData IFC file list: https://github.com/bimdata/BIMData-Research-and-Development/blob/master/pages/IFC_FILES.md
- Open IFC Model Repository, Auckland: https://openifcmodel.cs.auckland.ac.nz/ ; archived model pages: https://web.archive.org/web/2020/http://openifcmodel.cs.auckland.ac.nz/Model/Details/305 (also 301 to 308)
- opensourceBIM IFC-files: https://github.com/opensourceBIM/IFC-files ; opensourceBIM TestFiles: https://github.com/opensourceBIM/TestFiles
- IfcOpenShell files: https://github.com/IfcOpenShell/files
- KIT IFC Examples: https://www.ifcwiki.org/index.php?title=KIT_IFC_Examples
- Zenodo TalTech record: https://zenodo.org/records/15782433 ; GNI BIM Dataset: https://zenodo.org/records/19722012 ; RestageIFC: https://zenodo.org/records/21470658
- Autodesk Revit sample project pages: https://help.autodesk.com/cloudhelp/2024/ENU/Revit-GetStarted/files/GUID-61EF2F22-3A1F-4317-B925-1E85F138BE88.htm and https://help.autodesk.com/cloudhelp/2024/ENU/Revit-GetStarted/files/GUID-5DA26FF7-85B3-4E9D-A9F2-FADDD02D88D6.htm

**Schema (buildingSMART official pages, read via Internet Archive snapshots)**

- IFC4.3 ADD2 (base https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/):
  - IfcFireSuppressionTerminal.htm, IfcFireSuppressionTerminalType.htm, IfcFireSuppressionTerminalTypeEnum.htm
  - Pset_FireSuppressionTerminalTypeSprinkler.htm, Pset_FireSuppressionTerminalTypeCommon.htm
  - PEnum_SprinklerType.htm, PEnum_SprinklerActivation.htm, PEnum_SprinklerResponse.htm, PEnum_SprinklerBulbLiquidColour.htm
  - Pset_SpaceFireSafetyRequirements.htm
  - IfcDistributionSystemEnum.htm, IfcDistributionSystem.htm, IfcDistributionPort.htm, IfcRelConnectsPorts.htm
  - IfcMapConversion.htm, IfcProjectedCRS.htm
- IFC4 ADD2 TC1 (base https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/schema/):
  - ifcplumbingfireprotectiondomain/lexical/ifcfiresuppressionterminal.htm
  - ifcplumbingfireprotectiondomain/lexical/ifcfiresuppressionterminaltypeenum.htm
  - ifcplumbingfireprotectiondomain/pset/pset_firesuppressionterminaltypesprinkler.htm
  - ifcproductextension/pset/pset_spacefiresafetyrequirements.htm
  - ifcsharedbldgserviceelements/lexical/ifcdistributionsystemenum.htm
  - ifcrepresentationresource/lexical/ifcmapconversion.htm
- IFC2x3 TC1 (base https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/):
  - ifcplumbingfireprotectiondomain/lexical/ifcfiresuppressionterminaltype.htm
  - ifcplumbingfireprotectiondomain/lexical/ifcfiresuppressionterminaltypeenum.htm
  - psd/IfcPlumbingFireProtectionDomain/Pset_FireSuppressionTerminalTypeSprinkler.xml
  - ifcsharedbldgserviceelements/lexical/ifcflowterminal.htm
  - ifcproductextension/lexical/ifcsystem.htm
- Source tree: https://github.com/buildingSMART/IFC4.x-development (branch ifc4.3-main)

**Federation**

- buildingSMART Australasia, User Guide for Geo-referencing in IFC, v2.0 (2020): https://www.buildingsmart.org/wp-content/uploads/2020/02/User-Guide-for-Geo-referencing-in-IFC-v2.0.pdf
- Clemen, C. and Görne, H. (2019), "Level of Georeferencing (LoGeoRef) using IFC for BIM", Journal of Geodesy, Cartography and Cadastre, 10, 15-20.

Not re-read in this session (the live hosts returned a Cloudflare challenge): the IfcMapConversion and IfcProjectedCRS attribute lists are given from the published schema as known, and should be checked against the pages above before citing them in a paper.
