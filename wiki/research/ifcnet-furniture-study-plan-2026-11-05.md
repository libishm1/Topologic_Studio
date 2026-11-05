# Investigating furniture sub-type identification in IFC models from browser-computable geometry: licences, novelty, design and a pilot

Date: 2026-11-05
Status: phase 1 (research, design and pilot). No full training campaign was run; Section 6 gives its estimate for approval. New files only: `tools/identify-features.mjs`, `tools/identify-pilot.py`, this note and [`benchmark/results/2026-11-05-ifcnet-pilot.json`](../../benchmark/results/2026-11-05-ifcnet-pilot.json), which every number below traces to unless another file is named. No application file was changed.
Related: [ifc-identification-benchmark-2026-11-05.md](ifc-identification-benchmark-2026-11-05.md) (the baselines reused here), [ignition-sources-2026-11-05.md](ignition-sources-2026-11-05.md) (the ignition scenarios that define which furniture sub-types matter).

Conventions. Third-party IFC models and IFCNet's data and weights were used locally only; the repository holds aggregate numbers, never a file, an element name or a per-element result. Licences and terms were read on the source pages on 2026-11-05 and are quoted where short. Every DOI was resolved at Crossref (DataCite for arXiv and data DOIs) and again at OpenAlex on 2026-11-05. Text from papers and web pages was treated as data. The methods named after Poux's work were implemented here from the published papers; no code was copied from any repository.

---

## 0. Summary

The study asked whether IFCNet's shape classifier can be improved for furniture, whether the geometric feature methods of Poux and of the point-cloud literature identify furniture sub-types in IFC models, and whether such a model could be published under an open licence. The licences, read at source (Section 2), settle the last question first: IFCNet's weights and dataset carry no licence, so weights derived from them cannot be published openly; Poux's 3D-Data-Science code is CC BY-NC 4.0 for readers who do not own his book, so its methods were reimplemented from the papers; and openly licensed training data exist, in CC0 and CC BY furniture libraries, the CC BY and CC0 part of Objaverse, ABO under CC BY 4.0, BIMGEOM under CC0 and 18 of the 45 corpus models. The literature (Section 3) leaves a narrow gap: no indexed work identifies ignition-relevant furniture sub-types from IFC, computes eigenvalue features on IFC elements or runs such a classifier in the browser, while the fusion of names and geometry is already published.

The pilot (Section 5) held out every building in turn and gave three results. First, 71 geometric features, computed in a median 2.7 ms an element, identified the 18 element labels with accuracy 0.759 and macro-F1 0.669 without any name, against 0.413 and 0.349 for IFCNet and 0.622 and 0.498 for a linear probe on IFCNet's embedding; the eigenvalue, surface and segment features added 0.108 accuracy and 0.142 macro-F1 to the bounding box, with 95 % intervals that exclude zero. Second, the name rules followed by the feature model, used only where the rules fall back to the box, raised accuracy from 0.860 to 0.882 and macro-F1 from 0.829 to 0.873 on typed elements, but lowered accuracy on the hand-labelled proxies from 0.604 to 0.528, because the model never saw a proxy, an appliance or a sanitary fitting in training. Third, ten furniture sub-types were identified from geometry with accuracy 0.710 and macro-F1 0.594 on 649 elements from 19 buildings, no better than the bounding box alone (0.709 and 0.592) at this sample size; beds, refrigerators, counters and storage were separable, cooking appliances were not. IFCNet's embedding added nothing in either task, and the variant trained only on openly licensed models reached 0.53 accuracy and 0.24 macro-F1 on sub-types, for want of open furniture data.

The evidence therefore argues against fine-tuning IFCNet's weights, which would be neither more accurate on this evidence nor publishable, and for a small feature model trained on openly licensed IFC models and furniture libraries and published with its provenance. Section 6 estimates that campaign at a few CPU hours for the feature models and their ablations, against 58 hours a run (about a week for three seeds) if a DGCNN is also trained from scratch at 1,024 points; the dominant cost is acquiring, converting and auditing open furniture data, which is supervised work measured in weeks. Section 7 lists the decisions needed before it starts.

---

## 1. Aim and questions

The application identifies, in any IFC model a user opens in the browser, the elements that can start a fire, so that the fire and egress simulation can be started at a likely ignition point. The ignition review ranks ignition scenarios by their share of fire deaths, and the items it names include upholstered seating, beds and mattresses, cooking appliances, refrigerators, laundry appliances, heaters, lamps, electrical distribution and office workstations ([ignition-sources-2026-11-05.md](ignition-sources-2026-11-05.md)). Exporters rarely type these items: the earlier corpus check found no typed appliance in 45 models, and furniture arrives as IfcFurnishingElement or IfcBuildingElementProxy whose name is often the only evidence. The earlier benchmark showed that multilingual name rules identify 86 % of typed elements, that IFCNet's DGCNN shape classifier adds about 0.002 to them, and that no method separates furniture sub-types from shape.

The study asks three questions.

- Q1. Do per-element geometric features, specifically covariance eigenvalue features, surface-orientation statistics and the heights of horizontal surfaces above the floor, identify furniture sub-types when the name is absent or uninformative, and by how much do they improve on the element's bounding box alone?
- Q2. Does combining these features with the name rules and with a point network's embedding improve identification over the rules alone, under a split that keeps every building's elements on one side?
- Q3. Can a variant trained only on openly licensed data reach useful accuracy, so that its weights can be published under an open licence?

Hypotheses for the full campaign, fixed before it runs: H1, the full feature set beats the box-only features on sub-type macro-F1 with a building-level bootstrap interval that excludes zero; H2, the name rules followed by the feature model (a cascade) are at least as accurate as the rules alone on the typed elements and more accurate on proxies; H3, the openly licensed variant is within five points of macro-F1 of the variant trained on all local data, on the same held-out buildings.

---

## 2. Licences and terms

### 2.1 Code

Table 1. Code sources, as their pages read on 2026-11-05.

| Source | What it says | Consequence here |
|---|---|---|
| github.com/florentPoux/3D-Data-Science | GitHub's licence field: NOASSERTION. LICENSE: "This repository is licensed under a dual-license model depending on your ownership of the book". Book owners receive an MIT-style grant with the condition "Attribution to the original author (Florent Poux) and the book must be maintained in derivative works"; non-owners receive "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) ... NonCommercial: You may not use the material for commercial purposes". Some file headers say "License: MIT" and some files are marked "Review Only (Confidential)", which conflicts with the LICENSE. | No code was copied. The methods (eigenvalue features, PCA, RANSAC, region growing, voxelisation) were implemented from the papers cited in Section 4, which the repository's tutorials also follow. |
| github.com/florentPoux/3d-data-materials | GitHub's licence field: MIT. LICENSE: "MIT License, Copyright (c) 2026 Florent Poux". DATA_LICENSE.md: "The LICENSE file covers the code in this repository, and only the code"; its kit data are "Not MIT licensed", and its S3DIS-derived outputs repeat S3DIS's non-commercial terms. | The code could be reused with the MIT notice; none was needed. Its data are not used. |
| github.com/RWTH-E3D/ifcnet-models (IFCNet) | LICENSE: "The MIT License (MIT) Copyright (c) 2021, Christoph Emunds", covering "this software and associated documentation files". The README asks users of "the IFCNet dataset or code" to cite the paper and states no terms for the dataset or the weights. ifcnet.e3d.rwth-aachen.de, which serves both, states no licence, terms or copyright (re-confirmed). | The DGCNN architecture code may be reused under MIT. The weights and the dataset carry no grant. |

### 2.2 Training data

Table 2. Candidate training data. "Open weights" asks whether weights trained on the source could be published under MIT, Apache-2.0 or CC BY; the answers are conservative readings of the stated terms, not legal advice.

| Source | Terms as stated | Open weights? | Furniture and ignition items |
|---|---|---|---|
| IFCNet dataset and weights | No licence, terms or copyright stated; the paper describes "around 1000 IFC models ... collected from real-world projects, student works and online sources" | No grant: no | One IfcFurniture class (chairs to wardrobes), IfcLamp, IfcSpaceHeater, IfcSanitaryTerminal; no sub-types |
| BIMGEOM (Collins et al. 2021; Harvard Dataverse doi:10.7910/DVN/YK86XK) | Dataverse licence "CC0 1.0"; the paper calls it a "partially industry dataset" from 22 IFC files | Yes, on the CC0 text; provenance of the source models undocumented | One IfcFurnishingElement class (371 geometries); no sub-types |
| buildingSMART Certification-datasets (formerly Sample-Test-Files) | "This work is licensed under the Creative Commons Attribution 4.0 International License" | Yes, with attribution | Simple certification scenes |
| buildingSMART Community-Sample-Test-Files | README: contributors publish "under the Creative Commons Attribution 4.0 International License"; the Duplex and Medical-Dental folders state CC BY 4.0; Schependomlaan grants "permission to use the data for scientific and academic purposes" only | Duplex, Medical-Dental: yes, with attribution; Schependomlaan: no | Duplex: cabinets, sofas, beds, tables (Revit default families, see caveat below) |
| Objaverse 1.0 | "The use of the dataset as a whole is licensed under the ODC-By v1.0 license"; each object carries its own licence (CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA, CC0) in the metadata | Yes for objects licensed CC BY or CC0, with attribution and the ODC-By notice | LVIS categories include sofa, armchair, bed, mattress, desk, table, chair, cabinet, wardrobe, bookcase, stove, oven, microwave, refrigerator, washer, dishwasher, television, heater, lamp; no clothes dryer |
| Objaverse-XL | ODC-By for the whole; individual licences vary, and many rows have none | Only rows with a stated CC BY or CC0 licence | No category labels |
| Amazon Berkeley Objects (ABO) | Index page, read 2026-11-05: "A CC BY 4.0-licensed dataset of Amazon products"; earlier releases stated CC BY-NC 4.0 | Yes under the current CC BY 4.0, with attribution; keep a dated copy of the licence | Chairs, sofas, ottomans, tables, desks, cabinets, shelves, dressers, beds, mattresses, lamps; no appliances |
| Google Scanned Objects | CC BY 4.0 | Yes, with attribution | No full-size furniture |
| OmniObject3D | "released under the CC BY 4.0" | Yes, with attribution | Sofa, chair, table, cabinet, bed, lamp, microwave, fan; scale of the furniture unverified |
| Poly Haven | "Our assets are all licensed as CC0"; the website's terms forbid scraping, the API's terms allow use | Yes; download through the API | Sofas, armchairs, beds, desks, tables, shelves, stoves, microwave, heaters, TVs, lamps |
| Kenney Furniture Kit | "Creative Commons Zero, CC0" | Yes | Beds, sofas, desk, tables, cabinets, stove, microwave, fridge, washer, dryer, TVs, lamps |
| Quaternius packs | Pack files: CC0 1.0; the site's Quaternius Asset License v1.0 (2026-08-28) forbids redistributing assets "as a standalone asset" and states it is not retroactive | Probably yes for weights; do not redistribute the meshes | Beds, couches, chairs, tables, fridge, oven, washing machine, fireplace, lights |
| FreeCAD-library | LICENSE-Assets: every part is "licensed under the Creative Commons Attribution 3.0 Unported license"; scripts LGPL-2.1-or-later | Yes, with per-author attribution | Armchair, sofa, beds, closets, tables, desk, dishwasher, stove top, fridge, oven, microwave, washing machine, TV |
| IfcOpenShell's Bonsai furniture library | No licence in the file; the repository is GPL-3.0-or-later | Unclear: excluded | Furniture and appliance types as simple solids |
| ModelNet | "provided for the convenience of academic research only"; "the original authors hold the copyright of the CAD models" | No (silent on trained models; academic only) | Bed, chair, desk, sofa, table, wardrobe, lamp, monitor; no appliances |
| ShapeNet | "Researcher shall use the Database only for non-commercial research and educational purposes" | No (silent on trained models) | Chairs, sofas, tables, beds, cabinets, lamps, displays, washers, dishwashers |
| 3D-FUTURE and 3D-FRONT | "for scientific research purpose only"; section 3.3.3: "You shall not use the data sets, derivative works of the data sets, and the results obtained based on the data sets for commercialization or providing external services" | No: the only terms found that reach trained results explicitly | 34 furniture categories; no appliances |
| HSSD | "CC BY-NC 4.0"; licensed from Floorplanner "for academic use" | No (at most non-commercial) | The broadest appliance coverage |
| BuildingNet | "only for non-commercial research and educational purposes"; models from 3D Warehouse | No | One generic furniture label |
| NBS National BIM Library | Site terms forbid "any text or data mining or web scraping" | No | Manufacturer objects |
| BIMobject | Terms section 4.7(j): users may not use any content "to train or otherwise improve or enhance any artificial intelligence, machine learning ... systems" | No, explicitly | Manufacturer objects |
| Autodesk Revit content | The terms pages returned HTTP 403; secondary sources report no derivative works and a bar on machine-learning use | Treat as no | Revit default families |

Two caveats apply to the open sources. The Duplex model's furniture is Autodesk's metric default family content exported to IFC; buildingSMART distributes the file under CC BY 4.0, but the families' own terms are not addressed. Two licences changed recently (ABO from CC BY-NC to CC BY; Quaternius from CC0 to its own licence), so the campaign must archive a dated copy of every licence it relies on.

### 2.3 The local corpus

The benchmark's 45 models fall into two groups by the terms their sources state. Eighteen models, from 16 buildings, carry CC0, CC BY 4.0, MIT (generated buildings) or KIT's "unrestricted use" statement: the KIT FZK-Haus, Institute and Smiley West; IfcOpenHouse; buildingSMART's sample scene in two revisions; the Duplex in two exports and the dental clinic (NIBS Common BIM files, which buildingSMART's Community-Sample-Test-Files distribute under CC BY 4.0); the four Zenodo and buildingSMART-community models of the more2 batch (CC BY 4.0); three generated bim-whale buildings (MIT); and the two buildings generated for Studio (CC0). The other 27 have no stated licence that covers training (web-ifc's and xbim's test files, the youshengCode collection), are Autodesk sample projects redistributed under MIT, are GPL-3.0 test files (opensourceBIM TestFiles) or are CC BY-ND (opensourceBIM IFC-files). The list is `OPEN_MODELS` in `tools/identify-pilot.py`.

### 2.4 Which training paths allow open weights

- Path O, open. The DGCNN architecture (MIT) or the feature model, trained only on data whose terms grant reuse (CC0, CC BY, MIT, KIT's unrestricted use, and the CC BY or CC0 subset of Objaverse), can be published under an open licence. CC BY sources require an attribution file shipped with the weights, and Objaverse adds the ODC-By notice. CC BY-SA objects are excluded because a share-alike condition on derived weights is unresolved.
- Path I, IFCNet-derived. Weights fine-tuned from IFCNet's weights, a probe or head trained on IFCNet's frozen embedding, or any network trained on IFCNet's dataset are derivatives of material with no stated licence. The application ships IFCNet's own weights with credit by the user's decision; that decision does not make a derivative openly licensable. Such weights stay local, for comparison in the paper, unless IFCNet's authors grant a licence.
- Path N, non-commercial or research-only data. ModelNet, ShapeNet, 3D-FUTURE, 3D-FRONT, HSSD and BuildingNet are excluded from any published variant; 3D-FUTURE's terms exclude it explicitly.

The study is therefore designed with one variant on path O (Section 6.1, variant O-feat), trained and published with a licence and a per-file provenance manifest, and the IFCNet-derived variant kept as an unpublished comparison.

---

## 3. Prior work and the gap

Searches ran on OpenAlex (35 queries over titles and abstracts, 2015 to 2026), Crossref (45 bibliographic queries) and arXiv (15 queries) on 2026-11-05, over five themes: geometric deep learning for IFC classes, furniture in BIM, hybrids of names and geometry, eigenvalue features on BIM objects, and BIM-based fire risk. The queries and their hit counts are in the results file (`literatureSearch`) and the DOI checks in `doiVerification` (35 of 35 resolved at both registries); Table 3 lists the closest work.

Table 3. Closest prior work (DOIs verified at Crossref or DataCite and at OpenAlex).

| Work | What it does | Relation to this study |
|---|---|---|
| Emunds et al. 2021, IFCNet (10.48550/arXiv.2106.09712) | 19,613 single-entity IFC objects in 65 classes; IFCNetCore 20 classes; MVCNN, MeshNet and DGCNN (balanced accuracy 0.79 for DGCNN) | One furniture class; normalisation discards absolute size; fusion with semantics left to future work; fixed split |
| Emunds et al. 2022, SpaRSE-BIM (10.1016/j.aei.2022.101641) | Sparse 3D CNN on voxelised IFC geometry, on IFCNet and BIMGEOM | IFC classes only |
| Collins et al. 2021 (10.35490/ec3.2021.168) | Graph CNN on triangle meshes, up to 85 % accuracy; BIMGEOM | Shape only |
| Koo et al. 2019 (10.1016/j.autcon.2018.11.015); Koo et al. 2021 (10.1016/j.aei.2020.101200) | SVM on geometric and relational features, then MVCNN and PointNet for wall and door sub-types | The precedent for IFC sub-type classification; not furniture |
| Wu et al. 2022 (10.1061/(ASCE)CP.1943-5487.0001012) | Geometric, locational and metadata "invariant signatures" with classical learners | Hand-crafted features with metadata; structural classes |
| Utkucu et al. 2024 (10.1016/j.aei.2024.102503) | A stacked ensemble of keyword search, rule inference, geometric-feature learning and visual deep learning; 40 architectural and MEP classes, accuracy 0.91 | The closest precedent for the hybrid of names, rules and geometry |
| Liu et al. 2024, MMDL (10.1016/j.aei.2024.102458) | Graphical and non-graphical features fused in a multimodal network for fine-grained sub-categories | Fusion and fine-grained classes, not furniture |
| Tang et al. 2025 (10.3390/buildings15132347) | Random forest on 13 semantic, spatial and dimensional IFC features for Uniclass classes | Attributes plus dimensions; cross-validation not by project |
| Austern et al. 2024 (10.3390/buildings14020527) | Graph neural networks over element adjacency | Context rather than names |
| Luo et al. 2023 (10.1007/978-3-031-25082-8_23); Yang et al. 2026 (10.1016/j.autcon.2025.106660); Jing et al. 2026 (10.1016/j.aei.2026.105040) | Relational, IFC-graph and text-point-image fusion | Fusion of semantics with geometry, IFC classes |
| Yang et al. 2025, BIMCompNet (10.1145/3746027.3758238); Shi et al. 2026, BIM-JEPA (10.1016/j.autcon.2026.107075); Seydgar et al. 2024 (10.1109/ACCESS.2024.3436681) | A large multimodal IFC component dataset; self-supervised pretraining with released weights (MIT, gated); a benchmark of point networks | IFC classes; data licences of the pretraining sources unstated |
| Frías et al. 2022 (10.1061/(ASCE)CP.1943-5487.0001039) | Synthetic point clouds from manufacturer BIM objects, 21 indoor object classes, with a surface-variation channel | The nearest use of an eigenvalue feature on BIM-derived geometry, aimed at scans |
| Ding et al. 2025 (10.1016/j.psep.2025.107031); Elhami-Khorasani et al. 2020 (10.1007/s10694-020-00989-9) | Fuel-load surveys and fire risk from images, by recognising furniture | Furniture for fire, but from photographs, not IFC |
| Jiang et al. 2027 (10.1016/j.ress.2026.113173); Liang et al. 2026 (10.1016/j.dibe.2026.100869); Li and Guo 2026 (10.3390/buildings16132655) | BIM semantic enrichment for fire simulation and evacuation; image-to-BIM fire risk; ignition-source overlaps in 4D construction models | Fire-oriented BIM enrichment; whether Jiang et al. infer furniture or ignition items could not be confirmed from the metadata |

The evidence supports a narrow gap. No indexed work classifies IFC furniture into the sub-types that matter for ignition, none computes the set of covariance eigenvalue features on IFC element geometry to classify elements (Frías et al. (2022) used one of them, surface variation, on synthetic scans of BIM objects), and none runs an IFC classifier in the browser. The hybrid of names and geometry is not new: Utkucu et al. (2024), Liu et al. (2024) and Jing et al. (2026) already fuse attributes or text with shape, and Koo et al. (2019, 2021) already classify IFC sub-types. No abstract read reports a split that keeps each source model or building on one side, and no paper found documents the licence provenance of its IFC training data.

Table 4. Candidate claims and their strength on this evidence.

| Claim | Strength | What weakens it |
|---|---|---|
| Furniture sub-types relevant to ignition, identified from IFC geometry and names | Strong, if worded narrowly | Koo et al. sub-types (walls, doors); fine-grained classes in Liu et al.; image-based furniture recognition for fuel load (Ding et al., Elhami-Khorasani et al.); IFC4's PredefinedType enumerations already define the sub-types, so the task is recovering types the exporter omitted |
| Browser-computable eigenvalue and surface features for IFC elements, in milliseconds | Moderate | Hand-crafted IFC features are established (Koo, Wu, Tang, Utkucu); the eigenvalue features are textbook; the claim rests on the ablation against box-only features and on measured latency |
| Name rules, features and a point network's embedding, fused, under leave-buildings-out evaluation | Weak as an architecture; moderate as an evaluation protocol | Utkucu et al. 2024 and the other fusion papers |
| Openly licensed weights trained only on open data, with a per-file provenance manifest | Moderate, as an open-science contribution | BIM-JEPA and IFCNet release weights, without data licences |
| IFC element identification feeding ignition-source ranking for egress simulation | Strong to moderate | Jiang et al. 2027 (to be read in full before claiming), Liang et al. 2026, Ding et al. 2025 |

A paper can therefore claim, with the strongest footing, the first identification of ignition-relevant furniture sub-types from IFC geometry under a leave-buildings-out protocol, computed in the browser, with an openly licensed model and its provenance. The full texts of Utkucu et al., Liu et al., Jing et al. and Jiang et al. must be read before submission; their abstracts were partly available only from secondary pages.

---

## 4. Features and fusion

### 4.1 Per-element features

`elementFeatures` in `tools/identify-features.mjs` computes 71 values from an element's triangles in the viewer's frame (Y up, metres) and its storey's floor level, as the application already holds them. Table 5 lists them by group. The implementation is new; the definitions follow the cited papers.

Table 5. The feature set.

| Group (count) | Features | Source of the definition |
|---|---|---|
| Box and placement (9) | Length and width along the element's principal plan axes, height, height of the bottom above the storey floor, top height, log volume, width over length, height over length, whether the floor is known | `boxOf` in `identify.js` (the benchmark's box rule) |
| Mesh (3) | Log surface area, surface area over the box's surface, log triangle count | Defined here |
| Covariance eigenvalues (21) | Of 1,024 area-weighted surface samples: linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, change of curvature, log eigenvalue sum, verticality of the main axis and of the normal (1 - abs(n_y)); and, for 384 samples, the same features over their 16 nearest samples, as means and standard deviations, the shares of linear, planar and scattered neighbourhoods, mean local verticality and curvature | West et al. 2004; Demantké et al. 2011, 2012; Weinmann et al. 2015; Hackel et al. 2016; Poux and Billen 2019 |
| Surfaces (25) | Area shares of upward, downward, vertical and slanted faces; mean abs(n_y); orientation entropy over 12 bins; the share of surface in eight height bands of the element; the share of upward-facing area in eight bands of height above the floor (edges at 0.15, 0.32, 0.55, 0.68, 0.82, 1.0 and 1.6 m: seats, tables and desks, worktops, shelves), the number of distinct upward levels, the top level's share and the main level's height | Normals as in Poux and Billen 2019 (pure horizontal and vertical relations); height bands chosen here from furniture conventions |
| Segments and voxels (13) | Planar segments by region growing on the mesh (edge-adjacent triangles joined when their normals differ by less than 5 degrees and each centroid lies within 1 cm of the other's plane): counts above 1 % and 5 % of the area, the largest and three largest shares, the share covered; connected parts and the largest part's share; occupancy of an 8 x 8 x 8 grid on the element's box, of a 16 x 16 plan grid, and of four height layers | Region growing after Poux et al. 2022; voxel occupancy after Poux and Billen 2019 |

Region growing on the mesh replaces RANSAC on points. An IFC element's triangles already carry exact normals and adjacency, so coplanar regions can be grown in one pass over the edges with a union-find structure; RANSAC would re-estimate planes from samples at a cost that grows with the number of planes. Both are named in Poux's work; Poux and Billen (2019) avoid region growing on large scans for its cost, which does not arise on single elements.

On the corpus the features took a median 2.66 ms an element (90th percentile 3.95 ms, 99th 6.67 ms, largest 59 ms) over 13,536 elements in Node 24 on one thread; a second extraction, run while the pilot's models were fitting on 15 threads, took a median 5.93 ms and produced identical features, rule answers and sub-types for every element (`reextraction` in the results file). Node runs Chrome's JavaScript engine, so browser times should be of the same order; they were not measured in a browser in this phase. For comparison, the pruned DGCNN's embedding cost 47.7 ms an element in native onnxruntime on eight threads, and 0.21 to 0.26 s an element on one WASM thread in the benchmark.

### 4.2 Fusion with the rules and a point network

Three ways of combining the sources were compared in the pilot, each keeping the browser's limits in mind (a model must export to ONNX, stay small, and never delay the rules' answer):

- a cascade: the rules answer when a PredefinedType or a name decides, and the feature model answers when the rules fall back to the box or find nothing;
- a stacked model: one gradient-boosted classifier whose inputs are the 71 features, the rules' label and the source of the rules' answer (PredefinedType, name, box or none);
- the network's embedding as further inputs: the 256-wide penultimate layer and the 20 logits of the pruned DGCNN, added to the features (or, alone, to a logistic-regression probe).

The classifier is scikit-learn's histogram gradient boosting (300 rounds, learning rate 0.08, 31 leaves, L2 regularisation 1.0, class weights balanced), chosen before any result was seen and not tuned. It exports to ONNX with skl2onnx (Section 5.6), the format of onnxruntime-web, which the application already loads for IFCNet; whether onnxruntime-web's WASM build runs the exported tree operator remains to be checked.

---

## 5. Pilot

### 5.1 Data and label sets

Task A reuses the benchmark's evaluation set unchanged: 6,066 typed elements and 159 hand-labelled proxies from 45 models, with the IFC class mapped to 18 labels as the truth for typed elements. The rules' answers were reproduced exactly (6,225 of 6,225 elements), and IFCNet's answers from the embedding run matched the benchmark's on 6,210 of 6,225 elements; the run used batches of eight against the benchmark's batches of one, and the 15 differences are near-ties.

Task B builds a furniture sub-type set from the whole corpus. The corpus holds 2,009 furniture-class elements and no element typed as an appliance, and 574 proxies whose names give a furniture or appliance sub-type. A sub-type comes from the IFC4 PredefinedType where the exporter set it, otherwise from the element's name, object type or type name through an ordered multilingual vocabulary (`furnitureSubtype` in `tools/identify-features.mjs`: lamps, then appliances, upholstered seating, beds, desks, counters, chairs, tables and storage). Of the 2,009 furniture elements, 1,536 received a sub-type and 473 have names that state none. Repeated instances were thinned to at most three per family (same model, same name stem without digits, same size to 5 cm), and a sub-type was kept when it had at least 15 instances from at least three buildings, a rule fixed before any model was scored. Table 6 gives the outcome.

Table 6. Furniture and appliance sub-types in the corpus.

| Sub-type | Ignition scenario it serves | Instances | After thinning | Families | Buildings | Kept |
|---|---|---|---|---|---|---|
| storage (cabinets, wardrobes, shelves) | none | 872 | 266 | 119 | 12 | yes |
| chair | none | 411 | 55 | 22 | 8 | yes |
| desk | waste bin at a workstation | 271 | 24 | 10 | 9 | yes |
| counter | none (marks a kitchen) | 144 | 100 | 71 | 7 | yes |
| table | none | 136 | 90 | 50 | 9 | yes |
| electronics (TV, monitor, computer, printer) | TV and electronics; office equipment | 66 | 25 | 14 | 4 | yes |
| cooking (cooktop, oven, microwave) | unattended cooking | 41 | 24 | 13 | 5 | yes |
| seat-upholstered (sofa, armchair) | smoking materials on upholstery | 38 | 27 | 14 | 6 | yes |
| fridge | refrigerator or freezer | 20 | 20 | 11 | 5 | yes |
| bed | smoking materials on bedding; candles; chargers | 18 | 18 | 11 | 6 | yes |
| dishwasher | dishwasher | 16 | 11 | 6 | 4 | no |
| laundry | tumble dryer; washing machine | 10 | 10 | 7 | 4 | no |
| lamp (furniture class) | electrical distribution and lighting | 65 | 22 | 8 | 2 | no |
| other furnishings | none | 2 | 2 | 2 | 2 | no |

The kept set has 649 elements from 19 buildings, and only 3 of them took their sub-type from a PredefinedType; the rest are named, so the sub-type truth is the vocabulary's reading of a name and carries its errors. The application's own ignition patterns (`subtypeOf` in `identify.js`) find, in the same records, extension leads (82 elements, all in one model), workstations (74, seven models), cooking appliances (33, three), upholstered seating (31, seven), distribution boards (26, two), electronics (25, four), beds (18, seven), refrigerators (17, four), washing machines (9, three), dishwashers (7, two), e-bikes (2, one), one dryer and one fireplace, and nothing for candles, space heaters by name, small lithium-ion devices and chargers, office equipment such as copiers, Christmas trees, boilers or battery rooms (`ignitionNameMatches`). The corpus therefore cannot support dishwashers, laundry appliances, fireplaces and stoves, heaters as furnishings, micromobility, chargers, candles, Christmas trees or battery rooms; heaters, lights, distribution boards and boilers are covered at the label level of Task A (heater, lighting, mep), not as sub-types.

### 5.2 Protocol

Every split held out whole buildings. Repeated exports of one design were grouped first (the FZK-Haus in three exports, Smiley West in two, the Revit advanced sample in three, the Duplex in two and buildingSMART's sample scene in two), which leaves 38 buildings from the 45 models. The buildings were dealt to five folds, largest first, each to the fold with the fewest elements; each fold's model was trained on the other four folds' typed elements (Task A) or sub-type elements (Task B) and predicted its own, and each hand-labelled proxy was predicted by the model of its building's fold. "Names withheld" means that Name, ObjectType and LongName were blank for the method; the rules without names then decide from PredefinedType and the box only, the case of an element named by a code. Accuracy and macro-F1 over the labels present are computed as in the benchmark. The 95 % intervals are for the difference between two methods on the same elements, from 1,000 bootstrap resamples of buildings. The classifiers' settings were fixed before any score was seen (Section 4.2), and the logistic-regression probe on the DGCNN's 256-wide layer used standardised inputs, C = 0.5 and balanced class weights. The openly licensed variant was trained on the 18 open models (16 buildings) and tested on the buildings that have no openly licensed export at all, so that no other export of a training building leaks into the test.

### 5.3 Element labels (Task A)

Table 7 reports every method on the benchmark's elements. The first block repeats the benchmark's baselines, the second withholds names from every method, and the third gives names to the rules.

Table 7. Accuracy / macro-F1 on the typed elements (6,066) and the hand-labelled proxies (159), leave-buildings-out.

| Method | Typed elements | Proxies | Furniture F1, typed |
|---|---|---|---|
| Rules (benchmark) | 0.860 / 0.829 | 0.604 / 0.544 | 0.734 |
| Rules, then IFCNet at a probability of 0.8 (benchmark) | 0.862 / 0.831 | 0.598 / 0.543 | 0.749 |
| IFCNet DGCNN, zero-shot (benchmark) | 0.413 / 0.349 | 0.277 / 0.201 | 0.714 |
| Names withheld: rules (PredefinedType and box) | 0.701 / 0.630 | 0.189 / 0.077 | 0.583 |
| Names withheld: box features only | 0.651 / 0.527 | 0.145 / 0.113 | 0.765 |
| Names withheld: box and eigenvalue features | 0.731 / 0.607 | 0.138 / 0.185 | 0.824 |
| Names withheld: box and surface features | 0.726 / 0.601 | 0.126 / 0.136 | 0.819 |
| Names withheld: box, segment and voxel features | 0.733 / 0.612 | 0.094 / 0.114 | 0.809 |
| Names withheld: all 71 features | 0.759 / 0.669 | 0.113 / 0.167 | 0.833 |
| Names withheld: DGCNN linear probe | 0.622 / 0.498 | 0.119 / 0.140 | 0.731 |
| Names withheld: features and DGCNN embedding | 0.759 / 0.625 | 0.132 / 0.140 | 0.821 |
| Names withheld: features and rules, stacked | 0.819 / 0.712 | 0.176 / 0.228 | 0.820 |
| Rules, then features (cascade) | 0.882 / 0.873 | 0.528 / 0.533 | 0.885 |
| Rules, then features and embedding (cascade) | 0.884 / 0.852 | 0.547 / 0.560 | 0.895 |
| Features and rules, stacked | 0.874 / 0.783 | 0.365 / 0.357 | 0.859 |
| Features, rules and embedding, stacked | 0.872 / 0.734 | 0.346 / 0.332 | 0.853 |

Without names, the features identified typed elements better than every other method that reads geometry. Differences below are the point estimates from Table 7, with the bootstrap's 95 % interval in brackets. Against the box features alone the features gained 0.108 accuracy (0.072 to 0.139) and 0.142 macro-F1 (0.077 to 0.188); each group of richer features (eigenvalues, surfaces, segments and voxels) gained about eight points of accuracy over the box on its own, and together they gained eleven. Against IFCNet's zero-shot answer the features gained 0.346 accuracy (0.297 to 0.400) and 0.320 macro-F1 (0.272 to 0.408), and against a probe trained on IFCNet's embedding with the same splits they gained 0.137 accuracy (0.104 to 0.165) and 0.171 macro-F1 (0.108 to 0.214). Adding the embedding to the features changed accuracy by 0.000 (minus 0.015 to 0.017) and macro-F1 by minus 0.044 (minus 0.071 to 0.008). For an element named by a code, the features raised the name-less rules from 0.701 to 0.819 accuracy (gain 0.118, from 0.073 to 0.160) and from 0.630 to 0.712 macro-F1 (gain 0.082, from 0.019 to 0.192).

With names, the cascade is the only combination that improved on the rules on typed elements by both measures: accuracy rose by 0.022 (0.006 to 0.037) and macro-F1 by 0.044 (0.013 to 0.083). The gain sits where the rules fall back to the box: on those 1,624 typed elements the rules scored 0.579 / 0.397 and the features 0.663 / 0.601, and over all typed elements the cascade raised the F1 of stairs from 0.793 to 0.937, of lights from 0.493 to 0.941 and of furniture from 0.734 to 0.885. The stacked model raised accuracy but lost macro-F1, because it learned to trust the rules' label and dropped rare labels. On the proxies every learned model did worse than the rules: the cascade lost 0.076 accuracy (minus 0.133 to minus 0.022), and on the 68 proxies where the rules fall back to the box the rules scored 0.235 accuracy and the features 0.059. The cause is the training set rather than the features: no typed element in the corpus is an appliance or a sanitary fitting, so 38 of the 159 proxies carry labels the models never saw, and the proxies' "other" (people, trees, site marks) is not the typed "other" (members, plates, footings).

### 5.4 Furniture sub-types (Task B)

Table 8 reports the sub-type models on the 649 elements; all withhold names, since the truth was read from names.

Table 8. Furniture sub-types, 10 labels, 649 elements from 19 buildings, leave-buildings-out: accuracy / macro-F1, and F1 for the ignition-relevant sub-types.

| Method | All | Bed | Seat-upholstered | Desk | Cooking | Fridge | Electronics |
|---|---|---|---|---|---|---|---|
| Majority class (storage) | 0.410 / 0.058 | 0 | 0 | 0 | 0 | 0 | 0 |
| Box features only | 0.709 / 0.592 | 0.895 | 0.208 | 0.714 | 0.000 | 0.809 | 0.600 |
| Box and mesh features | 0.741 / 0.614 | 0.865 | 0.364 | 0.735 | 0.125 | 0.652 | 0.433 |
| Box and eigenvalue features | 0.686 / 0.592 | 0.821 | 0.346 | 0.889 | 0.059 | 0.844 | 0.378 |
| Box and surface features | 0.693 / 0.546 | 0.432 | 0.550 | 0.667 | 0.057 | 0.667 | 0.254 |
| Box, segment and voxel features | 0.678 / 0.563 | 0.865 | 0.377 | 0.524 | 0.047 | 0.735 | 0.474 |
| All 71 features | 0.710 / 0.594 | 0.850 | 0.537 | 0.528 | 0.154 | 0.792 | 0.319 |
| DGCNN linear probe | 0.633 / 0.515 | 0.667 | 0.490 | 0.522 | 0.333 | 0.275 | 0.278 |
| Features and DGCNN embedding | 0.698 / 0.589 | 0.842 | 0.512 | 0.588 | 0.229 | 0.744 | 0.344 |

The results do not support H1 at pilot scale. The full feature set and the box alone differ by 0.001 accuracy (interval minus 0.109 to 0.064) and 0.002 macro-F1 (minus 0.057 to 0.093); the eigenvalue features added to the box differ from it by minus 0.023 accuracy and 0.000 macro-F1, with intervals as wide. The features beat the DGCNN probe on accuracy by 0.077 (0.023 to 0.135), and the embedding added nothing to them (minus 0.012, interval minus 0.056 to 0.035). Per sub-type, the full features separated storage (F1 0.862), beds (0.850), counters (0.849), refrigerators (0.792), chairs (0.655) and upholstered seats (0.537, against 0.208 from the box), and failed on cooking appliances (0.154), which have the size and shape of the base cabinets they stand among, on electronics (0.319) and on tables (0.395), which the model confused with electronics in 28 of 90 cases. The intervals are wide because the test set is small: ten labels, 649 elements and 19 buildings, with 18 beds and 20 refrigerators. IFCNet itself, asked about these 649 elements, answered IfcFurniture for 323 and mostly IfcSlab, duct fittings, sanitary terminals or beams for the rest. On the 473 furniture elements whose names state no sub-type, the full model's answers were storage 193, chair 106, desk 85, counter 27, bed 17, upholstered seat 16, cooking 11, table 9, refrigerator 7 and electronics 2; these are unscored, and they are the elements the study is for.

### 5.5 The openly licensed variant

Trained on the 2,256 typed elements of the 16 openly licensed buildings and tested on the 3,529 typed elements of the 22 buildings with no open export, the feature model scored 0.700 accuracy and 0.625 macro-F1, against 0.754 and 0.721 for the cross-validated model trained on all other buildings, and 0.856 and 0.829 for the rules, on the same elements. The loss of 0.054 accuracy and 0.096 macro-F1 is larger than the five points that H3 allows. For sub-types the open buildings hold only 145 thinned elements from five buildings, and none of them is a cooking appliance, a refrigerator or an electronic device; trained on them, the model scored 0.53 accuracy and 0.24 macro-F1 on 500 elements from 13 other buildings, against 0.67 and 0.58 for the cross-validated model. The open IFC corpus is therefore too small for sub-types, and the open variant depends on the furniture libraries of Table 2.

### 5.6 Size and cost

The sub-type model (ten labels, 300 rounds) exported to ONNX at 1.36 MB, gave the same answers as scikit-learn on all 649 elements and took 0.054 ms an element in native onnxruntime; the 18-label model is 8.83 MB, larger than the pruned DGCNN (1.45 MB), because 300 rounds of 18 trees make 5,400 trees. Whether onnxruntime-web's WASM build runs the ai.onnx.ml tree operator was not checked in this phase. A five-fold fit of the 18-label feature model took 30.8 s and the probe 3.2 s. Training the DGCNN from a random start, on this CPU with eight threads, cost 0.94 s a sample at 2,048 points with 40 neighbours (IFCNet's setting) and 0.21 s at 1,024 points with 20 neighbours (`dgcnnTrainingCost`).

---

## 6. Full campaign: design and estimate

### 6.1 Variants

- O-feat: the 71 features with a small classifier (gradient boosting capped in size, or a two-layer MLP), trained only on openly licensed data, and the cascade of Section 4.2. Published with a licence, an attribution file and a per-file provenance manifest. This is the variant the pilot supports.
- O-net (optional): IFCNet's DGCNN architecture (MIT) trained from a random start on openly licensed data only, pruned and exported as in the benchmark, its embedding fused with the features. Publishable, but the pilot found that IFCNet's own embedding added nothing to the features, so it is worth its cost only if O-feat falls short on shapes that boxes and features cannot describe.
- I (comparison, unpublished): IFCNet-derived models, the probe and a head on IFCNet's frozen embedding, and optionally a short fine-tune, kept local as a baseline unless IFCNet's authors license their weights.
- All-local: the feature model trained on every local model, as the ceiling against which O-feat is judged (H3).

The test set stays IFC: the held-out buildings of the corpus, plus openly licensed IFC models acquired for the campaign and set aside before any model sees them. Furniture libraries are training data only, because their meshes (game assets, product scans, CAD parts) are not what IFC exporters produce.

### 6.2 Data that is missing

- Open IFC furniture. The 18 open models hold 145 thinned sub-type elements and no cooking appliance, refrigerator or electronics. More open IFC models with furniture are needed for training and, above all, for an untouched final test: candidates are the CC BY 4.0 Zenodo models already identified for the corpus but not kept (BIM4EEB and others), further buildingSMART community files, and models built in Bonsai from CC0 or CC BY furniture meshes.
- Open furniture meshes for the sub-types and the unsupported ignition items: Kenney's furniture kit (CC0: fridge, washer, dryer, stove), Poly Haven through its API (CC0: sofas, beds, heaters, stoves), FreeCAD-library (CC BY 3.0: dishwasher, laundry machines, oven, fridge, TV), ABO (CC BY 4.0: chairs, sofas, beds, mattresses, desks, storage, lamps) and the CC BY or CC0 objects of Objaverse's furniture and appliance categories. Download sizes were not measured in this phase; the first campaign step measures them and stops at a disk budget the user sets.
- Labels. The sub-type truth is a name read by a vocabulary. A hand audit of a stratified sample (for example 300 elements) is needed before any claim, and the 473 unnamed furniture elements of the corpus can become a hand-labelled test set from their geometry and room, which is the case the study is for.
- Proxies. The training set needs proxies, appliances, sanitary fittings and site objects, since their absence caused the cascade's loss on the hand-labelled proxies; open meshes of people, trees and vehicles (CC0 libraries) and the open models' proxies would fill it.
- Context. Cooking appliances are indistinguishable from base cabinets by shape; the room an element stands in (a kitchen by the IfcSpace name, which survives when element names are codes) and its neighbours (a cooktop sits in a counter run) are the features that can separate them.

### 6.3 Compute and time

Table 9. Estimate for the full campaign on this machine (i7-14700, CPU only), from the pilot's measurements; N is the number of training objects and E the number of epochs, both assumptions.

| Step | Basis | Estimate |
|---|---|---|
| Features for the corpus and about 20,000 library meshes | 2.7 to 5.9 ms an element (median), plus loading each mesh (not measured) | under 1 hour |
| O-feat: about 40 five-fold fits (ablations, three seeds, variants) and bootstrap intervals | 31 s a five-fold fit of the 18-label model | under 1 hour |
| Probe and head on IFCNet's embedding (variant I) | 47.7 ms an element for the embedding, 3.2 s a probe | about 15 minutes |
| O-net at 1,024 points, k = 20, N = 10,000, E = 100 | 0.21 s a sample (eight threads) | 58 hours a run; about 7 days for three seeds |
| O-net at 2,048 points, k = 40, same N and E | 0.94 s a sample | 262 hours (11 days) a run |
| Variant I full fine-tune, 6,715 corpus elements, 30 epochs at 2,048 points | 0.94 s a sample | 53 hours |
| Data acquisition, conversion, licence manifest, label audit | supervised work | two to four weeks |

Without O-net, the computing is a few CPU hours and the campaign's duration is set by the data work. With O-net at 1,024 points, add about a week of CPU for three seeds, run at night or at reduced threads while the other user's job runs; at 2,048 points O-net is not practical on this machine. The feature time doubled in the second extraction, which ran beside the pilot's own model fitting, so every timing above can double when the machine is busy.

### 6.4 Risks

- Domain gap: library meshes differ from IFC families in tessellation, detail and parts. Mitigation: train on libraries, test only on IFC, and report the gain from library data on the IFC test set.
- Scale: several libraries are not in metres (game assets), while the height features need metres. Mitigation: keep only sources with real-world scale or rescale by documented category heights, and record which.
- Label noise in the truth from names, and in the hand labels. Mitigation: the audit of Section 6.2 and a second labeller on a subset.
- Small test sets give wide intervals (Table 8). Mitigation: more open IFC buildings with furniture, and intervals reported throughout.
- Overfitting the design to the corpus by repeated evaluation. Mitigation: freeze the features and settings before the final test set is opened, and keep H1 to H3 as written in Section 1.
- Licence drift (ABO and Quaternius changed their terms recently). Mitigation: archive the licence page with each download and record it in the manifest.
- Browser cost: the 18-label boosted model is 8.83 MB. Mitigation: cap rounds and leaves or use a small MLP, and check that onnxruntime-web runs the exported operator.
- CPU contention from the other user's job.

### 6.5 Novelty after the pilot

The pilot sharpens the claims of Table 4. The strongest is now an evaluation finding: under a leave-buildings-out protocol, browser-computable geometric features beat a published IFC point network and a probe on its embedding by wide margins, and a names-first cascade improves the rules on typed elements. The furniture sub-type claim stays strong as a task but needs the campaign's data before it shows more than the bounding box can. A paper that reports the negative results with the positive ones (the embedding adds nothing, learned models hurt proxies without proxy training data, open-only training is data-limited) and publishes the openly licensed model with its provenance would be new on each count named in Section 3.

---

## 7. Decisions for the user

1. Scope: O-feat only (a few CPU hours and two to four weeks of data work), or O-feat with O-net at 1,024 points (add about a week of CPU for three seeds). The pilot supports O-feat only.
2. Data: which open sources to download (Kenney, Poly Haven through its API, FreeCAD-library, ABO, the CC BY and CC0 part of Objaverse, BIMGEOM, more open IFC models) and the disk budget, outside OneDrive.
3. The licence of the published model: CC BY 4.0 (natural when CC BY data are used, with an attribution file) or MIT (code-like, still with an attribution file for CC BY data); and whether to restrict training to CC0 data to avoid attribution altogether, at the cost of coverage.
4. Whether to write to IFCNet's authors for a licence on their weights and dataset; with one, variant I could be published too, but the pilot gives no accuracy reason to need it.
5. The sub-type vocabulary and the kept set (Table 6), and whether to hand-label the 473 unnamed furniture elements as the final test set.
6. Deployment rule: the pilot shows that the cascade helps typed elements and hurts proxies until proxies are in training; whether the application should use the features only for typed furniture-class elements for now.
7. Venue: Automation in Construction or Advanced Engineering Informatics for the classification method, or Fire Safety Journal or Fire Technology if the ignition ranking is the centre.

---

## 8. Limits

- The sub-type truth comes from names for 646 of 649 elements, read by a vocabulary written for this study; it has not been audited, and the confusion of tables with electronics suggests that some names mislead.
- The sub-type test is small (649 elements, 19 buildings), so differences of a few points are not resolved (Table 8).
- The hand-labelled proxies are the benchmark's: 159 elements labelled from names by the benchmark's author and dominated by "other".
- Settings were fixed before scoring and not tuned, so each learned model may be below its best; the comparison between them is fair, but their absolute scores are conservative.
- Feature timings come from Node, not a browser, and doubled when the machine was busy.
- The ONNX tree model was run in native onnxruntime only.
- The licence readings are conservative interpretations of stated terms, not legal advice; Autodesk's terms pages could not be read, and several Elsevier abstracts were available only from secondary pages, so the full texts of Utkucu et al., Liu et al., Jing et al. and Jiang et al. remain to be read.
- IFCNet's embedding was used frozen; a fine-tune could make it more useful, but would still be an unlicensed derivative.

---

## 9. Reproducing

1. The benchmark's data (`node tools/identify-bench.mjs extract` and `prepare`, with the hand labels) in `~/ifcnet-src/bench-data`, as in the benchmark note.
2. `node tools/identify-features.mjs extract --bench ~/ifcnet-src/bench-data --out ~/ifcnet-src/pilot`: features, rule answers and sub-type labels for every element of the benchmark's dataset and every furniture element of the corpus (about 3 ms an element).
3. In the venv (`~/venvs/ifcnet`, with scikit-learn 1.9.1 and skl2onnx 1.20.0 added): `python -I tools/identify-pilot.py embed` (the pruned DGCNN's embedding, from `~/ifcnet-src/onnx/pruned-fp32-n2048-b8.onnx`), `timetrain`, then `run --threads 15`, which writes `benchmark/results/2026-11-05-ifcnet-pilot.json` (about 12 minutes; the numbers are the same on every run, seed 20261105).
4. A second extraction to another folder, then `python -I tools/identify-pilot.py recheck --check DIR`, adds the reproducibility check and the ignition-pattern census to the results file. The `literatureSearch` and `doiVerification` blocks were added from the search log and the DOI checks of this phase and are not regenerated by the scripts.

---

## References

Every DOI below was resolved on 2026-11-05 at Crossref (DataCite for 10.48550 and 10.7910 and 10.2312) and at OpenAlex.

- Austern, G., Bloch, T., Abulafia, Y. (2024). Incorporating context into BIM-derived data: leveraging graph neural networks for building element classification. Buildings 14(2), 527. https://doi.org/10.3390/buildings14020527
- Collins, F. C., Braun, A., Ringsquandl, M., Hall, D. M., Borrmann, A. (2021). Assessing IFC classes with means of geometric deep learning on different graph encodings. EC3 2021. https://doi.org/10.35490/EC3.2021.168
- Collins, F. C. et al. (2021). BIMGEOM [dataset]. Harvard Dataverse. https://doi.org/10.7910/DVN/YK86XK
- Deitke, M. et al. (2022). Objaverse: a universe of annotated 3D objects. arXiv. https://doi.org/10.48550/arXiv.2212.08051
- Demantké, J., Mallet, C., David, N., Vallet, B. (2011). Dimensionality based scale selection in 3D lidar point clouds. Int. Arch. Photogramm. Remote Sens. XXXVIII-5/W12, 97-102. https://doi.org/10.5194/isprsarchives-XXXVIII-5-W12-97-2011
- Demantké, J., Vallet, B., Paparoditis, N. (2012). Streamed vertical rectangle detection in terrestrial laser scans for facade database production. ISPRS Annals I-3, 99-104. https://doi.org/10.5194/isprsannals-I-3-99-2012
- Ding, Y. et al. (2025). Automatic assessment of fuel load and fire risk via digitized database and intelligent computer vision. Process Safety and Environmental Protection 197, 107031. https://doi.org/10.1016/j.psep.2025.107031
- Elhami-Khorasani, N., Salado Castillo, J. G., Gernay, T. (2020). A digitized fuel load surveying methodology using machine vision. Fire Technology. https://doi.org/10.1007/s10694-020-00989-9
- Emunds, C., Pauen, N., Richter, V., Frisch, J., van Treeck, C. (2021). IFCNet: a benchmark dataset for IFC entity classification. EG-ICE 2021, arXiv. https://doi.org/10.48550/arXiv.2106.09712
- Emunds, C., Pauen, N., Richter, V., Frisch, J., van Treeck, C. (2022). SpaRSE-BIM: classification of IFC-based geometry via sparse convolutional neural networks. Advanced Engineering Informatics 53, 101641. https://doi.org/10.1016/j.aei.2022.101641
- Frías, E., Pinto, J., Sousa, R., Lorenzo, H., Díaz-Vilariño, L. (2022). Exploiting BIM objects for synthetic data generation toward indoor point cloud classification using deep learning. Journal of Computing in Civil Engineering. https://doi.org/10.1061/(ASCE)CP.1943-5487.0001039
- Hackel, T., Wegner, J. D., Schindler, K. (2016). Fast semantic segmentation of 3D point clouds with strongly varying density. ISPRS Annals III-3, 177-184. https://doi.org/10.5194/isprs-annals-III-3-177-2016
- Jiang, Wei, Zhang (2027). Fire evacuation design optimization of public buildings based on building information modeling semantic enrichment for fire simulation requirements. Reliability Engineering and System Safety 277, 113173. https://doi.org/10.1016/j.ress.2026.113173
- Jing et al. (2026). Tri-modal unified representation-driven IFC element classification. Advanced Engineering Informatics. https://doi.org/10.1016/j.aei.2026.105040
- Koo, B., La, S., Cho, N.-W., Yu, Y. (2019). Using support vector machines to classify building elements for checking the semantic integrity of building information models. Automation in Construction. https://doi.org/10.1016/j.autcon.2018.11.015
- Koo, B., Jung, R., Yu, Y. (2021). Automatic classification of wall and door BIM element subtypes using 3D geometric deep neural networks. Advanced Engineering Informatics 47, 101200. https://doi.org/10.1016/j.aei.2020.101200
- Li and Guo (2026). Fire hazard identification in large-scale 4-dimensional building information models: a voxelization-based approach. Buildings 16(13), 2655. https://doi.org/10.3390/buildings16132655
- Liang et al. (2026). From image to fire safety: an image-driven framework for as-is BIM reconstruction and fire risk assessment of existing buildings via semantic guidance. Developments in the Built Environment. https://doi.org/10.1016/j.dibe.2026.100869
- Liu et al. (2024). Automatic fine-grained BIM element classification using multi-modal deep learning (MMDL). Advanced Engineering Informatics. https://doi.org/10.1016/j.aei.2024.102458
- Luo et al. (2023). A geometric-relational deep learning framework for BIM object classification. ECCV 2022 Workshops, LNCS. https://doi.org/10.1007/978-3-031-25082-8_23
- Poux, F., Neuville, R., Nys, G.-A., Billen, R. (2018). 3D point cloud semantic modelling: integrated framework for indoor spaces and furniture. Remote Sensing 10, 1412. https://doi.org/10.3390/rs10091412
- Poux, F., Billen, R. (2019). Voxel-based 3D point cloud semantic segmentation: unsupervised geometric and relationship featuring vs deep learning methods. ISPRS International Journal of Geo-Information 8(5), 213. https://doi.org/10.3390/ijgi8050213
- Poux, F., Mattes, C., Selman, Z., Kobbelt, L. (2022). Automatic region-growing system for the segmentation of large point clouds. Automation in Construction 138, 104250. https://doi.org/10.1016/j.autcon.2022.104250
- Poux, F. (2025). 3D Data Science with Python. O'Reilly Media. ISBN 978-1-0981-6133-0.
- Seydgar, Poirier, Motamedi (2024). Comparative evaluation of deep neural network performance for point cloud-based IFC object classification. IEEE Access. https://doi.org/10.1109/ACCESS.2024.3436681
- Shi et al. (2026). Self-supervised learning for BIM element classification using a joint embedding predictive architecture. Automation in Construction. https://doi.org/10.1016/j.autcon.2026.107075
- Tang, Bito, Shide (2025). Automatic classification of BIM object based on IFC data using the Uniclass classification standard. Buildings 15(13), 2347. https://doi.org/10.3390/buildings15132347
- Utkucu, D., Ying, H., Wang, Z., Sacks, R. (2024). Classification of architectural and MEP BIM objects for building performance evaluation. Advanced Engineering Informatics 61, 102503. https://doi.org/10.1016/j.aei.2024.102503
- Weinmann, M., Jutzi, B., Hinz, S., Mallet, C. (2015). Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers. ISPRS Journal of Photogrammetry and Remote Sensing 105, 286-304. https://doi.org/10.1016/j.isprsjprs.2015.01.016
- West, K. F. et al. (2004). Context-driven automated target detection in 3D data. Proc. SPIE 5426, 133-143. https://doi.org/10.1117/12.542536
- Wu, J., Akanbi, T., Zhang, J. (2022). Constructing invariant signatures for AEC objects to support BIM-based analysis automation through object classification. Journal of Computing in Civil Engineering. https://doi.org/10.1061/(ASCE)CP.1943-5487.0001012
- Yang et al. (2025). BIMCompNet: multimodal dataset for geometric deep learning in building information model. ACM Multimedia 2025. https://doi.org/10.1145/3746027.3758238
- Yang et al. (2026). Lightweight framework for IFC element classification using multi-view and heterogeneous graph with decision-level fusion. Automation in Construction 181. https://doi.org/10.1016/j.autcon.2025.106660
