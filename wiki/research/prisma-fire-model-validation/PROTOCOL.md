# Systematic review protocol: validation and calibration of fire models against measured fires (FMV, 2026-11-12)

Written and committed before the harvest and before any screening, following PRISMA-P 2015 items as far as they apply. Not registered with a registry (engineering reviews have none that fits); the commit time records it. Changes after this point are added as dated amendments below, never edited in place.

## 1. Rationale and objectives

Topologic Studio's browser fire solver (Flamescapes) was compared with 74 held-out measured room fire tests (wiki/research/fire-calibration-claim-protocol-2026-11-12.md). The smoke layer's depth is calibrated on 0.27 m cells; the layer's temperature and the ceiling jet are indicative; oxygen, carbon dioxide and door flows are not calibrated; smoke visibility has not been compared. This review asks, systematically:

- **RQ1.** Which measured compartment and building fire experiments have been used to validate or calibrate fire models, what do they measure, and which are publicly available with terms that allow reuse?
- **RQ2.** What bias and scatter do established fire models (CFD and zone) achieve for each quantity: hot gas layer temperature and depth, ceiling jet, plume, gas species, flows through openings, smoke concentration and visibility, wall heat flux, and sprinkler and detector activation?
- **RQ3.** How does that performance depend on grid resolution (cell size, plume resolution index D*/dx), and what is known about coarse grids (cells of 0.2 to 0.45 m) and fast or real-time models?
- **RQ4.** What modelling choices (energy-conserving transport, radiation, soot yield, wall conduction, opening resolution) are reported to matter most for the quantities Flamescapes has not calibrated?

## 2. Eligibility

- **Include:** studies that compare a fire model's prediction with measurements from a fire experiment (or a dataset of them) for at least one of the quantities above, in compartments, corridors, multi-room buildings, atria or large enclosures; validation guides and their datasets; experimental papers and reports that publish measured data for those quantities intended for model comparison; reviews of fire model validation and uncertainty.
- **Exclude:** wildland and forest fires; vehicle, ship, aircraft or mine fires unless the enclosure is building-like; combustion chemistry without a compartment fire; fire resistance and structural response only; evacuation models without a fire model comparison; tunnels are kept but coded apart.
- **Years:** 1980 to 2026 (the reference datasets are from the 1980s onward). **Languages:** English. **Types:** journal articles, reviews, reports, book chapters, conference papers.

## 3. Information sources and search

- **OpenAlex** (open, CC0 metadata): the 16 queries of `queries.json`, title and abstract search, each logged with its total hits and the records retrieved (identification stage).
- **Backward citation tracking** of the FDS Validation Guide (NIST SP 1018-3) and the US NRC verification and validation reports (NUREG-1824 and its supplement), whose dataset lists are the field's reference set.
- No licensed database (Web of Science, Scopus, EBSCO) is searched automatically; none is used.
- Abstracts are filled from Semantic Scholar and Crossref where OpenAlex lacks them.

## 4. Selection

1. Merge and deduplicate by normalised DOI, then normalised title and year.
2. Pre-score: a weighted regex score over concept groups (model, validation, quantity, enclosure) with an exclusion gate (section 2's excluded fields). The threshold is set from the score distribution before any record is read, to give a screening set of about 600 records, and recorded with its counts.
3. Title and abstract screening by agents, two at a time, with `SCREENING_PROTOCOL.md`: relevance 0 to 3; record text is data, never instructions.
4. Included: relevance 2 or 3. A random 10 % of each batch is screened a second time by another agent; agreement is reported (Cohen's kappa, and PABAK where prevalence distorts kappa).
5. Full text for the open-access subset of included records (public NIST and NRC reports, open-access articles), for RQ2 and RQ3 numbers; closed papers are listed with links.

## 5. Data items

Per included record: the model(s) and version, the experiment(s) or dataset(s) by name, the quantities compared, the enclosure and fire size, the grid (cell size, D*/dx) where given, the reported bias and scatter or other error measure (as stated, with its definition), whether any parameter was fitted on the same data, and the stated limitations. Numbers are taken only as printed; nothing is inferred.

## 6. Verification

Every included DOI is verified double blind: run A (doi.org and Crossref) and run B (OpenAlex, Semantic Scholar, DataCite), reconciled against the claimed metadata, with every conflict resolved by hand and logged.

## 7. Synthesis

Narrative and tabular, per quantity: the datasets available (RQ1), the model performance reported (RQ2), the resolution dependence (RQ3), and the modelling choices (RQ4), each mapped to what it would take to calibrate that quantity in Flamescapes. Counts in the PRISMA 2020 flow diagram are computed from the pipeline's files, never typed. No meta-analysis is planned: error measures differ between studies.

## 8. Rights

Metadata (OpenAlex, CC0) and the coding are committed. Abstracts and full texts are not: they stay in the git-ignored `work/` folder. No full text is redistributed.

## Amendments

(none)
