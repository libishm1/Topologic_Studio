# Flamescapes: Simulating the Wildlandfire Urban Interface (transcription)

Wrede, H., Wagner, A. R., Pałubicki, W., Michels, D. L., Pirk, S. (2026). Flamescapes: Simulating the Wildlandfire Urban Interface. ACM Transactions on Graphics 45(6), Article 246. https://doi.org/10.1145/3842518

The DOI is as printed on the paper; Crossref did not resolve it on 2026-11-05 (the project page says "DOI (coming soon)").

Licence: Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/), as stated on the paper's first page. © 2026 the authors.
Changes: transcribed from the authors' PDF (project page https://helgewrede.github.io/flamescapes/, retrieved 2026-11-05) to Markdown on 2026-11-05; equations retyped in LaTeX; figures replaced by their captions; layout changed. Transcription errors are ours; cite the original.

Transcription conventions: page breaks of the original are marked with HTML comments such as `<!-- page 246:5 -->`, placed at the paragraph boundary nearest to the break; figure captions are placed at a paragraph boundary close to where the figure sits in the original; plot axis labels are given in brackets after a caption only where they carry numbers; equation numbers are kept as `\tag{n}`; Algorithm 1 is reproduced as a numbered code block with its mathematical content repeated in LaTeX below it. Unreadable glyphs would be marked `[?]` (none were needed). Points where the original may contain a typo or inconsistency are listed, unchanged in the text, under "Transcriber's notes" at the end.

The companion transcription of the supplementary material is `flamescapes-2026-supplement.md`; the equation catalogue is `flamescapes-2026-equations.json`.

---

<!-- page 246:1 -->

# Flamescapes: Simulating the Wildlandfire Urban Interface

HELGE WREDE\*, Kiel University, Germany
ANTON R. WAGNER\*, Kiel University, Germany
WOJTEK PAŁUBICKI, Adam Mickiewicz University, Poland
DOMINIK L. MICHELS, KAUST, KSA
SÖREN PIRK, Kiel University, Germany

> Figure 1: An urban wildfire simulated with our framework: by leveraging a fuel-centric representation for modeling the environment, our method allows us to simulate the combustion of vegetation and buildings, while considering several fire propagation mechanisms.

We present a physically based simulation framework for wildland–urban interface (WUI) fires that couples material-aware urban scenes with interactive combustion simulation. Motivated by the dominant WUI exposure pathways of direct flame contact, thermal radiation, and ember transport, our framework combines heterogeneous solid combustion, radiative heat transfer, ember transport, and explicit flame-front propagation. Urban scenes are generated from OpenStreetMap footprints using a procedural pipeline that represents buildings as combustible 3D structures composed of semantic atomic modules with material and fuel annotations. Building on a chemistry-based hybrid Eulerian–Lagrangian fire solver, we model material-dependent heating, conduction, pyrolysis, oxidation, fuel depletion, and ember generation. A hierarchical fuel graph couples semantic scene structure to the solver's solid fuel grid, enabling subcomponent-level mesh degradation. We demonstrate structure-to-structure ignition, radiative preheating, material-dependent burning, ember-driven spot fires, and validate the method through ablations and comparisons to real-world combustion experiments.

CCS Concepts: • **Computing methodologies** → **Physical simulation**.

Additional Key Words and Phrases: Combustion, Fire, Fluid Dynamics, Level of Detail, Numerical Simulation, Physics-based Modeling, Wildfires.

**ACM Reference Format:**
Helge Wrede, Anton R. Wagner, Wojtek Pałubicki, Dominik L. Michels, and Sören Pirk. 2026. Flamescapes: Simulating the Wildlandfire Urban Interface. *ACM Trans. Graph.* 45, 6, Article 246 (December 2026), 19 pages. https://doi.org/10.1145/3842518

\*Both authors contributed equally to this research.

Authors' Contact Information: Helge Wrede, helge.wrede@email.uni-kiel.de, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany; Anton R. Wagner, awa@informatik.uni-kiel.de, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany; Wojtek Pałubicki, wojciech.palubicki@amu.edu.pl, Adam Mickiewicz University, 61-614, Poznań, Poland; Dominik L. Michels, dominik.michels@kaust.edu.sa, KAUST, Thuwal 23955, KSA; Sören Pirk, soeren.pirk@gmail.com, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany.

This work is licensed under a Creative Commons Attribution 4.0 International License.
© 2026 Copyright held by the owner/author(s).
ACM 1557-7368/2026/12-ART246
https://doi.org/10.1145/3842518

<!-- page 246:2 -->

## 1 Introduction

Simulating fire spread and combustion at the wildland–urban interface (WUI) presents significant challenges due to the complex interplay of fluid dynamics, thermodynamics, heterogeneous fuels, and the complexity of the urban layout. WUI fires occur where natural vegetation meets human development, and their frequency and severity have surged with expanding urbanization and climatic extremes [Cunningham et al. 2025; Spyratos et al. 2007] – the recent Eaton and Palisades Fires in Southern California are testaments for this threat. Driven by extreme winds and drought, these fires burned tens of thousands of acres, destroyed thousands of homes, and displaced hundreds of thousands of residents, marking them among the most devastating urban fire disasters in U.S. history [CALFIRE 2025; FireRescue 2026]. While substantial progress has been made in modeling wildfire spread across landscapes and fuels [Valdivia et al. 2026], existing approaches often abstract or omit detailed representations of urban geometry, materials, and structure-level combustion. Furthermore, they cannot be used to generate high-quality visuals of urban wildfires; however, accurately modeling the interactions of fire and complex structures at the scale of cities – also with a focus on visual quality – is critical for risk assessment, mitigation planning, and resilience strategies.

Computer graphics has made substantial progress in simulating and rendering fire, smoke, and turbulent fluids [Kim et al. 2008; Nguyen et al. 2002; Nielsen et al. 2019; Stam 1999], but most methods simplify combustion chemistry or focus on isolated effects such as gaseous flames and visually driven fire. Recent wildfire simulations model vegetation combustion, heat transfer, and landscape-scale propagation with impressive realism [Hädrich et al. 2021; Kokosza et al. 2024; Liu et al. 2025], yet typically avoid the geometric and material complexity of urban structures. Other graphics work addresses related phenomena, including water, phase changes, multi-fluid interaction, and solid melting or burning [Braun et al. 2025; Clavet et al. 2005; Losasso et al. 2006a,b; Mihalef et al. 2006]. However, coupling reactive fire dynamics with heterogeneous solid materials and complex urban geometry remains largely unaddressed.

In this work, we present a physically based simulation framework for WUI fires that couples combustion dynamics with multi-scale, material-aware modeling of buildings and vegetation. Our approach builds on an interactive chemistry-based hybrid Eulerian–Lagrangian fire solver [Wrede et al. 2025] and extends it to WUI scenarios with heterogeneous solid combustion, radiative heat transfer, material-dependent heat conduction, ember transport, and explicit flame-front propagation. This combination is motivated by the dominant WUI exposure pathways of direct flame contact, thermal radiation, and ember transport [Purnomo et al. 2024]. By explicitly modeling heat exchange between gases and solids, as well as pyrolysis, oxidation, fuel depletion, and ember generation in heterogeneous solid fuels, our framework captures street-scale fire spread across vegetation, terrain, and complex urban structures, including radiative preheating of facades, delayed ignition through thermal conduction, material-dependent structural burning, structure-to-structure ignition, and ember-driven spot fires.

We procedurally generate 3D urban scenes from OpenStreetMap (OSM) [OpenStreetMap contributors 2017] data by converting building footprints, street layouts, vegetation, and ground materials into combustion-ready scene components. Because solid combustion requires buildings to burn as smaller material-dependent parts rather than monolithic meshes, our pipeline represents buildings as semantic atomic modules with material and fuel annotations. These modules, together with vegetation elements, are organized into a hierarchical fuel graph that stores material properties, fuel mass, and combustion state, is voxelized into the solver's solid fuel grid, and maps fuel depletion back to visible mesh updates. This coupling enables material-dependent fire–structure interaction and progressive subcomponent-level burn-down in large-scale WUI scenes.

In Fig. 1, we demonstrate a large-scale WUI simulation where flames propagate from surrounding vegetation into a dense residential neighborhood. Fire spreads across terrain, structures, and streets, while embers ignite secondary fires on rooftops and within urban lots. In summary, our contributions are: (1) an interactive WUI fire simulation framework that extends a chemistry-based hybrid Eulerian–Lagrangian fire solver with heterogeneous solid combustion, radiative heat transfer, material-dependent conduction, ember transport, and explicit flame-front propagation; (2) a procedural pipeline for generating combustion-ready urban environments from OSM data, including buildings composed of semantic atomic modules with material and fuel annotations; (3) a hierarchical fuel graph that enables mapping fuel consumption to subcomponent-level mesh degradation; and (4) a set of experiments showing street-scale WUI fire behavior, including structure-to-structure ignition, radiative preheating, material-dependent burning, and wind-driven ember spot fires.

## 2 Related Work

Fire and combustion simulation has long been studied across computer graphics, computational fluid dynamics, and fire science. Foundational work in combustion theory and flow modeling is presented in the texts of Merci and Beji [2016], Kroos and Potter [2014], Peters [2000], and Bridson's introduction to fluid solvers [Bridson 2015]. Most fire solvers model fluid motion using incompressible or weakly compressible Navier–Stokes equations, typically discretized via semi-Lagrangian advection and pressure projection schemes [Versteeg and Malalasekera 2007]. Realistic fire behavior further requires coupling fluid motion with thermodynamics and chemical kinetics, where combustion reactions are commonly described using Arrhenius-type rate laws [McNaught and Wilkinson 1997] and the associated heat release is computed from thermochemical principles [Schmidt-Rohr 2011].

*Methods in Computer Graphics.* In computer graphics, fire and smoke are typically simulated using grid-based CFD solvers [Bridson and Müller-Fischer 2007], which can capture both laminar and turbulent flame behavior [Hong et al. 2007; Nguyen et al. 2002; Stam 1999] and visually compelling smoke dynamics [Fedkiw et al. 2001; Pan and Manocha 2017]. A broad spectrum of techniques has been developed, ranging from physically based combustion and flame models [Nguyen et al. 2002; Nielsen et al. 2022, 2019; Pegoraro and Parker 2006] to artist-driven and controllable methods [Aguilera and Johansson 2019; Kim et al. 2017; Lamorlette and Foster 2002], as well as particle-based approaches [Horvath and Geiger 2009]. Vorticity confinement [Bridson 2015] is widely used to enhance visual detail in under-resolved turbulent regions. Another research direction focuses on the integration of combustion dynamics with reconstructed environments [Feng et al. 2025; Nazarenus et al. 2026; Shen et al. 2026]. Despite this progress, many graphics systems either simplify combustion chemistry, focus primarily on gaseous flames, or treat solid fuels and the surrounding scene geometry in a limited way.

<!-- page 246:3 -->

*Physics-based Fire Simulations.* Physics-based fire simulators aim to reproduce underlying thermodynamic and chemical phenomena governing combustion, typically incorporating convection, conduction, radiation, and finite-rate chemical kinetics [Merci and Beji 2016]. Radiative heat transfer is often modeled via Stefan–Boltzmann-type exchange and can be approximated using techniques such as ray casting to estimate radiative heating of nearby fuels [Stam and Fiume 1995]. Such approaches can produce realistic flame behavior and energy distribution and are widely used in high-fidelity visual effects [Nielsen et al. 2022, 2019; Pegoraro and Parker 2006] and robotics [Wagner et al. 2026]. However, these models are designed for predictive, offline simulation and may be too expensive for interactive or large-scale scene-driven exploration common in graphics workflows.

*Combustion of Solids.* Modeling the combustion of solids introduces additional challenges due to phase changes, insulating char layers, and geometry-dependent heat transfer. Several methods leverage volumetric grids to track disconnected fire fronts [Liu et al. 2012, 2009; Melek and Keyser 2002; Zhao et al. 2003] and fire spread across surfaces [Chiba et al. 1994]. Hong et al. [2010] model combustion under complex geometric constraints, while Pirk et al. [2017] focus on burning geometrically complex plant models. Stomakhin et al. [2014] propose a point-based heat transfer approach for melting and solidification, though without real-time interaction and diffusion at scale. More recently, Liu et al. [2025] incorporate signed distance fields for efficient surface queries and to model insulation effects caused by char layers, enabling more general wooden structure combustion.

*Wildfire Simulations.* Wildfire modeling has a long history in the fire science community, with solvers that span empirical and physics-based approaches. Widely used tools include the Fire Dynamics Simulator (FDS) [McGrattan et al. 2025a,b] and wildfire-oriented systems such as WFDS [Mell et al. 2007], WRF-Fire [Coen et al. 2013], CAWFE [Coen 2013], ABWiSE [Katan and Perez 2021], FIRETEC [Linn et al. 2002], OpenFOAM [Lapointe et al. 2020], and QUIC-Fire [Linn et al. 2020]. Neither of these focus on computer graphics applications nor interactive simulations. They also do not provide any means for generating buildings or environments and only have limited capabilities for integrating geo-referenced data. Bakhshaii and Johnson [2019] review trade-offs between physical accuracy and computational cost across these families of approaches. In computer graphics, recent work has pushed physically based wildfire simulation toward interactive rates and improved geometric fidelity. Fire in Paradise [Hädrich et al. 2021] introduces mesoscale wildfire simulation over detailed vegetation geometry, while Scintilla [Kokosza et al. 2024] models combustible vegetation with rich visual and physical behavior. These advances are highly effective for forest and vegetation-dominated scenarios, but do not support urban wildfires. Appx. A.1 summarizes the scope of state-of-the-art fire and wildfire solvers and shows how our framework differs by coupling interactive WUI fire simulation with procedural combustion-ready urban modeling, semantic fuel decomposition, heterogeneous solid combustion, and subcomponent-level mesh degradation.

> Figure 2: Overview of the modeling pipeline: At initialization (solid arrows) we use OSM building polygons and street center lines as input (a). We then compute an SDF (b) for placing street and vegetation assets into the scene (d) and procedurally generate 3D models of buildings with WFC and SSA algorithms (c). The scene is then hierarchically organized into a fuel graph (e), meshed (f) and then voxelized to obtain a solid fuel grid (h) for our combustion solver. To enable the element-wise combustion of buildings and vegetation during simulation (dashed arrows) we update the durability (g, 1) and the fuel graph (2) to identify combusted mesh elements (3).

*Procedural Modeling of Urban Landscapes.* Procedural modeling provides a natural foundation for building large, heterogeneous urban environments that can be paired with physically based simulation. Early work such as Parish and Müller's procedural modeling of cities [Parish and Müller 2001] demonstrated how road networks, parcels, and building placement can be generated from high-level constraints. For architecture, Müller et al. [2006] introduced grammar-based building generation that remains foundational for urban content creation pipelines. A broader overview of procedural world building techniques is provided by Smelik et al. [2014], covering terrains, roads, vegetation, and city-scale modeling with an emphasis on controllability and interactivity. More recently, Niese et al. [2022] introduced procedural placement models for vegetation in urban layouts. These works motivate data-driven and procedural pipelines for constructing urban scenes with controllable structure and semantics – a prerequisite for WUI simulation where urban geometry, materials, and land use patterns crucially affect fire spread and ember exposure.

## 3 Overview

Figure 2 summarizes the pipeline from procedural urban modeling to coupled combustion simulation. Our goal is to represent the same environment both as renderable geometry and as simulation fuel. To this end, we organize buildings, vegetation, and ground elements into a hierarchical fuel graph that stores material properties, fuel mass, and combustion state at multiple semantic scales. This graph is voxelized into a solid fuel grid for heat transfer and combustion, while remaining linked to the original mesh components so that fuel depletion can drive visible burn-down.

<!-- page 246:4 -->

Starting from OSM building footprints and street centerlines, we construct a two-dimensional signed distance field (SDF) to classify roads, sidewalks, driveways, and vegetated regions, and place vegetation and urban assets to create heterogeneous fuel distributions. In parallel, we generate combustion-ready buildings from each footprint. Since solid combustion acts on smaller structural elements with distinct materials and burn behavior, we use Wave Function Collapse (WFC) to synthesize wall modules and the Straight Skeleton Algorithm (SSA) to generate roof modules, producing buildings as combustible subcomponents rather than monolithic surface meshes.

The resulting modules are assembled into the fuel graph, where atomic elements are grouped into larger semantic components such as wall sections, roofs, and buildings. At initialization, the graph is voxelized into a solid fuel grid aligned with the Cartesian simulation grid, providing the material-dependent state required for conduction, pyrolysis, oxidation, and solid–gas heat exchange. During simulation, updated voxel fuel states are mapped back through the hierarchy using per-module durability updates, allowing individual mesh components to be damaged or removed as they burn.

The gas phase builds on the chemistry-based hybrid solver of Wrede et al. [2025]. We couple this solver to the solid fuel grid through heat and mass exchange: solid combustion depletes fuel, releases heat, injects gaseous products, and emits embers, while the gas solver transports heat and species. We leverage the hybrid approach to propagate flame fronts and ember particles through the scene. Radiation is modeled with rays and coupled to the Eulerian simulation. We furthermore link solver-side combustion to visible subcomponent-level destruction.

## 4 Methodology

In this section, we describe the components of our WUI framework, beginning with the procedural modeling and hierarchical organization of urban scenes into a fuel graph, followed by the details of our combustion dynamics with material specific solid combustion, radiation, conduction and embers.

### 4.1 Procedural Urban Scene Modeling

We procedurally generate combustion-ready urban scenes from OpenStreetMap building footprints and street centerlines by converting buildings, ground, and vegetation into material-aware simulation inputs. Since our solid-combustion model operates on smaller material-dependent components, buildings are represented as collections of atomic modules rather than surface meshes. Wall segments are discretized into 2D grids and assigned structural or facade states, such as studs, beams, windows, and doors, using WFC [Alaka and Bidarra 2023; Karth and Smith 2017] with adjacency constraints that enforce plausible architectural layouts. The WFC rules and the placement of structural elements are based on official building codes [International Code Council, Inc. 2020]. WFC is a popular procedural content generation method for synthesizing structured scene layouts and we use it as an approach to obtain combustion-ready building components from footprint geometry. Roofs are generated separately with the SSA [Aichholzer and Aurenhammer 1996; Cheng and Vigneron 2007] by extending footprints to form overhangs and deriving roof elements such as rafters and shingles. The surrounding urban environment is represented with a 2D signed distance field, which classifies roads, sidewalks, driveways, and vegetated areas. Vegetation is placed in suitable regions using Poisson disk sampling to control density and avoid overlap. Together, these steps produce material-annotated urban scenes whose building elements can be inserted into the fuel graph and later removed or damaged independently during combustion. Figure 3 shows examples of generated environments. Further details on the WFC and SSA are provided in the supplementary material.

> Figure 3: We procedurally model urban scenes from OSM data and based on a combustion-ready representation. With our framework we can generate urban scenes that resemble real geographic locations. Here we show top-down views (a, b) of urban scenes, and close-up renderings of houses (c, d).

### 4.2 Organization of Fuel

To efficiently represent complex urban environments for fire simulation, we introduce a hierarchical organization of scene geometry to represent fuel, that we refer to as *fuel graph*. At the lowest level, all buildings and urban elements are decomposed into atomic material modules that represent units of combustible matter. Each atomic module encodes physical properties such as material type, density, molar mass, thermal conductivity, specific heat capacity, and Arrhenius parameters for combustion and pyrolysis. They serve as the smallest resolvable unit of fuel within the simulation. The numerical values assigned to each atomic module are summarized in Tabs. 4, 3, and are defined using representative room-temperature material properties. We manually assign the material properties (wood, steel, stone, etc.) to object classes (walls, roofs, vegetation, etc.) prior to the simulation.

The atomic modules are aggregated into higher-level semantic structures that correspond to meaningful architectural components, such as studs, beams, walls, floors, and roof assemblies. An illustration of a fuel graph describing the hierarchical relationship between modules is shown in Fig. 4. For example, individual material blocks may be grouped to form steel beams (Fig. 4b, d) or wooden studs, shingles are aggregated into roof sections, and wall layers are summarized into composite wall elements (Fig. 4a, c, e). At each level of the hierarchy, aggregated nodes maintain physically meaningful summaries of their constituent modules, including effective fuel mass, surface area, thermal properties, and combustion state which we retrieve from literature (Appx. A.3). In addition to buildings, vegetation is discretized directly into the volumetric fuel grid, where plant geometry is voxelized into combustible solid cells with material properties corresponding to the vegetation type, enabling uniform treatment of building and vegetation fuels for combustion.

<!-- page 246:5 -->

> Figure 4: Organization of fuel: To efficiently represent complex urban environments for combustion dynamics, we use a hierarchical data structure for buildings that we refer to as fuel graph (f). At the lowest level, we use atomic modules (a,b), which represent the smallest resolvable units of combustible material and encode physical and thermochemical properties. The atomic modules are aggregated into higher-level wall panels and studs (c,d), corresponding to semantically meaningful architectural components. At the next highest level, these components are combined to form a complete wall (e) and at the highest level entire houses (not shown). A fuel graph enables physically meaningful aggregation of thermal properties and enables fire-related queries – such as heat transfer, ignition, and mass loss – to be evaluated at multiple semantic and spatial resolutions.

#### 4.2.1 Transforming Scene Geometry into the Solid Fuel Grid

The combustion solver operates on a uniform Cartesian *solid fuel grid* aligned with the Eulerian simulation grid, with cell size $\Delta x$. Each voxel stores gas composition (species mass fractions), solid composition (material volume fractions), and the corresponding thermophysical parameters required by conduction and reactions (Appx. A.3). A voxel is treated as *solid* if its total solid density exceeds a threshold, where solid density is computed from the stored solid volume fractions and the per-material densities. In other words, the occupancy flag is derived from composition rather than stored as an independent label. A voxel is treated as a *porous solid* if the density exceeds a smaller threshold, but stays below the solid threshold.

The scene geometry is written into the solid fuel grid once, before the simulation starts. We first reset all grid fields to zero. We then process the full set of instanced scene meshes, including building modules and vegetation instances. Each mesh already has a material identifier assigned during procedural generation. This identifier is mapped to one of the material classes used by the combustion solver.

For every voxel, we estimate how much of the voxel volume is occupied by each material. This is done by casting a fixed number of rays through the voxel [Ling et al. 2025], with $N = 64$ by default. The rays are placed using a low-discrepancy sampling pattern [Ahmed et al. 2025] so that their origins are evenly distributed over the voxel faces. Each ray then travels inward along the normal direction of the selected face. As a ray passes through the voxel, it records where it enters and exits mesh geometry. At each surface crossing, the ray switches between being outside and inside solid material. The segment lengths for which the ray is inside a material are accumulated separately for each material class. For example, if a ray passes through wood for part of its path and then through glass for another part, the corresponding wood and glass lengths are added to separate totals. Back-face hits outside of the voxel imply a fully contained solid voxel, while front-face hits imply an empty voxel. The material at each hit is obtained from the material identifier of the intersected mesh instance. After all rays have been processed, the accumulated in-material ray lengths are divided by the total ray length inside the voxel. This gives an estimate of the volume fraction of each material in that voxel.

In our scenes, foliage is frequently represented by thin billboard geometry, which has no physical thickness. Without correction, such leaves would contribute almost no solid volume to the fuel grid. To avoid this, when a ray intersects sparse biomass, we add an artificial solid length controlled by a configurable leaf-thickness parameter $\delta_l$. This gives foliage a finite effective fuel volume. Furthermore, some meshes are not watertight. In such cases, a ray may encounter a back-facing surface outside the voxel and incorrectly change its inside–outside state. To reduce this error, back-face hits beyond the voxel extent are checked with a reverse-direction ray query against the same instance. This validation prevents such hits from incorrectly classifying empty space as solid material. The result of this voxelization step is a set of per-material volume fractions for every voxel:

$$
\varphi_i = \frac{\sum_{r=1}^{N} l_{i,r}}{\sum_{r=1}^{N} d_r},
\tag{1}
$$

where $N$ is the number of rays per voxel, $l_{i,r}$ is the accumulated length of ray $r$ inside material $i$ and $d_r$ is the side-length of the voxel.

Moisture content is added as a linear correction based on the amount of dense (e.g. wood) and sparse (e.g. leaves) biomass in the voxel, since these materials represent the main moisture-carrying fuels.

$$
\varphi_{moisture} = r_d\, \varphi_{\mathrm{dense}} + r_s\, \varphi_{\mathrm{sparse}},
\tag{2}
$$

where $r_d$ and $r_s$ are the dense and sparse biomass moisture ratios. These volume fractions are then converted into mass fractions using the density of each material. After this conversion, each voxel contains the material volume fractions, moisture contribution, and derived properties required by the combustion solver. Thermophysical parameters are then evaluated directly from the resulting per-voxel material compositions.

*Modeling of Interior Fuel.* As we do not aim to model the interior layout and furniture configuration of houses, we cannot meaningfully represent the distribution of fuel load with atomic modules. Therefore, we statistically model interior fuel distributions based on empirically measured residential fire load densities. Let $q''$ denote the fire load density in MJ/m², and let $A$ be the building floor area. For each building, we first sample a total fire load density $q''_b$ from the residential distribution [Džolev et al. 2021] and the material properties of the indoor content [Johra 2021], and compute the corresponding total available combustion energy

$$
Q_b = q''_b A\,.
\tag{3}
$$

<!-- page 246:6 -->

> Figure 5: Combustion Dynamics: Overview of our solid-gas combustion solver. Solid fuels (b) are represented on a volumetric solid fuel grid (a) and evolve through pyrolysis and solid combustion (Sec. 4.5), producing embers and gaseous fuels (c). Gases are represented as a gas fuel grid used for gaseous combustion (Sec. 4.4) and flame-front propagation (Sec. 4.7). Thermal evolution (d) couples solids and gases via advection (Sec. 4.3), conduction (Sec. 4.9), convection, and radiation (Sec. 4.6), while chemical evolution (e) is handled through advection and diffusion. Particle-based embers (b, Sec. 4.8) provide a discrete transport mechanism across the domain, and the resulting fields are passed to the volumetric fluid renderer (f, Sec. 5.2) for each update step.

We convert this energy into an equivalent solid fuel mass using an effective net heat of combustion $\Delta_c H^{\mathrm{eff}}$ for mixed residential contents,

$$
M_b = \frac{Q_b}{\Delta_c H^{\mathrm{eff}}}\,.
\tag{4}
$$

$M_b$ represents the total initial combustible mass assigned to the building interior.

We distribute $M_b$ stochastically over interior solid cells. Let $\mathcal{I}_b$ be the set of interior solid voxels for building $b$, with $N_b = |\mathcal{I}_b|$. We define a per-voxel fuel mass random variable $m_{b,t}$ (in kg per voxel) and model interior heterogeneity by sampling

$$
m_{b,t} \sim \mathcal{N}(\mu_b, \sigma_b^2)\,, \qquad t \in \mathcal{I}_b\,.
\tag{5}
$$

We choose $(\mu_b, \sigma_b)$ per building such that the expected total mass matches the fire load implied mass and the variance is bounded by the reported spread of residential fire loads. Concretely, we constrain $\mu_b = M_b/N_b$ by the mass budget and set $\sigma_b$ as a building-specific dispersion parameter derived from the empirical variability in Džolev et al. [2021]. The resulting discrete field $(m_{b,t})$ of sampling fuel masses is mapped to the solid fuel grid as the initial remaining fuel mass for interior cells, providing per-building variability in total fuel load consistent with measured residential fire loads.

After the hierarchical fuel graph has been constructed and all material properties and aggregate fuel quantities have been assigned, the entire scene is discretized into a unified volumetric fuel grid. The combustion solver then updates this grid over time to model heat transfer, ignition, and fuel mass loss. To synchronize the rendered building meshes with the evolving voxel fuel state, we compute a per-module *durability* scalar from the fuel grid. For each atomic module (or each higher-level aggregate node in the fuel graph), we precompute a maximum durability $D_{\max}$ as the sum of the module's combustible fuel mass stored in voxels. During simulation, we recompute the mass sum to obtain the current durability $D(t)$. If $D(t) < \tau\, D_{\max}$, with a fixed threshold $\tau$, the corresponding mesh instance is flagged as destroyed and removed from rendering (and from subsequent durability tests). The destruction threshold $\tau$ is deliberately set at a low $\tau = 0.05$. This largely avoids fires originating from the fuel grid where the mesh has already been deleted. Furthermore, a small fade-out (alpha reduction) is added to reduce visual popping from disappearing meshes. This durability test does not affect combustion: it is only a visibility value derived from the voxel fuel grid.

### 4.3 Fluid Dynamics

An overview of our combustion model is shown in Fig. 5; the symbols and their respective units are shown in Appx. A.4. The simulation domain is discretized on a staggered Cartesian grid. Velocity components are stored on their respective cell face, while all other quantities are collocated at the cell center. Gas-phase quantities include velocity and chemical species mass fractions, while solid-phase cells store material volume fractions. The density and thermodynamic properties (temperature, heat capacity, thermal conductivity) are stored for each phase separately. Both phases are coupled through boundary fluxes and source terms. The different phase properties are denoted by the subscripts ${}_g$ for gas and ${}_s$ for solid.

Modifying the model proposed by Wrede et al. [2025], the gas flow is described by the Navier-Stokes equations with buoyancy effects and thermal expansion incorporated via an equation of state that couples density and temperature according to Nielsen et al. [2022]:

$$
\frac{D\boldsymbol{u}}{Dt} = -\frac{\nabla p}{\rho_g} - \boldsymbol{g}\left(1 - \frac{\rho_{amb}}{\rho_g}\right),
\tag{6}
$$

where $\boldsymbol{u}$ is the velocity field, $\rho_g$ is the gas density, $p$ is the pressure, $\boldsymbol{g}$ the effective gravity vector, and $\rho_{amb}$ the reference density (e.g. air at ambient temperature and pressure). Gas density–temperature coupling is then addressed with

$$
\rho_g = \rho_{\mathrm{amb}} \left(R T_g \sum_{i=0}^{n} \frac{Y_i}{M_i}\right)^{-1},
\tag{7}
$$

where $Y_i$ is the mass fraction of species, $R$ the universal gas constant, $T_g$ the gas temperature and $M_i$ the species specific molar mass.

This formulation is appropriate because fire-induced and wind-driven velocities in our scenarios remain in a low-Mach-number regime where high-speed compressibility effects are negligible. The dominant influence of combustion on the flow arises from temperature-driven buoyancy and expansion. Since fire spread, heat transfer, and ember transport are governed primarily by advection and thermal effects, this formulation captures the relevant physics of conflagrations while enabling computational efficiency and high visual fidelity.

<!-- page 246:7 -->

To model different types of solid structures, we introduce two types of solid-phase voxels. *Solid* cells, which fully block the flow of gases and *porous* cells, which only partially block gas flow. The state is determined by the amount of mass within the voxels. This allows us to model the gas flow interacting with vegetation structures that can not be resolved by the grid, without starving small air movements. The *solid* state is enforced with a Dirichlet boundary during the pressure projection step while the *porous* state uses an upper velocity bound.

Our hybrid, Eulerian-Lagrangian representation allows us to store and advect various different properties. All gas and solid properties are stored on the grid. Flame-front particles are represented by a position, velocity and temperature. Ember particles carry mass as well. Radiation rays only propagate thermal energy. For the spawning of and sampling between particles, rays and the grid, each particle or ray only interacts with one cell at a time, avoiding the need for interpolation.

### 4.4 Gas Combustion

Solid and gas temperatures evolve similarly. The solid and the solid-gas interface progression are described in the next sections. For the gas, we add a volumetric radiation term

$$
\rho_g C_{p,g} \frac{\partial T_g}{\partial t} = \nabla \cdot (k_g \nabla T_g) - \nabla q_{\mathrm{rad}} + \dot{q}_{c,g}\,,
\tag{8}
$$

where $C_{p,g}$ is the specific heat capacity of the mixture, $k_g$ is the mixture dependent thermal conductivity, $\nabla q_{rad}$ is the volumetric radiation (Eqs. 17) and $\dot{q}_{c,g}$ is the combustion heat source in the gas phase (Eqs. 9). Volumetric radiation losses depend on local soot concentration. Emission and absorption are handled separately.

We keep the gas combustion heat source term and combustion rate from Wrede et al. [2025]:

$$
\dot{q}_{c,g} = -\Delta_{c,g} H^0 v_{c,g},
\tag{9}
$$

with $\Delta_{c,g}H^0$ for the heat of combustion and $v_{c,g}$ as the combustion rate. The combustion rate $v_{c,g}$ is given by an Arrhenius-type expression

$$
v_{c,g} = A_{c,g} \exp\left(\frac{-E_{a,c,g}}{R T_g}\right) c_f^{a} c_0^{b}\,,
\tag{10}
$$

in which $A_{c,g}$ is the pre-exponential factor, and $E_{a,c,g}$ the activation energy for gaseous combustion. The molar concentrations of fuel and oxidizer $c_f$ and $c_o$ are raised to empirical exponents $a$ and $b$.

The chemical species evolution is modified to:

$$
\frac{\partial Y_i}{\partial t} + \boldsymbol{u} \cdot \nabla Y_i = D_i \nabla^2 Y_i + \dot{m}_{c,g} + \dot{m}_{e,s} + \dot{m}_{p,s} + \dot{m}_{c,s}\,,
\tag{11}
$$

where $Y_i$ is the mass fraction of species $i$, $D_i$ the species specific diffusion coefficient, $\dot{m}_{c,g}$ a gas combustion source term (Eq. 12), $\dot{m}_{e,s}$ a novel solid evaporation source term, $\dot{m}_{p,s}$ a novel solid pyrolysis source term (Eq. 14), and $\dot{m}_{c,s}$ a novel solid combustion source term (Eq. 15). For details on the solid pyrolysis and combustion, see Sec. 4.5.

> Figure 6: Visualizations of a burning tree (a), our ray-based radiation (b) as well as of particles for flame front propagation (c) and embers (d). Notably radiation is capable of transporting heat downwards and laterally but falls of quickly and is blocked by occluders, while the flame reaches primarily upwards. Flame front particles allow us to maintain a coherent flame shape which remains attached to the burning tree, even at low grid resolutions. Ember particle are more scattered and transport energy across larger distances and behind occluders.

The gas combustion species source term was left unchanged and is modeled by

$$
\dot{m}_{c,g} = \frac{v_{c,g}\, q_i M_i}{\rho_g}\,,
\tag{12}
$$

in which $q_i$ denotes the stoichiometric coefficient for species $i$.

### 4.5 Solid Combustion

We extend the aforementioned method with a solid combustion model, in which solid fuel is represented by material-specific volume fractions $\varphi_i$ stored in each voxel cell and a local solid temperature $T_s$. The specific heat capacity and thermal conductivity as well as the density are derived from the composition within the cell. The temperature progression is described in detail in Sec. 4.9. The solid density $\rho_s$ and specific heat capacity $C_{p,s}$ are given by

$$
\rho_s = \sum_{i=0}^{n} \varphi_i \rho_i\,, \qquad C_{p,s} = \frac{\sum_{i=0}^{n} \varphi_i C_{p,i}}{\sum_{i=0}^{n} \varphi_i}\,,
\tag{13}
$$

where $\rho_i$ is the material specific density and $C_{p,i}$ the material specific heat capacity. The thermal conductivity is determined analogous to the specific heat capacity.

Combustion proceeds in three stages: (1) evaporation of any moisture, (2) oxygen-limited pyrolysis, which converts solid fuel into gaseous pyrolysate, and (3) oxidation of solid char (or remaining fuel) in the presence of oxygen. All three stages couple back to the gas through mass source terms for species as well as through heat release and consumption.

<!-- page 246:8 -->

> Figure 7: A Temporal progression of an urban wildfire. The fire starts as a small cluster of burning houses (a-c) and subsequently develops to a larger fire (d, e) until it spreads across multiple blocks of houses (f-h).

*Evaporation (with moisture).* In solid cells which contain a moisture fraction, evaporation is taking place before either pyrolysis or combustion. This process uses the evaporation model presented by Wrede et al. [2025] applied to moisture within the solids. The thermal energy stored in the solid is used to evaporate moisture which is converted into water vapor and injected into the gas $\dot{m}_{e,s}$. The solid thermal energy is reduced accordingly, represented by the sink term $\dot{q}_{e,s}$.

*Pyrolysis (without oxygen).* In regions with insufficient oxygen, heated fuel undergoes pyrolysis and releases gaseous volatiles. We also model this with the Arrhenius equation to obtain the pyrolysis reaction rate $v_{p,s}$, analogous to the gas reaction rate. This activates as temperature rises and is suppressed as oxygen becomes available. Pyrolysis consumes solid fuel mass and injects gaseous products into the cell via the species source term $\dot{m}_{p,s}$, while contributing a thermal sink $\dot{q}_{p,s}$ to the solid energy equation:

$$
\dot{m}_{p,s} = \frac{M_i}{\rho_i}\, v_{p,s}\,, \qquad \dot{q}_{p,s} = \Delta_{p,s} H^0\, v_{p,s}\,,
\tag{14}
$$

where $M_i$ is the molar mass of material $i$, $\rho_i$ the density of material $i$ and $\Delta_{p,s}H^0$ the heat of pyrolysis. This formulation captures pre-ignition devolatilization and enables the buildup of flammable gases that can subsequently burn in the gas phase.

*Oxidation (with oxygen).* When oxygen is present, solid oxidation consumes fuel and oxygen and releases heat. We use the Arrhenius based reaction rate $v_{c,s}$ with corresponding heat release in the solid and a coupled gaseous species source term applied to the gas-phase mixture:

$$
\dot{m}_{c,s} = \frac{M_i}{\rho_i}\, v_{c,s}\,, \qquad \dot{q}_{c,s} = \Delta_{c,s} H^0\, v_{c,s}\,,
\tag{15}
$$

where $\Delta_{c,s}H^0$ denotes the lower heat of combustion.

Evaporation, pyrolysis and oxidation continuously inject gaseous fuel species into the gas. Gas-phase combustion then consumes these volatiles based on local mixture fraction and oxygen availability, producing heat that feeds back into the solid through the interface fluxes. This two-way coupling enables drying of wet biomass, oxygen-starved smoldering via pyrolysis, and rapid flaming once adequate oxygen and temperature are present.

### 4.6 Radiation

Radiative heat transfer provides a short to medium range coupling mechanism between flames, hot gases, and solid fuel surfaces. We model radiation using a hybrid surface-ray formulation that captures diffuse emission from hot solids and combustion regions. Contrary to previous work [Wrede et al. 2025], the full radiative transfer is simulated. It is based on thermal radiative transport approximated by Monte-Carlo ray tracing [Bati et al. 2023], which allows us to propagate thermal radiation while remaining efficient for large-scale urban scenes.

*Surface emission.* Solid fuel surfaces emit thermal radiation once heated. For each solid cell adjacent to gas, radiative emission is modeled as diffuse (Lambertian) and proportional to the local surface temperature:

$$
q_{\mathrm{rad},s} = \epsilon\sigma\left(T_s^4 - T_{\mathrm{amb}}^4\right),
\tag{16}
$$

<!-- page 246:9 -->

where $T_s$ is the solid surface temperature, $\epsilon$ the material emissivity, $\sigma$ the Stefan-Boltzmann constant, and $T_{\mathrm{amb}}$ the ambient temperature. This radiative flux is applied as an energy loss term in the solid and as a corresponding gain in the surrounding gas and nearby solids hit by the radiation.

To transport radiative energy, we spawn rays on the outward-facing hemisphere of the emitting surface. Ray directions are sampled according to Lambert's cosine law, ensuring energy conservation and physically plausible angular distributions. Each ray carries a fraction of the emitted radiative energy and propagates straight through the domain.

*Volumetric emission and absorption.* In addition to surface emission, hot, sooty combustion gases contribute to volumetric radiation. This effect is modeled as a local energy loss term in the gas temperature equation:

$$
-\nabla q_{\mathrm{rad}} = 4 Y_s \kappa_s \sigma\left(T_g^4 - T_{\mathrm{amb}}^4\right),
\tag{17}
$$

where $Y_s$ is the soot mass fraction and $\kappa_s$ an effective absorption coefficient. This formulation captures net radiative cooling of hot gas regions without explicitly resolving wavelength-dependent emission and absorption.

Sooty combustion gases also absorb incoming radiation. For each incoming radiation ray, the probability of absorbing it $P_{\mathrm{abs}} = 1 - \exp(-\kappa_s Y_s \ell)$ is also depending on the absorption coefficient $\kappa$, the soot mass fraction $Y_s$ and path segment length $\ell$. This allows flames and hot plumes to radiatively preheat distant fuel elements, even in the absence of direct convection or conduction.

*Radiative coupling to solids.* When a radiation ray intersects a solid surface, its remaining energy is deposited into the corresponding solid cell as a heat source term in the solid temperature evolution equation. This mechanism enables radiative preheating of building facades, vegetation, and interior structural elements through openings, contributing to delayed ignition and structure-to-structure fire spread.

Radiation rays are terminated after interaction with a solid surface or sooty combustion gas. This hybrid treatment balances physical plausibility with computational efficiency, allowing radiation to act as a dominant heat transfer mode at short to medium ranges without requiring a full global radiosity or view-factor computation.

Figure 6 (b) illustrates the resulting temporal heating pattern under radiative exposure. A burning tree radiatively heats nearby structural elements that act as occluders.

### 4.7 Flame Front Propagation

Grid-based combustion models tend to confine ignition and reaction rates to locally resolved temperature fields, which can artificially slow flame propagation when the grid resolution is insufficient to capture thin reaction zones. In particular, ignition fronts may stall at cell boundaries, and the rapid transport of high-temperature states observed in real flames is underrepresented. To address this limitation, we introduce flame-front particles as a lightweight, Lagrangian mechanism for transporting ignition-relevant thermal energy across neighboring cells. These particles are markers for the high-frequency propagation of flames and do not conserve energy.

> Figure 8: Comparison of two wildfire scenarios illustrating the effect of vegetation clearance on building ignition. In the dense vegetation case (a-c), the availability of continuous fuel enables fire to propagate to the structure, resulting in building ignition. In contrast, no vegetation around the structure (d-f) interrupts fuel continuity and prevents ignition. This demonstrates the protective effect of empty space around houses in WUI scenarios.

When combustion occurs within a grid cell, flame-front particles are spawned and emitted uniformly in all six directions of a cell. Each particle stores the temperature of its originating cell and acts as a carrier of thermal energy. Flame-front particles are also spawned when a solid cell undergoes a combustion, to ignite any nearby gaseous fuel. Their velocity is determined by the fuel specific flame-front speed and affected by the surrounding gas velocity. This models flames accelerating along the gas flow, but also allows flames to propagate against the flow. This allows flames to progress downwards, against the rising hot air. During subsequent gaseous combustion evaluations, the particle temperature supersedes the local cell temperature whenever it is higher, enabling ignition to propagate beyond the immediately burning region. Flame-front particles are removed once they enter cells in which no combustion is triggered, preventing spurious propagation. Together, this process produces an avalanche-like behavior: every combusting cell generates six flame-front particles, each of which can ignite neighboring cells and recursively spawn additional particles, allowing the flame front to rapidly advance through the domain (Fig. 6c).

### 4.8 Sparks and Embers

We integrate ember generation and transport using the model of Kokosza et al. [2024]. Embers are treated as an additional Lagrangian particle set (Fig. 6,c). They are spawned from solid burning biomass cells at a specified rate. They are advected according to the underlying gas velocity field scaled by a drag factor. Each ember stores its mass and temperature. As the embers move through the air, they combust and radiate heat, ensuring an energy balance which controls the lifetime and effectiveness of the ember depending on its trajectory and surrounding air velocity.

Embers deposit their heat when they intersect a solid cell. At contact with combustible solids, the ember immediately triggers a small pyrolysis and combustion according to its carried mass and temperature. This process releases heat and potentially gaseous fuel. Furthermore, embers act like flame-front particles, as gaseous fuel is ignited if a hot enough ember is present. Combined, these processes allow us to model long range heat transport, spot ignition and smoldering, complementing the radiative heat transfer.

<!-- page 246:10 -->

### 4.9 Solid Temperature Evolution

Heat transport inside solid fuel elements is modeled by heat conduction on the voxelized solid domain. For each solid cell, temperature evolves according to

$$
\rho_s C_{p,s} \frac{\partial T_s}{\partial t} = \nabla \cdot (k_s \nabla T_s) + \dot{q}_{e,s} + \dot{q}_{c,s} + \dot{q}_{p,s}\,,
\tag{18}
$$

where $k_s$ denotes the thermal conductivity, $\dot{q}_{e,s}$ the evaporation source term, $\dot{q}_{c,s}$ heat release by solid oxidation (Eq. 15), and $\dot{q}_p$ correspond to the (endothermic) pyrolysis term as defined above (Eq. 14).

At solid-gas interfaces, conductive heat flux is balanced against convective and radiative exchange with the gas:

$$
-k_s \nabla T_s \cdot \mathbf{n} = h\left(T_s - T_g\right) + \epsilon\sigma\left(T_s^4 - T_{\mathrm{amb}}^4\right),
\tag{19}
$$

with outward normal $\mathbf{n}$, neighboring gas temperature $T_g$, heat transfer coefficient $h$, emissivity $\epsilon$, and Stefan–Boltzmann constant $\sigma$. The conductive heat flux is also applied within porous cells, to transfer energy between the solid and the gas flowing through the cell. To account for small turbulent effects and complex surfaces, which are not resolvable at realistic sizes, we introduce the heat transfer coefficient to accelerate the interface heat flux. These boundary conditions allows sub-surface heating to delay ignition while enabling radiative preheating and surface cooling consistent with the coupled gas solver.

## 5 Algorithmics and Implementation

Our simulation framework is implemented in `Rust` and relies on `WGPU` as the primary GPU abstraction layer, with compute kernels written in `WGSL`. The overall system architecture is built on the `shipyard` entity-component system, which enables structured and highly parallel management of simulation state. User interaction and visualization are provided through `winit` for cross-platform windowing. The results shown in this paper were produced on different GPUs. Depending on the simulation size, either an NVIDIA RTX 4090 or RTX Pro 6000 Blackwell GPU was used. The implementation closely follows the numerical formulation outlined in Alg. 1, which summarizes the full simulation loop and integrates an Eulerian fluid solver for gas and solid heat transport, ray casting for thermal radiation, particle dynamics for the flame-front and embers, and thermochemical models for evaporation, pyrolysis, phase transitions, and combustion.

**ALGORITHM 1: Hybrid combustion simulation.**

**Input:** Initialized Eulerian grid with gas/solid compositions and gas/solid flags; empty particle set.
**Output:** Updated grid and particle state for each frame.

```
    // - Initialization -
 1  forall Grid Cells do
 2  |   Initialize u ← u_0, T_s, T_g ← T_amb, Y_i ← Y_i^0;
 3  Voxelize scene geometry: update φ_i (Eqs. 1, 2);
 4  Distribute interior fuel: update φ_i (Eqs. 3, 4, 5);
 5  Update gas density: ρ_g (Eq. 7);
 6  Update properties for all compositions: ρ, C_p, k (Eqs. 13);
    // - Update Loop -
 7  forall Frames do
    |   // - Lagrangian Particle Step -
 8  |   Spawn new ember particles;
 9  |   Spawn new flame front particles;
10  |   forall Particles do
11  |   |   Apply drag from the velocity grid to the ember particles;
12  |   |   Apply velocity from the grid to the flame front particles;
13  |   |   Update positions of all particles;
14  |   |   Enforce boundary conditions;
    |   // - Thermal Radiation -
15  |   Cast radiation rays (Eqs. 16, 17);
    |   // - Particle-to-Grid Transfer -
16  |   forall Grid Cells do
17  |   |   Sample ember particles to deposit thermal energy;
18  |   |   Sample flame front particles to propagate the flame front;
    |   // - Eulerian Grid Simulation Step -
19  |   forall Grid Cells do
20  |   |   Advection: update u, T_g, Y_i (Eq. 6);
21  |   |   Buoyancy: apply force to u;
22  |   |   Apply vorticity confinement;
23  |   |   Diffuse composition: update Y_i (Eq. 11);
24  |   |   Evaporation of moisture: compute v_{e,s}, q̇_{e,s}, Y_i, φ_i;
25  |   |   Oxidation of solids: compute v_{c,s}, q̇_{c,s}, Y_i, φ_i (Eq. 15);
26  |   |   Pyrolysis of solids: compute v_{p,s}, q̇_{p,s}, Y_i, φ_i (Eq. 14);
27  |   |   Combustion of gases: compute v_{c,g}, q̇_{c,g}, Y_i (Eqs. 9, 10, 12);
28  |   |   Update occupancy based on compositions;
29  |   |   Update properties for all compositions: ρ, C_p, k (Eqs. 13);
30  |   |   Conduct temperature: update T_g, T_s (Eqs. 8, 18, 19);
31  |   |   Update gas density: ρ_g (Eq. 7);
32  |   |   Enforce boundary conditions;
33  |   |   Solve pressure projection (Eq. 6);
34  |   Remove embers with expired lifetime;
35  |   Remove flame front particles outside of the flame front;
36  |   Update the durability according to the solid combustion process;
```

Mathematical content of the algorithm lines in LaTeX: line 2, $\boldsymbol{u} \leftarrow \boldsymbol{u}_0$, $T_s, T_g \leftarrow T_{amb}$, $Y_i \leftarrow Y_i^0$; lines 3 and 4, $\varphi_i$; line 5, $\rho_g$; line 6, $\rho, C_p, k$; line 20, $\boldsymbol{u}, T_g, Y_i$; line 24, $v_{e,s}, \dot{q}_{e,s}, Y_i, \varphi_i$; line 25, $v_{c,s}, \dot{q}_{c,s}, Y_i, \varphi_i$; line 26, $v_{p,s}, \dot{q}_{p,s}, Y_i, \varphi_i$; line 27, $v_{c,g}, \dot{q}_{c,g}, Y_i$; line 30, $T_g, T_s$.

### 5.1 Simulation Loop

The combustion simulation mainly uses two types of storages, 3D textures and storage arrays. The textures are used to store scalar and vector grid values, while the storage arrays store particle related quantities. The grid-particle and particle-grid mapping uses both representations to provide quick retrieval when sampling between the two. Hardware acceleration is used for the radiation ray casting. Furthermore, fast particle spawning as well as grid-particle mapping require a fast and parallel particle sorting step in between simulation steps. The grid is initialized with typical air composition at atmospheric pressure and at 300K. Before the simulation is started, the grid is seeded from the procedural geometric model and runtime voxelization to determine which cells store what type of mass and in what quantities. No particles exist at the time of initialization.

Alg. 1 summarizes the per-frame update of the coupled Eulerian grid and Lagrangian particle systems. We follow an update structure consisting of a particle step, particle-to-grid transfer, and an Eulerian grid step, all executed once per rendered frame. For every grid cell, we initialize the Eulerian state variables to their defined initial conditions (lines 1-2). Specifically, the velocity field $\boldsymbol{u}$, temperatures $T_s$ and $T_g$, species mass fractions $Y_i$, and density $\rho_g$ are set to $\boldsymbol{u}_0$, $T_0$, $Y_i^0$, and $\rho_0$, respectively. The voxelization step updates solid materials according to the scene geometry (line 3) followed by the distribution of interior fuel (line 4). The gas and solid properties are then updated accordingly (lines 5-6).

<!-- page 246:11 -->

> Figure 9: Ablation of WUI fire-spread pathways for a vegetation-to-structure ignition: Each row shows a temporal sequence with a different combination of direct flame contact, radiative heat transfer, and ember transport enabled or disabled. With direct flame contact alone, fire spreads only when the flame reaches the building surface (a). Adding radiation increases heating of the facade and roof, leading to faster and more sustained structural ignition (b). Without direct flame contact and radiation, the fire remains localized to the vegetation and does not ignite the structure (c). Radiation alone can ignite the building across a gap between vegetation and structure (d), while embers introduce non-local spot ignitions on the roof (e). When all mechanisms are enabled, the combined effects produce the strongest and most persistent structure fire (f). Please note that the images show an orthographic projection of a 3D simulation.

At the beginning of each frame (line 7), we spawn two particle sets (lines 8-9). Ember particles are emitted from burning solid cells according to the solid combustion model and are assigned an initial position at the neighboring cell and initialized with the local velocity. Flame-front particles are emitted to represent rapid ignition transport and are seeded from cells that recently underwent either gas-phase or solid-phase combustion.

Each particle is then advanced (lines 10-14). Ember particles receive a velocity update from the Eulerian velocity field (line 11). In practice, we sample the grid velocity at the particle position, and apply a first-order drag model that relaxes ember velocity toward the local gas velocity. Flame-front particles spawn with a combustion dependent velocity and are advected based on the local grid velocity (line 12), see Sec. 4.7 for more details. All particles update their positions using their velocities (line 13), and boundary conditions are enforced (line 14). Boundary handling clamps or reflects particles at domain boundaries and removes particles that enter fully solid regions, depending on particle type.

To simulate the radiative heat transfer, radiation rays are emitted from hot gas regions and from solid surface cells that satisfy the emission criterion described in Sec. 4.6 (line 15). The energy is deposited at the intersection of the ray and an absorbing sooty cell or the surface of a solid cell, which is accumulated as an additive source term in the temperature update.

After the particle update and radiation step, we deposit particle effects back onto the Eulerian grid (lines 16-18). Ember particles deposit thermal energy into a solid surface cell (line 17), when intersecting the corresponding cell. Flame-front particles update an ignition proxy on the grid (line 18) used by the combustion model to accelerate ignition where particle temperature exceeds the local cell temperature. We provide pseudocode for the radiation and particle-to-grid transfer in the supplementary material.

<!-- page 246:12 -->

> Figure 10: Temporal progression of urban wildfire simulations with (a-d) and without (e-h) ember transport under wind-heavy conditions. When embers are enabled, wind advects hot particles ahead of the flame front, leading to ignitions that can even overcome fuel breaks, such as streets (c, d). In the absence of embers, fire propagation remains more localized (g, h), highlighting the combined role of wind and ember transport in driving long-range ignition in urban wildfire scenarios. The top-down temperature maps for scenarios with (i-l) and without (m-p) embers provide a more detailed view of the fire propagation.

> Figure 11: Visualization of fire propagation at different levels of the hierarchical fuel graph. At the coarsest building-level, it is only possible to remove large structural components, such as walls (a-c). The second row (d-f) visualizes the intermediate level consisting of wall panels and studs, revealing more detailed structural degradation. The third row (g-i) shows the finest geometric level, where individual atomic modules such as panels, doors, windows, and roof elements burn and disappear independently. The fourth row (j-l) shows a temperature slice through the building over the same sequence, illustrating how heat develops inside the structure and drives fuel depletion across the hierarchy.

We then advance the Eulerian state for each grid cell (lines 19-33). We advect $\boldsymbol{u}$, $T_g$, and $Y_i$ using a semi-Lagrangian scheme (line 20). Buoyancy is applied as a body force to the velocity field based on the density (line 21), followed by vorticity confinement to restore small-scale rotational motion damped by semi-Lagrangian advection (line 22). Species mass fractions are diffused (line 23) using an explicit discretization of the Laplacian based on local temperature differences.

Chemical and thermochemical source terms are then evaluated. All of the solid steps are based on the available energy and their respective reaction rates, modifying the thermal energy of the solid and injecting gaseous mass fractions $Y_i$. Moisture is evaporated from the solids first (line 24). Afterwards, solid combustion is computed (line 25), exchanging oxygen with residual and carbon dioxide, while combusting solid mass. Next, solid pyrolysis (line 26) produces gaseous fuel and residual species. Finally, gas combustion is computed (line 27) by evaluating the reaction rate $v_{c,g}$ and updating species mass fractions $Y_i$ and the associated heat release in the gas.

After chemistry, we update occupancy and cell flags from the current composition fields (line 28). We update the solid per-cell thermophysical properties (line 29), including density $\rho$, heat capacity $C_p$, and conductivity $k$, as functions of local material composition. This step recomputes solid versus gas status and updates interface masks used by boundary exchange, based on remaining solid mass and material-specific burnout rules. Temperature is conducted through the updated cells and at the solid-gas interface (line 30). The gas density is then updated based on the temperature according to the equation of state. Boundary conditions are enforced (line 32). Finally, we enforce a divergence free velocity field by solving a pressure projection and subtracting the pressure gradient from the velocity field using a red-black Gauss Seidel (line 33). At the end of each frame, we remove particles whose lifetime has expired (line 34). We additionally cull flame-front particles that are no longer consistent with the active burning region (line 35), for example if they reside in cells that did not ignite and no longer satisfy the flame-front criterion.

<!-- page 246:13 -->

> Figure 12: Effect of wind and biomass on urban wildfire progression. Under high wind an urban wildfire reaches a cluster of trees (a) and produces a large number of embers that are transported into the urban scene (b) and cause the wildfire to progress (c). Increased amounts of biomass results in a more vigorous fire (d) with stronger flames and increased heat release (e), which consequently leads to an intense urban wildfire (f).

### 5.2 Visualization

We visualize the simulated combustion dynamics using a Monte-Carlo path tracer to capture the complex light transport associated with fire, smoke, and hot gases [Pharr et al. 2016]. The path tracer samples multiple time-varying 3D scalar fields, including fuel, oxygen, vapor, temperature, and residual reaction history, enabling the visualization of tightly coupled solid, liquid, and gaseous phenomena. Volume rendering is done with an importance sampled integration after every bounce using transmittance and free-flight estimates [Peters 2025], which we extend with an extra integration over the emission field. Denoising of the accumulated radiance is done using statistical error reduction for Monte Carlo renderings [Sakai et al. 2025]. Embers are rendered by training tracing rays from the particle to the camera position to then splat them onto the image. Their radiance is derived from their temperature and then further attenuated or fully blocked by volumetric effects or occluding solids. Most figures shown in the paper are rendered with 128 samples per pixels. Flame appearance is modeled based on the planckian blackbody radiation and the chemiluminescence scheme from Wrede et al. [2025].

## 6 Results and Validation

In this section, we evaluate the behavior and performance of our urban wildfire simulation framework through a series of qualitative and quantitative experiments, and validate its ability to reproduce key fire spread phenomena.

> Figure 13: The same two scenes as in Figure 12 are shown as temperature (inferno color map), wood (brown color) and leaves (green color) from a top down view. The higher wind scene develops a tongue-like fireline shape, whereas the low-wind but high-biomass scene has a more diffuse development and higher temperatures.

Our solver captures the coupled dynamics of gas-phase combustion, solid fuel degradation, and heat transfer, which allows us to model complex WUI scenarios (Figs. 1, 7, 12). In Fig. 11, we illustrate the visual difference of how a fire progressively consumes available fuel, resulting in increasing structural damage of a building at the coarsest (top), intermediate (middle) and highest (bottom) level of our fuel graph representation. The combustion simulation is carried out at the lowest level of detail – the resulting temperature is visualized as a slice of the voxel space (Fig. 11, j-l).

In Fig. 8 we compare two wildfire scenarios that illustrate the influence of vegetation clearance on structure ignition. In the dense vegetation case, continuous fuel connectivity allows the fire to propagate directly to the building, ultimately resulting in ignition. In contrast, introducing a vegetation-free margin around the structure disrupts fuel continuity and prevents fire spread to the building, highlighting the effectiveness of defensible space in mitigating structure ignition in WUI fires.

<!-- page 246:14 -->

Fig. 10 shows the temporal progression of urban wildfire simulations with and without ember transport under strong wind conditions. When ember transport is enabled, wind-driven embers are advected ahead of the primary flame front, triggering secondary ignitions and substantially accelerating fire spread. In contrast, disabling embers confines combustion to regions directly connected by continuous fuel, resulting in more localized fire growth. Consequently, the road acts as a natural fire break.

Figures 12 and 13 illustrate how environmental conditions and fuel availability affect WUI fire development in a procedurally generated urban scene. Under strong wind, the fire is rapidly advected from surrounding vegetation toward the neighborhood. Once the flame front reaches a cluster of trees, burning vegetation produces many embers that are transported ahead of the active fire line, where they ignite secondary spot fires and accelerate spread into the urban area (Fig. 12a–c). Increasing the available biomass increases flame intensity, heat release, and smoke production, resulting in a more vigorous and persistent fire (Fig. 12d–f). In Fig. 13 we show the same scenarios as top-down renderings of temperature and remaining biomass. In the high-wind case, the fire line develops a tongue-like shape aligned with the wind direction, indicating rapid directional spread and ember-driven ignitions ahead of the main fire line. In the lower-wind, high-biomass case, spread is less directional but produces broader regions of elevated temperature, as the denser fuel sustains combustion for longer and releases more heat locally. Together, these examples demonstrate that our framework can capture qualitatively different WUI fire behaviors caused by wind-driven ember transport and biomass-driven fire intensity.

Finally, a temporal sequence of a complex urban wildfire is shown in Fig. 7. The fire initiates in a small cluster of structures (a–c), then intensifies and expands through structure-to-structure ignition (d–e), and eventually propagates across multiple residential blocks (f–h). The simulation captures the combined effects of radiative preheating, material-dependent combustion, direct flame contact, and ember-driven spot ignitions in a dense urban layout.

### 6.1 Ablation of Fire-spread Pathways

We evaluate the contribution of the three dominant WUI exposure pathways – direct flame contact, thermal radiation, and ember transport – in a controlled vegetation-to-structure ignition scenario (Fig. 9). In this setup we place a burning tree near a combustible building and vary which spread mechanisms are active. With only direct flame contact enabled, ignition occurs only after the flame physically reaches the building surface, resulting in localized and comparatively delayed structural burning (a). Enabling radiation in addition to direct flame contact increases preheating of the facade and roof, producing earlier ignition and a more sustained structure fire (b). When both direct flame contact and radiation are disabled, the fire remains confined to the vegetation and does not bridge the separation gap (c). In contrast, radiation alone can ignite the structure without physical flame contact, demonstrating its role as a non-contact heating mechanism (d). Ember transport introduces a different nonlocal pathway by producing spot ignitions on the roof, even away from the active flame front (e). When all three mechanisms are enabled, their combined effects lead to the strongest and most persistent structure fire, illustrating that WUI fire spread emerges from the interaction of multiple exposure pathways rather than from a single mechanism (f).

> Figure 14: Radiative heat-flux validation using NISTIR 6546 Sample Problem 1. We simulated 12 m a pool-fire with target probes placed at the NIST acceptable-separation distances (6m, 10m, 50m, 55m) for combustible structures and people. The measured radiative heat flux over time at representative failure and safety distances. For combustible structures, the flux exceeds the NIST threshold of 31.5 kW/m² at 6 m but remains below it at 10 m. For people, the flux exceeds the exposure threshold of 1.4 kW/m² at 50 m but remains below it at 55 m. These results reproduce the expected separation-distance behavior of the NIST benchmark.
>
> (Plot annotations, transcribed for reference: left panel "Combustible Structures Threshold", y axis "Radiative Heat Flux (kW/m²)" 0 to 40, x axis "Simulation Time (s)" 0 to 30, legend "Building - failure (6 m)", "Building - failure (10 m)", "NIST threshold = 31.5 kW/m²"; right panel "People Exposure Threshold", y axis "Radiative Heat Flux (kW/m²)" 0.2 to 1.6, x axis "Simulation Time (s)" 0 to 30, legend "People - failure (50 m)", "People - safety (55 m)", "NIST threshold = 1.4 kW/m²".)

### 6.2 Comparisons to Real-World Fire Experiments

We assess the physical plausibility of our combustion solver through three real-world comparisons. First, the NISTIR 6546 pool-fire setup evaluates whether our radiation model reproduces threshold-based separation distances for people and combustible structures. Second, the NIST dry Douglas-fir tree experiment evaluates vegetation combustion through heat-release behavior and radiative exposure. Third, the wind-driven building-to-building fire-spread experiment evaluates coupled structural burning, heat transfer, and damage under realistic exterior fire exposure. Parameter values for the experiments are provided in the supplementary material.

*Radiative heat flux validation.* We replicate Sample Problem 1 from NISTIR 6546, which considers a gasoline storage tank located in a circular dike with a diameter of 12 m. The NIST procedure asks whether a pool fire would expose nearby buildings or people to thermal radiation above the acceptable-separation-distance thresholds of 31.5 kW/m² for buildings and 1.4 kW/m² for people. For gasoline, NIST uses a heat release rate per unit area of 2400 kW/m² and reports acceptable separation distances of less than 10 m for buildings and 55 m for people, measured from the edge of the dike [McGrattan et al. 2000]. We simulate the same 12 m pool-fire scenario and measure radiative heat flux at representative target distances. As shown in Fig. 14, the simulated flux exceeds the building threshold at 6 m but remains below it at 10 m, while for people it exceeds the exposure threshold at 50 m and remains below it at 55 m. These results reproduce the qualitative NIST separation-distance behavior and indicate that our radiation model captures the expected near-field hazard levels for this benchmark.

<!-- page 246:15 -->

> Figure 15: Temporal progression of a simulated dry fir tree inspired by the NIST [Hoehler et al. 2020]. The sequence illustrates the combustion of a fir tree from ignition (a), flame growth (b, c), and peak heat release behavior (d) to the full combustion (e, f). Our simulation reproduces the characteristic fire growth dynamics and heat release trends of dry vegetation under controlled ignition conditions.

*Vegetation combustion behavior.* To test the combustion behavior of vegetation we reproduced an experiment from the NIST Fire Calorimetry Database [Bryant and Bundy 2019; Hoehler et al. 2020; NIST 2026]. In this benchmark, a dried Douglas-fir Christmas tree is ignited in a small room, with measured heat release behavior recorded over time. We observe heat release behavior in our simulations that is qualitatively and quantitatively consistent with the NIST experiments as shown in Fig. 15. In particular, the graph shown in Fig.16 (left) shows a similar trend and comparable values for the heat release rate over the whole duration of the burning tree. While our model reproduces the overall fire growth characteristics observed in the experiments, we do not attempt to replicate the full experimental environment in detail. Specifically, the surrounding furnishings present in the NIST living-room setup are not explicitly modeled, as our focus is on validating material-dependent combustion behavior and heat release dynamics rather than room-scale fire evolution. The ignition starts rather slow, with a sharp increase towards the peak at 30 s, before slowly declining towards a stable smoldering at 50 s. We stop the experiment before the surrounding furniture catches on fire at around 65 seconds. This experiment shows that our simulation is capable of reproducing the combustion behavior of dry vegetation.

> Figure 16: Left: Comparison of the heat release rate over time for a dry pine tree fire measured in the NIST experiment (black dashed line) and our simulated scenario (red line). Both curves exhibit very similar temporal fire behavior characteristics, including rapid ignition, a pronounced peak heat release phase, and subsequent decay, indicating qualitative and quantitative agreement in combustion dynamics. Right: wind-driven structure exposure results (Fig. 17). Our sensor measurements are similar to those reported in Hedayati et al. [2026] (p. 28, Tab. 2, Test 13). This setup evaluates the coupling between solid combustion, wind-driven gas transport, radiative heat transfer, and structural exposure.
>
> (Plot annotations, transcribed for reference: left panel "Dry Pine Tree", y axis "Heat Release Rate (kW)" with ticks 2000 to 8000, x axis "Time (s)" 0 to 60, legend "NIST", "Ours"; right panel "Radiative Heat Flux Time History", y axis "Heat Flux (kW/m²)" 0 to 60, x axis "Duration (min)" 0 to 6.)

*Wind-driven structure exposure.* To validate our solid combustion model we replicated Test 13 from the wind-driven building-to-building fire-spread experiments of Hedayati et al. [2026]. The original experiment used a wood shed as the source of fire and a target building placed 6.1 m (20 ft) downwind at a 45° orientation, under a mean wind speed of 43.9 ± 3.3 km/h (27.3 ± 2.1 mph). The target included combustible open eaves and tempered windows, and the reported outcome was envelope damage rather than full structure ignition or destruction. The measured peak heat flux was approximately 33 kW/m² at about 23 min; vinyl glazing detached at 11:41, and one tempered window pane failed at 23:05, while the frame and full window assembly did not fail. We reproduce this configuration in simulation to evaluate whether our coupled solid-combustion, radiation, and gas-transport model produces comparable wind-driven exposure and damage progression. The original experiment burned roughly 21 t of wood within a 40 min time frame. We recreated a scaled down version, but ensured the same mass loss rate to keep the radiative heat flux comparable. As in the real experiment, our simulation captures an initial build-up phase, followed by a gradual increase in heat flux at the target surface, localized envelope damage under sustained exposure, and no full building ignition. The sensor measurements of our simulation are shown in Fig. 16, the experiment setup is shown in Fig. 17.

> Figure 17: Wind-driven structure-to-structure fire-spread validation: We simulate a wood source structure with a downwind target structure and heat-flux sensors placed on the target facade. We then measure the radiative heat-flux at multiple target sensors over time. The simulated fire plume is driven toward the target building, producing sustained radiative exposure and localized envelope heating without immediate full target ignition. The measured sensor values are reported in Fig. 16, right.

*Runtime performance.* In Tab. 1 we show the results of a performance analysis of our hierarchical fuel representation across a range of scene configurations with varying geometric complexity and fuel distributions. By organizing buildings and vegetation into a multi-level hierarchy, the solver efficiently aggregates material properties and combustion state at different semantic resolutions, reducing the cost of per-module queries in large scenes.

<!-- page 246:16 -->

**Table 1.** Performance timings are evaluated at the three LOD Levels (Fine, Medium, Coarse) shown in Fig. 4 applied to the whole scene and a mixture of LODs based on distance to the camera (adaptive). Mesh to Voxel queries are called *Update* (one per frame) in the table. *Render* is the time taken to render one image at 128 samples per pixel and *Sim Step* is a single simulation advancement. Fig. 7 could not be rendered at the medium LOD due to memory constraints (OOM) while the fine LOD *did not finish* (DNF) scene construction at all within 30 minutes. Fig. 12 similarly did not finish constructing at the finest LOD. For Fig. 7, the adaptive LOD uses only the coarse LODs due to the large distance from the camera to the scene.

| Figure | LOD | Update (ms) | Render (ms) | Sim Step (ms) | Grid Dimensions | GPU |
|---|---|---|---|---|---|---|
| Fig. 1 | Fine | 1,118 | 34,371 | 165 | 600 x 375 x 150 | RTX Pro 6000 |
| | Medium | 302 | 24,115 | | | |
| | Coarse | 249 | 6,211 | | | |
| | Adaptive | 280 | 20,765 | | | |
| Fig. 7 | Fine | DNF | DNF | 17 | 250 x 250 x 60 | RTX 4090 |
| | Medium | 11,654 | OOM | | | |
| | Coarse | 98 | 3,317 | | | |
| | Adaptive | 98 | 3,317 | | | |
| Fig. 10 | Fine | 3,077 | 16,019 | 53 | 240 x 240 x 100 | RTX 4090 |
| | Medium | 108 | 5,225 | | | |
| | Coarse | 78 | 2,614 | | | |
| | Adaptive | 419 | 7,688 | | | |
| Fig. 11 | Fine | 120 | 4,334 | 277 | 510 x 510 x 225 | RTX Pro 6000 |
| | Medium | 164 | 4,752 | | | |
| | Coarse | 62 | 3,008 | | | |
| | Adaptive | 104 | 4.180 | | | |
| Fig. 12 | Fine | DNF | DNF | 196 | 400 x 400 x 200 | RTX Pro 6000 |
| | Medium | 522 | 19,624 | | | |
| | Coarse | 394 | 16,116 | | | |
| | Adaptive | 412 | 16,097 | | | |

(In the original, the Sim Step, Grid Dimensions and GPU cells span the four LOD rows of each figure.)

## 7 Discussion and Limitations

Our framework demonstrates that a physically based fire simulation can be effectively combined with procedurally generated, semantically rich urban environments to model WUI fires at street scale. A key strength of our approach is the hierarchical fuel representation, which enables efficient simulation of large and complex scenes while retaining fine-grained detail. The fuel graph provides a way to map fuel depletion back to visible mesh updates and thereby enables the interactive simulation of large-scale scenarios. Coupled with procedural modeling from OSM data, this allows rapid construction of plausible urban environments that preserve structural semantics relevant to fire behavior, such as roof composition, facade materials, the interior fuel load distribution and vegetation proximity. Our method enables unprecedented visual analysis, albeit with a bias towards graphics rather than engineering-grade physics.

Despite these strengths, our system has several important limitations. First, while the combustion and heat transfer models are physically motivated, they necessarily rely on simplified kinetics and material parameters. Real-world building materials exhibit highly variable ignition behavior, degradation, and structural failure modes that are not fully captured by our current solid combustion model. In particular, effects such as structural collapse, glass breakage, and complex multilayer wall assemblies are approximated or omitted, which can influence airflow and fire spread in real fires. Second, although our solver incorporates buoyancy-driven flow, we currently do not consider a detailed atmospheric coupling such as cloud formation or rain. Third, the procedural generation of urban environments, while flexible and data-driven, produces approximations of real buildings rather than exact reconstructions. Architectural details and construction quality is approximated, which limits the fidelity of site-specific analysis. Similarly, our vegetation models abstract biological variability and moisture content, both of which play a significant role in real wildfire behavior. Additionally, while a low destruction threshold value prevents visually detached flames, it can preserve mostly burned structures longer than necessary, leading to visually delayed building degradation.

From a computational perspective, achieving interactive performance requires carefully balancing between resolution, physical accuracy, and visual fidelity. While the hierarchical fuel graph enables scalable rendering, extreme scenarios with very dense urban scenes or large numbers of embers can still become computationally demanding. The simulation runs at interactive rates, while path-traced rendering remains the primary bottleneck. A faster rendering approach could enable an interactive end-to-end pipeline. Further optimization, adaptive meshing, and multi-GPU acceleration are promising directions to improve scalability.

## 8 Conclusion

We have presented a physically based simulation framework for WUI fires that couples fluid dynamics, combustion, and heat transfer with procedurally generated, material-aware urban environments. By proposing a novel hybrid Eulerian–Lagrangian fire solver with radiative heat transfer, heterogeneous heat conduction, solid fuel combustion, and ember transport, our approach captures key WUI fire phenomena at street scale, including structure-to-structure ignition, ember-driven spot fires, and the influence of vegetation layout and urban geometry on fire behavior. A central contribution of this work is the hierarchical fuel representation, which organizes buildings and vegetation from atomic modules into higher-level semantic components, enabling scalable simulation and efficient coupling between fire dynamics and complex urban scenes. We see several avenues for future work, including a more careful integration with atmospheric boundary conditions, improved material calibration against empirical fire data, and adaptive refinement strategies for extreme-scale simulations. To increase the realism even further, one could model interior fuel based on generated layouts, add support for different building codes or model vegetation placement according to remote sensing data. Our framework provides a visually detailed and interactive means of exploring how wildfires spread from vegetation into urban environments, potentially supporting education, risk assessment, and the study of mitigation and resilience strategies. Given its current approximations of materials, structures, vegetation, and atmospheric conditions, it is intended for physically informed visual analysis rather than safety-critical or site-specific prediction. We believe our approach opens new opportunities for the graphics-driven investigation of WUI fires and for developing tools that support urban planning, resilience research, and the development of simulators for autonomous agents.

## Acknowledgments

We thank the reviewers for their valuable comments and suggestions. This work is supported by ERC grant #101170158 - WildfireTwins.

<!-- page 246:17 -->

## References

- R. K. Agarwal. 1985. On the use of the arrhenius equation to describe cellulose and wood pyrolysis. *Thermochimica Acta* 91 (1985), 343–349. doi:10.1016/0040-6031(85)85227-8
- G. Aguilera and J. Johansson. 2019. Avengers: Endgame, a new approach for combustion simulations. In *ACM SIGGRAPH 2019 Talks (SIGGRAPH '19)*. ACM, Article 39.
- A. G. M. Ahmed, M. Pharr, V. Ostromoukhov, and H. Huang. 2025. SZ Sequences: Binary-Based (0, 2 q)-Sequences. *ACM Transactions on Graphics (TOG)* 44, 6 (2025), 1–14.
- O. Aichholzer and F. Aurenhammer. 1996. Straight Skeletons for General Polygonal Figures in the Plane. In *Proceedings of the Second Annual International Conference on Computing and Combinatorics (COCOON '96)*. Springer-Verlag, 117–126.
- S. Alaka and R. Bidarra. 2023. Hierarchical Semantic Wave Function Collapse. In *Proceedings of the 18th International Conference on the Foundations of Digital Games* (Lisbon, Portugal) *(FDG '23)*. Association for Computing Machinery, New York, NY, USA, Article 68, 10 pages. doi:10.1145/3582437.3587209
- A. Bakhshaii and E.A. Johnson. 2019. A review of a new generation of wildfire–atmosphere modeling. *Canadian Journal of Forest Research* 49, 6 (2019), 565–574.
- M. Bati, S. Blanco, C. Coustet, V. Eymet, V. Forest, R. Fournier, J. Gautrais, N. Mellado, M. Paulin, and B. Piaud. 2023. Coupling conduction, convection and radiative transfer in a single path-space: Application to infrared rendering. *ACM Transactions on Graphics (TOG)* 42, 4 (2023), 1–20.
- B Braun, J Bender, and N Thuerey. 2025. Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation. *ACM Trans. on Graph, (TOG)* 44, 4 (2025), 1–23.
- R. Bridson. 2015. *Fluid simulation for computer graphics*. AK Peters/CRC Press.
- R. Bridson and M. Müller-Fischer. 2007. Fluid simulation. In *ACM SIGGRAPH 2007 Courses*. ACM, 1–81.
- R. A. Bryant and M. F. Bundy. 2019. *The NIST 20 MW Calorimetry Measurement System for Large-Fire Research*. Technical Note (NIST TN) 2077. National Institute of Standards and Technology, Gaithersburg, MD, USA. doi:10.6028/NIST.TN.2077
- CALFIRE. 2025. California Department of Forestry and Fire Protection - 2025 Fire Season Incident Archive. https://www.fire.ca.gov/incidents/2025. (2025). Incident information and seasonal wildfire outlook including the Eaton and related fires.
- S.-W. Cheng and A. Vigneron. 2007. Motorcycle Graphs and Straight Skeletons. *Algorithmica* 47, 2 (Feb. 2007), 159–182.
- N. Chiba, K. Muraoka, H. Takahashi, and M. Miura. 1994. Two-dimensional visual simulation of flames, smoke and the spread of fire. *The Journal of Visualization and Computer Animation* 5, 1 (1994), 37–53.
- S. Clavet, P. Beaudoin, and P. Poulin. 2005. Particle-based viscoelastic fluid simulation. In *ACM SIGGRAPH/Eurographics SCA*. 219–228.
- J. L. Coen. 2013. *Modeling Wildland Fires :.* National Center for Atmospheric Research (NCAR),, Boulder, CO :. 2013-02-04.
- J. L. Coen, M. Cameron, J. Michalakes, E. G. Patton, P. J. Riggan, and K. M. Yedinak. 2013. WRF-Fire: Coupled Weather–Wildland Fire Modeling with the Weather Research and Forecasting Model. *Journal of Applied Meteorology and Climatology* 52, 1 (2013), 16 – 38.
- C. X. Cunningham, John T. Abatzoglou, Todd M. Ellis, Grant J. Williamson, and David M. J. S. Bowman. 2025. Wildfires will intensify in the wildland-urban interface under near-term warming. *Communications Earth & Environment* 6, 1 (09 Jul 2025), 542. doi:10.1038/s43247-025-02475-y
- I. Džolev, M. Laban, and S. Draganić. 2021. Survey based fire load assessment and impact analysis of fire load increment on fire development in contemporary dwellings. *Safety Science* 135 (2021), 105094. doi:10.1016/j.ssci.2020.105094
- Engineering ToolBox. 2026. Material Properties. https://www.engineeringtoolbox.com/material-properties-t_24.html. Accessed 2026.
- R. Fedkiw, J. Stam, and H. W. Jensen. 2001. Visual simulation of smoke. In *Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '01)*. Association for Computing Machinery, New York, NY, USA, 15–22.
- Y. Feng, X. Feng, Y. Shang, Y. Jiang, C. Yu, Z. Zong, T. Shao, H. Wu, K. Zhou, C. Jiang, and Y. Yang. 2025. Gaussian splashing: Unified particles for versatile motion synthesis and rendering. In *Proceedings of the computer vision and pattern recognition conference*. 518–529.
- FireRescue. 2026. By the Numbers: 1 year after the Palisades and Eaton fires devastated Calif. https://www.firerescue1.com/wildfire-and-wildland-urban-interface/by-the-numbers-1-year-after-the-palisades-and-eaton-fires-devastated-calif. (2026). Statistical overview of the 2025 Los Angeles wildfires.
- T. Hädrich, D. T. Banuti, W. Pałubicki, S. Pirk, and D. L. Michels. 2021. Fire in Paradise: Mesoscale Simulation of Wildfires. *ACM Trans. Graph.* 40, 4, Article 163 (July 2021), 15 pages.
- F. Hedayati, D. Gorham, X. Monroy, M. Marandi, E. Sluder, M. Gollner, W. Cui, and A. Merhi. 2026. Wind-Driven Building-to-Building Fire Spread: Experimental Results and Probabilistic Modeling. *Fire Technology* 62, 2 (28 Jan 2026), 32. doi:10.1007/s10694-025-01854-3
- M. S. Hoehler, M. F. Bundy, L. A. DeLauter, R. L. Materese, L. Gerskovic, and J. R. Garcia. 2020. *Fire Hazards of Dry Versus Watered Christmas Trees*. Technical Note (NIST TN) 2131. National Institute of Standards and Technology, Gaithersburg, MD, USA. doi:10.6028/NIST.TN.2131
- J.-M. Hong, T. Shinar, and R. Fedkiw. 2007. Wrinkled flames and cellular patterns. *ACM Trans. Graph.* 26, 3 (July 2007), 47–es.
- Y. Hong, D. Zhu, X. Qiu, and Z. Wang. 2010. Geometry-based control of fire simulation. *The Visual Computer* 26, 9 (01 Sep 2010), 1217–1228.
- C. Horvath and W. Geiger. 2009. Directable, High-resolution Simulation of Fire on the GPU. *ACM Trans. Graph.* 28, 3, Article 41 (2009), 8 pages.
- IES. 2026. Table 6: Thermal Conductivity, Specific Heat Capacity and Density. https://help.iesve.com/ve2021/table_6_thermal_conductivity__specific_heat_capacity_and_density.htm. Accessed 2026.
- International Code Council, Inc. 2020. *Chapter 6: Framing — Studs and Plates*. International Code Council, Country Club Hills, IL, USA, Chapter 6. https://codes.iccsafe.org/content/MNIFGMNRC2020P1 [Retrieved 05.08.2026].
- H. Johra. 2019. *Thermal properties of common building materials*. Technical Report 216. Department of Civil Engineering, Aalborg University, Aalborg, Denmark. Accessed 2026.
- H. Johra. 2021. *Thermal properties of building materials – Review and database*. Technical Report 289. Department of the Built Environment, Aalborg University, Aalborg, Denmark. doi:10.54337/aau456230861 Accessed 2026.
- I. Karth and A. M. Smith. 2017. WaveFunctionCollapse is constraint solving in the wild. In *Proceedings of the 12th International Conference on the Foundations of Digital Games* (Hyannis, Massachusetts) *(FDG '17)*. Association for Computing Machinery, New York, NY, USA, Article 68, 10 pages. doi:10.1145/3102071.3110566
- J. Katan and L. Perez. 2021. ABWiSE v1.0: toward an agent-based approach to simulating wildfire spread. *Natural Hazards and Earth System Sciences* 21, 10 (2021), 3141–3160.
- T. Kim, E. Hong, J. Im, D. Yang, Y. Kim, and C.-H. Kim. 2017. Visual simulation of fire-flakes synchronized with flame. *The Visual Computer* 33 (06 2017).
- T. Kim, N. Thürey, D. James, and M. Gross. 2008. Wavelet Turbulence for Fluid Simulation. *ACM Trans. Graph.* 27, 3, Article 50 (2008), 6 pages.
- A. Kokosza, H. Wrede, D. Gonzalez E., M. Makowski, D. Liu, D. L. Michels, S. Pirk, and W. Palubicki. 2024. Scintilla: Simulating Combustible Vegetation for Wildfires. *ACM Trans. Graph.* 43, 4, Article 70 (July 2024), 21 pages.
- K. A. Kroos and M. C. Potter. 2014. *Thermodynamics for Engineers*. Cengage Learning.
- A. Lamorlette and N. Foster. 2002. Structural modeling of flames for a production environment. In *Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '02)*. ACM, New York, NY, USA, 729–735.
- C. Lapointe, N. Wimer, J. Glusman, A. Makowiecki, J. Daily, G. Rieker, and P. Hamlington. 2020. Efficient simulation of turbulent diffusion flames in OpenFOAM using adaptive mesh refinement. *Fire Safety Journal* 111 (01 2020), 102934.
- S. Ling, A. Madan, N. Sharp, and A. Jacobson. 2025. Uniform Sampling of Surfaces by Casting Rays. *Computer Graphics Forum* 44, 5 (2025), e70202.
- R.R. Linn, S.L. Goodrick, S. Brambilla, M.J. Brown, R.S. Middleton, J.J. O'Brien, and J.K. Hiers. 2020. QUIC-fire: A fast-running simulation tool for prescribed fire planning. *Environmental Modelling & Software* 125 (2020), 104616.
- R. Linn, J. Reisner, J. Colman, and J. Winterkamp. 2002. Studying wildfire behavior using FIRETEC. *International Journal of Wildland Fire* 11 (11 2002), 233–246.
- D. Liu, J. Klein, F. Rist, W. Pałubicki, S. Pirk, and D. Michels. 2025. FLAMEFORGE: Combustion Simulation of Wooden Structures. *Proc. ACM Comput. Graph. Interact. Tech.* 8, 4, Article 50 (Aug. 2025), 19 pages. doi:10.1145/3747855
- S. Liu, T. An, Z. Gong, and I. Hagiwara. 2012. *Physically based simulation of solid objects' burning*. Springer-Verlag, Berlin, Heidelberg, 110–120.
- S. Liu, Q. Liu, T. An, J. Sun, and Q. Peng. 2009. Physically based simulation of thin-shell objects' burning. *Vis. Comput.* 25, 5–7 (April 2009), 687–696.
- F. Losasso, G. Irving, E. Guendelman, and R. Fedkiw. 2006a. Melting and burning solids into liquids and gases. *IEEE Transactions on Visualization and Computer Graphics* 12, 3 (2006), 343–352.
- F. Losasso, T. Shinar, A. Selle, and R. Fedkiw. 2006b. Multiple interacting liquids. *ACM Trans. Graph.* 25, 3 (July 2006), 812–819.
- K. McGrattan, S. Hostikka, J. Floyd, R. McDermott, M. Vanella, E. Mueller, and C. Paul. 2025a. Fire Dynamics Simulator Technical Reference Guide Volume 1: Mathematical Model. doi:10.6028/NIST.SP.1018 [Version 6.10.1].
- K. McGrattan, S. Hostikka, J. Floyd, R. McDermott, M. Vanella, E. Mueller, and C. Paul. 2025b. Fire Dynamics Simulator, User's Guide. doi:10.6028/NIST.SP.1019 [Version 6.10.1].
- K. B. McGrattan, H. R. Baum, and A. Hamins. 2000. *Thermal Radiation from Large Pool Fires*. Technical Report NISTIR 6546. National Institute of Standards and Technology.
- A. D. McNaught and A. Wilkinson. 1997. *IUPAC Gold Book: Compendium of Chemical Terminology*. Blackwell Scientific Publications. https://goldbook.iupac.org
- Z. Melek and J. Keyser. 2002. Interactive simulation of fire. *Pacific Graphics* (2002), 431–432.
- W. Mell, M. Jenkins, J. Gould, and P. Cheney. 2007. A Physics Based Approach to Modeling Grassland Fires. *International Journal of Wildland Fire* (2007).
- B. Merci and T. Beji. 2016. Modeling and simulation of transport phenomena in fire engineering. *Fire Safety Journal* 80 (2016), 12–22.
- V. Mihalef, B. Unlusu, D. Metaxas, M. Sussman, and M. Hussaini. 2006. Physics based boiling simulation. 317–324.

<!-- page 246:18 -->

- P. Müller, P. Wonka, S. Haegler, A. Ulmer, and L. Van Gool. 2006. Procedural modeling of buildings. In *ACM SIGGRAPH 2006 Papers* (Boston, Massachusetts) *(SIGGRAPH '06)*. ACM, 614–623. doi:10.1145/1179352.1141931
- J. Nazarenus, D. Michels, W. Palubicki, S. Kou, F.-L. Zhang, S. Pirk, and R. Koch. 2026. Gaussians on Fire: High-Frequency Reconstruction of Flames. In *European Conference on Computer Vision (ECCV)*.
- D. Quang Nguyen, R. Fedkiw, and H. W. Jensen. 2002. Physically Based Modeling and Animation of Fire. *ACM Trans. Graph.* 21, 3 (2002), 721–728.
- M. B. Nielsen, M. Bojsen-Hansen, K. Stamatelos, and R. Bridson. 2022. Physics-based combustion simulation. *ACM Transactions on Graphics (TOG)* 41, 5 (2022), 1–21.
- M. B. Nielsen, K. Stamatelos, M. Bojsen-Hansen, and R. Bridson. 2019. Physics-based combustion simulation in bifrost. In *ACM SIGGRAPH 2019 Talks*. 1–2.
- T. Niese, S. Pirk, M. Albrecht, B. Benes, and O. Deussen. 2022. Procedural Urban Forestry. *ACM Trans. Graph.* 41, 2, Article 20 (March 2022), 18 pages. doi:10.1145/3502220
- NIST. 2026. *NIST Fire Calorimetry Database (FCD)*. doi:10.18434/mds2-2314
- OpenStreetMap contributors. 2017. Planet dump retrieved from https://planet.osm.org . https://www.openstreetmap.org.
- Z. Pan and D. Manocha. 2017. Efficient Solver for Spacetime Control of Smoke. *ACM Trans. Graph.* 36, 5, Article 162 (July 2017), 13 pages.
- Y. I. H. Parish and P. Müller. 2001. Procedural modeling of cities. In *Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '01)*. Association for Computing Machinery, New York, NY, USA, 301–308.
- V. Pegoraro and S. G. Parker. 2006. Physically-based Realistic Fire Rendering. *EG Nat. Phenom.* (2006), 51–59.
- C. Peters. 2025. Jackknife Transmittance and MIS Weight Estimation. *ACM Trans. Graph.* 44, 6, Article 204 (Dec. 2025), 16 pages. doi:10.1145/3763273
- N. Peters. 2000. *Turbulent Combustion*. Cambridge University Press.
- M. Pharr, W. Jakob, and G. Humphreys. 2016. *Physically Based Rendering: From Theory to Implementation* (3rd ed.). Morgan Kaufmann Publishers Inc.
- S. Pirk, M. Jarząbek, T. Hädrich, D. L. Michels, and W. Palubicki. 2017. Interactive wood combustion for botanical tree models. *ACM Trans. Graph. (TOG)* 36, 6 (2017), 1–12.
- D. M. J. Purnomo, Y. Qin, M. Theodori, M. Zamanialaei, C. Lautenberger, A. Trouvé, and M. J. Gollner. 2024. Integrating an urban fire model into an operational wildland fire model to simulate one dimensional wildland–urban interface fires: a parametric study. *International Journal of Wildland Fire* 33, 10 (10 2024), WF24102. arXiv:https://connectsci.au/wf/article-pdf/doi/10.1071/WF24102/154668/wf24102.pdf doi:10.1071/WF24102
- J. Rath, M.G. Wolfinger, G. Steiner, G. Krammer, F. Barontini, and V. Cozzani. 2003. Heat of wood pyrolysis. *Fuel* 82, 1 (2003), 81–91. doi:10.1016/S0016-2361(02)00138-2
- H. Sakai, C. Freude, M. Wimmer, and D. Hahn. 2025. Statistical Error Reduction for Monte Carlo Rendering. In *Proceedings of the SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers '25)*. ACM, New York, NY, USA, Article 50, 12 pages. doi:10.1145/3757377.3763995
- K. Schmidt-Rohr. 2011. The heat of combustion—getting it right. *Journal of Chemical Education* 88, 8 (2011), 1099–1104.
- Q. Shen, N. Tao, Q. Dai, T. Chen, M. Qin, Y. Zhang, M. Chu, W. Chen, and B. Chen. 2026. FieryGS: In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting. In *The Fourteenth International Conference on Learning Representations*. https://openreview.net/forum?id=ziKFH7whvy
- R. M. Smelik, T. Tutenel, R. Bidarra, and B. Benes. 2014. A Survey on Procedural Modelling for Virtual Worlds. *Computer Graphics Forum* 33, 6 (2014), 31–50.
- V. Spyratos, Patrick S. Bourgeron, and Michael Ghil. 2007. Development at the wildland–urban interface and the mitigation of forest-fire risk. *Proceedings of the National Academy of Sciences* 104, 36 (2007), 14272–14276. doi:10.1073/pnas.0704488104
- J. Stam. 1999. Stable Fluids. *Proc. of ACM SIGGRAPH* (1999), 121–128.
- J. Stam and E. Fiume. 1995. Depicting fire and other gaseous phenomena using diffusion processes. In *Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '95)*. ACM, New York, NY, USA, 129–136.
- A. Stomakhin, C. Schroeder, C. Jiang, L. Chai, J. Teran, and A. Selle. 2014. Augmented MPM for phase-change and varied materials. *ACM Trans. Graph.* 33, 4, Article 138 (July 2014), 11 pages.
- J. Valdivia, X. Xi, A. Simeoni, and J. L. Urban. 2026. Towards modeling tree-to-tree fire spread in wildland urban-interface (WUI) fires. *Fire Safety Journal* 159 (2026), 104550.
- H. Versteeg and W. Malalasekera. 2007. *An Introduction to Computational Fluid Dynamics*. Pearson Education. https://books.google.de/books?id=dlC8MgEACAAJ
- A. R. Wagner, M. B. Rao, H. Wrede, S. Pirk, and X. Xiao. 2026. Fire as a Service: Augmenting Robot Simulators with Thermally and Visually Accurate Fire Dynamics. In *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.
- H. Wrede, A. Wagner, S. M. Mahfuz, W. Palubicki, D. Michels, and S. Pirk. 2025. Fire-X: Extinguishing Fire with Stoichiometric Heat Release. *ACM Trans. Graph.* 44, 6, Article 269 (Dec. 2025), 17 pages. doi:10.1145/3763338
- Y. Zhao, X. Wei, Z. Fan, A. Kaufman, and H. Qin. 2003. Voxels on Fire. In *Proceedings of the 14th IEEE Visualization 2003 (VIS'03) (VIS '03)*. IEEE Computer Society, USA, 36.

## A Appendix

### A.1 Solver Comparison

**Table 2.** High-level comparison of fire and combustion solvers. ✓ indicates supported functionality, ∼ partial or limited support, ✗ unsupported or not a primary focus, and ✓★ indicates functionality that is supported and forms a central contribution of our method. The table compares high-level scope and modeling support rather than individual implementation details.

| Capability | FDS [McGrattan et al. 2025b] | WFDS [Mell et al. 2007] | QUIC [Linn et al. 2020] | Paradise [Hädrich et al. 2021] | Scintilla [Kokosza et al. 2024] | Fire-X [Wrede et al. 2025] | Ours |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ***Overall scope*** | | | | | | | |
| Large outdoor fire scenes | ∼ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Urban / WUI fire scenarios | ∼ | ✓ | ∼ | ✗ | ✗ | ✗ | ✓★ |
| Interactive graphics simulation | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| ***Fire and heat modeling*** | | | | | | | |
| Gas flow and combustion | ✓ | ✓ | ∼ | ✓ | ✓ | ✓ | ✓ |
| Conduction / radiation | ✓ | ✓ | ∼ | ∼ | ∼ | ∼ | ✓ |
| Radiative ignition / preheating | ✓ | ✓ | ∼ | ∼ | ∼ | ✗ | ✓★ |
| ***Fuel types*** | | | | | | | |
| Vegetation as fuel | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Buildings as combustible solids | ✓ | ∼ | ✗ | ✗ | ✗ | ✗ | ✓★ |
| Material-dependent burning | ✓ | ∼ | ∼ | ✗ | ✗ | ✗ | ✓★ |
| ***Fire spread mechanisms*** | | | | | | | |
| Ember / firebrand transport | ∼ | ∼ | ∼ | ✗ | ✓ | ✗ | ✓ |
| Ember-driven ignition | ∼ | ∼ | ∼ | ✗ | ✓ | ✗ | ✓ |
| Explicit flame-front spreading | ✓ | ✓ | ∼ | ∼ | ∼ | ✗ | ✓★ |
| ***Scene modeling*** | | | | | | | |
| Automatic large-scene generation | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓★ |
| Urban scene generation from maps | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓★ |

### A.2 Combustion and Pyrolysis Properties

**Table 3.** Combustion values for biomass fuel. Additional parameter values can be found in Wrede et al. [2025], A3.

| Property | Default Value | Units |
|---|--:|:-:|
| Solid Pyrolysis Activation Energy | 9.22e4 | $J/mol$ |
| Solid Pyrolysis Arrhenius Factor | 4.78e4 | $1/s$ |
| Solid Heat of Pyrolysis | 2.50e05 | $J/kg$ |
| Solid Combustion Arrhenius Factor | 1.80e8 | $1/s$ |
| Solid Combustion Activation Energy | 9.22e4 | $J/mol$ |
| Solid Heat of Combustion | -2.80e6 | $J/mol$ |
| Moisture Heat of Evaporation | 4.07e4 | $J/mol$ |
| Gas Activation Energy | 1.26e5 | $J/mol$ |
| Gas Arrhenius Factor | 8.30e5 | $1/s$ |
| Gas Fuel Reaction Exponent | -0.30 | 1 |
| Gas Oxygen Reaction Exponent | 1.30 | 1 |
| Gas Heat of Combustion | -8.34e5 | $J/mol$ |

### A.3 Material Properties

**Table 4.** Thermophysical properties used in our simulation. Values are representative room-temperature references. Molar masses $M$ for composites (wood, leaves, glass, stone) are assumptions based on dominant constituents rather than material constants. We derived these values from various sources [Agarwal 1985; Engineering ToolBox 2026; IES 2026; Johra 2019; Rath et al. 2003].

| Material | $\rho$ [kg/m³] | $c_p$ [J/(kg·K)] | $k$ [W/(m·K)] | $M$ [kg/mol] |
|---|--:|--:|--:|--:|
| Steel | 7850 | 470 | 52.5 | 0.05585 |
| Wood | 600 | 2000 | 0.15 | 0.16214 |
| Leaves | 60 | 1800 | 0.06 | 0.16214 |
| Stone | 2700 | 800 | 2.5 | 0.06008 |
| Glass | 2500 | 840 | 1.0 | 0.06000 |
| Moisture | 1000 | 4184 | 0.6 | 0.01802 |

<!-- page 246:19 -->

### A.4 Model Parameters

The subscript ${}_s$ denotes solid properties. Most of the properties without any subscript are tracked separately for each phase.

| Symbol | Meaning (units) |
|---|---|
| $\boldsymbol{u}$ | Velocity field (m s⁻¹) |
| $t$ | Time (s) |
| $\rho$ | Local density (kg m⁻³) |
| $p$ | Pressure (Pa) |
| $\boldsymbol{g}$ | Gravitational acceleration vector (m s⁻²) |
| $T$ | Temperature (K) |
| $T_{\mathrm{amb}}$ | Reference ambient temperature (K) |
| $v_{p,s}$ | Rate of pyrolysis (kg m⁻³ s⁻¹) |
| $\Delta_{p,s}H^{o}$ | Heat of pyrolysis (J mol⁻¹ or J kg⁻¹) |
| $\Delta_{c}H^{o}$ | Lower heat of combustion (J mol⁻¹ or J kg⁻¹) |
| $C_p$ | Specific heat capacity at constant pressure (J kg⁻¹ K⁻¹) |
| $k$ | Thermal conductivity (gas mixture or material composition) (W m⁻¹ K⁻¹) |
| $v_c$ | Rate of combustion (mol m⁻³ s⁻¹) |
| $A_{c,s}$ | Arrhenius pre-exponential factor for the combustion (units vary based on reaction order and concentration units) |
| $E_{a,c}$ | Activation energy for the combustion (J mol⁻¹) |
| $R$ | Universal gas constant (8.314 J mol⁻¹ K⁻¹) |
| $c_f, c_o$ | Molar concentrations of fuel and oxidizer, respectively (mol m⁻³) |
| $a, b$ | Empirical reaction order exponents for fuel and oxidizer, respectively (dimensionless) |
| $D_i$ | Diffusion coefficient of species $i$ (m² s⁻¹) |
| $Y_i$ | Mass fraction of species $i$ (dimensionless) |
| $\varphi_i$ | Volume fraction of material $i$ (dimensionless) |
| $\dot{q}_c$ | Combustion heat source term (J s⁻¹) |
| $\dot{q}_{e,s}$ | Evaporation heat source term (J s⁻¹) |
| $\Delta q_{rad}$ | Volumetric radiation (gas) (J s⁻¹) |
| $\dot{m}_{e,s}$ | Evaporation mass fraction source term (s⁻¹) |
| $\dot{m}_{c,s}$ | Combustion mass fraction source term (s⁻¹) |
| $\dot{m}_{p,s}$ | Pyrolysis mass fraction source term (s⁻¹) |
| $q_i$ | Stoichiometric coefficient for species $i$ (dimensionless) |
| $M_i$ | Molar mass of species $i$ or material $i$ (kg mol⁻¹) |
| $\epsilon$ | Emissivity (dimensionless) |
| $\sigma$ | Stefan-Boltzmann constant (5.67 × 10⁻⁸ W m⁻² K⁻⁴) |
| $\kappa$ | Volumetric absorption coefficient (dimensionless) |
| $\dot{q}_p$ | Pyrolysis heat sink term (J s⁻¹) |
| $\Delta x$ | Cell size (m) |
| $h$ | Heat transfer coefficient (dimensionless) |
| $\delta_l$ | Leaf thickness parameter (m) |

(In the original this list is typeset as two aligned columns without a table header.)

### A.5 Simulation Parameters

**Table 5.** Configurable simulation parameters for our model. The origin is used to differentiate between parameters based on physical properties, numerical precision constraints and scene specific parameters.

| Parameter | Range / Value | Units | Origin |
|---|--:|:-:|--:|
| ***Combustion & Pyrolysis*** | | | |
| Gas fuel type | Methane, Propane, etc. | — | Scene |
| Combustion CO₂ fraction | [0.850, 0.995] | — | Scene |
| Combustion residual fraction | [0.0, 0.150] | — | Scene |
| Pyrolysis fuel fraction | [0.850, 1.0] | — | Scene |
| Pyrolysis residual fraction | [0.0, 0.150] | — | Scene |
| Gas residual reaction fraction | [0.0, 0.500] | — | Scene |
| ***Heat Transfer*** | | | |
| Convection coefficient | [0.0, 200.0] | — | Numerical |
| Radiation ray energy | [0.0, 200.0] | J | Numerical |
| Residual absorption coefficient | [500, 4000] | — | Scene |
| ***Fluid Dynamics*** | | | |
| Vorticity confinement strength | [0.0, 50.0] | — | Scene |
| Wind velocity $(x, y, z)$ | [−15.0, 15.0] | m s⁻¹ | Scene |
| Porous solid velocity multiplier | [0.10, 0.50] | — | Scene |
| Ambient temperature | 300 | K | Physical |
| Ambient gas density | 1.1726 | kg m⁻³ | Physical |
| Gravitational acceleration | −9.81 | m s⁻² | Physical |
| Porous solid density threshold | [0.5, 10.0] | kg m⁻³ | Scene |
| Solid density threshold | [59.0, 1000.0] | kg m⁻³ | Scene |
| ***Particle System*** | | | |
| Particle capacity | [1024, 1 048 576] | — | Numerical |
| Ember spawn rate | [0.0, 10.0] | — | Scene |
| Ember lifetime | [1.0, 120.0] | s | Scene |
| Ember mass | [10⁻⁶, 0.2] | kg | Physical |
| Ember emissivity | 0.95 | — | Physical |
| Ember mass loss rate | [10⁻⁶, 0.06] | kg s⁻¹ | Scene |
| Ember grid velocity factor | [0.03, 0.50] | — | Scene |
| Flame-front lifetime | [0.5, 5.0] | s | Numerical |
| Flame-front speed | [1.0, 10.0] | m s⁻¹ | Physical |
| ***Procedural Scene Generation*** | | | |
| OSM request size | [0, 200] | m | Scene |
| Vegetation density | [0.0, 0.60] | — | Scene |
| Leaf thickness | [10⁻⁶, 0.01] | m | Scene |
| Minimum house distance | [0.0, 12.5] | m | Scene |
| Minimum road distance | [0.0, 2.45] | m | Scene |
| ***Computational Domain*** | | | |
| Fixed timestep | [0.004, 0.10] | s | Numerical |
| Maximum simulation time | [0.0, 300.0] | s | Scene |
| Grid resolution $(x, y, z)$ | [128 × 128 × 80] – [600 × 600 × 300] | voxels | Numerical |
| Domain size $(x, y, z)$ | [16 × 16 × 8] – [600 × 600 × 300] | m | Scene |

---

## Unreadable glyphs

None. Every equation, table cell and symbol was legible in renders of the PDF at 300 to 550 dpi, so no `[?]` markers were needed.

## Transcriber's notes

These notes are ours, not the authors'. The transcription above keeps the original wording, symbols and signs unchanged; the points below flag places where the original looks like a typo or is internally inconsistent. Page numbers are article pages (246:n), which equal the PDF page numbers.

1. **Oxidizer subscript and heat-of-reaction superscript (p. 7, p. 19).** Eq. (10) prints the oxidizer concentration as $c_0$ (digit zero), while the sentence after it and the A.4 symbol list use $c_o$ (letter o). Likewise the equations print $H^0$ (zero) while A.4 prints $\Delta_{p,s}H^{o}$ and $\Delta_c H^{o}$ (letter o).
2. **Radiation operator (p. 7, p. 9, p. 19).** Eqs. (8) and (17) write the radiation term as $\nabla q_{\mathrm{rad}}$, a gradient of a scalar, where an energy balance needs a divergence of a flux. The A.4 list names the same quantity $\Delta q_{rad}$.
3. **Sign of the volumetric radiation term (p. 7, p. 9).** Eq. (8) has $-\nabla q_{\mathrm{rad}}$ on its right-hand side, and Eq. (17) sets $-\nabla q_{\mathrm{rad}} = 4Y_s\kappa_s\sigma(T_g^4 - T_{\mathrm{amb}}^4)$, which is positive for hot gas. Substituted literally, hot gas would gain energy, while the text calls the term "a local energy loss term" that "captures net radiative cooling".
4. **Arrhenius symbol indices (p. 7, p. 19).** Eq. (10) uses $A_{c,g}$ and $E_{a,c,g}$; the A.4 list gives $A_{c,s}$ and $E_{a,c}$ for the same roles.
5. **Signs of the solid heat terms (p. 7, p. 8, p. 10, p. 18).** Eq. (9) carries a minus sign, $\dot{q}_{c,g} = -\Delta_{c,g}H^0 v_{c,g}$, and Table 3 gives a negative gas heat of combustion ($-8.34\mathrm{e}5$ J/mol), so gas-phase heat release is positive. Eqs. (14) and (15) have no minus sign. With the Table 3 values, $\dot{q}_{c,s} = \Delta_{c,s}H^0 v_{c,s}$ with $-2.80\mathrm{e}6$ J/mol is negative (a sink), although the text says solid oxidation "releases heat"; and $\dot{q}_{p,s} = \Delta_{p,s}H^0 v_{p,s}$ with $+2.50\mathrm{e}05$ J/kg is positive and enters Eq. (18) with a plus sign, although the text calls pyrolysis an endothermic "thermal sink". Either a different sign convention applies to the solid terms or a minus sign is missing.
6. **Units of the solid mass source terms (p. 8, p. 19).** A.4 gives the pyrolysis rate $v_{p,s}$ in kg m⁻³ s⁻¹ and the combustion rate $v_c$ in mol m⁻³ s⁻¹. Then $\dot{m}_{c,s} = (M_i/\rho_i)\,v_{c,s}$ has units s⁻¹ as listed, but $\dot{m}_{p,s} = (M_i/\rho_i)\,v_{p,s}$ has units kg mol⁻¹ s⁻¹, not the s⁻¹ listed in A.4. In addition, Eqs. (14) and (15) use the solid material's $M_i$ and $\rho_i$ (a rate of solid volume fraction), yet the terms are added to the gas mass-fraction equation (11); the conversion between the two is not stated.
7. **Units of heat source terms (p. 19).** A.4 lists $\dot{q}_c$, $\dot{q}_{e,s}$, $\Delta q_{rad}$ and $\dot{q}_p$ in J s⁻¹ (W), but Eqs. (8) and (18) are written per unit volume, which requires W m⁻³.
8. **Equation of state (p. 6).** Eq. (7) reads $\rho_g = \rho_{\mathrm{amb}}\,(R T_g \sum_i Y_i/M_i)^{-1}$. Since $R T_g \sum_i Y_i/M_i$ has units J kg⁻¹, the ideal-gas law needs a pressure (Pa) in place of $\rho_{\mathrm{amb}}$ for the result to be a density; with $\rho_{\mathrm{amb}} = 1.1726$ kg m⁻³ (Table 5) the right-hand side is not dimensionally a density unless a normalization is implied.
9. **Buoyancy sign (p. 6, p. 19).** Eq. (6) has $-\boldsymbol{g}(1 - \rho_{amb}/\rho_g)$. If $\boldsymbol{g}$ points downward (Table 5 lists gravitational acceleration as $-9.81$ m s⁻²), this term accelerates hot gas ($\rho_g < \rho_{amb}$) downward; the usual Boussinesq-type form is $+\boldsymbol{g}(1 - \rho_{amb}/\rho_g)$. The result depends on a sign convention for $\boldsymbol{g}$ that the paper does not state.
10. **Absorption coefficient (p. 9, p. 19).** Eq. (17) and $P_{\mathrm{abs}}$ use $\kappa_s$, the following sentence says "absorption coefficient $\kappa$", and A.4 lists $\kappa$ as dimensionless; inside $\exp(-\kappa_s Y_s \ell)$ with $\ell$ a length, $\kappa_s$ needs units of m⁻¹. Also, $Y_s$ (soot mass fraction) uses the subscript $s$, which A.4 reserves for solid properties.
11. **Pyrolysis heat symbol (p. 10).** The sentence after Eq. (18) refers to "$\dot{q}_p$" while the equation uses $\dot{q}_{p,s}$ (A.4 also lists $\dot{q}_p$).
12. **Heat transfer coefficient units (p. 10, p. 19).** A.4 lists $h$ as dimensionless, but $h\,(T_s - T_g)$ in Eq. (19) must be a heat flux, so $h$ needs W m⁻² K⁻¹. The Table 5 entry "Convection coefficient", range [0.0, 200.0] with no unit, is probably this $h$ (our reading; the paper does not link the two names).
13. **Initial temperature symbol (p. 10, p. 11).** Alg. 1 line 2 initializes $T_s, T_g \leftarrow T_{amb}$; the text of Sec. 5.1 says the temperatures are set to $T_0$.
14. **Figure panel reference (p. 9).** Sec. 4.8 refers to embers as "(Fig. 6,c)", but the Fig. 6 caption assigns flame-front particles to panel (c) and embers to panel (d).
15. **Species naming in the tree experiment (p. 14, p. 15).** Sec. 6.2 and Fig. 15 describe a dry Douglas-fir (fir) Christmas tree; the Fig. 16 caption and its plot title say "dry pine tree".
16. **Table 1 number format (p. 16).** Fig. 11, Adaptive, Render is printed "4.180" with a decimal point, where every other entry uses a thousands comma; 4,180 ms is the likely intended value.
17. **Minor wording slips, kept as printed.** Fig. 6 caption "falls of quickly" (probably "falls off"); Fig. 7 caption begins "A Temporal progression"; Sec. 5.2 "rendered by training tracing rays" (stray word); Fig. 14 caption "We simulated 12 m a pool-fire" (word order).
18. **Ranges in Table 5 versus the experiment tables.** Several values used in the supplement's validation tables fall outside the ranges given in Table 5 (p. 19); these are listed in the transcriber's notes of `flamescapes-2026-supplement.md`.
