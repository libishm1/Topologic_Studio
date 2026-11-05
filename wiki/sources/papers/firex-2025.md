# Fire-X: Extinguishing Fire with Stoichiometric Heat Release (transcription)

Wrede, H., Wagner, A. R., Mahfuz, S. M., Pałubicki, W., Michels, D. L., Pirk, S. (2025). Fire-X: Extinguishing Fire with Stoichiometric Heat Release. ACM Transactions on Graphics 44(6), Article 268. https://doi.org/10.1145/3763338

Licence: Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/), as stated on the paper's first page. © 2025 the authors.
Changes: transcribed from the authors' PDF (project page https://helgewrede.github.io/firex/, retrieved 2026-11-05) to Markdown on 2026-11-05; equations retyped in LaTeX; figures replaced by their captions; layout changed. Transcription errors are ours; cite the original.

Transcription conventions: text is in reading order (left column, then right column). Running heads and page footers are omitted; page starts are marked with HTML comments (`<!-- p. N -->`). Words hyphenated at line ends are rejoined, except real compounds. Figure captions and floats are placed at the nearest paragraph break after their position on the page. Text added by the transcriber is marked "Transcriber:". The equation catalogue is in `firex-2025-equations.json` next to this file. Notes on suspected typos and inconsistencies are at the end ("Transcriber's notes"); the transcription itself is unchanged.

---

<!-- p. 1 -->

HELGE WREDE, Kiel University, Germany
ANTON R. WAGNER, Kiel University, Germany
SARKER MIRAZ MAHFUZ, Kiel University, Germany
WOJTEK PAŁUBICKI, Adam Mickiewicz University, Poland
DOMINIK L. MICHELS, KAUST, KSA
SÖREN PIRK, Kiel University, Germany

> Figure 1: A complex scene of fire and water interactions generated with our framework. A large fire affecting three cars is extinguished with a water jet, causing large amounts of smoke and vapor. Our combustion model enables simulating multi-species thermodynamics and the extinction of flames.

We present a novel combustion simulation framework to model fire phenomena across solids, liquids, and gases. Our approach extends traditional fluid solvers by incorporating multi-species thermodynamics and reactive transport for fuel, oxygen, nitrogen, carbon dioxide, water vapor, and residuals. Combustion reactions are governed by stoichiometry-dependent heat release, allowing an accurate simulation of premixed and diffusive flames with varying intensity and composition. We support a wide range of scenarios including jet fires, water suppression (sprays and sprinklers), fuel evaporation, and starvation conditions. Our framework enables interactive heat sources, fire detectors, and realistic rendering of flames (e.g., laminar-to-turbulent transitions and blue-to-orange color shifts). Our key contributions include the tight coupling of species dynamics with thermodynamic feedback, evaporation modeling, and a hybrid SPH-grid representation for the efficient simulation of extinguishing fires. We validate our method through numerous experiments that demonstrate its versatility in both indoor and outdoor fire scenarios.

CCS Concepts: • **Computing methodologies** → **Physical simulation**; **Interactive simulation**.

Additional Key Words and Phrases: Combustion, Evaporation, Extinguishing, Fire Simulation, Fluid Simulation, Heat Release, Stoichiometry.

**ACM Reference Format:**
Helge Wrede, Anton R. Wagner, Sarker Miraz Mahfuz, Wojtek Pałubicki, Dominik L. Michels, and Sören Pirk. 2025. Fire-X: Extinguishing Fire with Stoichiometric Heat Release. *ACM Trans. Graph.* 44, 6, Article 268 (December 2025), 17 pages. https://doi.org/10.1145/3763338

Authors' Contact Information: Helge Wrede, helge.wrede@email.uni-kiel.de, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany; Anton R. Wagner, awa@informatik.uni-kiel.de, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany; Sarker Miraz Mahfuz, smm@informatik.uni-kiel.de, Kiel University, Christian-Albrechts-Platz 4, 24118 Kiel, Germany; Wojtek Pałubicki, wojciech.palubicki@amu.edu.pl, Adam Mickiewicz University, 61-614, Poznań, Poland; Dominik L. Michels, dominik.michels@kaust.edu.sa, KAUST, Thuwal 23955, KSA; Sören Pirk, soeren.pirk@gmail.com, Kiel University, 24118 Kiel, Germany.

This work is licensed under a Creative Commons Attribution 4.0 International License.
© 2025 Copyright held by the owner/author(s).
ACM 1557-7368/2025/12-ART268
https://doi.org/10.1145/3763338

<!-- p. 2 -->

## 1 Introduction

Simulating combustion in computer graphics poses substantial challenges due to the interplay of fluid dynamics, thermodynamics, and chemical kinetics across multiple phases – solids, liquids, and gases. Modeling combustion and fire effects is critical in many domains: from visual storytelling in games and films [Nielsen et al. 2019], to safety training [Kinateder et al. 2014], fire suppression planning [Cao et al. 2020], and wildfire research [Hädrich et al. 2021; Kokosza et al. 2024]. Furthermore, modeling different extinguishing behaviors is important for capturing the wide range of fire-suppression phenomena encountered in real-world scenarios. Whether it is a sprinkler diffusing fine water droplets, a fire extinguisher dispersing foam or gas, or a direct hose stream dousing flames, each method interacts with combustion in distinct thermo- and fluid-dynamic ways. Visually differentiating between extinguishing mechanisms – such as steam plumes from evaporation or darkening flames due to oxygen deprivation – adds to the visual fidelity and immersion to virtual scenes.

While computer graphics has made significant progress in flame rendering and fluid animation [Kim et al. 2008; Nguyen et al. 2002; Stam 1999], most existing techniques either simplify combustion chemistry or focus solely on gaseous flames, ignoring phase transitions, multi-fuel reactions, or water-based suppression. On the other hand, physics-based approaches for fire provide high-fidelity models but are designed for predictive simulation and not for interactive applications [Merci and Beji 2022; Nielsen et al. 2022]. Other approaches focus on simulating water [Clavet et al. 2005] also undergoing temperature changes [Mihalef et al. 2006], the interaction of multiple fluids [Losasso et al. 2006a], and the melting and burning of solids into liquids and gases [Losasso et al. 2006b]. The coupling of reactive multi-phase flow, with varying fuel compositions and evaporation, remains largely unaddressed in an interactive or controllable setting.

In this work, we present a physically-based combustion model which addresses the simulation of multiple species at interactive rates. Compared to other combustion models that focus on specific aspects of combustion phenomena, our method trades descriptive complexity for a more integrated combustion simulation. We show that combustion across solids, liquids, and gases can be unified within an efficient hybrid framework that couples Eulerian grids for thermochemical fields with Lagrangian smoothed particle hydrodynamics (SPH) for fluid motion and droplet dynamics. This enables the simulation of a broad range of scenarios, such as premixed flames, water sprays, candle and jet fires, and suppression via sprinklers and direct water jets. Unlike previous models, our framework explicitly tracks major species – fuel, oxygen, carbon dioxide, nitrogen, water vapor, and residuals – allowing stoichiometry-dependent heat release, oxygen starvation, and phase transition effects such as evaporation. Our approach extends the state-of-the-art of simulating combustion by modeling complete and incomplete combustion, along with visual cues such as flame color shifts and smoke as well as vapor transitions. Moreover, our solver efficiently couples chemical kinetics and thermodynamics with a particle-grid representation to maintain high accuracy with interactive runtime performance. Additionally, we render realistic flame and smoke behaviors, modulated by environmental conditions and user-defined parameters.

In Fig. 1, we show a complex simulation showcasing the dynamic interaction between fire and water. Three vehicles are engulfed in intense flames which are being extinguished by a high-pressure water jet. The interaction of fire and water results in large smoke and vapor clouds. Our novel combustion model captures multi-species thermodynamics, allowing realistic visualization of flame extinction and complex thermal behaviors in reactive flows.

In summary, our contributions are (1) a hybrid combustion simulation framework combining SPH with an Eulerian thermochemical grid, (2) a stoichiometry-aware reaction and heat release model spanning multiple combustion species, (3) a coupled liquid-gas-solid model for extinguishing dynamics, and (4) an interactive parameter space for rendering and controlling fire behaviors in various scenarios.

## 2 Related Work

Simulating fire and combustion has been a long-standing research topic in both computer graphics and physics-based modeling. Seminal books in combustion modeling and fluid dynamics include Merci and Beji [2016], Kross and Potter [2014], Peters [2000], as well as Bridson's introduction to fluid solvers [Bridson 2015]. Fluid motion in most fire models is described by the incompressible Navier-Stokes equations, often discretized using semi-Lagrangian advection and pressure projection [Versteeg and Malalasekera 2007].

In computer graphics, fire is typically modeled using computational fluid dynamics (CFD) and grid-based fluid solvers [Bridson and Müller-Fischer 2007], which effectively capture both laminar and turbulent flame behaviors [Hong et al. 2007; Nguyen et al. 2002; Stam 1999], as well as smoke dynamics [Fedkiw et al. 2001; Pan and Manocha 2017]. Various approaches have been developed for the visual modeling of fire, including physically-based methods [Nguyen et al. 2002; Nielsen et al. 2022, 2019; Pegoraro and Parker 2006], techniques that emphasize the physical properties of flames [Nguyen et al. 2001], methods that prioritize artistic control [Aguilera and Johansson 2019; Kim et al. 2017; Lamorlette and Foster 2002], and models that rely on particle-based representations [Horvath and Geiger 2009]. Futhermore, vorticity confinement [Bridson 2015] is a widely used method to enhance the visual fidelity of flames.

In physics-based fire simulations, the modeling process aims to accurately replicate the underlying thermodynamic and chemical phenomena governing combustion. These simulations include detailed representations of fluid motion, heat transfer, and combustion reactions, incorporating processes such as convection, conduction, radiation, and finite-rate chemical kinetics [Merci and Beji 2016]. Radiative heat transfer is commonly handled through models based on the Stefan-Boltzmann law, with some approaches using ray casting techniques to approximate radiative heating of the surrounding fuel [Stam and Fiume 1995]. By integrating these physical principles, such models can produce realistic flame behavior and energy distribution, making them suitable for high-fidelity applications in both visual effects and scientific visualization [Nielsen et al. 2022, 2019; Pegoraro and Parker 2006]. Chemical reactions in fire are often <!-- p. 3 --> modeled using modified Arrhenius equations as formalized in the IUPAC Gold Book [1997], while the heat of combustion is computed following thermochemical approaches [Schmidt-Rohr 2011].

> Figure 2: Overview of our multiphase framework: our liquid-gas-solid model enables simulating the thermodynamics of solids, liquids and gases. We use a Lagrangian representation for solids (a) and liquids (b) and an Eulerian representation for gases (d). To maintain all states of matter, we synchronize solid and fluid particles with the gas grid in two update steps (c, e).

More extreme combustion phenomena, such as explosions and detonations, require compressible flow models [Ihm et al. 2004]. Kwatra et al. [2009] proposed a method that dynamically transitions between compressible and incompressible regimes, making it suitable for simulating shock waves and pressure-driven deflagrations in graphics contexts. Particle-based explosion models, such as that of Feldman et al. [2003], combine Eulerian air simulation with Lagrangian fuel particles and approximate heat transfer. Advection and diffusion are typically handled using semi-Lagrangian schemes [Stam 1999], while pressure projection and multigrid solvers are used for velocity field correction. Tools like AMGCL [2019] and PyBullet [2021] as well as methods for fast fluid simulations [Rabbani et al. 2022] support scalable fluid-rigid interaction and large-scale solver efficiency. Sparse volumetric data structures such as OpenVDB [2013] facilitate the efficient handling of complex simulation domains at high resolution.

While we do not focus on simulating the combustion of solids, several approaches leverage volumetric grids to enable the tracking of disconnected, propagating fire fronts [Liu et al. 2012, 2009; Melek and Keyser 2002; Zhao et al. 2003] and support fire spreading across surfaces [Chiba et al. 1994]. Hong et al. [2010] present a method capable of modeling combustion under complex geometric constraints, while Pirk et al. [2017] focus on the combustion of geometrically complex plant models. Stomakhin et al. [2014] introduce a point-based method for simulating heat transfer to capture the melting and solidification of materials. Their approach models the thermodynamic and mechanical behavior of various substances but lacks support for real-time interaction and heat diffusion. More recently, Liu et al. [2025] use signed distance fields to support the combustion of general wooden structures. Their approach enables to efficiently query the surface information required to compute the insulating effect caused by the char layer.

Physically-based fire simulators have been developed to model the complex interactions between combustion, fluid flow, and heat transfer, particularly in engineering and safety-critical environments. The Fire Dynamics Simulator (FDS), is a widely used computational fluid dynamics tool for simulating fire-driven flows in buildings and enclosures [McGrattan et al. 2013]. It incorporates low-Mach number Navier-Stokes equations, radiation transport, combustion models, and detailed treatment of smoke and heat propagation. Several methods aim at simulating combustion for wildfire scenarios, including WFDS [Mell et al. 2007], WRF-Fire [Coen et al. 2013], CAWFE [Coen 2013], ABWiSE [Katan and Perez 2021], FIRETEC [Linn et al. 2002], OpenFOAM[Lapointe et al. 2020], and QUIC-Fire [Linn et al. 2020]. Bakhshaii and Johnson [2019] offer a comprehensive review of modeling strategies across empirical, physical, and hybrid domains, highlighting the trade-offs between real-time capability and physical accuracy.

Unlike the existing methods, which primarily focus on either high-fidelity offline simulation or artist-driven visual effects, we introduce a method that allows the interactive simulation of extinguishing behaviors of flames. Furthermore, we devise a principled physics model that can simulate stoichiometric reactions that accurately capture heat release across multiple combustion regimes, including diffusion and premixed flames.

## 3 Overview

Our hybrid multiphase fluid model is expressed in two different spatial domains. We use a continuous Cartesian coordinate system and a discrete uniform grid composed of voxels to capture gas, liquid, and solid interactions relevant for combustion scenarios (see Fig. 2). The discrete simulation domain represents either gaseous or solid regions, using a binary occupancy flag. SPH particles representing liquids are embedded in the continuous spatial domain and are synchronized with the grid simulation.

At initialization, the solid geometry usually defined as meshes is encoded as a dense point cloud (**solids**, Fig. 2a). Each point populates a corresponding grid cell, which is flagged as solid and used to enforce boundary conditions for velocity, temperature, and species transport during simulation. A user can specify emitters to generate SPH particles for fluids. Particles contribute mass, temperature, and velocity to the underlying grid using conservative sampling schemes (**liquids**, Fig. 2b, c). The particles explicitly represent the liquid phase (e.g., water or ethanol) and model its behavior using SPH forces to capture realistic fluid dynamics. Our SPH simulation describes surface tension, pressure projection, and collision using a double density relaxation scheme. Each gaseous grid cell stores thermodynamic quantities including velocity, density, temperature, and a vector of chemical species mass fractions $Y_i$ (Fig. 3).

<!-- p. 4 -->

Our grid solver describes diffusion, pressure projection, liquid evaporation, advection, buoyancy, combustion, vorticity confinement, thermal properties, density, and species transport. Gas density is dynamically updated from local temperature and species composition using a temperature-dependent ideal gas law formulation describing chemical species evolution. This allows our model to express stochiometric combustion over a range of different gas mixtures (**gases**, Fig. 2d).

Solid cells contribute to heat conduction and boundary enforcement; liquid particles exchange mass and energy with the gas phase through heat-driven evaporation (Fig. 2e). Particles whose lifetime has expired (e.g., due to complete evaporation) are removed. This two-way coupling handles multiphase interfaces and phenomena such as heat conduction from solid walls into fluid, or energy-driven phase transitions between liquid and vapor. Our evaporation model includes both continuous heat-based mass loss and discrete droplet-based evaporation tied to local energy budgets (Fig. 3).

Our approach supports modeling of thermodynamic processes including conduction, and chemical kinetics using modified Arrhenius equations, and supports grid-based combustion dynamics such as flame propagation, soot production, and volumetric expansion due to temperature increase. The model is designed to accommodate both physical approximations of deflagration scenarios and artist-tuned stylizations, depending on the application domain.

## 4 Methodology

In this section, we present our multiphase combustion model. We begin by outlining the equations for fluid motion, thermodynamics, and species transport, including extensions for combustion-driven buoyancy and heat generation. Finally, we detail the evaporation model, which describes liquid–gas phase transitions based on thermal energy balance and droplet-level resolution. A list of parameters is included in Appendix A.1.

### 4.1 General

The simulation domain defines velocity, temperature, species fractions, and material states, described by coupled PDEs for fluid dynamics, thermodynamics, and species transport as illustrated in Fig. 3. To model natural convection (i.e., buoyancy-driven flow), the incompressible Navier-Stokes equation is adapted to ideal gas flows using the Boussinesq approximation and hydrostatic pressure is neglected. We obtain the following momentum equation:

$$
\frac{D\boldsymbol{u}}{Dt} = \nu \nabla^2 \boldsymbol{u} - \frac{1}{\rho}\nabla p - \boldsymbol{g}\left(1 - \frac{T_{\mathrm{amb}}}{T}\right) . \tag{1}
$$

Here, $\boldsymbol{u}$ denotes the velocity field, $\nu$ the kinematic viscosity, $\rho$ the fluid's local density, $p$ pressure, $T$ temperature, and $T_{\mathrm{amb}}$ the reference ambient temperature. The right term with effective gravity $\boldsymbol{g}$ introduces buoyancy based on temperature difference, enabling plume rise and convection effects critical for combustion-driven flows.

Thermal transport is governed by a modified heat equation which has been extended with radiative cooling and a combustion energy source term. We obtain

$$
\frac{DT}{Dt} = \frac{k}{\rho C_p}\nabla^2 T + \frac{\epsilon\sigma}{C_p}\left(T_{\mathrm{amb}}^4 - T^4\right) + \frac{J_c}{\rho C_p} , \tag{2}
$$

in which $k$ denotes thermal conductivity, $C_p$ specific heat at constant pressure, $\epsilon$ emissivity, and $\sigma$ the Stefan-Boltzmann constant. The radiative term accounts for the net exchange of energy with the environment. The source term $J_c$ (see Eq. 6) incorporates the local heat release rate from combustion.

> Figure 3: Schematic representation of the multiphysics model, illustrating the interdependent processes governing fire plume dynamics. The model couples natural convection, thermal transport, chemical species evolution, evaporation, and combustion.

Transcriber: Fig. 3 contains five boxed equations, transcribed here with their printed labels (not numbered in the paper):

Thermal Transport (Eqs. 2, 6):

$$
\frac{\partial T}{\partial t} + \boldsymbol{u}\cdot\nabla T = \frac{k}{\rho C_p}\nabla^2 T + \frac{\epsilon\sigma}{C_p}\left(T_{\mathrm{amb}}^4 - T^4\right) - \frac{\Delta_c H^0}{\rho C_p}\varphi v_c
$$

Natural Convection (Eq. 1):

$$
\frac{D\boldsymbol{u}}{Dt} = \nu\nabla^2\boldsymbol{u} - \frac{1}{\rho}\nabla p - \boldsymbol{g}\left(1 - \frac{T_{\mathrm{amb}}}{T}\right)
$$

Chemical Species Evolution (Eqs. 3, 11):

$$
\frac{\partial Y_i}{\partial t} + \boldsymbol{u}\cdot\nabla Y_i = D_i\nabla^2 Y_i + \Gamma_i\rho_l + \frac{q_i M_i}{\rho} v_c
$$

Evaporation (Eq. 9):

$$
\frac{\partial \rho_l}{\partial t} = -\frac{\rho_l C_p (T_l - T_{\mathrm{amb}})}{C_p(T_B - T_{\mathrm{amb}}) + \Delta_v H^0}
$$

Combustion (Eq. 5):

$$
v_c = A T^n \exp\left(\frac{-E_a}{RT}\right) c_f^a\, c_o^b
$$

The evolution of chemical species is handled per-species, i.e.,

$$
\frac{DY_i}{Dt} = D_i \nabla^2 Y_i + \frac{v_c q_i M_i}{\rho} + S_i , \tag{3}
$$

in which $Y_i$ denotes mass fraction of species $i$, $D_i$ diffusion coefficient, and $S_i$ represents the source term from the phase change.

### 4.2 Combustion

We model combustion using a simplified global one-step reaction mechanism. The heat of combustion is approximated from the empirical formula of the fuel:

$$
\Delta_c H^o \approx -417(c + 0.25h - 0.5o) . \tag{4}
$$

This estimates the lower heating value (measured in kJ mol$^{-1}$) for hydrocarbon fuels of composition $C_c H_h O_o N_n$ [Schmidt-Rohr 2015].

The combustion rate follows an Arrhenius-type expression

$$
v_c = A T^n \exp\left(\frac{-E_a}{RT}\right) c_f^a\, c_o^b , \tag{5}
$$

in which $v_c$ denotes the volumetric reaction rate, $T$ temperature, $R$ universal gas constant, $A$ pre-exponential factor, and $E_a$ activation energy. The molar concentrations of fuel and oxidizer $c_f$ and $c_o$ are raised to empirical exponents $a$ and $b$. The resulting combustion energy rate per unit volume is given by

$$
J_c = -\varphi v_c \Delta_c H^0 \tag{6}
$$

which occurs in Eq. 2 feeding back the heat release into the thermal equation. We use $\varphi$ as a weighting factor for adjusting the heat efficiency of the combustion (e.g., in the case of an incomplete combustion).

<!-- p. 5 -->

> Figure 4: Visualization of various hydrocarbons: we simulate the the establishment of a flame 800 ms after ignition for Acetylene (a), Butane (b), Cyclopropane (c), Propane (d), Methane (e), and Ethylene (f). Our framework enables simulating stoichiometric mixtures of various fuels. The combustion patterns reflect differences in turbulence, stability, and height, related to molecule properties.

Moreover, species evolution from the combustion reaction is modeled by

$$
\frac{\partial Y_i}{\partial t} = \frac{v_c q_i M_i}{\rho} , \tag{7}
$$

in which $q_i$ denotes the stoichiometric coefficient for species $i$ with molar mass $M_i$.

To complete the system, we relate the gas density to pressure, temperature, and mixture composition using the ideal gas law:

$$
\rho = p_{\mathrm{amb}} \left( T R \sum_i \frac{Y_i}{,} M_i \right)^{-1} , \tag{8}
$$

where $R$ is the universal gas constant. This formulation accounts for varying mixture composition and temperature effects in a physically grounded manner.

### 4.3 Evaporation Model

To model phase change from liquid to gas, we implement an energy-driven evaporation scheme. The liquid density $\rho_l$ decreases according to the local energy surplus:

$$
\frac{\partial \rho_l}{\partial t} = -\frac{\rho_l C_p (T_l - T_{\mathrm{amb}})}{C_p (T_B - T_{\mathrm{amb}}) + \Delta_v H^0} , \tag{9}
$$

in which $T_l$ denotes the current liquid temperature, $T_B$ boiling temperature, $C_p$ liquid specific heat capacity, and $\Delta_v H^0$ latent heat of vaporization. Droplet conduction accelerates the heat transfer between the air and the liquid as we assume a spray of droplets instead of a laminar flow:

$$
T_{dc} = \frac{\rho_l}{\rho_d}\frac{A_d}{d_d}(T_l - T) , \quad \frac{\partial T}{\partial t} = \frac{k T_{dc}}{\rho C_p} , \quad \frac{\partial T_l}{\partial t} = -\frac{k T_{dc}}{\rho C_p} . \tag{10}
$$

The droplet's diameter is denoted with $d_d$, its surface area with $A_d$, and its density with $\rho_d$. Please note that the latter one does not correspond to the liquid density, but instead to the amount of liquid in the form of droplets.

The mass fraction of a specific species changes based on the evaporated density and an evaporation coefficient. This decouples the evaporation process from the actual liquid and gas compositions, and enables the evaporation of different mixtures:

$$
\frac{\partial Y_i}{\partial t} = \Gamma_i \rho_l (= S_i) , \tag{11}
$$

in which $\Gamma_i$ denotes the evaporation coefficient for a specific species.

## 5 Algorithmics and Implementation

Our simulation framework is implemented in `Rust` and uses `WGPU` as the primary GPU abstraction layer for compute shaders written in `WGSL`. The system architecture is based on the `shipyard` entity-component system, providing structured, parallel simulation state management. User interaction and visualization are handled through `winit` for windowing and `egui` for the GUI layer. All results presented in this paper were generated on a workstation equipped with an Intel i9-13900K CPU, 128 GB of RAM, and an NVIDIA RTX 4090 GPU. The implementation closely follows the numerical model described in Algo. 1 which summarizes the complete simulation loop, incorporating Eulerian fluid dynamics for gas and heat transport, Lagrangian SPH for liquid behavior, and thermochemical models for evaporation, phase change, and combustion.

To differentiate between all the different parts of the system, we settled on a few different subscripts. The subscript $s$ denotes properties that are only used for the correct behavior of the SPH system and therefore only appear in particle to particle interactions. The subscript $p$ is used for particle properties that can are exchanged with the grid. Both quantities, marked with either $s$ or $p$, are stored in the particles. Liquid properties that are stored in the grid are denoted by $l$. All other properties that are stored in the grid are written without any subscript.

### 5.1 Data Structures and Initialization

The simulation domain is defined on a voxel grid, where each cell is flagged as *gas*, *solid*, or *gas+liquid*. Scalar fields such as temperature $T$, species mass fractions $Y_i$, and pressure $p$ are stored at cell centers, while velocity $\boldsymbol{u}$ is represented in a staggered marker-and-cell (MAC) grid layout with components stored at face centers [Bridson 2015]. Solid obstacles are initialized as uniformly distributed particles, which allows for particle-particle collisions. These are then sampled onto the grid and cells are flagged as a solid accordingly. This dual representation of solids enables both the particles and the grid to interact with solids. Dirichlet boundary conditions (fixed values) on solid walls are applied for velocity and Neumann (zero-gradient) conditions for pressure. Slip conditions are enforced with mirrored ghost cells [Bridson 2015]. For temperature, we impose fixed wall <!-- p. 6 --> values and apply one-sided derivatives near boundaries. The simulation uses a fixed timestep constrained by Courant–Friedrichs–Lewy (CFL) criteria for both advection and diffusion.

SPH particles can be introduced into user-defined regions of a continuous Cartesian coordinate system and are colocated with the voxel grid. Each particle carries velocity $\boldsymbol{u}_p$, temperature $T_p$, mass $m_p$, and thermophysical properties such as specific heat and vaporization enthalpy. SPH-to-grid interpolation is performed using compact support kernels and normalization factors to ensure conservative transfer of physical quantities. A list of SPH parameters is included in the Appendix (Tab. A.1.2).

### 5.2 Smoothed Particle Hydrodynamics

Liquid behavior such as water spray, suppression, and droplet dynamics is handled using SPH. We use pseudo-pressure and pseudo-density formulations from Clavet et al. [2005]. Particle to particle interactions are handled within the smoothing radius $h_s$. The neighborhoods of the particles are denoted with $N_s(i)$ with $r_{s_{ij}} \le h_s$ and $r_{s_{ij}}$ as the Euclidean distance between two particles. Each particle's local density is estimated by

$$
\rho_{s_i} = \sum_{j \in N_s(i)} \left(1 - \frac{r_{s_{ij}}}{h_s}\right)^2 . \tag{12}
$$

Furthermore, a near-density term is computed with a cubic kernel. This allows for surface tension effects and particle collision avoidance due to the flat gradient of the quadratic smoothing kernel. It is defined by

$$
\rho_{s_i}^{\mathrm{near}} = \sum_{j \in N_s(i)} \left(1 - \frac{r_{ij}}{h_s}\right)^3 . \tag{13}
$$

These are then converted into pseudo-pressures

$$
p_{s_i} = k_s(\rho_{s_i} - \rho_{s_0}) , \quad p_{s_i}^{\mathrm{near}} = k_s^{\mathrm{near}} \rho_{s_i}^{\mathrm{near}} , \tag{14}
$$

where $k_s$ and $k_s^{\mathrm{near}}$ are stiffness coefficients and $\rho_{s_0}$ is the rest density. Pseudo-pressure forces between particles are calculated as

$$
F_{s_i}^{\mathrm{press}} = -\sum_{j \in N_s(i)} p_{s_j}\left(1 - \frac{r_{s_{ij}}}{h_s}\right) + p_{s_j}^{\mathrm{near}}\left(1 - \frac{r_{s_{ij}}}{h_s}\right)^2 \tag{15}
$$

and viscous damping is added through a pairwise radial velocity difference

$$
\Delta\boldsymbol{u}_{s_{ij}}^{\mathrm{vis}} = (\boldsymbol{u}_{p_i} - \boldsymbol{u}_{p_j}) \cdot \hat{r}_{s_{ij}} , \tag{16}
$$

$$
\boldsymbol{a}_{s_i}^{\mathrm{vis}} = (1 - \frac{r_{s_{ij}}}{h_s})(\alpha \boldsymbol{u}_{s_{ij}}^{\mathrm{vis}} + \beta (\Delta\boldsymbol{u}_{s_{ij}}^{\mathrm{vis}})^2)\hat{r}_{s_{ij}} , \tag{17}
$$

where $\alpha$ and $\beta$ control linear and quadratic damping, respectively.

### 5.3 Particle-Grid Sampling

To exchange information between SPH particles and the Eulerian grid, we use a weighted sampling approach based on a simple distance kernel. This sampling uses a different smoothing radius, denoted $h$, with the corresponding neighborhood $N$ containing all particles that satisfy $r_{ij} \le h$. The distance here is the Euclidean distance between positions in the grid and particle positions. We begin by defining a linear distance-based weight between particle $i$ and grid cell $j$, i.e.,

$$
w_{ij} = 1 - \frac{r_{ij}}{h} . \tag{18}
$$

> Figure 5: Four different species of small flame combustion: a clean combustion, common for a Bunsen burner and premixed combustion (a), a flame with a few traces of residuals that show visible glowing (b), an intermediate flame species where the bottom burns near complete combustion but water vapor and glowing residuals are already present in the upper parts of the flame (c), and an even more turbulent flame which shows large amounts water vapor and residuals, leading to colors typical for diffuse combustion (d).

The mass of particles is used to accumulate the total liquid density contribution to grid cell $j$:

$$
\rho_{l_j} = \sum_{i \in N(j)} m_{p_i} w_{ij} . \tag{19}
$$

To normalize contributions, we compute a sampling normalization factor for the liquid density:

$$
\omega_j^{\rho} = \sum_{i \in N(j)} w_{ij} , \tag{20}
$$

which allows us to estimate the particle mass $m_i$ by inverting the density contribution from grid to particles:

$$
m_{p_i} = \left( \sum_{j \in N(i)} w_{ij} \right)^{-1} \sum_{j \in N(i)} \rho_{l_j} \frac{w_{ij}}{\omega_j^{\rho}} . \tag{21}
$$

Temperature values are interpolated similarly. First, we compute a normalization factor for liquid temperature sampling:

$$
\omega_j^{\mathsf{T}} = \sum_{i \in N(j)} m_{p_i} w_{ij} . \tag{22}
$$

Then, the liquid temperature at grid cell $j$ is computed from nearby particles:

$$
T_{l_j} = \sum_{i \in N(j)} T_{p_i} m_{p_i} w_{ij} / \omega_j^{\mathsf{T}} . \tag{23}
$$

To transfer liquid temperatures back from the grid to particles, we use a normalized interpolation:

$$
T_{p_i} = \left( \sum_{j \in N(i)} \frac{m_{p_i} w_{ij}}{\omega_j^{\mathsf{T}}} \right)^{-1} \sum_{j \in N(i)} T_{l_j} \frac{m_{p_i} w_{ij}}{\omega_j^{\mathsf{T}}} . \tag{24}
$$

<!-- p. 7 -->

The velocity of the liquid is also sampled, but only from the particles to the grid. This allows us to model different effects, like the Venturi effect and the rapid increase in air velocity after the evaporation of water. However, since we only sample in one direction, we do not need to ensure consistency between both directions, which simplifies the problem. Using the average velocity of all neighboring cells is sufficient, as we scale the liquid velocity with the sampled density when applying it to the gas velocity:

$$
\boldsymbol{u}_{l_j} = \frac{1}{|N(j)|} \sum_{i \in N(j)} \boldsymbol{u}_{p_i} . \tag{25}
$$

### 5.4 Droplet based Evaporation

The evaporation, just like the combustion, is done on the grid. This is possible because we have bidirectional sampling of the liquid properties between the grid and the SPH system. Before the evaporation is calculated, the liquid properties are sampled from the particles at the beginning of the grid update step and stored in the grid as the liquid properties. After the evaporation, the particle properties are sampled from the grid again before the end of the grid update step. Since we model the liquid as a spray of droplets, we can partially evaporate a liquid cell. To handle droplet-based evaporation explicitly, we define the total available energy in the liquid cell as

$$
E_l = (T_l - T_{\mathrm{amb}}) C_p \rho_l V_c \tag{26}
$$

and the energy required to evaporate one droplet by

$$
E_d = (C_p (T_B - T_{\mathrm{amb}}) + \Delta_v H^0) \rho_d V_d . \tag{27}
$$

Assuming $n_d$ total droplets, the energy balance equation becomes

$$
E_l = E_d n_d , \quad \rho_l = \rho_d n_d . \tag{28}
$$

The number of droplets $n_e = E_l / E_d$, which can be evaporated, is then computed based on the energy budget and the resulting evaporated liquid density by $\rho_e = n_e \rho_d V_d / V_c$. This formulation allows droplets to evaporate incrementally based on the available local energy while ensuring both energy and mass conservation across the grid and the SPH domain.

After evaporation, the new mixture density ratio is computed to adjust thermodynamic properties, i.e.,

$$
\rho' = \frac{\rho}{\rho + \Delta\rho_l} , \tag{29}
$$

the gas temperature is adjusted to reflect the thermal mixing with vapor at boiling point, i.e.,

$$
T' = (1 - \rho') T_B + \rho' T , \tag{30}
$$

and species mass fractions $Y_i$ are rescaled proportionally, i.e.,

$$
Y_i' = \rho' Y_i . \tag{31}
$$

We also use the liquid velocity to model a change in gas velocity similar to the rapid expansion of evaporated water. This is done by using the evaporated density fraction and using it to interpolate between the gas and the liquid velocity:

$$
\boldsymbol{u}' = (1 - \rho') \boldsymbol{u}_l + \rho' \boldsymbol{u} . \tag{32}
$$

> Figure 6: Visualization of flame behavior: a flame in a contained environment is undergoing oxygen starvation from an early stage (top row) to a later stage (bottom row). The images show the RGB rendering (a, e), the temperature field (b, f), fuel (green) and oxygen (blue) (c, g), and CO$_2$ (d, h). In the later stage (bottom row), the flame becomes less intense, temperature drops, and a significant decrease in oxygen concentration is observed which leads to the extinction of the flame.

### 5.5 Simulation Loop

The main simulation loop has been implemented according to Algo. 1. All scalar and vector fields on the grid and particles are initialized to default or user-provided values (Lines 1–4). This includes setting $\boldsymbol{u}$, $T$, $\rho$, $Y_i$ on the grid, and initializing $\boldsymbol{u}_p$, $T_p$, $m_p$ for each particle. Pseudo-density and near-density are computed for each particle (Eqs. 12 and 13), followed by pressure forces (Eqs. 14 and 15), viscosity forces (Eq. 17), and advection. Boundary enforcement at walls is handled using ghost particles and velocity reflection schemes (Lines 5–12). SPH particles contribute to the grid via weighted kernel sampling of mass, temperature, and velocity (Eqs. 19, 23 and 25). The weighting functions are normalized per cell to maintain conservation (Lines 13–14).

The gas solver runs a sequence of updates: semi-Lagrangian advection, Boussinesq buoyancy, chemical reaction modeling (Eqs. 1 and 3), radiative thermal exchange (Eq. 2), and thermal as well as species diffusion. The Arrhenius equation is checked for each cell to combust the available fuel and oxygen, resulting in changes in temperature and species composition (Eqs. 4, 5 and 7). In grid cells containing a hot liquid, thermodynamic conditions are evaluated to trigger evaporation (Eqs. 9 to 11 and 26 to 31). Adjustments are made to local temperature, species mass fractions, and particle mass. Incompressibility is enforced by solving a pressure Poisson equation using a GPU-based iterative solver (Lines 15-22). The grid fields are interpolated back to SPH particles (Lines 23–24), updating their mass and temperature values using conservative kernels (Eqs. 21 <!-- p. 8 --> and 24). Particles that are fully evaporated or exceed user-defined lifespans are discarded from the simulation (Lines 25-26).

**ALGORITHM 1:** Hybrid combustion simulation.

Transcriber: line numbers and indentation as printed; bold vectors are written plainly inside the code block (u, v_p, x_p are vectors).

```text
    Input: Initialized Eulerian grid with gas/solid flags; initialized SPH
           particle set.
    Output: Updated grid and particle state for each frame.
 1  forall Grid Cells do
 2      Initialize u ← u_0, T ← T_0, Y_i ← Y_i^0, ρ ← ρ_0;
 3  forall SPH Particles do
 4      Initialize v_p ← v_0, T_p ← T_0, m_p ← m_0;
 5  forall Frames do
        // — Lagrangian SPH Particle Step —
 6      Spawn new particles;
 7      forall SPH Particles do
 8          Compute density ρ_p, near-density ρ_p^near;
 9          Compute pressure p_p, near-pressure p_p^near;
10          Apply pressure and viscosity forces;
11          Update velocity and position v_p, x_p;
12          Enforce boundary conditions;
        // — Particle-to-Grid Transfer —
13      forall Grid Cells do
14          Sample ρ_l, T, u, from SPH using kernel w_ij;
        // — Eulerian Grid Simulation Step —
15      forall Grid Cells do
16          Advection: update u, T, Y_i with semi-Lagrangian scheme;
17          Buoyancy: apply Boussinesq force to u;
18          Combustion: compute v_c, J_c, update T, Y_i;
19          Diffusion: apply Laplacian on T, Y_i;
20          Thermal radiation: apply radiative thermal exchange;
21          Evaporation: update ρ, ρ_l, T, T_l, Y_i;
22          Solve pressure projection to enforce incompressibility;
        // — Grid-to-Particle Transfer —
23      forall SPH Particles do
24          Interpolate m_p and T_p from grid;
25      Remove particles with expired lifetime;
26      Remove particles without any remaining mass;
```

### 5.6 Visualization

We use ray marching augmented with global illumination (GI) techniques to render dynamic imagery of combustion phenomena. The ray marcher samples a set of 3D scalar fields including residual history for sooting, CO$_2$, fuel, oxygen, liquid density, temperature, and vapor, to visualize complex interactions of solids, liquids and gases. Meshes are rendered via BVH-accelerated ray tracing with support for physically based material for rendering (PBR). We handle transparent materials like water and glass through index-of-refraction tracking and refraction modeling using Snell's law, which allows us to render droplets and laminar flow. For scenes involving complex meshes (Figs. 1, 12, 18) we employ a Monte-Carlo path tracer, which marches through the volume grid before every bounce to integrate emission, absorption, and scattering into radiance [Pharr et al. 2016]. We then apply a denoising step to lower the necessary sample count for high quality renderings [Áfra 2025].

> Figure 7: A parameter space exploration of the water vapor and residual reaction coefficient. Starting from an entirely transparent flame (bottom left) we show the increase of residuals from left to right (horizontal axis) and the increase of vapor from bottom to top (vertical axis).

Flame coloration is determined through a combination of blackbody radiation and chemiluminescence modeling [Stewart and Johnson 2016]. The latter captures the spectral contributions of radicals like $CH*$ (blue-green light) and $C_2*$ (green light), which are inferred from local combustion activity and fuel richness, modulated by temperature. Residuals and soot, are tracked in a 3D texture and contribute both as light blockers at low temperatures and as emissive blackbody sources for hot gases. The soot map is used to darken the sooted surfaces, which enhances the realism of the material being exposed to combustion. For the figures rendered without path tracing, fire glow is instead further intensified by a bloom post-processing pass. Volumetric effects include colorless smoke that attenuates light and steam, which cast shadow rays to add depth and softness.

## 6 Results and Validation

To demonstrate the capabilities of our framework, we present results and validation obtained from various simulation experiments. <!-- p. 9 holds Figs. 8 to 10 only; the text continues on p. 10 --> Specifically, we show that our method reproduces key physical behaviors in combustion, evaporation, and fluid coupling scenarios.

<!-- p. 9 -->

> Figure 8: Water extinguishing experiments: we use two nozzle types to generate a laminar (a)-(h) and a spray (i-p) type of water stream to extinguish a flame and show the impact of aiming the water at the top of the flame (a-d, i-l) and at the bottom of the flame (e-h, m-p). For each fire-water interaction we show the average spatio-temporal temperatures (d, h, l, p) which show the overall effectiveness of extinquishing a flame during an experiment. A laminar water stream directed at the top of a flame does not impact the fire which leads to an overall high temperature. A spray stream directed at the bottom of the fire immediately stops the fire resulting in an overall lower temperature (p).

> Figure 9: A series showing the combustion of methane and oxygen in a nitrogen environment. Oxygen is visualized in blue and fuel in green. The injected rate of oxygen is the same for the whole experiment, while the amount of fuel is ramped up from a mass fraction of 0.1 to 1.0 mixed with nitrogen. Additionally, a heat source is set up below the two emitters. (a, b) show the combustion with a fuel lean mixture, (c, d) show the stoichiometric mixture of fuel and oxygen and (e, f) show a fuel rich mixture caused by an injection of pure fuel mixing with pure oxygen.

> Figure 10: Liquid fuel: As we simulate the thermodynamics between gases and liquids our framework enables simulating liquid fuel. A liquid fuel (ethanol) is emitted into a flame (a) and ignites (b). The burning liquid collides with obstacles in the scene (c) and continues to burn until all fuel is evaporated (d).

<!-- p. 10 -->

**Table 1.** Simulation parameters for Acetylene, Butane, Cyclopropane, Propane, Methane, and Ethylene combustion. Key quantities include fuel diffusivity $D_f$, Arrhenius factor $A$, reaction exponents $a$, $b$, heat of combustion $\Delta_c H^o$, stoichiometric coefficients $q_f$, $q_o$, $q_{CO_2}$, $q_{H_2O}$, and fuel-specific values $M_f$, $C_p$, $k$. Note that values for $D_f$, $\Delta_c H^o$, and $M_f$ are scaled by factors indicated in their respective column headers. See Sec. 4 for definitions.

| Gas | $D_f$ $(\cdot 10^{-5})$ | $A$ | $a$ | $b$ | $\Delta_c H^o$ $(\cdot 10^{6})$ | $q_f$ | $q_o$ | $q_{CO_2}$ | $q_{H_2O}$ | $M_f$ $(\cdot 10^{-2})$ | $C_p$ | $k$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acetylene ($C_2H_2$) | 1.46 | $6.50 \times 10^{12}$ | 0.50 | 1.25 | −1.0425 | −2.0 | −5.0 | 4.0 | 2.0 | 2.600 | 63 | 0.024 |
| Butane ($C_4H_{10}$) | 1.00 | $7.40 \times 10^{11}$ | 0.15 | 1.60 | −2.7105 | −2.0 | −13.0 | 8.0 | 10.0 | 5.812 | 1720 | 0.015 |
| Cyclopropane ($C_3H_6$) | 1.14 | $4.20 \times 10^{11}$ | −0.10 | 1.85 | −1.8765 | −2.0 | −9.0 | 6.0 | 6.0 | 4.200 | 114 | 0.015 |
| Propane ($C_3H_8$) | 1.14 | $8.60 \times 10^{11}$ | 0.10 | 1.65 | −2.0850 | −1.0 | −5.0 | 3.0 | 4.0 | 4.410 | 1670 | 0.016 |
| Methane ($CH_4$) | 2.10 | $8.30 \times 10^{5}$ | −0.30 | 1.30 | −0.8340 | −1.0 | −2.0 | 1.0 | 2.0 | 1.604 | 2230 | 0.034 |
| Ethylene ($C_2H_4$) | 1.63 | $2.00 \times 10^{12}$ | 0.10 | 1.65 | −1.2510 | −1.0 | −3.0 | 2.0 | 2.0 | 2.805 | 1550 | 0.019 |

### 6.1 Results

We organize our qualitative experiments into topical groups, each exploring different aspects of combustion dynamics, multiphase interactions, or system behavior under physically plausible conditions. The experiments range from fine-scale features such as flame structure to complex scene-level behavior. In the Appendix, we include a table with parameter value ranges for user-controlled parameters (Tab. 3) and fixed parameters calibrated with values found in literature (Tab. 4).

#### 6.1.1 Combustion and Flame Behavior.

To evaluate how our framework handles fuel-dependent combustion dynamics, we simulate stoichiometric mixtures of various hydrocarbons. Specifically, we compare flame development 800 ms after ignition for six common fuels: Acetylene, Butane, Cyclopropane, Propane, Methane, and Ethylene (Tab. 1 for parameter values). In Fig. 4 we show the resulting flame and smoke plume structures. Although all cases begin with identical ignition conditions and domain setup, we set different Arrhenius equation parameter values based on the fuel and chemical properties for all gases. Acetylene (Fig. 4a) produces a tall, narrow flame with strong vertical acceleration, indicating high combustion velocity and low molecular weight. In contrast, Butane (Fig. 4b) and Propane (Fig. 4d) form broader, more turbulent plumes, consistent with their heavier hydrocarbon structure. Cyclopropane (Fig. 4c) exhibits structured vortex rings and moderate turbulence. Methane (Fig. 4e) yields a clean and vertically stable flame. Ethylene (Fig. 4f) generates more complex turbulence and brighter flame fronts due to its increased chemical reactivity.

In Fig. 5 we show four different types of small flame combustion, illustrating the range of flame appearance our method can simulate. We calibrate combustion parameter values such as buoyancy, combustion heat efficiency $\varphi$, radiation coefficient $\epsilon$, residual produced $q_{residual}$, and vorticity confinement strength for each experiment. Fig. 5a shows a clean combustion, typical of a Bunsen burner with premixed fuel. In Fig. 5b a flame with minor traces of glowing residuals is shown. A near-complete combustion occurs at the base while water vapor and glowing residuals appear in the upper regions (Fig. 5c). A more turbulent flame is shown in Fig. 5d, which is characterized by substantial amounts of water vapor and residuals.

> Figure 11: Simulation showing fire suppression at a window using principles of fluid dynamics. Flames and hot gases vent from the window (a). A water stream is directed outward from the window (b), creating a high-velocity flow that induces a low-pressure zone outside (Bernoulli Principle). This draws heat, smoke, and flames out of the room while limiting air entrainment into the structure. The window opening acts as a constriction (Venturi Effect), accelerating the outward flow and enhancing the removal of hot gases (c). The fire is effectively suppressed as interior temperatures drop and oxygen supply is reduced (d).

To further investigate the expressiveness of our model, we conduct a parameter space exploration by varying the water vapor (vertical axis) and the residual reaction coefficients (horizontal axis) in Fig. 7. The bottom-left corner corresponds to the minimal configuration where both coefficients are set to zero. Increasing the water vapor coefficient vertically (bottom to top) introduces progressively more visible steam and white combustion products. Increasing the residual coefficient horizontally (left to right) adds dark soot and incomplete combustion effects. While these coefficients do not influence the combustion process itself, they modulate the final appearance of the combustion products. This parameter space exploration illustrates our framework's capability in smoothly capturing the visual transition from clean, complete combustion to sooty, oxygen-limited burns. Note that the scale of the residual coefficient is logarithmic (0, 0.01, 0.1, 1, 10), whereas the water vapor coefficient is linear (0, 2.5, 5, 7.5, 10).

To further analyze combustion dynamics under varying fuel–oxygen ratios, we simulate the combustion of methane and oxygen in a nitrogen environment while varying the fuel mass fraction. In Fig. 9 we visualize this process using color-coded species: oxygen in blue, fuel in green, and flame intensity in yellow-orange. Across three scenarios, the oxygen flow remains constant, while the fuel concentration increases from a lean to a rich mixture. In the first two frames (Fig. 9a, b), combustion occurs under fuel-lean conditions, producing small and intermittent flames. In Fig. 9c and Fig. 9d we show the equal mixture of fuel and oxygen, resulting in strong, vertically stable flames and optimal combustion efficiency. Finally, <!-- p. 11 --> in Fig. 9e and Fig. 9f, the excess fuel leads to a rich mixture, where incomplete combustion reduces flame visibility and promotes soot formation.

> Figure 12: Two frames of a timeseries showing the extinction of a complex fire. Three vehicles are vigorously burning (a). The fire is then suppressed by a high-pressure water jet (b), producing dense clouds of smoke and vapor. Our advanced combustion model supports multi-species thermodynamics and accurately simulates flame extinction dynamics.

> Figure 13: Ablation study of the dispersion and displacement terms of our model. Enabling both terms causes water to evaporate while the water generates a drag on the fire and vapor (a). Disabling the displacement term leads to a vapor cloud that is not affected by the water (b). Only enabling the displacement term shows that the fire is dragged in the direction of the water (c). Disabling both terms leaves only the diffuse temperature exchange, which is too slow to have an effect on the fast interaction of flames and water.

#### 6.1.2 Liquid–Fire Interaction and Evaporation.

We further analyze the interaction between water jets and flame behavior by comparing laminar and spray nozzles, each aimed either at the top or bottom of the flame. As shown in Fig. 8a–d, a laminar stream directed toward the top of the flame has a limited extinguishing effect. The flame remains largely intact (Fig. 8a–c), and the spatio-temporal temperature map (Fig. 8d) shows only a localized temperature drop with high overall residual heat. In contrast, when the same laminar stream is aimed at the base of the flame (Fig. 8e–h), it disrupts the combustion zone more effectively, with Fig. 8h indicating lower core temperatures and a cooling effect extending upward.

The spray nozzle produces a finer, more distributed stream that further improves the extinguishing. When directed at the top of the flame (Fig. 8i–l), it produces dense vapor and drag, but still fails to fully suppress combustion (Fig. 8l). Finally, the spray directed at the bottom of the flame (Fig. 8m–p) results in complete flame suppression and the thermal field in Fig. 8p shows a significant reduction in overall temperature.

To analyze evaporation-driven combustion, we simulate a scenario in which a stream of liquid fuel is injected into an existing flame. As shown in Fig. 10a, the fuel begins to evaporate upon contact with surrounding hot gases, forming a vapor-rich region near the jet. This vapor mixes with ambient oxygen and ignites, extending the existing flame in direction of the jet (Fig. 10b). In Fig. 10c, the ignited liquid fuel continues to collide with a wall.

#### 6.1.3 Physics Probes and Diagnostics.

To examine how our simulations can assist in fire suppression, we simulate the use of a high-velocity stream directed at a burning window opening. As shown in Fig. 11a, flames and hot gases naturally vent outward from the structure. In Fig. 11b, a water jet is introduced in front of the window, generating a fast-moving outflow that lowers the local pressure outside the window based on the Bernoulli principle. This pressure drop draws flames and hot gases out of the room. The window itself acts as a geometric constriction, and as illustrated in Fig. 11c, this narrowing accelerates the exiting flow in accordance with the Venturi effect. As a result, smoke and thermal energy are more efficiently expelled. Finally, Fig. 11d shows that interior flames weaken significantly as oxygen inflow is reduced and overall temperatures drop. This diagnostic case demonstrates how directed airflow can passively assist fire mitigation.

To study combustion under limited oxygen conditions, we simulate a small flame undergoing progressive oxygen starvation. Fig. 6 shows visualizations at two distinct stages: early (top row, Fig. 6a–d) and late (bottom row, Fig. 6e–h). RGB renderings in Fig. 6a and 6e illustrate a clear decrease in flame luminosity and size over time. The corresponding temperature fields (Fig. 6b, f) confirm this drop, with lower peak temperatures and reduced vertical heat transport. Fuel and oxygen distributions (green and blue, respectively) in Fig. 6c and 6g show that the available oxygen diminishes significantly in the later stage. Finally, the CO$_2$ field in Fig. 6d and 6h confirms incomplete combustion and suppressed chemical reaction. This diagnostic setup validates our model's ability to simulate extinction phenomena resulting from local oxygen depletion.

<!-- p. 12 -->

> Figure 14: Comparison of our simulation and FDS [McGrattan et al. 2013]. We use the experiment setup described in NBSIR 79-1910 [McCaffrey 1979] to measure the average temperature of all sensors over a period of 30 seconds for flames with an energy of 14 kW (a) and 57 kW (d). We measure the mean and standard deviation of temperatures measured at each sensor for flames of 14 kW and 57 kW for FDS (b, e) and for our simulation (c, f). We also compared FDS with our simulation on a sprinkler setup for which we measure the average temperature (g) as well as the mean and standard deviation for FDS (h) and our simulation (i). The results indicate that our framework is able to closely match the simulation results of FDS for both experiments.

> Figure 15: Renderings of the scene setup for the enclosure fire experiment described in NBSIR 79-1910 [McCaffrey 1979] (a) and a sprinkler experiment (a-d) that we use to compare our model to FDS.

#### 6.1.4 Emergency Response Devices.

Fig. 18 illustrates a fire emergency scenario in a residential kitchen. A stovetop ignition initiates a spreading flame Fig. 18a, which triggers a smoke detector and activates an overhead sprinkler system Fig. 18b. Our simulation captures the subsequent interaction between the water spray and the hot combustion gases, showing both flame suppression and vapor generation. This example highlights the potential of physics-based fire simulation as an in silico test bed for evaluating real-world fire response strategies.

#### 6.1.5 Multiphase Scene-Level Experiments.

In Fig. 12 we show two frames from a time series illustrating the extinction of a complex fire scenario. In the initial frame (a), three vehicles are burning. The combustion leads to pronounced flame structures and smoke. In the subsequent frame (b), the fire is actively being suppressed by a high-pressure water jet, resulting in dense plumes of smoke and vapor.

With our multi-species thermodynamics model it is possible to realistically simulate the interaction between fire and extinguishing agents, capturing the nuanced dynamics of flame extinction and post-combustion behavior.

In Fig. 16 we show the annealing process of a metal rod. Initially, a flame is applied (Fig. 16a) and gradually heats the rod (Fig. 16b, c) until it begins to glow visibly. Once the flame is removed (Fig. 16d), the rod continues to emit a glow as it slowly cools down, eventually returning to its original state (Fig. 16e, f).

### 6.2 Validation

We evaluated our combustion model qualitatively and quantitatively, including comparisons of our simulation results with an established solver for combustion, visual comparisons to previous approaches, and ablations studies of different components of our framework. Runtime performance measures are shown in the Appendix (Tab. 2).

#### 6.2.1 Comparison to McCaffrey and FDS.

We conducted a series of comparative experiments against both real measurements of combustion experiments as well as established combustion solvers (FDS [McGrattan et al. 2013]). The first experiment replicates the enclosure fire test described in NBSIR 79-1910 [McCaffrey 1979] conducted in a 1.5m × 1.5m × 3.9m sized compartment (see Fig. 15a). A natural gas burner with a heat release rate of approximately 14.4 to 57.5 kW was used, producing a steady diffusion flame. The flame was placed on a pedestal of 0.72 m height. To capture the vertical <!-- p. 13 --> temperature profile and development of the hot gas layer 21 temperature sensors were installed in a vertical line at heights ranging from 0.05m to 3.0m from the pedestal. We use this data for validating the thermal and fluid dynamic behavior of our solver. In Fig. 14 we show a comparative analysis of FDS and our framework on this setup. Specifically, we analyze the average temperature recorded by all sensors over a continuous 30-second interval for flame heat release rates of 14 kW and 57 kW, as illustrated in Fig. 14a and Fig. 14d, respectively. Blue lines indicate our temperature evolution which closely conforms to the red lines resulting from the equivalent FDS simulation. Beyond average temperatures, we further assess the spatial temperature distribution by computing the mean and standard deviation of the temperature at each individual sensor location. This analysis is conducted separately for the 14 kW (Fig. 14a-c) and 57 kW (Fig. 14d-f) flame scenarios for FDS and our model. These results reveal a strong correlation between our simulation outputs and those generated by FDS, thereby indicating that our framework is able to replicate complex thermal behaviors observed in standard fire scenarios.

In Fig. 17, we show the centerline temperature rise ($\Delta T$) as a function of height ($z$) above the burner for various heat release rates ($Q$), comparing results from our thermodynamics model with those from FDS. The plot illustrates how temperature varies along the vertical axis of the flame for different fire sizes (14.4 kW, 33 kW, 57.5 kW), highlighting the thermal behavior within the plume. The results demonstrate that our model not only reproduces the experimentally observed temperature rise (NBSIR 79-1910 [McCaffrey 1979]) with high fidelity but also shows strong agreement with the FDS simulation outputs.

In a second experiment, we simulate a sprinkler activation scenario, comparing the fire suppression dynamics, spray interaction, and temperature decay between our solver and FDS. We use the same 1.5m × 1.5m × 3.9m sized compartment with a pedestal (0.72m height) and a 33 kW flame. After 15 seconds a water sprinkler (with 300 liters/min) is started, which causes the extinction of the flame. A visualization of our setup is shown in Fig. 15a-d. The average temperatures of FDS and our simulation as well as the mean and standard deviations of individual sensors are shown in Fig. 14g-i. These results show that our framework is able to extinguish fire with similar temperature characteristics as FDS.

#### 6.2.2 Qualitative Validation.

To assess the visual fidelity of our method, we performed a qualitative comparison against the flame results shown in Nielsen et al. [2022]. Fig. 19 illustrates this comparison across both large and small flame regimes. In Fig. 19a, we show the result from Nielsen et al. [2022], and in Fig. 19b, our framework produces a similarly detailed large-scale flame, including fine turbulent structures. This demonstrates that our model supports high-resolution flames without requiring an explicit signed distance field to track the flame front. However, in the small-flame regime, Fig. 19c and Fig. 19d show real flames with sharp, thin flame fronts such as those from Bunsen burners, which are well-represented in Nielsen et al.'s [2022] method. By contrast, Fig. 19e shows that our approach is less suited for this regime due to its lack of flame front modeling, resulting in a more diffuse appearance.

> Figure 16: Annealing of a metal rod: a flame is initiated (a) and slowly increases the temperature of a metal rod (b) until it starts to glow (c). After the flame is turned off (d), the rod remains glowing (e) until it is entirely cooled off (f).

> Figure 17: We compare FDS to our thermodynamics model by measuring the centerline temperature rise (ΔT) versus height (z) for various heat release rates (Q). This figure illustrates how the temperature along the flame's centerline changes with height above the burner for different fire sizes. The results indicate that our framework closely matches the measured rise in temperature as well as the simulated results of FDS.

Transcriber: Fig. 17 is a log-log plot of ΔT (°C) against $z/Q^{2/5}$ (m•W$^{-2/5}$), with series 14.4 kW, 33.0 kW, 57.5 kW (FDS) and 14.4 kW, 33.0 kW, 57.5 kW (Ours), and a dashed reference line labelled $(z/Q^{2/5})^{\eta}$ with regions marked "η = 0" (Flame), "η = -1" (Intermittent) and "η = -5/3" (Plume).

#### 6.2.3 Dispersion Ablation.

To evaluate the significance of dispersion and displacement forces in water–fire interaction, we conduct an ablation study comparing four configurations. As shown in Fig. 13a, enabling both the dispersion and displacement terms produces the expected interaction: water evaporates rapidly, forming vapor, and simultaneously pushes back the flame through drag. In Fig. 13b, we disable the displacement term while keeping dispersion active. This leads to a vapor cloud forming correctly, but with no coupling to the flame, which continues burning undisturbed. In Fig. 13c, we disable dispersion but retain displacement. The flame is physically deflected by the incoming spray, yet vapor formation is nearly absent, indicating that momentum exchange alone is insufficient for suppression. Finally, Fig. 13d disables both terms, leaving only passive heat transfer. In this case, the flame remains stable, with neither visible vapor nor dynamic suppression. This comparison illustrates that both dispersion and displacement are essential for simulating fast and realistic flame response and visible water-driven disruption.

<!-- p. 14 -->

> Figure 18: Emergency response devices: A stovetop in a kitchen is catching fire (a). After a while a fire detector detects the smoke and starts a water sprinkler (b) to extinguish the fire (c).

> Figure 19: Comparison to Nielsen et al. [2022]: Our method enables generating larger flames (b) with a similar degree of complexity (a). As we do not explicitly compute the flame front with a signed distance field, our method (e) does not allow simulating smaller flames (e.g. Bunsen burner) with the same degree of detail as shown for real flames (c), or (d) Nielsen et al. [2022].

## 7 Discussion and Limitations

We present a physically grounded framework for combustion simulation with support for chemical species, stoichiometric reactions, and residual formation. Conceptually, our approach is related to other species-transport fire models [Merci and Beji 2016; Nielsen et al. 2022]. However, our method enables multi-phase interactions between gas, liquid, and static solid materials, with droplet-based evaporation and species-dependent rendering that reflect underlying combustion chemistry. Compared to prior methods, our model advances the state-of-the-art by unifying chemically driven combustion with multi-phase simulation and species-aware visualization in a single framework. Unlike artistic fire models, our approach explicitly tracks species mass fractions and supports stoichiometric control over reaction products. This enables simulations that differentiate between complete and incomplete combustion based on molecular composition. The inclusion of droplet-based evaporation within a species-coupled solver allows realistic interactions between flames and water – capturing both energy exchange and visible effects such as vaporization and suppression. This allows to test fire suppression strategies in the safe confines of a computer simulation. Furthermore, the physically motivated rendering pipeline ties temperature and species content to appearance, providing interpretable visual outputs rooted in combustion chemistry. While grounded in existing fluid simulation techniques, the integration of these components enables new types of fire–fluid interactions with a degree of physical plausibility not previously demonstrated in real-time or artist-controllable systems.

While effective for a range of scenarios, the framework has several limitations. Solids are treated as static obstacles without thermal deformation or melting. The flame front is not explicitly modeled, leading to smoother transitions and limited accuracy in laminar or small-scale flames. Furthermore, we do not not explicitly simulate radiation in a detailed manner while heat transfer in liquids is subject to diffusion artifacts due to temperature sampling. Only one fuel and liquid type is supported per simulation, with globally uniform reaction parameters; localized variation in combustion completeness is currently not modeled. Additionally, all species are advected identically, ignoring differential effects such as CO$_2$ stratification.

## 8 Conclusion and Future Work

We have presented a unified, hybrid framework for simulating multiphase combustion and fire suppression, with a combined Eulerian and Lagrangian representation to capture complex interactions across solid, liquid, and gas phases. By explicitly modeling key combustion species and phase transitions as well as by modeling stoichiometry-aware heat release, our method supports simulating diverse scenarios, including open flame propagation, water-based flame extinction, and fire dynamics responsive to the environment. Moreover, our framework bridges the gap between high-fidelity combustion models and the efficiency required for interactive graphics.

With a novel hybrid multi-species thermodynamics model, we see several avenues for future work. First, our current model simplifies certain chemical formulations and assumes predefined reaction parameters; incorporating adaptive or learned chemical kinetics could improve realism for a broader class of fuels. Second, we would like to evaluate the different hydrocarbon fuels in more detail. A comparison with real shapes and combustion behaviors would enable us to calibrate our approach even further. Third, enhanced modeling of radiative heat transfer and soot formation would further increase visual fidelity, especially in dense smoke scenarios. Another interesting direction would be to couple our solver with methods for describing geometric deformations of combustion. Finally, integrating this system into real-time engines or VR environments, where user interaction and sensory feedback (e.g., heat, sound) play a crucial role, remains an exciting frontier. We believe our work lays a strong foundation for next-generation fire simulation tools that are both physically grounded and artistically expressive, applicable in visual media, safety training, and scientific visualization.

<!-- p. 15 -->

## Acknowledgments

We thank the reviewers for their valuable comments and suggestions. This work is supported by ERC grant 101170158 - WildfireTwins.

## References

- A. T. Áfra. 2025. Intel® Open Image Denoise. https://www.openimagedenoise.org.
- G. Aguilera and J. Johansson. 2019. Avengers: Endgame, a new approach for combustion simulations. In ACM SIGGRAPH 2019 Talks (SIGGRAPH '19). ACM, Article 39.
- A. Bakhshaii and E.A. Johnson. 2019. A review of a new generation of wildfire–atmosphere modeling. Canadian Journal of Forest Research 49, 6 (2019), 565–574.
- R. Bridson. 2015. Fluid simulation for computer graphics. AK Peters/CRC Press.
- R. Bridson and M. Müller-Fischer. 2007. Fluid simulation. In ACM SIGGRAPH 2007 Courses. ACM, 1–81.
- G. Cao, J. Railio, E. F. Curd, M. Hyttinen, P. Liu, H. M. Mathisen, D. Belkowska-Woloczko, M. Justo-Alonso, P. White, C. Coxon, and T. A. Wenaas. 2020. Chapter 9 - Air-handling processes. In Industrial Ventilation Design Guidebook (Second Edition). Academic Press, 417–496.
- N. Chiba, K. Muraoka, H. Takahashi, and M. Miura. 1994. Two-dimensional visual simulation of flames, smoke and the spread of fire. The Journal of Visualization and Computer Animation 5, 1 (1994), 37–53.
- S. Clavet, P. Beaudoin, and P. Poulin. 2005. Particle-based viscoelastic fluid simulation. In ACM SIGGRAPH/Eurographics SCA. 219–228.
- J L. Coen. 2013. Modeling Wildland Fires :. National Center for Atmospheric Research (NCAR),, Boulder, CO :. 2013-02-04.
- J. L. Coen, M. Cameron, J. Michalakes, E. G. Patton, P. J. Riggan, and K. M. Yedinak. 2013. WRF-Fire: Coupled Weather–Wildland Fire Modeling with the Weather Research and Forecasting Model. Journal of Applied Meteorology and Climatology 52, 1 (2013), 16 – 38.
- E. Coumans and Y. Bai. 2016–2021. PyBullet, a Python module for physics simulation for games, robotics and machine learning. http://pybullet.org.
- D. Demidov. 2019. AMGCL: An efficient, flexible, and extensible algebraic multigrid implementation. Lobachevskii Journal of Mathematics 40 (2019), 535–546.
- R. Fedkiw, J. Stam, and H. W. Jensen. 2001. Visual simulation of smoke. In Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '01). Association for Computing Machinery, New York, NY, USA, 15–22.
- B. Feldman, J. M. Cohen, and J. F. Hughes. 2003. Animating Suspended Particle Explosions. ACM Transactions on Graphics (TOG) 22, 3 (2003), 708–715.
- T. Hädrich, D. T. Banuti, W. Pałubicki, S. Pirk, and D. L Michels. 2021. Fire in paradise: Mesoscale simulation of wildfires. ACM Trans. on Graph. (TOG) 40, 4 (2021), 1–15.
- J.-M. Hong, T. Shinar, and R. Fedkiw. 2007. Wrinkled flames and cellular patterns. ACM Trans. Graph. 26, 3 (July 2007), 47–es.
- Y. Hong, D. Zhu, X. Qiu, and Z. Wang. 2010. Geometry-based control of fire simulation. The Visual Computer 26, 9 (01 Sep 2010), 1217–1228.
- C. Horvath and W. Geiger. 2009. Directable, High-resolution Simulation of Fire on the GPU. ACM Trans. Graph. 28, 3, Article 41 (2009), 8 pages.
- I. Ihm, B. Kang, and D. Cha. 2004. Animation of reactive gaseous fluids through chemical kinetics. In ACM SIGGRAPH/Eurographics SCA. Eurographics Association, 203–212.
- J. Katan and L. Perez. 2021. ABWiSE v1.0: toward an agent-based approach to simulating wildfire spread. Natural Hazards and Earth System Sciences 21, 10 (2021), 3141–3160.
- T. Kim, E. Hong, J. Im, D. Yang, Y. Kim, and C.-H. Kim. 2017. Visual simulation of fire-flakes synchronized with flame. The Visual Computer 33 (06 2017).
- T. Kim, N. Thürey, D. James, and M. Gross. 2008. Wavelet Turbulence for Fluid Simulation. ACM Trans. Graph. 27, 3, Article 50 (2008), 6 pages.
- M. Kinateder, E. Ronchi, D. Nilsson, M. Kobes, M. Müller, P. Pauli, and A. Mühlberger. 2014. Virtual reality for fire evacuation research. In 2014 Federated Conference on Computer Science and Information Systems. 313–321.
- A. Kokosza, H. Wrede, D. Gonzalez E., M. Makowski, D. Liu, D. L. Michels, S. Pirk, and W. Palubicki. 2024. Scintilla: Simulating Combustible Vegetation for Wildfires. ACM Trans. Graph. 43, 4, Article 70 (July 2024), 21 pages.
- K. A. Kroos and M. C. Potter. 2014. Thermodynamics for Engineers. Cengage Learning.
- N. Kwatra, I. Essa, A. Schödl, N. Thuerey, C. Wojtan, and G. Turk. 2009. A Method for Avoiding the Acoustic CFL Condition for Compressible Flow. ACM Transactions on Graphics (TOG) 28, 5 (2009), 1–8.
- A. Lamorlette and N. Foster. 2002. Structural modeling of flames for a production environment. In Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '02). ACM, New York, NY, USA, 729–735.
- C. Lapointe, N. Wimer, J. Glusman, A. Makowiecki, J. Daily, G. Rieker, and P. Hamlington. 2020. Efficient simulation of turbulent diffusion flames in OpenFOAM using adaptive mesh refinement. Fire Safety Journal 111 (01 2020), 102934.
- D. R. Lide. 1995. CRC handbook of chemistry and physics: a ready-reference book of chemical and physical data. CRC press.
- R.R. Linn, S.L. Goodrick, S. Brambilla, M.J. Brown, R.S. Middleton, J.J. O'Brien, and J.K. Hiers. 2020. QUIC-fire: A fast-running simulation tool for prescribed fire planning. Environmental Modelling Software 125 (2020), 104616.
- R. Linn, J. Reisner, J. Colman, and J. Winterkamp. 2002. Studying wildfire behavior using FIRETEC. International Journal of Wildland Fire 11 (11 2002), 233–246.
- P. J. Linstrom and W. G. Mallard. 2025. NIST Chemistry WebBook. NIST Standard Reference Database Number 69. Retrieved May 23, 2025.
- D. Liu, J. Klein, F. Rist, W. Pałubicki, S. Pirk, and D. L. Michels. 2025. FlameForge: Combustion of Generalized Wooden Structures. ACM SIGGRAPH / Eurographics Symposium on Computer Animation (2025).
- S. Liu, T. An, Z. Gong, and I. Hagiwara. 2012. Physically based simulation of solid objects' burning. Springer-Verlag, Berlin, Heidelberg, 110–120.
- S. Liu, Q. Liu, T. An, J. Sun, and Q. Peng. 2009. Physically based simulation of thin-shell objects' burning. Vis. Comput. 25, 5–7 (April 2009), 687–696.
- F. Losasso, G. Irving, E. Guendelman, and R. Fedkiw. 2006a. Melting and burning solids into liquids and gases. IEEE Transactions on Visualization and Computer Graphics 12, 3 (2006), 343–352.
- F. Losasso, T. Shinar, A. Selle, and R. Fedkiw. 2006b. Multiple interacting liquids. ACM Trans. Graph. 25, 3 (July 2006), 812–819.
- B J. McCaffrey. 1979. Purely Buoyant Diffusion Flames: Some Experimental Results. Final Report NBSIR 79-1910. National Bureau of Standards, Washington, D.C. https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nbsir79-1910.pdf
- K. McGrattan, R. McDermott, C. Weinschenk, and G. Forney. 2013. Fire Dynamics Simulator Users Guide, Sixth Edition.
- A. D. McNaught and A. Wilkinson. 1997. IUPAC Gold Book: Compendium of Chemical Terminology. Blackwell Scientific Publications. https://goldbook.iupac.org
- Z. Melek and J. Keyser. 2002. Interactive simulation of fire. Pacific Graphics (2002), 431–432.
- W. Mell, M. Jenkins, J. Gould, and P. Cheney. 2007. A Physics Based Approach to Modeling Grassland Fires. International Journal of Wildland Fire (2007).
- B. Merci and T. Beji. 2016. Modeling and simulation of transport phenomena in fire engineering. Fire Safety Journal 80 (2016), 12–22.
- B. Merci and T. Beji. 2022. Fluid mechanics aspects of fire and smoke dynamics in enclosures. CRC press.
- V. Mihalef, B. Unlusu, D. Metaxas, M. Sussman, and M. Hussaini. 2006. Physics based boiling simulation. 317–324.
- K. Museth. 2013. VDB: High-resolution sparse volumes with dynamic topology. ACM transactions on graphics (TOG) 32, 3 (2013), 1–22.
- D. Quang Nguyen, R. Fedkiw, and H. W. Jensen. 2002. Physically Based Modeling and Animation of Fire. ACM Trans. Graph. 21, 3 (2002), 721–728.
- D. Q. Nguyen, R. P. Fedkiw, and M. Kang. 2001. A Boundary Condition Capturing Method for Incompressible Flame Discontinuities. J. Comput. Phys. 172, 1 (2001), 71–98.
- M. B. Nielsen, M. Bojsen-Hansen, K. Stamatelos, and R. Bridson. 2022. Physics-based combustion simulation. ACM Transactions on Graphics (TOG) 41, 5 (2022), 1–21.
- M. B. Nielsen, K. Stamatelos, M. Bojsen-Hansen, and R. Bridson. 2019. Physics-based combustion simulation in bifrost. In ACM SIGGRAPH 2019 Talks. 1–2.
- Z. Pan and D. Manocha. 2017. Efficient Solver for Spacetime Control of Smoke. ACM Trans. Graph. 36, 5, Article 162 (July 2017), 13 pages.
- V. Pegoraro and S. G. Parker. 2006. Physically-based Realistic Fire Rendering. EG Nat. Phenom. (2006), 51–59.
- N. Peters. 2000. Turbulent Combustion. Cambridge University Press.
- M. Pharr, W. Jakob, and G. Humphreys. 2016. Physically Based Rendering: From Theory to Implementation (3rd ed.). Morgan Kaufmann Publishers Inc.
- S. Pirk, M. Jarząbek, T. Hädrich, D. L. Michels, and W. Palubicki. 2017. Interactive wood combustion for botanical tree models. ACM Trans. Graph. (TOG) 36, 6 (2017), 1–12.
- A. H. Rabbani, J.-P. Guertin, D. Rioux-Lavoie, A. Schoentgen, K. Tong, A. Sirois-Vigneux, and D. Nowrouzezahrai. 2022. Compact Poisson Filters for Fast Fluid Simulation. In ACM SIGGRAPH 2022. ACM, New York, NY, USA, Article 35, 9 pages.
- K. Schmidt-Rohr. 2011. The heat of combustion—getting it right. Journal of Chemical Education 88, 8 (2011), 1099–1104.
- K. Schmidt-Rohr. 2015. Why combustions are always exothermic, yielding about 418 kJ per mole of O2. Journal of Chemical Education 92, 12 (2015), 2094–2099.
- J. Stam. 1999. Stable Fluids. Proc. of ACM SIGGRAPH (1999), 121–128.
- J. Stam and E. Fiume. 1995. Depicting fire and other gaseous phenomena using diffusion processes. In Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '95). ACM, New York, NY, USA, 129–136.
- S.M. Stewart and R.B. Johnson. 2016. Blackbody Radiation: Computational Aids and Numerical Methods. Taylor & Francis.
- A. Stomakhin, C. Schroeder, C. Jiang, L. Chai, J. Teran, and A. Selle. 2014. Augmented MPM for phase-change and varied materials. ACM Trans. Graph. 33, 4, Article 138 (July 2014), 11 pages.
- H. Versteeg and W. Malalasekera. 2007. An Introduction to Computational Fluid Dynamics e-book. Pearson Education. https://books.google.de/books?id=dlC8MgEACAAJ
- C. K. Westbrook and F. L. Dryer. 1981. Simplified reaction mechanisms for the oxidation of hydrocarbon fuels in flames. Combustion science and technology 27, 1-2 (1981), 31–43.
- Y. Zhao, X. Wei, Z. Fan, A. Kaufman, and H. Qin. 2003. Voxels on Fire. In Proceedings of the 14th IEEE Visualization 2003 (VIS'03) (VIS '03). IEEE Computer Society, USA, 36.

<!-- p. 16 -->

## A Appendix

### A.1 List of Symbols

#### A.1.1 Model Parameters.

| Symbol | Description |
|---|---|
| $\boldsymbol{u}$ | Velocity field (m s$^{-1}$) |
| $\nu$ | Kinematic viscosity (m$^2$ s$^{-1}$) |
| $\rho$ | Fluid's local density (kg m$^{-3}$) |
| $p$ | Pressure (Pa) |
| $T$ | Temperature (K) |
| $T_{\mathrm{amb}}$ | Reference ambient temperature (K) |
| $\boldsymbol{g}$ | Gravitational acceleration vector (m s$^{-2}$) |
| $k$ | Thermal conductivity (W m$^{-1}$ K$^{-1}$) |
| $C_p$ | Specific heat at constant pressure (gas) or liquid specific heat capacity (liquid) (J kg$^{-1}$ K$^{-1}$) |
| $\epsilon$ | Emissivity (dimensionless) |
| $\sigma$ | Stefan-Boltzmann constant ($5.67 \times 10^{-8}$ W m$^{-2}$ K$^{-4}$) |
| $J_c$ | Volumetric heat release rate from combustion (W m$^{-3}$) |
| $Y_i$ | Mass fraction of species $i$ (dimensionless) |
| $D_i$ | Diffusion coefficient of species $i$ (m$^2$ s$^{-1}$) |
| $S_i$ | Source term from phase change for species $i$ (s$^{-1}$) |
| $\Delta_c H^o$ | Standard heat of combustion (J mol$^{-1}$ or J kg$^{-1}$) |
| $c, h, o, n$ | Number of carbon, hydrogen, oxygen, nitrogen atoms in fuel molecule (dimensionless, for empirical $\Delta_c H^o$) |
| $v_c$ | Volumetric reaction rate (typically mol m$^{-3}$ s$^{-1}$) |
| $A$ | Arrhenius pre-exponential factor (units vary based on reaction order and concentration units) |
| $n$ | Temperature exponent in Arrhenius equation (dimensionless) |
| $E_a$ | Activation energy (J mol$^{-1}$) |
| $R$ | Universal gas constant (8.314 J mol$^{-1}$ K$^{-1}$) |
| $c_f, c_o$ | Molar concentrations of fuel and oxidizer, respectively (mol m$^{-3}$) |
| $a, b$ | Empirical reaction order exponents for fuel and oxidizer, respectively (dimensionless) |
| $\varphi$ | Combustion heat efficiency (dimensionless) |
| $q_i$ | Stoichiometric coefficient for species $i$ (dimensionless) |
| $M_i$ | Molar mass of species $i$ (kg mol$^{-1}$) |
| $p_{\mathrm{amb}}$ | Ambient pressure (Pa) (used in ideal gas law context) |
| $\rho_l$ | Liquid density (kg m$^{-3}$) |
| $T_l$ | Current liquid temperature (K) |
| $T_B$ | Boiling temperature (K) |
| $\Delta_v H^0$ | Latent heat of vaporization (J kg$^{-1}$) |
| $T_{dc}$ | Intermediate term for droplet conduction calculation (K m) |
| $d_d$ | Droplet diameter (m) |
| $A_d$ | Droplet surface area (m$^2$) |
| $\rho_d$ | Intrinsic density of the liquid making up a droplet (kg m$^{-3}$) |
| $\Gamma_i$ | Evaporation coefficient for species $i$ (m$^3$ kg$^{-1}$ s$^{-1}$) |
| $\rho_e$ | Evaporated liquid density (mass of liquid evaporated per unit volume of gas) (kg m$^{-3}$) |
| $E_l$ | Total available energy in a liquid cell (J) |
| $V_c$ | Volume of a grid cell (m$^3$) |
| $E_d$ | Energy required to evaporate one droplet (J) |
| $V_d$ | Volume of one droplet (m$^3$) |
| $n_d$ | Total number of droplets in a cell (dimensionless) |
| $n_e$ | Number of droplets that can be evaporated from a cell (dimensionless) |
| $\boldsymbol{u}_l$ | Liquid velocity (sampled to grid) (m s$^{-1}$) |

#### A.1.2 Algorithmic and SPH Parameters.

| Symbol | Description |
|---|---|
| $\boldsymbol{u}_p$ | SPH particle velocity (m s$^{-1}$) |
| $T_p$ | SPH particle temperature (K) |
| $m_p$ | SPH particle mass (kg) |
| $h_s$ | SPH smoothing radius (for particle-particle interactions) (m) |
| $r_{s_{ij}}$ | Euclidean distance between SPH particles $i$ and $j$ (m) |
| $\rho_{s_i}$ | SPH particle local pseudo-density (sum of kernel values, dimensionless as per Eq. 12) |
| $\rho_{s_i}^{\mathrm{near}}$ | SPH particle near pseudo-density (sum of kernel values, dimensionless as per Eq. 13) |
| $p_{s_i}$ | SPH particle pseudo-pressure (specific energy form) (m$^2$ s$^{-2}$) |
| $p_{s_i}^{\mathrm{near}}$ | SPH particle near pseudo-pressure (specific energy form) (m$^2$ s$^{-2}$) |
| $k_s$ | SPH stiffness coefficient (e.g., related to speed of sound squared) (m$^2$ s$^{-2}$) |
| $k_s^{\mathrm{near}}$ | SPH near-stiffness coefficient (m$^2$ s$^{-2}$) |
| $\rho_{s_0}$ | SPH reference pseudo-density number (dimensionless) |
| $\boldsymbol{a}_{s_i}^{\mathrm{press}}$ | Acceleration on SPH particle $i$ due to pseudo-pressure forces (m s$^{-2}$) (assuming $F_{s_i}^{\mathrm{press}}$ in text is $m_i \boldsymbol{a}_{s_i}^{\mathrm{press}}$) |
| $\Delta\boldsymbol{u}_{s_{ij}}^{\mathrm{vis}}$ | Pairwise radial velocity difference for SPH viscosity (m s$^{-1}$) |
| $\hat{r}_{s_{ij}}$ | Unit vector from SPH particle $i$ to $j$ (dimensionless) |
| $\boldsymbol{a}_{s_i}^{\mathrm{vis}}$ | Viscous acceleration on SPH particle $i$ (m s$^{-2}$) |
| $\alpha$ | SPH linear viscosity damping coefficient (s$^{-1}$) |
| $\beta$ | SPH quadratic viscosity damping coefficient (m$^{-1}$) |
| $h$ | Smoothing radius for particle-grid sampling (m) |
| $r_{ij}$ | Euclidean distance between SPH particle $i$ and grid cell center $j$ (m) |
| $w_{ij}$ | Linear distance-based weight for particle-grid sampling (dimensionless) |
| $\rho_{l_j}$ | Total liquid density contribution to grid cell $j$ from SPH particles (kg m$^{-3}$) |
| $\omega_j^{\rho}$ | Sampling normalization factor for liquid density at grid cell $j$ (dimensionless) |
| $\omega_j^{\mathsf{T}}$ | Sampling normalization factor for liquid temperature at grid cell $j$ (kg) |
| $T_{l_j}$ | Liquid temperature at grid cell $j$ from SPH particles (K) |

### A.2 Runtime Performance

**Table 2.** Performance characteristics of our method. For a set of figures we show the number of voxels, the average number of active particles (AP), the number of vertices (NV), the simulation time of the grid (STG), the simulation time of particles (STP), and the render time (RT).

| Figure | # of Voxels | AP | NV | STG | STP | RT |
|---|---|---:|---:|---:|---:|---:|
| 1 | 400×300×200 | 175k | 27k | 65ms | 3ms | 960ms |
| 4 | 128×128×256 | - | - | 29ms | - | 10ms |
| 8a-d | 300×150×300 | 6.7k | - | 37ms | 8ms | 23ms |
| 8i-j | 300×150×300 | 6.7k | - | 36ms | 6ms | 35ms |
| 11 | 160×200×240 | 57k | 3k | 21ms | 2ms | 52ms |
| 18 | 210×150×90 | 106k | 1.5M | 7ms | 15ms | 872ms |

<!-- p. 17 -->

### A.3 Parameter Tables

In Tab. 4 we provide values for parameters that depend on the physical and chemical properties of materials, liquids and gases – these parameters are fixed. In Tab. 3 we provide values for parameters that we used to generate the results in this paper. These parameters can be configured to account for various different scenarios such as for complete combustion, different liquid spray configurations, etc.

**Table 3.** Configurable Parameters

| Simulation Parameters | Value |
|---|---|
| Delta Time | 1 / 120 s |
| Ambient Temperature | 300.0 K |
| **Particle System Parameters** | **Value** |
| Particle Capacity | 131072 - 1048576 |
| Update Multiplier | 1 - 4 |
| Smoothing Radius | 0.06 - 0.4 m |
| Stiffness | 0.004 - 0.005 |
| Near Stiffness | 0.01 - 0.02 |
| Rest Density (pseudo) | 1.0 - 300.0 |
| Linear Impulse | 0.0 |
| Quadratic Impulse | 0.4 |
| **Grid Parameter** | **Value** |
| Grid Size | 64×64×64 - 200×300×400 |
| Grid Length | 0.1 - 10.0 m |
| Pressure Iterations | 64 - 128 |
| **Thermal Parameters** | **Value** |
| Radiation coefficient | 0.0 - 6.0 |
| Density temperature coupling limit | 300 - 3000 K |
| **Liquid Parameters** | **Value** |
| Liquid Droplet Diameter | 0.0005 - 0.005 m |
| Liquid Displacement Factor | 0.0 - 0.4 |
| **Vorticity confinement** | **Value** |
| Strength | 0.0 - 50.0 |
| Lower velocity threshold | 0.0 - 0.1 |
| Upper velocity threshold | 0.0 - 5.0 |
| Lower temperature threshold | 301 K |
| **Combustion Parameters** | **Value** |
| Carbon dioxide reaction coefficients | 0.0 - 10.0 |
| Water vapor reaction coefficients | 0.0 - 10.0 |
| Residual reaction coefficient | 0.0 - 10.0 |
| Heat Efficiency | 0.0 - 1.0 |
| **Particle Emitter Parameter** | **Value** |
| Mass | 0.1 - 1.0 kg |
| Velocity | 0.0 - 10.0 m s$^{-1}$ |
| Lifetime | 0.0 - 120.0 s |
| Frequency | 10.0 - 100.0 Hz |
| Temperature | 300.0 K |
| Spray Angle | 0.0 - 180.0 ° |
| **Grid Emitter Parameter** | **Value** |
| Fuel Mass Fraction | 0.1 - 1.0 |
| Oxygen Mass Fraction | 0.1 - 1.0 |
| Nitrogen Mass Fraction | 0.1 - 1.0 |
| Temperature | 300 - 1500 K |
| Velocity | 0.0 - 10.0 m s$^{-1}$ |

**Table 4.** Fixed Parameters

| Gas Parameters | Reference |
|---|---|
| Density | Air (depending on temperature) |
| Species diffusion coefficients | Usually in Air or Nitrogen [Lide 1995] |
| Specific heat capacity | Mixture temperature averaged [Linstrom and Mallard 2025] |
| Thermal conductivity | Mixture at ambient temperature [Linstrom and Mallard 2025] |
| **Liquid Parameters** | **Reference** |
| Thermal conductivity | Water or Ethanol [Linstrom and Mallard 2025] |
| Specific heat capacity | Water or Ethanol [Linstrom and Mallard 2025] |
| Boiling Temperature | Water or Ethanol [Linstrom and Mallard 2025] |
| Heat of vaporization | Water or Ethanol [Linstrom and Mallard 2025] |
| **Solid Parameters** | **Reference** |
| Thermal conductivity | Aluminum or Iron [Linstrom and Mallard 2025] |
| Specific heat capacity | Aluminum or Iron [Linstrom and Mallard 2025] |
| Boiling Temperature | Aluminum or Iron [Linstrom and Mallard 2025] |
| **Combustion Parameters** | **Reference** |
| Activation energy | Fuel type [Westbrook and Dryer 1981] |
| Arrhenius pre-exponential factor | Fuel type [Westbrook and Dryer 1981] |
| Arrhenius temperature exponent | 0 [Westbrook and Dryer 1981] |
| Lower heat of combustion | Fuel type [Schmidt-Rohr 2015] |
| Fuel reaction exponent | Fuel type [Westbrook and Dryer 1981] |
| Oxygen reaction exponent | Fuel type [Westbrook and Dryer 1981] |
| Fuel reaction coefficients | Fuel type |
| Oxygen reaction coefficients | Fuel type |
| Nitrogen reaction coefficients | 0 |
| Fuel molar mass | Fuel type [Linstrom and Mallard 2025] |
| Fuel heat capacity | Fuel type temperature averaged [Linstrom and Mallard 2025] |
| Fuel thermal conductivity | Fuel type at ambient temperature [Linstrom and Mallard 2025] |

---

## Unreadable glyphs

None. Every glyph was legible in 250 to 800 dpi renders of the PDF pages; no `[?]` markers were needed.

## Transcriber's notes

These notes record suspected typos, inconsistencies and gaps. The transcription above reproduces the paper as printed. Units quoted below come from the paper's Appendix A.1 unless stated otherwise.

1. p. 4, Eq. 2: the radiative term $\frac{\epsilon\sigma}{C_p}(T_{\mathrm{amb}}^4 - T^4)$ does not have units of K s$^{-1}$. With $\sigma$ in W m$^{-2}$ K$^{-4}$ and $C_p$ in J kg$^{-1}$ K$^{-1}$ it has units kg K m$^{-2}$ s$^{-1}$; a factor of (area per volume)/$\rho$ would be needed. Sec. 6.1.1 (p. 10) calls $\epsilon$ a "radiation coefficient" and Table 3 (p. 17) lists "Radiation coefficient 0.0 - 6.0", so in practice $\epsilon$ seems to act as a tunable coefficient rather than an emissivity bounded by 1 (Appendix A.1.1 still calls it "Emissivity").
2. p. 4 to 5, notation: Eq. 4, Table 1 and Appendix A.1.1 write the heat of combustion with a letter o superscript ($\Delta_c H^o$); Eq. 6 and Fig. 3 write it with a digit zero ($\Delta_c H^0$). The latent heat is always $\Delta_v H^0$. Transcribed as printed.
3. p. 4 and p. 10, Eq. 4 against Table 1: the text says Eq. 4 gives kJ mol$^{-1}$; Table 1 lists $\Delta_c H^o$ "(·10$^6$)", i.e. in J mol$^{-1}$. All six Table 1 values equal $-417(c + 0.25h)$ kJ mol$^{-1}$ exactly (methane: $-417 \times 2 = -834$ kJ mol$^{-1}$ = $-0.8340 \times 10^6$ J mol$^{-1}$; butane: $-417 \times 6.5 = -2710.5$ kJ mol$^{-1}$), so they are per mole of fuel. This is a consistency check, not an error.
4. p. 4 to 5 and p. 10, Eqs. 6 and 7 with Table 1: Eq. 6 multiplies $v_c$ by $\Delta_c H^0$, which by Eq. 4 is per mole of fuel, with no stoichiometric factor; Eq. 7 removes $|q_f|$ moles of fuel per unit of $v_c$. Table 1 gives $q_f = -2$ for acetylene, butane and cyclopropane, so for these fuels each unit of reaction progress consumes two moles of fuel but releases the heat of one, unless $v_c$ is meant per mole of fuel (the paper does not say).
5. p. 5, Eq. 8: the fraction is printed with a comma as its denominator, $\frac{Y_i}{,}M_i$, so the sum reads as $Y_i$ times $M_i$. The mixture ideal-gas law needs $\sum_i Y_i/M_i$ (i.e. $\rho = p_{\mathrm{amb}}/(RT\sum_i Y_i/M_i)$); the printed form also has the wrong units. Most likely the comma replaced $M_i$ in the denominator (LaTeX typo). Transcribed as printed.
6. p. 5, Eq. 9: the right-hand side has units of a density (kg m$^{-3}$), not a density per unit time; a rate factor (for example 1/time step) is implicit. The numerator uses $T_l - T_{\mathrm{amb}}$ while the denominator uses $T_B - T_{\mathrm{amb}}$, so evaporation proceeds below the boiling point whenever $T_l > T_{\mathrm{amb}}$.
7. p. 5, Eq. 10: with $k$ in W m$^{-1}$ K$^{-1}$ and $T_{dc}$ in K m, $kT_{dc}/(\rho C_p)$ has units K m$^3$ s$^{-1}$, not K s$^{-1}$ (a 1/volume factor is missing). The liquid-temperature equation divides by the same $\rho C_p$ as the gas equation, so the two rates are equal and opposite; this conserves energy only if gas and liquid heat capacities per volume are equal.
8. p. 5, Eq. 11: the text says the species change is "based on the evaporated density", but the equation (and Fig. 3) uses the liquid density $\rho_l$, not the evaporated density $\rho_e$ defined on p. 7.
9. p. 5, p. 7 and p. 16, meaning of $\rho_d$: p. 5 says $\rho_d$ "does not correspond to the liquid density, but instead to the amount of liquid in the form of droplets", while Appendix A.1.1 says "Intrinsic density of the liquid making up a droplet (kg m$^{-3}$)". Eq. 28 ($\rho_l = \rho_d n_d$) fits the first reading; Eq. 27 ($E_d = \ldots \rho_d V_d$) and $\rho_e = n_e \rho_d V_d / V_c$ (p. 7) fit the second.
10. p. 6, Eq. 13: the kernel uses $r_{ij}$, whereas Eqs. 12, 15 and 17 use $r_{s_{ij}}$. Appendix A.1.2 defines $r_{ij}$ as the particle to grid-cell distance of Sec. 5.3, so $r_{s_{ij}}$ is presumably meant.
11. p. 6, Eq. 15: $F_{s_i}^{\mathrm{press}}$ is written as a scalar, with no direction vector (for example $\hat{r}_{s_{ij}}$); the summation is printed without brackets around both terms. Appendix A.1.2 says $F_{s_i}^{\mathrm{press}}$ is assumed to be $m_i \boldsymbol{a}_{s_i}^{\mathrm{press}}$, but it also gives $p_{s_i}$ in m$^2$ s$^{-2}$, which makes the printed sum a specific energy rather than a force.
12. p. 6, Eqs. 16 and 17: the linear damping term in Eq. 17 is printed as $\alpha \boldsymbol{u}_{s_{ij}}^{\mathrm{vis}}$, without the $\Delta$ used in Eq. 16 and in the quadratic term; the units in Appendix A.1.2 ($\alpha$ in s$^{-1}$, $\beta$ in m$^{-1}$) work for $\alpha\Delta u$ and $\beta\Delta u^2$, so $\Delta\boldsymbol{u}_{s_{ij}}^{\mathrm{vis}}$ is presumably meant. Eq. 16 is a dot product (a scalar) but is typeset bold. Eq. 17 also uses $r_{s_{ij}}$ and $\hat{r}_{s_{ij}}$ on the right-hand side but has no sum over $j$ on the left-hand side, so the per-pair accumulation is implicit.
13. p. 6, Eqs. 19 to 21: by Eq. 19, $\rho_{l_j}$ is a sum of masses times dimensionless weights and so has units kg, whereas Appendix A.1.2 gives kg m$^{-3}$ (no division by the cell volume is shown). Consequently Eq. 21 returns a density, not a mass, unless such a factor is applied. The text before Eq. 21 calls the estimated mass $m_i$; the equation uses $m_{p_i}$.
14. p. 7, Eq. 29: $\Delta\rho_l$ is not defined anywhere; from the surrounding text it is presumably the evaporated density $\rho_e$. Before Eq. 32 the text calls $\rho'$ "the evaporated density fraction", but by Eq. 29 $\rho'$ is the remaining gas fraction ($\rho' = 1$ when nothing evaporates, giving $\boldsymbol{u}' = \boldsymbol{u}$ in Eq. 32 and $T' = T$ in Eq. 30).
15. p. 7, Eq. 31: scaling every $Y_i$ by $\rho'$ leaves $\sum_i Y_i' = \rho' < 1$; the vapor mass fraction must be added elsewhere (Eq. 11 is the only species source from evaporation), but the paper does not state how the sum is restored.
16. p. 7 and p. 8, Algorithm 1 against the text: Algorithm 1 writes the particle velocity as $\boldsymbol{v}_p$, $\boldsymbol{v}_0$ (Lines 4 and 11), while Secs. 5.1 and 5.5 and Appendix A.1.2 use $\boldsymbol{u}_p$; Line 8 writes $\rho_p$, $\rho_p^{\mathrm{near}}$ where Eqs. 12 and 13 use $\rho_{s_i}$, $\rho_{s_i}^{\mathrm{near}}$ (the subscript convention on p. 5 reserves $s$ for SPH-only quantities); Line 14 samples $T$ where Eq. 23 defines $T_{l_j}$. Sec. 5.5 attributes "chemical reaction modeling" to "Eqs. 1 and 3", but Eq. 1 is the momentum equation.
17. p. 10, Table 1, $C_p$ column: Table 1 gives no units. The values for butane (1720), propane (1670), methane (2230) and ethylene (1550) are of the size of mass-specific heat capacities in J kg$^{-1}$ K$^{-1}$ near ambient temperature, matching Appendix A.1.1. The values for acetylene (63) and cyclopropane (114) are one to two orders of magnitude smaller, the size of molar heat capacities in J mol$^{-1}$ K$^{-1}$ at elevated temperature. Table 4 says fuel heat capacity is "temperature averaged". Possible mixed units; not resolvable from the paper.
18. p. 10, Table 1, missing quantities: there is no activation-energy column, although Eq. 5 needs $E_a$ and Table 4 refers it to Westbrook and Dryer [1981]; the units of $A$ are not stated (Appendix A.1.1: "units vary"). Table 4 sets the temperature exponent $n = 0$. Transcriber's recollection, not checked against the source here: the $A$, $a$, $b$ values match the single-step entries of Westbrook and Dryer (1981), which use $E_a = 30$ kcal mol$^{-1}$ for these fuels and cm-mol-s units for $A$; in that table the C$_3$H$_6$ entry ($A = 4.2 \times 10^{11}$, $a = -0.1$, $b = 1.85$) is propene, whereas the paper labels C$_3$H$_6$ as cyclopropane.
19. p. 10 and p. 17, product coefficients: the Table 1 product coefficients are exactly stoichiometric (for example $2\,C_2H_2 + 5\,O_2 \to 4\,CO_2 + 2\,H_2O$), but Table 3 lists "Carbon dioxide reaction coefficients", "Water vapor reaction coefficients" and "Residual reaction coefficient" as configurable from 0.0 to 10.0, and Sec. 6.1.1 says the vapor and residual coefficients "do not influence the combustion process itself". If such values enter Eq. 7, $\sum_i q_i M_i \neq 0$ and species mass is not conserved; no molar mass is given for the residual species.
20. p. 1, p. 12 and p. 14, detectors and sprinklers: the paper mentions "fire detectors" (abstract), a smoke detector that activates a sprinkler (Sec. 6.1.4, Fig. 18), and a sprinkler of "300 liters/min" started after 15 s (Sec. 6.2.1), but gives no detector model, threshold, response time or sprinkler spray model beyond the particle emitter parameters of Table 3.
21. p. 11 and p. 13, Fig. 13 and Sec. 6.2.3: the "dispersion" and "displacement" terms are not tied to numbered equations. Displacement appears to correspond to the velocity blend of Eq. 32 (Table 3: "Liquid Displacement Factor 0.0 - 0.4"); dispersion is not defined explicitly.
22. p. 17, Table 4: "Boiling Temperature" is listed under Solid Parameters (Aluminum or Iron), although Sec. 7 says solids are static obstacles without melting.
23. p. 16, Table 2: the row "8i-j" does not match the Fig. 8 panel ranges (i-l, i-p); possibly "8i-l" was meant.
24. Prose typos kept as printed: "Kross and Potter [2014]" (p. 2; the reference list has "K. A. Kroos"), "Futhermore" (p. 2), "OpenFOAM[Lapointe" without space (p. 3), "stochiometric" (p. 4), "can are exchanged" (p. 5), "the the establishment" (p. 5, Fig. 4), "large amounts water vapor" (p. 6, Fig. 5), "extinquishing" (p. 9, Fig. 8), "ablations studies" (p. 12), "do not not" (p. 14). In the references: "(NCAR),," and "Environmental Modelling Software" (an ampersand appears to be missing; the PDF shows a double space).
