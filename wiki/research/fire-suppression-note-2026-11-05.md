# Note for a future fire suppression model (2026-11-05)

Status: a note, not a design. Nothing here is implemented. Every source marked "to read" must be read and its DOI checked at Crossref before any value is used.

## Why it fits

The Flamescapes port already carries most of what suppression acts on:

- the gas temperature and its species, water vapour among them;
- the fuel's moisture, which dries before the fuel pyrolyses;
- the oxygen that the gas combustion consumes;
- the soot that sets the visibility.

Fire-X (Wrede et al. 2025, doi:10.1145/3763338), whose gas chemistry the port uses, is a paper about extinguishing. Its liquid model is transcribed in `wiki/sources/papers/firex-2025.md` and `firex-2025-equations.json`:

- the liquid density in a cell and its continuous evaporation (Section 4.3);
- droplet conduction with the gas;
- droplet-based evaporation (Section 5.4): the cell's sensible energy above ambient divided by the energy to heat and vaporise one droplet gives the number of droplets evaporated;
- mixing of the gas temperature and velocity after evaporation (the displacement effect).

The transcription records its gaps. The paper gives no fire or smoke detector model and no sprinkler spray model beyond the particle emitter parameters of its Table 3. Its validation case switches a 300 L/min spray on at 15 s.

## What a suppression model needs

1. **Where the sprinklers are.**
   - IFC `IfcFireSuppressionTerminal` with predefined type SPRINKLER, and its type object. Their property set (activation temperature, response class, coverage area, discharge flow rate; the IFC4 `Pset_FireSuppressionTerminalTypeSprinkler`, to verify against the IFC4 documentation) supplies the parameters where the exporter wrote them.
   - The Model step would list them, as it lists exit doors.
   - Most models in the local corpus are architectural and will have none, so a user placement tool (like the fire origin) is needed too.
2. **When each one opens.** A heat-responsive link with a response time index (RTI) heated by the gas around it. The model used by NIST's FDS and DETACT is the place to start (to read: the FDS Technical Reference Guide's sprinkler activation section and Heskestad and Bill's RTI work, 1988). The gas temperature and velocity at the sprinkler's cell come from the solver, so activation is a per-sprinkler ODE stepped with the solver.
3. **The spray.**
   - Water particles from each open head, following Fire-X's emitter (Table 3) and droplet evaporation (Section 5.4).
   - On the GPU, the particle buffer already holds flame-front particles and embers. Water droplets would be a third kind with their own update kernel.
   - Effects:
     - evaporation cools the gas and adds water vapour, which dilutes the oxygen;
     - droplets reaching a fuel surface add to its moisture, which the existing drying step must remove before pyrolysis resumes.
4. **Manual suppression.** `fireClasses.js` already lists the suitable and unsuitable agents per fire class (water, foam, CO2, dry powder, wet chemical, and for class D special powders). A later step could model one portable extinguisher discharge per class as an agent source at a point and time. Water on class K or a lithium-ion cell (L) must show the wrong outcome, not a success.
5. **Egress effects.**
   - Cooling lowers buoyancy, so smoke may fall into the lower layer and visibility at head height can drop when a sprinkler opens. The forecast and the visibility reading must show this, not hide it.
   - The plan's dose (heat plus ISO 13571 CO) then needs no change. It reads the same fields.
6. **Embers.** Droplets wetting a target cell raise the heat an ember needs to start a spot fire, through the target's moisture. The embers switch (on or off) must stay independent of the suppression switch.

## Verification to plan for

- **Energy bookkeeping.** Liquid sensible energy, latent heat and gas energy must balance over a step, as the liquid fuel pool already does (`chemistry_liquid_energy` in `formal/fireegress/FireEgress/FlamescapesPort.lean`). The same form of theorem should hold for water.
- **Activation.** The RTI ODE's exact solution for a constant gas temperature gives a closed-form activation time to test against.
- **GPU parity.** The same one-step and 200-step agreement checks as the solver (`tools/flamescapes-gpu-check.mjs`).
- **Validation.** Fire-X's own validation case (300 L/min at 15 s), then published sprinklered room fire tests (to find and read).

## Order of work

Sprinklers read from the IFC and listed, then activation (cheap, useful alone: "this head opens at t = 74 s"), then the spray on the CPU, then on the GPU, then manual suppression. Each step needs its sources read before code, as the fire classes did.
