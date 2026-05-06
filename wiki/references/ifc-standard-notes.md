# IFC Standard Notes

This page records the IFC facts used by roadmap and implementation planning.

## Current Official Version

buildingSMART lists IFC `4.3.2.0` as the latest official version, also published as ISO `16739-1:2024`.

Practical implication:

- Future IFC work should target IFC 4.3.2.0 terminology unless the project explicitly targets IFC2x3, IFC4, or an IFC5 development branch.

## IFC Is More Than A File

buildingSMART describes IFC as a schema, documentation, property and quantity set definitions, and exchange/serialization mechanisms. The common `.ifc` file is a STEP Physical File Format encoding.

Practical implication:

- A lightweight app profile can optimize extraction, but it should not be described as full IFC compliance unless it validates against official IFC requirements or a defined model view.

## Architecture Layers

The IFC 4.3 documentation describes layered schema architecture:

- Resource layer
- Core layer
- Interoperability layer
- Domain layer

Practical implication:

- Egress extraction should rely on stable building element and spatial concepts, but avoid deep assumptions about every domain-specific entity.

## Current App Subset

The current browser extraction uses a pragmatic subset:

- Slabs
- Stairs and stair flights
- Doors
- Walls
- Spaces and storeys are collected but not central to graph generation yet

Future work should make this subset explicit as an internal profile before broadening it.

