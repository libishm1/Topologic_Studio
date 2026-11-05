/**
 * Imperative wrapper around the That Open engine.
 *
 * React owns none of the 3D lifecycle. The component mounts a div, hands it to
 * this class once, and afterwards only calls methods. Classic drove the engine
 * from a dozen `useEffect` hooks whose dependency arrays included `upAxis`,
 * `flipY` and `flipZ`, so changing a checkbox tore down and rebuilt the whole
 * IFC model. Here a transform change is a matrix update.
 */

import * as THREE from "three";
import {
  Components,
  FragmentsManager,
  IfcLoader,
  Raycasters,
  RendererMode,
  SimpleCamera,
  SimpleRenderer,
  SimpleScene,
  Worlds,
} from "@thatopen/components";

import { hashFile, readCached, writeCached, writeCachedSemantics } from "../lib/fragmentCache.js";
import { PhaseTimer, mark } from "../lib/perf.js";
import { AIR_TERMINAL_NAME, aggregatedParts, collectCategoryIds, elementRecords, extractMeshes, withPredefinedType } from "./categories.js";
import { SCENE_COLOURS, heatColor, paintDeadlines } from "./heat.js";
import { coreItems } from "../core/cores.js";
import { centroidOf } from "../core/centroid.js";

/** Categories that are not building elements: the spatial structure, openings, annotations. */
const NOT_ELEMENTS = /^IFC(PROJECT|SITE|BUILDING|BUILDINGSTOREY|OPENINGELEMENT|ANNOTATION|GRID|VIRTUALELEMENT|GRIDAXIS)$|TYPE$/i;

/** Self-hosted by default; `npm run sync-assets` keeps these matching the
 *  installed package versions. Both are served from public/, under the
 *  site's base path: GitHub Pages serves the app from /Topologic_Studio/,
 *  where the root-relative "/wasm/" was a 404 and no IFC would open. The
 *  WASM path is made a full URL so web-ifc takes it as absolute. */
const BASE = import.meta.env.BASE_URL || "/";
const WASM_PATH =
  import.meta.env.VITE_WEBIFC_WASM_PATH ||
  (typeof window !== "undefined" ? new URL(`${BASE}wasm/`, window.location.href).href : `${BASE}wasm/`);
const WORKER_PATH = import.meta.env.VITE_FRAGMENTS_WORKER || `${BASE}fragments/worker.mjs`;

const OVERLAY = {
  path: "path",
  dynamicPath: "dynamicPath",
  graph: "graph",
  rlPath: "rlPath",
  navGraph: "navGraph",
  navRoute: "navRoute",
  outlines: "outlines",
  flames: "flames",
  embers: "embers",
  smoke: "smoke",
};

/** A soft round sprite for glowing particles, drawn once on a canvas. */
function glowTexture() {
  const size = 64;
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(size, size) : Object.assign(document.createElement("canvas"), { width: size, height: size });
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.6)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/** Glow colour of a particle at T kelvin: dull red to yellow-white, as hot bodies look. */
const GLOW_STOPS = [0x7f1d1d, 0xdc2626, 0xf97316, 0xfacc15, 0xfef3c7].map((hex) => new THREE.Color(hex));
function glowColor(target, T) {
  const x = Math.min(1, Math.max(0, (T - 800) / 1000)) * (GLOW_STOPS.length - 1);
  const i = Math.min(GLOW_STOPS.length - 2, Math.floor(x));
  return target.copy(GLOW_STOPS[i]).lerp(GLOW_STOPS[i + 1], x - i);
}

/** Overlays the section cut also clips; routes and markers stay whole. */
const CLIPPED = new Set([OVERLAY.outlines, OVERLAY.graph, OVERLAY.navGraph]);



/** Alternative routes are drawn as overlays alt0, alt1, ... */
const ALT_PREFIX = "alt";

/** A LOD1 room's tint by class (core/lod1.js); rooms nobody walks in are grey. */
const LOD1_ROOM_COLOURS = {
  Unit: 0xcbd5e1,
  Office: 0xc7d2fe,
  Commercial: 0xfbcfe8,
  Corridor: 0xfde68a,
  Arcade: 0xfde68a,
  Stair: 0xfdba74,
  Ramp: 0xfdba74,
  Toilet: 0x93c5fd,
  Service: 0xd6d3d1,
  Parking: 0xe5e7eb,
};
const LOD1_DOOR_COLOURS = { exit: 0x16a34a, door: 0x92400e };

/**
 * One flat-shaded geometry from many extracted meshes ({positions, indices}),
 * unindexed so each face keeps its own normal; `colourOf(mesh)` gives a
 * per-mesh colour (vertex colours), or none.
 */
function mergedMeshes(meshes, colourOf = null) {
  let count = 0;
  for (const m of meshes) count += m.indices.length;
  const positions = new Float32Array(count * 3);
  const colours = colourOf ? new Float32Array(count * 3) : null;
  const colour = new THREE.Color();
  let o = 0;
  for (const m of meshes) {
    if (colourOf) colour.setHex(colourOf(m));
    for (let k = 0; k < m.indices.length; k += 1, o += 3) {
      const v = m.indices[k] * 3;
      positions[o] = m.positions[v];
      positions[o + 1] = m.positions[v + 1];
      positions[o + 2] = m.positions[v + 2];
      if (colours) {
        colours[o] = colour.r;
        colours[o + 1] = colour.g;
        colours[o + 2] = colour.b;
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (colours) geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  geometry.computeVertexNormals();
  return geometry;
}

export class ViewerManager {
  constructor() {
    this.components = null;
    this.world = null;
    this.fragments = null;
    this.ifcLoader = null;
    this.raycaster = null;

    this.model = null;
    this.modelObject = null;
    this.modelId = null;
    this.categoryIds = null;
    // A LOD1 model drawn without an IFC (loadLod1): its egress geometry.
    this.lod1 = null;

    this.overlays = new Map();
    this.markers = {};
    this.disposed = false;
    this.ready = false;

    this._container = null;
    this._pickMode = null;
    this._onPick = null;
    this._pointerDown = null;
    this._pointerStart = null;
    this._resizeObserver = null;
    this._theme = "dark";
    // The layer colouring the graph, if any: { opacity, redraw? }.
    this._colouring = null;
  }

  // ------------------------------------------------------------- lifecycle

  async init(container, { theme = "dark", onPick } = {}) {
    if (this.components) return;
    this._container = container;
    this._onPick = onPick;
    this._theme = theme;

    const components = new Components();
    this.components = components;

    const worlds = components.get(Worlds);
    const world = worlds.create();
    world.scene = new SimpleScene(components);
    world.renderer = new SimpleRenderer(components, container);
    // No engine logo over the model (the engine's own switch); That Open
    // Engine is credited in the README and on the Methods page (MIT).
    world.renderer.showLogo = false;
    world.camera = new SimpleCamera(components);
    this.world = world;

    const scene = world.scene.three;
    scene.up.set(0, 1, 0);
    if (world.camera.three?.up) world.camera.three.up.set(0, 1, 0);

    this._buildEnvironment(scene);
    this._buildMarkers(scene);

    components.init();

    // Phones report a device pixel ratio of 3 and more; the engine caps it at
    // 2, which on a touch screen still shades 4x the pixels of 1x for little
    // visible gain. 1.5 keeps edges crisp and frames fast.
    if (window.matchMedia?.("(pointer: coarse)").matches) {
      world.renderer.three?.setPixelRatio?.(Math.min(window.devicePixelRatio || 1, 1.5));
    }

    this.raycaster = components.get(Raycasters).get(world);

    world.camera.controls?.setLookAt(12, 10, 12, 0, 0, 0);
    world.camera.controls?.update?.(0);

    this._attachPointer(container);
    this._attachResize(container);
    this.applyTheme(theme);

    const fragments = components.get(FragmentsManager);
    this._fragmentsWorkerUrl = await resolveWorkerUrl();
    fragments.init(this._fragmentsWorkerUrl);
    this.fragments = fragments;

    const ifcLoader = components.get(IfcLoader);
    await ifcLoader.setup({
      autoSetWasm: false,
      wasm: { path: WASM_PATH, absolute: WASM_PATH.startsWith("http") },
    });
    ifcLoader.settings.webIfc.COORDINATE_TO_ORIGIN = true;
    this.ifcLoader = ifcLoader;

    this.ready = true;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.setDrawRate(0);
    this._glowSprite?.dispose();
    this._glowSprite = null;
    this.ready = false;
    this._abortImport?.();
    this._warmImport?.worker.terminate();
    this._warmImport = null;

    this._resizeObserver?.disconnect();
    this._resizeObserver = null;

    if (this._container && this._pointerDown) {
      this._container.removeEventListener("pointerdown", this._pointerStart);
      this._container.removeEventListener("pointerup", this._pointerDown);
    }

    this.overlays.forEach((object) => this._destroyObject(object));
    this.overlays.clear();

    try {
      this.components?.dispose();
    } catch {
      /* the engine is going away regardless */
    }

    this.components = null;
    this.world = null;
    this.fragments = null;
    this.ifcLoader = null;
    this.model = null;
    this.modelObject = null;
  }

  // ------------------------------------------------------------ scene setup

  _buildEnvironment(scene) {
    scene.add(new THREE.AmbientLight(0xffffff, 1.6));

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(12, 20, 14);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xffffff, 0.7);
    fill.position.set(-14, 8, -10);
    scene.add(fill);

    this.grid = new THREE.GridHelper(60, 60);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.35;
    this.grid.userData.pickIgnore = true;
    scene.add(this.grid);
  }

  _buildMarkers(scene) {
    // Exit and fire origin sit close together in hue, so they are also
    // distinguished by shape: a sphere for the two egress endpoints, a spiked
    // octahedron for the ignition point. Colour alone would not separate them
    // for a colour-blind viewer either.
    const make = (color, shape = "sphere") => {
      const geometry =
        shape === "spike"
          ? new THREE.OctahedronGeometry(0.3, 0)
          : new THREE.SphereGeometry(0.22, 20, 20);
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false }),
      );
      // A white outline keeps the marker legible against the graph, which is
      // itself a dense field of saturated colour. BackSide means only the far
      // hemisphere draws, so it reads as a rim rather than a veil over the
      // marker's own colour.
      const halo = new THREE.Mesh(
        shape === "spike"
          ? new THREE.OctahedronGeometry(0.38, 0)
          : new THREE.SphereGeometry(0.3, 20, 20),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          side: THREE.BackSide,
          transparent: true,
          opacity: 0.85,
          depthTest: false,
          depthWrite: false,
        }),
      );
      mesh.add(halo);
      mesh.renderOrder = 999;
      halo.renderOrder = 998;
      mesh.visible = false;
      mesh.userData.pickIgnore = true;
      scene.add(mesh);
      return mesh;
    };
    this.markers.start = make(0x22c55e);
    this.markers.exit = make(0xf97316);
    this.markers.fire = make(0xff2d00, "spike");
  }

  applyTheme(theme) {
    this._theme = theme;
    if (!this.world) return;
    const dark = theme === "dark";
    this.world.scene.three.background = new THREE.Color(dark ? 0x0d1219 : 0xeef2f7);
    if (this.grid) {
      this.grid.material.color = new THREE.Color(dark ? 0x223049 : 0xcbd5e1);
      this.grid.material.needsUpdate = true;
    }
    const outlines = this.overlays.get(OVERLAY.outlines);
    if (outlines) outlines.material.color = new THREE.Color(this._outlineColor());
    // Heat and deadline colours have a ramp per background.
    this._colouring?.redraw?.();
  }

  _outlineColor() {
    return this._theme === "dark" ? 0xcbd5e1 : 0x334155;
  }

  // ------------------------------------------------------ outlines, section

  /**
   * The building's edges as lines. Ghosted, the model was flat translucent
   * faces with no shading, which read as a 2D diagram, especially on a phone;
   * outlining walls, slabs, stairs, doors and openings makes the 3D form read
   * in every appearance. Built from the meshes extracted for the graph, so
   * they sit exactly on it; faces meeting at under 30 degrees are merged.
   */
  setOutlines(meshes) {
    if (!this.world) return;
    const chunks = [];
    let total = 0;
    for (const mesh of meshes || []) {
      if (!mesh?.positions?.length || !mesh.indices?.length) continue;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
      geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
      const edges = new THREE.EdgesGeometry(geometry, 30);
      const array = edges.getAttribute("position").array;
      chunks.push(array);
      total += array.length;
      geometry.dispose();
      edges.dispose();
    }
    if (!total) {
      this._setOverlay(OVERLAY.outlines, null);
      return;
    }
    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(merged, 3));
    const material = new THREE.LineBasicMaterial({
      color: this._outlineColor(),
      transparent: true,
      opacity: 0.6,
    });
    const lines = new THREE.LineSegments(geometry, material);
    lines.renderOrder = 2;
    this._setOverlay(OVERLAY.outlines, lines);
  }

  /**
   * Cut the building above `height` (scene units, up is +y), as a plan view
   * does, or show it whole with null. The model, its outlines and the graph
   * are clipped; routes and markers are not.
   */
  setSectionHeight(height) {
    this._section = Number.isFinite(height) ? new THREE.Plane(new THREE.Vector3(0, -1, 0), height) : null;
    if (this.modelObject) this._applyClip(this.modelObject);
    for (const name of CLIPPED) {
      const object = this.overlays.get(name);
      if (object) this._applyClip(object);
    }
    // A cut seen from eye level is just a facade: look from above at an
    // angle unless the view already looks down by more than 20 degrees.
    const camera = this.world?.camera?.three;
    const target = this.world?.camera?.controls?.getTarget?.(new THREE.Vector3());
    if (this._section && camera && target) {
      const dir = camera.position.clone().sub(target).normalize();
      if (dir.y < Math.sin(THREE.MathUtils.degToRad(20))) this.setStandardView("iso");
    }
  }

  _applyClip(object) {
    const planes = this._section ? [this._section] : [];
    object.traverse((child) => {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        if (!material) continue;
        material.clippingPlanes = planes;
        material.needsUpdate = true;
      }
    });
  }

  // ------------------------------------------------------------------ input

  _attachPointer(container) {
    // Distinguish a click from an orbit drag: firing a raycast on every
    // pointerup made rotating the model reposition the start marker.
    let downAt = null;

    this._pointerStart = (event) => {
      downAt = { x: event.clientX, y: event.clientY, t: performance.now() };
    };

    this._pointerDown = async (event) => {
      if (!this._pickMode || !downAt) return;
      const moved = Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y);
      const elapsed = performance.now() - downAt.t;
      downAt = null;
      if (moved > 5 || elapsed > 600) return;

      // With a graph built, picks land on the graph only: a ray into the model
      // hits a wall or the roof as often as a floor, and the point then has to
      // be snapped to the graph anyway.
      if (this._graphNodes?.length) {
        const point = this._pickGraphNode(event);
        if (point) this._onPick?.(this._pickMode, point, null);
        return;
      }
      try {
        const hit = await this.raycaster?.castRay();
        if (!hit?.point) return;
        const point = [hit.point.x, hit.point.y, hit.point.z];
        this._onPick?.(this._pickMode, point, hit);
      } catch {
        /* a missed ray is not an error worth surfacing */
      }
    };

    container.addEventListener("pointerdown", this._pointerStart);
    container.addEventListener("pointerup", this._pointerDown);
  }

  /**
   * The graph node under the pointer: among nodes within 12 px of it on
   * screen, the one nearest the camera, which is the one you see. Screen space
   * rather than a world-space radius, so picking feels the same at any zoom.
   */
  _pickGraphNode(event, tolerancePx = 12) {
    const nodes = this._graphNodes;
    const camera = this.world?.camera?.three;
    if (!nodes || !camera || !this._container) return null;
    const rect = this._container.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const v = new THREE.Vector3();
    const tol2 = tolerancePx * tolerancePx;
    let best = -1;
    let bestDepth = Infinity;
    for (let i = 0; i < nodes.length; i += 3) {
      v.set(nodes[i], nodes[i + 1], nodes[i + 2]).project(camera);
      if (v.z < -1 || v.z > 1) continue; // behind the camera or past the far plane
      const sx = ((v.x + 1) / 2) * rect.width;
      const sy = ((1 - v.y) / 2) * rect.height;
      const d2 = (sx - px) * (sx - px) + (sy - py) * (sy - py);
      if (d2 <= tol2 && v.z < bestDepth) {
        bestDepth = v.z;
        best = i;
      }
    }
    return best < 0 ? null : [nodes[best], nodes[best + 1], nodes[best + 2]];
  }

  _attachResize(container) {
    // SimpleRenderer listens to window resize, which misses the panel being
    // collapsed or the window being split.
    this._resizeObserver = new ResizeObserver(() => {
      this.world?.renderer?.resize?.();
      this.world?.camera?.updateAspect?.();
    });
    this._resizeObserver.observe(container);
  }

  setPickMode(mode) {
    this._pickMode = mode || null;
  }

  // ------------------------------------------------------------------- load

  /**
   * Load an IFC file, using the fragment cache when the same bytes have been
   * seen before.
   *
   * `prepared`: {fragments, semantics} converted ahead of time (the examples,
   * scripts/prepare-examples.mjs). The file is then not read: only its name
   * and size are used, and nothing is converted or cached.
   *
   * @returns {Promise<{cached: boolean, phases: object, stats: object,
   *          semantics: Promise<object>|null}>}  semantics: what the IFC says
   *          about egress (core/ifcSemantics.js), from the cache or read by
   *          the import worker; they may come after the model is drawn. Null
   *          when no worker could read them.
   */
  async loadIfc(file, { onProgress, signal, prepared = null } = {}) {
    if (!this.ready) throw new Error("Viewer is still starting up.");
    // The page stays responsive while a model converts, so another file can
    // be opened meanwhile; the older load is dropped ("cancelled"). Its
    // conversion is aborted, and every step after an await checks that it is
    // still the latest load: a model it finishes loading after a newer load
    // has begun is disposed, not added to the scene beside the new one.
    this._abortImport?.();
    const token = (this._loadToken = (this._loadToken || 0) + 1);
    const current = () => {
      if (signal?.aborted || token !== this._loadToken || this.disposed) throw new Error("cancelled");
    };

    const timer = new PhaseTimer("ifc.load");
    const report = (stage, detail) => onProgress?.({ stage, detail });

    let key = null;
    let buffer = null;
    if (!prepared) {
      report("reading", file.name);
      ({ key, buffer } = await timer.run("hash", () => hashFile(file)));
      current();
    }

    this.clearModel();

    const record = prepared
      ? { buffer: prepared.fragments, semantics: prepared.semantics }
      : await timer.run("cacheLookup", () => readCached(key));
    current();
    let fragmentBytes = record?.buffer ?? null;
    const cached = Boolean(fragmentBytes);
    let semantics = null;

    if (!cached) {
      report("converting", "Converting IFC to fragments");
      fragmentBytes = await timer.run("convert", async () => {
        const job = await this._importJob(buffer, file.name, {
          onProgress: (p) => report("converting", p),
          signal,
        });
        current();
        const { fragments: converted, ifc } = job;
        if (converted) {
          semantics = job.semantics;
          // What IfcLoader.load sets and names a model it converted. The id
          // must differ from the cached path's: clearModel does not wait for
          // the old model's disposal, whose last step drops the thread route
          // by id, so reopening the file from the cache under the same id
          // could lose the new model's route and never finish loading.
          this.fragments.core.settings.autoCoordinate = false;
          // Loading transfers the buffer to the fragments worker; the copy
          // is what gets cached.
          const model = await this.fragments.core.load(converted.slice(), {
            modelId: file.name,
            raw: false,
          });
          this._adoptIfCurrent(model, current);
          return converted;
        }
        // No worker could convert it here: convert on this thread, as before.
        const model = await this.ifcLoader.load(
          new Uint8Array(ifc),
          false,
          file.name,
          { processData: { progressCallback: (p) => report("converting", p) } },
        );
        this._adoptIfCurrent(model, current);
        // Serialise what we just built so the next load skips the parse.
        return model.getBuffer(false);
      });

      report("caching", "Saving to local cache");
      // The fragments now; the semantics when the worker has read them.
      const reading = semantics;
      void timer
        .run("cacheWrite", () =>
          writeCached(key, fragmentBytes, {
            name: file.name,
            bytes: file.size,
          }),
        )
        .then((written) => written && reading?.then((s) => writeCachedSemantics(key, s)))
        .catch(() => {});
    } else {
      report("cached", prepared ? "Loading the model" : "Loading cached fragments");
      if (record.semantics) {
        semantics = Promise.resolve(record.semantics);
      } else if (buffer) {
        // Cached without them: read them beside the model's load.
        semantics = (await this._importJob(buffer, file.name, { signal, convert: false })).semantics;
        void semantics.then((s) => writeCachedSemantics(key, s)).catch(() => {});
      }
      const model = await timer.run("loadFragments", () =>
        this.fragments.core.load(fragmentBytes, {
          // A prepared model has no hash; the load's own number keeps
          // its id apart from a model of the same name still being disposed.
          modelId: prepared ? `${file.name}#${token}` : `${file.name}-${key.slice(0, 8)}`,
          raw: false,
        }),
      );
      this._adoptIfCurrent(model, current);
    }

    current();
    // From here on the model is the one adopted above; a newer load's
    // clearModel removes it if this one is overtaken.
    const model = this.model;

    report("rendering", "Drawing the model");
    await timer.run("firstFrame", async () => {
      await this.fragments.core.update(true);
      this.fitToModel();
    });
    current();

    // Categories come from the fragments model, not a second IFC parse.
    report("indexing", "Reading IFC categories");
    const categoryIds = await timer.run("categories", () => collectCategoryIds(model));
    current();
    this.categoryIds = categoryIds;

    // IfcSpace volumes are rooms, not built fabric: drawn, they wrap the
    // building in tinted boxes (the Duplex had one above its roof). Hidden
    // from view only; the graph still reads their floors.
    if (categoryIds.spaces?.length && model.setVisible) {
      try {
        await model.setVisible(categoryIds.spaces, false);
        await this.fragments.core.update(true);
      } catch (error) {
        console.warn("[viewer] Could not hide IfcSpace volumes:", error);
      }
      current();
    }

    const phases = timer.finish({
      cached,
      prepared: Boolean(prepared),
      file: file.name,
      bytes: file.size,
    });

    return {
      cached,
      prepared: Boolean(prepared),
      semantics,
      phases: phases.detail,
      totalMs: phases.ms,
      stats: {
        floors: this.categoryIds.floors.length,
        stairs: this.categoryIds.stairs.length,
        doors: this.categoryIds.doors.length,
        walls: this.categoryIds.walls.length,
        spaces: this.categoryIds.spaces.length,
        storeys: this.categoryIds.storeys.length,
      },
    };
  }

  /**
   * Run a job in src/workers/ifc-import.worker.js, with the importer settings
   * `init` gave the IfcLoader.
   *
   * Converting (`convert` true), resolves to `{ fragments, semantics }` once
   * the fragments are back: `semantics` is a promise for what the same worker
   * reads next (core/ifcSemantics.js), and the worker ends when it settles.
   * Resolves to `{ ifc }` when no worker can run the conversion here (no
   * module workers, its script failed to load, or web-ifc could not start in
   * it); `ifc` is then the IFC buffer, still whole, for the caller to convert
   * on this thread. The buffer is only transferred once the worker has
   * loaded, and comes back if web-ifc fails to start. Rejects with a message
   * for the user when the conversion itself fails.
   *
   * With `convert` false, only reads the semantics: resolves to
   * `{ semantics }` at once.
   *
   * The conversion, or else the semantics, reject with "cancelled" when
   * `signal` aborts, another load starts or the viewer is disposed.
   */
  _importJob(buffer, name, { onProgress, signal, convert = true } = {}) {
    if (signal?.aborted) return Promise.reject(new Error("cancelled"));

    let resolveJob, rejectJob, resolveSemantics, rejectSemantics;
    const job = new Promise((resolve, reject) => {
      resolveJob = resolve;
      rejectJob = reject;
    });
    const semantics = new Promise((resolve, reject) => {
      resolveSemantics = resolve;
      rejectSemantics = reject;
    });
    // Not every caller reads them.
    semantics.catch(() => {});

    const noWorker = (reason) => {
      rejectSemantics(new Error(`The IFC reader could not start in a worker (${reason}).`));
      resolveJob(convert ? { ifc: buffer } : { semantics });
      return job;
    };
    if (typeof Worker === "undefined") return noWorker("no module workers");

    // Take the worker warmed while the page was idle, if it loaded; its
    // "ready" may already have come, or still be on its way to the handler
    // set below.
    let worker;
    let ready = false;
    const warm = this._warmImport;
    this._warmImport = null;
    if (warm && !warm.failed) {
      worker = warm.worker;
      ready = warm.ready;
    } else {
      warm?.worker.terminate();
      try {
        worker = this._newImportWorker();
      } catch (error) {
        console.warn("[viewer] IFC import worker unavailable; converting on the main thread.", error);
        return noWorker(error?.message || "it did not start");
      }
    }

    const wasm = this._wasmSettings();
    const webIfc = { ...this.ifcLoader.settings.webIfc };

    let started = false;
    // The fragments are back, or none were asked for: what fails from here
    // on is the semantics.
    let converted = !convert;
    let ended = false;
    const end = () => {
      if (ended) return;
      ended = true;
      worker.terminate();
      signal?.removeEventListener("abort", abort);
      if (this._abortImport === abort) this._abortImport = null;
      this._scheduleWarmImport();
    };
    const fail = (error) => {
      end();
      if (!converted) rejectJob(error);
      rejectSemantics(error);
    };
    const abort = () => fail(new Error("cancelled"));
    const unavailable = (reason, ifc) => {
      if (convert) console.warn(`[viewer] IFC import worker unavailable (${reason}); converting on the main thread.`);
      end();
      buffer = ifc;
      noWorker(reason);
    };
    const post = () => {
      started = true;
      if (convert) worker.postMessage({ type: "convert", bytes: buffer, name, wasm, webIfc }, [buffer]);
      else worker.postMessage({ type: "semantics", bytes: buffer, wasm }, [buffer]);
    };

    worker.onmessage = ({ data }) => {
      switch (data?.type) {
        case "ready":
          post();
          break;
        case "unavailable":
          unavailable(data.reason, data.bytes);
          break;
        case "progress":
          onProgress?.(data.value);
          break;
        case "done":
          converted = true;
          resolveJob({ fragments: data.bytes, semantics });
          break;
        case "failed":
          fail(new Error(data.message));
          break;
        case "semantics":
          end();
          resolveSemantics(data.result);
          break;
        case "semanticsFailed":
          end();
          rejectSemantics(new Error(data.message));
          break;
        default:
          break;
      }
    };
    worker.onerror = (event) => {
      event.preventDefault?.();
      if (!started) {
        unavailable(event.message || "its script did not load", buffer);
      } else {
        fail(
          new Error(
            `The IFC ${converted ? "reader" : "converter"} stopped while reading ${name}, usually because the ` +
              "browser ran out of memory. Close other tabs and try again, or open the file on a desktop browser.",
          ),
        );
      }
    };
    worker.onmessageerror = () =>
      fail(new Error(`What the IFC worker read from ${name} could not be passed back to the page. Try again.`));

    signal?.addEventListener("abort", abort, { once: true });
    this._abortImport = abort;
    if (ready) post();
    if (!convert) resolveJob({ semantics });
    return job;
  }

  /** web-ifc's WASM directory, as a full URL: the worker's own URL is under
   *  assets/, against which a relative path would be read. */
  _wasmSettings() {
    const { path, absolute } = this.ifcLoader.settings.wasm;
    return absolute ? { path, absolute } : { path: new URL(path, window.location.href).href, absolute: true };
  }

  _newImportWorker() {
    return new Worker(new URL("../workers/ifc-import.worker.js", import.meta.url), {
      type: "module",
      name: "ifc-import",
    });
  }

  /**
   * Start loading an import worker because a file is about to be opened: the
   * pointer or focus is on a file button or an example, or a file is dragged
   * over the page. Warming at every page load made every visitor download its
   * module, model or not (review m9). Started at once, not when the page is
   * next idle: on a phone the tap comes a moment before the file does.
   */
  warmImport({ convert = true } = {}) {
    if (!this.ready) return;
    this._prefetch(this._fragmentsWorkerUrl);
    if (!convert) return;
    this._prefetch(new URL("web-ifc.wasm", this._wasmSettings().path).href);
    this._scheduleWarmImport({ now: true });
  }

  /**
   * Fetch into the HTTP cache what a first open otherwise asks for one after
   * the other: web-ifc's WASM (477 KB gzipped), which the import worker
   * requests only once its own module has loaded, and the fragments worker
   * (283 KB), which the viewer starts only once the conversion is done. In
   * turn they cost a phone on 4G about a second of each first open.
   */
  _prefetch(url) {
    this._prefetched ||= new Set();
    if (!url || this._prefetched.has(url) || typeof fetch !== "function") return;
    this._prefetched.add(url);
    fetch(url)
      .then((response) => response.arrayBuffer())
      .catch(() => {});
  }

  /**
   * Load an import worker, so its module (about 645 KB gzipped, its own
   * fragments and web-ifc) is in memory before the next file is opened: in a
   * worker the first conversion otherwise paid that download, about 0.2-0.3
   * s on a desktop and more on a phone. Asked for by warmImport, and after
   * each job, when the page is idle, for the next file.
   */
  _scheduleWarmImport({ now = false } = {}) {
    // A start already waiting for idle time is overtaken by one asked for now.
    if (this._warmImport || (this._warmPending && !now) || this.disposed || typeof Worker === "undefined") return;
    this._warmPending = true;
    const start = () => {
      this._warmPending = false;
      if (this._warmImport || this.disposed) return;
      try {
        const worker = this._newImportWorker();
        const warm = { worker, ready: false, failed: false };
        worker.onmessage = ({ data }) => {
          if (data?.type === "ready") warm.ready = true;
        };
        worker.onerror = (event) => {
          event.preventDefault?.();
          warm.failed = true;
        };
        this._warmImport = warm;
      } catch {
        // No module workers here: conversions run on the main thread.
      }
    };
    if (now) start();
    else if (typeof requestIdleCallback === "function") requestIdleCallback(start, { timeout: 4000 });
    else setTimeout(start, 1500);
  }

  /** Adopt a model a load just produced, or dispose of it if that load has
   *  been overtaken (`current` throws "cancelled"). */
  _adoptIfCurrent(model, current) {
    try {
      current();
    } catch (error) {
      const id = model?.modelId;
      if (id !== undefined) {
        Promise.resolve()
          .then(() => this.fragments?.core?.disposeModel?.(id))
          .catch(() => {});
      }
      throw error;
    }
    this._adoptModel(model);
  }

  _adoptModel(model) {
    if (!model) throw new Error("The IFC file produced no fragments model.");
    this.model = model;
    model.graphicsQuality = 1;
    model.frozen = false;

    const object = model.object || (model instanceof THREE.Object3D ? model : null);
    if (!object) throw new Error("The fragments model has no renderable geometry.");

    this.modelObject = object;
    if (this.world.camera.three && model.useCamera) {
      model.useCamera(this.world.camera.three);
    }
    this.world.scene.three.add(object);
  }

  /**
   * Draw a LOD1 model with no IFC behind it, from its egress geometry
   * (core/lod1.js lod1ToEgressGeometry, in the viewer frame): the walkable
   * rooms as translucent prisms tinted by class, the rooms nobody walks in
   * grey, and the slabs, walls, stairs and doors (exits green) solid. Each
   * group is merged into one mesh. The geometry is kept, so the graph, the
   * fire and the two-zone model read it as they read an IFC's.
   */
  loadLod1(geometry) {
    if (!this.ready) throw new Error("Viewer is still starting up.");
    // An IFC still converting is dropped, as when another file is opened.
    this._abortImport?.();
    this._loadToken = (this._loadToken || 0) + 1;
    this.clearModel();
    const lod1 = geometry.lod1 || {};
    const classOf = new Map((lod1.rooms || []).map((r) => [r.itemId, r.class]));
    const kindOf = new Map((lod1.doors || []).map((d) => [d.itemId, d.kind]));
    const group = new THREE.Group();
    group.name = "lod1";
    const add = (meshes, material, colourOf = null) => {
      const list = (meshes || []).filter((m) => m?.indices?.length);
      if (!list.length) return;
      const mesh = new THREE.Mesh(mergedMeshes(list, colourOf), material);
      group.add(mesh);
    };
    const solid = (color) => new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide });
    const glass = (options) =>
      new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide, ...options });
    add(geometry.floors, solid(0xe2e8f0));
    add(geometry.walls, solid(0xf8fafc));
    add(geometry.stairs, solid(0xf59e0b));
    add(
      (geometry.doors || []).filter((m) => (m.kind ?? kindOf.get(m.itemId)) !== "opening"),
      solid(0xffffff),
      (m) => LOD1_DOOR_COLOURS[m.kind ?? kindOf.get(m.itemId)] ?? LOD1_DOOR_COLOURS.door,
    );
    add(geometry.spaces, glass({ vertexColors: true }), (m) => LOD1_ROOM_COLOURS[classOf.get(m.itemId)] ?? LOD1_ROOM_COLOURS.Unit);
    add(lod1.shafts, glass({ color: 0x9ca3af, opacity: 0.25 }));
    this.world.scene.three.add(group);
    this.modelObject = group;
    this.lod1 = geometry;
    // Picked like a model's surfaces before a graph is built (castRay reads them).
    group.children.forEach((mesh) => this.world.meshes?.add?.(mesh));
    this.fitToModel();
  }

  clearModel() {
    if (this.lod1 && this.modelObject) {
      for (const mesh of [...this.modelObject.children]) {
        this.world?.meshes?.delete?.(mesh);
        this.modelObject.remove(mesh);
        this._destroyObject(mesh);
      }
    }
    this.lod1 = null;
    if (this.modelObject) {
      this.world?.scene?.three?.remove(this.modelObject);
      this.modelObject = null;
    }
    if (this.model) {
      try {
        this.fragments?.core?.disposeModel?.(this.model.modelId);
      } catch {
        /* already gone */
      }
      this.model = null;
    }
    this.categoryIds = null;
    this.clearOverlays();
  }

  /**
   * Pull the geometry the sampler worker needs.
   * One pass per category, straight out of the fragments already in memory.
   */
  async extractEgressGeometry() {
    // A LOD1 model is its egress geometry.
    if (this.lod1) return this.lod1;
    if (!this.model || !this.categoryIds) {
      throw new Error("Load an IFC model first.");
    }
    const done = mark("ifc.extractGeometry");
    const world = this.modelObject?.matrixWorld?.clone() || null;
    const ids = this.categoryIds;

    // Coverings join the floors only when typed FLOORING; a failed attribute
    // query just drops them rather than letting ceilings in.
    const flooring = await withPredefinedType(this.model, ids.coverings || [], "FLOORING").catch(
      () => [],
    );

    // Slabs typed ROOF are walkable surfaces but not floors: 7% of C20's
    // graph and 14% of ISSUE_159's, and two of C20's separate parts
    // (wiki/research/ifc-handling-survey-2026-10-27.md, item 2). Untyped roofs
    // are still caught by the height rule in lattice.js.
    const roofs = new Set(await withPredefinedType(this.model, ids.floors, "ROOF").catch(() => []));
    const floorIds = roofs.size ? ids.floors.filter((id) => !roofs.has(id)) : ids.floors;

    // Parts aggregated into stairs and ramps (IfcRelAggregates) are stair
    // geometry: S_Office's IfcStairs have none of their own, only proxies for
    // flights and landings. Slabs, walls and doors among the parts stay in
    // their own groups.
    const partsOf = await aggregatedParts(this.model, ids.stairs).catch(() => new Map());
    const own = new Set([...ids.floors, ...(ids.coverings || []), ...ids.walls, ...ids.doors, ...ids.stairs]);
    const stairPartIds = [...new Set([...partsOf.values()].flat())].filter((id) => !own.has(id));

    // Vertical cores (src/core/cores.js): stairs, lifts and escalators by type,
    // and rooms or proxies by name. Stair and room meshes are extracted
    // anyway; the others are extracted for their footprint.
    const records = await elementRecords(this.model, [
      ...ids.stairs,
      ...(ids.spaces || []),
      ...(ids.transport || []),
      ...(ids.proxies || []),
    ]).catch(() => []);
    for (const record of records) record.parts = partsOf.get(record.itemId) || [];
    const cores = coreItems(records);
    const extracted = new Set([...ids.stairs, ...stairPartIds, ...(ids.spaces || [])]);
    const coreMeshIds = cores.map((core) => core.itemId).filter((id) => !extracted.has(id));

    const [floors, stairs, doors, walls, spaces, obstacles, coreMeshes, roofMeshes] = await Promise.all([
      extractMeshes(this.model, [...floorIds, ...flooring], world),
      extractMeshes(this.model, [...ids.stairs, ...stairPartIds], world),
      extractMeshes(this.model, ids.doors, world),
      extractMeshes(this.model, ids.walls, world),
      extractMeshes(this.model, ids.spaces || [], world),
      extractMeshes(this.model, ids.obstacles || [], world),
      extractMeshes(this.model, coreMeshIds, world),
      // Roof slabs are not walked on, but the Flamescapes solver needs them
      // as the top storey's ceiling.
      extractMeshes(this.model, [...roofs, ...(ids.roofElements || [])], world),
    ]);

    done({
      floors: floors.length,
      stairs: stairs.length,
      doors: doors.length,
      walls: walls.length,
      spaces: spaces.length,
      obstacles: obstacles.length,
      roofs: roofs.size,
      stairParts: stairPartIds.length,
      cores: cores.length,
    });
    // Air terminals, for the fire solver's ventilation: their centres.
    const flowTerminals = await elementRecords(this.model, ids.flowTerminals || []).catch(() => []);
    const terminalIds = [...(ids.airTerminals || []), ...flowTerminals.filter((r) => AIR_TERMINAL_NAME.test(r.name)).map((r) => r.itemId)];
    const terminals = [...(await this.positionsOf(terminalIds).catch(() => new Map())).values()];
    return { floors, stairs, doors, walls, spaces, obstacles, roofs: roofs.size, roofMeshes, cores, coreMeshes, terminals };
  }

  /**
   * Elements and rooms the ignition matcher reads (src/core/ignition.js):
   * names, classes and predefined types of the candidate classes and the
   * proxies, and the rooms' names. Positions come later, for the ranked
   * candidates only (positionsOf).
   */
  async ignitionData() {
    // A LOD1 model has rooms but no furniture or appliances: rooms only.
    if (this.lod1) {
      const rooms = (this.lod1.lod1?.rooms || []).filter((r) => r.walkable).map((r) => ({ itemId: r.itemId, type: "IFCSPACE", name: r.name, longName: r.name }));
      return { records: [], rooms };
    }
    if (!this.model || !this.categoryIds) throw new Error("Load an IFC model first.");
    const ids = this.categoryIds;
    const [records, rooms] = await Promise.all([
      elementRecords(this.model, [...(ids.ignition || []), ...(ids.proxies || [])]).catch(() => []),
      elementRecords(this.model, ids.spaces || []).catch(() => []),
    ]);
    return { records, rooms };
  }

  /**
   * Every element, for identification (src/core/identify.js): class, names
   * and predefined type. Generic elements (IfcBuildingElementProxy and the
   * like) also get their box, its bottom's height above the floor below (the
   * highest door bottom under it), and their meshes, for the shape classifier.
   */
  async identifyData() {
    if (this.lod1) throw new Error("A rooms (LOD1) model has no elements to identify: open an IFC.");
    if (!this.model || !this.categoryIds) throw new Error("Load an IFC model first.");
    // identify.js carries the ignition catalogue's name patterns: loaded when asked for.
    const { GENERIC_CLASSES, boxOf } = await import("../core/identify.js");
    const categories = ((await this.model.getCategories()) || []).filter((c) => !NOT_ELEMENTS.test(c));
    const found = categories.length ? await this.model.getItemsOfCategories(categories.map((c) => new RegExp(`^${c}$`, "i"))) : {};
    const ids = [...new Set(Object.values(found || {}).flat())];
    const records = await elementRecords(this.model, ids);
    const generic = records.filter((r) => GENERIC_CLASSES.has(String(r.type).toUpperCase()));
    const world = this.modelObject?.matrixWorld?.clone() || null;
    const [meshes, doorMeshes] = await Promise.all([
      extractMeshes(this.model, generic.map((r) => r.itemId), world),
      extractMeshes(this.model, this.categoryIds.doors, world),
    ]);
    const group = (list) => {
      const by = new Map();
      for (const mesh of list) {
        if (!by.has(mesh.itemId)) by.set(mesh.itemId, []);
        by.get(mesh.itemId).push(mesh);
      }
      return by;
    };
    const levels = [...group(doorMeshes).values()].map((list) => boxOf(list)?.bottom).filter(Number.isFinite);
    const meshesOf = group(meshes);
    for (const r of generic) {
      const box = boxOf(meshesOf.get(r.itemId));
      if (!box) continue;
      r.dims = box.dims;
      const below = levels.filter((z) => z <= box.bottom + 0.2);
      r.above = below.length ? box.bottom - Math.max(...below) : null;
    }
    return { records, meshesOf };
  }

  /** Centroids (viewer frame) of some items' geometry, as a Map itemId -> [x, y, z]. */
  async positionsOf(itemIds) {
    const out = new Map();
    if (!itemIds?.length || this.lod1) return out;
    const world = this.modelObject?.matrixWorld?.clone() || null;
    const meshes = await extractMeshes(this.model, itemIds, world);
    const byItem = new Map();
    for (const mesh of meshes) {
      if (!byItem.has(mesh.itemId)) byItem.set(mesh.itemId, []);
      byItem.get(mesh.itemId).push(mesh);
    }
    for (const [itemId, list] of byItem) {
      const c = centroidOf(list);
      if (c) out.set(itemId, c);
    }
    return out;
  }

  // --------------------------------------------------------------- overlays

  _destroyObject(object) {
    if (!object) return;
    this.world?.scene?.three?.remove(object);
    object.geometry?.dispose?.();
    if (Array.isArray(object.material)) {
      object.material.forEach((m) => m.dispose?.());
    } else {
      object.material?.dispose?.();
    }
  }

  _setOverlay(name, object) {
    const previous = this.overlays.get(name);
    if (previous) this._destroyObject(previous);
    if (object) {
      object.userData.pickIgnore = true;
      if (CLIPPED.has(name)) this._applyClip(object);
      this.world.scene.three.add(object);
      this.overlays.set(name, object);
    } else {
      this.overlays.delete(name);
    }
  }

  clearOverlays() {
    this.overlays.forEach((object) => this._destroyObject(object));
    this.overlays.clear();
    // Picking reads these; a new model must not be picked against the old graph.
    this._graphNodes = null;
    this._graphEdges = null;
    this._colouring = null;
  }

  /**
   * Draw an egress route.
   *
   * Built as a tube, not a line. `LineBasicMaterial.linewidth` is ignored by
   * every major browser (it is capped at 1px by the WebGL spec on most
   * platforms), so a route drawn with THREE.Line renders as a hairline thread
   * lost among thousands of graph edges - which defeats the point of showing
   * it. A tube has real geometric thickness that scales with the model.
   */
  setPath(points, { color = 0xdc2626, radius = null, dynamic = false, name = null } = {}) {
    name = name ?? (dynamic ? OVERLAY.dynamicPath : OVERLAY.path);
    if (!points || points.length < 2) {
      this._setOverlay(name, null);
      this._updateGraphEmphasis();
      return;
    }

    const vectors = points
      .filter((p) => p && p.length >= 3)
      .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (vectors.length < 2) {
      this._setOverlay(name, null);
      this._updateGraphEmphasis();
      return;
    }

    // Scale the tube to the model so it reads on a corridor and on a campus.
    let tubeRadius = radius;
    if (tubeRadius === null) {
      const box = this._modelBox();
      const extent = box
        ? Math.max(...box.getSize(new THREE.Vector3()).toArray())
        : 20;
      tubeRadius = Math.max(0.04, Math.min(0.35, extent * 0.006));
    }

    const curve = new THREE.CatmullRomCurve3(vectors, false, "catmullrom", 0.15);
    const segments = Math.min(1200, Math.max(vectors.length * 4, 24));
    const geometry = new THREE.TubeGeometry(curve, segments, tubeRadius, 8, false);
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
      depthWrite: false,
    });
    const tube = new THREE.Mesh(geometry, material);
    tube.renderOrder = dynamic ? 902 : 901;
    this._setOverlay(name, tube);
    this._updateGraphEmphasis();
  }

  /**
   * Draw alternative routes, replacing any drawn before.
   *
   * @param routes [{ points, color }], best first. The best is drawn thickest
   *               so the ranking reads without a legend.
   */
  setRoutes(routes) {
    for (const name of [...this.overlays.keys()]) {
      if (name.startsWith(ALT_PREFIX)) this._setOverlay(name, null);
    }
    const box = this._modelBox();
    const extent = box ? Math.max(...box.getSize(new THREE.Vector3()).toArray()) : 20;
    const base = Math.max(0.04, Math.min(0.35, extent * 0.006));
    (routes || []).forEach((route, i) => {
      this.setPath(route.points, {
        color: route.color,
        name: `${ALT_PREFIX}${i}`,
        radius: i === 0 ? base : base * 0.7,
      });
    });
    this._updateGraphEmphasis();
  }

  /**
   * Draw another graph over ours - TopologicPy's Navigation graph - so the two
   * estimates of the walkable space can be compared by eye.
   */
  setReferenceGraph(nodes, edges, { color = 0x84cc16 } = {}) {
    if (!nodes?.length || !edges?.length) {
      this._setOverlay(OVERLAY.navGraph, null);
      return;
    }
    const positions = new Float32Array(edges.length * 3);
    for (let i = 0; i < edges.length; i += 1) {
      const n = edges[i] * 3;
      positions[i * 3] = nodes[n];
      positions[i * 3 + 1] = nodes[n + 1];
      positions[i * 3 + 2] = nodes[n + 2];
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.6,
      depthTest: false,
    });
    const segments = new THREE.LineSegments(geometry, material);
    segments.renderOrder = 900;
    this._setOverlay(OVERLAY.navGraph, segments);
  }

  /** Navigation's route, drawn like ours in its own overlay. */
  setReferenceRoute(points, { color = 0x65a30d } = {}) {
    this.setPath(points, { color, name: OVERLAY.navRoute });
  }

  /**
   * The loaded model's coordination matrix, as 16 numbers (column-major).
   *
   * web-ifc moves the model near the origin (COORDINATE_TO_ORIGIN) and That
   * Open records the move. The server needs it to put TopologicPy's graph,
   * which lives in IFC world coordinates, into this frame: exactly for an
   * unrotated model, to about 1e-3 for a rotated one, since That Open rounds
   * the rotation (formal/fireegress Frames.lean).
   */
  async getCoordinationMatrix() {
    try {
      const matrix = await this.model?.getCoordinationMatrix?.();
      return matrix?.elements ? Array.from(matrix.elements) : null;
    } catch {
      return null;
    }
  }

  /**
   * Fade the graph back when a route is on screen.
   *
   * 17k edges at full strength drown out the one line the user actually asked
   * for, so the graph steps aside while a route is displayed. A layer that
   * colours the graph (heat, burning, deadlines) is itself what is being
   * read, and keeps its own strength; when it is removed the graph goes back
   * to the strength the routes call for, not always the full one (review m11).
   */
  _updateGraphEmphasis() {
    const segments = this.overlays.get(OVERLAY.graph);
    if (!segments) return;
    const hasPath = [...this.overlays.keys()].some(
      (name) =>
        name === OVERLAY.path ||
        name === OVERLAY.dynamicPath ||
        name === OVERLAY.navRoute ||
        name.startsWith(ALT_PREFIX),
    );
    segments.material.opacity = this._colouring ? this._colouring.opacity : hasPath ? 0.16 : 0.45;
    segments.material.needsUpdate = true;
  }

  /**
   * Draw the navigation graph from typed arrays.
   *
   * One LineSegments with a vertex-colour attribute, so recolouring for the
   * fire overlay is an in-place buffer write rather than a rebuild.
   */
  setGraph(nodes, edges, kinds) {
    if (!nodes?.length || !edges?.length) {
      this._setOverlay(OVERLAY.graph, null);
      this._graphNodes = null;
      this._graphEdges = null;
      return;
    }

    const edgeCount = edges.length / 2;
    const positions = new Float32Array(edgeCount * 6);
    const colors = new Float32Array(edgeCount * 6);

    const base = new THREE.Color(0x3b82f6);
    const stair = new THREE.Color(0x8b5cf6);
    const door = new THREE.Color(0xf59e0b);
    const lift = new THREE.Color(0x14b8a6);
    const pick = (kind) => (kind === 1 ? stair : kind === 2 ? door : kind === 3 ? lift : base);

    for (let e = 0; e < edgeCount; e += 1) {
      const a = edges[e * 2];
      const b = edges[e * 2 + 1];
      const o = e * 6;
      positions[o] = nodes[a * 3];
      positions[o + 1] = nodes[a * 3 + 1];
      positions[o + 2] = nodes[a * 3 + 2];
      positions[o + 3] = nodes[b * 3];
      positions[o + 4] = nodes[b * 3 + 1];
      positions[o + 5] = nodes[b * 3 + 2];

      const ca = pick(kinds?.[a] ?? 0);
      const cb = pick(kinds?.[b] ?? 0);
      colors[o] = ca.r;
      colors[o + 1] = ca.g;
      colors[o + 2] = ca.b;
      colors[o + 3] = cb.r;
      colors[o + 4] = cb.g;
      colors[o + 5] = cb.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.45,
    });
    const segments = new THREE.LineSegments(geometry, material);
    this._setOverlay(OVERLAY.graph, segments);

    this._graphNodes = nodes;
    this._graphEdges = edges;
    this._graphKinds = kinds;
    this._baseColors = colors.slice();
    this._colouring = null;
    this._updateGraphEmphasis();
  }

  /** The heat ramp and the deadline map's "none" for the scene's background. */
  _sceneColours() {
    return this._theme === "dark" ? SCENE_COLOURS.dark : SCENE_COLOURS.light;
  }

  /**
   * Put the graph back in its own colours, or note the layer now colouring
   * it: its strength, and how to redraw it in the other theme's ramp.
   */
  _setColouring(colouring, colors, attribute) {
    if (!colouring) colors.set(this._baseColors);
    attribute.needsUpdate = true;
    this._colouring = colouring;
    this._updateGraphEmphasis();
  }

  /** Recolour graph edges from a node temperature map, in place. */
  setTemperatures(temperatures, { ambient = 20, peak = 120 } = {}) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (!segments || !this._graphEdges) return;

    const attribute = segments.geometry.getAttribute("color");
    if (!attribute) return;
    const colors = attribute.array;

    if (!temperatures || temperatures.size === 0) {
      this._setColouring(null, colors, attribute);
      return;
    }

    const edgeCount = this._graphEdges.length / 2;
    const scratch = new THREE.Color();
    const { stops } = this._sceneColours();

    for (let e = 0; e < edgeCount; e += 1) {
      const o = e * 6;
      for (let side = 0; side < 2; side += 1) {
        const node = this._graphEdges[e * 2 + side];
        const temp = temperatures.get(node);
        const offset = o + side * 3;
        if (temp === undefined || temp <= ambient + 1) {
          colors[offset] = this._baseColors[offset];
          colors[offset + 1] = this._baseColors[offset + 1];
          colors[offset + 2] = this._baseColors[offset + 2];
        } else {
          heatColor(scratch, temp, ambient, peak, stops);
          colors[offset] = scratch.r;
          colors[offset + 1] = scratch.g;
          colors[offset + 2] = scratch.b;
        }
      }
    }
    this._setColouring(
      { opacity: 0.75, redraw: () => this.setTemperatures(temperatures, { ambient, peak }) },
      colors,
      attribute,
    );
  }

  /**
   * Recolour graph edges by each node's escape deadline (seconds): the heat
   * ramp reversed, so little time left reads as hot. Grey: no departure on
   * the grid gets out within the dose limit (-Infinity). null restores the
   * graph's own colours, and so does a map that is not one value per node of
   * this graph (one made before a rebuild: indexed by the new graph's nodes it
   * read past its end and threw, review C1).
   */
  setDeadlines(deadlines, { max = 300 } = {}) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (!segments || !this._graphEdges) return;
    const attribute = segments.geometry.getAttribute("color");
    if (!attribute) return;
    const colors = attribute.array;
    if (!paintDeadlines(colors, this._graphEdges, this._graphNodes.length / 3, deadlines, { max, ...this._sceneColours() })) {
      this._setColouring(null, colors, attribute);
      return;
    }
    this._setColouring({ opacity: 0.8, redraw: () => this.setDeadlines(deadlines, { max }) }, colors, attribute);
  }

  /**
   * How often the view draws. 0 (the default) draws every frame. While the
   * fire solver computes on the GPU it shares it with this view: drawing
   * `fps` times a second, and whenever the camera moves, gives the solver
   * back most of what drawing took (Duplex, 120 s fire on an integrated GPU:
   * 72 s with drawing every frame, 53 s with none).
   */
  setDrawRate(fps) {
    const renderer = this.world?.renderer;
    clearInterval(this._drawTimer);
    this._drawTimer = null;
    if (this._onCameraMove) this.world?.camera?.controls?.removeEventListener?.("update", this._onCameraMove);
    if (!renderer) return;
    if (!(fps > 0)) {
      renderer.mode = RendererMode.AUTO;
      return;
    }
    renderer.mode = RendererMode.MANUAL;
    this._onCameraMove ||= () => {
      if (this.world?.renderer) this.world.renderer.needsUpdate = true;
    };
    this.world.camera?.controls?.addEventListener?.("update", this._onCameraMove);
    this._drawTimer = setInterval(() => {
      renderer.needsUpdate = true;
    }, 1000 / fps);
    renderer.needsUpdate = true;
  }

  /**
   * The smoke as a cloud (smokeCloud in core/flamescapesRun.js): one soft grey
   * point per block of cells, as opaque as light crossing the block finds it,
   * pale where thin and darker where thick. `strength` scales it: lighter in
   * the heat view, so the graph's colours show through. One layer for the
   * whole run, rewritten in place like the particles.
   */
  setSmokeCloud(cloud, { strength = 1 } = {}) {
    const count = cloud?.opacity?.length || 0;
    if (!count) {
      this._setOverlay(OVERLAY.smoke, null);
      return;
    }
    let points = this.overlays.get(OVERLAY.smoke);
    if (!points || points.userData.capacity < count) {
      const capacity = Math.max(2048, count, 2 * (points?.userData.capacity || 0));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(THREE.DynamicDrawUsage));
      // Four components: three.js reads the fourth as each point's alpha.
      geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage));
      const material = new THREE.PointsMaterial({
        size: 1,
        map: this._glowSprite || (this._glowSprite = glowTexture()),
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        sizeAttenuation: true,
      });
      points = new THREE.Points(geometry, material);
      points.renderOrder = 9;
      points.frustumCulled = false;
      points.userData.capacity = capacity;
      this._setOverlay(OVERLAY.smoke, points);
    }
    // A sprite fades to its edge, so twice the block and a little more closes the gaps.
    points.material.size = cloud.size * 2.2;
    points.material.opacity = strength;
    const position = points.geometry.getAttribute("position");
    const color = points.geometry.getAttribute("color");
    position.array.set(cloud.positions);
    const colors = color.array;
    for (let q = 0; q < count; q += 1) {
      const alpha = cloud.opacity[q] / 255;
      const shade = 0.6 - 0.4 * alpha;
      colors[q * 4] = shade;
      colors[q * 4 + 1] = shade;
      colors[q * 4 + 2] = shade;
      colors[q * 4 + 3] = alpha;
    }
    position.needsUpdate = true;
    color.needsUpdate = true;
    points.geometry.setDrawRange(0, count);
  }

  /**
   * Recolour graph edges by visibility through smoke (metres, one value per
   * node): clear air keeps the graph's own colour, and the edge greys and
   * darkens as visibility falls from `clear` to `dark` metres.
   */
  setSmoke(visibility, { clear = 30, dark = 1 } = {}) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (!segments || !this._graphEdges) return;
    const attribute = segments.geometry.getAttribute("color");
    if (!attribute) return;
    const colors = attribute.array;
    if (!visibility || visibility.length !== this._graphNodes.length / 3) {
      this._setColouring(null, colors, attribute);
      return;
    }
    const light = this._theme === "dark" ? new THREE.Color(0xd4d4d8) : new THREE.Color(0xa1a1aa);
    const black = this._theme === "dark" ? new THREE.Color(0x52525b) : new THREE.Color(0x18181b);
    const scratch = new THREE.Color();
    const edgeCount = this._graphEdges.length / 2;
    const span = Math.log(clear / dark);
    for (let e = 0; e < edgeCount; e += 1) {
      for (let side = 0; side < 2; side += 1) {
        const node = this._graphEdges[e * 2 + side];
        const offset = e * 6 + side * 3;
        const v = visibility[node];
        if (!(v < clear)) {
          colors[offset] = this._baseColors[offset];
          colors[offset + 1] = this._baseColors[offset + 1];
          colors[offset + 2] = this._baseColors[offset + 2];
          continue;
        }
        // Darker as visibility falls, on a log scale (10 m is a third of the way).
        const t = Math.min(1, Math.max(0, Math.log(clear / Math.max(v, dark)) / span));
        scratch.copy(light).lerp(black, t);
        colors[offset] = scratch.r;
        colors[offset + 1] = scratch.g;
        colors[offset + 2] = scratch.b;
      }
    }
    this._setColouring({ opacity: 0.85, redraw: () => this.setSmoke(visibility, { clear, dark }) }, colors, attribute);
  }

  /**
   * Draw the Flamescapes solver's particles as glowing points: flame-front
   * particles larger and dimmer, embers small and bright, both coloured by
   * temperature from dull red (800 K) to yellow-white (1800 K). null clears.
   */
  setParticles(snapshot) {
    if (!snapshot || !snapshot.temps?.length) {
      this._setOverlay(OVERLAY.flames, null);
      this._setOverlay(OVERLAY.embers, null);
      return;
    }
    const sprite = this._glowSprite || (this._glowSprite = glowTexture());
    for (const [kind, name, size, opacity] of [
      [0, OVERLAY.flames, 0.45, 0.5],
      [1, OVERLAY.embers, 0.12, 1],
    ]) {
      let count = 0;
      for (let q = 0; q < snapshot.kinds.length; q += 1) if (snapshot.kinds[q] === kind) count += 1;
      // One layer per kind for the whole run: a frame rewrites its buffers and
      // reallocates only when the count outgrows them (frames arrive every few
      // seconds of playback, each with thousands of particles).
      let points = this.overlays.get(name);
      if (!points || points.userData.capacity < count) {
        const capacity = Math.max(1024, count, 2 * (points?.userData.capacity || 0));
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(THREE.DynamicDrawUsage));
        geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(THREE.DynamicDrawUsage));
        const material = new THREE.PointsMaterial({
          size,
          map: sprite,
          vertexColors: true,
          transparent: true,
          opacity,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          sizeAttenuation: true,
        });
        points = new THREE.Points(geometry, material);
        points.renderOrder = 10;
        // The particles move every frame; culling would need a new bounding sphere each time.
        points.frustumCulled = false;
        points.userData.capacity = capacity;
        this._setOverlay(name, points);
      }
      const position = points.geometry.getAttribute("position");
      const color = points.geometry.getAttribute("color");
      const positions = position.array;
      const colors = color.array;
      const scratch = new THREE.Color();
      let o = 0;
      for (let q = 0; q < snapshot.kinds.length; q += 1) {
        if (snapshot.kinds[q] !== kind) continue;
        positions.set(snapshot.positions.subarray(q * 3, q * 3 + 3), o * 3);
        glowColor(scratch, snapshot.temps[q]);
        colors[o * 3] = scratch.r;
        colors[o * 3 + 1] = scratch.g;
        colors[o * 3 + 2] = scratch.b;
        o += 1;
      }
      position.needsUpdate = true;
      color.needsUpdate = true;
      points.geometry.setDrawRange(0, count);
    }
  }

  /** Recolour graph edges from a set of burning node indices. */
  setBurningNodes(burning) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (!segments || !this._graphEdges) return;
    const attribute = segments.geometry.getAttribute("color");
    if (!attribute) return;
    const colors = attribute.array;

    if (!burning || burning.size === 0) {
      this._setColouring(null, colors, attribute);
      return;
    }

    const fire = new THREE.Color(0xff4500);
    const edgeCount = this._graphEdges.length / 2;
    for (let e = 0; e < edgeCount; e += 1) {
      const o = e * 6;
      for (let side = 0; side < 2; side += 1) {
        const node = this._graphEdges[e * 2 + side];
        const offset = o + side * 3;
        if (burning.has(node)) {
          colors[offset] = fire.r;
          colors[offset + 1] = fire.g;
          colors[offset + 2] = fire.b;
        } else {
          colors[offset] = this._baseColors[offset];
          colors[offset + 1] = this._baseColors[offset + 1];
          colors[offset + 2] = this._baseColors[offset + 2];
        }
      }
    }
    this._setColouring({ opacity: 0.7 }, colors, attribute);
  }

  setGraphVisible(visible) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (segments) segments.visible = visible;
  }

  setModelVisible(visible) {
    if (this.modelObject) this.modelObject.visible = visible;
  }

  /**
   * How the IFC geometry is drawn: "solid", "ghost" or "hidden".
   *
   * Ghost mode exists because of a plain usability problem: the navigation
   * graph lives inside the building, so with solid geometry the user builds a
   * graph and sees nothing change - the roof and walls hide all of it. Ghosting
   * keeps the building as context while letting the graph read through.
   */
  setModelAppearance(mode) {
    const object = this.modelObject;
    if (!object) return;

    if (mode === "hidden") {
      object.visible = false;
      return;
    }
    object.visible = true;

    const ghost = mode === "ghost";
    object.traverse((child) => {
      if (!child.isMesh && !child.isLineSegments) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        if (!material) continue;
        // Remember the material's own settings once, so leaving ghost mode
        // restores exactly what the loader configured rather than a guess.
        if (material.userData.__originalOpacity === undefined) {
          material.userData.__originalOpacity = material.opacity;
          material.userData.__originalTransparent = material.transparent;
          material.userData.__originalDepthWrite = material.depthWrite;
        }
        if (ghost) {
          material.transparent = true;
          material.opacity = 0.18;
          // Without this the ghosted shell still occludes the graph behind it.
          material.depthWrite = false;
        } else {
          material.opacity = material.userData.__originalOpacity;
          material.transparent = material.userData.__originalTransparent;
          material.depthWrite = material.userData.__originalDepthWrite;
        }
        material.needsUpdate = true;
      }
    });
    if (this._section) this._applyClip(object);
  }

  setMarker(name, point) {
    const marker = this.markers[name];
    if (!marker) return;
    if (!point || point.length < 3) {
      marker.visible = false;
      return;
    }
    marker.position.set(point[0], point[1], point[2]);
    marker.visible = true;
  }

  // ------------------------------------------------------------------ camera

  fitToModel() {
    const box = this._modelBox();
    if (!box) return;
    this._frame(box);
  }

  fitToPoints(points) {
    if (!points?.length) return;
    const box = new THREE.Box3();
    points.forEach((p) => box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2])));
    if (box.isEmpty()) return;
    box.expandByScalar(2);
    this._frame(box);
  }

  _modelBox() {
    if (this.modelObject) {
      const box = new THREE.Box3().setFromObject(this.modelObject);
      if (!box.isEmpty()) return box;
    }
    if (this.model?.box instanceof THREE.Box3 && !this.model.box.isEmpty()) {
      return this.model.box.clone();
    }
    return null;
  }

  _frame(box) {
    const controls = this.world?.camera?.controls;
    const camera = this.world?.camera?.three;
    if (!camera) return;

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const extent = Math.max(size.x, size.y, size.z) || 10;

    // Keep the far plane ahead of the model or large sites clip away.
    if (camera.far < extent * 12) {
      camera.far = extent * 12;
      camera.near = Math.max(0.01, extent / 5000);
      camera.updateProjectionMatrix();
    }
    if (this.grid) {
      this.grid.position.set(center.x, box.min.y - 0.05, center.z);
    }

    // Keep the current direction (the iso one on first load).
    const dir = camera.position.clone().sub(controls?.getTarget?.(new THREE.Vector3()) || center);
    if (dir.lengthSq() < 1e-9) dir.set(1, 0.8, 1);
    this._fitAlong(box, dir);
  }

  /**
   * Look at the box from `dir` (towards the camera), close enough that every
   * corner is in view. camera-controls' fitToBox snaps the camera to the
   * nearest axis, which turned the iso view into a flat elevation; this keeps
   * the direction and solves the distance from the field of view instead.
   */
  _fitAlong(box, dir) {
    const controls = this.world?.camera?.controls;
    const camera = this.world?.camera?.three;
    if (!camera) return;
    const center = box.getCenter(new THREE.Vector3());
    const back = dir.clone().normalize();
    const forward = back.clone().negate();
    const worldUp = Math.abs(back.y) > 0.999 ? new THREE.Vector3(0, 0, -1) : new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(forward, worldUp).normalize();
    const up = new THREE.Vector3().crossVectors(right, forward).normalize();
    const vHalf = THREE.MathUtils.degToRad(camera.fov || 50) / 2;
    const hHalf = Math.atan(Math.tan(vHalf) * (camera.aspect || 1));
    let distance = 0;
    for (const x of [box.min.x, box.max.x]) {
      for (const y of [box.min.y, box.max.y]) {
        for (const z of [box.min.z, box.max.z]) {
          const v = new THREE.Vector3(x, y, z).sub(center);
          const toward = v.dot(back);
          distance = Math.max(
            distance,
            Math.abs(v.dot(right)) / Math.tan(hHalf) + toward,
            Math.abs(v.dot(up)) / Math.tan(vHalf) + toward,
          );
        }
      }
    }
    distance = Math.max(distance * 1.06, 1);
    const eye = center.clone().addScaledVector(back, distance);
    if (controls?.setLookAt) {
      controls.setLookAt(eye.x, eye.y, eye.z, center.x, center.y, center.z, true);
    } else {
      camera.position.copy(eye);
      camera.lookAt(center);
    }
  }

  /** Snap to an axis view. `which` is one of top/front/back/left/right/iso. */
  setStandardView(which) {
    const box = this._modelBox() || new THREE.Box3(
      new THREE.Vector3(-5, -5, -5),
      new THREE.Vector3(5, 5, 5),
    );
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    const directions = {
      top: [0, 1, 0.0001],
      front: [0, 0, 1],
      back: [0, 0, -1],
      left: [-1, 0, 0],
      right: [1, 0, 0],
      iso: [1, 0.8, 1],
    };
    const dir = directions[which] || directions.iso;
    this._fitAlong(box, new THREE.Vector3(...dir));
  }
}

/**
 * Resolve the fragments worker URL, preferring the self-hosted copy.
 *
 * `FragmentsManager.getWorker()` fetches the worker from
 * `unpkg.com/@thatopen/fragments@<version>/dist/worker/worker.mjs` at runtime.
 * That makes booting the viewer depend on a third-party CDN, breaks offline and
 * air-gapped use, and when the fetch fails the symptom is a bare
 * `Uncaught SyntaxError: Unexpected token '<'` from the worker — because the
 * failed request resolved to an HTML error page.
 *
 * `npm run sync-assets` copies the matching worker into public/, so the normal
 * path is a same-origin fetch. The CDN call is kept only as a last resort.
 */
async function resolveWorkerUrl() {
  try {
    const response = await fetch(WORKER_PATH, { method: "HEAD" });
    const type = response.headers.get("content-type") || "";
    if (response.ok && !type.includes("text/html")) {
      return new URL(WORKER_PATH, window.location.href).href;
    }
    console.warn(
      `[viewer] ${WORKER_PATH} is missing or is not a script. ` +
        "Run `npm run sync-assets`. Falling back to the CDN copy.",
    );
  } catch (error) {
    console.warn(`[viewer] Could not reach ${WORKER_PATH}:`, error);
  }

  const fallback = await FragmentsManager.getWorker();
  if (typeof fallback !== "string" || !fallback) {
    throw new Error(
      "Could not obtain the fragments worker. Run `npm run sync-assets` in " +
        "topologicpy-web-frontend so it is served from /fragments/worker.mjs.",
    );
  }
  return fallback;
}
