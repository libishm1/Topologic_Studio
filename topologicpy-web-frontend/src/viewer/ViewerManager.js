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
  SimpleCamera,
  SimpleRenderer,
  SimpleScene,
  Worlds,
} from "@thatopen/components";

import { hashFile, readCached, writeCached } from "../lib/fragmentCache.js";
import { PhaseTimer, mark } from "../lib/perf.js";
import { collectCategoryIds, extractMeshes } from "./categories.js";

/** Self-hosted by default; `npm run sync-wasm` keeps it matching web-ifc. */
const WASM_PATH = import.meta.env.VITE_WEBIFC_WASM_PATH || "/wasm/";

const OVERLAY = {
  path: "path",
  dynamicPath: "dynamicPath",
  graph: "graph",
  rlPath: "rlPath",
};

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
    world.camera = new SimpleCamera(components);
    this.world = world;

    const scene = world.scene.three;
    scene.up.set(0, 1, 0);
    if (world.camera.three?.up) world.camera.three.up.set(0, 1, 0);

    this._buildEnvironment(scene);
    this._buildMarkers(scene);

    components.init();

    this.raycaster = components.get(Raycasters).get(world);

    world.camera.controls?.setLookAt(12, 10, 12, 0, 0, 0);
    world.camera.controls?.update?.(0);

    this._attachPointer(container);
    this._attachResize(container);
    this.applyTheme(theme);

    const fragments = components.get(FragmentsManager);
    // getWorker() hands back a blob URL for the worker matching the installed
    // @thatopen/fragments build. Classic pointed at the package's internal
    // dist path with `new URL(...)`, which 3.4 no longer exports at all.
    fragments.init(await FragmentsManager.getWorker());
    this.fragments = fragments;

    const ifcLoader = components.get(IfcLoader);
    await ifcLoader.setup({
      autoSetWasm: false,
      wasm: { path: WASM_PATH, absolute: WASM_PATH.startsWith("http") },
    });
    ifcLoader.settings.webIfc.COORDINATE_TO_ORIGIN = true;
    ifcLoader.settings.webIfc.OPTIMIZE_PROFILES = true;
    this.ifcLoader = ifcLoader;

    this.ready = true;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.ready = false;

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
    const make = (color) => {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 20, 20),
        new THREE.MeshBasicMaterial({ color, depthTest: false }),
      );
      mesh.renderOrder = 999;
      mesh.visible = false;
      mesh.userData.pickIgnore = true;
      scene.add(mesh);
      return mesh;
    };
    this.markers.start = make(0x22c55e);
    this.markers.exit = make(0xf97316);
    this.markers.fire = make(0xff4500);
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
   * @returns {Promise<{cached: boolean, phases: object, stats: object}>}
   */
  async loadIfc(file, { onProgress, signal } = {}) {
    if (!this.ready) throw new Error("Viewer is still starting up.");

    const timer = new PhaseTimer("ifc.load");
    const report = (stage, detail) => onProgress?.({ stage, detail });

    report("reading", file.name);
    const { key, buffer } = await timer.run("hash", () => hashFile(file));
    if (signal?.aborted) throw new Error("cancelled");

    this.clearModel();

    let fragmentBytes = await timer.run("cacheLookup", () => readCached(key));
    const cached = Boolean(fragmentBytes);

    if (!cached) {
      report("converting", "Converting IFC to fragments");
      fragmentBytes = await timer.run("convert", async () => {
        const model = await this.ifcLoader.load(
          new Uint8Array(buffer),
          false,
          file.name,
          { processData: { progressCallback: (p) => report("converting", p) } },
        );
        this._adoptModel(model);
        // Serialise what we just built so the next load skips the parse.
        return model.getBuffer(false);
      });

      report("caching", "Saving to local cache");
      void timer
        .run("cacheWrite", () =>
          writeCached(key, fragmentBytes, {
            name: file.name,
            bytes: file.size,
          }),
        )
        .catch(() => {});
    } else {
      report("cached", "Loading cached fragments");
      const model = await timer.run("loadFragments", () =>
        this.fragments.core.load(fragmentBytes, {
          modelId: `${file.name}-${key.slice(0, 8)}`,
          raw: false,
        }),
      );
      this._adoptModel(model);
    }

    if (signal?.aborted) throw new Error("cancelled");

    report("rendering", "Drawing the model");
    await timer.run("firstFrame", async () => {
      await this.fragments.core.update(true);
      this.fitToModel();
    });

    // Categories come from the fragments model, not a second IFC parse.
    report("indexing", "Reading IFC categories");
    this.categoryIds = await timer.run("categories", () =>
      collectCategoryIds(this.model),
    );

    const phases = timer.finish({
      cached,
      file: file.name,
      bytes: file.size,
    });

    return {
      cached,
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

  clearModel() {
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
    if (!this.model || !this.categoryIds) {
      throw new Error("Load an IFC model first.");
    }
    const done = mark("ifc.extractGeometry");
    const world = this.modelObject?.matrixWorld?.clone() || null;
    const ids = this.categoryIds;

    const [floors, stairs, doors, walls] = await Promise.all([
      extractMeshes(this.model, ids.floors, world),
      extractMeshes(this.model, ids.stairs, world),
      extractMeshes(this.model, ids.doors, world),
      extractMeshes(this.model, ids.walls, world),
    ]);

    done({
      floors: floors.length,
      stairs: stairs.length,
      doors: doors.length,
      walls: walls.length,
    });
    return { floors, stairs, doors, walls };
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
      this.world.scene.three.add(object);
      this.overlays.set(name, object);
    } else {
      this.overlays.delete(name);
    }
  }

  clearOverlays() {
    this.overlays.forEach((object) => this._destroyObject(object));
    this.overlays.clear();
  }

  setPath(points, { color = 0xdc2626, width = 4, dynamic = false } = {}) {
    const name = dynamic ? OVERLAY.dynamicPath : OVERLAY.path;
    if (!points || points.length < 2) {
      this._setOverlay(name, null);
      return;
    }
    const vectors = points
      .filter((p) => p && p.length >= 3)
      .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (vectors.length < 2) {
      this._setOverlay(name, null);
      return;
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(vectors);
    const material = new THREE.LineBasicMaterial({
      color,
      linewidth: width,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    });
    const line = new THREE.Line(geometry, material);
    line.renderOrder = 900;
    this._setOverlay(name, line);
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
    const pick = (kind) => (kind === 1 ? stair : kind === 2 ? door : base);

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
  }

  /** Recolour graph edges from a node temperature map, in place. */
  setTemperatures(temperatures, { ambient = 20, peak = 120 } = {}) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (!segments || !this._graphEdges) return;

    const attribute = segments.geometry.getAttribute("color");
    if (!attribute) return;
    const colors = attribute.array;

    if (!temperatures || temperatures.size === 0) {
      colors.set(this._baseColors);
      attribute.needsUpdate = true;
      segments.material.opacity = 0.45;
      return;
    }

    const edgeCount = this._graphEdges.length / 2;
    const scratch = new THREE.Color();

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
          heatColor(scratch, temp, ambient, peak);
          colors[offset] = scratch.r;
          colors[offset + 1] = scratch.g;
          colors[offset + 2] = scratch.b;
        }
      }
    }
    attribute.needsUpdate = true;
    segments.material.opacity = 0.75;
  }

  /** Recolour graph edges from a set of burning node indices. */
  setBurningNodes(burning) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (!segments || !this._graphEdges) return;
    const attribute = segments.geometry.getAttribute("color");
    if (!attribute) return;
    const colors = attribute.array;

    if (!burning || burning.size === 0) {
      colors.set(this._baseColors);
      attribute.needsUpdate = true;
      segments.material.opacity = 0.45;
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
    attribute.needsUpdate = true;
    segments.material.opacity = 0.7;
  }

  setGraphVisible(visible) {
    const segments = this.overlays.get(OVERLAY.graph);
    if (segments) segments.visible = visible;
  }

  setModelVisible(visible) {
    if (this.modelObject) this.modelObject.visible = visible;
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

    if (controls?.fitToBox) {
      controls.fitToBox(box, true, { paddingTop: 1, paddingBottom: 1 });
      return;
    }
    const offset = new THREE.Vector3(1, 0.8, 1).normalize().multiplyScalar(extent * 1.7);
    camera.position.copy(center.clone().add(offset));
    camera.lookAt(center);
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
    const distance = (Math.max(size.x, size.y, size.z) || 10) * 1.8;

    const directions = {
      top: [0, 1, 0.0001],
      front: [0, 0, 1],
      back: [0, 0, -1],
      left: [-1, 0, 0],
      right: [1, 0, 0],
      iso: [1, 0.8, 1],
    };
    const dir = directions[which] || directions.iso;
    const eye = new THREE.Vector3(...dir).normalize().multiplyScalar(distance).add(center);

    const controls = this.world?.camera?.controls;
    if (controls?.setLookAt) {
      controls.setLookAt(eye.x, eye.y, eye.z, center.x, center.y, center.z, true);
    } else if (this.world?.camera?.three) {
      this.world.camera.three.position.copy(eye);
      this.world.camera.three.lookAt(center);
    }
  }
}

/** Blue -> cyan -> green -> yellow -> red across the temperature range. */
function heatColor(target, temp, min, max) {
  const t = Math.max(0, Math.min(1, (temp - min) / Math.max(1e-6, max - min)));
  if (t < 0.25) return target.setRGB(0, t / 0.25, 1);
  if (t < 0.5) return target.setRGB(0, 1, 1 - (t - 0.25) / 0.25);
  if (t < 0.75) return target.setRGB((t - 0.5) / 0.25, 1, 0);
  return target.setRGB(1, 1 - (t - 0.75) / 0.25, 0);
}
