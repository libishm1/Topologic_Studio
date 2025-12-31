// src/IFCViewer.jsx
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  Components,
  Worlds,
  SimpleScene,
  SimpleCamera,
  SimpleRenderer,
  FragmentsManager,
  IfcLoader,
} from "@thatopen/components";

const DEFAULT_WASM_PATH = "https://unpkg.com/web-ifc@0.0.73/";

const fitCameraToModel = (world, model, object) => {
  const camera = world.camera?.three;
  if (!camera) return;
  const controls = world.camera?.controls || null;

  let box = null;
  if (object) {
    const candidate = new THREE.Box3().setFromObject(object);
    if (!candidate.isEmpty()) {
      box = candidate;
    }
  }
  if (!box && model?.box instanceof THREE.Box3 && !model.box.isEmpty()) {
    box = model.box.clone();
  }
  if (!box) return;

  const size = new THREE.Vector3();
  box.getSize(size);
  const center = new THREE.Vector3();
  box.getCenter(center);

  const maxSize = Math.max(size.x, size.y, size.z);
  const fitDistance = maxSize * 1.6 || 10;
  const offset = new THREE.Vector3(1, 1, 1).normalize().multiplyScalar(fitDistance);

  if (camera.far < fitDistance * 10) {
    camera.far = fitDistance * 10;
    camera.updateProjectionMatrix();
  }

  if (controls?.fitToBox) {
    controls.fitToBox(box, true);
    return;
  }

  camera.position.copy(center.clone().add(offset));
  if (controls?.target) {
    controls.target.copy(center);
    controls.update();
  } else if (camera.lookAt) {
    camera.lookAt(center);
  }
};

const resolveUpAxis = (object, upAxis) => {
  if (upAxis === "y" || upAxis === "z") return upAxis;
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return null;
  const size = new THREE.Vector3();
  box.getSize(size);
  if (size.z > size.y * 1.2) return "z";
  return "y";
};

const applyUpAxis = (object, axis, flipY, flipZ) => {
  object.rotation.set(0, 0, 0);
  object.scale.set(1, 1, 1);
  if (axis === "z") {
    object.rotation.set(Math.PI / 2, 0, 0);
  }
  if (flipY) {
    object.scale.y = -1;
  }
  if (flipZ) {
    object.scale.z = -1;
  }
  object.updateMatrixWorld(true);
};

const forceDoubleSide = (object) => {
  object.traverse((child) => {
    if (!child.isMesh || !child.material) return;
    if (Array.isArray(child.material)) {
      child.material.forEach((mat) => {
        mat.side = THREE.DoubleSide;
        mat.needsUpdate = true;
      });
    } else {
      child.material.side = THREE.DoubleSide;
      child.material.needsUpdate = true;
    }
  });
};

const waitForModelReady = async (model, fragments, maxMs = 8000) => {
  if (!model) return;
  const start = performance.now();
  while (model.isBusy && performance.now() - start < maxMs) {
    await fragments.core.update(true);
    await new Promise((resolve) => setTimeout(resolve, 60));
  }
};

export default function IFCViewer({
  file,
  pickMode,
  onPick,
  startPoint,
  exitPoint,
  upAxis = "auto",
  invertOrbit = false,
  flipY = false,
  flipZ = false,
}) {
  const containerRef = useRef(null);
  const componentsRef = useRef(null);
  const worldRef = useRef(null);
  const modelRef = useRef(null);
  const modelIdRef = useRef(null);
  const startMarkerRef = useRef(null);
  const exitMarkerRef = useRef(null);
  const pickModeRef = useRef(pickMode);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  useEffect(() => {
    pickModeRef.current = pickMode;
  }, [pickMode]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const components = new Components();
    componentsRef.current = components;

    const worlds = components.get(Worlds);
    const world = worlds.create();
    world.scene = new SimpleScene(components);
    world.renderer = new SimpleRenderer(components, container);
    world.camera = new SimpleCamera(components);
    worldRef.current = world;

    world.scene.three.up.set(0, 1, 0);
    const threeCamera = world.camera.three;
    if (threeCamera?.up) {
      threeCamera.up.set(0, 1, 0);
    }

    world.scene.three.background = new THREE.Color(0xf8fafc);
    world.scene.three.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(10, 10, 10);
    world.scene.three.add(dir);

    const grid = new THREE.GridHelper(30, 30);
    world.scene.three.add(grid);
    world.scene.three.add(new THREE.AxesHelper(2));

    const startMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x22c55e })
    );
    startMarker.visible = false;
    const exitMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xf97316 })
    );
    exitMarker.visible = false;
    world.scene.three.add(startMarker);
    world.scene.three.add(exitMarker);
    startMarkerRef.current = startMarker;
    exitMarkerRef.current = exitMarker;

    if (world.camera.controls?.up?.set) {
      world.camera.controls.up.set(0, 1, 0);
    }
    if (world.camera.controls) {
      world.camera.controls.azimuthRotateSpeed = invertOrbit ? -1 : 1;
    }
    if (world.camera.controls?.setLookAt) {
      world.camera.controls.setLookAt(10, 10, 10, 0, 0, 0);
      world.camera.controls.update();
    }

    components.init();

    let active = true;
    const initIfc = async () => {
      try {
        const fragments = components.get(FragmentsManager);
        const workerUrl = new URL(
          "@thatopen/fragments/dist/Worker/worker.mjs",
          import.meta.url
        );
        fragments.init(workerUrl);

        const ifcLoader = components.get(IfcLoader);
        const wasmPath = import.meta.env.VITE_WEBIFC_WASM_PATH || DEFAULT_WASM_PATH;
        const absolute =
          wasmPath.startsWith("http") || wasmPath.startsWith("/");
        await ifcLoader.setup({
          autoSetWasm: false,
          wasm: { path: wasmPath, absolute },
        });
        ifcLoader.settings.webIfc.COORDINATE_TO_ORIGIN = false;
        ifcLoader.settings.webIfc.USE_FAST_BOOLS = false;
        ifcLoader.settings.webIfc.OPTIMIZE_PROFILES = false;

        if (active) {
          setReady(true);
        }
      } catch (err) {
        if (!active) return;
        setError(err?.message || "IFC viewer init failed.");
      }
    };

    initIfc();

    return () => {
      active = false;
      components.dispose();
      componentsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const world = worldRef.current;
    if (!world?.camera?.controls) return;
    world.camera.controls.azimuthRotateSpeed = invertOrbit ? -1 : 1;
    world.camera.controls.update();
  }, [invertOrbit]);

  useEffect(() => {
    const startMarker = startMarkerRef.current;
    if (!startMarker) return;
    if (!startPoint || startPoint.length < 3) {
      startMarker.visible = false;
      return;
    }
    startMarker.position.set(startPoint[0], startPoint[1], startPoint[2]);
    startMarker.visible = true;
  }, [startPoint]);

  useEffect(() => {
    const exitMarker = exitMarkerRef.current;
    if (!exitMarker) return;
    if (!exitPoint || exitPoint.length < 3) {
      exitMarker.visible = false;
      return;
    }
    exitMarker.position.set(exitPoint[0], exitPoint[1], exitPoint[2]);
    exitMarker.visible = true;
  }, [exitPoint]);

  useEffect(() => {
    if (!ready) return;
    if (!file) {
      setStatus("idle");
      setError(null);
      return;
    }

    let cancelled = false;
    const loadIfc = async () => {
      setStatus("loading");
      setError(null);
      try {
        const buffer = await file.arrayBuffer();
        if (cancelled) return;
        const components = componentsRef.current;
        const world = worldRef.current;
        if (!components || !world) return;
        const ifcLoader = components.get(IfcLoader);
        const fragments = components.get(FragmentsManager);

        if (modelIdRef.current) {
          try {
            await fragments.core.disposeModel(modelIdRef.current);
          } catch (disposeErr) {
            console.warn("Failed to dispose previous IFC model.", disposeErr);
          }
          modelIdRef.current = null;
        }

        if (modelRef.current) {
          world.scene.three.remove(modelRef.current);
          if (typeof modelRef.current.dispose === "function") {
            modelRef.current.dispose();
          }
          modelRef.current = null;
        }

        const model = await ifcLoader.load(
          new Uint8Array(buffer),
          false,
          file.name
        );
        if (cancelled) return;
        const resolvedModel = model || Array.from(fragments.list.values()).pop();
        if (!resolvedModel) {
          setError("IFC model failed to load.");
          setStatus("idle");
          return;
        }
        resolvedModel.graphicsQuality = 1;
        resolvedModel.frozen = false;
        const object =
          resolvedModel.object ||
          (resolvedModel instanceof THREE.Object3D
            ? resolvedModel
            : resolvedModel?.three || resolvedModel?.mesh || resolvedModel?.group || null);
        if (!object) {
          setError("IFC model has no renderable geometry.");
          setStatus("idle");
          return;
        }
        const axis = resolveUpAxis(object, upAxis);
        if (axis) {
          applyUpAxis(object, axis, flipY, flipZ);
        }
        forceDoubleSide(object);
        const threeCamera = world.camera.three;
        if (threeCamera && resolvedModel.useCamera) {
          resolvedModel.useCamera(threeCamera);
        }
        modelRef.current = object;
        if (resolvedModel.modelId) {
          modelIdRef.current = resolvedModel.modelId;
        }
        world.scene.three.add(object);
        await fragments.core.update(true);
        await new Promise((resolve) => {
          requestAnimationFrame(() => resolve());
        });
        await fragments.core.update(true);
        await waitForModelReady(resolvedModel, fragments);
        fitCameraToModel(world, resolvedModel, object);
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err?.message || "Failed to load IFC.");
        setStatus("idle");
      }
    };

    loadIfc();
    return () => {
      cancelled = true;
    };
  }, [file, ready, upAxis, flipY, flipZ]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handlePointerDown = (event) => {
      if (!pickModeRef.current) return;
      const world = worldRef.current;
      const model = modelRef.current;
      if (!world || !model) return;

      const camera =
        world.camera?.three || world.camera?.camera || world.camera;
      const renderer =
        world.renderer?.three || world.renderer?.renderer || world.renderer;
      if (!camera || !renderer) return;

      const rect = container.getBoundingClientRect();
      const pointer = new THREE.Vector2();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(model, true);
      if (!hits.length) return;

      const hit = hits[0];
      const point = [hit.point.x, hit.point.y, hit.point.z];
      if (pickModeRef.current === "start") {
        const marker = startMarkerRef.current;
        if (marker) {
          marker.position.set(point[0], point[1], point[2]);
          marker.visible = true;
        }
      } else if (pickModeRef.current === "exit") {
        const marker = exitMarkerRef.current;
        if (marker) {
          marker.position.set(point[0], point[1], point[2]);
          marker.visible = true;
        }
      }
      onPick?.(pickModeRef.current, point, hit);
    };

    container.addEventListener("pointerdown", handlePointerDown);
    return () => {
      container.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [onPick]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
      {!file && (
        <div className="viewer-placeholder">
          <div className="viewer-placeholder-card">
            <h2>IFC viewer (Fragments)</h2>
            <p>Load an IFC file to view it with the That Open engine.</p>
          </div>
        </div>
      )}
      {status === "loading" && (
        <div className="viewer-overlay">
          <span>Loading IFC...</span>
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}
    </div>
  );
}
