// src/IFCViewer.jsx
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  IFCSLAB,
  IFCSLABSTANDARDCASE,
  IFCSLABELEMENTEDCASE,
  IFCSTAIR,
  IFCSTAIRFLIGHT,
  IFCCOVERING,
  IFCDOOR,
  IFCSPACE, 
  IFCWALL,
  IFCBUILDINGSTOREY,
} from "web-ifc";
import {
  Components,
  Worlds,
  SimpleScene,
  SimpleCamera,
  SimpleRenderer,
  FragmentsManager,
  IfcLoader,
  Raycasters,
} from "@thatopen/components";

const DEFAULT_WASM_PATH = "https://unpkg.com/web-ifc@0.0.73/";

const PERF_LOG = (import.meta.env.VITE_PERF_LOG ?? (import.meta.env.DEV ? "1" : "")) === "1";

const perfMark = (label) => {
  if (!PERF_LOG) return () => {};
  const t0 = performance.now();
  return (extra) => {
    const ms = performance.now() - t0;
    if (extra) console.log(`PERF ${label} ${ms.toFixed(1)}ms`, extra);
    else console.log(`PERF ${label} ${ms.toFixed(1)}ms`);
  };
};

const payloadByteSize = (geometryArray) => {
  if (!Array.isArray(geometryArray)) return 0;
  let bytes = 0;
  for (const g of geometryArray) {
    if (g?.vertices?.length) bytes += g.vertices.length * 4;
    if (g?.indices?.length) bytes += g.indices.length * 4;
    if (g?.normals?.length) bytes += g.normals.length * 4;
  }
  return bytes;
};

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

const vectorToArray = (vector) => {
  if (!vector || typeof vector.size !== "function") return [];
  const result = [];
  for (let i = 0; i < vector.size(); i += 1) {
    result.push(vector.get(i));
  }
  return result;
};

const uniqueIds = (ids) => Array.from(new Set(ids));

const isValidIfcModelId = (modelId) =>
  Number.isInteger(modelId) && modelId >= 0;

const temperatureToColor = (temp, minTemp = 20, maxTemp = 120) => {
  // Map temperature to color gradient: blue (cold) -> green -> yellow -> orange -> red (hot)
  const normalized = Math.max(0, Math.min(1, (temp - minTemp) / (maxTemp - minTemp)));

  if (normalized < 0.25) {
    // Blue to Cyan (20-45°C)
    const t = normalized / 0.25;
    return new THREE.Color(0, t, 1);
  } else if (normalized < 0.5) {
    // Cyan to Green (45-70°C)
    const t = (normalized - 0.25) / 0.25;
    return new THREE.Color(0, 1, 1 - t);
  } else if (normalized < 0.75) {
    // Green to Yellow (70-95°C)
    const t = (normalized - 0.5) / 0.25;
    return new THREE.Color(t, 1, 0);
  } else {
    // Yellow to Red (95-120°C)
    const t = (normalized - 0.75) / 0.25;
    return new THREE.Color(1, 1 - t, 0);
  }
};

const collectIfcIds = (ifcApi, modelId) => {
  const slabTypes = [IFCSLAB, IFCSLABSTANDARDCASE, IFCSLABELEMENTEDCASE];
  const stairTypes = [IFCSTAIR, IFCSTAIRFLIGHT];
  const coveringTypes = [IFCCOVERING];

  const slabIds = slabTypes.flatMap((typeId) =>
    vectorToArray(ifcApi.GetLineIDsWithType(modelId, typeId, true))
  );
  const stairIds = stairTypes.flatMap((typeId) =>
    vectorToArray(ifcApi.GetLineIDsWithType(modelId, typeId, true))
  );
  const coveringIds = coveringTypes.flatMap((typeId) =>
    vectorToArray(ifcApi.GetLineIDsWithType(modelId, typeId, true))
  );
  const doorIds = vectorToArray(
    ifcApi.GetLineIDsWithType(modelId, IFCDOOR, true)
  );
  const spaceIds = vectorToArray(
    ifcApi.GetLineIDsWithType(modelId, IFCSPACE, true)
  );
  const wallIds = vectorToArray(
    ifcApi.GetLineIDsWithType(modelId, IFCWALL, true)
  );
  const storeyIds = vectorToArray(
    ifcApi.GetLineIDsWithType(modelId, IFCBUILDINGSTOREY, true)
  );

  return {
    slabs: uniqueIds(slabIds),
    stairs: uniqueIds(stairIds),
    coverings: uniqueIds(coveringIds),
    doors: uniqueIds(doorIds),
    spaces: uniqueIds(spaceIds),
    walls: uniqueIds(wallIds),
    storeys: uniqueIds(storeyIds),
  };
};

const mergeMeshData = (meshDataList, extraTransform = null) => {
  const positions = [];
  const indices = [];
  const normals = [];
  let indexOffset = 0;

  meshDataList.forEach((meshData) => {
    if (!meshData?.positions || !meshData?.indices) return;
    const baseMatrix = meshData.transform || new THREE.Matrix4();
    const matrix = extraTransform
      ? extraTransform.clone().multiply(baseMatrix)
      : baseMatrix;
    const normalMatrix = new THREE.Matrix3().getNormalMatrix(matrix);
    const pos = meshData.positions;
    const idx = meshData.indices;
    const meshNormals = meshData.normals || null;

    for (let i = 0; i < pos.length; i += 3) {
      const v = new THREE.Vector3(pos[i], pos[i + 1], pos[i + 2]);
      v.applyMatrix4(matrix);
      positions.push(v.x, v.y, v.z);

      if (meshNormals) {
        const rawX = meshNormals[i];
        const rawY = meshNormals[i + 1];
        const rawZ = meshNormals[i + 2];
        const n = new THREE.Vector3(rawX, rawY, rawZ);
        if (meshNormals instanceof Int16Array) {
          n.divideScalar(32767);
        }
        n.applyMatrix3(normalMatrix).normalize();
        normals.push(n.x, n.y, n.z);
      }
    }

    for (let i = 0; i < idx.length; i += 1) {
      indices.push(idx[i] + indexOffset);
    }
    indexOffset += pos.length / 3;
  });

  return {
    vertices: positions,
    indices,
    normals: normals.length ? normals : null,
  };
};

const buildGeometryPayload = async (model, localIds, extraTransform = null) => {
  if (!model || !localIds || !localIds.length) return [];
  if (typeof model.getItemsGeometry !== "function") {
    console.warn("Model does not support getItemsGeometry.");
    return [];
  }
  try {
    const geometrySets = await model.getItemsGeometry(localIds);
    if (!geometrySets || !Array.isArray(geometrySets)) {
      console.warn("Invalid geometry sets returned.");
      return [];
    }

    return localIds
      .map((id, index) => {
        const meshDataList = geometrySets[index] || [];
        const merged = mergeMeshData(meshDataList, extraTransform);
        if (!merged.vertices || merged.vertices.length === 0) {
          return null;
        }
        return {
          expressID: id,
          vertices: merged.vertices,
          indices: merged.indices,
          normals: merged.normals,
        };
      })
      .filter(Boolean);
  } catch (error) {
    console.error("Error in buildGeometryPayload:", error);
    return [];
  }
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
  onEgressDataExtracted,
  pathPoints,
  graphEdges,
  graphEdgeIds,
  egressRequestId = 0,
  startPoint,
  exitPoint,
  upAxis = "auto",
  invertOrbit = false,
  flipY = false,
  flipZ = false,
  meshVisible = true,
  fireNodes = [],
  fireTemperatures = {},
  fireUseTemperature = false,
  dynamicPath = null,
}) {
  const containerRef = useRef(null);
  const componentsRef = useRef(null);
  const worldRef = useRef(null);
  const modelRef = useRef(null);
  const ifcModelRef = useRef(null);
  const modelIdRef = useRef(null);
  const raycasterRef = useRef(null);
  const startMarkerRef = useRef(null);
  const exitMarkerRef = useRef(null);
  const pathLineRef = useRef(null);
  const dynamicPathLineRef = useRef(null);
  const graphLinesRef = useRef(null);
  const sceneReadyRef = useRef(false);
  const egressIdsRef = useRef(null);
  const egressModelIdRef = useRef(null);
  const pickModeRef = useRef(pickMode);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  useEffect(() => {
    pickModeRef.current = pickMode;
  }, [pickMode]);

  useEffect(() => {
    if (!egressRequestId || !onEgressDataExtracted) return;
    const model = ifcModelRef.current;
    const ids = egressIdsRef.current;
    if (!model || !ids) return;

    const run = async () => {
      // Helper: yield to the browser between heavy tasks so the scene stays navigable
      const yieldFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

      try {
        const extraTransform = model.matrixWorld?.clone() || null;
        const doneFloors = perfMark(`getItemsGeometry floors (n=${ids.slabs?.length || 0})`);
        const allFloorGeometry = await buildGeometryPayload(
          model,
          ids.slabs || [],
          extraTransform
        );
        doneFloors({ items: allFloorGeometry.length, bytes: payloadByteSize(allFloorGeometry) });
        await yieldFrame();
        const floorGeometry = allFloorGeometry;

        const doneStairs = perfMark(`getItemsGeometry stairs (n=${ids.stairs?.length || 0})`);
        const stairGeometry = await buildGeometryPayload(
          model,
          ids.stairs || [],
          extraTransform
        );
        doneStairs({ items: stairGeometry.length, bytes: payloadByteSize(stairGeometry) });
        await yieldFrame();
        const doneDoors = perfMark(`getItemsGeometry doors (n=${ids.doors?.length || 0})`);
        const doorGeometry = await buildGeometryPayload(
          model,
          ids.doors || [],
          extraTransform
        );
        doneDoors({ items: doorGeometry.length, bytes: payloadByteSize(doorGeometry) });
        await yieldFrame();
        const doneWalls = perfMark(`getItemsGeometry walls (n=${ids.walls?.length || 0})`);
        const wallGeometry = await buildGeometryPayload(
          model,
          ids.walls || [],
          extraTransform
        );
        doneWalls({ items: wallGeometry.length, bytes: payloadByteSize(wallGeometry) });
        onEgressDataExtracted({
          modelId: egressModelIdRef.current,
          ids,
          floors: floorGeometry,
          stairs: stairGeometry,
          doors: doorGeometry,
          walls: wallGeometry,
        });
      } catch (egressErr) {
        console.warn("IFC egress extraction failed.", egressErr);
        onEgressDataExtracted({
          modelId: egressModelIdRef.current,
          ids,
          floors: [],
          stairs: [],
          doors: [],
          walls: [],
          egressError: egressErr?.message || "IFC egress extraction failed.",
        });
      }
    };

    run();
  }, [egressRequestId, onEgressDataExtracted]);

  useEffect(() => {
    if (!ready || !sceneReadyRef.current) return;
    const world = worldRef.current;
    if (!world) return;
    let scene;
    try {
      scene = world.scene?.three;
    } catch {
      return;
    }
    if (!scene) return;

    if (pathLineRef.current) {
      scene.remove(pathLineRef.current);
      if (pathLineRef.current.geometry) {
        pathLineRef.current.geometry.dispose();
      }
      if (pathLineRef.current.material) {
        pathLineRef.current.material.dispose();
      }
      pathLineRef.current = null;
    }

    if (!pathPoints || pathPoints.length < 2) {
      return;
    }

    const points = pathPoints
      .filter((p) => pathPoints && p && p.length >= 3)
      .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (points.length < 2) return;

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xef4444,
      linewidth: 3,
      transparent: true,
      opacity: 0.9,
    });
    const line = new THREE.Line(geometry, material);
    line.userData.pickIgnore = true;

    // Apply the same transformations as the IFC model (upAxis, flipY, flipZ)
    const axis = resolveUpAxis(null, upAxis);
    if (axis) {
      applyUpAxis(line, axis, flipY, flipZ);
    }

    scene.add(line);
    pathLineRef.current = line;
  }, [pathPoints, ready, upAxis, flipY, flipZ]);

  // Dynamic path rendering (magenta line for hazard-weighted rerouting)
  useEffect(() => {
    if (!ready || !sceneReadyRef.current) return;
    const world = worldRef.current;
    if (!world) return;
    let scene;
    try {
      scene = world.scene?.three;
    } catch {
      return;
    }
    if (!scene) return;

    // Clean up existing dynamic path
    if (dynamicPathLineRef.current) {
      scene.remove(dynamicPathLineRef.current);
      if (dynamicPathLineRef.current.geometry) {
        dynamicPathLineRef.current.geometry.dispose();
      }
      if (dynamicPathLineRef.current.material) {
        dynamicPathLineRef.current.material.dispose();
      }
      dynamicPathLineRef.current = null;
    }

    if (!dynamicPath || dynamicPath.length < 2) {
      return;
    }

    // Create dynamic path line (magenta, distinct from static red path)
    const points = dynamicPath
      .filter((p) => p && p.length >= 3)
      .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (points.length < 2) return;

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xff00ff, // Magenta
      linewidth: 5,
      transparent: true,
      opacity: 0.95,
    });
    const line = new THREE.Line(geometry, material);
    line.userData.pickIgnore = true;

    // Apply transformations
    const axis = resolveUpAxis(null, upAxis);
    if (axis) {
      applyUpAxis(line, axis, flipY, flipZ);
    }

    scene.add(line);
    dynamicPathLineRef.current = line;
  }, [dynamicPath, ready, upAxis, flipY, flipZ]);

  // Build / rebuild graph wire mesh when edges change
  useEffect(() => {
    if (!ready || !sceneReadyRef.current) return;
    const world = worldRef.current;
    if (!world) return;
    let scene;
    try {
      scene = world.scene?.three;
    } catch {
      return;
    }
    if (!scene) return;

    if (graphLinesRef.current) {
      scene.remove(graphLinesRef.current);
      if (graphLinesRef.current.geometry) {
        graphLinesRef.current.geometry.dispose();
      }
      if (graphLinesRef.current.material) {
        graphLinesRef.current.material.dispose();
      }
      graphLinesRef.current = null;
    }

    if (!graphEdges || !Array.isArray(graphEdges) || graphEdges.length === 0) {
      return;
    }

    const positions = [];
    const colors = [];
    const defaultColor = new THREE.Color(0x3b82f6);
    graphEdges.forEach((edge) => {
      if (edge && edge.length === 2) {
        const [p1, p2] = edge;
        if (p1 && p1.length >= 3 && p2 && p2.length >= 3) {
          positions.push(p1[0], p1[1], p1[2]);
          positions.push(p2[0], p2[1], p2[2]);
          // Two vertices per edge segment — both start with default color
          colors.push(defaultColor.r, defaultColor.g, defaultColor.b);
          colors.push(defaultColor.r, defaultColor.g, defaultColor.b);
        }
      }
    });

    if (positions.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.4,
      linewidth: 1,
    });
    const lineSegments = new THREE.LineSegments(geometry, material);
    lineSegments.userData.pickIgnore = true;

    const axis = resolveUpAxis(null, upAxis);
    if (axis) {
      applyUpAxis(lineSegments, axis, flipY, flipZ);
    }

    scene.add(lineSegments);
    graphLinesRef.current = lineSegments;
  }, [graphEdges, ready, upAxis, flipY, flipZ]);

  // Update graph wire colors when fire state changes (temperature or node-based)
  useEffect(() => {
    const lines = graphLinesRef.current;
    if (!lines) return;
    const geom = lines.geometry;
    let colorAttr = geom.getAttribute('color');

    // If color attribute doesn't exist yet (edge rebuild happened), create it
    const posAttr = geom.getAttribute('position');
    if (!posAttr) return;
    const vertCount = posAttr.count;
    if (!colorAttr || colorAttr.count !== vertCount) {
      const arr = new Float32Array(vertCount * 3);
      colorAttr = new THREE.Float32BufferAttribute(arr, 3);
      geom.setAttribute('color', colorAttr);
    }

    const hasEdgeIds = graphEdgeIds && Array.isArray(graphEdgeIds);
    const defaultColor = new THREE.Color(0x3b82f6);
    const tempKeys = fireTemperatures ? Object.keys(fireTemperatures).length : 0;
    const hasTemps = fireUseTemperature && tempKeys > 0;
    const hasNodes = fireNodes && fireNodes.length > 0;

    if ((!hasTemps && !hasNodes) || !hasEdgeIds) {
      // Reset all edges to default blue
      for (let i = 0; i < vertCount; i++) {
        colorAttr.setXYZ(i, defaultColor.r, defaultColor.g, defaultColor.b);
      }
      colorAttr.needsUpdate = true;
      lines.material.opacity = 0.4;
      lines.material.needsUpdate = true;
      return;
    }

    // Clamp loop to whichever is smaller — edge_ids or vertex pairs available
    const maxEdges = Math.min(graphEdgeIds.length, Math.floor(vertCount / 2));

    if (hasTemps) {
      // Temperature mode: gradient coloring per-vertex
      const ambientTemp = 20;
      let vertIdx = 0;
      for (let i = 0; i < maxEdges; i++) {
        const [idA, idB] = graphEdgeIds[i];
        const tA = fireTemperatures[idA] ?? ambientTemp;
        const tB = fireTemperatures[idB] ?? ambientTemp;
        const cA = tA > ambientTemp + 1 ? temperatureToColor(tA) : defaultColor;
        const cB = tB > ambientTemp + 1 ? temperatureToColor(tB) : defaultColor;
        colorAttr.setXYZ(vertIdx, cA.r, cA.g, cA.b);
        colorAttr.setXYZ(vertIdx + 1, cB.r, cB.g, cB.b);
        vertIdx += 2;
      }
    } else {
      // Node-based mode: edges with burning endpoints turn orange/red
      const burningSet = new Set(fireNodes);
      const fireColor = new THREE.Color(0xff4500); // orange-red
      let vertIdx = 0;
      for (let i = 0; i < maxEdges; i++) {
        const [idA, idB] = graphEdgeIds[i];
        const aFire = burningSet.has(idA);
        const bFire = burningSet.has(idB);
        const cA = aFire ? fireColor : defaultColor;
        const cB = bFire ? fireColor : defaultColor;
        colorAttr.setXYZ(vertIdx, cA.r, cA.g, cA.b);
        colorAttr.setXYZ(vertIdx + 1, cB.r, cB.g, cB.b);
        vertIdx += 2;
      }
    }
    colorAttr.needsUpdate = true;
    lines.material.opacity = 0.7;
    lines.material.needsUpdate = true;
  }, [fireTemperatures, fireUseTemperature, fireNodes, graphEdgeIds]);

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

    const raycasters = components.get(Raycasters);
    const raycaster = raycasters.get(world);
    raycasterRef.current = raycaster;

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
    startMarker.userData.isPickMarker = true;
    const exitMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xf97316 })
    );
    exitMarker.visible = false;
    exitMarker.userData.isPickMarker = true;
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
    sceneReadyRef.current = true;

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
        ifcLoader.settings.webIfc.USE_FAST_BOOLS = true;
        ifcLoader.settings.webIfc.OPTIMIZE_PROFILES = true;

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
      sceneReadyRef.current = false;
      components.dispose();
      componentsRef.current = null;
      worldRef.current = null;
      modelRef.current = null;
      ifcModelRef.current = null;
      modelIdRef.current = null;
      raycasterRef.current = null;
    };
  }, []);

  useEffect(() => {
    const world = worldRef.current;
    if (!world?.camera?.controls) return;
    world.camera.controls.azimuthRotateSpeed = invertOrbit ? -1 : 1;
    world.camera.controls.update();
  }, [invertOrbit]);

  useEffect(() => {
    const model = modelRef.current;
    if (!model) return;
    model.visible = meshVisible;
  }, [meshVisible]);

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
        const doneRead = perfMark(`file.arrayBuffer (${file.name}, ${file.size} bytes)`);
        const buffer = await file.arrayBuffer();
        doneRead();
        if (cancelled) return;
        const components = componentsRef.current;
        const world = worldRef.current;
        if (!components || !world) return;
        const ifcLoader = components.get(IfcLoader);
        const fragments = components.get(FragmentsManager);

        if (modelRef.current) {
          world.scene.three.remove(modelRef.current);
          modelRef.current = null;
        }
        ifcModelRef.current = null;
        modelIdRef.current = null;

        const doneLoad = perfMark(`ifcLoader.load (${file.name})`);
        const model = await ifcLoader.load(
          new Uint8Array(buffer),
          false,
          file.name
        );
        doneLoad();
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
        ifcModelRef.current = resolvedModel;
        if (resolvedModel.modelId) {
          modelIdRef.current = resolvedModel.modelId;
        }
        world.scene.three.add(object);
        const doneFirstFrame = perfMark("model visible (fragments update + fit)");
        await fragments.core.update(true);
        fitCameraToModel(world, resolvedModel, object);
        doneFirstFrame();
        // Model is visible and navigable immediately
        setStatus(onEgressDataExtracted ? "processing" : "ready");

        if (onEgressDataExtracted) {
          // Use requestAnimationFrame + setTimeout to let the browser
          // render the scene and register pointer events before heavy work
          requestAnimationFrame(() => {
            setTimeout(async () => {
              if (cancelled) return;
              try {
                const doneIds = perfMark("collectIfcIds");
                const ifcApi = ifcLoader.webIfc;
                const collectIds = (modelId) => {
                  if (!ifcApi || !isValidIfcModelId(modelId)) return null;
                  try {
                    return collectIfcIds(ifcApi, modelId);
                  } catch {
                    return null;
                  }
                };
                let ifcModelId =
                  resolvedModel?.modelId ??
                  resolvedModel?.modelID ??
                  modelIdRef.current ??
                  null;
                let shouldClose = false;
                let ids = collectIds(ifcModelId);

                if (!ids) {
                  ifcModelId = await ifcLoader.readIfcFile(new Uint8Array(buffer));
                  shouldClose = true;
                  ids = collectIds(ifcModelId);
                }

                if (!ids) {
                  throw new Error("IFC egress ID extraction failed.");
                }
                doneIds({
                  slabs: ids.slabs?.length,
                  stairs: ids.stairs?.length,
                  doors: ids.doors?.length,
                  walls: ids.walls?.length,
                  spaces: ids.spaces?.length,
                  storeys: ids.storeys?.length,
                });

                egressModelIdRef.current = ifcModelId;
                egressIdsRef.current = ids;
                onEgressDataExtracted({
                  modelId: ifcModelId,
                  ids,
                });
                if (shouldClose && ifcApi?.CloseModel) {
                  ifcApi.CloseModel(ifcModelId);
                }
              } catch (egressErr) {
                console.warn("IFC egress id extraction failed.", egressErr);
                onEgressDataExtracted?.({
                  modelId: egressModelIdRef.current,
                  ids: null,
                  egressError:
                    egressErr?.message || "IFC egress id extraction failed.",
                });
              } finally {
                setStatus("ready");
              }
            }, 50);
          });
        }
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

    const handlePointerDown = async () => {
      if (!pickModeRef.current) {
        return; // Not in picking mode
      }

      const raycaster = raycasterRef.current;
      if (!raycaster) {
        return;
      }

      try {
        const result = await raycaster.castRay();

        if (!result || !result.point) {
          return; // No hit
        }

        const point = [result.point.x, result.point.y, result.point.z];

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

        if (onPick) {
          onPick(pickModeRef.current, point, result);
        }
      } catch (err) {
        console.error("Raycast failed:", err);
      }
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
      {status === "processing" && (
        <div className="viewer-processing-badge">
          Extracting building data...
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}
    </div>
  );
}
