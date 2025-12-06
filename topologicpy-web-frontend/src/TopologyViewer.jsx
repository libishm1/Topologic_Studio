// src/TopologyViewer.jsx
import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Plural keys in parents object (TopologicPy export)
const PARENTS_LEVEL_ORDER = [
  "cellComplexes",
  "cells",
  "shells",
  "faces",
  "wires",
  "edges",
  "vertices",
];

// Map plural parents keys -> singular level label for UI
const PLURAL_TO_SINGULAR = {
  clusters: "Cluster",
  cellComplexes: "CellComplex",
  cells: "Cell",
  shells: "Shell",
  faces: "Face",
  wires: "Wire",
  edges: "Edge",
  vertices: "Vertex",
};

// Inverse map: singular -> plural
const SINGULAR_TO_PLURAL = Object.fromEntries(
  Object.entries(PLURAL_TO_SINGULAR).map(([pl, si]) => [si, pl])
);

// ---- Helpers to parse colour / opacity from dictionary ----
function parseColorSpec(spec, fallbackColor) {
  if (!spec) return fallbackColor.clone();
  const color = new THREE.Color();

  // String CSS colour
  if (typeof spec === "string") {
    try {
      color.set(spec);
      return color;
    } catch {
      return fallbackColor.clone();
    }
  }

  // Array: [r, g, b] or [r, g, b, a]
  if (Array.isArray(spec) && spec.length >= 3) {
    let [r, g, b] = spec;
    const is0to255 =
      Math.max(Math.abs(r), Math.abs(g), Math.abs(b)) > 1.0;

    if (is0to255) {
      r /= 255.0;
      g /= 255.0;
      b /= 255.0;
    }

    color.setRGB(r, g, b);
    return color;
  }

  // Object with r/g/b
  if (
    typeof spec === "object" &&
    spec !== null &&
    ["r", "g", "b"].every((k) => spec[k] !== undefined)
  ) {
    let { r, g, b } = spec;
    const is0to255 =
      Math.max(Math.abs(r), Math.abs(g), Math.abs(b)) > 1.0;
    if (is0to255) {
      r /= 255.0;
      g /= 255.0;
      b /= 255.0;
    }
    color.setRGB(r, g, b);
    return color;
  }

  return fallbackColor.clone();
}

function extractAlphaFromColorSpec(spec) {
  if (Array.isArray(spec) && spec.length >= 4) {
    let a = spec[3];
    if (a > 1.0) a /= 255.0;
    return a;
  }
  if (
    typeof spec === "object" &&
    spec !== null &&
    spec.a !== undefined
  ) {
    let a = spec.a;
    if (a > 1.0) a /= 255.0;
    return a;
  }
  return null;
}

export default function TopologyViewer({ data, selection, onSelectionChange }) {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);

  const facesGroupRef = useRef(null);
  const edgesGroupRef = useRef(null);
  const vertsGroupRef = useRef(null);

  const raycasterRef = useRef(new THREE.Raycaster());
  const pointerRef = useRef(new THREE.Vector2());

  const faceMeshByIdRef = useRef(new Map());
  const dataRef = useRef(null);

  const lastHitRef = useRef({
    baseUid: null,
    parentsKey: null,
    index: 0,
    chain: [],
    point: null,
  });

  const selectionRef = useRef(null);

  useEffect(() => {
    dataRef.current = data;
  }, [data]);

  useEffect(() => {
    selectionRef.current = selection;
  }, [selection]);

  // ---------- SCENE INIT (once) ----------
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || mount.offsetWidth || 800;
    const height = mount.clientHeight || mount.offsetHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    scene.up.set(0, 0, 1); // Z-up

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.up.set(0, 0, 1);
    camera.position.set(5, 5, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 10, 10);
    scene.add(dirLight);

    const grid = new THREE.GridHelper(30, 30);
    grid.rotation.x = Math.PI / 2; // XY-plane
    scene.add(grid);

    const axes = new THREE.AxesHelper(2);
    scene.add(axes);

    const facesGroup = new THREE.Group();
    const edgesGroup = new THREE.Group();
    const vertsGroup = new THREE.Group();
    scene.add(facesGroup);
    scene.add(edgesGroup);
    scene.add(vertsGroup);

    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    controlsRef.current = controls;
    facesGroupRef.current = facesGroup;
    edgesGroupRef.current = edgesGroup;
    vertsGroupRef.current = vertsGroup;

    raycasterRef.current.params.Line.threshold = 0.01;

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    const handleResize = () => {
      if (!mount || !rendererRef.current || !cameraRef.current) return;
      const w = mount.clientWidth || mount.offsetWidth || width;
      const h = mount.clientHeight || mount.offsetHeight || height;
      const cam = cameraRef.current;
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    const handlePointerDown = (event) => {
      const renderer = rendererRef.current;
      const camera = cameraRef.current;
      const facesGroup = facesGroupRef.current;
      const edgesGroup = edgesGroupRef.current;
      const vertsGroup = vertsGroupRef.current;
      const topo = dataRef.current;

      if (!renderer || !camera || !facesGroup || !topo) return;

      const rect = renderer.domElement.getBoundingClientRect();
      pointerRef.current.x =
        ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointerRef.current.y =
        -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(pointerRef.current, camera);

      const pickFromGroup = (group, topoType) => {
        if (!group) return null;
        const hits = raycasterRef.current.intersectObjects(
          group.children,
          false
        );
        if (hits.length === 0) return null;
        const hit = hits[0];
        const obj = hit.object;
        const uid = obj.userData.uid;
        if (!uid) return null;
        return {
          uid,
          topoType,
          parents: obj.userData.parents || {},
          point: hit.point.clone(),
        };
      };

      let baseHit =
        pickFromGroup(facesGroup, "Face") ||
        pickFromGroup(edgesGroup, "Edge") ||
        pickFromGroup(vertsGroup, "Vertex");

      // Background click: clear selection & reset
      if (!baseHit) {
        onSelectionChange?.(null);
        lastHitRef.current = {
          baseUid: null,
          parentsKey: null,
          index: 0,
          chain: [],
          point: null,
        };
        return;
      }

      const { uid: baseUid, topoType, parents, point } = baseHit;

      // Build chain from parents according to priority order
      const chain = [];
      for (const levelKey of PARENTS_LEVEL_ORDER) {
        const arr = parents[levelKey] || [];
        for (const pid of arr) {
          chain.push({
            levelPlural: levelKey,
            uid: pid,
          });
        }
      }

      const selfPlural = SINGULAR_TO_PLURAL[topoType] || null;
      if (selfPlural) {
        const already = chain.some(
          (item) => item.levelPlural === selfPlural && item.uid === baseUid
        );
        if (!already) {
          chain.push({
            levelPlural: selfPlural,
            uid: baseUid,
          });
        }
      }

      if (chain.length === 0) {
        lastHitRef.current = {
          baseUid,
          parentsKey: JSON.stringify(parents || {}),
          index: 0,
          chain: [],
          point,
        };
        onSelectionChange?.({
          uid: baseUid,
          level: topoType,
        });
        return;
      }

      const seen = new Set();
      const dedupedChain = [];
      for (const item of chain) {
        const key = `${item.levelPlural}:${item.uid}`;
        if (seen.has(key)) continue;
        seen.add(key);
        dedupedChain.push(item);
      }

      const parentsKey = JSON.stringify(parents || {});
      const sameContext =
        lastHitRef.current.baseUid === baseUid &&
        lastHitRef.current.parentsKey === parentsKey;

      const currentSelection = selectionRef.current;
      let chosen;
      let index = 0;

      if (sameContext) {
        // same region: cycle
        index = (lastHitRef.current.index + 1) % dedupedChain.length;
        chosen = dedupedChain[index];
      } else if (currentSelection) {
        // new region, but keep same level as current selection if possible
        const desiredPlural = SINGULAR_TO_PLURAL[currentSelection.level];
        let idx = -1;
        if (desiredPlural) {
          idx = dedupedChain.findIndex(
            (item) => item.levelPlural === desiredPlural
          );
        }
        if (idx === -1) idx = 0;
        index = idx;
        chosen = dedupedChain[idx];
      } else {
        // no active selection: start at top of chain
        index = 0;
        chosen = dedupedChain[0];
      }

      lastHitRef.current = {
        baseUid,
        parentsKey,
        index,
        chain: dedupedChain,
        point,
      };

      const singularLevel =
        PLURAL_TO_SINGULAR[chosen.levelPlural] || topoType || "Unknown";

      onSelectionChange?.({
        uid: chosen.uid,
        level: singularLevel,
      });
    };

    renderer.domElement.addEventListener("pointerdown", handlePointerDown);

    return () => {
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.dispose();
      if (renderer.domElement.parentElement === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [onSelectionChange]);

  // ---------- BUILD GEOMETRY WHEN DATA CHANGES ----------
  useEffect(() => {
    const facesGroup = facesGroupRef.current;
    const edgesGroup = edgesGroupRef.current;
    const vertsGroup = vertsGroupRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const faceMeshById = faceMeshByIdRef.current;

    faceMeshById.clear();

    if (!facesGroup || !edgesGroup || !vertsGroup || !data) return;

    const clearGroup = (group) => {
      while (group.children.length > 0) {
        const child = group.children.pop();
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      }
    };

    clearGroup(facesGroup);
    clearGroup(edgesGroup);
    clearGroup(vertsGroup);

    const baseFaceColor = new THREE.Color(0xcccccc);
    const baseEdgeColor = new THREE.Color(0x222222);
    const baseVertColor = new THREE.Color(0x000000);

    // --- 1. Build vertex lookup ---
    const vertexById = new Map();

    const addVertexFromObj = (v) => {
      if (!v) return;
      const id = v.uid ?? v.uuid;
      if (!id) return;
      const coord = v.coordinates || v.Coordinates;
      if (!Array.isArray(coord) || coord.length < 3) return;
      vertexById.set(id, {
        uid: id,
        x: coord[0],
        y: coord[1],
        z: coord[2],
        dictionary: v.dictionary || {},
        parents: v.parents || {},
      });
    };

    if (Array.isArray(data.vertices)) {
      data.vertices.forEach(addVertexFromObj);
    }

    // --- 2. Faces ---
    if (Array.isArray(data.faces)) {
      data.faces.forEach((face) => {
        const id = face.uid ?? face.uuid;
        if (!id) return;

        const tris = face.triangles || [];
        if (!Array.isArray(tris) || tris.length === 0) return;

        const positions = [];
        tris.forEach((tri) => {
          if (Array.isArray(tri) && tri.length === 3) {
            tri.forEach((p) => {
              positions.push(p[0], p[1], p[2]);
            });
          }
        });
        if (positions.length < 9) return;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(positions, 3)
        );
        geometry.computeVertexNormals();

        const dict = face.dictionary || {};

        const colorSpec =
          face.color ??
          dict.color ??
          dict.Color ??
          dict.faceColor ??
          dict.FaceColor;

        let color = parseColorSpec(colorSpec, baseFaceColor);

        // Opacity: face.opacity > dict.opacity / alpha > alpha from colour spec > default
        let opacity =
          typeof face.opacity === "number"
            ? face.opacity
            : typeof dict.opacity === "number"
            ? dict.opacity
            : typeof dict.Opacity === "number"
            ? dict.Opacity
            : typeof dict.alpha === "number"
            ? dict.alpha
            : typeof dict.Alpha === "number"
            ? dict.Alpha
            : null;

        if (opacity == null) {
          const alphaFromColor = extractAlphaFromColorSpec(colorSpec);
          opacity =
            typeof alphaFromColor === "number" ? alphaFromColor : 0.6;
        }

        if (opacity > 1.0) opacity = 1.0;
        if (opacity < 0.0) opacity = 0.0;

        const material = new THREE.MeshStandardMaterial({
        color,
        transparent: opacity < 1.0,
        opacity,
        side: THREE.DoubleSide,
      });
        
        // Store original appearance for later reset
        material.userData = material.userData || {};
        material.userData.baseColor = color.clone();
        material.userData.baseOpacity = opacity;



        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData.uid = id;
        mesh.userData.topoType = "Face";
        mesh.userData.parents = face.parents || {};

        facesGroup.add(mesh);
        faceMeshById.set(id, mesh);
      });
    }

    // --- 3. Edges ---
    let edgeSource = [];
    if (Array.isArray(data.edges) && data.edges.length > 0) {
      edgeSource = data.edges;
    } else if (Array.isArray(data.raw)) {
      edgeSource = data.raw.filter(
        (e) => e.type === "Edge" && Array.isArray(e.vertices)
      );
    }

    edgeSource.forEach((edge) => {
      const id = edge.uid ?? edge.uuid;
      if (!id) return;

      const vIds = edge.vertices || [];
      if (vIds.length < 2) return;
      const v0 = vertexById.get(vIds[0]);
      const v1 = vertexById.get(vIds[1]);
      if (!v0 || !v1) return;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(
          [v0.x, v0.y, v0.z, v1.x, v1.y, v1.z],
          3
        )
      );

      const dict = edge.dictionary || {};
      const edgeColorSpec =
        edge.color ??
        dict.color ??
        dict.Color ??
        dict.edgeColor ??
        dict.EdgeColor;

      const color = parseColorSpec(edgeColorSpec, baseEdgeColor);

      const material = new THREE.LineBasicMaterial({
        color,
        linewidth: 1,
        depthTest: false, // on top of faces
      });

      const line = new THREE.Line(geometry, material);
      line.userData.uid = id;
      line.userData.topoType = "Edge";
      line.userData.parents = edge.parents || {};

      edgesGroup.add(line);
    });

    // --- 4. Vertices ---
    if (vertexById.size > 0) {
      const vertGeom = new THREE.SphereGeometry(0.02, 10, 10);
      vertexById.forEach((v) => {
        const dict = v.dictionary || {};
        const vertColorSpec =
          v.color ??
          dict.color ??
          dict.Color ??
          dict.vertexColor ??
          dict.VertexColor;

        const color = parseColorSpec(vertColorSpec, baseVertColor);

        const material = new THREE.MeshBasicMaterial({
          color,
          depthTest: false, // on top
        });
        const mesh = new THREE.Mesh(vertGeom, material);
        mesh.position.set(v.x, v.y, v.z);
        mesh.userData.uid = v.uid;
        mesh.userData.topoType = "Vertex";
        mesh.userData.parents = v.parents || {};

        vertsGroup.add(mesh);
      });
    }

    // --- 5. Recenter camera ---
    if (camera && vertexById.size > 0) {
      const xs = [];
      const ys = [];
      const zs = [];
      vertexById.forEach((v) => {
        xs.push(v.x);
        ys.push(v.y);
        zs.push(v.z);
      });
      const center = new THREE.Vector3(
        (Math.min(...xs) + Math.max(...xs)) / 2,
        (Math.min(...ys) + Math.max(...ys)) / 2,
        (Math.min(...zs) + Math.max(...zs)) / 2
      );
      camera.position.set(center.x + 5, center.y + 5, center.z + 5);
      camera.lookAt(center);
      if (controls) {
        controls.target.copy(center);
        controls.update();
      }
    } else if (camera) {
      camera.position.set(5, 5, 5);
      camera.lookAt(0, 0, 0);
      if (controls) {
        controls.target.set(0, 0, 0);
        controls.update();
      }
    }
  }, [data]);

// ---------- HIGHLIGHTING BASED ON SELECTION ----------
    useEffect(() => {
      const faceMeshById = faceMeshByIdRef.current;
      const highlightColor = new THREE.Color(0xff0000);

      // 1) Reset all faces to their stored base colour + opacity
      faceMeshById.forEach((mesh) => {
        const mat = mesh.material;
        if (!mat) return;

        const baseColor =
          mat.userData?.baseColor instanceof THREE.Color
            ? mat.userData.baseColor
            : null;
        const baseOpacity =
          typeof mat.userData?.baseOpacity === "number"
            ? mat.userData.baseOpacity
            : 0.6;

        if (baseColor && mat.color) {
          mat.color.copy(baseColor);
        }
        mat.opacity = baseOpacity;
        mat.transparent = baseOpacity < 1.0;
      });

      // No active selection → nothing highlighted
      if (!selection) return;

      const pluralKey = SINGULAR_TO_PLURAL[selection.level];
      if (!pluralKey) return;

      // 2) If the selection is a Face: highlight just that one
      if (selection.level === "Face") {
        const mesh = faceMeshById.get(selection.uid);
        if (mesh && mesh.material && mesh.material.color) {
          mesh.material.color.copy(highlightColor);

          const baseOpacity =
            typeof mesh.material.userData?.baseOpacity === "number"
              ? mesh.material.userData.baseOpacity
              : 0.6;

          mesh.material.opacity = Math.max(baseOpacity, 0.6);
          mesh.material.transparent = mesh.material.opacity < 1.0;
        }
        return;
      }

      // 3) Higher-level selection: highlight all faces whose parents contain this uid
      faceMeshById.forEach((mesh) => {
        const mat = mesh.material;
        if (!mat || !mat.color) return;

        const parents = mesh.userData.parents || {};
        const ids = parents[pluralKey] || [];
        if (Array.isArray(ids) && ids.includes(selection.uid)) {
          mat.color.copy(highlightColor);

          const baseOpacity =
            typeof mat.userData?.baseOpacity === "number"
              ? mat.userData.baseOpacity
              : 0.6;

          mat.opacity = Math.max(baseOpacity, 0.9);
          mat.transparent = mat.opacity < 1.0;
        }
      });
    }, [selection]);


  return (
    <div
      ref={mountRef}
      style={{ width: "100%", height: "100%", overflow: "hidden" }}
    />
  );
}
