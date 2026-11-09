/**
 * The people of the egress animation (core/crowd.js simulateCrowd), drawn
 * and moved on the GPU: each person's walk is uploaded once, as a texture of
 * segments (start s, end s, from point, to point) and one of points, and the
 * vertex shader finds each person's segment for the playback time by
 * bisection and places them on it. The page does no work per frame beyond
 * setting the time, so thousands of people cost one draw call.
 *
 * They are drawn through the building, as the graph is, after everything
 * else and on a cleared depth buffer, so a figure's own limbs still hide
 * one another; a storey section clips them with the model.
 *
 * Two looks: dots, or simple figures (a body, a head, arms and legs that
 * swing while walking). Colour says what each person is doing: grey before
 * setting off, light blue walking, amber held up by the crowd or a door for
 * more than half a second, green out at a gathering place; red is still
 * inside when the run ended. Someone who left a full gathering place
 * (core/crowd.js goneAt) sinks a little and fades over FADE seconds.
 */
import * as THREE from "three";

// Texels a row of the data textures (WebGL 2 allows at least 2048).
const W = 2048;

export const CROWD_COLOURS = { waiting: 0x94a3b8, walking: 0x38bdf8, held: 0xf59e0b, out: 0x22c55e, inside: 0xef4444 };

const LOOKUP = `
precision highp float;
precision highp sampler2D;
uniform sampler2D segs;
uniform sampler2D pts;
uniform float time;
attribute float first;
attribute float count;
attribute float outAt;
const int W = ${W};
attribute float goneAt;
const float FADE = 2.5;
// 0 there, rising to 1 as they fade after leaving a full gathering place.
float fadeOf() { return clamp((time - goneAt) / FADE, 0.0, 1.0); }
vec4 seg(int k) { return texelFetch(segs, ivec2(k % W, k / W), 0); }
vec3 pt(float i) { int k = int(i + 0.5); return texelFetch(pts, ivec2(k % W, k / W), 0).xyz; }
// 0 before setting off, 1 walking, 2 held up, 3 out, 4 inside at the end.
void locate(out vec3 p, out vec3 dir, out float state) {
  int lo = int(first + 0.5), n = int(count + 0.5);
  vec4 s0 = seg(lo);
  if (time < s0.x) { p = pt(s0.z); dir = pt(s0.w) - p; state = 0.0; return; }
  int a = 0, b = n - 1;
  // Bisection, bounded for the GPU compilers that want it: 2^24 segments a person.
  for (int it = 0; it < 24 && a < b; it++) { int m = (a + b + 1) / 2; if (seg(lo + m).x <= time) a = m; else b = m - 1; }
  vec4 s = seg(lo + a);
  vec3 A = pt(s.z), B = pt(s.w);
  dir = B - A;
  if (time <= s.y && s.y > s.x) { p = mix(A, B, (time - s.x) / (s.y - s.x)); state = 1.0; return; }
  p = B;
  if (a == n - 1) { state = outAt < 1e29 ? 3.0 : 4.0; return; }
  // A short pause between steps is walking; a long one, held up.
  state = seg(lo + a + 1).x - s.y > 0.5 ? 2.0 : 1.0;
}
uniform vec3 colours[5];
vec3 colourOf(float state) { return colours[int(state + 0.5)]; }
#include <clipping_planes_pars_vertex>
`;

const DOT_VERTEX = `${LOOKUP}
uniform float size;
varying vec3 vColour;
varying float vFade;
void main() {
  vec3 p, dir; float state;
  locate(p, dir, state);
  vColour = colourOf(state);
  vFade = fadeOf();
  vec4 mvPosition = modelViewMatrix * vec4(p + vec3(0.0, 0.9 - 0.4 * vFade, 0.0), 1.0);
  gl_PointSize = clamp(size / -mvPosition.z, 3.0, 18.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;
const DOT_FRAGMENT = `
#include <clipping_planes_pars_fragment>
varying vec3 vColour;
varying float vFade;
void main() {
  #include <clipping_planes_fragment>
  if (vFade >= 1.0) discard;
  vec2 c = gl_PointCoord - 0.5;
  float r = dot(c, c);
  if (r > 0.25) discard;
  gl_FragColor = vec4(mix(vColour, vColour * 0.55, smoothstep(0.12, 0.25, r)), 1.0 - vFade);
}`;

const FIGURE_VERTEX = `${LOOKUP}
uniform float wall;
attribute float part;
attribute float pivot;
attribute float tone;
attribute float phase;
varying vec3 vColour;
varying float vShade;
varying float vFade;
void main() {
  vec3 p, dir; float state;
  locate(p, dir, state);
  vFade = fadeOf();
  p.y -= 0.4 * vFade;
  // The shirt says what they are doing; trousers and skin stay as they are.
  vColour = tone == 0.0 ? colourOf(state) : tone == 1.0 ? vec3(0.2, 0.24, 0.32) : vec3(0.93, 0.78, 0.65);
  // Arms and legs swing about the hip or shoulder while walking.
  float swing = state == 1.0 ? 0.55 * sin(wall * 11.0 + phase) : 0.0;
  float side = part == 1.0 || part == 4.0 ? 1.0 : -1.0;
  float angle = part == 0.0 ? 0.0 : swing * side * (part > 2.5 ? 0.8 : 1.0);
  vec3 q = position;
  vec3 nrm = normal;
  float c = cos(angle), s = sin(angle);
  q.yz = vec2(pivot + (q.y - pivot) * c - q.z * s, (q.y - pivot) * s + q.z * c);
  nrm.yz = vec2(nrm.y * c - nrm.z * s, nrm.y * s + nrm.z * c);
  // Facing the way they walk: local +z turns to the segment's direction.
  vec2 f = length(dir.xz) > 1e-4 ? normalize(dir.xz) : vec2(0.0, 1.0);
  vec3 world = vec3(q.x * f.y + q.z * f.x, q.y, -q.x * f.x + q.z * f.y);
  vec3 n = vec3(nrm.x * f.y + nrm.z * f.x, nrm.y, -nrm.x * f.x + nrm.z * f.y);
  vShade = 0.62 + 0.38 * max(dot(n, normalize(vec3(0.35, 1.0, 0.25))), 0.0);
  vec4 mvPosition = modelViewMatrix * vec4(p + world, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;
const FIGURE_FRAGMENT = `
#include <clipping_planes_pars_fragment>
varying vec3 vColour;
varying float vShade;
varying float vFade;
void main() {
  #include <clipping_planes_fragment>
  if (vFade >= 1.0) discard;
  gl_FragColor = vec4(vColour * vShade, 1.0 - vFade);
}`;

/** A float texture of `rows` of four numbers, W texels wide. */
function dataTexture(values, stride) {
  const items = values.length / stride;
  const height = Math.max(1, Math.ceil(items / W));
  const data = new Float32Array(W * height * 4);
  for (let i = 0; i < items; i += 1) for (let c = 0; c < stride; c += 1) data[i * 4 + c] = values[i * stride + c];
  const texture = new THREE.DataTexture(data, W, height, THREE.RGBAFormat, THREE.FloatType);
  // Read texel by texel (texelFetch): no filtering.
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

/**
 * A person 1.75 m tall facing +z, feet at the origin: parts 0 body and head,
 * 1 and 2 legs, 3 and 4 arms; tones 0 shirt, 1 trousers, 2 skin.
 */
function figureGeometry() {
  const pieces = [];
  const add = (g, part, pivot, tone) => pieces.push({ g, part, pivot, tone });
  const box = (w, h, d, x, y, z, part, pivot, tone) => add(new THREE.BoxGeometry(w, h, d).translate(x, y, z), part, pivot, tone);
  box(0.36, 0.58, 0.21, 0, 1.15, 0, 0, 0, 0);
  add(new THREE.SphereGeometry(0.115, 10, 8).translate(0, 1.6, 0.01), 0, 0, 2);
  box(0.14, 0.88, 0.15, -0.095, 0.44, 0, 1, 0.88, 1);
  box(0.14, 0.88, 0.15, 0.095, 0.44, 0, 2, 0.88, 1);
  box(0.09, 0.62, 0.1, -0.235, 1.12, 0, 3, 1.42, 0);
  box(0.09, 0.62, 0.1, 0.235, 1.12, 0, 4, 1.42, 0);
  const position = [], normal = [], part = [], pivot = [], tone = [];
  for (const { g, part: k, pivot: y, tone: c } of pieces) {
    const flat = g.index ? g.toNonIndexed() : g;
    position.push(...flat.attributes.position.array);
    normal.push(...flat.attributes.normal.array);
    for (let v = 0; v < flat.attributes.position.count; v += 1) {
      part.push(k);
      pivot.push(y);
      tone.push(c);
    }
  }
  return { position: Float32Array.from(position), normal: Float32Array.from(normal), part: Float32Array.from(part), pivot: Float32Array.from(pivot), tone: Float32Array.from(tone) };
}

/**
 * The crowd as one object, dots or figures. Its `time` (s of the walk) is
 * read from `clock()` before every draw.
 */
export function crowdObject(crowd, { style = "figures", clock = () => 0, colours: own = {} } = {}) {
  const { people, segments, points, first, count, outAt } = crowd;
  const segs = dataTexture(segments, 4);
  const pts = dataTexture(points, 3);
  const colours = ["waiting", "walking", "held", "out", "inside"].map((k) => new THREE.Color(own[k] ?? CROWD_COLOURS[k]));
  const uniforms = { segs: { value: segs }, pts: { value: pts }, time: { value: 0 }, wall: { value: 0 }, size: { value: 260 }, colours: { value: colours } };
  const perPerson = (geometry, Attribute) => {
    geometry.setAttribute("first", new Attribute(Float32Array.from(first), 1));
    geometry.setAttribute("count", new Attribute(Float32Array.from(count), 1));
    geometry.setAttribute("outAt", new Attribute(Float32Array.from(outAt, (t) => (Number.isFinite(t) ? t : 1e30)), 1));
    geometry.setAttribute("goneAt", new Attribute(Float32Array.from(crowd.goneAt ?? new Float32Array(people).fill(Infinity), (t) => (Number.isFinite(t) ? t : 1e30)), 1));
  };
  let object;
  if (style === "dots") {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(people * 3), 3));
    perPerson(geometry, THREE.BufferAttribute);
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: DOT_VERTEX, fragmentShader: DOT_FRAGMENT, clipping: true, transparent: true });
    object = new THREE.Points(geometry, material);
  } else {
    const shape = figureGeometry();
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(shape.position, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(shape.normal, 3));
    geometry.setAttribute("part", new THREE.BufferAttribute(shape.part, 1));
    geometry.setAttribute("pivot", new THREE.BufferAttribute(shape.pivot, 1));
    geometry.setAttribute("tone", new THREE.BufferAttribute(shape.tone, 1));
    perPerson(geometry, THREE.InstancedBufferAttribute);
    // Each walks out of step with the others.
    geometry.setAttribute("phase", new THREE.InstancedBufferAttribute(Float32Array.from({ length: people }, (_, i) => (i * 2.399963) % (2 * Math.PI)), 1));
    geometry.instanceCount = people;
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader: FIGURE_VERTEX, fragmentShader: FIGURE_FRAGMENT, clipping: true, transparent: true });
    object = new THREE.Mesh(geometry, material);
  }
  object.frustumCulled = false;
  // Transparent and late, so drawn after the model; its depth cleared first.
  object.renderOrder = 960;
  object.onBeforeRender = (renderer) => {
    renderer.clearDepth();
    uniforms.time.value = clock();
    uniforms.wall.value = performance.now() / 1000;
  };
  const dispose = object.geometry.dispose.bind(object.geometry);
  object.geometry.dispose = () => {
    segs.dispose();
    pts.dispose();
    dispose();
  };
  return object;
}
