/** Centroid of a set of meshes' vertices ({positions} in any one frame), or null if there are none. */
export function centroidOf(meshes) {
  let x = 0, y = 0, z = 0, n = 0;
  for (const mesh of meshes || []) {
    const p = mesh.positions;
    for (let v = 0; v < p.length; v += 3) {
      x += p[v];
      y += p[v + 1];
      z += p[v + 2];
      n += 1;
    }
  }
  return n ? [x / n, y / n, z / n] : null;
}
