/**
 * Walkable-surface sampling worker.
 *
 * The architectural change versus Classic. Classic extracted category geometry
 * on the main thread, serialised every triangle to JSON and posted tens of
 * megabytes to Python, which then resampled it into a few thousand points.
 *
 * The browser already has the geometry in memory to draw it, so sampling here
 * costs one pass over buffers we already hold, off the main thread, and the
 * upload becomes the point cloud itself: typically a couple of hundred
 * kilobytes instead of tens of megabytes.
 *
 * Input geometry arrives as transferable typed arrays, so handing it to the
 * worker is a pointer move rather than a copy.
 */

const AXIS = { x: 0, y: 1, z: 2 };

function axisIndex(upAxis) {
  return AXIS[upAxis] ?? 2;
}

function horizontalAxes(upAxis) {
  const up = axisIndex(upAxis);
  return [0, 1, 2].filter((i) => i !== up);
}

/**
 * Sample one mesh's walkable triangles onto a barycentric lattice.
 * Mirrors the server-side rule so the two paths agree.
 */
function sampleMesh(mesh, options, out) {
  const { positions, indices } = mesh;
  if (!positions || !indices || positions.length < 9 || indices.length < 3) return;

  const up = axisIndex(options.upAxis);
  const minUp = Math.cos((Math.max(0, Math.min(89, options.maxSlopeDeg)) * Math.PI) / 180);
  const spacing = Math.max(options.spacing, 1e-3);
  const excludeAbove = options.excludeAbove;

  const triangles = Math.floor(indices.length / 3);
  for (let t = 0; t < triangles; t += 1) {
    if (out.count >= options.maxPoints) return;

    const i0 = indices[t * 3] * 3;
    const i1 = indices[t * 3 + 1] * 3;
    const i2 = indices[t * 3 + 2] * 3;
    if (i0 + 2 >= positions.length || i1 + 2 >= positions.length || i2 + 2 >= positions.length) {
      continue;
    }

    const ax = positions[i0], ay = positions[i0 + 1], az = positions[i0 + 2];
    const bx = positions[i1], by = positions[i1 + 1], bz = positions[i1 + 2];
    const cx = positions[i2], cy = positions[i2 + 1], cz = positions[i2 + 2];

    if (excludeAbove !== null && excludeAbove !== undefined) {
      const a = up === 0 ? ax : up === 1 ? ay : az;
      const b = up === 0 ? bx : up === 1 ? by : bz;
      const c = up === 0 ? cx : up === 1 ? cy : cz;
      if (a >= excludeAbove && b >= excludeAbove && c >= excludeAbove) continue;
    }

    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const mag = Math.hypot(nx, ny, nz);
    if (mag <= 2e-6) continue;

    // Signed, not absolute. You stand on an upward-facing surface; a slab's
    // underside and a ceiling are just as "horizontal" but are not walkable.
    // Treating them as walkable produced a second point sheet under every
    // storey, which the neighbour search then braced into a space-frame truss.
    const signedUp = (up === 0 ? nx : up === 1 ? ny : nz) / mag;
    const upComponent = options.requireUpward ? signedUp : Math.abs(signedUp);
    if (upComponent < minUp) continue;

    const area = 0.5 * mag;
    const steps = Math.max(1, Math.min(48, Math.ceil(Math.sqrt(area) / spacing)));

    for (let u = 0; u <= steps; u += 1) {
      for (let v = 0; v <= steps - u; v += 1) {
        const alpha = u / steps;
        const beta = v / steps;
        const gamma = 1 - alpha - beta;
        if (gamma < 0) continue;
        out.push(
          alpha * ax + beta * bx + gamma * cx,
          alpha * ay + beta * by + gamma * cy,
          alpha * az + beta * bz + gamma * cz,
        );
        if (out.count >= options.maxPoints) return;
      }
    }
  }
}

/** Growable float32 point buffer with voxel deduplication on read. */
function createSink(capacity) {
  let data = new Float32Array(Math.max(capacity * 3, 3072));
  let count = 0;
  return {
    get count() {
      return count;
    },
    push(x, y, z) {
      if ((count + 1) * 3 > data.length) {
        const grown = new Float32Array(data.length * 2);
        grown.set(data);
        data = grown;
      }
      const o = count * 3;
      data[o] = x;
      data[o + 1] = y;
      data[o + 2] = z;
      count += 1;
    },
    /** Deduplicate onto a voxel grid; triangle fans overlap heavily at seams. */
    decimate(cell) {
      if (count === 0) return new Float32Array(0);
      if (!cell || cell <= 0) return data.slice(0, count * 3);
      const seen = new Set();
      const out = new Float32Array(count * 3);
      let kept = 0;
      const inv = 1 / cell;
      for (let i = 0; i < count; i += 1) {
        const o = i * 3;
        const key =
          `${Math.round(data[o] * inv)},` +
          `${Math.round(data[o + 1] * inv)},` +
          `${Math.round(data[o + 2] * inv)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const k = kept * 3;
        out[k] = data[o];
        out[k + 1] = data[o + 1];
        out[k + 2] = data[o + 2];
        kept += 1;
      }
      return out.slice(0, kept * 3);
    },
  };
}

/**
 * Collapse each vertical column of points down to its walking surface.
 *
 * A slab is a solid: sampling accepts both its top face and its underside,
 * because a downward-facing normal is just as "horizontal" as an upward one and
 * IFC winding is not reliable enough to tell them apart. That produced two
 * parallel sheets about a slab-thickness apart, and the neighbour search then
 * braced them together into a space-frame truss instead of a flat floor mesh.
 * Ceilings (IFCCOVERING) add further phantom sheets.
 *
 * For each horizontal cell the points are sorted by height and split into
 * clusters wherever the gap exceeds `gap`; each cluster keeps only its topmost
 * point. A slab's top and underside fall in one cluster and collapse to the
 * top - the surface you actually stand on - while separate storeys stay
 * separate because they are far more than `gap` apart.
 */
function collapseColumns(flat, cellSize, upAxis, gap) {
  const count = Math.floor(flat.length / 3);
  if (count === 0) return flat;

  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const inv = 1 / Math.max(cellSize, 1e-3);

  const columns = new Map();
  for (let i = 0; i < count; i += 1) {
    const o = i * 3;
    const key = `${Math.round(flat[o + h0] * inv)},${Math.round(flat[o + h1] * inv)}`;
    let bucket = columns.get(key);
    if (!bucket) {
      bucket = [];
      columns.set(key, bucket);
    }
    bucket.push(i);
  }

  const out = new Float32Array(count * 3);
  let kept = 0;
  const emit = (index) => {
    const o = index * 3;
    const k = kept * 3;
    out[k] = flat[o];
    out[k + 1] = flat[o + 1];
    out[k + 2] = flat[o + 2];
    kept += 1;
  };

  for (const bucket of columns.values()) {
    bucket.sort((a, b) => flat[a * 3 + up] - flat[b * 3 + up]);
    let top = bucket[0];
    for (let n = 1; n < bucket.length; n += 1) {
      const current = bucket[n];
      if (flat[current * 3 + up] - flat[top * 3 + up] > gap) {
        emit(top); // previous cluster ended; keep its highest point
      }
      top = current;
    }
    emit(top);
  }

  return out.slice(0, kept * 3);
}

/**
 * Bottom-centre of each door opening: one navigation waypoint per door.
 *
 * Meshes are aggregated by `itemId` first. An IFC door is typically several
 * meshes (frame, leaf, glazing); treating each as its own waypoint produced
 * three or four nodes per opening, which inflated the door count and clustered
 * redundant nodes in the doorway.
 */
function doorWaypoints(meshes, upAxis) {
  const up = axisIndex(upAxis);
  const byItem = new Map();

  for (const mesh of meshes) {
    const p = mesh.positions;
    if (!p || p.length < 9) continue;
    // Fall back to a per-mesh key when the caller did not tag the item.
    const key = mesh.itemId ?? `mesh:${byItem.size}`;
    let acc = byItem.get(key);
    if (!acc) {
      acc = { sum: [0, 0, 0], count: 0, minUp: Infinity };
      byItem.set(key, acc);
    }
    const n = Math.floor(p.length / 3);
    for (let i = 0; i < n; i += 1) {
      const o = i * 3;
      acc.sum[0] += p[o];
      acc.sum[1] += p[o + 1];
      acc.sum[2] += p[o + 2];
      const u = p[o + up];
      if (u < acc.minUp) acc.minUp = u;
    }
    acc.count += n;
  }

  const out = [];
  for (const acc of byItem.values()) {
    if (!acc.count) continue;
    const coord = [
      acc.sum[0] / acc.count,
      acc.sum[1] / acc.count,
      acc.sum[2] / acc.count,
    ];
    coord[up] = acc.minUp;
    out.push(coord);
  }
  return out;
}

/** Each wall reduced to a 2D centreline plus its vertical extent. */
function wallSegments(meshes, upAxis) {
  const up = axisIndex(upAxis);
  const [h0, h1] = horizontalAxes(upAxis);
  const out = [];
  for (const mesh of meshes) {
    const p = mesh.positions;
    if (!p || p.length < 9) continue;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    const n = Math.floor(p.length / 3);
    for (let i = 0; i < n; i += 1) {
      const o = i * 3;
      for (let a = 0; a < 3; a += 1) {
        const value = p[o + a];
        if (value < lo[a]) lo[a] = value;
        if (value > hi[a]) hi[a] = value;
      }
    }
    const span0 = hi[h0] - lo[h0];
    const span1 = hi[h1] - lo[h1];
    const mid0 = (lo[h0] + hi[h0]) / 2;
    const mid1 = (lo[h1] + hi[h1]) / 2;
    const segment =
      span0 >= span1
        ? [[lo[h0], mid1], [hi[h0], mid1]]
        : [[mid0, lo[h1]], [mid0, hi[h1]]];
    out.push({
      segment,
      thickness: Math.min(span0, span1),
      up_min: lo[up],
      up_max: hi[up],
    });
  }
  return out;
}

function maxHeight(groups, upAxis) {
  const up = axisIndex(upAxis);
  let best = -Infinity;
  for (const meshes of groups) {
    for (const mesh of meshes) {
      const p = mesh.positions;
      if (!p) continue;
      for (let i = up; i < p.length; i += 3) {
        if (p[i] > best) best = p[i];
      }
    }
  }
  return Number.isFinite(best) ? best : null;
}

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    const {
      floors = [],
      stairs = [],
      doors = [],
      walls = [],
      upAxis = "y",
      floorSpacing = 0.5,
      maxPoints = 40000,
      decimateCell = 0,
      // Height difference that still counts as the same slab. Comfortably
      // above a slab thickness or floor build-up, comfortably below a storey.
      columnGap = 0.9,
    } = payload || {};

    const t0 = performance.now();

    // Roof slabs are geometrically walkable but are not floors; anything
    // within a metre of the model's ceiling is excluded.
    const ceiling = maxHeight([floors, stairs], upAxis);
    const excludeAbove = ceiling === null ? null : ceiling - 1.0;

    const stairSpacing = Math.max(floorSpacing * 0.3, 0.15);
    const stairBudget = Math.floor(maxPoints * 0.4);

    /**
     * Sample a group, preferring upward-facing surfaces.
     *
     * Some exporters wind their faces inconsistently. If insisting on upward
     * normals yields nothing at all, the model's winding cannot be trusted, so
     * fall back to accepting either orientation rather than returning an empty
     * graph. The fallback is reported so the caller knows the result is coarser.
     */
    const sampleGroup = (meshes, options, capacity) => {
      let sink = createSink(capacity);
      for (const mesh of meshes) {
        sampleMesh(mesh, { ...options, requireUpward: true }, sink);
      }
      if (sink.count === 0 && meshes.length > 0) {
        sink = createSink(capacity);
        for (const mesh of meshes) {
          sampleMesh(mesh, { ...options, requireUpward: false }, sink);
        }
        return { sink, windingFallback: sink.count > 0 };
      }
      return { sink, windingFallback: false };
    };

    const stairResult = sampleGroup(
      stairs,
      { upAxis, spacing: stairSpacing, maxSlopeDeg: 45, maxPoints: stairBudget, excludeAbove },
      4096,
    );
    const stairSink = stairResult.sink;

    const floorResult = sampleGroup(
      floors,
      {
        upAxis,
        spacing: floorSpacing,
        maxSlopeDeg: 10,
        maxPoints: maxPoints - stairSink.count,
        excludeAbove,
      },
      16384,
    );
    const floorSink = floorResult.sink;

    const cell = decimateCell > 0 ? decimateCell : floorSpacing * 0.8;

    // Collapse floors to one sheet per storey. Stairs are left alone: their
    // treads are meant to sit at different heights, and merging them would
    // flatten the flight.
    const floorRaw = floorSink.decimate(cell);
    const floorPoints = collapseColumns(floorRaw, cell, upAxis, columnGap);
    const stairPoints = stairSink.decimate(Math.min(cell, stairSpacing));

    const result = {
      floorPoints,
      stairPoints,
      doorPoints: doorWaypoints(doors, upAxis),
      walls: wallSegments(walls, upAxis),
      stats: {
        floorSamples: floorSink.count,
        stairSamples: stairSink.count,
        floorPoints: floorPoints.length / 3,
        stairPoints: stairPoints.length / 3,
        collapsedAway: (floorRaw.length - floorPoints.length) / 3,
        windingFallback: floorResult.windingFallback || stairResult.windingFallback,
        ms: performance.now() - t0,
      },
    };

    self.postMessage({ id, ok: true, result }, [
      result.floorPoints.buffer,
      result.stairPoints.buffer,
    ]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
