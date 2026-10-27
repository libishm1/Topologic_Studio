"""Plot a lattice graph dumped by tools/lattice-check.mjs --dump, one panel per storey.

    python tools/plot_lattice.py graph.json out.png

Nodes are coloured by connected component (walls on), so a building that has
fallen apart shows as several colours. Walls are grey lines, blocked edges
red, doors black squares. Y is up in the dump (web-ifc's frame), so plan is
x against z.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

data = json.load(open(sys.argv[1]))
p = np.array(data["points"]).reshape(-1, 3)
e = np.array(data["edges"]).reshape(-1, 2)
blocked = np.array(data["blocked"], bool)
kinds = np.array(data["kinds"])
n = len(p)
open_e = e[~blocked]
_, labels = connected_components(
    coo_matrix((np.ones(len(open_e)), (open_e[:, 0], open_e[:, 1])), shape=(n, n)), directed=False
)
sizes = np.bincount(labels)
rank = np.argsort(-sizes)
colour_of = np.empty_like(rank)
colour_of[rank] = np.arange(len(rank))

for r, lab in enumerate(rank[:12]):
    m = labels == lab
    print(
        f"part {r}: {m.sum():5d} nodes  height {p[m, 1].min():5.2f}..{p[m, 1].max():5.2f}  "
        f"x {p[m, 0].min():6.1f}..{p[m, 0].max():6.1f}  z {p[m, 2].min():6.1f}..{p[m, 2].max():6.1f}  "
        f"stair {int((kinds[m] == 1).sum())} door {int((kinds[m] == 2).sum())}"
    )

# Storeys: cluster floor-node heights with a 1.5 m gap. Stair nodes are left
# out, since they fill the gap between storeys and would merge them.
h = p[:, 1]
order = np.sort(np.unique(np.round(h[kinds != 1], 1)))
bands, current = [], [order[0]]
for v in order[1:]:
    if v - current[-1] > 1.5:
        bands.append(current)
        current = [v]
    else:
        current.append(v)
bands.append(current)
panels = [(b[0] - 0.05, b[-1] + 0.05) for b in bands]

fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6.5), squeeze=False)
cmap = plt.get_cmap("tab20")
for ax, (lo, hi) in zip(axes[0], panels):
    sel = (h >= lo) & (h <= hi)
    for w in data["walls"]:
        if w["up_min"] < hi + 1.0 and w["up_max"] > lo - 0.5:
            (x0, z0), (x1, z1) = w["segment"]
            ax.plot([x0, x1], [z0, z1], color="0.6", lw=max(0.5, w["thickness"] * 8), zorder=1)
    for (a, b), bl in zip(e, blocked):
        if sel[a] and sel[b]:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 2], p[b, 2]], color="red" if bl else "0.8", lw=0.4, zorder=2)
    ax.scatter(p[sel, 0], p[sel, 2], s=4, c=[cmap(colour_of[l] % 20) for l in labels[sel]], zorder=3)
    st = sel & (kinds == 1)
    ax.scatter(p[st, 0], p[st, 2], s=14, facecolors="none", edgecolors="purple", lw=0.6, zorder=4)
    for d in data["doors"]:
        if lo - 1 <= d["sill"] + 0.75 <= hi + 1:
            ax.plot(d["x"], d["y"], "ks", ms=4, zorder=5)
    ax.set_title(f"{data['file']}  y {lo:.1f}..{hi:.1f}  ({sel.sum()} nodes)", fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
fig.suptitle(f"{len(sizes)} parts, largest {sizes.max() / n:.0%}  (colour = part; purple rings = stair; red = blocked)", fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
print(sys.argv[2])
