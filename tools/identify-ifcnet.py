"""Port IFCNet's DGCNN geometry classifier to ONNX for the browser, optimise
it, and check every variant against the float32 PyTorch reference.

IFCNet: Emunds, Pauen, Richter, Frisch and van Treeck (2021), "IFCNet: A
Benchmark Dataset for IFC Entity Classification", EG-ICE 2021,
arXiv:2106.09712; code https://github.com/RWTH-E3D/ifcnet-models (MIT). The
DGCNN below is that repository's src/models/models.py definition (itself
after https://github.com/WangYueFt/dgcnn), kept as the reference.

The pretrained weights are NOT in this repository: the MIT licence covers the
code, and the weights' terms are unstated (see
wiki/research/ifc-identification-benchmark-2026-11-05.md). Download
dgcnn_model.zip from https://ifcnet.e3d.rwth-aachen.de/ yourself and unzip it.
Then, in a venv outside OneDrive with CPU PyTorch, onnx, onnxruntime, onnxsim
and onnxconverter-common:

  python tools/identify-ifcnet.py export --weights DIR --out DIR
  python tools/identify-ifcnet.py verify --weights DIR --out DIR --ply DIR [--corpus FILE.bin]
  python tools/identify-ifcnet.py fixtures --weights DIR --out DIR --corpus FIXTURES.bin

`export` writes the ONNX variants and DIR/manifest.json. `verify` scores the
reference and every variant on IFCNetCore's test split (the PLY release, 2,379
objects) and on corpus point clouds written by `node tools/identify-bench.mjs
extract` (FILE.bin, float32 [M, 2048, 3]), and writes DIR/verify.json:
accuracy, balanced accuracy, weighted F1, the share of predictions that
change against the reference, the largest per-class recall drop, and the
largest logit difference.

The shipped graph ("fused") is an exact algebraic rewrite of DGCNN's
EdgeConv. Each EdgeConv is a 1x1 convolution over [x_j - x_i, x_i] for the k
neighbours j of point i, then batch norm, LeakyReLU and a max over j. With
the batch norm folded into the weights (A, D, b), the pre-activation is
A x_j + D x_i + b, and LeakyReLU is increasing, so

    max_j LeakyReLU(A x_j + D x_i + b) = LeakyReLU(max_j (A x_j) + D x_i + b).

A x is computed once per point instead of once per edge: the convolutions
cost N x C_in x C_out instead of N x k x 2 C_in x C_out (k = 40), and the
[N, k, 2 C_in] edge tensor is never built. Only the k-nearest-neighbour
search and the gather stay per edge.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = [
    "IfcAirTerminal", "IfcBeam", "IfcCableCarrierFitting", "IfcCableCarrierSegment", "IfcDoor",
    "IfcDuctFitting", "IfcDuctSegment", "IfcFurniture", "IfcLamp", "IfcOutlet", "IfcPipeFitting",
    "IfcPipeSegment", "IfcPlate", "IfcRailing", "IfcSanitaryTerminal", "IfcSlab", "IfcSpaceHeater",
    "IfcStair", "IfcValve", "IfcWall",
]
K = 40
EMB = 1024
# Folded weights below this are dead channels (DGCNNFused).
PRUNE = 1e-6
EPS = 1e-5


# ---------------------------------------------------------------- reference

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20):
    batch_size, _, num_points = x.size()
    idx = knn(x, k=k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNN(nn.Module):
    """IFCNet's DGCNN, unchanged (input [B, 3, N])."""

    def __init__(self, dropout=0.5, k=K, emb_dims=EMB, output_channels=20):
        super().__init__()
        self.k = k
        self.bn1, self.bn2 = nn.BatchNorm2d(64), nn.BatchNorm2d(64)
        self.bn3, self.bn4 = nn.BatchNorm2d(128), nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        lrelu = lambda: nn.LeakyReLU(negative_slope=0.2)  # noqa: E731
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), self.bn1, lrelu())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), self.bn2, lrelu())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), self.bn3, lrelu())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), self.bn4, lrelu())
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1, bias=False), self.bn5, lrelu())
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        b = x.size(0)
        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1)[0]
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1)[0]
        x = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        x = torch.cat((F.adaptive_max_pool1d(x, 1).view(b, -1), F.adaptive_avg_pool1d(x, 1).view(b, -1)), 1)
        x = self.dp1(F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2))
        x = self.dp2(F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2))
        return self.linear3(x)


class PointsMajor(nn.Module):
    """The reference with the browser's input layout, [B, N, 3]."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, points):
        return self.net(points.permute(0, 2, 1))


def load_reference(weights_dir: Path) -> DGCNN:
    params = json.loads((weights_dir / "DGCNNParams.json").read_text())
    net = DGCNN(params["dropout"], params["k"], params["embedding_dim"], len(CLASSES))
    state, _ = torch.load(weights_dir / "DGCNNWeights+Optimizer", map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return net.eval()


# ---------------------------------------------------------------- fused

def flush(t):
    """Subnormal weights to zero: they change nothing and make CPUs take a
    slow path (the folded weights had 0.9 M subnormals; fp32 ran 6 times
    slower than the same graph with dequantised weights)."""
    t = t.detach().to(torch.float32)
    return torch.where(t.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(t), t)


def bn_affine(bn: nn.BatchNorm1d | nn.BatchNorm2d):
    a = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    return a, bn.bias.detach() - a * bn.running_mean.detach()


def leaky(x):
    return torch.where(x >= 0, x, 0.2 * x)


class DGCNNFused(nn.Module):
    """The same function, rewritten: points-major [B, N, C], batch norms folded,
    EdgeConv as one per-point projection, a gather and a max (module doc).

    prune: drop the channels training left dead. Weight decay drove many
    batch-norm scales to zero (bn5: 858 of 1,024; bn4: 204 of 256), so those
    channels output LeakyReLU(bias) whatever the input: a constant. A constant
    channel adds nothing to the neighbour distances, passes through the
    neighbour max unchanged, and its max- and mean-pool are itself, so its
    contribution to every later layer folds into that layer's bias. A channel
    is dropped when all its folded weights are below `prune` in magnitude.
    """

    def __init__(self, ref: DGCNN, prune: float | None = None, k: int | None = None):
        super().__init__()
        # k: neighbours per point. The weights do not depend on it; with fewer
        # points, fewer neighbours keep the neighbourhood the size training saw
        # (k = 40 of 2,048 points is about k = 20 of 1,024).
        self.k = k or ref.k
        self.edge = nn.ParameterList()
        dd = torch.float64
        P = lambda t: nn.Parameter(flush(t).to(torch.float32).contiguous(), requires_grad=False)  # noqa: E731
        self.pruned = {}
        keep_in = torch.ones(3, dtype=torch.bool)
        const_in = torch.zeros(3, dtype=dd)
        outputs = []  # (keep mask, constants) of x1..x4
        for i, (conv, bn) in enumerate(((ref.conv1, ref.bn1), (ref.conv2, ref.bn2), (ref.conv3, ref.bn3), (ref.conv4, ref.bn4))):
            w = conv[0].weight.detach()[:, :, 0, 0].to(dd)
            c = w.shape[1] // 2
            a, b = (t.to(dd) for t in bn_affine(bn))
            A = a[:, None] * w[:, :c]
            D = a[:, None] * (w[:, c:] - w[:, :c])
            # Dropped (constant) inputs: their A and D terms are constants.
            b = b + (A[:, ~keep_in] + D[:, ~keep_in]) @ const_in[~keep_in]
            A, D = A[:, keep_in], D[:, keep_in]
            dead = torch.maximum(A.abs().amax(1), D.abs().amax(1)) < prune if prune else torch.zeros(A.shape[0], dtype=torch.bool)
            const = leaky(b) * dead
            keep = ~dead
            self.pruned[f"edge{i + 1}"] = [int(keep.sum()), int(keep.numel())]
            # One matmul gives both projections: [C_in kept, 2 C_out kept].
            self.edge.append(P(torch.cat([A[keep], D[keep]], 0).t()))
            self.edge.append(P(b[keep]))
            outputs.append((keep, const))
            keep_in, const_in = keep, const
        keep_cat = torch.cat([k for k, _ in outputs])
        const_cat = torch.cat([v for _, v in outputs])
        a5, b5 = (t.to(dd) for t in bn_affine(ref.bn5))
        W5 = a5[:, None] * ref.conv5[0].weight.detach()[:, :, 0].to(dd)
        b5 = b5 + W5[:, ~keep_cat] @ const_cat[~keep_cat]
        W5 = W5[:, keep_cat]
        dead5 = W5.abs().amax(1) < prune if prune else torch.zeros(W5.shape[0], dtype=torch.bool)
        const5 = leaky(b5) * dead5
        keep5 = ~dead5
        self.pruned["conv5"] = [int(keep5.sum()), int(keep5.numel())]
        self.w5, self.b5 = P(W5[keep5].t()), P(b5[keep5])
        # linear1 reads [max-pool, mean-pool]; a constant channel pools to itself.
        a6, b6 = (t.to(dd) for t in bn_affine(ref.bn6))
        W6 = a6[:, None] * ref.linear1.weight.detach().to(dd)
        keep6_in = torch.cat([keep5, keep5])
        const6_in = torch.cat([const5, const5])
        b6 = b6 + W6[:, ~keep6_in] @ const6_in[~keep6_in]
        W6 = W6[:, keep6_in]
        dead6 = W6.abs().amax(1) < prune if prune else torch.zeros(W6.shape[0], dtype=torch.bool)
        const6 = leaky(b6) * dead6
        keep6 = ~dead6
        self.pruned["linear1"] = [int(keep6.sum()), int(keep6.numel())]
        self.w6, self.b6 = P(W6[keep6].t()), P(b6[keep6])
        a7, b7 = (t.to(dd) for t in bn_affine(ref.bn7))
        W7 = a7[:, None] * ref.linear2.weight.detach().to(dd)
        b7 = a7 * ref.linear2.bias.detach().to(dd) + b7 + W7[:, ~keep6] @ const6[~keep6]
        self.w7, self.b7 = P(W7[:, keep6].t()), P(b7)
        self.w8 = P(ref.linear3.weight.detach().t())
        self.b8 = P(ref.linear3.bias.detach())

    def edge_conv(self, x, w, b):
        bsz, n, _ = x.shape
        # k nearest neighbours in feature space, self included, as in the reference.
        g = torch.matmul(x, x.transpose(1, 2))
        xx = (x * x).sum(-1, keepdim=True)
        idx = (2 * g - xx - xx.transpose(1, 2)).topk(self.k, dim=-1)[1]
        cout = w.shape[1] // 2
        p = torch.matmul(x, w)
        a, d = p[..., :cout], p[..., cout:]
        flat = idx + (torch.arange(bsz, device=x.device) * n).view(-1, 1, 1)
        nb = a.reshape(bsz * n, cout)[flat.reshape(-1)].view(bsz, n, self.k, cout)
        return F.leaky_relu(nb.max(dim=2)[0] + d + b, 0.2)

    def forward(self, points):
        x1 = self.edge_conv(points, self.edge[0], self.edge[1])
        x2 = self.edge_conv(x1, self.edge[2], self.edge[3])
        x3 = self.edge_conv(x2, self.edge[4], self.edge[5])
        x4 = self.edge_conv(x3, self.edge[6], self.edge[7])
        x = F.leaky_relu(torch.matmul(torch.cat((x1, x2, x3, x4), -1), self.w5) + self.b5, 0.2)
        x = torch.cat((x.max(dim=1)[0], x.mean(dim=1)), -1)
        x = F.leaky_relu(torch.matmul(x, self.w6) + self.b6, 0.2)
        x = F.leaky_relu(torch.matmul(x, self.w7) + self.b7, 0.2)
        return torch.matmul(x, self.w8) + self.b8


# ---------------------------------------------------------------- export

def export(args):
    import onnx

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_reference(Path(args.weights))
    fused = DGCNNFused(ref).eval()
    manifest = {"classes": CLASSES, "k": K, "input": "points [B, N, 3] float32, centred on the vertex mean and scaled to the unit sphere", "output": "logits [B, 20]", "variants": []}

    def save(model, name, n, b, graph):
        path = out / f"{name}.onnx"
        x = torch.rand(b, n, 3) * 2 - 1
        torch.onnx.export(model, (x,), str(path), input_names=["points"], output_names=["logits"], opset_version=17, dynamo=False)
        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        manifest["variants"].append({"name": name, "points": n, "batch": b, "graph": graph, "precision": "fp32", "bytes": path.stat().st_size})
        return path

    # The unchanged graph, and the rewrite without pruning, for what each step buys.
    save(PointsMajor(ref), "naive-fp32-n2048-b1", 2048, 1, "naive")
    save(DGCNNFused(ref).eval(), "fused-fp32-n1024-b1", 1024, 1, "fused")
    pruned = DGCNNFused(ref, prune=PRUNE).eval()
    manifest["pruned"] = pruned.pruned
    for n, k in ((512, 40), (1024, 40), (2048, 40), (512, 10), (1024, 20)):
        net = pruned if k == K else DGCNNFused(ref, prune=PRUNE, k=k).eval()
        tag = f"n{n}" if k == K else f"n{n}k{k}"
        for b in (1, 8):
            raw = save(net, f"pruned-raw-{tag}-b{b}", n, b, "fused+pruned")
            manifest["variants"][-1]["k"] = k
            for v in optimise(raw, out, n, b):
                v["k"] = k
                manifest["variants"].append(v)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for v in manifest["variants"]:
        print(f"{v['name']:32s} {v['bytes'] / 1e6:7.2f} MB")


def optimise(raw: Path, out: Path, n: int, b: int):
    """Simplified fp32, int8 (dynamic, MatMul weights) and fp16 from one raw export."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    variants = []
    prefix, base = raw.stem.split("-raw-")
    sim_path = out / f"{prefix}-fp32-{base}.onnx"
    model = onnx.load(str(raw))
    try:
        from onnxsim import simplify
        model, ok = simplify(model)
        assert ok, "onnxsim check failed"
        how = "onnxsim"
    except ImportError:
        how = "none"
    onnx.save(model, str(sim_path))
    variants.append({"name": sim_path.stem, "points": n, "batch": b, "graph": "fused+pruned+" + how, "precision": "fp32", "bytes": sim_path.stat().st_size})

    q_path = out / f"{prefix}-int8-{base}.onnx"
    quantize_dynamic(str(sim_path), str(q_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=False)
    variants.append({"name": q_path.stem, "points": n, "batch": b, "graph": "fused+pruned+" + how, "precision": "int8 dynamic (MatMul weights, per-tensor)", "bytes": q_path.stat().st_size})

    try:
        from onnxconverter_common import float16
        h = float16.convert_float_to_float16(onnx.load(str(sim_path)), keep_io_types=True)
        h_path = out / f"{prefix}-fp16-{base}.onnx"
        onnx.save(h, str(h_path))
        variants.append({"name": h_path.stem, "points": n, "batch": b, "graph": "fused+pruned+" + how, "precision": "fp16 (fp32 inputs and outputs)", "bytes": h_path.stat().st_size})
    except ImportError:
        pass

    # Weight-only int8: int8 initialisers with a DequantizeLinear in front of
    # each MatMul, so the arithmetic stays fp32 and only the download shrinks.
    w_path = out / f"{prefix}-w8-{base}.onnx"
    weight_only_int8(onnx.load(str(sim_path)), w_path)
    variants.append({"name": w_path.stem, "points": n, "batch": b, "graph": "fused+pruned+" + how, "precision": "int8 weights, fp32 compute", "bytes": w_path.stat().st_size})
    return variants


def weight_only_int8(model, path: Path):
    import onnx
    from onnx import helper, numpy_helper

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    # Weight name -> the axis of its output channels (MatMul [K, N]: 1; Gemm with transB: 0).
    weights = {}
    for n in g.node:
        if n.op_type in ("MatMul", "Gemm") and len(n.input) > 1 and n.input[1] in inits:
            trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
            weights[n.input[1]] = 0 if (n.op_type == "Gemm" and trans_b) else 1
    new_nodes = []
    for name in sorted(weights):
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        if w.ndim != 2:
            continue
        axis = weights[name]
        # Symmetric scale per output channel.
        scale = np.maximum(np.abs(w).max(axis=1 - axis), 1e-12) / 127.0
        shape = (1, -1) if axis == 1 else (-1, 1)
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        g.initializer.remove(inits[name])
        g.initializer.extend([
            numpy_helper.from_array(q, name + "_q"),
            numpy_helper.from_array(scale.astype(np.float32), name + "_scale"),
            numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "_zp"),
        ])
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale", name + "_zp"], [name], axis=axis, name=name + "_dq"))
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


# ---------------------------------------------------------------- verify

def read_ply_points(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    count = next(int(line.split()[2]) for line in lines[:end] if line.startswith("element vertex"))
    return np.array([[float(v) for v in line.split()[:3]] for line in lines[end + 1:end + 1 + count]], dtype=np.float32)


def ifcnet_test_split(ply_root: Path, cache: Path):
    if cache.exists():
        d = np.load(cache)
        return d["points"], d["labels"]
    files = sorted(ply_root.glob("**/test/*.ply"))
    pts = np.stack([read_ply_points(f) for f in files])
    labels = np.array([CLASSES.index(f.parts[-3]) for f in files])
    np.savez_compressed(cache, points=pts, labels=labels)
    return pts, labels


def metrics(labels, pred, ncls=len(CLASSES)):
    labels, pred = np.asarray(labels), np.asarray(pred)
    acc = float((labels == pred).mean())
    rec, f1, sup = [], [], []
    for c in range(ncls):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        if tp + fn == 0:
            continue
        r = tp / (tp + fn)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
        sup.append(tp + fn)
    sup = np.array(sup)
    return {"accuracy": acc, "balancedAccuracy": float(np.mean(rec)), "weightedF1": float((np.array(f1) * sup).sum() / sup.sum()), "macroF1": float(np.mean(f1)), "recall": rec}


def run_torch(model, pts, batch=16):
    outs = []
    with torch.no_grad():
        for i in range(0, len(pts), batch):
            outs.append(model(torch.from_numpy(pts[i:i + batch])).numpy())
    return np.concatenate(outs)


def run_onnx(path: Path, pts, batch):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    outs = []
    for i in range(0, len(pts), batch):
        chunk = pts[i:i + batch]
        pad = batch - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, 0)])
        outs.append(sess.run(None, {"points": chunk})[0][:batch - pad])
    return np.concatenate(outs)


def stratified(labels, per_class, seed=0):
    rng = np.random.default_rng(seed)
    pick = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        pick.extend(rng.permutation(idx)[:per_class].tolist())
    return np.array(sorted(pick))


def verify(args):
    """Every variant against the float32 PyTorch reference.

    The reference is IFCNet's own graph (DGCNN above) at 2,048 points, on a
    stratified sample of IFCNetCore's test split (`--per-class` objects per
    class) and of the corpus (`--corpus-count`): IFCNet's graph runs at about
    a second an object on this CPU even with subnormals flushed, so the full
    split would take hours. The unpruned rewrite (DGCNNFused) is the same-N
    reference for 512 and 1,024 points.
    """
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    base = load_reference(Path(args.weights))
    naive, fused = PointsMajor(base), DGCNNFused(base).eval()
    fused20, fused10 = DGCNNFused(base, k=20).eval(), DGCNNFused(base, k=10).eval()
    pts, labels = ifcnet_test_split(Path(args.ply), out / "ifcnet-test.npz")
    pick = stratified(labels, args.per_class)
    rng = np.random.default_rng(0)
    perm = rng.permutation(pts.shape[1])
    sets = {"ifcnet-test": (pts[pick], labels[pick])}
    if args.corpus:
        corpus = np.fromfile(args.corpus, dtype=np.float32).reshape(-1, 2048, 3)
        cpick = np.sort(np.random.default_rng(1).permutation(len(corpus))[:args.corpus_count])
        sets["corpus"] = (corpus[cpick], None)
        np.save(out / "corpus-pick.npy", cpick)
    result = {"note": verify.__doc__.strip(), "perClass": args.per_class, "sets": {}, "variants": {}}
    reference = {}
    for name, (p, lab) in sets.items():
        for n, k, model in ((2048, K, naive), (2048, K, fused), (1024, K, fused), (512, K, fused), (1024, 20, fused20), (512, 10, fused10)):
            key = (name, n, k, "naive" if model is naive else "fused")
            sub = np.ascontiguousarray(p[:, perm[:n]]) if n < 2048 else p
            t = time.perf_counter()
            logits = run_torch(model, sub, batch=8)
            reference[key] = logits
            entry = {"secondsPerObject": (time.perf_counter() - t) / len(p), "count": len(p)}
            if lab is not None:
                entry.update({k: v for k, v in metrics(lab, logits.argmax(1)).items() if k != "recall"})
            if key[3] == "fused" and n == 2048:
                r = reference[(name, 2048, K, "naive")]
                entry["vsIFCNetGraph"] = {"maxAbsLogitDiff": float(np.abs(logits - r).max()), "changed": float((logits.argmax(1) != r.argmax(1)).mean())}
            result["sets"][f"{name}/{key[3]}-n{n}-k{k}"] = entry
            print(name, key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}, flush=True)
    for v in manifest["variants"]:
        if v["batch"] != 1 or "-raw-" in v["name"]:
            continue
        path = out / f"{v['name']}.onnx"
        n, k = v["points"], v.get("k", K)
        row = {"bytes": v["bytes"], "points": n, "k": k, "precision": v["precision"], "graph": v["graph"]}
        for name, (p, lab) in sets.items():
            sub = np.ascontiguousarray(p[:, perm[:n]]) if n < 2048 else p
            t = time.perf_counter()
            logits = run_onnx(path, sub, 1)
            secs = (time.perf_counter() - t) / len(p)
            same = reference[(name, n, k, "fused")]
            ref = reference[(name, 2048, K, "naive")]
            stats = {
                "secondsPerObjectOrtCpu": secs,
                "nan": bool(np.isnan(logits).any()),
                "maxAbsLogitDiffSameN": float(np.abs(logits - same).max()),
                "changedVsSameN": float((logits.argmax(1) != same.argmax(1)).mean()),
                "changedVsReference": float((logits.argmax(1) != ref.argmax(1)).mean()),
            }
            if lab is not None:
                m = metrics(lab, logits.argmax(1))
                m0 = metrics(lab, ref.argmax(1))
                stats.update({k: m[k] for k in ("accuracy", "balancedAccuracy", "weightedF1", "macroF1")})
                drops = np.array(m0["recall"]) - np.array(m["recall"])
                worst = int(drops.argmax())
                stats["largestClassRecallDrop"] = {"class": CLASSES[worst], "drop": float(drops[worst])}
            row[name] = stats
        result["variants"][v["name"]] = row
        print(v["name"], json.dumps(row), flush=True)
        (out / "verify.json").write_text(json.dumps(result, indent=2))
    (out / "verify.json").write_text(json.dumps(result, indent=2))


def fixtures(args):
    """Logits of every batch-1 variant (onnxruntime, CPU) for a few clouds, the
    first n points of each: the browser must give the same (identify-bench.html)."""
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    clouds = np.fromfile(args.corpus, dtype=np.float32).reshape(-1, 2048, 3)
    result = {}
    for v in manifest["variants"]:
        if v["batch"] != 1 or "-raw-" in v["name"]:
            continue
        n = v["points"]
        sub = np.ascontiguousarray(clouds[:, :n])
        logits = run_onnx(out / f"{v['name']}.onnx", sub, 1)
        result[v["name"]] = {"items": [{"points": [round(float(x), 7) for x in sub[i].ravel()], "logits": [float(x) for x in logits[i]]} for i in range(len(sub))]}
    (out / "fixtures.json").write_text(json.dumps(result))
    print("fixtures for", len(result), "variants")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "verify", "fixtures"])
    ap.add_argument("--weights", required=True, help="folder with DGCNNWeights+Optimizer and DGCNNParams.json")
    ap.add_argument("--out", required=True, help="output folder (keep it outside the repository)")
    ap.add_argument("--ply", help="IFCNetCore PLY release root (verify)")
    ap.add_argument("--corpus", help="corpus points, float32 [M, 2048, 3] (verify)")
    ap.add_argument("--per-class", type=int, default=20, help="IFCNetCore test objects per class (verify)")
    ap.add_argument("--corpus-count", type=int, default=400, help="corpus elements (verify)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.set_flush_denormal(True)
    {"export": export, "verify": verify, "fixtures": fixtures}[args.command](args)


if __name__ == "__main__":
    main()
