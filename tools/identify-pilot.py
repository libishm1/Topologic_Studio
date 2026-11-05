"""Pilot for the IFCNet furniture study: geometric features, the name rules and
IFCNet's DGCNN embedding, scored with leave-buildings-out splits on the local
corpus. Plan and results: wiki/research/ifcnet-furniture-study-plan-2026-11-05.md.

    python -I tools/identify-pilot.py embed [--bench DIR] [--pilot DIR] [--onnx FILE] [--threads 8]
        Runs the pruned DGCNN (tools/identify-ifcnet.py, batch 8, 2,048 points)
        on every element of PILOT/features.json and keeps three of its tensors:
        the pooled embedding (max and mean of the 166 kept channels), the
        256-wide penultimate layer and the 20 logits. Writes PILOT/embed.npz.
    python -I tools/identify-pilot.py timetrain [--pilot DIR] [--threads 8]
        Times DGCNN training steps on this CPU from a random initialisation
        (no IFCNet weights: the openly licensed variant trains from scratch),
        for the full campaign's estimate. Writes PILOT/train-timing.json.
    python -I tools/identify-pilot.py recheck --check DIR [--pilot DIR] [--out FILE]
        Compares PILOT/features.json with a second extraction in DIR (run
        `node tools/identify-features.mjs extract --out DIR` again) and adds
        the comparison, the second run's timing and a count of the ignition
        scenarios the corpus's names reach to the results file.
    python -I tools/identify-pilot.py run [--bench DIR] [--pilot DIR] [--out FILE] [--threads 15]
        The experiments (section 5 of the plan; five folds fitted in parallel,
        threads shared between them). Writes aggregate numbers only to
        benchmark/results/2026-11-05-ifcnet-pilot.json; run recheck after it.

BENCH (default ~/ifcnet-src/bench-data) holds the benchmark's dataset.json,
points.bin, eval.json, hard-labels.json and preds-*.json; PILOT (default
~/ifcnet-src/pilot) holds the output of `node tools/identify-features.mjs
extract`. Both hold third-party names and IFCNet data and stay outside the
repository; the results file holds counts and scores, never a name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
LABELS = ["wall", "slab", "roof", "door", "window", "stair", "railing", "column", "beam", "covering",
          "furniture", "appliance", "sanitary", "heater", "lighting", "mep", "space", "other"]
SEED = 20261105

# Buildings that appear more than once in the corpus (another export of the same
# design): they are kept in one group so no building is on both sides of a split.
SAME_BUILDING = {
    "AC20-FZK-Haus": "fzk-haus", "ISSUE_005_haus": "fzk-haus", "osb-testfiles-adt-fzk-haus": "fzk-haus",
    "kit-smiley-west": "smiley-west", "osb-testfiles-allplan-smiley-west": "smiley-west",
    "advanced_model": "revit-advanced", "AdvancedProject": "revit-advanced", "Ifc4_Revit_ARC": "revit-advanced",
    "duplex-apartment": "duplex", "bsc-duplex": "duplex",
    "bsi-ifc4-Building-Architecture": "bsi-scene", "buildingsmart-sample-scene": "bsi-scene",
}

# Licence of each corpus model as its source states it (benchmark/manifest.json
# and the more2 batch's source notes). "open": CC0, CC BY 4.0, MIT or KIT's
# "unrestricted use"; everything else (no stated licence, GPL, CC BY-ND, Autodesk
# sample projects) is kept out of the openly licensed variant. The Duplex's
# furniture is Autodesk default family content (plan, section 2.2).
OPEN_MODELS = {
    "AC20-FZK-Haus": "KIT unrestricted use", "C20-Institute-Var-2": "KIT unrestricted use",
    "IfcOpenHouse_IFC4": "CC0-1.0", "dental_clinic": "CC-BY-4.0 (buildingSMART community samples)",
    "bsi-ifc4-Building-Architecture": "CC-BY-4.0 (buildingSMART)", "buildingsmart-sample-scene": "CC-BY-4.0 (buildingSMART)",
    "duplex-apartment": "CC-BY-4.0 (buildingSMART community samples)", "bsc-duplex": "CC-BY-4.0 (buildingSMART community samples)",
    "office-two-storey": "CC0-1.0 (generated)", "courtyard-ring": "CC0-1.0 (generated)",
    "LargeBuilding": "MIT (generated)", "ManySimpleWalls": "MIT (generated)", "TallBuilding": "MIT (generated)",
    "bsc-esplanades": "CC-BY-4.0", "zenodo-taltech-ds3": "CC-BY-4.0", "zenodo-klassiqua-office-1970": "CC-BY-4.0",
    "zenodo-balconies-a5": "CC-BY-4.0", "kit-smiley-west": "KIT unrestricted use",
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stem(model: str) -> str:
    return model.split("/")[0]


def building(model: str) -> str:
    return SAME_BUILDING.get(stem(model), stem(model))


def is_open(model: str) -> bool:
    return stem(model) in OPEN_MODELS


# ---------------------------------------------------------------- scoring

def score(truth, pred, labels=None):
    """Accuracy, macro-F1 over the labels present in the truth (as tools/identify-bench.mjs), per-label F1."""
    truth = list(truth)
    pred = list(pred)
    present = [l for l in (labels or sorted(set(truth))) if l in set(truth)]
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(truth, pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1 = {}
    for l in present:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l])
        f1[l] = 2 * P * R / (P + R) if P + R else 0.0
    acc = sum(tp.values()) / len(truth) if truth else 0.0
    return {"n": len(truth), "accuracy": round(acc, 4), "macroF1": round(float(np.mean(list(f1.values()))) if f1 else 0.0, 4),
            "f1": {k: round(v, 3) for k, v in f1.items()}}


def bootstrap_diff(truth, a, b, groups, labels=None, reps=1000, seed=SEED):
    """Cluster bootstrap over buildings of (b - a) in accuracy and macro-F1: mean and 95 % interval."""
    rng = np.random.default_rng(seed)
    truth, a, b, groups = map(np.asarray, (truth, a, b, groups))
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    da, dm = [], []
    for _ in range(reps):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        sa, sb = score(truth[pick], a[pick], labels), score(truth[pick], b[pick], labels)
        da.append(sb["accuracy"] - sa["accuracy"])
        dm.append(sb["macroF1"] - sa["macroF1"])
    q = lambda xs: [round(float(np.percentile(xs, 2.5)), 4), round(float(np.percentile(xs, 97.5)), 4)]
    return {"accuracy": {"mean": round(float(np.mean(da)), 4), "ci95": q(da)}, "macroF1": {"mean": round(float(np.mean(dm)), 4), "ci95": q(dm)}}


# ---------------------------------------------------------------- data

def load(bench: Path, pilot: Path):
    feats = json.loads((pilot / "features.json").read_text(encoding="utf8"))
    names = feats["featureNames"]
    recs = feats["records"]
    by_id = {r["id"]: r for r in recs}
    ds = json.loads((bench / "dataset.json").read_text(encoding="utf8"))["records"]
    hard = json.loads((bench / "hard-labels.json").read_text(encoding="utf8"))
    ev = json.loads((bench / "eval.json").read_text(encoding="utf8"))["records"]
    # eval.json's order: the dataset's typed records, then its hand-labelled generic ones (prepare in identify-bench.mjs).
    order = [r for r in ds if r["label"]] + [r for r in ds if r["id"] in hard]
    assert len(order) == len(ev), (len(order), len(ev))
    for r, e in zip(order, ev):
        assert r["model"] == e["model"] and r["name"] == e["name"] and r["dims"] == e["dims"], "eval order differs from the dataset"
    return feats, names, recs, by_id, order, ev, hard


def family_key(r):
    text = (r.get("typeName") or r.get("objectType") or r.get("name") or "").lower()
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"[^a-zäöüøæåéèà]+", " ", text).strip()
    dims = tuple(round(d / 0.05) for d in r["dims"])
    return (r["model"], text, dims)


# ---------------------------------------------------------------- embed

def embed(args):
    import onnx
    import onnxruntime as ort

    bench, pilot = Path(args.bench), Path(args.pilot)
    feats = json.loads((pilot / "features.json").read_text(encoding="utf8"))
    recs = feats["records"]
    ds_pts = np.memmap(bench / "points.bin", dtype=np.float32, mode="r").reshape(-1, 2048, 3)
    ex_pts = np.fromfile(pilot / "extra-points.bin", dtype=np.float32).reshape(-1, 2048, 3)
    model = onnx.load(args.onnx)
    g = model.graph
    gemm = {n.input[1]: n for n in g.node if n.op_type == "Gemm"}
    pooled, hidden = gemm["w6"].input[0], gemm["w8"].input[0]
    for name in (pooled, hidden):
        g.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    out_names = ["logits", pooled, hidden]
    pts = np.stack([ds_pts[r["datasetIndex"]] if "datasetIndex" in r else ex_pts[r["extraIndex"]] for r in recs]).astype(np.float32)
    n = len(pts)
    res = {k: [] for k in ("logits", "pooled", "hidden")}
    t0 = time.time()
    for i in range(0, n, 8):
        batch = pts[i:i + 8]
        pad = 8 - len(batch)
        if pad:
            batch = np.concatenate([batch, np.repeat(batch[-1:], pad, axis=0)])
        lo, po, hi = sess.run(out_names, {"points": batch})
        keep = 8 - pad
        res["logits"].append(lo[:keep]); res["pooled"].append(po[:keep]); res["hidden"].append(hi[:keep])
        if i % 800 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f} s", flush=True)
    secs = time.time() - t0
    np.savez(pilot / "embed.npz", ids=np.array([r["id"] for r in recs]), **{k: np.concatenate(v) for k, v in res.items()},
             seconds=secs, threads=args.threads)
    print(f"{n} elements in {secs:.0f} s ({1000 * secs / n:.1f} ms an element, batch 8, {args.threads} threads)")


# ---------------------------------------------------------------- training cost

def timetrain(args):
    import importlib.util

    import torch

    spec = importlib.util.spec_from_file_location("identify_ifcnet", Path(__file__).with_name("identify-ifcnet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(SEED)
    out = {"threads": args.threads, "torch": torch.__version__, "configs": []}
    for points, k, batch in ((2048, 40, 16), (1024, 20, 16), (1024, 20, 32)):
        torch.set_num_threads(args.threads)
        net = mod.DGCNN(k=k, output_channels=14)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x = torch.randn(batch, 3, points)
        y = torch.randint(0, 14, (batch,))
        times = []
        for step in range(6):
            t0 = time.time()
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(x), y)
            loss.backward()
            opt.step()
            if step >= 2:
                times.append(time.time() - t0)
        net.eval()
        with torch.no_grad():
            t0 = time.time()
            net(x)
            infer = time.time() - t0
        cfg = {"points": points, "k": k, "batch": batch, "secondsPerStep": round(float(np.median(times)), 3),
               "msPerSampleTrain": round(1000 * float(np.median(times)) / batch, 1), "msPerSampleInfer": round(1000 * infer / batch, 1)}
        out["configs"].append(cfg)
        print(cfg, flush=True)
    Path(args.pilot, "train-timing.json").write_text(json.dumps(out, indent=1), encoding="utf8")


# ---------------------------------------------------------------- experiments

def hgb(seed=SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=seed)


def probe(seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", random_state=seed))


def folds(groups, k=5):
    """Leave-buildings-out folds: buildings sorted by element count, dealt greedily to the smallest fold."""
    count = Counter(groups)
    load_ = [0] * k
    fold_of = {}
    for gname, c in sorted(count.items(), key=lambda kv: (-kv[1], kv[0])):
        f = int(np.argmin(load_))
        fold_of[gname] = f
        load_[f] += c
    return fold_of


def _fit_fold(make, X_tr, y_tr, X_te, X_ex):
    m = make()
    m.fit(X_tr, y_tr)
    return m.predict(X_te), (m.predict(X_ex) if X_ex is not None and len(X_ex) else None)


def cross_predict(make, X, y, groups, fold_of, X_extra=None, groups_extra=None, jobs=5):
    """Leave-buildings-out predictions for every row of X (and of X_extra, by the fold of its building).
    The folds are fitted in parallel processes."""
    from joblib import Parallel, delayed

    y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    pred_extra = np.empty(len(X_extra), dtype=object) if X_extra is not None else None
    fid = np.array([fold_of[g] for g in groups])
    fid_extra = np.array([fold_of.get(g, -1) for g in groups_extra]) if X_extra is not None else None
    fs = sorted(set(fid))
    outs = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_fit_fold)(make, X[fid != f], y[fid != f], X[fid == f], X_extra[fid_extra == f] if X_extra is not None else None) for f in fs)
    for f, (p, pe) in zip(fs, outs):
        pred[fid == f] = p
        if pe is not None:
            pred_extra[fid_extra == f] = pe
    if X_extra is not None and (fid_extra == -1).any():
        m = make()
        m.fit(X, y)
        pred_extra[fid_extra == -1] = m.predict(X_extra[fid_extra == -1])
    return pred, pred_extra


def onehot(values, vocab):
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    pos = {v: i for i, v in enumerate(vocab)}
    for i, v in enumerate(values):
        if v in pos:
            out[i, pos[v]] = 1
    return out


def run(args):
    # Five folds fit in parallel processes, each with a share of the threads.
    os.environ["OMP_NUM_THREADS"] = str(max(1, args.threads // 5))
    bench, pilot = Path(args.bench), Path(args.pilot)
    feats, names, recs, by_id, order, ev, hard = load(bench, pilot)
    emb = np.load(pilot / "embed.npz")
    emb_row = {i: k for k, i in enumerate(emb["ids"].tolist())}
    F = np.array([r["f"] for r in recs], dtype=np.float64)
    rec_row = {r["id"]: k for k, r in enumerate(recs)}
    results = {"date": "2026-11-05", "seed": SEED, "featureNames": names, "featureTiming": feats["timing"], "node": feats["node"]}
    groups_all = sorted({building(r["model"]) for r in recs})
    results["corpus"] = {"models": len({r["model"] for r in recs}), "buildings": len(groups_all),
                         "recordsWithFeatures": len(recs), "openModels": len({r["model"] for r in recs if is_open(r["model"])})}
    results["embedding"] = {"seconds": float(emb["seconds"]), "threads": int(emb["threads"]), "n": int(len(emb["ids"])),
                            "msPerElement": round(1000 * float(emb["seconds"]) / len(emb["ids"]), 2)}

    # Feature groups for the ablation.
    gidx = lambda lst: [names.index(n) for n in lst]
    box = ["len", "wid", "hgt", "above", "top", "logVol", "widLen", "hgtLen", "floorKnown"]
    mesh = ["logArea", "areaRatio", "logTris"]
    eigen = ["linearity", "planarity", "scattering", "omnivariance", "anisotropy", "eigenentropy", "curvChange", "logEigSum",
             "axisVert", "verticality"] + [n for n in names if n.startswith("loc") or n.startswith("share")]
    surf = ["nUp", "nDown", "nSide", "nSlant", "meanAbsNy", "orientEntropy"] + [n for n in names if n.startswith("hRel") or n.startswith("upH")] + ["upLevels", "upTopShare", "upMainHeight"]
    segvox = ["segN1", "segN5", "segTop1", "segTop3", "segCover", "partsN", "partsTop1", "occ8", "planFill", "layer0", "layer1", "layer2", "layer3"]
    assert sorted(box + mesh + eigen + surf + segvox) == sorted(names), "feature groups do not cover the features"
    groups_def = {"box": box, "box+mesh": box + mesh, "box+eigen": box + eigen, "box+surfaces": box + surf,
                  "box+segments,voxels": box + segvox, "all": names}
    results["featureGroups"] = {k: len(v) for k, v in groups_def.items()}

    # ------------------------------------------------ task A: 18 labels, the benchmark's elements
    rows = [rec_row[r["id"]] for r in order]
    truth = np.array([e["truth"] for e in ev], dtype=object)
    subset = np.array([e["subset"] for e in ev])
    models = [e["model"] for e in ev]
    grp = np.array([building(m) for m in models])
    XA = F[rows]
    E_hidden = emb["hidden"][[emb_row[r["id"]] for r in order]]
    E_logits = emb["logits"][[emb_row[r["id"]] for r in order]]
    preds_rules = json.loads((bench / "preds-rules.json").read_text(encoding="utf8"))
    rules = np.array([p["label"] for p in preds_rules], dtype=object)
    rules_by = [p["by"] for p in preds_rules]
    ifcnet = np.array([p["label"] for p in json.loads((bench / "preds-ifcnet-pruned-fp32-n2048-b1.json").read_text(encoding="utf8"))], dtype=object)
    hyb = np.array([p["label"] for p in json.loads((bench / "preds-rules+ifcnet-pruned-fp32-n2048-b1@0.8.json").read_text(encoding="utf8"))], dtype=object)
    # The rules here (features.json) and the benchmark's must agree.
    same = sum(by_id[r["id"]]["rules"]["label"] == rules[i] for i, r in enumerate(order))
    results["checks"] = {"rulesReproduced": f"{same} of {len(order)}"}
    # The embedding run's logits must give the benchmark's IFCNet answers (same points, same graph, batch 8 not 1).
    ifc_label = ["mep", "beam", "mep", "mep", "door", "mep", "mep", "furniture", "lighting", "mep", "mep", "mep", "other", "railing",
                 "sanitary", "slab", "heater", "stair", "mep", "wall"]
    same_ifc = sum(ifc_label[int(i)] == ifcnet[k] for k, i in enumerate(E_logits.argmax(1)))
    results["checks"]["ifcnetReproduced"] = f"{same_ifc} of {len(order)}"
    rules_noname = np.array([by_id[r["id"]]["rulesNoName"]["label"] for r in order], dtype=object)
    rules_noname_by = [by_id[r["id"]]["rulesNoName"]["by"] for r in order]

    prim = subset == "primary"
    fold_of = folds(list(grp[prim]))
    results["taskA"] = {"folds": 5, "split": "leave buildings out (5 folds; train on the typed elements of the other buildings)",
                        "n": {"primary": int(prim.sum()), "hard": int((~prim).sum())}, "methods": {}}
    TA = results["taskA"]["methods"]

    def add(name, pred, note=""):
        log(name)
        TA[name] = {"primary": score(truth[prim], pred[prim], LABELS), "hard": score(truth[~prim], pred[~prim], LABELS), "note": note}

    add("rules (benchmark)", rules, "src/core/identify.js, from preds-rules.json")
    add("IFCNet DGCNN zero-shot (benchmark)", ifcnet, "pruned fp32 2,048 points, from the benchmark")
    add("rules, then IFCNet at 0.8 (benchmark)", hyb, "from the benchmark")
    add("rules without names", rules_noname, "identify() given PredefinedType, LongName and the box only")

    def cv(make, X, label):
        p, ph = cross_predict(make, X[prim], truth[prim], grp[prim], fold_of, X[~prim], grp[~prim])
        out = np.empty(len(truth), dtype=object)
        out[prim], out[~prim] = p, ph
        return out

    t0 = time.time()
    f_only = cv(hgb, XA, "features")
    results["taskA"]["secondsFeaturesModel"] = round(time.time() - t0, 1)
    add("features only (HGB)", f_only, "names withheld; 71 features")
    abl = {}
    for gname in ("box", "box+eigen", "box+surfaces", "box+segments,voxels"):
        abl[gname] = cv(hgb, XA[:, gidx(groups_def[gname])], gname)
        add(f"features only, {gname} (HGB)", abl[gname], "ablation")
    Xr = np.hstack([XA, onehot(rules, LABELS), onehot(rules_by, ["predefined", "name", "box", "none"])])
    stacked = cv(hgb, Xr, "features+rules")
    add("features + rules, stacked (HGB)", stacked, "rules' label and source as inputs")
    cascade = np.array([r if b in ("predefined", "name") else f for r, b, f in zip(rules, rules_by, f_only)], dtype=object)
    add("rules, then features (cascade)", cascade, "the rules answer when a PredefinedType or a name decides; the features model otherwise")
    Xrn = np.hstack([XA, onehot(rules_noname, LABELS), onehot(rules_noname_by, ["predefined", "name", "box", "none"])])
    f_rnn = cv(hgb, Xrn, "f+rnn")
    add("features + rules without names, stacked (HGB)", f_rnn, "names withheld everywhere")
    t0 = time.time()
    p_probe = cv(probe, E_hidden.astype(np.float64), "probe")
    results["taskA"]["secondsProbe"] = round(time.time() - t0, 1)
    add("DGCNN linear probe (256-d penultimate layer)", p_probe, "logistic regression on IFCNet's frozen embedding; names withheld")
    Xfe = np.hstack([XA, E_hidden, E_logits])
    f_emb = cv(hgb, Xfe, "f+emb")
    add("features + DGCNN embedding (HGB)", f_emb, "names withheld")
    Xfre = np.hstack([Xr, E_hidden, E_logits])
    add("features + rules + DGCNN embedding, stacked (HGB)", cv(hgb, Xfre, "all"), "")
    cascade_e = np.array([r if b in ("predefined", "name") else f for r, b, f in zip(rules, rules_by, f_emb)], dtype=object)
    add("rules, then features + embedding (cascade)", cascade_e, "")

    # Bootstrap intervals for the differences that carry the claims.
    results["taskA"]["bootstrap"] = {
        "stacked - rules (primary)": bootstrap_diff(truth[prim], rules[prim], stacked[prim], grp[prim], LABELS),
        "stacked - rules (hard)": bootstrap_diff(truth[~prim], rules[~prim], stacked[~prim], grp[~prim], LABELS),
        "cascade - rules (primary)": bootstrap_diff(truth[prim], rules[prim], cascade[prim], grp[prim], LABELS),
        "cascade - rules (hard)": bootstrap_diff(truth[~prim], rules[~prim], cascade[~prim], grp[~prim], LABELS),
        "features - box only (primary)": bootstrap_diff(truth[prim], abl["box"][prim], f_only[prim], grp[prim], LABELS),
        "features + rules without names - rules without names (primary)": bootstrap_diff(truth[prim], rules_noname[prim], f_rnn[prim], grp[prim], LABELS),
        "features - IFCNet zero-shot (primary)": bootstrap_diff(truth[prim], ifcnet[prim], f_only[prim], grp[prim], LABELS),
        "features - DGCNN probe (primary)": bootstrap_diff(truth[prim], p_probe[prim], f_only[prim], grp[prim], LABELS),
        "features+embedding - features (primary)": bootstrap_diff(truth[prim], f_only[prim], f_emb[prim], grp[prim], LABELS),
    }
    # Where the rules fall back to the box or find nothing: the elements the geometry is for.
    fb = np.array([b in ("box", "none") for b in rules_by])
    results["taskA"]["rulesFallback"] = {
        "primary": {"n": int((fb & prim).sum()), "rules": score(truth[fb & prim], rules[fb & prim], LABELS),
                    "features": score(truth[fb & prim], f_only[fb & prim], LABELS), "featuresEmbedding": score(truth[fb & prim], f_emb[fb & prim], LABELS)},
        "hard": {"n": int((fb & ~prim).sum()), "rules": score(truth[fb & ~prim], rules[fb & ~prim], LABELS),
                 "features": score(truth[fb & ~prim], f_only[fb & ~prim], LABELS), "featuresEmbedding": score(truth[fb & ~prim], f_emb[fb & ~prim], LABELS)},
    }

    # Open-data variant: train on the openly licensed models only; test on the buildings with no openly
    # licensed model at all (another export of an open building would leak into the test).
    open_buildings = {building(m) for m in models if is_open(m)}
    op = np.array([is_open(m) for m in models])
    closed = np.array([building(m) not in open_buildings for m in models])
    tr, te = prim & op, prim & closed
    log("task A open variant")
    if tr.any() and te.any():
        m = hgb(); m.fit(XA[tr], truth[tr]); p_open = m.predict(XA)
        results["taskA"]["openVariant"] = {
            "train": {"elements": int(tr.sum()), "buildings": len(set(grp[tr]))}, "test": {"elements": int(te.sum()), "buildings": len(set(grp[te]))},
            "featuresTrainedOnOpen": score(truth[te], p_open[te], LABELS),
            "featuresCrossValidatedSameElements": score(truth[te], f_only[te], LABELS),
            "rulesSameElements": score(truth[te], rules[te], LABELS),
            "hardSubsetNotOpen": score(truth[~prim & closed], p_open[~prim & closed], LABELS),
            "hardSubsetNotOpenRules": score(truth[~prim & closed], rules[~prim & closed], LABELS),
        }

    # ------------------------------------------------ task B: furniture sub-types
    FURN = {"IFCFURNITURE", "IFCFURNISHINGELEMENT", "IFCSYSTEMFURNITUREELEMENT"}
    APPL = {"IFCELECTRICAPPLIANCE", "IFCAUDIOVISUALAPPLIANCE", "IFCCOMMUNICATIONSAPPLIANCE"}
    GEN = {"IFCBUILDINGELEMENTPROXY", "IFCFLOWTERMINAL", "IFCDISTRIBUTIONELEMENT", "IFCDISTRIBUTIONFLOWELEMENT", "IFCBUILDINGELEMENT", "IFCELEMENT", "IFCPROXY"}
    pool = [r for r in recs if (r["cls"] in FURN or r["cls"] in APPL) or (r["cls"] in GEN and r["subtype"] and r["subtype"] not in ("lamp", "furniture-other"))]
    census = {
        "furnitureClassElements": sum(r["cls"] in FURN for r in recs),
        "applianceClassElements": sum(r["cls"] in APPL for r in recs),
        "furnitureClassWithSubtype": sum(r["cls"] in FURN and bool(r["subtype"]) for r in recs),
        "furnitureClassSubtypeByPredefined": sum(r["cls"] in FURN and r["subtypeBy"] == "predefined" for r in recs),
        "genericWithFurnitureSubtype": sum(r["cls"] in GEN and bool(r["subtype"]) and r["subtype"] not in ("lamp", "furniture-other") for r in recs),
    }
    # At most three instances of one family (same model, name stem and size to 5 cm).
    rng = np.random.default_rng(SEED)
    fam = defaultdict(list)
    for r in pool:
        if r["subtype"]:
            fam[family_key(r)].append(r)
    dedup = []
    for key in sorted(fam, key=str):
        items = fam[key]
        pick = rng.permutation(len(items))[:3]
        dedup += [items[i] for i in sorted(pick)]
    raw_counts = Counter(r["subtype"] for r in pool if r["subtype"])
    counts = Counter(r["subtype"] for r in dedup)
    model_counts = {s: len({building(r["model"]) for r in dedup if r["subtype"] == s}) for s in counts}
    fam_counts = Counter(r["subtype"] for r in {family_key(r): r for r in dedup}.values())
    MIN_N, MIN_B = 15, 3
    supported = sorted(s for s in counts if counts[s] >= MIN_N and model_counts[s] >= MIN_B)
    census.update({"rawBySubtype": dict(raw_counts), "dedupBySubtype": dict(counts), "familiesBySubtype": dict(fam_counts),
                   "buildingsBySubtype": model_counts, "rule": f"supported when at least {MIN_N} instances (at most 3 per family) from at least {MIN_B} buildings",
                   "supported": supported})
    B = [r for r in dedup if r["subtype"] in supported]
    yB = np.array([r["subtype"] for r in B], dtype=object)
    gB = np.array([building(r["model"]) for r in B])
    XB = F[[rec_row[r["id"]] for r in B]]
    EB_hidden = emb["hidden"][[emb_row[r["id"]] for r in B]].astype(np.float64)
    EB_logits = emb["logits"][[emb_row[r["id"]] for r in B]]
    foldB = folds(list(gB))
    results["taskB"] = {"census": census, "n": len(B), "buildings": len(set(gB)), "labels": supported,
                        "byPredefined": int(sum(r["subtypeBy"] == "predefined" for r in B)), "methods": {}}
    TB = results["taskB"]["methods"]

    def cvB(make, X):
        p, _ = cross_predict(make, X, yB, gB, foldB)
        return p

    maj = Counter(yB).most_common(1)[0][0]
    TB["majority class"] = score(yB, np.array([maj] * len(yB), dtype=object), supported)
    pB = {}
    for gname, cols in groups_def.items():
        pB[gname] = cvB(hgb, XB[:, gidx(cols)])
        TB[f"features {gname} (HGB)"] = score(yB, pB[gname], supported)
        log(f"task B {gname}")
    pB["probe"] = cvB(probe, EB_hidden)
    TB["DGCNN linear probe (256-d)"] = score(yB, pB["probe"], supported)
    pB["f+emb"] = cvB(hgb, np.hstack([XB, EB_hidden, EB_logits]))
    TB["features + DGCNN embedding (HGB)"] = score(yB, pB["f+emb"], supported)
    # IFCNet's own answer on these elements (it has one furniture class).
    ifc_cls = ["IfcAirTerminal", "IfcBeam", "IfcCableCarrierFitting", "IfcCableCarrierSegment", "IfcDoor", "IfcDuctFitting", "IfcDuctSegment",
               "IfcFurniture", "IfcLamp", "IfcOutlet", "IfcPipeFitting", "IfcPipeSegment", "IfcPlate", "IfcRailing", "IfcSanitaryTerminal",
               "IfcSlab", "IfcSpaceHeater", "IfcStair", "IfcValve", "IfcWall"]
    top = Counter(ifc_cls[i] for i in EB_logits.argmax(1))
    results["taskB"]["ifcnetTop1OnTheseElements"] = dict(top.most_common())
    results["taskB"]["bootstrap"] = {
        "all - box": bootstrap_diff(yB, pB["box"], pB["all"], gB, supported),
        "box+eigen - box": bootstrap_diff(yB, pB["box"], pB["box+eigen"], gB, supported),
        "all - probe": bootstrap_diff(yB, pB["probe"], pB["all"], gB, supported),
        "features+embedding - all": bootstrap_diff(yB, pB["all"], pB["f+emb"], gB, supported),
    }
    # Confusion of the full feature model.
    conf = {t: dict(Counter(p for tt, p in zip(yB, pB["all"]) if tt == t)) for t in supported}
    results["taskB"]["confusionAll"] = conf
    # Open-data variant for sub-types.
    log("task B open variant")
    opB = np.array([is_open(r["model"]) for r in B])
    clB = np.array([building(r["model"]) not in open_buildings for r in B])
    if opB.any() and clB.any():
        present_tr = set(yB[opB])
        m = hgb(); m.fit(XB[opB], yB[opB]); p = m.predict(XB[clB])
        results["taskB"]["openVariant"] = {"train": {"elements": int(opB.sum()), "buildings": len(set(gB[opB])), "bySubtype": dict(Counter(yB[opB]))},
                                           "test": {"elements": int(clB.sum()), "buildings": len(set(gB[clB])), "bySubtype": dict(Counter(yB[clB]))},
                                           "featuresTrainedOnOpen": score(yB[clB], p, supported),
                                           "featuresCrossValidatedSameElements": score(yB[clB], pB["all"][clB], supported),
                                           "labelsMissingFromOpenTraining": sorted(set(supported) - present_tr)}
    # Unknown furniture: furniture-class elements whose name and type say nothing.
    unknown = [r for r in recs if r["cls"] in FURN and not r["subtype"]]
    results["taskB"]["unknownFurnitureElements"] = len(unknown)
    if unknown:
        m = hgb(); m.fit(XB, yB)
        pu = m.predict(F[[rec_row[r["id"]] for r in unknown]])
        results["taskB"]["unknownFurniturePredicted"] = dict(Counter(pu))

    # Model size: the full feature model on all sub-type elements.
    import pickle
    m = hgb(); m.fit(XB, yB)
    results["taskB"]["modelPickleBytes"] = len(pickle.dumps(m))
    try:
        # skl2onnx 1.20 hands onnx a list mixing Python ints and numpy bools for one tree attribute,
        # which onnx 1.23 refuses; cast such lists to ints (the values are 0 and 1 either way).
        import onnx.helper as onnx_helper
        if not getattr(onnx_helper.make_attribute, "_bool_cast", False):
            _orig = onnx_helper.make_attribute

            def _make_attribute(key, value, *a, **k):
                if isinstance(value, (list, tuple, np.ndarray)) and any(isinstance(v, (bool, np.bool_)) for v in value):
                    value = [int(v) for v in value]
                return _orig(key, value, *a, **k)
            _make_attribute._bool_cast = True
            onnx_helper.make_attribute = _make_attribute
        from skl2onnx import to_onnx
        onx = to_onnx(m, XB[:1].astype(np.float32), options={id(m): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
        results["taskB"]["modelOnnxBytes"] = len(onx.SerializeToString())
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        po = sess.run(None, {sess.get_inputs()[0].name: XB.astype(np.float32)})[0]
        results["taskB"]["onnxAgreesWithSklearn"] = f"{int((po == m.predict(XB)).sum())} of {len(XB)}"
        one = XB[:1].astype(np.float32)
        name = sess.get_inputs()[0].name
        for _ in range(20):
            sess.run(None, {name: one})
        ts = []
        for i in range(200):
            t0 = time.perf_counter(); sess.run(None, {name: XB[i % len(XB):i % len(XB) + 1].astype(np.float32)}); ts.append(time.perf_counter() - t0)
        results["taskB"]["onnxMsPerElementNative"] = round(1000 * float(np.median(ts)), 3)
        mA = hgb(); mA.fit(XA[prim], truth[prim])
        onxA = to_onnx(mA, XA[:1].astype(np.float32), options={id(mA): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
        results["taskA"]["modelOnnxBytes"] = len(onxA.SerializeToString())
        results["taskA"]["modelPickleBytes"] = len(pickle.dumps(mA))
    except Exception as exc:  # skl2onnx not installed or the converter refused
        results["taskB"]["modelOnnx"] = f"not exported: {type(exc).__name__}: {exc}"[:200]

    timing_file = pilot / "train-timing.json"
    if timing_file.exists():
        results["dgcnnTrainingCost"] = json.loads(timing_file.read_text(encoding="utf8"))
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in results.items() if k in ("taskA", "taskB")}, indent=1)[:6000])
    print("wrote", out)


def recheck(args):
    a = json.loads(Path(args.pilot, "features.json").read_text(encoding="utf8"))
    b = json.loads(Path(args.check, "features.json").read_text(encoding="utf8"))
    ra, rb = a["records"], b["records"]
    same_ids = len(ra) == len(rb) and all(x["id"] == y["id"] for x, y in zip(ra, rb))
    fa, fb = np.array([x["f"] for x in ra]), np.array([y["f"] for y in rb])
    out = Path(args.out)
    results = json.loads(out.read_text(encoding="utf8"))
    results["reextraction"] = {
        "sameRecords": bool(same_ids), "maxFeatureDifference": float(np.abs(fa - fb).max()) if same_ids else None,
        "sameRuleAnswers": sum(x["rules"] == y["rules"] and x["rulesNoName"] == y["rulesNoName"] for x, y in zip(ra, rb)),
        "sameSubtypes": sum(x["subtype"] == y["subtype"] for x, y in zip(ra, rb)),
        "timing": b["timing"], "note": "second extraction, run while the pilot's model fitting used 15 threads",
    }
    # Which ignition scenarios the corpus's names reach at all (src/core/ignitionCatalog.js patterns, via identify.js).
    hits = defaultdict(lambda: {"elements": 0, "models": set()})
    for r in ra:
        if r.get("ignition"):
            hits[r["ignition"]]["elements"] += 1
            hits[r["ignition"]]["models"].add(r["model"])
    results["ignitionNameMatches"] = {
        "note": "elements of features.json (the benchmark's sample and every furniture element) whose Name or ObjectType matches a scenario",
        "byScenario": {k: {"elements": v["elements"], "models": len(v["models"])} for k, v in sorted(hits.items(), key=lambda kv: -kv[1]["elements"])},
    }
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print(results["reextraction"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "timetrain", "run", "recheck"):
        p = sub.add_parser(name)
        p.add_argument("--bench", default=str(HOME / "ifcnet-src" / "bench-data"))
        p.add_argument("--pilot", default=str(HOME / "ifcnet-src" / "pilot"))
        p.add_argument("--threads", type=int, default=8)
    sub.choices["embed"].add_argument("--onnx", default=str(HOME / "ifcnet-src" / "onnx" / "pruned-fp32-n2048-b8.onnx"))
    for name in ("run", "recheck"):
        sub.choices[name].add_argument("--out", default=str(REPO / "benchmark" / "results" / "2026-11-05-ifcnet-pilot.json"))
    sub.choices["recheck"].add_argument("--check", required=True)
    args = ap.parse_args()
    {"embed": embed, "timetrain": timetrain, "run": run, "recheck": recheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
