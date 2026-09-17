"""Fit linear and 2-layer MLP probes from feature sets to labels; write metrics.

    native/artifacts/rl/venv/bin/python native/rl/research/probes/probe_fit.py --features raw fused gru_state

Feature sets (see FEATURES): the raw compact observation (`raw`, and `raw_now` = the
undelayed encode, `raw_nopos` = raw with the absolute position obs[8:10] zeroed), the
encoder intermediates from acts-<tag>.npy (`flat`, `players`, `slots`, `entities`,
`slimes`, `pooled`, `fused`, `gru_state`, `gru_out`), the candidate families
(`rel` = features.relational_features, `rays16/32/32x3`) and `+`-joined unions.

Every label is fitted at once: the linear probe is one multi-output linear layer
(each output is an independent linear probe), the MLP probe is a grouped MLP with a
separate 256-unit hidden layer per label (einsum over the label axis), so labels
never share parameters. Binary labels use BCE and report AUC / accuracy / majority
accuracy; regression labels are standardised on the train split and report R^2.
Games are split 80/20 by the `split` metadata column (merge.py). Standardisation of
features uses train statistics only.
"""
import argparse
import json
import math
import time
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import layout  # noqa: E402
import schema  # noqa: E402
import features as F  # noqa: E402
from activations import SLICES  # noqa: E402
from generate import OUT  # noqa: E402

# label name -> (kind, expression). Expressions are evaluated on the label/meta table.
BINARY = ["hit_15", "hit_30", "hit_60", "hit_inflight_30", "hit_inflight_60", "fall_30", "fall_60", "squish_60",
          "any_death_30", "any_death_60", "my_shot_hits_60", "own_arrow_in_flight", "lands_dry", "terrain_within_dash",
          "grounded", "foe_grounded", "foe_dash_ready", "foe_bow_ready", "foe_platform_ready", "real_death_60", "win", "opp_alive"]
REGRESSION = {
    "drop": "clip(drop, 0, 64)", "edge_distance": "clip(edge_distance, -32, 32)", "terrain_min_ray": "clip(terrain_min_ray, 0, 64)",
    "support": "clip(support, -32, 32)",
    "idle_dx10": "idle_x10 - x", "idle_dy10": "idle_y10 - y", "idle_dx30": "idle_x30 - x", "idle_dy30": "idle_y30 - y",
    "idle_opp_dx10": "idle_opp_x10 - opp_x", "idle_opp_dy10": "idle_opp_y10 - opp_y",
    "idle_opp_dx30": "idle_opp_x30 - opp_x", "idle_opp_dy30": "idle_opp_y30 - opp_y",
    "real_dx12": "real_x12 - x", "real_dy12": "real_y12 - y",
    "real_opp_dx12": "real_opp_x12 - opp_x", "real_opp_dy12": "real_opp_y12 - opp_y",
    "real_opp_dx32": "real_opp_x32 - opp_x", "real_opp_dy32": "real_opp_y32 - opp_y",
    "vx": "vx", "vy": "vy", "foe_vx": "foe_vx", "foe_vy": "foe_vy",
    "foe_dash_cd": "foe_dash_cd", "foe_bow_cd": "foe_bow_cd", "foe_platform_cd": "foe_platform_cd",
    "return_disc": "return_disc", "outcome_sign": "2 * win - 1", "log_ticks_to_end": "log1p(ticks_to_end)",
    "abs_x": "x", "abs_y": "y",
}
MULTICLASS = {"map": ("map", 5)}  # one-vs-rest binary heads, reported as argmax accuracy
# Privileged inputs for the critic-ceiling probe (`priv` feature part): idle-fork futures, live geometry, opponent state.
PRIV_FEATURES = ["hit_15", "hit_30", "hit_60", "fall_30", "fall_60", "squish_60", "any_death_60", "my_shot_hits_60", "own_arrow_in_flight",
                 "idle_death_tick / 60", "(idle_x30 - x) / 32", "(idle_y30 - y) / 32", "(idle_opp_x30 - opp_x) / 32", "(idle_opp_y30 - opp_y) / 32",
                 "grounded", "clip(drop, 0, 64) / 64", "clip(edge_distance, -32, 32) / 32", "lands_dry", "terrain_within_dash",
                 "vx / 64", "vy / 64", "foe_vx / 64", "foe_vy / 64", "foe_grounded", "foe_dash_ready", "foe_bow_ready", "foe_platform_ready",
                 "foe_dash_cd", "foe_bow_cd", "foe_platform_cd", "x / 64", "y / 64", "opp_x / 64", "opp_y / 64", "opp_alive", "tick / 1800"]
POS_COLS = (schema.COMPACT_OFFSETS["prefix"] + 8, schema.COMPACT_OFFSETS["prefix"] + 9)


class Data:
    def __init__(self, dataset, acts_tag="024", maps=None, controllers=None):
        self.dataset = Path(dataset)
        self.index = json.loads((self.dataset / "index.json").read_text())
        self.obs = np.load(self.dataset / "obs.npy", mmap_mode="r")
        now = self.dataset / "obs_now.npy"  # deleted once the undelayed probes were done (disk budget); regenerate with generate.py + merge.py
        self.obs_now = np.load(now, mmap_mode="r") if now.exists() else None
        self.meta = np.load(self.dataset / "meta.npy", mmap_mode="r")
        self.labels = np.load(self.dataset / "labels.npy", mmap_mode="r")
        self.rays = {k: np.load(self.dataset / f"{k}.npy", mmap_mode="r") for k in schema.RAY_SETS if (self.dataset / f"{k}.npy").exists()}
        acts_path = self.dataset / f"acts-{acts_tag}.npy"
        self.acts = np.load(acts_path, mmap_mode="r") if acts_path.exists() else None
        self.meta_cols = {k: i for i, k in enumerate(schema.META)}
        self.label_cols = {k: i for i, k in enumerate(schema.LABELS)}
        keep = np.ones(self.index["rows"], dtype=bool)
        if maps is not None:
            keep &= np.isin(self.meta[:, self.meta_cols["map"]], maps)
        if controllers is not None:
            keep &= np.isin(self.meta[:, self.meta_cols["controller"]], controllers)
        split = self.meta[:, self.meta_cols["split"]]
        self.train = np.flatnonzero(keep & (split == 0))
        self.test = np.flatnonzero(keep & (split == 1))

    def table(self, rows):
        env = {k: self.labels[rows, i].astype(np.float64) for k, i in self.label_cols.items()}
        env.update({k: self.meta[rows, i].astype(np.float64) for k, i in self.meta_cols.items() if k not in env})
        env.update(clip=np.clip, log1p=np.log1p, abs=np.abs)
        return env

    def targets(self, rows, names):
        """(n, L) float32 with NaN for missing; list of kinds."""
        env = self.table(rows)
        cols, kinds = [], []
        for name in names:
            if name in BINARY:
                cols.append(env[name]); kinds.append("binary")
            elif name in REGRESSION:
                cols.append(eval(REGRESSION[name], {}, env)); kinds.append("regression")
            elif name in MULTICLASS:
                col, k = MULTICLASS[name]
                for c in range(k):
                    cols.append(np.where(np.isnan(env[col]), np.nan, (env[col] == c).astype(np.float64))); kinds.append("binary")
            else:
                raise KeyError(name)
        return np.stack(cols, 1).astype(np.float32), kinds

    def features(self, name, rows):
        parts = []
        for part in name.split("+"):
            if part == "raw":
                parts.append(self.obs[rows].astype(np.float32))
            elif part == "raw_now":
                parts.append(self.obs_now[rows].astype(np.float32))
            elif part == "raw_nopos":
                x = self.obs[rows].astype(np.float32); x[:, list(POS_COLS)] = 0; parts.append(x)
            elif part == "pos":
                parts.append(self.obs[rows][:, list(POS_COLS)].astype(np.float32))
            elif part == "opp_id":  # one-hot of the opponent controller (never observable; the outcome-variance decomposition)
                ids = self.meta[rows, self.meta_cols["opponent_controller"]]
                parts.append((ids[:, None] == np.arange(16)[None]).astype(np.float32))
            elif part == "priv":  # the privileged labels themselves as inputs (the critic-ceiling experiment)
                env = self.table(rows)
                cols = [np.clip(eval(e, {}, env), -4, 4) for e in PRIV_FEATURES]
                parts.append(np.nan_to_num(np.stack(cols, 1), nan=0.0).astype(np.float32))
            elif part == "rel":
                parts.append(F.relational_features(self.obs[rows]))
            elif part in self.rays:
                parts.append(self.rays[part][rows].astype(np.float32))
            elif part == "pooled":
                parts.append(self.acts[rows, 128:384].astype(np.float32))
            elif part in SLICES:
                parts.append(self.acts[rows, SLICES[part]].astype(np.float32))
            else:
                raise KeyError(part)
        return np.concatenate(parts, 1) if len(parts) > 1 else parts[0]


class GroupedMLP(torch.nn.Module):
    """L independent 2-layer MLPs (d -> hidden -> 1) evaluated in one einsum."""

    def __init__(self, d, L, hidden=256):
        super().__init__()
        bound = 1 / math.sqrt(d)
        self.w1 = torch.nn.Parameter(torch.empty(L, d, hidden).uniform_(-bound, bound))
        self.b1 = torch.nn.Parameter(torch.zeros(L, hidden))
        self.w2 = torch.nn.Parameter(torch.empty(L, hidden).uniform_(-1 / math.sqrt(hidden), 1 / math.sqrt(hidden)))
        self.b2 = torch.nn.Parameter(torch.zeros(L))

    def forward(self, x):
        L, d, hidden = self.w1.shape
        # one matmul against the (d, L*hidden) weight: no per-label copy of x
        h = torch.relu((x @ self.w1.permute(1, 0, 2).reshape(d, L * hidden)).view(-1, L, hidden) + self.b1)
        return (h * self.w2).sum(-1) + self.b2


def auc(score, y):
    """Mann-Whitney AUC."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(pred, y, kind):
    ok = ~np.isnan(y)
    pred, y = pred[ok].astype(np.float64), y[ok].astype(np.float64)
    if len(y) == 0:
        return dict(n=0)
    if kind == "binary":
        p = 1 / (1 + np.exp(-pred))
        rate = y.mean()
        out = dict(n=int(len(y)), positive_rate=float(rate), auc=auc(pred, y), accuracy=float(((p > 0.5) == (y > 0.5)).mean()),
                   majority=float(max(rate, 1 - rate)),
                   log_loss=float(-np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))),
                   baseline_log_loss=float(-(rate * np.log(rate + 1e-7) + (1 - rate) * np.log(1 - rate + 1e-7))))
        # average precision
        order = np.argsort(-pred); ys = y[order]
        tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
        out["average_precision"] = float((prec * ys).sum() / max(ys.sum(), 1))
        return out
    var = y.var()
    return dict(n=int(len(y)), r2=float(1 - np.mean((pred - y) ** 2) / var) if var > 0 else float("nan"),
                rmse=float(np.sqrt(np.mean((pred - y) ** 2))), std=float(np.sqrt(var)))


def fit(data, feature_name, label_names, device, epochs=6, batch=4096, chunk=32768, hidden=256, seed=0, max_train=0, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train, test = data.train, data.test
    if max_train and len(train) > max_train:
        train = np.sort(rng.choice(train, max_train, replace=False))
    # standardisation + target statistics on a train subsample
    sample = np.sort(rng.choice(train, min(len(train), 100_000), replace=False))
    xs = data.features(feature_name, sample)
    mean = xs.mean(0); std = xs.std(0); std[std < 1e-3] = 1.0
    ys, kinds = data.targets(sample, label_names)
    ymean = np.nanmean(ys, 0); ystd = np.nanstd(ys, 0); ystd[ystd < 1e-6] = 1.0
    is_reg = np.array([k == "regression" for k in kinds])
    ymean[~is_reg] = 0; ystd[~is_reg] = 1
    d, L = xs.shape[1], ys.shape[1]
    t = lambda a: torch.as_tensor(a, device=device)
    mean_t, std_t, ymean_t, ystd_t, is_reg_t = t(mean), t(std), t(ymean), t(ystd), t(is_reg)
    linear = torch.nn.Linear(d, L).to(device)
    mlp = GroupedMLP(d, L, hidden).to(device)
    steps = epochs * math.ceil(len(train) / batch)
    opts = [torch.optim.Adam(linear.parameters(), lr=3e-3, weight_decay=1e-6), torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-6)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, steps) for o in opts]

    def loss_fn(pred, y):
        ok = ~torch.isnan(y)
        y0 = torch.where(ok, y, torch.zeros_like(y))
        per = torch.where(is_reg_t[None], (pred - y0) ** 2, torch.nn.functional.binary_cross_entropy_with_logits(pred, y0, reduction="none"))
        per = torch.where(ok, per, torch.zeros_like(per))
        return (per.sum(0) / ok.sum(0).clamp(min=1)).sum()

    started = time.time()
    chunks = [train[i:i + chunk] for i in range(0, len(train), chunk)]
    for epoch in range(epochs):
        order = rng.permutation(len(chunks))
        total = 0.0; count = 0
        for ci in order:
            rows = chunks[ci]
            x = (t(data.features(feature_name, rows)) - mean_t) / std_t
            y = (t(data.targets(rows, label_names)[0]) - ymean_t) / ystd_t
            perm = torch.randperm(len(rows), device=device)
            for b in range(0, len(rows), batch):
                idx = perm[b:b + batch]
                xb, yb = x[idx], y[idx]
                for model, opt, sched in zip((linear, mlp), opts, scheds):
                    loss = loss_fn(model(xb), yb)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                    total += float(loss.detach()); count += 1
        log(f"  {feature_name} epoch {epoch + 1}/{epochs} loss {total / max(count, 1):.4f} ({time.time() - started:.0f}s)")
    # evaluation on the test games
    preds = {"linear": [], "mlp": []}; ys_all = []
    with torch.no_grad():
        for b in range(0, len(test), chunk):
            rows = test[b:b + chunk]
            x = (t(data.features(feature_name, rows)) - mean_t) / std_t
            ys_all.append(data.targets(rows, label_names)[0])
            preds["linear"].append((linear(x) * ystd_t + ymean_t).cpu().numpy())
            preds["mlp"].append(np.concatenate([(mlp(x[b:b + batch]) * ystd_t + ymean_t).cpu().numpy() for b in range(0, len(x), batch)]))
    y = np.concatenate(ys_all)
    result = dict(features=feature_name, dim=int(d), train_rows=int(len(train)), test_rows=int(len(test)), seconds=time.time() - started, labels={})
    col = 0
    for name in label_names:
        if name in MULTICLASS:
            k = MULTICLASS[name][1]
            for model in preds:
                p = np.concatenate(preds[model])[:, col:col + k]
                yy = y[:, col:col + k]; ok = ~np.isnan(yy[:, 0])
                truth = yy[ok].argmax(1); guess = p[ok].argmax(1)
                counts = np.bincount(truth, minlength=k)
                result["labels"].setdefault(name, {})[model] = dict(n=int(ok.sum()), accuracy=float((truth == guess).mean()),
                                                                   majority=float(counts.max() / max(ok.sum(), 1)), kind="multiclass")
            col += k
        else:
            for model in preds:
                p = np.concatenate(preds[model])[:, col]
                m = metrics(p, y[:, col], kinds[col]); m["kind"] = kinds[col]
                result["labels"].setdefault(name, {})[model] = m
            col += 1
    return result


FEATURES = ["raw", "raw_now", "raw_nopos", "flat", "players", "slots", "entities", "slimes", "pooled", "fused", "gru_state", "gru_out",
            "rel", "raw+rel", "rays16", "rays32", "rays32x3", "raw+rays16", "raw+rays32", "raw+rays32x3", "raw+rel+rays32",
            "raw_nopos+rays32", "raw_nopos+rel", "pos"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--features", nargs="+", default=FEATURES)
    parser.add_argument("--labels", nargs="+", default=BINARY + list(REGRESSION) + list(MULTICLASS))
    parser.add_argument("--acts-tag", default="024")
    parser.add_argument("--maps", type=int, nargs="*", default=None)
    parser.add_argument("--controllers", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None, help="results json (default <dataset>/results/probes-<acts-tag>.json)")
    parser.add_argument("--gpu-fraction", type=float, default=0.12, help="torch per-process GPU memory cap (fraction of the device)")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--skip-done", action="store_true", help="skip feature sets already in the results json")
    args = parser.parse_args()
    if args.device.startswith("cuda"):  # the machine is shared with a live training job: hard cap at ~1.5 GB
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    data = Data(args.dataset, args.acts_tag, args.maps, args.controllers)
    out = args.out or args.dataset / "results" / f"probes-{args.acts_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    print(f"train rows {len(data.train)} test rows {len(data.test)} device {args.device}")
    for feature_name in args.features:
        if any(p in SLICES or p == "pooled" for p in feature_name.split("+")) and data.acts is None:
            print(f"skip {feature_name}: no acts-{args.acts_tag}.npy"); continue
        if args.skip_done and feature_name in results:
            continue
        results[feature_name] = fit(data, feature_name, args.labels, args.device, epochs=args.epochs, max_train=args.max_train,
                                    batch=args.batch, chunk=args.chunk)
        out.write_text(json.dumps(results, indent=1))
        lab = results[feature_name]["labels"]
        show = [f"{k}: lin {v['linear'].get('auc', v['linear'].get('r2', v['linear'].get('accuracy'))):.3f} mlp {v['mlp'].get('auc', v['mlp'].get('r2', v['mlp'].get('accuracy'))):.3f}"
                for k, v in lab.items() if k in ("hit_inflight_30", "fall_30", "drop", "return_disc", "map")]
        print(f"{feature_name} [{results[feature_name]['dim']}] {results[feature_name]['seconds']:.0f}s | " + " | ".join(show))


if __name__ == "__main__":
    main()
