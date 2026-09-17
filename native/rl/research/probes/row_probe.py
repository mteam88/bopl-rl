"""Per-entity-row probe: can one row, on its own, say "this arrow hits me"?

The set encoder applies one bias-free linear map to each entity row and then a
max over rows, so whatever the encoder knows about a specific arrow has to be a
linear function of that arrow's 34 features. This probe fits linear and MLP
classifiers on hostile projectile rows (features = the row, optionally + the
relational features of features.py and the observer's velocity) to the label
"this row is the arrow that kills me in the idle fork within 60 ticks"
(meta.killer_row). Train/test split by game.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import schema  # noqa: E402
import features as F  # noqa: E402
from probe_fit import metrics  # noqa: E402
from generate import OUT  # noqa: E402


def collect(obs, meta, rows, chunk=65536):
    mc = {k: i for i, k in enumerate(schema.META)}
    raw, rel, sv, y, states = [], [], [], [], []
    for b in range(0, len(rows), chunk):
        r = rows[b:b + chunk]
        compact = obs[r].astype(np.float32)
        hostile = F.hostile_projectile_rows(compact)  # (n, 32)
        n_idx, row_idx = np.nonzero(hostile)
        if len(n_idx) == 0:
            continue
        ent = schema.block(compact, "entities")
        relf = F.entity_relational(compact).reshape(len(compact), schema.ENTITY_CAP, F.REL_PER_ROW)
        selfv = F.self_velocity(compact)
        killer = meta[r, mc["killer_row"]]
        raw.append(ent[n_idx, row_idx]); rel.append(relf[n_idx, row_idx]); sv.append(selfv[n_idx])
        y.append((killer[n_idx] == row_idx).astype(np.float32)); states.append(r[n_idx])
    return np.concatenate(raw), np.concatenate(rel), np.concatenate(sv), np.concatenate(y), np.concatenate(states)


def fit(x_train, y_train, x_test, y_test, hidden=256, epochs=8, seed=0):
    torch.manual_seed(seed)
    mean, std = x_train.mean(0), x_train.std(0); std[std < 1e-3] = 1
    xt = torch.as_tensor((x_train - mean) / std); yt = torch.as_tensor(y_train)
    xe = torch.as_tensor((x_test - mean) / std)
    out = {}
    for name, model in (("linear", torch.nn.Linear(xt.shape[1], 1)),
                        ("mlp", torch.nn.Sequential(torch.nn.Linear(xt.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)))):
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        steps = epochs * (len(xt) // 4096 + 1); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(epochs):
            perm = torch.randperm(len(xt))
            for b in range(0, len(xt), 4096):
                idx = perm[b:b + 4096]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt[idx]).squeeze(-1), yt[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            pred = model(xe).squeeze(-1).numpy()
        out[name] = metrics(pred, y_test, "binary")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--max-states", type=int, default=600_000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    obs = np.load(args.dataset / "obs.npy", mmap_mode="r"); meta = np.load(args.dataset / "meta.npy", mmap_mode="r")
    mc = {k: i for i, k in enumerate(schema.META)}
    split = meta[:, mc["split"]]
    rng = np.random.default_rng(0)
    train_rows = np.sort(rng.choice(np.flatnonzero(split == 0), args.max_states, replace=False))
    test_rows = np.flatnonzero(split == 1)
    tr = collect(obs, meta, train_rows); te = collect(obs, meta, test_rows)
    result = dict(train_rows=int(len(tr[3])), test_rows=int(len(te[3])), positive_rate_test=float(te[3].mean()), sets={})
    variants = {
        "row": (tr[0], te[0]),
        "row+selfvel": (np.concatenate([tr[0], tr[2]], 1), np.concatenate([te[0], te[2]], 1)),
        "row+rel": (np.concatenate([tr[0], tr[1]], 1), np.concatenate([te[0], te[1]], 1)),
        "rel": (tr[1], te[1]),
        "row_pos_vel_only": (tr[0][:, 9:13], te[0][:, 9:13]),
        "rel_closing_miss_only": (tr[1][:, [0, 6, 8]], te[1][:, [0, 6, 8]]),
    }
    for name, (a, b) in variants.items():
        result["sets"][name] = fit(a, tr[3], b, te[3])
        print(name, {k: round(v["auc"], 3) for k, v in result["sets"][name].items()},
              {k: round(v["average_precision"], 3) for k, v in result["sets"][name].items()})
    (args.dataset / "results").mkdir(exist_ok=True)
    (args.dataset / "results" / "row-probe.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
