"""The checkpoint's own value head against the realised discounted return.

For rows decided by a checkpoint controller, value_online is the value the
policy computed at that step (delayed observation, its recurrent state along the
real rollout) and return_disc is sum_k gamma^k r_{t+k} along the same rollout
(gamma = schema.GAMMA, the training gamma). Reports explained variance,
correlation, calibration by value decile and by time-to-end, and a
calibration/scatter PNG.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import schema  # noqa: E402
from generate import OUT, CONTROLLERS  # noqa: E402


GAE_LAMBDA = 0.9  # train config of 024 (checkpoint sidecar: gamma 0.995, gae_lambda 0.9, horizon 128)


def explained_variance(v, g):
    return float(1 - np.var(g - v) / np.var(g))


def training_targets(meta, rewards, values, sel):
    """Per (game, player) sequence of the selected rows: the n-step bootstrapped returns
    (n = 128, 512 decisions, bootstrapped with the policy's own value at the cut, no
    bootstrap past the episode end) and the full-episode TD(lambda) return that PPO's
    GAE regresses the value head to."""
    mc = {k: i for i, k in enumerate(schema.META)}
    rows = np.flatnonzero(sel)
    keys = meta[rows, mc["game"]].astype(np.int64) * 2 + meta[rows, mc["player"]]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(rows)]])
    out = {"boot_128": np.empty(len(rows)), "boot_512": np.empty(len(rows)), "td_lambda": np.empty(len(rows))}
    g = schema.GAMMA
    for s, e in zip(starts, ends):
        r = rewards[s:e]; v = values[s:e]; T = e - s
        full = np.zeros(T + 1)  # full[t] = sum_k gamma^k r[t+k]
        for t in range(T - 1, -1, -1):
            full[t] = r[t] + g * full[t + 1]
        for n in (128, 512):
            acc = full[:T].copy()
            if T > n:  # G_n[t] = sum_{k<n} gamma^k r[t+k] + gamma^n v[t+n] for t+n < T
                t = np.arange(T - n)
                acc[t] = full[t] - g ** n * full[t + n] + g ** n * v[t + n]
            out[f"boot_{n}"][s:e] = acc
        lam = np.zeros(T)
        for t in range(T - 1, -1, -1):
            lam[t] = r[t] + (g * ((1 - GAE_LAMBDA) * v[t + 1] + GAE_LAMBDA * lam[t + 1]) if t + 1 < T else 0.0)
        out["td_lambda"][s:e] = lam
    return out


def analyse(labels, meta, controller, name, out_dir):
    lc = {k: i for i, k in enumerate(schema.LABELS)}; mc = {k: i for i, k in enumerate(schema.META)}
    sel = (meta[:, mc["controller"]] == controller)
    v = labels[sel, lc["value_online"]].astype(np.float64); g = labels[sel, lc["return_disc"]].astype(np.float64)
    tte = labels[sel, lc["ticks_to_end"]].astype(np.float64); timeout = labels[sel, lc["timeout"]]
    r = labels[sel, lc["reward"]].astype(np.float64)
    win, loss = labels[sel, lc["win"]], labels[sel, lc["loss"]]
    targets = training_targets(meta, r, v, sel)
    ok = ~np.isnan(v) & ~np.isnan(g)
    v, g, tte, timeout = v[ok], g[ok], tte[ok], timeout[ok]
    outcome = np.where(win[ok] > 0, 1.0, -1.0)  # loss and timeout both pay -1
    result = dict(controller=name, rows=int(ok.sum()), explained_variance=explained_variance(v, g),
                  correlation=float(np.corrcoef(v, g)[0, 1]), value_mean=float(v.mean()), return_mean=float(g.mean()),
                  value_std=float(v.std()), return_std=float(g.std()), mse=float(np.mean((v - g) ** 2)),
                  timeout_fraction=float(np.nanmean(timeout)), gamma=schema.GAMMA, gae_lambda=GAE_LAMBDA)
    result["ev_by_target"] = dict(realised_return=result["explained_variance"],
                                  **{k: explained_variance(v, t[ok]) for k, t in targets.items()},
                                  outcome_sign=explained_variance(v, outcome))
    result["corr_by_target"] = dict(realised_return=result["correlation"], **{k: float(np.corrcoef(v, t[ok])[0, 1]) for k, t in targets.items()},
                                    outcome_sign=float(np.corrcoef(v, outcome)[0, 1]))
    result["target_std"] = dict(realised_return=float(g.std()), **{k: float(t[ok].std()) for k, t in targets.items()}, outcome_sign=float(outcome.std()))
    # the trivially self-consistent check: the value one decision later, discounted, plus the reward
    v_next = np.concatenate([v[1:], [0.0]]); same_seq = np.concatenate([np.diff(meta[sel][ok][:, mc["game"]] * 2 + meta[sel][ok][:, mc["player"]]) == 0, [False]])
    td0 = r[ok] + schema.GAMMA * np.where(same_seq, v_next, 0.0)
    result["ev_by_target"]["td0"] = explained_variance(v, td0)
    result["corr_by_target"]["td0"] = float(np.corrcoef(v, td0)[0, 1]); result["target_std"]["td0"] = float(td0.std())
    # calibration by value decile
    edges = np.quantile(v, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 9)
    result["calibration"] = [dict(value=float(v[bins == b].mean()), realised=float(g[bins == b].mean()), n=int((bins == b).sum()))
                             for b in range(10)]
    # by time to end (near the end the return is essentially the outcome)
    horizons = [(0, 60), (60, 120), (120, 240), (240, 480), (480, 960), (960, 1e9)]
    result["by_ticks_to_end"] = []
    for lo, hi in horizons:
        m = (tte >= lo) & (tte < hi)
        if m.sum() > 100:
            result["by_ticks_to_end"].append(dict(lo=lo, hi=hi if hi < 1e9 else None, n=int(m.sum()), ev=explained_variance(v[m], g[m]),
                                                  corr=float(np.corrcoef(v[m], g[m])[0, 1]), return_std=float(g[m].std())))
    # a sign test: does the critic know who is winning at all?
    dec = g > 0
    result["sign_accuracy"] = float(((v > 0) == dec).mean()); result["sign_majority"] = float(max(dec.mean(), 1 - dec.mean()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        idx = np.random.default_rng(0).choice(len(v), min(len(v), 20000), replace=False)
        ax[0].scatter(v[idx], g[idx], s=2, alpha=0.2); ax[0].plot([-1, 1], [-1, 1], "k--", lw=1)
        ax[0].set_xlabel("value head"); ax[0].set_ylabel("realised discounted return"); ax[0].set_title(f"{name}: EV={result['explained_variance']:.3f}")
        cal = result["calibration"]
        ax[1].plot([c["value"] for c in cal], [c["realised"] for c in cal], "o-"); ax[1].plot([-1, 1], [-1, 1], "k--", lw=1)
        ax[1].set_xlabel("mean value (decile)"); ax[1].set_ylabel("mean realised return"); ax[1].set_title("calibration")
        fig.tight_layout(); fig.savefig(out_dir / f"critic-{name}.png", dpi=110); plt.close(fig)
    except ImportError:
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    args = parser.parse_args()
    labels = np.load(args.dataset / "labels.npy", mmap_mode="r"); meta = np.load(args.dataset / "meta.npy", mmap_mode="r")
    out_dir = args.dataset / "results"; out_dir.mkdir(exist_ok=True)
    results = []
    for i, c in enumerate(CONTROLLERS):
        if c["kind"] == "checkpoint" and c["name"].startswith("024"):
            results.append(analyse(labels, meta, i, c["name"], out_dir))
            print(json.dumps({k: v for k, v in results[-1].items() if not isinstance(v, list)}))
    ceiling = out_dir / "critic-ceiling.json"  # probe_fit.py --controllers 1 --labels return_disc outcome_sign win ... --out critic-ceiling.json
    if ceiling.exists():
        table = {}
        for feats, block in json.loads(ceiling.read_text()).items():
            table[feats] = {lab: {model: {k: v for k, v in m.items() if k in ("r2", "auc", "accuracy", "majority", "n")}
                                  for model, m in per.items()} for lab, per in block["labels"].items()}
        results.append(dict(controller="ceiling-probes (024-final rows, test games)", note=(
            "MLP/linear probes fitted to the same targets from the raw delayed observation, the privileged labels (priv), the "
            "opponent controller one-hot (opp_id, never observable) and the GRU state; r2 of return_disc / outcome_sign is the EV"),
            probes=table))
    (out_dir / "critic.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
