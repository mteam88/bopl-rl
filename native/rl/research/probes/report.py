"""Markdown tables and PNG plots from the probe results (results/*.json) for PROBES.md.

    native/artifacts/rl/venv/bin/python native/rl/research/probes/report.py > /tmp/tables.md
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
from generate import OUT  # noqa: E402

KEY_BINARY = ["hit_15", "hit_30", "hit_60", "hit_inflight_30", "hit_inflight_60", "my_shot_hits_60", "fall_30", "fall_60", "squish_60",
              "any_death_60", "lands_dry", "terrain_within_dash", "grounded", "foe_dash_ready", "foe_bow_ready", "foe_platform_ready",
              "real_death_60", "win", "own_arrow_in_flight", "opp_alive"]
KEY_REGRESSION = ["drop", "edge_distance", "support", "terrain_min_ray", "vx", "vy", "foe_vx", "foe_vy", "foe_dash_cd", "foe_bow_cd",
                  "foe_platform_cd", "idle_dx10", "idle_dy10", "idle_dx30", "idle_dy30", "idle_opp_dx10", "idle_opp_dy10", "idle_opp_dx30",
                  "idle_opp_dy30", "real_dx12", "real_dy12", "real_opp_dx12", "real_opp_dy12", "real_opp_dx32", "real_opp_dy32",
                  "return_disc", "outcome_sign", "log_ticks_to_end", "abs_x", "abs_y"]


def score(m):
    if m.get("kind") == "binary":
        return m.get("auc", float("nan"))
    if m.get("kind") == "multiclass":
        return m.get("accuracy", float("nan"))
    return m.get("r2", float("nan"))


def table(results, features, labels, model, title, extra=None):
    feats = [f for f in features if f in results]
    lines = [f"**{title}** ({model} probe; binary = test AUC, regression = test R^2, map = accuracy)", "",
             "| label | " + " | ".join(feats) + (" | " + extra[0] if extra else "") + " |", "|---" * (len(feats) + 1 + (1 if extra else 0)) + "|"]
    for lab in labels:
        if not any(lab in results[f]["labels"] for f in feats):
            continue
        row = [lab]
        for f in feats:
            m = results[f]["labels"].get(lab, {}).get(model)
            row.append(f"{score(m):.3f}" if m and not np.isnan(score(m)) else "")
        if extra:
            row.append(extra[1](lab))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


RARE = ["hit_15", "hit_30", "hit_60", "hit_inflight_30", "hit_inflight_60", "my_shot_hits_60", "fall_30", "fall_60", "squish_60", "any_death_60"]


def ap_table(results, features, model, title):
    feats = [f for f in features if f in results]
    lines = [f"**{title}** ({model} probe; test average precision; random = the positive rate)", "",
             "| label | positive rate | " + " | ".join(feats) + " |", "|---" * (len(feats) + 2) + "|"]
    for lab in RARE:
        m0 = results[feats[0]]["labels"].get(lab, {}).get(model, {})
        row = [lab, f"{m0.get('positive_rate', float('nan')):.4f}"]
        for f in feats:
            m = results[f]["labels"].get(lab, {}).get(model, {})
            row.append(f"{m['average_precision']:.3f}" if "average_precision" in m else "")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def base_rates(results, feature, labels):
    def f(lab):
        m = results[feature]["labels"].get(lab, {}).get("linear", {})
        if m.get("kind") == "binary":
            return f"{m.get('positive_rate', float('nan')):.3f} / n={m.get('n', 0)}"
        if m.get("kind") == "multiclass":
            return f"maj {m.get('majority', float('nan')):.3f}"
        return f"std {m.get('std', float('nan')):.2f} / n={m.get('n', 0)}"
    return ("positive rate / n", f)


def heatmap(results, features, labels, model, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = [f for f in features if f in results]
    labs = [l for l in labels if any(l in results[f]["labels"] for f in feats)]
    grid = np.full((len(labs), len(feats)), np.nan)
    for i, lab in enumerate(labs):
        for j, f in enumerate(feats):
            m = results[f]["labels"].get(lab, {}).get(model)
            if m:
                grid[i, j] = score(m)
    fig, ax = plt.subplots(figsize=(1.1 * len(feats) + 3, 0.32 * len(labs) + 1.5))
    im = ax.imshow(grid, vmin=0.5 if all(results[feats[0]]["labels"][l][model].get("kind") == "binary" for l in labs) else 0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=45, ha="right"); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    for i in range(len(labs)):
        for j in range(len(feats)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if grid[i, j] < 0.8 else "k")
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def perturb_plot(res_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(res_dir.glob("perturb-*.json"))
    if not files:
        return ""
    fig, axes = plt.subplots(1, len(files), figsize=(5.5 * len(files), 4), squeeze=False)
    lines = ["| input shifted | maps | " + " | ".join(f"KL @{d:+.1f}" for d in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)) + " | dV @+-0.1 | dV @+-0.3 |", "|---" * 10 + "|"]
    for ax, f in zip(axes[0], files):
        r = json.loads(f.read_text())
        tag = f.stem.replace("perturb-024-final-", "")
        for name, per in r["inputs"].items():
            ds = sorted(float(d) for d in per)
            ax.plot(ds, [per[str(d)]["kl_mean"] for d in ds], "o-", label=name)
            dv1 = np.mean([per["0.1"]["dvalue_mean"], per["-0.1"]["dvalue_mean"]]); dv3 = np.mean([per["0.3"]["dvalue_mean"], per["-0.3"]["dvalue_mean"]])
            lines.append(f"| {name} | {tag} | " + " | ".join(f"{per[str(d)]['kl_mean']:.4f}" for d in ds) + f" | {dv1:.4f} | {dv3:.4f} |")
        ax.set_yscale("log"); ax.set_xlabel("shift of the input (obs units; 0.1 = 6.4 world units for positions)"); ax.set_ylabel("mean KL(base || shifted), nats")
        ax.set_title(f"024-final, {tag}: {r['sequences']} sequences, {r['steps']} steps"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(res_dir / "perturb.png", dpi=120); plt.close(fig)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--results", default="probes-024.json")
    args = parser.parse_args()
    res_dir = args.dataset / "results"
    results = json.loads((res_dir / args.results).read_text())
    encoder = ["raw", "raw_now", "flat", "players", "slots", "entities", "slimes", "pooled", "fused", "gru_state", "gru_out"]
    families = ["raw", "rel", "raw+rel", "rays16", "rays32", "rays32x3", "raw+rays16", "raw+rays32", "raw+rays32x3", "raw+rel+rays32"]
    position = ["raw", "raw_nopos", "pos", "raw_nopos+rays32", "raw_nopos+rel", "fused", "gru_state"]
    out = []
    for model in ("linear", "mlp"):
        out.append(ap_table(results, encoder, model, "Input vs encoder: rare labels, average precision"))
        out.append(ap_table(results, families + ["raw_nopos+rays32"], model, "Feature families: rare labels, average precision"))
        out.append(table(results, encoder, KEY_BINARY + ["map"], model, f"Input vs encoder: binary labels", base_rates(results, "raw", KEY_BINARY + ["map"])))
        out.append(table(results, encoder, KEY_REGRESSION, model, f"Input vs encoder: regression labels", base_rates(results, "raw", KEY_REGRESSION)))
        out.append(table(results, families, KEY_BINARY, model, f"Feature families: binary labels"))
        out.append(table(results, families, KEY_REGRESSION, model, f"Feature families: regression labels"))
        out.append(table(results, position, KEY_BINARY + ["map"], model, f"Absolute position ablation: binary labels"))
        out.append(table(results, position, KEY_REGRESSION, model, f"Absolute position ablation: regression labels"))
        heatmap(results, encoder, KEY_BINARY, model, res_dir / f"heat-encoder-binary-{model}.png", f"AUC by feature set ({model})")
        heatmap(results, encoder, KEY_REGRESSION, model, res_dir / f"heat-encoder-regression-{model}.png", f"R^2 by feature set ({model})")
        heatmap(results, families, KEY_BINARY, model, res_dir / f"heat-families-binary-{model}.png", f"AUC with candidate features ({model})")
        heatmap(results, families, KEY_REGRESSION, model, res_dir / f"heat-families-regression-{model}.png", f"R^2 with candidate features ({model})")
    out.append(perturb_plot(res_dir))
    # the encoder comparison across checkpoints / the position ablation of the encoder itself
    for other in ("probes-024-nopos.json", "probes-early.json"):
        if (res_dir / other).exists():
            o = json.loads((res_dir / other).read_text())
            merged = {**{f"{k}@{other[7:-5]}": v for k, v in o.items()}, **{f"{k}@024": results[k] for k in o if k in results}}
            feats = sorted(merged)
            out.append(table(merged, feats, KEY_BINARY + ["map"], "mlp", f"Encoder features: 024 vs {other[7:-5]} (binary)"))
            out.append(table(merged, feats, KEY_REGRESSION, "mlp", f"Encoder features: 024 vs {other[7:-5]} (regression)"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
