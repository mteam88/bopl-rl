"""Sensitivity of the policy to its absolute position input (memorisation probe b).

Runs the native CPU policy along recorded (game, player) sequences of a
checkpoint controller, reproducing its recurrent state, and at every step
compares the action distribution and value under the true observation with the
same observation where one input is shifted by +-delta and everything else is
fixed. Inputs compared:

  position      obs[8], obs[9]           self x/64, y/64 (the absolute position)
  opp_relpos    players row 1 cols 9,10  opponent position relative to self, /64
  time_left     obs[13]                  episode ticks remaining fraction
  null_count    obs[14+id]               source count of an ability never present
                                         in the data (its weights never saw a gradient)

KL is KL(base || perturbed) summed over the three heads (aim 256, buttons 16,
hold 5) on unmasked logits; value change is |v' - v|.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import layout  # noqa: E402
import schema  # noqa: E402
from native_policy import load, HEAD_BOUNDS  # noqa: E402
from generate import OUT, FINAL, CONTROLLERS  # noqa: E402

DELTAS = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)


def kl(base, other):
    total = torch.zeros(base.shape[0])
    for h in range(layout.NUM_ATNS):
        a, b = base[:, HEAD_BOUNDS[h]:HEAD_BOUNDS[h + 1]], other[:, HEAD_BOUNDS[h]:HEAD_BOUNDS[h + 1]]
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        total += (la.exp() * (la - lb)).sum(-1)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=FINAL)
    parser.add_argument("--controller", default="024-final")
    parser.add_argument("--sequences", type=int, default=192)
    parser.add_argument("--maps", type=int, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    obs = np.load(args.dataset / "obs.npy", mmap_mode="r"); meta = np.load(args.dataset / "meta.npy", mmap_mode="r")
    mc = {k: i for i, k in enumerate(schema.META)}
    ctrl = [c["name"] for c in CONTROLLERS].index(args.controller)
    keys = meta[:, mc["game"]].astype(np.int64) * 2 + meta[:, mc["player"]]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(keys)]])
    ok = meta[starts, mc["controller"]] == ctrl
    if args.maps is not None:
        ok &= np.isin(meta[starts, mc["map"]], args.maps)
    sel = np.flatnonzero(ok)
    sel = np.random.default_rng(0).choice(sel, min(args.sequences, len(sel)), replace=False)
    policy, _ = load(args.checkpoint)
    p = schema.COMPACT_OFFSETS["prefix"]
    # the null feature: an ability count column that is zero everywhere in a sample
    sample = obs[np.random.default_rng(1).choice(len(obs), 20000, replace=False)].astype(np.float32)
    counts = np.abs(sample[:, p + 14:p + 14 + layout.EMBEDDING_IDS]).max(0)
    null_id = int(np.flatnonzero(counts == 0)[0])
    pl = schema.COMPACT_OFFSETS["players"] + layout.PLAYER_FEATURES  # players row 1 = the opponent
    inputs = {"position": [p + 8, p + 9], "opp_relpos": [pl + 9, pl + 10], "time_left": [p + 13], "null_count": [p + 14 + null_id]}
    # scan the sequences in lockstep
    lengths = ends[sel] - starts[sel]
    L = int(lengths.max())
    state = policy.initial(len(sel))
    kls = {k: {d: [] for d in DELTAS} for k in inputs}; dvs = {k: {d: [] for d in DELTAS} for k in inputs}
    base_entropy = []
    for t in range(L):
        active = np.flatnonzero(lengths > t)
        rows = starts[sel][active] + t
        compact = obs[rows].astype(np.float32)
        full = torch.from_numpy(schema.expand(compact))
        st = state[:, active]
        raw, nxt = policy.step_raw(full, st)
        logits, values = policy.game_logits(raw)
        state[:, active] = nxt
        lp = logits[:, :layout.ACT_AIM].log_softmax(-1); base_entropy.append(float(-(lp.exp() * lp).sum(-1).mean()))
        # all perturbations of this step in one batch
        variants, tags = [], []
        for name, cols in inputs.items():
            for d in DELTAS:
                c = compact.copy(); c[:, cols] = np.clip(c[:, cols] + d, -1, 1); variants.append(c); tags.append((name, d))
        big = torch.from_numpy(schema.expand(np.concatenate(variants)))
        st_rep = st.repeat(1, len(variants), 1)
        raw2, _ = policy.step_raw(big, st_rep)
        logits2, values2 = policy.game_logits(raw2)
        n = len(active)
        for i, (name, d) in enumerate(tags):
            kls[name][d].append(kl(logits, logits2[i * n:(i + 1) * n]).numpy())
            dvs[name][d].append((values2[i * n:(i + 1) * n] - values).abs().numpy())
    result = dict(controller=args.controller, checkpoint=str(args.checkpoint), sequences=int(len(sel)), steps=int(lengths.sum()),
                  null_ability_id=null_id, aim_entropy_nats=float(np.mean(base_entropy)), inputs={})
    for name in inputs:
        result["inputs"][name] = {}
        for d in DELTAS:
            k = np.concatenate(kls[name][d]); v = np.concatenate(dvs[name][d])
            result["inputs"][name][str(d)] = dict(kl_mean=float(k.mean()), kl_median=float(np.median(k)), kl_p90=float(np.quantile(k, 0.9)),
                                                  dvalue_mean=float(v.mean()), dvalue_p90=float(np.quantile(v, 0.9)))
        print(name, {d: round(result["inputs"][name][str(d)]["kl_mean"], 4) for d in DELTAS},
              "dV", {d: round(result["inputs"][name][str(d)]["dvalue_mean"], 4) for d in DELTAS})
    out = args.out or args.dataset / "results" / f"perturb-{args.controller}.json"
    out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
