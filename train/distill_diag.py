"""Where does a distilled net differ from its base, by how much the search's candidates differed?

    venv/bin/python train/distill_diag.py BASE.bin STUDENT.bin --data DIR [--seed 0 --holdout 0.05 --games 20]

Replays the held-out games of the same split train/distill.py used and, per decision, bins the
search's candidate return spread (max - min mean return over the k candidates). Per bin:

    agree    argmax of the net == the search's chosen action (all three heads)
    chosen   log-prob of the chosen action under the net
    pref     the net's probability of the chosen candidate given it plays one of the k
             candidates (joint over heads): did the net learn to *rank* the candidates
             like the search did, where the ranking mattered?
    rank     Spearman correlation of the net's candidate log-probs with the candidates'
             mean returns (how much the net orders the candidates by search return)
    q_gain   expected candidate return under the student's candidate distribution minus the
             base's (x1000), next to the search's own (its pick's return minus the base's
             expectation); on held-out games the rollout noise is independent of the nets

If pref/rank rise only on the low-spread bins, the student learned the search's sampling
pattern and not its selection.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'eval'), str(HERE.parent / 'native/tools')]
import layout  # noqa: E402
from distill import Games, load_policy, masked_log_softmax  # noqa: E402

BINS = [0, 0.005, 0.02, 0.05, 0.2, 10]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('base', type=Path)
    p.add_argument('student', type=Path)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--holdout', type=float, default=0.05)
    p.add_argument('--games', type=int, default=20, help='held-out games to replay')
    p.add_argument('--newest', action='store_true',
                   help='replay the newest --games games of games.jsonl instead of the split\'s held-out ones '
                        '(games recorded after the student was trained are unseen whatever its split was)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    base = load_policy(a.base, device)
    student = load_policy(a.student, device)
    games = Games(a.data, 8, a.seed, a.holdout)
    rows = games.holdout[:a.games]
    if a.newest:
        import json
        rows = []
        for d_ in a.data:
            for line in (d_ / 'games.jsonl').read_text().splitlines():
                info = json.loads(line)
                if (d_ / f"g{info['seed']}.npz").exists():
                    rows.append((d_ / f"g{info['seed']}.npz", info['decisions']))
        rows = rows[-a.games:]
    acc = {}  # bin -> list of per-row arrays
    for bucket in games.stream(rows, shuffle=False):
        S, T = bucket['valid'].shape
        state = {'b': base.initial(S).to(device), 's': student.initial(S).to(device)}
        for i in range(T):
            v = torch.from_numpy(bucket['valid'][:, i]).to(device)
            if not bool(v.any()):
                break
            x = torch.from_numpy(bucket['obs'][:, i]).to(device).float()
            m = torch.from_numpy(bucket['mask'][:, i]).to(device)
            cand = torch.from_numpy(bucket['cand'][:, i]).to(device)
            out = {}
            with torch.no_grad():
                for key, net in (('b', base), ('s', student)):
                    r, new = net.step_raw(x, state[key])
                    logits, _ = net.game_logits(r)
                    out[key] = masked_log_softmax(logits, m)
                    state[key] = torch.where(v[None, :, None], new, state[key])
                valid = cand[..., 3] > 0
                q = cand[..., 4]
                spread = q.masked_fill(~valid, -1e9).amax(1) - q.masked_fill(~valid, 1e9).amin(1)
                chosen = cand[..., 6] > 0
                ca = (cand[..., 0] * chosen).sum(1).long(); cb = (cand[..., 1] * chosen).sum(1).long(); ch = (cand[..., 2] * chosen).sum(1).long()
                aim = cand[..., 0].long().clamp(0, 255); btn = cand[..., 1].long().clamp(0, 15); hold = cand[..., 2].long().clamp(0, 4)
                rec = dict(spread=spread, valid=v)
                for key, logp in out.items():
                    rec['agree_' + key] = ((logp[:, :256].argmax(1) == ca) & (logp[:, 256:272].argmax(1) == cb) & (logp[:, 272:].argmax(1) == ch)).float()
                    rec['chosen_' + key] = (logp.gather(1, ca[:, None]) + logp.gather(1, 256 + cb[:, None]) + logp.gather(1, 272 + ch[:, None])).squeeze(1)
                    # joint log-prob of each candidate (duplicates in the Gumbel set are rare; keep them)
                    cl = logp.gather(1, aim) + logp.gather(1, 256 + btn) + logp.gather(1, 272 + hold)
                    cl = cl.masked_fill(~valid, -1e9)
                    cp = torch.softmax(cl, 1)
                    rec['pref_' + key] = (cp * chosen).sum(1)
                    rec['cl_' + key] = cl
                    rec['eq_' + key] = (cp * q.masked_fill(~valid, 0)).sum(1)
                rec['q'] = q; rec['cvalid'] = valid
                rec['qc'] = (q * chosen).sum(1)
                for k_, t in rec.items():
                    acc.setdefault(k_, []).append(t.cpu().numpy())
    d = {k: np.concatenate(v) for k, v in acc.items()}
    ok = d['valid']
    d = {k: v[ok] for k, v in d.items()}

    def spearman(cl, q, valid):
        out = []
        for i in range(len(q)):
            w = valid[i]
            if w.sum() < 3:
                out.append(np.nan); continue
            a_ = cl[i][w]; b_ = q[i][w]
            if np.ptp(b_) == 0:
                out.append(np.nan); continue
            ra = np.argsort(np.argsort(a_)); rb = np.argsort(np.argsort(b_))
            out.append(np.corrcoef(ra, rb)[0, 1])
        return np.array(out)
    rank_b = spearman(d['cl_b'], d['q'], d['cvalid']); rank_s = spearman(d['cl_s'], d['q'], d['cvalid'])
    print(f"{len(rows)} held-out games, {ok.sum()} decisions")
    print(f"{'spread':>14} {'n':>7} {'frac':>6} | {'agree b':>8} {'agree s':>8} | {'chosen b':>9} {'chosen s':>9} | {'pref b':>7} {'pref s':>7} | {'rank b':>7} {'rank s':>7} | {'qg s':>6} {'qg srch':>7}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (d['spread'] >= lo) & (d['spread'] < hi)
        if sel.sum() == 0:
            continue
        print(f"[{lo:.3f},{hi:.3f}) {sel.sum():7d} {sel.mean():6.3f} | {d['agree_b'][sel].mean():8.3f} {d['agree_s'][sel].mean():8.3f} "
              f"| {d['chosen_b'][sel].mean():9.3f} {d['chosen_s'][sel].mean():9.3f} | {d['pref_b'][sel].mean():7.3f} {d['pref_s'][sel].mean():7.3f} "
              f"| {np.nanmean(rank_b[sel]):7.3f} {np.nanmean(rank_s[sel]):7.3f} "
              f"| {1000 * (d['eq_s'][sel] - d['eq_b'][sel]).mean():6.2f} {1000 * (d['qc'][sel] - d['eq_b'][sel]).mean():7.2f}")
    sel = np.ones(len(d['spread']), bool)
    print(f"{'all':>14} {sel.sum():7d} {1.0:6.3f} | {d['agree_b'].mean():8.3f} {d['agree_s'].mean():8.3f} "
          f"| {d['chosen_b'].mean():9.3f} {d['chosen_s'].mean():9.3f} | {d['pref_b'].mean():7.3f} {d['pref_s'].mean():7.3f} "
          f"| {np.nanmean(rank_b):7.3f} {np.nanmean(rank_s):7.3f} "
          f"| {1000 * (d['eq_s'] - d['eq_b']).mean():6.2f} {1000 * (d['qc'] - d['eq_b']).mean():7.2f}")


if __name__ == '__main__':
    main()
