"""How reliable is an n=1 search label? Screened-relabel games (eval/search_record.py
--label-budget) hold, on each relabelled decision, the screen's candidates (`screen`, n=1)
and the relabel's (`cand`, more rollouts). The two searches draw their own candidates; over
the actions both drew, correlate the within-decision return deviations, and score the
screen's pick (argmax of its n=1 returns over the shared actions) against the screen's
prior-weighted mean twice: on the screen's own returns (biased up by the noise it selected
on) and on the relabel's independent returns (an unbiased estimate of the selection gain).

    venv/bin/python train/label_noise.py DIR [--games 100]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', type=Path)
    p.add_argument('--games', type=int, default=100)
    a = p.parse_args(argv)
    infos = [json.loads(line) for line in (a.data / 'games.jsonl').read_text().splitlines()][:a.games]
    pairs, shared, decisions, labeled, spread_s, spread_l = [], [], 0, 0, [], []
    own, indep = [], []
    for info in infos:
        d = np.load(a.data / f"g{info['seed']}.npz")
        decisions += d['labeled'].size
        lab = d['labeled']
        labeled += int(lab.sum())
        for ci, si in zip(d['cand'][lab], d['screen'][lab]):
            qa = {tuple(x[:3]): x[4] for x in ci if x[3] > 0}
            qb = {tuple(x[:3]): x[4] for x in si if x[3] > 0}
            pb = {tuple(x[:3]): x[5] for x in si if x[3] > 0}
            spread_l.append(np.ptp(list(qa.values()))); spread_s.append(np.ptp(list(qb.values())))
            keys = [k for k in qa if k in qb]
            shared.append(len(keys))
            if len(keys) >= 2:
                x = np.array([qa[k] for k in keys]); y = np.array([qb[k] for k in keys])
                pairs += list(zip(x - x.mean(), y - y.mean()))
                w = np.exp(np.array([pb[k] for k in keys]) - max(pb[k] for k in keys)); w /= w.sum()
                i = y.argmax()
                own.append(y[i] - (w * y).sum()); indep.append(x[i] - (w * x).sum())
    pairs = np.array(pairs)
    print(f'{len(infos)} games, {decisions} seat-decisions, {labeled} relabelled ({labeled / max(decisions, 1):.3f})')
    print(f'shared candidates per relabelled decision {np.mean(shared):.2f}; screen spread median {np.median(spread_s):.4f}, '
          f'relabel spread median {np.median(spread_l):.4f}')
    print(f'within-decision return deviations, relabel vs screen: r = {np.corrcoef(pairs.T)[0, 1]:.3f} over {len(pairs)} pairs')
    print(f'screen pick gain over the prior-weighted mean of shared candidates ({len(own)} decisions): '
          f'{np.mean(own) * 1000:.2f} on its own returns, {np.mean(indep) * 1000:.2f} '
          f'(+-{np.std(indep) / np.sqrt(len(indep)) * 1000:.2f}) on the relabel x1e-3')


if __name__ == '__main__':
    main()
