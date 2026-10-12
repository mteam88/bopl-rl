"""Can the search's candidate ranking be predicted from what the net sees?

    venv/bin/python train/q_probe.py BASE.bin --data DIR [DIR ...] --games 600 --min-spread 0.02

Replays recorded games through the base (teacher-forced, as train/distill.py) and keeps,
on decisions whose candidate return spread is >= --min-spread, the actor trunk output y
(what every policy head reads), the 48-float privileged block the critic adds, and the k
candidates with their returns (`cand`, or the relabel where `labeled`). Then fits a
small MLP q(features, action) on 90% of the games, scored only on within-decision
deviations (the ranking is all a policy can use), and reports on the held-out 10%:

    r        correlation of predicted and recorded within-decision return deviations
    gain     return of the probe's argmax candidate minus the base's expected candidate
             return, next to the search's own (argmax of the recorded returns; biased up
             by rollout noise) and the base's argmax

for features y, y + privileged, and the base's own candidate log-probs (prior). If y can't
rank the candidates, no fine-tune of the linear policy head on y will.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'eval')]
import layout  # noqa: E402
from distill import Games, load_policy, masked_log_softmax, candidate_logp  # noqa: E402


def collect(base, games, rows, device, min_spread):
    out = dict(y=[], priv=[], act=[], q=[], valid=[], prior=[], game=[])
    for gi in range(0, len(rows), 16):
        chunk = rows[gi:gi + 16]
        b = Games.load(chunk)
        labeled = None
        S, T = b['valid'].shape
        state = base.initial(S).to(device)
        for i in range(T):
            v = torch.from_numpy(b['valid'][:, i]).to(device)
            if not bool(v.any()):
                break
            x = torch.from_numpy(b['obs'][:, i]).to(device).float()
            m = torch.from_numpy(b['mask'][:, i]).to(device)
            c = torch.from_numpy(b['cand'][:, i]).to(device)
            with torch.no_grad():
                y, new = base.trunk(x, state)
                logits = y @ base.w['actor'].t()
                lg, _ = base.game_logits(torch.cat([logits, logits[:, :1]], -1) if logits.shape[1] == base.width else logits)
                lp = masked_log_softmax(lg, m)
                cl, cv = candidate_logp(lp, c)
            state = torch.where(v[None, :, None], new, state)
            q = c[..., 4]
            spread = q.masked_fill(~cv, -1e9).amax(1) - q.masked_fill(~cv, 1e9).amin(1)
            keep = v & (cv.sum(1) >= 2) & (spread >= min_spread)
            if not bool(keep.any()):
                continue
            k = keep
            out['y'].append(y[k].half().cpu()); out['priv'].append(x[k, layout.V4_PRIVILEGED_OFFSET:layout.V4_PRIVILEGED_OFFSET + layout.V4_PRIVILEGED].half().cpu())
            out['act'].append(c[k][..., :3].long().cpu()); out['q'].append(q[k].cpu()); out['valid'].append(cv[k].cpu())
            out['prior'].append(cl[k].cpu())
            out['game'].append((torch.arange(S, device=device)[k] // 2 + gi).cpu())
        print(f'{min(gi + 16, len(rows))}/{len(rows)} games, {sum(len(t) for t in out["q"])} decisions kept', flush=True)
    return {k: torch.cat(v) for k, v in out.items()}


class Probe(torch.nn.Module):
    def __init__(self, width, hidden=256):
        super().__init__()
        self.aim = torch.nn.Embedding(layout.ACT_AIM, 32)
        self.btn = torch.nn.Embedding(layout.ACT_BUTTONS, 16)
        self.hold = torch.nn.Embedding(layout.ACT_HOLD, 8)
        self.state = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, hidden), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden + 56, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                                        torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, feat, act):   # feat (N, W), act (N, K, 3) -> (N, K)
        h = self.state(feat)
        K = act.shape[1]
        a = torch.cat([self.aim(act[..., 0].clamp(0, 255)), self.btn(act[..., 1].clamp(0, 15)), self.hold(act[..., 2].clamp(0, 4))], -1)
        return self.head(torch.cat([h[:, None].expand(-1, K, -1), a], -1)).squeeze(-1)


def centered(q, valid):
    n = valid.sum(1, keepdim=True).clamp_min(1)
    mean = (q * valid).sum(1, keepdim=True) / n
    return (q - mean) * valid


def score(pred, d, idx):
    q, valid, prior = d['q'][idx], d['valid'][idx], d['prior'][idx]
    pq, rq = centered(pred, valid), centered(q, valid)
    r = float((pq * rq).sum() / (pq.pow(2).sum().sqrt() * rq.pow(2).sum().sqrt()).clamp_min(1e-12))
    pb = torch.softmax(prior.masked_fill(~valid, -1e9), 1)
    eb = (pb * q * valid).sum(1)
    pick = lambda s: q.gather(1, s.masked_fill(~valid, -1e9).argmax(1, keepdim=True)).squeeze(1)
    # Soft improvement: pi'(c) ∝ pi_base(c) exp(beta * standardised prediction), the policy a student could play.
    z = pq / pq[valid].std().clamp_min(1e-12)
    soft = {beta: float(((torch.softmax((prior + beta * z).masked_fill(~valid, -1e9), 1) * q * valid).sum(1) - eb).mean())
            for beta in (0.25, 0.5, 1, 2)}
    score.soft = soft
    return r, float((pick(pred) - eb).mean()), float((pick(q) - eb).mean()), float((pick(prior) - eb).mean())


def fit(feat, d, train, test, device, epochs, seed=0):
    torch.manual_seed(seed)
    model = Probe(feat.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    q, valid, act = d['q'].to(device), d['valid'].to(device), d['act'].to(device)
    feat = feat.float().to(device)
    scale = float(centered(q, valid)[valid].std())
    best = None
    for epoch in range(epochs):
        perm = train[torch.randperm(len(train))]
        model.train()
        for j in range(0, len(perm), 1024):
            b = perm[j:j + 1024].to(device)
            pred = model(feat[b], act[b])
            loss = ((centered(pred, valid[b]) - centered(q[b], valid[b]) / scale) ** 2 * valid[b]).sum() / valid[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pt = torch.cat([model(feat[test[j:j + 4096].to(device)], act[test[j:j + 4096].to(device)]) for j in range(0, len(test), 4096)]).cpu()
            ptr = torch.cat([model(feat[train[j:j + 4096].to(device)], act[train[j:j + 4096].to(device)]) for j in range(0, min(len(train), 40960), 4096)]).cpu()
        dc = {k: v.cpu() for k, v in d.items()}
        r_te = score(pt, dc, test)
        r_tr = score(ptr, dc, train[:40960])
        r_te = score(pt, dc, test)
        soft = ' '.join(f'b{b}:{v * 1000:+.2f}' for b, v in score.soft.items())
        print(f'  epoch {epoch + 1}: train r {r_tr[0]:.3f} | held-out r {r_te[0]:.3f}, argmax gain {r_te[1] * 1000:.2f} '
              f'(search {r_te[2] * 1000:.2f}, base argmax {r_te[3] * 1000:.2f}), soft gain {soft} x1e-3', flush=True)
        if best is None or r_te[0] > best[0]:
            best = r_te
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('base', type=Path)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--games', type=int, default=600)
    p.add_argument('--min-spread', type=float, default=0.02)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache', type=Path, help='save/load the collected features here')
    a = p.parse_args(argv)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    if a.cache and a.cache.exists():
        d = torch.load(a.cache)
    else:
        base = load_policy(a.base, device)
        games = Games(a.data, 16, 0, 0.0001)
        rows = (games.holdout + games.train)[:a.games]
        d = collect(base, games, rows, device, a.min_spread)
        if a.cache:
            torch.save(d, a.cache)
    n_games = int(d['game'].max()) + 1
    test_games = torch.arange(n_games)[torch.arange(n_games) % 10 == 0]
    is_test = torch.isin(d['game'], test_games)
    train, test = torch.nonzero(~is_test).squeeze(1), torch.nonzero(is_test).squeeze(1)
    print(f'{len(d["q"])} decisions (spread >= {a.min_spread}), {len(train)} train, {len(test)} held out ({len(test_games)} games)')
    r, g, gs, gb = score(d['prior'][test].float(), d, test)
    print(f'prior (base log-probs): held-out r {r:.3f}, base argmax gain {gb * 1000:.2f}, search {gs * 1000:.2f} x1e-3')
    for name, feat in (('y', d['y']), ('y + privileged', torch.cat([d['y'], d['priv']], 1))):
        print(f'{name}:')
        best = fit(feat, d, train, test, device, a.epochs)
        print(f'{name}: best held-out r {best[0]:.3f}, gain {best[1] * 1000:.2f} of the search\'s {best[2] * 1000:.2f} x1e-3')


if __name__ == '__main__':
    main()
