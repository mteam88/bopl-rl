"""Expert iteration, the training half: fine-tune a v4 checkpoint toward the search's
improved policy on recorded games (eval/search_record.py).

    venv/bin/python train/distill.py BASE.bin --data DIR [DIR ...] --out RUN \\
        --epochs 2 --games-per-batch 32 --lr 1e-4 --beta 10 --kl 0.1 --value 1.0

Loss, per recorded decision (both seats, teacher-forced replay of the recorded
observation rows so the recurrent state is the student's own):

  improved policy   pi'(a) = pi_base(a) w(a) / Z over the joint action space, where the
                    k searched candidates get w(c) = exp(beta (q(c) - q_bar)) from their mean
                    rollout return q(c) (q_bar is the prior-weighted mean over the candidates)
                    and every other action keeps w = 1, so pi' = pi_base where the search saw
                    nothing and sharpens toward the candidates whose returns stood out.
  policy term       sum over heads KL(marginal_h(pi') || pi_student_h): the student factorises
                    aim x buttons x hold, and the closest product to pi' is its marginals
  --target onehot   (default) replaces pi' by the one-hot of the action the search chose: the search
                    is a deterministic argmax over k prior samples under common random numbers, so
                    the one-hot is its exact output; the soft target needs a return scale (beta)
                    and on this game the candidates' returns differ by ~0.003 at the median, which
                    no single beta turns into both a meaningful and a bounded weight
  --target listwise KL(t || softmax_c log pi_student(c)) over the k candidates only, with
                    t(c) ∝ pi_base(c) exp(beta q(c)): moves only the net's preference *among* the
                    searched candidates (the search's selection), not its absolute log-probs
                    (the onehot target mostly taught the net the search's flatter sampling,
                    docs/EI.md section 5); equal returns -> t = base preference -> no gradient
  anchor            kl * sum over heads KL(pi_base_h || pi_student_h)
  value             value * (v_student - v_root_base)^2, keeps the critic where it was
                    (the critic is privileged; PPO can recalibrate it later)

Masks are the recorded ones (latched buttons/hold zeroed), applied to base, target and
student alike. Writes RUN/policy-<step>.bin (+ .json sidecar, native checkpoint format,
loadable by every evaluator) after every epoch and RUN/log.jsonl.
"""
import argparse
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'eval')]
import layout  # noqa: E402
import policy_v4  # noqa: E402
from native_checkpoint import write_checkpoint  # noqa: E402
from native_policy import load  # noqa: E402

HEADS = [(0, layout.ACT_AIM), (layout.ACT_AIM, layout.ACT_AIM + layout.ACT_BUTTONS),
         (layout.ACT_AIM + layout.ACT_BUTTONS, layout.LOGITS)]
NEG = -1e9


def load_policy(path, device):
    policy, meta = load(Path(path).resolve())
    flat = torch.from_numpy(np.array(policy.weights, dtype=np.float32)).to(device)
    return policy_v4.PolicyV4(flat, meta['hidden'], meta['num_layers'], meta['aim_fourier'], meta['aim_ar'], **meta['v4_shape'])


def masked_log_softmax(logits, mask):
    """Per-head log-softmax over allowed actions (mask 1 = allowed); disallowed -> very negative."""
    out = torch.empty_like(logits)
    for lo, hi in HEADS:
        x = logits[..., lo:hi].masked_fill(mask[..., lo:hi] == 0, NEG)
        out[..., lo:hi] = torch.log_softmax(x, -1)
    return out


def improved_targets(base_logp, cand, mask, beta, clip):
    """Per-head marginals of the improved joint policy.

    base_logp (S, 277) masked log-probs of the base; cand (S, K, 8) search candidate rows
    (aim, buttons, hold, visits, mean, prior, chosen, var); rows with visits 0 are padding.
    Returns (targets (S, 277) probabilities per head, stats dict)."""
    S, K, _ = cand.shape
    aim = cand[..., 0].long().clamp(0, layout.ACT_AIM - 1)
    btn = cand[..., 1].long().clamp(0, layout.ACT_BUTTONS - 1)
    hold = cand[..., 2].long().clamp(0, layout.ACT_HOLD - 1)
    valid = cand[..., 3] > 0
    q = cand[..., 4]
    # Joint prior of each candidate under the base (masked heads): p(c) = p_aim p_btn p_hold.
    lp = (base_logp.gather(1, aim) + base_logp.gather(1, layout.ACT_AIM + btn)
          + base_logp.gather(1, layout.ACT_AIM + layout.ACT_BUTTONS + hold))
    p = torch.exp(lp) * valid
    p_sum = p.sum(1, keepdim=True).clamp_min(1e-12)
    q_bar = (p * q).sum(1, keepdim=True) / p_sum
    w = torch.exp((beta * (q - q_bar)).clamp(-clip, clip)) * valid + (~valid)   # padding: w = 1 (no effect)
    delta = p * (w - 1)                                                          # (S, K) mass change per candidate
    Z = 1 + delta.sum(1, keepdim=True)
    base_p = torch.exp(base_logp)
    targets = base_p.clone()
    targets.scatter_add_(1, aim, delta)
    targets.scatter_add_(1, layout.ACT_AIM + btn, delta)
    targets.scatter_add_(1, layout.ACT_AIM + layout.ACT_BUTTONS + hold, delta)
    targets = (targets / Z).clamp_min(0)
    for lo, hi in HEADS:   # renormalise each head (clamp and fp error)
        targets[:, lo:hi] = targets[:, lo:hi] / targets[:, lo:hi].sum(1, keepdim=True).clamp_min(1e-12)
    targets = targets * (mask != 0)
    # How far the target moved from the base, per decision (sum over heads of KL(target || base)).
    moved = sum((targets[:, lo:hi] * (torch.log(targets[:, lo:hi].clamp_min(1e-12)) - base_logp[:, lo:hi])).sum(1)
                for lo, hi in HEADS)
    return targets, dict(moved=moved, cand_mass=p_sum.squeeze(1), q_spread=(q.masked_fill(~valid, 0).amax(1) - q.masked_fill(~valid, 1e9).amin(1)))


def onehot_targets(base_logp, cand, mask):
    """Per-head one-hot of the search's chosen candidate (the search policy is a deterministic argmax
    over its k candidates under common random numbers, so this is its exact output)."""
    S = cand.shape[0]
    chosen = cand[..., 6] > 0
    aim = (cand[..., 0] * chosen).sum(1).long().clamp(0, layout.ACT_AIM - 1)
    btn = (cand[..., 1] * chosen).sum(1).long().clamp(0, layout.ACT_BUTTONS - 1)
    hold = (cand[..., 2] * chosen).sum(1).long().clamp(0, layout.ACT_HOLD - 1)
    targets = torch.zeros_like(base_logp)
    rows = torch.arange(S, device=base_logp.device)
    targets[rows, aim] = 1
    targets[rows, layout.ACT_AIM + btn] = 1
    targets[rows, layout.ACT_AIM + layout.ACT_BUTTONS + hold] = 1
    targets = targets * (mask != 0)
    moved = -(base_logp[rows, aim] + base_logp[rows, layout.ACT_AIM + btn] + base_logp[rows, layout.ACT_AIM + layout.ACT_BUTTONS + hold])
    valid = cand[..., 3] > 0
    q = cand[..., 4]
    return targets, dict(moved=moved, cand_mass=torch.exp(base_logp).new_zeros(S),
                         q_spread=(q.masked_fill(~valid, -1e9).amax(1) - q.masked_fill(~valid, 1e9).amin(1)))


def candidate_logp(logp, cand):
    """(S, K) joint log-prob of each candidate under masked per-head log-probs; padding -> NEG."""
    valid = cand[..., 3] > 0
    aim = cand[..., 0].long().clamp(0, layout.ACT_AIM - 1)
    btn = cand[..., 1].long().clamp(0, layout.ACT_BUTTONS - 1)
    hold = cand[..., 2].long().clamp(0, layout.ACT_HOLD - 1)
    cl = (logp.gather(1, aim) + logp.gather(1, layout.ACT_AIM + btn)
          + logp.gather(1, layout.ACT_AIM + layout.ACT_BUTTONS + hold))
    return cl.masked_fill(~valid, NEG), valid


def listwise_targets(base_cl, cand, valid, beta):
    """Target over the k candidates only: t(c) ∝ p_base(c) exp(beta q(c)), the completed-Q
    improved policy restricted to the searched set. Where the returns are equal it is the base's
    own preference, so coin-flip decisions give no gradient; the rest of the action space is left
    to the anchor."""
    q = cand[..., 4]
    return torch.softmax(base_cl + (beta * q).masked_fill(~valid, 0), 1)


def kl_heads(p_logp_or_p, student_logp, mask, probs=False):
    """Sum over heads of KL(p || student) given p as probabilities (probs=True) or log-probs."""
    total = 0
    for lo, hi in HEADS:
        m = mask[:, lo:hi] != 0
        if probs:
            p = p_logp_or_p[:, lo:hi]
            logp = torch.log(p.clamp_min(1e-12))
        else:
            logp = p_logp_or_p[:, lo:hi]
            p = torch.exp(logp)
        total = total + (p * (logp - student_logp[:, lo:hi]) * m).sum(1)
    return total


class Games:
    """Streams buckets of games (decompressed in a background thread) sorted by length."""
    def __init__(self, dirs, games_per_batch, seed, holdout=0.05):
        rows = []
        for d in dirs:
            for line in (Path(d) / 'games.jsonl').read_text().splitlines():
                info = json.loads(line)
                path = Path(d) / f"g{info['seed']}.npz"
                if path.exists():
                    rows.append((path, info['decisions']))
        rng = np.random.default_rng(seed)
        rng.shuffle(rows)
        n_hold = max(1, int(len(rows) * holdout))
        self.holdout, self.train = rows[:n_hold], rows[n_hold:]
        self.games_per_batch = games_per_batch
        self.rng = rng
        self.decisions = sum(t for _, t in self.train)

    def buckets(self, rows, shuffle):
        rows = sorted(rows, key=lambda r: r[1])
        chunks = [rows[i:i + self.games_per_batch] for i in range(0, len(rows), self.games_per_batch)]
        if shuffle:
            self.rng.shuffle(chunks)
        return chunks

    @staticmethod
    def load(chunk):
        """-> dict of padded arrays: obs (S, T, 4812) f16, mask (S, T, 277) u8, cand (S, T, K, 8) f32,
        root_value (S, T) f32, valid (S, T) bool; S = 2 * games (seat-major within game)."""
        datas = [np.load(path) for path, _ in chunk]
        T = max(d['obs'].shape[0] for d in datas)
        S = 2 * len(datas)
        K = datas[0]['cand'].shape[2]
        obs = np.zeros((S, T, layout.V4_OBS_SIZE), np.float16)
        mask = np.zeros((S, T, layout.LOGITS), np.uint8)
        cand = np.zeros((S, T, K, 8), np.float32)
        value = np.zeros((S, T), np.float32)
        valid = np.zeros((S, T), bool)
        labeled = np.zeros((S, T), bool)
        for g, d in enumerate(datas):
            t = d['obs'].shape[0]
            for seat in (0, 1):
                s = 2 * g + seat
                obs[s, :t] = d['obs'][:, seat]
                mask[s, :t] = d['mask'][:, seat]
                cand[s, :t] = d['cand'][:, seat]
                value[s, :t] = d['root_value'][:, seat]
                valid[s, :t] = True
                labeled[s, :t] = d['labeled'][:, seat] if 'labeled' in d.files else True
        return dict(obs=obs, mask=mask, cand=cand, value=value, valid=valid, labeled=labeled)

    def stream(self, rows, shuffle):
        q = queue.Queue(maxsize=2)
        chunks = self.buckets(rows, shuffle)

        def produce():
            for chunk in chunks:
                q.put(self.load(chunk))
            q.put(None)
        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            yield item


def run_bucket(student, base, bucket, device, a, train, optimizer=None):
    """One bucket of sequences, time-major in chunks of a.bptt steps. Returns summed stats."""
    S, T = bucket['valid'].shape
    state_s = student.initial(S).to(device)
    state_b = base.initial(S).to(device)
    stats = dict(rows=0.0, policy=0.0, anchor=0.0, value=0.0, moved=0.0, agree_student=0.0, agree_base=0.0,
                 chosen_student=0.0, chosen_base=0.0, q_spread=0.0,
                 cand_rows=0.0, pref_student=0.0, pref_base=0.0, q_gain=0.0, q_gain_search=0.0)
    for t0 in range(0, T, a.bptt):
        t1 = min(T, t0 + a.bptt)
        obs = torch.from_numpy(bucket['obs'][:, t0:t1]).to(device, non_blocking=True)
        mask = torch.from_numpy(bucket['mask'][:, t0:t1]).to(device, non_blocking=True)
        cand = torch.from_numpy(bucket['cand'][:, t0:t1]).to(device, non_blocking=True)
        value = torch.from_numpy(bucket['value'][:, t0:t1]).to(device, non_blocking=True)
        valid = torch.from_numpy(bucket['valid'][:, t0:t1]).to(device, non_blocking=True)
        labeled = torch.from_numpy(bucket['labeled'][:, t0:t1] if a.labeled_only
                                   else np.ones((S, t1 - t0), bool)).to(device, non_blocking=True)
        loss_total = 0
        state_s = state_s.detach()
        for i in range(t1 - t0):
            v = valid[:, i]
            if not bool(v.any()):
                break
            x = obs[:, i].float()
            m = mask[:, i]
            with torch.no_grad():
                rows_b, new_b = base.step_raw(x, state_b)
                logits_b, _ = base.game_logits(rows_b)
                logp_b = masked_log_softmax(logits_b, m)
                cl_b, cvalid = candidate_logp(logp_b, cand[:, i])
                if a.target == 'soft':
                    targets, tstats = improved_targets(logp_b, cand[:, i], m, a.beta, a.clip)
                elif a.target == 'listwise':
                    targets = listwise_targets(cl_b, cand[:, i], cvalid, a.beta)
                    log_t = torch.log(targets.clamp_min(1e-12))
                    tstats = dict(moved=(targets * (log_t - torch.log_softmax(cl_b, 1))).sum(1),
                                  q_spread=onehot_targets(logp_b, cand[:, i], m)[1]['q_spread'])
                else:
                    targets, tstats = onehot_targets(logp_b, cand[:, i], m)
            with torch.set_grad_enabled(train):
                rows_s, new_s = student.step_raw(x, state_s)
                logits_s, value_s = student.game_logits(rows_s)
                logp_s = masked_log_softmax(logits_s, m)
                cl_s, _ = candidate_logp(logp_s, cand[:, i])
                if a.target == 'listwise':
                    # KL(t || softmax over candidates of the student): zero where t is the base's preference
                    policy_term = (targets * (log_t - torch.log_softmax(cl_s, 1))).sum(1) * cvalid.any(1)
                else:
                    policy_term = kl_heads(targets, logp_s, m, probs=True)
                anchor_term = kl_heads(logp_b, logp_s, m)
                value_term = (value_s - value[:, i]) ** 2
                policy_term = policy_term * labeled[:, i]
                if a.weight_spread > 0:
                    policy_term = policy_term * (tstats['q_spread'] / a.weight_spread).clamp(0, 1)
                per_row = policy_term + a.kl * anchor_term + a.value * value_term
                loss = (per_row * v).sum() / max(1, int(v.sum()))
                if train:
                    loss_total = loss_total + loss
            # Padded rows keep a frozen state (they never become valid again: padding is a tail).
            vv = v.unsqueeze(0).unsqueeze(-1)
            state_s = torch.where(vv, new_s, state_s)
            state_b = torch.where(vv, new_b, state_b)
            with torch.no_grad():
                chosen = cand[:, i][..., 6] > 0
                c_aim = (cand[:, i][..., 0] * chosen).sum(1).long()
                c_btn = (cand[:, i][..., 1] * chosen).sum(1).long()
                c_hold = (cand[:, i][..., 2] * chosen).sum(1).long()
                def agree(logp):
                    return ((logp[:, 0:256].argmax(1) == c_aim) & (logp[:, 256:272].argmax(1) == c_btn)
                            & (logp[:, 272:277].argmax(1) == c_hold)).float()
                def chosen_logp(logp):
                    return (logp.gather(1, c_aim[:, None]) + logp.gather(1, 256 + c_btn[:, None])
                            + logp.gather(1, 272 + c_hold[:, None])).squeeze(1)
                vf = v.float()
                # Selection among the searched candidates: the probability of the search's pick given
                # the net plays one of the k (pref), and the expected candidate return of the net's
                # candidate distribution minus the base's (q_gain, in return units; on held-out games
                # the rollout noise in q is independent of the net, so this is an unbiased measure of
                # how much of the search's selection the net took over). q_gain_search: the search's own.
                qc = cand[:, i][..., 4].masked_fill(~cvalid, 0)
                ps, pb = torch.softmax(cl_s.detach(), 1), torch.softmax(cl_b, 1)
                eb = (pb * qc).sum(1)
                has = cvalid.any(1).float() * vf * labeled[:, i]
                stats['cand_rows'] += float(has.sum())
                stats['pref_student'] += float(((ps * chosen).sum(1) * has).sum())
                stats['pref_base'] += float(((pb * chosen).sum(1) * has).sum())
                stats['q_gain'] += float((((ps * qc).sum(1) - eb) * has).sum())
                stats['q_gain_search'] += float((((qc * chosen).sum(1) - eb) * has).sum())
                stats['rows'] += float(vf.sum())
                stats['policy'] += float((policy_term * vf).sum())
                stats['anchor'] += float((anchor_term * vf).sum())
                stats['value'] += float((value_term * vf).sum())
                stats['moved'] += float((tstats['moved'] * vf).sum())
                stats['q_spread'] += float((tstats['q_spread'] * vf).sum())
                stats['agree_student'] += float((agree(logp_s) * vf).sum())
                stats['agree_base'] += float((agree(logp_b) * vf).sum())
                stats['chosen_student'] += float((chosen_logp(logp_s) * vf).sum())
                stats['chosen_base'] += float((chosen_logp(logp_b) * vf).sum())
        if train and torch.is_tensor(loss_total):
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([student.flat], a.clip_grad)
            optimizer.step()
    return stats


def summarise(stats):
    n, nc = max(stats['rows'], 1), max(stats['cand_rows'], 1)
    per_cand = ('pref_student', 'pref_base', 'q_gain', 'q_gain_search')
    return {k: (v if k in ('rows', 'cand_rows') else v / (nc if k in per_cand else n)) for k, v in stats.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('base', type=Path)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--games-per-batch', type=int, default=32, help='games per bucket (2 sequences each)')
    p.add_argument('--bptt', type=int, default=32, help='decisions per truncated-BPTT chunk')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--target', choices=('soft', 'onehot', 'listwise'), default='onehot',
                   help='onehot: the search\'s chosen action per head; soft: the completed-Q improved policy (beta) '
                        'marginalised onto the heads; listwise: the completed-Q policy over the k candidates only')
    p.add_argument('--beta', type=float, default=10.0, help='soft target: return -> log-weight scale of the improved policy')
    p.add_argument('--clip', type=float, default=4.0, help='clip of beta*(q - q_bar) in nats')
    p.add_argument('--weight-spread', type=float, default=0.0,
                   help='if > 0, weight each decision\'s policy term by min(1, (max q - min q) / this)')
    p.add_argument('--labeled-only', action='store_true',
                   help='policy term and candidate stats only on relabelled decisions (search_record --label-budget)')
    p.add_argument('--kl', type=float, default=0.1, help='anchor weight: KL(base || student)')
    p.add_argument('--value', type=float, default=1.0, help='value anchor weight')
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--holdout', type=float, default=0.05)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-games', type=int, help='use only this many training games (quick checks)')
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    policy, meta = load(a.base.resolve())
    shape = meta['v4_shape']
    base = load_policy(a.base, device)
    base_flat = base.flat
    student_flat = base_flat.clone().requires_grad_(True)
    student = policy_v4.PolicyV4(student_flat, meta['hidden'], meta['num_layers'], meta['aim_fourier'], meta['aim_ar'], **shape)
    optimizer = torch.optim.Adam([student_flat], lr=a.lr, betas=(0.9, 0.999))
    games = Games(a.data, a.games_per_batch, a.seed, a.holdout)
    if a.max_games:
        games.train = games.train[:a.max_games]
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / 'log.jsonl').open('a')
    config = dict(meta['config'])
    config['distill'] = dict(base=str(a.base.resolve()), base_step=meta['step'], data=[str(d) for d in a.data],
                             games=len(games.train), decisions=games.decisions, epochs=a.epochs, lr=a.lr, target=a.target, beta=a.beta,
                             clip=a.clip, kl=a.kl, value=a.value, weight_spread=a.weight_spread, bptt=a.bptt, labeled_only=a.labeled_only,
                             games_per_batch=a.games_per_batch)
    print(f'{len(games.train)} training games ({games.decisions} decisions), {len(games.holdout)} held out; device {device}', flush=True)

    def evaluate(tag, step):
        agg = None
        for bucket in games.stream(games.holdout, shuffle=False):
            s = run_bucket(student, base, bucket, device, a, train=False)
            agg = s if agg is None else {k: agg[k] + s[k] for k in agg}
        rec = dict(event='holdout', tag=tag, step=step, **summarise(agg))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n'); log.flush()
        return rec

    step = 0
    evaluate('epoch0', step)
    for epoch in range(1, a.epochs + 1):
        t_epoch = time.time()
        agg = None
        for b, bucket in enumerate(games.stream(games.train, shuffle=True)):
            t0 = time.time()
            s = run_bucket(student, base, bucket, device, a, train=True, optimizer=optimizer)
            agg = s if agg is None else {k: agg[k] + s[k] for k in agg}
            step += int(s['rows'])
            rec = dict(event='train', epoch=epoch, bucket=b, step=step, seconds=round(time.time() - t0, 1), **summarise(s))
            log.write(json.dumps(rec) + '\n'); log.flush()
            if b % 5 == 0:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        rec = dict(event='epoch', epoch=epoch, step=step, seconds=round(time.time() - t_epoch), **summarise(agg))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        log.write(json.dumps(rec) + '\n'); log.flush()
        evaluate(f'epoch{epoch}', step)
        game_steps = int(meta['step']) + step
        path = a.out / f'policy-{game_steps:09d}.bin'
        if path.exists():
            path.unlink(); Path(str(path) + '.json').unlink(missing_ok=True)
        weights = student_flat.detach().cpu().numpy().astype(np.float32)
        config['distill'].update(epoch=epoch, distilled_decisions=step)
        write_checkpoint(path, lambda name: weights.tofile(name), obs_size=layout.V4_OBS_SIZE, hidden=meta['hidden'],
                         num_layers=meta['num_layers'], architecture=layout.V4_ARCHITECTURE, game_steps=game_steps,
                         learner_steps=int(meta['learner_steps']) + step, config=config,
                         aim_fourier=meta['aim_fourier'], aim_ar=meta['aim_ar'], v4_shape=shape)
        print(f'wrote {path}', flush=True)
    log.close()


if __name__ == '__main__':
    main()
