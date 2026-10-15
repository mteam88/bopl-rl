"""Behaviour cloning of human play (eval/human_dataset.py rows) into a v4 checkpoint.

    venv/bin/python train/human_bc.py f5.bin --data native/artifacts/rl/human-bc/data-2p --out RUN \\
        --epochs 12 --lr 3e-4 --kl 0 --value 1.0 [--random-init]

Loss per recorded decision of one human seat (teacher-forced: the student's recurrent
state runs over the recorded observation rows):

  policy   sum over heads of -log pi_student_h(a_h), with the recorded mask (a latched
           decision has one legal button combo and hold 0, so those heads add nothing)
  anchor   kl * sum over heads KL(pi_f5_h || pi_student_h) (needs f5's logits; skipped at kl 0)
  value    value * (v_student - v_f5)^2 at the f5 value logged with the row, so the critic
           stays where f5 had it (the RL phase recalibrates it)

The train/val split is by session (--val-sessions), so the val games come from sittings
the net never saw. Every epoch logs, on val, per head: NLL of the human's action, argmax
accuracy, and (aim) the median angle error of the argmax on directional decisions,
next to f5's own numbers on the same rows (logged by the dataset). Buttons and hold count
only on free (unlatched) decisions. Keeps RUN/best.bin (lowest val NLL) and
RUN/policy-<step>.bin per epoch (native format, loadable by every evaluator).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'eval'), str(HERE)]
import layout  # noqa: E402
import policy_v4  # noqa: E402
from distill import HEADS, kl_heads, load_policy, masked_log_softmax  # noqa: E402
from native_checkpoint import write_checkpoint  # noqa: E402
from native_policy import load  # noqa: E402

VAL_SESSIONS = ('old-20260916-165454', 'mac-20261010-003004', 'mac-20260917-000139')
NAMES = ('aim', 'buttons', 'hold')


def rows_of(dirs, faithful=True):
    out = []
    for d in dirs:
        for line in (Path(d) / 'games.jsonl').read_text().splitlines():
            r = json.loads(line)
            if faithful and not r['faithful']:
                continue
            r['npz'] = str(Path(d) / r['path'])
            out.append(r)
    return out


def load_bucket(rows):
    """Padded arrays for a list of seat rows: obs (S,T,4812) f16, mask (S,T,277) u8, action (S,T,3),
    latched, value_f5, f5_logp (S,T,3), valid (S,T)."""
    ds = [np.load(r['npz']) for r in rows]
    S, T = len(ds), max(d['obs'].shape[0] for d in ds)
    out = dict(obs=np.zeros((S, T, layout.V4_OBS_SIZE), np.float16), mask=np.zeros((S, T, layout.LOGITS), np.uint8),
               action=np.zeros((S, T, 3), np.int64), latched=np.zeros((S, T), bool), value=np.zeros((S, T), np.float32),
               f5_logp=np.zeros((S, T, 3), np.float32), valid=np.zeros((S, T), bool))
    for s, d in enumerate(ds):
        t = d['obs'].shape[0]
        out['obs'][s, :t] = d['obs']; out['mask'][s, :t] = d['mask']; out['action'][s, :t] = d['action']
        out['latched'][s, :t] = d['latched']; out['value'][s, :t] = d['f5_value']; out['f5_logp'][s, :t] = d['f5_logp']
        out['valid'][s, :t] = True
    return out


def angle_err(pred, true):
    """Degrees between aim bins (both directional: 1..255)."""
    d = (pred - true).abs().float() * (360.0 / 255.0)
    return torch.minimum(d, 360.0 - d)


def run_bucket(student, base, b, device, a, train, optimizer=None, collect=None):
    S, T = b['valid'].shape
    st = student.initial(S).to(device)
    sb = base.initial(S).to(device) if base is not None else None
    stats = {k: 0.0 for k in ('rows', 'loss', 'anchor', 'value')}
    for t0 in range(0, T, a.bptt):
        t1 = min(T, t0 + a.bptt)
        g = lambda k, dt=None: torch.from_numpy(b[k][:, t0:t1]).to(device, non_blocking=True)
        obs, mask, act, lat, val, valid = g('obs'), g('mask'), g('action'), g('latched'), g('value'), g('valid')
        f5lp = g('f5_logp')
        st = st.detach()
        total = 0
        for i in range(t1 - t0):
            v = valid[:, i]
            if not bool(v.any()):
                break
            x, m = obs[:, i].float(), mask[:, i]
            with torch.set_grad_enabled(train):
                rows, new = student.step_raw(x, st)
                logits, value = student.game_logits(rows)
                lp = masked_log_softmax(logits, m)
                idx = act[:, i] + torch.tensor([0, layout.ACT_AIM, layout.ACT_AIM + layout.ACT_BUTTONS], device=device)
                head_lp = lp.gather(1, idx)                       # (S, 3)
                nll = -head_lp.sum(1)
                per = nll + a.value * (value - val[:, i]) ** 2
                anchor = torch.zeros_like(nll)
                if base is not None and a.kl > 0:
                    with torch.no_grad():
                        rb, nb = base.step_raw(x, sb)
                        lb = masked_log_softmax(base.game_logits(rb)[0], m)
                    anchor = kl_heads(lb, lp, m)
                    per = per + a.kl * anchor
                    sb = torch.where(v[None, :, None], nb, sb)
                loss = (per * v).sum() / max(1, int(v.sum()))
                if train:
                    total = total + loss
            st = torch.where(v[None, :, None], new, st)
            vf = v.float()
            stats['rows'] += float(vf.sum()); stats['loss'] += float((nll.detach() * vf).sum())
            stats['anchor'] += float((anchor.detach() * vf).sum())
            stats['value'] += float((((value.detach() - val[:, i]) ** 2) * vf).sum())
            if collect is not None:
                with torch.no_grad():
                    am = [lp[:, lo:hi].argmax(1) for lo, hi in HEADS]
                    keep = v.cpu().numpy()
                    collect.append(dict(
                        keep=keep, latched=lat[:, i].cpu().numpy(), action=act[:, i].cpu().numpy(),
                        nll=-head_lp.detach().cpu().numpy(), argmax=torch.stack(am, 1).cpu().numpy(),
                        f5_nll=-f5lp[:, i].cpu().numpy(),
                        p_neutral=lp[:, 0].exp().cpu().numpy(),
                        p_press=1 - lp[:, layout.ACT_AIM].exp().cpu().numpy()))
        if train and torch.is_tensor(total):
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([student.flat], a.clip_grad)
            optimizer.step()
    return stats


def boot_ci(groups, fn, n=1000, seed=0):
    """Session-cluster bootstrap of fn over concatenated groups -> (value, lo, hi)."""
    rng = np.random.default_rng(seed)
    keys = list(groups)
    def f(ks):
        return fn({k2: np.concatenate([groups[k][k2] for k in ks]) for k2 in groups[keys[0]]})
    bs = [f(rng.choice(keys, len(keys))) for _ in range(n)]
    return float(f(keys)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def val_metrics(per_session, ci=False):
    """Per-head NLL/accuracy for the student and f5 (free decisions for buttons/hold)."""
    def head(h, what):
        def fn(d):
            sel = d['keep'] & (~d['latched'] if h > 0 else True)
            if what == 'nll': return d['nll'][sel, h].mean()
            if what == 'f5_nll': return d['f5_nll'][sel, h].mean()
            if what == 'acc': return (d['argmax'][sel, h] == d['action'][sel, h]).mean()
            if what == 'aim_err':
                dsel = sel & (d['action'][:, 0] > 0) & (d['argmax'][:, 0] > 0)
                return float(np.median(angle_err(torch.from_numpy(d['argmax'][dsel, 0]), torch.from_numpy(d['action'][dsel, 0])).numpy())) if dsel.any() else float('nan')
        return fn
    out = {}
    agg = lambda fn: (boot_ci(per_session, fn) if ci else (fn({k: np.concatenate([per_session[s][k] for s in per_session]) for k in next(iter(per_session.values()))}),))
    for h, name in enumerate(NAMES):
        for what in ('nll', 'f5_nll', 'acc'):
            out[f'{name}_{what}'] = agg(head(h, what))
    out['aim_err_deg'] = agg(head(0, 'aim_err'))
    out['total_nll'] = agg(lambda d: d['nll'][d['keep']].sum(1).mean())
    out['f5_total_nll'] = agg(lambda d: d['f5_nll'][d['keep']].sum(1).mean())
    out['p_neutral'] = agg(lambda d: d['p_neutral'][d['keep']].mean())
    out['human_neutral'] = agg(lambda d: (d['action'][d['keep'], 0] == 0).mean())
    out['p_press'] = agg(lambda d: d['p_press'][d['keep'] & ~d['latched']].mean())
    out['human_press'] = agg(lambda d: (d['action'][d['keep'] & ~d['latched'], 1] != 0).mean())
    return {k: (v[0] if not ci else v) for k, v in out.items()}


def collect_val(student, val_rows, device, a):
    per = {}
    for i in range(0, len(val_rows), a.seqs):
        chunk = val_rows[i:i + a.seqs]
        col = []
        run_bucket(student, None, load_bucket(chunk), device, a, train=False, collect=col)
        # split the batch rows back to sessions
        for s, r in enumerate(chunk):
            d = {k: np.concatenate([c[k][s:s + 1] for c in col]) for k in col[0]}
            d['keep'] = d['keep'].astype(bool); d['latched'] = d['latched'].astype(bool)
            g = per.setdefault(r['session'], [])
            g.append(d)
    return {s: {k: np.concatenate([d[k] for d in ds]) for k in ds[0]} for s, ds in per.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('base', type=Path, help='f5 (init and layout; anchor and value target)')
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seqs', type=int, default=32, help='seat sequences per bucket')
    p.add_argument('--bptt', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--cosine', action='store_true', help='cosine-decay the learning rate to 10%% over the run')
    p.add_argument('--kl', type=float, default=0.0, help='anchor weight: KL(f5 || student)')
    p.add_argument('--value', type=float, default=1.0)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--random-init', action='store_true', help='start from a fresh init (f5 layout) instead of f5')
    p.add_argument('--val-sessions', default=','.join(VAL_SESSIONS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--gpu-fraction', type=float, default=0.2, help='cap of this process on the shared GPU')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--train', default='all',
                   help="comma list of v4 tensors to train (eval/layout.policy_v4_tensors names, e.g. 'actor' keeps "
                        "f5's trunk and critic and fits only the decoder); 'all' trains everything")
    p.add_argument('--reinit', default='', help="comma list of v4 tensors to reset to a fresh init before training (e.g. 'actor')")
    p.add_argument('--max-seats', type=int, help='train/val on this many seats each (quick checks)')
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    policy, meta = load(a.base.resolve())
    shape = meta['v4_shape']
    base = load_policy(a.base, device)
    if a.random_init:
        init = torch.from_numpy(policy_v4.init_weights(meta['hidden'], meta['num_layers'], meta['aim_fourier'], meta['aim_ar'],
                                                       seed=a.seed, **shape)).to(device)
    else:
        init = base.flat.clone()
    if a.reinit:
        fresh = torch.from_numpy(policy_v4.init_weights(meta['hidden'], meta['num_layers'], meta['aim_fourier'], meta['aim_ar'],
                                                        seed=a.seed, **shape)).to(device)
        offset = 0
        for name, (r, c) in layout.policy_v4_tensors(meta['hidden'], meta['num_layers'], meta['aim_fourier'], meta['aim_ar'],
                                                     **{**layout.V4_DEFAULTS, **shape}):
            if name in a.reinit.split(','):
                init[offset:offset + r * c] = fresh[offset:offset + r * c]
            offset += layout.aligned(r * c)
    flat = init.clone().requires_grad_(True)
    student = policy_v4.PolicyV4(flat, meta['hidden'], meta['num_layers'], meta['aim_fourier'], meta['aim_ar'], **shape)
    opt = torch.optim.Adam([flat], lr=a.lr)
    if a.train != 'all':
        names = set(a.train.split(','))
        unknown = names - {n for n, _ in student.tensors}
        if unknown:
            raise SystemExit(f'unknown tensors {sorted(unknown)}')
        keep = torch.zeros_like(flat, dtype=torch.bool)
        offset = 0
        for name, (r, c) in student.tensors:
            if name in names:
                keep[offset:offset + r * c] = True
            offset += layout.aligned(r * c)
        flat.register_hook(lambda g: g * keep)
    rows = rows_of(a.data)
    val_s = set(a.val_sessions.split(','))
    train_rows = [r for r in rows if r['session'] not in val_s]
    val_rows = [r for r in rows if r['session'] in val_s]
    if a.max_seats:
        train_rows, val_rows = train_rows[:a.max_seats], val_rows[:a.max_seats]
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / 'log.jsonl').open('a')
    n_train = sum(r['decisions'] for r in train_rows)
    print(f'{len(train_rows)} train seats ({n_train} decisions), {len(val_rows)} val seats '
          f'({sum(r["decisions"] for r in val_rows)}); init {"random" if a.random_init else a.base.name}; device {device}', flush=True)
    config = dict(meta['config'])
    config['human_bc'] = dict(base=str(a.base.resolve()), base_step=meta['step'], data=[str(d) for d in a.data],
                              train_seats=len(train_rows), decisions=n_train, val_sessions=sorted(val_s), lr=a.lr, kl=a.kl,
                              value=a.value, random_init=a.random_init, cosine=a.cosine, train=a.train, reinit=a.reinit, bptt=a.bptt, seqs=a.seqs)

    def write(path, step, extra):
        if path.exists():
            path.unlink(); Path(str(path) + '.json').unlink(missing_ok=True)
        weights = flat.detach().cpu().numpy().astype(np.float32)
        cfg = dict(config); cfg['human_bc'] = dict(config['human_bc'], **extra)
        game_steps = int(meta['step'])
        write_checkpoint(path, lambda name: weights.tofile(name), obs_size=layout.V4_OBS_SIZE, hidden=meta['hidden'],
                         num_layers=meta['num_layers'], architecture=layout.V4_ARCHITECTURE, game_steps=game_steps + step,
                         learner_steps=int(meta['learner_steps']) + step, config=cfg,
                         aim_fourier=meta['aim_fourier'], aim_ar=meta['aim_ar'], v4_shape=shape)

    def evaluate(epoch, step):
        per = collect_val(student, val_rows, device, a)
        m = val_metrics(per)
        rec = dict(event='val', epoch=epoch, step=step, **{k: float(v) for k, v in m.items()})
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        log.write(json.dumps(rec) + '\n'); log.flush()
        return rec, per

    step, best = 0, (math.inf, None)
    rec, _ = evaluate(0, 0)
    for epoch in range(1, a.epochs + 1):
        t = time.time()
        order = sorted(train_rows, key=lambda r: r['decisions'])
        chunks = [order[i:i + a.seqs] for i in range(0, len(order), a.seqs)]
        rng.shuffle(chunks)
        agg = {}
        for k, chunk in enumerate(chunks):
            if a.cosine:
                frac = ((epoch - 1) * len(chunks) + k) / (a.epochs * len(chunks))
                for group in opt.param_groups:
                    group['lr'] = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            s = run_bucket(student, base, load_bucket(chunk), device, a, train=True, optimizer=opt)
            agg = {k: agg.get(k, 0) + v for k, v in s.items()}
            step += int(s['rows'])
        n = max(1, agg['rows'])
        tr = dict(event='train', epoch=epoch, step=step, seconds=round(time.time() - t), nll=agg['loss'] / n,
                  anchor=agg['anchor'] / n, value=agg['value'] / n)
        print(json.dumps(tr), flush=True); log.write(json.dumps(tr) + '\n'); log.flush()
        rec, per = evaluate(epoch, step)
        path = a.out / f'policy-epoch{epoch:02d}.bin'
        write(path, step, dict(epoch=epoch, val_nll=rec['total_nll']))
        if rec['total_nll'] < best[0]:
            best = (rec['total_nll'], epoch)
            write(a.out / 'best.bin', step, dict(epoch=epoch, val_nll=rec['total_nll']))
            ci = val_metrics(per, ci=True)
            (a.out / 'best.json').write_text(json.dumps(dict(epoch=epoch, step=step, metrics=ci), indent=1))
    print(f'best epoch {best[1]} val nll {best[0]:.3f}', flush=True)
    log.close()


if __name__ == '__main__':
    main()
