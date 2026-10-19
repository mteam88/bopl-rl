"""Carry a trained checkpoint's weights into a changed policy v4 architecture.

    native/artifacts/rl/venv/bin/python train/surgery.py OLD.bin NEW.bin --hidden 384
    native/artifacts/rl/venv/bin/python train/surgery.py OLD.bin NEW.bin --widen 2 --widen-init zero   # actor exact

Every v4 tensor is named and shaped by eval/layout.py, so surgery is a walk over
two segmentations of the same flat buffer: same name and shape, copy; same name,
new shape, copy the overlap and initialise the rest; new name, initialise; gone,
drop it. The report of which is which is the output that matters, so it always
prints. The result is written by native_checkpoint.write_checkpoint, the path
training publishes through, and loads with _C.load_weights like any checkpoint.

What a plain rectangle copy would get wrong, and this does not: most v4 axes are
concatenations. mingru rows are three stacked gates, fusion columns are the flat
projection then four (max || mean) pools, critic_in columns are the trunk then
the privileged observation, and actor rows are aim block(s) then buttons then
hold. Widen `hidden` with old[:rows, :cols] and the highway gate lands on the
update gate and the privileged inputs slide into the trunk: a checkpoint that
loads, validates, and plays like noise. Axes are copied segment by named segment.

v4 only (layout.V4_ARCHITECTURE). The v3 and puffernet-mingru blobs have no named
segmentation to walk -- layout.policy_weights is a float count and nothing more --
and v4 is the architecture being changed, so those are refused rather than guessed.
Both checkpoints are read through the layout module as it is now, so this moves
weights between architectures, not between observation layouts: changing V4_FLAT
or a set's width edits layout.py and invalidates the source's obs_size.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval')]
import layout
from native_checkpoint import write_checkpoint

# Fan-in scaled uniform bound per tensor, read off the trainer's own initialiser
# (bopl_v4.cu enc_init/dec_init, PufferLib models.cu mingru_init_weights, through
# kernels.cu puf_kaiming_init): U(-gain/sqrt(cols), gain/sqrt(cols)). New weights
# get exactly the distribution a fresh run of the target architecture would give
# them -- including the two deliberately small ones, ffn_out and critic_b2, whose
# 0.1 keeps a new residual branch near identity instead of shouting over the
# trained trunk. Zeros would be a different, unmotivated choice: dead ReLU rows in
# the set encoders and a permanently symmetric embedding.
GAINS = {'embedding': 1.0, 'flat': math.sqrt(2.0), 'fusion': 1.0,
         'ffn_in': math.sqrt(2.0), 'ffn_out': 0.1, 'actor': 1.0,
         'critic_in': math.sqrt(2.0), 'critic_b1': math.sqrt(2.0),
         'critic_b2': 0.1, 'critic_out': 1.0}
ARCH_KEYS = ('hidden', 'num_layers', 'aim_fourier', 'aim_ar') + tuple(layout.V4_DEFAULTS)


def gain(name):
    if name.startswith('mingru'):
        return 1.0
    if name.endswith('_w1') or name.endswith('_w2'):
        return math.sqrt(2.0)
    return GAINS[name]


def architecture(hidden, num_layers, aim_fourier=4, aim_ar=0, shape=None):
    """Every parameter policy_v4_tensors reads, defaults filled in."""
    arch = dict(layout.V4_DEFAULTS, hidden=int(hidden), num_layers=int(num_layers),
                aim_fourier=int(aim_fourier), aim_ar=int(aim_ar))
    arch.update({k: int(v) for k, v in (shape or {}).items() if k in layout.V4_DEFAULTS})
    return arch


def v4_shape(arch):
    """The subset write_checkpoint records as v4_shape."""
    return {k: arch[k] for k in layout.V4_DEFAULTS}


def tensors(arch):
    """name -> (flat offset, (rows, cols)), in policy_v4_tensors order."""
    out, offset = {}, 0
    rest = {k: v for k, v in arch.items() if k not in ('hidden', 'num_layers')}
    for name, shape in layout.policy_v4_tensors(arch['hidden'], arch['num_layers'], **rest):
        out[name] = (offset, shape)
        offset += layout.aligned(shape[0] * shape[1])
    return out


def segments(name, arch):
    """(row segments, column segments) as (label, size, scale) triples.

    An axis that is a concatenation names its parts so a copy matches part to
    part. `scale` is what a stored float is worth to the forward pass, so a copy
    across a rescaled basis can keep the function it computed.
    """
    rows = [('', None, 1.0)]
    cols = [('', None, 1.0)]
    if name == 'fusion':                     # policy_v4.trunk: [flat, per-set max || mean]
        cols = [('flat', arch['flat_dim'], 1.0)]
        cols += [(f'{name}_{pool}', arch['set_dim'], 1.0) for name, *_ in layout.V4_SETS for pool in ('max', 'mean')]
    elif name == 'critic_in':                # policy_v4.value: [trunk, privileged]
        cols = [('trunk', arch['hidden'], 1.0)]
        if arch['critic_privileged']:
            cols.append(('privileged', layout.V4_PRIVILEGED, 1.0))
    elif name.startswith('mingru'):          # policy_v4.mingru_step: combined.chunk(3, -1)
        rows = [(gate, arch['hidden'], 1.0) for gate in ('candidate', 'update', 'highway')]
    elif name == 'actor':                    # policy_v4.game_logits: aim block(s), buttons, hold
        rows = aim_segments(arch) + [('buttons', layout.ACT_BUTTONS, 1.0), ('hold', layout.ACT_HOLD, 1.0)]
        if not arch['critic_dim']:
            rows.append(('value', 1, 1.0))
    return rows, cols


def aim_segments(arch):
    """Decoder aim rows: one block per button combo under aim_ar, otherwise one.

    A Fourier block is a constant row plus 2K harmonic coefficients read through a
    basis carrying 1/sqrt(K) (policy_v4.fourier_aim_matrix), so that factor is the
    block's scale: copying K=4 into K=8 rescales the coefficients and the aim
    distribution comes out unchanged. Raw 256-logit blocks share no label with
    Fourier blocks -- the coefficients are not comparable -- so K to 0 reinitialises.
    """
    harmonics, blocks = arch['aim_fourier'], layout.ACT_BUTTONS if arch['aim_ar'] else 1
    out = []
    for block in range(blocks):
        if harmonics:
            out += [(f'aim{block}.constant', 1, 1.0),
                    (f'aim{block}.harmonics', 2 * harmonics, 1.0 / math.sqrt(harmonics))]
        else:
            out.append((f'aim{block}.logits', layout.ACT_AIM, 1.0))
    return out


# The tail of a grown Fourier aim block starts at zero, so raising K starts as
# exactly the aim distribution the source had and learns the finer harmonics from
# there. Nothing else zero-extends: a new hidden unit whose incoming row is zero
# sits at the ReLU's dead side and never earns a gradient, so those keep the
# trainer's init even though it perturbs what the source computed.
ZERO_EXTENDED = '.harmonics'


def overlaps(target_axis, source_axis, target_span, source_span):
    """(target slice, source slice, factor, zeroed tail) per segment both axes name.

    The lone unlabelled segment spans the axis. An aim block the source does not
    have falls back to its block 0: turning aim_ar on gives every button combo the
    unconditional head it is a refinement of, rather than sixteen fresh ones.
    """
    source, offset = {}, 0
    for label, size, scale in source_axis:
        size = source_span if size is None else size
        source[label] = (offset, size, scale)
        offset += size
    offset = 0
    for label, size, scale in target_axis:
        size = target_span if size is None else size
        found = source.get(label)
        if found is None and label.startswith('aim'):
            found = source.get('aim0' + label[label.index('.'):])
        if found is not None:
            start, source_size, source_scale = found
            width = min(size, source_size)
            tail = slice(offset + width, offset + size) if label.endswith(ZERO_EXTENDED) and size > width else None
            yield slice(offset, offset + width), slice(start, start + width), source_scale / scale, tail
        offset += size


def graft(name, source, target, source_arch, target_arch):
    """Copy every segment both architectures name. Returns the floats copied."""
    source_rows, source_cols = segments(name, source_arch)
    target_rows, target_cols = segments(name, target_arch)
    copied = 0
    for rows, source_rows_slice, row_factor, tail in overlaps(target_rows, source_rows, target.shape[0], source.shape[0]):
        for cols, source_cols_slice, col_factor, _ in overlaps(target_cols, source_cols, target.shape[1], source.shape[1]):
            block = source[source_rows_slice, source_cols_slice]
            factor = row_factor * col_factor
            target[rows, cols] = block if factor == 1.0 else block * np.float32(factor)
            copied += block.size
        if tail is not None:
            target[tail] = 0.0
    return copied


def initialise(arch, seed):
    """The flat buffer a fresh run of this architecture would have started from."""
    rng = np.random.default_rng(seed)
    flat = np.empty(layout.policy_v4_weights(arch['hidden'], arch['num_layers'],
                                             **{k: v for k, v in arch.items() if k not in ('hidden', 'num_layers')}),
                    dtype='<f4')
    for name, (offset, (rows, cols)) in tensors(arch).items():
        bound = gain(name) / math.sqrt(cols)
        flat[offset:offset + rows * cols] = rng.uniform(-bound, bound, rows * cols)
    return flat


def surgery(weights, source_arch, target_arch, seed=0):
    """(flat target weights, report rows).

    A report row is (name, source shape or None, target shape or None, action,
    floats copied, floats in the target tensor).
    """
    source_tensors, target_tensors = tensors(source_arch), tensors(target_arch)
    target = initialise(target_arch, seed)
    report = []
    for name, (offset, shape) in target_tensors.items():
        view = target[offset:offset + shape[0] * shape[1]].reshape(shape)
        if name not in source_tensors:
            report.append((name, None, shape, 'initialised', 0, view.size))
            continue
        source_offset, source_shape = source_tensors[name]
        source_view = weights[source_offset:source_offset + source_shape[0] * source_shape[1]].reshape(source_shape)
        copied = graft(name, source_view, view, source_arch, target_arch)
        action = 'copied' if copied == view.size == source_view.size else 'resized'
        report.append((name, source_shape, shape, action, copied, view.size))
    for name, (_, shape) in source_tensors.items():
        if name not in target_tensors:
            report.append((name, shape, None, 'dropped', 0, 0))
    return target, report


def widen(weights, source_arch, factor, noise=0.01, seed=0):
    """(flat target weights, report rows): hidden grown `factor`-fold by duplicating units.

    Net2WiderNet for the v4 trunk. Target unit j is a copy of source unit j % hidden:
    rows that write a hidden unit (fusion, every mingru gate, ffn_out) are tiled, and
    columns that read the trunk (mingru, ffn_in, actor, critic_in's trunk segment) are
    tiled and divided by `factor`, so a consumer sums the copies back to the original.
    A tiled vector has the original's mean and variance, so the layer norm after the
    fusion and the one before the FFN pass it through unchanged: the policy computes
    exactly what the source did. The critic does not: its layer norm spans the trunk
    and the privileged observation together, and tiling changes their proportions.

    Exact copies receive identical gradients forever (Muon's orthogonalised update
    keeps duplicated rows duplicated too), so the copies' incoming rows get Gaussian
    noise of `noise` times the tensor's std; that is the only departure from the
    source's policy.
    """
    if factor < 2:
        raise ValueError('widen needs a factor of at least 2')
    rng = np.random.default_rng(seed)
    hidden = source_arch['hidden']
    target_arch = dict(source_arch, hidden=hidden * factor)
    source_tensors, target_tensors = tensors(source_arch), tensors(target_arch)
    target = np.zeros(layout.policy_v4_weights(target_arch['hidden'], target_arch['num_layers'],
                                               **{k: v for k, v in target_arch.items() if k not in ('hidden', 'num_layers')}),
                      dtype='<f4')

    def tile_rows(w, blocks):
        """Duplicate each of `blocks` stacked hidden-sized row blocks; noise on the copies."""
        parts = []
        for block in np.split(w, blocks):
            copies = np.tile(block, (factor, 1))
            copies[hidden:] += rng.normal(0.0, noise * float(w.std()), copies[hidden:].shape).astype(w.dtype)
            parts.append(copies)
        return np.concatenate(parts)

    def tile_cols(w, span):
        return np.concatenate([np.tile(w[:, :span], (1, factor)) / np.float32(factor), w[:, span:]], 1)

    report = []
    for name, (offset, shape) in target_tensors.items():
        source_offset, source_shape = source_tensors[name]
        w = weights[source_offset:source_offset + source_shape[0] * source_shape[1]].reshape(source_shape)
        if name == 'fusion' or name == 'ffn_out':
            out = tile_rows(w, 1)
        elif name.startswith('mingru'):
            out = tile_rows(tile_cols(w, hidden), 3)
        elif name in ('ffn_in', 'actor', 'critic_in'):
            out = tile_cols(w, hidden)
        else:
            out = w
        assert out.shape == shape, (name, out.shape, shape)
        target[offset:offset + out.size] = out.ravel()
        report.append((name, source_shape, shape, 'copied' if out is w else 'duplicated', w.size, out.size))
    return target, target_arch, report


def widen_zero(weights, source_arch, factor, noise=0.01, seed=0):
    """(flat target weights, target arch, report rows): hidden grown `factor`-fold, new units silent.

    The fusion rows are tiled, so the layer norm after the fusion sees the same mean and
    variance and the source's units get exactly the input they had. Every source mingru
    row reads only the source's units (zero columns on the new ones), so the source's
    recurrent state evolves as before. Every reader of the trunk (actor, critic_in's
    trunk segment) starts with zero columns for the new units, so the actor computes
    exactly what the source did, whatever the new units hold.

    What they hold matters to the critic: its layer norm spans [trunk, privileged], so
    the new units enter its mean and variance. Fresh random units moved the value by
    half its spread (value RMSE 0.09 vs sd 0.18 on l2-main-136002m). So each new unit's
    mingru rows are a copy of its source unit's, reading the fusion copy, plus Gaussian
    noise of `noise` times the tensor's std over the whole row: the trunk the critic
    sees is the source's tiled, as in widen. The copies are not symmetric with their
    sources even without noise, since their outgoing columns differ (zero vs trained).
    An FFN's layer norm over the trunk would have the critic's problem, so ffn_dim > 0
    is refused.
    """
    if factor != 2:
        raise ValueError('zero widening adds one copy of the units; use factor 2')
    if source_arch['ffn_dim']:
        raise ValueError('zero widening cannot keep the FFN layer norm exact; use ffn_dim 0')
    rng = np.random.default_rng(seed)
    hidden = source_arch['hidden']
    wide = hidden * factor
    target_arch = dict(source_arch, hidden=wide)
    source_tensors, target_tensors = tensors(source_arch), tensors(target_arch)
    target = np.zeros(layout.policy_v4_weights(wide, target_arch['num_layers'],
                                               **{k: v for k, v in target_arch.items() if k not in ('hidden', 'num_layers')}),
                      dtype='<f4')

    def mingru(w):
        parts, zeros = [], np.zeros((hidden, hidden), w.dtype)
        for block in np.split(w, 3):
            copy = np.concatenate([zeros, block], 1)
            copy += rng.normal(0.0, noise * float(w.std()), copy.shape).astype(w.dtype)
            parts += [np.concatenate([block, zeros], 1), copy]
        return np.concatenate(parts)

    def zero_cols(w):
        return np.concatenate([w[:, :hidden], np.zeros((w.shape[0], wide - hidden), w.dtype), w[:, hidden:]], 1)

    report = []
    for name, (offset, shape) in target_tensors.items():
        source_offset, source_shape = source_tensors[name]
        w = weights[source_offset:source_offset + source_shape[0] * source_shape[1]].reshape(source_shape)
        if name == 'fusion':
            out, action = np.tile(w, (factor, 1)), 'tiled'
        elif name.startswith('mingru'):
            out, action = mingru(w), 'grown'
        elif name in ('actor', 'critic_in'):
            out, action = zero_cols(w), 'zero-cols'
        else:
            out, action = w, 'copied'
        assert out.shape == shape, (name, out.shape, shape)
        target[offset:offset + out.size] = out.ravel()
        report.append((name, source_shape, shape, action, w.size, out.size))
    return target, target_arch, report


def read_checkpoint(path):
    """(architecture, metadata, flat weights) of a published v4 checkpoint."""
    path = Path(path)
    metadata = json.loads(Path(str(path) + '.json').read_text())
    if metadata.get('architecture') != layout.V4_ARCHITECTURE:
        raise ValueError(f"Surgery reads {layout.V4_ARCHITECTURE} checkpoints, not {metadata.get('architecture')!r}")
    if metadata.get('obs_size') != layout.V4_OBS_SIZE:
        raise ValueError(f"Checkpoint reads {metadata.get('obs_size')} observation floats, layout.py now writes {layout.V4_OBS_SIZE}")
    weights = np.fromfile(path, dtype='<f4')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('weights_sha256'):
        raise ValueError(f'Checkpoint weights differ from the sidecar hash: {path}')
    arch = architecture(metadata['hidden'], metadata['num_layers'], metadata['aim_fourier'],
                        metadata['aim_ar'], metadata.get('v4_shape'))
    expected = layout.policy_v4_weights(arch['hidden'], arch['num_layers'],
                                        **{k: v for k, v in arch.items() if k not in ('hidden', 'num_layers')})
    if weights.size != expected:
        raise ValueError(f'Checkpoint holds {weights.size} floats, its architecture implies {expected}')
    return arch, metadata, weights


def format_report(report, source_arch, target_arch, source):
    changed = {k: (source_arch[k], target_arch[k]) for k in ARCH_KEYS if source_arch[k] != target_arch[k]}
    lines = [f'surgery {source}',
             '  ' + (', '.join(f'{k} {was} -> {now}' for k, (was, now) in changed.items()) or 'identical architecture'),
             '',
             f"{'tensor':<12} {'source':>14} {'target':>14}  {'action':<12} {'floats kept':>21}"]
    shape = lambda s: '-' if s is None else f'{s[0]}x{s[1]}'
    kept = total = 0
    for name, was, now, action, copied, size in report:
        kept, total = kept + copied, total + size
        share = '' if not size else f' ({100.0 * copied / size:5.1f}%)'
        lines.append(f'{name:<12} {shape(was):>14} {shape(now):>14}  {action:<12} {copied:>12,}{share}')
    lines += ['', f"{'total':<12} {'':>14} {'':>14}  {'':<12} {kept:>12,}"
                  f"{'' if not total else f' ({100.0 * kept / total:5.1f}%)'} of {total:,}"]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Carry a v4 checkpoint into a changed v4 architecture.')
    parser.add_argument('source', help='an existing .bin with its .bin.json sidecar')
    parser.add_argument('target', help='the .bin to write; it must not exist')
    for key in ARCH_KEYS:
        parser.add_argument('--' + key.replace('_', '-'), type=int, help=f'target {key} (default: the source\'s)')
    parser.add_argument('--seed', type=int, default=0, help='seed for the weights that have to be initialised')
    parser.add_argument('--widen', type=int, help='instead: grow hidden this many times by duplicating units, '
                                                  'keeping the policy (see widen); no other change allowed')
    parser.add_argument('--widen-noise', type=float, default=0.01, help='symmetry-breaking noise on the copies, x tensor std')
    parser.add_argument('--widen-init', choices=('copy', 'zero'), default='copy',
                        help='copy: duplicate units (Net2WiderNet, see widen); zero: fresh units with zero '
                             'outgoing weights, the actor exactly preserved (see widen_zero)')
    args = parser.parse_args(argv)

    source_arch, metadata, weights = read_checkpoint(args.source)
    if args.widen:
        if any(getattr(args, k) is not None for k in ARCH_KEYS):
            parser.error('--widen sets hidden itself and changes nothing else')
        if args.widen_init == 'zero':
            target, target_arch, report = widen_zero(weights, source_arch, args.widen, args.widen_noise, args.seed)
        else:
            target, target_arch, report = widen(weights, source_arch, args.widen, args.widen_noise, args.seed)
    else:
        target_arch = dict(source_arch, **{k: v for k, v in vars(args).items() if k in source_arch and v is not None})
        target, report = surgery(weights, source_arch, target_arch, args.seed)
    print(format_report(report, source_arch, target_arch, args.source))

    kept = sum(row[4] for row in report)
    config = dict(metadata['config'])
    config.update({k: v for k, v in (('hidden', target_arch['hidden']), ('layers', target_arch['num_layers']),
                                     ('aim_fourier', target_arch['aim_fourier']), ('aim_ar', target_arch['aim_ar']),
                                     ('v4_shape', v4_shape(target_arch))) if k in config})
    config['surgery'] = dict(source=str(Path(args.source).resolve()), source_sha256=metadata['weights_sha256'],
                             changed={k: [source_arch[k], target_arch[k]] for k in ARCH_KEYS
                                      if source_arch[k] != target_arch[k]},
                             kept_floats=kept, total_floats=int(target.size), seed=args.seed)
    if args.widen:
        config['surgery'].update(widen=args.widen, widen_init=args.widen_init, widen_noise=args.widen_noise)
    written = write_checkpoint(args.target, target.tofile, obs_size=layout.V4_OBS_SIZE,
                               hidden=target_arch['hidden'], num_layers=target_arch['num_layers'],
                               architecture=layout.V4_ARCHITECTURE, game_steps=metadata['step'],
                               learner_steps=metadata['learner_steps'], config=config,
                               aim_fourier=target_arch['aim_fourier'], aim_ar=target_arch['aim_ar'],
                               v4_shape=v4_shape(target_arch))
    print(f"\nwrote {args.target}  {written['weights_sha256']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
