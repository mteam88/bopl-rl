"""Policy diagnostics captured from the observation used to choose an action."""
import gzip
import json

import torch


def decision(tick, logits, value, action):
    if tuple(logits.shape) != (272,) or tuple(action.shape) != (2,):
        raise ValueError('Expected one Bopl policy row and its two action heads')
    if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
        raise ValueError('Nonfinite evaluation policy output')
    aim, buttons = [part.softmax(-1) for part in logits.split([256, 16])]
    return dict(t=int(tick), value=float(value), buttons=buttons.tolist(),
                aim=aim.tolist(), action=action.tolist())


def write(path, step, side, decisions, checkpoint_identity, observation_delay_ticks=9):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(dict(schema=1, step=step, side=side, decision_ticks=4,
                       aim_bins=256, checkpoint_identity=checkpoint_identity,
                       tick_semantics='action-decision time', observation_delay_ticks=observation_delay_ticks,
                       own_previous_actions='immediate', aim_zero='no joystick direction',
                       decisions=decisions), stream, allow_nan=False, separators=(',', ':'))
    temporary.replace(path)
