"""Policy diagnostics captured from the observation used to choose an action."""
import gzip
import json

import torch

import layout


def decision(tick, logits, value, action, mask=None):
    """One policy row. logits may carry -inf where the environment's action mask
    (1 = allowed) disallowed a choice; mask marks a latched decision."""
    if tuple(logits.shape) != (layout.LOGITS,) or tuple(action.shape) != (layout.NUM_ATNS,):
        raise ValueError(f'Expected one Bopl policy row and its {layout.NUM_ATNS} action heads')
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(value).all():
        raise ValueError('Nonfinite evaluation policy output')
    aim, buttons, hold = [part.softmax(-1) for part in logits.split(layout.ACTION_SIZES)]
    if not (torch.isfinite(aim).all() and torch.isfinite(buttons).all() and torch.isfinite(hold).all()):
        raise ValueError('An action head has no allowed choice')
    latched = mask is not None and not all(int(m) for m in mask[layout.ACT_AIM:])
    return dict(t=int(tick), value=float(value), buttons=buttons.tolist(),
                aim=aim.tolist(), hold=hold.tolist(), action=action.tolist(), latched=latched)


def write(path, step, side, decisions, checkpoint_identity, observation_delay_ticks=9):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(dict(schema=1, step=step, side=side, decision_ticks=4,
                       aim_bins=layout.ACT_AIM, hold_lengths=list(layout.HOLD_LENGTHS),
                       action_heads=['aim', 'buttons', 'hold'], checkpoint_identity=checkpoint_identity,
                       tick_semantics='action-decision time', observation_delay_ticks=observation_delay_ticks,
                       own_previous_actions='immediate', aim_zero='no joystick direction',
                       hold='decisions the sampled button state stays latched (aim is never latched)',
                       latched='button and hold heads were masked to the latched state for this decision',
                       decisions=decisions), stream, allow_nan=False, separators=(',', ':'))
    temporary.replace(path)
