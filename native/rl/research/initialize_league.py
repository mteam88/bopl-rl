"""Lift a learned Bow policy into the unrestricted DAP button distribution.

Preserve its Bow marginal exactly; give the three previously excluded buttons
independent, nonzero initial probabilities. Every output remains trainable.
This is initialization, not an inference-time mask or scripted action.
"""
import argparse
import hashlib
import math
from pathlib import Path
import torch
from policy import load


def expand(policy, extra_probability):
    if not 0 < extra_probability < .5:
        raise ValueError('Extra-button probability must be in (0, .5)')
    with torch.no_grad():
        weight = policy.actor.weight.clone()
        bias = policy.actor.bias.clone()
        for buttons in range(16):
            source = 256 + (buttons & 4)
            extras = sum(bool(buttons & bit) for bit in (1, 2, 8))
            prior = extras * math.log(extra_probability) + (3-extras) * math.log1p(-extra_probability)
            policy.actor.weight[256+buttons].copy_(weight[source])
            policy.actor.bias[256+buttons].copy_(bias[source] + prior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra-probability', type=float, default=.001)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists')
    model, source = load(args.source)
    if model.action_mode != 'absolute' or model.input_mode != 'full':
        raise ValueError('Expected the absolute-action, full-input Bow teacher')
    expand(model, args.extra_probability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=1, policy=model.state_dict(), obs_size=model.obs_size,
                    hidden=model.hidden, task='full-game', action_mode='absolute', input_mode='full',
                    initialization=dict(source=str(args.source.resolve()),
                                        sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                                        extra_button_probability=args.extra_probability,
                                        source_task=source.get('task', 'imitation'))), args.output)


if __name__ == '__main__':
    main()
