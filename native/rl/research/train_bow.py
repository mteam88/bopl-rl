"""Bounded Bow curriculum using the full-game Policy and shared recurrent PPO.

The learner controls one player; the other sends neutral inputs. The first
curriculum exposes aim and Bow held/released, without jump/dash/platform.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.distributions import Categorical
from bow_task import Task, Trial
from environment import configure, ROOT, OBS_SIZE
from policy import Policy
from ppo import update
from train import advantages, atomic_json


def bow_distribution(logits, actions=None, timing=False):
    aim = Categorical(logits=logits[..., :256], validate_args=False)
    button = Categorical(logits=logits[..., [256, 260]], validate_args=False)
    if actions is None:
        angles = aim.sample()
        buttons = torch.zeros_like(angles) if timing else 4 * button.sample()
        actions = torch.stack([angles, buttons], -1)
    logp = aim.log_prob(actions[..., 0])
    entropy = aim.entropy()
    if not timing:
        logp = logp + button.log_prob(actions[..., 1] // 4)
        entropy = entropy + button.entropy()
    return actions, logp, entropy


def evaluate(policy, trial, seed, count=32, timing=False, output=None):
    """Sampling seeds measure action randomness; fixed starts are not new layouts."""
    device = next(policy.parameters()).device
    results = []
    saved_success = False
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        for mode in ('sample', 'argmax'):
            for i in range(count if mode == 'sample' else 1):
                torch.manual_seed(seed + i)
                obs = trial.reset(seed + i)
                memory = policy.initial(1, device)
                while not trial.done:
                    with torch.no_grad():
                        logits, _, memory = policy.step(torch.as_tensor(obs[None], device=device), memory)
                        if mode == 'sample':
                            chosen = bow_distribution(logits, timing=timing)[0][0].cpu().numpy()
                        else:
                            chosen = (int(logits[0, :256].argmax()), 0 if timing else 4 * int(logits[0, [256, 260]].argmax()))
                    obs, _, _, result, _ = trial.step(chosen)
                result.update(mode=mode, action_seed=seed + i)
                results.append(result)
                if output is not None and (i == 0 or mode == 'argmax'):
                    trial.replay(output / f'{mode}-{i}.rep')
                if output is not None and result['success'] and not saved_success:
                    trial.replay(output / 'first-hit.rep')
                    saved_success = True
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map', type=int, default=0)
    parser.add_argument('--side', type=int, choices=(0, 1), default=0)
    parser.add_argument('--limit', type=int, default=240)
    parser.add_argument('--assistance', choices=('none', 'timing'), default='none')
    parser.add_argument('--hold-ticks', type=int, default=64)
    parser.add_argument('--start-move-ticks', type=int, default=0)
    parser.add_argument('--opponent', choices=('idle', 'approach'), default='idle')
    parser.add_argument('--steps', type=int, default=262144)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--entropy', type=float, default=.01)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--load', type=Path, help='Weights only; fresh optimizer and game trials')
    parser.add_argument('--load-optimizer', action='store_true', help='Also load Adam state; game trials and recurrent memory still restart')
    parser.add_argument('--hold-prior', type=float, help='Initialize Bow head bias to this hold probability; no fixed action durations')
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1 or args.entropy < 0 or args.lr <= 0:
        parser.error('Positive steps/workers/learning rate and nonnegative entropy required')
    if args.load_optimizer and not args.load:
        parser.error('--load-optimizer requires --load')
    if args.hold_prior is not None and not 0 < args.hold_prior < 1:
        parser.error('Hold prior must be between zero and one')
    task = Task(map_id=args.map, side=args.side, limit=args.limit,
                assistance=args.assistance, hold_ticks=args.hold_ticks,
                start_move_ticks=args.start_move_ticks, opponent=args.opponent)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    configure(out / 'workers')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda'
    policy = Policy(OBS_SIZE).to(device)
    if args.load:
        prior = torch.load(args.load, map_location=device, weights_only=False)
        if prior.get('input_mode', 'full') != 'full' or prior.get('action_mode', 'absolute') != 'absolute':
            raise ValueError('Bow warm starts require full inputs and absolute action logits')
        policy.load_state_dict(prior['policy'])
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    if args.load_optimizer:
        optimizer.load_state_dict(prior['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
    if args.hold_prior is not None:
        with torch.no_grad():
            policy.actor.bias[256] = np.log(1 - args.hold_prior)
            policy.actor.bias[260] = np.log(args.hold_prior)
    config = dict(task=asdict(task), training_seed=args.seed, steps=args.steps, workers=args.workers,
                  entropy=args.entropy, action_scope='256 stick codes and Bow held/released',
                  action_repeat=4, observation_delay=9, obs_size=OBS_SIZE, hidden=128,
                  horizon=128, sequence=32, epochs=3, minibatch_segments=32,
                  gamma=.995, gae_lambda=.95, learning_rate=args.lr,
                  reward=dict(hit=1, death=-.25, other=0), termination='first arrow hit or either death or time limit',
                  warm_start=str(args.load.resolve()) if args.load else None,
                  load_optimizer=args.load_optimizer, hold_prior=args.hold_prior,
                  warm_start_sha256=hashlib.sha256(args.load.read_bytes()).hexdigest() if args.load else None,
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(Path(__file__).parent.glob('*.py'))},
                  assembly_sha256=hashlib.sha256((ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll').read_bytes()).hexdigest(),
                  research_library_sha256=hashlib.sha256((ROOT/'native/artifacts/rl/libbopl_research.so').read_bytes()).hexdigest())
    for relative in config['sources']:
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    atomic_json(out / 'config.json', config)
    timing = args.assistance == 'timing'
    def action_distribution(logits, actions=None):
        return bow_distribution(logits, actions, timing)
    def checkpoint(step):
        path = out / f'policy-{step:09d}.pt'
        torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                        obs_size=OBS_SIZE, hidden=128, config=config, task='bow', step=step), path)
        return path
    trials = []
    evaluator = None
    started = time.monotonic()
    step = 0
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            trials = list(pool.map(lambda _: Trial(task), range(args.workers)))
            evaluator = Trial(task)
            observations = np.stack(list(pool.map(lambda pair: pair[1].reset(args.seed + pair[0]), enumerate(trials))))
            memory = policy.initial(args.workers, device)
            reset_before = torch.ones(args.workers, dtype=torch.bool, device=device)
            checkpoint(0)
            initial = evaluate(policy, evaluator, 100001, timing=timing)
            atomic_json(out / 'initial-evaluation.json', initial)
            episodes = []
            with (out / 'metrics.jsonl').open('w', buffering=1) as metrics, (out / 'episodes.jsonl').open('w', buffering=1) as episode_log:
                while step < args.steps:
                    rollout_started = time.monotonic()
                    n, tmax, length = args.workers, 128, 32
                    obs = torch.empty(tmax, n, OBS_SIZE, device=device)
                    actions = torch.empty(tmax, n, 2, dtype=torch.long, device=device)
                    old_logp, values, rewards, dones = [torch.empty(tmax, n, device=device) for _ in range(4)]
                    resets = torch.empty(tmax, n, dtype=torch.bool, device=device)
                    memories = torch.empty(tmax // length, 1, n, 128, device=device)
                    completed = []
                    for t in range(tmax):
                        obs[t].copy_(torch.as_tensor(observations, device=device))
                        resets[t] = reset_before
                        if t % length == 0:
                            memories[t // length].copy_(memory)
                        with torch.no_grad():
                            logits, value, memory = policy.step(obs[t], memory)
                            chosen, logp, _ = action_distribution(logits)
                        actions[t], old_logp[t], values[t] = chosen, logp, value
                        results = list(pool.map(lambda pair: pair[0].step(pair[1]), zip(trials, chosen.cpu().numpy())))
                        rewards[t] = torch.tensor([r[1] for r in results], device=device)
                        dones[t] = torch.tensor([r[2] for r in results], device=device)
                        reset_before = dones[t].bool()
                        memory *= (~reset_before).view(1, -1, 1)
                        for i, result in enumerate(results):
                            observations[i] = result[0]
                            if result[2]:
                                record = dict(step=step + t * n + i + 1, worker=i, **result[3])
                                completed.append(record)
                                episode_log.write(json.dumps(record) + '\n')
                        ended = [i for i, r in enumerate(results) if r[2]]
                        fresh = list(pool.map(lambda i: trials[i].reset(args.seed + step + t * n + i + 1), ended))
                        for i, observation in zip(ended, fresh):
                            observations[i] = observation
                    with torch.no_grad():
                        _, last_value, _ = policy.step(torch.as_tensor(observations, device=device), memory)
                        adv = advantages(rewards, values, dones, last_value)
                        returns = adv + values
                    if not torch.isfinite(obs).all() or not torch.isfinite(adv).all():
                        raise RuntimeError('Nonfinite rollout')
                    rollout_seconds = time.monotonic() - rollout_started
                    losses = update(policy, optimizer, obs, actions, old_logp, values, returns, adv,
                                    torch.ones_like(resets), resets, memories, length, 3, 32,
                                    entropy_coefficient=args.entropy, action_distribution=action_distribution)
                    step += tmax * n
                    episodes.extend(completed)
                    recent = episodes[-256:]
                    row = dict(step=step, elapsed=time.monotonic() - started, episodes=len(episodes),
                               rollout_seconds=rollout_seconds, update_seconds=time.monotonic()-rollout_started-rollout_seconds,
                               recent_trials=len(recent), success_rate=sum(r['success'] for r in recent)/max(1,len(recent)),
                               total_hits=sum(r['hits'] for r in episodes),
                               losses=np.mean(losses, axis=0).tolist())
                    metrics.write(json.dumps(row, allow_nan=False) + '\n')
                    if step % (tmax * n * 16) == 0:
                        print(json.dumps(row), flush=True)
                        checkpoint(step)
                final_path = checkpoint(step)
                media = out / 'evaluation'
                media.mkdir()
                final = evaluate(policy, evaluator, 200001, count=128, timing=timing, output=media)
                atomic_json(out / 'final-evaluation.json', final)
                atomic_json(out / 'result.json', dict(status='complete', step=step, episodes=len(episodes),
                            elapsed=time.monotonic()-started, checkpoint=str(final_path),
                            initial_successes=sum(r['success'] for r in initial if r['mode']=='sample'),
                            final_successes=sum(r['success'] for r in final if r['mode']=='sample'),
                            initial_trials=32, final_trials=128,
                            argmax_success=final[-1]['success']))
    finally:
        for trial in trials:
            trial.close()
        if evaluator is not None:
            evaluator.close()


if __name__ == '__main__':
    main()
