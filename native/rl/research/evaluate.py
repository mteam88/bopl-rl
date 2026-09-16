"""Fixed-seed, side-swapped evaluation with downloadable native and visual replays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from environment import Environment,configure
from native_policy import load,masked,sample,decide
from common import atomic_json
import decision_dump
import layout
from telemetry import FIELDS,summarize
from evaluation_protocol import checkpoint_identity,map_ids,opponents,schedule,scripted_names,scripted_opponent,DEFAULT_EVAL_SCRIPTED

def random_action(rng):
    """Scripted opponent without a hold output: hold index 0 (length 1) keeps every decision free."""
    return torch.tensor([rng.integers(layout.ACT_AIM),rng.integers(layout.ACT_BUTTONS),0])

def wilson(wins,n):
    if not n: return [0,1]
    z=1.96; p=wins/n; denom=1+z*z/n
    middle=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0,middle-half),min(1,middle+half)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--games-per-opponent',type=int,default=8)
    parser.add_argument('--heldout-games',type=int,default=4)
    parser.add_argument('--evaluation-maps',type=map_ids,default=[0])
    parser.add_argument('--heldout-maps',type=map_ids,default=[1,2])
    parser.add_argument('--opponents',type=Path,help='Manifest of frozen trained opponents and SHA256 identities')
    parser.add_argument('--frozen-games',type=int,default=16)
    parser.add_argument('--scripted-opponents',type=scripted_names,default=DEFAULT_EVAL_SCRIPTED,
                        help='Comma-separated scripted bot names (kind-level, e.g. mixed-hard,archer-medium); empty disables')
    parser.add_argument('--pool',type=Path,default=None,
                        help='Reference pool.json for in-run Elo (default: reference_pool.DEFAULT_POOL, pool v7, if present)')
    parser.add_argument('--seed-base',type=int,default=10001)
    parser.add_argument('--heldout-seed-base',type=int,default=20001)
    parser.add_argument('--selfplay-seed',type=int,default=30001)
    parser.add_argument('--suite',choices=('diagnostic','final'),default='diagnostic')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--game',type=Path,help='Unity player (otherwise BOPL_RL_GAME or LinuxIL2CPP)')
    args=parser.parse_args()
    if min(args.games_per_opponent,args.heldout_games,args.workers)<1:
        parser.error('Positive game and worker counts required')
    if args.games_per_opponent%2 or args.heldout_games%2:
        parser.error('An even number of games is required for side-swapped seed pairs')
    torch.set_num_threads(1)
    checkpoint_id = checkpoint_identity(args.checkpoint)
    baseline_id = checkpoint_identity(args.initial)
    policy,checkpoint=load(args.checkpoint)
    initial,_=load(args.initial)
    frozen=opponents(args.opponents)
    models={'initial':initial}
    for item in frozen:
        models[item['name']]=load(Path(item['path']))[0]
    # Every participant reads its own observation layout (v3 or v4, layout.py); the
    # environment writes each side's row in its checkpoint's layout below.
    step=checkpoint['step']; config=checkpoint['config']
    observation_delay=config.get('observation_delay_ticks',9)
    if set(args.heldout_maps)&set(config['training_maps']):
        raise ValueError('A held-out map was used for training')
    tasks,group_keys=schedule(args.evaluation_maps,args.heldout_maps,frozen,args.games_per_opponent,
                              args.heldout_games,args.frozen_games,args.seed_base,args.heldout_seed_base,args.selfplay_seed,
                              scripted=args.scripted_opponents)
    out=args.run/('evaluations' if args.suite=='diagnostic' else 'final-evaluations')/f'{step:09d}'
    if (out/'result.json').exists():
        previous=json.loads((out/'result.json').read_text())
        expected={(t['opponent'],t['map'],t['seed'],t['side']) for t in tasks}
        actual={(t['opponent'],t['map'],t['seed'],t['side']) for t in previous['games']}
        if (previous.get('schema') != 2 or expected!=actual or len(previous['games'])!=len(tasks)
            or previous.get('checkpoint_identity') != checkpoint_id
            or previous.get('baseline_identity') != baseline_id
            or previous.get('frozen_opponents') != frozen
            or previous.get('decision_dump_schema') != 1):
            raise ValueError('Existing evaluation differs from the requested checkpoint, opponents or schedule')
        print(f'Evaluation {step} already complete',flush=True)
        emit_elo(args, previous)
        return
    out.mkdir(parents=True,exist_ok=True)
    configure(out/'workers', args.game)
    started=time.monotonic()
    # Each thread owns its game and its sampling generator; the model weights
    # are immutable. Reuse each game process across tasks to avoid startup cost.
    buckets=[tasks[i::args.workers] for i in range(min(args.workers,len(tasks)))]
    def rows_for(model,rows,layouts):
        # A model reads both rows at its own width; a row the environment wrote in another
        # checkpoint's layout (v3 versus v4, layout.py) is zeroed: its actions are never used.
        batch=rows[:,:model.obs_size].copy()
        mine=layout.layout_of(model.architecture)
        for p in range(2):
            if layouts[p]!=mine: batch[p]=0
        return torch.from_numpy(batch)
    def work(bucket):
        results=[]
        with Environment(seed=90001+tasks.index(bucket[0]),limit=config['episode_ticks'],delay=observation_delay, reward=config.get('reward')) as env:
            for task in bucket:
                env.reset(task['seed'],task['map'])
                overflow_before=env.observation_overflow
                state=policy.initial(2,'cpu')
                opponent=models.get(task['opponent'])
                other_state=opponent.initial(2,'cpu') if opponent is not None else None
                # Scripted bots decide inside the C environment from the true
                # game state; the caller's action floats for them are ignored.
                scripted=scripted_opponent(task['opponent'])
                env.bot(0,0); env.bot(1,0)
                if scripted is not None: env.bot(1-task['side'],*scripted)
                # Each side's row in its checkpoint's observation layout (a scripted or random
                # opponent's row stays v3).
                env.set_layout(task['side'],layout.layout_of(policy.architecture))
                env.set_layout(1-task['side'],layout.layout_of(opponent.architecture) if opponent is not None else 3)
                layouts=[env.layout_of(0),env.layout_of(1)]
                generator=torch.Generator().manual_seed(task['seed']*7+task['side'])
                rng=np.random.default_rng(task['seed']*11+task['side'])
                frames=[env.frame()] if task['save'] else []
                mind=[]
                button_counts=np.zeros(4,dtype=np.int64); hold_counts=np.zeros(layout.ACT_HOLD,dtype=np.int64)
                latched_decisions=0; decisions=0; events=[]
                while not env.ended:
                    rows=env.rows.copy()
                    mask=env.mask.copy()
                    with torch.no_grad():
                        # native_policy.decide samples as the trainer does (buttons before aim when the aim head
                        # is autoregressive); the logits are those the actions were sampled from.
                        logits,values,actions,state=decide(policy,rows_for(policy,rows,layouts),mask,state,generator)
                        other=1-task['side']
                        if opponent is not None:
                            _,_,old_actions,other_state=decide(opponent,rows_for(opponent,rows,layouts),mask,other_state,generator)
                            actions[other]=old_actions[other]
                        elif task['opponent']=='random':
                            actions[other]=random_action(rng)
                    if task['save']:
                        side=task['side']
                        mind.append(decision_dump.decision(env.state.header.episode_ticks,
                                    logits[side],values[side],actions[side],mask[side]))
                    env.step(actions.numpy())
                    events.append(env.telemetry[task['side']].copy())
                    buttons=int(actions[task['side'],1])  # masked to the latched state while a hold is in force
                    for i in range(4): button_counts[i]+=bool(buttons&(1<<i))
                    if mask[task['side'],layout.ACT_AIM+layout.ACT_BUTTONS:].all(): hold_counts[int(actions[task['side'],2])]+=1  # free decision (the hold head is unmasked; the button head may be arrow-release masked)
                    else: latched_decisions+=1
                    decisions+=1
                    if task['save']:
                        frames[-1]['actions']=actions.tolist()
                        frames.append(env.frame())
                final=env.frame(); me=final['players'][task['side']]
                timeout=bool(final['flags']&2); win=bool(me['flags']&2) and not timeout
                result={**task,'win':win,'timeout':timeout,'kills':me['kills'],'deaths':me['deaths'],
                        'observation_overflow':env.observation_overflow-overflow_before,
                        'ticks':final['t'],'final_players':final['players'],'behavior':summarize(np.asarray(events)),
                        'action_rates':dict(zip(['jump','dash','arrow','platform'],(button_counts/max(1,decisions)).tolist())),
                        # Hold head: share of free (unlatched) decisions choosing each hold length, as the trainer's hold_* log.
                        'hold_rates':dict(zip([str(n) for n in layout.HOLD_LENGTHS],(hold_counts/max(1,hold_counts.sum())).tolist())),
                        'latched_fraction':latched_decisions/max(1,decisions)}
                if task['save']:
                    replay=out/(task['name']+'.rep'); env.replay(replay)
                    visual=out/(task['name']+'.json')
                    dump=out/(task['name']+'.mind.json.gz')
                    decision_dump.write(dump,step,task['side'],mind,checkpoint_id,observation_delay)
                    atomic_json(visual,dict(schema=1,step=step,task=task,result=result,frames=frames,
                                           rendering='physics state replay',tick_rate=60,decision_rate=15,
                                           actions_timing='observation-before-action'))
                    result.update(replay=replay.name,visual=visual.name,mind=dump.name,
                                  replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
                results.append(result)
                print(json.dumps(dict(step=step,**result)),flush=True)
        return results
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        results=[result for bucket in workers.map(work,buckets) for result in bucket]
    groups=[]
    for opponent,map_id in group_keys:
        selected=[r for r in results if r['opponent']==opponent and r['map']==map_id]
        wins=sum(r['win'] for r in selected); n=len(selected)
        groups.append(dict(opponent=opponent,map=map_id,games=n,wins=wins,win_rate=wins/n,
                           map_role='trained' if map_id in config['training_maps'] else 'untrained',
                           interval=wilson(wins,n),timeouts=sum(r['timeout'] for r in selected),
                           seed_pairs=n//2,
                           behavior_totals={key:sum(r['behavior']['totals'][key] for r in selected) for key in FIELDS},
                           mean_kills=float(np.mean([r['kills'] for r in selected])),
                           mean_deaths=float(np.mean([r['deaths'] for r in selected])),
                           mean_duration=float(np.mean([r['ticks']/60 for r in selected]))))
    if (checkpoint_identity(args.checkpoint) != checkpoint_id
        or checkpoint_identity(args.initial) != baseline_id or opponents(args.opponents) != frozen):
        raise ValueError('Checkpoint or frozen opponents changed during evaluation')
    report=dict(schema=2,decision_dump_schema=1,step=step,suite=args.suite,checkpoint=args.checkpoint.name,
                checkpoint_identity=checkpoint_id, baseline_identity=baseline_id,
                frozen_opponents=frozen,scripted_opponents=args.scripted_opponents,
                evaluation_maps=args.evaluation_maps,heldout_maps=args.heldout_maps,
                training_maps=config['training_maps'],
                seed_base=args.seed_base,heldout_seed_base=args.heldout_seed_base,selfplay_seed=args.selfplay_seed,
                interval_note='Descriptive Wilson interval; does not adjust for correlation within seed pairs',
                elapsed=time.monotonic()-started,groups=groups,games=results)
    atomic_json(out/'result.json',report)
    emit_elo(args, report)
    print(json.dumps(report),flush=True)


def emit_elo(args, report):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sweep'))
    import reference_pool
    pool_path = args.pool or (reference_pool.DEFAULT_POOL if reference_pool.DEFAULT_POOL.is_file() else None)
    if not pool_path:
        return
    pool = reference_pool.load_pool(pool_path)
    rating = reference_pool.write_evaluation_rating(
        args.run, report, pool, args.checkpoint, game=args.game)
    if rating:
        print(json.dumps(dict(elo=rating['rating'], interval=rating['interval'],
                              games=rating['games'], source=rating.get('source'),
                              pool_sha8=reference_pool.pool_sha8(pool))), flush=True)


if __name__=='__main__': main()
