"""Playable agents and Bradley-Terry fits for the frozen reference pool.

Model / Random / Scripted decide in games; bradley_terry fits a round robin
(used when building a pool); bradley_terry_fixed rates one candidate against
members whose ratings stay put. Timeouts count as draws; random is anchored
at 0 Elo.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import numpy as np
from evaluation_protocol import scripted_opponent
import layout
# environment.py loads the research library on import; it is imported where games
# are played so the Bradley-Terry fits and participants stay importable without it
# (sweep/reference_pool.py and its tests).

HEAD_BOUNDS=[sum(layout.ACTION_SIZES[:i]) for i in range(layout.NUM_ATNS+1)]  # logits row offsets per head
ELO_SCALE=400/math.log(10)


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


class Model:
    """A native checkpoint participant (research/native_policy.py contract)."""
    def __init__(self,name,path,kind,step=None,library=None):
        from native_policy import load
        self.name=name; self.kind=kind; self.step=step; self.path=Path(path)
        self.policy,self.meta=load(self.path,**(dict(library=library) if library else {}))
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()

    def start(self,task,side,rng):
        import torch
        state=self.policy.initial(2,'cpu')
        policy=self.policy
        def head(row,index):
            part=row[HEAD_BOUNDS[index]:HEAD_BOUNDS[index+1]]
            weights=np.exp(part-part.max()); weights/=weights.sum()
            return rng.choice(len(part),p=weights)
        def act(observation,mask):
            nonlocal state
            with torch.no_grad():
                batch=torch.from_numpy(np.ascontiguousarray(observation,dtype=np.float32))
                rows,state=policy.step_raw(batch,state)
                logits,_=policy.game_logits(rows)
            # Sample as the trainer does: the environment's action mask (1 = allowed) zeroes latched choices;
            # an autoregressive aim head is materialised for the sampled button combo before aim is drawn.
            allowed=np.asarray(mask[side])!=0
            row=np.where(allowed,as_numpy(logits)[side],-np.inf)
            action=[0,head(row,1),0]
            if policy.aim_ar:
                combos=np.zeros(rows.shape[0],dtype=np.int32); combos[side]=action[1]
                with torch.no_grad(): logits,_=policy.game_logits(rows,combos)
                row=np.where(allowed,as_numpy(logits)[side],-np.inf)
            action[0]=head(row,0); action[2]=head(row,2)
            return np.asarray(action,dtype=np.int64)
        return act


class Random:
    name='random'; kind='anchor'; step=None; sha256=None

    def start(self,task,side,rng):
        # No hold output: hold index 0 (length 1) leaves every decision's buttons free.
        return lambda observation,mask: np.asarray([rng.integers(layout.ACT_AIM),rng.integers(layout.ACT_BUTTONS),0],dtype=np.int64)


class Scripted:
    """A hand-coded bot anchor (bopl_bots.h) deciding inside the C environment."""
    kind='anchor'; step=None; sha256=None

    def __init__(self,name):
        self.name=name; self.bot=scripted_opponent(name)

    def attach(self,env,side):
        env.bot(side,*self.bot)

    def start(self,task,side,rng):
        # The environment ignores this player's action floats while the bot is pinned.
        return lambda observation,mask: np.zeros(layout.NUM_ATNS,dtype=np.int64)


def bradley_terry(names,records,anchor):
    """MLE ratings in Elo units with the anchor fixed at 0.

    records: {(a,b): [wins_a, wins_b, draws]} with a,b participant names.
    One virtual draw per played pair regularizes undefeated participants.
    """
    index={name:i for i,name in enumerate(names)}
    ratings=np.zeros(len(names))
    pairs=[(index[a],index[b],wa+d/2+.5,wb+d/2+.5) for (a,b),(wa,wb,d) in records.items()]
    # Gradient ascent on the log-likelihood. Its curvature per participant is at most
    # games/4, so a step of 2/max_games converges from any start (a step scaled by the
    # pair count diverged for pools of a few members with many games per pair).
    games=np.zeros(len(names))
    for a,b,score_a,score_b in pairs:
        games[a]+=score_a+score_b; games[b]+=score_a+score_b
    step=2/max(1,games.max())
    for _ in range(5000):
        gradient=np.zeros_like(ratings)
        for a,b,score_a,score_b in pairs:
            expected=1/(1+math.exp(ratings[b]-ratings[a]))
            gradient[a]+=score_a-(score_a+score_b)*expected
            gradient[b]+=score_b-(score_b+score_a)*(1-expected)
        ratings+=gradient*step
        ratings-=ratings[index[anchor]]
        if np.abs(gradient).max()<1e-9: break
    return {name:ratings[i]*ELO_SCALE for name,i in index.items()}


def bradley_terry_fixed(records,ratings,virtual_draw=.5):
    """MLE Elo of one candidate against opponents whose ratings stay fixed.

    records: {opponent: [wins, losses, draws]} from the candidate's side;
    ratings: {opponent: Elo}. The reference pool is never re-rated by the
    candidate's games (OpenAI Five's frozen reference rating); a virtual draw
    per opponent (virtual_draw wins each way) keeps an undefeated candidate
    finite. The likelihood is concave in one variable, so Newton converges.
    """
    pairs=[(ratings[name]/ELO_SCALE,w+d/2+virtual_draw,w+l+d+2*virtual_draw)
           for name,(w,l,d) in records.items() if w+l+d>0]
    if not pairs: return 0.0
    r=sum(p[0] for p in pairs)/len(pairs)
    for _ in range(100):
        gradient=0.0; curvature=0.0
        for opponent,score,games in pairs:
            expected=1/(1+math.exp(opponent-r))
            gradient+=score-games*expected
            curvature-=games*expected*(1-expected)
        step=gradient/curvature
        r-=step
        if abs(step)<1e-12: break
    return r*ELO_SCALE


def pair_tasks(a,b,maps,games_per_map,seed_base):
    """Side-swapped fixed-seed games between two participants on each map."""
    if games_per_map%2 or games_per_map<2: raise ValueError('games per map must be even and positive')
    tasks=[]
    for map_id in maps:
        for pair in range(games_per_map//2):
            seed=seed_base+pair
            tasks.append(dict(a=a,b=b,map=map_id,seed=seed,side=0))
            tasks.append(dict(a=a,b=b,map=map_id,seed=seed,side=1))
    return tasks


def play_games(by_name,tasks,limit,workers,seed_base,delay=9):
    """Play tasks (dicts a, b, map, seed, side) between participants keyed by name.

    Each participant provides start(task, side, rng) -> act(observation, mask)
    and, for scripted bots, attach(env, side). Returns each task with its
    ticks and outcome: 'a', 'b', or 'draw' (a timeout, or neither/both won).
    Deterministic for a given (a, b, map, seed, side) and participant set.
    """
    from environment import Environment
    def play(bucket):
        results=[]
        with Environment(seed=seed_base+len(tasks),limit=limit,delay=delay) as env:
            for task in bucket:
                env.reset(task['seed'],task['map'])
                rng=np.random.default_rng((task['seed'],task['map'],task['side'],
                                           int(hashlib.sha256((task['a']+task['b']).encode()).hexdigest()[:8],16)))
                env.bot(0,0); env.bot(1,0)  # detach any bot from the previous task
                for name,side in ((task['a'],task['side']),(task['b'],1-task['side'])):
                    attach=getattr(by_name[name],'attach',None)
                    if attach: attach(env,side)
                actors={task['side']:by_name[task['a']].start(task,task['side'],rng),
                        1-task['side']:by_name[task['b']].start(task,1-task['side'],rng)}
                while not env.ended:
                    observation=env.obs.copy(); mask=env.mask.copy()
                    env.step(np.stack([actors[0](observation,mask),actors[1](observation,mask)]))
                final=env.frame()
                timeout=bool(final['flags']&2)
                a_won=bool(final['players'][task['side']]['flags']&2) and not timeout
                b_won=bool(final['players'][1-task['side']]['flags']&2) and not timeout
                results.append(dict(task,ticks=final['t'],
                                    outcome='draw' if timeout or a_won==b_won else ('a' if a_won else 'b')))
        return results
    if not tasks: return []
    buckets=[tasks[i::workers] for i in range(min(workers,len(tasks)))]
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        return [game for bucket in pool.map(play,buckets) for game in bucket]


def tally(games):
    """{(a, b): [wins_a, wins_b, draws]} from play_games results."""
    records={}
    for game in games:
        row=records.setdefault((game['a'],game['b']),[0,0,0])
        row[{'a':0,'b':1,'draw':2}[game['outcome']]]+=1
    return records
