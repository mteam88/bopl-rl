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
# (league check).

HEAD_BOUNDS=[sum(layout.ACTION_SIZES[:i]) for i in range(layout.NUM_ATNS+1)]  # logits row offsets per head
ELO_SCALE=400/math.log(10)
DEATH_FIELDS=('deaths_environment','deaths_opponent','deaths_self','deaths_other')  # telemetry.EVENT_FIELDS


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def sample_head(rng,row,index,temperature):
    """rng.choice(len(part), p=softmax(part / temperature)) for one head of a masked logits row,
    drawing the same index from the same rng state (numpy Generator.choice: float64 cdf of p,
    normalised by its last entry, searched right of one rng.random()) without choice's
    per-call validation."""
    part=row[HEAD_BOUNDS[index]:HEAD_BOUNDS[index+1]]
    if temperature[index]!=1.0: part=part/temperature[index]   # x / 1.0 is x
    weights=np.exp(part-part.max()); weights/=weights.sum()
    cdf=np.cumsum(weights,dtype=np.float64); cdf/=cdf[-1]
    return int(cdf.searchsorted(rng.random(),side='right'))


class Model:
    """A native checkpoint participant (eval/native_policy.py contract). Only this seat's
    row is evaluated (batch 1: rows are independent, so its logits and state are
    bit-identical to a batch holding both seats), through the evaluator's fast path;
    the critic is skipped unless a subclass sets needs_value."""
    needs_value=False

    def __init__(self,name,path,kind,step=None,library=None):
        from native_policy import load
        self.name=name; self.kind=kind; self.step=step; self.path=Path(path)
        self.policy,self.meta=load(self.path,**(dict(library=library) if library else {}))
        self.policy.use_fast_path(value=self.needs_value)
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.layout=layout.layout_of(self.meta.get('architecture'))
        from native_policy import writer_ablation
        self.ablate=writer_ablation(self.policy)

    def attach(self,env,side):
        # The environment writes this player's row in the checkpoint's observation layout
        # (v3 or v4, layout.py) and v4 writer family (v4_shape.ablate), so v3 pool members
        # and v4 candidates of different families share a game.
        env.set_layout(side,self.layout,ablate=self.ablate)

    def start(self,task,side,rng):
        from native_policy import Stepper
        policy=self.policy; stepper=Stepper(policy)
        temperature=getattr(self,'temperature',None) or (1.0,1.0,1.0)
        def act(observation,mask):
            # observation: both rows at the padded stride; only this side's row is in this
            # checkpoint's layout.
            stepper.step(observation[side:side+1,:policy.obs_size])
            # Sample as the trainer does: the environment's action mask (1 = allowed) zeroes latched choices;
            # an autoregressive aim head is materialised for the sampled button combo before aim is drawn.
            allowed=np.asarray(mask[side])!=0
            row=np.where(allowed,stepper.game_logits()[0],-np.inf)
            action=[0,sample_head(rng,row,1,temperature),0]
            if policy.aim_ar:
                row=np.where(allowed,stepper.game_logits(action[1])[0],-np.inf)
            action[0]=sample_head(rng,row,0,temperature); action[2]=sample_head(rng,row,2,temperature)
            return np.asarray(action,dtype=np.int64)
        return act


class Ensemble:
    """N native checkpoints playing one seat as one player. Each member keeps its own
    recurrent state and reads the same observation row; every head is sampled from the
    mean of the members' masked logits (a log-linear pool: the normalised geometric mean
    of their distributions). The heads are independent given the row, so the pooled joint
    still factorises per head, and it is exactly the product-of-experts of the members'
    joint policies; averaging probabilities per head would not be their mixture.
    With aim_ar members the buttons are drawn first, then the aim blocks of the drawn
    combo are pooled. seconds/decisions time act() (all members) for the cost report."""
    step=None; path=None

    def __init__(self,name,paths,kind,library=None):
        import threading
        self.name=name; self.kind=kind
        self.members=[Model(f'{name}[{i}]',p,kind,library=library) for i,p in enumerate(paths)]
        first=self.members[0]
        for m in self.members[1:]:
            # One observation row per seat: every member must read the layout and writer family it is written in.
            if (m.layout,m.ablate,m.policy.obs_size,m.policy.aim_ar)!=(first.layout,first.ablate,first.policy.obs_size,first.policy.aim_ar):
                raise ValueError(f'ensemble members {first.path} and {m.path} read different observation rows or aim heads')
        self.layout=first.layout; self.ablate=first.ablate
        self.sha256=hashlib.sha256('+'.join(m.sha256 for m in self.members).encode()).hexdigest()
        self.seconds=0.; self.decisions=0; self.lock=threading.Lock()

    def attach(self,env,side):
        env.set_layout(side,self.layout,ablate=self.ablate)

    def start(self,task,side,rng):
        import time
        from native_policy import Stepper
        steppers=[Stepper(m.policy) for m in self.members]
        width=self.members[0].policy.obs_size; aim_ar=self.members[0].policy.aim_ar
        temperature=getattr(self,'temperature',None) or (1.0,1.0,1.0)
        def act(observation,mask):
            t0=time.perf_counter()
            allowed=np.asarray(mask[side])!=0
            for s in steppers: s.step(observation[side:side+1,:width])
            pooled=lambda combo=0: np.mean([s.game_logits(combo)[0] for s in steppers],axis=0)
            row=np.where(allowed,pooled(),-np.inf)
            action=[0,sample_head(rng,row,1,temperature),0]
            if aim_ar:
                row=np.where(allowed,pooled(action[1]),-np.inf)
            action[0]=sample_head(rng,row,0,temperature); action[2]=sample_head(rng,row,2,temperature)
            with self.lock:
                self.seconds+=time.perf_counter()-t0; self.decisions+=1
            return np.asarray(action,dtype=np.int64)
        return act

    def fork_stats(self):
        return (self.seconds,self.decisions)

    def merge_fork_stats(self,delta):
        with self.lock:
            self.seconds+=delta[0]; self.decisions+=delta[1]

    def summary(self):
        return dict(members=[str(m.path) for m in self.members],decisions=self.decisions,
                    ms_per_decision=1000*self.seconds/max(self.decisions,1))


def participant(name,spec,kind='candidate'):
    """A player from a command-line spec: 'random', a checkpoint path, 'member:<league name>',
    or 'ensemble:<spec>+<spec>+...' whose parts are checkpoint paths or member: names."""
    def checkpoint(part):
        if not part.startswith('member:'):
            return Path(part).resolve()
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[1]/'league'))
        import registry
        rows=registry.by_name(registry.load())
        if part[7:] not in rows or not rows[part[7:]].get('path'):
            raise ValueError(f'{part}: no checkpoint member of that name in the league')
        return registry.ROOT/rows[part[7:]]['path']
    if spec=='random':
        return Random()
    if spec.startswith('ensemble:'):
        return Ensemble(name,[checkpoint(p) for p in spec[9:].split('+')],kind)
    return Model(name,checkpoint(spec),kind)


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


def bradley_terry_fit(names, records, anchor, virtual_draw=.5, ridge=1e-6):
    """Vectorized MLE + Wald 95% CI, same model as bradley_terry.

    One virtual draw (virtual_draw wins each way) per played pair, independent
    of the real game count: a 480-game pair is barely moved, a 6-game pair
    stays finite. Sparse Newton (scipy) converges the joint MLE; Wald intervals
    use the diagonal of the observed information so a few-thousand-player fit
    stays off the dense n^3 inverse. Full dense inversion is used only when
    n is small enough that the off-diagonal terms are cheap.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    names = list(names)
    index = {name: i for i, name in enumerate(names)}
    empty = {name: 0.0 for name in names}
    empty_ci = {name: [0.0, 0.0] for name in names}
    if anchor not in index:
        return dict(ratings=empty, intervals=empty_ci, iterations=0, ci='none')
    rows = []
    for (a, b), (wa, wb, d) in records.items():
        if a not in index or b not in index or a == b:
            continue
        if wa + wb + d <= 0:
            continue
        rows.append((index[a], index[b], wa + d / 2 + virtual_draw, wb + d / 2 + virtual_draw))
    n = len(names)
    if not rows:
        return dict(ratings=empty, intervals=empty_ci, iterations=0, ci='none')
    a_idx = np.array([row[0] for row in rows], dtype=np.int32)
    b_idx = np.array([row[1] for row in rows], dtype=np.int32)
    score_a = np.array([row[2] for row in rows], dtype=np.float64)
    score_b = np.array([row[3] for row in rows], dtype=np.float64)
    games = score_a + score_b
    ratings = np.zeros(n)
    ai = index[anchor]
    keep = np.array([i for i in range(n) if i != ai], dtype=np.int32)
    remap = np.full(n, -1, dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    m = len(keep)

    def expected_and_gradient(values):
        expected = 1 / (1 + np.exp(np.clip(values[b_idx] - values[a_idx], -40, 40)))
        ga = score_a - games * expected
        gb = score_b - games * (1 - expected)
        gradient = np.bincount(a_idx, ga, minlength=n) + np.bincount(b_idx, gb, minlength=n)
        gradient[ai] = 0.0
        return expected, gradient

    def information(expected):
        weight = games * expected * (1 - expected)
        rows_i = np.concatenate([a_idx, b_idx, a_idx, b_idx, keep])
        cols_i = np.concatenate([a_idx, b_idx, b_idx, a_idx, keep])
        data = np.concatenate([weight, weight, -weight, -weight, np.full(m, ridge)])
        ok = (remap[rows_i] >= 0) & (remap[cols_i] >= 0)
        return coo_matrix(
            (data[ok], (remap[rows_i[ok]], remap[cols_i[ok]])), shape=(m, m)
        ).tocsr(), weight

    iterations = 0
    expected, gradient = expected_and_gradient(ratings)
    for iterations in range(1, 41):
        info, _ = information(expected)
        try:
            step = spsolve(info, gradient[keep])
        except Exception:
            player_games = np.bincount(a_idx, games, minlength=n) + np.bincount(b_idx, games, minlength=n)
            step = gradient[keep] * (2 / max(1.0, float(player_games.max())))
        if np.max(np.abs(step)) < 1e-10:
            ratings[keep] += step
            ratings -= ratings[ai]
            break
        # Damped if a Newton step would explode (first steps on a flat start).
        scale = 1.0
        peak = float(np.max(np.abs(step)))
        if peak > 8:
            scale = 8 / peak
        ratings[keep] += step * scale
        ratings -= ratings[ai]
        expected, gradient = expected_and_gradient(ratings)
        if np.max(np.abs(gradient)) < 1e-8:
            break
    expected, gradient = expected_and_gradient(ratings)
    info, weight = information(expected)
    se = np.zeros(n)
    ci_method = 'wald-diag'
    diag = np.bincount(a_idx, weight, minlength=n) + np.bincount(b_idx, weight, minlength=n)
    se[keep] = 1 / np.sqrt(np.maximum(diag[keep] + ridge, 1e-18))
    if m <= 400:
        try:
            dense = info.toarray()
            cov = np.linalg.inv(dense)
            se[keep] = np.sqrt(np.maximum(np.diag(cov), 0.0))
            ci_method = 'wald'
        except np.linalg.LinAlgError:
            pass
    z = 1.959963984540054
    elo = ratings * ELO_SCALE
    lo = (ratings - z * se) * ELO_SCALE
    hi = (ratings + z * se) * ELO_SCALE
    lo[ai] = 0.0
    hi[ai] = 0.0
    return dict(
        ratings={name: float(elo[i]) for name, i in index.items()},
        intervals={name: [float(lo[i]), float(hi[i])] for name, i in index.items()},
        iterations=iterations,
        ci=ci_method,
    )


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


def pair_tasks(a,b,maps,games_per_map,seed_base,lineups=()):
    """Side-swapped fixed-seed games between two participants on each map.

    lineups are indices into a loadout set (evaluation_protocol.resolve_loadouts
    'lineups'); each one gets its own games_per_map games on every map, so the
    count is per lineup. Without them the schedule is the fixed-loadout one.
    """
    if games_per_map%2 or games_per_map<2: raise ValueError('games per map must be even and positive')
    tasks=[]
    for map_id in maps:
        for lineup in (lineups or (None,)):
            for pair in range(games_per_map//2):
                task=dict(a=a,b=b,map=map_id,seed=seed_base+pair,side=0)
                if lineup is not None: task['lineup']=lineup
                tasks.append(task)
                tasks.append(dict(task,side=1))
    return tasks


def side_pinned_tasks(a,b,maps,games_per_map,seed_base,lineup_by_side):
    """Fixed-seed games where each SIDE carries its own pinned loadout row.

    pair_tasks couples both sides to one lineup, which is right when the row is
    mirrored and wrong when it is not: a row fixes the kit each seat plays, so
    swapping sides under one asymmetric row swaps the kits too. Here the caller
    supplies {0: row_index, 1: row_index} and the side chooses the row, which is
    what lets both participants keep their own kit in every game.

    Still side-swapped and still even: half the games put `a` in seat 0 under
    lineup_by_side[0], half put it in seat 1 under lineup_by_side[1].
    """
    if games_per_map%2 or games_per_map<2: raise ValueError('games per map must be even and positive')
    if set(lineup_by_side)!={0,1}: raise ValueError('lineup_by_side needs a row for side 0 and side 1')
    tasks=[]
    for map_id in maps:
        for pair in range(games_per_map//2):
            for side in (0,1):
                tasks.append(dict(a=a,b=b,map=map_id,seed=seed_base+pair,side=side,
                                  lineup=lineup_by_side[side]))
    return tasks


def play_games(by_name,tasks,limit,workers,seed_base,delay=9,on_result=None,loadouts=None,style=None,processes=False):
    """Play tasks (dicts a, b, map, seed, side) between participants keyed by name.

    Each participant provides start(task, side, rng) -> act(observation, mask)
    and, for scripted bots, attach(env, side). Returns each task with its
    ticks and outcome: 'a', 'b', or 'draw' (a timeout, or neither/both won); a draw
    also carries draw_kind: 'timeout', 'both-dead' or 'neither-won'.
    Deterministic for a given (a, b, map, seed, side) and participant set.

    loadouts (evaluation_protocol.resolve_loadouts) replaces the fixed 2,4,14
    draw; a task may name a `lineup` index into its set to pin exactly that row
    for its game, and every game then carries the six ability ids it played as
    `lineup_ids`. Without loadouts nothing changes: the C sampler stays in fixed
    mode, draws no extra randomness, and the games are the ones played before.

    style (dict data=<style.bin bytes>, terms=[names]) measures a compiled style in
    every game (Environment.install_style: tallied, never paid, so the games are the
    same ones): each result gets `style`, per seat {term: [pay, fires]},
    `style_decisions` per seat, and `event_support` (1 when the game backend emitted
    events, which event terms need).

    processes plays each worker's games in a forked process instead of a thread
    (forked): the same games, without GIL contention. Participants then must not rely
    on state shared across games beyond fork_stats, and on_result is not supported.
    """
    if processes and on_result:
        raise ValueError('on_result would run in the forked workers; play with threads')
    from environment import Environment
    rows=(loadouts or {}).get('lineups') or []
    def play(bucket):
        results=[]
        with Environment(seed=seed_base+len(tasks),limit=limit,delay=delay,loadouts=loadouts) as env:
            if style is not None and env.install_style(style['data'])!=len(style['terms']):
                raise ValueError('style term names do not match the program')
            pinned=False
            for task in bucket:
                # bopl_draw_config picks a set row uniformly at random, so the only way
                # to force one is to hand the environment a set of exactly that row --
                # verbatim, with no mirroring layered on what the file already says.
                # It must happen before the reset that draws the episode. Doing it per
                # task is free next to the episode: the call is a memcpy of at most
                # 1.5 kB plus a validation pass (96 pinned games cost the same wall
                # clock as 96 unpinned ones), so nothing here caches the installed row.
                index=task.get('lineup')
                if index is not None:
                    env.configure_loadouts(dict(loadouts,lineups=[rows[index]],mirror_share=0.)); pinned=True
                elif pinned:
                    env.configure_loadouts(loadouts); pinned=False
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
                    observation=env.rows.copy(); mask=env.mask.copy()
                    env.step(np.stack([actors[0](observation,mask),actors[1](observation,mask)]))
                final=env.frame()
                timeout=bool(final['flags']&2)
                a_won=bool(final['players'][task['side']]['flags']&2) and not timeout
                b_won=bool(final['players'][1-task['side']]['flags']&2) and not timeout
                result=dict(task,ticks=final['t'],
                            outcome='draw' if timeout or a_won==b_won else ('a' if a_won else 'b'))
                # Per participant [a, b]: deaths by cause (telemetry deaths_*) and the
                # sim's cause_of_death code of the last one, None if it never died
                # (eval/readout.py deaths reads the codes).
                result['deaths']=[]
                for seat in (task['side'],1-task['side']):
                    counts={k[7:]:final['telemetry'][seat][k] for k in DEATH_FIELDS}
                    result['deaths'].append(dict(counts,code=final['players'][seat]['cause_of_death'] if any(counts.values()) else None))
                # A draw is either the episode limit or a trade; they mean opposite things
                # about a policy, so say which without changing how a draw is scored.
                if result['outcome']=='draw':
                    result['draw_kind']='timeout' if timeout else ('both-dead' if a_won else 'neither-won')
                if style is not None:
                    seats=env.style_tallies()
                    result['style']=[{name:[seat['pay'][t],seat['fires'][t]] for t,name in enumerate(style['terms'])}
                                     for seat in seats]
                    result['style_decisions']=[seat['decisions'] for seat in seats]
                    result['event_support']=seats[0]['event_support']
                if loadouts is not None:
                    played=env.loadouts
                    result['lineup_ids']=played[0]+played[1]
                if on_result:
                    on_result(result)
                results.append(result)
        return results
    if not tasks: return []
    buckets=[tasks[i::workers] for i in range(min(workers,len(tasks)))]
    if processes and len(buckets)>1:
        return forked(play,buckets,by_name)
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        return [game for bucket in pool.map(play,buckets) for game in bucket]


def forked(play,buckets,by_name):
    """[game for bucket in map(play, buckets)], each bucket in its own forked process.

    Threads share one GIL, and the per-decision Python of a game is about a third of
    its CPU once the policy runs on the fast path, so six threads cannot use six
    cores; forked workers can. The games are the same (each task seeds its own rng;
    a bucket's environment is seeded as in the threaded path). State a participant
    accumulates while playing stays in the child unless it defines fork_stats() (a
    tuple of running totals) and merge_fork_stats(delta), which the parent calls with
    what the child's games added."""
    import os,pickle,signal,sys,traceback
    children=[]
    try:
        for bucket in buckets:
            r,w=os.pipe()
            pid=os.fork()
            if pid==0:
                os.close(r); code=0
                try:
                    counted={n:p for n,p in by_name.items() if hasattr(p,'fork_stats')}
                    before={n:p.fork_stats() for n,p in counted.items()}
                    played=play(bucket)
                    delta={n:tuple(x-y for x,y in zip(p.fork_stats(),before[n])) for n,p in counted.items()}
                    payload=pickle.dumps((None,played,delta))
                except BaseException:
                    payload=pickle.dumps((traceback.format_exc(),None,None)); code=1
                with os.fdopen(w,'wb') as f: f.write(payload)
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(code)
            os.close(w); children.append((pid,r))
        games=[]
        for pid,r in children:
            # A finished child blocks on its full pipe until read; reading in order cannot deadlock.
            with os.fdopen(r,'rb') as f: error,played,stats=pickle.loads(f.read() or pickle.dumps(('worker died',None,None)))
            os.waitpid(pid,0)
            if error: raise RuntimeError(f'game worker {pid} failed:\n{error}')
            for name,totals in stats.items(): by_name[name].merge_fork_stats(totals)
            games+=played
        children=[]
        return games
    finally:
        for pid,_ in children:
            try: os.kill(pid,signal.SIGKILL); os.waitpid(pid,0)
            except (ProcessLookupError,ChildProcessError): pass


def tally(games):
    """{(a, b): [wins_a, wins_b, draws]} from play_games results."""
    records={}
    for game in games:
        row=records.setdefault((game['a'],game['b']),[0,0,0])
        row[{'a':0,'b':1,'draw':2}[game['outcome']]]+=1
    return records
