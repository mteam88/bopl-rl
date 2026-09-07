"""Reconcile all frozen matches, then plot the three-condition comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'A':'#61798d','B':'#df872d','C':'#287f72'}
LABELS={'A':'A · Self-play','B':'B · Script mixture','C':'C · Bow initialization + mixture'}


def read(path):return json.loads(Path(path).read_text())


def paired_interval(left,right,seed=510):
    """Resample training seeds and paired action-seed blocks within fixed cells."""
    difference=right.mean(-1)-left.mean(-1)  # training seed, cell, pair
    n,c,p=difference.shape
    rng=np.random.default_rng(seed)
    samples=5000
    training=rng.integers(n,size=(samples,n))
    pairs=rng.integers(p,size=(samples,n,c,p))
    values=difference[training[:,:,None,None],np.arange(c)[None,None,:,None],pairs].mean((1,2,3))
    return dict(mean=float(difference.mean()),by_seed=difference.mean((1,2)).tolist(),
                interval=np.quantile(values,[.025,.975]).tolist(),bootstrap_replicates=samples)


def evaluation(path,protocol,suite):
    r=read(path);spec=protocol[suite]
    # DeclareGameUndecided awards WonThisRound to tied last deaths. Preserve
    # raw artifacts, but a shared award is a draw for the frozen win metric.
    shared_awards=[]
    for game in r['games']:
        if game['win'] and game['loss']:
            assert not game['timeout'] and all(p['stocks']==0 for p in game['final_players'])
            shared_awards.append(game['name'])
            game['win']=game['loss']=False
    names=[b['name'] for b in spec['bots']]
    cells=[(m,name) for m in spec['maps'] for name in names]
    array=np.zeros((len(cells),spec['pairs_per_cell'],2))
    seen=set();by_cell=[]
    for game in r['games']:
        m=game['map'];name=game['opponent']['spec']['name'];side=game['side']
        pair=game['seed']-spec['seed_base']-m*1000-names.index(name)*100
        key=(m,name,pair,side)
        if key in seen or not 0<=pair<spec['pairs_per_cell']:raise ValueError('Duplicate or unexpected task')
        seen.add(key)
        assert game['step_mode']=='sampled'
        assert not (game['win'] and (game['loss'] or game['timeout']))
        array[cells.index((m,name)),pair,side]=game['win']
    assert len(seen)==len(cells)*spec['pairs_per_cell']*2
    for i,(m,name) in enumerate(cells):
        games=[g for g in r['games'] if g['map']==m and g['opponent']['spec']['name']==name]
        by_cell.append(dict(map=m,opponent=name,win_rate=float(array[i].mean()),
                            games=len(games),wins=sum(g['win'] for g in games),
                            timeouts=sum(g['timeout'] for g in games)))
    n=len(r['games'])
    return array,dict(games=n,shared_award_draws=shared_awards,wins=sum(g['win'] for g in r['games']),
                      losses=sum(g['loss'] for g in r['games']),timeouts=sum(g['timeout'] for g in r['games']),
                      draws=sum(not g['win'] and not g['loss'] for g in r['games']),
                      win_rate=float(array.mean()),kills=sum(g['kills'] for g in r['games']),
                      deaths=sum(g['deaths'] for g in r['games']),
                      hits=sum(g['behavior']['totals']['arrow_hits'] for g in r['games']),
                      shots=sum(g['behavior']['totals']['arrows_fired'] for g in r['games']),
                      environment_deaths=sum(g['behavior']['totals']['deaths_environment'] for g in r['games']),
                      opponent_environment_deaths=sum(g['opponent_behavior']['totals']['deaths_environment'] for g in r['games']),
                      elapsed=r['elapsed'],cells=by_cell)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.artifacts;out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=read(root/'protocol.json')
    expected=(root/'protocol.sha256').read_text().strip()
    assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==expected
    results={};arrays={};curves={}
    for run in sorted(protocol['runs'],key=lambda r:(r['condition'],r['seed'])):
        name=run['name'];r=read(root/'runs'/name/'result.json');config=read(root/'runs'/name/'config.json')
        assert r['status']=='complete' and r['learner_steps']>=protocol['learner_decisions']
        assert r['learner_steps']<protocol['learner_decisions']+protocol['horizon']*protocol['workers']*2
        assert config['training_maps']==protocol['training_maps'] and config['learning_rate']==protocol['learning_rate']
        assert config['assembly_sha256']==protocol['assembly_sha256']
        for f,h in protocol['training_source_sha256'].items():
            assert config['source_sha256']['native/rl/research/'+f]==h,(name,f)
        metrics=[json.loads(line) for line in (root/'runs'/name/'metrics.jsonl').read_text().splitlines()]
        exposure=Counter()
        for m in metrics:
            exposure.update(m['opponent_game_decisions'])
            assert not m['env'].get('game_errors',0) and not m['env'].get('observation_overflow',0)
            assert m['learner_fraction']==.75
        curves[name]=metrics
        result=dict(condition=run['condition'],seed=run['seed'],learner_decisions=r['learner_steps'],
                    all_player_decisions=r['step'],updates=r['updates'],matches=r['matches'],
                    training_seconds=r['elapsed'],wall_seconds=r['wall_seconds'],
                    opponent_game_decisions=dict(exposure),
                    retention=read(root/'retention'/name/'result.json'))
        # Keep trial-level retention artifacts outside the compact summary.
        result['retention'].pop('games')
        for suite in ('development','final'):
            array,aggregate=evaluation(root/'evaluations'/name/suite/'result.json',protocol,suite)
            arrays[name,suite]=array;result[suite]=aggregate
        results[name]=result
    totals={};comparisons={}
    for suite in ('development','final'):
        for condition in 'ABC':
            names=[name for name,r in results.items() if r['condition']==condition]
            matrix=np.stack([arrays[name,suite] for name in names])
            totals[condition,suite]=matrix
        comparisons[suite]={f'{right}-{left}':paired_interval(totals[left,suite],totals[right,suite])
                            for left,right in [('A','B'),('B','C')]}
    references={}
    for name in ('untrained-s73','bow-initial'):
        path=root/'references'/name/'result.json'
        if path.exists():_,references[name]=evaluation(path,protocol,'development')
    summary=dict(schema=1,protocol_sha256=expected,runs=results,comparisons=comparisons,
                 development_references=references,
                 total_learner_decisions=sum(r['learner_decisions'] for r in results.values()),
                 total_matches=sum(r[s]['games'] for r in results.values() for s in ('development','final')),
                 interval_note='Descriptive hierarchical bootstrap of three training seeds and paired action-seed blocks within fixed map/opponent cells. Three training seeds limit uncertainty estimates.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,suite in zip(axes[0],('development','final')):
        for i,c in enumerate('ABC'):
            rates=totals[c,suite].mean((1,2,3))*100
            ax.bar(i,rates.mean(),color=COLORS[c],alpha=.8,width=.6)
            ax.scatter(i+np.array([-.12,0,.12]),rates,color='black',s=24,zorder=3)
            ax.text(i,max(rates.max(),rates.mean())+1,f'{rates.mean():.1f}%',ha='center')
        ax.set(xticks=range(3),xticklabels=['A','B','C'],ylabel='Wins / all matches (%)',
               title=f'{suite.title()}: final checkpoints, all 3 seeds',ylim=(0,min(100,max(30,max(totals[c,suite].mean((1,2,3)).max()*100 for c in 'ABC')+12))))
    for name,style in [('untrained-s73','--'),('bow-initial',':')]:
        if name not in references:continue
        axes[0,0].axhline(references[name]['win_rate']*100,linestyle=style,color='#8055a0',
                          linewidth=1,label=f'{name} starting reference (one policy)')
    if references:axes[0,0].legend(fontsize=7,loc='upper left')
    ax=axes[1,0]
    for i,c in enumerate('ABC'):
        rows=[r for r in results.values() if r['condition']==c]
        rates=np.array([r['retention']['successes']/r['retention']['trials']*100 for r in rows])
        ax.bar(i,rates.mean(),color=COLORS[c],alpha=.8,width=.6)
        ax.scatter(i+np.array([-.12,0,.12]),rates,color='black',s=24,zorder=3)
    ax.axhline(100,color=COLORS['C'],linestyle=':',label='C before PPO: 32/32')
    ax.set(xticks=range(3),xticklabels=['A','B','C'],ylabel='Successful first shots (%)',ylim=(-2,108),title='Retention under unrestricted controls')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    for name,metrics in curves.items():
        c=results[name]['condition'];x=np.array([m['learner_steps'] for m in metrics])/1e6
        hits=np.cumsum([m['behavior']['totals']['arrow_hits'] for m in metrics])
        ax.plot(x,hits,color=COLORS[c],alpha=.65,linewidth=1,label=LABELS[c] if results[name]['seed']==73 else None)
    ax.set(xlabel='Learner decisions (millions)',ylabel='Cumulative learner arrow hits',title='Training behavior · opponent mixtures differ')
    ax.legend(fontsize=8)
    fig.suptitle('Scripted opponents and Bow initialization · frozen 3 × 3 comparison',fontsize=16)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'overview.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    spec=protocol['final'];names=[s['name'].replace('-test','') for s in spec['bots']]
    vmax=max(10,max(totals[c,'final'].mean((0,2,3)).max()*100 for c in 'ABC'))
    for ax,c in zip(axes,'ABC'):
        matrix=totals[c,'final'].mean((0,2,3)).reshape(len(spec['maps']),len(names))*100
        im=ax.imshow(matrix,vmin=0,vmax=vmax,cmap='YlGnBu',aspect='auto')
        for (y,x),v in np.ndenumerate(matrix):ax.text(x,y,f'{v:.1f}',ha='center',va='center',color='white' if v>vmax*.6 else 'black')
        ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(3),yticklabels=[f'Map {m}' for m in spec['maps']],title=LABELS[c])
        ax.tick_params(axis='x',rotation=30)
    fig.colorbar(im,ax=axes,label='Wins / all matches (%)',shrink=.8)
    fig.suptitle('Reserved maps and opponents · averages across 3 training seeds',fontsize=15)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'final-cells.{ext}',dpi=160)
    for svg in out.glob('*.svg'):svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(dict(matches=summary['total_matches'],learner_decisions=summary['total_learner_decisions'],comparisons=comparisons),indent=2))


if __name__=='__main__':main()
