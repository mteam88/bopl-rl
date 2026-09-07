"""Summarize every fixed A/B continuation milestone and direct pairing."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_data import aggregate, outcome

SEEDS=(73,173,273)
COLORS={'A':'#537b9c','B':'#c47c32'}


def read(path):
    return json.loads(path.read_text())


def compact(report):
    value=aggregate(report)
    value['kills']=sum(g['kills'] for g in report['games'])
    value['timeouts']=sum(bool(g['timeout']) for g in report['games'])
    value['shared_winner_flags']=sum(bool(g['win'] and g['loss']) for g in report['games'])
    for player,key in [('learner','behavior'),('opponent','opponent_behavior')]:
        for field in ('deaths_environment','deaths_opponent','deaths_self'):
            value[player+'_'+field]=sum(g[key]['totals'][field] for g in report['games'])
    return value


def cube(reports,protocol,suite):
    spec=protocol[suite];maps=spec['maps'];bots=[b['name'] for b in spec['bots']]
    result=np.full((3,len(bots),len(maps),spec['pairs_per_cell'],2),np.nan)
    for i,report in enumerate(reports):
        for g in report['games']:
            bot=g['opponent']['spec']['name'];pair=int(g['name'].split('-p')[-1].split('-')[0])
            result[i,bots.index(bot),maps.index(g['map']),pair,g['side']]=outcome(g)[0]
    assert np.isfinite(result).all(), "Incomplete script grid"
    return result


def interval(values):
    return [float(x) for x in np.quantile(values,[.025,.975])]


def paired_difference(a,b,seed=5151):
    """Resample training seeds and paired action-seed blocks within fixed cells."""
    delta=(b-a).mean(axis=-1);rng=np.random.default_rng(seed);samples=[]
    for _ in range(5000):
        chosen=delta[rng.integers(0,len(delta),len(delta))]
        indices=rng.integers(0,chosen.shape[-1],size=chosen.shape)
        samples.append(float(np.take_along_axis(chosen,indices,axis=-1).mean()))
    return dict(mean=float(delta.mean()),by_seed=delta.mean(axis=tuple(range(1,delta.ndim))).tolist(),interval95=interval(samples))


def cross_cube(out,protocol,milestone):
    spec=protocol['crossplay'];data=np.full((3,3,len(spec['maps']),spec['pairs_per_cell'],2,3),np.nan);counts=[]
    for ai,a in enumerate(SEEDS):
        for bi,b in enumerate(SEEDS):
            report=read(out/'evaluations'/f'crossplay-{milestone}-A{a}-B{b}'/'result.json')
            counts.append(dict(A_seed=a,B_seed=b,**compact(report)))
            for g in report['games']:
                pair=int(g['name'].split('-p')[-1].split('-')[0])
                data[ai,bi,spec['maps'].index(g['map']),pair,g['side']]=outcome(g)
    assert np.isfinite(data).all() and np.all(data.sum(axis=-1)==1), "Incomplete crossplay grid"
    return data,counts


def cross_difference(delta):
    """Resample A and B seed axes separately, then paired game seeds per map."""
    delta=delta.mean(axis=-1);rng=np.random.default_rng(6151);samples=[]
    for _ in range(5000):
        selected=delta[rng.integers(0,3,3)][:,rng.integers(0,3,3)]
        indices=rng.integers(0,selected.shape[-1],size=selected.shape)
        samples.append(float(np.take_along_axis(selected,indices,axis=-1).mean()))
    return dict(mean=float(delta.mean()),interval95=interval(samples))


def save(fig,out,stem):
    for ext in ('png','svg','pdf'):fig.savefig(out/f'{stem}.{ext}',dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.artifacts;out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=read(root/'protocol.json');common=read(root/'fallback-checkpoint.json')['milestone'];balanced=tuple(m for m in (1000000,2000000,3000000) if m<=common);parent=root.parent/'opponent-league';runs=[];cubes={};cross={};crossdata={}
    for condition in 'AB':
        for milestone in (1000000,2000000,3000000,4000000):
            for suite in (('development','transfer') if milestone in (1000000,common,4000000) else ('development',)):
                reports=[]
                for seed in SEEDS:
                    name=f'{condition}-s{seed}'
                    path=parent/'evaluations'/name/('final' if suite=='transfer' else suite) if milestone==1000000 else root/'evaluations'/f'{name}-{milestone}-{suite}'
                    if not (path/'result.json').exists():continue
                    report=read(path/'result.json');reports.append(report)
                    runs.append(dict(run=name,condition=condition,seed=seed,milestone=milestone,suite=suite,**compact(report)))
                if len(reports)==3:cubes[condition,milestone,suite]=cube(reports,protocol,suite)
    contrasts={}
    for milestone in (1000000,2000000,3000000,4000000):
        for suite in (('development','transfer') if milestone in (1000000,common,4000000) else ('development',)):
            if ('A',milestone,suite) not in cubes or ('B',milestone,suite) not in cubes:continue
            contrasts[f'B_minus_A_{milestone}_{suite}']=paired_difference(cubes['A',milestone,suite],cubes['B',milestone,suite])
    for c in 'AB':
        for suite in ('development','transfer'):
            contrasts[f'{c}_{common//1000000}M_minus_1M_{suite}']=paired_difference(cubes[c,1000000,suite],cubes[c,common,suite])
        if (c,4000000,'development') in cubes:contrasts[f'{c}_4M_minus_3M_development']=paired_difference(cubes[c,3000000,'development'],cubes[c,4000000,'development'])
    for milestone in (1000000,common):
        values,cells=cross_cube(root,protocol,milestone);crossdata[milestone]=values
        cross[milestone]=dict(games=864,A_wins=int(values[...,0].sum()),B_wins=int(values[...,1].sum()),draws=int(values[...,2].sum()),
                              cells=cells,B_minus_A=cross_difference(values[...,1]-values[...,0]))
    partial=[]
    for a in SEEDS:
        for b in SEEDS:
            file=root/'evaluations'/f'crossplay-4000000-A{a}-B{b}'/'result.json'
            partial.append(dict(A_seed=a,B_seed=b,result=compact(read(file)) if file.exists() else None))
    cross[4000000]=dict(complete=False,cells=partial,note='Missing B seed 73; no balanced aggregate or inference.')
    cross_change=cross_difference(crossdata[common][...,1]-crossdata[1000000][...,1])
    training=[]
    for r in protocol['runs']:
        directory=root/'runs'/r['name'];finished=(directory/'result.json').exists()
        result=read(directory/('result.json' if finished else 'status.json'))
        training.append(dict(name=r['name'],status='complete' if finished else 'interrupted',learner_steps=result['learner_steps'],new_learner_steps=result['learner_steps']-r['source_learner_steps'],wall_seconds=result.get('wall_seconds',result.get('elapsed'))))
    summary=dict(schema=1,complete=False,available_work_complete=True,common_checkpoint=common,incidents=read(root/'training-incidents.json'),evaluation=read(root/'evaluation-status.json'),training=training,script_results=runs,contrasts=contrasts,crossplay=cross,crossplay_B_win_change=cross_change,
                 uncertainty='Descriptive 95% bootstrap intervals, 5000 resamples. Script contrasts pair training seeds and within-cell game-seed blocks (both sides). Crossplay resamples the A and B training-seed axes separately and game-seed blocks within each map, pairing milestones for change. Maps and opponents remain fixed. Three seeds per condition cannot establish broad generalization.',
                 draw_rule='Timeouts and shared winner flags are draws; all matches stay in win-rate denominators.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for c in 'AB':
        y=np.array([cubes[c,m,'development'].mean(axis=(1,2,3,4))*100 for m in balanced])
        for i,seed in enumerate(SEEDS):axes[0,0].plot([m/1e6 for m in balanced],y[:,i],color=COLORS[c],alpha=.3,linewidth=1)
        axes[0,0].plot([m/1e6 for m in balanced],y.mean(1),'-o',color=COLORS[c],label=c,linewidth=2.5)
        t=np.array([cubes[c,m,'transfer'].mean(axis=(1,2,3,4))*100 for m in (1000000,common)])
        for i in range(3):axes[0,1].plot([1,common/1e6],t[:,i],color=COLORS[c],alpha=.3,linewidth=1)
        axes[0,1].plot([1,common/1e6],t.mean(1),'-o',color=COLORS[c],label=c,linewidth=2.5)
    for row in (r for r in runs if r['milestone']==4000000 and r['suite']=='development'):
        axes[0,0].scatter([4],[row['win_rate']*100],edgecolors=COLORS[row['condition']],facecolors='none',s=35)
    for ax,title in zip(axes[0],['Development: every fixed checkpoint','Reserved maps: same inspected benchmark']):
        ax.set(title=title,xlabel='Million learner decisions',ylabel='Win rate (%)',xticks=[1,2,3,4],ylim=(0,100));ax.legend();ax.grid(alpha=.15)
    ax=axes[1,0];bottom=np.zeros(2)
    for label,color,key in [('A wins',COLORS['A'],'A_wins'),('B wins',COLORS['B'],'B_wins'),('Draws','#b5b2ac','draws')]:
        y=np.array([cross[m][key]/864*100 for m in (1000000,common)])
        ax.bar([0,1],y,bottom=bottom,label=label,color=color)
        for i,v in enumerate(y):
            if v>5:ax.text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center')
        bottom+=y
    ax.set(title='A versus B: all nine seed pairings',xticks=[0,1],xticklabels=['1M',f'{common//1000000}M'],ylabel='Share of all 864 matches (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncols=3)
    ax=axes[1,1]
    for c,offset in [('A',-.18),('B',.18)]:
        rows=[r for r in runs if r['condition']==c and r['suite']=='transfer'];vals=[]
        for m in (1000000,common):
            selected=[r for r in rows if r['milestone']==m];vals.append(sum(r['hits'] for r in selected)/len(selected)/384)
        ax.bar(np.arange(2)+offset,vals,width=.34,color=COLORS[c],label=c)
    ax.set(title='Arrow hits against reserved-map scripts',xticks=[0,1],xticklabels=['1M',f'{common//1000000}M'],ylabel='Learner hits / match');ax.legend()
    fig.suptitle('Bounded longer training · A self-play / B script mixture\nThin lines: each seed. Bold lines: three-seed mean. Hollow 4M points: incomplete cohort.',fontsize=14)
    save(fig,out,'overview')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,m in zip(axes,(1000000,common)):
        matrix=crossdata[m][...,1].mean(axis=(2,3,4))*100
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlOrBr')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>65 else '#242424')
        ax.set(title=f'{m//1000000}M learner decisions',xticks=range(3),xticklabels=SEEDS,yticks=range(3),yticklabels=SEEDS,xlabel='B training seed',ylabel='A training seed')
    fig.colorbar(im,ax=axes,label='B wins / all 96 matches in pairing (%)',shrink=.8)
    fig.suptitle('Direct crossplay: all pairings, six maps, both sides')
    save(fig,out,'crossplay-pairings')
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    changes={c:(cubes[c,common,'transfer']-cubes[c,1000000,'transfer']).mean(axis=(0,3,4))*100 for c in 'AB'}
    extent=max(10,float(np.ceil(max(np.abs(x).max() for x in changes.values())/5)*5))
    labels=[b['name'].replace('-test','') for b in protocol['transfer']['bots']]
    for row,c in enumerate('AB'):
        values=cubes[c,common,'transfer'].mean(axis=(0,3,4))*100
        for col,data in enumerate((values,changes[c])):
            ax=axes[row,col]
            im=ax.imshow(data,vmin=0 if col==0 else -extent,vmax=100 if col==0 else extent,cmap='YlOrBr' if col==0 else 'RdBu')
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    dark=(data[i,j]>65) if col==0 else abs(data[i,j])>extent*.65
                    ax.text(j,i,f'{data[i,j]:.1f}%' if col==0 else f'{data[i,j]:+.1f}',ha='center',va='center',color='white' if dark else '#222222')
            ax.set(title=f'{c}: {common//1000000}M win rate' if col==0 else f'{c}: change from 1M (pp)',xticks=range(len(protocol['transfer']['maps'])),xticklabels=protocol['transfer']['maps'],yticks=range(len(labels)),yticklabels=labels,xlabel='Reserved map')
            fig.colorbar(im,ax=ax,shrink=.7,label='Wins / all games (%)' if col==0 else 'Percentage points')
    fig.suptitle('Every reserved-map / script cell · mean of all three training seeds')
    save(fig,out,'transfer-cells')
    for svg in out.glob('*.svg'):svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(dict(contrasts=contrasts,crossplay={m:{k:v for k,v in c.items() if k!='cells'} for m,c in cross.items()}),indent=2))


if __name__=='__main__':main()
