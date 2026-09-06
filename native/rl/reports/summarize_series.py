"""Summarize a saved series and draw portable figures without loading a policy.

Use a Python environment with NumPy and Matplotlib. The training environment is
not changed. Final intervals resample whole side-swapped game-seed pairs; they
describe evaluation uncertainty for each fixed policy, not training-seed variance.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GROUPS=(('random',0),('initial',0),('random',1),('random',2))
COLORS=('#2846d4','#e87350','#71816a')


def read(path):
    return json.loads(path.read_text())


def finite(value):
    if isinstance(value,dict): return all(map(finite,value.values()))
    if isinstance(value,list): return all(map(finite,value))
    return not isinstance(value,float) or math.isfinite(value)


def interval(games,opponent,map_id,rng):
    pairs={}
    for game in games:
        if (game['opponent'],game['map'])!=(opponent,map_id): continue
        pairs.setdefault(game['seed'],{})[game['side']]=float(game['win'])
    if not pairs or any(set(pair)!={0,1} for pair in pairs.values()):
        raise ValueError('Final evaluation must contain complete side-swapped pairs')
    values=np.array([sum(pair.values())/2 for pair in pairs.values()])
    samples=rng.choice(values,size=(5000,len(values)),replace=True).mean(axis=1)
    return np.quantile(samples,[.025,.975]).tolist()


def summarize(directory,allow_partial):
    plan=read(directory/'plan.json'); state=read(directory/'status.json')
    if state['status']!='complete' and not allow_partial:
        raise ValueError('Series has not completed; use --allow-partial only for a labeled preview')
    rng=np.random.default_rng(1949); runs=[]; histories=[]
    for member in plan['runs']:
        root=directory.parent/member['name']; file=root/'metrics.jsonl'
        rows=[]
        if file.exists():
            for index,line in enumerate(file.read_text().splitlines()):
                try: rows.append(json.loads(line))
                except ValueError:
                    if not allow_partial: raise
        if not rows:
            if allow_partial: continue
            raise ValueError(f'Missing training metrics for {member["name"]}')
        if not all(map(finite,rows)): raise ValueError('Nonfinite training metrics')
        errors=max(row['env'].get('game_errors',0) for row in rows)
        overflow=max(row['env'].get('observation_overflow',0) for row in rows)
        if errors or overflow: raise ValueError('Game errors or observation overflow')
        loop_seconds=sum(row['rollout_seconds']+row['train_seconds'] for row in rows)
        run=dict(**member,step=rows[-1]['step'],matches=rows[-1]['matches'],updates=len(rows),
                 learner_decisions=sum(row['behavior']['decisions'] for row in rows),
                 checks=dict(all_metrics_finite=True,game_errors=errors,observation_overflow=overflow),
                 loop_seconds=loop_seconds,loop_decisions_per_second=rows[-1]['step']/loop_seconds,
                 median_decisions_per_second=statistics.median(row['sps'] for row in rows),
                 diagnostics=[dict(step=report['step'],groups=report['groups'])
                              for p in sorted(root.glob('evaluations/*/result.json')) if (report:=read(p))],
                 native_videos=len(list(root.glob('evaluations/*/selfplay.mp4'))),
                 native_replays=len(list(root.glob('evaluations/*/*.rep')))+len(list(root.glob('final-evaluations/*/*.rep'))))
        result=root/'result.json'
        run['training_result']=read(result) if result.exists() else None
        if not allow_partial and (not run['training_result'] or run['training_result']['status']!='complete' or run['step']<plan['steps']):
            raise ValueError('Training budget did not complete')
        reports=sorted(root.glob('final-evaluations/*/result.json'))
        run['final']=None
        if reports:
            report=read(reports[-1]); groups=[]
            if report['step']!=run['step'] or report['baseline_sha256']!=plan['baseline_sha256']:
                raise ValueError('Final checkpoint or baseline identity differs from the plan')
            for group in report['groups']:
                games=[g for g in report['games'] if (g['opponent'],g['map'])==(group['opponent'],group['map'])]
                expected=plan['final_games'] if group['map']==0 else plan['final_heldout_games']
                if len(games)!=expected or len(games)!=group['games'] or sum(g['win'] for g in games)!=group['wins'] or sum(g['timeout'] for g in games)!=group['timeouts']:
                    raise ValueError('Final group does not match its per-game results and declared budget')
                groups.append(dict(**group,pair_bootstrap_interval=interval(report['games'],group['opponent'],group['map'],rng)))
            run['final']=dict(step=report['step'],groups=groups,checkpoint_sha256=report['checkpoint_sha256'],
                              baseline_sha256=report['baseline_sha256'],seed_base=report['seed_base'],
                              heldout_seed_base=report['heldout_seed_base'],elapsed=report['elapsed'])
        elif not allow_partial: raise ValueError('Missing final evaluation')
        runs.append(run); histories.append(rows)
    summary=dict(schema=1,status=state['status'],name=plan['name'],git_revision=plan['git_revision'],
                 reporting_versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
                 source_identity=plan['source_identity'],baseline_sha256=plan['baseline_sha256'],
                 elapsed=state['elapsed'],total_decisions=sum(run['step'] for run in runs),
                 total_learner_decisions=sum(run['learner_decisions'] for run in runs),
                 total_matches=sum(run['matches'] for run in runs),runs=runs,
                 interval_method='95% percentile bootstrap over side-swapped game-seed pairs, 5000 draws, RNG seed 1949; conditional on each fixed trained policy. Training-seed results are reported separately.')
    return summary,histories


def plot(summary,histories,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.labelcolor':'#354052','text.color':'#182236','axes.edgecolor':'#cfd3d8'})
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.7),facecolor='#f5f3ed')
    fig.subplots_adjust(top=.80,bottom=.14,left=.075,right=.975,hspace=.42,wspace=.25)
    fig.text(.075,.94,'DASH / ARROW / PLATFORM',fontsize=22,weight='bold')
    fig.text(.075,.9,'Three training seeds · 10 million decisions each · unchanged rewards and observations',fontsize=11)
    if summary['status']!='complete': fig.text(.975,.94,'PREVIEW · SERIES RUNNING',ha='right',fontsize=10,color='#a1482b')
    for ax in axes.flat:
        ax.set_facecolor('#ffffff'); ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#e5e7eb',linewidth=.7); ax.set_axisbelow(True)
    for i,(run,rows) in enumerate(zip(summary['runs'],histories)):
        color=COLORS[i%len(COLORS)]; label=f'Seed {run["seed"]}'
        for col,opponent in enumerate(('initial','random')):
            points=[(item['step']/1e6,next(g['win_rate'] for g in item['groups'] if g['opponent']==opponent and g['map']==0)*100) for item in run['diagnostics']]
            axes[0,col].plot(*zip(*points),marker='o',linewidth=2.2,markersize=5,color=color,label=label)
        window=min(100,len(rows)); kernel=np.ones(window)
        deaths=np.convolve([row['behavior']['totals']['deaths_environment'] for row in rows],kernel,'valid')
        minutes=np.convolve([row['behavior']['seconds']/60 for row in rows],kernel,'valid')
        axes[1,0].plot([row['step']/1e6 for row in rows[window-1:]],deaths/minutes,color=color,linewidth=1.8)
        if run['final']:
            groups=[next(g for g in run['final']['groups'] if (g['opponent'],g['map'])==key) for key in GROUPS]
            values=np.array([g['win_rate']*100 for g in groups]); bounds=np.array([g['pair_bootstrap_interval'] for g in groups])*100
            axes[1,1].bar(np.arange(4)+(i-1)*.24,values,width=.21,color=color,
                          yerr=[values-bounds[:,0],bounds[:,1]-values],error_kw=dict(elinewidth=1,capsize=2,ecolor='#344052'))
    for col,title in enumerate(('Against the starting policy','Against random play')):
        ax=axes[0,col];ax.set_title(title,loc='left',weight='bold',pad=12)
        ax.set(xlabel='Training decisions (millions)',ylabel='Diagnostic win rate (%)',xlim=(0,10.2),ylim=(0,100))
        ax.set_xticks([0,1,3,5,10]);ax.axhline(50,color='#8b94a4',linestyle=':',linewidth=1)
    axes[1,0].set_title('Learning to survive',loc='left',weight='bold',pad=12)
    axes[1,0].set(xlabel='Training decisions (millions)',ylabel='Environment deaths / player-minute',xlim=(0,10.2),ylim=(0,None))
    axes[1,1].set_title('Final evaluation on fresh game seeds',loc='left',weight='bold',pad=12)
    axes[1,1].set(ylim=(0,100),ylabel='Final win rate (%)')
    axes[1,1].set_xticks(range(4),['Random\ntrained map','Starting policy\ntrained map','Random\nuntrained 1','Random\nuntrained 2'],fontsize=8)
    axes[1,1].axhline(50,color='#8b94a4',linestyle=':',linewidth=1)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.068,.883),ncol=3,frameon=False)
    fig.text(.075,.063,'Diagnostics: 64 games per opponent. Final: 256 per trained-map opponent, 64 per untrained map. Timeouts count as non-wins.',fontsize=9)
    fig.text(.075,.04,'Final error bars resample side-swapped seed pairs (95%). Survival curves use trailing 100-update windows. These are weak fixed opponents, not human benchmarks.',fontsize=8.5)
    fig.savefig(output/'overview.png',dpi=160,facecolor=fig.get_facecolor())
    fig.savefig(output/'overview.svg',facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    summary,histories=summarize(args.series.resolve(),args.allow_partial)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plot(summary,histories,args.output)
    print(json.dumps(dict(status=summary['status'],output=str(args.output),seeds=[r['seed'] for r in summary['runs']])))


if __name__=='__main__': main()
