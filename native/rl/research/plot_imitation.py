"""Plot the small demonstration experiment from its recorded results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--completed-only',action='store_true')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads((args.run/p).read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
if not args.completed_only:
    one=read('one-replay-2000/result.json');six=read('six-replay/result.json')
    loop=read('closed-loop/result.json');cadence=read('cadence/result.json')
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axs[0,0]
    for folder,label,color in [('one-replay-2000','One replay','#156f82'),('six-replay','Six replays','#d87529')]:
        rows=[json.loads(x) for x in (args.run/folder/'metrics.jsonl').read_text().splitlines()]
        ax.plot([x['epoch'] for x in rows],[x['train']['loss'] for x in rows],color=color,label=label+' / train')
        ax.plot([x['epoch'] for x in rows],[x['validation']['loss'] for x in rows],color=color,ls='--',label=label+' / held out')
    ax.set(yscale='log',xlabel='Optimization steps',ylabel='Action cross-entropy',title='A. It can memorize; transfer remains poor')
    ax.legend(fontsize=9,frameon=False)
    ax=axs[0,1]
    values=[one['final']['train']['joint_accuracy'],six['final']['train']['joint_accuracy'],six['final']['validation']['joint_accuracy'],six['persistence_baseline']['validation']['joint_accuracy']]
    labels=['One replay\ntrain','Six replays\ntrain','Six replays\nheld out','Repeat previous\nheld out']
    ax.bar(labels,np.array(values)*100,color=['#156f82','#d87529','#d87529','#9b9b9b'])
    for i,v in enumerate(values):ax.text(i,v*100+2,f'{v:.1%}',ha='center')
    ax.set(ylim=(0,115),ylabel='Exact joint action accuracy (%)',title='B. Check against a simple persistence baseline')
    ax=axs[1,0]
    for i,(folder,label,color) in enumerate([('human60','Human 60 Hz','#156f82'),('human15','Human 15 Hz','#789cac'),('one','One replay fit','#d87529'),('six','Six replay fit','#deac86'),('initial','Initial GRU','#9b9b9b')]):
        rows=[r for r in loop['games'] if r['mode']==folder]
        hits=sum(r['arrow_hits'] for r in rows);shots=sum(r['arrows_fired'] for r in rows)
        ax.bar(i,hits,color=color);ax.text(i,hits+.15,f'{hits}/{shots}',ha='center')
    ax.set(xticks=range(5),xticklabels=['Human\n60 Hz','Human\n15 Hz','One replay\nfit','Six replay\nfit','Initial\nGRU'],ylim=(0,9.5),ylabel='Arrow hits (label: hits/shots)',title='C. Four matches with the policy controlling the game')
    ax=axs[1,1]
    variants=[('human60','60 Hz'),('human30','30 Hz'),('human15','15 Hz'),('aim15','Stick only\n15 Hz'),('buttons15','Buttons only\n15 Hz'),('human15phase3','15 Hz\nshifted 3 ticks')]
    for i,(variant,label) in enumerate(variants):
        hits=sum(r['arrow_hits'] for r in cadence['games'] if r['variant']==variant)
        ax.bar(i,hits,color='#156f82' if i<2 else '#9b9b9b');ax.text(i,hits+.15,str(hits),ha='center')
    ax.set(xticks=range(len(variants)),xticklabels=[x[1] for x in variants],ylim=(0,9.5),ylabel='Arrow hits across four replays',title='D. Resampling changes behavior before learning')
    fig.suptitle('Human imitation: the first eight solo recordings',fontsize=19,fontweight='bold')
    fig.supxlabel('Same 128-unit GRU. Recorded inputs for the idle opponent. Small diagnostic, not a competitive benchmark.',fontsize=10)
    for extension in ('png','svg','pdf'):fig.savefig(args.output/f'overview.{extension}',dpi=170)
    summary=dict(one_replay=one,six_replays=six,closed_loop=loop,cadence=cadence,parity=read('parity/result.json'))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

# Optional continuation after the user finished the recording session.
if (args.run/'completed-closed-loop/result.json').exists():
    old=read('six-replay-held10.json');new=read('thirtyseven-replay-held10.json')
    completed=read('completed-closed-loop/result.json')
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    labels=['6 replays','37 replays','Repeat\nprevious']
    values=[old['model']['joint_accuracy'],new['model']['joint_accuracy'],new['persistence']['joint_accuracy']]
    ax=axes[0];ax.bar(labels,np.array(values)*100,color=['#789cac','#156f82','#9b9b9b'])
    for i,v in enumerate(values):ax.text(i,v*100+1.2,f'{v:.1%}',ha='center')
    ax.set(ylim=(0,65),ylabel='Exact joint action accuracy (%)',title='A. Same 10 held-out matches')
    ax=axes[1]
    values=[old['model']['nonneutral_direction_mae_degrees'],new['model']['nonneutral_direction_mae_degrees'],new['persistence']['nonneutral_direction_mae_degrees']]
    ax.bar(labels,values,color=['#789cac','#156f82','#9b9b9b'])
    for i,v in enumerate(values):ax.text(i,v+1,f'{v:.1f}°',ha='center')
    ax.set(ylim=(0,51),ylabel='Mean circular error, both sticks nonneutral',title='B. Direction prediction improves')
    ax=axes[2]
    labels=['Human\n60 Hz','Human\n15 Hz','6 replay\nmodel','37 replay\nmodel']
    for i,mode in enumerate(['human60','human15','six','thirtyseven']):
        rows=[r for r in completed['games'] if r['mode']==mode]
        hits=sum(r['arrow_hits'] for r in rows);shots=sum(r['arrows_fired'] for r in rows)
        ax.bar(i,hits,color=['#156f82','#789cac','#deac86','#d87529'][i]);ax.text(i,hits+.2,f'{hits}/{shots}',ha='center')
    ax.set(xticks=range(4),xticklabels=labels,ylim=(0,16),ylabel='Arrow hits (label: hits/shots)',title='C. Control on 8 held-out maps')
    fig.suptitle('Completed batch: 47 controller recordings, 47 maps',fontsize=18,fontweight='bold')
    fig.supxlabel('37 training matches; 10 excluded from training. Same session and human. Fixed opponent inputs; argmax policies.',fontsize=10)
    for extension in ('png','svg','pdf'):fig.savefig(args.output/f'completed-overview.{extension}',dpi=170)
    complete_summary=dict(plan=read('completed-batch-plan.json'),recordings=read('completed-recordings.json'),
        old_model_new39=read('six-replay-new39.json'),old_model_held10=old,new_model_held10=new,
        new_fit=read('thirtyseven-replay/result.json'),closed_loop=completed)
    (args.output/'completed-summary.json').write_text(json.dumps(complete_summary,indent=2)+'\n')
