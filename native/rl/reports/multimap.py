"""Per-map figures for the controlled multi-map experiment."""
import matplotlib.pyplot as plt
import numpy as np

COLORS=('#2846d4','#e87350','#71816a')


def style(axes):
    for ax in np.asarray(axes).flat:
        ax.set_facecolor('white');ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#e5e7eb',linewidth=.7);ax.set_axisbelow(True)


def save(fig,output,name):
    for suffix in ('png','svg'): fig.savefig(output/f'{name}.{suffix}',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)


def bars(ax,runs,keys,labels):
    width=.72/max(1,len(runs))
    for i,run in enumerate(runs):
        if not run['final']: continue
        by_key={(g['opponent'],g['map']):g for g in run['final']['groups']}
        groups=[by_key[key] for key in keys]
        values=np.array([g['win_rate'] for g in groups])*100
        bounds=np.array([g['pair_bootstrap_interval'] for g in groups])*100
        x=np.arange(len(groups))+(i-(len(runs)-1)/2)*width
        ax.bar(x,values,width=width*.88,color=COLORS[i%len(COLORS)],label=f"Seed {run['seed']}",
               yerr=[values-bounds[:,0],bounds[:,1]-values],error_kw=dict(elinewidth=1,capsize=2,ecolor='#344052'))
    ax.set_xticks(range(len(keys)),labels,fontsize=9)
    ax.set(ylim=(0,100),ylabel='Final win rate (%)')
    ax.axhline(50,color='#8b94a4',linestyle=':',linewidth=1)


def plot(summary,histories,output):
    runs=summary['runs'];maps=summary['evaluation_maps'];frozen=summary['frozen_opponents']
    names={p['name'] for p in frozen};budget=summary['budget']/1e6
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#182236',
                         'axes.labelcolor':'#354052','axes.edgecolor':'#cfd3d8'})
    fig,axes=plt.subplots(len(maps),2,figsize=(13.5,3*len(maps)+2),squeeze=False,facecolor='#f5f3ed')
    fig.subplots_adjust(top=.86,bottom=.1,left=.075,right=.975,hspace=.45,wspace=.23)
    fig.text(.075,.95,'LEARNING ACROSS MAPS',fontsize=22,weight='bold')
    fig.text(.075,.915,f"{len(runs)} training seeds · {budget:g}M decisions each · unchanged rewards and observations",fontsize=11)
    if summary['status']!='complete': fig.text(.975,.95,'PREVIEW · SERIES RUNNING',ha='right',color='#a1482b')
    style(axes)
    for row,map_id in enumerate(maps):
        ax=axes[row,0]
        for i,run in enumerate(runs):
            points=[]
            for item in run['diagnostics']:
                groups=[g for g in item['groups'] if g['map']==map_id and g['opponent'] in names]
                if groups: points.append((item['step']/1e6,100*sum(g['wins'] for g in groups)/sum(g['games'] for g in groups)))
            if points: ax.plot(*zip(*points),color=COLORS[i%3],marker='o',linewidth=2,label=f"Seed {run['seed']}")
        ax.set_title(f'Map {map_id} · against previous trained policies',loc='left',weight='bold',pad=12)
        ax.set(xlim=(0,max(.01,budget*1.02)),ylim=(0,100),xlabel='Training decisions (millions)',ylabel='Diagnostic mean win rate (%)')
        ax.axhline(50,color='#8b94a4',linestyle=':',linewidth=1)
        opponent_ids=['random','initial',*[p['name'] for p in frozen]]
        labels=['Random','Starting',*[p['name'].replace('trained-s','Old seed\n') for p in frozen]]
        bars(axes[row,1],runs,[(p,map_id) for p in opponent_ids],labels)
        axes[row,1].set_title(f'Map {map_id} · fresh final game seeds',loc='left',weight='bold',pad=12)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.068,.898),ncol=len(runs),frameon=False)
    fig.text(.075,.044,'Left: mean across the three frozen previous policies. Right: each opponent separately; 95% intervals resample whole side-swapped seed pairs.',fontsize=9)
    fig.text(.075,.024,'Timeouts count as non-wins. Training seeds remain separate. These comparisons do not measure human-level ability.',fontsize=9)
    save(fig,output,'overview')

    fig,axes=plt.subplots(1,2,figsize=(13.5,5.6),facecolor='#f5f3ed')
    fig.subplots_adjust(top=.76,bottom=.2,left=.075,right=.975,wspace=.25)
    fig.text(.075,.92,'TRANSFER & SURVIVAL',fontsize=22,weight='bold')
    if summary['status']!='complete': fig.text(.975,.92,'PREVIEW · SERIES RUNNING',ha='right',color='#a1482b')
    style(axes)
    bars(axes[0],runs,[('random',m) for m in summary['heldout_maps']],
         [f'Map {m}\nuntrained' for m in summary['heldout_maps']])
    axes[0].set_title('Against random play on unseen maps',loc='left',weight='bold',pad=12)
    for i,(run,rows) in enumerate(zip(runs,histories)):
        window=min(100,len(rows));kernel=np.ones(window)
        deaths=np.convolve([r['behavior']['totals']['deaths_environment'] for r in rows],kernel,'valid')
        minutes=np.convolve([r['behavior']['seconds']/60 for r in rows],kernel,'valid')
        axes[1].plot([r['step']/1e6 for r in rows[window-1:]],deaths/minutes,color=COLORS[i%3],linewidth=1.8,label=f"Seed {run['seed']}")
    axes[1].set_title('Environment deaths during training',loc='left',weight='bold',pad=12)
    axes[1].set(xlim=(0,max(.01,budget*1.02)),ylim=(0,None),xlabel='Training decisions (millions)',ylabel='Deaths / learner-minute')
    fig.legend(*axes[1].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.068,.87),ncol=len(runs),frameon=False)
    fig.text(.075,.065,'Held-out maps never enter training. Final intervals resample seed pairs. Survival curves use trailing 100-update windows.',fontsize=9)
    save(fig,output,'transfer')
