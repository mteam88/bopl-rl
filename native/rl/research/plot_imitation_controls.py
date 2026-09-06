"""Recipe iteration plots from fixed experiment artifacts; no model evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
read=lambda path:json.loads((args.root/path).read_text())
fits={name:read(name+'/result.json') for name in ['zero','previous','delta15','full30','delta30','balanced30','delta-one']}
full15=json.loads((args.root.parent/'imitation/thirtyseven-replay/result.json').read_text())
scores={name:read(name+'-score.json') for name in ['delta30','balanced30']}
evals={name:read(name+'/result.json') for name in ['eval15','eval30','eval-balanced','scripted']}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
ax=axes[0,0];labels=['Full state','No inputs','Previous only','Predict changes']
values=[full15['final']['validation']['joint_accuracy'],*[fits[n]['final']['validation']['joint_accuracy'] for n in ['zero','previous','delta15']]]
ax.bar(labels,np.array(values)*100,color=['#9b9b9b','#bfbfbf','#789cac','#156f82'])
ax.axhline(54.0138,color='#d87529',ls='--',label='Repeat previous: 54%')
for i,v in enumerate(values):ax.text(i,100*v+1,f'{v:.1%}',ha='center')
ax.set(ylim=(0,65),ylabel='Exact action accuracy (%)',title='A. Simple 15 Hz baselines expose poor transfer');ax.legend(frameon=False)
ax=axes[0,1];labels=['Absolute','Changes','Balanced changes']
values=[fits[n]['final']['validation']['joint_accuracy'] for n in ['full30','delta30','balanced30']]
ax.bar(labels,np.array(values)*100,color=['#9b9b9b','#156f82','#d87529'])
ax.axhline(63.9702,color='#555',ls='--',label='Repeat previous: 64%')
for i,v in enumerate(values):ax.text(i,100*v+1,f'{v:.1%}',ha='center')
ax.set(ylim=(0,75),ylabel='Exact action accuracy (%)',title='B. At 30 Hz, 64% can mean never press Bow');ax.legend(frameon=False,loc='upper left')
ax=axes[1,0];x=np.arange(2)
for i,event in enumerate(['bow_press','bow_release']):
 values=[scores[n]['events'][event]['recall']*100 for n in ['delta30','balanced30']]
 ax.bar(x+(i-.5)*.3,values,width=.3,label=event.replace('bow_','').title(),color=['#156f82','#d87529'][i])
ax.set(xticks=x,xticklabels=['Changes','Balanced changes'],ylim=(0,60),ylabel='Same-decision Bow event recall (%)',title='C. Weighting recovers some events, with many false calls');ax.legend(frameon=False)
ax=axes[1,1]
rows=[('Human 60 Hz',evals['eval30'],'human60'),('Human 30 Hz',evals['eval30'],'human30'),('30 Hz absolute',evals['eval30'],'full'),('30 Hz changes',evals['eval30'],'delta'),('Balanced',evals['eval-balanced'],'balanced'),('Scripted Bow',evals['scripted'],'point-hold-release')]
for i,(label,data,mode) in enumerate(rows):
 selected=[r for r in data['games'] if r['mode']==mode];hits=sum(r['arrow_hits'] for r in selected);shots=sum(r['arrows_fired'] for r in selected)
 ax.bar(i,hits,color='#156f82' if i<2 else '#d87529');ax.text(i,hits+.2,f'{hits}/{shots}',ha='center')
ax.set(xticks=range(len(rows)),xticklabels=['Human\n60 Hz','Human\n30 Hz','Absolute','Changes','Balanced','Scripted\nBow'],ylim=(0,16),ylabel='Arrow hits (label: hits/shots)',title='D. Eight development maps, real game control')
fig.suptitle('Control and prediction experiments: check behavior, not accuracy alone',fontsize=17,fontweight='bold')
fig.supxlabel('Same GRU and data split. Development matches are reused for iteration; this is not an independent final test.',fontsize=10)
for ext in ['png','svg','pdf']:fig.savefig(args.output/f'overview.{ext}',dpi=170)
summary=dict(plan=read('plan.json'),fits=fits,baseline=full15,scores=scores,evaluations=evals)
(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

# Inspect exact inputs immediately before the network, plus its predictions.
d=np.load(args.root/'balanced30-score-537-p2.npz');meta=json.loads(str(d['metadata']))
actions=d['target'];prediction=d['predicted'];observations=d['inputs']
press=np.flatnonzero((actions[:,1]&2)>0);idx=int(press[0]) if len(press) else 0
obs=observations[idx];tick=meta['origin']+idx*meta['repeat'];t=np.arange(len(actions))*meta['repeat']/60
fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained')
ax=axes[0];entities=obs[898:2914].reshape(96,21);active=entities[:,0]>.5
ax.scatter(entities[active,8]*64,entities[active,9]*64,s=20,color='#9b9b9b',label='Encoded entity centers')
ax.scatter([0],[0],s=90,color='#156f82',label='Self');ax.scatter([obs[10+148+9]*64],[obs[10+148+10]*64],s=90,color='#d87529',label='Opponent')
ax.set(xlabel='Relative x',ylabel='Relative y',title=f'Encoded world at tick {tick-9}; action target at tick {tick}');ax.legend(frameon=False,ncol=3)
ax=axes[1]
for array,label,color in [(actions,'Human target','#156f82'),(prediction,'Balanced model','#d87529')]:
 angle=(array[:,0].astype(float)-1)*360/255;angle[array[:,0]==0]=np.nan;ax.plot(t,angle,label=label,color=color,lw=1)
ax.set(ylabel='Nonneutral stick angle (degrees)',xlabel='Seconds after startup',title='Exact target and prediction on replay 537');ax.legend(frameon=False)
ax=axes[2]
for i,bit in enumerate([0,1,2,3]):
 for array,color,offset in [(actions,'#156f82',0),(prediction,'#d87529',.14)]:
  ax.step(t,i+((array[:,1]&(1<<bit))>0)*.55+offset,where='post',color=color,lw=1)
ax.set(yticks=np.arange(4)+.25,yticklabels=['Jump','Bow','Platform','Dash'],xlabel='Seconds after startup',title='Button holds: human target (blue), model (orange)')
fig.suptitle('Inspect the actual network input and target',fontsize=18,fontweight='bold')
for ext in ['png','svg','pdf']:fig.savefig(args.output/f'prediction-inspection.{ext}',dpi=170)
