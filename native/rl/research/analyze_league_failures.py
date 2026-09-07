"""Describe fixed failure traces without treating input statistics as causal proof."""
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def quantiles(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,
                p10=float(np.quantile(a,.1)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None)


def held_lengths(buttons):
    lengths=[];run=0
    for held in buttons&4:
        if held:run+=4
        elif run:lengths.append(run);run=0
    if run:lengths.append(run)
    return lengths


def analyze_player(side,frames,actions,events):
    shot_rows=[];charge_times=[];last_activation=None;cancelled=0
    for i,event in enumerate(events[:,side]):
        if event[1]:last_activation=frames[i+1]['t']
        if event[3]:
            if last_activation is not None:
                charge_times.append(frames[i+1]['t']-last_activation);last_activation=None
            previous_ids={e['id'] for e in frames[i]['entities']}
            arrows=[e for e in frames[i+1]['entities'] if e['owner']==side+1 and e['layer'] in (9,14) and e['id'] not in previous_ids]
            row=dict(decision=i,tick=frames[i+1]['t'],projectiles_observed=len(arrows),speed=None,heading_error_degrees=None)
            if len(arrows)==1:
                arrow=arrows[0];target=frames[i+1]['players'][1-side]
                velocity=np.array([arrow['vx'],arrow['vy']]);delta=np.array([target['x']-arrow['x'],target['y']-arrow['y']])
                speed=float(np.linalg.norm(velocity));row.update(speed=speed,projectile_id=arrow['id'],distance=float(np.linalg.norm(delta)))
                if target['flags']&1 and speed>.1 and np.linalg.norm(delta)>.1:
                    row['heading_error_degrees']=math.degrees(math.acos(float(np.clip(velocity@delta/(speed*np.linalg.norm(delta)),-1,1))))
            shot_rows.append(row)
        if event[5:9].sum() and last_activation is not None:
            cancelled+=1;last_activation=None
    angles=actions[:,side,0];held=(actions[:,side,1]&4)>0
    turns=[]
    for i in range(1,len(actions)):
        if held[i] and held[i-1] and angles[i] and angles[i-1]:
            d=abs(int(angles[i])-int(angles[i-1]));turns.append(min(d,255-d)*360/255)
    durations=held_lengths(actions[:,side,1]);totals=events[:,side].sum(0)
    distances=[math.hypot(f['players'][0]['x']-f['players'][1]['x'],f['players'][0]['y']-f['players'][1]['y']) for f in frames if all(p['flags']&1 for p in f['players'])]
    speeds=[r['speed'] for r in shot_rows if r['speed'] is not None]
    errors=[r['heading_error_degrees'] for r in shot_rows if r['heading_error_degrees'] is not None]
    return dict(shots=int(totals[3]),hits=int(totals[4]),bow_activations=int(totals[1]),
                environment_deaths=int(totals[5]),opponent_deaths=int(totals[6]),self_deaths=int(totals[7]),other_deaths=int(totals[8]),
                bow_input_hold_ticks=durations,hold_summary=quantiles(durations),
                hold_runs_at_most_8_ticks=sum(d<=8 for d in durations),
                activation_to_fire_ticks=charge_times,activation_to_fire_summary=quantiles(charge_times),
                activation_interrupted_by_death=cancelled,projectile_speed_summary=quantiles(speeds),
                shot_rows=shot_rows,observed_projectile_speeds=speeds,
                heading_errors_degrees=errors,headings_over_90_degrees=sum(x>90 for x in errors),
                input_aim_change_while_bow_held=quantiles(turns),distance_summary=quantiles(distances))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-name',default='failure-review')
    parser.add_argument('--matched-later',action='store_true',help='Same cases at a later checkpoint, without reselecting losses')
    args=parser.parse_args();later_label=args.review_name.removeprefix('failure-review-').upper();root=args.artifacts/args.review_name;out=args.output;out.mkdir(parents=True,exist_ok=True)
    review=json.loads((root/'result.json').read_text());rows=[];plots=[]
    for case in review['cases']:
        directory=root/(case['run']+'-'+case['case']);data=np.load(directory/'decisions.npz')
        frames=json.load(gzip.open(directory/'frames.json.gz','rt'))
        learner=analyze_player(case['side'],frames,data['actions'],data['events'])
        opponent=analyze_player(1-case['side'],frames,data['actions'],data['events'])
        rows.append(dict(**case,learner=learner,opponent=opponent))
        if case['run'].startswith('B'):plots.append((case,frames,data['events'],learner))
    groups={}
    for condition in ('A','B','script'):
        members=[r['learner'] for r in rows if r['run'].startswith(condition)] if condition!='script' else [r['opponent'] for r in rows]
        holds=[v for r in members for v in r['bow_input_hold_ticks']];speeds=[v for r in members for v in r['observed_projectile_speeds']];errors=[v for r in members for v in r['heading_errors_degrees']]
        groups[condition]=dict(cases=len(members),**{k:sum(r[k] for r in members) for k in ('shots','hits','bow_activations','environment_deaths','opponent_deaths','self_deaths','other_deaths','hold_runs_at_most_8_ticks','headings_over_90_degrees')},
                               holds=quantiles(holds),projectile_speeds=quantiles(speeds),headings_measured=len(errors),
                               activation_to_fire=quantiles([v for r in members for v in r['activation_to_fire_ticks']]))
    summary=dict(schema=1,cases=rows,groups=groups,
                 scope=(f'The same 18 cases selected for B losses at 1M and their matched A cases, now at {later_label} regardless of outcome. No reselection; conditional diagnostic sample.' if args.matched_later else '18 selected B losses and their matched A cases, stratified by training seed, map and ranged style. Conditional diagnostic sample, not an unbiased strength estimate.'),
                 caveats=['Held input duration includes times outside active Bow; it is not exact charge duration.',
                          'Activation and shot events are resolved to four ticks.',
                          'Projectile speed is measured in the first state after a firing event, not exact muzzle speed. Missing projectiles are not automatically misses or blocked shots.',
                          'Heading error is to the current target center; leading, gravity, inherited velocity and collisions can matter. More than 90 degrees means initial motion away from that center, not proof of intent.',
                          'Telemetry and full current state are recorded for analysis only; policies receive the unchanged delayed observation.'])
    (out/'failure-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,key,title,ylabel in [(axes[0],'bow_input_hold_ticks','Held Bow inputs','Ticks per contiguous hold'),(axes[1],'observed_projectile_speeds','Observed new projectiles','World units / second')]:
        samples=[]
        for c in ('A','B','script'):
            members=[r['learner'] for r in rows if r['run'].startswith(c)] if c!='script' else [r['opponent'] for r in rows]
            samples.append([v for r in members for v in r[key]])
        ax.boxplot(samples,tick_labels=['A','B','Scripts'],showfliers=False)
        ax.set(title=title,ylabel=ylabel)
    ax=axes[2];x=np.arange(2)
    environmental=np.array([groups[c]['environment_deaths'] for c in 'AB']);combat=np.array([groups[c]['opponent_deaths'] for c in 'AB'])
    own=np.array([groups[c]['self_deaths'] for c in 'AB'])
    ax.bar(x,environmental,label='Environment',color='#bf7845');ax.bar(x,combat,bottom=environmental,label='Opponent',color='#627f98')
    ax.bar(x,own,bottom=environmental+combat,label='Self',color='#9b8aaf')
    ax.set(xticks=x,xticklabels=['A','B'],title='Deaths in matched selected cases',ylabel='Deaths');ax.legend()
    fig.suptitle(f'The same ranged-opponent cases at {later_label} · no loss reselection' if args.matched_later else 'Ranged-opponent failure review · 18 B losses + matched A cases')
    for ext in ('png','svg','pdf'):fig.savefig(out/f'failure-overview.{ext}',dpi=150)
    plt.close(fig)
    for m in (1,2,5):
        fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained')
        for case,frames,events,learner in plots:
            if case['map']!=m:continue
            row=(73,173,273).index(int(case['run'].split('s')[1]));col=0 if case['case'].startswith('archer') else 1
            ax=axes[row,col];side=case['side'];positions=np.array([[[p['x'],p['y']] for p in f['players']] for f in frames])
            for player,color in [(1-side,'#a1a9b2'),(side,'#cf7d2c')]:
                xy=positions[:,player];segments=np.stack([xy[:-1],xy[1:]],1)
                valid=np.linalg.norm(xy[1:]-xy[:-1],axis=1)<8
                ax.add_collection(LineCollection(segments[valid],colors=color,linewidths=.8,alpha=.7))
            shots=np.flatnonzero(events[:,side,3])+1;deaths=np.flatnonzero(events[:,side,5:9].sum(1))+1
            ax.scatter(*positions[shots,side].T,marker='^',s=22,color='#2f6c87',label='Own shot')
            ax.scatter(*positions[deaths,side].T,marker='x',s=55,color='#a63437',label='Own death')
            ax.scatter(*positions[0,side],marker='o',s=30,color='#cf7d2c',label='Learner start')
            ax.autoscale();ax.set_aspect('equal',adjustable='datalim')
            ax.set(title=f"{case['run']} · {'archer' if col==0 else 'evasive'} · {learner['shots']} shots / {learner['hits']} hits",xlabel='World x',ylabel='World y')
        axes[0,0].legend(fontsize=7)
        label=f'same selected B cases at {later_label}' if args.matched_later else 'selected B failures'
        fig.suptitle(f'Map {m}: {label} · orange learner / gray opponent\nPaths omit large respawn jumps; no platform geometry is inferred',fontsize=14)
        for ext in ('png','svg','pdf'):fig.savefig(out/f'failure-map-{m}.{ext}',dpi=130)
        plt.close(fig)
    for svg in out.glob('failure-*.svg'):svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
