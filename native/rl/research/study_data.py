"""Compact read-only views of nested opponent experiments for the dashboard."""
from functools import lru_cache
import json
import os
from pathlib import Path
import time


@lru_cache(maxsize=256)
def _json(path,modified,size):
    return json.loads(Path(path).read_text())


def read(path,default=None):
    path=Path(path)
    try:
        stat=path.stat();return _json(str(path),stat.st_mtime_ns,stat.st_size)
    except FileNotFoundError:return default


@lru_cache(maxsize=32)
def _metrics(path,modified,size):
    lines=Path(path).read_text().splitlines();rows=[]
    for index,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except ValueError:
            if index!=len(lines)-1:raise
    return rows


def metrics(path):
    try:
        stat=path.stat();return _metrics(str(path),stat.st_mtime_ns,stat.st_size)
    except FileNotFoundError:return []


def outcome(game):
    win=bool(game['win'] and not game['loss'] and not game['timeout'])
    loss=bool(game['loss'] and not game['win'] and not game['timeout'])
    return win,loss,not win and not loss


def aggregate(report):
    if not report:return None
    games=report['games'];counts=[0,0,0];opponents={}
    for game in games:
        values=outcome(game);counts=[x+int(v) for x,v in zip(counts,values)]
        desc=game['opponent'];name=desc.get('spec',{}).get('name',desc.get('kind','opponent'))
        cell=opponents.setdefault((name,game['map']),dict(opponent=name,map=game['map'],games=0,wins=0,losses=0,draws=0,hits=0,deaths=0))
        cell['games']+=1;cell['wins']+=values[0];cell['losses']+=values[1];cell['draws']+=values[2]
        cell['hits']+=game['behavior']['totals']['arrow_hits'];cell['deaths']+=game['deaths']
    return dict(games=len(games),wins=counts[0],losses=counts[1],draws=counts[2],
                win_rate=counts[0]/len(games) if games else 0,
                hits=sum(g['behavior']['totals']['arrow_hits'] for g in games),
                shots=sum(g['behavior']['totals']['arrows_fired'] for g in games),
                deaths=sum(g['deaths'] for g in games),
                cells=sorted(opponents.values(),key=lambda c:(c['map'],c['opponent'])))


def catalog(root,extra):
    entries={name:root/name for name in ('opponent-league','opponent-extension') if (root/name/'protocol.json').is_file()}
    entries.update(extra)
    return {name:path for name,path in entries.items() if (path/'protocol.json').is_file()}


def status(directory,source_steps=0):
    s=dict(read(directory/'status.json',{'status':'queued','learner_steps':source_steps}))
    if s['status']=='running':
        try:os.kill(s['pid'],0)
        except (ProcessLookupError,KeyError):s['status']='interrupted'
    return s


def payload(name,directory,root):
    protocol=read(directory/'protocol.json');extension='total_learner_target' in protocol
    parent=root/'opponent-league' if extension else directory
    target=protocol.get('total_learner_target',protocol.get('learner_decisions',1000000))
    fallback=read(directory/'fallback-checkpoint.json',{}) if extension else {}
    rows=[]
    for run in sorted(protocol['runs'],key=lambda r:(r['condition'],r['seed'])):
        path=directory/'runs'/run['name'];source=parent/'runs'/run['name']
        retry=directory/('retry-'+run['name'])
        if extension and (retry/'status.json').exists():path=retry
        old_metrics=metrics(source/'metrics.jsonl') if extension else []
        current=metrics(path/'metrics.jsonl');all_metrics=old_metrics+current
        s=status(path,run.get('source_learner_steps',0));latest=current[-1] if current else old_metrics[-1] if old_metrics else {}
        checkpoints={}
        milestones=(1000000,2000000,3000000,4000000) if extension else (1000000,)
        for milestone in milestones:
            if milestone==1000000:
                base=parent/'evaluations'/run['name'];dev=read(base/'development/result.json');transfer=read(base/'final/result.json')
            else:
                dev=read(directory/'evaluations'/f"{run['name']}-{milestone}-development"/'result.json')
                transfer=read(directory/'evaluations'/f"{run['name']}-{milestone}-transfer"/'result.json')
            checkpoints[str(milestone)]=dict(development=aggregate(dev),transfer=aggregate(transfer),unavailable=milestone>s.get('learner_steps',0) and s['status'] in ('failed','interrupted'))
        curve=[];hits=shots=env_deaths=self_deaths=combat_deaths=0
        stride=max(1,len(all_metrics)//180)
        for index,m in enumerate(all_metrics):
            totals=m.get('behavior',{}).get('totals',{});hits+=totals.get('arrow_hits',0);shots+=totals.get('arrows_fired',0)
            env_deaths+=totals.get('deaths_environment',0);self_deaths+=totals.get('deaths_self',0);combat_deaths+=totals.get('deaths_opponent',0)
            if index%stride==0 or index==len(all_metrics)-1:
                curve.append(dict(steps=m.get('learner_steps',m['step']*.75),hits=hits,shots=shots,
                                  environment_deaths=env_deaths,self_deaths=self_deaths,opponent_deaths=combat_deaths,
                                  sps=m['sps']*m.get('learner_fraction',.75),entropy=m.get('entropy')))
        last_file=path/'metrics.jsonl';age=time.time()-last_file.stat().st_mtime if last_file.exists() else None
        rows.append(dict(name=run['name'],condition=run['condition'],seed=run['seed'],status=s['status'],
                         learner_steps=s.get('learner_steps',latest.get('learner_steps',run.get('source_learner_steps',0))),target=target,
                         source_learner_steps=run.get('source_learner_steps',0),
                         sps=latest.get('sps',0)*latest.get('learner_fraction',.75) if s['status']=='running' else None,
                         elapsed=s.get('elapsed',0),update_age_seconds=age,
                         failure=s.get('failure'),action_rates=latest.get('action_rates',{}),
                         kl=latest.get('kl'),entropy=latest.get('entropy'),
                         curve=curve,checkpoints=checkpoints,
                         config_url=f'/studies/{name}/runs/{run["name"]}/config.json' if (path/'config.json').exists() else None))
    crossplays={}
    if extension:
        for milestone in sorted(set(protocol['crossplay']['milestones']+[fallback['milestone']] if fallback else protocol['crossplay']['milestones'])):
            cells=[];a_wins=b_wins=draws=games=0
            for a in protocol['crossplay']['seeds']:
                for b in protocol['crossplay']['seeds']:
                    r=aggregate(read(directory/'evaluations'/f'crossplay-{milestone}-A{a}-B{b}'/'result.json'))
                    unavailable=any(row['name'] in (f'A-s{a}',f'B-s{b}') and row['checkpoints'][str(milestone)]['unavailable'] for row in rows)
                    cells.append(dict(a=a,b=b,result=r,unavailable=unavailable))
                    if r:a_wins+=r['wins'];b_wins+=r['losses'];draws+=r['draws'];games+=r['games']
            crossplays[str(milestone)]=dict(cells=cells,games=games,expected_games=864,A_wins=a_wins,B_wins=b_wins,draws=draws,complete=games==864,unavailable=any(c['unavailable'] for c in cells))
    training=read(directory/'training-status.json',{}) if extension else read(directory/'series-status.json',{})
    evaluating=read(directory/'evaluation-status.json',{}) if extension else read(directory/'parallel-status.json',{})
    state='complete' if all(r['status']=='complete' for r in rows) and evaluating.get('status')=='complete' else 'evaluating' if all(r['status']=='complete' for r in rows) else 'training'
    if any(r['status'] in ('failed','interrupted') for r in rows) or training.get('status')=='failed' or evaluating.get('status')=='failed':state='needs attention'
    if any(r['status']=='running' for r in rows):state='training'
    elif training.get('status')=='complete_with_failures':state='finished with an interrupted run' if evaluating.get('status','').startswith('complete') else 'evaluating available checkpoints'
    active_games=0
    for job in evaluating.get('active',[]):
        active_games+=sum(p.name not in ('result.json','schedule.json') for p in (directory/'evaluations'/job['name']).glob('*.json'))
    completed_games=sum(j.get('games',0) for j in evaluating.get('completed',[]))
    clips=[]
    candidates=[('crossplay-1m','1M · A seed 73 vs B seed 73','Yellow: A · Purple: B · Map 0'),
                ('crossplay-3m','3M · A seed 73 vs B seed 73','Yellow: A · Purple: B · Map 0'),
                ('crossplay-4m','4M · A seed 73 vs B seed 73','Yellow: A · Purple: B · Map 0'),
                ('b-archer-failure','1M · B seed 73 vs archer','Purple: B · Yellow: script · Map 1'),
                ('b-evasive-failure','1M · B seed 73 vs evasive','Yellow: B · Purple: script · Map 1')]
    if not extension:candidates=[(f'{c}-s73',f'{c} seed 73 · 1M vs archer','Yellow: learner · Purple: script · Map 0') for c in 'ABC']
    for stem,title,detail in candidates:
        if (directory/'videos'/(stem+'.mp4')).exists():
            clips.append(dict(title=title,detail=detail,url=f'/studies/{name}/videos/{stem}.mp4',poster=f'/studies/{name}/videos/{stem}.jpg'))
    failure=read(directory/'dashboard/failure-summary.json')
    failure_final=read(directory/'dashboard/failure-summary-4m.json')
    retention=[]
    if not extension:
        for r in rows:
            value=read(directory/'retention'/r['name']/'result.json')
            if value:retention.append(dict(run=r['name'],successes=value['successes'],trials=value['trials']))
    links=[dict(label='Frozen protocol',url=f'/studies/{name}/protocol.json')]
    for file,label in [('dashboard/REPORT.md','Completed report'),('dashboard/overview.png','Comparison figure'),('dashboard/failure-overview.png','Failure measurements')]:
        if (directory/file).exists():links.append(dict(label=label,url=f'/studies/{name}/{file}'))
    return dict(name=name,kind='extension' if extension else 'league',title='A/B continuation' if extension else 'Scripted opponents + Bow initialization',
                status=state,runs=rows,total_learner_steps=sum(r['learner_steps'] for r in rows),total_target=len(rows)*target,
                training=training,evaluation=dict(status=evaluating.get('status','queued'),completed_suites=len(evaluating.get('completed',[])),
                expected_suites=evaluating.get('expected_suites',42 if extension else 18),completed_games=completed_games,active_games=active_games,
                expected_games=evaluating.get('expected_games',6624 if extension else 4752),unavailable_games=sum(j.get('games',0) for j in evaluating.get('unavailable',[])),error=evaluating.get('error')),
                crossplay=crossplays,fallback=fallback,incidents=read(directory/'training-incidents.json',{}),clips=clips,failure_groups=failure['groups'] if failure else None,
                failure_groups_4m=failure_final['groups'] if failure_final else None,retention=retention,links=links,
                training_maps=protocol['training_maps'],transfer_maps=protocol.get('transfer',protocol.get('final',{})).get('maps',[]),updated_at=time.time())
