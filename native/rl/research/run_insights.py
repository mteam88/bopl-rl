"""Replay-derived behavior insights for the dashboard, cached per evaluation.

Computed from the saved games' physics JSONs (decision-level actions for both
players), never from live game processes. Results are written once per
completed evaluation directory as insights.json; requests only pay for at most
one uncached directory, so history backfills progressively while a run stays
responsive.
"""
import json
import math
from pathlib import Path
import threading

SCHEMA=2
ARROW_BIT=2
# World-coordinate occupancy grid: 2 game units per cell.
GRID_X0,GRID_Y0,GRID_W,GRID_H,GRID_CELL=-60,-36,60,36,2
_lock=threading.Lock()


def circular_degrees(a,b):
    delta=abs(a-b)%256
    return min(delta,256-delta)*360/256


def median(values):
    if not values: return None
    ordered=sorted(values); n=len(ordered)
    middle=ordered[n//2]
    return (ordered[n//2-1]+middle)/2 if n%2==0 else middle


def game_insights(visual,side):
    """Gauge samples and per-map heatmap points for one saved game."""
    frames=visual['frames']
    holds=[]; run_length=0; aim_changes=[]; previous_angle=None
    positions=[]; deaths=[]
    for index,frame in enumerate(frames):
        action=frame.get('actions')
        if action:
            angle,buttons=action[side]
            if previous_angle is not None:
                aim_changes.append(circular_degrees(angle,previous_angle))
            previous_angle=angle
            if buttons>>ARROW_BIT&1: run_length+=1
            elif run_length: holds.append(run_length/15); run_length=0
        me=frame['players'][side]
        if me['flags']&1: positions.append((me['x'],me['y']))
        if index:
            before=frames[index-1]['players'][side]
            if me['deaths']>before['deaths']:
                deaths.append([round(before['x'],1),round(before['y'],1)])
    # A hold still running at the end is censored, not completed.
    return holds,aim_changes,positions,deaths


def evaluate_directory(directory):
    """Compute insights for one evaluation directory (result.json required)."""
    report=json.loads((directory/'result.json').read_text())
    holds=[]; aim_changes=[]; games=0
    maps={}
    for game in report['games']:
        if not game.get('visual'): continue
        path=directory/game['visual']
        if not path.is_file(): continue
        visual=json.loads(path.read_text())
        game_holds,game_aim,positions,deaths=game_insights(visual,game['side'])
        holds+=game_holds; aim_changes+=game_aim; games+=1
        cell=maps.setdefault(str(game['map']),dict(occupancy={},deaths=[],games=0,outline=None))
        cell['games']+=1
        cell['deaths']+=deaths
        for x,y in positions:
            column=min(GRID_W-1,max(0,int((x-GRID_X0)/GRID_CELL)))
            row=min(GRID_H-1,max(0,int((y-GRID_Y0)/GRID_CELL)))
            key=row*GRID_W+column
            cell['occupancy'][key]=cell['occupancy'].get(key,0)+1
        if cell['outline'] is None:
            cell['outline']=[dict(x=e['x'],y=e['y'],rx=e['rx'],ry=e['ry'],ux=e['ux'],uy=e['uy'],radius=e['radius'])
                             for e in visual['frames'][0]['entities']
                             if e['kind'] in (1,2) and not e['owner'] and not e['flags']&8]
    for cell in maps.values():
        cell['occupancy']=[[int(k),v] for k,v in sorted(cell['occupancy'].items())]
    return dict(schema=SCHEMA,step=report['step'],games=games,
                grid=dict(x0=GRID_X0,y0=GRID_Y0,w=GRID_W,h=GRID_H,cell=GRID_CELL),
                gauges=dict(bow_hold_seconds=median(holds),completed_holds=len(holds),
                            aim_change_degrees=median(aim_changes),decisions=len(aim_changes)),
                maps=maps)


def collect(run_directory,budget=1):
    """Insights for every evaluation of a run, newest first.

    Uses insights.json sidecars; computes at most `budget` missing directories
    per call (newest first) so one request never scans a whole long run.
    """
    rows=[]
    with _lock:
        for result in sorted(run_directory.glob('evaluations/*/result.json'),reverse=True):
            directory=result.parent
            sidecar=directory/'insights.json'
            try:
                value=json.loads(sidecar.read_text())
                if value.get('schema')!=SCHEMA: raise ValueError
            except (OSError,ValueError):
                if budget<1:
                    continue
                budget-=1
                try: value=evaluate_directory(directory)
                except (OSError,ValueError,KeyError): continue
                temporary=sidecar.with_suffix('.json.tmp')
                try:
                    temporary.write_text(json.dumps(value)+'\n'); temporary.replace(sidecar)
                except OSError: pass
            rows.append(dict(value,directory=directory.name))
    rows.sort(key=lambda r:r['step'])
    return rows
