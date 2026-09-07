"""Render the initial and most recent evaluation while a run is active.

Coalesces intermediate checkpoints if rendering falls behind; native .rep and
physics replays remain available for every completed evaluation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from common import atomic_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); run=args.run.resolve(); rendered=[]
    while True:
        results=sorted(run.glob('evaluations/*/result.json'))
        candidates=[]
        for file in results:
            report=json.loads(file.read_text()); game=next(g for g in report['games'] if g['name']=='selfplay')
            video=file.parent/'selfplay.mp4'
            if rendered and report['step']<=max(rendered): continue
            if not video.exists(): candidates.append((report['step'],file.parent,game,video))
        status=json.loads((run/'status.json').read_text()).get('status') if (run/'status.json').exists() else 'starting'
        if candidates:
            # Preserve the beginning, then keep the latest available policy.
            step,directory,game,video=candidates[0] if not rendered else candidates[-1]
            seconds=game['ticks']/60+6
            atomic_json(run/'video-status.json',dict(status='rendering',step=step,rendered=rendered))
            subprocess.run([sys.executable,str(Path(__file__).with_name('render_video.py')),
                            str(directory/game['replay']),str(video),'--seconds',str(seconds)],check=True)
            rendered.append(step)
            if status in ('complete','stopped','failed') and step==max(r[0] for r in candidates): break
        elif status in ('complete','stopped','failed'): break
        else: time.sleep(3)
    atomic_json(run/'video-status.json',dict(status='complete',rendered=rendered))

if __name__=='__main__': main()
