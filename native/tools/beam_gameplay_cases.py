"""Generate six-player Beam replay setups for the shared C# replay codec.

Held input requests emission; short presses request charge cancellation.
Player-5/6 Beam exits can fail on missing charge sounds in the pinned game.
The four-beams variant keeps six players and gives players 5/6 idle Dash slots. These are
input scripts, not injected beam bodies or modified gameplay state.
"""
import argparse
import json
from pathlib import Path
from generate_full_scenarios import scenario


def generate(directory, ticks=900, stocks=5):
    directory.mkdir(parents=True, exist_ok=True)
    source=scenario(6, [[28]]*6, 83114, 0, stocks=stocks)
    for mode in ('tap','held','stationary','four-beams'):
        setup={k:v for k,v in source.items() if k!='frames'}
        setup['players']=[dict(p,abilities=[28]) for p in source['players']]
        if mode=='four-beams':
            for player in setup['players'][4:]:player['abilities']=[2]
        frames=[]
        for tick in range(ticks):
            frame=[]
            for player in range(6):
                age=tick-240
                angle=0 if age<0 or mode in ('stationary','four-beams') else ([32,65,96,65,32,96][player] if mode=='held' else [1,65,128,192,224,32][(player+age//90)%6])
                frame.append(dict(seqNumber=tick+1,joystickAngle=angle,jump=mode=='tap' and age>=0 and age%120<3,
                    ab1=age>=0 and (mode!='tap' or age%90<30) and (mode!='four-beams' or player<4),ab2=False,ab3=False))
            frames.append(frame)
        setup['frames']=frames
        (directory/f'six-beams-{mode}.json').write_text(json.dumps(setup)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--ticks',type=int,default=900)
    parser.add_argument('--stocks',type=int,default=5)
    args=parser.parse_args(); generate(args.directory,args.ticks,args.stocks)
