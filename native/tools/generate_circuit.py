"""Generate an open-loop replay that drives both slimes around platform perimeters.

Rust is used only to choose test inputs. Unity subsequently executes the saved
inputs independently, and its states are the oracle for the resulting golden.
"""
import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from bopl_native import World

def main():
    frames=[[[0,0],[0,0]] for _ in range(240)]
    with World() as world:
        for _ in range(1200):
            if world.terminal:break
            actions=[]
            for index,p in enumerate(world.state()['players']):
                x,y=p['up']['y'],-p['up']['x']
                if index: x,y=-x,-y
                angle=math.atan2(y,x)%(2*math.pi)
                byte=round(angle/(2*math.pi)*255)%255+1
                actions.append([byte,0])
            frames.append(actions);world.step(actions)
    path=Path(__file__).resolve().parents[1]/'fixtures/surface-circuit.json'
    path.write_text(json.dumps(dict(setup=dict(seed=12345,level=0),profile='surface-circuit-v1',frames=frames),separators=(',',':'))+'\n')
    print(len(frames),'frames')
if __name__=='__main__':main()
