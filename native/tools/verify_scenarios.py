"""Build scripted or stochastic replays, capture Unity, export and check C ABI goldens."""
import argparse,subprocess,sys
from pathlib import Path
from reference import ROOT,capture
from golden import export,verify

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixtures',nargs='+',type=Path);p.add_argument('--output',type=Path,default=ROOT/'native/artifacts/scenarios');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for fixture in args.fixtures:
        import json
        count=len(json.loads(fixture.read_text())['frames'])
        replay=fixture.with_suffix('.rep');trace=args.output/(fixture.stem+'.jsonl.gz');golden=ROOT/'native/fixtures/reference'/(fixture.stem+'.jsonl.gz')
        subprocess.run(['dotnet',str(ROOT/'native/tools/replay/bin/Release/net8.0/replay.dll'),str(fixture),str(replay)],check=True)
        capture(argparse.Namespace(replay=replay,output=trace,ticks=count,batch=16,timeout=180))
        export(argparse.Namespace(trace=trace,output=golden));verify([golden])
if __name__=='__main__':main()
