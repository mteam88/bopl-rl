"""Compare Unity with itself using complete real or mutated replay files."""
import argparse,json,subprocess,sys
from pathlib import Path
from reference import ROOT,capture,compare

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('replays',nargs='+',type=Path);p.add_argument('--ticks',type=int,default=1200);p.add_argument('--output',type=Path,default=ROOT/'native/artifacts/reference-suite');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);results=[]
    for replay in args.replays:
        row={'replay':str(replay),'requested_ticks':args.ticks,'scope':'Unity self-comparison; not Rust parity'}
        try:
            metadata=json.loads(subprocess.check_output(["dotnet",str(ROOT/"native/tools/replay/bin/Release/net8.0/replay.dll"),"inspect",str(replay)],text=True))
            if metadata.get("error"):raise ValueError(metadata["error"])
            if not metadata["complete_player_inputs"]:raise ValueError("legacy replay is missing player 5/6 inputs")
            if metadata["has_unresolved_ability"]:raise ValueError("legacy replay has ability ID 255; original loadout cannot be recovered from this file")
            row["metadata"]=metadata
            ticks=min(args.ticks,metadata["ticks"])
            paths=[]
            for batch in (1,16):
                output=args.output/(replay.stem+f'-batch{batch}.jsonl.gz');paths.append(output)
                capture(argparse.Namespace(replay=replay,output=output,ticks=ticks,batch=batch,timeout=180))
            row['raw_match']=compare(*paths)==0
            row['gameplay_match']=compare(*paths,gameplay=True)==0
            row['excluded_fields']=list(json.loads((ROOT/'native/data/presentation-fields.json').read_text())['fields'])
            row['match']=row['gameplay_match']
        except Exception as error:row.update(match=False,error=str(error))
        results.append(row);print(json.dumps(row),flush=True)
        (args.output/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    return int(any(not r['match'] for r in results))
if __name__=='__main__':sys.exit(main())
