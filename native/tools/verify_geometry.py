"""Run original collision queries in Unity and compare the Rust port exactly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OPS = "circle_ray box_ray rounded_ray circle_overlap box_overlap rounded_overlap closest surface_normal surface_closest surface_local surface_position surface_move".split()
ONE = 1 << 32


def cases(count, seed):
    rng = random.Random(seed)
    for i in range(count):
        case = dict(center=[rng.randrange(-20*ONE,20*ONE) for _ in range(2)],
            extent=[rng.randrange(ONE//10,10*ONE) for _ in range(2)],
            origin=[rng.randrange(-25*ONE,25*ONE) for _ in range(2)],
            angle=rng.randrange(-26986075409,26986075409),
            direction=rng.randrange(-26986075409,26986075409),
            radius=rng.randrange(0,3*ONE), player_radius=rng.randrange(ONE//10,2*ONE), local=rng.randrange(0,ONE),speed=rng.randrange(-30*ONE,30*ONE))
        if i % 4 == 0:
            case["angle"] = 0
            case["direction"] = rng.choice([0,6746518852,13493037704,20239556556])
            x,y = case["center"]
            ex,ey = case["extent"]
            case["origin"] = [x+rng.choice([-ex,0,ex])+rng.choice([-1,0,1]),y+rng.choice([-ey,0,ey])+rng.choice([-1,0,1])]
        for operation in OPS:
            yield dict(case, operation=operation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=67484)
    args = parser.parse_args()
    directory = ROOT / "native/artifacts/geometry"
    directory.mkdir(parents=True,exist_ok=True)
    queries = list(cases(args.cases,args.seed))
    input_path = directory / "input.jsonl"
    input_path.write_text("".join(json.dumps(q,separators=(",",":"))+"\n" for q in queries))
    output_path = directory / "unity.jsonl"
    if output_path.exists(): output_path.unlink()
    env = dict(os.environ,BOPL_KERNEL_INPUT=str(input_path),BOPL_KERNEL_OUTPUT=str(output_path))
    for key in ("BOPL_TRACE","BOPL_TRACE_TICKS","BOPL_TRACE_BATCH"):
        env.pop(key,None)
    binary = ROOT / "Builds/Linux/BoplBattle.x86_64"
    with (directory/"stdout.log").open("w") as stdout:
        subprocess.run([str(binary),str(ROOT/"native/fixtures/movement.rep"),"-batchmode","-nographics","-logFile",str(directory/"unity.log")],
            env=env,stdout=stdout,stderr=subprocess.STDOUT,check=True,timeout=180)
    if "REFERENCE_KERNEL_COMPLETE" not in (directory/"unity.log").read_text():
        raise ValueError("Unity kernel probe did not complete")
    with input_path.open("rb") as source, (directory/"rust.jsonl").open("wb") as output:
        subprocess.run([str(ROOT/"native/target/release/kernel-probe")],stdin=source,stdout=output,check=True)
    left=[json.loads(line) for line in output_path.read_text().splitlines()]
    right=[json.loads(line) for line in (directory/"rust.jsonl").read_text().splitlines()]
    if len(left)!=len(queries) or len(right)!=len(queries):
        raise ValueError(f"incomplete corpus: {len(queries)} cases, Unity {len(left)}, Rust {len(right)}")
    failures=[]
    counts={op:{"cases":0,"mismatches":0,"hits":0} for op in OPS}
    for query,a,b in zip(queries,left,right):
        count=counts[query["operation"]]
        count["cases"]+=1
        if a is True or (isinstance(a,dict) and a.get("hit")):count["hits"]+=1
        if a!=b:
            count["mismatches"]+=1
            if len(failures)<12:failures.append({"case":query,"unity":a,"rust":b})
    report={"match":not failures,"cases":len(queries),"seed":args.seed,"operations":counts,"first_failures":failures,
        "input_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "game_assembly_sha256":hashlib.sha256((binary.parent/"BoplBattle_Data/Managed/Assembly-CSharp.dll").read_bytes()).hexdigest()}
    (directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return bool(failures)


if __name__=="__main__":sys.exit(main())
