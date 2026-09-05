"""Compare RNG output and internal state with the pinned game assembly."""
import argparse,gzip,hashlib,json,random,subprocess
from reference import ROOT

FIXTURE=ROOT/'native/fixtures/random.jsonl.gz'

def verify(queries,expected):
    result=subprocess.run([str(ROOT/'native/target/release/random-probe')],input=''.join(json.dumps(q)+'\n' for q in queries),text=True,capture_output=True,check=True)
    actual=[json.loads(line) for line in result.stdout.splitlines()]
    if actual!=expected:
        for index,(a,b) in enumerate(zip(actual,expected)):
            if a!=b:raise ValueError(json.dumps({'seed':queries[index]['seed'],'query':queries[index],'rust':a,'unity':b}))
        raise ValueError('Incomplete RNG output')
    print(json.dumps({'random_match':True,'seeds':len(queries),'draws':sum(len(q['operations']) for q in queries)}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--golden',action='store_true');args=p.parse_args()
    if args.golden:
        with gzip.open(FIXTURE,'rt') as f:records=[json.loads(l) for l in f]
        if records[0]['schema']!='bopl-random-golden-v1' or records[0]['seeds']!=len(records)-1 or records[0]['draws']!=sum(len(r['query']['operations']) for r in records[1:]):raise ValueError('Invalid RNG golden')
        verify([r['query'] for r in records[1:]],[r['expected'] for r in records[1:]]);return
    rng=random.Random(67484);queries=[]
    for seed in [0,1,37,12345,2**32-1]+[rng.randrange(2**32) for _ in range(60)]:
        ops=[]
        for i in range(1000):
            kind=rng.choice(['int','uint','fix'])
            low,high={'int':(-2**31,2**31),'uint':(0,2**32),'fix':(-2**63,2**63)}[kind]
            a,b=rng.randrange(low,high),rng.randrange(low,high)
            if i%5==0:b=a
            if kind=='fix' and i%3==0:b=(a & ~0xffffffff)|rng.randrange(2**32)
            ops.append({'kind':kind,'min':a,'max':b})
        queries.append({'seed':seed,'operations':ops})
    assembly=ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll'
    result=subprocess.run(['dotnet',str(ROOT/'native/tools/random-oracle/bin/Release/net8.0/random-oracle.dll')],input=''.join(json.dumps(q)+'\n' for q in queries),text=True,capture_output=True,check=True)
    expected=[json.loads(l) for l in result.stdout.splitlines()]
    if len(expected)!=len(queries):raise ValueError('Incomplete original RNG output')
    verify(queries,expected)
    directory=ROOT/'native/artifacts/random';directory.mkdir(exist_ok=True)
    header={'schema':'bopl-random-golden-v1','game_assembly_sha256':hashlib.sha256(assembly.read_bytes()).hexdigest(),'seeds':len(queries),'draws':65000}
    records=[header,*[{'query':q,'expected':e} for q,e in zip(queries,expected)]]
    (directory/'report.json').write_text(json.dumps(header,indent=2)+'\n')
    # Keep explicit boundary seeds and a full mixed-history sequence per seed.
    FIXTURE.write_bytes(gzip.compress((''.join(json.dumps(r,separators=(',',':'))+'\n' for r in [{**header,'seeds':5,'draws':5000},*records[1:6]])).encode(),mtime=0))

if __name__=='__main__':main()
