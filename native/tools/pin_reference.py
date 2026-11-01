"""Reproduce the baseline file manifest from an immutable Git commit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse',args.commit+'^{commit}'],cwd=ROOT/'game',text=True).strip()
    files={}
    with subprocess.Popen(['git','archive',commit,'Assets','ProjectSettings','Packages'],cwd=ROOT/'game',stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
            for entry in archive:
                if entry.isfile():files[entry.name]=hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
        if process.wait()!=0:raise RuntimeError('git archive failed')
    files=dict(sorted(files.items()))
    manifest={'schema':'bopl-source-pin-v1','commit':commit,'content_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'files':files}
    path=ROOT/'native/data/full-reference.json'
    if args.verify:
        if json.loads(path.read_text())!=manifest:raise ValueError('Committed pin does not match Git source')
        if (ROOT/'game/Assets/Resources/SimulationReference.txt').read_text().strip()!=manifest['content_sha256']:raise ValueError('Runtime pin differs')
    else:
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        (ROOT/'game/Assets/Resources/SimulationReference.txt').write_text(manifest['content_sha256']+'\n')
    print(json.dumps({'reference':commit,'files':len(files),'sha256':manifest['content_sha256'],'verified':args.verify}))

if __name__=='__main__':main()
