"""Standalone reproduction of the exp12 crash (torch 2.14.0+cu130, CUDA graphs).

cuBLAS caches one workspace per (handle, stream). The first graph captured on a
stream allocates that workspace inside its private memory pool; later captures
on the same stream reuse the cached pointer, so their kernels read memory owned
by the first graph. Freeing the first graph (and letting the allocator unmap
its pool) leaves the others reading freed memory: illegal memory access.

    python graph_workspace_repro.py full base        # crashes
    python graph_workspace_repro.py full clearbefore # passes (the fix used in turn.py / inference.py)
    python graph_workspace_repro.py full fresh       # passes (a new stream per capture)
    python graph_workspace_repro.py gru base         # passes (no cuBLAS GEMM in the graph)

Run from the repository root with the RL venv; needs native/rl/research on sys.path.
"""
import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from policy import Policy

fn,variant=sys.argv[1],sys.argv[2]   # fn: full|linear|gru|sample ; variant: base|clearbefore|noempty|fresh
torch.manual_seed(0); dev='cuda'
pol=Policy(300).to(dev).eval()
obs=torch.randn(128,300,device=dev); state=torch.zeros(1,128,128,device=dev)
def make():
    p=copy.deepcopy(pol)
    if fn=='full': return lambda: p.step(obs,state)
    if fn=='linear':
        w=torch.randn(128,300,device=dev); return lambda: torch.nn.functional.linear(obs,w)
    if fn=='gru': return lambda: p.memory(obs[:,:128].unsqueeze(0),state)
    if fn=='sample': return lambda: torch.multinomial(torch.softmax(obs[:,:16],-1),1)
    raise SystemExit(f'unknown fn {fn}')
one_stream=torch.cuda.Stream()
def capture(run):
    stream=torch.cuda.Stream() if variant=='fresh' else one_stream; main=torch.cuda.current_stream(); stream.wait_stream(main)
    with torch.cuda.stream(stream),torch.no_grad():
        run(); g=torch.cuda.CUDAGraph()
        if variant=='clearbefore': torch._C._cuda_clearCublasWorkspaces()
        g.capture_begin(capture_error_mode='thread_local')
        try: out=run()
        finally: g.capture_end()
    main.wait_stream(stream); torch.cuda.synchronize(); return g,out
graphs={}
for round in range(30):
    for i in range(6):
        if i not in graphs: graphs[i]=capture(make())
    for _ in range(10):
        for i in graphs: graphs[i][0].replay()
    torch.cuda.synchronize()
    for i in (round%6,(round+3)%6): del graphs[i]
    if variant!='noempty': torch.cuda.empty_cache()
    for _ in range(10):
        for i in graphs: graphs[i][0].replay()
    torch.cuda.synchronize()
print(fn,variant,'DONE')
