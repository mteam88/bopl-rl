"""Children of a function in a perf profile: perf script | callees.py <substr> [depth]
Per sample whose stack holds the function, credit the frame `depth` levels below it
(1 = its direct callee; samples in the function itself count as <self>)."""
import sys, re, collections
target, depth = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
total = hit = 0; c = collections.Counter(); stack = []
def flush():
    global total, hit
    if not stack: return
    total += 1
    for i, f in enumerate(stack):           # stack[0] = leaf
        if target in f:
            hit += 1
            j = i - depth
            c['<self>' if j < 0 else stack[j]] += 1
            break
for line in sys.stdin:
    line = line.rstrip('\n')
    if not line.strip(): flush(); stack = []; continue
    if line[0] in ' \t':
        m = re.match(r'\s+[0-9a-f]+\s+(.+)$', line)
        if m: stack.append(re.sub(r'\+0x[0-9a-f]+$', '', m.group(1)))   # inlined frames keep their ' (inlined)' tag
flush()
print(f'{target}: {hit}/{total} samples = {100*hit/max(total,1):.1f}%')
for f, n in c.most_common(25): print(f'{100*n/max(total,1):6.2f}%  {f[:150]}')
