"""Per-buffer-step GPU timeline from a BOPL_CUPTI_TRACE dump (cupti_profile.cpp).

A rollout step on a buffer stream runs from the first H2D of the step (issued
by the buffer thread after the env step) to the end of the D2H of the actions.
This script finds the buffer streams (the ones with D2H memcpys), splits each
into steps at the D2H ends, prints the distribution of the GPU-side step
window and the kernel/memcpy inventory of one representative (median-length)
step: every record on the buffer stream with the gap before it, then the
records of the other streams (bank branches) inside the same window.

    python cupti_timeline.py /tmp/trace.txt [--stream ID] [--steps N]
"""
import argparse
import collections
import statistics


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            s, e, st, k, name = line.rstrip('\n').split(' ', 4)
            recs.append((int(s), int(e), int(st), k, name))
    recs.sort()
    return recs


def short(name):
    if name.startswith('memcpy_') or name.startswith('memset'):
        return name
    if name.startswith('_Z'):
        # crude demangle: skip _Z<len>name
        i = 2
        while i < len(name) and name[i].isdigit():
            i += 1
        n = int(name[2:i]) if i > 2 else 0
        return name[i:i + n] if n else name[:40]
    return name[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--stream', type=int, default=None)
    ap.add_argument('--steps', type=int, default=200, help='steady-state steps to summarise (from the end of the trace)')
    ap.add_argument('--all-streams', action='store_true', help='also list the records of every other stream in the window')
    a = ap.parse_args()
    recs = load(a.trace)
    by_stream = collections.defaultdict(list)
    for r in recs:
        by_stream[r[2]].append(r)
    # A step ends at the D2H of the actions (per-bank chain) or at the sampler
    # kernel of the batched chain (BOPL_BATCHED_FWD with zero-copy: no memcpys).
    def is_end(r):
        return r[3] == 'D' or (r[3] == 'K' and 'br_sample' in r[4])
    d2h = {st: sum(1 for r in rs if is_end(r)) for st, rs in by_stream.items()}
    buffer_streams = sorted([st for st, n in d2h.items() if n > 10], key=lambda s: -d2h[s])
    print('streams:', {st: (len(rs), d2h[st]) for st, rs in sorted(by_stream.items())})
    print('buffer streams (step-end count):', [(st, d2h[st]) for st in buffer_streams])
    stream = a.stream if a.stream is not None else buffer_streams[0]
    rs = by_stream[stream]
    # Steps: end at each step-end record; start at the first record after the previous one.
    steps = []
    start_i = 0
    for i, r in enumerate(rs):
        if is_end(r):
            seg = rs[start_i:i + 1]
            steps.append(seg)
            start_i = i + 1
    steps = steps[-a.steps - 5:-5] if len(steps) > a.steps + 5 else steps
    windows = [(seg[-1][1] - seg[0][0]) / 1e3 for seg in steps]
    busy = []
    for seg in steps:
        b = 0
        last = 0
        for s, e, *_ in seg:
            fr = max(s, last)
            if e > fr:
                b += e - fr
            last = max(last, e)
        busy.append(b / 1e3)
    print(f'\nstream {stream}: {len(steps)} steps; GPU window (first record start -> step end) '
          f'median {statistics.median(windows):.1f} us, p10 {sorted(windows)[len(windows)//10]:.1f}, '
          f'p90 {sorted(windows)[len(windows)*9//10]:.1f}; stream busy median {statistics.median(busy):.1f} us; '
          f'records/step {statistics.median(len(s) for s in steps):.0f}')
    # Interval from one step's D2H end to the next step's first H2D start = host side (env step + launch).
    host = [(steps[i + 1][0][0] - steps[i][-1][1]) / 1e3 for i in range(len(steps) - 1)]
    print(f'host gap between steps (step end -> next first record) median {statistics.median(host):.1f} us')
    # Representative step: the one with the median window.
    order = sorted(range(len(steps)), key=lambda i: windows[i])
    seg = steps[order[len(order) // 2]]
    t0, t1 = seg[0][0], seg[-1][1]
    print(f'\nrepresentative step: window {(t1 - t0) / 1e3:.1f} us, {len(seg)} records on stream {stream}')
    print(f'{"start_us":>9} {"dur_us":>7} {"gap_us":>7} kind name')
    last = t0
    for s, e, st, k, name in seg:
        print(f'{(s - t0) / 1e3:9.1f} {(e - s) / 1e3:7.1f} {(s - last) / 1e3:7.1f} {k} {short(name)}')
        last = e
    # Other streams' records in the window.
    others = collections.defaultdict(list)
    for r in recs:
        if r[2] != stream and r[0] >= t0 and r[1] <= t1:
            others[r[2]].append(r)
    total = len(seg)
    for st, ors in sorted(others.items()):
        total += len(ors)
        kinds = collections.Counter(short(r[4]) for r in ors)
        span = (ors[-1][1] - ors[0][0]) / 1e3
        print(f'\nstream {st} in window: {len(ors)} records, span {span:.1f} us, '
              f'first at +{(ors[0][0] - t0) / 1e3:.1f} us, last end at +{(ors[-1][1] - t0) / 1e3:.1f} us')
        if a.all_streams:
            last = ors[0][0]
            for s, e, _, k, name in ors:
                print(f'{(s - t0) / 1e3:9.1f} {(e - s) / 1e3:7.1f} {(s - last) / 1e3:7.1f} {k} {short(name)}')
                last = e
        else:
            print('   ', dict(kinds))
    print(f'\nall records in window (all streams): {total}')
    # Kernel inventory over the summarised steps on this stream and all streams in their windows.
    inv = collections.Counter()
    dur = collections.Counter()
    for sg in steps:
        w0, w1 = sg[0][0], sg[-1][1]
        for r in recs:
            if w0 <= r[0] and r[1] <= w1:
                inv[short(r[4])] += 1
                dur[short(r[4])] += r[1] - r[0]
    n = len(steps)
    print(f'\ninventory per step, all streams, averaged over {n} steps of stream {stream} (other buffers\' work overlaps: divide by buffers):')
    print(f'{"count":>7} {"sum_us":>8} {"avg_us":>7} name')
    for name, c in sorted(inv.items(), key=lambda kv: -dur[kv[0]]):
        print(f'{c / n:7.1f} {dur[name] / n / 1e3:8.1f} {dur[name] / c / 1e3:7.1f} {name}')


if __name__ == '__main__':
    main()
