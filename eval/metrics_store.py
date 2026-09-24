"""Incremental JSONL chart views with bounded history and exact counters."""
from collections import OrderedDict, deque
from pathlib import Path
import json
import os
import threading

from packed import open_text, resolved

POINTS = 2400
TAIL = 120
CACHE_RUNS = 8


class Metrics:
    def __init__(self, identity, points=POINTS, tail=TAIL):
        self.identity = identity
        self.points = points
        self.tail_size = tail
        self.offset = 0
        self.count = 0
        self.stride = 1
        self.body = []
        self.tail = deque()
        self.totals = {}
        self.map_matches = {}
        self.signature = None

    def append(self, row):
        for key, value in row.get('behavior', {}).get('totals', {}).items():
            self.totals[key] = self.totals.get(key, 0) + value
        # Match counters are already cumulative; behavior totals are per update.
        if 'map_matches' in row:
            self.map_matches = dict(row['map_matches'])
        # Keep cumulative counters alongside samples: study curves must not sum
        # sampled per-update counters and silently undercount the missing rows.
        self.tail.append((self.count, row, dict(self.totals)))
        self.count += 1
        if len(self.tail) > self.tail_size:
            sample = self.tail.popleft()
            if sample[0] % self.stride == 0:
                self.body.append(sample)
            if len(self.body) > self.points - self.tail_size:
                self.stride *= 2
                self.body = [sample for sample in self.body if sample[0] % self.stride == 0]

    def snapshot(self):
        samples = self.body + list(self.tail)
        return dict(rows=[s[1] for s in samples], cumulative=[s[2] for s in samples],
                    count=self.count, totals=dict(self.totals), map_matches=dict(self.map_matches))


class MetricsStore:
    def __init__(self, capacity=CACHE_RUNS, points=POINTS, tail=TAIL):
        self.capacity = capacity
        self.points = points
        self.tail_size = tail
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def read(self, path):
        path = Path(path)
        found = resolved(path)
        if found != path:
            return self.read_packed(path, found)
        with self.lock:
            try:
                handle = path.open('rb')
            except FileNotFoundError:
                self.entries.pop(path, None)
                return Metrics(None).snapshot()
            with handle:
                stat = os.fstat(handle.fileno())
                identity = (stat.st_dev, stat.st_ino)
                signature = (stat.st_size, stat.st_mtime_ns)
                state = self.entries.pop(path, None)
                if (state is None or state.identity != identity or stat.st_size < state.offset
                        or (state.signature != signature and state.signature is not None
                            and stat.st_size <= state.signature[0])):
                    state = Metrics(identity, self.points, self.tail_size)
                self.entries[path] = state
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
                if signature != state.signature:
                    handle.seek(state.offset)
                    # Iterate lines, never read/split the entire file. Stop at
                    # the observed size so an active writer cannot extend a scan.
                    while handle.tell() < stat.st_size:
                        line = handle.readline()
                        if not line.endswith(b'\n') or handle.tell() > stat.st_size:
                            break
                        row = json.loads(line)  # a malformed complete line is an error
                        state.append(row)
                        state.offset = handle.tell()
                    state.signature = signature
                return state.snapshot()

    def read_packed(self, path, found):
        """An archived stream cannot be appended to: parse it once, then serve the cache."""
        with self.lock:
            try:
                stat = found.stat()
            except FileNotFoundError:
                self.entries.pop(path, None)
                return Metrics(None).snapshot()
            identity = (stat.st_dev, stat.st_ino)
            signature = (stat.st_size, stat.st_mtime_ns)
            state = self.entries.pop(path, None)
            if state is None or state.identity != identity or state.signature != signature:
                state = Metrics(identity, self.points, self.tail_size)
                with open_text(found) as handle:
                    for line in handle:
                        if not line.endswith('\n'):
                            break  # a torn tail, as in the plain scan above
                        state.append(json.loads(line))
                state.signature = signature
            self.entries[path] = state
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return state.snapshot()


store = MetricsStore()
