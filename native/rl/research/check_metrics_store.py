"""Check dashboard sampling against exact JSONL totals and writer transitions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from metrics_store import MetricsStore, POINTS, TAIL
from run_insights import game_insights


def main():
    with TemporaryDirectory() as directory:
        path = Path(directory)/'metrics.jsonl'
        records = [dict(step=i, map_matches={'0': i * 3},
                        behavior={'totals': {'arrow_hits': i % 7}}) for i in range(10000)]
        path.write_text(''.join(json.dumps(row)+'\n' for row in records))
        store = MetricsStore(capacity=2)
        result = store.read(path)
        assert result['count'] == len(records)
        assert len(result['rows']) <= POINTS
        assert result['rows'][-TAIL:] == records[-TAIL:]
        assert result['rows'][0] == records[0]
        assert result['map_matches']['0'] == records[-1]['map_matches']['0']
        prefix = [0]
        for row in records: prefix.append(prefix[-1]+row['behavior']['totals']['arrow_hits'])
        for row, cumulative in zip(result['rows'], result['cumulative']):
            assert cumulative['arrow_hits'] == prefix[row['step']+1]
        assert store.read(path) == result
        with path.open('ab') as out: out.write(b'{"step":10000')
        assert store.read(path) == result
        with path.open('ab') as out: out.write(b'}\n')
        assert store.read(path)['count'] == 10001
        with path.open('ab') as out: out.write(b'bad middle\n{"step":10001}\n')
        try: store.read(path)
        except ValueError: pass
        else: raise AssertionError('Malformed complete line was accepted')
        path.write_text('{"step":5}\n')
        assert store.read(path)['rows'] == [{'step': 5}]
        path.write_text('{"step":6}\n')
        assert store.read(path)['rows'] == [{'step': 6}]
        replacement = path.with_suffix('.new')
        replacement.write_text('{"step":7}\n'); replacement.replace(path)
        assert store.read(path)['rows'] == [{'step': 7}]
        for i in range(3):
            file=path.with_name(f'{i}.jsonl'); file.write_text('{"step":1}\n'); store.read(file)
        assert len(store.entries) == 2
        path.unlink()
        assert store.read(path)['count'] == 0
    player=dict(flags=1,x=0,y=0,deaths=0)
    frames=[dict(actions=[[10,4,16]],players=[player]),dict(actions=[[20,0,1]],players=[player])]
    extended=game_insights({'frames':frames},0)
    for frame in frames: frame['actions'][0]=frame['actions'][0][:2]
    assert game_insights({'frames':frames},0) == extended
    print('PASS: bounded samples, exact totals/tail, append, torn tail, corruption, truncation, replacement, eviction, v2 replay actions')


if __name__ == '__main__': main()
