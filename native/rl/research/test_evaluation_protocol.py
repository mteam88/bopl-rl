"""Protect opponent identity and complete paired-game coverage without starting Unity."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_protocol import map_ids,opponents,schedule


class EvaluationProtocolTests(unittest.TestCase):
    def test_legacy_schedule_is_unchanged(self):
        tasks,groups=schedule([0],[1,2],[],8,4,16,10001,20001,30001)
        self.assertEqual(groups,[('random',0),('initial',0),('random',1),('random',2)])
        self.assertEqual(len(tasks),25)
        self.assertEqual(tasks[0],dict(opponent='random',map=0,seed=10001,side=0,save=True,name='random-0'))
        self.assertEqual(tasks[15]['seed'],10004)
        self.assertEqual(tasks[-1]['name'],'selfplay')
        self.assertEqual(sum(t['save'] for t in tasks),5)

    def test_multimap_final_has_each_opponent_on_each_map(self):
        frozen=[dict(name=f'trained-s{s}') for s in (73,173,273)]
        tasks,groups=schedule([0,1,2],[3,4],frozen,256,64,128,50001,60001,70001)
        self.assertEqual(len(tasks),2817)
        self.assertEqual(len(groups),17)
        counts=Counter((t['opponent'],t['map']) for t in tasks)
        pairs=defaultdict(set)
        for task in tasks:
            if task['opponent']=='selfplay': continue
            pairs[task['opponent'],task['map'],task['seed']].add(task['side'])
        self.assertTrue(all(sides=={0,1} for sides in pairs.values()))
        self.assertEqual(len({t['name'] for t in tasks}),len(tasks))
        for m in range(3):
            for opponent in ('random','initial'): self.assertEqual(counts[opponent,m],256)
            for f in frozen: self.assertEqual(counts[f['name'],m],128)
        for m in (3,4): self.assertEqual(counts['random',m],64)
        diagnostic,_=schedule([0,1,2],[3,4],frozen,32,16,16,10001,20001,30001)
        identity=lambda t:(t['opponent'],t['map'],t['seed'],t['side'])
        self.assertFalse(set(map(identity,tasks))&set(map(identity,diagnostic)))

    def test_invalid_maps_and_counts_fail(self):
        for value in ('','0,0','-1','31'):
            with self.assertRaises(ValueError): map_ids(value)
        with self.assertRaises(ValueError): schedule([0,1],[1],[],8,4,16,1,2,3)
        with self.assertRaises(ValueError): schedule([0],[1],[],7,4,16,1,2,3)

    def test_manifest_detects_changed_weights_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); weights=root/'policy.pt';weights.write_bytes(b'frozen checkpoint')
            item=dict(name='trained-s73',path=weights.name,sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
            manifest=root/'opponents.json';manifest.write_text(json.dumps(dict(schema=1,opponents=[item])))
            self.assertEqual(opponents(manifest)[0]['path'],str(weights))
            weights.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError,'changed'): opponents(manifest)
            weights.write_bytes(b'frozen checkpoint')
            manifest.write_text(json.dumps(dict(schema=1,opponents=[item,item])))
            with self.assertRaisesRegex(ValueError,'distinct'): opponents(manifest)


if __name__=='__main__': unittest.main()
