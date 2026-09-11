"""Protect opponent identity and complete paired-game coverage without starting Unity."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_protocol import (SCRIPTED_KINDS,SCRIPTED_NAMES,checkpoint_identity,map_ids,opponents,schedule,
                                 scripted_names,scripted_opponent)


class EvaluationProtocolTests(unittest.TestCase):
    def test_paired_schedule_covers_each_group(self):
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

    def test_scripted_opponents_schedule_on_every_map_with_paired_sides(self):
        scripted=['mixed-hard','archer-medium']
        tasks,groups=schedule([0],[1,2],[],8,4,16,10001,20001,30001,scripted=scripted)
        counts=Counter((t['opponent'],t['map']) for t in tasks)
        for name in scripted:
            self.assertEqual(counts[name,0],8)
            for m in (1,2): self.assertEqual(counts[name,m],4)
            self.assertIn((name,0),groups); self.assertIn((name,1),groups)
        pairs=defaultdict(set)
        for task in tasks:
            if task['opponent'] in scripted:
                pairs[task['opponent'],task['map'],task['seed']].add(task['side'])
        self.assertTrue(pairs and all(sides=={0,1} for sides in pairs.values()))
        self.assertEqual(len({t['name'] for t in tasks}),len(tasks))
        # One replay saved per scripted group, like every other opponent.
        for name in scripted:
            self.assertEqual(sum(t['save'] for t in tasks if t['opponent']==name),3)
        # Without scripted opponents the schedule is exactly the old one.
        old,_=schedule([0],[1,2],[],8,4,16,10001,20001,30001)
        identity=lambda t:(t['opponent'],t['map'],t['seed'],t['side'])
        self.assertEqual([identity(t) for t in old],
                         [identity(t) for t in tasks if t['opponent'] not in scripted])

    def test_scripted_names_parse_and_reject(self):
        self.assertEqual(scripted_opponent('mixed-hard'),(4,2))
        self.assertEqual(scripted_opponent('archer-easy'),(1,0))
        # v2 kinds: version suffix on the kind, level still last (dashboard label "mixed-v2 (hard)").
        self.assertEqual(scripted_opponent('mixed-v2-hard'),(9,2))
        self.assertEqual(scripted_opponent('archer-v2-easy'),(6,0))
        for name in ('random','initial','trained-s73','mixed','mixed-extreme','bow-hard','mixed-v3-hard'):
            self.assertIsNone(scripted_opponent(name))
        self.assertEqual(scripted_names('mixed-hard,evasive-medium'),['mixed-hard','evasive-medium'])
        self.assertEqual(scripted_names('mixed-v2-hard,mixed-hard'),['mixed-v2-hard','mixed-hard'])
        self.assertEqual(scripted_names(''),[])
        for value in ('mixed','mixed-hard,mixed-hard','pursuer-hardest'):
            with self.assertRaises(ValueError): scripted_names(value)
        self.assertEqual(len(SCRIPTED_NAMES),3*len(SCRIPTED_KINDS))
        self.assertEqual(scripted_opponent('search-medium'),(5,1))
        # v1 ids are frozen (scores stay comparable); search then the v2 kinds appended.
        self.assertEqual([scripted_opponent(f'{k}-hard')[0] for k in ('archer','pursuer','evasive','mixed')],[1,2,3,4])
        self.assertEqual([scripted_opponent(f'{k}-v2-hard')[0] for k in ('archer','pursuer','evasive','mixed')],[6,7,8,9])

    def test_invalid_maps_and_counts_fail(self):
        for value in ('','0,0','-1','31'):
            with self.assertRaises(ValueError): map_ids(value)
        with self.assertRaises(ValueError): schedule([0,1],[1],[],8,4,16,1,2,3)
        with self.assertRaises(ValueError): schedule([0],[1],[],7,4,16,1,2,3)

    def test_manifest_freezes_weights_and_metadata_and_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'policy.bin'
            weights.write_bytes(b'frozen checkpoint')
            sidecar = root / 'policy.bin.json'
            metadata = dict(schema=1, backend='pufferlib-native', hidden=128,
                            weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
            sidecar.write_text(json.dumps(metadata))
            item = dict(name='trained-s73', path=weights.name, **checkpoint_identity(weights))
            manifest = root / 'opponents.json'
            manifest.write_text(json.dumps(dict(schema=2, opponents=[item])))
            self.assertEqual(opponents(manifest)[0]['path'], str(weights))
            weights.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'metadata hash'):
                opponents(manifest)
            weights.write_bytes(b'frozen checkpoint')
            sidecar.write_text(json.dumps(dict(metadata, hidden=256)))
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                opponents(manifest)
            sidecar.write_text(json.dumps(metadata))
            for reserved in ('search-hard', 'mixed-v2-easy'):  # level suffix reserved for scripted kinds
                manifest.write_text(json.dumps(dict(schema=2, opponents=[dict(item, name=reserved)])))
                with self.assertRaisesRegex(ValueError, 'level suffix'):
                    opponents(manifest)
            manifest.write_text(json.dumps(dict(schema=2, opponents=[item, item])))
            with self.assertRaisesRegex(ValueError, 'distinct'):
                opponents(manifest)
            # Scripted bot names are reserved alongside random/initial/selfplay.
            manifest.write_text(json.dumps(dict(schema=2, opponents=[dict(item, name='mixed-hard')])))
            with self.assertRaisesRegex(ValueError, 'distinct safe names'):
                opponents(manifest)

    def test_native_identity_requires_a_valid_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.bin'
            path.write_bytes(b'weights')
            with self.assertRaises(FileNotFoundError):
                checkpoint_identity(path)
            Path(str(path) + '.json').write_text(json.dumps(dict(schema=1, backend='unknown')))
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                checkpoint_identity(path)
            with self.assertRaisesRegex(ValueError, 'Native checkpoints'):
                checkpoint_identity(path.with_suffix('.weights'))


if __name__=='__main__': unittest.main()
