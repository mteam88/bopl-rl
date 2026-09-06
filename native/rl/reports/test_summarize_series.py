"""Reporting fixtures are synthetic; they protect completeness checks, not playing strength."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from summarize_series import interval,summarize


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.series=self.root/'study';self.series.mkdir()
        run=self.root/'run';run.mkdir()
        frozen=[dict(name=f'trained-s{s}',sha256=f'fixture-{s}') for s in (73,173,273)]
        self.write(self.series/'plan.json',dict(name='study',git_revision='synthetic',source_identity={},baseline_sha256='baseline',
            steps=4096,runs=[dict(name='run',seed=73)],training_maps=[0,1,2],evaluation_maps=[0,1,2],heldout_maps=[3,4],
            final_games=2,final_frozen_games=2,final_heldout_games=2,frozen_opponents=frozen))
        self.write(self.series/'status.json',dict(status='complete',elapsed=2))
        self.write(run/'result.json',dict(status='complete',step=4096))
        self.metric=dict(step=4096,matches=6,sps=2048,rollout_seconds=1,train_seconds=1,
                         env={},behavior=dict(decisions=3072),map_matches={'0':2,'1':2,'2':2})
        (run/'metrics.jsonl').write_text(json.dumps(self.metric)+'\n')
        keys=[(opponent,m) for m in range(3) for opponent in ('random','initial',*[p['name'] for p in frozen])]+[('random',3),('random',4)]
        groups=[dict(opponent=opponent,map=m,games=2,wins=1,win_rate=.5,timeouts=0) for opponent,m in keys]
        games=[dict(opponent=opponent,map=m,seed=50001,side=side,win=side==0,timeout=False) for opponent,m in keys for side in (0,1)]
        self.final=run/'final-evaluations/000004096/result.json';self.final.parent.mkdir(parents=True)
        self.report=dict(step=4096,baseline_sha256='baseline',checkpoint_sha256='candidate',frozen_opponents=frozen,
                         seed_base=50001,heldout_seed_base=60001,elapsed=1,groups=groups,games=games)
        self.write(self.final,self.report)

    def tearDown(self): self.temporary.cleanup()

    @staticmethod
    def write(path,value): path.write_text(json.dumps(value)+'\n')

    def test_accepts_complete_multimap_groups(self):
        report,_=summarize(self.series,False)
        self.assertEqual(report['total_decisions'],4096)
        self.assertEqual(len(report['runs'][0]['final']['groups']),17)
        self.assertEqual(report['runs'][0]['map_matches'],{'0':2,'1':2,'2':2})

    def test_rejects_a_missing_opponent_map_group(self):
        self.report['groups'].pop();self.write(self.final,self.report)
        with self.assertRaisesRegex(ValueError,'missing or duplicating'): summarize(self.series,False)

    def test_rejects_incorrect_map_totals(self):
        self.metric['map_matches']['2']=1
        (self.root/'run/metrics.jsonl').write_text(json.dumps(self.metric)+'\n')
        with self.assertRaisesRegex(ValueError,'conserve'): summarize(self.series,False)

    def test_rejects_duplicate_side_in_bootstrap(self):
        games=[dict(opponent='random',map=0,seed=1,side=0,win=True)]*2
        with self.assertRaisesRegex(ValueError,'Duplicate'): interval(games,'random',0,np.random.default_rng(1))


if __name__=='__main__': unittest.main()
