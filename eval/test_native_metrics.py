import copy
import json
import unittest

from native_metrics import empty_counters, update
from telemetry import FIELDS


class NativeMetricsTests(unittest.TestCase):
    def test_real_decisions_exclude_padding_and_keep_unfinished_events(self):
        before = empty_counters()
        now = copy.deepcopy(before)
        now.update(learner_steps=24, game_steps=32, buttons=[12,0,24,6], kills=1,
                   opponent_game_decisions=8)
        now['telemetry'][FIELDS.index('ticks')] = 96
        now['telemetry'][FIELDS.index('arrows_fired')] = 3
        log = dict(agent_steps=88, epoch=1, env={}, loss=dict(entropy=8., entropy_aim=5., entropy_buttons=3.))
        row = update(log,now,before,elapsed=2,previous_elapsed=0,
                     entropy_coefficient=.01,learning_rate=.001)
        self.assertEqual((row['step'],row['learner_steps'],row['native_agent_steps']),(32,24,88))
        self.assertEqual((row['sps'],row['learner_sps'],row['learner_fraction']),(16,12,.75))
        self.assertEqual(row['matches'],0)
        self.assertEqual(row['behavior']['totals']['arrows_fired'],3)
        self.assertEqual(row['behavior']['totals']['kills'],1)
        self.assertEqual(row['action_rates']['arrow'],1)
        self.assertIsNone(row['explained_variance'])
        self.assertEqual(before,empty_counters())
        # Next interval contains two learner timeouts from one self-play match.
        after=copy.deepcopy(now)
        after.update(learner_steps=48,game_steps=64,matches=1,map_matches={'0':1},opponent_game_decisions=16)
        after['telemetry'][FIELDS.index('ticks')]+=96
        for field,value in [('episodes',2),('timeouts',2),('reward_outcome',-1)]:
            after['telemetry'][FIELDS.index(field)]+=value
        row=update(log,after,now,elapsed=4,previous_elapsed=2,
                   entropy_coefficient=.009,learning_rate=.001)
        self.assertEqual(row['behavior']['totals']['arrows_fired'],0)
        self.assertEqual(row['behavior']['totals']['reward_outcome'],-1)
        self.assertEqual(row['behavior']['totals']['episodes'],2)
        self.assertEqual(row['matches'],1)
        self.assertEqual(row['mean_reward'],-1/24)
        json.dumps(row,allow_nan=False)

    def test_reject_inconsistent_counters(self):
        now=empty_counters();now.update(learner_steps=2,game_steps=1)
        with self.assertRaises(ValueError):
            update({},now,empty_counters(),elapsed=1,previous_elapsed=0,
                   entropy_coefficient=.01,learning_rate=.001)

    def test_nonfinite_training_loss_is_failure_not_missing_data(self):
        now=empty_counters();now.update(learner_steps=1,game_steps=2)
        kwargs=dict(elapsed=1,previous_elapsed=0,entropy_coefficient=.01,learning_rate=.001)
        with self.assertRaises(FloatingPointError):
            update({'loss':{'policy':float('nan')}},now,empty_counters(),**kwargs)
        row=update({'loss':{'explained_variance':float('nan')}},now,empty_counters(),**kwargs)
        self.assertIsNone(row['explained_variance'])


if __name__=='__main__':unittest.main()
