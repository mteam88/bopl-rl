"""A matching prefix must never certify a truncated or malformed actor trace."""
import copy
import unittest
from actor_stream import Protocol


class StreamCompleteness(unittest.TestCase):
    def setUp(self):
        self.query = dict(actions=[{}], runtime=dict(actors=[{}], scheduler=dict(inputs=[{}], players=[])))
        self.frame = dict(actors=[{}], scene={}, report=dict(phases=[], events=[], pushes=[], physics=None), ticks=1, time=0, level_time=0, inputs=[{}])
        self.row = dict(case=0, action=0, state=self.frame)
        self.end = dict(case=0, result=dict(completed=1))

    def test_complete_success_and_partial_failure(self):
        p = Protocol([self.query]); p.accept(self.row); p.accept(self.end)
        self.assertEqual(p.finish()['ticks'], 1)
        p = Protocol([self.query])
        p.accept(dict(case=0, result=dict(completed=0, action=0, error='null_reference', failed=self.frame)))
        self.assertEqual(p.finish()['errors'], dict(null_reference=1))

    def test_matching_prefix_without_end_is_incomplete(self):
        p = Protocol([self.query]); p.accept(self.row)
        with self.assertRaises(ValueError): p.finish()

    def test_order_counts_and_extra_data(self):
        invalid = [dict(self.row, case=1), dict(self.row, action=1), dict(self.row, action=True)]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError): Protocol([self.query]).accept(row)
        for end in [dict(case=0, result=dict(completed=0)), dict(case=0, result=dict(completed=True))]:
            p = Protocol([self.query]); p.accept(self.row)
            with self.assertRaises(ValueError): p.accept(end)
        p = Protocol([self.query]); p.accept(self.row)
        with self.assertRaises(ValueError): p.accept(self.row)
        p.accept(self.end)
        with self.assertRaises(ValueError): p.accept(self.end)

    def test_failure_requires_full_partial_state(self):
        for result in [dict(error='null_reference', completed=0, action=0),
                       dict(error='null_reference', completed=0, action=0, failed={}),
                       dict(error='null_reference', completed=1, action=1, failed=self.frame)]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                Protocol([self.query]).accept(dict(case=0, result=result))
        row = copy.deepcopy(self.row); del row['state']['report']['phases']
        with self.assertRaises(ValueError): Protocol([self.query]).accept(row)


if __name__ == '__main__':
    unittest.main()
