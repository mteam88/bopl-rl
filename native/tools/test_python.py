"""Contract checks for handle ownership, invalid input, and reset through ctypes."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from bopl_native import World

class Contract(unittest.TestCase):
    def test_reset_and_rejected_inputs(self):
        with World() as world:
            initial=world.state()
            for action in ([[256,0],[0,0]],[[0,4],[0,0]],[[0,0]], [[-1,0],[0,0]]):
                with self.assertRaises(ValueError):world.step(action)
                self.assertEqual(initial,world.state())
            world.step([[1,0],[128,0]]);self.assertNotEqual(initial,world.state())
            world.reset();self.assertEqual(initial,world.state())
        world.close()
        with self.assertRaises(RuntimeError):world.state()
    def test_terminal_requires_reset(self):
        with World() as world:
            # Reuse a source-backed trajectory whose first death is in the corpus.
            import gzip,json
            fixture=Path(__file__).resolve().parents[1]/'fixtures/reference/random-1.jsonl.gz'
            with gzip.open(fixture,'rt') as f:
                next(f)
                for line in f:world.step(json.loads(line)['actions'])
            self.assertTrue(world.terminal);state=world.state()
            with self.assertRaises(RuntimeError):world.step([[0,0],[0,0]])
            self.assertEqual(state,world.state());world.reset();self.assertFalse(world.terminal)
if __name__=='__main__':unittest.main()
