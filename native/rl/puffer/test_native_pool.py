"""Match-lifetime checks independent of CUDA or game startup."""
import unittest
from native_pool import SnapshotPool


class Vector:
    def __init__(self, references):
        self.references = references
        self.mask = None

    def bank_references(self):
        return self.references

    def set_valid_banks(self, mask):
        assert mask
        self.mask = mask


class Backend:
    def __init__(self):
        self.loads = []

    def load_frozen_bank(self, trainer, bank, path):
        self.loads.append((bank, path))


class PoolTests(unittest.TestCase):
    def test_held_snapshot_is_retired_then_replaced_after_last_match(self):
        backend, vector = Backend(), Vector([2, 0])
        pool = SnapshotPool(backend, None, vector, 2, "initial.bin")
        self.assertEqual(pool.request("one.bin", 100)["event"], "loaded")
        vector.references = [2, 1]
        self.assertEqual(pool.request("two.bin", 200)["event"], "retiring")
        self.assertEqual(vector.mask, 2)
        self.assertEqual(backend.loads, [(0, "initial.bin"), (1, "one.bin")])
        vector.references = [1, 3]
        self.assertIsNone(pool.service())
        # A newer checkpoint can arrive while the last old match still runs.
        self.assertIsNone(pool.request("three.bin", 300))
        vector.references = [0, 4]
        self.assertEqual(pool.service()["step"], 300)
        self.assertEqual(backend.loads[-1], (0, "three.bin"))
        self.assertEqual(vector.mask, 3)
        self.assertEqual(pool.steps, [100, 300])

    def test_unreferenced_bank_can_be_replaced_without_retirement(self):
        backend, vector = Backend(), Vector([1, 0])
        pool = SnapshotPool(backend, None, vector, 2, "initial.bin")
        pool.request("one.bin", 100)
        vector.references = [0, 2]
        self.assertEqual(pool.request("two.bin", 200)["event"], "loaded")
        self.assertEqual(vector.mask, 3)
        self.assertEqual(backend.loads[-1], (0, "two.bin"))


if __name__ == "__main__":
    unittest.main()
