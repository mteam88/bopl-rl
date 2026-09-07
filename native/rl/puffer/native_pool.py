"""Keep frozen snapshots unchanged until every match using them has ended."""
from pathlib import Path


class SnapshotPool:
    def __init__(self, backend, trainer, vector, banks, initial, step=0):
        if banks < 2 or banks > 8:
            raise ValueError("A refreshing historical pool requires two to eight banks")
        self.backend, self.trainer, self.vector = backend, trainer, vector
        self.loaded = [None] * banks
        self.valid = {0}
        self.pending = None
        self.draining = None
        self._load(0, (Path(initial), step))
        self._publish()

    def _publish(self):
        self.vector.set_valid_banks(sum(1 << bank for bank in self.valid))

    def _load(self, bank, snapshot):
        path, step = snapshot
        self.backend.load_frozen_bank(self.trainer, bank, str(path))
        self.loaded[bank] = (path, step)

    @property
    def steps(self):
        return sorted(item[1] for item in self.loaded if item is not None)

    def request(self, path, step):
        """Retain the newest pending snapshot if match completion is delayed."""
        self.pending = (Path(path), step)
        return self.service()

    def service(self):
        """Call only between rollouts, while the environment workers are idle."""
        if self.pending is None:
            return None
        references = self.vector.bank_references()
        if len(references) != len(self.loaded):
            raise RuntimeError("Frozen bank reference count changed shape")
        bank = self.draining
        if bank is not None and references[bank]:
            return None
        if bank is None:
            available = [i for i, count in enumerate(references) if count == 0]
            if available:
                bank = min(available, key=lambda i: -1 if self.loaded[i] is None else self.loaded[i][1])
            else:
                # Prevent new matches from selecting the oldest bank. Matches
                # already using it retain its weights until their own reset.
                bank = min(self.valid, key=lambda i: self.loaded[i][1])
                self.valid.remove(bank)
                self.draining = bank
                self._publish()
                return dict(event="retiring", bank=bank, step=self.loaded[bank][1],
                            pending_step=self.pending[1], references=references[bank])
        snapshot = self.pending
        self._load(bank, snapshot)
        self.valid.add(bank)
        self.pending = None
        self.draining = None
        self._publish()
        return dict(event="loaded", bank=bank, step=snapshot[1], path=str(snapshot[0]))
