"""Validated learner/frozen routing for one native PufferLib trainer."""
import ctypes as C

import numpy as np

import counters


class NativeVector:
    """Native routing of `workers` two-player games over `buffers` buffers and `banks` frozen banks.

    `every` is binding.c's historical_every: every `every`-th environment is a
    historical one (learner vs a frozen bank, 1 + banks slots), the rest are
    learner-vs-learner self-play (2 slots). 2 (the default) alternates them, so
    2/3 of the learner's rows are self-play; 1 makes every game a historical
    one, for an exploiter that trains only against its frozen target(s). A
    group is `every - 1` self-play games followed by one historical game.

    `team_every` is binding.c's: 0 none, else the self-play game of every
    `team_every`-th group is a 2v2 team game of four learner players (every 2
    only). Groups are counted over the whole vector, so it must divide the
    groups of a buffer for every buffer to hold the same rows.
    """
    def __init__(self, backend, workers, buffers, banks, every=2, team_every=0):
        if every not in (1, 2):
            raise ValueError('historical_every must be 1 or 2')
        if workers < 2 or workers % (every * buffers) or not 2 <= banks <= 8:
            raise ValueError('Each buffer requires complete self-play/historical groups and 2..8 banks')
        if (workers // every) % banks:
            raise ValueError('Historical games must be dealt evenly to the banks')
        self.backend, self.workers, self.buffers, self.banks = backend, workers, buffers, banks
        self.every = every
        self.groups = workers // (every * buffers)   # groups per buffer
        if team_every < 0 or (team_every and (every != 2 or self.groups % team_every)):
            raise ValueError('team_every needs historical_every 2 and must divide the groups of a buffer')
        self.team_every = team_every
        # Players of each group's self-play game (4 in a team group), over one buffer's groups.
        self.selfplay_players = [4 if team_every and g % team_every == team_every - 1 else 2 for g in range(self.groups)]
        # Learner rows of one group: every row of each self-play game, slot 0 of the historical one.
        self.group_primaries = [p * (every - 1) + 1 for p in self.selfplay_players]
        self.team_rows = sum(p for p in self.selfplay_players if p == 4) * buffers
        # Physical decisions per vector step: two players a game, two more in each team game.
        self.physical_per_step = 2 * workers + self.team_rows // 2
        self.agents_per_buffer = sum(self.group_primaries) + self.groups * banks
        self.total_agents = self.agents_per_buffer * buffers
        self.historical_per_bank = workers // every // banks
        # Compact row layout: a historical environment's
        # opponent plays on one of its `banks` slots for the environment's whole
        # life, so 3 of 4 opponent rows never hold a decision. When every
        # buffer deals its historical games evenly to the banks, each buffer's
        # rows are packed [learner rows | bank 0 live rows | ... | bank n-1
        # live rows | idle rows]; the trainer's bank_layout covers the live
        # prefix and the idle tail is never uploaded, run through the policy or
        # sampled. Otherwise (groups % banks != 0) the legacy layout puts every
        # opponent row of bank b in bank b's slice.
        self.compact = self.groups % banks == 0
        self.primary_rows = sum(self.group_primaries)
        self.learner_rows = self.primary_rows * buffers
        self.live_per_bank = self.groups // banks if self.compact else self.groups
        self.live_rows = self.primary_rows + banks * self.live_per_bank
        self.lib = counters.bind(C.CDLL(backend.__file__))
        for name, types in {
            'bopl_vec_handles': [C.POINTER(C.c_void_p), C.c_int],
            'bopl_vec_reset': [C.c_void_p],
            'bopl_vec_layout': [C.c_void_p, C.POINTER(C.c_int), C.c_int],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types, C.c_int

    def layout(self):
        out = np.empty((self.workers, 22), dtype=np.int32)
        count = self.lib.bopl_vec_layout(C.c_void_p(self.gpu_obs_ptr),
                                        out.ctypes.data_as(C.POINTER(C.c_int)), out.size)
        if count != self.workers:
            raise RuntimeError('Native environment count changed')
        return out

    def expected_layout(self):
        """binding.c's environments as columns [num_agents, players, banks, tag, active_bank] of layout().

        Historical game j (every `every`-th environment) is pinned to bank
        j % banks and carries PufferLib's selfplay tag 1 + bank.
        """
        expected = []
        for j in range(self.workers // self.every):
            bank, players = j % self.banks, self.selfplay_players[j % self.groups]
            expected += [[players, players, 0, 0, -1]] * (self.every - 1) + [[1 + self.banks, 2, self.banks, 1 + bank, bank]]
        return expected

    def permutation(self, before):
        """The trainer's agent permutation (slot -> physical row) for the layout() rows `before`."""
        if before[:, [0, 1, 2, 3, 5]].tolist() != self.expected_layout():
            raise RuntimeError('Expected self-play/historical groups with bank-pinned historical games')
        perm = np.full(self.total_agents, -1, dtype=np.int32)
        slot = 0
        for buffer in range(self.buffers):
            primary, frozen, idle = [], [[] for _ in range(self.banks)], []
            for group in range(self.groups):
                own = self.group_primaries[group]
                primary.extend(range(slot, slot + own))
                active = int(before[self.every * (buffer * self.groups + group) + self.every - 1, 5])
                for bank in range(self.banks):
                    (frozen[bank] if not self.compact or bank == active else idle).append(slot + own + bank)
                slot += own + self.banks
            if self.compact and any(len(rows) != self.live_per_bank for rows in frozen):
                raise RuntimeError('Historical games are not dealt evenly to the banks within a buffer')
            order = primary + [row for group in frozen for row in group] + idle
            perm[order] = np.arange(self.agents_per_buffer) + buffer * self.agents_per_buffer
        if sorted(perm.tolist()) != list(range(self.total_agents)):
            raise RuntimeError('Invalid native agent permutation')
        return perm

    def attach(self, trainer):
        handles = (C.c_void_p * 1)()
        if self.lib.bopl_vec_handles(handles, 1) != 1 or self.backend.num_envs(trainer) != self.workers:
            raise RuntimeError('Training requires exactly one vector with the requested game count')
        self.gpu_obs_ptr = handles[0]
        before = self.layout()
        perm = self.permutation(before)
        self.backend.set_agent_perm(trainer, perm)
        self.backend.set_env_tags(trainer, np.ascontiguousarray(before[:, 3], dtype=np.int32))
        after, slot = self.layout(), 0
        for row in after:
            size = int(row[0])
            if row[8:8+size].tolist() != perm[slot:slot+size].tolist():
                raise RuntimeError('Native environment did not install the trainer permutation')
            slot += size

    def reset(self):
        if self.lib.bopl_vec_reset(C.c_void_p(self.gpu_obs_ptr)):
            raise RuntimeError('Native post-routing reset failed')

    def snapshot(self):
        return counters.snapshot(self.lib, self)
