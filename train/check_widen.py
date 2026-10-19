"""Does a widened checkpoint still compute its source's policy? Measured on real games.

    native/artifacts/rl/venv/bin/python train/check_widen.py SOURCE.bin OPPONENT.bin WIDE.bin [WIDE.bin ...] \\
        --pairs 4 --out check.json

SOURCE plays OPPONENT on the all-combos schedule (league/all_combos.py: random kits,
side-swapped seed pairs) through the CPU evaluator. Every WIDE net is stepped on the
same observation rows as SOURCE, carrying its own recurrent state, so a drift that
compounds through the state is measured where it would happen in play. Per decision:
KL(source || wide) of each action head under the environment's mask (the factorised
policy's KL is their sum), and the value difference. Teacher-forced on SOURCE's
actions, so this measures the function, not game outcomes (all_combos.py does that).
"""
import argparse
import json
from pathlib import Path
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'league'), str(ROOT / 'eval'), str(ROOT / 'native/tools')]
import all_combos  # noqa: E402
import elo_ladder  # noqa: E402
import games as league_games  # noqa: E402
import layout  # noqa: E402
import map_catalogue  # noqa: E402
import registry  # noqa: E402
from native_policy import load  # noqa: E402

HEADS = ('aim', 'buttons', 'hold')


def log_softmax(row):
    finite = np.isfinite(row)
    out = np.full(row.shape, -np.inf)
    top = row[finite].max()
    out[finite] = row[finite] - top - np.log(np.exp(row[finite] - top).sum())
    return out


def head_kl(p_row, q_row):
    lp, lq = log_softmax(p_row), log_softmax(q_row)
    keep = np.isfinite(lp)
    return float((np.exp(lp[keep]) * (lp[keep] - lq[keep])).sum())


class Shadowed(elo_ladder.Model):
    """SOURCE as a player, with the wide nets stepped beside it on its observations."""
    needs_value = True

    def __init__(self, name, path, wides):
        super().__init__(name, path, 'candidate')
        self.wides = [(Path(p).name, load(Path(p).resolve())[0]) for p in wides]
        self.lock = threading.Lock()
        self.rows = {label: [] for label, _ in self.wides}

    def start(self, task, side, rng):
        import torch
        policy = self.policy
        state = policy.initial(2, 'cpu')
        wide_states = [wide.initial(2, 'cpu') for _, wide in self.wides]
        bounds = np.cumsum([0, *layout.ACTION_SIZES])

        def head(row, index):
            part = row[bounds[index]:bounds[index + 1]]
            weights = np.exp(part - part.max())
            return rng.choice(len(part), p=weights / weights.sum())

        def act(observation, mask):
            nonlocal state
            batch = np.zeros((2, policy.obs_size), dtype=np.float32)
            batch[side] = observation[side, :policy.obs_size]
            allowed = np.asarray(mask[side]) != 0
            with torch.no_grad():
                batch = torch.from_numpy(batch)
                rows, state = policy.step_raw(batch, state)
                logits, value = policy.game_logits(rows)
                source = np.where(allowed, elo_ladder.as_numpy(logits)[side].astype(np.float64), -np.inf)
                found = []
                for i, (label, wide) in enumerate(self.wides):
                    wide_rows, wide_states[i] = wide.step_raw(batch, wide_states[i])
                    wide_logits, wide_value = wide.game_logits(wide_rows)
                    other = np.where(allowed, elo_ladder.as_numpy(wide_logits)[side].astype(np.float64), -np.inf)
                    kls = [head_kl(source[bounds[h]:bounds[h + 1]], other[bounds[h]:bounds[h + 1]]) for h in range(3)]
                    found.append((label, kls + [float(wide_value[side] - value[side]), float(value[side])]))
            with self.lock:
                for label, row in found:
                    self.rows[label].append(row)
            return np.asarray([head(source, 0), head(source, 1), head(source, 2)], dtype=np.int64)
        return act


def summary(rows):
    a = np.asarray(rows)
    kl = a[:, :3].sum(1)
    out = dict(decisions=int(len(a)), kl_mean=float(kl.mean()), kl_p99=float(np.quantile(kl, .99)),
               kl_max=float(kl.max()))
    for h, name in enumerate(HEADS):
        out[f'kl_{name}_mean'] = float(a[:, h].mean())
        out[f'kl_{name}_max'] = float(a[:, h].max())
    dv = a[:, 3]
    out.update(value_rmse=float(np.sqrt((dv ** 2).mean())), value_absmax=float(np.abs(dv).max()),
               value_sd=float(a[:, 4].std()), value_corr=float(np.corrcoef(a[:, 4], a[:, 4] + dv)[0, 1]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path)
    p.add_argument('opponent', type=Path)
    p.add_argument('wide', nargs='+', type=Path)
    p.add_argument('--pairs', type=int, default=4)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seed-base', type=int, default=211001)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)

    built = {'A': Shadowed('A', a.source.resolve(), a.wide),
             'B': elo_ladder.Model('B', a.opponent.resolve(), 'candidate')}
    league_games.configure(a.out.parent / 'check-widen-workers')
    todo = all_combos.tasks(a.pairs, a.seed_base, map_catalogue.parse_map_ids(registry.MAPS))
    played = league_games.play(built, todo, a.workers, a.seed_base)
    result = dict(source=str(a.source), opponent=str(a.opponent), seed_base=a.seed_base, games=len(played),
                  wide={label: summary(rows) for label, rows in built['A'].rows.items()})
    print(json.dumps(result, indent=1))
    a.out.write_text(json.dumps(result, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
