"""The RL map catalogue: RL map index <-> Unity level (native/data/map-set.json).

The Rust sim addresses maps by RL index (0-30, dense, stable); the game, the
Unity player's BOPL_LEVEL_SET hook and the human result rows address Unity
levels (0-123, scene 6+level, Assets/Scenes/FFA/Level{level+1}.unity). RL
indices 0-4 are levels 0-4; the more-maps catalogue appends 26 grass levels.
native/tools/extract_rollout_data.py writes the table and the campaign split.

CLI, for shells (the Mac launcher):
  map_set.py levels training|held-out|all|<rl,indices>   -> Unity levels, comma separated
  map_set.py rl <levels>                                  -> RL indices (levels outside the catalogue dropped)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / 'native/data/map-set.json'


def load(path=TABLE):
    return json.loads(Path(path).read_text())


def levels(table=None):
    """RL index -> Unity level."""
    table = table or load()
    return {m['rl']: m['level'] for m in table['maps']}


def rl_indices(table=None):
    """Unity level -> RL index."""
    return {level: rl for rl, level in levels(table).items()}


def role_of_level(level, table=None):
    """'training' | 'held-out' for catalogue levels, 'unported' for the rest."""
    table = table or load()
    for m in table['maps']:
        if m['level'] == level:
            return m['role']
    return 'unported'


def split(table=None):
    table = table or load()
    return {k: list(v) for k, v in table['split'].items()}


def parse_maps(spec, table=None):
    """A --maps argument -> RL indices: a role name, 'all', or a comma list."""
    table = table or load()
    if spec in ('training', 'held-out', 'held_out'):
        return list(table['split']['training' if spec == 'training' else 'held_out'])
    if spec == 'all':
        return [m['rl'] for m in table['maps']]
    return [int(x) for x in spec.split(',') if x.strip()]


def main(argv):
    if len(argv) < 3 or argv[1] not in ('levels', 'rl'):
        print(__doc__, file=sys.stderr)
        return 2
    table = load()
    if argv[1] == 'levels':
        table_levels = levels(table)
        out = [table_levels[rl] for rl in parse_maps(argv[2], table)]
    else:
        back = rl_indices(table)
        out = [back[int(x)] for x in argv[2].split(',') if x.strip() and int(x) in back]
    print(','.join(str(x) for x in out))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
