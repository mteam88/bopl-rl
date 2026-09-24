"""The RL map catalogue: RL map index <-> Unity level (native/data/map-set.json).

The Rust sim addresses maps by RL index (dense and stable, 0-123); the game, the
Unity player's BOPL_LEVEL_SET hook and the human result rows address Unity
levels (0-123, scene 6+level, Assets/Scenes/FFA/Level{level+1}.unity). RL
indices are the same authored level IDs. native/data/map-set.json carries the
catalogue and the retained campaign split.

CLI, for shells (the Mac launcher):
  map_set.py levels training|held-out|all|<rl,indices>   -> Unity levels, comma separated
  map_set.py rl <levels>                                  -> RL indices (levels outside the catalogue dropped)
"""
import json
import sys
from pathlib import Path
from map_catalogue import parse_map_ids

ROOT = Path(__file__).resolve().parents[1]
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


def level_types(table=None):
    """RL index -> LevelType: grass 0, snow 1, space 2.

    Only space is mechanical. It halves the gravity modifier on players
    (PlayerPhysics.AddGravityFactor) and on ropes (DetPhysics.cs:374), removes
    water entirely (the water checks at DetPhysics.cs:1137,1180 are skipped) and
    puts the lower blast boundary at Camera_YMin instead of the water plane.
    Generic physics bodies are NOT affected - gravityAccel is a constant with no
    level-type factor - so projectile arcs are the same on every map. Snow
    differs from grass only cosmetically: no LevelType.snow test exists in the
    game's C# or in the port, so nothing should branch on 1 versus 0.
    """
    table = table or load()
    return {m['rl']: m['level_type'] for m in table['maps']}


def space_maps(table=None):
    """RL indices whose level type is space."""
    return [rl for rl, kind in level_types(table).items() if kind == 2]


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
    return parse_map_ids(spec)


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
