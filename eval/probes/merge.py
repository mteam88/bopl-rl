"""Merge per-game npz files into memmapped arrays plus an index.

    native/artifacts/rl/venv/bin/python eval/probes/merge.py

Writes <dataset>/{obs,obs_now,meta,labels,rays16,rays32,rays32x3}.npy (rows in
game order, player-major inside a game) and <dataset>/index.json. The split
column of meta is filled here: test games are those with (game * 2654435761)
% 5 == 0 (about 20%, fixed by game id so every later step agrees).
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import schema  # noqa: E402
from generate import OUT  # noqa: E402

ARRAYS = ("obs", "obs_now", "meta", "labels", "rays16", "rays32", "rays32x3")


def test_game(game):
    return (int(game) * 2654435761) % 5 == 0


def merge_one(dataset, games_dir, name):
    """Rebuild one array from a regenerated games dir, checking every game's rows against index.json."""
    index = json.loads((dataset / "index.json").read_text())
    files = sorted(Path(games_dir).glob("*.npz"))
    expected = {g["game"]: g for g in index["game_index"]}
    if len(files) != len(expected):
        raise SystemExit(f"{len(files)} games regenerated, index has {len(expected)}")
    out = None
    for f in files:
        with np.load(f) as d:
            meta = d["meta"]; g = expected[int(meta[0, schema.META.index("game")])]
            if meta.shape[0] != g["rows"]:
                raise SystemExit(f"game {g['game']}: {meta.shape[0]} rows regenerated vs {g['rows']} in the index")
            block = d[name]
            if out is None:
                out = np.lib.format.open_memmap(dataset / f"{name}.npy", mode="w+", dtype=block.dtype, shape=(index["rows"], block.shape[1]))
            out[g["start"]:g["start"] + g["rows"]] = block
    out.flush()
    print(json.dumps(dict(array=name, rows=index["rows"], width=out.shape[1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--only", default=None, help="rebuild just this array from --games-dir (alignment-checked)")
    parser.add_argument("--games-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.only:
        merge_one(args.dataset, args.games_dir or args.dataset / "games", args.only)
        return
    files = sorted((args.dataset / "games").glob("*.npz"))
    if not files:
        raise SystemExit("no games")
    shapes, counts = {}, []
    for f in files:
        with np.load(f) as d:
            counts.append(d["meta"].shape[0])
            if not shapes:
                shapes = {k: (d[k].shape[1], d[k].dtype.str) for k in ARRAYS}
    total = int(sum(counts))
    outputs = {k: np.lib.format.open_memmap(args.dataset / f"{k}.npy", mode="w+", dtype=np.dtype(shapes[k][1]), shape=(total, shapes[k][0]))
               for k in ARRAYS}
    split_col = schema.META.index("split")
    game_col = schema.META.index("game")
    offset, games = 0, []
    for f, n in zip(files, counts):
        with np.load(f) as d:
            for k in ARRAYS:
                block = d[k]
                if k == "meta":
                    block = block.copy()
                    block[:, split_col] = int(test_game(block[0, game_col]))
                outputs[k][offset:offset + n] = block
            games.append(dict(game=int(d["meta"][0, game_col]), start=offset, rows=n, test=test_game(d["meta"][0, game_col])))
        offset += n
    for k in ARRAYS:
        outputs[k].flush()
    index = dict(rows=total, games=len(files), arrays={k: dict(width=shapes[k][0], dtype=shapes[k][1]) for k in ARRAYS},
                 meta=list(schema.META), labels=list(schema.LABELS), ray_sets=schema.RAY_SETS, compact_size=schema.COMPACT_SIZE,
                 game_index=games)
    (args.dataset / "index.json").write_text(json.dumps(index))
    print(json.dumps(dict(rows=total, games=len(files), test_rows=int(sum(g["rows"] for g in games if g["test"])))))


if __name__ == "__main__":
    main()
