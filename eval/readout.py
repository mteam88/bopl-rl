#!/usr/bin/env python3
"""One-shot readout of a training run against baselines.

    python eval/readout.py <run dir> [--baseline NAME=<run dir>]... [--json out.json]

A run dir is a sweep trial (.../sweeps/<sweep>/<trial>) holding config.json,
metrics.jsonl, evaluation.log, evaluations/<step>/ and league/.
Prints, for the run and each baseline at the same evaluation steps:

  1. league ratings (league/policy-<step>.json, refitted on the members' current
     ratings): rating, interval, games, and the mean / median / IQM of the last
     three 'band' ratings, which is what the sweep scores;
  2. win rates on training maps vs held-out maps at each evaluation (the
     diagnostic suite's groups, all opponents pooled), per opponent optional;
  3. the value head: explained variance of the critic vs the realised
     gamma-return over the evaluation games with decision dumps
     (evaluations/<step>/*.mind.json.gz), next to the trainer's in-run
     explained_variance (vs its bootstrapped TD(lambda) target) at the same
     step, plus value / return standard deviations;
  4. the death taxonomy of the learner's losses (squish / fall / own arrow /
     opponent arrow / timeout / other) and of its kills, per evaluation.

Baselines default to 024 (overnight-9/024) and the 024 seed floor runs found
under sweeps/seed-024-s*/000.
"""
import argparse
import collections
import gzip
import json
import math
import os
import statistics
import sys
from pathlib import Path

import map_catalogue
import packed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "league"))
import candidate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sweeps_dirs():
    """Sweeps roots to search, in order: $BOPL_SWEEPS, the base project's when
    this tree is a rift (.rifts/<project>/<rift>; runs are written there), then
    this tree's own."""
    out = []
    if os.environ.get("BOPL_SWEEPS"):
        out.append(Path(os.environ["BOPL_SWEEPS"]))
    parts = ROOT.parts
    if ".rifts" in parts:
        i = parts.index(".rifts")
        out.append(Path(*parts[:i]) / parts[i + 1] / "native/artifacts/rl/sweeps")
    out.append(ROOT / "native/artifacts/rl/sweeps")
    return [d for d in out if d.is_dir()]


SWEEPS_DIRS = sweeps_dirs()
SWEEPS = SWEEPS_DIRS[0]


def reference():
    """The league's current member ratings, fitted once per readout."""
    global REFERENCE
    if REFERENCE is None:
        REFERENCE = candidate.reference()
    return REFERENCE


REFERENCE = None


def resolve(path):
    """A run dir given absolutely or relative to a sweeps root."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    for root in SWEEPS_DIRS:
        if (root / path).exists():
            return root / path
    return SWEEPS / path
DEFAULT_BASELINES = [("024", "overnight-9/024")]
GAMMA = 0.995   # train/bopl.ini; per decision, the unit the critic is trained in
ROLE = {"trained": "trained", "untrained": "heldout", "heldout": "heldout"}   # the diagnostic suite's map_role


# ---- loading ----
def read_jsonl(path):
    out = []
    if not packed.exists(path):
        return out
    with packed.open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


class Run:
    def __init__(self, name, path):
        self.name = name
        self.path = resolve(path)
        if not self.path.exists():
            raise SystemExit(f"{name}: no run at {path}")
        raw_config = json.loads((self.path / "config.json").read_text()) if (self.path / "config.json").exists() else {}
        raw_status = json.loads((self.path / "status.json").read_text()) if (self.path / "status.json").exists() else {}
        if (self.path / "result.json").exists():
            raw_status = json.loads((self.path / "result.json").read_text())
        self.catalogue = map_catalogue.catalogue_from_records(raw_config, raw_status)
        self.config = map_catalogue.normalize_record(raw_config, raw_status)
        self.status = map_catalogue.normalize_record(raw_status, raw_config)
        self.metrics = [map_catalogue.normalize_record(row, raw_config, raw_status)
                        for row in read_jsonl(self.path / "metrics.jsonl")]
        log = [map_catalogue.normalize_record(row, raw_config, raw_status)
               for row in read_jsonl(self.path / "evaluation.log")]
        self.diagnostics = {d["step"]: d for d in log if d.get("suite") == "diagnostic"}
        self.league = self._league()
        self.results = self._results()

    @property
    def steps(self):
        return sorted(self.diagnostics)

    def learner_ratio(self):
        """learner steps per native step (the learner's share of the agent rows)."""
        for m in reversed(self.metrics):
            if m.get("step") and m.get("learner_steps"):
                return m["learner_steps"] / m["step"]
        return 0.75

    def _league(self):
        """The run's league curve (league candidate.curve), [] when nothing is rated yet."""
        if not (self.path / "league").is_dir():
            return []
        return candidate.curve(self.path, reference())

    def _results(self):
        """The sweep driver's observations for this trial (results.jsonl beside the trial), if any."""
        results = self.path.parent / "results.jsonl"
        for r in read_jsonl(results):
            if r.get("id") == self.path.name:
                return r
        return None

    def metric_at(self, step, key, window=20):
        """Mean of `key` over the last `window` metric rows at or before `step`."""
        rows = [m[key] for m in self.metrics if m.get("step", 0) <= step and m.get(key) is not None]
        rows = rows[-window:]
        return statistics.fmean(rows) if rows else None

    # ---- evaluation games ----
    def win_rates(self, step):
        """{'trained': (wins, games), 'heldout': (wins, games), 'timeouts': n} from the diagnostic groups."""
        d = self.diagnostics.get(step)
        out = {"trained": [0, 0], "heldout": [0, 0], "timeouts": 0}
        if not d:
            return out
        for g in d.get("groups", []):
            role = self.map_role(g)
            out[role][0] += g["wins"]
            out[role][1] += g["games"]
            out["timeouts"] += g.get("timeouts", 0)
        return out

    def win_rates_by_opponent(self, step):
        d = self.diagnostics.get(step)
        out = collections.defaultdict(lambda: {"trained": [0, 0], "heldout": [0, 0]})
        if not d:
            return out
        for g in d.get("groups", []):
            role = self.map_role(g)
            out[g["opponent"]][role][0] += g["wins"]
            out[g["opponent"]][role][1] += g["games"]
        return out

    def map_role(self, group):
        map_id = group.get('map')
        if map_id in self.config.get('heldout_maps', ()):
            return 'heldout'
        if map_id in self.config.get('training_maps', ()):
            return 'trained'
        return ROLE.get(group.get('map_role', 'trained'), 'trained')

    def deaths(self, step):
        """Learner deaths and kills by cause over the evaluation's games.
        Learner: cause_of_death 1 = squished, 2 with an environment death = fall
        (water / blast), 2 with a self death = own arrow, 0 with an opponent
        death = opponent arrow (cross-tabulated on 024's 1600 losses); the
        opponent's telemetry is not recorded, so its accidents are one bucket."""
        d = self.diagnostics.get(step)
        me = collections.Counter()
        foe = collections.Counter()
        games = 0
        if not d:
            return me, foe, games
        for g in d.get("games", []):
            games += 1
            mine = g["final_players"][g["side"]]
            theirs = g["final_players"][1 - g["side"]]
            totals = g.get("behavior", {}).get("totals", {})
            if g.get("timeout"):
                me["timeout"] += 1
            elif g.get("deaths") or mine.get("stocks", 1) == 0 and not mine.get("flags", 1) & 1:
                code = mine.get("cause_of_death")
                if code == 1:
                    me["squish"] += 1
                elif totals.get("deaths_self"):
                    me["own arrow"] += 1
                elif totals.get("deaths_environment") or code == 2:
                    me["fall"] += 1
                elif totals.get("deaths_opponent") or code == 0:
                    me["opp arrow"] += 1
                else:
                    me["other"] += 1
            if g.get("kills") or (theirs.get("stocks", 1) == 0 and not theirs.get("flags", 1) & 1):
                code = theirs.get("cause_of_death")
                foe["squish" if code == 1 else "accident" if code == 2 else "arrow" if code == 0 else "other"] += 1
        return me, foe, games

    def value_ev(self, step, gamma=GAMMA):
        """Explained variance of the critic vs the realised discounted return
        over the evaluation's saved games (decision dumps). The return at
        decision t is gamma^(N-1-t) times the terminal reward (outcome plus the
        kill / death shaping; one stock, so kills and deaths end the game).
        Returns dict(ev, n, value_std, return_std, value_mean, return_mean, games) or None."""
        folder = self.path / "evaluations" / str(step)
        d = self.diagnostics.get(step)
        if not d or not folder.is_dir():
            return None
        reward = self.config.get("reward", {"win": 1, "loss": -1, "timeout": -1, "kill": 0, "death": 0})
        values, returns = [], []
        games = 0
        for g in d.get("games", []):
            mind = folder / f"{g['name']}.mind.json.gz"
            if not mind.exists():
                continue
            try:
                with gzip.open(mind, "rt") as f:
                    decisions = json.load(f).get("decisions", [])
            except (OSError, json.JSONDecodeError):
                continue
            v = [x["value"] for x in decisions if "value" in x]
            if not v:
                continue
            terminal = reward["timeout"] if g.get("timeout") else (reward["win"] if g["win"] else reward["loss"])
            terminal += reward.get("kill", 0) * g.get("kills", 0) + reward.get("death", 0) * g.get("deaths", 0)
            n = len(v)
            values.extend(v)
            returns.extend(terminal * gamma ** (n - 1 - t) for t in range(n))
            games += 1
        if len(values) < 2:
            return None
        rm = statistics.fmean(returns)
        var_r = sum((r - rm) ** 2 for r in returns)
        resid = sum((a - b) ** 2 for a, b in zip(values, returns))
        return dict(ev=1 - resid / var_r if var_r > 0 else float("nan"), n=len(values), games=games,
                    value_std=statistics.pstdev(values), return_std=statistics.pstdev(returns),
                    value_mean=statistics.fmean(values), return_mean=rm)


# ---- formatting ----
def fmt(x, width=7, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " " * (width - 1) + "-"
    if isinstance(x, float):
        return f"{x:{width}.{digits}f}"
    return f"{x:{width}}"


def pct(wins_games):
    w, g = wins_games
    return f"{100 * w / g:5.1f}%" if g else "    - "


def iqm(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    lo, hi = n / 4, 3 * n / 4
    total = weight = 0.0
    for i, x in enumerate(xs):
        w = max(0.0, min(i + 1, hi) - max(i, lo))
        total += w * x
        weight += w
    return total / weight if weight else None


def section(title):
    print()
    print(title)
    print("-" * len(title))


def league_scores(runs):
    section("1. League rating by checkpoint (rating, 95% interval, games, source); the sweep scores the IQM of the last three 'band' ratings")
    print("  source: band = the scorer's full adaptive rating (headline); in-run = the trainer's small rating of every"
          " evaluation checkpoint. Both are refitted on the members' current ratings.")
    for run in runs:
        if not run.league:
            print(f"  {run.name}: none yet")
            continue
        print(f"  {run.name}:")
        for d in run.league:
            lo, hi = d["interval"]
            print(f"    learner {(d['learner_steps'] or 0) / 1e6:7.0f}M  rating {fmt(d['rating'], 7, 1)}  [{fmt(lo, 6, 1)}, {fmt(hi, 6, 1)}]"
                  f"  games {d['games']:4d}  {d['source']}")
        band = [d["rating"] for d in run.league if d["source"] == "band"][-3:]
        if band:
            print(f"    last {len(band)} band: mean {fmt(statistics.fmean(band), 6, 1)}  median {fmt(statistics.median(band), 6, 1)}  IQM {fmt(iqm(band), 6, 1)}")
        if run.results and run.results.get("observations"):
            obs = run.results["observations"]
            print(f"    sweep observations: " + ", ".join(f"{o['score']:.1f}@{o['steps'] / 1e6:.0f}M" for o in obs) + f"  status {run.results.get('status')}")


def win_rates(runs, steps, by_opponent=False):
    section("2. Win rate, training maps vs held-out maps, all opponents pooled")
    ratio = runs[0].learner_ratio()
    print(f"  {'learner':>9}  " + "  ".join(f"{r.name:>26}" for r in runs))
    print(f"  {'':>9}  " + "  ".join(f"{'trained  heldout   gap':>26}" for _ in runs))
    for step in steps:
        cells = []
        for r in runs:
            w = r.win_rates(step)
            if not w["trained"][1] and not w["heldout"][1]:
                cells.append(f"{'-':>26}")
                continue
            gap = (w["trained"][0] / w["trained"][1] - w["heldout"][0] / w["heldout"][1]) * 100 if w["trained"][1] and w["heldout"][1] else None
            cells.append(f"{pct(w['trained'])}  {pct(w['heldout'])}  {fmt(gap, 5, 1)}pt")
        print(f"  {step * ratio / 1e6:8.0f}M  " + "  ".join(cells))
    if by_opponent:
        last = steps[-1]
        print(f"\n  per opponent at the last common evaluation ({last * ratio / 1e6:.0f}M learner steps): trained / heldout")
        opponents = sorted({o for r in runs for o in r.win_rates_by_opponent(last)})
        for o in opponents:
            cells = []
            for r in runs:
                w = r.win_rates_by_opponent(last).get(o)
                cells.append(f"{pct(w['trained'])} / {pct(w['heldout'])}" if w else f"{'-':>15}")
            print(f"  {o:>20}  " + "  ".join(f"{c:>26}" for c in cells))


def value_section(runs, steps, gamma):
    section(f"3. Value head: explained variance vs the realised gamma-return (gamma {gamma}, evaluation games with decision dumps) | in-run EV vs the bootstrapped target")
    ratio = runs[0].learner_ratio()
    print(f"  {'learner':>9}  " + "  ".join(f"{r.name:>38}" for r in runs))
    print(f"  {'':>9}  " + "  ".join(f"{'realised EV   in-run EV  V std  R std':>38}" for _ in runs))
    for step in steps:
        cells = []
        for r in runs:
            ev = r.value_ev(step, gamma)
            inrun = r.metric_at(step, "explained_variance")
            if ev is None and inrun is None:
                cells.append(f"{'-':>38}")
                continue
            cells.append(f"{fmt(ev['ev'] if ev else None, 8, 3)} ({ev['games'] if ev else 0:3d}g)  {fmt(inrun, 8, 3)}  {fmt(ev['value_std'] if ev else None, 5, 2)}  {fmt(ev['return_std'] if ev else None, 5, 2)}")
        print(f"  {step * ratio / 1e6:8.0f}M  " + "  ".join(cells))


def deaths_section(runs, steps):
    section("4. Death taxonomy over the evaluation games: how the learner lost (per 100 games) and how its kills happened")
    ratio = runs[0].learner_ratio()
    causes = ["squish", "fall", "own arrow", "opp arrow", "timeout", "other"]
    kill_causes = ["arrow", "squish", "accident", "other"]
    for r in runs:
        print(f"  {r.name}:")
        print(f"    {'learner':>9}  {'games':>5}  {'losses':>6} | " + " ".join(f"{c:>9}" for c in causes) + " || kills " + " ".join(f"{c:>8}" for c in kill_causes))
        for step in steps:
            me, foe, games = r.deaths(step)
            if not games:
                continue
            losses = sum(me.values())
            row = " ".join(f"{100 * me[c] / games:8.1f}%" for c in causes)
            kills = " ".join(f"{100 * foe[c] / games:7.1f}%" for c in kill_causes)
            print(f"    {step * ratio / 1e6:8.0f}M  {games:5d}  {losses:6d} | {row} || {sum(foe.values()):5d} {kills}")


def throughput(run):
    if not run.metrics:
        return
    last = run.metrics[-1]
    recent = run.metrics[-50:]
    mean = lambda k: statistics.fmean(m[k] for m in recent if m.get(k) is not None) if any(m.get(k) is not None for m in recent) else None
    st = run.status
    print(f"  {run.name}: {st.get('status', '?')}, step {last.get('step', 0) / 1e6:.0f}M native / {last.get('learner_steps', 0) / 1e6:.0f}M learner"
          f" of {st.get('target_learner_steps', run.config.get('steps', 0)) / 1e6:.0f}M, elapsed {st.get('elapsed', last.get('elapsed', 0)) / 3600:.2f} h;"
          f" last 50 updates: sps {fmt(mean('sps'), 8, 0)} learner_sps {fmt(mean('learner_sps'), 8, 0)}"
          f" train {fmt(mean('train_seconds'), 6, 3)} s rollout {fmt(mean('rollout_seconds'), 6, 3)} s wall {fmt(mean('wall_seconds'), 6, 3)} s/update;"
          f" set_overflow {last.get('set_overflow')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run dir (sweep trial), absolute or relative to native/artifacts/rl/sweeps")
    p.add_argument("--name", default=None, help="label for the run (default: <sweep>/<trial>)")
    p.add_argument("--baseline", action="append", default=None, metavar="NAME=DIR",
                   help="baseline run(s); default 024=overnight-9/024 plus every seed-024-s*/000")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--by-opponent", action="store_true", help="also print per-opponent win rates at the last common evaluation")
    p.add_argument("--all-steps", action="store_true", help="print every baseline evaluation step, not only the run's")
    p.add_argument("--json", default=None, help="write the collected numbers here as JSON")
    a = p.parse_args()

    run_path = resolve(a.run)
    name = a.name or "/".join(run_path.resolve().parts[-2:])
    runs = [Run(name, run_path)]
    baselines = a.baseline
    if baselines is None:
        baselines = [f"{n}={d}" for n, d in DEFAULT_BASELINES if resolve(d).exists()]
        seen = set()
        for root in SWEEPS_DIRS:
            for d in sorted(root.glob("seed-024-s*/000")):
                if d.parent.name not in seen:
                    seen.add(d.parent.name)
                    baselines.append(f"{d.parent.name.replace('seed-024-', '024-')}={d}")
    for spec in baselines:
        n, _, d = spec.partition("=")
        if not d:
            n, d = Path(spec).name, spec
        try:
            runs.append(Run(n, d))
        except SystemExit as e:
            print(f"skipping baseline: {e}", file=sys.stderr)

    steps = runs[0].steps
    if a.all_steps:
        steps = sorted({s for r in runs for s in r.steps})
    if not steps:
        print(f"{runs[0].name}: no evaluations yet")

    c = runs[0].config
    print(f"Readout of {runs[0].name}: encoder {c.get('encoder')}, hidden {c.get('hidden')}, layers {c.get('layers')}, seed {c.get('seed')},"
          f" map catalogue {c.get('map_catalogue')}, training maps {c.get('training_maps')},"
          f" held-out {c.get('heldout_maps')}, v4 shape {c.get('v4_shape')}")
    for r in runs:
        throughput(r)
        for resumed in r.config.get('resumed', []):
            # train.py --resume restores weights, pool, frozen banks, counters and
            # schedules; Muon's momentum buffers and the recurrent state restart cold.
            print(f"  {r.name}: resumed at {resumed['at_learner_steps']/1e6:.0f}M learner steps from"
                  f" {Path(resumed['checkpoint']).name} (steps {resumed['steps']}); the optimizer momentum"
                  f" restarted cold there, the one discontinuity of the recipe")
    league_scores(runs)
    if steps:
        win_rates(runs, steps, a.by_opponent)
        value_section(runs, steps, a.gamma)
        deaths_section(runs, steps)

    if a.json:
        out = {}
        for r in runs:
            out[r.name] = dict(
                path=str(r.path), status=r.status, config=r.config,
                league=[dict(learner_steps=d["learner_steps"], rating=d["rating"], interval=d["interval"], games=d["games"],
                             source=d["source"]) for d in r.league],
                win_rates={str(s): r.win_rates(s) for s in r.steps},
                value={str(s): r.value_ev(s, a.gamma) for s in r.steps},
                deaths={str(s): dict(zip(("learner", "kills", "games"), (dict(m), dict(f), g))) for s in r.steps for m, f, g in [r.deaths(s)]},
                in_run_ev={str(s): r.metric_at(s, "explained_variance") for s in r.steps},
            )
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
