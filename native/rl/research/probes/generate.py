"""Generate the probe dataset: Rust-sim rollouts with privileged labels.

Each game is written as one .npz under <out>/games; `merge.py` turns them
into the memmapped dataset the probes read. Deterministic: a game is a
function of (seed, map, controllers).

    nice -n 15 native/artifacts/rl/venv/bin/python native/rl/research/probes/generate.py --workers 4
"""
import argparse
import json
from multiprocessing import Process
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "sweep"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import probe_env  # noqa: E402  (must precede `environment`)
from probe_env import ProbeEnvironment, IDLE_NAMES, GEOMETRY_NAMES  # noqa: E402
from environment import configure  # noqa: E402
from native_policy import load, decide  # noqa: E402
from evaluation_protocol import scripted_opponent  # noqa: E402
import layout  # noqa: E402
import schema  # noqa: E402
from telemetry import FIELDS  # noqa: E402

ROOT = HERE.parents[3]
ART = ROOT / "native/artifacts/rl"
GAME = ROOT / "native/target/release/libbopl_sim.so"
POOL = ART / "sweeps/anchors/pool-v7/pool.json"
FINAL = ART / "sweeps/overnight-9/024/checkpoints/policy-2313256960.bin"
EARLY = ART / "sweeps/overnight-9/024/checkpoints/policy-320012288.bin"
OUT = ART / "probes/dataset"
MAPS = (0, 1, 2)

# Controller table (index = meta 'controller'); kind: random | checkpoint | scripted.
CONTROLLERS = [dict(name="random", kind="random"),
               dict(name="024-final", kind="checkpoint", path=str(FINAL)),
               dict(name="024-early-320m", kind="checkpoint", path=str(EARLY))]
_pool = json.loads(POOL.read_text())
for member in _pool["members"]:
    if member["kind"] == "checkpoint":
        CONTROLLERS.append(dict(name=member["name"], kind="checkpoint", path=str(POOL.parent / member["path"])))
for name in ("archer-hard", "pursuer-hard", "evasive-hard", "mixed-hard", "mixed-medium", "search-medium"):
    CONTROLLERS.append(dict(name=name, kind="scripted"))
INDEX = {c["name"]: i for i, c in enumerate(CONTROLLERS)}


def schedule(maps=MAPS, scale=1.0, base_game=0, base_seed=500_000):
    """The game list: (game id, controller index per player, map, seed). Side-swapped seed pairs.

    The default is the training-map set (maps 0-2). `heldout_schedule` is the same plan at a quarter
    of the volume on maps 3-4, which 024 never trained on, for the map-memorisation probes."""
    games = []
    pool_names = [c["name"] for c in CONTROLLERS if c["kind"] == "checkpoint" and c["name"].startswith("v3b")]
    scripted = ["archer-hard", "pursuer-hard", "evasive-hard", "mixed-hard", "mixed-medium"]
    plan = [("024-final", name, 96) for name in pool_names]
    plan += [("024-final", name, 96) for name in scripted]
    plan += [("024-final", "search-medium", 48), ("024-final", "random", 24), ("024-final", "024-final", 192),
             ("024-final", "024-early-320m", 192)]
    plan += [("024-early-320m", name, 24) for name in pool_names]
    plan += [("024-early-320m", name, 24) for name in scripted]
    seed = base_seed
    for focal, other, count in plan:
        count = max(2, int(round(count * scale / 2)) * 2)
        for i in range(count):
            side = i % 2
            controllers = [INDEX[focal], INDEX[other]] if side == 0 else [INDEX[other], INDEX[focal]]
            games.append(dict(game=base_game + len(games), controllers=controllers, map=maps[(i // 2) % len(maps)],
                              seed=seed + i // 2, focal=focal, other=other))
        seed += 1000
    return games


def heldout_schedule():
    return schedule(maps=(3, 4), scale=0.25, base_game=10_000, base_seed=900_000)


class Runner:
    def __init__(self, out, log_dir):
        configure(log_dir, GAME)
        self.out = Path(out)
        self.policies = {}
        self.meta = json.loads(Path(str(FINAL) + ".json").read_text())
        reward = self.meta["config"]["reward"]
        self.env = ProbeEnvironment(seed=1, limit=self.meta["config"]["episode_ticks"], delay=self.meta["config"]["observation_delay_ticks"], reward=reward)
        if not self.env.forks():
            raise RuntimeError("the probe labels need the forkable Rust sim")
        self.reward_columns = [FIELDS.index(k) for k in ("reward_outcome", "reward_kills", "reward_deaths")]

    def policy(self, index):
        spec = CONTROLLERS[index]
        if spec["path"] not in self.policies:
            self.policies[spec["path"]] = load(spec["path"])[0]
        return self.policies[spec["path"]]

    def play(self, game):
        env = self.env
        env.reset(game["seed"], game["map"])
        controllers = game["controllers"]
        env.bot(0, 0); env.bot(1, 0)
        states, rng = {}, np.random.default_rng(game["seed"] * 11 + 1)
        generator = torch.Generator().manual_seed(game["seed"] * 7 + 3)
        for p in range(2):
            spec = CONTROLLERS[controllers[p]]
            if spec["kind"] == "scripted":
                env.bot(p, *scripted_opponent(spec["name"]))
            elif spec["kind"] == "checkpoint":
                states[p] = self.policy(controllers[p]).initial(1, "cpu")
        rows = dict(obs=[], obs_now=[], meta=[], labels=[], **{k: [] for k in schema.RAY_SETS})
        rewards, decision = [], 0
        while not env.ended:
            obs = env.obs.copy(); mask = env.mask.copy()
            live, observed = env.live_state_pointer(), env.observed_state_pointer()
            live_state = live.contents
            actions = np.zeros((2, layout.NUM_ATNS), dtype=np.float32)
            values = np.full(2, np.nan, dtype=np.float32)
            for p in range(2):
                spec = CONTROLLERS[controllers[p]]
                if spec["kind"] == "checkpoint":
                    with torch.no_grad():
                        _, value, action, states[p] = decide(self.policy(controllers[p]), torch.from_numpy(obs[p:p + 1]),
                                                             mask[p:p + 1], states[p], generator)
                    actions[p] = action[0].numpy(); values[p] = float(value[0])
                elif spec["kind"] == "random":
                    actions[p] = [rng.integers(layout.ACT_AIM), rng.integers(layout.ACT_BUTTONS), 0]
            # Privileged labels from the live state, before the decision is applied.
            future = env.idle_future(schema.IDLE_HORIZON)
            obs_now = env.pack(live)
            obs_now[:, layout.OBS_PREVIOUS_OFFSET:] = obs[:, layout.OBS_PREVIOUS_OFFSET:]
            compact_obs, dropped = schema.compact(obs)
            compact_now, dropped_now = schema.compact(obs_now)
            for p in range(2):
                o = 1 - p
                geometry = env.geometry(live, p)
                ids_obs, n_obs = env.entity_rows(observed, p)
                ids_now, n_now = env.entity_rows(live, p)
                f = dict(zip(IDLE_NAMES, future[p])); fo = dict(zip(IDLE_NAMES, future[o])); g = dict(zip(GEOMETRY_NAMES, geometry))
                killer = int(f["killer_id"])
                killer_row = int(np.flatnonzero(ids_obs[:n_obs] == killer)[0]) if killer and (ids_obs[:n_obs] == killer).any() else -1
                killer_row_now = int(np.flatnonzero(ids_now[:n_now] == killer)[0]) if killer and (ids_now[:n_now] == killer).any() else -1
                death, cause = f["death_tick"], int(f["cause"])
                died = death >= 0
                label = dict(
                    idle_death_tick=death, idle_cause=cause,
                    hit_15=float(died and cause == 3 and death <= 15), hit_30=float(died and cause == 3 and death <= 30),
                    hit_60=float(died and cause == 3), self_hit_60=float(died and cause == 4),
                    hit_inflight_30=float(died and cause == 3 and death <= 30 and f["killer_known"] > 0),
                    hit_inflight_60=float(died and cause == 3 and f["killer_known"] > 0), killer_known=f["killer_known"],
                    fall_30=float(died and cause == 1 and death <= 30), fall_60=float(died and cause == 1),
                    squish_60=float(died and cause == 2), any_death_30=float(died and death <= 30), any_death_60=float(died),
                    own_arrow_in_flight=f["own_arrow_in_flight"],
                    my_shot_hits_60=float(f["own_arrow_in_flight"] > 0 and fo["death_tick"] >= 0 and int(fo["cause"]) == 3),
                    idle_x10=f["x10"], idle_y10=f["y10"], idle_x30=f["x30"], idle_y30=f["y30"],
                    idle_opp_x10=fo["x10"], idle_opp_y10=fo["y10"], idle_opp_x30=fo["x30"], idle_opp_y30=fo["y30"],
                    **{k: g[k] for k in GEOMETRY_NAMES if k != "view_ok"},
                    x=live_state.players[p].x * probe_env.FIX, y=live_state.players[p].y * probe_env.FIX,
                    opp_x=live_state.players[o].x * probe_env.FIX, opp_y=live_state.players[o].y * probe_env.FIX,
                    opp_alive=float(live_state.players[o].flags & 1),
                    value_online=values[p])
                rows["labels"].append(np.array([label.get(k, np.nan) for k in schema.LABELS], dtype=np.float32))
                latched = int(not mask[p, layout.ACT_AIM + layout.ACT_BUTTONS:].all())
                rows["meta"].append(np.array([game["game"], decision, p, controllers[p], controllers[o], game["map"], game["seed"],
                                              live_state.header.episode_ticks, actions[p, 0], actions[p, 1], actions[p, 2], latched,
                                              killer_row, killer_row_now, n_obs, n_now, dropped[p], dropped_now[p], 0, 0], dtype=np.int32))
                rows["obs"].append(compact_obs[p].astype(np.float16)); rows["obs_now"].append(compact_now[p].astype(np.float16))
                for name, (n, extra, hits) in schema.RAY_SETS.items():
                    rows[name].append(env.rays(live, p, n, extra, hits).reshape(-1).astype(np.float16))
            env.step(actions)
            rewards.append(env.telemetry[:, self.reward_columns].sum(1).copy())
            decision += 1
        return self.finish(game, rows, np.asarray(rewards, dtype=np.float32), decision)

    def finish(self, game, rows, rewards, decisions):
        """Realised-future labels along the rollout, then one npz per game."""
        final = self.env.state
        timeout = bool(final.header.flags & 2)
        labels = np.stack(rows["labels"]).reshape(decisions, 2, len(schema.LABELS))
        meta = np.stack(rows["meta"]).reshape(decisions, 2, len(schema.META))
        L = {k: i for i, k in enumerate(schema.LABELS)}
        ticks = meta[:, 0, schema.META.index("tick")].astype(np.float32)
        end_tick = float(final.header.episode_ticks)
        for p in range(2):
            o = 1 - p
            me = final.players[p]
            win = bool(me.flags & 2) and not timeout
            x, y = labels[:, p, L["x"]], labels[:, p, L["y"]]
            ox, oy = labels[:, o, L["x"]], labels[:, o, L["y"]]

            def shift(a, k):
                s = np.full(decisions, np.nan, dtype=np.float32)
                if decisions > k:
                    s[:-k] = a[k:]
                return s
            labels[:, p, L["real_x12"]] = shift(x, 3); labels[:, p, L["real_y12"]] = shift(y, 3)
            labels[:, p, L["real_opp_x12"]] = shift(ox, 3); labels[:, p, L["real_opp_y12"]] = shift(oy, 3)
            labels[:, p, L["real_opp_x32"]] = shift(ox, 8); labels[:, p, L["real_opp_y32"]] = shift(oy, 8)
            # My real death: the player is dead at the end unless it won or timed out (one life per round).
            died = not bool(me.flags & 1)
            death_tick = end_tick if died else np.inf
            labels[:, p, L["real_death_tick"]] = death_tick - ticks
            labels[:, p, L["real_death_60"]] = ((death_tick - ticks) <= 60).astype(np.float32)
            telemetry = final.telemetry[p]
            cause = 0 if not died else (2 if me.cause_of_death == 1 else 1 if telemetry.deaths_environment else 3 if telemetry.deaths_opponent else 4 if telemetry.deaths_self else 5)
            labels[:, p, L["real_cause"]] = cause
            labels[:, p, L["win"]] = float(win); labels[:, p, L["loss"]] = float(not win and not timeout); labels[:, p, L["timeout"]] = float(timeout)
            labels[:, p, L["ticks_to_end"]] = end_tick - ticks
            r = rewards[:, p]
            labels[:, p, L["reward"]] = r
            ret = np.zeros(decisions, dtype=np.float64); acc = 0.0
            for t in range(decisions - 1, -1, -1):
                acc = r[t] + schema.GAMMA * acc; ret[t] = acc
            labels[:, p, L["return_disc"]] = ret
        meta[-1, :, schema.META.index("terminal")] = 1
        # Player-major order inside a game (the MinGRU scan reads contiguous sequences).
        order = np.arange(decisions * 2).reshape(decisions, 2).T.reshape(-1)
        out = dict(obs=np.stack(rows["obs"])[order], obs_now=np.stack(rows["obs_now"])[order],
                   meta=meta.reshape(-1, len(schema.META))[order], labels=labels.reshape(-1, len(schema.LABELS))[order],
                   **{k: np.stack(rows[k])[order] for k in schema.RAY_SETS})
        path = self.out / "games" / f"{game['game']:05d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        os.replace(tmp, path)
        return dict(game=game["game"], decisions=decisions, ticks=int(end_tick), timeout=timeout,
                    winner=[int(bool(final.players[p].flags & 2)) for p in range(2)], focal=game["focal"], other=game["other"])


def worker(index, workers, games, out):
    torch.set_num_threads(1)
    os.nice(15)
    runner = Runner(out, Path(out) / "workers" / f"{index}")
    for game in games[index::workers]:
        path = Path(out) / "games" / f"{game['game']:05d}.npz"
        if path.exists():
            continue
        started = time.monotonic()
        result = runner.play(game)
        result["seconds"] = round(time.monotonic() - started, 2)
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="only the first N games of the schedule")
    parser.add_argument("--heldout", action="store_true", help="the maps 3-4 supplement (game ids from 10000)")
    args = parser.parse_args()
    games = heldout_schedule() if args.heldout else schedule()
    if args.limit:
        games = games[:args.limit]
    (args.out / "games").mkdir(parents=True, exist_ok=True)
    name = "schedule-heldout.json" if args.heldout else "schedule.json"
    (args.out / name).write_text(json.dumps(dict(controllers=CONTROLLERS, games=games), indent=1))
    processes = [Process(target=worker, args=(i, args.workers, games, args.out)) for i in range(args.workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode:
            raise SystemExit(f"worker failed with {p.exitcode}")


if __name__ == "__main__":
    main()
