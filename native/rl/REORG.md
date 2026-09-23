# The reorganization, and the decisions behind it

The repo grew by accretion: one directory per experiment, dead Python left
behind whenever an approach was abandoned, and measurement records filed in
three different places. This is the approved plan to reorganize it around
AlphaStar's workflow, and the standing decisions that shape it. It exists so
that the plan survives a session ending.

Approved 2026-09-23. Little backwards compatibility is required: the work is in
git, and retraining the best checkpoints takes a few hours.

## Target layout

| directory | holds |
|---|---|
| `sim/` | the simulator: the game's C# (the oracle) and the Rust port |
| `train/` | the trainer, the policy, PFSP opponent selection |
| `league/` | the league registry and its members - a permanent artifact in git |
| `eval/` | evaluation, the gates, the divergence instrument |
| `board/` | the dashboard |
| `docs/` | everything written down |

## Decisions

**One generalist, not a population of mains.** A single main agent, on the
expectation that it ends up with a better grasp of dynamic strategy than
several narrower mains would.

**Exploiters start from random init, not from a checkpoint.** An exploiter
seeded from the generalist inherits the generalist's habits, and we want
exploitive ability combinations that do *not* transfer the way we would guess.
Random init is the point, not a cost.

**The league is a permanent artifact in git.** It grows: exploiters get added,
and its strength grows with them. It is not rebuilt per run. Ratings are only
comparable within one (sim hash, reward semantics, loadout) condition, which is
why the registry keys payoff counts by condition.

**Promotion stays implicit.** No hardcoded promotion rule. What is needed is the
framework for organizing what sits in which pool, plus the recognition that
pools saturate - which has already happened here.

**No held-out abilities as the generalization split.** The abilities are very
different from each other and hard to learn; holding out nine of 38 tests
whether a policy can learn nine new abilities, not whether it learned the game.
Maps are the generalization axis.

**Surgery stays available but stays simple.** It is a worse fit for this problem
than it was for AlphaStar, so it should be as simple as possible.

**q32 is a measurement instrument, not a goal** (2026-09-23). See the
"Number type" section of `RUST_ROLLOUTS.md` for the evidence and the working
rule that follows from it.

## Deletion inventory

Deleting is most of the work. Verified dead, to be removed as the reorg lands:

- ~3,200 lines of dead Python now, ~4,350 once PFSP replaces the old opponent
  selection (`--swap-winrate`, the banks, `--elo-init`/`--elo-k`)
- `review/` - 9 files of prose from a completed code review
- `reports/` - removed 2026-09-23; it was a single stale `.pyc`
- `evidence/` - 38 dated measurement records, ~2.0M. **Decided: delete**
  (2026-09-23). It is the only *data* category in the inventory, so it needs a
  deliberate call rather than being swept up. Two things to respect when doing
  it: the records are committed and pushed, so `git log` remains the recovery
  path, and **ten documents link into `evidence/` and `review/`** -
  `NATIVE_HANDOFF.md`, `MANY_WORLDS.md`, `SIM_OVERHAUL.md`, `THROUGHPUT.md`,
  `ABILITY_PORT_GAPS.md`, `README.md`, `autoresearch/notes.md`,
  `sim/PORTING.md`. Fix those links in the same change that removes the
  directories, which is why this waits for the reorg rather than happening
  separately.

## Order of work

1. The reorg itself: move to the layout above, execute the deletion inventory.
2. PFSP into the trainer, replacing the old opponent selection.
3. Minimal surgery.

Then the first generalist run over randomised loadouts, the first random-init
exploiter, and time-to-exploit as the headline metric.

## Naming

Simple and easy to understand. No version numbers, no `_v2` suffixes, no
backwards-compatibility shims. If a name needs a number to disambiguate it from
its predecessor, the predecessor should have been deleted.
