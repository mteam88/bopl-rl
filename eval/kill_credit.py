"""Who a death is credited to, the way training credits it under layout v4.2 (writer bit 16).

The game's own killer (event `actor`, and the kill counters and RlTelemetry deaths_* that follow
it) credits nobody for a Guided Missile's blast or a cast Spike: their explosions keep
PlayerOwnerId 0. The sim names the caster in the death event's `creator` (sim/abi/bopl_rl.h),
and train/bopl_puffer.h pays the caster's kill and charges an own Missile/Spike as a self death.
Every offense metric that reads "credited kills" goes through `credit`, so evaluation counts
what v4.2 training rewards; `game_credit` is the game's own attribution, for rows recorded
before events were kept (all_combos --games-out deaths counts).
"""
DEATH = 1   # BoplRlEvent.kind of a death (BOPL_EVENT_DEATH)


def credit(actor, creator, subject):
    """'opponent', 'self' or 'none' for a death of player id `subject`."""
    who = actor or creator
    return 'none' if not who or who == 0xFFFFFFFF else 'self' if who == subject else 'opponent'


def game_credit(actor, subject):
    return credit(actor, 0, subject)


def deaths(events, ids):
    """[(seat, event)] for the death events among `events` whose subject is one of `ids` (seat order)."""
    return [(ids.index(e.subject), e) for e in events if e.kind == DEATH and e.subject in ids]


def credited_kills(events, ids):
    """Per seat, the deaths of the other seats credited to it (`credit`), from one step's events."""
    out = [0] * len(ids)
    for seat, e in deaths(events, ids):
        who = e.actor or e.creator
        if credit(e.actor, e.creator, e.subject) == 'opponent' and who in ids:
            out[ids.index(who)] += 1
    return out
