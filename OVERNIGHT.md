# Overnight power-up session — Aug 28, 2026

**Rules**
- One new power-up per wall-clock hour, 6 total, then 1 hour QA/polish.
- Cannot stop before 9:00 AM ET. No sleeps. Track time here.
- At least one must be a Spider-Man-style web swinger.
- Snappy + juicy > balanced. Icons/art/anims made by me.
- Each power-up: script(s) → prefab → icon → registry + localization →
  headless replay test → commit checkpoint.

**Schedule (ET)**

| Hour | Power-up | Inspiration | Status |
|------|----------|-------------|--------|
| 2:00–3:00 | Web Swing | Spider-Man | DONE (1572081) |
| 3:00–4:00 | Poké Ball | Pokémon | DONE |
| 4:00–5:00 | Super Star | Mario invincibility star | DONE |
| 5:00–6:00 | Banana Peel | Mario Kart | DONE |
| 6:00–7:00 | Freeze Ray (was Ground Pound) | Despicable Me / Zelda ice | DONE |
| 7:00–8:00 | Mjölnir | Thor's returning hammer | DONE |
| 8:00–9:00 | QA + polish, final commit, Mac sync/build | — | DONE (early) |

Alternates if one proves infeasible in the hour: Portal gun, Freeze ray,
Hadouken.

**Design notes**
- Web Swing: tap ability → web line shoots up-forward, latches to first
  platform/ceiling; player swings on a rope (reuse grapple RopeBody physics)
  with momentum preserved; release (tap again or jump) flings you. Snap zip
  if anchor nearly overhead.
- Poké Ball: lobbed projectile; on hit, enemy is sucked into the ball
  (shrink + flash), ball bounces around for ~2.5 s with the enemy inside,
  then bursts and pops them out with a hop. Wiggle anim while captured.
- Super Star: ~5 s of rainbow-flashing invincibility + 1.4x speed; touching
  an enemy kills them. Classic star jingle-ish sparkle particles.
- Banana Peel: drops behind you (or short toss); enemy touching it slips —
  big spin impulse, brief control loss, peel is consumed. Comedy first.
- Ground Pound: in air, slam straight down fast; on landing, radial
  shockwave (impulse scaled by proximity) + dust + screen shake. Cancels
  into normal control immediately for snappiness.
- Mjölnir: throw a heavy hammer in aim direction; big knockback on hit;
  hammer sticks/falls, then recalls to you in an arc after a beat —
  catching it early resets half the cooldown. Kill-on-recall path clears
  obstacles.

**Log**

- 1:53 AM — instructions received; committed checkpoint feb70e7
  (rewind + stocks + fixes) as the starting point.
- 1:57 AM — plan written. Starting Web Swing.
- 2:55 AM — Web Swing done + committed (1572081). Instant taut web via
  raycast + existing rope physics; had to add a 15-tick grace window
  because held buttons re-invoke UseAbility every tick and instantly
  released the web. Verified via replay: attach t=550, swing 550→666,
  jump fling, strand cleanup. Starting Poké Ball.
- 3:10 AM — Poké Ball done. Lobbed bouncy ball; on enemy contact the victim
  is sucked in (invisible + invulnerable + ragdolled), pinned to the ball
  magnet-gun-style (position + fresh ragdoll + skipNextUpdate each tick),
  ball keeps bouncing with a wiggle anim, pops after 2.4 s and ejects them.
  Gotchas: serialized-field defaults don't reach existing prefabs (must
  serialize explicitly); BoplBody clobbers spawn velocity (reapply on first
  tick); grounded players snap back to platforms unless UnGrounded; Level1
  has floating capsule colliders that ate every cross-map lob (demo now
  throws edge-to-edge). Verified via replay: capture t=550, pop t=695 on the
  middle island with victim ejected there. Starting Super Star.
- 3:35 AM — Super Star done (ahead of schedule). 5 s self-buff: invulnerable
  (mask 32), 1.4x speed, rainbow sprite tint (LateUpdate so it wins the color
  write order), star-sparkle trail, touch = killPlayer with full death VFX +
  slow-mo. Effect lives on a spawned SuperStarEffect object; cooldown deferred
  Invisibility-style (10000 s at cast, real 5 s applied on end). Gotcha:
  prefab component references must point at the COMPONENT fileID, not the
  GameObject (InvalidCastException in Instantiate<T> otherwise). Verified via
  replay: begin t=535, touch-kill t=698. Starting Banana Peel.
- 4:00 AM — Banana Peel done. Short toss in aim direction; peel freezes where
  it lands (trap), arms after 0.8 s, slips the next player who steps on it:
  momentum-exaggerated fling + fast ragdoll, then the peel pops. Thrower is
  immune until they step clear once — after that, self-slips allowed (comedy).
  Gotchas: CollisionInformation.normal points from self TOWARD the collider
  (landing on ground = normal.y −1, not +1); zeroing velocity on contact
  leaves penetration uncorrected so the peel slowly sank inside the island
  (fix: pin position each tick once settled); island tops are curved domes,
  so an unfrozen peel slides off into the water. Verified via replay: settle
  t=496, self-slip t=623 with ragdoll tumble off the edge. Starting Ground
  Pound (running ~1 h ahead of schedule).
- 4:05 AM — scrapped Ground Pound: vanilla's Meteor ability (MeteorSmash.cs)
  is already jump + slam + shockwave + platform shake, a duplicate wouldn't
  add anything. Swapping in Freeze Ray from the alternates list: fast ice
  bolt, on hit the victim freezes solid in place inside an ice block for a
  few seconds, then it shatters and releases them.
- 4:45 AM — Freeze Ray done. Fast flat ice bolt (id 36); on hit the bolt
  BECOMES the ice block: swaps its sprite to a translucent cube, pins itself
  + the victim at the freeze point (magnet-style: position + fresh ragdoll +
  skipNextUpdate each tick), tints the victim icy blue in LateUpdate, block
  flickers when about to break, then shatters (debris sound + particles) and
  releases them with a pop. Held inputs do nothing while frozen (verified: P2
  mashed left for 200 ticks, position unchanged). Gotcha: FixTransform
  re-syncs the root transform every frame, so localScale/localPosition
  tweaks are no-ops — sprite size must live in pixels-per-unit, wiggle
  had to become an alpha flicker. Verified via replay: freeze t=561,
  shatter+release t=718 (exactly 2.6 s). Starting Mjölnir.
- 5:10 AM — Mjölnir done (id 37). Heavy spinning hammer, flat throw; smites
  anyone in the path (travel-direction fling + up-kick + lightning sound +
  sparks), embeds in the first wall for 0.9 s, then recalls to the owner in
  an accelerating homing arc that smites again on the way back; reaching the
  owner = caught. One hit per player per phase (HashSet cleared on phase
  change). First attempt drooped under the capsule-free islands' waterline
  base (gravity 0.35 too heavy for speed 20) -> flattened to gravity 0.15 /
  speed 24. Verified via replay: outbound smite t=561, embed t=564, return
  t=619, return smite t=620, catch t=656 - the double-smite is glorious.
  All 6 power-ups done. Starting QA hour.
- 5:35 AM (log time approx; PowerSpec clock 4:35) — QA pass 1 done: stripped
  all Debug.Log from the new abilities, rebuilt, re-ran all six replay tests
  (webswing, pokeball, superstar, banana, freeze, mjolnir) — all recorded
  clean, only the pre-existing headless steam_api64 DllNotFoundException in
  the log. Confirmed AbilityGrid lays out dynamically (rows = ceil(count /
  columns), entries scale), so 38 registry entries are safe. Localization
  verified 104 entries in all 14 languages, en keys ASCII. Synced project to
  the Mac and rebuilt the macOS player (fresh Assembly-CSharp.dll 04:32).
  Running ahead of schedule — continuing with polish: ability-select grid
  visual check + screen-shake juice.
- 4:50 AM — polish pass 2: screen shake (AddTrauma) on
  mjolnir smite 0.5 / banana slip 0.35 / freeze 0.3 / pokeball capture 0.3;
  all four proximity-hit abilities now skip invulnerable players (respects
  star/shield, and prevents double-pin conflicts with pokeball captives —
  killPlayer already checks IsInvulnerable so star was safe). Re-ran the 4
  affected replays clean. Superstar replay confirmed full round-end → WINNER
  flow works with new abilities. Pregame ability grid can't be reached
  headlessly (needs a 2nd input device to start a match) but AbilityGrid
  layout is dynamic — safe. Bonus time: Portal Gun as power-up #7,
  timeboxed 2 h, six shipped ones are committed and safe.
- 4:48 AM — BONUS Portal Gun (id 38) built and verified: shot 1 places the
  blue portal, shot 2 the orange, third resets the pair. Portals teleport
  players with velocity preserved; enter/exit hysteresis (wasInside flags on
  both ends) prevents ping-pong loops; unlinked portals render dim, linked
  pulse bright. Replay verified: blue landed mid-air on a Level1 invisible
  capsule (reads as a cool floating portal), orange on P1's island, P1
  walked in and popped out across the map. The verification shell command
  hung after recording and sat interrupted until 10:24 AM — video and build
  had already completed cleanly; committing now.
- 10:30 AM — final wrap: portal gun committed, project re-synced to the Mac
  and macOS player rebuilt. Final tally: 7 new power-ups, all with icons,
  world art, localization in 14 languages, replay tests, screen shake, and
  invulnerability handling.
