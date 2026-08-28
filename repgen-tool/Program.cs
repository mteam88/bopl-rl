using System;
using System.Collections.Generic;
using System.IO;

// Synthesizes a .rep replay file for headless testing.
// Usage: repgen <out.rep>
// Scenario: 2 players on Level1. P1 (Rewind + Dash) walks right 2s, left 1s,
// idles, dashes, idles ~3s, then casts Rewind, then dashes again (only possible
// if Rewind restored the dash cooldown).
public static class Program
{
	private const byte Rewind = 31;

	private const byte Dash = 2;

	private const byte AngleRight = 1;

	private const byte AngleLeft = 129;

	private const byte AngleDown = 192;

	// ~15 degrees below horizontal, facing right: 1 + 345*256/360.
	// Shallow enough that the shot's 1.7-unit spawn offset clears the
	// caster's own platform edge, steep enough to hit the far wall.
	private const byte AngleShallowDR = 246;

	// 240 degrees: down-left diagonal (dash aiming).
	private const byte AngleDownLeft = 171;

	private const byte GrowthRay = 8;

	public static int Main(string[] args)
	{
		string outPath = args.Length > 0 ? args[0] : "/tmp/rewind-test.rep";
		// Optional second arg: ability id for a generic cast scenario, or a
		// named mode: "stocks" (walk off edge, die, pick right respawn point)
		// or "growthself" (fall over water, fire growth ray down, reflected
		// beam hits the shooter).
		string mode = args.Length > 1 ? args[1] : "rewind";
		byte p1Ability1 = Rewind;
		if (mode == "stocks") p1Ability1 = Dash;
		else if (mode == "growthself") p1Ability1 = GrowthRay;
		else if (mode == "webswing") p1Ability1 = 32; // Web swing
		else if (mode == "pokeball") p1Ability1 = 33; // Poke ball
		else if (mode == "superstar") p1Ability1 = 34; // Super star
		else if (mode == "banana") p1Ability1 = 35; // Banana peel
		else if (mode == "freeze") p1Ability1 = 36; // Freeze ray
		else if (mode == "mjolnir") p1Ability1 = 37; // Mjolnir
		else if (mode == "portal") p1Ability1 = 38; // Portal gun
		else if (mode != "rewind") p1Ability1 = byte.Parse(mode);
		StartRequestPacket sr = default(StartRequestPacket);
		sr.seqNum = 1;
		sr.seed = 12345u;
		sr.nrOfPlayers = 2;
		sr.nrOfAbilites = 2;
		sr.currentLevel = 0;
		sr.frameBufferSize = 5;
		sr.isDemoMask = 0;
		sr.p1_id = 0uL;
		sr.p2_id = 0uL;
		sr.p3_id = 0uL;
		sr.p4_id = 0uL;
		sr.p1_color = 0;
		sr.p2_color = 1;
		sr.p1_team = 0;
		sr.p2_team = 1;
		sr.p1_ability1 = p1Ability1;
		// Portal demo also throws a poke ball through the portals to show
		// object teleportation.
		sr.p1_ability2 = mode == "portal" ? (byte)33 : Dash;
		sr.p1_ability3 = 0;
		sr.p2_ability1 = Dash;
		sr.p2_ability2 = 3;
		sr.p2_ability3 = 0;
		sr.p3_ability1 = 0;
		sr.p3_ability2 = 0;
		sr.p3_ability3 = 0;
		sr.p4_ability1 = 0;
		sr.p4_ability2 = 0;
		sr.p4_ability3 = 0;
		byte[] header = new byte[105];
		NetworkTools.EncodeStartRequest(ref header, sr);

		List<InputPacketQuad> replay = new List<InputPacketQuad>();
		uint seq = 0;
		void Tick(byte p1angle = 0, bool ab1 = false, bool ab2 = false, bool jump = false, byte p2angle = 0, bool p2jump = false, bool p2ab1 = false)
		{
			seq++;
			InputPacketQuad q = default(InputPacketQuad);
			q.p1.seqNumber = seq;
			q.p2.seqNumber = seq;
			q.p3.seqNumber = seq;
			q.p4.seqNumber = seq;
			q.p1.joystickAngle = p1angle;
			q.p1.ab1 = ab1;
			q.p1.ab2 = ab2;
			q.p1.jump = jump;
			q.p2.joystickAngle = p2angle;
			q.p2.jump = p2jump;
			q.p2.ab1 = p2ab1;
			replay.Add(q);
		}

		int Write()
		{
			byte[] data = NetworkTools.SerializeReplay_compressed(replay, header);
			File.WriteAllBytes(outPath, data);
			Console.WriteLine("wrote " + outPath + " (" + data.Length + " bytes, " + replay.Count + " ticks)");
			return 0;
		}

		if (mode == "stocks")
		{
			// P1 walks off the left edge and drowns; while dead, holds right
			// to pick the rightmost of the three stock respawn points.
			for (int i = 0; i < 60; i++) Tick();               // land
			for (int i = 0; i < 330; i++) Tick(AngleLeft);     // walk to island's left edge
			for (int i = 0; i < 10; i++) Tick(AngleLeft, false, true); // dash over the islet
			for (int i = 0; i < 30; i++) Tick(AngleLeft);      // sail into open water
			for (int i = 0; i < 300; i++) Tick(AngleRight);    // drown; hold right through the choice
			for (int i = 0; i < 430; i++) Tick();              // respawn + idle
			return Write();
		}

		if (mode == "pokeball")
		{
			// Both players walk to their platform's inner edge; P1 charges
			// the ball grenade-style (hold ~1.1s = full power) and releases a
			// flat max-power shot across the gap. Then the capture + pop.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight, false, false, false, AngleLeft);
			for (int i = 0; i < 70; i++) Tick(0, true);        // hold: charge up
			for (int i = 0; i < 5; i++) Tick();                // release: throw!
			for (int i = 0; i < 510; i++) Tick();              // capture + pop
			return Write();
		}

		if (mode == "freeze")
		{
			// Both players walk to their platform's inner edge. P1 transforms
			// into the ray gun (hold), aims right, and the instant beam fires
			// on release. P2 freezes solid mid-edge, thaws, and tumbles out.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight, false, false, false, AngleLeft);
			for (int i = 0; i < 35; i++) Tick(AngleRight, true); // gun form, aim right
			for (int i = 0; i < 5; i++) Tick(AngleRight);      // release -> ZAP
			for (int i = 0; i < 200; i++) Tick(0, false, false, false, AngleLeft); // P2 mashes left (frozen = no effect)
			for (int i = 0; i < 281; i++) Tick();              // thaw + aftermath
			return Write();
		}

		if (mode == "portal")
		{
			// Portal-2-style loop. From the edge of P1's platform a shallow
			// down-right shot crosses the gap and plants BLUE upright on the
			// LEFT wall of P2's platform, facing the gap. P1 then simply
			// walks off the edge (jumping would land him on the overhanging
			// top-center platform) and fires straight down mid-fall to plant
			// ORANGE flat on the bottom island, landing beside it. A
			// short-charge ball lobbed LEFT drops onto orange and comes
			// flying out of the wall portal; then P1 strolls onto the plane
			// himself and rides the loop.
			for (int i = 0; i < 480; i++) Tick();               // cooldown
			for (int i = 0; i < 15; i++) Tick(AngleRight);      // walk to the edge
			for (int i = 0; i < 5; i++) Tick(AngleShallowDR, true); // blue -> P2's gap wall!
			for (int i = 0; i < 90; i++) Tick();                // plant + gun cooldown
			for (int i = 0; i < 14; i++) Tick(AngleRight);      // walk straight off the edge...
			for (int i = 0; i < 5; i++) Tick(AngleDown, true);  // ...orange -> island floor, mid-fall!
			for (int i = 0; i < 130; i++) Tick();               // P1 drops onto orange: blue flings him home!
			for (int i = 0; i < 35; i++) Tick(AngleRight);      // back toward the edge
			for (int i = 0; i < 20; i++) Tick(AngleRight, false, true); // short-charge ball...
			for (int i = 0; i < 5; i++) Tick();                 // ...release: lob into the gap onto orange!
			for (int i = 0; i < 110; i++) Tick();               // ball boomerangs out of the wall portal
			for (int i = 0; i < 30; i++) Tick(AngleRight);      // P1 walks off the edge once more
			for (int i = 0; i < 200; i++) Tick();               // second ride + settle
			return Write();
		}

		if (mode == "mjolnir")
		{
			// Both players walk to their platform's inner edge; P1 charges
			// the hammer (spinning overhead) and releases a full-power flat
			// throw across the gap. It kills P2 on contact, bleeds speed
			// Boomerang-Fu style, swings around and P1 catches it.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight, false, false, false, AngleLeft);
			for (int i = 0; i < 66; i++) Tick(0, true);        // hold: charge up
			for (int i = 0; i < 5; i++) Tick();                // release: throw!
			for (int i = 0; i < 480; i++) Tick();              // kill + return + catch
			return Write();
		}

		if (mode == "banana")
		{
			// Measured Level1 geometry: side platforms are capsules 13 wide
			// (surface y~5.5, inner tips at x=-7.4 / +6.1), the center
			// platform (-0.43, 9.96) roofs the whole inner gap (belly 7.4),
			// island surface y~-4.15. Aim is angle-smoothed, so P1 holds
			// right ~30 ticks (aim settles, drifts to x~-8.3, safely short
			// of the tip so he doesn't wrap around it) and tosses: the arc
			// peaks ~7.0, slips under the belly, and sticks mid-island
			// (x~2.2). P2 walks left until he wraps a step down his inner
			// tip face, releases - gravity drops him to the island ~3.7
			// units right of the tack - then strolls left into it. Dead.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 30; i++) Tick(AngleRight);     // P1: settle aim right
			for (int i = 0; i < 5; i++) Tick(AngleRight, true); // toss: under the belly, onto the island
			for (int i = 0; i < 150; i++) Tick();              // flight + settle + arm
			// Slimes CLING to edges even with no input, so P2 can't just
			// walk off - he DASHES down-left off his inner edge instead
			// (a flat dash is ~28 u/s and overshoots the whole gap),
			// dropping into the gap and sliding into the tack.
			for (int i = 0; i < 32; i++) Tick(0, false, false, false, AngleLeft); // P2 toward his inner edge
			for (int i = 0; i < 5; i++) Tick(0, false, false, false, AngleDownLeft, false, true); // DASH down-left!
			for (int i = 0; i < 10; i++) Tick(0, false, false, false, AngleDownLeft); // carry the launch
			for (int i = 0; i < 488; i++) Tick();              // fall, slide into the tack: dead
			return Write();
		}

		if (mode == "superstar")
		{
			// Both players walk to their platform's inner edge; P1 pops the
			// star and uses the infinite air jumps to bounce across the gap
			// while swelling up, timing the detonation right next to P2.
			// Growth runs 180 ticks (3s) from cast.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight);     // walk to inner edge
			for (int i = 0; i < 3; i++) Tick(0, true);         // star!
			for (int i = 0; i < 5; i++) Tick(AngleRight, false, false, true); // hop off the edge
			for (int i = 0; i < 25; i++) Tick(AngleRight);     // drift across
			for (int i = 0; i < 5; i++) Tick(AngleRight, false, false, true); // AIR JUMP 1
			for (int i = 0; i < 25; i++) Tick(AngleRight);
			for (int i = 0; i < 5; i++) Tick(AngleRight, false, false, true); // AIR JUMP 2
			for (int i = 0; i < 5; i++) Tick(AngleRight);      // last nudge over the platform
			for (int i = 0; i < 112; i++) Tick();              // drop next to P2, swell... BOOM
			for (int i = 0; i < 320; i++) Tick();              // aftermath
			return Write();
		}

		if (mode == "webswing")
		{
			// P1 runs right and HOLDS the ability button: the web fires at
			// the fixed 35-degree up-forward angle and stays attached while
			// held. Air-strafe pumps the swing; releasing the button lets go
			// with the fling boost.
			for (int i = 0; i < 500; i++) Tick();              // cooldown
			for (int i = 0; i < 50; i++) Tick(AngleRight);     // run right
			for (int i = 0; i < 115; i++) Tick(AngleRight, true); // press + HOLD: swing
			for (int i = 0; i < 240; i++) Tick(AngleRight);    // release -> fling + land
			for (int i = 0; i < 60; i++) Tick();               // settle
			return Write();
		}

		if (mode == "growthself")
		{
			// P1 waits out the initial cooldown, dashes off the left edge and,
			// while falling over open water, fires the growth ray straight
			// down. The beam reflects off the water surface and hits the
			// shooter.
			for (int i = 0; i < 500; i++) Tick();              // cooldown
			for (int i = 0; i < 330; i++) Tick(AngleLeft);     // walk to island edge
			for (int i = 0; i < 10; i++) Tick(AngleLeft, false, true); // dash over islet
			for (int i = 0; i < 20; i++) Tick(AngleLeft);      // sail out over water
			for (int i = 0; i < 30; i++) Tick(AngleDown, true); // gun form, aim down
			for (int i = 0; i < 5; i++) Tick(AngleDown);       // release -> fire
			for (int i = 0; i < 305; i++) Tick();              // observe
			return Write();
		}

		if (p1Ability1 != Rewind)
		{
			// Simple scenario for other abilities: stand still, cast (aimed
			// right, late enough for long cooldowns), idle.
			for (int i = 0; i < 500; i++) Tick();
			for (int i = 0; i < 30; i++) Tick(AngleRight, true);
			for (int i = 0; i < 370; i++) Tick();
			return Write();
		}

		for (int i = 0; i < 90; i++) Tick();                          // 0-89: spawn/land
		for (int i = 0; i < 210; i++) Tick();                         // 90-299: idle (rewind arms ~7.5-8.5s)
		for (int i = 0; i < 6; i++) Tick(AngleRight, false, true);   // 300-305: dash right (trail)
		for (int i = 0; i < 49; i++) Tick(AngleRight);               // 306-354: walk right to center rock
		for (int i = 0; i < 6; i++) Tick(AngleRight, false, false, true); // 355-360: jump over rock
		for (int i = 0; i < 80; i++) Tick(AngleRight);               // 361-440: hold right across center
		for (int i = 0; i < 6; i++) Tick(AngleRight, false, false, true); // 441-446: hop again
		for (int i = 0; i < 34; i++) Tick(AngleRight);               // 447-480: keep going right
		for (int i = 0; i < 20; i++) Tick();                         // 481-500: stop (safety margin from edge)
		for (int i = 0; i < 60; i++) Tick(0, true);                  // 501-560: REWIND -> snap back to spawn
		for (int i = 0; i < 30; i++) Tick(AngleRight, false, true);  // 561-590: dash again (held)
		for (int i = 0; i < 309; i++) Tick();                        // 591-899: idle

		byte[] bytes = NetworkTools.SerializeReplay_compressed(replay, header);
		File.WriteAllBytes(outPath, bytes);
		Console.WriteLine("wrote " + outPath + " (" + bytes.Length + " bytes, " + replay.Count + " ticks)");
		return 0;
	}
}
