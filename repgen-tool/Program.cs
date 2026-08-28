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
		sr.p1_ability2 = Dash;
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
		void Tick(byte p1angle = 0, bool ab1 = false, bool ab2 = false, bool jump = false, byte p2angle = 0)
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
			// P2 walks left toward the middle while P1 lobs a poke ball at
			// them; then everyone waits out the capture + pop.
			// Level1: both players walk to their platform's inner edge
			// (wrapping around it, below the floating capsule), then P1
			// fires a flat shot across the gap.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight, false, false, false, AngleLeft);
			for (int i = 0; i < 5; i++) Tick(AngleRight, true); // throw!
			for (int i = 0; i < 580; i++) Tick();              // capture + pop
			return Write();
		}

		if (mode == "freeze")
		{
			// Like the pokeball demo: both players walk to their platform's
			// inner edge, P1 fires a flat bolt across the gap. P2 freezes
			// solid mid-edge, thaws, and tumbles back out.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight, false, false, false, AngleLeft);
			for (int i = 0; i < 5; i++) Tick(AngleRight, true); // fire!
			for (int i = 0; i < 20; i++) Tick();               // bolt flight
			for (int i = 0; i < 200; i++) Tick(0, false, false, false, AngleLeft); // P2 mashes left (frozen = no effect)
			for (int i = 0; i < 296; i++) Tick();              // thaw + aftermath
			return Write();
		}

		if (mode == "portal")
		{
			// P1 walks to the inner edge, fires the blue portal up-right
			// (lands on/above the centre island), then the orange portal
			// flat-left onto their own island, walks left into the orange
			// and pops out across the map.
			const byte AngleUpRight = 33;
			for (int i = 0; i < 480; i++) Tick();               // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight);      // walk to edge
			for (int i = 0; i < 5; i++) Tick(AngleUpRight, true); // blue!
			for (int i = 0; i < 95; i++) Tick();                // flight + place
			for (int i = 0; i < 5; i++) Tick(AngleLeft, true);  // orange!
			for (int i = 0; i < 75; i++) Tick();                // place + link
			for (int i = 0; i < 90; i++) Tick(AngleLeft);       // walk into orange
			for (int i = 0; i < 283; i++) Tick();               // pop out + settle
			return Write();
		}

		if (mode == "mjolnir")
		{
			// Both players walk to their platform's inner edge; P1 hurls the
			// hammer flat across the gap. It smites P2, embeds in their
			// island, then boomerangs home and P1 catches it.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight, false, false, false, AngleLeft);
			for (int i = 0; i < 5; i++) Tick(AngleRight, true); // throw!
			for (int i = 0; i < 548; i++) Tick();              // smite + return + catch
			return Write();
		}

		if (mode == "banana")
		{
			// P1 tosses a peel ahead on their own platform, steps away to
			// clear the thrower immunity, then walks back into it and eats
			// the slip. Self-inflicted slapstick.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 5; i++) Tick(AngleRight, true); // toss!
			for (int i = 0; i < 60; i++) Tick();               // settle + arm
			for (int i = 0; i < 30; i++) Tick(AngleLeft);      // step clear (immunity)
			for (int i = 0; i < 15; i++) Tick();               // beat
			for (int i = 0; i < 70; i++) Tick(AngleRight);     // walk back into it
			for (int i = 0; i < 384; i++) Tick();              // tumble + aftermath
			return Write();
		}

		if (mode == "superstar")
		{
			// Both players walk to their platform's inner edge; P1 pops the
			// star, jumps off the inner face and air-strafes across the gap
			// at star speed; the touch kills P2.
			for (int i = 0; i < 480; i++) Tick();              // cooldown
			for (int i = 0; i < 55; i++) Tick(AngleRight);     // walk to inner edge
			for (int i = 0; i < 30; i++) Tick(AngleRight, true); // star!
			for (int i = 0; i < 4; i++) Tick(AngleRight, false, false, true); // launch off the wall face
			for (int i = 0; i < 56; i++) Tick(AngleRight);     // fly across onto the capsule
			for (int i = 0; i < 6; i++) Tick(AngleRight, false, false, true); // fling off it
			for (int i = 0; i < 6; i++) Tick(AngleRight);      // brief strafe...
			for (int i = 0; i < 55; i++) Tick();               // ...drop onto P2's platform
			for (int i = 0; i < 45; i++) Tick(AngleLeft);      // walk into P2
			for (int i = 0; i < 264; i++) Tick();              // aftermath

			return Write();
		}

		if (mode == "webswing")
		{
			// P1 walks right, webs up-forward, pumps the swing, then releases
			// with a jump fling.
			const byte AngleUpRight = 33; // ~45 degrees
			for (int i = 0; i < 500; i++) Tick();              // cooldown
			for (int i = 0; i < 50; i++) Tick(AngleRight);     // run right
			for (int i = 0; i < 5; i++) Tick(AngleUpRight, true); // web!
			for (int i = 0; i < 110; i++) Tick(AngleRight);    // swing + pump
			for (int i = 0; i < 6; i++) Tick(AngleRight, false, false, true); // jump release
			for (int i = 0; i < 240; i++) Tick();              // fly + land
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
