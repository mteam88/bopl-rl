#!/usr/bin/env python3
"""Random 'human' for the live shadow test: drives virtual pads (vpad FIFOs)
and/or the keyboard of eval/shadow_live/session.sh with a jittery stream of
sticks, jumps and ability buttons, so the recorded human packets cover
movement, aiming, every ability slot and deaths.

  human.py --seconds 60 --pads 1,2 [--keyboard] [--seed 1] [--dir /tmp/shadow-live]

Never presses start/back/guide (pause) or Escape."""
import argparse
import math
import os
import random
import subprocess
import time

PAD_BUTTONS = ["a", "b", "x", "y", "lb", "rb"]
KEYS = ["space", "a", "d", "w", "s", "j", "k", "l", "q", "e", "Shift_L"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--pads", default="1")
    ap.add_argument("--keyboard", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dir", default=os.environ.get("SHADOW_LIVE_DIR", "/tmp/shadow-live"))
    ap.add_argument("--display", default=os.environ.get("SHADOW_LIVE_DISPLAY", ":97"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pads = [p for p in args.pads.split(",") if p]
    fifos = {p: open(os.path.join(args.dir, f"pad{p}.fifo"), "w", buffering=1) for p in pads}
    held = {p: set() for p in pads}
    keys_held = set()
    env = dict(os.environ, DISPLAY=args.display)
    end = time.time() + args.seconds
    while time.time() < end:
        for p, fifo in fifos.items():
            angle = rng.uniform(0, 2 * math.pi)
            mag = rng.choice([0, 1, 1, 1])
            fifo.write(f"lx {int(32000 * mag * math.cos(angle))}\nly {int(32000 * mag * math.sin(angle))}\n")
            for b in PAD_BUTTONS:
                want = rng.random() < (0.35 if b == "a" else 0.12)
                if want != (b in held[p]):
                    fifo.write(f"{b} {int(want)}\n")
                    (held[p].add if want else held[p].discard)(b)
            if rng.random() < 0.1:
                fifo.write(f"rt {rng.choice([0, 255])}\n")
        if args.keyboard:
            for k in KEYS:
                want = rng.random() < 0.2
                if want != (k in keys_held):
                    subprocess.run(["xdotool", "keydown" if want else "keyup", k], env=env, check=False)
                    (keys_held.add if want else keys_held.discard)(k)
            if rng.random() < 0.3:
                subprocess.run(["xdotool", "mousemove", str(rng.randint(100, 1180)), str(rng.randint(80, 640))], env=env, check=False)
            if rng.random() < 0.15:
                subprocess.run(["xdotool", "click", rng.choice(["1", "3"])], env=env, check=False)
        time.sleep(rng.uniform(0.05, 0.25))
    for p, fifo in fifos.items():
        fifo.write("lx 0\nly 0\nrt 0\n")
        for b in held[p]:
            fifo.write(f"{b} 0\n")
        fifo.close()
    for k in keys_held:
        subprocess.run(["xdotool", "keyup", k], env=env, check=False)


if __name__ == "__main__":
    main()
