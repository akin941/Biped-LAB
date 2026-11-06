"""WP4-14: send velocity commands to a running ``biped-runtime --command-udp``.

Usage:
  uv run python scripts/teleop.py --port 9870 --hold 0.15 0 0 --seconds 5
  uv run python scripts/teleop.py --port 9870 --keyboard   # w/s a/d q/e, space = stop, x = quit

Sends at ``--rate`` Hz. Stopping this script stops the robot: the runtime's gate
ramps the command to zero 0.5 s after the last packet.
"""

from __future__ import annotations

import argparse
import time

from bipedlab.command_channel import CommandSender, KeyTeleop


def hold(tx: CommandSender, cmd: list[float], seconds: float, rate: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        tx.send(*cmd)
        time.sleep(1.0 / rate)


def keyboard(tx: CommandSender, rate: float) -> None:
    import curses

    def loop(scr: curses.window) -> None:
        scr.nodelay(True)
        tele = KeyTeleop()
        while True:
            key = scr.getch()
            if key == ord("x"):
                return
            if key != -1:
                tele.press(chr(key))
            vx, vy, wz = tele.command()
            scr.erase()
            scr.addstr(0, 0, "w/s ileri-geri  a/d sol-sag  q/e donus  bosluk dur  x cik")
            scr.addstr(2, 0, f"vx {vx:+.3f} m/s   vy {vy:+.3f} m/s   wz {wz:+.3f} rad/s")
            tx.send(vx, vy, wz)
            time.sleep(1.0 / rate)

    curses.wrapper(loop)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--rate", type=float, default=20.0, help="packets per second")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hold", type=float, nargs=3, metavar=("VX", "VY", "WZ"))
    mode.add_argument("--keyboard", action="store_true")
    p.add_argument("--seconds", type=float, default=5.0, help="with --hold")
    args = p.parse_args()
    tx = CommandSender(args.host, args.port)
    try:
        if args.keyboard:
            keyboard(tx, args.rate)
        else:
            hold(tx, args.hold, args.seconds, args.rate)
    finally:
        tx.close()


if __name__ == "__main__":
    main()
