"""Command-line control of a running Kinefly2 (run_rig).

    python -m kinefly2.apps.ctl status
    python -m kinefly2.apps.ctl start --fly 017 --protocol YAMV [--genotype G --notes "..." --no-video]
    python -m kinefly2.apps.ctl stop
    python -m kinefly2.apps.ctl event '{"trial": 1, "event": "stim_on"}'
    python -m kinefly2.apps.ctl snapshot
    python -m kinefly2.apps.ctl zero [--camera front]     # this posture = zero (landmarks re-captured)
    python -m kinefly2.apps.ctl --host <rig-ip> status
"""
import argparse
import json
import sys

from ..client import Kinefly2Client, Kinefly2Error


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=5558, help='command port (rig.yaml command_port)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    sub.add_parser('ping')
    sub.add_parser('stop')
    sub.add_parser('snapshot')
    z = sub.add_parser('zero')
    z.add_argument('--camera', default=None)
    s = sub.add_parser('start')
    s.add_argument('--fly', dest='fly_id')
    s.add_argument('--protocol')
    s.add_argument('--genotype')
    s.add_argument('--notes')
    s.add_argument('--video', dest='record_video', action='store_true', default=None)
    s.add_argument('--no-video', dest='record_video', action='store_false')
    e = sub.add_parser('event')
    e.add_argument('text', help='event text (JSON recommended)')
    args = ap.parse_args()

    k = Kinefly2Client(args.host, command_port=args.port)
    try:
        if args.cmd == 'start':
            r = k.start(fly_id=args.fly_id, protocol=args.protocol, genotype=args.genotype,
                        notes=args.notes, record_video=args.record_video)
        elif args.cmd == 'stop':
            r = k.stop()
        elif args.cmd == 'event':
            r = k.event(args.text)
        elif args.cmd == 'snapshot':
            r = k.snapshot()
        elif args.cmd == 'zero':
            r = k.zero(args.camera)
        elif args.cmd == 'ping':
            r = k.ping()
        else:
            r = k.status()
    except Kinefly2Error as e:
        print('error: %s' % e, file=sys.stderr)
        return 1
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
