"""Continuous real-time tracking with script-triggered recordings.

Kinefly2 stays on for hours: one pipeline thread per camera keeps tracking,
the merged fly state is published over ZMQ, the GUI shows live overlays. A
**recording** (HDF5 fly state + optional raw video + events) is a session
that experiment scripts start and stop over the command socket -- see
kinefly2.client -- or the GUI `record` button, or the CLI (apps/ctl.py):

    python -m kinefly2.apps.run_rig config/rig.yaml                      # cameras on, idle
    python -m kinefly2.apps.ctl start --fly 017 --protocol YAMV          # from any terminal/script
    python -m kinefly2.apps.ctl stop

The experiment template (`--exp`, default experiments/default.yaml if it
exists) supplies data_root, genotype, video/quality/event settings; each
recording overrides fields (fly id, protocol, notes, video on/off).

    python -m kinefly2.apps.run_rig config/rig.yaml --cameras bottom     # front camera off
    python -m kinefly2.apps.run_rig config/rig.yaml --record-now --fly 017 --protocol YAMV
                                                                          # one-shot: record from start to exit
"""
import argparse
import os
import sys
import time

import cv2
import yaml

from ..camera.base import make_source
from ..core.experiment import Experiment
from ..core.merger import StateMerger
from ..core.pipeline import CameraPipeline
from ..core.session import RecordingSession
from ..gui.app import TrackerView
from ..output.command_server import CommandServer
from ..output.event_receiver import EventReceiver
from ..output.zmq_pub import ZmqPublisher

DEFAULT_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'experiments', 'default.yaml')


def parse_cameras(rig, rig_dir, only=None):
    """rig.yaml camera entries: name -> path, or name -> {config, enabled}."""
    cameras = {}
    for name, entry in rig['cameras'].items():
        if isinstance(entry, dict):
            enabled = entry.get('enabled', True)
            path = entry['config']
        else:
            enabled, path = True, entry
        if only is not None:
            enabled = name in only
        if enabled:
            cameras[name] = os.path.join(rig_dir, path)
    if only:
        missing = set(only) - set(cameras)
        if missing:
            raise SystemExit('Unknown camera(s) %s; rig.yaml defines %s'
                             % (sorted(missing), sorted(rig['cameras'])))
    return cameras


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rig_config', help='rig.yaml naming the cameras and ports')
    ap.add_argument('--cameras', nargs='+', metavar='NAME',
                    help='Use only these cameras (e.g. --cameras bottom)')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--no-commands', action='store_true', help='Disable the command socket')
    g = ap.add_argument_group('experiment template (defaults for recordings)')
    g.add_argument('--exp', metavar='YAML', help='Experiment template (default: experiments/default.yaml)')
    g.add_argument('--data-root', dest='data_root')
    g.add_argument('--genotype')
    g.add_argument('--events-port', dest='events_port', type=int)
    g.add_argument('--video', dest='record_video', action='store_true', default=None)
    g.add_argument('--no-video', dest='record_video', action='store_false')
    r = ap.add_argument_group('one-shot recording (record from start until exit)')
    r.add_argument('--record-now', action='store_true')
    r.add_argument('--fly', dest='fly_id')
    r.add_argument('--protocol')
    r.add_argument('--notes')
    args = ap.parse_args()

    with open(args.rig_config) as f:
        rig = yaml.safe_load(f)
    rig_dir = os.path.dirname(os.path.abspath(args.rig_config))
    cameras = parse_cameras(rig, rig_dir, args.cameras)
    if not cameras:
        raise SystemExit('No cameras enabled.')
    primary = rig.get('primary', 'bottom')
    if primary not in cameras:
        primary = next(iter(cameras))
        print('Primary camera not enabled; using %s as primary.' % primary)
    use_gui = not args.no_gui and rig.get('use_gui', True)

    # Experiment template: rig-wide defaults for recordings.
    exp_path = args.exp or (DEFAULT_TEMPLATE if os.path.exists(DEFAULT_TEMPLATE) else None)
    template = Experiment.load(exp_path, {'data_root': args.data_root, 'genotype': args.genotype,
                                          'events_port': args.events_port,
                                          'record_video': args.record_video})

    merger = StateMerger(primary=primary)
    if rig.get('zmq_port'):
        merger.add_sink(ZmqPublisher(port=int(rig['zmq_port'])))

    views, pipelines = {}, {}
    for name, cfg_path in cameras.items():
        view = TrackerView(name, cfg_path, use_gui=use_gui)
        with open(cfg_path) as f:
            cam_cfg = (yaml.safe_load(f) or {}).get('camera') or {}
        source = make_source(cam_cfg)
        source.open()
        views[name] = view
        pipelines[name] = CameraPipeline(view, source, merger, render=use_gui)

    events = EventReceiver(template.events_port) if template.events_port else None
    session = RecordingSession(template, merger, pipelines, views=views, events=events,
                               config_paths=[args.rig_config, exp_path] + list(cameras.values()))
    commands = None
    if not args.no_commands and rig.get('command_port', 5558):
        commands = CommandServer(session.handle_command, port=int(rig.get('command_port', 5558)))

    def toggle_record():
        try:
            if session.recording:
                session.stop()
            else:
                session.start()
        except Exception as e:
            print('record: %s' % e)
    def log_zeroed(name):
        def cb(summary):
            if events is not None:
                events.log('{"event": "zeroed", "camera": "%s", "detail": "%s"}' % (name, summary))
        return cb
    for name, view in views.items():
        view.on_record = toggle_record
        view.on_zeroed = log_zeroed(name)

    print('Kinefly2 up: cameras %s, primary %s' % (', '.join(cameras), primary))
    if rig.get('zmq_port'):
        print('  fly state: ZMQ PUB tcp://*:%s' % rig['zmq_port'])
    if commands is not None:
        print('  commands:  ZMQ REP tcp://*:%d  (kinefly2.client / apps/ctl.py)' % commands.port)
    if events is not None:
        print('  events:    ZMQ PULL tcp://*:%d' % events.port)
    print('  recordings -> %s  (template %s)' % (os.path.expanduser(template.data_root), exp_path))

    for p in pipelines.values():
        p.start()
    if args.record_now:
        session.start(fly_id=args.fly_id, protocol=args.protocol, notes=args.notes)

    try:
        while True:
            if any(p.eof or p.error for p in pipelines.values()):
                break
            if any(v.bQuit for v in views.values()):
                break
            rec = session.status().get('session') if session.recording else None
            for v in views.values():
                v.rec_status = rec
            if use_gui:
                for p in pipelines.values():
                    img = p.latest_display()
                    if img is not None:
                        cv2.imshow(p.view.window_name, img)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.5)
                line = '  '.join('%s: %d' % (n, p.n_frames) for n, p in pipelines.items())
                if rec:
                    line += '   REC %s %.0fs' % (rec['name'], rec['duration'])
                print(line, end='\r')
    except KeyboardInterrupt:
        pass
    finally:
        print()
        try:
            session.close()             # Stops an active recording cleanly.
        except Exception as e:
            print('stop recording: %s' % e)
        if commands is not None:
            commands.close()
        if events is not None:
            events.close()
        for p in pipelines.values():
            p.stop()
        for p in pipelines.values():
            p.source.close()
        merger.close()
        cv2.destroyAllWindows()

    for n, p in pipelines.items():
        print('  %s: %d frames tracked, %d reconnects' % (n, p.n_frames, p.n_reconnects))
    if session.last_summary:
        s = session.last_summary
        print('  last recording: %s  %.1f s, %d rows, %d events -> %s' % (
            s['name'], s['duration'], s['rows'], s['events'], s['output_dir']))


if __name__ == '__main__':
    sys.exit(main())
