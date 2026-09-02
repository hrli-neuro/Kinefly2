"""Real-time multi-camera tracking with merged state output.

One pipeline thread per camera; the GUI (imshow/waitKey) stays in the main
thread. The merged fly state is published over ZMQ and logged to HDF5.
The primary camera (bottom view) drives the output loop; the future LED-panel
closed-loop driver attaches as another sink.

Per-experiment settings (recording name, data folder, video on/off, event
port) come from an experiment YAML plus command-line overrides:

    python -m kinefly2.apps.run_rig config/rig.yaml --exp experiments/default.yaml --fly 018
    python -m kinefly2.apps.run_rig config/rig.yaml --exp experiments/default.yaml --no-video
    python -m kinefly2.apps.run_rig config/rig.yaml --cameras bottom      # front camera off

Without --exp (or any experiment flag) the legacy behavior applies: an
auto-named flystate_<timestamp>.h5 in rig.yaml's log_dir, no video.
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
from ..gui.app import TrackerView
from ..output.event_receiver import EventReceiver
from ..output.logger import Hdf5Logger
from ..output.video_recorder import VideoRecorder
from ..output.zmq_pub import ZmqPublisher


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
    ap.add_argument('rig_config', help='rig.yaml naming the cameras and outputs')
    ap.add_argument('--cameras', nargs='+', metavar='NAME',
                    help='Use only these cameras (e.g. --cameras bottom)')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--no-log', action='store_true', help='Disable the HDF5 fly-state log')
    g = ap.add_argument_group('experiment (per-recording settings)')
    g.add_argument('--exp', metavar='YAML', help='Experiment YAML (see experiments/default.yaml)')
    g.add_argument('--fly', dest='fly_id', help='Fly id, e.g. 017')
    g.add_argument('--genotype')
    g.add_argument('--protocol', help='Stimulus / open-loop / closed-loop protocol name')
    g.add_argument('--notes')
    g.add_argument('--date', help='YYMMDD (default: today)')
    g.add_argument('--data-root', dest='data_root')
    g.add_argument('--video', dest='record_video', action='store_true', default=None,
                   help='Record raw MJPEG video per camera')
    g.add_argument('--no-video', dest='record_video', action='store_false',
                   help='Do not record video for this recording')
    g.add_argument('--events-port', dest='events_port', type=int)
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

    overrides = {k: getattr(args, k) for k in Experiment.FIELDS if hasattr(args, k)}
    experiment_mode = args.exp is not None or any(v is not None for v in overrides.values())

    merger = StateMerger(primary=primary)
    if rig.get('zmq_port'):
        merger.add_sink(ZmqPublisher(port=int(rig['zmq_port'])))

    logger = recorders = events = None
    exp = None
    recorders = {}
    if experiment_mode:
        exp = Experiment.load(args.exp, overrides)
        exp.make_output_dir()
        exp.snapshot_configs([args.rig_config, args.exp] + list(cameras.values()))
        exp.write_resolved()
        print('Recording: %s\n  -> %s' % (exp.run_name, exp.output_dir))
        if exp.log_flystate and not args.no_log:
            logger = Hdf5Logger(exp.path('_flystate.h5'), cameras=list(cameras))
            merger.add_sink(logger)
        if exp.events_port:
            events = EventReceiver(exp.events_port, logger=logger,
                                   jsonl_path=exp.path('_events.jsonl'))
            print('  events: ZMQ PULL on port %d' % exp.events_port)
    elif not args.no_log and rig.get('log_dir'):
        log_dir = os.path.join(rig_dir, rig['log_dir'])
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, time.strftime('flystate_%Y%m%d_%H%M%S.h5'))
        logger = Hdf5Logger(path, cameras=list(cameras))
        merger.add_sink(logger)
        print('Logging to %s' % path)

    views = {}
    pipelines = []
    for name, cfg_path in cameras.items():
        view = TrackerView(name, cfg_path, use_gui=use_gui)
        with open(cfg_path) as f:
            cam_cfg = (yaml.safe_load(f) or {}).get('camera') or {}
        source = make_source(cam_cfg)
        source.open()
        recorder = None
        if exp is not None and exp.record_video:
            fps = getattr(source, 'fps', None) or cam_cfg.get('fps') or 30.0
            recorder = VideoRecorder(exp.path('_%s.avi' % name), fps=fps,
                                     quality=exp.video_quality,
                                     sidecar_path=exp.path('_%s_frames.csv' % name))
            recorders[name] = recorder
        views[name] = view
        pipelines.append(CameraPipeline(view, source, merger, render=use_gui,
                                        recorder=recorder))
    if recorders:
        print('  video: %s' % ', '.join(os.path.basename(r.path) for r in recorders.values()))

    for p in pipelines:
        p.start()
    t_start = time.monotonic()

    try:
        while True:
            if any(p.eof or p.error for p in pipelines):
                break
            if any(v.bQuit for v in views.values()):
                break
            if use_gui:
                for p in pipelines:
                    img = p.latest_display()
                    if img is not None:
                        cv2.imshow(p.view.window_name, img)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.5)
                print('  '.join('%s: %d frames' % (p.view.name, p.n_frames)
                                for p in pipelines), end='\r')
    except KeyboardInterrupt:
        pass
    finally:
        for p in pipelines:
            p.stop()
        for p in pipelines:
            p.source.close()
        if events is not None:
            events.close()
        for r in recorders.values():
            r.close()
        merger.close()
        cv2.destroyAllWindows()

    dur = time.monotonic() - t_start
    print('\nDone: %.1f s' % dur)
    for p in pipelines:
        line = '  %s: %d frames tracked' % (p.view.name, p.n_frames)
        r = recorders.get(p.view.name)
        if r is not None:
            line += ', %d video frames written' % r.n_written
            if r.n_dropped:
                line += ', %d DROPPED (disk too slow)' % r.n_dropped
        print(line)
    if logger is not None:
        print('  fly state: %d rows, %d events -> %s' % (logger.n, logger.n_events, logger.path))
    if exp is not None:
        print('  folder: %s' % exp.output_dir)


if __name__ == '__main__':
    sys.exit(main())
