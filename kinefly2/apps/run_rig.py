"""Real-time multi-camera tracking with merged state output.

One pipeline thread per camera; the GUI (imshow/waitKey) stays in the main
thread. The merged fly state is published over ZMQ and logged to HDF5.
The primary camera (--primary, default 'bottom') drives the output loop; the
future LED-panel closed-loop driver attaches as another sink in rig.yaml.

    python -m kinefly2.apps.run_rig config/rig.yaml
    python -m kinefly2.apps.run_rig config/rig.yaml --no-gui
"""
import argparse
import os
import sys
import time

import cv2
import yaml

from ..camera.base import make_source
from ..core.merger import StateMerger
from ..core.pipeline import CameraPipeline
from ..gui.app import TrackerView
from ..output.logger import Hdf5Logger
from ..output.zmq_pub import ZmqPublisher


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rig_config', help='rig.yaml naming the cameras and outputs')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--no-log', action='store_true')
    args = ap.parse_args()

    with open(args.rig_config) as f:
        rig = yaml.safe_load(f)
    rig_dir = os.path.dirname(os.path.abspath(args.rig_config))

    cameras = rig['cameras']            # name -> path to per-camera YAML
    primary = rig.get('primary', 'bottom')
    use_gui = not args.no_gui and rig.get('use_gui', True)

    merger = StateMerger(primary=primary)
    if rig.get('zmq_port'):
        merger.add_sink(ZmqPublisher(port=int(rig['zmq_port'])))
    logger = None
    if not args.no_log and rig.get('log_dir'):
        log_dir = os.path.join(rig_dir, rig['log_dir'])
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, time.strftime('flystate_%Y%m%d_%H%M%S.h5'))
        logger = Hdf5Logger(path, cameras=list(cameras))
        merger.add_sink(logger)
        print('Logging to %s' % path)

    views = {}
    pipelines = []
    for name, cfg_path in cameras.items():
        cfg_path = os.path.join(rig_dir, cfg_path)
        view = TrackerView(name, cfg_path, use_gui=use_gui)
        with open(cfg_path) as f:
            cam_cfg = (yaml.safe_load(f) or {}).get('camera') or {}
        source = make_source(cam_cfg)
        source.open()
        views[name] = view
        pipelines.append(CameraPipeline(view, source, merger, render=use_gui))

    for p in pipelines:
        p.start()

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
        merger.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
