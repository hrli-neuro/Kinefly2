"""Offline single-camera tracking (development / analysis mode).

Runs the full tracker on a video file, synchronously in the main thread (which
is also what macOS HighGUI requires). Use the GUI to place hinges and masks;
edits are saved back to the camera YAML, so a config tuned here works
unchanged on the rig.

    python -m kinefly2.apps.run_offline config/camera_bottom.yaml
    python -m kinefly2.apps.run_offline config/camera_bottom.yaml --csv out.csv --loop
"""
import argparse
import math
import sys

import cv2

from ..camera.base import make_source
from ..gui.app import TrackerView

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config', help='Per-camera YAML (with camera: and tracker: sections)')
    ap.add_argument('--name', default=None, help='Camera name (default: from config or filename)')
    ap.add_argument('--csv', default=None, help='Write per-frame angles to this CSV file')
    ap.add_argument('--video', default=None, help='Override camera.path with this video file')
    ap.add_argument('--loop', action='store_true', help='Loop the video')
    ap.add_argument('--pace', action='store_true', help='Play at the video frame rate')
    ap.add_argument('--no-gui', action='store_true')
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    cam_cfg = dict(config.get('camera') or {})
    if args.video:
        cam_cfg['kind'] = 'video'
        cam_cfg['path'] = args.video
    if args.loop:
        cam_cfg['loop'] = True
    if args.pace:
        cam_cfg['pace'] = True

    name = args.name or config.get('name') or 'camera'
    view = TrackerView(name, args.config, use_gui=not args.no_gui)

    csv_file = None
    if args.csv:
        csv_file = open(args.csv, 'w')
        csv_file.write('frame,t,head,left,right,abdomen,aux\n')

    source = make_source(cam_cfg)
    source.open()
    try:
        while not view.bQuit:
            frame = source.read()
            if frame is None:
                break
            state = view.process(frame)

            if state is not None and csv_file is not None:
                def a(part):
                    ps = state.parts.get(part)
                    return ps.angle if ps else math.nan
                aux = state.parts.get('aux')
                csv_file.write('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (
                    frame.frame_id, state.t_camera, a('head'), a('left'),
                    a('right'), a('abdomen'), aux.intensity if aux else math.nan))

            if not args.no_gui:
                img = view.render()
                if img is not None:
                    cv2.imshow(view.window_name, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        source.close()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
