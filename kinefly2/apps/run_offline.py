"""Offline single-camera tracking (development / tuning the rig config).

Runs the full tracker on a video file, synchronously in the main thread (which
is also what macOS HighGUI requires). Use the GUI to place hinges and masks;
edits are saved back to the camera YAML, so a config tuned here works
unchanged on the rig.

    python -m kinefly2.apps.run_offline config/camera_bottom.yaml
    python -m kinefly2.apps.run_offline config/camera_bottom.yaml --csv out.csv --loop

To ANALYZE recorded videos -- pick files, tune on a paused or slowed video,
results in their own never-overwritten folder, rig config untouched -- use
kinefly2.apps.analyze_videos instead.
"""
import argparse
import math
import os
import sys

import cv2

from ..camera.base import make_source
from ..camera.video_source import median_background
from ..gui.app import TrackerView

import yaml


def make_median_background(name, config_path, cam_cfg, n_samples=41):
    """Median of frames sampled across the whole video: anything that moves
    (wings, legs) vanishes; the static scene remains."""
    if cam_cfg.get('kind', 'video') != 'video' or not cam_cfg.get('path'):
        print('--make-bg needs a video file (camera kind: video, or --video FILE).')
        return 1
    try:
        bg, n_used, n_total = median_background(cam_cfg['path'], n_samples)
    except IOError as e:
        print(e)
        return 1
    out = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                       'background_%s.png' % name)
    cv2.imwrite(out, bg)
    print('Wrote %s (median of %d of %d frames)' % (out, n_used, n_total))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config', help='Per-camera YAML (with camera: and tracker: sections)')
    ap.add_argument('--name', default=None, help='Camera name (default: from config or filename)')
    ap.add_argument('--csv', default=None, help='Write per-frame angles to this CSV file')
    ap.add_argument('--video', default=None, help='Override camera.path with this video file')
    ap.add_argument('--loop', action='store_true', help='Loop the video')
    ap.add_argument('--pace', action='store_true', help='Play at the video frame rate')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--make-bg', action='store_true',
                    help='Compute a background image as the temporal median of '
                         'the video (the moving wings/legs vanish, the static '
                         'body/tether/illumination stay), save it as this '
                         "camera's background, and exit. Then enable "
                         "'subtract' for the wings in the GUI.")
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

    if args.make_bg:
        return make_median_background(name, args.config, cam_cfg)

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
