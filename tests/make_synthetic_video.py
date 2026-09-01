"""Generate a synthetic bottom-view fly video with known ground-truth angles,
plus a matching tracker config and ground-truth CSV. Used by test_tracking.py
to verify the ported algorithms end-to-end without hardware.

Geometry (352x352, y down, fly bright on black):
  body axis vertical, head up: head hinge (176,96), abdomen hinge (176,256)
  wing hinges: left (156,176), right (196,176)
  left wing: filled sector, leading edge at 140+30*sin(2*pi*1*t) deg (image frame),
             intensity ramp so the leading edge has the strongest gradient
  right wing: mirror image
  head: bright disc above the head hinge with a dark notch rotating
        20*sin(2*pi*0.7*t) deg about straight-up
"""
import csv
import math
import os

import cv2
import numpy as np
import yaml

W = H = 352
FPS = 90.0
N_FRAMES = 300
HEAD_HINGE = (176.0, 96.0)
ABD_HINGE = (176.0, 256.0)
LEFT_HINGE = (156.0, 176.0)
RIGHT_HINGE = (196.0, 176.0)
WING_R = 75.0
HEAD_R = 26.0

HERE = os.path.dirname(os.path.abspath(__file__))


def sector(img, hinge, a0_deg, a1_deg, r, color):
    """Filled pie sector from hinge between image-frame angles a0..a1 (deg)."""
    angles = np.linspace(math.radians(a0_deg), math.radians(a1_deg), 32)
    pts = [hinge] + [(hinge[0] + r * math.cos(a), hinge[1] + r * math.sin(a))
                     for a in angles]
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)


def draw_frame(t):
    img = np.zeros((H, W), dtype=np.uint8)

    # Wings first (body drawn over their roots).
    aL = 140.0 + 30.0 * math.sin(2 * math.pi * 1.0 * t)     # left leading edge, deg
    aR = 180.0 - aL                                          # mirrored
    # Intensity ramp from trailing (95 deg) to leading edge: strongest gradient
    # sits at the leading edge, which is what the edge tracker should find.
    for frac, level in ((1.0, 60), (0.66, 120), (0.33, 200)):
        span = (aL - 95.0) * frac
        sector(img, LEFT_HINGE, aL - span, aL, WING_R, level)
        spanR = (85.0 - aR) * frac
        sector(img, RIGHT_HINGE, aR, aR + spanR, WING_R, level)

    # Body.
    cv2.ellipse(img, (176, 176), (26, 90), 0, 0, 360, 255, -1)

    # Head: an eccentric ellipse (wider than long, like a real head) that
    # rotates as a whole about the neck hinge, plus antenna-like spots, so the
    # entire annulus content moves with the head angle.
    # The head must be smaller than its tracking wedge (as in real use), so the
    # rotating silhouette boundary is the registration feature.
    aH = -90.0 + 15.0 * math.sin(2 * math.pi * 0.7 * t)      # head direction, deg
    cx, cy = HEAD_HINGE
    hx = cx + 13.0 * math.cos(math.radians(aH))
    hy = cy + 13.0 * math.sin(math.radians(aH))
    cv2.ellipse(img, ((hx, hy), (18, 22), aH + 90.0), 255, -1)
    for da, r_feat, sz, level in ((-18, 17, 2, 0), (18, 17, 2, 0), (0, 19, 2, 80)):
        a = math.radians(aH + da)
        px = int(cx + r_feat * math.cos(a))
        py = int(cy + r_feat * math.sin(a))
        cv2.circle(img, (px, py), sz, level, -1)

    return img, aL, aR, aH


def main():
    vid_dir = os.path.join(HERE, 'videos')
    os.makedirs(vid_dir, exist_ok=True)
    vid_path = os.path.join(vid_dir, 'synthetic_bottom.avi')
    csv_path = os.path.join(vid_dir, 'synthetic_bottom_truth.csv')
    cfg_path = os.path.join(vid_dir, 'synthetic_bottom.yaml')

    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H),
                             isColor=False)
    with open(csv_path, 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['frame', 't', 'left_deg', 'right_deg', 'head_deg'])
        for i in range(N_FRAMES):
            t = i / FPS
            img, aL, aR, aH = draw_frame(t)
            writer.write(img)
            cw.writerow([i, '%.6f' % t, '%.4f' % aL, '%.4f' % aR, '%.4f' % aH])
    writer.release()

    # Tracker config. Body axis points up: angleBody_i = -pi/2, so
    # body-frame angle = image-frame angle + pi/2 (wrapped to [-pi, pi)).
    d2r = math.radians
    tracker = {
        'scale_image': 1.0,
        'n_edges_max': 1,
        'rc_background': 1000.0,
        'wingbeat_min': 180,
        'wingbeat_max': 220,
        'gui': {
            'windows': False,
            'symmetric': False,
            'head': {'track': True, 'subtract_bg': False, 'stabilize': False,
                     'hinge': {'x': HEAD_HINGE[0], 'y': HEAD_HINGE[1]},
                     'radius_outer': 28.0, 'radius_inner': 8.0,
                     'angle_hi': d2r(45.0), 'angle_lo': d2r(-45.0)},
            'abdomen': {'track': False, 'subtract_bg': False, 'stabilize': False,
                        'hinge': {'x': ABD_HINGE[0], 'y': ABD_HINGE[1]},
                        'radius_outer': 20.0, 'radius_inner': 10.0,
                        'angle_hi': d2r(225.0), 'angle_lo': d2r(135.0)},
            # Windows chosen to include only the moving leading edge
            # (image 105..185 deg left; -5..75 deg right; +90 for body frame).
            'left': {'track': True, 'subtract_bg': False, 'stabilize': False,
                     'hinge': {'x': LEFT_HINGE[0], 'y': LEFT_HINGE[1]},
                     'radius_outer': 70.0, 'radius_inner': 30.0,
                     'angle_hi': d2r(-85.0), 'angle_lo': d2r(-165.0)},
            'right': {'track': True, 'subtract_bg': False, 'stabilize': False,
                      'hinge': {'x': RIGHT_HINGE[0], 'y': RIGHT_HINGE[1]},
                      'radius_outer': 70.0, 'radius_inner': 30.0,
                      'angle_hi': d2r(165.0), 'angle_lo': d2r(85.0)},
            'aux': {'track': False, 'subtract_bg': False,
                    'center': {'x': 50.0, 'y': 50.0},
                    'radius1': 30.0, 'radius2': 20.0, 'angle': 0.0},
            'axis': {'track': False,
                     'pt1': {'x': 176.0, 'y': 96.0},
                     'pt2': {'x': 176.0, 'y': 256.0}},
        },
        'head': {'tracker': 'area', 'autozero': True, 'threshold': 0.0,
                 'feathering': 0.25, 'saturation_correction': False},
        'abdomen': {'tracker': 'tip', 'autozero': True, 'threshold': 0.0,
                    'feathering': 0.25, 'saturation_correction': False},
        'left': {'tracker': 'edge', 'autozero': True, 'threshold': 0.0,
                 'feathering': 0.25, 'saturation_correction': False},
        'right': {'tracker': 'edge', 'autozero': True, 'threshold': 0.0,
                  'feathering': 0.25, 'saturation_correction': False},
        'aux': {'tracker': 'intensity', 'saturation_correction': False},
    }
    config = {'name': 'synthetic',
              'camera': {'kind': 'video', 'path': vid_path, 'fps': FPS},
              'tracker': tracker}
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print('Wrote %s (%d frames), %s, %s' % (vid_path, N_FRAMES, csv_path, cfg_path))


if __name__ == '__main__':
    main()
