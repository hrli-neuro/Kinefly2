"""End-to-end verification of the ported tracking core on synthetic video.

Runs the full headless pipeline (VideoFileSource -> TrackerView -> Fly) on the
synthetic bottom-view video and checks that the tracked left/right wing angles
and head rotation correlate with the known ground truth. This is the
milestone-1 parity gate: it exercises the polar transforms, the edge tracker,
the area tracker, and the config/scale plumbing.

Run:  python -m pytest tests/ -v      (or python tests/test_tracking.py)
"""
import csv
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

VID_CFG = os.path.join(HERE, 'videos', 'synthetic_bottom.yaml')
TRUTH = os.path.join(HERE, 'videos', 'synthetic_bottom_truth.csv')


def ensure_synthetic():
    if not (os.path.exists(VID_CFG) and os.path.exists(TRUTH)):
        subprocess.check_call([sys.executable, os.path.join(HERE, 'make_synthetic_video.py')])


def run_tracker():
    from kinefly2.camera.base import make_source
    from kinefly2.gui.app import TrackerView
    import yaml

    with open(VID_CFG) as f:
        cam_cfg = yaml.safe_load(f)['camera']

    view = TrackerView('synthetic', VID_CFG, use_gui=False)
    rows = []
    with make_source(cam_cfg) as source:
        while True:
            frame = source.read()
            if frame is None:
                break
            state = view.process(frame)
            if state is None:
                continue
            rows.append((frame.frame_id,
                         state.parts['left'].angle,
                         state.parts['right'].angle,
                         state.parts['head'].angle))
    return np.array(rows, dtype=float)


def load_truth():
    with open(TRUTH) as f:
        r = csv.DictReader(f)
        rows = [(int(d['frame']), float(d['left_deg']), float(d['right_deg']),
                 float(d['head_deg'])) for d in r]
    return np.array(rows, dtype=float)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return 0.0
    return abs(np.corrcoef(a[m], b[m])[0, 1])


def test_synthetic_tracking():
    ensure_synthetic()
    tracked = run_tracker()
    truth = load_truth()

    assert len(tracked) >= 0.9 * len(truth), \
        'Tracked only %d of %d frames' % (len(tracked), len(truth))

    # Align on frame id, skip the settling period (autozero, background).
    n = min(len(tracked), len(truth))
    skip = 20
    tr = tracked[skip:n]
    gt = truth[skip:n]

    c_left = corr(tr[:, 1], np.radians(gt[:, 1]))
    c_right = corr(tr[:, 2], np.radians(gt[:, 2]))
    c_head = corr(tr[:, 3], np.radians(gt[:, 3]))
    frac_valid = np.isfinite(tr[:, 1:3]).mean()

    print('valid wing measurements: %.0f%%' % (100 * frac_valid))
    print('correlation with ground truth: left %.3f  right %.3f  head %.3f'
          % (c_left, c_right, c_head))

    assert frac_valid > 0.9, 'Too many missing wing measurements'
    assert c_left > 0.95, 'Left wing tracking does not follow ground truth (r=%.3f)' % c_left
    assert c_right > 0.95, 'Right wing tracking does not follow ground truth (r=%.3f)' % c_right
    assert c_head > 0.8, 'Head tracking does not follow ground truth (r=%.3f)' % c_head


if __name__ == '__main__':
    test_synthetic_tracking()
    print('OK')
