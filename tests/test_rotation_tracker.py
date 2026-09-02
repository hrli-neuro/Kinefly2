"""RotationTracker (template rotation search) vs the area tracker on small
head rotations, on in-memory synthetic frames with known ground truth."""
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

VID_CFG = os.path.join(HERE, 'videos', 'synthetic_bottom.yaml')


def ensure_synthetic():
    subprocess.check_call([sys.executable, os.path.join(HERE, 'make_synthetic_video.py')])


def run(tracker, amp_deg, n=300, search_deg=8):
    import yaml
    from tests import make_synthetic_video as syn
    from kinefly2.camera.base import Frame
    from kinefly2.gui.app import TrackerView

    syn.HEAD_AMP_DEG = amp_deg
    view = TrackerView('rot', VID_CFG, use_gui=False)
    view.params['head']['tracker'] = tracker
    view.params['head']['autozero'] = False
    view.params['head']['search_deg'] = search_deg
    view.fly = type(view.fly)(view.scale_params(view.params, view.scale))
    view.fly.set_params(view.scale_params(view.params, view.scale))
    view.fly.bInvertColor = False
    view.fly.bInvertColorValid = True
    view.fly.bInvertColorAuto = False

    out, truth, dts = [], [], []
    for i in range(n):
        t = i / syn.FPS
        img, aL, aR, aH = syn.draw_frame(t)
        fr = Frame(image=img, t_camera=t, t_host=time.monotonic(), frame_id=i)
        t0 = time.perf_counter()
        st = view.process(fr)
        dts.append(time.perf_counter() - t0)
        out.append(st.parts['head'].angle)
        truth.append(math.radians(aH + 90.0))
    return np.array(out), np.array(truth), 1e3 * np.median(dts[20:])


def signed_corr(a, b, skip=5):
    a, b = a[skip:], b[skip:]
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


def test_rotation_tracker_small_angles():
    ensure_synthetic()
    # Sign convention must match the area tracker (checked on large rotations).
    area15, truth15, _ = run('area', 15.0)
    rot15, _, _ = run('rotation', 15.0, search_deg=20)
    c_area = signed_corr(area15, truth15)
    c_rot15 = signed_corr(rot15, truth15)
    print('15 deg: area r=%+.3f  rotation r=%+.3f' % (c_area, c_rot15))
    assert abs(c_rot15) > 0.98
    assert np.sign(c_rot15) == np.sign(c_area), 'sign convention differs from area tracker'

    # Small angles: this is what the area tracker cannot resolve.
    area3, truth3, _ = run('area', 3.0)
    rot3, _, dt = run('rotation', 3.0)
    c_area3 = signed_corr(area3, truth3)
    c_rot3 = signed_corr(rot3, truth3)
    slope = np.polyfit(truth3[5:], rot3[5:], 1)[0]
    print('3 deg:  area r=%+.3f  rotation r=%+.3f  gain %.2f  cost %.2f ms/frame' % (
        c_area3, c_rot3, slope, dt))
    assert abs(c_rot3) > 0.95, 'rotation tracker r=%.3f at 3 deg' % c_rot3
    assert 0.8 < abs(slope) < 1.2, 'rotation tracker gain %.2f (should be ~1)' % slope


if __name__ == '__main__':
    test_rotation_tracker_small_angles()
    print('OK')
