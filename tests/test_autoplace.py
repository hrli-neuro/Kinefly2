"""Auto-placement of tracking regions.

Tier A: shift+rotate the synthetic frame by a known rigid transform and check
that registration to the reference moves the hinges onto the transformed truth.
Tier B: silhouette bootstrap on the synthetic bottom view lands the hinges
near the true body geometry.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

VID_CFG = os.path.join(HERE, 'videos', 'synthetic_bottom.yaml')
VID = os.path.join(HERE, 'videos', 'synthetic_bottom.avi')


def ensure_synthetic():
    subprocess.check_call([sys.executable, os.path.join(HERE, 'make_synthetic_video.py')])


def load_frame(i=0):
    cap = cv2.VideoCapture(VID)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_register_known_transform():
    ensure_synthetic()
    import yaml
    from kinefly2.gui import autoplace
    params = yaml.safe_load(open(VID_CFG))['tracker']
    ref = load_frame(0)
    for deg, dx, dy in ((0, 12, -7), (6, -15, 9), (-9, 20, 14)):
        M = cv2.getRotationMatrix2D((176.0, 176.0), -deg, 1.0)   # image rotates by +deg (y down)
        M[0, 2] += dx
        M[1, 2] += dy
        cur = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))
        W, method = autoplace.register(ref, cur, autoplace.body_mask(params, ref.shape))
        moved, rot = autoplace.transform_params(params, W)
        for part in ('head', 'abdomen', 'left', 'right'):
            p = params['gui'][part]['hinge']
            truth = M @ np.array([p['x'], p['y'], 1.0])
            got = moved['gui'][part]['hinge']
            err = math.hypot(got['x'] - truth[0], got['y'] - truth[1])
            assert err < 2.0, '%s hinge off by %.2f px (transform %s, method %s)' % (part, err, (deg, dx, dy), method)
        rot_err = abs(math.degrees(rot) - math.degrees(math.atan2(M[1, 0], M[0, 0])))
        assert rot_err < 1.0, 'rotation off by %.2f deg' % rot_err
        print('transform %s: method %s, max hinge error < 2 px, rot err %.2f deg' % ((deg, dx, dy), method, rot_err))


def test_silhouette_bootstrap():
    ensure_synthetic()
    import yaml
    from kinefly2.gui import autoplace
    from kinefly2.gui.app import tracker_defaults
    from tests.make_synthetic_video import HEAD_HINGE, ABD_HINGE, LEFT_HINGE, RIGHT_HINGE
    img = load_frame(0)
    # Synthetic fly is bright on black (invertcolor False); the real bottom view
    # is the opposite, and the code path is identical apart from the threshold polarity.
    out, info = autoplace.silhouette_bootstrap(img, tracker_defaults(), invertcolor=False)
    body_len = math.hypot(HEAD_HINGE[0] - ABD_HINGE[0], HEAD_HINGE[1] - ABD_HINGE[1])
    g = out['gui']
    # Body axis: head above abdomen (smaller y).
    assert g['head']['hinge']['y'] < g['abdomen']['hinge']['y'], 'head/abdomen swapped'
    axis_angle = math.degrees(math.atan2(g['head']['hinge']['y'] - g['abdomen']['hinge']['y'],
                                         g['head']['hinge']['x'] - g['abdomen']['hinge']['x']))
    assert abs(axis_angle + 90) < 5, 'body axis angle %.1f' % axis_angle
    for part, truth in (('head', HEAD_HINGE), ('abdomen', ABD_HINGE),
                        ('left', LEFT_HINGE), ('right', RIGHT_HINGE)):
        h = g[part]['hinge']
        err = math.hypot(h['x'] - truth[0], h['y'] - truth[1]) / body_len
        assert err < 0.25, '%s hinge %.0f%% of body length from truth' % (part, 100 * err)
        print('bootstrap %s: %.0f%% of body length from truth' % (part, 100 * err))


if __name__ == '__main__':
    test_register_known_transform()
    test_silhouette_bootstrap()
    print('OK')
