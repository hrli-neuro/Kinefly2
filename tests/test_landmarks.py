"""LandmarkTracker: roll and yaw separation on a synthetic front-view head
driven by INDEPENDENT roll and yaw signals; comparison with the hinge-based
rotation tracker, which the same yaw must corrupt."""
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FPS = 90.0


def make_config(path, landmarks, tracker='landmarks'):
    import yaml
    from kinefly2.gui.app import tracker_defaults
    from tests.make_synthetic_video import FRONT_NECK
    p = tracker_defaults()
    p['invertcolor'] = False
    p['head'].update({'tracker': tracker, 'autozero': False, 'template_px': 17,
                      'search_px': 20, 'min_score': 0.5, 'search_deg': 25})
    g = p['gui']
    g['head'].update({'track': True, 'hinge': {'x': FRONT_NECK[0], 'y': FRONT_NECK[1]},
                      'radius_inner': 20.0, 'radius_outer': 100.0,
                      'angle_lo': -1.2, 'angle_hi': 1.2,
                      'landmarks': [{'x': float(x), 'y': float(y)} for (x, y) in landmarks]})
    g['abdomen']['hinge'] = {'x': FRONT_NECK[0], 'y': FRONT_NECK[1] + 60}   # body axis: up
    for part in ('left', 'right', 'abdomen', 'aux', 'axis'):
        g[part]['track'] = False
    with open(path, 'w') as f:
        yaml.safe_dump({'name': 'front', 'camera': {'kind': 'video', 'path': 'none'},
                        'tracker': p}, f)


def run(tracker, roll_amp, yaw_amp, n=360):
    """Returns (roll_truth, yaw_truth, roll_est_rad, shift_x_px, ms_per_frame).

    The yaw signal is rendered as a lateral shift of the head, so the tracker's
    x displacement (roll-induced part removed) is what should follow it -- no
    yaw ANGLE is computed, by design."""
    from tests.make_synthetic_video import draw_front_frame
    from kinefly2.camera.base import Frame
    from kinefly2.gui.app import TrackerView

    d = tempfile.mkdtemp()
    cfg = os.path.join(d, 'front.yaml')
    img0, lm0 = draw_front_frame(0.0, 0.0)
    make_config(cfg, [lm0[k] for k in ('eye_l', 'eye_r', 'patch', 'ant_l', 'ant_r')], tracker)
    view = TrackerView('front', cfg, use_gui=False)
    rolls, yaws, est_roll, est_shift, dts = [], [], [], [], []
    for i in range(n):
        t = i / FPS
        roll = roll_amp * math.sin(2 * math.pi * 0.7 * t)
        yaw = yaw_amp * math.sin(2 * math.pi * 0.45 * t + 1.0)     # independent of roll
        img, _ = draw_front_frame(roll, yaw)
        t0 = time.perf_counter()
        st = view.process(Frame(image=img, t_camera=t, t_host=time.monotonic(), frame_id=i))
        dts.append(time.perf_counter() - t0)
        part = st.parts['head']
        est_roll.append(part.angles[0] if len(part.angles) > 0 else np.nan)
        est_shift.append(part.gradients[0] if len(part.gradients) >= 3 else np.nan)
        rolls.append(math.radians(roll))
        yaws.append(math.radians(yaw))
        if i == 0:
            view.render()
    return (np.array(rolls), np.array(yaws), np.array(est_roll), np.array(est_shift),
            1e3 * np.median(dts[10:]))


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 10 else np.nan


def test_landmarks_separate_roll_and_yaw():
    for roll_amp in (3.0, 20.0):
        rolls, yaws, er, shift, dt = run('landmarks', roll_amp, 15.0)
        c_rr, c_ry, c_sy = corr(er, rolls), corr(er, yaws), corr(shift, yaws)
        gain = np.polyfit(rolls[5:], er[5:], 1)[0]
        valid = np.isfinite(er).mean()
        print('landmarks roll %2.0f deg + yaw 15 deg: roll r=%+.3f gain %.2f | roll~yaw r=%+.3f | '
              'shift~yaw r=%+.3f | valid %.0f%% | %.2f ms' % (roll_amp, c_rr, gain, c_ry, c_sy,
                                                              100 * valid, dt))
        assert valid > 0.98
        assert c_rr > 0.98, 'roll not tracked (r=%.3f)' % c_rr
        assert 0.9 < gain < 1.1, 'roll gain %.2f' % gain
        assert abs(c_ry) < 0.15, 'roll estimate leaks yaw (r=%.3f)' % c_ry
        assert abs(c_sy) > 0.95, 'head translation does not follow yaw (r=%.3f)' % c_sy
        assert dt < 1.5, 'too slow: %.2f ms' % dt

    # A pure roll must leave (almost) no translation once the roll-induced
    # centroid shift about the hinge has been removed.
    rolls, _, er, shift, _ = run('landmarks', 20.0, 0.0)
    span = np.nanmax(shift[5:]) - np.nanmin(shift[5:])
    print('landmarks roll 20 deg, no yaw: residual translation span %.2f px' % span)
    assert span < 3.0, 'pure roll leaks into translation (%.2f px)' % span

    # The hinge-based rotation tracker must be corrupted by the same yaw.
    rolls, yaws, er, _, dt = run('rotation', 20.0, 15.0)
    c_rr, c_ry = corr(er, rolls), corr(er, yaws)
    print('rotation  roll 20 deg + yaw 15 deg: roll r=%+.3f | roll~yaw r=%+.3f | %.2f ms' % (c_rr, c_ry, dt))
    assert abs(c_ry) > 0.2 or c_rr < 0.95, 'rotation tracker unexpectedly immune to yaw'


def test_rigid_fit_and_matching():
    from kinefly2.tracking.tracker_landmarks import rigid_fit
    P = np.array([[0, 0], [10, 0], [10, 8], [0, 8], [5, 4]], float)
    th = math.radians(12.0)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    Q = (R @ P.T).T + np.array([3.0, -2.0])
    theta, scale, dc, resid = rigid_fit(P, Q)
    assert abs(theta - th) < 1e-9 and resid.max() < 1e-9
    assert np.allclose(dc, Q.mean(0) - P.mean(0))
    theta2, scale2, _, _ = rigid_fit(P, 1.5 * Q, similarity=True)
    assert abs(scale2 - 1.5) < 1e-9 and abs(theta2 - th) < 1e-9


def test_zero_recaptures_templates():
    from tests.make_synthetic_video import draw_front_frame
    from kinefly2.camera.base import Frame
    from kinefly2.gui.app import TrackerView
    d = tempfile.mkdtemp(); cfg = os.path.join(d, 'front.yaml')
    img0, lm0 = draw_front_frame(0.0, 0.0)
    make_config(cfg, [lm0[k] for k in ('eye_l', 'eye_r', 'patch', 'ant_l', 'ant_r')])
    view = TrackerView('front', cfg, use_gui=False)
    view.process(Frame(image=img0, t_camera=0, t_host=time.monotonic(), frame_id=0))
    img1, _ = draw_front_frame(10.0, 0.0)               # head rolled 10 deg
    st = view.process(Frame(image=img1, t_camera=0.1, t_host=time.monotonic(), frame_id=1))
    assert st.parts['head'].kind == 'landmarks'
    assert len(st.parts['head'].angles) == 1, 'angles must hold the roll only'
    assert abs(math.degrees(st.parts['head'].angles[0]) - 10) < 1.0
    summary = view.zero_all()                           # this (rolled) posture = zero
    assert 'head 5/5' in summary, summary
    st = view.process(Frame(image=img1, t_camera=0.2, t_host=time.monotonic(), frame_id=2))
    assert abs(math.degrees(st.parts['head'].angles[0])) < 0.3, st.parts['head'].angles
    img2, _ = draw_front_frame(13.0, 0.0)               # 3 deg beyond the new zero
    st = view.process(Frame(image=img2, t_camera=0.3, t_host=time.monotonic(), frame_id=3))
    assert abs(math.degrees(st.parts['head'].angles[0]) - 3) < 0.7, st.parts['head'].angles
    print('zero OK')


if __name__ == '__main__':
    test_rigid_fit_and_matching()
    test_landmarks_separate_roll_and_yaw()
    test_zero_recaptures_templates()
    print('OK')
