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


def run(tracker, roll_amp, yaw_amp, n=360, pitch_amp=0.0, landmarks=None):
    """Returns (truth, estimates) as a dict of arrays plus ms_per_frame.

    Yaw and pitch are rendered as lateral and vertical shifts of the head --
    that is what they look like end-on from the front -- so the tracker's x and
    y displacement (roll-induced part removed) are what should follow them. No
    yaw or pitch ANGLE is computed unless lever_px has been calibrated."""
    from tests.make_synthetic_video import draw_front_frame
    from kinefly2.camera.base import Frame
    from kinefly2.gui.app import TrackerView

    d = tempfile.mkdtemp()
    cfg = os.path.join(d, 'front.yaml')
    img0, lm0 = draw_front_frame(0.0, 0.0)
    pts = landmarks or [lm0[k] for k in ('eye_l', 'eye_r', 'patch', 'ant_l', 'ant_r')]
    make_config(cfg, pts, tracker)
    view = TrackerView('front', cfg, use_gui=False)
    out = {k: [] for k in ('roll', 'yaw', 'pitch', 'est_roll', 'est_dx', 'est_dy',
                           'est_axial', 'est_lateral', 'sigma')}
    dts = []
    # Drive the three rotations at harmonics of the run length, so the truth
    # signals are exactly orthogonal over the window and any correlation left
    # between channels is real cross-talk rather than correlated inputs.
    span = n / FPS
    for i in range(n):
        t = i / FPS
        roll = roll_amp * math.sin(2 * math.pi * (3 / span) * t)
        yaw = yaw_amp * math.sin(2 * math.pi * (2 / span) * t + 1.0)
        pitch = pitch_amp * math.sin(2 * math.pi * (1 / span) * t + 2.1)
        img, _ = draw_front_frame(roll, yaw, pitch)
        t0 = time.perf_counter()
        st = view.process(Frame(image=img, t_camera=t, t_host=time.monotonic(), frame_id=i))
        dts.append(time.perf_counter() - t0)
        part = st.parts['head']
        g = part.gradients
        out['est_roll'].append(part.angles[0] if len(part.angles) > 0 else np.nan)
        out['est_dx'].append(g[0] if len(g) >= 3 else np.nan)
        out['est_dy'].append(g[1] if len(g) >= 3 else np.nan)
        ex = part.extra or {}
        out['est_axial'].append(ex.get('d_axial', np.nan))
        out['est_lateral'].append(ex.get('d_lateral', np.nan))
        out['sigma'].append(ex.get('roll_sigma', np.nan))
        out['roll'].append(math.radians(roll))
        out['yaw'].append(math.radians(yaw))
        out['pitch'].append(math.radians(pitch))
        if i == 0:
            view.render()
    return {k: np.array(v) for k, v in out.items()}, 1e3 * np.median(dts[10:])


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 10 else np.nan


def test_landmarks_separate_roll_and_yaw():
    for roll_amp in (3.0, 20.0):
        r, dt = run('landmarks', roll_amp, 15.0)
        rolls, yaws, er, shift = r['roll'], r['yaw'], r['est_roll'], r['est_dx']
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
    r, _ = run('landmarks', 20.0, 0.0)
    shift = r['est_dx']
    span = np.nanmax(shift[5:]) - np.nanmin(shift[5:])
    print('landmarks roll 20 deg, no yaw: residual translation span %.2f px' % span)
    assert span < 3.0, 'pure roll leaks into translation (%.2f px)' % span

    # The hinge-based rotation tracker must be corrupted by the same yaw.
    r, dt = run('rotation', 20.0, 15.0)
    er, rolls, yaws = r['est_roll'], r['roll'], r['yaw']
    c_rr, c_ry = corr(er, rolls), corr(er, yaws)
    print('rotation  roll 20 deg + yaw 15 deg: roll r=%+.3f | roll~yaw r=%+.3f | %.2f ms' % (c_rr, c_ry, dt))
    assert abs(c_ry) > 0.2 or c_rr < 0.95, 'rotation tracker unexpectedly immune to yaw'


def test_pitch_is_the_vertical_channel():
    """Pitch is seen end-on from the front, so it appears as a vertical shift,
    scaled by the same lever arm as yaw. Roll, yaw and pitch must come out on
    three separate channels without leaking into each other."""
    r, dt = run('landmarks', 8.0, 12.0, pitch_amp=10.0)
    c_pitch = corr(r['est_axial'], r['pitch'])
    c_yaw = corr(r['est_lateral'], r['yaw'])
    cross = (corr(r['est_axial'], r['yaw']), corr(r['est_lateral'], r['pitch']),
             corr(r['est_roll'], r['pitch']))
    print('roll 8 + yaw 12 + pitch 10 deg: dy~pitch r=%+.3f | dx~yaw r=%+.3f | '
          'roll r=%+.3f | cross-talk dy~yaw %+.3f dx~pitch %+.3f roll~pitch %+.3f | %.2f ms'
          % (c_pitch, c_yaw, corr(r['est_roll'], r['roll']), cross[0], cross[1], cross[2], dt))
    assert c_pitch > 0.95, 'vertical channel does not follow pitch (r=%.3f)' % c_pitch
    assert c_yaw > 0.95, 'horizontal channel does not follow yaw (r=%.3f)' % c_yaw
    assert corr(r['est_roll'], r['roll']) > 0.95
    for name, c in zip(('dy~yaw', 'dx~pitch', 'roll~pitch'), cross):
        assert abs(c) < 0.3, '%s cross-talk r=%.3f' % (name, c)

    # Yaw and pitch share one lever arm: the px-per-radian slope must match.
    from tests.make_synthetic_video import FRONT_DEPTH
    ky = np.polyfit(np.sin(r['yaw'][5:]), r['est_lateral'][5:], 1)[0]
    kp = np.polyfit(np.sin(r['pitch'][5:]), r['est_axial'][5:], 1)[0]
    print('lever arm: from yaw %.1f px/rad, from pitch %.1f px/rad, true %.1f'
          % (abs(ky), abs(kp), FRONT_DEPTH))
    assert abs(abs(ky) - abs(kp)) / abs(ky) < 0.25, (ky, kp)
    for k in (ky, kp):
        assert abs(abs(k) - FRONT_DEPTH) / FRONT_DEPTH < 0.25, (k, FRONT_DEPTH)


def test_roll_sigma_reports_precision():
    """The reported roll uncertainty must be in the right ballpark: comparable
    to how much the estimate actually scatters about the truth."""
    r, _ = run('landmarks', 8.0, 12.0, pitch_amp=10.0)
    err = r['est_roll'][5:] - r['roll'][5:]
    actual = np.nanstd(err - np.nanmean(err))
    claimed = np.nanmedian(r['sigma'][5:])
    print('roll_sigma: claimed %.2f deg, actual scatter %.2f deg'
          % (np.degrees(claimed), np.degrees(actual)))
    assert np.isfinite(claimed) and 0 < claimed
    assert 0.1 < claimed / actual < 10.0, (claimed, actual)


def test_ambiguous_layout_is_caught_and_contained():
    """Regression for a real failure: four landmarks placed on a near-rectangle
    map onto themselves under a 180 deg rotation, so the fit cannot tell 0 from
    180 and flips. The residual stays LOW through this (a consistent swap is
    still perfectly rigid), so it must be caught geometrically instead."""
    from kinefly2.tracking.tracker_landmarks import layout_metrics, safe_search_px

    # The exact layout that broke the real front camera.
    bad = [(232.0, 160.0), (236.0, 202.0), (202.0, 161.0), (201.0, 207.0)]
    L = layout_metrics(bad)
    print('broken rectangle: min_dist %.1f px, rms_radius %.1f px, ambiguity %.1f px'
          % (L['min_dist'], L['rms_radius'], L['ambiguity']))
    assert L['ambiguity'] < 0.25 * L['rms_radius'], 'ambiguity check missed the rectangle'

    # The matcher must not be able to reach a neighbouring landmark.
    safe = safe_search_px(L['min_dist'], 25)
    assert safe + 25 // 2 < L['min_dist'], (safe, L['min_dist'])

    # An asymmetric layout of the same size must NOT be flagged.
    good = [(235.7, 158.6), (246.4, 199.7), (208.8, 172.0), (218.7, 204.4), (189.1, 185.5)]
    G = layout_metrics(good)
    assert G['ambiguity'] > 0.25 * G['rms_radius'], 'false alarm on a good layout'
    print('good layout: ambiguity %.1f px vs threshold %.1f px -> not flagged'
          % (G['ambiguity'], 0.25 * G['rms_radius']))

    # End to end: with the clamp in place the ambiguous layout must no longer
    # produce impossible roll angles, and must announce itself.
    from tests.make_synthetic_video import draw_front_frame
    img0, lm0 = draw_front_frame(0.0, 0.0)
    c = np.mean([lm0[k] for k in ('eye_l', 'eye_r')], axis=0)
    rect = [(c[0] - 15, c[1] - 21), (c[0] + 15, c[1] - 21),
            (c[0] - 15, c[1] + 21), (c[0] + 15, c[1] + 21)]
    r, _ = run('landmarks', 8.0, 12.0, pitch_amp=6.0, landmarks=rect)
    worst = np.nanmax(np.abs(np.degrees(r['est_roll'])))
    print('rectangle layout, end to end: worst |roll| %.1f deg' % worst)
    assert worst < 60.0, 'ambiguous layout still flips (worst %.1f deg)' % worst


def test_lever_px_produces_angles():
    """With lever_px set, the same constant turns both channels into angles."""
    from kinefly2.tracking.tracker_landmarks import LandmarkTracker
    from tests.make_synthetic_video import draw_front_frame, FRONT_DEPTH
    import copy

    img0, lm0 = draw_front_frame(0.0, 0.0)
    pts = [lm0[k] for k in ('eye_l', 'eye_r', 'patch', 'ant_l', 'ant_r')]
    base = {'gui': {'head': {'track': True,
                             'hinge': {'x': 176.0, 'y': 230.0},
                             'landmarks': [{'x': float(x), 'y': float(y)} for x, y in pts]}},
            'head': {'template_px': 17, 'search_px': 12, 'min_score': 0.4}}

    # Without lever_px the angles must stay NaN rather than be guessed.
    tr = LandmarkTracker(name='head', params=copy.deepcopy(base))
    tr.update(1 / 90., img0, False)
    img, _ = draw_front_frame(0.0, 10.0, 6.0)
    tr.update(1 / 90., img, False)
    assert not np.isfinite(tr.state.extra['yaw']), tr.state.extra
    assert not np.isfinite(tr.state.extra['pitch']), tr.state.extra

    p = copy.deepcopy(base)
    p['head']['lever_px'] = FRONT_DEPTH
    tr = LandmarkTracker(name='head', params=p)
    tr.update(1 / 90., img0, False)
    tr.update(1 / 90., img, False)
    yaw_deg = np.degrees(tr.state.extra['yaw'])
    pitch_deg = np.degrees(tr.state.extra['pitch'])
    print('lever_px %.1f: true yaw 10.0 pitch 6.0 -> measured yaw %.1f pitch %.1f deg'
          % (FRONT_DEPTH, yaw_deg, pitch_deg))
    assert abs(abs(yaw_deg) - 10.0) < 3.0, yaw_deg
    # Positive pitch carries the head dorsally, which is +axis, so the sign
    # must come out positive rather than merely the right magnitude.
    assert abs(pitch_deg - 6.0) < 3.0, pitch_deg


def _lm_tracker(view, axis_deg, pts, lever=None, hinge=(176.0, 230.0)):
    from kinefly2.tracking.tracker_landmarks import LandmarkTracker
    p = {'view': view,
         'gui': {'head': {'track': True,
                          'hinge': {'x': hinge[0], 'y': hinge[1]},
                          'axis_deg': axis_deg,
                          'landmarks': [{'x': float(x), 'y': float(y)} for x, y in pts]}},
         'head': {'template_px': 17, 'search_px': 12, 'min_score': 0.4}}
    if lever:
        p['head']['lever_px'] = lever
    return LandmarkTracker(name='head', params=p)


def test_view_decides_which_rotation_is_exact():
    """The same tracker means different anatomy in each view: a ventral camera
    measures yaw exactly, a frontal one roll. The other two channels come from
    the displacement and stay NaN until lever_px is calibrated."""
    from tests.make_synthetic_video import draw_front_frame

    img0, lm0 = draw_front_frame(0.0, 0.0)
    pts = [lm0[k] for k in ('eye_l', 'eye_r', 'patch', 'ant_l', 'ant_r')]
    img, _ = draw_front_frame(6.0, 10.0, 4.0)
    for view, exact, others in (('ventral', 'yaw', ('roll', 'pitch')),
                                ('frontal', 'roll', ('yaw', 'pitch'))):
        tr = _lm_tracker(view, -90.0, pts)
        tr.update(1 / 90., img0, False)
        tr.update(1 / 90., img, False)
        e = tr.state.extra
        assert e['channels']['inplane'] == exact, e['channels']
        assert np.isfinite(e[exact]), (view, e)
        assert np.isclose(e[exact], tr.state.angles[0]), (view, e[exact], tr.state.angles[0])
        for name in others:
            assert not np.isfinite(e[name]), (view, name, e[name])
        assert np.isfinite(e['d_axial']) and np.isfinite(e['d_lateral'])
        print('view=%-8s exact channel = %-5s (= the in-plane angle, %+.2f deg); '
              '%s NaN until lever_px' % (view, exact, np.degrees(e[exact]), '/'.join(others)))


def test_axis_rotates_the_decomposition():
    """axis_deg is what makes the split anatomical rather than image-aligned:
    rotating it by 90 deg must swap the axial and lateral channels, and by
    180 deg must flip their signs."""
    from tests.make_synthetic_video import draw_front_frame

    img0, lm0 = draw_front_frame(0.0, 0.0)
    pts = [lm0[k] for k in ('eye_l', 'eye_r', 'patch', 'ant_l', 'ant_r')]
    img, _ = draw_front_frame(0.0, 12.0, 8.0)

    def channels(axis_deg):
        tr = _lm_tracker('frontal', axis_deg, pts)
        tr.update(1 / 90., img0, False)
        tr.update(1 / 90., img, False)
        return tr.state.extra['d_axial'], tr.state.extra['d_lateral']

    a0, l0 = channels(-90.0)
    a90, l90 = channels(0.0)          # axis rotated by +90 deg
    a180, l180 = channels(90.0)       # axis rotated by 180 deg
    print('axis -90: axial %+6.2f lateral %+6.2f | axis 0: %+6.2f %+6.2f | axis +90: %+6.2f %+6.2f'
          % (a0, l0, a90, l90, a180, l180))
    # Rotating the axis by +90 deg carries the old lateral onto the new axial,
    # and the old axial onto the negated new lateral.
    assert np.isclose(a90, l0, atol=0.05) and np.isclose(l90, -a0, atol=0.05)
    assert np.isclose(a180, -a0, atol=0.05) and np.isclose(l180, -l0, atol=0.05)


def test_pitch_is_axial_in_both_views():
    """Pitch is the axial channel whichever way the camera looks, and the two
    displacement channels share one lever arm."""
    r, _ = run('landmarks', 4.0, 12.0, pitch_amp=10.0)
    # run() drives the synthetic head with the front generator; the axis points
    # dorsally (-90 deg), so the axial channel is -dy in image terms.
    c_axial = corr(r['est_axial'], r['pitch'])
    c_lat = corr(r['est_lateral'], r['yaw'])
    print('axial~pitch r=%+.3f | lateral~yaw r=%+.3f' % (c_axial, c_lat))
    assert abs(c_axial) > 0.95 and abs(c_lat) > 0.95


def test_parts_and_wings_are_independent():
    """The separation the rig depends on: each landmark part fits its own
    points with its own templates, and the wing angles no longer move when a
    head or abdomen hinge does."""
    from kinefly2.tracking.tracker_landmarks import LandmarkTracker
    from tests.make_synthetic_video import draw_front_frame

    # Two landmark parts on the same image, far apart. Moving one part's
    # landmarks/hinge must not change the other's output at all.
    img0, lm0 = draw_front_frame(0.0, 0.0)
    img, _ = draw_front_frame(5.0, 8.0, 4.0)
    head_pts = [lm0[k] for k in ('eye_l', 'eye_r', 'patch', 'ant_l', 'ant_r')]

    def head_out(hinge, axis):
        tr = _lm_tracker('frontal', axis, head_pts, hinge=hinge)
        tr.update(1 / 90., img0, False)
        tr.update(1 / 90., img, False)
        return tr.state.angles[0], tr.state.extra['d_axial'], tr.state.extra['d_lateral']

    a1 = head_out((176.0, 230.0), -90.0)
    a2 = head_out((176.0, 230.0), -90.0)
    assert np.isfinite(a1[0]), 'the fit failed; the test set is bad, not the tracker'
    assert a1 == a2, 'not deterministic'
    # The in-plane angle needs no rotation centre, so moving the hinge far away
    # must leave it bit-identical.
    a3 = head_out((40.0, 40.0), -90.0)
    assert a3[0] == a1[0], (a1[0], a3[0])
    print('in-plane angle is hinge-independent: %+.6f rad either way' % a1[0])

    # Wing angles must not depend on the head/abdomen hinges once the body
    # axis is frozen in the config.
    from kinefly2.tracking.fly import Fly
    from kinefly2.gui.app import tracker_defaults
    import copy as _copy
    base = tracker_defaults()
    base['body_axis_deg'] = 90.0
    f1 = Fly(_copy.deepcopy(base))
    moved = _copy.deepcopy(base)
    moved['gui']['abdomen']['hinge'] = {'x': 60, 'y': 300}
    moved['gui']['head']['hinge'] = {'x': 300, 'y': 20}
    f2 = Fly(moved)
    f1.set_params(_copy.deepcopy(base)); f2.set_params(moved)
    assert f1.left.angleBody_i == f2.left.angleBody_i, (f1.left.angleBody_i, f2.left.angleBody_i)
    assert f1.right.angleBody_i == f2.right.angleBody_i
    print('wing body frame unchanged by hinge moves: %.6f rad both' % f1.left.angleBody_i)


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
    test_pitch_is_the_vertical_channel()
    test_roll_sigma_reports_precision()
    test_ambiguous_layout_is_caught_and_contained()
    test_lever_px_produces_angles()
    test_view_decides_which_rotation_is_exact()
    test_axis_rotates_the_decomposition()
    test_pitch_is_axial_in_both_views()
    test_parts_and_wings_are_independent()
    test_zero_recaptures_templates()
    print('OK')
