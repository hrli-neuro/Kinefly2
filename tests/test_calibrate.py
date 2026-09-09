"""The lever-arm fit must recover a known constant, and must refuse to
produce one from a recording that cannot support it."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinefly2.analysis import calibrate


def synth(lever_px=90.0, yaw_amp_deg=12.0, noise_px=1.5, n=6000, seed=1):
    """Front-camera shift produced by a known yaw, plus measurement noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 60.0
    yaw = np.radians(yaw_amp_deg) * np.sin(2 * np.pi * 0.3 * t)
    yaw += np.radians(0.3) * rng.standard_normal(n)      # reference noise
    shift = lever_px * np.sin(yaw) + 4.0                 # 4 px arbitrary offset
    shift += noise_px * rng.standard_normal(n)
    return shift, yaw


def test_recovers_known_lever():
    for lever in (60.0, 90.0, 140.0):
        shift, yaw = synth(lever_px=lever)
        fit = calibrate.fit_lever(shift, yaw, n_boot=120)
        ok, why = calibrate.check_fit(fit)
        err = abs(fit['lever_px'] - lever) / lever
        assert ok, (lever, why)
        assert err < 0.05, (lever, fit['lever_px'], err)
        lo, hi = fit['ci']
        assert lo <= lever <= hi, (lever, fit['ci'])
        print('lever %5.1f -> fitted %6.1f px/rad (%.1f%% error), r2 %.3f, CI %.1f..%.1f'
              % (lever, fit['lever_px'], 100 * err, fit['r2'], lo, hi))


def test_rejects_outliers():
    """A few wild frames must not set the scale."""
    shift, yaw = synth(lever_px=90.0)
    rng = np.random.default_rng(7)
    bad = rng.choice(len(shift), 200, replace=False)
    shift[bad] += 60.0 * rng.standard_normal(200)      # lost landmarks
    fit = calibrate.fit_lever(shift, yaw, n_boot=120)
    assert abs(fit['lever_px'] - 90.0) / 90.0 < 0.08, fit['lever_px']
    print('with 3%% wild frames -> %.1f px/rad (robust)' % fit['lever_px'])


def test_refuses_when_yaw_barely_moves():
    """The real cam1/cam2 case: yaw std 0.35 deg. No constant may be adopted."""
    shift, yaw = synth(lever_px=90.0, yaw_amp_deg=0.5, noise_px=5.0)
    fit = calibrate.fit_lever(shift, yaw, n_boot=120)
    ok, why = calibrate.check_fit(fit)
    assert not ok, fit
    assert any('yaw spans' in w for w in why), why
    print('yaw range %.2f deg -> REJECTED: %s' % (fit['yaw_range_deg'], why[0]))


def test_refuses_uncorrelated():
    """Shift that has nothing to do with the reference yaw."""
    rng = np.random.default_rng(3)
    _, yaw = synth(yaw_amp_deg=12.0)
    shift = 8.0 * rng.standard_normal(len(yaw))
    fit = calibrate.fit_lever(shift, yaw, n_boot=120)
    ok, why = calibrate.check_fit(fit)
    assert not ok, fit
    print('uncorrelated shift -> REJECTED: %s' % why[0])


def _two_camera_log(n=3000, lever=90.0, seed=2):
    """A log shaped like Hdf5Logger's: a ventral camera measuring yaw exactly
    with no displacement channels, and a frontal one measuring roll exactly."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 60.0
    yaw = np.radians(12.0) * np.sin(2 * np.pi * 0.3 * t)
    nan = np.full(n, np.nan)
    return {
        'merged/t_host': t,
        'bottom/head_yaw': yaw, 'bottom/head_roll': nan.copy(),
        'bottom/head_pitch': nan.copy(),
        'bottom/head_d_lateral': nan.copy(), 'bottom/head_d_axial': nan.copy(),
        'front/head_roll': np.radians(5.0) * np.sin(2 * np.pi * 0.17 * t),
        'front/head_yaw': nan.copy(), 'front/head_pitch': nan.copy(),
        'front/head_d_lateral': lever * np.sin(yaw) + 1.2 * rng.standard_normal(n),
        'front/head_d_axial': 3.0 * rng.standard_normal(n),
    }


def test_ventral_cannot_be_a_calibration_target():
    """A ventral camera has no lateral channel, so nothing can calibrate it -
    but it is still the reference that calibrates the frontal one."""
    log = _two_camera_log()
    shift, ref, name = calibrate.lever_inputs(log, 'front', reference='bottom')
    assert name == 'yaw' and np.isfinite(shift).any()
    fit = calibrate.fit_lever(shift, ref, n_boot=100)
    ok, why = calibrate.check_fit(fit)
    assert ok, why
    assert abs(fit['lever_px'] - 90.0) / 90.0 < 0.05, fit['lever_px']
    print('front calibrated by the bottom camera: %.1f px/rad (true 90.0)' % fit['lever_px'])

    try:
        calibrate.lever_inputs(log, 'bottom', reference='front')
    except ValueError as e:
        assert 'no lateral channel' in str(e), str(e)
        print('bottom refused as a target: %s...' % str(e)[:58])
    else:
        raise AssertionError('a ventral camera must not be calibratable')


def test_apply_lever_roundtrip():
    """The same constant converts both channels; small angles round-trip."""
    yaw = np.radians(np.linspace(-15, 15, 200))
    shift = 90.0 * np.sin(yaw)
    back = calibrate.apply_lever(shift, 90.0)
    assert np.allclose(back, yaw, atol=1e-9)
    pitch = calibrate.apply_lever(np.array([9.0, -9.0]), 90.0)
    assert abs(np.degrees(pitch[0]) - 5.74) < 0.05, np.degrees(pitch)
    print('apply_lever round-trips; 9 px at lever 90 = %.2f deg' % np.degrees(pitch[0]))


def test_epoch_mask():
    log = {'merged/t_host': np.linspace(0, 10, 601),
           'events/t_host': np.array([2.0, 4.0, 6.0, 8.0]),
           'events/text': ['stim_on cw', 'stim_off', 'stim_on ccw', 'stim_off']}
    m = calibrate.epochs_from_events(log, 'stim_on')
    t = log['merged/t_host']
    assert m[(t >= 2.0) & (t < 4.0)].all() and m[(t >= 6.0) & (t < 8.0)].all()
    assert not m[(t >= 4.0) & (t < 6.0)].any() and not m[t < 2.0].any()
    print('epoch mask selects %d of %d rows' % (m.sum(), len(m)))


if __name__ == '__main__':
    test_recovers_known_lever()
    test_rejects_outliers()
    test_refuses_when_yaw_barely_moves()
    test_refuses_uncorrelated()
    test_ventral_cannot_be_a_calibration_target()
    test_apply_lever_roundtrip()
    test_epoch_mask()
    print('OK')
