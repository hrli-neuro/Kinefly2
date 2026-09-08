"""Post-hoc calibration of the front camera's yaw/pitch lever arm.

Why this exists
---------------
Seen from the front, the head's roll is an in-plane rotation and is measured
exactly -- it needs no rotation centre and no scale. The other two rotations are
seen end-on and appear only as a shift of the head:

    head_dx_corr  <- YAW    (rotation about the dorsoventral axis)
    head_dy_corr  <- PITCH  (rotation about the lateral axis)

Both scale by the SAME constant: the head's anterior-posterior depth from the
neck joint, in pixels. Put the neck at the origin and the landmark centroid at
(lateral, depth, height); a small yaw moves the centroid horizontally by
-yaw*depth and a small pitch moves it vertically by +pitch*depth. The lateral
and height offsets do not enter either one. So one number converts both
channels, and this module fits it.

That depth points along the front camera's optical axis, so a frontal view
cannot measure it. The reference has to come from outside:

  * the bottom camera's head angle -- yaw is an in-plane rotation there, so it
    is already a true angle and needs no lever arm of its own;
  * an external ground truth (a rotation stage stepping through known angles),
    which is the only source that also checks the bottom camera itself;
  * spontaneous yaw pooled over many recordings, if it spans enough range.

Nothing here runs in real time. The pixel channels are what get logged, so the
constant stays revisable and can be re-applied to old recordings -- which is
the reason not to bake it into the recording. Once it is known, putting
`lever_px:` in the head's section of the camera YAML makes Kinefly2 log
head_yaw / head_pitch and show degrees in the GUI.

Typical use, from a notebook:

    from kinefly2.analysis import calibrate
    log = calibrate.load_log('.../260819_..._flystate.h5')
    fit = calibrate.fit_lever(log['front/head_dx_corr'], log['bottom/head_angle'])
    ok, why = calibrate.check_fit(fit)
    print(calibrate.report(fit, ok, why))
"""
import math

import numpy as np


def load_log(path, cameras=('bottom', 'front')):
    """Read an Hdf5Logger file into a flat dict of arrays.

    Returns every '<camera>/<column>' dataset plus 'merged/*', and the event
    table under 'events/*'. Angles stay in radians, as stored.
    """
    import h5py

    out = {}
    with h5py.File(path, 'r') as f:
        for group in list(cameras) + ['merged']:
            if group not in f:
                continue
            for col in f[group]:
                out['%s/%s' % (group, col)] = f['%s/%s' % (group, col)][:]
        if 'events' in f:
            out['events/t_host'] = f['events/t_host'][:]
            out['events/text'] = [t.decode() if isinstance(t, bytes) else t
                                  for t in f['events/text'][:]]
    return out


def epochs_from_events(log, substring, duration=None):
    """Boolean mask over log rows selecting epochs that follow an event whose
    text contains `substring`, for restricting a fit to stimulus trials.

    Each matching event opens an epoch that runs until the next event (or for
    `duration` seconds, if given).
    """
    t = np.asarray(log['merged/t_host'], dtype=float)
    mask = np.zeros(len(t), dtype=bool)
    ev_t = np.asarray(log.get('events/t_host', []), dtype=float)
    ev_x = list(log.get('events/text', []))
    for i, (te, text) in enumerate(zip(ev_t, ev_x)):
        if substring not in text:
            continue
        end = te + duration if duration is not None else (
            ev_t[i + 1] if i + 1 < len(ev_t) else t[-1] if len(t) else te)
        mask |= (t >= te) & (t < end)
    return mask


def _huber_lstsq(X, y, delta=1.5, iters=25):
    """Huber-weighted least squares: outlying frames (a lost landmark, a leg
    across the head) get down-weighted instead of dragging the slope."""
    w = np.ones(len(y))
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        Xw = X * w[:, None]
        beta, *_ = np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        a = np.abs(r) / (delta * s)
        w = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-9))
    return beta, w


def fit_lever(shift_px, yaw_rad, mask=None, n_boot=400, seed=0):
    """Fit shift_px = lever_px * sin(yaw) + offset.

    `shift_px` is the front camera's head_dx_corr and `yaw_rad` the reference
    yaw angle (radians), sampled on the same rows. sin() rather than the angle
    itself because the projected shift of a point rotating about the neck is
    what the camera sees; the two agree to 2% out to 20 degrees.

    Returns a dict with lever_px, offset_px, r2, n, yaw_range_deg, and a 95%
    bootstrap CI on lever_px. Fitting is robust (Huber), so a few bad frames
    do not set the scale for everything.
    """
    x = np.asarray(shift_px, dtype=float)
    y = np.asarray(yaw_rad, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    x, y = x[good], y[good]
    if len(x) < 100:
        return {'lever_px': math.nan, 'offset_px': math.nan, 'r2': math.nan,
                'n': int(len(x)), 'yaw_range_deg': 0.0, 'ci': (math.nan, math.nan)}

    y = y - np.median(y)                     # the reference has an arbitrary zero
    s = np.sin(y)
    X = np.column_stack([s, np.ones_like(s)])
    beta, w = _huber_lstsq(X, x)
    pred = X @ beta
    ss_res = float((w * (x - pred) ** 2).sum())
    ss_tot = float((w * (x - np.average(x, weights=w)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        b, _ = _huber_lstsq(X[idx], x[idx], iters=8)
        boot[i] = b[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {'lever_px': float(beta[0]), 'offset_px': float(beta[1]), 'r2': float(r2),
            'n': int(len(x)),
            'yaw_range_deg': float(np.degrees(np.percentile(y, 99) - np.percentile(y, 1))),
            'ci': (float(lo), float(hi))}


def check_fit(fit, min_range_deg=5.0, min_r2=0.5, max_ci_frac=0.20, min_n=1000):
    """Decide whether a fit may be adopted as the calibration constant.

    A lever arm fitted from a recording where the fly barely moved its head is
    not a measurement -- it is noise with units, and it would silently rescale
    every yaw and pitch number derived from it afterwards. Returns
    (ok, reasons); `reasons` lists every gate that failed.
    """
    why = []
    if not np.isfinite(fit.get('lever_px', math.nan)):
        return False, ['no usable frames']
    if fit['n'] < min_n:
        why.append('only %d frames (need %d)' % (fit['n'], min_n))
    if fit['yaw_range_deg'] < min_range_deg:
        why.append('reference yaw spans only %.2f deg (need %.1f): too little '
                   'head movement to set a scale'
                   % (fit['yaw_range_deg'], min_range_deg))
    if not np.isfinite(fit['r2']) or fit['r2'] < min_r2:
        why.append('r2 = %.3f (need %.2f): the shift does not follow the '
                   'reference yaw' % (fit['r2'], min_r2))
    lo, hi = fit['ci']
    if np.isfinite(lo) and abs(fit['lever_px']) > 0:
        frac = (hi - lo) / abs(fit['lever_px'])
        if frac > max_ci_frac:
            why.append('95%% CI is %.0f%% of the estimate (need <%.0f%%)'
                       % (100 * frac, 100 * max_ci_frac))
    if fit['lever_px'] <= 0:
        why.append('lever_px = %.1f is not positive; check the sign convention '
                   'of the reference yaw' % fit['lever_px'])
    return (not why), why


def report(fit, ok=None, why=None):
    """Human-readable summary, including the line to paste into the config."""
    if ok is None:
        ok, why = check_fit(fit)
    lo, hi = fit['ci']
    lines = [
        'lever_px    = %.1f px/rad  (95%% CI %.1f .. %.1f)' % (fit['lever_px'], lo, hi),
        '            = %.2f px per degree' % (fit['lever_px'] * math.pi / 180),
        'r2          = %.3f   over n = %d frames' % (fit['r2'], fit['n']),
        'yaw range   = %.2f deg' % fit['yaw_range_deg'],
    ]
    if np.isfinite(fit['lever_px']) and fit['lever_px'] > 0:
        lines.append('resolution  = 1 px of shift is %.2f deg of yaw or pitch'
                     % (180 / math.pi / fit['lever_px']))
    if ok:
        lines += ['', 'ACCEPTED. Put this in the head: section of the front camera YAML:',
                  '    lever_px: %.1f' % fit['lever_px'],
                  'Then head_yaw and head_pitch are logged in radians and the GUI',
                  'shows degrees. dy_corr uses the same constant as dx_corr.']
    else:
        lines += ['', 'REJECTED -- do not adopt this constant:']
        lines += ['  * ' + w for w in why]
        lines += ['The pixel channels remain valid and are unaffected; only the',
                  'conversion to angles is withheld.']
    return '\n'.join(lines)


def apply_lever(shift_px, lever_px):
    """Convert a displacement channel to an angle in radians.

    Works on either channel: dx_corr gives yaw, dy_corr gives pitch.
    """
    x = np.asarray(shift_px, dtype=float)
    return np.arcsin(np.clip(x / float(lever_px), -1.0, 1.0))
