"""Post-hoc calibration of a camera's lever arm, turning its two displacement
channels into angles.

Why this exists
---------------
Each camera view measures ONE head rotation exactly -- the in-plane one, which
needs no rotation centre and no scale -- and sees the other two only end-on, as
a shift of the head:

    ventral camera (below):  yaw exact;  nothing else -- its lever arm (the
                             dorsoventral depth) is too short, and the other two
                             channels measured noise, so they are not reported
    frontal camera (front):  roll exact; pitch = axial shift, yaw = lateral

Both shifts scale by the SAME constant: the depth of the landmark centroid from
the joint along that camera's optical axis, in pixels. Put the joint at the
origin with the centroid at (lateral, depth, height); a small rotation about
either in-image axis moves the centroid by (angle x depth), and the other two
offsets cancel out. So one number converts both channels, and this module fits
it -- once per camera.

That depth points along the optical axis, so the view cannot measure it. The
reference has to come from outside:

  * the OTHER camera, whose exact in-plane angle is precisely the rotation this
    camera sees as its lateral channel (see lever_inputs);
  * an external ground truth (a rotation stage stepping through known angles),
    which is the only source that also checks the cameras themselves;
  * spontaneous movement pooled over many recordings, if it spans enough range.

Nothing here runs in real time. The pixel channels are what get logged, so the
constant stays revisable and can be re-applied to old recordings -- which is
the reason not to bake it into the recording. Once known, putting `lever_px:`
in the head's section of the camera YAML makes Kinefly2 log all three
rotations as angles and show degrees in the GUI.

Because each view measures a DIFFERENT rotation exactly, the two cameras
calibrate each other: the front camera's lateral channel is yaw, which the
bottom camera measures exactly, and the bottom camera's lateral channel is
roll, which the front camera measures exactly. Pitch then falls out of both
independently -- a genuine cross-check rather than an assumption.

Typical use, from a notebook:

    from kinefly2.analysis import calibrate
    log = calibrate.load_log('.../260819_..._flystate.h5')
    shift, ref, name = calibrate.lever_inputs(log, 'front', reference='bottom')
    fit = calibrate.fit_lever(shift, ref)
    print(calibrate.report(fit))
"""
import math

import numpy as np


def load_log(path, cameras=('bottom', 'front')):
    """Read an Hdf5Logger file into a flat dict of arrays.

    Returns every '<camera>/<column>' dataset plus 'merged/*', and the event
    table under 'events/*'. Angles stay in radians, as stored.

    'attrs' holds the file attributes -- the code version, the measurement
    conventions, the lever arm and landmark positions in force when it was
    recorded. Worth reading before comparing two files: the wing zero changed
    on 2026-09-10 without any config changing with it.
    """
    import json

    import h5py

    out = {'attrs': {}}
    with h5py.File(path, 'r') as f:
        for key, value in f.attrs.items():
            try:
                out['attrs'][key] = json.loads(value)
            except (ValueError, TypeError):
                out['attrs'][key] = value
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


def exact_channel(log, camera):
    """Which rotation this camera measures exactly, and its values (rad).

    Each view measures one rotation as a true in-plane angle -- yaw for a
    ventral camera, roll for a frontal one -- and that is the only anatomical
    column that is finite before calibration."""
    for name in ('roll', 'pitch', 'yaw'):
        v = log.get('%s/head_%s' % (camera, name))
        if v is not None and np.isfinite(np.asarray(v, dtype=float)).any():
            return name, np.asarray(v, dtype=float)
    return None, None


def lever_inputs(log, camera, reference):
    """Inputs for fitting `camera`'s lever arm: (shift_px, reference_rad, name).

    The two cameras calibrate each other. A camera's LATERAL channel carries
    exactly the rotation the other camera measures as its in-plane angle --
    a frontal camera's lateral channel is yaw, which a ventral camera measures
    exactly, and vice versa. So the pair matches up on its own, and this
    refuses the combination where it does not (two cameras of the same view
    cannot calibrate one another).
    """
    ref_name, ref = exact_channel(log, reference)
    own_name, _ = exact_channel(log, camera)
    lateral = np.asarray(log.get('%s/head_d_lateral' % camera, []), dtype=float)
    if lateral.size and not np.isfinite(lateral).any():
        raise ValueError(
            '%s has no lateral channel to calibrate: a ventral camera reports '
            'only its exact in-plane angle, because its lever arm is too short '
            'to measure the other two rotations. It is still a valid reference '
            'for calibrating a frontal camera.' % camera)
    if ref_name is None:
        raise ValueError('%s has no exact angle: is it running a landmarks '
                         'tracker with a view set?' % reference)
    if ref_name == own_name:
        raise ValueError('%s and %s measure the same rotation (%s) exactly, so '
                         'neither can calibrate the other. Use cameras with '
                         'different views.' % (camera, reference, ref_name))
    return lateral, ref, ref_name


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
    """Fit shift_px = lever_px * sin(reference) + offset.

    `shift_px` is the camera's lateral channel (head_d_lateral) and `yaw_rad`
    the reference angle in radians, sampled on the same rows; use
    lever_inputs() to pair them up. sin() rather than the angle itself because
    the projected shift of a point rotating about the joint is what the camera
    sees; the two agree to 2% out to 20 degrees.

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


def provenance_lines(log):
    """What produced this recording, for the head of a report. Older files
    predate the stamping and say so rather than guessing."""
    attrs = (log or {}).get('attrs') or {}
    version = attrs.get('code_version')
    if not version:
        return ['recorded by:  unknown (this file predates provenance stamping)']
    lines = ['recorded by:  Kinefly2 %s' % version]
    conv = attrs.get('conventions') or {}
    if conv:
        lines.append('conventions:  %s' % ', '.join('%s=%s' % kv for kv in sorted(conv.items())))
    return lines


def report(fit, ok=None, why=None, log=None):
    """Human-readable summary, including the line to paste into the config."""
    if ok is None:
        ok, why = check_fit(fit)
    lo, hi = fit['ci']
    lines = []
    if log is not None:
        lines += provenance_lines(log) + ['']
    lines += [
        'lever_px    = %.1f px/rad  (95%% CI %.1f .. %.1f)' % (fit['lever_px'], lo, hi),
        '            = %.2f px per degree' % (fit['lever_px'] * math.pi / 180),
        'r2          = %.3f   over n = %d frames' % (fit['r2'], fit['n']),
        'yaw range   = %.2f deg' % fit['yaw_range_deg'],
    ]
    if np.isfinite(fit['lever_px']) and fit['lever_px'] > 0:
        lines.append('resolution  = 1 px of shift is %.2f deg of yaw or pitch'
                     % (180 / math.pi / fit['lever_px']))
    if ok:
        lines += ['', 'ACCEPTED. This constant belongs to THIS fly. Put it in the',
                  'experiment YAML (not the camera config, where the next fly would',
                  'inherit it):', '',
                  '    lever_px:',
                  '      front: %.1f' % fit['lever_px'], '',
                  'calibrate_fly --write <experiment.yaml> does it for you. Both',
                  'displacement channels then become angles through this one number.']
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
