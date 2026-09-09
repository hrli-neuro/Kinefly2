"""Fit one fly's lever arm from that fly's own recording.

    python -m kinefly2.apps.calibrate_fly <recording_dir_or_h5>
    python -m kinefly2.apps.calibrate_fly <recording_dir> --write experiments/today.yaml

A frontal camera measures head ROLL exactly (an in-plane rotation) but sees
pitch and yaw only end-on, as a shift of the head in pixels. Converting those
to angles needs one constant per camera -- the depth of the landmark centroid
from the neck along that camera's optical axis -- and that constant changes with
head size, mounting and landmark placement, so it is per fly.

The reference is the OTHER camera: a ventral camera measures yaw exactly, which
is precisely what the frontal camera sees on its lateral channel. No stimulus
is needed; spontaneous head movement over a normal recording is usually plenty
(a test recording calibrated at r2 = 0.96 from 32 degrees of spontaneous yaw).
Run this on the fly's own recording rather than a separate calibration block:
that guarantees the constant belongs to the landmark set the data was taken
with, which a separate session cannot.

Two things are checked before a constant is handed back:

  * the statistical gates in analysis.calibrate.check_fit -- a lever arm fitted
    from a fly that barely moved is noise with units, and would silently
    rescale every angle derived from it;
  * the landmark positions stored in the recording against the camera config's
    current ones. Moving a landmark shifts the centroid and so the lever arm;
    a constant carried over from before the move is silently wrong.

The axis is fitted too. `axis_deg` decides which image direction counts as
pitch, and getting it wrong swaps pitch and yaw without any other symptom, so
it is estimated from the data (the direction that puts the reference rotation
on the lateral channel) and compared with the config.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

from ..analysis import calibrate
from ..tracking.tracker_landmarks import axis_vectors


def find_log(path):
    """Accept a recording directory or the HDF5 itself."""
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(path, '*_flystate.h5')))
    if not hits:
        raise SystemExit('no *_flystate.h5 in %s' % path)
    return hits[0]


def recorded_landmarks(log_path):
    """Landmark positions stamped into the recording, or {} for older files."""
    import h5py

    with h5py.File(log_path, 'r') as f:
        raw = f.attrs.get('landmarks')
    if raw is None:
        return {}
    return json.loads(raw if isinstance(raw, str) else raw.decode())


def config_landmarks(config_path, part='head'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    g = ((cfg.get('tracker') or {}).get('gui') or {}).get(part) or {}
    return [[float(p['x']), float(p['y'])] for p in (g.get('landmarks') or [])]


def landmarks_match(a, b, tol=0.5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and (a.size == 0 or np.abs(a - b).max() <= tol)


def fit_axis(log, camera, reference):
    """Image direction that puts the reference rotation on the lateral channel.

    Scans axis_deg and returns the angle maximising |corr(lateral, reference)|,
    with that correlation. The result is a line, not a direction: adding 180
    gives the same split with both signs flipped.
    """
    ref_name, ref = calibrate.exact_channel(log, reference)
    dx = np.asarray(log.get('%s/head_dx_corr' % camera, []), dtype=float)
    dy = np.asarray(log.get('%s/head_dy_corr' % camera, []), dtype=float)
    if ref is None or not dx.size:
        return None, np.nan
    d = np.column_stack([dx, dy])
    m = np.isfinite(ref) & np.isfinite(dx) & np.isfinite(dy)
    if m.sum() < 100:
        return None, np.nan
    best = (None, -1.0)
    for deg in np.arange(-180.0, 180.0, 1.0):
        _, lat = axis_vectors(deg)
        r = abs(np.corrcoef((d @ lat)[m], ref[m])[0, 1])
        if np.isfinite(r) and r > best[1]:
            best = (float(deg), float(r))
    return best


def axis_lines_agree(a, b, tol_deg=20.0):
    """Compare two axis angles as undirected lines (180 apart is the same)."""
    if a is None or b is None:
        return None
    d = abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)
    return d <= tol_deg, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording', help='Recording directory, or the *_flystate.h5 itself')
    ap.add_argument('--camera', default='front', help='Camera to calibrate (default: front)')
    ap.add_argument('--reference', default='bottom',
                    help='Camera whose exact in-plane angle is the reference (default: bottom)')
    ap.add_argument('--config', default=None,
                    help='Camera YAML to check landmarks against '
                         '(default: config/camera_<camera>.yaml)')
    ap.add_argument('--write', default=None,
                    help='Experiment YAML to write lever_px into')
    ap.add_argument('--epochs', default=None,
                    help='Restrict the fit to event epochs whose text contains this')
    ap.add_argument('--force', action='store_true',
                    help='Write even if the landmarks no longer match the recording')
    args = ap.parse_args()

    log_path = find_log(args.recording)
    print('recording: %s' % log_path)
    log = calibrate.load_log(log_path, cameras=(args.camera, args.reference))

    try:
        shift, ref, ref_name = calibrate.lever_inputs(log, args.camera, args.reference)
    except (ValueError, KeyError) as e:
        raise SystemExit('cannot calibrate: %s' % e)
    print('reference:  %s measures %s exactly' % (args.reference, ref_name.upper()))

    mask = calibrate.epochs_from_events(log, args.epochs) if args.epochs else None
    if mask is not None:
        print('restricted to %d frames in "%s" epochs' % (mask.sum(), args.epochs))

    fit = calibrate.fit_lever(shift, ref, mask=mask)
    ok, why = calibrate.check_fit(fit)
    print()
    print(calibrate.report(fit, ok, why))

    # Is the axis pointing the right way? A wrong axis swaps pitch and yaw and
    # has no other symptom, so check it rather than trusting it.
    cfg_path = args.config or os.path.join('config', 'camera_%s.yaml' % args.camera)
    best_deg, best_r = fit_axis(log, args.camera, args.reference)
    cfg_axis = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = yaml.safe_load(f) or {}
        cfg_axis = (((c.get('tracker') or {}).get('gui') or {}).get('head') or {}).get('axis_deg')
    print()
    if best_deg is None:
        print('axis: not enough data to check')
    else:
        print('axis: data prefers axis_deg %+.0f (|r| %.3f on the lateral channel)'
              % (best_deg, best_r))
        agree = axis_lines_agree(cfg_axis, best_deg)
        if cfg_axis is None:
            print('      %s sets no axis_deg' % cfg_path)
        elif agree[0]:
            print('      config has %+.1f, %.0f deg off that line - fine'
                  % (cfg_axis, agree[1]))
        else:
            print('      *** config has %+.1f, %.0f deg off that line. Pitch and yaw '
                  'are probably swapped; drag the axis arrow or set axis_deg %+.0f.'
                  % (cfg_axis, agree[1], best_deg))

    # Do the recording's landmarks still match the rig? A constant fitted for
    # one set of points does not apply to another.
    rec_lm = (recorded_landmarks(log_path).get(args.camera) or {}).get('head')
    stale = False
    if rec_lm is None:
        print('\nlandmarks: not stamped in this recording - cannot check staleness')
    elif not os.path.exists(cfg_path):
        print('\nlandmarks: %s not found - cannot check staleness' % cfg_path)
    else:
        now = config_landmarks(cfg_path)
        if landmarks_match(rec_lm, now):
            print('\nlandmarks: unchanged since the recording (%d points)' % len(now))
        else:
            stale = True
            print('\n*** landmarks: %s now has %d points; the recording had %d, in '
                  'different places.\n    A lever arm belongs to the points it was fitted '
                  'with -- moving one shifts the\n    centroid and so the arm -- meaning '
                  'this constant does not describe the setup\n    you have now. Either '
                  'calibrate from a recording made with the current\n    landmarks, or put '
                  'them back where they were.'
                  % (cfg_path, len(now), len(rec_lm)))

    if not args.write:
        return 0 if ok else 1
    if not ok:
        raise SystemExit('\nnot written: the fit did not pass its gates')
    if stale and not args.force:
        raise SystemExit('\nnot written: landmarks changed since the recording (--force to override)')

    with open(args.write) as f:
        text = f.read()
    doc = yaml.safe_load(text) or {}
    lever = dict(doc.get('lever_px') or {})
    lever[args.camera] = round(float(fit['lever_px']), 1)
    doc['lever_px'] = lever
    with open(args.write, 'w') as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print('\nwrote lever_px: {%s: %.1f} into %s' % (args.camera, lever[args.camera], args.write))
    return 0


if __name__ == '__main__':
    sys.exit(main())
