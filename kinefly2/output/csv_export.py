"""Flat CSV copies of a fly-state HDF5, for a quick look in a spreadsheet,
pandas or MATLAB (readtable).

One row per merged frame. Per camera (`group='bottom'`), the columns are that
camera's datasets without a prefix -- seq, t_camera, t_host, ..., left_angle,
head_yaw, ... -- which is what an offline analysis writes as <run>_<cam>.csv.
The rows of two such files from one pair analysis line up: row N of each is
frame N (the cameras ran in lockstep). Without a group, every per-frame
dataset is written, named <group>_<column> (merged_seq, bottom_left_angle, ...).

Values are exactly as stored: angles in RADIANS, positions in pixels, times in
seconds, NaN where a camera or part had no value. Events are not per-frame and
stay in the HDF5 (/events).
"""
import csv
import math

import numpy as np

from ..analysis import calibrate

# Columns that identify the row come first in each group, the rest follow in
# the file's order.
LEADING = ('seq', 't_camera', 't_host', 't_wall', 't_done', 'age', 'stim', 'stim_t_wall')


def _cell(v):
    if not math.isfinite(v):
        return 'NaN'
    if v == int(v) and abs(v) < 2 ** 53:
        return str(int(v))
    return repr(float(v))       # shortest exact form: no digits lost on t_wall


def h5_to_csv(h5_path, csv_path, group=None, exclusive=True):
    """Write the CSV -- one camera's columns with `group`, else everything.
    Returns (rows, columns). `exclusive` refuses to replace an existing file."""
    log = calibrate.load_log(h5_path)
    n = len(log.get('merged/seq', []))
    if group is not None and group not in (log.get('cameras') or []):
        raise ValueError('no camera %r in %s (it has %s)'
                         % (group, h5_path, ', '.join(log.get('cameras') or []) or 'none'))
    groups = [group] if group is not None else ['merged'] + list(log.get('cameras') or [])
    names, arrays = [], []
    for g in groups:
        cols = [k.split('/', 1)[1] for k in log
                if isinstance(k, str) and k.startswith(g + '/')]
        cols = [c for c in LEADING if c in cols] + [c for c in cols if c not in LEADING]
        for col in cols:
            a = np.asarray(log['%s/%s' % (g, col)])
            if a.ndim != 1 or len(a) != n or a.dtype.kind not in 'fiub':
                continue
            names.append(col if group is not None else '%s_%s' % (g, col))
            arrays.append(a.astype(float))
    with open(csv_path, 'x' if exclusive else 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            w.writerow([_cell(a[i]) for a in arrays])
    return n, len(names)
