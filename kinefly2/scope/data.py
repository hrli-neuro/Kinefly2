"""The traces, wherever they come from: the channel groups a viewer draws, a
saved file read into columns, and the live stream buffered as columns.

No Qt here -- the browser dashboard uses this module too -- and no OpenCV: on
Linux the OpenCV wheels bring their own Qt, which must never share a process
with the Scope's.

Every source ends in one shape: a time axis (the primary camera's frame clock,
seconds) and display channels keyed like the live stream -- 'bottom/left',
'front/head_roll', 'stim', ... -- with angles in DEGREES and NaN (None in a live
row) where there is no value. A saved file and the live stream must produce the
same keys and values, or every feature of a viewer has to be written twice.
"""
import csv
import json
import math
import os
import threading
from dataclasses import dataclass, field

import numpy as np

PARTS = ('left', 'right', 'head', 'abdomen')   # drawn first; extras follow in order
ROW_META = ('t', 't_host', 'seq', 'rec', 'stale')   # per-row fields that are not channels


def clean(x):
    """JSON has no NaN or Infinity. Python's json emits bare `NaN`, which
    JSON.parse rejects outright -- one uncalibrated channel would kill the
    whole stream -- so non-finite values become null and are drawn as gaps."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def axis_time(state):
    """The display timestamp for one published state.

    The primary camera's frame clock, which is the only clock that means the
    same thing live and in replay. A publisher that does not say which camera
    is primary (an older run_rig) falls back to the host clock."""
    primary = state.get('primary')
    cams = state.get('cameras') or {}
    cs = cams.get(primary) if primary else None
    tc = (cs or {}).get('t_camera')
    if isinstance(tc, (int, float)) and math.isfinite(tc):
        return float(tc)
    return float(state['t_host'])


def _wbf(freq):
    """Wingbeat frequency for display: 0 Hz is the detector saying it found
    none (and what an untracked aux part reports), so it is a gap, not 0."""
    return freq if isinstance(freq, (int, float)) and freq and math.isfinite(freq) else None


def row_from_state(state):
    """Flatten one published state into the channels a viewer plots."""
    row = {'t': axis_time(state), 't_host': float(state['t_host']),
           'seq': state['seq'], 'stim': state.get('stim'),
           'rec': state.get('rec')}
    # How old each camera's contribution was at the merge, in ms, and which
    # cameras were too old to trust. A stale camera's measurements arrive
    # here already blanked, so this is what says the gap is a stalled camera
    # rather than a fly that stopped moving.
    ages = state.get('ages') or {}
    limit = state.get('stale_after_s')
    stale = [c for c, a in sorted(ages.items())
             if a is not None and limit is not None and a > limit]
    row['stale'] = ','.join(stale) if stale else None
    for cam, age in ages.items():
        row['%s/lag_ms' % cam] = None if age is None else age * 1e3
    for cam, cs in (state.get('cameras') or {}).items():
        parts = cs.get('parts') or {}
        # Whatever the camera actually tracks, not a fixed list: a ventral
        # camera may declare extra parts (hind legs) that a frontal one does not.
        names = list(PARTS) + [p for p in parts if p not in PARTS and p != 'aux']
        for part in names:
            ps = parts.get(part) or {}
            a = (ps.get('angles') or [None])[0]
            row['%s/%s' % (cam, part)] = None if a is None else math.degrees(a)
            # Edge trackers report how many edges their annulus found -- the
            # check that says whether an angle is trustworthy at all. Emitted
            # for every part, null where the tracker has no such notion, so a
            # live row and a row read back from a file carry the same keys.
            row['%s/%s_n_edges' % (cam, part)] = (ps.get('extra') or {}).get('n_edges')
        head = parts.get('head') or {}
        extra = head.get('extra') or {}
        # The anatomical channels, named by the tracker itself. Exactly one is
        # the exact in-plane angle; a frontal camera fills the other two once
        # its lever arm is calibrated, a ventral one never does.
        for name in ('roll', 'pitch', 'yaw'):
            v = extra.get(name)
            row['%s/head_%s' % (cam, name)] = None if v is None else math.degrees(v)
        for name in ('d_axial', 'd_lateral'):        # uncalibrated fallbacks, px
            row['%s/head_%s' % (cam, name)] = extra.get(name)
        row['%s/head_q' % cam] = head.get('intensity')
        row['%s/wbf' % cam] = _wbf((parts.get('aux') or {}).get('freq'))
    return clean(row)


# ------------------------------------------------------------------- channels

@dataclass(frozen=True)
class Trace:
    key: str
    label: str
    color: str
    alt: str = None          # drawn instead when `key` has no data (uncalibrated,
    alt_unit: str = None     # or a head tracked by a hinge tracker, not landmarks)


@dataclass(frozen=True)
class Group:
    """One panel: its traces share an axis and a unit."""
    id: str
    title: str
    unit: str
    traces: tuple = ()
    range: tuple = (-1.0, 1.0)
    hidden: bool = False     # starts hidden: a diagnostic, not a measurement
    match: str = None        # instead of fixed traces, one per key ending in this


GROUPS = (
    Group('wsa', 'Wing stroke amplitude', 'deg',
          (Trace('bottom/left', 'L', '#ee5555'), Trace('bottom/right', 'R', '#55bbee')), (0, 180)),
    Group('lmr', 'L − R', 'deg', (Trace('L-R', 'L−R', '#ffd84a'),), (-60, 60)),
    Group('lpr', 'L + R', 'deg', (Trace('L+R', 'L+R', '#ffaa77'),), (0, 360), hidden=True),
    Group('abd', 'Abdomen bending', 'deg', (Trace('bottom/abdomen', 'abd', '#ff8888'),), (-45, 45)),
    Group('legs', 'Hind legs', 'deg',
          (Trace('bottom/leg_l3', 'L3', '#6aa8ff'), Trace('bottom/leg_r3', 'R3', '#a8dcff')),
          (-120, 120)),
    Group('byaw', 'Head yaw (bottom, exact)', 'deg',
          (Trace('bottom/head_yaw', 'yaw', '#7fe07f', alt='bottom/head', alt_unit='deg'),),
          (-30, 30)),
    Group('froll', 'Head roll (front, exact)', 'deg',
          (Trace('front/head_roll', 'roll', '#c890ff', alt='front/head', alt_unit='deg'),),
          (-30, 30)),
    Group('fpitch', 'Head pitch (front)', 'deg',
          (Trace('front/head_pitch', 'pitch', '#80c8ff', alt='front/head_d_axial', alt_unit='px'),),
          (-20, 20)),
    Group('fyaw', 'Head yaw (front)', 'deg',
          (Trace('front/head_yaw', 'yaw', '#4fc79a', alt='front/head_d_lateral', alt_unit='px'),),
          (-30, 30)),
    Group('stim', 'Stimulus', 'V', (Trace('stim', 'stim', '#ffa060'),), (-1, 11)),
    Group('wbf', 'Wingbeat frequency', 'Hz', (Trace('bottom/wbf', 'wbf', '#ff9966'),), (150, 250)),
    Group('q', 'Landmark match quality', '', (), (0, 1), match='/head_q'),
    Group('edges', 'Edges found per annulus', '', (), (0, 4), hidden=True, match='_n_edges'),
    Group('lag', 'Camera lag', 'ms', (), (0, 60), hidden=True, match='/lag_ms'),
)
PALETTE = ('#7fb27f', '#b27fb2', '#e0b060', '#60b0e0', '#e07070', '#a0a0ff', '#c0c060')
DERIVED = {'L-R': ('bottom/left', 'bottom/right', np.subtract),
           'L+R': ('bottom/left', 'bottom/right', np.add)}


def series(columns, key):
    """The values of one key: a column, or one derived from two."""
    if key in DERIVED:
        a, b, op = DERIVED[key]
        if a in columns and b in columns:
            return op(columns[a], columns[b])
        return None
    return columns.get(key)


def has_data(a):
    return a is not None and len(a) > 0 and bool(np.isfinite(a).any())


def unit_of(key):
    """The unit of a channel no group names, from its key. Only camera
    channels have known units; a generic CSV column ('csv/...') has none."""
    cam, _, tail = key.partition('/')
    if not tail or cam == 'csv':
        return ''
    if tail.startswith('head_d_'):
        return 'px'
    if tail == 'lag_ms':
        return 'ms'
    if tail == 'wbf':
        return 'Hz'
    if tail == 'head_q' or tail.endswith('_n_edges'):
        return ''
    return 'deg'            # a part's angle, or one of the head's rotations


def resolve(group, columns):
    """The traces a group draws from `columns`: [(label, color, key, unit)].
    A fixed trace whose key has no data falls back to its `alt`; a `match`
    group has one trace per matching key that carries data."""
    out = []
    if group.match:
        keys = sorted(k for k in columns if k.endswith(group.match) and has_data(columns[k]))
        for i, k in enumerate(keys):
            label = k[:-len(group.match)].replace('/', ' ')     # 'bottom', 'bottom left'
            out.append((label or k, PALETTE[i % len(PALETTE)], k, group.unit))
        return out
    for tr in group.traces:
        if has_data(series(columns, tr.key)):
            out.append((tr.label, tr.color, tr.key, group.unit))
        elif tr.alt and has_data(series(columns, tr.alt)):
            out.append((tr.label, tr.color, tr.alt, tr.alt_unit or group.unit))
    return out


def claimed_keys(columns):
    """Keys some group draws (or could): everything else becomes an extra panel."""
    keys = set()
    for g in GROUPS:
        if g.match:
            keys.update(k for k in columns if k.endswith(g.match))
        for tr in g.traces:
            keys.update(DERIVED[tr.key][:2] if tr.key in DERIVED else (tr.key,))
            if tr.alt:
                keys.add(tr.alt)
    return keys


def extra_groups(columns):
    """A panel for every channel with data that no group claims: another
    camera's wings, a part only this rig has, a column of a generic CSV."""
    claimed = claimed_keys(columns)
    out = []
    for i, k in enumerate(sorted(columns)):
        if k in claimed or not has_data(columns[k]):
            continue
        a = columns[k][np.isfinite(columns[k])]
        lo, hi = float(a.min()), float(a.max())
        pad = (hi - lo) * 0.1 or abs(hi) * 0.1 or 1.0
        out.append(Group('x:' + k, k, unit_of(k), (Trace(k, k.split('/')[-1],
                                                         PALETTE[i % len(PALETTE)]),),
                         (lo - pad, hi + pad)))
    return out


# ------------------------------------------------------------------ recordings

@dataclass
class Recording:
    """A saved recording read into display columns."""
    name: str
    path: str
    t: np.ndarray                   # display axis: the primary camera's frame clock, s
    t_host: np.ndarray
    seq: np.ndarray
    columns: dict                   # key -> float array (degrees for angles), NaN = gap
    stale: list = field(default_factory=list)    # per row: stale camera names or None
    events: list = field(default_factory=list)   # [{'t': s, 'text': str}]
    info: dict = field(default_factory=dict)


def run_name(path):
    name = os.path.basename(path)
    for suffix in ('_flystate.h5', '.h5', '.csv'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def columns_from_log(log, path, name=None):
    """Turn a load_log()-shaped dict ('<group>/<column>' arrays, 'cameras',
    'attrs', 'events/*') into a Recording. The HDF5 names a few channels
    differently from the live stream -- `head_q` is stored as `head_score`,
    `wbf` as `aux_freq` -- and angles are radians on disk, degrees here."""
    attrs = log.get('attrs') or {}
    rig = attrs.get('rig') or {}
    cams = list(log.get('cameras') or [])
    primary = rig.get('primary')
    primary_known = primary in cams
    if not primary_known:
        # Written before the `rig` attribute existed. Guessing the primary is
        # usually right on a two-camera rig, but it IS a guess, and it decides
        # whose frame clock the x-axis is -- so say so rather than imply the
        # file stated it.
        primary = cams[0] if cams else None
    stale_after_s = rig.get('stale_after_s')

    def col(key):
        a = log.get(key)
        return None if a is None else np.asarray(a, dtype=float)

    # A per-camera CSV has no merged group: that camera's host clock stands in.
    t_host = col('merged/t_host')
    if t_host is None and primary is not None:
        t_host = col('%s/t_host' % primary)
    t_host = np.zeros(0) if t_host is None else t_host
    n = int(t_host.size)

    # The display axis: the primary camera's frame clock. A camera can be
    # missing from individual rows, so bridge its gaps.
    axis = None
    if primary is not None:
        tc = col('%s/t_camera' % primary)
        if tc is not None and tc.size == n:
            ok = np.isfinite(tc)
            if ok.sum() >= 2:
                idx = np.arange(n)
                axis = np.interp(idx, idx[ok], tc[ok]) if not ok.all() else tc
    if axis is None:
        axis = t_host

    columns = {}

    def put(key, name, scale=None):
        a = col(name)
        if a is None or a.size != n:
            return
        if scale == 'deg':
            a = np.degrees(a)
        elif scale is not None:
            a = a * scale
        columns[key] = a

    put('stim', 'merged/stim')
    for cam in cams:
        # Column names in the file are the authority on what this camera tracked.
        extra = sorted({k.split('/', 1)[1][:-6] for k in log
                        if isinstance(k, str) and k.startswith(cam + '/') and k.endswith('_angle')}
                       - set(PARTS))
        for part in list(PARTS) + extra:
            put('%s/%s' % (cam, part), '%s/%s_angle' % (cam, part), 'deg')
            put('%s/%s_n_edges' % (cam, part), '%s/%s_n_edges' % (cam, part))
        for nm in ('roll', 'pitch', 'yaw'):
            put('%s/head_%s' % (cam, nm), '%s/head_%s' % (cam, nm), 'deg')
        for nm in ('d_axial', 'd_lateral'):
            put('%s/head_%s' % (cam, nm), '%s/head_%s' % (cam, nm))
        put('%s/head_q' % cam, '%s/head_score' % cam)
        put('%s/wbf' % cam, '%s/aux_freq' % cam)
        if '%s/wbf' % cam in columns:
            w = columns['%s/wbf' % cam]
            columns['%s/wbf' % cam] = np.where(w == 0, np.nan, w)
        put('%s/lag_ms' % cam, '%s/age' % cam, 1e3)

    # Which cameras were blanked as stale on each row. The threshold is not
    # recoverable from the columns alone, which is why it is stamped in `rig`.
    stale = [None] * n
    if stale_after_s:
        flags = {}
        for cam in cams:
            a = col('%s/age' % cam)
            if a is not None and a.size == n:
                flags[cam] = np.isfinite(a) & (a > stale_after_s)
        if flags:
            for i in np.nonzero(np.any(np.vstack(list(flags.values())), axis=0))[0]:
                stale[int(i)] = ','.join(c for c in cams if c in flags and flags[c][i])

    seq = col('merged/seq')
    if seq is None and primary is not None:
        seq = col('%s/seq' % primary)
    seq = np.zeros(n) if seq is None or seq.size != n else seq

    # Events are stamped on the host clock: map them onto the display axis
    # through the rows themselves.
    events = []
    ev_t = col('events/t_host')
    if ev_t is not None and ev_t.size and n > 1:
        mapped = np.interp(ev_t, t_host, axis)
        events = [{'t': float(a), 'text': str(x)} for a, x in zip(mapped, log.get('events/text', []))]

    name = name or run_name(path)
    info = {'name': name, 'path': os.path.abspath(path), 'primary': primary,
            'primary_known': primary_known, 'cameras': cams,
            'stale_after_s': stale_after_s, 'n': n,
            'span': float(axis[-1] - axis[0]) if n else 0.0,
            'code_version': attrs.get('code_version'),
            'conventions': attrs.get('conventions'),
            'lever_px': attrs.get('lever_px'),
            'source': attrs.get('source')}
    return Recording(name=name, path=os.path.abspath(path), t=axis, t_host=t_host, seq=seq,
                     columns=columns, stale=stale, events=events, info=info)


def _number(s):
    try:
        return float(s)
    except ValueError:
        return math.nan


def log_from_csv(path):
    """Read one of our CSV exports into the load_log() shape, or None when it
    is some other CSV. Two kinds: the per-camera <run>_<cam>.csv (unprefixed
    columns; the camera is the last _token of the file name) and the combined
    export (merged_*, <cam>_* columns; camera names without underscores)."""
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    header, body = rows[0], rows[1:]
    data = np.array([[_number(x) for x in r] for r in body], dtype=float) \
        if body else np.zeros((0, len(header)))
    cols = {h: data[:, i] for i, h in enumerate(header) if i < data.shape[1]}
    if any(h.startswith('merged_') for h in header):
        log = {'attrs': {}}
        cams = []
        for h, a in cols.items():
            group, _, rest = h.partition('_')
            log['%s/%s' % (group, rest)] = a
            if group != 'merged' and group not in cams:
                cams.append(group)
        log['cameras'] = cams
        return log
    if 't_camera' in header and 'seq' in header:
        cam = run_name(path).rsplit('_', 1)[-1]
        log = {'attrs': {'rig': {'primary': cam, 'cameras': [cam]}}, 'cameras': [cam]}
        for h, a in cols.items():
            log['%s/%s' % (cam, h)] = a
        return log
    return None


TIME_COLUMNS = ('t', 'time', 'time_s', 't_camera', 'seconds')


def generic_csv(path):
    """Any other CSV: numeric columns against the first time-like column (or
    the row number), drawn as they are -- no unit conversion is guessed."""
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    if not rows or len(rows) < 2:
        raise ValueError('%s has no data rows' % path)
    header, body = rows[0], rows[1:]
    data = np.array([[_number(x) for x in r[:len(header)]] + [math.nan] * (len(header) - len(r))
                     for r in body], dtype=float)
    tcol = next((i for i, h in enumerate(header) if h.strip().lower() in TIME_COLUMNS), None)
    t = data[:, tcol] if tcol is not None else np.arange(len(body), dtype=float)
    columns = {'csv/%s' % h.strip(): data[:, i] for i, h in enumerate(header)
               if i != tcol and np.isfinite(data[:, i]).any()}
    name = run_name(path)
    info = {'name': name, 'path': os.path.abspath(path), 'primary': None,
            'primary_known': False, 'cameras': [], 'n': len(t),
            'span': float(np.nanmax(t) - np.nanmin(t)) if len(t) else 0.0,
            'time_column': header[tcol] if tcol is not None else None}
    return Recording(name=name, path=os.path.abspath(path), t=t, t_host=t.copy(),
                     seq=np.arange(len(t), dtype=float), columns=columns,
                     stale=[None] * len(t), info=info)


def load_recording(path):
    """A saved recording: an HDF5 fly state (a live recording or an offline
    analysis), one of our CSV exports, or any numeric CSV."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.h5', '.hdf5'):
        from ..analysis import calibrate
        return columns_from_log(calibrate.load_log(path), path)
    if ext == '.csv':
        log = log_from_csv(path)
        return columns_from_log(log, path) if log is not None else generic_csv(path)
    raise ValueError('cannot read %s: expected an .h5 or .csv file' % path)


# ------------------------------------------------------------------------ live

class ColumnRing:
    """Rows appended as columns into fixed-size numpy rings, so drawing slices
    arrays instead of walking thousands of dicts."""

    def __init__(self, capacity):
        self.cap = int(capacity)
        self.clear()

    def clear(self):
        self.n = 0
        self.t = np.full(self.cap, np.nan)
        self.cols = {}

    def __len__(self):
        return min(self.n, self.cap)

    def append(self, row):
        i = self.n % self.cap
        self.t[i] = row['t']
        for a in self.cols.values():
            a[i] = np.nan
        for k, v in row.items():
            if k in ROW_META or v is None or isinstance(v, str):
                continue
            a = self.cols.get(k)
            if a is None:
                a = self.cols[k] = np.full(self.cap, np.nan)
            a[i] = v
        self.n += 1

    def _runs(self):
        """The held rows as (start, stop) slices of the ring, oldest first:
        one run until the ring wraps, two after. Each is sorted by time."""
        if self.n <= self.cap:
            return [(0, self.n)]
        s = self.n % self.cap
        return [(s, self.cap), (0, s)] if s else [(0, self.cap)]

    def window(self, keys, lo=-np.inf, hi=np.inf):
        """(t, {key: values}) for the rows with lo <= t <= hi, oldest first.
        A binary search per run, so the cost is the window, not the ring."""
        parts = []
        for a, b in self._runs():
            t = self.t[a:b]
            i0 = a + int(np.searchsorted(t, lo, 'left'))
            i1 = a + int(np.searchsorted(t, hi, 'right'))
            if i1 > i0:
                parts.append(np.arange(i0, i1))
        idx = np.concatenate(parts) if parts else np.zeros(0, dtype=int)
        out = {}
        for k in keys:
            src = [k] if k not in DERIVED else list(DERIVED[k][:2])
            if all(s in self.cols for s in src):
                out[k] = series({s: self.cols[s][idx] for s in src}, k)
        return self.t[idx], out

    def span(self):
        if not self.n:
            return None
        oldest = 0 if self.n <= self.cap else self.n % self.cap
        return float(self.t[oldest]), float(self.t[(self.n - 1) % self.cap])


class LiveBuffer:
    """The live source: states from run_rig's ZMQ publisher, flattened by
    row_from_state and kept as columns for `history_s`.

    Markers are found as rows arrive: a stimulus change is an onset (`stim` is
    held between changes), and `rec` changing is a recording starting or
    stopping. A frame clock that jumps backwards means run_rig was restarted:
    the buffer starts afresh instead of drawing the new run on top of the old
    one, where it would fall outside the window."""

    def __init__(self, address, history_s=600.0, max_rate_hz=200.0, connect=True):
        self.address = address
        self.ring = ColumnRing(int(history_s * max_rate_hz))
        self.lock = threading.Lock()
        self.markers = []              # [(t, kind, label)]
        self.last = None               # newest row, for the REC / STALE badges
        self.n_rx = 0
        self.resets = 0
        self.t_origin = None           # first frame-clock time of this run
        self.error = None
        self._stop = False
        self.thread = None
        if connect:
            self.thread = threading.Thread(target=self._run, daemon=True, name='scope-live')
            self.thread.start()

    def _run(self):
        try:
            import zmq
            sock = zmq.Context.instance().socket(zmq.SUB)
            sock.setsockopt(zmq.RCVTIMEO, 300)
            sock.setsockopt_string(zmq.SUBSCRIBE, '')
            sock.connect(self.address)
        except Exception as e:           # a bad address must not kill the viewer
            self.error = str(e)
            return
        try:
            while not self._stop:
                try:
                    msg = sock.recv_string()
                except zmq.Again:
                    continue
                try:
                    self.push(row_from_state(json.loads(msg)))
                except (ValueError, KeyError, TypeError):
                    continue
        finally:
            sock.close(linger=0)

    def push(self, row):
        with self.lock:
            prev = self.last
            if prev is not None and row['t'] < prev['t'] - 1.0:
                self.ring.clear()
                self.markers = [(row['t'], 'reset', 'restart')]
                self.resets += 1
                self.t_origin = row['t']
                prev = None
            if self.t_origin is None:
                self.t_origin = row['t']
            if prev is not None:
                if row.get('stim') is not None and row.get('stim') != prev.get('stim'):
                    self.markers.append((row['t'], 'stim', '%g' % row['stim']))
                if row.get('rec') != prev.get('rec'):
                    self.markers.append((row['t'], 'rec', 'REC ' + row['rec'] if row.get('rec')
                                         else 'stop'))
            elif row.get('rec'):
                self.markers.append((row['t'], 'rec', 'REC ' + row['rec']))
            self.ring.append(row)
            self.last = row
            self.n_rx += 1
            span = self.ring.span()
            while self.markers and span and self.markers[0][0] < span[0]:
                self.markers.pop(0)

    def window(self, keys, lo=-np.inf, hi=np.inf):
        with self.lock:
            return self.ring.window(keys, lo, hi)

    def span(self):
        with self.lock:
            return self.ring.span()

    def keys(self):
        with self.lock:
            return set(self.ring.cols)

    def status(self):
        with self.lock:
            last = self.last or {}
            return {'n': self.n_rx, 'rec': last.get('rec'), 'stale': last.get('stale'),
                    'markers': list(self.markers), 'origin': self.t_origin,
                    'error': self.error}

    def close(self):
        self._stop = True
