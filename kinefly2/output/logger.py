"""HDF5 fly-state + event logger (readable from Python and MATLAB: h5read).

Layout, one row per merged state (angles in RADIANS; the GUI displays degrees):
    /merged/seq, /merged/t_host, /merged/t_wall
    /merged/stim                            the stimulus code in force on this frame --
                                            the voltage the panels are driven with, set by
                                            the experiment script (Kinefly2Client.stim) and
                                            HELD until it changes, so every row is labelled
                                            with the stimulus that was on. NaN before the
                                            first call.
    /merged/stim_t_wall                     time.time() when that value last changed, so
                                            stimulus update latency stays visible
    /<camera>/seq, t_camera, t_host, t_done
    /<camera>/age                           how old this camera's contribution was when the
                                            row was merged, in seconds. ~0 for the primary
                                            camera (its frame triggered the merge); for a
                                            secondary it is under one frame interval when
                                            healthy. Past the rig's `stale_after_s` the
                                            merger blanks that camera's measurements to NaN
                                            rather than repeating them, and this column is
                                            how long the gap lasted. Comparing `seq` between
                                            two cameras is the other check: with a hardware
                                            trigger (or lockstep replay) they advance
                                            together.
    /<camera>/<part>_angle, <part>_radius   for head, abdomen, left, right, plus any extra
                                            parts the camera declares (`parts:` in its YAML --
                                            the ventral view's hind legs, leg_l3/leg_r3)
    /<camera>/<part>_n_edges                edge trackers only: how many edges that annulus
                                            found this frame. One is what a correctly placed
                                            sector gives; more means it is catching a second
                                            leg or the body margin, and the angle is then a
                                            plausible-looking wrong answer. NaN for other trackers.
    /<camera>/aux_intensity, /<camera>/aux_freq
    /<camera>/head_dx, head_dy              head translation, px (landmarks tracker)
    /<camera>/head_dx_corr, head_dy_corr    same, with the in-plane rotation's own
                                            centroid shift removed, in IMAGE axes.
    /<camera>/head_d_axial, head_d_lateral  the same shift resolved along the part's own
                                            long axis and across it (px). Axial is PITCH;
                                            across it is YAW in a frontal view. A VENTRAL
                                            camera leaves both NaN by design: its lever arm
                                            (dorsoventral depth) is too short, and those
                                            channels measured noise (SNR 1.9 and 1.2). The
                                            column set is kept uniform across cameras, so
                                            all-NaN here is expected, not a bug.
    /<camera>/head_roll, head_pitch, head_yaw
                                            the three rotations under fixed anatomical
                                            names, radians, so analysis need not know the
                                            view. Exactly one is always finite -- the exact
                                            in-plane angle (yaw for a ventral camera, roll
                                            for a frontal one). A frontal camera fills the
                                            other two once `lever_px` is calibrated; a
                                            ventral camera never does.
    /<camera>/head_fit_residual, head_score landmark fit quality
    /<camera>/head_roll_sigma               roll uncertainty, radians (residual / (spread
                                            * sqrt(N))): the precision the current
                                            landmark layout can actually deliver
Events (from EventReceiver or log_event), one row per event:
    /events/t_host, /events/t_wall, /events/text

t_host is time.monotonic() (shared by all rows and events in this file);
t_wall is time.time() so other programs' timestamps can be aligned.

Writes happen on a dedicated thread fed by a queue so disk I/O can never stall
the tracking loop: live, a full queue drops the row (counted in n_dropped)
rather than wait. Offline analysis passes block=True instead -- nothing there is
real time, and every frame must land -- and mode='x', so an existing file is
never truncated. Missing cameras/parts are NaN.
"""
import math
import queue
import threading
import time

import h5py
import numpy as np

from .base import FlystateSink

PARTS = ('head', 'abdomen', 'left', 'right')   # default when a camera does not say
HEAD_EXTRA = ('dx', 'dy', 'dx_corr', 'dy_corr', 'fit_residual', 'score',
              'd_axial', 'd_lateral', 'roll', 'pitch', 'yaw', 'roll_sigma')
# Anatomical channels the landmark tracker fills; which one is the exact
# in-plane angle depends on the camera view (ventral -> yaw, frontal -> roll).
LM_EXTRA = ('d_axial', 'd_lateral', 'roll', 'pitch', 'yaw', 'roll_sigma')
CHUNK = 1024


class Hdf5Logger(FlystateSink):
    def __init__(self, path, cameras, metadata=None, parts=None, mode='w', block=False):
        """`parts` maps camera -> the angle-bearing parts it tracks, so a camera
        declaring extra parts (hind legs on the ventral view) gets columns for
        them. Cameras not named fall back to the standard four.

        `mode` is h5py's: 'w' truncates an existing file, 'x' refuses to open
        one. `block` makes handle() wait for the writer when the queue is full
        instead of dropping the row."""
        self.path = str(path)
        self.cameras = list(cameras)
        self.parts = {cam: tuple((parts or {}).get(cam) or PARTS) for cam in self.cameras}
        self.block = bool(block)
        self.queue = queue.Queue(maxsize=10000)
        self.file = h5py.File(self.path, mode)
        self.datasets = {}
        self.n = 0
        self.n_events = 0
        self.n_dropped = 0          # Rows lost to a full queue (never when blocking).
        self.closed = False

        def add(name, dtype='f8'):
            self.datasets[name] = self.file.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(CHUNK,))

        add('merged/seq')
        add('merged/t_host')
        add('merged/t_wall')
        add('merged/stim')          # stimulus code in force on this frame
        add('merged/stim_t_wall')   # when it last changed (time.time())
        for cam in self.cameras:
            for col in ('seq', 't_camera', 't_host', 't_done', 'age'):
                add('%s/%s' % (cam, col))
            for part in self.parts[cam]:
                add('%s/%s_angle' % (cam, part))
                add('%s/%s_radius' % (cam, part))
                add('%s/%s_n_edges' % (cam, part))
            add('%s/aux_intensity' % cam)
            add('%s/aux_freq' % cam)
            for col in HEAD_EXTRA:          # landmarks tracker only (NaN otherwise)
                add('%s/head_%s' % (cam, col))
        add('events/t_host')
        add('events/t_wall')
        add('events/text', h5py.string_dtype())

        # Provenance as file attributes: what the derived angles depend on.
        # `landmarks` records the points each lever arm was meant for, so a
        # constant reused after the landmarks moved can be caught.
        if metadata:
            import json
            for key, value in metadata.items():
                self.file.attrs[key] = json.dumps(value)

        self.thread = threading.Thread(target=self._writer, daemon=True,
                                       name='hdf5-logger')
        self.thread.start()

    def handle(self, merged):
        if self.closed:
            return
        row = {'merged/seq': merged.seq, 'merged/t_host': merged.t_host,
               'merged/t_wall': time.time(),
               'merged/stim': merged.stim, 'merged/stim_t_wall': merged.stim_t_wall}
        for cam in self.cameras:
            cs = merged.cameras.get(cam)
            if cs is None:
                for col in ('seq', 't_camera', 't_host', 't_done', 'age'):
                    row['%s/%s' % (cam, col)] = math.nan
                for part in self.parts[cam]:
                    row['%s/%s_angle' % (cam, part)] = math.nan
                    row['%s/%s_radius' % (cam, part)] = math.nan
                    row['%s/%s_n_edges' % (cam, part)] = math.nan
                row['%s/aux_intensity' % cam] = math.nan
                row['%s/aux_freq' % cam] = math.nan
                for col in HEAD_EXTRA:
                    row['%s/head_%s' % (cam, col)] = math.nan
                continue
            row['%s/seq' % cam] = cs.seq
            row['%s/t_camera' % cam] = cs.t_camera
            row['%s/t_host' % cam] = cs.t_host
            row['%s/t_done' % cam] = cs.t_done
            # A stale camera arrives here with no parts at all (the merger
            # blanked it), so every measurement below falls through to NaN
            # while seq/t_camera/age stay to say what happened.
            row['%s/age' % cam] = merged.age(cam)
            for part in self.parts[cam]:
                ps = cs.parts.get(part)
                row['%s/%s_angle' % (cam, part)] = ps.angle if ps else math.nan
                row['%s/%s_radius' % (cam, part)] = ps.radius if ps else math.nan
                # Edge trackers only: how many edges the annulus found. Anything
                # but the expected count means the sector caught something else,
                # and the angle above is then a plausible-looking wrong answer.
                row['%s/%s_n_edges' % (cam, part)] = (
                    (ps.extra or {}).get('n_edges', math.nan) if ps else math.nan)
            aux = cs.parts.get('aux')
            row['%s/aux_intensity' % cam] = aux.intensity if aux else math.nan
            row['%s/aux_freq' % cam] = aux.freq if aux else math.nan
            # Landmark tracker extras: head translation in px (raw and with the
            # roll-induced shift removed), fit quality. NaN for other trackers.
            hd = cs.parts.get('head')
            lm = (hd is not None and hd.kind == 'landmarks'
                  and len(hd.radii) >= 2 and len(hd.gradients) >= 3)
            row['%s/head_dx' % cam] = hd.radii[0] if lm else math.nan
            row['%s/head_dy' % cam] = hd.radii[1] if lm else math.nan
            row['%s/head_dx_corr' % cam] = hd.gradients[0] if lm else math.nan
            row['%s/head_dy_corr' % cam] = hd.gradients[1] if lm else math.nan
            row['%s/head_fit_residual' % cam] = hd.gradients[2] if lm else math.nan
            row['%s/head_score' % cam] = hd.intensity if lm else math.nan
            # yaw/pitch are NaN until lever_px is calibrated; roll_sigma always.
            ex = hd.extra if lm else {}
            for col in LM_EXTRA:
                row['%s/head_%s' % (cam, col)] = ex.get(col, math.nan)
        if self.block:
            self.queue.put(('state', row))
            return
        try:
            self.queue.put_nowait(('state', row))
        except queue.Full:
            self.n_dropped += 1   # Losing a log row beats stalling the closed loop.

    def log_event(self, text, t_host=None, t_wall=None):
        """Record an experiment event (thread-safe; never blocks unless the
        logger was made blocking)."""
        if self.closed:
            return
        item = ('event', t_host if t_host is not None else time.monotonic(),
                t_wall if t_wall is not None else time.time(), str(text))
        if self.block:
            self.queue.put(item)
            return
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            pass

    def _put(self, name, i, value):
        ds = self.datasets[name]
        if ds.shape[0] <= i:
            ds.resize((i + CHUNK,))
        ds[i] = value

    def _writer(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            if item[0] == 'state':
                i = self.n
                self.n += 1
                for name, value in item[1].items():
                    self._put(name, i, value)
                if self.n % CHUNK == 0:
                    self.file.flush()
            else:
                _, t_host, t_wall, text = item
                i = self.n_events
                self.n_events += 1
                self._put('events/t_host', i, t_host)
                self._put('events/t_wall', i, t_wall)
                self._put('events/text', i, text)
                self.file.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        # A blocking logger has promised every row, so it waits for all of them
        # to be written; a live one gives up rather than hang the shutdown.
        self.thread.join(timeout=None if self.block else 30)
        for name, ds in self.datasets.items():
            ds.resize((self.n_events if name.startswith('events/') else self.n,))
        self.file.close()
