"""HDF5 fly-state + event logger (readable from Python and MATLAB: h5read).

Layout, one row per merged state:
    /merged/seq, /merged/t_host, /merged/t_wall
    /<camera>/seq, t_camera, t_host, t_done
    /<camera>/<part>_angle, <part>_radius   for head, abdomen, left, right
    /<camera>/aux_intensity, /<camera>/aux_freq
Events (from EventReceiver or log_event), one row per event:
    /events/t_host, /events/t_wall, /events/text

t_host is time.monotonic() (shared by all rows and events in this file);
t_wall is time.time() so other programs' timestamps can be aligned.

Writes happen on a dedicated thread fed by a queue so disk I/O can never stall
the tracking loop. Missing cameras/parts are NaN.
"""
import math
import queue
import threading
import time

import h5py
import numpy as np

from .base import FlystateSink

PARTS = ('head', 'abdomen', 'left', 'right')
CHUNK = 1024


class Hdf5Logger(FlystateSink):
    def __init__(self, path, cameras):
        self.path = str(path)
        self.cameras = list(cameras)
        self.queue = queue.Queue(maxsize=10000)
        self.file = h5py.File(self.path, 'w')
        self.datasets = {}
        self.n = 0
        self.n_events = 0
        self.closed = False

        def add(name, dtype='f8'):
            self.datasets[name] = self.file.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(CHUNK,))

        add('merged/seq')
        add('merged/t_host')
        add('merged/t_wall')
        for cam in self.cameras:
            for col in ('seq', 't_camera', 't_host', 't_done'):
                add('%s/%s' % (cam, col))
            for part in PARTS:
                add('%s/%s_angle' % (cam, part))
                add('%s/%s_radius' % (cam, part))
            add('%s/aux_intensity' % cam)
            add('%s/aux_freq' % cam)
        add('events/t_host')
        add('events/t_wall')
        add('events/text', h5py.string_dtype())

        self.thread = threading.Thread(target=self._writer, daemon=True,
                                       name='hdf5-logger')
        self.thread.start()

    def handle(self, merged):
        if self.closed:
            return
        row = {'merged/seq': merged.seq, 'merged/t_host': merged.t_host,
               'merged/t_wall': time.time()}
        for cam in self.cameras:
            cs = merged.cameras.get(cam)
            if cs is None:
                for col in ('seq', 't_camera', 't_host', 't_done'):
                    row['%s/%s' % (cam, col)] = math.nan
                for part in PARTS:
                    row['%s/%s_angle' % (cam, part)] = math.nan
                    row['%s/%s_radius' % (cam, part)] = math.nan
                row['%s/aux_intensity' % cam] = math.nan
                row['%s/aux_freq' % cam] = math.nan
                continue
            row['%s/seq' % cam] = cs.seq
            row['%s/t_camera' % cam] = cs.t_camera
            row['%s/t_host' % cam] = cs.t_host
            row['%s/t_done' % cam] = cs.t_done
            for part in PARTS:
                ps = cs.parts.get(part)
                row['%s/%s_angle' % (cam, part)] = ps.angle if ps else math.nan
                row['%s/%s_radius' % (cam, part)] = ps.radius if ps else math.nan
            aux = cs.parts.get('aux')
            row['%s/aux_intensity' % cam] = aux.intensity if aux else math.nan
            row['%s/aux_freq' % cam] = aux.freq if aux else math.nan
        try:
            self.queue.put_nowait(('state', row))
        except queue.Full:
            pass  # Losing a log row beats stalling the closed loop.

    def log_event(self, text, t_host=None, t_wall=None):
        """Record an experiment event (thread-safe; never blocks)."""
        if self.closed:
            return
        item = ('event', t_host if t_host is not None else time.monotonic(),
                t_wall if t_wall is not None else time.time(), str(text))
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
        self.thread.join(timeout=30)
        for name, ds in self.datasets.items():
            ds.resize((self.n_events if name.startswith('events/') else self.n,))
        self.file.close()
