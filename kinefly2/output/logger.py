"""HDF5 fly-state logger (readable from Python and MATLAB >= R2011: h5read).

Layout, one row per merged state:
    /merged/seq, /merged/t_host
    /<camera>/seq, t_camera, t_host, t_done
    /<camera>/<part>_angle, <part>_radius   for head, abdomen, left, right
    /<camera>/aux_intensity, /<camera>/aux_freq

Writes happen on a dedicated thread fed by a queue so disk I/O can never stall
the tracking loop. Missing cameras/parts are NaN.
"""
import math
import queue
import threading

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

        def add(name):
            self.datasets[name] = self.file.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype='f8', chunks=(CHUNK,))

        add('merged/seq')
        add('merged/t_host')
        for cam in self.cameras:
            for col in ('seq', 't_camera', 't_host', 't_done'):
                add('%s/%s' % (cam, col))
            for part in PARTS:
                add('%s/%s_angle' % (cam, part))
                add('%s/%s_radius' % (cam, part))
            add('%s/aux_intensity' % cam)
            add('%s/aux_freq' % cam)

        self.thread = threading.Thread(target=self._writer, daemon=True,
                                       name='hdf5-logger')
        self.thread.start()

    def handle(self, merged):
        row = {'merged/seq': merged.seq, 'merged/t_host': merged.t_host}
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
            self.queue.put_nowait(row)
        except queue.Full:
            pass  # Losing a log row beats stalling the closed loop.

    def _writer(self):
        while True:
            row = self.queue.get()
            if row is None:
                return
            i = self.n
            self.n += 1
            for name, value in row.items():
                ds = self.datasets[name]
                if ds.shape[0] <= i:
                    ds.resize((i + CHUNK,))
                ds[i] = value
            if self.n % CHUNK == 0:
                self.file.flush()

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=5)
        for ds in self.datasets.values():
            ds.resize((self.n,))
        self.file.close()
