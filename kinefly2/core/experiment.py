"""Per-experiment settings: the one place that changes between recordings.

This is also where each fly's `lever_px` lives -- the constant that turns the
front camera's displacement channels into pitch and yaw angles. It is per fly
because head size, mounting and landmark placement all change it, and it is
deliberately NOT in the camera config so that one fly's constant can never be
inherited by the next. Fit it from the fly's own recording with
kinefly2.apps.calibrate_fly.

An experiment YAML (see experiments/default.yaml) plus command-line overrides
resolve to a recording name following the lab convention
    {date}_{genotype}_{fly_id}_{protocol}       e.g. 260819_XHCS_XHCS_017_YAMV
and an output folder
    <data_root>/<date>/<name>/                  (suffix _2, _3, ... if it exists)
holding the HDF5 fly state, optional MJPEG video per camera with frame-index
sidecars, the stimulus event log, the resolved settings, and a snapshot of
the rig/camera configs used — everything needed to re-analyze the recording.
"""
import dataclasses
import os
import shutil
import time
from dataclasses import dataclass, field

import yaml


@dataclass
class Experiment:
    data_root: str = '~/Data/Kinefly2'
    date: str = 'auto'            # YYMMDD; 'auto' = today
    genotype: str = 'genotype'
    fly_id: str = '000'
    protocol: str = 'protocol'
    notes: str = ''
    record_video: bool = True     # per-recording raw-video switch
    video_quality: int = 95       # MJPEG quality, 1-100
    log_flystate: bool = True
    events_port: int = 5557       # ZMQ PULL for stimulus events; null/0 = off
    lever_px: dict = None         # {camera: px/rad} for THIS fly; see calibrate_fly.
                                  # Absent -> pitch/yaw stay in px, never guessed.
    source_path: str = None       # the YAML this came from, if any
    output_dir: str = field(default=None, repr=False)

    FIELDS = ('data_root', 'date', 'genotype', 'fly_id', 'protocol', 'notes',
              'record_video', 'video_quality', 'log_flystate', 'events_port',
              'lever_px')

    @classmethod
    def load(cls, path=None, overrides=None):
        """Build from a YAML file (optional) with non-None overrides winning."""
        values = {}
        if path:
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            values.update({k: v for k, v in loaded.items() if k in cls.FIELDS})
            values['source_path'] = os.path.abspath(path)
        for k, v in (overrides or {}).items():
            if v is not None and k in cls.FIELDS:
                values[k] = v
        exp = cls(**values)
        if exp.date in (None, 'auto', ''):
            exp.date = time.strftime('%y%m%d')
        exp.date = str(exp.date)
        exp.fly_id = str(exp.fly_id)
        if not exp.events_port:
            exp.events_port = None
        # The lever arm is per fly, so it lives here rather than in the camera
        # config: a constant fitted for one fly must never leak into the next.
        if isinstance(exp.lever_px, dict):
            exp.lever_px = {str(k): float(v) for k, v in exp.lever_px.items()
                            if v is not None}
        elif exp.lever_px is not None:
            raise ValueError('lever_px must map camera name -> px/rad, '
                             'e.g. {front: 48.2}; got %r' % (exp.lever_px,))
        return exp

    @property
    def name(self):
        return '%s_%s_%s_%s' % (self.date, self.genotype, self.fly_id, self.protocol)

    def make_output_dir(self):
        base = os.path.join(os.path.expanduser(self.data_root), self.date)
        os.makedirs(base, exist_ok=True)
        candidate = os.path.join(base, self.name)
        n = 2
        while os.path.exists(candidate):
            candidate = os.path.join(base, '%s_%d' % (self.name, n))
            n += 1
        os.makedirs(candidate)
        self.output_dir = candidate
        return candidate

    @property
    def run_name(self):
        """Name including any collision suffix (folder basename)."""
        return os.path.basename(self.output_dir) if self.output_dir else self.name

    def path(self, suffix):
        """Path of an output file: <output_dir>/<run_name><suffix>."""
        return os.path.join(self.output_dir, self.run_name + suffix)

    def snapshot_configs(self, paths):
        cfg_dir = os.path.join(self.output_dir, 'configs')
        os.makedirs(cfg_dir, exist_ok=True)
        for p in paths:
            if p and os.path.exists(p):
                shutil.copy2(p, os.path.join(cfg_dir, os.path.basename(p)))
        return cfg_dir

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.FIELDS}
        d['name'] = self.run_name
        d['output_dir'] = self.output_dir
        d['source_path'] = self.source_path
        d['started'] = time.strftime('%Y-%m-%d %H:%M:%S')
        return d

    def write_resolved(self):
        with open(self.path('_experiment.yaml'), 'w') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
