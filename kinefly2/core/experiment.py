"""Per-experiment settings: the one place that changes between recordings.

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
    source_path: str = None       # the YAML this came from, if any
    output_dir: str = field(default=None, repr=False)

    FIELDS = ('data_root', 'date', 'genotype', 'fly_id', 'protocol', 'notes',
              'record_video', 'video_quality', 'log_flystate', 'events_port')

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
