"""Recording sessions: attach/detach the recording sinks (HDF5 logger, video
writers, event log) while tracking, the live ZMQ stream and the GUI keep
running. This is what lets Kinefly2 stay on for hours while an experiment
script starts and stops recordings.

Also implements the command protocol used by the command socket, the GUI
record button and the CLI (see handle_command).
"""
import os
import threading
import time

import cv2

from .experiment import Experiment
from ..output.logger import Hdf5Logger
from ..output.video_recorder import VideoRecorder
from ..version import provenance


class RecordingSession:
    def __init__(self, template, merger, pipelines, views=None, events=None,
                 config_paths=()):
        """template: Experiment with the per-rig defaults (data_root, genotype,
        video settings); each start() overrides fields and resolves the date.
        pipelines: {camera name: CameraPipeline}; views: {name: TrackerView}."""
        self.template = template
        self.merger = merger
        self.pipelines = dict(pipelines)
        self.views = dict(views or {})
        self.events = events
        self.config_paths = [p for p in config_paths if p]
        self.lock = threading.RLock()
        self.active = None          # dict while recording
        self.last_summary = None
        self.n_sessions = 0

    # ---------------------------------------------------------------- control

    @property
    def recording(self):
        return self.active is not None

    def start(self, **overrides):
        with self.lock:
            if self.active is not None:
                raise RuntimeError('already recording %s' % self.active['exp'].run_name)
            values = {k: getattr(self.template, k) for k in Experiment.FIELDS}
            values['date'] = 'auto'
            values.update({k: v for k, v in overrides.items()
                           if v is not None and k in Experiment.FIELDS})
            exp = Experiment.load(None, values)
            exp.source_path = self.template.source_path
            exp.make_output_dir()
            exp.snapshot_configs(self.config_paths)
            exp.write_resolved()

            # This fly's lever arm, applied to the trackers for the duration of
            # the recording so the GUI shows degrees and the log carries angles.
            # Cameras not named keep pixels; a ventral view ignores it.
            applied = {}
            for cam, view in self.views.items():
                if hasattr(view, 'set_lever_px'):
                    parts = view.set_lever_px((exp.lever_px or {}).get(cam))
                    if (exp.lever_px or {}).get(cam) and parts:
                        applied[cam] = exp.lever_px[cam]
            if applied:
                print('REC lever_px: %s' % ', '.join('%s=%.1f' % kv for kv in applied.items()))

            logger = None
            if exp.log_flystate:
                # Stamp what the numbers depend on: the constant used, and the
                # landmark positions it was meant for. A lever arm fitted for a
                # different set of points is silently wrong, so calibrate_fly
                # compares these before reusing one.
                # `rig` is what a reader needs to reconstruct the recording's
                # own time base: which camera drove the merge (so whose
                # t_camera the rows belong to) and the staleness threshold that
                # decided which values were blanked.
                meta = {'lever_px': exp.lever_px or {}, 'landmarks': {},
                        'rig': {'primary': self.merger.primary,
                                'cameras': list(self.pipelines),
                                # What each camera actually delivered, not the
                                # config's request (see PySpinSource).
                                'fps': {c: getattr(p.source, 'fps', None)
                                        for c, p in self.pipelines.items()},
                                'parts': {c: v.part_names() for c, v in self.views.items()
                                          if hasattr(v, 'part_names')},
                                'stale_after_s': self.merger.stale_after_s},
                        **provenance()}
                for cam, view in self.views.items():
                    if hasattr(view, 'landmark_geometry'):
                        meta['landmarks'][cam] = view.landmark_geometry()
                parts = {cam: view.part_names() for cam, view in self.views.items()
                         if hasattr(view, 'part_names')}
                logger = Hdf5Logger(exp.path('_flystate.h5'), cameras=list(self.pipelines),
                                    metadata=meta, parts=parts)
                self.merger.add_sink(logger)
            recorders = {}
            if exp.record_video:
                for name, p in self.pipelines.items():
                    fps = getattr(p.source, 'fps', None) or 30.0
                    rec = VideoRecorder(exp.path('_%s.avi' % name), fps=fps,
                                        quality=exp.video_quality,
                                        sidecar_path=exp.path('_%s_frames.csv' % name))
                    recorders[name] = rec
                    p.recorder = rec
            if self.events is not None:
                self.events.attach(logger, exp.path('_events.jsonl'))
            self.active = {'exp': exp, 'logger': logger, 'recorders': recorders,
                           't_start': time.monotonic(), 'wall_start': time.time(),
                           'frames_start': {n: p.n_frames for n, p in self.pipelines.items()},
                           'stale_start': self.merger.stale_counts()}
            self.n_sessions += 1
            # Announce it on the live stream, so a dashboard can say what it is
            # watching and anchor its time axis to the start of the run.
            self.merger.set_recording(exp.run_name)
            print('REC start: %s -> %s' % (exp.run_name, exp.output_dir))
            return {'name': exp.run_name, 'output_dir': exp.output_dir}

    def stop(self):
        with self.lock:
            if self.active is None:
                raise RuntimeError('not recording')
            a = self.active
            self.merger.set_recording(None)
            for name, p in self.pipelines.items():
                p.recorder = None
            if self.events is not None:
                self.events.detach()
            for rec in a['recorders'].values():
                rec.close()
            if a['logger'] is not None:
                self.merger.remove_sink(a['logger'])
                a['logger'].close()
            summary = self._summary(a, final=True)
            self.active = None
            self.last_summary = summary
            msg = 'REC stop: %s  %.1f s, %d rows, %d events' % (
                summary['name'], summary['duration'], summary['rows'], summary['events'])
            if summary['stale']:
                msg += '  STALE %s' % ', '.join('%s=%d rows' % kv
                                                for kv in sorted(summary['stale'].items()))
            if summary['log_dropped']:
                msg += '  LOG DROPPED %d rows (disk too slow)' % summary['log_dropped']
            print(msg)
            return summary

    def _summary(self, a, final=False):
        exp = a['exp']
        s = {'name': exp.run_name, 'output_dir': exp.output_dir,
             'duration': time.monotonic() - a['t_start'],
             'rows': a['logger'].n if a['logger'] is not None else 0,
             'events': a['logger'].n_events if a['logger'] is not None else 0,
             # Rows the logger had to drop because the disk fell behind.
             'log_dropped': a['logger'].n_dropped if a['logger'] is not None else 0,
             'video': {n: {'written': r.n_written, 'dropped': r.n_dropped}
                       for n, r in a['recorders'].items()},
             'frames': {n: p.n_frames - a['frames_start'][n] for n, p in self.pipelines.items()}}
        # Rows in which a camera was too old to trust and was logged as NaN.
        # Anything but zero here is worth knowing before the fly is discarded.
        now = self.merger.stale_counts()
        s['stale'] = {n: now.get(n, 0) - a['stale_start'].get(n, 0)
                      for n in set(now) | set(a['stale_start'])}
        s['stale'] = {n: v for n, v in s['stale'].items() if v}
        return s

    def status(self):
        with self.lock:
            st = {'recording': self.active is not None,
                  'sessions': self.n_sessions,
                  'cameras': {n: p.status() for n, p in self.pipelines.items()},
                  'template': {'data_root': self.template.data_root,
                               'genotype': self.template.genotype,
                               'fly_id': self.template.fly_id,
                               'protocol': self.template.protocol}}
            if self.active is not None:
                st['session'] = self._summary(self.active)
            elif self.last_summary is not None:
                st['last_session'] = self.last_summary
            return st

    def snapshot(self):
        """Save the current frame of every camera as PNG (into the recording
        folder if recording, else into <data_root>/snapshots)."""
        with self.lock:
            if self.active is not None:
                d = self.active['exp'].output_dir
                prefix = self.active['exp'].run_name + '_'
            else:
                d = os.path.join(os.path.expanduser(self.template.data_root), 'snapshots')
                prefix = ''
            os.makedirs(d, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            paths = []
            for name, view in self.views.items():
                img = view.imgUnscaled
                if img is None:
                    continue
                path = os.path.join(d, '%ssnapshot_%s_%s.png' % (prefix, name, stamp))
                cv2.imwrite(path, img)
                paths.append(path)
            return paths

    def close(self):
        with self.lock:
            if self.active is not None:
                self.stop()

    # ------------------------------------------------------------- commands

    def handle_command(self, req):
        """Command protocol (dict in, dict out). Used by the ZMQ command
        socket; errors are returned as {'ok': False, 'error': ...}."""
        try:
            cmd = req.get('cmd')
            if cmd == 'start':
                fields = {k: req.get(k) for k in Experiment.FIELDS}
                return {'ok': True, **self.start(**fields)}
            elif cmd == 'stop':
                return {'ok': True, 'summary': self.stop()}
            elif cmd == 'status':
                return {'ok': True, **self.status()}
            elif cmd == 'event':
                text = req.get('text')
                if text is None:
                    return {'ok': False, 'error': "event needs 'text'"}
                if self.events is None:
                    return {'ok': False, 'error': 'events disabled'}
                logged = self.events.log(text if isinstance(text, str) else __import__('json').dumps(text))
                return {'ok': True, 'logged': logged, 'recording': self.recording}
            elif cmd == 'snapshot':
                return {'ok': True, 'paths': self.snapshot()}
            elif cmd == 'zero':
                out = {}
                for name, view in self.views.items():
                    if req.get('camera') in (None, name):
                        out[name] = view.zero_all()
                return {'ok': True, 'zeroed': out}
            elif cmd == 'stim':
                # The stimulus code now in force. Held until it changes, so
                # every frame carries the value that was on.
                if 'value' not in req:
                    return {'ok': False, 'error': "stim needs 'value'"}
                out = self.merger.set_stim(req.get('value'))
                return {'ok': True, **out, 'recording': self.recording}
            elif cmd == 'ping':
                return {'ok': True, 'time': time.time()}
            else:
                return {'ok': False, 'error': 'unknown cmd %r' % cmd}
        except Exception as e:
            return {'ok': False, 'error': str(e)}
