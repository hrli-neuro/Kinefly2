"""Analyze recorded videos: one job is one video, or the bottom+front pair of
one recording, tracked from frame 0 into the same fly-state HDF5 a live
recording writes.

A job runs in three steps, and the split is what keeps the results honest:

  prepare_setup  a PRIVATE tuning folder: each camera's YAML is a copy of the
                 seed (the rig config, or the previous analysis) pointed at the
                 video, with the video's own median background. Nothing is
                 written where results go, so abandoning a setup leaves nothing
                 behind -- and the rig config is never touched.
  finalize       claims the output folder (<name>, or <name>_001, ... -- see
                 core/naming.py) and copies the tuned files into it.
  run_analysis   fresh trackers from those files, from frame 0, cameras in
                 lockstep, every frame logged. The tuning session contributes no
                 value to the data: trackers carry history (the background
                 model, autozero, landmark prediction), and a tuning session is
                 all pauses, seeks and loops.

A finished analysis is one flat folder, every file labeled with the run and,
where it belongs to one camera, the camera:

    <save folder>/<run>/
      <run>_flystate.h5            both cameras, frame-paired: the layout of a live
                                   recording (the Scope, dashboard --file and
                                   calibrate_fly read it as they are)
      <run>_<cam>.csv              one camera's columns (output/csv_export.py)
      <run>_traces.png             every trace, drawn by the Scope (scope/launch.py)
      <run>_analysis.yaml          what was analyzed, how far it got, provenance
      <run>_<cam>_config.yaml      regions and landmarks exactly as used
      <run>_<cam>_background.png   <run>_<cam>_reference.png
      <run>_<cam>_<part>_templates.npz
"""
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from ..camera.video_source import VideoFileSource, median_background
from ..gui.app import TrackerView
from ..output.csv_export import h5_to_csv
from ..output.logger import Hdf5Logger
from ..scope.launch import render_figure
from ..version import provenance
from .merger import StateMerger
from .naming import reserve_dir
from .pipeline import CameraPipeline


@dataclass
class VideoJob:
    """One analysis: a video, or a bottom+front pair of the same recording."""
    name: str                        # As requested; the folder may become name_001.
    videos: dict                     # camera -> video path
    fps: float = None                # Override the container's frame rate.
    seeds: dict = field(default_factory=dict)   # camera -> YAML the tuning started from
    setup_dir: str = None            # Private tuning folder, until finalize().
    out_dir: str = None              # <save folder>/<run_name>, after finalize().
    run_name: str = None             # The folder's name: name, or name_001, ...

    @property
    def cameras(self):
        return list(self.videos)

    def stem(self, cam):
        """Path prefix of one camera's files: <tuning folder>/<cam> while it is
        tuned, <out_dir>/<run_name>_<cam> once finalized."""
        prefix = '%s_%s' % (self.run_name, cam) if self.out_dir else cam
        return os.path.join(self.out_dir or self.setup_dir, prefix)

    def config_path(self, cam):
        return self.stem(cam) + '_config.yaml'

    def path(self, suffix):
        """An output file: <out_dir>/<run_name><suffix>."""
        return os.path.join(self.out_dir, self.run_name + suffix)


def seed_reference(seed, cam):
    """The reference frame that goes with a seed config, or None: the
    <stem>_reference.png beside an analysis's <stem>_config.yaml, else the
    reference_<cam>.png beside a rig config (or inside the configs/ folder of
    an analysis made before the folders were flattened)."""
    if seed.endswith('_config.yaml'):
        path = seed[:-len('_config.yaml')] + '_reference.png'
    else:
        path = os.path.join(os.path.dirname(seed), 'reference_%s.png' % cam)
    return path if os.path.exists(path) else None


def prepare_setup(job, seeds, n_bg_samples=41, log=print):
    """Build the job's private tuning folder and return it.

    Per camera: the seed's tracker settings pointed at the video, the seed's
    reference frame (what `autoplace` registers against), and the video's own
    median background -- a recording has no frame without the fly, and the
    abdomen is unmeasurable without subtraction.

    Landmark templates are deliberately NOT carried over. The head's zero is
    the posture they were cut at, and templates from another video (another
    fly, another mounting) still match well enough to report angles --
    relative to the wrong posture. They are cut from this video instead: on
    the first frame shown, whenever a landmark is placed or moved, or with
    `zero`."""
    missing = [c for c in job.videos if not (seeds.get(c) and os.path.exists(seeds[c]))]
    if missing:
        raise ValueError('no camera config to start %s from' % ', '.join(missing))
    root = tempfile.mkdtemp(prefix='kinefly2_setup_')
    job.setup_dir, job.out_dir, job.run_name = root, None, None
    job.seeds = {c: os.path.abspath(seeds[c]) for c in job.videos}
    try:
        for cam, video in job.videos.items():
            with open(job.seeds[cam]) as f:
                doc = yaml.safe_load(f) or {}
            camera = {'kind': 'video', 'path': os.path.abspath(video)}
            if job.fps:
                camera['fps'] = float(job.fps)
            with open(job.config_path(cam), 'x') as f:
                yaml.safe_dump({'name': cam, 'camera': camera,
                                'tracker': doc.get('tracker') or {}}, f, sort_keys=False)
            ref = seed_reference(job.seeds[cam], cam)
            if ref:
                shutil.copy2(ref, job.stem(cam) + '_reference.png')
            bg, n_used, n_total = median_background(video, n_bg_samples)
            cv2.imwrite(job.stem(cam) + '_background.png', bg)
            log('%s: background = median of %d of %d frames of %s'
                % (cam, n_used, n_total, os.path.basename(video)))
    except Exception:
        discard_setup(job)
        raise
    return root


def discard_setup(job):
    """Forget a job's tuning (skipped, or the queue was quit)."""
    if job.setup_dir:
        shutil.rmtree(job.setup_dir, ignore_errors=True)
    job.setup_dir = None


def finalize(job, out_parent):
    """Claim the output folder -- <name>, or <name>_001, ... when taken -- and
    copy the tuned files into it, each labeled with the run: the tuning's
    <cam>_config.yaml becomes <run>_<cam>_config.yaml. Returns the folder."""
    src = job.setup_dir
    out_dir = reserve_dir(out_parent, job.name)
    run = os.path.basename(out_dir)
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.bak'):             # autoplace's undo copies
            shutil.copy2(os.path.join(src, fn), os.path.join(out_dir, '%s_%s' % (run, fn)))
    discard_setup(job)
    job.out_dir, job.run_name = out_dir, run
    return out_dir


def _write_yaml(path, record, exclusive):
    with open(path, 'x' if exclusive else 'w') as f:
        yaml.safe_dump(record, f, sort_keys=False)


def _progress(pipeline, total, t0):
    done = pipeline.n_frames
    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if total and rate > 0 and done <= total else None
    return {'done': done, 'total': total, 'elapsed': elapsed, 'fps': rate, 'eta': eta}


def _tracked_fractions(h5_path, views):
    """Per camera, the fraction of rows with a finite angle for each part it
    tracked: a part that lost its target shows at once, not in analysis."""
    import h5py

    out = {}
    with h5py.File(h5_path, 'r') as f:
        for cam, view in views.items():
            fr = {}
            for part in view.partnames:
                key = '%s/%s_angle' % (cam, part)
                if view.params['gui'][part]['track'] and key in f and len(f[key]):
                    fr[part] = round(float(np.isfinite(f[key][:]).mean()), 4)
            out[cam] = fr
    return out


def fmt_time(s):
    s = max(0.0, float(s))
    return '%d:%02d' % (s // 60, s % 60)


def progress_line(prog):
    s = '%d' % prog['done']
    if prog['total']:
        s += '/%d (%d%%)' % (prog['total'], 100.0 * prog['done'] / prog['total'])
    s += '  %.0f fps' % prog['fps']
    if prog['eta'] is not None:
        s += '  ETA %s' % fmt_time(prog['eta'])
    return s


def run_analysis(job, primary=None, csv=True, figure=True, show=None,
                 render_every_s=1 / 15.0, log=print):
    """Track every frame of the job's videos, from the first, into
    <out_dir>/<run_name>_flystate.h5, then write a CSV per camera and the
    traces figure. Returns the record that is also written to
    <run_name>_analysis.yaml.

    Fresh trackers are built from the configs in the output folder, so what is
    analyzed is exactly what is saved there. Two cameras run in lockstep --
    frame N of one is merged with frame N of the other -- and staleness
    blanking is off: in lockstep a camera's 'age' measures tracking time, not
    missing data, and blanking would throw away good frames. The logger blocks
    rather than drop a row.

    show(pipelines, progress) -> bool, if given, is called from this (the main)
    thread ~30 times a second to draw and to poll for a cancel, and returns
    True to cancel. Without it the run is headless. A cancelled or failed run
    keeps what it wrote and says so in the record."""
    cams = job.cameras
    primary = primary if primary in cams else cams[0]
    started = time.strftime('%Y-%m-%d %H:%M:%S')
    views, sources = {}, {}
    try:
        for cam in cams:
            views[cam] = TrackerView(cam, job.config_path(cam), use_gui=False,
                                     replay=True, toolbar=False, asset_stem=job.stem(cam))
            src = VideoFileSource(job.videos[cam], fps=job.fps)
            src.open()
            sources[cam] = src
    except Exception:
        for s in sources.values():
            s.close()
        raise
    n_frames = {c: int(s.n_frames) for c, s in sources.items()}
    fps = {c: round(float(s.fps), 4) for c, s in sources.items()}
    warnings = []
    if len(set(n_frames.values())) > 1:
        warnings.append('videos differ in length (%s); lockstep stops at the shortest'
                        % ', '.join('%s %d' % kv for kv in n_frames.items()))
    if len(set(round(v, 2) for v in fps.values())) > 1:
        warnings.append('videos differ in frame rate (%s)'
                        % ', '.join('%s %.2f' % kv for kv in fps.items()))
    for w in warnings:
        log('WARNING: %s' % w)
    known = [n for n in n_frames.values() if n]
    total = min(known) if known else 0

    parts = {c: v.part_names() for c, v in views.items()}
    source = {'mode': 'offline video analysis',
              'videos': {c: os.path.abspath(job.videos[c]) for c in cams},
              'n_frames': n_frames, 'fps': fps}
    meta = {'lever_px': {},
            'landmarks': {c: v.landmark_geometry() for c, v in views.items()},
            'rig': {'primary': primary, 'cameras': cams, 'parts': parts,
                    'stale_after_s': None},
            'source': source,
            **provenance()}
    h5_path = job.path('_flystate.h5')
    yaml_path = job.path('_analysis.yaml')
    record = {'name': job.run_name, 'requested_name': job.name,
              'output_dir': job.out_dir, 'status': 'running',
              'videos': {c: {'path': source['videos'][c], 'frames': n_frames[c], 'fps': fps[c]}
                         for c in cams},
              'primary': primary,
              'seed': dict(job.seeds),
              'started': started,
              'warnings': warnings,
              'provenance': provenance()}
    merger = StateMerger(primary=primary, stale_after_s=None)
    try:
        logger = Hdf5Logger(h5_path, cameras=cams, metadata=meta, parts=parts,
                            mode='x', block=True)
        merger.add_sink(logger)
        # Written now as well as at the end: a run that dies midway still
        # leaves a record saying it never finished.
        _write_yaml(yaml_path, record, exclusive=True)
    except Exception:
        for s in sources.values():
            s.close()
        merger.close()
        raise

    barrier = threading.Barrier(len(cams)) if len(cams) > 1 else None
    pipelines = {c: CameraPipeline(views[c], sources[c], merger, render=show is not None,
                                   barrier=barrier, is_primary=(c == primary),
                                   render_every_s=render_every_s)
                 for c in cams}
    cancelled = False
    t0 = time.monotonic()
    t_print = None
    try:
        for p in pipelines.values():
            p.start()
        while any(p.thread.is_alive() for p in pipelines.values()):
            prog = _progress(pipelines[primary], total, t0)
            if show is not None:
                if show(pipelines, prog):
                    cancelled = True
                    break
            else:
                time.sleep(0.05)
                if time.monotonic() - (t_print or t0) > 1.0:
                    t_print = time.monotonic()
                    print('  %s  %s   ' % (job.run_name, progress_line(prog)), end='\r', flush=True)
    except KeyboardInterrupt:
        cancelled = True
    finally:
        for p in pipelines.values():
            p.stop()
        for s in sources.values():
            s.close()
        merger.close()          # Closes the logger once every queued row is written.
    if t_print is not None:
        print()
    duration = time.monotonic() - t0
    # Templates a tracker cut for itself on frame 0 (no tuning session saved
    # any) are this analysis's head zero: keep them with its configs.
    for view in views.values():
        for part in view.landmark_parts():
            path = view.templates_path(part)
            if not os.path.exists(path):
                getattr(view.fly, part).save_templates(path)

    errors = {c: repr(p.error) for c, p in pipelines.items() if p.error is not None}
    complete = not cancelled and not errors and any(p.eof for p in pipelines.values())
    record['status'] = 'complete' if complete else ('cancelled' if cancelled else 'failed')
    record['completed'] = complete
    record['frames_analyzed'] = {c: int(p.n_frames) for c, p in pipelines.items()}
    record['rows'] = int(logger.n)
    record['log_dropped'] = int(logger.n_dropped)
    if errors:
        record['errors'] = errors
    record['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    record['duration_s'] = round(duration, 2)
    record['tracked'] = _tracked_fractions(h5_path, views) if logger.n else {}
    files = {'flystate': os.path.basename(h5_path)}
    # The HDF5 is the record; a copy or a figure that fails is a note on it.
    if csv and logger.n:
        files['csv'] = []
        for cam in cams:
            try:
                h5_to_csv(h5_path, job.path('_%s.csv' % cam), group=cam)
                files['csv'].append(os.path.basename(job.path('_%s.csv' % cam)))
            except Exception as e:
                record['warnings'].append('%s CSV not written: %s' % (cam, e))
                log('WARNING: %s CSV not written: %s' % (cam, e))
    if figure and logger.n:
        ok, why = render_figure(h5_path, job.path('_traces.png'))
        if ok:
            files['figure'] = os.path.basename(job.path('_traces.png'))
        else:
            record['warnings'].append(why)
            log('WARNING: %s' % why)
    files['configs'] = [os.path.basename(job.config_path(c)) for c in cams]
    record['files'] = files
    _write_yaml(yaml_path, record, exclusive=False)
    return record


def summary_lines(record):
    """A few lines for the terminal after each analysis."""
    name = record['name']
    head = '== %s -> %s' % (record['requested_name'], record['output_dir'])
    if name != record['requested_name']:
        head += '   (%s was taken, so this is %s)' % (record['requested_name'], name)
    lines = [head]
    frames = record.get('frames_analyzed') or {}
    dur = record.get('duration_s') or 0.0
    rows = record.get('rows', 0)
    lines.append('   %s: %d rows from %s in %.1f s (%.0f frames/s)' % (
        record['status'].upper(), rows,
        ', '.join('%s %d frames' % kv for kv in frames.items()), dur,
        rows / dur if dur > 0 else 0.0))
    tracked = record.get('tracked') or {}
    bits = []
    for cam, fr in tracked.items():
        if fr:
            bits.append('%s %s' % (cam, '  '.join('%s %.1f%%' % (p, 100 * v) for p, v in fr.items())))
    if bits:
        lines.append('   finite angles: ' + '  |  '.join(bits))
    for w in record.get('warnings') or []:
        lines.append('   WARNING: ' + w)
    for cam, e in (record.get('errors') or {}).items():
        lines.append('   ERROR %s: %s' % (cam, e))
    if record.get('log_dropped'):
        lines.append('   LOG DROPPED %d rows' % record['log_dropped'])
    files = record.get('files') or {}
    shown = [files.get('flystate')] + list(files.get('csv') or []) + [files.get('figure')]
    lines.append('   files: %s, %s_analysis.yaml, per-camera config/background/reference'
                 % (', '.join(f for f in shown if f), name))
    return lines
