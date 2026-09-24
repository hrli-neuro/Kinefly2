"""Analyze recorded videos -- no camera needed.

    python -m kinefly2.apps.analyze_videos                  # opens the launcher window
    python -m kinefly2.apps.analyze_videos A_cam1.mp4 A_cam2.mp4 --out ~/Data/offline
    python -m kinefly2.apps.analyze_videos *.avi --out DIR --no-gui    # batch, no tuning

A job is one video, or the bottom+front pair of one recording (paired by name:
X_cam1 + X_cam2, X_bottom + X_front; cam N is the N-th camera in rig.yaml).
Each job is first TUNED -- the usual camera windows with a player to pause,
slow down (0.25x-2x), step and scrub -- then ANALYZED from frame 0 with fresh
trackers into

    <out>/<name>/   <name>_flystate.h5, <name>_flystate.csv, <name>_analysis.yaml, configs/

A name that already exists in <out> gets _001, _002, ...: nothing is ever
overwritten. The rig config is never modified either: the first job starts from
it (or from a previous analysis, --from), each later job from the job before,
and every analysis keeps the exact configs it used. See README section 2.1.2.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import traceback

import cv2
import yaml

from ..core.naming import group_videos, preview_name
from ..core.offline import (VideoJob, discard_setup, finalize, prepare_setup,
                            run_analysis, summary_lines)
from ..gui.player import AnalysisDisplay, PlayerPanel, SetupSession
from ..scope import launch as scope
from .run_rig import parse_cameras

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RIG = os.path.join(ROOT, 'config', 'rig.yaml')


def load_rig(rig_path):
    """{camera: its YAML} for every camera in rig.yaml, in rig order -- enabled
    or not, since a camera switched off on the rig can still have videos -- and
    the primary camera."""
    with open(rig_path) as f:
        rig = yaml.safe_load(f) or {}
    names = list(rig.get('cameras') or {})
    if not names:
        raise SystemExit('%s names no cameras' % rig_path)
    cams = parse_cameras(rig, os.path.dirname(os.path.abspath(rig_path)), only=names)
    primary = rig.get('primary') if rig.get('primary') in cams else names[0]
    return cams, primary


def analysis_seeds(folder, cameras):
    """{camera: YAML} from a previous analysis: its <run>_<cam>_config.yaml,
    or configs/camera_<cam>.yaml in one made before the folders were
    flattened."""
    seeds = {}
    for cam in cameras:
        flat = sorted(glob.glob(os.path.join(glob.escape(folder), '*_%s_config.yaml' % cam)))
        old = os.path.join(folder, 'configs', 'camera_%s.yaml' % cam)
        if flat:
            seeds[cam] = flat[0]
        elif os.path.exists(old):
            seeds[cam] = old
    if not seeds:
        raise SystemExit('no <run>_<camera>_config.yaml (or configs/camera_<camera>.yaml) in %s'
                         % folder)
    return seeds


def run_launcher(rig_path):
    """Show the launcher window (in its own process: Tk and OpenCV's windows
    must not share the macOS event loop). Returns the settings, or None."""
    fd, out = tempfile.mkstemp(prefix='kinefly2_jobs_', suffix='.json')
    os.close(fd)
    env = dict(os.environ)
    env['PYTHONPATH'] = ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    try:
        r = subprocess.run([sys.executable, '-m', 'kinefly2.apps.analyze_launcher',
                            '--rig', rig_path, '--out-json', out], env=env)
        if r.returncode != 0:
            return None
        with open(out) as f:
            text = f.read()
        return json.loads(text) if text.strip() else None
    finally:
        if os.path.exists(out):
            os.remove(out)


def close_windows():
    cv2.destroyAllWindows()
    for _ in range(5):          # macOS: let the windows actually go away
        cv2.waitKey(1)


def run_queue(settings, rig_cams, primary, gui=True, setup=True, fps=None):
    """Tune and analyze each job in turn. Returns the analysis records."""
    out_dir = os.path.abspath(os.path.expanduser(settings['out_dir']))
    jobs = [VideoJob(name=j['name'], videos=dict(j['videos']), fps=fps) for j in settings['jobs']]
    seed = settings.get('seed') or {}
    seeds = analysis_seeds(seed['path'], rig_cams) if seed.get('kind') == 'analysis' else {}
    panel = PlayerPanel() if gui else None
    records = []
    print('%d job%s -> %s' % (len(jobs), '' if len(jobs) == 1 else 's', out_dir))
    try:
        for i, job in enumerate(jobs):
            tag = '%d/%d' % (i + 1, len(jobs))
            print('\n[%s] %s: %s' % (tag, job.name, ', '.join(
                '%s=%s' % (c, os.path.basename(v)) for c, v in job.videos.items())))
            start = {c: seeds.get(c) or rig_cams.get(c) for c in job.videos}
            try:
                prepare_setup(job, start)
            except Exception as e:
                print('[%s] %s: cannot start: %s' % (tag, job.name, e))
                continue
            if setup:
                session = SetupSession(job, primary=primary, speed=settings.get('speed', 0.5),
                                       saves_as=os.path.join(out_dir, preview_name(out_dir, job.name)),
                                       title=tag, panel=panel)
                try:
                    result = session.run()
                except Exception:
                    traceback.print_exc()
                    result = 'skip'
                if result != 'analyze':
                    discard_setup(job)
                    print('[%s] %s: %s' % (tag, job.name,
                                           'skipped' if result == 'skip' else 'stopped here'))
                    if result == 'quit':
                        break
                    continue
            finalize(job, out_dir)
            display = None
            if gui:
                for cam, video in job.videos.items():
                    try:
                        cv2.setWindowTitle('kinefly2: %s' % cam, '%s: %s   [analyzing %s]'
                                           % (cam, os.path.basename(video), tag))
                    except cv2.error:
                        pass
                display = AnalysisDisplay(panel, title='%s -> %s' % (job.name, job.run_name))
            try:
                record = run_analysis(job, primary=primary, csv=settings.get('csv', True),
                                      show=display)
            except Exception:
                traceback.print_exc()
                print('[%s] %s: analysis failed; whatever was written is in %s'
                      % (tag, job.name, job.out_dir))
                continue
            records.append(record)
            print('\n'.join(summary_lines(record)))
            # Every trace of this analysis, in the Scope: a new tab in the
            # window already open, or a new window that stays after this exits.
            if gui and record['rows']:
                if not scope.open_scope(job.path('_flystate.h5')):
                    print('   (install the Scope to see the traces here: %s)' % scope.INSTALL_HINT)
            # The next job starts from this one's tuning.
            for cam in job.videos:
                seeds[cam] = job.config_path(cam)
            if display is not None and display.quit:
                break
    finally:
        for job in jobs:
            discard_setup(job)
        if gui:
            close_windows()
    if records:
        print('\n%d %s written:' % (len(records), 'analysis' if len(records) == 1 else 'analyses'))
        for r in records:
            print('  %-9s %s' % (r['status'], r['output_dir']))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('videos', nargs='*', help='Video files (none: open the launcher window)')
    ap.add_argument('--out', help='Folder to save results in (required with videos)')
    ap.add_argument('--name', help='Name for the analysis (only with a single job)')
    ap.add_argument('--rig', default=DEFAULT_RIG, help='rig.yaml naming the cameras (default: %(default)s)')
    ap.add_argument('--from', dest='seed_dir', metavar='DIR',
                    help='Start tuning from this previous analysis instead of the rig config')
    ap.add_argument('--camera', help='Camera for videos whose name does not say (default: primary)')
    ap.add_argument('--fps', type=float, help="Override the videos' frame rate")
    ap.add_argument('--speed', type=float, default=0.5, help='Setup playback speed (default 0.5)')
    ap.add_argument('--no-csv', action='store_true', help='Do not write the CSV copy')
    ap.add_argument('--no-setup', action='store_true',
                    help='Skip tuning: analyze with the starting configs as they are')
    ap.add_argument('--no-gui', action='store_true', help='No windows at all (implies --no-setup)')
    args = ap.parse_args()

    rig_cams, primary = load_rig(args.rig)
    if args.camera and args.camera not in rig_cams:
        ap.error('--camera %s: %s names %s' % (args.camera, args.rig, ', '.join(rig_cams)))
    gui = not args.no_gui
    setup = gui and not args.no_setup

    if args.videos:
        if not args.out:
            ap.error('--out is required when videos are given')
        missing = [v for v in args.videos if not os.path.isfile(v)]
        if missing:
            ap.error('not found: %s' % ', '.join(missing))
        jobs = group_videos(args.videos, list(rig_cams), args.camera or primary)
        if args.name:
            if len(jobs) != 1:
                ap.error('--name needs exactly one job; these videos make %d' % len(jobs))
            jobs[0]['name'] = args.name
        settings = {'jobs': jobs, 'out_dir': args.out, 'csv': not args.no_csv,
                    'speed': args.speed,
                    'seed': {'kind': 'analysis', 'path': args.seed_dir} if args.seed_dir
                    else {'kind': 'rig'}}
        records = run_queue(settings, rig_cams, primary, gui=gui, setup=setup, fps=args.fps)
        return 0 if records and all(r['completed'] for r in records) else 1

    if not gui:
        ap.error('--no-gui needs video files on the command line')
    # The launcher comes back after each queue, until it is closed.
    while True:
        settings = run_launcher(args.rig)
        if settings is None:
            return 0
        run_queue(settings, rig_cams, primary, gui=True, setup=setup, fps=args.fps)


if __name__ == '__main__':
    sys.exit(main())
