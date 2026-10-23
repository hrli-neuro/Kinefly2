"""Starting the Scope from other programs, without importing Qt here.

analyze_videos and run_rig drive OpenCV windows, and the Scope's Qt must live
in a process of its own: on Linux the OpenCV wheels carry their own Qt, and two
Qts in one process fight. So those programs start the Scope with these helpers,
and the figure saved after each analysis is drawn by a short-lived offscreen
Scope process.
"""
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LIVE = 'tcp://localhost:5556'
INSTALL_HINT = 'pip install "PySide6<6.12" pyqtgraph'   # 6.12.0 crashes the Scope on Python 3.10


def available():
    """True when the Scope can run: its libraries (PySide6, pyqtgraph) are
    installed and the Scope itself is part of this checkout."""
    return all(importlib.util.find_spec(m) is not None
               for m in ('PySide6', 'pyqtgraph', 'kinefly2.apps.scope'))


def _env(**extra):
    env = dict(os.environ, **extra)
    env['PYTHONPATH'] = ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def _cmd(*args):
    return [sys.executable, '-m', 'kinefly2.apps.scope'] + [str(a) for a in args]


def open_scope(*paths, live=None):
    """Show files and/or a live stream in the Scope: in the Scope window that
    is already open, if there is one, else in a new one that outlives the
    caller. Returns False when the Scope is not installed."""
    if not available():
        return False
    args = [os.path.abspath(p) for p in paths] + (['--live', live] if live else [])
    subprocess.Popen(_cmd(*args), env=_env(), stdout=subprocess.DEVNULL,
                     start_new_session=True)
    return True


def render_figure(h5_path, out_path, timeout=180):
    """Save a figure of every trace in a recording, drawn offscreen by the
    Scope. Returns (ok, message) and never raises: a missing figure must not
    fail the analysis it belongs to."""
    if not available():
        return False, 'no figure: the Scope is not installed (%s)' % INSTALL_HINT
    try:
        r = subprocess.run(_cmd('--figure', out_path, h5_path),
                           env=_env(QT_QPA_PLATFORM='offscreen'),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, 'no figure: %s' % e
    if r.returncode == 0 and os.path.exists(out_path):
        return True, ''
    tail = [l for l in r.stdout.decode(errors='replace').strip().splitlines() if l][-3:]
    return False, 'no figure: %s' % (' | '.join(tail) or 'exit status %d' % r.returncode)
