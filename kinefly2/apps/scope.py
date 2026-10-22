"""Kinefly2 Scope: live and saved traces, Axoscope-style.

    python -m kinefly2.apps.scope                           # empty window: File > Open, or drop files
    python -m kinefly2.apps.scope run_flystate.h5 other.h5  # one tab each
    python -m kinefly2.apps.scope --live                    # this machine's run_rig
    python -m kinefly2.apps.scope --live tcp://<rig-ip>:5556
    python -m kinefly2.apps.scope --figure out.png run_flystate.h5   # save a figure, no window

Stacked, time-linked panels -- wing stroke amplitude, L-R, head yaw / roll /
pitch, abdomen, legs, stimulus, ... -- with an overview strip, two measurement
cursors and the stimulus/recording/event markers. Opening files while a Scope
is already running hands them to that window as new tabs (this is how each
offline analysis shows up in one window); --new opens a separate window.

Needs PySide6 and pyqtgraph (pip install "PySide6<6.12" pyqtgraph). README section 2.4.1.
"""
import argparse
import os
import sys

from ..scope.launch import DEFAULT_LIVE, INSTALL_HINT


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='Recordings to open (.h5 or .csv)')
    ap.add_argument('--live', nargs='?', const=DEFAULT_LIVE, metavar='ADDR',
                    help="Watch run_rig's fly-state stream (default %s)" % DEFAULT_LIVE)
    ap.add_argument('--figure', metavar='OUT',
                    help='Save a figure (.png/.svg/.pdf) of every trace in the one recording '
                         'given, without opening a window')
    ap.add_argument('--new', action='store_true',
                    help='Open a new window even if a Scope is already running')
    args = ap.parse_args()

    try:
        import pyqtgraph  # noqa: F401
        from PySide6 import QtWidgets
    except ImportError as e:
        raise SystemExit('The Scope needs PySide6 and pyqtgraph: %s  (%s)' % (INSTALL_HINT, e))

    if args.figure:
        if len(args.files) != 1:
            ap.error('--figure needs exactly one recording')
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from ..scope.export import render_figure
        render_figure(args.files[0], args.figure)
        print('wrote %s' % args.figure)
        return 0

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('Kinefly2 Scope')
    from ..scope.window import InstanceServer, MainWindow, send_to_running

    paths = [os.path.abspath(p) for p in args.files]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        ap.error('not found: %s' % ', '.join(missing))
    if not args.new and send_to_running(paths, args.live):
        return 0                   # the Scope that is already open took them
    win = MainWindow()
    server = None if args.new else InstanceServer(win)   # noqa: F841  (kept alive with win)
    win.show()
    win.open_paths(paths)
    if args.live:
        win.open_live(args.live)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
