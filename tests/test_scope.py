"""Kinefly2 Scope: the shared data layer (files, CSVs, channel groups, the live
buffer) without Qt, then the window itself in an offscreen subprocess (as it
runs in practice, away from OpenCV) -- skipped where PySide6/pyqtgraph are not
installed."""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

VID_CFG = os.path.join(HERE, 'videos', 'synthetic_bottom.yaml')
VID = os.path.join(HERE, 'videos', 'synthetic_bottom.avi')


def ensure_synthetic():
    subprocess.check_call([sys.executable, os.path.join(HERE, 'make_synthetic_video.py')])


def pair_analysis():
    """A bottom+front offline analysis of the synthetic video (the same video
    standing in for both cameras). Returns the job."""
    from kinefly2.core.offline import VideoJob, finalize, prepare_setup, run_analysis
    job = VideoJob(name='scopetest', videos={'bottom': VID, 'front': VID})
    prepare_setup(job, {'bottom': VID_CFG, 'front': VID_CFG}, log=lambda *a: None)
    finalize(job, tempfile.mkdtemp())
    rec = run_analysis(job, figure=False, log=lambda *a: None)
    assert rec['completed'], rec
    return job


# ------------------------------------------------------------------ data layer

def test_h5_and_csv_exports_read_the_same():
    """Whatever file of an analysis is opened -- the HDF5, a per-camera CSV,
    or the combined CSV -- the Scope must show the same numbers."""
    ensure_synthetic()
    from kinefly2.output.csv_export import h5_to_csv
    from kinefly2.scope.data import load_recording
    job = pair_analysis()
    h5 = load_recording(job.path('_flystate.h5'))
    assert h5.name == job.run_name and h5.info['primary'] == 'bottom' and h5.info['primary_known']
    assert len(h5.t) == 300 and abs(h5.t[1] - h5.t[0] - 1 / 90.0) < 1e-6
    for cam in ('bottom', 'front'):
        c = load_recording(job.path('_%s.csv' % cam))
        assert c.info['cameras'] == [cam], c.info['cameras']
        for part in ('left', 'right', 'head'):
            k = '%s/%s' % (cam, part)
            assert np.allclose(c.columns[k], h5.columns[k], equal_nan=True), k
        assert np.allclose(c.t, h5.t)
    both = os.path.join(tempfile.mkdtemp(), 'both.csv')
    h5_to_csv(job.path('_flystate.h5'), both)
    b = load_recording(both)
    assert sorted(b.info['cameras']) == ['bottom', 'front']
    assert np.allclose(b.columns['front/left'], h5.columns['front/left'], equal_nan=True)
    # Angles are degrees on screen: the synthetic wing sweeps +-30 deg.
    left = h5.columns['bottom/left'][np.isfinite(h5.columns['bottom/left'])]
    assert 40 < np.ptp(left) < 80, np.ptp(left)


def test_generic_csv_plots_every_numeric_column():
    from kinefly2.scope.data import extra_groups, load_recording
    p = os.path.join(tempfile.mkdtemp(), 'my_trial.csv')
    with open(p, 'w') as f:
        f.write('time,signal,other,note\n')
        for i in range(50):
            f.write('%.3f,%.4f,%d,x\n' % (i * 0.01, math.sin(i / 5.0), i))
    r = load_recording(p)
    assert r.info['time_column'] == 'time' and abs(r.t[10] - 0.1) < 1e-9
    assert sorted(r.columns) == ['csv/other', 'csv/signal'], sorted(r.columns)   # 'note' is text
    groups = extra_groups(r.columns)
    assert [g.title for g in groups] == ['csv/other', 'csv/signal']
    assert all(g.unit == '' for g in groups), 'no unit is guessed for a generic column'


def test_groups_follow_what_the_file_holds():
    """Fallbacks and extras: a head tracked by a hinge tracker (no landmark
    yaw) still shows under head yaw; an uncalibrated pitch shows in px; a
    channel no group names gets its own panel; 0 Hz is no wingbeat."""
    from kinefly2.scope.data import GROUPS, extra_groups, resolve, row_from_state, series
    n = 20
    cols = {'bottom/left': np.full(n, 50.0), 'bottom/right': np.full(n, 40.0),
            'bottom/head': np.linspace(-5, 5, n), 'bottom/head_yaw': np.full(n, np.nan),
            'front/head_pitch': np.full(n, np.nan), 'front/head_d_axial': np.full(n, 2.0),
            'front/left': np.full(n, 70.0), 'bottom/head_q': np.full(n, 0.9),
            'front/head_q': np.full(n, 0.8)}
    g = {x.id: x for x in GROUPS}
    assert resolve(g['byaw'], cols) == [('yaw', '#7fe07f', 'bottom/head', 'deg')]
    assert resolve(g['fpitch'], cols)[0][2:] == ('front/head_d_axial', 'px')
    assert [t[0] for t in resolve(g['q'], cols)] == ['bottom', 'front']
    assert resolve(g['lmr'], cols)[0][2] == 'L-R'
    assert np.allclose(series(cols, 'L-R'), 10.0)          # derived from the two wings
    assert [x.title for x in extra_groups(cols)] == ['front/left']
    assert extra_groups(cols)[0].unit == 'deg'
    live = row_from_state({'seq': 0, 't_host': 1.0, 'primary': 'bottom', 'cameras': {
        'bottom': {'t_camera': 0.5, 'parts': {'aux': {'freq': 0.0}}}}})
    assert live['bottom/wbf'] is None, '0 Hz must be a gap, not a frequency'


def test_live_buffer_markers_ring_and_restart():
    from kinefly2.scope.data import LiveBuffer
    buf = LiveBuffer('tcp://unused', history_s=1.0, max_rate_hz=50.0, connect=False)  # 50 rows
    for i in range(120):
        buf.push({'t': 10 + i * 0.01, 't_host': float(i), 'seq': i, 'stale': None,
                  'rec': 'run1' if 30 <= i < 90 else None,
                  'stim': 1.0 if i < 60 else 2.0, 'bottom/left': float(i)})
    t, cols = buf.window(['bottom/left', 'L-R'])
    assert len(t) == 50 and np.all(np.diff(t) > 0), 'the ring wraps, oldest first'
    assert cols['bottom/left'][0] == 70 and 'L-R' not in cols
    kinds = [(k, lbl) for _, k, lbl in buf.status()['markers']]
    assert kinds == [('rec', 'stop')], kinds      # older markers left with their rows
    assert buf.status()['origin'] == 10.0
    # run_rig restarted: the frame clock goes back -- a fresh buffer.
    buf.push({'t': 0.0, 't_host': 200.0, 'seq': 0, 'stale': None, 'rec': None, 'stim': 3.0,
              'bottom/left': 1.0})
    assert buf.resets == 1 and len(buf.window(['bottom/left'])[0]) == 1
    assert buf.status()['origin'] == 0.0 and buf.status()['markers'][0][1] == 'reset'


def test_scope_modules_never_load_opencv():
    """The Scope's Qt must not share a process with OpenCV's (on Linux the
    OpenCV wheels bring their own Qt), so nothing the Scope imports may pull
    cv2 in -- nor may the helpers analyze_videos and run_rig use to start it."""
    from kinefly2.scope import launch
    mods = ['kinefly2.scope.data', 'kinefly2.scope.launch', 'kinefly2.apps.analyze_launcher']
    if launch.available():
        mods += ['kinefly2.scope.view', 'kinefly2.scope.window', 'kinefly2.scope.export',
                 'kinefly2.apps.scope']
    code = 'import sys\n' + '\n'.join('import %s' % m for m in mods) + \
        "\nassert 'cv2' not in sys.modules, 'cv2 was imported'\nprint('ok')"
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
    assert r.returncode == 0 and r.stdout.decode().strip().endswith('ok'), r.stdout.decode()


# ------------------------------------------------------------------ the window

GUI = r'''
import json, math, os, sys, tempfile, threading, time
import numpy as np
from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication(['test'])
from kinefly2.scope.data import load_recording
from kinefly2.scope.export import export_view
from kinefly2.scope.window import InstanceServer, MainWindow, send_to_running
h5, other, server_name = sys.argv[1], sys.argv[2], sys.argv[3]
out = {}
settings = QtCore.QSettings(os.path.join(tempfile.mkdtemp(), 's.ini'), QtCore.QSettings.Format.IniFormat)
win = MainWindow(settings); win.resize(1200, 900); win.show()
server = InstanceServer(win, name=server_name)
def pump(n=5):
    for _ in range(n):
        app.processEvents(); time.sleep(0.02)
(v,) = win.open_paths([h5]); pump()
out['panels'] = [p.group.id for p in v.panels]
# Zoomed to half a second the curves still hold their samples.
v.set_x_range(1.0, 1.5); pump()
out['zoomed_points'] = min(int(np.isfinite(p.curves[0].getData()[1]).sum()) for p in v.panels)
# Cursor readouts are the recording's own samples.
v.set_x_range(0.0, 3.3); win.a_cursors.trigger(); v.set_cursor(0, 0.5); v.set_cursor(1, 2.0); pump()
rec = load_recording(h5)
x = rec.t - rec.t[0]
want = rec.columns['bottom/left'][int(np.argmin(np.abs(x - 2.0)))]
row = [r for r in v.readout() if r[0].endswith('· L')][0]
out['cursor_ok'] = bool(abs(row[3] - want) < 1e-9)
seg = rec.columns['bottom/left'][(x >= 0.5) & (x <= 2.0)]
out['mean_ok'] = bool(abs(row[5] - np.nanmean(seg)) < 1e-9)
# Exports in every format.
d = tempfile.mkdtemp()
for ext in ('png', 'svg', 'pdf'):
    export_view(v, os.path.join(d, 'fig.' + ext))
    out['export_' + ext] = os.path.getsize(os.path.join(d, 'fig.' + ext))
# Another program hands a file to this window.
out['handoff'] = send_to_running([other], name=server_name)
pump(10)
out['tabs_after_handoff'] = win.tabs.count()
# Live: a fake run_rig publisher.
import zmq
port = int(sys.argv[4])
pub = zmq.Context.instance().socket(zmq.PUB); pub.bind('tcp://127.0.0.1:%d' % port)
stop = []
def publish():
    i = 0
    while not stop:
        t = i / 90.0
        pub.send_string(json.dumps({'seq': i, 't_host': time.monotonic(), 'primary': 'bottom',
            'stim': float(int(t) % 2), 'rec': 'live1', 'ages': {'bottom': 0.001}, 'stale_after_s': 0.055,
            'cameras': {'bottom': {'t_camera': t, 'parts': {
                'left': {'angles': [math.radians(60 + 20 * math.sin(t))]},
                'right': {'angles': [math.radians(55)]}}}}}))
        i += 1; time.sleep(1 / 90.0)
th = threading.Thread(target=publish, daemon=True); th.start()
lv = win.open_live('tcp://127.0.0.1:%d' % port)
end = time.monotonic() + 3.0
while time.monotonic() < end:
    app.processEvents(); time.sleep(0.01)
out['live_rx'] = lv.source.n_rx
out['live_panels'] = [p.group.id for p in lv.panels]
out['live_points'] = int(np.isfinite(lv.panels[0].curves[0].getData()[1]).sum()) if lv.panels else 0
stop.append(1); th.join()
win.close()
print('RESULT ' + json.dumps(out))
'''


def test_scope_window_offscreen():
    ensure_synthetic()
    from kinefly2.scope import launch
    if not launch.available():
        print('PySide6/pyqtgraph not installed; skipping the Scope window test')
        return
    job = pair_analysis()
    other = job.path('_bottom.csv')
    r = subprocess.run([sys.executable, '-c', GUI, job.path('_flystate.h5'), other,
                        'kinefly2-scope-test-%d' % os.getpid(), '5741'],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=dict(os.environ, QT_QPA_PLATFORM='offscreen'), timeout=240)
    text = r.stdout.decode()
    lines = [l for l in text.splitlines() if l.startswith('RESULT ')]
    assert r.returncode == 0 and lines, text[-3000:]
    out = json.loads(lines[-1][len('RESULT '):])
    assert {'wsa', 'lmr', 'byaw'} <= set(out['panels']), out['panels']
    assert out['zoomed_points'] >= 40, 'zoomed in, curves must keep their samples: %s' % out
    assert out['cursor_ok'] and out['mean_ok'], out
    assert out['export_png'] > 20000 and out['export_svg'] > 10000 and out['export_pdf'] > 5000, out
    assert out['handoff'] and out['tabs_after_handoff'] == 2, out
    assert out['live_rx'] > 150 and 'wsa' in out['live_panels'] and out['live_points'] > 100, out
    print('scope window: %s' % out)


def test_figure_command():
    ensure_synthetic()
    from kinefly2.scope import launch
    if not launch.available():
        print('PySide6/pyqtgraph not installed; skipping the figure test')
        return
    job = pair_analysis()
    png = job.path('_traces.png')
    ok, why = launch.render_figure(job.path('_flystate.h5'), png)
    assert ok, why
    import cv2
    img = cv2.imread(png)
    assert img is not None and img.shape[1] == 1600 and img.shape[0] > 400, None if img is None else img.shape
    assert img.mean() > 200, 'a light figure: mostly white'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('%s OK' % name)
    print('OK')
