"""Analyzing recorded videos (apps/analyze_videos): names that never overwrite,
grouping files into jobs, seeking, the player's pacing, a scripted tuning
session, and the analysis itself -- one camera and a lockstep pair -- against
the synthetic ground truth. No window is opened: the setup loop is driven
through a stand-in for cv2's window functions.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

VID_CFG = os.path.join(HERE, 'videos', 'synthetic_bottom.yaml')
VID = os.path.join(HERE, 'videos', 'synthetic_bottom.avi')


def ensure_synthetic():
    subprocess.check_call([sys.executable, os.path.join(HERE, 'make_synthetic_video.py')])


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def quiet(*args, **kwargs):
    pass


def run_job(out, name, videos, seeds, **kwargs):
    from kinefly2.core.offline import VideoJob, finalize, prepare_setup, run_analysis
    job = VideoJob(name=name, videos=videos)
    prepare_setup(job, seeds, log=quiet)
    finalize(job, out)
    kwargs.setdefault('figure', False)       # drawn by the Scope; see test_scope.py
    return job, run_analysis(job, log=quiet, **kwargs)


# ------------------------------------------------------------------ naming

def test_reserve_dir_never_overwrites():
    from kinefly2.core.naming import preview_name, reserve_dir
    root = tempfile.mkdtemp()
    first = reserve_dir(root, 'fly1')
    with open(os.path.join(first, 'result.txt'), 'w') as f:
        f.write('first result')
    assert preview_name(root, 'fly1') == 'fly1_001'
    second = reserve_dir(root, 'fly1')
    third = reserve_dir(root, 'fly1')
    assert [os.path.basename(p) for p in (first, second, third)] == ['fly1', 'fly1_001', 'fly1_002']
    assert os.listdir(second) == [] and os.listdir(third) == [], 'a claimed folder starts empty'
    with open(os.path.join(first, 'result.txt')) as f:
        assert f.read() == 'first result'
    # A file of that name blocks it too -- claimed as a folder, never touched.
    open(os.path.join(root, 'fly2'), 'w').close()
    assert os.path.basename(reserve_dir(root, 'fly2')) == 'fly2_001'
    # Two jobs of one name in the same queue get two folders.
    assert preview_name(root, 'new', taken=['new']) == 'new_001'
    # A name cannot climb out of the save folder.
    assert os.path.dirname(reserve_dir(root, '../escape')) == root


def test_group_videos_pairs_one_recording():
    from kinefly2.core.naming import group_videos
    cams = ['bottom', 'front']
    jobs = group_videos(['/d/260910_fly119_cam1.mp4', '/d/260910_fly119_cam2.mp4',
                         '/d/video_fly048_020.mov', '/d/r_front.avi', '/d/r_bottom.avi',
                         '/e/solo_front.avi', '/d/fly_cam10.mp4'], cams, 'bottom')
    got = [(j['name'], list(j['videos'])) for j in jobs]
    assert got == [('260910_fly119', ['bottom', 'front']), ('video_fly048_020', ['bottom']),
                   ('r', ['bottom', 'front']), ('solo', ['front']), ('fly_cam10', ['bottom'])], got
    assert jobs[0]['videos'] == {'bottom': '/d/260910_fly119_cam1.mp4',
                                 'front': '/d/260910_fly119_cam2.mp4'}
    # One name in two folders is two recordings, not a pair.
    assert len(group_videos(['/a/x_bottom.avi', '/b/x_front.avi'], cams)) == 2
    # A second video for a camera starts a new job instead of replacing the first.
    assert len(group_videos(['/a/x_bottom.avi', '/a/x_bottom.mp4'], cams)) == 2


def test_launcher_model():
    from kinefly2.apps.analyze_launcher import LauncherModel
    root = tempfile.mkdtemp()
    os.mkdir(os.path.join(root, 'A'))             # an earlier result called A
    vids = []
    for n in ('A_cam1.mp4', 'A_cam2.mp4', 'B.mov'):
        vids.append(os.path.join(root, n))
        open(vids[-1], 'w').close()
    m = LauncherModel(['bottom', 'front'], 'bottom', {'out_dir': root})
    assert m.add_videos(vids) == 2
    m.add_videos([vids[2]])                       # the same video again: its own folder
    assert m.previews() == ['A_001', 'B', 'B_001'], m.previews()
    assert m.problems() == []
    m.rename(2, 'B/../x')
    assert m.jobs[2]['name'] == 'B_.._x'
    m.swap(0)
    assert m.jobs[0]['videos']['bottom'].endswith('A_cam2.mp4')
    m.swap(1)
    assert list(m.jobs[1]['videos']) == ['front']
    json.dumps(m.settings(os.path.join(ROOT, 'config', 'rig.yaml')))
    m.seed_kind, m.seed_path = 'analysis', root
    assert any('config' in p for p in m.problems()), m.problems()


LAUNCHER_SMOKE = r'''
import json, os, sys, tempfile
import tkinter as tk
class HiddenTk(tk.Tk):          # built and driven, never shown
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.withdraw()
tk.Tk = HiddenTk
from kinefly2.apps import analyze_launcher as L
assert 'cv2' not in sys.modules, 'the launcher must not load OpenCV'
root = tempfile.mkdtemp()
os.mkdir(os.path.join(root, 'A'))
vids = [os.path.join(root, n) for n in ('A_cam1.mp4', 'A_cam2.mp4', 'B.mov')]
for v in vids:
    open(v, 'w').close()
m = L.LauncherModel(['bottom', 'front'], 'bottom', {})
w = L.LauncherWindow(m, sys.argv[1])
assert w.start_btn.instate(['disabled'])
m.add_videos(vids)
w.out_var.set(root)
rows = [w.tree.item(i, 'values') for i in w.tree.get_children()]
assert rows[0][-1].startswith('A_001') and rows[1][-1] == 'B', rows
assert not w.start_btn.instate(['disabled']), w.status.cget('text')
w.tree.selection_set(w.tree.get_children()[1])
w.on_select()
w.name_var.set('B2')
w.apply_name()
w.swap()
w.root.update()
w.start()
print(json.dumps(w.result))
'''


def test_launcher_window_builds():
    """The Tk window itself: built hidden in its own process (as it runs in
    production), rows, collision preview, rename and swap."""
    r = subprocess.run([sys.executable, '-c', LAUNCHER_SMOKE,
                        os.path.join(ROOT, 'config', 'rig.yaml')],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode()
    if r.returncode and 'TclError' in out and 'display' in out.lower():
        print('no display: launcher window not tested')
        return
    assert r.returncode == 0, out
    result = json.loads(out.strip().splitlines()[-1])
    assert [j['name'] for j in result['jobs']] == ['A', 'B2'], result['jobs']
    assert list(result['jobs'][1]['videos']) == ['front'], 'swap moved the single video'


# ------------------------------------------------------------------ player

def test_seek_is_exact():
    ensure_synthetic()
    from kinefly2.camera.video_source import VideoFileSource, median_background
    src = VideoFileSource(VID)
    src.open()
    assert src.n_frames == 300
    frames = [f.image.copy() for f in iter(src.read, None)]
    for k in (0, 1, 150, 298, 299, 7):
        src.seek(k)
        f = src.read()
        assert f.frame_id == k and abs(f.t_camera - k / src.fps) < 1e-9, (k, f.frame_id)
        assert np.array_equal(f.image, frames[k]), 'seek to %d gave another frame' % k
    src.close()
    bg, n_used, n_total = median_background(VID)
    assert (n_used, n_total) == (41, 300) and bg.shape == frames[0].shape


def test_transport_pacing():
    from kinefly2.gui.player import Transport
    t = Transport(300, 90.0, speed=0.5)
    assert abs(t.interval - 1 / 45.0) < 1e-12
    t.play(0.0)
    moved = sum(t.tick(k * 0.001) for k in range(1, 1501))      # 1.5 s in 1 ms steps
    assert moved == 67 and t.index == 67, moved                  # 45 frames/s
    # A stall does not make it race through frames to catch up.
    assert t.tick(5.0) and t.index == 68 and not t.tick(5.0)
    t.step(1)
    assert not t.playing and t.index == 69, 'stepping pauses'
    t.seek(10 ** 6)
    assert t.index == 299
    t.play(10.0)
    assert t.tick(10.1) and t.index == 0, 'loops at the end'
    t.faster(), t.faster(), t.faster()
    assert t.speed == 2.0
    t.slower()
    assert t.speed == 1.0
    t.jump(-1.0)
    assert t.index == 0
    t.seek(100)
    t.jump(1.0)
    assert t.index == 190
    t.truncate(150)
    assert t.n == 150 and t.index == 149


def test_keys():
    from kinefly2.gui.player import key_action
    assert key_action(-1) is None
    assert key_action(ord(' ')) == ('toggle',)
    assert key_action(13) == ('analyze',) and key_action(27) == ('skip',)
    assert key_action(63234) == ('step', -1) and key_action(65363) == ('step', 1)
    assert key_action(ord('q')) == ('quit',) and key_action(ord('Q')) is None
    assert key_action(ord(']')) == ('faster',) and key_action(ord('<')) == ('jump', -1.0)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeGui:
    """Stands in for cv2's window functions: keeps what was shown, hands out
    scripted key presses (a step may be a function that does mouse work
    first), and advances the fake clock by each wait."""

    def __init__(self, clock, script):
        self.clock = clock
        self.script = list(script)
        self.callbacks, self.shown, self.titles = {}, {}, {}
        self.idle = 0

    def namedWindow(self, name, flags=0):
        pass

    def moveWindow(self, name, x, y):
        pass

    def setWindowTitle(self, name, title):
        self.titles[name] = title

    def setMouseCallback(self, name, fn, param=None):
        self.callbacks[name] = fn

    def imshow(self, name, img):
        self.shown[name] = img

    def waitKeyEx(self, ms):
        self.clock.t += ms / 1000.0
        if not self.script:
            self.idle += 1
            assert self.idle < 500, 'the session outlived its script'
            return -1
        step = self.script.pop(0)
        if callable(step):
            step = step(self)
        return -1 if step is None else step

    def mouse(self, window, event, x, y):
        self.callbacks[window](event, int(round(x)), int(round(y)), 0, None)


def test_setup_session_scripted():
    """Play at 0.5x, step, drag a hinge while paused, scrub, withdraw an
    esc, ANALYZE -- then the analysis runs with exactly the tuned config, and
    the seed (rig) config is untouched throughout."""
    ensure_synthetic()
    import yaml
    from kinefly2.core.offline import VideoJob, finalize, prepare_setup, run_analysis
    from kinefly2.gui import player
    from kinefly2.gui.player import PlayerPanel, SetupSession

    seed_before = sha(VID_CFG)
    WIN = 'kinefly2: bottom'
    seen = {}

    def record(key, then=None):
        # Note the state at the start of an iteration, optionally handing out a
        # key in the same one (a step of its own would let playback go on).
        def step(gui):
            seen[key] = (session.transport.index, session.transport.playing)
            return then
        return step

    def drag_left_hinge(gui):
        h = session.views['bottom'].params['gui']['left']['hinge']
        seen['hinge'] = (h['x'], h['y'])
        gui.mouse(WIN, cv2.EVENT_LBUTTONDOWN, h['x'], h['y'])
        gui.mouse(WIN, cv2.EVENT_MOUSEMOVE, h['x'] + 6, h['y'] + 3)
        gui.mouse(WIN, cv2.EVENT_LBUTTONUP, h['x'] + 6, h['y'] + 3)

    def click_bar(frac):
        def step(gui):
            x0, y0, x1, y1 = session.panel.bar
            x, y = x0 + frac * (x1 - x0), (y0 + y1) / 2.0
            gui.mouse(PlayerPanel.WINDOW, cv2.EVENT_LBUTTONDOWN, x, y)
            gui.mouse(PlayerPanel.WINDOW, cv2.EVENT_LBUTTONUP, x, y)
        return step

    script = ([ord(' ')] + [None] * 20 + [record('played', then=ord('.'))]
              + [record('stepped')] + [None] * 5 + [record('held')]
              + [drag_left_hinge, None]
              + [click_bar(0.5), None, record('scrubbed')]
              + [27, ord('.'), 27, None, record('esc withdrawn')]
              + [13])
    clock = FakeClock()
    gui = FakeGui(clock, script)
    job = VideoJob(name='tuned', videos={'bottom': VID})
    prepare_setup(job, {'bottom': VID_CFG}, log=quiet)
    session = SetupSession(job, primary='bottom', speed=0.5, saves_as='somewhere',
                           title='1/1', panel=PlayerPanel(gui), gui=gui, clock=clock)
    assert session.run() == 'analyze'

    played, playing = seen['played']
    assert playing and 15 <= played <= 21, seen['played']          # ~1 frame per wait at 0.5x
    assert seen['stepped'] == (played + 1, False), seen
    assert seen['held'] == seen['stepped'], 'paused stays on its frame'
    assert seen['scrubbed'][0] == 150, seen['scrubbed']             # half-way along the bar
    assert seen['esc withdrawn'] == (151, False), 'esc, step, esc must not skip'
    assert gui.shown[WIN].ndim == 3 and gui.shown[PlayerPanel.WINDOW].shape == (134, 700, 3)
    assert 'setup 1/1' in gui.titles[WIN]
    assert gui.callbacks[WIN] is player._ignore_mouse, 'handles locked once tuning ends'

    tuned = yaml.safe_load(open(job.config_path('bottom')))['tracker']['gui']['left']['hinge']
    x0, y0 = seen['hinge']
    assert (tuned['x'], tuned['y']) == (x0 + 6, y0 + 3), tuned
    assert sha(VID_CFG) == seed_before, 'tuning must never touch the seed config'

    out = tempfile.mkdtemp()
    finalize(job, out)
    final = yaml.safe_load(open(os.path.join(out, 'tuned', 'tuned_bottom_config.yaml')))
    assert final['tracker']['gui']['left']['hinge'] == tuned
    rec = run_analysis(job, log=quiet, csv=False, figure=False)
    assert rec['completed'] and rec['rows'] == 300, rec


def test_skip_and_quit_need_two_presses():
    ensure_synthetic()
    from kinefly2.core.offline import VideoJob, discard_setup, prepare_setup
    from kinefly2.gui.player import PlayerPanel, SetupSession
    for keys, want in (([27, None, 27], 'skip'), ([ord('q'), ord('q')], 'quit')):
        clock = FakeClock()
        gui = FakeGui(clock, keys)
        job = VideoJob(name='x', videos={'bottom': VID})
        prepare_setup(job, {'bottom': VID_CFG}, log=quiet)
        s = SetupSession(job, panel=PlayerPanel(gui), gui=gui, clock=clock)
        assert s.run() == want
        discard_setup(job)
        assert job.setup_dir is None


# ------------------------------------------------------------------ analysis

def test_offline_analysis_single_camera():
    ensure_synthetic()
    import h5py
    import yaml
    from tests.test_tracking import corr, load_truth

    seed_before = sha(VID_CFG)
    out = tempfile.mkdtemp()
    job, rec = run_job(out, 'syn', {'bottom': VID}, {'bottom': VID_CFG})
    assert rec['completed'] and rec['status'] == 'complete', rec
    assert rec['rows'] == 300 and rec['log_dropped'] == 0, rec
    assert job.out_dir == os.path.join(out, 'syn')
    # One flat folder, every file labeled with the run and its camera.
    assert sorted(os.listdir(job.out_dir)) == [
        'syn_analysis.yaml', 'syn_bottom.csv', 'syn_bottom_background.png',
        'syn_bottom_config.yaml', 'syn_bottom_reference.png', 'syn_flystate.h5'], \
        sorted(os.listdir(job.out_dir))
    cfg = yaml.safe_load(open(os.path.join(job.out_dir, 'syn_bottom_config.yaml')))
    assert cfg['camera'] == {'kind': 'video', 'path': VID}, cfg['camera']

    h5 = os.path.join(job.out_dir, 'syn_flystate.h5')
    with h5py.File(h5, 'r') as f:
        left = f['bottom/left_angle'][:]
        assert list(f['bottom/seq'][:3]) == [0, 1, 2]
        assert json.loads(f.attrs['source'])['videos']['bottom'] == VID
        assert json.loads(f.attrs['rig'])['primary'] == 'bottom'
    truth = load_truth()
    r = corr(left[20:], np.radians(truth[20:300, 1]))
    assert r > 0.99, 'offline analysis does not follow ground truth (r=%.3f)' % r

    # The per-camera CSV: that camera's columns, unprefixed, the h5's values.
    with open(os.path.join(job.out_dir, 'syn_bottom.csv')) as f:
        rows = list(csv.reader(f))
    assert rows[0][:2] == ['seq', 't_camera'] and len(rows) == 301, rows[0][:4]
    assert not any(h.startswith(('bottom_', 'merged_')) for h in rows[0]), rows[0]
    col = rows[0].index('left_angle')
    assert np.allclose([float(x[col]) for x in rows[1:]], left, equal_nan=True)

    record = yaml.safe_load(open(os.path.join(job.out_dir, 'syn_analysis.yaml')))
    assert record['status'] == 'complete' and record['videos']['bottom']['frames'] == 300
    assert sha(VID_CFG) == seed_before, 'the seed (rig) config must never be modified'

    # The same name again: a new folder, and the first result untouched.
    before = (sha(h5), os.path.getmtime(h5))
    job2, rec2 = run_job(out, 'syn', {'bottom': VID}, {'bottom': VID_CFG})
    assert job2.out_dir == os.path.join(out, 'syn_001') and rec2['name'] == 'syn_001'
    assert os.path.exists(os.path.join(job2.out_dir, 'syn_001_flystate.h5'))
    assert (sha(h5), os.path.getmtime(h5)) == before, 'first result was touched'
    print('offline single camera OK: r=%.4f, second run -> %s' % (r, rec2['name']))


def test_offline_pair_lockstep():
    ensure_synthetic()
    import h5py
    out = tempfile.mkdtemp()
    job, rec = run_job(out, 'pair', {'bottom': VID, 'front': VID},
                       {'bottom': VID_CFG, 'front': VID_CFG}, csv=False)
    assert rec['completed'] and rec['rows'] == 300, rec
    assert sorted(os.listdir(job.out_dir)) == [
        'pair_analysis.yaml', 'pair_bottom_background.png', 'pair_bottom_config.yaml',
        'pair_bottom_reference.png', 'pair_flystate.h5', 'pair_front_background.png',
        'pair_front_config.yaml', 'pair_front_reference.png'], sorted(os.listdir(job.out_dir))
    with h5py.File(os.path.join(job.out_dir, 'pair_flystate.h5'), 'r') as f:
        assert np.array_equal(f['bottom/seq'][:], f['front/seq'][:]), \
            'lockstep: frame N of one camera must pair with frame N of the other'
        assert np.isfinite(f['front/left_angle'][:]).mean() > 0.9, 'front was blanked'


def test_cancel_keeps_what_was_written():
    ensure_synthetic()
    import h5py
    import yaml
    from kinefly2.core.offline import VideoJob, finalize, prepare_setup, run_analysis
    from kinefly2.gui.player import AnalysisDisplay, PlayerPanel
    clock = FakeClock()
    gui = FakeGui(clock, [ord('q'), ord('q')])
    display = AnalysisDisplay(PlayerPanel(gui), gui=gui, clock=clock)
    job = VideoJob(name='cut', videos={'bottom': VID})
    prepare_setup(job, {'bottom': VID_CFG}, log=quiet)
    finalize(job, tempfile.mkdtemp())
    rec = run_analysis(job, show=display, log=quiet, figure=False)
    assert rec['status'] == 'cancelled' and not rec['completed'] and display.quit, rec
    with h5py.File(job.path('_flystate.h5'), 'r') as f:
        assert f['merged/seq'].shape[0] == rec['rows']
    assert yaml.safe_load(open(job.path('_analysis.yaml')))['status'] == 'cancelled'


def test_command_line_headless():
    ensure_synthetic()
    d = tempfile.mkdtemp()
    rig = os.path.join(d, 'rig.yaml')
    with open(rig, 'w') as f:
        f.write('cameras:\n  bottom: %s\nprimary: bottom\n' % VID_CFG)
    out = os.path.join(d, 'out')
    cmd = [sys.executable, '-m', 'kinefly2.apps.analyze_videos', VID, '--out', out,
           '--rig', rig, '--no-gui']
    for _ in range(2):
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert r.returncode == 0, r.stdout.decode()
    # synthetic_bottom.avi names its camera, so the job is called 'synthetic'.
    assert sorted(os.listdir(out)) == ['synthetic', 'synthetic_001'], os.listdir(out)
    from kinefly2.scope import launch
    if launch.available():          # the figure comes with every analysis
        assert os.path.getsize(os.path.join(out, 'synthetic', 'synthetic_traces.png')) > 10000


def test_seeds_from_either_folder_layout():
    """A previous analysis seeds the next one whether it is a flat folder or
    one made before the folders were flattened (configs/camera_<cam>.yaml),
    and its reference frame comes along either way."""
    import shutil
    from kinefly2.apps.analyze_videos import analysis_seeds
    from kinefly2.core.offline import seed_reference
    flat, old = tempfile.mkdtemp(), tempfile.mkdtemp()
    shutil.copy(VID_CFG, os.path.join(flat, 'fly7_bottom_config.yaml'))
    open(os.path.join(flat, 'fly7_bottom_reference.png'), 'w').close()
    os.mkdir(os.path.join(old, 'configs'))
    shutil.copy(VID_CFG, os.path.join(old, 'configs', 'camera_bottom.yaml'))
    open(os.path.join(old, 'configs', 'reference_bottom.png'), 'w').close()
    s_flat = analysis_seeds(flat, ['bottom', 'front'])
    s_old = analysis_seeds(old, ['bottom', 'front'])
    assert list(s_flat) == ['bottom'] and s_flat['bottom'].endswith('fly7_bottom_config.yaml')
    assert list(s_old) == ['bottom'] and s_old['bottom'].endswith('camera_bottom.yaml')
    assert seed_reference(s_flat['bottom'], 'bottom').endswith('fly7_bottom_reference.png')
    assert seed_reference(s_old['bottom'], 'bottom').endswith('reference_bottom.png')
    bare = os.path.join(tempfile.mkdtemp(), 'camera_bottom.yaml')
    shutil.copy(VID_CFG, bare)
    assert seed_reference(bare, 'bottom') is None          # a rig config without one


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('%s OK' % name)
    print('OK')
