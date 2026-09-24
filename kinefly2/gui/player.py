"""The offline player: pause, slow motion, stepping and seeking on recorded
videos, with every camera of a job on the same frame.

    Transport        which frame, playing or not, how fast. Pure state -- no
                     windows -- so its pacing is testable with a fake clock.
    PlayerPanel      the player window: transport buttons, a seek bar, a status
                     line. Clicks become actions for the session to consume.
    SetupSession     tuning one job: the usual camera windows (toolbar, handles,
                     +H landmarks, autoplace, zero) driven by the player, until
                     ANALYZE, skip or quit.
    AnalysisDisplay  the same windows while the analysis runs: overlay only,
                     and the player turned into a progress bar.

Everything that touches a window goes through `gui` (cv2 by default), so the
whole setup loop can be driven by a script in the tests.

Two rules keep the setup view showing what the analysis will see. The running
background model adapts with the time between frames, and a frame that does not
follow the previous one -- a seek, a step back, the loop back to frame 0, or a
paused frame tracked again (which counts as a whole second) -- would let it soak
up the current frame, fly and all. So (1) a paused frame is tracked again only
after an edit, never continuously, and (2) any such frame restarts the model
from the background image first. Plain playback adapts normally, as the
analysis does.
"""
import copy
import math
import os
import time

import cv2
import numpy as np

from ..camera.video_source import VideoFileSource
from . import autoplace
from .app import TrackerView

SPEEDS = (0.25, 0.5, 1.0, 2.0)
CONFIRM_S = 3.0                 # A second Esc/q within this long confirms.
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fmt_clock(s):
    s = round(max(0.0, float(s)), 1)      # first, so 59.96 s reads 1:00.0, not 0:60.0
    return '%d:%04.1f' % (s // 60, s % 60)


def fmt_speed(s):
    return ('%g' % s) + 'x'


# --------------------------------------------------------------------- state

class Transport:
    """Playback state for one job: the frame index, and when the next one is
    due. `tick(now)` advances at fps * speed without ever bursting through
    frames to catch up, and loops at the end."""

    def __init__(self, n_frames, fps, speed=0.5, loop=True):
        self.n = max(1, int(n_frames))
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.speed = self.nearest(speed)
        self.loop = loop
        self.index = 0
        self.playing = False
        self.t_next = None

    @staticmethod
    def nearest(speed):
        return min(SPEEDS, key=lambda v: abs(v - float(speed)))

    @property
    def interval(self):
        return 1.0 / (self.fps * self.speed)

    def play(self, now):
        self.playing = True
        self.t_next = now + self.interval

    def pause(self):
        self.playing = False
        self.t_next = None

    def toggle(self, now):
        if self.playing:
            self.pause()
        else:
            self.play(now)

    def seek(self, index):
        self.index = int(min(max(0, int(index)), self.n - 1))

    def step(self, k):
        """Single frames pause first, so the frame stays put to work on."""
        self.pause()
        self.seek(self.index + k)

    def jump(self, seconds):
        self.seek(self.index + int(round(seconds * self.fps)))

    def set_speed(self, speed, now=None):
        self.speed = self.nearest(speed)
        if self.playing and now is not None:
            self.t_next = now + self.interval

    def faster(self, now=None):
        self.set_speed(SPEEDS[min(SPEEDS.index(self.speed) + 1, len(SPEEDS) - 1)], now)

    def slower(self, now=None):
        self.set_speed(SPEEDS[max(SPEEDS.index(self.speed) - 1, 0)], now)

    def tick(self, now):
        """Advance one frame if one is due; True when the index moved."""
        if not self.playing or now < self.t_next:
            return False
        nxt = self.index + 1
        if nxt >= self.n:
            if not self.loop:
                self.pause()
                return False
            nxt = 0
        self.index = nxt
        self.t_next += self.interval
        if self.t_next < now:
            # Tracking and drawing took longer than a frame: go on from now
            # rather than race through frames to catch up.
            self.t_next = now + self.interval
        return True

    def wait_s(self, now):
        """Seconds until the next frame is due; None while paused."""
        return None if not self.playing else max(0.0, self.t_next - now)

    def truncate(self, n):
        """The video ended early (its container over-reported the length)."""
        self.n = max(1, int(n))
        self.seek(self.index)


# waitKeyEx codes for the arrow/Home/End keys: macOS (Cocoa), Linux (GTK/Qt), Windows.
_LEFT = (63234, 65361, 2424832)
_RIGHT = (63235, 65363, 2555904)
_HOME = (63273, 65360, 2359296)
_END = (63275, 65367, 2293760)
_KEYS = {' ': ('toggle',), ',': ('step', -1), '.': ('step', 1),
         '<': ('jump', -1.0), '>': ('jump', 1.0), '[': ('slower',), ']': ('faster',),
         '0': ('home',), '\r': ('analyze',), '\n': ('analyze',),
         '\x1b': ('skip',), 'q': ('quit',)}

KEY_HELP = ('space play/pause   arrows or , . step   < > 1 s   [ ] speed   '
            'enter ANALYZE   esc x2 skip   q x2 quit')


def key_action(key):
    """The action a key press asks for, or None."""
    if key is None or key < 0:
        return None
    if key in _LEFT:
        return ('step', -1)
    if key in _RIGHT:
        return ('step', 1)
    if key in _HOME:
        return ('home',)
    if key in _END:
        return ('end',)
    return _KEYS.get(chr(key & 0xFF))


def _ignore_mouse(*args):
    """Installed on the camera windows while an analysis runs: a click then
    must not move a handle, and no mouse-up may rewrite the configs."""


# --------------------------------------------------------------------- window

class PlayerPanel:
    """The player window. Draws itself from a Transport; clicks and seek-bar
    drags are queued as actions (pop_actions)."""

    WINDOW = 'kinefly2: player'
    W, H = 700, 134
    BG = (38, 38, 38)
    FG = (235, 235, 235)
    DIM = (150, 150, 150)
    ACTIVE = (0, 150, 255)        # BGR orange: the speed in use, play while playing
    GO = (60, 150, 40)            # ANALYZE
    WARN = (80, 80, 255)
    NOTE = (0, 190, 255)          # BGR amber: things about this job worth knowing

    SETUP = (('|<', ('home',)), ('-1s', ('jump', -1.0)), ('-1', ('step', -1)),
             ('play', ('toggle',)), ('+1', ('step', 1)), ('+1s', ('jump', 1.0)), None,
             ('0.25x', ('speed', 0.25)), ('0.5x', ('speed', 0.5)),
             ('1x', ('speed', 1.0)), ('2x', ('speed', 2.0)), None,
             ('ANALYZE', ('analyze',)), ('skip', ('skip',)), ('quit', ('quit',)))
    ANALYSIS = (('cancel', ('skip',)), ('quit', ('quit',)))

    def __init__(self, gui=cv2):
        self.gui = gui
        self.actions = []
        self.pressed = None
        self.dragging = False
        self.mode = None
        self.buttons = []
        self.bar = (12, 42, self.W - 12, 56)
        self.img = None
        self.opened = False

    def open(self, x=None, y=None):
        self.gui.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        self.gui.setMouseCallback(self.WINDOW, self.on_mouse)
        if x is not None:
            self.gui.moveWindow(self.WINDOW, int(x), int(y))
        self.opened = True

    def set_mode(self, mode):
        """'setup' (transport controls) or 'analysis' (progress, cancel)."""
        self.mode = mode
        self.buttons = self._layout(self.SETUP if mode == 'setup' else self.ANALYSIS)
        self.actions, self.pressed, self.dragging = [], None, False

    @staticmethod
    def _layout(spec):
        x, y, h = 12, 8, 24
        out = []
        for item in spec:
            if item is None:
                x += 14
                continue
            label, action = item
            widest = 'pause' if action == ('toggle',) else label
            w = max(30, cv2.getTextSize(widest, FONT, 0.45, 1)[0][0] + 16)
            out.append((label, action, (x, y, x + w, y + h)))
            x += w + 4
        return out

    def pop_actions(self):
        a, self.actions = self.actions, []
        return a

    # ------------------------------------------------------------ mouse

    @staticmethod
    def _inside(rect, x, y):
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def _seek_to(self, x):
        x0, _, x1, _ = self.bar
        frac = min(1.0, max(0.0, (x - x0) / float(x1 - x0)))
        # Keep only the newest seek of a drag: the frame is loaded once.
        self.actions = [a for a in self.actions if a[0] != 'seek'] + [('seek', frac)]

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for label, action, rect in self.buttons:
                if self._inside(rect, x, y):
                    self.pressed = label
                    return
            x0, y0, x1, y1 = self.bar
            if self.mode == 'setup' and x0 - 4 <= x <= x1 + 4 and y0 - 6 <= y <= y1 + 6:
                self.dragging = True
                self._seek_to(x)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._seek_to(x)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.pressed is not None:
                for label, action, rect in self.buttons:
                    if label == self.pressed and self._inside(rect, x, y):
                        self.actions.append(action)
            if self.dragging:
                self._seek_to(x)
            self.pressed, self.dragging = None, False

    # ------------------------------------------------------------ drawing

    def _button(self, img, rect, text, fill, pressed):
        x0, y0, x1, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20) if pressed else (170, 170, 170), 1)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        off = 1 if pressed else 0
        cv2.putText(img, text, ((x0 + x1 - tw) // 2 + off, (y0 + y1 + th) // 2 + off),
                    FONT, 0.45, self.FG, 1, cv2.LINE_AA)

    def _bar(self, img, frac, marker):
        x0, y0, x1, y1 = self.bar
        cv2.rectangle(img, (x0, y0), (x1, y1), (75, 75, 75), -1)
        xp = int(round(x0 + min(1.0, max(0.0, frac)) * (x1 - x0)))
        cv2.rectangle(img, (x0, y0), (xp, y1), (150, 110, 50), -1)
        if marker:
            cv2.line(img, (xp, y0 - 4), (xp, y1 + 4), self.FG, 2)

    def _text(self, img, s, y, color=None):
        cv2.putText(img, s, (12, y), FONT, 0.45, color or self.FG, 1, cv2.LINE_AA)

    def draw_setup(self, transport, saves_as='', prompt='', notes=()):
        img = np.full((self.H, self.W, 3), self.BG, np.uint8)
        for label, action, rect in self.buttons:
            text, fill = label, (85, 85, 85)
            if action == ('toggle',) and transport.playing:
                text, fill = 'pause', self.ACTIVE
            elif action[0] == 'speed' and action[1] == transport.speed:
                fill = self.ACTIVE
            elif action == ('analyze',):
                fill = self.GO
            self._button(img, rect, text, fill, self.pressed == label)
        self._bar(img, transport.index / float(max(1, transport.n - 1)), True)
        last = transport.n - 1
        self._text(img, 'frame %d / %d    %s / %s    %s    %s' % (
            transport.index, last, fmt_clock(transport.index / transport.fps),
            fmt_clock(last / transport.fps), fmt_speed(transport.speed),
            'PLAYING' if transport.playing else 'PAUSED'), 76)
        if prompt:
            self._text(img, prompt, 94, self.WARN)
        elif saves_as:
            self._text(img, 'saves as  ' + saves_as, 94, self.DIM)
        if notes:
            self._text(img, '   '.join(notes), 110, self.NOTE)
        self._text(img, KEY_HELP, 128, self.DIM)
        self.img = img
        self.gui.imshow(self.WINDOW, img)
        return img

    def draw_progress(self, prog, title='', prompt=''):
        img = np.full((self.H, self.W, 3), self.BG, np.uint8)
        for label, action, rect in self.buttons:
            self._button(img, rect, label, (85, 85, 85), self.pressed == label)
        x = self.buttons[-1][2][2] + 16 if self.buttons else 12
        cv2.putText(img, 'ANALYZING  ' + title, (x, 26), FONT, 0.45, self.FG, 1, cv2.LINE_AA)
        frac = prog['done'] / float(prog['total']) if prog.get('total') else 0.0
        self._bar(img, frac, False)
        s = 'frame %d' % prog['done']
        if prog.get('total'):
            s += ' / %d  (%d%%)' % (prog['total'], 100.0 * frac)
        s += '    %.0f frames/s' % prog['fps']
        if prog.get('eta') is not None:
            s += '    %s left' % fmt_clock(prog['eta'])
        self._text(img, s, 76)
        self._text(img, prompt or 'handles are locked while analyzing', 94,
                   self.WARN if prompt else self.DIM)
        self._text(img, 'esc x2 cancel (keeps what was written)   q x2 cancel and quit', 128, self.DIM)
        self.img = img
        self.gui.imshow(self.WINDOW, img)
        return img


class _Confirm:
    """Esc and q act only when pressed twice within CONFIRM_S."""

    PROMPTS = {'skip': 'press esc (or skip) again to skip this video',
               'quit': 'press q (or quit) again to stop -- nothing more is analyzed',
               'cancel': 'press esc (or cancel) again to cancel this analysis'}

    def __init__(self):
        self.pending = None

    def ask(self, kind, now):
        if self.pending and self.pending[0] == kind and now < self.pending[1]:
            self.pending = None
            return True
        self.pending = (kind, now + CONFIRM_S)
        return False

    def prompt(self, now):
        if self.pending and now < self.pending[1]:
            return self.PROMPTS.get(self.pending[0], '')
        self.pending = None
        return ''


# ------------------------------------------------------------------- sessions

class SetupSession:
    """Tune one job before it is analyzed. run() returns 'analyze', 'skip' or
    'quit'; on 'analyze' the landmark templates in use are saved next to the
    configs, so the analysis starts from exactly the zero that was shown."""

    def __init__(self, job, primary=None, speed=0.5, saves_as='', title='',
                 panel=None, gui=cv2, clock=time.monotonic):
        self.job = job
        self.primary = primary if primary in job.videos else job.cameras[0]
        self.speed = speed
        self.saves_as = saves_as
        self.title = title
        self.gui = gui
        self.clock = clock
        self.panel = panel or PlayerPanel(gui)
        self.views, self.sources = {}, {}
        self.frames, self.next_pos = {}, {}
        self.shown = None
        self.dirty = set()
        self.confirm = _Confirm()
        self.handlers = {}
        self.transport = None
        self.notes = []             # Shown in the player window for this job.

    # ------------------------------------------------------------ frames

    def _goto(self, index):
        """Put frame `index` on every camera and track it. False if a video
        has no such frame."""
        frames = {}
        for cam, src in self.sources.items():
            if self.next_pos[cam] != index:
                src.seek(index)
            fr = src.read()
            if fr is None:
                self.next_pos[cam] = -1
                return False
            frames[cam] = fr
            self.next_pos[cam] = index + 1
        follows = self.shown is not None and index == self.shown + 1
        self.frames, self.shown = frames, index
        for cam, view in self.views.items():
            if not follows:
                view.reset_background_model()
            view.process(frames[cam])
        self.dirty.clear()
        return True

    def _show(self, index):
        for _ in range(3):
            if self._goto(index):
                return
            # The container promised more frames than the video has.
            self.transport.truncate(index)
            index = 0 if (self.transport.playing and self.transport.loop) else self.transport.index
        raise IOError('cannot read frame %d of %s' % (index, ', '.join(self.job.videos.values())))

    def _retrack(self):
        """Track the held frame again after an edit (paused only; playing,
        the next frame comes soon anyway)."""
        if not self.transport.playing:
            for cam in sorted(self.dirty):
                view = self.views[cam]
                if not view.bMousing:
                    view.reset_background_model()
                    view.process(self.frames[cam])
        self.dirty.clear()

    def _fit_regions(self):
        """Regions tuned on a video of another size can sit outside this one,
        where their handles cannot be grabbed: bring them inside, and say where
        they came from -- autoplace cannot help across sizes."""
        for cam, view in self.views.items():
            shape = self.frames[cam].image.shape
            if not autoplace.points_inside(view.params, shape):
                view.apply_params(copy.deepcopy(view.params), 'regions moved inside')
                self.dirty.add(cam)
                self.notes.append('%s: regions were outside this video, moved inside' % cam)
            ref = cv2.imread(view.filenameReference, cv2.IMREAD_GRAYSCALE)
            if ref is not None and ref.shape != shape[:2]:
                view.statusText = 'tuned on %dx%d' % (ref.shape[1], ref.shape[0])
                self.notes.append('%s: regions come from a %dx%d video, this is %dx%d'
                                  ' - place them by hand' % (cam, ref.shape[1], ref.shape[0],
                                                             shape[1], shape[0]))
        for note in self.notes:
            print(note)

    # ------------------------------------------------------------ windows

    def _mouse(self, cam):
        def handler(event, x, y, flags, param):
            view = self.views.get(cam)
            if view is None:
                return
            view.on_mouse(event, x, y, flags, param)
            if event == cv2.EVENT_LBUTTONUP:
                self.dirty.add(cam)
        return handler

    def _install_callbacks(self):
        # Again every second: a camera window closed with its close button
        # comes back on the next imshow, but without its mouse handler.
        for cam, view in self.views.items():
            try:
                self.gui.setMouseCallback(view.window_name, self.handlers[cam])
            except cv2.error:
                pass

    def open(self):
        for cam in self.job.cameras:
            self.views[cam] = TrackerView(cam, self.job.config_path(cam), use_gui=False,
                                          replay=True, asset_stem=self.job.stem(cam))
            src = VideoFileSource(self.job.videos[cam], fps=self.job.fps)
            src.open()
            self.sources[cam] = src
            self.next_pos[cam] = 0
        counts = [s.n_frames for s in self.sources.values() if s.n_frames]
        n = min(counts) if counts else 10 ** 9      # unknown: found at the end
        self.transport = Transport(n, self.sources[self.primary].fps, self.speed)
        if not self._goto(0):
            raise IOError('no frames in %s' % ', '.join(self.job.videos.values()))
        self._fit_regions()
        x = 20
        for cam, view in self.views.items():
            win = view.window_name
            self.gui.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            self.handlers[cam] = self._mouse(cam)
            self.gui.setMouseCallback(win, self.handlers[cam])
            self.gui.setWindowTitle(win, '%s: %s   [setup %s]' % (
                cam, os.path.basename(self.job.videos[cam]), self.title))
            self.gui.moveWindow(win, x, 20)
            x += self.frames[cam].image.shape[1] + 24
        h = max(f.image.shape[0] for f in self.frames.values())
        self.panel.open(20, h + 80)
        self.panel.set_mode('setup')

    def close(self):
        for src in self.sources.values():
            src.close()
        for view in self.views.values():
            try:
                self.gui.setMouseCallback(view.window_name, _ignore_mouse)
            except cv2.error:
                pass

    # ------------------------------------------------------------ loop

    def _act(self, action, now):
        kind = action[0]
        t = self.transport
        if kind not in ('skip', 'quit'):
            self.confirm.pending = None     # Anything else withdraws a pending esc/q.
        if kind == 'toggle':
            t.toggle(now)
        elif kind == 'step':
            t.step(action[1])
        elif kind == 'jump':
            t.jump(action[1])
        elif kind == 'seek':
            t.seek(round(action[1] * (t.n - 1)))
        elif kind == 'home':
            t.seek(0)
        elif kind == 'end':
            t.seek(t.n - 1)
        elif kind == 'speed':
            t.set_speed(action[1], now)
        elif kind == 'faster':
            t.faster(now)
        elif kind == 'slower':
            t.slower(now)
        elif kind == 'analyze':
            return 'analyze'
        elif kind in ('skip', 'quit'):
            if self.confirm.ask(kind, now):
                return kind
        return None

    def _draw(self, now):
        for view in self.views.values():
            img = view.render()
            if img is not None:
                self.gui.imshow(view.window_name, img)
        self.panel.draw_setup(self.transport, self.saves_as, self.confirm.prompt(now),
                              self.notes)

    def run(self):
        """Returns 'analyze', 'skip' or 'quit'."""
        try:
            self.open()
            result = self._loop()
            if result == 'analyze':
                for view in self.views.values():
                    view.persist_templates()
            return result
        finally:
            self.close()

    def _loop(self):
        self._draw(self.clock())
        t_callbacks = self.clock()
        while True:
            wait = self.transport.wait_s(self.clock())
            # Round up: waking a fraction of a millisecond early would spend a
            # whole loop finding the frame not yet due.
            delay = 30 if wait is None else int(min(30, max(1, math.ceil(wait * 1000))))
            key = self.gui.waitKeyEx(delay)
            now = self.clock()
            actions = self.panel.pop_actions()
            a = key_action(key)
            if a is not None:
                actions.insert(0, a)
            for a in actions:
                result = self._act(a, now)
                if result:
                    return result
            if any(v.bQuit for v in self.views.values()):     # the toolbar's exit
                return 'quit'
            self.transport.tick(now)
            if self.transport.index != self.shown:
                self._show(self.transport.index)
            elif self.dirty:
                self._retrack()
            self._draw(now)
            if now - t_callbacks > 1.0:
                self._install_callbacks()
                t_callbacks = now


class AnalysisDisplay:
    """The `show` callback for core.offline.run_analysis: each camera window
    shows its pipeline's latest overlay, the player window a progress bar.
    Returns True to cancel; `quit` says whether the queue should stop too."""

    def __init__(self, panel, title='', gui=cv2, clock=time.monotonic):
        self.panel = panel
        self.title = title
        self.gui = gui
        self.clock = clock
        self.quit = False
        self.confirm = _Confirm()
        if not self.panel.opened:           # no setup ran before this analysis
            self.panel.open(20, 20)
        self.panel.set_mode('analysis')

    def __call__(self, pipelines, prog):
        for p in pipelines.values():
            img = p.latest_display()
            if img is not None:
                self.gui.imshow(p.view.window_name, img)
        now = self.clock()
        prompt = self.confirm.prompt(now)
        if self.confirm.pending and self.confirm.pending[0] == 'skip':
            prompt = _Confirm.PROMPTS['cancel']
        self.panel.draw_progress(prog, self.title, prompt)
        key = self.gui.waitKeyEx(30)
        now = self.clock()
        actions = self.panel.pop_actions()
        a = key_action(key)
        if a is not None:
            actions.insert(0, a)
        for a in actions:
            if a[0] in ('skip', 'quit') and self.confirm.ask(a[0], now):
                self.quit = a[0] == 'quit'
                return True
        return False
