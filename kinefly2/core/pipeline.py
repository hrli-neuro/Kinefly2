"""Per-camera capture -> track thread.

Each CameraPipeline blocks on its FrameSource, runs the TrackerView, submits
the resulting CameraState to the StateMerger, and parks the newest annotated
frame where the GUI thread can pick it up (dropping display frames is fine;
tracking frames are never dropped).

Built for continuous operation: an optional VideoRecorder can be attached and
detached at any time (recording sessions), and a live camera that stops
delivering frames is reconnected instead of ending the run.
"""
import threading
import time
import traceback


class CameraPipeline:
    RECONNECT_S = 2.0

    def __init__(self, view, source, merger=None, render=True, recorder=None):
        self.view = view
        self.source = source
        self.merger = merger
        self.render = render
        self.recorder = recorder      # Optional VideoRecorder; swap at runtime.
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='pipeline-%s' % view.name)
        self.running = False
        self.eof = False              # Video file ended (live cameras never set this).
        self.down = False             # Live camera not delivering; reconnecting.
        self.error = None
        self._display_lock = threading.Lock()
        self._display_img = None
        self.n_frames = 0
        self.n_reconnects = 0
        self.t_last_frame = None

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)

    def latest_display(self):
        with self._display_lock:
            return self._display_img

    def status(self):
        age = None if self.t_last_frame is None else time.monotonic() - self.t_last_frame
        return {'frames': self.n_frames, 'fps': round(self.view.hzF, 1),
                'last_frame_age_s': None if age is None else round(age, 3),
                'down': self.down, 'reconnects': self.n_reconnects,
                'eof': self.eof, 'error': None if self.error is None else str(self.error),
                'recording': self.recorder is not None}

    def _reconnect(self):
        """Live source stopped delivering: close/reopen until it works."""
        self.down = True
        print('%s: camera not delivering frames; reconnecting...' % self.view.name)
        while self.running:
            time.sleep(self.RECONNECT_S)
            try:
                self.source.close()
            except Exception:
                pass
            try:
                self.source.open()
                self.n_reconnects += 1
                self.down = False
                print('%s: camera reconnected' % self.view.name)
                return True
            except Exception as e:
                print('%s: reconnect failed: %s' % (self.view.name, e))
        return False

    def _run(self):
        try:
            while self.running and not self.view.bQuit:
                frame = self.source.read()
                if frame is None:
                    if getattr(self.source, 'reconnectable', False):
                        if self._reconnect():
                            continue
                        return
                    self.eof = True
                    return
                self.t_last_frame = time.monotonic()
                rec = self.recorder
                if rec is not None:
                    rec.write(frame)
                state = self.view.process(frame)
                self.n_frames += 1
                if state is not None and self.merger is not None:
                    self.merger.submit(state)
                if self.render:
                    img = self.view.render()
                    if img is not None:
                        with self._display_lock:
                            self._display_img = img
        except Exception as e:
            self.error = e
            traceback.print_exc()
            self.running = False
