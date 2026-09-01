"""Per-camera capture -> track thread.

Each CameraPipeline blocks on its FrameSource, runs the TrackerView, submits
the resulting CameraState to the StateMerger, and parks the newest annotated
frame where the GUI thread can pick it up (dropping display frames is fine;
tracking frames are never dropped).
"""
import threading
import traceback


class CameraPipeline:
    def __init__(self, view, source, merger=None, render=True):
        self.view = view
        self.source = source
        self.merger = merger
        self.render = render
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='pipeline-%s' % view.name)
        self.running = False
        self.eof = False
        self.error = None
        self._display_lock = threading.Lock()
        self._display_img = None
        self.n_frames = 0

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)

    def latest_display(self):
        with self._display_lock:
            return self._display_img

    def _run(self):
        try:
            while self.running and not self.view.bQuit:
                frame = self.source.read()
                if frame is None:
                    self.eof = True
                    return
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
