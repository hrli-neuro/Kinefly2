"""Per-camera capture -> track thread.

Each CameraPipeline blocks on its FrameSource, runs the TrackerView, submits
the resulting CameraState to the StateMerger, and parks the newest annotated
frame where the GUI thread can pick it up (dropping display frames is fine;
tracking frames are never dropped).

Built for continuous operation: an optional VideoRecorder can be attached and
detached at any time (recording sessions), and a live camera that stops
delivering frames is reconnected instead of ending the run.

LOCKSTEP (file playback only). Given a `barrier` shared with the other
pipelines, each camera waits for the others at every frame boundary, so bottom
frame N always merges with front frame N however differently the two views cost
to track. This is for replaying recorded video, where the files are already
frame-synchronised and the only thing that can break the pairing is the
tracker's own speed. It is deliberately NOT used with live cameras: there, one
camera hiccuping must not stall the other, and the hardware trigger is what
keeps the two frame clocks together.
"""
import threading
import time
import traceback


class CameraPipeline:
    RECONNECT_S = 2.0
    SYNC_TIMEOUT_S = 30.0         # Lockstep: give up rather than hang forever.

    def __init__(self, view, source, merger=None, render=True, recorder=None,
                 barrier=None, is_primary=False):
        self.view = view
        self.source = source
        self.merger = merger
        self.render = render
        self.recorder = recorder      # Optional VideoRecorder; swap at runtime.
        self.barrier = barrier        # Lockstep with the other pipelines (files only).
        self.is_primary = is_primary  # Only the primary emits the merged row.
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='pipeline-%s' % view.name)
        self.running = False
        self.eof = False              # Video file ended (live cameras never set this).
        self.down = False             # Live camera not delivering; reconnecting.
        self.desynced = False         # Lockstep partner left; this one stops too.
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
        if self.barrier is not None:
            self.barrier.abort()      # Release anyone parked at a frame boundary.
        self.thread.join(timeout=5)

    def latest_display(self):
        with self._display_lock:
            return self._display_img

    def status(self):
        age = None if self.t_last_frame is None else time.monotonic() - self.t_last_frame
        return {'frames': self.n_frames, 'fps': round(self.view.hzF, 1),
                'last_frame_age_s': None if age is None else round(age, 3),
                'down': self.down, 'reconnects': self.n_reconnects,
                'eof': self.eof, 'desynced': self.desynced,
                'error': None if self.error is None else str(self.error),
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

    def _sync(self):
        """Wait for the other pipelines at a frame boundary.

        False means the group has broken up -- a partner hit EOF, errored or
        quit and aborted the barrier -- so this pipeline should stop too rather
        than block. A timeout breaks the barrier the same way, which is what
        keeps a wedged tracker from hanging the whole app."""
        try:
            self.barrier.wait(timeout=self.SYNC_TIMEOUT_S)
            return True
        except threading.BrokenBarrierError:
            return False

    def _submit_lockstep(self, state):
        """Stage this frame, wait for every camera to stage its own, then let
        the primary emit the merged row.

        The barrier is crossed on EVERY frame, including one that produced no
        state (the GUI returns None while a handle is being dragged) -- skipping
        it would leave the partner waiting alone until the timeout.

        The second wait is not redundant: without it a fast camera could read
        and stage frame N+1 before the primary had emitted frame N, and the row
        would pair mismatched frames -- the exact bug lockstep exists to fix."""
        if state is not None and self.merger is not None:
            self.merger.stage(state)
        if not self._sync():
            return False
        if self.is_primary and state is not None and self.merger is not None:
            self.merger.emit()
        return self._sync()

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
                if self.barrier is not None:
                    if not self._submit_lockstep(state):
                        self.desynced = True
                        return
                elif state is not None and self.merger is not None:
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
        finally:
            # However this thread leaves -- EOF, error, quit -- a partner parked
            # at the barrier must not wait for a camera that is never coming
            # back. Files of unequal length therefore stop together, when the
            # shorter one ends.
            if self.barrier is not None:
                self.barrier.abort()
