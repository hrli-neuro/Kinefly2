"""Merge per-camera states and dispatch to output sinks.

The primary camera (the bottom view, which measures L/R wing stroke amplitude)
drives the loop: each of its updates produces a MergedFlyState that is handed
synchronously to every sink in the primary camera's processing thread. Fast
sinks (the future LED-panel serial write) therefore add no queue latency; slow
sinks (logging, ZMQ) must enqueue internally and return immediately.

Sinks can be added/removed at runtime (recording sessions attach and detach
the logger): the sink list is an immutable tuple swapped atomically.
"""
import math
import threading
import time

from .flystate import MergedFlyState


class StateMerger:
    def __init__(self, primary):
        self.primary = primary
        self.latest = {}
        self.sinks = ()
        self.seq = 0
        self.stim = math.nan
        self.stim_t_wall = math.nan
        self.lock = threading.Lock()

    def set_stim(self, value):
        """Set the stimulus code now in force -- the voltage the panels are
        being driven with, which is what identifies the stimulus.

        The value is HELD: every merged state from here on carries it until it
        changes, so 'which stimulus was on at frame N' is a direct lookup
        rather than something analysis has to reconstruct. The wall-clock time
        of the change is kept alongside it, so update latency stays visible.

        This is the seam an analog input would use: a DAQ reader calls exactly
        this, and nothing downstream can tell the difference.
        """
        value = math.nan if value is None else float(value)
        with self.lock:
            changed = not (value == self.stim or (math.isnan(value) and math.isnan(self.stim)))
            if changed:
                self.stim = value
                self.stim_t_wall = time.time()
        return {'stim': self.stim, 'changed': changed, 't_wall': self.stim_t_wall}

    def add_sink(self, sink):
        with self.lock:
            self.sinks = self.sinks + (sink,)

    def remove_sink(self, sink):
        with self.lock:
            self.sinks = tuple(s for s in self.sinks if s is not sink)

    def submit(self, cam_state):
        with self.lock:
            self.latest[cam_state.camera] = cam_state
            if cam_state.camera != self.primary:
                return None
            self.seq += 1
            merged = MergedFlyState(seq=self.seq, t_host=time.monotonic(),
                                    cameras=dict(self.latest),
                                    stim=self.stim, stim_t_wall=self.stim_t_wall)
            sinks = self.sinks
        for sink in sinks:
            sink.handle(merged)
        return merged

    def close(self):
        for sink in self.sinks:
            sink.close()
        self.sinks = ()
