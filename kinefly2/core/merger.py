"""Merge per-camera states and dispatch to output sinks.

The primary camera (the bottom view, which measures L/R wing stroke amplitude)
drives the loop: each of its updates produces a MergedFlyState that is handed
synchronously to every sink in the primary camera's processing thread. Fast
sinks (the future LED-panel serial write) therefore add no queue latency; slow
sinks (logging, ZMQ) must enqueue internally and return immediately.

Sinks can be added/removed at runtime (recording sessions attach and detach
the logger): the sink list is an immutable tuple swapped atomically.
"""
import threading
import time

from .flystate import MergedFlyState


class StateMerger:
    def __init__(self, primary):
        self.primary = primary
        self.latest = {}
        self.sinks = ()
        self.seq = 0
        self.lock = threading.Lock()

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
                                    cameras=dict(self.latest))
            sinks = self.sinks
        for sink in sinks:
            sink.handle(merged)
        return merged

    def close(self):
        for sink in self.sinks:
            sink.close()
        self.sinks = ()
