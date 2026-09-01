"""Merge per-camera states and dispatch to output sinks.

The primary camera (the bottom view, which measures L/R wing stroke amplitude)
drives the loop: each of its updates produces a MergedFlyState that is handed
synchronously to every sink in the primary camera's processing thread. Fast
sinks (the future LED-panel serial write) therefore add no queue latency; slow
sinks (logging, ZMQ) must enqueue internally and return immediately.
"""
import threading
import time

from .flystate import MergedFlyState


class StateMerger:
    def __init__(self, primary):
        self.primary = primary
        self.latest = {}
        self.sinks = []
        self.seq = 0
        self.lock = threading.Lock()

    def add_sink(self, sink):
        self.sinks.append(sink)

    def submit(self, cam_state):
        with self.lock:
            self.latest[cam_state.camera] = cam_state
            if cam_state.camera != self.primary:
                return None
            self.seq += 1
            merged = MergedFlyState(seq=self.seq, t_host=time.monotonic(),
                                    cameras=dict(self.latest))
        for sink in self.sinks:
            sink.handle(merged)
        return merged

    def close(self):
        for sink in self.sinks:
            sink.close()
