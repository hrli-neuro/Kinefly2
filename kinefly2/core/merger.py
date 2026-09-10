"""Merge per-camera states and dispatch to output sinks.

The primary camera (the bottom view, which measures L/R wing stroke amplitude)
drives the loop: each of its updates produces a MergedFlyState that is handed
synchronously to every sink in the primary camera's processing thread. Fast
sinks (the future LED-panel serial write) therefore add no queue latency; slow
sinks (logging, ZMQ) must enqueue internally and return immediately.

Sinks can be added/removed at runtime (recording sessions attach and detach
the logger): the sink list is an immutable tuple swapped atomically.

HOW THE TWO CAMERAS ARE PAIRED. This is latest-wins, not timestamp alignment:
a merged row carries whichever state each secondary camera delivered most
recently. When both cameras run at the same rate that is the right partner and
the pairing error is under one frame; it is also biased, because a secondary
sample is always older than the primary's, never newer. Two things follow:

  * on the rig, run both cameras at one frame rate with the hardware trigger
    (camera/pyspin_source.py::_apply_trigger) so frame N of one really is
    frame N of the other;
  * whatever the rate, `ages` records how old each contribution was, and past
    `stale_after_s` that camera's measurements are BLANKED instead of repeated.

The blanking matters more than it looks. A stalled camera whose last state is
copied forward produces a smooth, flat, entirely plausible trace that no
downstream analysis can distinguish from a fly holding still; NaN is honest,
and the age column says exactly how long it lasted.
"""
import dataclasses
import math
import threading
import time

from .flystate import MergedFlyState


class StateMerger:
    def __init__(self, primary, stale_after_s=None):
        self.primary = primary
        # Age past which a secondary camera's measurements are dropped rather
        # than repeated. None disables blanking (ages are still recorded).
        self.stale_after_s = math.nan if stale_after_s is None else float(stale_after_s)
        self.latest = {}
        self.sinks = ()
        self.seq = 0
        self.stim = math.nan
        self.stim_t_wall = math.nan
        self.n_stale = {}          # camera -> merged rows blanked as stale
        self.last_ages = {}        # camera -> age at the most recent merge
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

    def stale_counts(self):
        with self.lock:
            return dict(self.n_stale)

    def ages(self):
        with self.lock:
            return dict(self.last_ages)

    def stage(self, cam_state):
        """Record a camera's newest state without emitting a merged row.

        Split out from submit() so a caller can control the merge point: under
        lockstep replay every camera stages frame N, and only then does the
        primary emit, which is what makes bottom frame N pair with front
        frame N (see CameraPipeline)."""
        with self.lock:
            self.latest[cam_state.camera] = cam_state

    def emit(self):
        """Build one merged state from what is staged and hand it to the sinks."""
        with self.lock:
            t = time.monotonic()
            cameras, ages = {}, {}
            for name, cs in self.latest.items():
                age = t - cs.t_host
                ages[name] = age
                # The primary is never blanked: its state was measured from the
                # frame that triggered this merge, so a large age there means
                # slow processing, not stale data -- worth recording, not
                # worth destroying the row over.
                if (name != self.primary and self.stale_after_s == self.stale_after_s
                        and age > self.stale_after_s):
                    # Keep seq and the timestamps so the stale frame stays
                    # identifiable; drop the measurements. Every consumer
                    # already renders a missing part as NaN.
                    cameras[name] = dataclasses.replace(cs, parts={})
                    self.n_stale[name] = self.n_stale.get(name, 0) + 1
                else:
                    cameras[name] = cs
            self.seq += 1
            self.last_ages = ages
            merged = MergedFlyState(seq=self.seq, t_host=t, cameras=cameras,
                                    stim=self.stim, stim_t_wall=self.stim_t_wall,
                                    ages=ages, stale_after_s=self.stale_after_s)
            sinks = self.sinks
        for sink in sinks:
            sink.handle(merged)
        return merged

    def submit(self, cam_state):
        """Stage a camera's state, and emit a merged row if it is the primary's.
        The live path: each primary frame produces exactly one merged state."""
        self.stage(cam_state)
        if cam_state.camera != self.primary:
            return None
        return self.emit()

    def close(self):
        for sink in self.sinks:
            sink.close()
        self.sinks = ()
