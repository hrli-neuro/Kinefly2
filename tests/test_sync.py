"""How the two cameras are paired, and what happens when one falls behind.

A merged row carries whichever state each camera delivered most recently. Two
failure modes follow, and this file pins both:

  * a camera that STALLS keeps contributing its last state forever, producing a
    smooth flat trace no analysis can tell from a fly holding still. The merger
    blanks it past `stale_after_s`, and records how old it was either way;
  * two cameras replaying recorded FILES drift apart in frame index, because
    the two views cost different amounts to track -- nothing to do with the
    recording, which was frame-synchronised. A barrier holds them together.
"""
import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kinefly2.camera.base import Frame
from kinefly2.core.flystate import BodypartState, CameraState
from kinefly2.core.merger import StateMerger
from kinefly2.core.pipeline import CameraPipeline


class FakeSource:
    """A file-like source: a fixed number of frames, then EOF."""
    reconnectable = False

    def __init__(self, n, delay=0.0):
        self.n, self.delay, self.i = n, delay, 0

    def open(self):
        pass

    def read(self):
        if self.i >= self.n:
            return None
        if self.delay:
            time.sleep(self.delay)          # Stands in for a costlier tracker.
        f = Frame(image=None, t_camera=float(self.i), t_host=time.monotonic(),
                  frame_id=self.i)
        self.i += 1
        return f

    def close(self):
        pass


class FakeView:
    def __init__(self, name):
        self.name, self.bQuit = name, False

    def process(self, frame):
        return CameraState(camera=self.name, seq=frame.frame_id,
                           t_camera=frame.t_camera, t_host=frame.t_host,
                           t_done=time.monotonic(),
                           parts={'head': BodypartState(angles=[0.25])})

    def render(self):
        return None


class Collect:
    def __init__(self):
        self.rows = []

    def handle(self, merged):
        self.rows.append(merged)

    def close(self):
        pass


def run_pair(n_bottom, n_front, front_delay, lockstep, bottom_delay=0.001,
             timeout=20.0):
    """Two pipelines over fake sources; returns (merged rows, pipelines).

    The two delays stand in for the real asymmetry: the bottom view tracks four
    parts plus the wingbeat detector, the front view runs landmark matching, and
    they do not cost the same per frame."""
    merger = StateMerger(primary='bottom')
    sink = Collect()
    merger.add_sink(sink)
    barrier = threading.Barrier(2) if lockstep else None
    pipes = {}
    for name, n, delay in (('bottom', n_bottom, bottom_delay), ('front', n_front, front_delay)):
        pipes[name] = CameraPipeline(FakeView(name), FakeSource(n, delay), merger,
                                     render=False, barrier=barrier,
                                     is_primary=(name == 'bottom'))
    for p in pipes.values():
        p.start()
    t0 = time.monotonic()
    while any(p.thread.is_alive() for p in pipes.values()):
        if time.monotonic() - t0 > timeout:
            raise AssertionError('pipelines did not finish within %.0f s' % timeout)
        time.sleep(0.01)
    return sink.rows, pipes


def seq_gaps(rows):
    return [abs(r.cameras['bottom'].seq - r.cameras['front'].seq)
            for r in rows if 'front' in r.cameras]


def test_free_running_cameras_drift_apart():
    """The bug lockstep exists to fix: replay both files as fast as each
    tracker manages and the frame indices come apart, even though the two
    recordings are frame-for-frame aligned on disk."""
    rows, _ = run_pair(120, 120, front_delay=0.004, lockstep=False)
    gaps = seq_gaps(rows)
    assert gaps, 'no rows carried both cameras'
    assert max(gaps) > 5, 'expected drift without lockstep, max gap %d' % max(gaps)
    print('free-running: bottom and front drift up to %d frames apart' % max(gaps))


def test_lockstep_pairs_the_same_frame():
    rows, pipes = run_pair(120, 120, front_delay=0.004, lockstep=True)
    gaps = seq_gaps(rows)
    assert gaps, 'no rows carried both cameras'
    assert max(gaps) == 0, 'lockstep must pair frame N with frame N, max gap %d' % max(gaps)
    assert pipes['bottom'].n_frames == pipes['front'].n_frames == 120
    print('lockstep: %d rows, every one pairing frame N with frame N' % len(rows))


def test_lockstep_releases_a_waiting_partner():
    """Unequal files: the shorter ends first and must not leave the other
    parked at the barrier until the timeout."""
    t0 = time.monotonic()
    rows, pipes = run_pair(400, 12, front_delay=0.001, lockstep=True,
                           bottom_delay=0.001, timeout=10.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, 'took %.1f s: a partner was left waiting' % elapsed
    assert pipes['front'].eof, 'the short source should report EOF'
    assert pipes['bottom'].desynced, 'the long source should stop when its partner leaves'
    assert pipes['bottom'].n_frames < 400
    print('unequal lengths: stopped together after %d/%d frames in %.2f s'
          % (pipes['bottom'].n_frames, 400, elapsed))


def test_stale_camera_is_blanked_not_repeated():
    """A stalled camera's last state must not be copied forward: the row keeps
    its seq and timestamps, but the measurements go NaN."""
    merger = StateMerger(primary='bottom', stale_after_s=0.05)
    now = time.monotonic()

    def cam(name, seq, t_host):
        return CameraState(camera=name, seq=seq, t_camera=float(seq), t_host=t_host,
                           t_done=t_host, parts={'head': BodypartState(angles=[0.25])})

    # Fresh: nothing is touched.
    merger.stage(cam('front', 10, now))
    merged = merger.submit(cam('bottom', 10, now))
    assert merged.part('front', 'head').angle == 0.25
    assert not merged.is_stale('front'), merged.ages
    assert merged.ages['front'] < 0.05

    # The front camera stops; the bottom keeps going.
    time.sleep(0.08)
    merged = merger.submit(cam('bottom', 11, time.monotonic()))
    assert merged.is_stale('front'), merged.ages
    assert merged.part('front', 'head') is None, 'stale measurements must be dropped'
    assert merged.cameras['front'].seq == 10, 'the stale frame must stay identifiable'
    assert merged.ages['front'] > 0.05
    assert merger.stale_counts()['front'] == 1
    # The primary is never blanked -- its own age is processing time, not stale data.
    assert merged.part('bottom', 'head').angle == 0.25
    print('stale front blanked after %.0f ms; seq %d kept, angle NaN'
          % (merged.ages['front'] * 1e3, merged.cameras['front'].seq))


def test_no_threshold_never_blanks():
    """The live path is unchanged when no threshold is set: ages are recorded,
    nothing is dropped."""
    merger = StateMerger(primary='bottom')
    old = time.monotonic() - 5.0
    merger.stage(CameraState(camera='front', seq=1, t_camera=1.0, t_host=old, t_done=old,
                             parts={'head': BodypartState(angles=[0.25])}))
    merged = merger.submit(CameraState(camera='bottom', seq=9, t_camera=9.0,
                                       t_host=time.monotonic(), t_done=time.monotonic(),
                                       parts={'head': BodypartState(angles=[0.5])}))
    assert merged.part('front', 'head').angle == 0.25, 'nothing may be dropped'
    assert merged.ages['front'] > 4.9
    assert math.isnan(merged.stale_after_s)
    assert not merged.is_stale('front')
    assert merger.stale_counts() == {}
    print('no threshold: 5 s old state kept, age %.1f s recorded' % merged.ages['front'])


def test_age_reaches_the_log():
    """The age column and the NaN blanking must survive into the HDF5 file."""
    import tempfile

    import h5py
    import numpy as np

    from kinefly2.output.logger import Hdf5Logger

    path = os.path.join(tempfile.mkdtemp(), 'sync.h5')
    logger = Hdf5Logger(path, cameras=['bottom', 'front'])
    merger = StateMerger(primary='bottom', stale_after_s=0.05)
    merger.add_sink(logger)

    def cam(name, seq, t_host):
        return CameraState(camera=name, seq=seq, t_camera=float(seq), t_host=t_host,
                           t_done=t_host, parts={'head': BodypartState(angles=[0.25])})

    now = time.monotonic()
    merger.stage(cam('front', 3, now))
    merger.submit(cam('bottom', 3, now))
    time.sleep(0.08)
    merger.submit(cam('bottom', 4, time.monotonic()))
    time.sleep(0.2)
    logger.close()

    with h5py.File(path, 'r') as f:
        age = f['front/age'][:]
        angle = f['front/head_angle'][:]
        seq = f['front/seq'][:]
    assert len(age) == 2, len(age)
    assert age[0] < 0.05 and age[1] > 0.05, age
    assert np.isfinite(angle[0]) and not np.isfinite(angle[1]), angle
    assert list(seq) == [3, 3], seq
    print('log: front/age %.3f -> %.3f s, angle %.2f -> NaN, seq held at %d'
          % (age[0], age[1], angle[0], seq[1]))


if __name__ == '__main__':
    test_free_running_cameras_drift_apart()
    test_lockstep_pairs_the_same_frame()
    test_lockstep_releases_a_waiting_partner()
    test_stale_camera_is_blanked_not_repeated()
    test_no_threshold_never_blanks()
    test_age_reaches_the_log()
    print('OK')
