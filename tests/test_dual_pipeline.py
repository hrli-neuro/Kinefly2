"""Integration test: two camera pipelines -> StateMerger -> ZMQ + HDF5.

Exercises the full real-time architecture (threads, merger, sinks) with two
video-file sources standing in for the two FLIR cameras. Verifies that:
  - both pipelines run concurrently and the merger emits merged states,
  - a ZMQ subscriber receives merged states containing both cameras,
  - the HDF5 log holds matching rows with finite wing angles.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

VID_CFG = os.path.join(HERE, 'videos', 'synthetic_bottom.yaml')


def ensure_synthetic():
    # Always regenerate: interactive GUI runs on the synthetic config save
    # their edits back into the YAML, which would silently change this test.
    subprocess.check_call([sys.executable, os.path.join(HERE, 'make_synthetic_video.py')])


def test_dual_pipeline():
    ensure_synthetic()
    import yaml
    import zmq

    from kinefly2.camera.base import make_source
    from kinefly2.core.merger import StateMerger
    from kinefly2.core.pipeline import CameraPipeline
    from kinefly2.gui.app import TrackerView
    from kinefly2.output.logger import Hdf5Logger
    from kinefly2.output.zmq_pub import ZmqPublisher

    port = 5599
    log_path = os.path.join(tempfile.mkdtemp(), 'flystate.h5')

    merger = StateMerger(primary='bottom')
    merger.add_sink(ZmqPublisher(port=port))
    logger = Hdf5Logger(log_path, cameras=['bottom', 'front'])
    merger.add_sink(logger)

    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.connect('tcp://localhost:%d' % port)
    sub.setsockopt_string(zmq.SUBSCRIBE, '')
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    time.sleep(0.3)   # PUB/SUB slow-joiner.

    with open(VID_CFG) as f:
        cam_cfg = yaml.safe_load(f)['camera']

    pipelines = []
    for name in ('bottom', 'front'):
        view = TrackerView(name, VID_CFG, use_gui=False)
        source = make_source(dict(cam_cfg, pace=True))   # Pace at 90 Hz like the rig.
        source.open()
        pipelines.append(CameraPipeline(view, source, merger, render=False))

    received = []
    try:
        for p in pipelines:
            p.start()
        t_end = time.monotonic() + 2.5
        while time.monotonic() < t_end:
            try:
                received.append(json.loads(sub.recv_string()))
            except zmq.Again:
                break
    finally:
        for p in pipelines:
            p.stop()
        for p in pipelines:
            p.source.close()
        merger.close()
        sub.close(linger=0)

    for p in pipelines:
        assert p.error is None, 'Pipeline %s failed: %r' % (p.view.name, p.error)
        assert p.n_frames > 100, 'Pipeline %s processed only %d frames' % (p.view.name, p.n_frames)

    assert len(received) > 100, 'ZMQ subscriber received only %d states' % len(received)
    late = received[len(received) // 2]
    assert set(late['cameras']) == {'bottom', 'front'}, \
        'Merged state missing a camera: %s' % list(late['cameras'])
    left = late['cameras']['bottom']['parts']['left']['angles']
    assert left and np.isfinite(left[0]), 'No left wing angle in merged state'

    import h5py
    with h5py.File(log_path) as f:
        n = f['merged/seq'].shape[0]
        assert n > 100, 'HDF5 log has only %d rows' % n
        la = f['bottom/left_angle'][:]
        assert np.isfinite(la).mean() > 0.9, 'HDF5 wing angles mostly NaN'
    os.remove(log_path)
    print('dual pipeline OK: %d zmq states, %d log rows' % (len(received), n))


if __name__ == '__main__':
    test_dual_pipeline()
    print('OK')
