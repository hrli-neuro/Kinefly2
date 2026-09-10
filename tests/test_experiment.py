"""Per-experiment recording: naming, overrides, output folder contents,
video/sidecar/HDF5 alignment, stimulus events, and re-tracking a recording.
"""
import csv
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
    subprocess.check_call([sys.executable, os.path.join(HERE, 'make_synthetic_video.py')])


def test_naming_and_collision():
    from kinefly2.core.experiment import Experiment
    root = tempfile.mkdtemp()
    exp = Experiment.load(None, {'data_root': root, 'date': '260819', 'genotype': 'XHCS_XHCS',
                                 'fly_id': '017', 'protocol': 'YAMV'})
    assert exp.name == '260819_XHCS_XHCS_017_YAMV'
    d1 = exp.make_output_dir()
    assert d1 == os.path.join(root, '260819', '260819_XHCS_XHCS_017_YAMV')
    exp2 = Experiment.load(None, {'data_root': root, 'date': '260819', 'genotype': 'XHCS_XHCS',
                                  'fly_id': '017', 'protocol': 'YAMV'})
    d2 = exp2.make_output_dir()
    assert d2.endswith('_2') and exp2.run_name == '260819_XHCS_XHCS_017_YAMV_2'
    assert exp2.path('_flystate.h5').endswith('260819_XHCS_XHCS_017_YAMV_2_flystate.h5')
    auto = Experiment.load(None, {'data_root': root})
    assert len(auto.date) == 6 and auto.date.isdigit()


def test_yaml_and_override_precedence():
    from kinefly2.core.experiment import Experiment
    import yaml
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'exp.yaml')
    with open(p, 'w') as f:
        yaml.safe_dump({'genotype': 'G', 'fly_id': '017', 'protocol': 'P',
                        'record_video': True, 'events_port': 5557}, f)
    exp = Experiment.load(p, {'fly_id': '018', 'record_video': False, 'notes': None})
    assert exp.fly_id == '018' and exp.genotype == 'G' and exp.record_video is False
    assert exp.source_path == os.path.abspath(p)


def test_recording_pipeline_and_events():
    ensure_synthetic()
    import cv2
    import h5py
    import yaml
    import zmq

    from kinefly2.camera.base import make_source
    from kinefly2.core.experiment import Experiment
    from kinefly2.core.merger import StateMerger
    from kinefly2.core.pipeline import CameraPipeline
    from kinefly2.gui.app import TrackerView
    from kinefly2.output.event_receiver import EventReceiver
    from kinefly2.output.logger import Hdf5Logger
    from kinefly2.output.video_recorder import VideoRecorder

    root = tempfile.mkdtemp()
    exp = Experiment.load(None, {'data_root': root, 'genotype': 'syn', 'fly_id': '001',
                                 'protocol': 'test', 'events_port': 5598})
    exp.make_output_dir()
    exp.snapshot_configs([VID_CFG])
    exp.write_resolved()

    with open(VID_CFG) as f:
        cam_cfg = yaml.safe_load(f)['camera']
    cameras = ['bottom', 'front']
    logger = Hdf5Logger(exp.path('_flystate.h5'), cameras=cameras)
    merger = StateMerger(primary='bottom')
    merger.add_sink(logger)
    events = EventReceiver(exp.events_port, logger=logger, jsonl_path=exp.path('_events.jsonl'))

    pipelines, recorders = [], {}
    for name in cameras:
        view = TrackerView(name, VID_CFG, use_gui=False)
        source = make_source(dict(cam_cfg, pace=False))
        source.open()
        rec = VideoRecorder(exp.path('_%s.avi' % name), fps=source.fps, quality=95,
                            sidecar_path=exp.path('_%s_frames.csv' % name))
        recorders[name] = rec
        pipelines.append(CameraPipeline(view, source, merger, render=False, recorder=rec))

    push = zmq.Context.instance().socket(zmq.PUSH)
    push.connect('tcp://localhost:%d' % exp.events_port)
    time.sleep(0.2)
    for p in pipelines:
        p.start()
    push.send_string(json.dumps({'trial': 1, 'event': 'stim_on'}))
    t_end = time.monotonic() + 15
    while time.monotonic() < t_end and not all(p.eof for p in pipelines):
        time.sleep(0.05)
    push.send_string('{"event": "stim_off"}')
    time.sleep(0.5)
    for p in pipelines:
        p.stop()
        p.source.close()
    events.close()
    for r in recorders.values():
        r.close()
    merger.close()
    push.close(linger=0)

    assert all(p.error is None for p in pipelines)
    n_bottom = next(p.n_frames for p in pipelines if p.view.name == 'bottom')
    assert n_bottom == 300, n_bottom

    files = sorted(os.listdir(exp.output_dir))
    for suffix in ('_flystate.h5', '_bottom.avi', '_bottom_frames.csv', '_front.avi',
                   '_front_frames.csv', '_events.jsonl', '_experiment.yaml'):
        assert exp.run_name + suffix in files, (suffix, files)
    assert os.path.exists(os.path.join(exp.output_dir, 'configs', 'synthetic_bottom.yaml'))

    # Video frames == sidecar rows == recorder count == frames read.
    rec = recorders['bottom']
    cap = cv2.VideoCapture(exp.path('_bottom.avi'))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    with open(exp.path('_bottom_frames.csv')) as f:
        rows = list(csv.DictReader(f))
    assert rec.n_dropped == 0
    assert n_video == rec.n_written == len(rows) == 300, (n_video, rec.n_written, len(rows))
    assert [int(r['frame_id']) for r in rows] == list(range(300))

    with h5py.File(exp.path('_flystate.h5')) as f:
        assert 'bottom/head_dx_corr' in f, list(f['bottom'])
        # yaw/pitch angles exist but stay NaN until lever_px is calibrated:
        # the displacement channels are the data, an angle would be a guess.
        for col in ('head_yaw', 'head_pitch'):
            assert col in f['bottom'], list(f['bottom'])
            assert not np.isfinite(f['bottom/%s' % col][:]).any(), col
        seq = f['bottom/seq'][:]
        assert len(seq) == 300 and seq[0] == 0 and seq[-1] == 299
        assert np.isfinite(f['merged/t_wall'][:]).all()
        texts = [t.decode() if isinstance(t, bytes) else t for t in f['events/text'][:]]
        assert len(texts) == 2 and 'stim_on' in texts[0] and 'stim_off' in texts[1], texts
        assert f['events/t_host'][0] >= f['merged/t_host'][0] - 1.0
        left_live = f['bottom/left_angle'][:]
    with open(exp.path('_events.jsonl')) as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 2 and 'stim_on' in lines[0]['text']

    # Re-track the recorded MJPEG video and compare with the live HDF5 angles.
    view = TrackerView('retrack', VID_CFG, use_gui=False)
    src = make_source({'kind': 'video', 'path': exp.path('_bottom.avi'), 'fps': 90})
    src.open()
    left_re = []
    for fr in iter(src.read, None):
        st = view.process(fr)
        left_re.append(st.parts['left'].angle)
    src.close()
    left_re = np.array(left_re)
    m = np.isfinite(left_live) & np.isfinite(left_re)
    r = np.corrcoef(left_live[m], left_re[m])[0, 1]
    assert r > 0.99, 'Re-tracked MJPEG differs from live tracking (r=%.3f)' % r
    print('recording OK: %s  (retrack r=%.4f)' % (exp.output_dir, r))


def test_provenance_is_stamped_and_never_fails():
    """A recording must say which code and which conventions produced it. The
    config snapshot cannot: on 2026-09-10 the wing zero changed from the
    head-abdomen perpendicular to the wing-base line, shifting every wing angle
    by ~15 degrees while the YAML stayed byte-identical."""
    import yaml

    from kinefly2.core.experiment import Experiment
    from kinefly2.version import CONVENTIONS, code_version, provenance

    p = provenance()
    for key in ('code_version', 'kinefly2_version', 'conventions', 'python', 'host'):
        assert key in p, key
    assert p['conventions']['wing_zero'] == 'wing_base_line', p['conventions']
    assert CONVENTIONS['angle_units'] == 'radians'

    root = tempfile.mkdtemp()
    exp = Experiment.load(None, {'data_root': root, 'date': '260910',
                                 'genotype': 'syn', 'fly_id': '001', 'protocol': 'prov'})
    exp.make_output_dir()
    exp.write_resolved()
    with open(exp.path('_experiment.yaml')) as f:
        written = yaml.safe_load(f)
    assert written['provenance']['code_version'] == p['code_version']
    assert written['provenance']['conventions']['wing_zero'] == 'wing_base_line'

    # A copy with no git history still records -- it just says so, rather than
    # raising in the middle of starting a recording. Point the module at a
    # directory that is not a checkout and ask again.
    import kinefly2.version as v
    saved_file, saved_cache = v.__file__, v._cached
    try:
        v._cached = None
        v.__file__ = os.path.join(tempfile.mkdtemp(), 'version.py')
        fallback = v.code_version()
    finally:
        v.__file__, v._cached = saved_file, saved_cache
    assert fallback and 'no git' in fallback, fallback
    assert code_version() == p['code_version'], 'the cache must survive the probe'
    print('provenance: %s, conventions %s' % (
        p['code_version'], ', '.join('%s=%s' % kv for kv in sorted(p['conventions'].items()))))
    print('  no-git fallback: %s' % fallback)


if __name__ == '__main__':
    test_naming_and_collision()
    test_yaml_and_override_precedence()
    test_provenance_is_stamped_and_never_fails()
    test_recording_pipeline_and_events()
    print('OK')
