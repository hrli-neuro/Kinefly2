"""Continuous operation with script-triggered recordings: tracking runs
non-stop while a client starts/stops recording sessions over the command
socket, logs events, and queries status."""
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


def test_sessions_over_command_socket():
    ensure_synthetic()
    import h5py
    import yaml
    import zmq

    from kinefly2.camera.base import make_source
    from kinefly2.client import Kinefly2Client, Kinefly2Error
    from kinefly2.core.experiment import Experiment
    from kinefly2.core.merger import StateMerger
    from kinefly2.core.pipeline import CameraPipeline
    from kinefly2.core.session import RecordingSession
    from kinefly2.gui.app import TrackerView
    from kinefly2.output.command_server import CommandServer
    from kinefly2.output.event_receiver import EventReceiver

    root = tempfile.mkdtemp()
    template = Experiment.load(None, {'data_root': root, 'genotype': 'syn', 'fly_id': '000',
                                      'protocol': 'idle', 'events_port': 5687})
    with open(VID_CFG) as f:
        cam_cfg = yaml.safe_load(f)['camera']
    merger = StateMerger(primary='bottom')
    pipelines, views = {}, {}
    for name in ('bottom', 'front'):
        view = TrackerView(name, VID_CFG, use_gui=False)
        src = make_source(dict(cam_cfg, pace=True, loop=True))   # continuous, ~90 Hz
        src.open()
        views[name] = view
        pipelines[name] = CameraPipeline(view, src, merger, render=False)
    events = EventReceiver(template.events_port)
    session = RecordingSession(template, merger, pipelines, views=views, events=events,
                               config_paths=[VID_CFG])
    server = CommandServer(session.handle_command, port=5688)
    for p in pipelines.values():
        p.start()
    time.sleep(0.5)

    k = Kinefly2Client('localhost', command_port=5688)
    push = zmq.Context.instance().socket(zmq.PUSH)
    push.connect('tcp://localhost:%d' % template.events_port)
    try:
        st = k.status()
        assert st['recording'] is False and st['cameras']['bottom']['frames'] > 0

        # Events while idle are dropped.
        r = k.event({'event': 'idle'})
        assert r['logged'] is False and r['recording'] is False
        push.send_string('idle push'); time.sleep(0.2)
        assert events.n_dropped_idle >= 2 and events.n_events == 0

        # Stop while idle is an error, not a crash.
        try:
            k.stop(); assert False, 'stop while idle should fail'
        except Kinefly2Error:
            pass

        # Session 1. lever_px is per fly, so it travels with the start
        # command and is applied to the trackers for this recording only.
        frames_before = pipelines['bottom'].n_frames
        info = k.start(fly_id='017', protocol='YAMV', notes='first',
                       lever_px={'front': 48.2})
        assert info['name'].endswith('_syn_017_YAMV'), info
        try:
            k.start(fly_id='018'); assert False, 'start while recording should fail'
        except Kinefly2Error as e:
            assert 'already' in str(e)
        time.sleep(1.0)
        k.event({'trial': 1, 'event': 'stim_on'})
        push.send_string('{"trial": 1, "event": "pushed"}')
        time.sleep(0.5)
        st = k.status()
        assert st['recording'] and st['session']['name'] == info['name']
        assert st['session']['rows'] > 30, st['session']
        z = k.zero()
        assert set(z) == {'bottom', 'front'} and all('zeroed' in v for v in z.values()), z
        paths = k.snapshot()
        assert len(paths) == 2 and all(os.path.exists(p) for p in paths)
        # The stimulus code: held from the moment it is set, so every frame
        # from here on is labelled with the stimulus that was on.
        r = k.stim(2.5)
        assert r['stim'] == 2.5 and r['changed'] is True, r
        assert k.stim(2.5)['changed'] is False, 'setting the same value is not a change'
        time.sleep(0.4)
        r = k.stim(7.0)
        assert r['stim'] == 7.0 and r['changed'] is True, r
        time.sleep(0.4)

        # It must never leak into the camera config: that is where the next
        # fly would silently inherit it.
        with open(VID_CFG) as f:
            on_disk = yaml.safe_load(f) or {}
        for part in ('head', 'abdomen', 'left', 'right'):
            assert 'lever_px' not in ((on_disk.get('tracker') or {}).get(part) or {}), \
                'lever_px leaked into the camera config'
        summary = k.stop()
        assert summary['events'] == 2, summary   # 'zeroed' events only come from run_rig's callback
        assert summary['video']['bottom']['written'] > 30 and summary['video']['bottom']['dropped'] == 0
        assert pipelines['bottom'].n_frames > frames_before

        # lever_px round-trips: start command -> Experiment -> resolved YAML,
        # and is stamped into the recording along with the landmark positions
        # it belongs to, so a stale constant can be detected later.
        with open(os.path.join(summary['output_dir'],
                               summary['name'] + '_experiment.yaml')) as f:
            resolved = yaml.safe_load(f)
        assert resolved['lever_px'] == {'front': 48.2}, resolved.get('lever_px')
        with h5py.File(os.path.join(summary['output_dir'], summary['name'] + '_flystate.h5')) as f:
            assert json.loads(f.attrs['lever_px']) == {'front': 48.2}, dict(f.attrs)
            stim = f['merged/stim'][:]
            seen = set(np.unique(stim[np.isfinite(stim)]))
            assert {2.5, 7.0} <= seen, seen        # both steps landed in the log
            held = np.isfinite(stim).sum()
            assert held > 10, 'the value must be held on every frame, not just at the change'
            assert np.isfinite(f['merged/stim_t_wall'][:]).any()
            assert 'landmarks' in f.attrs, dict(f.attrs)
            assert f['merged/seq'].shape[0] == summary['rows']
            texts = [t.decode() if isinstance(t, bytes) else t for t in f['events/text'][:]]
            assert any('stim_on' in t for t in texts) and any('pushed' in t for t in texts), texts
            assert np.isfinite(f['bottom/left_angle'][:]).mean() > 0.9
        files = os.listdir(summary['output_dir'])
        assert any(x.endswith('_bottom.avi') for x in files) and any(x.endswith('_events.jsonl') for x in files)
        assert os.path.isdir(os.path.join(summary['output_dir'], 'configs'))

        # Session 2 without video, tracking never stopped in between.
        frames_between = pipelines['bottom'].n_frames
        with k.recording(fly_id='018', protocol='OL', record_video=False) as info2:
            time.sleep(0.6)
        assert info2['name'].endswith('_syn_018_OL')
        s2 = session.last_summary
        assert s2['video'] == {} and s2['rows'] > 20
        assert pipelines['bottom'].n_frames > frames_between + 40
        assert os.path.isdir(info2['output_dir']) and info2['output_dir'] != summary['output_dir']
        assert not any(x.endswith('.avi') for x in os.listdir(info2['output_dir']))
        assert k.status()['recording'] is False
        assert all(p.error is None for p in pipelines.values())
        print('sessions OK: %s | %s' % (summary['name'], s2['name']))
    finally:
        push.close(linger=0)
        k.close()
        server.close()
        session.close()
        events.close()
        for p in pipelines.values():
            p.stop()
            p.source.close()
        merger.close()


if __name__ == '__main__':
    test_sessions_over_command_socket()
    print('OK')
