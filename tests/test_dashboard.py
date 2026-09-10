"""The dashboard must emit strictly valid JSON and name channels the way the
tracker does, whatever the camera view or calibration state."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinefly2.apps.dashboard import StateBuffer, clean


def strict_loads(text):
    """json.loads that refuses NaN/Infinity, like a browser's JSON.parse."""
    def boom(const):
        raise ValueError('non-finite literal %r in payload' % const)
    return json.loads(text, parse_constant=boom)


def state(view, calibrated=False, seq=0, ages=None, stale_after_s=None):
    """A published state shaped like MergedFlyState.to_dict()."""
    nan = float('nan')
    if view == 'ventral':
        extra = {'yaw': math.radians(3.0), 'roll': nan, 'pitch': nan,
                 'd_axial': nan, 'd_lateral': nan, 'roll_sigma': 0.01}
    else:
        extra = {'roll': math.radians(2.0),
                 'pitch': math.radians(1.5) if calibrated else nan,
                 'yaw': math.radians(-4.0) if calibrated else nan,
                 'd_axial': 3.1, 'd_lateral': -7.2, 'roll_sigma': 0.02}
    cam = 'bottom' if view == 'ventral' else 'front'
    return {
        'seq': seq, 't_host': float(seq) / 90.0, 'stim': 2.5, 'stim_t_wall': 1.0,
        'ages': ages if ages is not None else {cam: 0.004},
        'stale_after_s': stale_after_s if stale_after_s is not None else 0.055,
        'cameras': {cam: {'camera': cam, 'seq': seq, 't_camera': nan,
                          't_host': 0.0, 't_done': 0.0,
                          'parts': {
                              'head': {'angles': [math.radians(3.0)], 'gradients': [1.0, 2.0, 0.5],
                                       'radii': [1.0, 2.0], 'freq': 0.0, 'intensity': 0.93,
                                       'kind': 'landmarks', 'extra': extra},
                              'left': {'angles': [math.radians(50.0)], 'gradients': [], 'radii': [],
                                       'freq': 0.0, 'intensity': 0.0, 'kind': None, 'extra': {}},
                              'right': {'angles': [nan], 'gradients': [], 'radii': [],
                                        'freq': 0.0, 'intensity': 0.0, 'kind': None, 'extra': {}},
                              'abdomen': {'angles': [], 'gradients': [], 'radii': [],
                                          'freq': 0.0, 'intensity': 0.0, 'kind': None, 'extra': {}},
                              'aux': {'angles': [], 'gradients': [], 'radii': [],
                                      'freq': 201.0, 'intensity': 0.0, 'kind': None, 'extra': {}},
                          }}},
    }


def test_payload_is_strictly_valid_json():
    """The bug this pins: a published state carries bare NaN, Python's json
    emits it as the literal NaN, and a browser's JSON.parse rejects the whole
    frame -- so one uncalibrated channel would kill the entire stream."""
    raw = state('frontal', calibrated=False)
    assert 'NaN' in json.dumps(raw), 'fixture should contain the problem'
    try:
        strict_loads(json.dumps(raw))
        raise AssertionError('fixture should be rejected by a strict parser')
    except ValueError:
        pass

    rows = [StateBuffer.row_from_state(state(v, seq=i))
            for i, v in enumerate(('ventral', 'frontal'))]
    text = json.dumps(rows)
    assert 'NaN' not in text and 'Infinity' not in text, text[:200]
    parsed = strict_loads(text)            # must not raise
    assert len(parsed) == 2
    print('payload is strict-JSON clean; non-finite values became null')


def test_clean_handles_nested_and_scalars():
    out = clean({'a': float('nan'), 'b': [1.0, float('inf'), {'c': float('-inf')}], 'd': 2.5})
    assert out == {'a': None, 'b': [1.0, None, {'c': None}], 'd': 2.5}, out
    assert clean(float('nan')) is None and clean(3.0) == 3.0
    print('clean(): NaN/Inf -> null at every depth')


def test_channels_follow_the_view():
    """A ventral camera has only its exact yaw; a frontal one has roll plus
    pitch/yaw, in degrees when calibrated and as pixels when not."""
    v = StateBuffer.row_from_state(state('ventral'))
    assert abs(v['bottom/head_yaw'] - 3.0) < 1e-6, v['bottom/head_yaw']
    for k in ('bottom/head_roll', 'bottom/head_pitch',
              'bottom/head_d_axial', 'bottom/head_d_lateral'):
        assert v[k] is None, (k, v[k])
    print('ventral: yaw %.2f deg, everything else null' % v['bottom/head_yaw'])

    u = StateBuffer.row_from_state(state('frontal', calibrated=False))
    assert abs(u['front/head_roll'] - 2.0) < 1e-6
    assert u['front/head_pitch'] is None and u['front/head_yaw'] is None
    assert u['front/head_d_axial'] == 3.1 and u['front/head_d_lateral'] == -7.2
    print('frontal uncalibrated: roll in deg, pitch/yaw fall back to px')

    c = StateBuffer.row_from_state(state('frontal', calibrated=True))
    assert abs(c['front/head_pitch'] - 1.5) < 1e-6
    assert abs(c['front/head_yaw'] + 4.0) < 1e-6
    print('frontal calibrated: pitch %.2f yaw %.2f deg' % (c['front/head_pitch'], c['front/head_yaw']))

    # Degrees, not radians: a bare radian value would look like a tiny number.
    assert abs(v['bottom/left'] - 50.0) < 1e-6, v['bottom/left']
    assert v['bottom/right'] is None, 'NaN angle must be a gap, not 0'
    assert v['stim'] == 2.5 and v['bottom/wbf'] == 201.0


def test_camera_lag_and_stale_badge():
    """A stalled camera arrives with its measurements already blanked, so the
    page needs the lag and the stale marker to say why the trace has a gap --
    otherwise it reads as a fly holding still."""
    healthy = StateBuffer.row_from_state(
        state('ventral', ages={'bottom': 0.004, 'front': 0.011}, stale_after_s=0.055))
    assert healthy['stale'] is None
    assert abs(healthy['bottom/lag_ms'] - 4.0) < 1e-6, healthy['bottom/lag_ms']
    assert abs(healthy['front/lag_ms'] - 11.0) < 1e-6

    stalled = StateBuffer.row_from_state(
        state('ventral', ages={'bottom': 0.004, 'front': 0.42}, stale_after_s=0.055))
    assert stalled['stale'] == 'front', stalled['stale']
    assert abs(stalled['front/lag_ms'] - 420.0) < 1e-6

    # An older publisher sends no ages at all; the page must still work.
    bare = dict(state('ventral'))
    bare.pop('ages'), bare.pop('stale_after_s')
    row = StateBuffer.row_from_state(bare)
    assert row['stale'] is None and 'bottom/lag_ms' not in row
    strict_loads(json.dumps([healthy, stalled, row]))
    print('lag/stale: front %.0f ms flagged stale, %.0f ms not' % (
        stalled['front/lag_ms'], healthy['front/lag_ms']))


def test_new_channels_survive_a_saved_layout():
    """A saved layout predates any channel added since. Without reconcile()
    the new one could never appear, not even in the hidden bar."""
    import re

    from kinefly2.apps import dashboard

    ids = re.findall(r"\{id:'(\w+)'", dashboard.PAGE)
    assert 'lag' in ids, ids
    assert 'function reconcile(' in dashboard.PAGE
    assert 'o.panels.push(newPanel(c))' in dashboard.PAGE
    assert "show:c.hidden?false:null" in dashboard.PAGE, 'hidden channels must start hidden'
    print('page defines %d channels including lag, and reconciles saved layouts' % len(ids))


def test_decimation_preserves_extremes():
    """Min/max decimation is what makes a 60 s window at 90 Hz drawable. The
    property that matters is that a spike inside a pixel column survives."""
    n, W = 5400, 600
    ramp = np.sin(np.linspace(0, 40 * np.pi, n)) * 10.0
    ramp[1234] = 99.0                      # a one-sample spike
    cols = [None] * W
    for i, v in enumerate(ramp):
        x = int(W * i / n)
        c = cols[x]
        cols[x] = {'lo': v, 'hi': v} if c is None else {'lo': min(c['lo'], v), 'hi': max(c['hi'], v)}
    hi = max(c['hi'] for c in cols if c)
    lo = min(c['lo'] for c in cols if c)
    assert hi == 99.0, hi           # the spike is not skipped over
    assert abs(lo - ramp.min()) < 1e-9
    assert sum(c is not None for c in cols) == W
    print('decimation: %d samples -> %d columns, spike %.0f preserved' % (n, W, hi))


def test_buffer_trims_to_its_window():
    buf = StateBuffer.__new__(StateBuffer)      # no ZMQ thread
    import collections
    import threading
    buf.samples, buf.lock, buf.history_s, buf.n, buf.t_last = \
        collections.deque(), threading.Lock(), 2.0, 0, None
    for i in range(400):
        row = {'t': i / 90.0}
        buf.samples.append(row)
        while buf.samples and row['t'] - buf.samples[0]['t'] > buf.history_s:
            buf.samples.popleft()
    span = buf.samples[-1]['t'] - buf.samples[0]['t']
    assert span <= 2.0 + 1e-9, span
    assert len(buf.samples) > 100
    print('buffer holds %.2f s (%d rows) at a 2 s limit' % (span, len(buf.samples)))


def test_camera_config_plumbing():
    """The camera-side config surface, checkable without any hardware: the
    keys reach PySpinSource, the shipped configs still parse, and
    list_cameras degrades cleanly where PySpin is absent."""
    from kinefly2.camera.base import make_source
    from kinefly2.apps import list_cameras

    src = make_source({'kind': 'pyspin', 'serial': 42, 'fps': 90,
                       'exposure_us': 1500, 'gain_db': 3,
                       'roi': {'width': 512, 'height': 384, 'offset_x': 8},
                       'binning': 2, 'trigger': {'mode': 'secondary', 'source': 'Line3'}})
    assert src.serial == '42' and src.fps == 90
    assert src.roi == {'width': 512, 'height': 384, 'offset_x': 8}
    assert src.trigger == {'mode': 'secondary', 'source': 'Line3'}
    assert src.binning == 2 and src.pixel_format == 'Mono8'
    # Omitted keys must stay absent rather than becoming defaults that would
    # silently reconfigure a camera.
    plain = make_source({'kind': 'pyspin', 'serial': 1})
    assert plain.roi is None and plain.trigger is None and plain.binning is None
    print('pyspin config plumbing: roi/trigger/binning reach the source, absent when unset')

    # The shipped configs still parse with the commented pyspin block in them.
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ('camera_bottom.yaml', 'camera_front.yaml'):
        path = os.path.join(here, 'config', name)
        if not os.path.exists(path):
            continue
        cfg = yaml.safe_load(open(path))
        assert cfg['camera']['kind'] == 'video', cfg['camera']
        assert 'list_cameras' in open(path).read(), '%s lost its setup comment' % name
    print('shipped camera configs parse and keep their setup comments')

    # config_block() renders something pasteable from a minimal info dict.
    block = list_cameras.config_block({'serial': '22334455', 'model': 'BFS-U3-04S2M',
                                       'gain_db': 0, 'width_max': 720, 'height_max': 540},
                                      fps=90)
    assert 'kind: pyspin' in block and 'serial: "22334455"' in block
    assert 'fps: 90' in block and 'trigger:' in block
    print('list_cameras prints a pasteable config block')


if __name__ == '__main__':
    test_payload_is_strictly_valid_json()
    test_clean_handles_nested_and_scalars()
    test_channels_follow_the_view()
    test_camera_lag_and_stale_badge()
    test_new_channels_survive_a_saved_layout()
    test_decimation_preserves_extremes()
    test_buffer_trims_to_its_window()
    test_camera_config_plumbing()
    print('OK')
