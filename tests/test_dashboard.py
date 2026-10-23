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


def state(view, calibrated=False, seq=0, ages=None, stale_after_s=None,
          primary=None, t_camera=None, rec=None):
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
        'primary': cam if primary is None else primary, 'rec': rec,
        'cameras': {cam: {'camera': cam, 'seq': seq,
                          't_camera': nan if t_camera is None else t_camera,
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


def test_time_axis_is_the_frame_clock():
    """The bug this pins: plotting the host clock silently rescales replay.
    Tracking a 60 fps file as fast as the machine manages ran 6.6x real time,
    so a '10 s' window was showing 66 s of video. The primary camera's
    t_camera is video time on a file and the device clock on a camera."""
    r = StateBuffer.row_from_state(state('ventral', seq=90, t_camera=12.5))
    assert r['t'] == 12.5, r['t']
    assert abs(r['t_host'] - 1.0) < 1e-9, r['t_host']

    # No t_camera (a source that cannot supply one) -> host clock, as before.
    r = StateBuffer.row_from_state(state('ventral', seq=90))
    assert abs(r['t'] - 1.0) < 1e-9, r['t']

    # An older publisher names no primary: fall back rather than guess.
    bare = dict(state('ventral', seq=90, t_camera=12.5))
    bare.pop('primary')
    r = StateBuffer.row_from_state(bare)
    assert abs(r['t'] - 1.0) < 1e-9, 'must not adopt a camera clock unasked'

    # A named primary that is not in the state must not crash either.
    odd = dict(state('ventral', seq=90, t_camera=12.5), primary='nosuchcam')
    assert abs(StateBuffer.row_from_state(odd)['t'] - 1.0) < 1e-9
    print('time axis: frame clock when the primary supplies one, host clock otherwise')


def test_saved_file_and_live_stream_agree():
    """One data model or two implementations -- this is the invariant that
    keeps it one. A recording read back must produce the same row keys the
    live stream publishes, or every feature needs writing twice."""
    import tempfile

    from kinefly2.apps.dashboard import rows_from_h5
    from kinefly2.core.flystate import BodypartState, CameraState
    from kinefly2.core.merger import StateMerger
    from kinefly2.output.logger import Hdf5Logger

    # Named like a real recording: file mode takes the run name from the
    # filename, so the '_flystate.h5' suffix stripping is part of this.
    path = os.path.join(tempfile.mkdtemp(), '260910_syn_001_view_flystate.h5')
    merger = StateMerger(primary='bottom', stale_after_s=0.05)
    logger = Hdf5Logger(path, cameras=['bottom', 'front'],
                        metadata={'rig': {'primary': 'bottom',
                                          'cameras': ['bottom', 'front'],
                                          'stale_after_s': 0.05}})
    merger.add_sink(logger)
    merger.set_recording('260910_syn_001_view')

    def cam(name, seq, t_host):
        extra = {'roll': 0.01, 'pitch': 0.02, 'yaw': 0.03,
                 'd_axial': 1.0, 'd_lateral': 2.0, 'roll_sigma': 0.01}
        return CameraState(camera=name, seq=seq, t_camera=seq / 90.0, t_host=t_host,
                           t_done=t_host, parts={
                               'head': BodypartState(angles=[0.05], gradients=[1., 2., .5],
                                                     radii=[1., 2.], intensity=0.9,
                                                     kind='landmarks', extra=extra),
                               'left': BodypartState(angles=[0.8]),
                               'right': BodypartState(angles=[0.7]),
                               'abdomen': BodypartState(angles=[0.1]),
                               'aux': BodypartState(freq=200.0)})

    import time as _t
    live_rows = []
    for i in range(20):
        now = _t.monotonic()
        merger.stage(cam('front', i, now))
        live_rows.append(StateBuffer.row_from_state(merger.submit(cam('bottom', i, now)).to_dict()))
    _t.sleep(0.2)
    logger.close()

    file_rows, info = rows_from_h5(path)
    assert len(file_rows) == 20, len(file_rows)
    assert info['primary'] == 'bottom' and info['primary_known']

    live_keys, file_keys = set(live_rows[5]), set(file_rows[5])
    assert live_keys == file_keys, (
        'live-only %s file-only %s' % (sorted(live_keys - file_keys),
                                       sorted(file_keys - live_keys)))

    # And the same values, not merely the same names.
    L, F = live_rows[5], file_rows[5]
    for k in ('bottom/left', 'bottom/head_roll', 'front/head_pitch',
              'bottom/head_q', 'bottom/wbf', 'bottom/head_d_axial'):
        assert abs(L[k] - F[k]) < 1e-6, (k, L[k], F[k])
    assert abs(F['t'] - 5 / 90.0) < 1e-9, F['t']      # frame clock, not host
    assert F['rec'] == '260910_syn_001_view'
    print('file and live agree on all %d keys, values match, t = frame clock' % len(live_keys))


def test_sweep_and_axis_arithmetic_in_js():
    """The navigation maths lives in the page, so run it in a real JS engine.
    Skipped where JavaScriptCore is absent (non-macOS)."""
    import re
    import subprocess

    from kinefly2.apps.dashboard import PAGE

    jsc = ('/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/'
           'Helpers/jsc')
    if not os.path.exists(jsc):
        print('jsc not present; skipping the JS sweep test')
        return

    def extract(name):
        m = re.search(r'function\s+%s\s*\(' % re.escape(name), PAGE)
        assert m, 'no function %s in the page' % name
        i, depth = PAGE.index('{', m.end() - 1), 0
        for j in range(i, len(PAGE)):
            if PAGE[j] == '{':
                depth += 1
            elif PAGE[j] == '}':
                depth -= 1
                if depth == 0:
                    return PAGE[m.start():j + 1]
        raise AssertionError('unbalanced braces in %s' % name)

    consts = '\n'.join(m.group(0) for m in
                       re.finditer(r'^const (TICKS)=.*?;$', PAGE, re.M))
    assert 'TICKS' in consts
    src = HARNESS_HEAD + consts + '\n' + \
        '\n'.join(extract(n) for n in ('bounds', 'rightEdge', 'tickStep', 'mmss')) + \
        HARNESS_BODY
    import tempfile
    js = os.path.join(tempfile.mkdtemp(), 'sweep.js')
    open(js, 'w').write(src)
    out = subprocess.run([jsc, js], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT).stdout.decode()
    assert 'FAILED' not in out and 'Exception' not in out, out
    print(out.strip().splitlines()[-1])


HARNESS_HEAD = """
var data=[], recT0=null, mode='live', recName=null;
var view={win:30, right:0, follow:true};
"""

HARNESS_BODY = r"""
function rows(t0,t1,step){ var a=[]; for(var t=t0;t<=t1+1e-9;t+=step) a.push({t:t,t_host:t}); return a; }
var fails=0;
function eq(got,want,msg){ if(Math.abs(got-want)>1e-6){ print('FAIL '+msg+': got '+got+' want '+want); fails++; } }

// The sweep: while the data are shorter than the window the right edge stays
// pinned one window after the start, so the trace fills from the LEFT.
view.win=30; view.follow=true;
data=rows(100,105,0.1);
eq(rightEdge(),130,'right pinned at first+win while filling');
var W=600, x=function(t){return W*(1-(rightEdge()-t)/view.win);};
eq(x(100),0,'oldest sample at the left edge');
eq(x(105),100,'newest sample 1/6 across');

// Changeover to scrolling is seamless.
data=rows(100,130,0.1); eq(rightEdge(),130,'exactly full: both rules agree');
data=rows(100,160,0.1); eq(rightEdge(),160,'scrolling follows the newest sample');
eq(x(160),600,'newest sample at the right edge');

// A recording restarts the sweep partway through the buffer.
recT0=140; data=rows(100,145,0.1);
eq(bounds().sweep,140,'recording start becomes the sweep origin');
eq(rightEdge(),170,'sweep restarts at the recording');
recT0=null;

// Scrolled back, clamped to the data at both ends.
data=rows(100,160,0.1); view.follow=false;
view.right=105; eq(rightEdge(),130,'cannot scroll before the oldest window');
view.right=999; eq(rightEdge(),160,'cannot scroll past the newest sample');
view.right=145; eq(rightEdge(),145,'a valid position stays put');

// An empty buffer must not produce NaN.
data=[]; view.follow=true; eq(rightEdge(),30,'empty buffer: no NaN');

// Ticks keep a readable number of divisions at every window.
[2,5,10,30,60,120,300].forEach(function(w){ view.win=w; var k=w/tickStep(w);
  if(!(k>=2&&k<=8)){ print('FAIL ticks for '+w+'s -> '+k); fails++; } });

// mm:ss, including the rounding edge that would otherwise print 1:60.
view.win=30;
if(mmss(0)!=='0:00'){print('FAIL mmss(0)='+mmss(0));fails++;}
if(mmss(75)!=='1:15'){print('FAIL mmss(75)='+mmss(75));fails++;}
if(mmss(119.6)!=='2:00'){print('FAIL mmss(119.6)='+mmss(119.6)+' (the 1:60 bug)');fails++;}
view.win=5;
if(mmss(65.4)!=='1:05.4'){print('FAIL mmss(65.4)='+mmss(65.4));fails++;}
print(fails ? (fails+' FAILED') : 'sweep, clamping, ticks and mm:ss all pass');
"""


JSC = '/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc'


def _page_function(name):
    import re

    from kinefly2.apps.dashboard import PAGE
    m = re.search(r'function\s+%s\s*\(' % re.escape(name), PAGE)
    assert m, 'no function %s in the page' % name
    i, depth = PAGE.index('{', m.end() - 1), 0
    for j in range(i, len(PAGE)):
        if PAGE[j] == '{':
            depth += 1
        elif PAGE[j] == '}':
            depth -= 1
            if depth == 0:
                return PAGE[m.start():j + 1]
    raise AssertionError('unbalanced braces in %s' % name)


def test_traces_survive_zooming_in():
    """The bug this pins: traces vanished whenever the view was zoomed in.
    The renderer binned samples per pixel column and lifted the pen at every
    column with no sample in it -- and zoomed in (on a 2x screen, 10 s of a
    60 fps video is 600 samples over ~2800 columns) every sample sits between
    empty columns, so each became a lone moveTo and nothing was stroked.
    Runs the page's own tracePath in JavaScriptCore; skipped where absent."""
    import subprocess
    import tempfile

    if not os.path.exists(JSC):
        print('jsc not present; skipping the trace-path test')
        return
    body = r"""
var fails=0;
function check(ok,msg){ if(!ok){ print('FAIL '+msg); fails++; } }
function rows(n,rate,f){ var a=[]; for(var i=0;i<n;i++) a.push({t:100+i/rate,'bottom/left':f(i)}); return a; }
// Zoomed in: 600 samples (10 s at 60 fps) across 2800 columns.
var sparse=rows(600,60,function(i){return Math.sin(i/20)*30+90;});
var s=tracePath(sparse,'bottom/left',100+599/60,10,2800);
check(s.length===1,'sparse: one unbroken line, got '+s.length+' segments');
check(s.length&&s[0].length===600,'sparse: every sample drawn, got '+(s.length?s[0].length:0));
// Zoomed out: 5400 samples across 600 columns, with a one-sample spike.
var dense=rows(5400,90,function(i){return i===1234?999:Math.sin(i/7)*10;});
var d=tracePath(dense,'bottom/left',100+5399/90,60,600);
var top=-1e9, pts=0; d.forEach(function(g){ pts+=g.length; g.forEach(function(p){ if(p[1]>top) top=p[1]; }); });
check(d.length===1,'dense: one line, got '+d.length);
check(top===999,'dense: the spike survives decimation, top '+top);
check(pts<=601*4,'dense: decimated to the columns (x=0..600), '+pts+' points');
// A missing value (a stale camera, an uncalibrated channel) is a real gap.
sparse[300]['bottom/left']=null;
var g=tracePath(sparse,'bottom/left',100+599/60,10,2800);
check(g.length===2,'a null splits the line in two, got '+g.length);
check(g.length===2&&g[0].length+g[1].length===599,'no sample lost around the gap');
// Samples outside the window are not drawn.
var w=tracePath(sparse,'bottom/left',100+599/60,1,2800);
check(w.length===1&&w[0].length===61,'window clip: 1 s holds 61 samples, got '+(w.length?w[0].length:0));
print(fails ? (fails+' FAILED') : 'trace path: sparse, dense, gaps and clipping all pass');
"""
    src = '\n'.join(_page_function(n) for n in ('val', 'tracePath')) + body
    js = os.path.join(tempfile.mkdtemp(), 'trace.js')
    open(js, 'w').write(src)
    out = subprocess.run([JSC, js], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT).stdout.decode()
    assert 'FAILED' not in out and 'Exception' not in out and 'Error' not in out, out
    print(out.strip().splitlines()[-1])


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


def test_buffer_trims_on_the_host_clock():
    """Trimming must not use the display axis. A looping video resets its
    frame clock, so a span measured on `t` goes negative and the buffer then
    never trims again -- it grows until the process dies."""
    import collections
    import threading

    def fresh(history_s=2.0):
        buf = StateBuffer.__new__(StateBuffer)          # no ZMQ thread
        buf.samples, buf.lock, buf.history_s, buf.n, buf.t_last = \
            collections.deque(), threading.Lock(), history_s, 0, None
        return buf

    buf = fresh()
    for i in range(400):
        buf.samples.append({'t': i / 90.0, 't_host': i / 90.0})
        buf.trim()
    span = buf.samples[-1]['t_host'] - buf.samples[0]['t_host']
    assert span <= 2.0 + 1e-9, span
    assert len(buf.samples) > 100
    print('buffer holds %.2f s (%d rows) at a 2 s limit' % (span, len(buf.samples)))

    # The looping-video case: the frame clock restarts, the host clock does not.
    buf = fresh()
    for i in range(400):
        buf.samples.append({'t': (i % 50) / 90.0, 't_host': i / 90.0})
        buf.trim()
    assert len(buf.samples) <= 200, 'a reset frame clock must still trim: %d rows' % len(buf.samples)
    span = buf.samples[-1]['t_host'] - buf.samples[0]['t_host']
    assert span <= 2.0 + 1e-9, span
    print('looping video: frame clock resets 8x, buffer still bounded (%d rows)' % len(buf.samples))


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

    # The shipped configs still parse with the commented pyspin block in them,
    # and on a rig, with that block swapped in for the video lines.
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ('camera_bottom.yaml', 'camera_front.yaml'):
        path = os.path.join(here, 'config', name)
        if not os.path.exists(path):
            continue
        cfg = yaml.safe_load(open(path))
        assert cfg['camera']['kind'] in ('video', 'pyspin'), cfg['camera']
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
    test_time_axis_is_the_frame_clock()
    test_saved_file_and_live_stream_agree()
    test_sweep_and_axis_arithmetic_in_js()
    test_decimation_preserves_extremes()
    test_buffer_trims_on_the_host_clock()
    test_camera_config_plumbing()
    print('OK')
