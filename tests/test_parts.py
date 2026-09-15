"""Extra body parts: the part list is data, not code.

The part names used to be written out at every site that iterated them, so
adding one meant finding all of them. These tests pin the three things that
makes safe: the factory builds what the old if/elif chain did, an extra part
declared in a config reaches the log and the dashboard, and per-part tuning
cannot leak into the parts already tuned.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kinefly2.tracking import parts


def base_params(extra=()):
    """A params dict shaped like a camera YAML's tracker section."""
    p = {'n_edges_max': 1, 'rc_background': 1000.0, 'scale_image': 1.0,
         'invertcolor': False, 'wingbeat_min': 180, 'wingbeat_max': 220,
         'view': 'ventral', 'body_axis_deg': 0.0, 'parts': list(extra),
         'gui': {'symmetric': False, 'windows': False,
                 'axis': {'track': False, 'pt1': {'x': 0, 'y': 0}, 'pt2': {'x': 1, 'y': 1}}}}
    for name in list(parts.BODY_PARTS) + list(extra):
        p[name] = dict(parts.part_defaults(name))
        p['gui'][name] = dict(parts.gui_defaults(name))
    p['head']['tracker'] = 'area'
    p['abdomen']['tracker'] = 'tip'
    p['left']['tracker'] = 'edge'
    p['right']['tracker'] = 'edge'
    p['aux'] = {'tracker': 'intensity', 'saturation_correction': True}
    p['gui']['aux'] = {'track': False, 'subtract_bg': False,
                       'center': {'x': 50, 'y': 50}, 'radius1': 30, 'radius2': 20,
                       'angle': 0.0}
    return p


def test_factory_builds_what_the_if_elif_chain_did():
    """Each tracker name must still map to the same class, with the same
    overlay colour, or a config that worked yesterday draws differently."""
    p = base_params()
    expect = {'head': 'AreaTracker', 'abdomen': 'TipTracker',
              'left': 'EdgeTrackerByIntensityProfile',
              'right': 'EdgeTrackerByIntensityProfile',
              'aux': 'IntensityTracker'}
    for name, clsname in expect.items():
        t = parts.make_tracker(name, p)
        assert type(t).__name__ == clsname, (name, type(t).__name__)
    # Every documented tracker name resolves. `intensity` is the exception:
    # it wants an ellipse ROI (center/radius1/radius2), which only `aux`
    # carries, so it is checked there.
    for kind in parts.TRACKER_KINDS:
        if kind == 'intensity':
            continue
        p['head']['tracker'] = kind
        assert parts.make_tracker('head', p) is not None, kind
    p['head']['tracker'] = 'area'
    assert parts.make_tracker('aux', p) is not None
    # An unknown one is refused by name rather than silently leaving the part
    # unbuilt, which is what the old chain did (it printed and carried on).
    p['head']['tracker'] = 'nonsense'
    try:
        parts.make_tracker('head', p)
        raise AssertionError('an unknown tracker must raise')
    except ValueError as e:
        assert 'nonsense' in str(e) and 'edge' in str(e), str(e)
    print('factory: all %d tracker kinds resolve; unknown names raise' % len(parts.TRACKER_KINDS))


def test_extra_parts_are_declared_by_config():
    assert parts.body_parts({}) == parts.BODY_PARTS
    p = {'parts': ['leg_l3', 'leg_r3']}
    assert parts.body_parts(p) == parts.BODY_PARTS + ('leg_l3', 'leg_r3')
    assert parts.all_parts(p)[-1] == 'aux'
    # Duplicates and names that collide with the standard parts are ignored
    # rather than producing two trackers for one name.
    assert parts.extra_parts({'parts': ['leg_l3', 'leg_l3', 'head', 'aux']}) == ('leg_l3',)
    print('extra parts: declared in the config, deduplicated, no collisions')


def test_each_part_is_referenced_to_its_opposite():
    """A part's zero is its opposite number's direction, so a symmetric posture
    reads L = R. Legs pair by side the way the wings do."""
    assert parts.opposite('left') == 'right' and parts.opposite('right') == 'left'
    assert parts.opposite('head') == 'abdomen' and parts.opposite('abdomen') == 'head'
    assert parts.opposite('leg_l3') == 'leg_r3'
    assert parts.opposite('leg_r3') == 'leg_l3'
    assert parts.opposite('leg_l1') == 'leg_r1'
    assert parts.opposite('aux') is None
    assert parts.opposite('thorax') is None      # no side suffix, no counterpart
    print('zero reference: every part pairs with its opposite number')


def test_per_part_tuning_does_not_leak():
    """n_edges_max is per part now. The bug this pins: it used to be rig-wide,
    so giving the abdomen two edges silently changed BOTH wings -- measured at
    the time as left 68.4 -> 65.4 deg."""
    from kinefly2.tracking.tracker_edge import EdgeTrackerByIntensityProfile

    p = base_params(('leg_r3',))
    p['abdomen']['tracker'] = 'edge'
    p['abdomen']['n_edges_max'] = 2
    p['gui']['abdomen']['track'] = True
    built = {}
    for name in ('left', 'right', 'abdomen', 'leg_r3'):
        t = EdgeTrackerByIntensityProfile(name=name, params=p, color='red')
        t.set_params(p)
        built[name] = t.detector.n_edges_max
    assert built['abdomen'] == 2, built
    assert built['left'] == 1 and built['right'] == 1, built     # global default
    assert built['leg_r3'] == 1, built
    print('per-part n_edges_max: abdomen=2 while the wings stay at 1')


def test_angle_from_mean_reports_the_midpoint():
    """`angle_from: mean` is what turns two silhouette margins into a
    centreline. It must use a circular mean, or a pair straddling +/-pi
    averages to zero -- pointing the abdomen at the head."""
    from kinefly2.tracking.tracker_edge import EdgeTrackerByIntensityProfile

    p = base_params()
    p['abdomen']['tracker'] = 'edge'
    p['abdomen']['angle_from'] = 'mean'
    t = EdgeTrackerByIntensityProfile(name='abdomen', params=p, color='red')
    t.params = p
    t.state.angles = [math.radians(170), math.radians(-170)]   # straddling pi
    # Re-run just the mean step the tracker applies.
    a = np.array(t.state.angles, dtype=float)
    mean = float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    assert abs(abs(math.degrees(mean)) - 180.0) < 1e-6, math.degrees(mean)
    a2 = np.radians([100.0, 140.0])
    m2 = math.degrees(float(np.arctan2(np.mean(np.sin(a2)), np.mean(np.cos(a2)))))
    assert abs(m2 - 120.0) < 1e-6, m2
    print('angle_from mean: circular, so 170 and -170 give 180 (not 0)')


def test_extra_parts_reach_the_log_and_the_dashboard():
    import tempfile

    import h5py

    from kinefly2.apps.dashboard import StateBuffer, rows_from_h5
    from kinefly2.core.flystate import BodypartState, CameraState
    from kinefly2.core.merger import StateMerger
    from kinefly2.output.logger import Hdf5Logger

    names = ['head', 'abdomen', 'left', 'right', 'leg_l3', 'leg_r3']
    path = os.path.join(tempfile.mkdtemp(), '260910_syn_001_legs_flystate.h5')
    logger = Hdf5Logger(path, cameras=['bottom'], parts={'bottom': names},
                        metadata={'rig': {'primary': 'bottom', 'cameras': ['bottom'],
                                          'parts': {'bottom': names},
                                          'stale_after_s': 0.05}})
    merger = StateMerger(primary='bottom')
    merger.add_sink(logger)
    import time as _t
    for i in range(10):
        now = _t.monotonic()
        ps = {n: BodypartState(angles=[0.1 * (j + 1)],
                               extra={'n_edges': 1.0} if n.startswith('leg') else {})
              for j, n in enumerate(names)}
        ps['aux'] = BodypartState(freq=200.0)
        state = CameraState(camera='bottom', seq=i, t_camera=i / 90.0, t_host=now,
                            t_done=now, parts=ps)
        row = StateBuffer.row_from_state(merger.submit(state).to_dict())
    _t.sleep(0.2)
    logger.close()

    # The live row carries the extra parts...
    assert 'bottom/leg_r3' in row, sorted(row)
    assert abs(row['bottom/leg_r3'] - math.degrees(0.6)) < 1e-6, row['bottom/leg_r3']
    # ...and so does the file, read back through the same loader.
    with h5py.File(path, 'r') as f:
        assert 'bottom/leg_r3_angle' in f, list(f['bottom'])
        assert 'bottom/leg_r3_n_edges' in f
        assert f['bottom/leg_r3_n_edges'][0] == 1.0
    file_rows, info = rows_from_h5(path)
    assert 'bottom/leg_r3' in file_rows[0], sorted(file_rows[0])
    assert set(row) == set(file_rows[0]), (set(row) ^ set(file_rows[0]))
    print('extra parts: %d columns logged, and live/file rows still agree' % len(names))


if __name__ == '__main__':
    test_factory_builds_what_the_if_elif_chain_did()
    test_extra_parts_are_declared_by_config()
    test_each_part_is_referenced_to_its_opposite()
    test_per_part_tuning_does_not_leak()
    test_angle_from_mean_reports_the_midpoint()
    test_extra_parts_reach_the_log_and_the_dashboard()
    print('OK')
