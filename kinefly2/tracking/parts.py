"""Which body parts exist, and which tracker each one gets.

One list, in one place. The part names used to be written out at every site
that iterated them -- the GUI alone had ten copies -- and Fly.__init__ repeated
a near-identical six-branch if/elif per part. Adding a part meant finding all
of them, so this module is what makes extra parts (the hind legs) a config
entry rather than an edit in a dozen files.

A camera YAML adds parts by naming them:

    tracker:
      parts: [leg_l3, leg_r3]     # extra angle-bearing parts on this camera
      leg_l3: {tracker: edge, threshold: 0.0}
      gui:
        leg_l3: {track: true, hinge: {x: 162, y: 196}, ...}

Extra parts are ordinary parts: hinge, annulus, angular window, a tracker
choice, and one angle out. Nothing downstream distinguishes them from a wing.
"""

# The parts every camera has. `aux` is separate because it is not angle-bearing
# -- it is an ellipse ROI for the wingbeat detector, with no hinge or annulus.
BODY_PARTS = ('head', 'abdomen', 'left', 'right')
AUX_PART = 'aux'

# tracker name -> (class name in its module, module, overlay colour)
TRACKER_KINDS = {
    'area':      ('AreaTracker',                   '.tracker_area',      'blue'),
    'edge':      ('EdgeTrackerByIntensityProfile', '.tracker_edge',      'red'),
    'tip':       ('TipTracker',                    '.tracker_tip',       'purple'),
    'landmarks': ('LandmarkTracker',               '.tracker_landmarks', 'green'),
    'rotation':  ('RotationTracker',               '.tracker_rotation',  'green'),
    'intensity': ('IntensityTracker',              '.tracker_intensity', 'hot_pink'),
}


def extra_parts(params):
    """Parts this camera declares beyond the standard four."""
    names = (params or {}).get('parts') or []
    out = []
    for n in names:
        n = str(n)
        if n in BODY_PARTS or n == AUX_PART or n in out:
            continue
        out.append(n)
    return tuple(out)


def body_parts(params):
    """Every angle-bearing part: the standard four plus any extras."""
    return BODY_PARTS + extra_parts(params)


def all_parts(params):
    """Everything with a tracker, `aux` included."""
    return body_parts(params) + (AUX_PART,)


def opposite(name):
    """The part a part's zero is measured against.

    Each part is referenced to its opposite number -- head to abdomen, each
    wing to the other wing, each leg to the leg on the other side -- so a
    symmetric posture reads L = R and no other hinge can shift it (README
    2.5.5). Returns None for a part with no counterpart.
    """
    fixed = {'head': 'abdomen', 'abdomen': 'head', 'left': 'right', 'right': 'left'}
    if name in fixed:
        return fixed[name]
    # Side-suffixed names (leg_l3 / leg_r3) pair with their mirror.
    for a, b in (('_l', '_r'), ('_r', '_l')):
        i = name.rfind(a)
        if i >= 0:
            return name[:i] + b + name[i + len(a):]
    return None


def make_tracker(name, params, bEqualizeHist=False):
    """Build the tracker a part's config asks for.

    Trackers are imported here rather than at module scope so that
    bodypart_motion can import this module for `opposite()` without a cycle.
    """
    kind = (params.get(name) or {}).get('tracker')
    if name == AUX_PART:
        kind = kind or 'intensity'
    spec = TRACKER_KINDS.get(kind)
    if spec is None:
        raise ValueError(
            "%s: tracker must be one of %s, got %r"
            % (name, ', '.join(sorted(TRACKER_KINDS)), kind))
    clsname, modname, color = spec
    import importlib
    mod = importlib.import_module(modname, package=__package__)
    cls = getattr(mod, clsname)
    if name == AUX_PART:
        color = 'yellow'
    return cls(name=name, params=params, color=color, bEqualizeHist=bEqualizeHist)


def part_defaults(name):
    """Tracker-section defaults for a part the config did not fully specify."""
    return {'tracker': 'edge', 'autozero': True, 'threshold': 0.0,
            'feathering': 0.25, 'saturation_correction': True}


def gui_defaults(name):
    """GUI-section defaults: off, with a placeholder annulus to be dragged."""
    return {'track': False, 'subtract_bg': False, 'stabilize': False,
            'hinge': {'x': 175, 'y': 150}, 'radius_outer': 20, 'radius_inner': 10,
            'angle_hi': 2.3562, 'angle_lo': 0.7854}
